"""Frozen Teacher collection with independently optimized Student PPO + AMP."""
import hashlib
import json
from pathlib import Path

import torch

from legged_gym.mixed_handoff import STUDENT_CONTROL, load_actor
from rsl_rl.algorithms.mixed_handoff_ppo import MixedHandoffPPO
from rsl_rl.modules import ActorCritic, AMP
from rsl_rl.modules.masked_amp import StudentAMPHistory, project_amp
from rsl_rl.storage.mixed_handoff_rollout_storage import MixedHandoffRolloutStorage


TRAINING_CONTRACT = dict(actor_obs=738, critic_obs=126, actions=29, history_frames=6,
                         amp_obs=540, amp_frame_features=54, amp_history=10,
                         expert_source="CarryWith-only", version=1)


def read_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def initialize_student(student, path, contract, device="cpu", allow_legacy_contract=False):
    source, spec = load_actor(path, contract, device, allow_legacy_contract)
    if spec["dims"] != [738, 512, 256, 256, 29]:
        raise ValueError("Student initialization requires 738->512->256->256->29 ELU Actor")
    student.actor.load_state_dict(source.state_dict(), strict=True)
    std = read_checkpoint(path)["model_state_dict"].get("std")
    std_source = "configured init_noise_std"
    if std is not None and std.shape == student.std.shape:
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("Checkpoint action std must be finite and positive")
        with torch.no_grad():
            student.std.copy_(std.to(device))
        std_source = "checkpoint model_state_dict.std"
    return dict(**file_identity(path), action_std=std_source)


class MixedHandoffOnPolicyRunner:
    def __init__(self, env, train_cfg, teacher_checkpoint, contract,
                 student_init_checkpoint=None, expert=None, log_dir=None,
                 device="cpu", allow_legacy_contract=False, initialize=True):
        self.env, self.train_cfg, self.contract = env, train_cfg, contract
        self.device, self.log_dir = torch.device(device), Path(log_dir) if log_dir else None
        self.cfg, amp_cfg = train_cfg["runner"], train_cfg["amp"]
        if not 0 <= amp_cfg["amp_coef"] <= 1 or amp_cfg["reward_scale"] < 0:
            raise ValueError("AMP reward requires 0 <= amp_coef <= 1 and nonnegative reward_scale")
        self.teacher, teacher_spec = load_actor(teacher_checkpoint, contract, device, allow_legacy_contract)
        if teacher_spec["dims"] != [738, 512, 256, 256, 29]:
            raise ValueError("Teacher must retain the audited 738->512->256->256->29 Actor")
        self.teacher_identity = file_identity(teacher_checkpoint)
        self.teacher_snapshot = {k: p.detach().clone() for k, p in self.teacher.state_dict().items()}
        ac = ActorCritic(738, 126, 6, 29, **train_cfg["policy"]).to(device)
        self.student_init = initialize_student(ac, student_init_checkpoint or teacher_checkpoint,
                                              contract, device, allow_legacy_contract) if initialize else None
        amp = AMP(540, amp_cfg["amp_coef"], device=device).to(device) if amp_cfg["enabled"] else None
        normalizer = None
        if amp is not None and expert is None:
            raise ValueError("Enabled AMP requires independent CarryWith-only expert motion")
        if amp is not None and amp_cfg["use_normalizer"]:
            from rsl_rl.utils.utils import AmpNormalizer
            normalizer = AmpNormalizer(540, device=device)
        self.alg = MixedHandoffPPO(ac, amp, normalizer, expert, **train_cfg["algorithm"])
        self.storage = MixedHandoffRolloutStorage(env.num_envs, self.cfg["num_steps_per_env"], device=device)
        self.history = StudentAMPHistory(env.num_envs, device)
        self.collection_iteration = 0
        self.teacher_steps = self.student_steps = 0
        self.handoff_successes = self.handoff_failures = 0
        self.writer = None
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
        self.obs = self.critic_obs = None

    def reset(self):
        obs, critic = self.env.reset()
        self.obs, self.critic_obs = obs.to(self.device).clone(), critic.to(self.device).clone()
        self.history.count.zero_()

    @torch.no_grad()
    def collect_step(self):
        env, ac = self.env, self.alg.actor_critic
        student = (env.handoff.phase.to(self.device) == STUDENT_CONTROL).clone()
        n = env.num_envs
        actions = torch.zeros(n, 29, device=self.device)
        # Teacher rows are placeholders, never sampled or evaluated by Student.
        values, log_prob = torch.zeros(n, device=self.device), torch.zeros(n, device=self.device)
        mu, sigma = torch.zeros_like(actions), torch.ones_like(actions)
        if (~student).any():
            actions[~student] = self.teacher(self.obs[~student])
        if student.any():
            sampled = ac.act(self.obs[student])
            actions[student] = sampled
            values[student] = ac.evaluate(self.critic_obs[student]).squeeze(-1)
            log_prob[student] = ac.get_actions_log_prob(sampled)
            mu[student], sigma[student] = ac.action_mean, ac.action_std
        obs, critic, task_reward, dones, infos, terminal_ids, terminal_critic, amp_state = env.step(actions.to(env.device))
        diag = infos["handoff"]
        executed_student = diag["executed_phase"].to(self.device) == STUDENT_CONTROL
        if not torch.equal(student, executed_student):
            raise RuntimeError("Pre-step phase and executed action ownership differ")
        obs, critic = obs.to(self.device).clone(), critic.to(self.device).clone()
        dones = dones.to(self.device).bool().clone()
        task_reward = task_reward.to(self.device).clone()
        timeouts = diag["timeout"].to(self.device).bool()
        failures = diag["failure"].to(self.device).bool()
        next_student = env.handoff.phase.to(self.device) == STUDENT_CONTROL
        next_critic = critic.clone()
        terminal_ids = terminal_ids.to(self.device).long()
        next_critic[terminal_ids] = terminal_critic.to(self.device).clamp(
            -env.cfg.normalization.clip_observations, env.cfg.normalization.clip_observations)
        next_values = torch.zeros_like(values)
        bootstrap = student & ~failures & ((~dones & next_student) | (dones & timeouts))
        if bootstrap.any():
            next_values[bootstrap] = ac.evaluate(next_critic[bootstrap]).squeeze(-1)
        amp_obs = project_amp(amp_state.to(self.device))
        amp_valid = self.history.step(student, dones)
        amp_reward = torch.zeros_like(task_reward)
        reward = task_reward.clone()
        enabled = self.alg.amp is not None
        if enabled and amp_valid.any():
            amp_reward[amp_valid] = self.alg.amp.predict_reward(amp_obs[amp_valid], self.alg.amp_normalizer).squeeze(-1) * self.train_cfg["amp"]["reward_scale"]
            coef = self.train_cfg["amp"]["amp_coef"]
            reward[amp_valid] = (1-coef) * task_reward[amp_valid] + coef * amp_reward[amp_valid]
        # Both inputs are pre-reset when used for timeouts. Smoothness excludes
        # every done and phase boundary, so reset actor observations are harmless.
        self.storage.add(observations=self.obs, next_observations=obs,
                         critic_observations=self.critic_obs, next_critic_observations=next_critic,
                         actions=torch.where(student[:, None], actions, 0.), mu=mu, sigma=sigma,
                         values=values, next_values=next_values, log_prob=log_prob,
                         rewards=reward, dones=dones, timeouts=timeouts, failures=failures,
                         student_valid=student, next_student=next_student, amp_valid=amp_valid,
                         amp_observations=amp_obs)
        self.obs, self.critic_obs = obs, critic
        self.teacher_steps += int((~student).sum())
        self.student_steps += int(student.sum())
        trigger = diag["handoff_trigger"].to(self.device).bool()
        failed = diag["handoff_failure"].to(self.device).bool()
        self.handoff_successes += int(trigger.sum())
        self.handoff_failures += int(failed.sum())
        samples = dict(task_reward=task_reward[student], amp_reward=amp_reward[student], combined_reward=reward[student],
                       amp_reward_valid=amp_reward[amp_valid] if enabled else amp_reward[:0],
                       handoff_time=diag["handoff_time"].to(self.device)[trigger])
        for k, v in diag.items():
            if k.startswith(("student/", "student_reward_raw/", "student_reward_weighted/")):
                samples[k] = v.to(self.device)[student].float().flatten()
        reasons = diag["student/termination_reason"].to(self.device).long()
        for bit, name in ((1, "box_drop"), (2, "separation"), (4, "grasp_loss"), (8, "tilt"), (16, "physical"), (32, "timeout")):
            samples["termination/" + name] = ((reasons[student & dones] & bit) != 0).float()
        return samples

    def assert_teacher_frozen(self):
        if self.teacher.training or any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError("Teacher is no longer frozen")
        if any(not torch.equal(p, self.teacher_snapshot[k]) for k, p in self.teacher.state_dict().items()):
            raise RuntimeError("Frozen Teacher weights changed")

    def learn(self, iterations, optimize=True, max_optimizer_updates=None):
        if self.obs is None:
            self.reset()
        start_updates = self.alg.optimizer_updates
        try:
            for _ in range(iterations):
                stats = {}
                for _ in range(self.cfg["num_steps_per_env"]):
                    for k, v in self.collect_step().items():
                        if v.numel():
                            sums = stats.setdefault(k, [0., 0., 0])
                            sums[0] += v.sum().item()
                            sums[1] += v.square().sum().item()
                            sums[2] += v.numel()
                count = int(self.storage.data["student_valid"][:self.storage.step].sum())
                amp_count = len(self.storage.amp_samples())
                self.storage.compute_returns(self.alg.gamma, self.alg.lam)
                if optimize:
                    metrics = self.alg.update(self.storage)
                else:
                    self.storage.clear()
                    metrics = dict(student_valid_count=count, amp_valid_count=amp_count, updated=0)
                self.assert_teacher_frozen()
                self.collection_iteration += 1
                attempts = self.handoff_successes + self.handoff_failures
                metrics.update(collection_iterations=self.collection_iteration, optimizer_updates=self.alg.optimizer_updates,
                               optimizer_steps=self.alg.optimizer_steps, teacher_steps=self.teacher_steps, student_steps=self.student_steps,
                               student_step_fraction=self.student_steps / max(1, self.teacher_steps+self.student_steps),
                               student_valid_ratio=count / (self.env.num_envs * self.cfg["num_steps_per_env"]),
                               amp_valid_fraction=amp_count / max(1, count),
                               handoff_success_rate=self.handoff_successes / max(1, attempts),
                               handoff_failure_rate=self.handoff_failures / max(1, attempts))
                for k, (total, square, number) in stats.items():
                    metrics["performance/" + k] = total / number
                    if "reward" in k:
                        metrics["performance/" + k + "_std"] = max(0., square / number - (total / number)**2)**.5
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(k, v, self.collection_iteration)
                    self.writer.flush()
                print(json.dumps(metrics, sort_keys=True), flush=True)
                if self.log_dir and self.collection_iteration % self.cfg["save_interval"] == 0:
                    self.save(self.log_dir / f"model_{self.collection_iteration}.pt")
                if max_optimizer_updates is not None and self.alg.optimizer_updates-start_updates >= max_optimizer_updates:
                    break
            if self.log_dir:
                self.save(self.log_dir / f"model_{self.collection_iteration}.pt")
        finally:
            if self.writer:
                self.writer.close()

    def save(self, path):
        normalizer = self.alg.amp_normalizer
        torch.save(dict(format="mixed_handoff_student_v1", scope1_contract=self.contract,
                        training_contract=TRAINING_CONTRACT, train_cfg=self.train_cfg,
                        teacher=self.teacher_identity, student_init=self.student_init,
                        expert=self.alg.motion_buffer.identity if self.alg.motion_buffer else None,
                        model_state_dict=self.alg.actor_critic.state_dict(),
                        amp_state_dict=self.alg.amp.state_dict() if self.alg.amp else None,
                        optimizer_state_dict=self.alg.optimizer.state_dict(),
                        normalizer={k: getattr(normalizer, k) for k in ("mean", "var", "count")} if normalizer else None,
                        collection_iteration=self.collection_iteration,
                        optimizer_updates=self.alg.optimizer_updates, optimizer_steps=self.alg.optimizer_steps,
                        learning_rate=self.alg.learning_rate,
                        teacher_steps=self.teacher_steps, student_steps=self.student_steps,
                        handoff_successes=self.handoff_successes, handoff_failures=self.handoff_failures), path)

    def load(self, path):
        checkpoint = read_checkpoint(path)
        if checkpoint.get("format") != "mixed_handoff_student_v1":
            raise ValueError("Full Student resume requires a mixed_handoff_student_v1 checkpoint; use --student_init_checkpoint for Teacher weights")
        if checkpoint["scope1_contract"] != self.contract or checkpoint["training_contract"] != TRAINING_CONTRACT:
            raise ValueError("Student resume observation/AMP contracts differ")
        if checkpoint["teacher"]["sha256"] != self.teacher_identity["sha256"]:
            raise ValueError("Student resume frozen Teacher identity differs")
        for key in ("algorithm", "policy", "amp"):
            saved_cfg, current_cfg = checkpoint["train_cfg"][key], self.train_cfg[key]
            if key == "amp":
                saved_cfg = {k: v for k, v in saved_cfg.items() if k != "expert_config"}
                current_cfg = {k: v for k, v in current_cfg.items() if k != "expert_config"}
            if saved_cfg != current_cfg:
                raise ValueError("Student resume configuration differs: " + key)
        for key in ("control", "sim", "handoff", "student_rewards", "student_termination", "domain_rand", "asset", "commands", "normalization", "dataset"):
            if checkpoint["train_cfg"].get("environment", {}).get(key) != self.train_cfg.get("environment", {}).get(key):
                raise ValueError("Student resume environment configuration differs: " + key)
        expert = self.alg.motion_buffer.identity if self.alg.motion_buffer else None
        if checkpoint["expert"] is not None:
            # Allow moving a checkout; source hashes, weights and timing must match.
            saved_expert = {k: v for k, v in checkpoint["expert"].items() if k != "config"}
            if expert is None or saved_expert != {k: v for k, v in expert.items() if k != "config"}:
                raise ValueError("Student resume expert identity differs")
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if self.alg.amp is not None:
            self.alg.amp.load_state_dict(checkpoint["amp_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.alg.amp_normalizer is not None:
            for k, v in checkpoint["normalizer"].items():
                setattr(self.alg.amp_normalizer, k, v.to(self.device))
        self.collection_iteration = checkpoint["collection_iteration"]
        self.alg.optimizer_updates = checkpoint["optimizer_updates"]
        self.alg.optimizer_steps = checkpoint["optimizer_steps"]
        self.alg.learning_rate = checkpoint["learning_rate"]
        self.student_init = checkpoint["student_init"]
        for k in ("teacher_steps", "student_steps", "handoff_successes", "handoff_failures"):
            setattr(self, k, checkpoint[k])
        # Simulator state is not restored: resume begins a real Teacher episode.
        self.obs = self.critic_obs = None
        self.history.count.zero_()
        self.storage.clear()
