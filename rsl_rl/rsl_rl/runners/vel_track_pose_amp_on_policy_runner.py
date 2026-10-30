"""Frozen Teacher collection with independently optimized Student PPO + Pose AMP."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from legged_gym.mixed_handoff import STUDENT_CONTROL, load_actor
from rsl_rl.algorithms.mixed_handoff_ppo import MixedHandoffPPO
from rsl_rl.modules import ActorCritic, AMP
from rsl_rl.modules.masked_pose_amp import (add_gated_pose_reward, classify_pose_mode,
    compute_pose_tracking_gate, condition_pose_observation, pose_amp_valid,
    project_pose_amp)
from rsl_rl.storage.vel_track_pose_amp_rollout_storage import VelTrackPoseAMPRolloutStorage


TRAINING_CONTRACT = dict(actor_obs=738, critic_obs=126, actions=29, history_frames=6,
                         amp_motion_obs=54, amp_discriminator_obs=55,
                         amp_frame_features=54, amp_history=1,
                         expert_source="Pose-HOLD+CarryWith-mode-conditioned", version=1)

GAIT_REWARD_UPGRADE = dict(gait_foot_slip=-0.05, gait_swing_clearance=0.08,
                           gait_foot_scuff=-0.05, gait_air_time=0.0)
STUDENT_REWARD_RETUNE = dict(carry_lin_vel_tracking=(3.0, 2.5),
                             carry_yaw_vel_tracking=(2.5, 3.0))
GAIT_CONFIG_UPGRADE = dict(contact_threshold_N=1.0, moving_threshold=0.1,
                           slip_tolerance_m_s=0.05, horizontal_speed_scale_m_s=0.30,
                           target_height_m=0.04, scuff_height_m=0.02,
                           air_time_min_s=0.06, air_time_range_s=0.20,
                           geometry_max_points=128, geometry_fit_tolerance_m=0.0005,
                           geometry_validation_tolerance_m=0.001)
GAIT_PHYSICAL_DIAGNOSTICS = (
    "student/stance_slip_m_s",
    "student/swing_clearance_m",
    "student/low_clearance_swing_fraction",
)


def resume_environment_section_matches(name, saved, current):
    """Allow only the audited gait addition and velocity-weight retune."""
    if saved == current:
        return True
    if name == "gait":
        return saved is None and current == GAIT_CONFIG_UPGRADE
    if name != "student_rewards" or not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    saved_scales = saved.get("scales", {})
    current_scales = current.get("scales", {})
    saved_gait = {key: saved_scales[key] for key in GAIT_REWARD_UPGRADE
                  if key in saved_scales}
    current_gait = {key: current_scales.get(key) for key in GAIT_REWARD_UPGRADE}
    if ((saved_gait and saved_gait != GAIT_REWARD_UPGRADE)
            or current_gait != GAIT_REWARD_UPGRADE):
        return False
    saved_base = {key: value for key, value in saved_scales.items() if key not in GAIT_REWARD_UPGRADE}
    current_base = {key: value for key, value in current_scales.items() if key not in GAIT_REWARD_UPGRADE}
    for key, (old, new) in STUDENT_REWARD_RETUNE.items():
        if saved_base.get(key) == old and current_base.get(key) == new:
            saved_base[key] = new
    saved_other = {key: value for key, value in saved.items() if key != "scales"}
    current_other = {key: value for key, value in current.items() if key != "scales"}
    return saved_base == current_base and saved_other == current_other


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


def initialize_student(student, path, device="cpu"):
    source, _ = load_actor(path, device)
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


class VelTrackPoseAMPOnPolicyRunner:
    def __init__(self, env, train_cfg, teacher_checkpoint,
                 student_init_checkpoint=None, expert=None, log_dir=None,
                 device="cpu", initialize=True):
        self.env, self.train_cfg = env, train_cfg
        self.device, self.log_dir = torch.device(device), Path(log_dir) if log_dir else None
        self.cfg, amp_cfg = train_cfg["runner"], train_cfg["amp"]
        # Retain runtime semantics that static network dimensions cannot check.
        self.environment_signature = dict(dof_names=list(getattr(env, "dof_names", [])),
                                          dt=getattr(env, "dt", .02))
        self.detailed_diagnostics = self.cfg.get("detailed_handoff_events", False)
        for name in ("num_steps_per_env", "save_interval", "max_iterations", "max_optimizer_updates"):
            value = self.cfg.get(name)
            if value is None and name in ("max_iterations", "max_optimizer_updates"):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        self.terminal_log_interval = self.cfg.get("terminal_log_interval", 100)
        if (isinstance(self.terminal_log_interval, bool)
                or not isinstance(self.terminal_log_interval, int) or self.terminal_log_interval < 1):
            raise ValueError("terminal_log_interval must be a positive integer")
        for retired in ("amp_coef", "reward_scale", "style_weight", "gate_floor"):
            if retired in amp_cfg:
                raise ValueError(retired + " is retired; Pose AMP uses pose_style_weight and pose_gate_floor")
        if (amp_cfg.get("num_obs") != 54 or amp_cfg.get("discriminator_num_obs") != 55
                or amp_cfg.get("num_one_step_obs") != 54 or amp_cfg.get("window_length") != 1):
            raise ValueError("Pose AMP requires the 54D pose / 55D discriminator / one-frame contract")
        for name in ("pose_style_weight", "pose_gate_floor", "stand_linear_threshold", "stand_yaw_threshold"):
            value = amp_cfg.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Invalid gated AMP parameter: " + name)
        if amp_cfg["pose_style_weight"] < 0 or not 0 <= amp_cfg["pose_gate_floor"] <= 1:
            raise ValueError("pose_style_weight must be nonnegative and pose_gate_floor in [0, 1]")
        if amp_cfg["stand_linear_threshold"] < 0 or amp_cfg["stand_yaw_threshold"] < 0:
            raise ValueError("AMP Stand thresholds must be nonnegative")
        self.teacher, _ = load_actor(teacher_checkpoint, device)
        self.teacher_identity = file_identity(teacher_checkpoint)
        self.teacher_snapshot = {k: p.detach().clone() for k, p in self.teacher.state_dict().items()}
        ac = ActorCritic(738, 126, 6, 29, **train_cfg["policy"]).to(device)
        self.student_init = initialize_student(ac, student_init_checkpoint or teacher_checkpoint,
                                              device) if initialize else None
        amp = AMP(55, 0., device=device).to(device) if amp_cfg["enabled"] else None
        normalizer = None
        if amp is not None and expert is None:
            raise ValueError("Enabled AMP requires an independent HOLD/CarryWith expert motion library")
        if amp is not None and amp_cfg["use_normalizer"]:
            from rsl_rl.utils.utils import AmpNormalizer
            normalizer = AmpNormalizer(55, device=device)
        self.alg = MixedHandoffPPO(ac, amp, normalizer, expert, **train_cfg["algorithm"])
        self.storage = VelTrackPoseAMPRolloutStorage(
            env.num_envs, self.cfg["num_steps_per_env"], device=device)
        self.collection_iteration = 0
        self.teacher_steps = self.student_steps = 0
        self.amp_steps = 0
        self.handoff_successes = self.handoff_failures = 0
        self._pending_counts = torch.zeros(5, dtype=torch.long, device=self.device)
        self.writer = None
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
        self.obs = self.critic_obs = None
        self.episode_teacher_steps = torch.zeros(env.num_envs, device=self.device)
        self.episode_student_steps = torch.zeros_like(self.episode_teacher_steps)
        self.episode_amp_steps = torch.zeros_like(self.episode_teacher_steps)
        # TensorBoard-only episode returns; never enter PPO or JSON diagnostics.
        self._tb_episode_returns = torch.zeros(env.num_envs, 3, device=self.device)
        self._tb_completed = []
        self._tb_episode_info = []

    def reset(self):
        obs, critic = self.env.reset()
        self.obs, self.critic_obs = obs.to(self.device).clone(), critic.to(self.device).clone()
        self.episode_teacher_steps.zero_()
        self.episode_student_steps.zero_()
        self.episode_amp_steps.zero_()
        self._tb_episode_returns.zero_()

    @torch.no_grad()
    def collect_step(self):
        env, ac = self.env, self.alg.actor_critic
        student = env.handoff.phase.to(self.device) == STUDENT_CONTROL
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
        task_reward = task_reward.to(self.device)
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
        amp_motion = project_pose_amp(amp_state.to(self.device))
        executed_command = diag["executed_command"].to(self.device)
        stand_mode = classify_pose_mode(executed_command,
            self.train_cfg["amp"]["stand_linear_threshold"],
            self.train_cfg["amp"]["stand_yaw_threshold"])
        amp_valid = pose_amp_valid(student, dones)
        amp_obs = condition_pose_observation(amp_motion, stand_mode)
        linear_error = diag["student/linear_tracking_error"].to(self.device)
        yaw_error = diag["student/yaw_tracking_error"].to(self.device)
        gate = compute_pose_tracking_gate(linear_error, yaw_error,
            env.cfg.student_rewards.carry_lin_vel_sigma,
            env.cfg.student_rewards.carry_yaw_vel_sigma,
            self.train_cfg["amp"]["pose_gate_floor"])
        amp_raw = torch.zeros_like(task_reward)
        enabled = self.alg.amp is not None
        if enabled and amp_valid.any():
            amp_raw[amp_valid] = self.alg.amp.predict_reward(
                amp_obs[amp_valid], self.alg.amp_normalizer).squeeze(-1)
        reward, amp_bonus = add_gated_pose_reward(task_reward, amp_raw, gate, amp_valid,
            getattr(env, "dt", .02), self.train_cfg["amp"]["pose_style_weight"])
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
        trigger = diag["handoff_trigger"].to(self.device).bool()
        failed = diag["handoff_failure"].to(self.device).bool()
        self._pending_counts += torch.stack(((~student).sum(), student.sum(), amp_valid.sum(),
                                            trigger.sum(), failed.sum()))
        samples = dict(task_reward=task_reward[student], amp_reward=amp_raw[student], combined_reward=reward[student],
                       amp_reward_valid=amp_raw[amp_valid] if enabled else amp_raw[:0],
                       amp_raw_valid=amp_raw[amp_valid] if enabled else amp_raw[:0],
                       tracking_gate=gate[amp_valid] if enabled else gate[:0],
                       amp_bonus=amp_bonus[amp_valid] if enabled else amp_bonus[:0],
                       pose_raw_nonzero=(amp_raw[amp_valid] != 0).float() if enabled else amp_raw[:0],
                       pose_bonus_abs=amp_bonus[amp_valid].abs() if enabled else amp_bonus[:0],
                       task_reward_abs_pose=task_reward[amp_valid].abs() if enabled else task_reward[:0],
                       pose_stand_raw=amp_raw[amp_valid & stand_mode] if enabled else amp_raw[:0],
                       pose_moving_raw=amp_raw[amp_valid & ~stand_mode] if enabled else amp_raw[:0],
                       handoff_time=diag["handoff_time"].to(self.device)[trigger])
        self.episode_teacher_steps += ~student
        self.episode_student_steps += student
        self.episode_amp_steps += amp_valid
        dt = getattr(env, "dt", .02)
        reached = self.episode_student_steps > 0
        segments = dones & reached
        samples.update({
            "gate/teacher_steps": self.episode_teacher_steps[trigger].clone(),
            "gate/teacher_warmup_s": (self.episode_teacher_steps[trigger] * dt).clone(),
            "episode/teacher_steps": self.episode_teacher_steps[dones].clone(),
            "episode/teacher_warmup_s": (self.episode_teacher_steps[dones] * dt).clone(),
            "episode/zero_student_fraction": (~reached[dones]).float(),
            # Include completed Teacher-only episodes with ratio zero. Counters
            # persist across rollouts and are cleared only for asynchronous dones.
            "episode_student_fraction": (self.episode_student_steps[dones] /
                (self.episode_teacher_steps[dones] + self.episode_student_steps[dones]).clamp_min(1)),
            "student/segment_frames": self.episode_student_steps[segments].clone(),
            "student/segment_duration_s": self.episode_student_steps[segments] * dt,
            "student/episode_reached_amp_fraction": (self.episode_amp_steps[segments] > 0).float(),
            # Timeout is a right-censored survival observation, even though its
            # episode has ended. Physical failures are observed survival events.
            "student/completed_survival_s": (self.episode_student_steps[segments & failures] * dt).clone(),
            "student/censored_timeout_s": (self.episode_student_steps[segments & timeouts & ~failures] * dt).clone(),
            "termination/teacher_before_handoff": failed[dones].float(),
            "termination/teacher_physical_failure": (failures & ~student)[dones].float(),
            "termination/episode_timeout": timeouts[dones].float(),
        })
        if self.writer:
            self._tb_episode_returns += torch.where(
                student[:, None], torch.stack((reward, task_reward, amp_bonus), dim=1), 0.)
            self._tb_completed.append(torch.cat((self._tb_episode_returns[segments],
                self.episode_student_steps[segments, None], self.episode_amp_steps[segments, None]), dim=1))
            self._tb_episode_returns[dones] = 0
            if "episode" in infos:
                self._tb_episode_info.append(infos["episode"])
        self.episode_teacher_steps[dones] = 0
        self.episode_student_steps[dones] = 0
        self.episode_amp_steps[dones] = 0
        # Sparse physical snapshots preserve vectors and actual actions, without
        # averaging Teacher placeholders into first-Student discontinuities.
        first = diag.get("first_student_step", torch.zeros_like(dones)).to(self.device).bool()
        physical = ("box_height", "box_clearance", "hand_contact", "hand_force", "hand_surface_distance",
                    "base_linear_velocity", "base_angular_velocity", "box_linear_velocity", "box_angular_velocity",
                    "box_tilt", "teacher_last_action", "student_first_action", "action_discontinuity",
                    "action_discontinuity_l2")
        for key in physical if self.detailed_diagnostics else ():
            if key in diag:
                mask = first if key.startswith(("student_first", "action_discontinuity")) else trigger
                samples["gate/" + key] = diag[key].to(self.device)[mask].float().flatten()
        if self.log_dir and self.detailed_diagnostics:
            event_ids = (trigger | first | dones).nonzero().flatten()
            if event_ids.numel():
                rows = {k: v.detach().index_select(0, event_ids.to(v.device)).cpu().tolist()
                        for k, v in diag.items()}
                events = event_ids.cpu().tolist()
                first_rows = first[event_ids].cpu().tolist()
                with (self.log_dir / "handoff_events.jsonl").open("a", encoding="utf-8") as stream:
                    for row, index in enumerate(events):
                        event = {k: values[row] for k, values in rows.items()}
                        event.update(collection_iteration=self.collection_iteration, env_id=index)
                        if not first_rows[row]:
                            for key in ("student_first_action", "action_discontinuity", "action_discontinuity_l2"):
                                event.pop(key, None)
                        stream.write(json.dumps(event) + "\n")
        for k, v in diag.items():
            if k.startswith(("student/", "student_reward_raw/", "student_reward_weighted/")):
                values = v.to(self.device)[student].float().flatten()
                if k in GAIT_PHYSICAL_DIAGNOSTICS:
                    values = values[torch.isfinite(values)]
                samples[k] = values
        for key, value in infos.get("episode", {}).items():
            if key.startswith("student/"):
                samples["episode/" + key] = torch.as_tensor(value, device=self.device).detach().reshape(-1)
        reasons = diag["student/termination_reason"].to(self.device).long()
        for bit, name in ((1, "box_drop"), (2, "separation"), (4, "grasp_loss"), (8, "tilt"), (16, "physical"), (32, "timeout")):
            samples["termination/" + name] = ((reasons[student & dones] & bit) != 0).float()
        return samples

    def _flush_counts(self):
        counts = self._pending_counts.cpu().tolist()
        for name, count in zip(("teacher_steps", "student_steps", "amp_steps",
                                "handoff_successes", "handoff_failures"), counts):
            setattr(self, name, getattr(self, name) + count)
        self._pending_counts.zero_()

    def assert_teacher_frozen(self, check_weights=True):
        if self.teacher.training or any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError("Teacher is no longer frozen")
        if check_weights and any(not torch.equal(p, self.teacher_snapshot[k]) for k, p in self.teacher.state_dict().items()):
            raise RuntimeError("Frozen Teacher weights changed")

    def _synchronize_timing(self):
        # perf_counter alone measures CUDA dispatch, not completed GPU work.
        devices = {torch.device(self.device), torch.device(self.env.device)}
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def log(self, metrics, total_iterations, elapsed_time, eta, final=False, width=80, pad=35):
        """Compact native PhysHSI panel; detailed metrics stay in file/TensorBoard."""
        def number(value, spec=".4f", suffix=""):
            if value is None or not math.isfinite(value):
                return "N/A"
            return f"{value:{spec}}{suffix}"

        def row(label, value):
            return f"{label + ':':>{pad}} {value}"

        title = f" Learning/Collection iteration {self.collection_iteration}/{total_iterations}"
        title += " (final) " if final else " "
        lines = ["#" * width, title.center(width), "",
                 row("Computation", number(metrics.get("Perf/total_fps"), ".0f", " steps/s")),
                 row("Collection time", number(metrics.get("Perf/collection_time"), ".3f", "s")),
                 row("Learning time", number(metrics.get("Perf/learning_time"), ".3f", "s")),
                 row("Value function loss", number(metrics.get("value"))),
                 row("Surrogate loss", number(metrics.get("surrogate"))),
                 row("Mean action noise std", number(metrics.get("Policy/mean_noise_std"), ".2f")),
                 row("Learning rate", number(metrics.get("learning_rate"), ".2e")),
                 row("PPO KL", number(metrics.get("kl"), ".6f")),
                 row("Mean Student task reward", number(metrics.get("performance/task_reward"))),
                 row("Mean raw Pose AMP reward", number(metrics.get("performance/amp_raw_valid"))),
                 row("Mean Pose AMP gate", number(metrics.get("performance/tracking_gate"))),
                 row("Mean Pose AMP bonus", number(metrics.get("performance/amp_bonus"))),
                 row("Student-valid ratio", number(metrics.get("student_valid_ratio"), ".2%")),
                 row("Cumulative Student ratio", number(metrics.get("student_step_fraction"), ".2%")),
                 row("Completed episode Student ratio", number(metrics.get("episode_student_fraction")
                     if metrics.get("episode_student_fraction_count", 0) else None, ".2%")),
                 row("AMP-valid fraction", number(metrics.get("amp_valid_fraction")
                                                 if metrics.get("student_valid_count", 0) else None, ".2%")),
                 row("Handoff success rate", number(metrics.get("handoff_success_rate")
                                                  if self.handoff_successes + self.handoff_failures else None, ".2%")),
                 row("Mean completed Student survival", number(metrics.get("performance/student/completed_survival_s"), ".3f", "s")),
                 row("Actual PPO optimizer updates", str(self.alg.optimizer_updates)),
                 "-" * width,
                 row("Total timesteps", str(self.teacher_steps + self.student_steps)),
                 row("Total time (this run)", number(elapsed_time, ".2f", "s")),
                 row("ETA (collection limit)", number(eta, ".1f", "s"))]
        print("\n".join(lines), flush=True)

    def learn(self, iterations, optimize=True, max_optimizer_updates=None):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if max_optimizer_updates is not None and (isinstance(max_optimizer_updates, bool)
                or not isinstance(max_optimizer_updates, int) or max_optimizer_updates < 1):
            raise ValueError("max_optimizer_updates must be None or a positive integer")
        if self.obs is None:
            self.reset()
        start_updates = self.alg.optimizer_updates
        total_iterations = self.collection_iteration + iterations
        run_start = time.perf_counter()
        try:
            for completed in range(1, iterations + 1):
                self._synchronize_timing()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                collection_start = time.perf_counter()
                stats = {}
                distributions = {}
                self._tb_completed.clear()
                self._tb_episode_info.clear()
                for _ in range(self.cfg["num_steps_per_env"]):
                    for k, v in self.collect_step().items():
                        if v.numel():
                            # Float64 accumulation matches the former Python
                            # double accumulator; reductions remain float32.
                            reduced = torch.stack((v.sum(), v.square().sum())).double()
                            if k not in stats:
                                stats[k] = [reduced, v.numel()]
                            else:
                                stats[k][0] += reduced
                                stats[k][1] += v.numel()
                            if k in ("handoff_time", "episode/teacher_steps", "episode/teacher_warmup_s",
                                     "student/segment_frames", "student/segment_duration_s",
                                     "student/completed_survival_s", "student/censored_timeout_s"):
                                distributions.setdefault(k, []).append(v.detach().float())
                self._synchronize_timing()
                learning_start = time.perf_counter()
                collection_time = learning_start - collection_start
                self.storage.compute_returns(self.alg.gamma, self.alg.lam)
                if optimize:
                    metrics = self.alg.update(self.storage)
                else:
                    count = int(self.storage.data["student_valid"][:self.storage.step].sum())
                    amp_count = int(self.storage.data["amp_valid"][:self.storage.step].sum())
                    self.storage.clear()
                    metrics = dict(student_valid_count=count, amp_valid_count=amp_count, updated=0)
                self._synchronize_timing()
                learning_time = time.perf_counter() - learning_start
                self.assert_teacher_frozen(check_weights=self.detailed_diagnostics)
                self._flush_counts()
                self.collection_iteration += 1
                count, amp_count = metrics["student_valid_count"], metrics["amp_valid_count"]
                attempts = self.handoff_successes + self.handoff_failures
                metrics.update(collection_iterations=self.collection_iteration, optimizer_updates=self.alg.optimizer_updates,
                               optimizer_steps=self.alg.optimizer_steps, teacher_steps=self.teacher_steps, student_steps=self.student_steps,
                               amp_valid_steps=self.amp_steps,
                               student_step_fraction=self.student_steps / max(1, self.teacher_steps+self.student_steps),
                               student_valid_ratio=count / (self.env.num_envs * self.cfg["num_steps_per_env"]),
                               amp_valid_fraction=amp_count / max(1, count),
                               cumulative_amp_valid_fraction=self.amp_steps / max(1, self.student_steps),
                               handoff_success_rate=self.handoff_successes / max(1, attempts),
                               handoff_failure_rate=self.handoff_failures / max(1, attempts))
                active = self.episode_student_steps > 0
                metrics.update(student_incomplete_segments=int(active.sum()),
                               student_incomplete_mean_frames=float(self.episode_student_steps[active].mean()) if active.any() else 0.,
                               teacher_incomplete_episodes=int((self.env.handoff.phase != STUDENT_CONTROL).sum()),
                               pending_first_student_segments=int(((self.env.handoff.phase == STUDENT_CONTROL) & ~active).sum()))
                reduced_stats = torch.stack([s[0] for s in stats.values()]).cpu().tolist() if stats else []
                for (k, (_, number)), (total, square) in zip(stats.items(), reduced_stats):
                    metrics["performance/" + k] = total / number
                    if "reward" in k:
                        metrics["performance/" + k + "_std"] = max(0., square / number - (total / number)**2)**.5
                pose_task = metrics.get("performance/task_reward_abs_pose")
                metrics["pose_amp_bonus_to_task_reward"] = (
                    metrics.get("performance/pose_bonus_abs", 0.) / max(pose_task, 1e-12)
                    if pose_task is not None else 0.)
                # These means/counts cover episodes/segments ending in THIS
                # rollout, including portions collected in earlier rollouts.
                metrics["episode_student_fraction"] = metrics.get("performance/episode_student_fraction", 0.)
                metrics["episode_student_fraction_count"] = stats.get("episode_student_fraction", (None, 0))[1]
                for key in ("student/segment_duration_s", "student/completed_survival_s", "student/censored_timeout_s"):
                    metrics[key + "/count"] = stats.get(key, (None, 0))[1]
                    if key in stats:
                        metrics[key] = metrics["performance/" + key]
                for k, batches in distributions.items():
                    values = torch.cat(batches).cpu()
                    metrics["distribution/" + k + "/count"] = values.numel()
                    for label, q in (("min", 0.), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1.)):
                        metrics["distribution/" + k + "/" + label] = float(torch.quantile(values, q))
                elapsed_time = time.perf_counter() - run_start
                reached_limit = (max_optimizer_updates is not None
                                 and self.alg.optimizer_updates-start_updates >= max_optimizer_updates)
                final = completed == iterations or reached_limit
                eta = 0. if final else elapsed_time / completed * (iterations - completed)
                metrics.update({"Perf/collection_time": collection_time, "Perf/learning_time": learning_time,
                                "Perf/gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(self.device)
                                if self.device.type == "cuda" else 0,
                                "Perf/total_fps": self.env.num_envs * self.cfg["num_steps_per_env"] /
                                                  max(collection_time + learning_time, 1e-9),
                                "Policy/mean_noise_std": self.alg.actor_critic.std.detach().mean().item(),
                                "learning_rate": self.alg.learning_rate})
                if self.writer:
                    # Fixed native-style whitelist. Full diagnostics stay in JSONL.
                    scalars = {
                        "Loss/value_function": metrics.get("value"),
                        "Loss/surrogate": metrics.get("surrogate"),
                        "Loss/learning_rate": metrics["learning_rate"],
                        "Loss/amp_expert_loss": metrics.get("amp_expert_loss"),
                        "Loss/amp_policy_loss": metrics.get("amp_policy_loss"),
                        "Train/student_ratio": metrics["student_valid_ratio"],
                        "Train/handoff_success_rate": metrics["handoff_success_rate"] if attempts else None,
                        # Observed failure endpoints only; censored timeouts stay in JSONL.
                        "Train/student_survival_s": metrics.get("student/completed_survival_s"),
                        "Policy/mean_noise_std": metrics["Policy/mean_noise_std"],
                        "Perf/total_fps": metrics["Perf/total_fps"],
                        "Perf/collection_time": metrics["Perf/collection_time"],
                        "Perf/learning_time": metrics["Perf/learning_time"],
                        "Train/mean_pose_amp_raw": metrics.get("performance/amp_raw_valid"),
                        "Train/mean_pose_amp_gate": metrics.get("performance/tracking_gate"),
                        "Train/mean_pose_amp_bonus": metrics.get("performance/amp_bonus"),
                        "Train/pose_amp_bonus_to_task_reward": metrics.get("pose_amp_bonus_to_task_reward"),
                        "Train/pose_amp_nonzero_fraction": metrics.get("performance/pose_raw_nonzero"),
                        "Train/mean_pose_amp_raw_stand": metrics.get("performance/pose_stand_raw"),
                        "Train/mean_pose_amp_raw_moving": metrics.get("performance/pose_moving_raw"),
                        "student/gait_foot_slip": metrics.get("performance/student/gait_foot_slip"),
                        "student/gait_swing_clearance": metrics.get("performance/student/gait_swing_clearance"),
                        "student/gait_foot_scuff": metrics.get("performance/student/gait_foot_scuff"),
                        "student/stance_slip_m_s": metrics.get("performance/student/stance_slip_m_s"),
                        "student/swing_clearance_m": metrics.get("performance/student/swing_clearance_m"),
                        "student/low_clearance_swing_fraction": metrics.get(
                            "performance/student/low_clearance_swing_fraction"),
                    }
                    if "amp_gradient_penalty" in metrics:
                        scalars["Loss/amp_loss"] = (metrics["amp_expert_loss"] +
                            metrics["amp_policy_loss"] + metrics["amp_gradient_penalty"])
                    ended = torch.cat(self._tb_completed)
                    if ended.numel():
                        # Same completed Student cohort, including timeouts, across
                        # rollout boundaries. Length is Student frames, not Teacher warmup.
                        combined, raw, amp, length, amp_frames = ended.mean(0).cpu().tolist()
                        scalars.update({"Train/mean_reward": combined, "Train/mean_raw_reward": raw,
                                        "Train/mean_episode_length": length})
                        # Every Student-owned frame is eligible. Disabled Pose AMP
                        # or no Student segment means there is no bonus curve.
                        if self.alg.amp is not None and amp_frames > 0:
                            scalars["Train/mean_pose_amp_reward"] = amp
                    for name in ("carry_lin_vel_tracking", "carry_yaw_vel_tracking", "carry_bilateral_contact",
                                 "lin_vel_z", "ang_vel_xy", "orientation", "dof_acc", "action_rate", "torques",
                                 "dof_vel", "dof_pos_limits", "dof_vel_limits", "torque_limits"):
                        key = "student/mean_step_reward/" + name
                        # Weight reset-batch means by completed Student episode count;
                        # exclude Teacher-only episodes. Match Policy1's per-second rew_*.
                        entries = [torch.stack((torch.as_tensor(info[key], device=self.device),
                                    torch.as_tensor(info["student/episodes_with_student"], device=self.device)))
                                   for info in self._tb_episode_info
                                   if key in info and "student/episodes_with_student" in info]
                        if entries:
                            entries = torch.stack(entries)
                            total, episodes = torch.stack(((entries[:, 0] * entries[:, 1]).sum(),
                                                            entries[:, 1].sum())).cpu().tolist()
                            if episodes > 0:
                                scalars["Episode/rew_" + name] = total / episodes / getattr(self.env, "dt", .02)
                    for tag, value in scalars.items():
                        if value is not None and math.isfinite(value):
                            self.writer.add_scalar(tag, value, self.collection_iteration)
                    self.writer.flush()
                if self.log_dir:
                    with (self.log_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metrics, sort_keys=True) + "\n")
                if self.collection_iteration % self.terminal_log_interval == 0 or final:
                    self.log(metrics, total_iterations, elapsed_time, eta, final=final)
                if self.log_dir and self.collection_iteration % self.cfg["save_interval"] == 0:
                    self.save(self.log_dir / f"model_{self.collection_iteration}.pt")
                if reached_limit:
                    break
            self.assert_teacher_frozen()
            if self.log_dir and self.collection_iteration % self.cfg["save_interval"] != 0:
                self.save(self.log_dir / f"model_{self.collection_iteration}.pt")
        finally:
            if self.writer:
                self.writer.close()

    def save(self, path):
        self._flush_counts()
        normalizer = self.alg.amp_normalizer
        torch.save(dict(format="vel_track_pose_amp_student_v1", environment_signature=self.environment_signature,
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
                        amp_steps=self.amp_steps,
                        handoff_successes=self.handoff_successes, handoff_failures=self.handoff_failures), path)

    def load(self, path):
        checkpoint = read_checkpoint(path)
        if checkpoint.get("format") != "vel_track_pose_amp_student_v1":
            raise ValueError("Full Student resume requires a vel_track_pose_amp_student_v1 checkpoint; use --student_init_checkpoint for Actor/std weights")
        if checkpoint["training_contract"] != TRAINING_CONTRACT:
            raise ValueError("Student resume observation/AMP contracts differ")
        saved_signature = checkpoint.get("environment_signature")
        if saved_signature is not None and saved_signature != self.environment_signature:
            raise ValueError("Student resume runtime environment signature differs")
        # Old Student exports may carry these runtime checks in scope1_contract.
        # Its synthetic feature descriptions are no longer a loading requirement.
        legacy = checkpoint.get("scope1_contract")
        if saved_signature is None and isinstance(legacy, dict):
            for key, current in self.environment_signature.items():
                if key in legacy and legacy[key] != current:
                    raise ValueError("Student resume runtime environment differs: " + key)
            angles = self.train_cfg.get("environment", {}).get("init_state", {}).get("default_joint_angles")
            if "default_joint_angles" in legacy and legacy["default_joint_angles"] != angles:
                raise ValueError("Student resume default joint angles differ")
        if checkpoint["teacher"]["sha256"] != self.teacher_identity["sha256"]:
            raise ValueError("Student resume frozen Teacher identity differs")
        for key in ("algorithm", "policy", "amp"):
            saved_cfg, current_cfg = checkpoint["train_cfg"][key], self.train_cfg[key]
            if key == "amp":
                saved_cfg = {k: v for k, v in saved_cfg.items() if k != "expert_config"}
                current_cfg = {k: v for k, v in current_cfg.items() if k != "expert_config"}
            if saved_cfg != current_cfg:
                raise ValueError("Student resume configuration differs: " + key)
        for key in ("control", "sim", "init_state", "handoff", "student_rewards", "gait",
                    "student_termination", "domain_rand", "asset", "commands", "normalization", "dataset"):
            saved = checkpoint["train_cfg"].get("environment", {}).get(key)
            current = self.train_cfg.get("environment", {}).get(key)
            if not resume_environment_section_matches(key, saved, current):
                raise ValueError("Student resume environment configuration differs: " + key)
        # Reset-window semantics belong to the environment contract; printing
        # reset diagnostics does not. Older exports have no training RSI field.
        def rsi_settings(config):
            settings = config.get("environment", {}).get("pickup_rsi", {})
            return {k: v for k, v in settings.items() if k != "log_resets"} if settings.get("enabled", False) else {}
        if rsi_settings(checkpoint["train_cfg"]) != rsi_settings(self.train_cfg):
            raise ValueError("Student resume environment configuration differs: pickup_rsi")
        expert = self.alg.motion_buffer.identity if self.alg.motion_buffer else None
        if (checkpoint["expert"] is None) != (expert is None):
            raise ValueError("Student resume expert identity differs")
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
        self.amp_steps = checkpoint.get("amp_steps", 0)
        self._pending_counts.zero_()
        # Simulator state is not restored: resume begins a real Teacher episode.
        self.obs = self.critic_obs = None
        self.storage.clear()
