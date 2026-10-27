"""CPU numerical tests of the real masked storage, PPO, AMP and runner.

Only simulator imports are adapted. ActorCritic, AMP, their losses, Adam and
MotionLib execute normally.
"""
import ast
import argparse
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "legged_gym"))
from test_mixed_handoff import MATH, classes_from_source, config_namespace
from legged_gym.mixed_handoff import load_actor
from rsl_rl.modules import ActorCritic, AMP
from rsl_rl.modules.masked_amp import project_amp, StudentAMPHistory
from rsl_rl.algorithms.mixed_handoff_ppo import MixedHandoffPPO

from rsl_rl.storage.mixed_handoff_rollout_storage import MixedHandoffRolloutStorage
from rsl_rl.runners.mixed_handoff_on_policy_runner import MixedHandoffOnPolicyRunner, initialize_student
from rsl_rl.utils.utils import AmpNormalizer


def motion_classes():
    package = ModuleType("scope3_motion")
    package.__path__ = [str(ROOT / "legged_gym/legged_gym/envs/motionlib")]
    utils = ModuleType("legged_gym.utils")
    utils.torch_utils = MATH
    with patch.dict(sys.modules, {"scope3_motion": package, "legged_gym.utils": utils}):
        from scope3_motion.motionlib_carrywith import CarryWithMotionLib, validate_carrywith_config
    return CarryWithMotionLib, validate_carrywith_config


CarryWithMotionLib, validate_carrywith_config = motion_classes()
CONTRACT = dict(actor_obs=738, critic_obs=126, actions=29, history_frames=6, activation="elu")


def actor_critic():
    return ActorCritic(738, 126, 6, 29, actor_hidden_dims=[512, 256, 256],
                       critic_hidden_dims=[32, 16], activation="elu", init_noise_std=.4)


def configuration(amp=True, steps=14, normalize=False):
    return dict(policy=dict(actor_hidden_dims=[512, 256, 256], critic_hidden_dims=[32, 16],
                            activation="elu", init_noise_std=.4),
                algorithm=dict(num_learning_epochs=1, num_mini_batches=2, schedule="fixed", learning_rate=2e-4),
                runner=dict(num_steps_per_env=steps, save_interval=1),
                amp=dict(enabled=amp, amp_coef=.25, reward_scale=.5, use_normalizer=normalize))


class Expert:
    identity = dict(source="CarryWith-only", files=[dict(name="synthetic", sha256="test")], weights=[5., 2., 5.], fps=60., control_dt=.02)

    def get_expert_obs(self, batch_size):
        return torch.randn(batch_size, 540) * .1


class TensorEnvironment:
    """Two asynchronous Teacher->Student episodes; terminal obs differs from reset."""
    num_envs, device = 2, "cpu"
    cfg = NS(normalization=NS(clip_observations=100.))

    def __init__(self):
        self.handoff = NS(phase=torch.zeros(2, dtype=torch.long))
        self.t = torch.zeros(2, dtype=torch.long)
        self.amp = torch.zeros(2, 600)

    def reset(self):
        self.handoff.phase.zero_()
        self.t.zero_()
        self.amp.zero_()
        return torch.zeros(2, 738), torch.zeros(2, 126)

    def step(self, actions):
        owned = self.handoff.phase.clone()
        self.t += 1
        triggered = (owned == 0) & (self.t >= torch.tensor([2, 4]))
        self.handoff.phase[triggered] = 1
        dones = self.t >= torch.tensor([16, 18])
        failure = dones & torch.tensor([True, False])
        timeout = dones & ~failure
        obs = torch.randn(2, 738) * .05
        critic = torch.randn(2, 126) * .05
        terminal_ids = dones.nonzero().flatten()
        terminal_critic = torch.full((len(terminal_ids), 126), 7.)
        obs[dones], critic[dones] = -9., -9.
        self.amp = torch.cat((self.amp[:, 60:], torch.ones(2, 60) * self.t[:, None]), -1)
        amp = self.amp.clone()
        diag = dict(executed_phase=owned, timeout=timeout, failure=failure,
                    handoff_trigger=triggered, handoff_failure=dones & (owned == 0),
                    handoff_time=torch.where(self.handoff.phase == 1, .04, -1.),
                    **{"student/termination_reason": failure.long()*16 + timeout.long()*32,
                       "student/linear_error": torch.tensor([.1, .2]),
                       "student_reward_weighted/carry_box_tilt": torch.tensor([-.01, -.02])})
        self.t[dones] = 0
        self.handoff.phase[dones] = 0
        self.amp[dones] = 0
        return obs, critic, torch.tensor([.12, .15]), dones, {"handoff": diag}, terminal_ids, terminal_critic, amp


def transition(n=2, **overrides):
    data = dict(observations=torch.randn(n, 738)*.1, next_observations=torch.randn(n, 738)*.1,
                critic_observations=torch.randn(n, 126)*.1, next_critic_observations=torch.randn(n, 126)*.1,
                actions=torch.randn(n, 29)*.1, mu=torch.zeros(n, 29), sigma=torch.ones(n, 29),
                amp_observations=torch.randn(n, 540)*.1, values=torch.zeros(n), next_values=torch.zeros(n),
                rewards=torch.ones(n), log_prob=torch.zeros(n), student_valid=torch.ones(n, dtype=torch.bool),
                amp_valid=torch.zeros(n, dtype=torch.bool), dones=torch.zeros(n, dtype=torch.bool),
                timeouts=torch.zeros(n, dtype=torch.bool), failures=torch.zeros(n, dtype=torch.bool),
                next_student=torch.ones(n, dtype=torch.bool))
    data.update(overrides)
    return data


def populated_storage(ac, amp_valid=True):
    storage = MixedHandoffRolloutStorage(2, 3)
    with torch.no_grad():
        for t in range(3):
            tr = transition(student_valid=torch.tensor([True, t > 0]), amp_valid=torch.tensor([amp_valid, amp_valid and t > 0]))
            ac.update_distribution(tr["observations"])
            tr["actions"] = ac.distribution.sample()
            tr["mu"], tr["sigma"] = ac.action_mean, ac.action_std
            tr["log_prob"] = ac.get_actions_log_prob(tr["actions"])
            tr["values"] = ac.evaluate(tr["critic_observations"]).squeeze(-1)
            tr["next_values"] = ac.evaluate(tr["next_critic_observations"]).squeeze(-1)
            storage.add(**tr)
    storage.compute_returns(.99, .95)
    return storage


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "teacher.pt"
        self.source = actor_critic()
        with torch.no_grad():
            self.source.std.fill_(.31)
            for p in self.source.critic.parameters():
                p.fill_(9.)
        torch.save(dict(model_state_dict=self.source.state_dict(), scope1_contract=CONTRACT), self.path)

    def tearDown(self):
        self.temp.cleanup()
        self.output.__exit__(None, None, None)

    def runner(self, amp=True, steps=14, normalize=False, initialize=True):
        return MixedHandoffOnPolicyRunner(TensorEnvironment(), configuration(amp, steps, normalize),
                                          self.path, CONTRACT, expert=Expert() if amp else None, initialize=initialize)

    def test_initialization_storage_critic_std_and_optimizer_isolation(self):
        runner = self.runner()
        student = runner.alg.actor_critic
        for teacher, actor in zip(runner.teacher.parameters(), student.actor.parameters()):
            self.assertTrue(torch.equal(teacher, actor))
            self.assertNotEqual(teacher.data_ptr(), actor.data_ptr())
        self.assertTrue(torch.equal(student.std, self.source.std))
        self.assertFalse(torch.equal(student.critic[0].weight, self.source.critic[0].weight))
        optimized = {id(p) for g in runner.alg.optimizer.param_groups for p in g["params"]}
        self.assertFalse(optimized & {id(p) for p in runner.teacher.parameters()})
        self.assertEqual(optimized, {id(p) for p in student.parameters()} | {id(p) for p in runner.alg.amp.parameters()})
        runner.assert_teacher_frozen()

    def test_missing_or_incompatible_std_uses_configured_std_and_bad_std_rejected(self):
        checkpoint = torch.load(self.path, weights_only=False)
        checkpoint["model_state_dict"].pop("std")
        torch.save(checkpoint, self.path)
        ac = actor_critic()
        metadata = initialize_student(ac, self.path, CONTRACT)
        self.assertTrue(torch.equal(ac.std, torch.full((29,), .4)))
        self.assertEqual(metadata["action_std"], "configured init_noise_std")
        checkpoint["model_state_dict"]["std"] = torch.full((29,), -1.)
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "positive"):
            initialize_student(ac, self.path, CONTRACT)

    def test_valid_ppo_and_amp_change_student_and_discriminator_only(self):
        runner = self.runner()
        ac, amp = runner.alg.actor_critic, runner.alg.amp
        before_actor = [p.clone() for p in ac.actor.parameters()]
        before_amp = [p.clone() for p in amp.parameters()]
        metrics = runner.alg.update(populated_storage(ac))
        self.assertEqual(metrics["student_valid_count"], 5)
        self.assertEqual(metrics["amp_valid_count"], 5)
        self.assertGreater(metrics["actor_gradient_norm"], 0.)
        self.assertGreater(metrics["critic_gradient_norm"], 0.)
        self.assertGreater(metrics["amp_gradient_norm"], 0.)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, ac.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_amp, amp.parameters())))
        runner.assert_teacher_frozen()
        self.assertEqual(runner.alg.optimizer_updates, 1)
        self.assertEqual(runner.alg.optimizer_steps, 2)

    def test_student_update_without_amp_valid_does_not_touch_discriminator(self):
        runner = self.runner()
        before = copy.deepcopy(runner.alg.amp.state_dict())
        with patch.object(runner.alg.amp, "compute_loss", side_effect=AssertionError("Invalid AMP batch")):
            metrics = runner.alg.update(populated_storage(runner.alg.actor_critic, amp_valid=False))
        self.assertEqual(metrics["amp_valid_count"], 0)
        self.assertEqual(metrics["updated"], 1)
        self.assertTrue(all(torch.equal(p, before[k]) for k, p in runner.alg.amp.state_dict().items()))

    def test_all_teacher_skips_adam_and_leaves_every_parameter_unchanged(self):
        runner = self.runner()
        storage = MixedHandoffRolloutStorage(2, 1)
        storage.add(**transition(student_valid=torch.zeros(2, dtype=torch.bool), rewards=torch.tensor([1e9, -1e9])))
        storage.compute_returns(.99, .95)
        before = copy.deepcopy(runner.alg.actor_critic.state_dict())
        with patch.object(runner.alg.optimizer, "step", side_effect=AssertionError("Empty update")):
            metrics = runner.alg.update(storage)
        self.assertEqual(metrics["updated"], 0)
        self.assertEqual(runner.alg.optimizer_updates, 0)
        self.assertTrue(all(p.grad is None for p in runner.alg.actor_critic.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in runner.alg.actor_critic.state_dict().items()))

    def test_teacher_rewards_actions_values_and_observations_cannot_change_ppo(self):
        ac = actor_critic()
        alg = MixedHandoffPPO(ac)
        baseline = populated_storage(ac)
        altered = copy.deepcopy(baseline)
        invalid = ~altered.data["student_valid"]
        for k in ("rewards", "actions", "values", "log_prob", "mu", "sigma", "observations", "critic_observations"):
            altered.data[k][invalid] = float("nan")
        altered.compute_returns(.99, .95)
        valid = baseline.data["student_valid"]
        torch.testing.assert_close(baseline.data["returns"][valid], altered.data["returns"][valid])
        b = next(baseline.batches(1, 1))
        torch.manual_seed(3)
        a = next(altered.batches(1, 1))
        # Sort for exact same random smoothness weights and sample order.
        ids_b = b["observations"][:, 0].argsort()
        ids_a = a["observations"][:, 0].argsort()
        b, a = {k: v[ids_b] for k, v in b.items()}, {k: v[ids_a] for k, v in a.items()}
        torch.manual_seed(4)
        loss1 = alg.ppo_loss(b)[0]
        torch.manual_seed(4)
        loss2 = alg.ppo_loss(a)[0]
        torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)

    def test_hand_calculated_gae_failure_timeout_reset_and_rollout_boundary(self):
        storage = MixedHandoffRolloutStorage(2, 5)
        # env0: Teacher -> Student -> Student failure -> Teacher -> Student.
        # env1: Student timeout -> Teacher -> Student continuing to rollout end.
        valid = [[0, 1], [1, 0], [1, 1], [0, 1], [1, 1]]
        for t in range(5):
            done = torch.tensor([t == 2, t == 0])
            timeout = torch.tensor([False, t == 0])
            storage.add(**transition(student_valid=torch.tensor(valid[t], dtype=torch.bool),
                                      rewards=torch.tensor([1., 2.]), values=torch.tensor([.5, 1.]),
                                      next_values=torch.tensor([4., 5.]), dones=done, timeouts=timeout,
                                      failures=done & ~timeout,
                                      next_student=torch.tensor([t in (0, 1, 3, 4), t != 0])))
        storage.compute_returns(.9, .8)
        # env0: t2 delta=.5, t1 delta=4.1 + .72*.5; t4 delta=4.1.
        # env1: t0 delta=5.5 (timeout); t4=5.5,t3=9.46,t2=12.3112.
        expected = torch.tensor([[0., 6.5], [4.96, 0.], [1., 13.3112], [0., 10.46], [4.6, 6.5]])
        torch.testing.assert_close(storage.data["returns"], expected)
        raw = torch.tensor([5.5, 4.46, .5, 12.3112, 9.46, 4.1, 5.5])
        self.assertAlmostEqual(storage.advantage_mean, raw.mean().item(), places=5)
        self.assertAlmostEqual(storage.data["advantages"][storage.data["student_valid"]].mean().item(), 0., places=6)
        self.assertAlmostEqual(storage.data["advantages"][storage.data["student_valid"]].std(unbiased=False).item(), 1., places=6)

    def test_single_advantage_is_finite_and_all_minibatches_include_last_transition(self):
        storage = MixedHandoffRolloutStorage(2, 1)
        storage.add(**transition(student_valid=torch.tensor([False, True])))
        storage.compute_returns(.99, .95)
        self.assertTrue(torch.isfinite(storage.data["advantages"]).all())
        self.assertEqual(storage.data["advantages"][0, 1].item(), 1.)
        self.assertEqual(sum(len(b["values"]) for b in storage.batches(4, 3)), 3)

    def test_failure_overlapping_timeout_never_bootstraps(self):
        storage = MixedHandoffRolloutStorage(2, 1)
        storage.add(**transition(dones=torch.tensor([True, True]), timeouts=torch.tensor([True, True]),
                                 failures=torch.tensor([True, False]), next_values=torch.tensor([100., 10.])))
        storage.compute_returns(.9, .8)
        torch.testing.assert_close(storage.data["returns"], torch.tensor([[1., 10.]]))

    def test_smoothness_never_interpolates_reset_or_teacher_pairs(self):
        ac = actor_critic()
        storage = populated_storage(ac)
        batch = next(storage.batches(1, 1))
        batch["dones"].fill_(True)
        batch["next_observations"].fill_(float("nan"))
        batch["next_critic_observations"].fill_(float("nan"))
        loss, metrics = MixedHandoffPPO(ac).ppo_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["smoothness"].item(), 0.)
        batch["dones"].fill_(False)
        batch["next_student"].fill_(False)
        self.assertTrue(torch.isfinite(MixedHandoffPPO(ac).ppo_loss(batch)[0]))

    def test_exact_amp_features_every_frame_and_velocity_invariance(self):
        original = torch.arange(1200.).reshape(2, 600)
        projected = project_amp(original).reshape(2, 10, 54)
        frames = original.reshape(2, 10, 60)
        self.assertTrue(torch.equal(projected[:, :, :48], frames[:, :, :48]))
        self.assertTrue(torch.equal(projected[:, :, 48:], frames[:, :, 54:]))
        frames[:, :, 48:54] = float("nan")
        torch.testing.assert_close(project_amp(frames.flatten(1)), projected.flatten(1))
        with self.assertRaises(ValueError):
            project_amp(torch.zeros(2, 540))

    def test_amp_history_first_nine_reset_and_cross_rollout(self):
        history = StudentAMPHistory(2, "cpu")
        self.assertFalse(history.step(torch.tensor([False, False]), torch.tensor([False, False])).any())
        for i in range(10):
            valid = history.step(torch.tensor([True, i >= 2]), torch.tensor([False, False]))
            self.assertEqual(valid.tolist(), [i == 9, False])
        self.assertEqual(history.step(torch.tensor([True, True]), torch.tensor([True, False])).tolist(), [True, False])
        self.assertEqual(history.count.tolist(), [0, 9])
        self.assertEqual(history.step(torch.tensor([False, True]), torch.tensor([False, False])).tolist(), [False, True])

    def test_real_runner_ownership_reward_blending_and_terminal_bootstrap(self):
        runner = self.runner(steps=18)
        runner.reset()
        with patch.object(runner.alg.amp, "predict_reward", return_value=torch.ones(2, 1)):
            # Predict returns variable batch sizes near asynchronous handoff.
            runner.alg.amp.predict_reward.side_effect = lambda x, _: torch.ones(len(x), 1)
            for _ in range(18):
                runner.collect_step()
        d = runner.storage.data
        self.assertEqual(d["student_valid"][:5].tolist(), [[False, False], [False, False], [True, False], [True, False], [True, True]])
        self.assertFalse(d["amp_valid"][:11, 0].any())
        self.assertTrue(d["amp_valid"][11, 0])
        self.assertFalse(d["amp_valid"][:13, 1].any())
        self.assertTrue(d["amp_valid"][13, 1])
        self.assertAlmostEqual(d["rewards"][10, 0].item(), .12, places=6)
        self.assertAlmostEqual(d["rewards"][11, 0].item(), .75*.12 + .25*.5, places=6)
        self.assertEqual(d["next_values"][15, 0].item(), 0.)  # failure
        with torch.no_grad():
            expected = runner.alg.actor_critic.evaluate(torch.full((1, 126), 7.)).item()
        self.assertAlmostEqual(d["next_values"][17, 1].item(), expected, places=6)
        self.assertFalse(d["amp_valid"][16:, 0].any())
        self.assertEqual(runner.history.count.tolist(), [0, 0])

    def test_normalizer_uses_raw_valid_samples_only(self):
        runner = self.runner(normalize=True)
        storage = populated_storage(runner.alg.actor_critic)
        storage.data["amp_observations"][~storage.data["amp_valid"]] = 1e10
        normalizer = runner.alg.amp_normalizer
        recorded = []
        update = normalizer.update
        with patch.object(normalizer, "update", side_effect=lambda x: (recorded.append(x.clone()), update(x))[1]):
            runner.alg.update(storage)
        self.assertEqual(len(recorded), 4)
        self.assertTrue(all(x.abs().max() < 2 for x in recorded))
        self.assertEqual(normalizer.mean.shape, (540,))
        self.assertTrue(torch.isfinite(normalizer.var).all())

    def test_resume_round_trip_and_existing_viewer_loader(self):
        runner = self.runner(normalize=True)
        runner.alg.update(populated_storage(runner.alg.actor_critic))
        runner.collection_iteration = 3
        saved = Path(self.temp.name) / "student.pt"
        runner.save(saved)
        actor, _ = load_actor(saved, CONTRACT)
        for a, b in zip(actor.parameters(), runner.alg.actor_critic.actor.parameters()):
            self.assertTrue(torch.equal(a, b))
        resumed = self.runner(normalize=True, initialize=False)
        resumed.load(saved)
        self.assertEqual(resumed.collection_iteration, 3)
        self.assertEqual(resumed.alg.optimizer_steps, runner.alg.optimizer_steps)
        self.assertEqual(resumed.alg.optimizer_updates, 1)
        for key, value in runner.alg.actor_critic.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.alg.actor_critic.state_dict()[key]))
        self.assertTrue(torch.equal(runner.alg.amp_normalizer.mean, resumed.alg.amp_normalizer.mean))
        self.assertTrue(torch.equal(runner.alg.actor_critic.std, resumed.alg.actor_critic.std))
        self.assertEqual(len(runner.alg.optimizer.state), len(resumed.alg.optimizer.state))
        resumed.alg.update(populated_storage(resumed.alg.actor_critic))
        resumed.assert_teacher_frozen()
        with self.assertRaisesRegex(ValueError, "Full Student resume"):
            resumed.load(self.path)

    def test_resume_rejects_teacher_expert_and_amp_contract_changes(self):
        runner = self.runner()
        path = Path(self.temp.name) / "student.pt"
        runner.save(path)
        checkpoint = torch.load(path, weights_only=False)
        for key, replacement, message in (("teacher", dict(sha256="changed"), "Teacher identity"),
                                           ("training_contract", {}, "contracts differ"),
                                           ("expert", dict(source="hold"), "expert identity")):
            changed = copy.deepcopy(checkpoint)
            changed[key] = replacement
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, message):
                runner.load(path)

    def test_bounded_collection_distinguishes_empty_collections_and_updates(self):
        runner = self.runner(amp=False, steps=2)
        runner.learn(10, max_optimizer_updates=1)
        self.assertEqual(runner.collection_iteration, 2)
        self.assertEqual(runner.alg.optimizer_updates, 1)
        self.assertEqual(runner.teacher_steps, 6)
        self.assertEqual(runner.student_steps, 2)
        runner.assert_teacher_frozen()


class ExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "expert.yaml"
        self.entries = [dict(file=f"carrywith{i}.pt", weight=w) for i, w in zip((1, 2, 3), (5., 2., 5.))]
        self.config.write_text(yaml.safe_dump(dict(motions=dict(carryWith=self.entries))))
        self.mapping = ROOT / "legged_gym/resources/config/joint_id.txt"
        self.names = [line.split()[1] for line in self.mapping.read_text().splitlines()]
        for i, n in zip((1, 2, 3), (20, 22, 25)):
            # Large, distinct base heights reveal any cross-clip interpolation.
            clip = dict(base_height=torch.arange(n).float() + 100*i,
                        base_position=torch.zeros(n, 3), base_quat=torch.tensor([0., 0., 0., 1.]).repeat(n, 1),
                        joint_position=torch.arange(29.).repeat(n, 1) * .01,
                        joint_velocity=torch.zeros(n, 29), link_position=torch.zeros(n, 6, 3),
                        box_pos_local=torch.tensor([.2, 0., .1]).repeat(n, 1))
            torch.save(clip, self.root / f"carrywith{i}.pt")

    def tearDown(self):
        self.temp.cleanup()

    def sampler(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return CarryWithMotionLib(self.config, str(self.mapping), self.names, 60, "cpu", .02, .7, **kwargs)

    def test_only_carrywith_weights_clip_boundaries_timing_and_feature_values(self):
        sampler = self.sampler()
        self.assertEqual(sampler.skills, ["carryWith"])
        self.assertEqual(sampler.motion_weights_tot.tolist(), [5., 2., 5.])
        obs = sampler.get_expert_obs(12000).reshape(-1, 10, 54)
        clip = (obs[:, :, 0] / 100).floor()
        self.assertTrue((clip == clip[:, :1]).all())
        probability = torch.bincount(clip[:, 0].long(), minlength=4)[1:].float() / len(obs)
        torch.testing.assert_close(probability, torch.tensor([5/12, 2/12, 5/12]), atol=.015, rtol=0)
        torch.testing.assert_close(obs[:, 1:, 0]-obs[:, :-1, 0], torch.full((len(obs), 9), 1.2), atol=5e-5, rtol=0)
        torch.testing.assert_close(obs[0, :, 1:30], torch.arange(29.).repeat(10, 1)*.01)
        torch.testing.assert_close(obs[0, :, 45:48], torch.tensor([.2, 0., .1]).repeat(10, 1))
        self.assertEqual(obs.flatten(1).shape, (12000, 540))

    def test_tracked_carrywith_dataset_loads_from_default_yaml_without_override(self):
        config = ROOT / "legged_gym/resources/config/carrybox_carrywith_only.yaml"
        _, _, files = validate_carrywith_config(config)
        expected = ROOT / "legged_gym/resources/dataset/dataset_carrybox/carryWith"
        self.assertEqual(files, [(expected / f"carrywith{i}.pt").resolve() for i in (1, 2, 3)])
        with contextlib.redirect_stdout(io.StringIO()):
            sampler = CarryWithMotionLib(config, str(self.mapping), self.names, 60, "cpu", .02, .7)
        self.assertEqual(sampler.motion_len_tot.tolist(), [256, 194, 323])
        self.assertEqual(sampler.motion_weights_tot.tolist(), [5., 2., 5.])
        obs = sampler.get_expert_obs(32)
        self.assertEqual(obs.shape, (32, 540))
        self.assertTrue(torch.isfinite(obs).all())

    def test_expert_uses_same_projection_and_motionlib_preprocessing(self):
        sampler = self.sampler()
        raw = torch.cat([sampler.motion_base_height, sampler.motion_dof_pos, sampler.motion_end_effector_pos,
                         sampler.motion_box_pos, sampler.motion_base_lin_vel, sampler.motion_base_ang_vel,
                         sampler.motion_base_rot], -1)
        with patch("torch.multinomial", return_value=torch.zeros(1, dtype=torch.long)), patch("torch.rand", return_value=torch.zeros(1)):
            obs = sampler.get_expert_obs(1)
        times = torch.arange(10) * 1.2
        w = (times-times.floor())[:, None]
        expected = raw[times.long()]*(1-w) + raw[times.ceil().long()]*w
        torch.testing.assert_close(obs, project_amp(expected.flatten()[None]))

    def test_missing_files_wrong_skill_and_short_clip_fail_clearly(self):
        (self.root / "carrywith3.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "carrywith3.pt"):
            self.sampler()
        self.config.write_text(yaml.safe_dump(dict(motions=dict(hold=self.entries))))
        with self.assertRaisesRegex(ValueError, "ONLY carryWith"):
            validate_carrywith_config(self.config)

    def test_temporally_insufficient_clip_rejected(self):
        path = self.root / "carrywith1.pt"
        data = torch.load(path, weights_only=False)
        torch.save({k: v[:5] for k, v in data.items()}, path)
        with self.assertRaisesRegex(ValueError, "too short"):
            self.sampler()


class TrainingRegressionTests(unittest.TestCase):
    def parse(self, argv):
        source = ROOT / "legged_gym/legged_gym/scripts/train_mixed_handoff.py"
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "get_args")

        def gym_parse(description):
            parser = argparse.ArgumentParser(description=description)
            parser.add_argument("--pipeline", default="gpu")
            parser.add_argument("--physx", dest="physics_engine", action="store_const", const=1, default=1)
            return parser.parse_args()

        namespace = dict(argparse=argparse, sys=sys, Path=Path, gymapi=NS(SIM_PHYSX=1),
                         gymutil=NS(parse_arguments=gym_parse), __doc__="Student training")
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        with patch.object(sys, "argv", ["train_mixed_handoff.py", *argv]):
            return namespace["get_args"]()

    def test_cli_supports_fresh_resume_smoke_flags_and_gym_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            path.touch()
            base = ["--teacher_checkpoint", str(path)]
            args = self.parse(base + ["--student_init_checkpoint", str(path), "--headless", "--no_amp",
                                     "--steps_per_env", "20", "--max_optimizer_updates", "1", "--pipeline", "cpu"])
            self.assertEqual(args.steps_per_env, 20)
            self.assertEqual(args.pipeline, "cpu")
            self.assertTrue(args.no_amp)
            self.assertTrue(args.headless)
            resumed = self.parse(base + ["--resume", "--resume_path", "student.pt"])
            self.assertTrue(resumed.resume)
            for options in (["--resume"], ["--resume_path", "student.pt"],
                            ["--resume", "--resume_path", "student.pt", "--student_init_checkpoint", str(path)],
                            ["--num_envs", "0"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.parse(base + options)

    def test_training_profile_preserves_viewer_rewards_commands_and_separate_reset(self):
        ns = dict(config_namespace)
        classes_from_source(ROOT / "legged_gym/legged_gym/envs/g1/carrybox_config.py", ["G1CfgPPO"], ns)
        classes_from_source(ROOT / "legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py",
                            ["MixedHandoffTrainingCfg", "MixedHandoffCfgPPO"], ns)
        training = ns["MixedHandoffTrainingCfg"]()
        viewer = ns["MixedHandoffCfg"]()
        self.assertEqual(training.env.num_envs, 16)
        self.assertFalse(training.env.test)
        self.assertEqual(viewer.env.num_envs, 1)
        self.assertTrue(viewer.env.test)
        self.assertEqual(training.asset.box.reset_mode, "random")
        self.assertEqual(training.asset.box.skill, ["pickUp"])
        self.assertEqual(training.asset.box.skill_init_prob, [1.])
        self.assertEqual(training.env.episode_length_s, 20.)
        self.assertEqual(training.handoff.stable_duration_s, .1)
        self.assertEqual(training.handoff.student_initial_stand_duration_s, 0.)
        self.assertEqual(viewer.asset.box.reset_mode, "default")
        self.assertEqual(viewer.handoff.stable_duration_s, .3)
        self.assertEqual(viewer.handoff.student_initial_stand_duration_s, .5)
        self.assertEqual(ns["MixedHandoffCfgPPO"]().runner.num_steps_per_env, 100)
        self.assertEqual(ns["MixedHandoffCfgPPO"]().runner.terminal_log_interval, 100)
        self.assertEqual(ns["MixedHandoffCfgPPO"]().runner.save_interval, 500)
        for group in ("student_rewards", "student_termination", "control", "sim", "domain_rand", "dataset"):
            def values(obj):
                return {k: values(getattr(obj, k)) if hasattr(getattr(obj, k), "__dict__") else getattr(obj, k)
                        for k in dir(obj) if not k.startswith("_")}
            self.assertEqual(values(getattr(training, group)), values(getattr(viewer, group)))
        registry = (ROOT / "legged_gym/legged_gym/utils/task_registry.py").read_text()
        self.assertIn("runner = HIMOnPolicyRunner", registry)
        script = (ROOT / "legged_gym/legged_gym/scripts/train_mixed_handoff.py").read_text()
        self.assertIn("runner = MixedHandoffOnPolicyRunner", script)


if __name__ == "__main__":
    unittest.main()
