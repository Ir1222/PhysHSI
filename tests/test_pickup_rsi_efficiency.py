"""CPU tests of real PickUp clips, native resets and command boundaries."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import yaml

from test_mixed_handoff import ROOT, HandoffState, HandoffSettings, MATH, classes_from_source, config_namespace
from test_student_carry import fixture
import test_mixed_handoff_training as training_tests
from test_mixed_handoff_training import (CarryWithMotionLib,
                                         MixedHandoffOnPolicyRunner, TensorEnvironment, configuration,
                                         actor_critic)
from legged_gym.pickup_rsi import pickup_window_bounds, sample_pickup_time, pickup_box_velocity


def training_config():
    ns = dict(config_namespace)
    classes_from_source(ROOT / "legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py",
                        ["MixedHandoffTrainingCfg"], ns)
    return ns["MixedHandoffTrainingCfg"]()


def configure(cfg, args):
    path = ROOT / "legged_gym/legged_gym/scripts/train_mixed_handoff.py"
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "configure_environment")
    ns = dict(MixedHandoffCfg=config_namespace["MixedHandoffCfg"])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
    ns["configure_environment"](cfg, args)


class PickupRSITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        config = Path(cls.folder.name) / "pickup.yaml"
        files = sorted((ROOT / "legged_gym/resources/dataset/dataset_carrybox/pickUp").glob("*.pt"))
        config.write_text(yaml.safe_dump(dict(motions=dict(pickUp=[dict(file=str(p), weight=4.) for p in files]))))
        NativeMotionLib = CarryWithMotionLib.__mro__[1]
        with contextlib.redirect_stdout(io.StringIO()):
            cls.motion = NativeMotionLib(str(config), str(ROOT / "legged_gym/resources/config/joint_id.txt"),
                                         [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()],
                                         60, "cpu", 10, [1., 1.], .7)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_real_clip_lengths_contacts_and_all_sampling_windows(self):
        m = self.motion
        self.assertEqual(m.motion_len["pickUp"].tolist(), [192, 178, 129])
        self.assertEqual(m.contact_index.tolist(), [107, 100, 42])
        ids = torch.arange(3).repeat(100)
        for offsets in ((-.8, -.1), (-.2, .2), (0., .5), (.5, 1.), (-100., 100.), (100., 101.)):
            with self.subTest(offsets=offsets):
                settings = NS(window="contact", contact_window_s=offsets)
                lo, hi = pickup_window_bounds(m, ids, "contact", offsets)
                times = sample_pickup_time(m, ids, settings)
                self.assertTrue(((times >= lo) & (times <= hi)).all())
                self.assertTrue(((times >= 0) & (times <= m.motion_len["pickUp"][ids] - 1)).all())
                for states in (m.get_motion_state("pickUp", ids, times), m.get_obj_motion_state("pickUp", ids, times)):
                    self.assertTrue(all(torch.isfinite(s).all() for s in states))
        for offsets in ((1., -1.), (float("nan"), 1.)):
            with self.assertRaises(ValueError):
                pickup_window_bounds(m, ids, "contact", offsets)

    def test_full_window_delegates_to_original_distribution(self):
        ids = torch.arange(3).repeat(100)
        torch.manual_seed(71)
        expected = self.motion.sample_time_rsi("pickUp", ids)
        torch.manual_seed(71)
        actual = sample_pickup_time(self.motion, ids, NS(window="full"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_robot_box_platform_share_clip_time_and_reference_velocities(self):
        env = fixture()
        env.cfg.pickup_rsi = training_config().pickup_rsi
        env.box_cfg = training_config().asset.box
        env.skill = ["pickUp"]
        env.motionlib = self.motion
        env.dof_vel = torch.zeros(2, 29)
        env._reset_ref_env_ids, env._reset_ref_motion_ids, env._reset_ref_motion_times = {}, {}, {}
        env._reset_default_env_ids = []
        env.platform_default_pos = torch.full((2, 3), -10.)
        env.thresh_tag, env.has_seen_tag = torch.zeros(2), torch.zeros(2, dtype=torch.bool)
        env.far_pos_offset = torch.zeros(2, 3)
        namespace = env._reset_ref_state_init.__func__.__globals__
        namespace.update(sample_pickup_time=sample_pickup_time, pickup_box_velocity=pickup_box_velocity)
        native = type(env).__mro__[1]._reset_boxes.__globals__
        native["torch_rand_float"] = lambda lo, hi, shape, device: lo + (hi-lo) * torch.rand(shape, device=device)
        native["quat_mul"] = MATH.quat_mul
        ids, times = torch.tensor([0, 2]), torch.tensor([101., 50.])
        with patch.object(env.motionlib, "sample_motions", return_value=ids), patch.dict(
                namespace, sample_pickup_time=lambda m, i, s: times):
            env._reset_ref_state_init(torch.arange(2))
        expected = self.motion.get_motion_state("pickUp", ids, times)
        torch.testing.assert_close(env.root_states[:, :3], expected[0] + env.env_origins)
        torch.testing.assert_close(env.root_states[:, 7:10], expected[2])
        torch.testing.assert_close(env.root_states[:, 10:13], expected[3])
        torch.testing.assert_close(env.dof_pos, expected[4])
        torch.testing.assert_close(env.dof_vel, expected[5])
        env._reset_boxes(torch.arange(2))
        pos, rot, platform, _ = self.motion.get_obj_motion_state("pickUp", ids, times)
        torch.testing.assert_close(env.box_states[:, :3], pos + env.env_origins)
        # Native 180-degree box yaw symmetry remains physically equivalent.
        self.assertTrue(torch.isfinite(env.box_states).all())
        self.assertTrue(platform[0] and not platform[1])
        self.assertAlmostEqual(env.platform_pos[0, 2].item(),
                               float(pos[0, 2] - env._box_size[0, 2]/2 - env._platform_height/2), places=6)
        torch.testing.assert_close(env.platform_pos[1], env.platform_default_pos[1])
        linear, angular = pickup_box_velocity(self.motion, ids, times)
        torch.testing.assert_close(env.box_states[:, 7:10], linear)
        torch.testing.assert_close(env.box_states[:, 10:13], angular)
        self.assertEqual(env.handoff.phase.tolist(), [0, 0])

    def test_disabled_pickup_uses_native_reset(self):
        env = fixture()
        with patch.object(type(env).__mro__[1], "_reset_ref_state_init") as native:
            env._reset_ref_state_init(torch.arange(2))
            native.assert_called_once()


class ImmediateCommandTests(unittest.TestCase):
    def test_gate_prepares_first_actor_command_and_executed_reward_is_unchanged(self):
        env = fixture()
        env.handoff = HandoffState(2, "cpu", .02, HandoffSettings(
            stable_duration_s=.1, student_initial_stand_duration_s=0., mode="fixed", fixed_command=(1., -.5, .7)))
        previous_history = torch.randn(2, 738)
        env.obs_buf = previous_history.clone()
        for step in range(5):
            env.executed_phase = env.handoff.phase.clone()
            env.executed_command = env.commands[:, :3].clone()
            env.episode_length_buf[:] = step + 1
            env.check_termination()
        self.assertEqual(env.executed_phase.tolist(), [0, 0])
        self.assertEqual(env.handoff.phase.tolist(), [1, 1])
        command = torch.tensor([[.016, -.012, .024]]).repeat(2, 1)
        torch.testing.assert_close(env.handoff.student_applied_command, command)
        actor, critic = env.compute_task_observations()
        torch.testing.assert_close(actor[:, -3:], command)
        torch.testing.assert_close(critic[:, -3:], command)
        # Advancing the gate never changes the executed Teacher command/history.
        torch.testing.assert_close(env.executed_command, torch.zeros(2, 3))
        torch.testing.assert_close(env.obs_buf, previous_history)
        env.executed_phase = env.handoff.phase.clone()
        env.executed_command = env.commands[:, :3].clone()
        env.actions[:] = .7
        env.check_termination()
        env.compute_reward()
        torch.testing.assert_close(env.executed_command, command)
        self.assertTrue(env.extras["handoff"]["first_student_step"].all())
        torch.testing.assert_close(env.extras["handoff"]["action_discontinuity"], torch.full((2, 29), .3))

    def test_both_durations_require_consecutive_physical_validity(self):
        for duration in (.1, .3):
            for invalid in ("height", "bilateral", "clearance", "done"):
                with self.subTest(duration=duration, invalid=invalid):
                    state = HandoffState(1, "cpu", .02, HandoffSettings(stable_duration_s=duration))
                    args = dict(height=torch.tensor([.8]), bilateral=torch.tensor([True]),
                                clearance=torch.tensor([.1]), done=torch.tensor([False]), episode_time=torch.tensor([1.]))
                    args[invalid] = torch.tensor([False if invalid == "bilateral" else True]) if invalid in ("bilateral", "done") else torch.tensor([0.])
                    for _ in range(100):
                        self.assertFalse(state.complete_step(**args).any())
                    args.update(height=torch.tensor([.8]), bilateral=torch.tensor([True]),
                                clearance=torch.tensor([.1]), done=torch.tensor([False]))
                    for _ in range(round(duration/.02)-1):
                        self.assertFalse(state.complete_step(**args).any())
                    self.assertTrue(state.complete_step(**args).all())

    def test_initial_observation_refresh_cannot_trigger_gate(self):
        env = fixture()
        env.handoff = HandoffState(2, "cpu", .02, HandoffSettings(stable_duration_s=.02))
        env._initializing_training_observations = True
        env.check_termination()
        self.assertEqual(env.handoff.phase.tolist(), [0, 0])
        self.assertEqual(env.handoff.stable_time.tolist(), [0., 0.])
        env._initializing_training_observations = False
        env.check_termination()
        self.assertEqual(env.handoff.phase.tolist(), [1, 1])


class EfficiencyConfigTests(unittest.TestCase):
    def parse(self, flags):
        with tempfile.TemporaryDirectory() as folder:
            teacher = Path(folder) / "teacher.pt"
            teacher.touch()
            return training_tests.TrainingRegressionTests().parse(["--teacher_checkpoint", str(teacher), *flags])

    def test_defaults_ablation_profiles_and_explicit_overrides(self):
        args = self.parse([])
        self.assertIsNone(args.episode_length_s)
        cfg = training_config()
        configure(cfg, args)
        self.assertEqual(cfg.env.episode_length_s, 30.)
        for name, expected in {"A": (False, 20., .3, .5), "B": (True, 20., .3, .5),
                               "C": (True, 20., .3, 0.), "D": (True, 20., .1, 0.),
                               "historical": (False, 60., .3, .5)}.items():
            cfg = training_config()
            configure(cfg, self.parse(["--ablation", name]))
            self.assertEqual((cfg.pickup_rsi.enabled, cfg.env.episode_length_s, cfg.handoff.stable_duration_s,
                              cfg.handoff.student_initial_stand_duration_s), expected)
        configure(cfg, self.parse(["--ablation", "D", "--episode_length_s", "12", "--reset_mode", "default",
                                  "--stable_duration_s", ".3", "--pickup_rsi_window", "full"]))
        self.assertFalse(cfg.pickup_rsi.enabled)
        self.assertEqual(cfg.env.episode_length_s, 12.)
        self.assertEqual(cfg.handoff.stable_duration_s, .3)
        self.assertEqual(cfg.pickup_rsi.window, "full")

    def test_survival_diagnostics_distinguish_failures_timeouts_and_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "teacher.pt"
            torch.save(dict(model_state_dict=actor_critic().state_dict()), checkpoint)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runner = MixedHandoffOnPolicyRunner(TensorEnvironment(), configuration(amp=False, steps=18),
                                                    checkpoint, log_dir=folder)
                runner.learn(1, optimize=False)
            self.assertFalse((Path(folder) / "handoff_events.jsonl").exists())
            self.assertEqual(runner.episode_student_steps.tolist(), [0., 0.])
            self.assertEqual(runner.alg.optimizer_updates, 0)
            metrics = json.loads((Path(folder) / "training_metrics.jsonl").read_text().splitlines()[-1])
            self.assertAlmostEqual(metrics["performance/episode/teacher_steps"], 3.)
            self.assertAlmostEqual(metrics["performance/student/completed_survival_s"], .28)
            self.assertAlmostEqual(metrics["performance/student/censored_timeout_s"], .28)
            self.assertEqual(metrics["performance/episode/zero_student_fraction"], 0.)
            self.assertEqual(metrics["performance/student/episode_reached_amp_fraction"], 1.)
            self.assertEqual(metrics["distribution/student/completed_survival_s/median"],
                             metrics["performance/student/completed_survival_s"])
            runner.assert_teacher_frozen()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                incomplete = MixedHandoffOnPolicyRunner(TensorEnvironment(), configuration(amp=False, steps=10),
                                                        checkpoint, log_dir=Path(folder) / "incomplete")
                incomplete.learn(1, optimize=False)
            metrics = json.loads((Path(folder) / "incomplete/training_metrics.jsonl").read_text().splitlines()[-1])
            self.assertEqual(metrics["student_incomplete_segments"], 2)
            self.assertEqual(metrics["student_incomplete_mean_frames"], 7.)
            self.assertNotIn("performance/student/completed_survival_s", metrics)
            self.assertNotIn("performance/student/censored_timeout_s", metrics)

    def test_early_student_failure_is_valid_ppo_without_fake_amp_or_handoff_failure(self):
        class ShortSegments(TensorEnvironment):
            def step(self, actions):
                student = self.handoff.phase == 1
                self.t[student] = torch.tensor([15, 17])[student]
                return super().step(actions)

        with tempfile.TemporaryDirectory() as folder:
            teacher = Path(folder) / "teacher.pt"
            torch.save(dict(model_state_dict=actor_critic().state_dict()), teacher)
            with contextlib.redirect_stdout(io.StringIO()):
                runner = MixedHandoffOnPolicyRunner(ShortSegments(), configuration(amp=False, steps=6), teacher)
                runner.reset()
                samples = [runner.collect_step() for _ in range(6)]
            self.assertGreater(int(runner.storage.data["student_valid"][:6].sum()), 0)
            self.assertEqual(int(runner.storage.data["amp_valid"][:6].sum()), 0)
            self.assertEqual(runner.handoff_failures, 0)
            lengths = torch.cat([s["student/segment_frames"] for s in samples])
            torch.testing.assert_close(lengths, torch.ones_like(lengths))

    def test_legacy_full_resume_and_new_rsi_window_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            teacher, saved = Path(folder) / "teacher.pt", Path(folder) / "student.pt"
            torch.save(dict(model_state_dict=actor_critic().state_dict()), teacher)
            with contextlib.redirect_stdout(io.StringIO()):
                old = MixedHandoffOnPolicyRunner(TensorEnvironment(), configuration(amp=False), teacher)
                old.save(saved)
                cfg = configuration(amp=False)
                cfg["environment"] = dict(pickup_rsi=dict(enabled=False, window="contact", contact_window_s=(-.2, .2)))
                compatible = MixedHandoffOnPolicyRunner(TensorEnvironment(), cfg, teacher)
                compatible.load(saved)
                cfg["environment"]["pickup_rsi"]["enabled"] = True
                with self.assertRaisesRegex(ValueError, "pickup_rsi"):
                    compatible.load(saved)


if __name__ == "__main__":
    unittest.main()
