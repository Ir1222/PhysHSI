"""Focused CPU contracts for the independent single-frame Pose AMP task."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "legged_gym"))

from rsl_rl.modules.masked_amp import project_amp
from rsl_rl.modules.masked_pose_amp import (
    KEEP_INDICES,
    add_gated_pose_reward,
    classify_pose_mode,
    compute_pose_tracking_gate,
    condition_pose_observation,
    pose_amp_valid,
    project_pose_amp,
)
from rsl_rl.runners.vel_track_pose_amp_on_policy_runner import (
    TRAINING_CONTRACT,
    VelTrackPoseAMPOnPolicyRunner,
)
from rsl_rl.storage.mixed_handoff_rollout_storage import MixedHandoffRolloutStorage
from rsl_rl.storage.vel_track_pose_amp_rollout_storage import VelTrackPoseAMPRolloutStorage
from test_mixed_handoff import MATH
from test_mixed_handoff_training import TensorEnvironment, actor_critic


def pose_motion_class():
    package = ModuleType("pose_motion")
    package.__path__ = [str(ROOT / "legged_gym/legged_gym/envs/motionlib")]
    utils = ModuleType("legged_gym.utils")
    utils.torch_utils = MATH
    with patch.dict(sys.modules, {"pose_motion": package, "legged_gym.utils": utils}):
        from pose_motion.motionlib_pose_amp import PoseAMPMotionLib
    return PoseAMPMotionLib


def pose_configuration(enabled=True, steps=4):
    return dict(
        policy=dict(actor_hidden_dims=[512, 256, 256], critic_hidden_dims=[32, 16],
                    activation="elu", init_noise_std=.4),
        algorithm=dict(num_learning_epochs=1, num_mini_batches=1,
                       schedule="fixed", learning_rate=2e-4),
        runner=dict(num_steps_per_env=steps, save_interval=10),
        amp=dict(enabled=enabled, num_obs=54, discriminator_num_obs=55,
                 num_one_step_obs=54, window_length=1,
                 pose_style_weight=2., pose_gate_floor=.5,
                 stand_linear_threshold=.05, stand_yaw_threshold=.05,
                 use_normalizer=False),
    )


class PoseExpert:
    identity = dict(source="Pose-HOLD+CarryWith-mode-conditioned", files=[],
                    weights={}, fps=60, control_dt=.02, pose_features=54,
                    pose_history=1, condition="stand=1,moving=0")

    def get_expert_obs(self, batch_size, stand_mode=None):
        pose = torch.zeros(batch_size, 54)
        mode = torch.zeros(batch_size, dtype=torch.bool) if stand_mode is None else stand_mode.cpu()
        return condition_pose_observation(pose, mode)


class PoseAMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test-tmp")
        self.teacher = Path(self.temp.name) / "teacher.pt"
        torch.save(dict(model_state_dict=actor_critic().state_dict()), self.teacher)

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, enabled=True, steps=4, initialize=True):
        return VelTrackPoseAMPOnPolicyRunner(
            TensorEnvironment(), pose_configuration(enabled, steps), self.teacher,
            expert=PoseExpert() if enabled else None, initialize=initialize)

    def test_old_temporal_contract_remains_541_and_new_contract_is_55(self):
        old = MixedHandoffRolloutStorage(1, 1)
        new = VelTrackPoseAMPRolloutStorage(1, 1)
        self.assertEqual(old.data["amp_observations"].shape[-1], 541)
        self.assertEqual(new.data["amp_observations"].shape[-1], 55)
        self.assertEqual(TRAINING_CONTRACT["amp_motion_obs"], 54)
        self.assertEqual(TRAINING_CONTRACT["amp_discriminator_obs"], 55)
        self.assertEqual(TRAINING_CONTRACT["amp_history"], 1)
        self.assertEqual(project_amp(torch.zeros(1, 600)).shape[-1], 540)

    def test_projection_uses_latest_frame_only_and_preserves_feature_order(self):
        history = torch.arange(600., dtype=torch.float).reshape(1, 600)
        expected = history[:, -60:][:, list(KEEP_INDICES)]
        torch.testing.assert_close(project_pose_amp(history), expected)
        changed = history.clone()
        changed[:, :-60] = torch.randn_like(changed[:, :-60]) * 1e6
        torch.testing.assert_close(project_pose_amp(changed), expected)
        latest = history[:, -60:].clone()
        latest[:, 48:54] = float("nan")
        torch.testing.assert_close(project_pose_amp(latest), expected)
        self.assertEqual(project_pose_amp(torch.zeros(2, 60)).shape, (2, 54))

    def test_mode_labels_include_yaw_only_as_moving(self):
        commands = torch.tensor([[0., 0., 0.], [.2, 0., 0.],
                                 [0., .2, 0.], [0., 0., .2]])
        stand = classify_pose_mode(commands, .05, .05)
        self.assertEqual(stand.tolist(), [True, False, False, False])
        conditioned = condition_pose_observation(torch.zeros(4, 54), stand)
        self.assertEqual(conditioned.shape, (4, 55))
        torch.testing.assert_close(conditioned[:, -1], stand.float())

    def test_first_student_transition_valid_and_teacher_never_valid(self):
        runner = self.runner()
        runner.reset()
        with patch.object(runner.alg.amp, "predict_reward",
                          side_effect=lambda x, _: torch.ones(len(x), 1)):
            runner.collect_step()  # Teacher
            runner.collect_step()  # Teacher transition that opens the gate
            samples = runner.collect_step()  # First Student-owned transition
        valid = runner.storage.data["amp_valid"][:3]
        student = runner.storage.data["student_valid"][:3]
        self.assertTrue(torch.equal(valid, student))
        self.assertFalse(valid[:2].any())
        self.assertEqual(valid[2].tolist(), [True, False])
        self.assertTrue((samples["amp_bonus"] > 0).all())
        self.assertTrue(torch.equal(pose_amp_valid(student[2], torch.ones(2, dtype=torch.bool)),
                                    student[2]))

    def test_pose_discriminator_trains_on_55d_student_and_expert_batches(self):
        runner = self.runner(steps=4)
        before = {k: v.clone() for k, v in runner.alg.amp.state_dict().items()}
        runner.learn(1)
        self.assertEqual(runner.alg.optimizer_updates, 1)
        self.assertEqual(runner.amp_steps, runner.student_steps)
        self.assertTrue(any(not torch.equal(value, before[key])
                            for key, value in runner.alg.amp.state_dict().items()))

    def test_soft_gate_floor_and_additive_task_reward(self):
        gate = compute_pose_tracking_gate(
            torch.tensor([0., 100.]), torch.tensor([0., 100.]), .25, .25, .5)
        torch.testing.assert_close(gate, torch.tensor([1., .5]))
        task = torch.tensor([.12, -.03, .7])
        raw = torch.tensor([1., .5, 1.])
        combined, bonus = add_gated_pose_reward(
            task, raw, torch.tensor([1., .5, 1.]),
            torch.tensor([True, True, False]), .02, 2.)
        torch.testing.assert_close(bonus, torch.tensor([.04, .01, 0.]))
        torch.testing.assert_close(combined, task + bonus)

    def test_checkpoint_format_is_isolated_and_temporal_resume_rejected(self):
        runner = self.runner(enabled=False)
        pose_path = Path(self.temp.name) / "pose.pt"
        runner.save(pose_path)
        saved = torch.load(pose_path, weights_only=False)
        self.assertEqual(saved["format"], "vel_track_pose_amp_student_v1")
        old_path = Path(self.temp.name) / "old.pt"
        torch.save({"format": "mixed_handoff_student_v1"}, old_path)
        with self.assertRaisesRegex(ValueError, "vel_track_pose_amp_student_v1"):
            runner.load(old_path)

    def test_config_registration_entrypoints_and_baseline_isolation(self):
        config = (ROOT / "legged_gym/legged_gym/envs/g1/carrybox_vel_track_pose_amp_config.py").read_text()
        registry = (ROOT / "legged_gym/legged_gym/envs/__init__.py").read_text()
        train = (ROOT / "legged_gym/legged_gym/scripts/train_vel_track_pose_amp.py").read_text()
        play = (ROOT / "legged_gym/legged_gym/scripts/play_vel_track_pose_amp.py").read_text()
        self.assertIn('experiment_name = "vel_track_pose_amp"', config)
        self.assertIn('run_name = "pose_amp_v1"', config)
        self.assertIn('pose_style_weight = 2.0', config)
        self.assertIn('pose_gate_floor = 0.5', config)
        self.assertIn('"carrybox_vel_track_pose_amp"', registry)
        self.assertIn('make_env("carrybox_vel_track_pose_amp"', train)
        self.assertIn('make_env("carrybox_vel_track_pose_amp"', play)
        self.assertIn('"carrybox_mixed_handoff"', registry)

    def test_tensorboard_pose_diagnostics_are_explicit(self):
        source = (ROOT / "rsl_rl/rsl_rl/runners/vel_track_pose_amp_on_policy_runner.py").read_text()
        for tag in ("Train/mean_pose_amp_raw", "Train/mean_pose_amp_gate",
                    "Train/mean_pose_amp_bonus", "Train/mean_raw_reward",
                    "Train/mean_reward", "Train/pose_amp_bonus_to_task_reward",
                    "Train/pose_amp_nonzero_fraction"):
            self.assertIn(tag, source)


class PoseExpertSamplingTests(unittest.TestCase):
    def test_policy_and_expert_pose_projection_are_identical(self):
        # Import through the same lightweight package adaptation used by the
        # existing motion-library regression tests.
        PoseAMPMotionLib = pose_motion_class()
        lib = PoseAMPMotionLib.__new__(PoseAMPMotionLib)
        lib.device = torch.device("cpu")
        lib.skills = ["carryWith", "hold"]
        lib.motion_weights = {"carryWith": torch.tensor([1.]), "hold": torch.tensor([1.])}
        lib.motion_len = {"carryWith": torch.tensor([1]), "hold": torch.tensor([1])}
        lib.motion_start_ids = {"carryWith": torch.tensor([0]), "hold": torch.tensor([1])}
        raw = torch.arange(120., dtype=torch.float).reshape(2, 60)
        lib.motion_base_height = raw[:, :1]
        lib.motion_dof_pos = raw[:, 1:30]
        lib.motion_end_effector_pos = raw[:, 30:45]
        lib.motion_box_pos = raw[:, 45:48]
        lib.motion_base_lin_vel = raw[:, 48:51]
        lib.motion_base_ang_vel = raw[:, 51:54]
        lib.motion_base_rot = raw[:, 54:60]
        modes = torch.tensor([False, True])
        expert = lib.get_expert_obs(2, modes)
        expected = torch.stack((project_pose_amp(raw[0]), project_pose_amp(raw[1])))
        torch.testing.assert_close(expert[:, :54], expected)
        torch.testing.assert_close(expert[:, -1], modes.float())

    def test_real_hold_and_carrywith_sources_sample_single_55d_poses(self):
        PoseAMPMotionLib = pose_motion_class()
        mapping = ROOT / "legged_gym/resources/config/joint_id.txt"
        names = [line.split()[1] for line in mapping.read_text().splitlines() if line.strip()]
        lib = PoseAMPMotionLib(
            ROOT / "legged_gym/resources/config/carrybox_hold_carrywith_amp.yaml",
            mapping, names, 60, "cpu", .02, .5)
        modes = torch.tensor([True, False, True, False])
        observations = lib.get_expert_obs(len(modes), modes)
        self.assertEqual(observations.shape, (4, 55))
        self.assertTrue(torch.isfinite(observations).all())
        torch.testing.assert_close(observations[:, -1], modes.float())
        self.assertEqual(lib.window_length, 1)
        self.assertEqual(lib.motion_weights["carryWith"].tolist(), [5., 2., 5.])
        self.assertEqual(lib.motion_weights["hold"].tolist(), [1., 1., 1., 1.])


if __name__ == "__main__":
    unittest.main()
