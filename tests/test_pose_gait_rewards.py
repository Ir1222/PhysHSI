"""CPU contracts for the Pose-AMP Student gait rewards and histories."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "legged_gym/legged_gym/envs/g1/carrybox_vel_track_pose_amp.py"
STUDENT_CONTROL = 1


def class_to_dict(value):
    return {name: getattr(value, name) for name in vars(value) if not name.startswith("_")}


class Base:
    pass


def gait_environment_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == "VelTrackPoseAMP")
    namespace = dict(torch=torch, F=F, math=math, LeggedRobot=Base,
                     STUDENT_CONTROL=STUDENT_CONTROL, class_to_dict=class_to_dict,
                     GAIT_REWARD_NAMES=("gait_foot_slip", "gait_swing_clearance",
                                        "gait_foot_scuff", "gait_air_time"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["VelTrackPoseAMP"]


Env = gait_environment_class()


def gait_config():
    return NS(contact_threshold_N=1., moving_threshold=.1, slip_tolerance_m_s=.05,
              horizontal_speed_scale_m_s=.3, target_height_m=.04, scuff_height_m=.02,
              air_time_min_s=.06, air_time_range_s=.2)


def state_fixture():
    env = Env.__new__(Env)
    env.num_envs, env.device, env.dt = 2, "cpu", .02
    env.cfg = NS(gait=gait_config())
    env.executed_phase = torch.tensor([0, STUDENT_CONTROL])
    env.executed_command = torch.tensor([[.4, 0., 0.], [.4, 0., 0.]])
    env.gait_feet_indices = torch.tensor([1, 2])
    env.rigid_body_states = torch.zeros(2, 3, 13)
    env.rigid_body_states[:, :, 6] = 1.
    env.rigid_body_states[:, 1:3, 2] = .055
    env.contact_forces = torch.zeros(2, 3, 3)
    env.gait_sole_points = torch.tensor([
        [[0., 0., -.035]],
        [[0., 0., -.035]],
    ])
    contact = torch.zeros(2, 2, dtype=torch.bool)
    env.gait_contact_raw = contact.clone()
    env.gait_previous_raw = contact.clone()
    env.gait_contact_filtered = contact.clone()
    env.gait_touchdown = contact.clone()
    env.gait_swing = contact.clone()
    env.gait_stance_seen = contact.clone()
    env.gait_air_time = torch.zeros(2, 2)
    env.gait_preceding_air_time = torch.zeros(2, 2)
    env.gait_horizontal_velocity = torch.zeros(2, 2, 2)
    env.gait_horizontal_speed = torch.zeros(2, 2)
    env.gait_clearance = torch.zeros(2, 2)
    env.gait_previous_owner = torch.zeros(2, dtype=torch.bool)
    env.gait_history_valid = torch.zeros(2, dtype=torch.bool)
    return env


class PoseGaitRewardTests(unittest.TestCase):
    def test_registry_scales_enabled_terms_once_and_drops_air_time(self):
        env = Env.__new__(Env)
        env.num_envs, env.device, env.dt = 2, "cpu", .02
        env.cfg = NS(student_rewards=NS(scales=NS(
            gait_foot_slip=-.05,
            gait_swing_clearance=.08,
            gait_foot_scuff=-.05,
            gait_air_time=0.,
        )))
        env.reward_names = []
        env._prepare_student_reward_function()
        self.assertEqual(env.student_reward_names,
                         ["gait_foot_slip", "gait_swing_clearance", "gait_foot_scuff"])
        self.assertEqual(env.student_reward_scales, {
            "gait_foot_slip": -.05 * env.dt,
            "gait_swing_clearance": .08 * env.dt,
            "gait_foot_scuff": -.05 * env.dt,
        })
        self.assertEqual(len(env.student_reward_names), len(set(env.student_reward_names)))

        env.reward_names = ["gait_foot_slip"]
        with self.assertRaisesRegex(ValueError, "already registered"):
            env._prepare_student_reward_function()

    def test_contact_threshold_and_owner_handoff_reset_use_executed_step(self):
        env = state_fixture()
        env.gait_air_time.fill_(9.)
        env.contact_forces[1, 1] = torch.tensor([.8, .8, 0.])  # Norm > 1 N.
        env.contact_forces[1, 2] = torch.tensor([0., 0., 1.])  # Exactly 1 N is not contact.
        env._update_gait_state()
        self.assertEqual(env.gait_contact_raw.tolist(), [[False, False], [True, False]])
        torch.testing.assert_close(env.gait_air_time[0], torch.zeros(2))
        torch.testing.assert_close(env.gait_air_time[1], torch.tensor([0., .02]))
        torch.testing.assert_close(env.gait_clearance, torch.full((2, 2), .02), atol=1e-7, rtol=0)

        # A Teacher->Student owner change discards inherited air/contact history,
        # then records only this actual Student-owned step.
        env.gait_air_time[0] = 7.
        env.gait_previous_raw[0] = True
        env.gait_contact_filtered[0] = True
        env.executed_phase[0] = STUDENT_CONTROL
        env._update_gait_state()
        torch.testing.assert_close(env.gait_air_time[0], torch.full((2,), .02))
        self.assertFalse(env.gait_touchdown[0].any())
        torch.testing.assert_close(env.gait_preceding_air_time[0], torch.zeros(2))

        env._reset_gait_history(torch.tensor([1]))
        self.assertFalse(env.gait_history_valid[1])
        self.assertEqual(env.gait_air_time[1].sum(), 0.)
        self.assertFalse(env.gait_contact_filtered[1].any())

    def test_reward_formulas_require_student_and_moving_command(self):
        env = state_fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.gait_horizontal_speed[:] = torch.tensor([.35, .15])
        env.gait_contact_filtered[:] = torch.tensor([True, False])
        env.gait_swing[:] = ~env.gait_contact_filtered
        env.gait_clearance[:] = torch.tensor([0., .02])
        env.gait_touchdown[:] = torch.tensor([False, True])
        env.gait_preceding_air_time[:] = torch.tensor([0., .16])

        torch.testing.assert_close(env._reward_gait_foot_slip(), torch.full((2,), .5))
        torch.testing.assert_close(env._reward_gait_swing_clearance(),
                                   torch.full((2,), 1 / 12), atol=1e-7, rtol=0)
        torch.testing.assert_close(env._reward_gait_foot_scuff(),
                                   torch.full((2,), math.exp(-1) / 4), atol=1e-7, rtol=0)
        torch.testing.assert_close(env._reward_gait_air_time(), torch.full((2,), .5))

        env.executed_command[0] = 0.
        env.executed_phase[1] = 0
        for name in ("foot_slip", "swing_clearance", "foot_scuff", "air_time"):
            self.assertEqual(getattr(env, "_reward_gait_" + name)().sum(), 0.)

    def test_physical_metrics_select_stance_swing_and_low_clearance_motion(self):
        env = state_fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.gait_horizontal_speed[:] = torch.tensor([.2, .15])
        env.gait_contact_filtered[:] = torch.tensor([True, False])
        env.gait_swing[:] = ~env.gait_contact_filtered
        env.gait_clearance[:] = torch.tensor([0., .01])
        metrics = env._gait_physical_diagnostics()
        self.assertTrue(torch.isnan(metrics["student/stance_slip_m_s"][:, 1]).all())
        torch.testing.assert_close(metrics["student/stance_slip_m_s"][:, 0], torch.full((2,), .2))
        torch.testing.assert_close(metrics["student/swing_clearance_m"][:, 1], torch.full((2,), .01))
        torch.testing.assert_close(metrics["student/low_clearance_swing_fraction"][:, 1],
                                   torch.ones(2))


if __name__ == "__main__":
    unittest.main()
