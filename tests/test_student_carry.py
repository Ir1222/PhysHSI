"""Scope 2 tests: real tensor/quaternion math and phase-aware environment hooks."""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from test_mixed_handoff import (
    environment_fixture, HandoffSettings, STUDENT_CONTROL,
    MixedHandoffCfg, MATH,
)


def quat(roll=0., pitch=0., yaw=0.):
    return MATH.quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw]))[0]


def fixture():
    env = environment_fixture(HandoffSettings(stable_duration_s=.02))
    # Dedicated torso body permits actual relative-rotation tests.
    env.student_body_indices["torso"] = 6
    env.rigid_body_states[:, 0, :3] = env.root_states[:, :3]
    env.rigid_body_states[:, 6, :3] = env.root_states[:, :3]
    env.rew_buf = torch.zeros(2)
    env.last_actions = torch.zeros(2, 29)
    env.last_dof_vel = torch.zeros(2, 29)
    env.torques = env.computed_torques = torch.zeros(2, 29)
    env.p_gains = torch.ones(29)
    env.dof_pos_limits = torch.tensor([[-2., 2.]]).repeat(29, 1)
    env.dof_vel_limits = torch.ones(29) * 10
    env.torque_limits = torch.ones(29) * 100
    env.penalised_contact_indices = [0]
    env.reward_names = ["walk_task", "action_rate"]
    env.reward_functions = [lambda: torch.tensor([2., 3.]), env._reward_action_rate]
    env.reward_scales = {"walk_task": env.dt, "action_rate": -.03 * env.dt}
    env.episode_sums = {name: torch.zeros(2) for name in env.reward_scales}
    return env


def raw_reward(env, name):
    return env.student_reward_functions[env.student_reward_names.index(name)]()


def set_scales(env, **scales):
    from types import SimpleNamespace
    env.cfg.student_rewards.scales = SimpleNamespace(**scales)
    env._prepare_student_reward_function()


def student_step(env):
    env.executed_phase[:] = STUDENT_CONTROL
    env.handoff.phase[:] = STUDENT_CONTROL
    env.check_termination()
    env.compute_reward()
    return env.extras["handoff"]


def update(env, surface_distance=None, box_bottom=None):
    # Geometry/contact tests use known near-box palm coordinates.
    env.hand_box_surface_distance = torch.zeros(2, 2) if surface_distance is None else surface_distance
    env.box_bottom = env.box_states[:, 2] - .125 if box_bottom is None else box_bottom
    env._update_student_carry_state()
    return env


class OwnershipTests(unittest.TestCase):
    def test_gate_step_keeps_exact_teacher_reward_and_first_student_has_no_teacher_terms(self):
        env = fixture()
        env.check_termination()  # Gate enters Student, but executed action was Teacher.
        self.assertTrue((env.handoff.phase == STUDENT_CONTROL).all())
        baseline = copy.copy(env)
        baseline.rew_buf = torch.zeros(2)
        baseline.episode_sums = {k: torch.zeros(2) for k in env.episode_sums}
        type(env).__mro__[1].compute_reward(baseline)
        env.compute_reward()
        torch.testing.assert_close(env.rew_buf, baseline.rew_buf)
        self.assertEqual(env.extras["handoff"]["student_total_reward"].abs().sum(), 0.)
        teacher_sums = {k: v.clone() for k, v in env.episode_sums.items()}
        info = student_step(env)
        self.assertEqual(info["teacher_total_reward"].abs().sum(), 0.)
        self.assertEqual(info["teacher_reward_raw/walk_task"].abs().sum(), 0.)
        torch.testing.assert_close(env.rew_buf, info["student_total_reward"])
        for k in env.episode_sums:
            torch.testing.assert_close(env.episode_sums[k], teacher_sums[k])
        for v in info.values():
            self.assertEqual(v.shape[0], env.num_envs)
        self.assertTrue(torch.isfinite(env.rew_buf).all())

    def test_asynchronous_ownership_and_no_double_count_of_regularization(self):
        env = fixture()
        env.executed_phase[1] = STUDENT_CONTROL
        env.handoff.phase[1] = STUDENT_CONTROL
        set_scales(env, action_rate=-.03)
        env.check_termination()
        env.compute_reward()
        expected = -(.4**2)*29*.03*env.dt
        self.assertAlmostEqual(env.rew_buf[1].item(), expected, places=6)
        self.assertEqual(env.student_steps.tolist(), [0., 1.])
        self.assertEqual(env.extras["handoff"]["student_reward_raw/action_rate"][0], 0.)

    def test_positive_clipping_and_teacher_termination_order(self):
        env = fixture()
        env.cfg.rewards.only_positive_rewards = True
        env.reward_functions = [lambda: torch.full((2,), -2.)]
        env.reward_names = ["walk_task"]
        env.reward_scales = {"walk_task": env.dt, "termination": -1.}
        env.episode_sums = {k: torch.zeros(2) for k in env.reward_scales}
        set_scales(env, action_rate=-1.)
        env.executed_phase[1] = STUDENT_CONTROL
        env.check_termination()
        env.reset_buf[0] = True
        env.compute_reward()
        torch.testing.assert_close(env.rew_buf, torch.tensor([-1., 0.]))

    def test_next_command_cannot_change_executed_step_reward(self):
        env = fixture()
        env.executed_command[:] = torch.tensor([.3, -.2, .1])
        info = student_step(env)
        expected = info["student_reward_raw/carry_lin_vel_tracking"].clone()
        # Both the next command and target may have advanced in check_termination.
        env.handoff.student_target_command[:] = 100.
        env.handoff.student_applied_command[:] = -100.
        env.commands[:] = 50.
        env.check_termination()
        env.compute_reward()
        torch.testing.assert_close(env.extras["handoff"]["student_reward_raw/carry_lin_vel_tracking"], expected)

    def test_episode_student_denominator_excludes_teacher_steps(self):
        env = fixture()
        env.episode_length_buf[:] = 2000
        student_step(env)
        student_step(env)
        expected = env.student_total_sum / 2
        torch.testing.assert_close(env.extras["handoff"]["episode_student_mean_reward"], expected)
        self.assertEqual(env.student_steps.tolist(), [2., 2.])

    def test_ordinary_reset_clears_history_and_logs_only_student_denominator(self):
        env = fixture()
        env.student_steps[:] = torch.tensor([2., 0.])
        env.student_total_sum[:] = torch.tensor([4., 0.])
        env.last_box_contacts[:] = True
        env.grasp_loss_time[:] = .2
        env.student_episode_sums["action_rate"][:] = torch.tensor([-.2, 0.])

        def base_reset(instance, ids):
            instance.extras["episode"] = {}

        with patch.object(type(env).__mro__[1], "reset_idx", base_reset):
            env.reset_idx(torch.tensor([0, 1]))
        episode = env.extras["episode"]
        self.assertEqual(episode["student/mean_step_reward/total"], 2.)
        self.assertEqual(episode["student/episodes_without_student"], 1.)
        self.assertEqual(episode["student/episodes_with_student"], 1.)
        self.assertAlmostEqual(episode["student/mean_step_reward/action_rate"].item(), -.1)
        self.assertFalse(env.last_box_contacts.any())
        self.assertEqual(env.grasp_loss_time.sum(), 0.)
        self.assertEqual(env.student_steps.sum(), 0.)

    def test_nonfinite_or_wrong_shape_student_rewards_fail_explicitly(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.check_termination()
        index = env.student_reward_names.index("carry_lin_vel_tracking")
        env.student_reward_functions[index] = lambda: torch.full((2,), float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            env.compute_reward()
        env.student_reward_functions[index] = lambda: torch.zeros(2, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            env.compute_reward()


class CoordinateTests(unittest.TestCase):
    def test_forward_lateral_and_yaw_tracking(self):
        for command in ((.6, 0., 0.), (0., -.4, 0.), (0., 0., .5)):
            env = fixture()
            env.executed_command[:] = torch.tensor(command)
            env.rigid_body_states[:, 0, 7:9] = torch.tensor(command[:2])
            env.rigid_body_states[:, 0, 12] = command[2]
            update(env)
            torch.testing.assert_close(raw_reward(env, "carry_lin_vel_tracking"), torch.ones(2))
            torch.testing.assert_close(raw_reward(env, "carry_yaw_vel_tracking"), torch.ones(2))

    def test_yaw_frame_ignores_roll_pitch_and_yaw_rate_uses_world_z(self):
        env = fixture()
        env.rigid_body_states[:, 0, 3:7] = quat(.4, -.5, math.pi/2)
        env.rigid_body_states[:, 0, 7:10] = torch.tensor([0., .6, .3])
        env.rigid_body_states[:, 0, 10:13] = torch.tensor([1., 2., .7])
        env.executed_command[:] = torch.tensor([.6, 0., .7])
        update(env)
        torch.testing.assert_close(env.student_carry_metrics["measured_velocity_yaw"], torch.tensor([[.6, 0.], [.6, 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(raw_reward(env, "carry_lin_vel_tracking"), torch.ones(2))
        torch.testing.assert_close(env.student_carry_metrics["measured_yaw_rate_world"], torch.full((2,), .7))
        full_local = MATH.quat_rotate_inverse(env.rigid_body_states[:, 0, 3:7], env.rigid_body_states[:, 0, 7:10])
        self.assertFalse(torch.allclose(full_local[:, :2], env.student_carry_metrics["measured_velocity_yaw"]))


class ContactTests(unittest.TestCase):
    def test_bilateral_filter_and_force_attribution(self):
        env = fixture()
        update(env)
        self.assertTrue((raw_reward(env, "carry_bilateral_contact") == 1).all())
        env.contact_forces[0, 8] = 0.
        update(env)  # One-frame OR filter preserves the contact for this frame.
        self.assertEqual(raw_reward(env, "carry_bilateral_contact")[0], 1.)
        update(env)
        self.assertEqual(raw_reward(env, "carry_bilateral_contact")[0], 0.)
        self.assertEqual(raw_reward(env, "carry_bilateral_contact")[1], 1.)
        update(env, surface_distance=torch.ones(2, 2), box_bottom=torch.ones(2))
        update(env, surface_distance=torch.ones(2, 2), box_bottom=torch.ones(2))
        self.assertFalse(env.student_carry_metrics["filtered_box_contact"].any())
        self.assertTrue(env.student_carry_metrics["raw_hand_contact"][1].all())


class TerminationTests(unittest.TestCase):
    def test_student_drop_and_separation_do_not_affect_teacher(self):
        env = fixture()
        env.box_states[0, 2] = .2
        env.box_states[1, 0] = 2.
        env.check_termination()
        self.assertFalse(env.reset_buf.any())
        env.executed_phase[:] = STUDENT_CONTROL
        env.handoff.phase[:] = STUDENT_CONTROL
        env.check_termination()
        self.assertTrue(env.reset_buf.all())
        self.assertEqual(env.student_carry_reason.tolist(), [1, 2])

    def test_grasp_loss_grace_flicker_and_independent_environments(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        update(env)
        env.contact_forces[0, [7, 8]] = 0
        update(env)
        self.assertFalse(env.student_carry_failure.any())
        for _ in range(14):
            update(env)
        self.assertFalse(env.student_carry_failure.any())
        for _ in range(3):
            update(env)
        self.assertEqual(env.student_carry_failure.tolist(), [True, False])
        env.contact_forces[0, [7, 8], 0] = 2
        update(env)
        self.assertEqual(env.grasp_loss_time[0], 0.)

    def test_observed_handoff_tilt_diagnostic_only_and_quaternion_sign_invariant(self):
        env = fixture()
        env.box_states[:, 3:7] = quat(roll=1.7434)
        env.executed_phase[:] = STUDENT_CONTROL
        update(env)
        torch.testing.assert_close(env.student_carry_metrics["box_tilt"], torch.full((2,), 1.7434), atol=1e-6, rtol=0)
        self.assertTrue(env.student_carry_metrics["box_tilt_exceeded"].all())
        self.assertFalse(env.student_carry_failure.any())
        env.box_states[:, 3:7] *= -1
        update(env)
        torch.testing.assert_close(env.student_carry_metrics["box_tilt"], torch.full((2,), 1.7434), atol=1e-6, rtol=0)
        env.cfg.student_termination.terminate_box_tilt = True
        update(env)
        self.assertTrue(env.student_carry_failure.all())
        self.assertTrue((env.student_carry_reason == 8).all())

    def test_student_failure_and_timeout_stay_separate_and_diagnostics_pre_reset(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.episode_length_buf[:] = 3001
        env.box_states[0, 2] = .2
        env.check_termination()
        info = env.extras["handoff"]
        self.assertEqual(info["failure"].tolist(), [True, False])
        self.assertTrue(info["timeout"].all())
        self.assertEqual(info["student/termination_reason"].tolist(), [33, 32])
        before = info["box_height"].clone()
        env.box_states[:] = 0.
        torch.testing.assert_close(info["box_height"], before)


if __name__ == "__main__":
    unittest.main()
