"""Scope 2 tests: real tensor/quaternion math and phase-aware environment hooks."""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from test_mixed_handoff import (
    environment_fixture, HandoffSettings, STUDENT_CONTROL,
    StudentCarry, StudentCarryCfg, MATH,
)


def quat(roll=0., pitch=0., yaw=0.):
    return MATH.quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw]))[0]


def fixture():
    env = environment_fixture(HandoffSettings(stable_duration_s=.02))
    # Dedicated torso body permits actual relative-rotation tests.
    env.student_carry.bodies["torso"] = 6
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


def student_step(env):
    env.executed_phase[:] = STUDENT_CONTROL
    env.handoff.phase[:] = STUDENT_CONTROL
    env.check_termination()
    env.compute_reward()
    return env.extras["handoff"]


def update(env):
    carry = env.student_carry
    # Geometry/contact tests use known near-box palm coordinates.
    carry.update(env, env.executed_phase == STUDENT_CONTROL,
                 torch.zeros(2, 2), env.box_states[:, 2] - .125)
    return carry


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
        env.cfg.student.scales = {"action_rate": -.03}
        env.check_termination()
        env.compute_reward()
        expected = -(.4**2)*29*.03*env.dt
        self.assertAlmostEqual(env.rew_buf[1].item(), expected, places=6)
        self.assertEqual(env.student_carry.steps.tolist(), [0., 1.])
        self.assertEqual(env.extras["handoff"]["student_reward_raw/action_rate"][0], 0.)

    def test_positive_clipping_and_teacher_termination_order(self):
        env = fixture()
        env.cfg.rewards.only_positive_rewards = True
        env.reward_functions = [lambda: torch.full((2,), -2.)]
        env.reward_names = ["walk_task"]
        env.reward_scales = {"walk_task": env.dt, "termination": -1.}
        env.episode_sums = {k: torch.zeros(2) for k in env.reward_scales}
        env.cfg.student.scales = {"action_rate": -1.}
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
        carry = env.student_carry
        expected = carry.total_sum / 2
        torch.testing.assert_close(env.extras["handoff"]["episode_student_mean_reward"], expected)
        self.assertEqual(carry.steps.tolist(), [2., 2.])

    def test_ordinary_reset_clears_history_and_logs_only_student_denominator(self):
        env = fixture()
        carry = env.student_carry
        carry.steps[:] = torch.tensor([2., 0.])
        carry.total_sum[:] = torch.tensor([4., 0.])
        carry.history_valid[:] = True
        carry.last_box_contact[:] = True
        carry.grasp_loss_time[:] = .2
        carry.episode_sums["action_rate"][:] = torch.tensor([-.2, 0.])

        def base_reset(instance, ids):
            instance.extras["episode"] = {}

        with patch.object(type(env).__mro__[1], "reset_idx", base_reset):
            env.reset_idx(torch.tensor([0, 1]))
        episode = env.extras["episode"]
        self.assertEqual(episode["student/mean_step_reward/total"], 2.)
        self.assertEqual(episode["student/episodes_without_student"], 1.)
        self.assertEqual(episode["student/episodes_with_student"], 1.)
        self.assertAlmostEqual(episode["student/mean_step_reward/action_rate"].item(), -.1)
        self.assertFalse(carry.history_valid.any())
        self.assertFalse(carry.last_box_contact.any())
        self.assertEqual(carry.grasp_loss_time.sum(), 0.)
        self.assertEqual(carry.steps.sum(), 0.)

    def test_nonfinite_or_wrong_shape_student_rewards_fail_explicitly(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.check_termination()
        env.student_carry.terms["carry_lin_vel_tracking"][:] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            env.compute_reward()
        env.student_carry.terms["carry_lin_vel_tracking"] = torch.zeros(2, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            env.compute_reward()


class CoordinateTests(unittest.TestCase):
    def test_forward_lateral_and_yaw_tracking(self):
        for command in ((.6, 0., 0.), (0., -.4, 0.), (0., 0., .5)):
            env = fixture()
            env.executed_command[:] = torch.tensor(command)
            env.rigid_body_states[:, 0, 7:9] = torch.tensor(command[:2])
            env.rigid_body_states[:, 0, 12] = command[2]
            carry = update(env)
            torch.testing.assert_close(carry.terms["carry_lin_vel_tracking"], torch.ones(2))
            torch.testing.assert_close(carry.terms["carry_yaw_vel_tracking"], torch.ones(2))

    def test_yaw_frame_ignores_roll_pitch_and_yaw_rate_uses_world_z(self):
        env = fixture()
        env.rigid_body_states[:, 0, 3:7] = quat(.4, -.5, math.pi/2)
        env.rigid_body_states[:, 0, 7:10] = torch.tensor([0., .6, .3])
        env.rigid_body_states[:, 0, 10:13] = torch.tensor([1., 2., .7])
        env.executed_command[:] = torch.tensor([.6, 0., .7])
        carry = update(env)
        torch.testing.assert_close(carry.metrics["measured_velocity_yaw"], torch.tensor([[.6, 0.], [.6, 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(carry.terms["carry_lin_vel_tracking"], torch.ones(2))
        torch.testing.assert_close(carry.metrics["measured_yaw_rate_world"], torch.full((2,), .7))
        full_local = MATH.quat_rotate_inverse(env.rigid_body_states[:, 0, 3:7], env.rigid_body_states[:, 0, 7:10])
        self.assertFalse(torch.allclose(full_local[:, :2], carry.metrics["measured_velocity_yaw"]))

    def test_zero_stillness_and_lateral_yaw_air_time(self):
        env = fixture()
        env.feet_air_time[:] = 1.
        env.first_contacts[:] = True
        carry = update(env)
        torch.testing.assert_close(carry.terms["zero_command_stillness"], torch.ones(2))
        self.assertEqual(carry.terms["feet_air_time"].sum(), 0.)
        env.executed_command[0, 1] = .2
        env.executed_command[1, 2] = .2
        carry = update(env)
        self.assertTrue(carry.metrics["moving_command"].all())
        self.assertTrue((carry.terms["feet_air_time"] > 0).all())
        self.assertEqual(carry.terms["zero_command_stillness"].sum(), 0.)


class ContactPostureTests(unittest.TestCase):
    def test_bilateral_filter_and_force_attribution(self):
        env = fixture()
        carry = update(env)
        self.assertTrue((carry.terms["carry_bilateral_contact"] == 1).all())
        env.contact_forces[0, 8] = 0.
        update(env)  # One-frame OR filter preserves the contact for this frame.
        self.assertEqual(carry.terms["carry_bilateral_contact"][0], 1.)
        update(env)
        self.assertEqual(carry.terms["carry_bilateral_contact"][0], 0.)
        self.assertEqual(carry.terms["carry_bilateral_contact"][1], 1.)
        carry.update(env, torch.ones(2, dtype=torch.bool), torch.ones(2, 2), torch.ones(2))
        carry.update(env, torch.ones(2, dtype=torch.bool), torch.ones(2, 2), torch.ones(2))
        self.assertFalse(carry.metrics["filtered_box_contact"].any())
        self.assertTrue(carry.metrics["raw_hand_contact"][1].all())

    def test_random_box_faces_and_surface_violation(self):
        env = fixture()
        env._box_size[:] = torch.tensor([[.3, .4, .2], [.4, .6, .3]])
        env.rigid_body_states[:, 7, :3] = env.box_states[:, :3]
        env.rigid_body_states[:, 8, :3] = env.box_states[:, :3]
        env.rigid_body_states[:, 7, 1] += env._box_size[:, 1]/2
        env.rigid_body_states[:, 8, 1] -= env._box_size[:, 1]/2
        carry = update(env)
        torch.testing.assert_close(carry.terms["carry_hand_box_surface"], torch.ones(2))
        env.rigid_body_states[0, 7, 1] += .2
        update(env)
        self.assertLess(carry.terms["carry_hand_box_surface"][0], 1.)
        self.assertEqual(carry.terms["carry_hand_box_surface"][1], 1.)

    def test_first_student_derivative_invalid_then_real_torso_frame_motion(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.rigid_body_states[:, 6, 3:7] = quat(yaw=math.pi/2)
        carry = update(env)
        self.assertFalse(carry.metrics["derivative_valid"].any())
        self.assertEqual(carry.metrics["hand_slip"].abs().sum(), 0.)
        self.assertEqual(carry.terms["carry_relative_velocity"].sum(), 0.)
        env.box_states[:, 1] += .02
        update(env)
        torch.testing.assert_close(carry.metrics["box_relative_velocity"][:, 0], torch.ones(2), atol=1e-5, rtol=0)
        carry.reset([0])
        update(env)
        self.assertEqual(carry.metrics["derivative_valid"].tolist(), [False, True])
        self.assertEqual(carry.metrics["box_relative_velocity"][0].abs().sum(), 0.)

    def test_slip_excludes_box_normal_motion_and_penalizes_tangent_motion(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        carry = update(env)
        update(env)
        torch.testing.assert_close(carry.terms["carry_hand_slip"], torch.ones(2))
        env.rigid_body_states[:, [7, 8], 1] += .1  # Box Y is face normal.
        update(env)
        torch.testing.assert_close(carry.metrics["hand_slip"], torch.zeros(2, 2))
        env.rigid_body_states[0, [7, 8], 0] += .1
        update(env)
        self.assertLess(carry.terms["carry_hand_slip"][0], 1.)
        self.assertEqual(carry.terms["carry_hand_slip"][1], 1.)

    def test_feasible_regions_and_joint_order(self):
        env = fixture()
        carry = env.student_carry
        env.dof_pos[:, carry.waist_ids] = torch.tensor(env.cfg.student.carry_waist_reference_target)
        env.rigid_body_states[:, 6, 3:7] = torch.tensor(env.cfg.student.carry_torso_pelvis_reference_quat)
        env.rigid_body_states[:, 1, 1] = .1
        env.rigid_body_states[:, 2, 1] = -.1
        env.rigid_body_states[:, 3, 1] = .1
        env.rigid_body_states[:, 4, 1] = -.1
        update(env)
        for name in ("carry_hip_posture", "carry_waist_reference", "carry_torso_pelvis_alignment",
                     "carry_feet_width", "carry_knee_width", "carry_foot_heading"):
            torch.testing.assert_close(carry.terms[name], torch.ones(2))
        previous_obs = env.obs_buf.clone()
        env.dof_pos[0, carry.hip_ids] += 1.
        env.dof_pos[0, carry.waist_ids] += 1.
        env.rigid_body_states[0, 1, 1] = 1.
        update(env)
        for name in ("carry_hip_posture", "carry_waist_reference", "carry_feet_width"):
            self.assertLess(carry.terms[name][0], carry.terms[name][1])
        torch.testing.assert_close(env.obs_buf, previous_obs)
        for field in ("carry_hip_joint_names", "carry_waist_joint_names"):
            cfg = StudentCarryCfg()
            setattr(cfg, field, tuple(reversed(getattr(cfg, field))))
            with self.assertRaisesRegex(ValueError, "joint order"):
                StudentCarry(env, cfg, carry.bodies)


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
        self.assertEqual(env.student_carry.reason.tolist(), [1, 2])

    def test_grasp_loss_grace_flicker_and_independent_environments(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        update(env)
        env.contact_forces[0, [7, 8]] = 0
        update(env)
        self.assertFalse(env.student_carry.failure.any())
        for _ in range(14):
            update(env)
        self.assertFalse(env.student_carry.failure.any())
        for _ in range(3):
            update(env)
        self.assertEqual(env.student_carry.failure.tolist(), [True, False])
        env.contact_forces[0, [7, 8], 0] = 2
        update(env)
        self.assertEqual(env.student_carry.grasp_loss_time[0], 0.)

    def test_observed_handoff_tilt_diagnostic_only_and_quaternion_sign_invariant(self):
        env = fixture()
        env.box_states[:, 3:7] = quat(roll=1.7434)
        env.executed_phase[:] = STUDENT_CONTROL
        carry = update(env)
        torch.testing.assert_close(carry.metrics["box_tilt"], torch.full((2,), 1.7434), atol=1e-6, rtol=0)
        self.assertTrue(carry.metrics["box_tilt_exceeded"].all())
        self.assertFalse(carry.failure.any())
        env.box_states[:, 3:7] *= -1
        update(env)
        torch.testing.assert_close(carry.metrics["box_tilt"], torch.full((2,), 1.7434), atol=1e-6, rtol=0)
        env.cfg.student.terminate_box_tilt = True
        update(env)
        self.assertTrue(carry.failure.all())
        self.assertTrue((carry.reason == 8).all())

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
        before = info["student/box_relative_position"].clone()
        env.box_states[:] = 0.
        torch.testing.assert_close(info["student/box_relative_position"], before)


if __name__ == "__main__":
    unittest.main()
