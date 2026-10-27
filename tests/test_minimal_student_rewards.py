"""Numerical contracts of the minimal Student objective and production path."""
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from test_student_carry import fixture, quat, raw_reward
from test_mixed_handoff import MATH, STUDENT_CONTROL
from handoff_regression_support import regression_environment, apply_sample
from test_mixed_handoff_training import (
    TensorEnvironment, MixedHandoffOnPolicyRunner, actor_critic, configuration, CONTRACT, Expert,
)


SCALES = dict(carry_lin_vel_tracking=3., carry_yaw_vel_tracking=2.5, carry_bilateral_contact=.5,
              lin_vel_z=-1., ang_vel_xy=-.05, orientation=-1., dof_acc=-1e-7,
              action_rate=-.03, torques=-1e-4, dof_vel=-2e-4, dof_pos_limits=-5.,
              dof_vel_limits=-1e-3, torque_limits=-.03)


class MinimalRewardTests(unittest.TestCase):
    def test_exact_scales_native_regularizers_and_removed_dependencies(self):
        env = fixture()
        self.assertEqual(env.student_reward_scales, {k: v*env.dt for k, v in SCALES.items()})
        for name in ("dof_acc", "action_rate", "torques", "dof_vel", "dof_pos_limits",
                     "dof_vel_limits", "torque_limits"):
            function = env.student_reward_functions[env.student_reward_names.index(name)]
            self.assertIs(function.__func__, getattr(type(env).__mro__[1], "_reward_" + name))
        self.assertEqual({name for name in dir(env.cfg.student_rewards) if not name.startswith("_")},
                         {"scales", "carry_lin_vel_sigma", "carry_yaw_vel_sigma"})
        for name in ("previous_hand_box_pos", "previous_box_relative_pos", "student_history_valid",
                     "carry_arm_indices", "carry_hip_indices", "carry_waist_indices", "hand_slip",
                     "box_relative_pos", "box_relative_velocity", "carry_feet_width"):
            self.assertFalse(hasattr(env, name), name)

    def test_stability_matches_native_formulas_for_tilted_moving_body(self):
        env = fixture()
        pelvis = env.rigid_body_states[:, 0]
        pelvis[:, 3:7] = quat(.4, -.5, .7)
        pelvis[:, 7:10] = torch.tensor([.7, -.3, .2])
        pelvis[:, 10:13] = torch.tensor([.3, -.2, .8])
        # Native post_physics_step refreshes these before rewards. Use exactly
        # that path to compare the explicit Student body reference helpers.
        env.base_lin_vel = MATH.quat_rotate_inverse(pelvis[:, 3:7], pelvis[:, 7:10])
        env.base_ang_vel = MATH.quat_rotate_inverse(pelvis[:, 3:7], pelvis[:, 10:13])
        env.projected_gravity = MATH.quat_rotate_inverse(pelvis[:, 3:7], env.gravity_vec)
        for name in ("lin_vel_z", "ang_vel_xy", "orientation"):
            expected = getattr(type(env).__mro__[1], "_reward_" + name)(env)
            torch.testing.assert_close(raw_reward(env, name), expected, atol=0, rtol=0)

    def test_debug_and_normal_rewards_contacts_failures_and_episode_sums_match(self):
        outputs = []
        for detailed in (False, True):
            env = regression_environment()
            env.detailed_diagnostics = detailed
            env.cfg.diagnostics.detailed = detailed
            if not detailed:
                for name in ("teacher_last_action", "student_first_action", "reset_count", "torso_index"):
                    delattr(env, name)
            frames = []
            for step in range(7):
                apply_sample(env, step)
                env.check_termination()
                env.compute_reward()
                diag = env.extras["handoff"]
                if not detailed:
                    self.assertFalse(any(k.startswith(("teacher_reward_", "student_reward_")) for k in diag))
                    self.assertNotIn("student/box_tilt", diag)
                    self.assertNotIn("action_discontinuity", diag)
                frames.append((env.rew_buf.clone(), env.reset_buf.clone(), env.handoff.phase.clone(),
                               env.student_carry_reason.clone(), env.hand_contact_filt.clone()))
                env.last_actions.copy_(env.actions)
                env.last_dof_vel.copy_(env.dof_vel)
            outputs.append((frames, copy.deepcopy(env.student_episode_sums)))
        for a, b in zip(outputs[0][0], outputs[1][0]):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        for key in SCALES:
            torch.testing.assert_close(outputs[0][1][key], outputs[1][1][key], atol=0, rtol=0)

    def test_one_contact_threshold_for_gate_reward_and_grasp_loss(self):
        env = fixture()
        env.handoff.settings.contact_force = 3.  # Fixture contact force = 2N.
        env.check_termination()
        self.assertFalse(env.handoff.phase.any())
        self.assertFalse(env.hand_contact_filt.any())
        self.assertEqual(raw_reward(env, "carry_bilateral_contact").sum(), 0.)
        env.executed_phase[:] = STUDENT_CONTROL
        for _ in range(17):
            env.check_termination()
        self.assertTrue((env.student_carry_reason == 4).all())

    def test_absent_posture_terms_cannot_enter_student_total(self):
        env = fixture()
        env.executed_phase[:] = STUDENT_CONTROL
        env.check_termination()
        expected = sum(raw_reward(env, name) * scale * env.dt for name, scale in SCALES.items())
        env.compute_reward()
        torch.testing.assert_close(env.rew_buf, expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(set(env.student_episode_sums), set(SCALES))

    def test_minimal_rewards_feed_bounded_real_ppo_amp_and_viewer_checkpoint(self):
        class RewardEnvironment(TensorEnvironment):
            def __init__(self):
                super().__init__()
                self.reward_env = fixture()
                self.reward_env.detailed_diagnostics = False

            def step(self, actions):
                result = list(super().step(actions))
                env = self.reward_env
                env.executed_phase.copy_(result[4]["handoff"]["executed_phase"])
                env.handoff.phase.copy_(env.executed_phase)
                env.actions.copy_(actions)
                env.check_termination()
                env.compute_reward()
                result[2] = env.rew_buf.clone()
                return tuple(result)

        torch.set_num_threads(1)
        for amp in (False, True):
            with self.subTest(amp=amp), tempfile.TemporaryDirectory() as folder, \
                    contextlib.redirect_stdout(io.StringIO()):
                path = Path(folder) / "teacher.pt"
                torch.manual_seed(17)
                torch.save(dict(model_state_dict=actor_critic().state_dict(), scope1_contract=CONTRACT), path)
                runner = MixedHandoffOnPolicyRunner(RewardEnvironment(), configuration(amp, 18), path,
                                                    CONTRACT, expert=Expert() if amp else None)
                runner.learn(2, max_optimizer_updates=2)
                self.assertEqual(runner.alg.optimizer_updates, 2)
                self.assertGreater(runner.student_steps, 0)
                self.assertGreater(runner.amp_steps, 0)
                runner.assert_teacher_frozen()
                checkpoint = Path(folder) / "student.pt"
                runner.save(checkpoint)
                from legged_gym.mixed_handoff import load_actor
                actor, _ = load_actor(checkpoint, CONTRACT)
                for actual, expected in zip(actor.parameters(), runner.alg.actor_critic.actor.parameters()):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
