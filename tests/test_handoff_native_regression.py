"""Compare native environment outputs to data frozen before the refactor."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from handoff_regression_support import regression_environment, apply_sample, snapshot
from test_student_carry import fixture


class NativeRegressionTests(unittest.TestCase):
    def assert_snapshot(self, actual, expected, path="snapshot"):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.assert_snapshot(actual[key], expected[key], path + "/" + key)
        else:
            exact = any(part in path for part in (
                "/raw_student/", "/metrics/", "/teacher_sums/",
                "/teacher_reward_raw/", "/teacher_reward_weighted/", "/student_reward_raw/"))
            torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected),
                                       atol=0. if exact else 5e-8, rtol=0. if exact else 1e-7, msg=path)

    def test_all_terms_metrics_boundaries_and_episode_statistics_match_scope2(self):
        golden = json.loads((Path(__file__).parent / "fixtures/mixed_handoff_scope2.json").read_text())
        self.assertEqual(golden["baseline_commit"], "6295c21694c793ff7f4f2f445f1331eba2313754")
        env = regression_environment()
        for step, expected in enumerate(golden["samples"]):
            with self.subTest(step=step):
                apply_sample(env, step)
                env.check_termination()
                env.compute_reward()
                raw = {name: function() for name, function in zip(
                    env.student_reward_names, env.student_reward_functions)}
                actual = snapshot(env, raw, env.student_carry_metrics, env.student_steps,
                                  env.student_total_sum, env.student_episode_sums)
                self.assert_snapshot(actual, expected)
                env.last_actions.copy_(env.actions)
                env.last_dof_vel.copy_(env.dof_vel)

        def base_reset(instance, ids):
            instance.extras["episode"] = {}

        with patch.object(type(env).__mro__[1], "reset_idx", base_reset):
            env.reset_idx(torch.tensor([0, 1]))
        self.assert_snapshot({k: v.tolist() for k, v in env.extras["episode"].items()}, golden["episode"])

    def test_native_preparation_dt_once_disabled_terms_and_teacher_separation(self):
        env = fixture()
        before = dict(env.reward_scales)
        self.assertEqual(env.cfg.student_rewards.scales.carry_lin_vel_tracking, 3.)
        for _ in range(2):
            env._prepare_student_reward_function()
            self.assertEqual(env.student_reward_scales["carry_lin_vel_tracking"], 3.*env.dt)
            self.assertEqual(env.cfg.student_rewards.scales.carry_lin_vel_tracking, 3.)
            self.assertEqual(env.reward_scales, before)
        env.cfg.student_rewards.scales.carry_hand_slip = 0.
        env._prepare_student_reward_function()
        self.assertNotIn("carry_hand_slip", env.student_reward_names)
        self.assertNotIn("carry_hand_slip", env.student_episode_sums)
        self.assertEqual(set(env.student_reward_names), set(env.student_reward_scales))
        cls = type(env)
        for name in ("lin_vel_z", "ang_vel_xy", "orientation", "feet_air_time",
                     "feet_slip", "feet_stumble", "feet_contact_forces"):
            self.assertNotIn("_reward_" + name, cls.__dict__)
            function = env.student_reward_functions[env.student_reward_names.index(name)]
            self.assertEqual(function.__name__, "_reward_student_" + name)
        self.assertFalse((Path(__file__).resolve().parents[1] / "legged_gym/legged_gym/student_carry.py").exists())

    def test_native_buffer_and_reward_initialization_order(self):
        from test_mixed_handoff import MixedHandoffCfg
        env = fixture()
        env.cfg.handoff = MixedHandoffCfg().handoff
        env.cfg.control = NS(upper_body_link="pelvis")
        env.num_obs, env.num_privileged_obs, env.num_actions = 738, 126, 29
        env.envs = env.actor_handles = [0]
        bodies = dict(pelvis=0, torso_link=6, left_rubber_hand=7, right_rubber_hand=8,
                      left_palm_link=7, right_palm_link=8, left_ankle_pitch_link=1,
                      right_ankle_pitch_link=2, left_knee_link=3, right_knee_link=4)
        env.gym = NS(find_actor_rigid_body_index=lambda e, a, name, domain: bodies[name])
        with patch.object(type(env).__mro__[1], "_init_buffers"), patch.dict(
                sys.modules, isaacgym=NS(gymapi=NS(DOMAIN_ENV=1))):
            env._init_buffers()
        env.reward_scales = {"action_rate": -.03}
        env._prepare_reward_function()
        self.assertEqual(env.reward_names, ["action_rate"])
        self.assertEqual(env.reward_scales["action_rate"], -.03*env.dt)
        self.assertEqual(len(env.student_reward_functions), 31)
        self.assertEqual(tuple(env.previous_hand_box_pos.shape), (2, 2, 3))
        self.assertEqual(env.student_body_indices["torso"], 6)

    def test_all_configuration_values_and_inherited_teacher_config_unchanged(self):
        from test_mixed_handoff import MixedHandoffCfg, config_namespace
        golden = json.loads((Path(__file__).parent / "fixtures/mixed_handoff_scope2.json").read_text())
        env = fixture()
        class_to_dict = env._prepare_student_reward_function.__func__.__globals__["class_to_dict"]
        for name, expected in golden["config"].items():
            group = (env.cfg.student_rewards if hasattr(env.cfg.student_rewards, name)
                     else env.cfg.student_termination)
            value = getattr(group, name)
            if name == "scales":
                value = class_to_dict(value)
            self.assertEqual(json.loads(json.dumps(value)), expected, name)
        self.assertEqual(class_to_dict(MixedHandoffCfg().rewards),
                         class_to_dict(config_namespace["G1Cfg"]().rewards))


if __name__ == "__main__":
    unittest.main()
