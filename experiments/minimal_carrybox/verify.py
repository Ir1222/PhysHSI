"""CPU checks of real config/methods/data; no Isaac Gym simulator is mocked as real.

Run from the repository root: python experiments/minimal_carrybox/verify.py
Imports needing Isaac Gym are bypassed with AST extraction. Tensor math,
configuration classes, observation/reset/reward methods and MotionLib are original.
Only simulator-facing reset calls are stubbed in the episode-reset check.
"""
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "legged_gym/legged_gym/envs"
BASELINE = "07bf22b"


def extract(path, namespace, names=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
                 and (names is None or getattr(n, "name", None) in names)]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


base = extract(ENV / "base/base_config.py", {"inspect": __import__("inspect")})
base = extract(ENV / "base/legged_robot_config.py", {"BaseConfig": base["BaseConfig"], "np": np})
config_ns = extract(ENV / "g1/carrybox_config.py", dict(base))
spec = importlib.util.spec_from_file_location("carrybox_test_torch_utils", ROOT / "legged_gym/legged_gym/utils/torch_utils.py")
tu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tu)
math_ns = {name: getattr(tu, name) for name in dir(tu) if not name.startswith("_")}
helper = ROOT / "legged_gym/legged_gym/utils/helpers.py"
class_to_dict = extract(helper, {}, {"class_to_dict"})["class_to_dict"]
robot_ns = extract(ENV / "g1/carrybox.py", dict(math_ns, BaseTask=object,
                   LeggedRobotCfg=base["LeggedRobotCfg"], class_to_dict=class_to_dict))
Robot = robot_ns["LeggedRobot"]


def original(path):
    return subprocess.check_output(["git", "show", BASELINE + ":" + path.as_posix()], cwd=ROOT).decode("utf-8")


class BaselineChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = config_ns["G1Cfg"]()

    def test_configuration_and_rewards(self):
        cfg = self.cfg
        self.assertEqual(cfg.asset.box.skill, ["loco", "pickUp"])
        self.assertEqual(cfg.asset.box.skill_init_prob, [0.8, 0.2])
        self.assertEqual((cfg.asset.box.reset_mode, cfg.asset.box.hybrid_init_prob), ("hybrid", 0.8))
        self.assertEqual((cfg.env.num_actor_obs, cfg.env.num_privileged_obs,
                          cfg.env.num_task_obs, cfg.env.num_actions, cfg.env.num_actor_history), (738, 126, 15, 29, 6))
        runner = config_ns["G1CfgPPO"]().runner
        self.assertFalse(runner.resume)
        self.assertIsNone(runner.resume_path)
        self.assertEqual(runner.experiment_name, "minimal_carrybox")
        obj = Robot.__new__(Robot)
        obj.cfg, obj.sim_params, obj.num_envs, obj.device = cfg, SimpleNamespace(dt=cfg.sim.dt), 4, "cpu"
        obj._parse_cfg(cfg)
        obj._prepare_reward_function()
        expected = {"walk_task": 1., "carryup_task": 1., "dof_acc": -1e-7,
                    "action_rate": -.03, "torques": -1e-4, "dof_vel": -2e-4,
                    "dof_pos_limits": -5., "dof_vel_limits": -1e-3, "torque_limits": -.03}
        self.assertEqual(set(obj.reward_names), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(obj.reward_scales[name], value * .02)
        self.assertEqual((cfg.rewards.scales.relocation_task, cfg.rewards.scales.standup_task), (0., 0.))
        print("\nNonzero config scales:", expected)
        print("Parsed effective scales (dt=0.02):", obj.reward_scales)
        print("Expected reset distribution: default=20%, loco RSI=64%, pickUp RSI=16%")

    def make_robot(self):
        obj = Robot.__new__(Robot)
        obj.cfg, obj.box_cfg, obj.camera_cfg = self.cfg, self.cfg.asset.box, self.cfg.asset.camera
        obj.num_envs, obj.device, obj.upper_body_index = 4, "cpu", 0
        obj.root_states = torch.zeros(4, 13)
        obj.box_states = torch.zeros(4, 13)
        obj.box_states[:, 0] = torch.tensor([1., .5, .7, 2.])
        obj.box_states[:, 6] = 1.
        obj.rigid_body_states = obj.root_states[:, None, :].clone()
        obj.rigid_body_states[:, :, 6] = 1.
        obj._box_size = torch.tensor([[.3, .3, .25]]).repeat(4, 1)
        obj.has_reached_box = torch.zeros(4, dtype=torch.bool)
        obj.robot2object_dist = torch.zeros(4)  # deliberately stale after reset
        obj.thresh_tag = torch.ones(4)
        obj.can_see_tag = torch.tensor([False, True, False, True])
        obj.has_seen_tag = torch.ones(4, dtype=torch.bool)
        obj.far_pos_offset = torch.ones(4, 3) * .1
        obj.default_zero_pos = torch.zeros(3)
        obj.default_quat = torch.tensor([0., 0., 0., 1.])
        obj.success_buf = torch.ones(4, dtype=torch.bool)
        return obj

    def test_observation_noise_latch_and_episode_reset(self):
        for noise in (False, True):
            obj = self.make_robot()
            obj.add_noise = noise
            actor, critic = obj.compute_task_observations()
            self.assertEqual(tuple(actor.shape), (4, 15))
            self.assertEqual(tuple(critic.shape), (4, 15))
            self.assertTrue(torch.isfinite(actor).all() and torch.isfinite(critic).all())
            torch.testing.assert_close(actor[:, -3:], critic[:, -3:])
            torch.testing.assert_close(actor[:, -3:], torch.tensor([[.85, 0., 0.], [0., 0., 0.], [.85, 0., 0.], [.85, 0., 0.]]))
            obj.box_states[:, 0] = 3.  # move away: stop must remain latched
            torch.testing.assert_close(obj.compute_task_observations()[0][1, -3:], torch.zeros(3))

            # Exercise the actual episode reset, isolating simulator calls only.
            obj.cfg.play_dataset = False
            for key in ("randomize_kp", "randomize_kd", "randomize_actuation_offset", "randomize_motor_strength", "delay"):
                setattr(obj.cfg.domain_rand, key, False)
            for name in ("_reset_actors", "_reset_boxes", "_reset_task", "_reset_env_tensors"):
                setattr(obj, name, lambda ids: None)
            for name in ("last_actions", "last_last_actions", "last_dof_vel", "last_torques", "feet_air_time", "joint_powers", "reset_buf", "episode_length_buf", "time_out_buf", "hfov_rad", "vfov_rad", "facing_angle"):
                setattr(obj, name, torch.zeros(4))
            obj.dof_pos, obj.default_dof_pos, obj.delay_buffer = torch.zeros(4, 29), torch.zeros(29), torch.zeros(5, 4, 29)
            obj.episode_sums, obj.extras = {}, {}
            obj.reset_idx(torch.tensor([1]))
            self.assertFalse(obj.has_reached_box[1])
            torch.testing.assert_close(obj.compute_task_observations()[0][1, -3:], torch.tensor([.85, 0., 0.]))
            obj.box_states[1, 0] = .1  # nearby RSI state should stop immediately
            torch.testing.assert_close(obj.compute_task_observations()[0][1, -3:], torch.zeros(3))

            # Exercise actual observation concatenation and terminal critic layout.
            obj.obs_scales = obj.cfg.normalization.obs_scales
            for name, size in (("base_ang_vel", 3), ("projected_gravity", 3), ("dof_vel", 29), ("end_effector_pos", 15), ("actions", 29), ("base_lin_vel", 3)):
                setattr(obj, name, torch.zeros(4, size))
            obj.obs_buf, obj.num_one_step_actor_obs = torch.zeros(4, 738), 123
            obj.noise_scale_vec = torch.zeros(108)
            obj.compute_observations()
            self.assertEqual(tuple(obj.obs_buf.shape), (4, 738))
            self.assertEqual(tuple(obj.privileged_obs_buf.shape), (4, 126))
            terminal = obj.compute_termination_observations(torch.tensor([0, 1]))
            self.assertEqual(tuple(terminal.shape), (2, 126))
            torch.testing.assert_close(obj.obs_buf[:, -3:], obj.privileged_obs_buf[:, -3:])

    def test_real_motionlib_and_expert_sampling(self):
        old = yaml.safe_load((ROOT / "legged_gym/resources/config/carrybox.yaml").read_text())["motions"]
        path = Path(self.cfg.dataset.motion_file.format(LEGGED_GYM_ROOT_DIR=ROOT / "legged_gym"))
        motions = yaml.safe_load(path.read_text())["motions"]
        self.assertEqual(list(motions), ["loco", "pickUp"])
        for name, entries in motions.items():
            self.assertEqual(entries, old[name])
            self.assertTrue(all((path.parent / e["file"]).is_file() for e in entries))
        ns = extract(ENV / "motionlib/motionlib_carrybox.py", dict(math_ns, os=__import__("os"), yaml=yaml, torch_utils=tu))
        mapping = ROOT / "legged_gym/resources/config/joint_id.txt"
        dofs = [line.split()[1] for line in mapping.read_text().splitlines()]
        lib = ns["MotionLib"](str(path), str(mapping), dofs, 60, "cpu", 10, [.95, 1.05], .7)
        self.assertEqual(lib.skills, ["loco", "pickUp"])
        self.assertEqual(lib.num_motion, {"loco": 11, "pickUp": 3})
        self.assertEqual(len(lib.motion_weights_tot), 14)
        expert = lib.get_expert_obs(128)
        self.assertEqual(tuple(expert.shape), (128, 600))
        self.assertTrue(torch.isfinite(expert).all())
        for skill in lib.skills:
            ids = lib.sample_motions(skill, 32)
            times = lib.sample_time_rsi(skill, ids)
            self.assertTrue(all(torch.isfinite(v).all() for v in lib.get_motion_state(skill, ids, times)))
            self.assertTrue(torch.isfinite(lib.get_amp_hist_obs(skill, ids, times)).all())
        print("Loaded real MotionLib: 11 loco + 3 pickUp; finite AMP expert batch", tuple(expert.shape))

    def test_original_training_and_reward_code_preserved(self):
        path = ENV / "g1/carrybox.py"
        before = original(path.relative_to(ROOT))
        after = path.read_text(encoding="utf-8")
        def methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
            return {n.name: ast.get_source_segment(source, n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        a, b = methods(before), methods(after)
        changed = {name for name in a if a[name] != b[name]}
        self.assertEqual(changed, {"compute_task_observations", "reset_idx", "_init_buffers"})
        self.assertEqual(set(a), set(b))
        # Every other pre-existing tracked file, including PPO/AMP/data, is identical.
        modified = subprocess.check_output(["git", "diff", "--name-only", BASELINE], cwd=ROOT, text=True).splitlines()
        allowed = {"legged_gym/legged_gym/envs/g1/carrybox.py", "legged_gym/legged_gym/envs/g1/carrybox_config.py"}
        for rel in modified:
            if rel not in allowed:
                tracked_before = subprocess.run(["git", "cat-file", "-e", BASELINE + ":" + rel], cwd=ROOT, capture_output=True)
                self.assertNotEqual(tracked_before.returncode, 0, "Unexpected original file modified: " + rel)
        print("All original robot methods preserved except observation + latch init/reset; original training/data untouched")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    unittest.main(verbosity=2)
