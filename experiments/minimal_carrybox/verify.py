"""CPU checks of real config/methods/data; no Isaac Gym simulator is mocked as real.

Run from the repository root: python experiments/minimal_carrybox/verify.py
Imports needing Isaac Gym are bypassed with AST extraction. Tensor math,
configuration classes, observation/reset/reward methods and MotionLib are original.
Only simulator-facing reset calls are stubbed in the episode-reset check.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "legged_gym/legged_gym/envs"
MANIFEST = json.loads(Path(__file__).with_name("original_manifest.json").read_text(encoding="utf-8"))


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


class BaselineChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = config_ns["G1Cfg"]()

    def test_configuration_and_rewards(self):
        cfg = self.cfg
        self.assertEqual(cfg.asset.box.skill, ["loco", "pickUp"])
        self.assertEqual(len(cfg.asset.box.skill_init_prob), len(cfg.asset.box.skill))
        self.assertTrue(all(p >= 0 for p in cfg.asset.box.skill_init_prob))
        self.assertAlmostEqual(sum(cfg.asset.box.skill_init_prob), 1.)
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
        expected = {"walk_task": 1., "carryup_task": 1., "pickup_success": 1., "dof_acc": -1e-7,
                    "action_rate": -.03, "torques": -1e-4, "dof_vel": -2e-4,
                    "dof_pos_limits": -5., "dof_vel_limits": -1e-3, "torque_limits": -.03}
        self.assertEqual(set(obj.reward_names), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(obj.reward_scales[name], value * .02)
        self.assertEqual((cfg.rewards.scales.relocation_task, cfg.rewards.scales.standup_task), (0., 0.))
        print("\nNonzero config scales:", expected)
        print("Parsed effective scales (dt=0.02):", obj.reward_scales)
        ref_prob = cfg.asset.box.hybrid_init_prob
        print("Expected reset distribution: default={:.0%}, loco RSI={:.0%}, pickUp RSI={:.0%}".format(
            1 - ref_prob, *(ref_prob * p for p in cfg.asset.box.skill_init_prob)))

    def make_robot(self):
        obj = Robot.__new__(Robot)
        obj.cfg, obj.box_cfg, obj.camera_cfg = self.cfg, self.cfg.asset.box, self.cfg.asset.camera
        obj.num_envs, obj.device, obj.upper_body_index = 4, "cpu", 0
        obj.root_states = torch.zeros(4, 13)
        obj.box_states = torch.zeros(4, 13)
        obj.box_states[:, 0] = torch.tensor([1., .5, .7, 2.])
        obj.box_states[:, 6] = 1.
        obj.episode_max_box_height = obj.box_states[:, 2].clone()
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
            obj.episode_max_box_height[:] = torch.tensor([.8, .9, 1., 1.1])
            def reset_boxes(ids):
                self.assertAlmostEqual(obj.extras["episode"]["box_height"].item(), .9)
                obj.box_states[ids, 2] = .4
            obj._reset_boxes = reset_boxes
            for name in ("last_actions", "last_last_actions", "last_dof_vel", "last_torques", "feet_air_time", "joint_powers", "reset_buf", "episode_length_buf", "time_out_buf", "hfov_rad", "vfov_rad", "facing_angle"):
                setattr(obj, name, torch.zeros(4))
            obj.dof_pos, obj.default_dof_pos, obj.delay_buffer = torch.zeros(4, 29), torch.zeros(29), torch.zeros(5, 4, 29)
            obj.dt = .02
            obj.episode_length_buf[:] = 2
            obj.episode_sums = {"pickup_success": torch.tensor([0., .04, 0., 0.])}
            obj.extras = {}
            obj.reset_idx(torch.tensor([1]))
            self.assertEqual(set(obj.extras["episode"]), {"box_height", "rew_pickup_success"})
            self.assertAlmostEqual(obj.extras["episode"]["rew_pickup_success"].item(), 1.)
            torch.testing.assert_close(obj.episode_max_box_height, torch.tensor([.8, .4, 1., 1.1]))
            self.assertEqual(obj.episode_sums["pickup_success"][1].item(), 0.)
            obj.reset_idx(torch.tensor([], dtype=torch.long))
            torch.testing.assert_close(obj.episode_max_box_height, torch.tensor([.8, .4, 1., 1.1]))
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

    def test_sparse_pickup_and_episode_height(self):
        obj = self.make_robot()
        cases = [(left, right, height) for left in (0., 1., 2.)
                 for right in (0., 1., 2.) for height in (.6, .72, .73)]
        obj.box_states = torch.zeros(len(cases), 13)
        obj.contact_forces = torch.zeros(len(cases), 5, 3)
        obj.hand_colli_indices = torch.tensor([1, 3])
        for i, (left, right, height) in enumerate(cases):
            obj.contact_forces[i, 1, 0] = -left
            obj.contact_forces[i, 3, 2] = right
            obj.box_states[i, 2] = height
        expected = torch.zeros(len(cases))
        expected[-1] = 1.  # only both forces > 1 N and height > 0.72 m
        torch.testing.assert_close(obj._reward_pickup_success(), expected)
        torch.testing.assert_close(obj._reward_pickup_success(), expected)  # per step, no latch
        obj.contact_forces[-1, 1] = torch.tensor([.8, .8, 0.])
        torch.testing.assert_close(obj._reward_pickup_success(), expected)  # vector norm

        # Execute the actual peak init/update statements without Isaac Gym.
        source = (ENV / "g1/carrybox.py").read_text(encoding="utf-8")
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        for name in ("_init_buffers", "post_physics_step"):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            stmt = next(n for n in method.body if isinstance(n, ast.Assign)
                        and "episode_max_box_height" in ast.get_source_segment(source, n.targets[0]))
            code = compile(ast.Module(body=[stmt], type_ignores=[]), name, "exec")
            if name == "_init_buffers":
                exec(code, {"self": obj, "torch": torch})
                obj.box_states[:, 2] += .1
            else:
                exec(code, {"self": obj, "torch": torch})
                obj.box_states[:, 2] -= .2
                exec(code, {"self": obj, "torch": torch})
                self.assertLess(method.body.index(stmt), next(i for i, n in enumerate(method.body)
                    if isinstance(n, ast.Expr) and "self.reset_idx(" in ast.get_source_segment(source, n)))
        torch.testing.assert_close(obj.episode_max_box_height, obj.box_states[:, 2] + .2)
        urdf = Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=ROOT / "legged_gym"))
        hands = [n.attrib["name"] for n in ET.parse(urdf).iter("link")
                 if self.cfg.asset.hand_colli_name in n.attrib["name"]]
        self.assertEqual(hands, ["left_rubber_hand", "right_rubber_hand"])
        self.assertFalse(self.cfg.asset.collapse_fixed_joints)

    def test_real_motionlib_and_expert_sampling(self):
        old = MANIFEST["motions"]
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
        after = path.read_text(encoding="utf-8")
        cls = next(n for n in ast.parse(after).body if isinstance(n, ast.ClassDef))
        a = MANIFEST["robot_methods"]
        b = {n.name: hashlib.sha256(ast.get_source_segment(after, n).encode()).hexdigest()
             for n in cls.body if isinstance(n, ast.FunctionDef)}
        changed = {name for name in a if a[name] != b[name]}
        self.assertEqual(changed, {"compute_task_observations", "reset_idx", "_init_buffers", "post_physics_step"})
        self.assertEqual(set(b) - set(a), {"_reward_pickup_success"})
        self.assertTrue(set(a) <= set(b))
        for rel, expected in MANIFEST["files"].items():
            path = ROOT / rel
            raw = path.read_bytes()
            if path.suffix.lower() in (".py", ".txt", ".urdf", ".yaml"):
                raw = raw.replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, rel)
        print("Original training/data/assets/rewards preserved; only observation, init/reset and peak-height tracking differ")

    def test_carrybox_only_package_and_asset_dependencies(self):
        registered = []
        registry = SimpleNamespace(register=lambda name, *args: registered.append(name))
        extract(ENV / "__init__.py", dict(task_registry=registry, G1CarryBox=Robot,
                G1CarryBoxCfg=config_ns["G1Cfg"], G1CarryBoxCfgPPO=config_ns["G1CfgPPO"]))
        self.assertEqual(registered, ["carrybox"])
        self.assertEqual({p.name for p in (ENV / "g1").glob("*.py")}, {"carrybox.py", "carrybox_config.py"})
        self.assertEqual({p.name for p in (ENV / "motionlib").glob("*.py")}, {"motionlib_carrybox.py"})
        resources = ROOT / "legged_gym/resources"
        expected = {rel for rel in MANIFEST["files"] if rel.startswith("legged_gym/resources/")}
        expected.add("legged_gym/resources/config/carrybox_loco_pickup.yaml")
        actual = {p.relative_to(ROOT).as_posix() for p in resources.rglob("*") if p.is_file()}
        self.assertEqual(actual, expected)
        urdf = Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=ROOT / "legged_gym"))
        for mesh in ET.parse(urdf).iter("mesh"):
            filename = mesh.attrib["filename"]
            path = urdf.parent / filename
            self.assertTrue(path.is_file(), filename)
            self.assertEqual(path.name, next(p.name for p in path.parent.iterdir() if p.name.lower() == path.name.lower()))
        print("Only carrybox registered; 14 motion files and all 35 required meshes present; no checkpoints/other resources")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    unittest.main(verbosity=2)
