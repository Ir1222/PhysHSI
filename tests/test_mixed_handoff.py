"""CPU tests run the real hooks with tensor fixtures, without importing Isaac Gym."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))
from legged_gym.mixed_handoff import (
    HandoffSettings, HandoffState, STUDENT_CONTROL, TEACHER_CONTROL,
    load_actor, sample_commands,
)


def classes_from_source(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def environment_fixture(settings=None):
    # Identity quaternion fixtures: only coordinate-independent history/command
    # ownership is tested here. Rotation/PhysX contact must be checked in viewer.
    namespace = {"torch": torch, "BaseTask": object, "LeggedRobotCfg": object,
                 "quat_rotate_inverse": lambda q, v: v,
                 "quat_conjugate": lambda q: q,
                 "quat_mul": lambda a, b: a,
                 "quat_to_tan_norm": lambda q: torch.zeros(len(q), 6)}
    directory = ROOT / "legged_gym/legged_gym/envs/g1"
    classes_from_source(directory / "carrybox.py", ["LeggedRobot"], namespace)
    namespace.update(HandoffSettings=HandoffSettings, HandoffState=HandoffState,
                     STUDENT_CONTROL=STUDENT_CONTROL)
    classes_from_source(directory / "carrybox_mixed_handoff.py", ["MixedHandoff"], namespace)
    env = namespace["MixedHandoff"].__new__(namespace["MixedHandoff"])
    env.num_envs, env.device, env.dt = 2, "cpu", .02
    env.cfg = NS(rewards=NS(thresh_robot2object=.7, target_speed_loco=.85),
                 normalization=NS(clip_actions=100.))
    env.handoff = HandoffState(2, "cpu", env.dt, settings or HandoffSettings())
    env.root_states = torch.zeros(2, 13)
    env.root_states[:, 2] = .8
    env.box_states = env.root_states.clone()
    env.box_states[:, 0] = .6
    env.box_states[:, 6] = 1
    env.rigid_body_states = torch.zeros(2, 10, 13)
    env.rigid_body_states[:, :, 2] = .8
    env.rigid_body_states[:, :, 6] = 1
    env.rigid_body_states[:, [7, 8], :3] = env.box_states[:, None, :3]
    env.upper_body_index, env.torso_index, env.head_index = 0, 0, 5
    env.gate_hand_indices = env.gate_palm_indices = [7, 8]
    env.hip_yaw_indices = [1, 2]
    env.termination_contact_indices = []
    env.contact_forces = torch.zeros(2, 10, 3)
    env.contact_forces[:, [7, 8], 0] = 2.
    env._box_size = torch.tensor([[.3, .3, .25]]).repeat(2, 1)
    env.env_origins = torch.zeros(2, 3)
    env.platform_pos = torch.zeros(2, 3)
    env._platform_height = .02
    env.projected_gravity_box = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    env.projected_gravity = env.projected_gravity_box.clone()
    env.roll = env.pitch = torch.zeros(2)
    env.object2goal_dist_xyz = torch.ones(2)
    env.box_cfg = NS(box_termination=False)
    env.test = True
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.max_episode_length = 3000
    env.teacher_last_action = torch.zeros(2, 29)
    env.student_first_action = torch.zeros(2, 29)
    env.reset_count = torch.zeros(2, dtype=torch.long)
    env.has_reached_box = torch.zeros(2, dtype=torch.bool)
    env.commands = torch.zeros(2, 4)
    env.actions = torch.full((2, 29), .4)
    env.executed_phase = env.handoff.phase.clone()
    env.executed_command = env.commands[:, :3].clone()
    env.executed_target = env.commands[:, :3].clone()
    env.extras = {}
    env.add_noise = False
    env.base_ang_vel = env.base_lin_vel = torch.zeros(2, 3)
    env.dof_pos = env.dof_vel = env.default_dof_pos = torch.zeros(2, 29)
    env.end_effector_pos = torch.zeros(2, 15)
    env.obs_scales = NS(ang_vel=.25, dof_pos=1., dof_vel=.05, lin_vel=2.)
    env.obs_buf = torch.arange(738.).repeat(2, 1)
    env.num_one_step_actor_obs = 123
    return env


class HandoffTests(unittest.TestCase):
    def advance(self, state, height=.8, bilateral=True, clearance=.2, done=False):
        n = len(state.phase)
        return state.complete_step(torch.full((n,), height), torch.full((n,), bilateral),
                                   torch.full((n,), clearance), torch.full((n,), done),
                                   torch.full((n,), 1.))

    def test_gate_requires_contact_height_clearance_and_continuity(self):
        state = HandoffState(1, "cpu", .02, HandoffSettings())
        for kwargs in ({"bilateral": False}, {"height": .72}, {"clearance": 0.}):
            for _ in range(20):
                self.assertFalse(self.advance(state, **kwargs).item())
        for _ in range(10):
            self.assertFalse(self.advance(state).item())
        self.advance(state, bilateral=False)
        for _ in range(14):
            self.assertFalse(self.advance(state).item())
        self.assertTrue(self.advance(state).item())
        self.assertEqual(state.phase.item(), STUDENT_CONTROL)
        self.assertEqual(state.student_applied_command.abs().sum(), 0)

    def test_terminal_step_does_not_switch(self):
        state = HandoffState(1, "cpu", .02, HandoffSettings(stable_duration_s=.02))
        self.assertFalse(self.advance(state, done=True).item())

    def test_zero_first_command_stand_then_rate_limit_every_change(self):
        state = HandoffState(1, "cpu", .02, HandoffSettings(
            stable_duration_s=.02, mode="fixed", fixed_command=(1., -.5, .7)))
        self.advance(state)
        for _ in range(24):
            self.assertEqual(state.student_applied_command.abs().sum(), 0)
            self.advance(state)
        self.advance(state)
        bound = torch.tensor([.8, .6, 1.2]) * .02
        for step in range(250):
            previous = state.student_applied_command.clone()
            if step == 120:
                state.student_target_command[:] = torch.tensor([-1., .5, -.7])
            self.advance(state)
            self.assertTrue(torch.all((state.student_applied_command - previous).abs() <= bound + 1e-7))
        state.reset([0])
        self.assertEqual(state.phase.item(), TEACHER_CONTROL)
        self.assertEqual(state.student_applied_command.abs().sum(), 0)

    def test_sampling_distribution_ranges_and_deadzone(self):
        torch.manual_seed(9)
        commands = sample_commands(30000, "cpu")
        nonzero = commands != 0
        counts = nonzero.sum(-1)
        self.assertAlmostEqual((counts == 0).float().mean().item(), .1, delta=.01)
        self.assertAlmostEqual((counts == 3).float().mean().item(), .6, delta=.015)
        for axis, (low, high) in enumerate(((-.6, 1.2), (-.5, .5), (-.7, .7))):
            self.assertTrue(((commands[:, axis] >= low) & (commands[:, axis] <= high)).all())
            specialist = (counts == 1) & nonzero[:, axis]
            self.assertAlmostEqual(specialist.float().mean().item(), .1, delta=.01)
            self.assertTrue((commands[specialist, axis].abs() >= .1 - 1e-7).all())

    def test_resampling_interval_and_stand_mode(self):
        state = HandoffState(1, "cpu", .02, HandoffSettings(
            stable_duration_s=.02, student_initial_stand_duration_s=0., mode="mixed"))
        self.advance(state)
        self.advance(state)
        self.assertTrue(4 <= state.resample_time.item() <= 6)
        target = state.student_target_command.clone()
        for _ in range(190):
            self.advance(state)
            torch.testing.assert_close(state.student_target_command, target)
        state.settings.mode = "stand"
        state.reset([0])
        self.advance(state)
        for _ in range(350):
            self.advance(state)
            self.assertEqual(state.student_applied_command.abs().sum(), 0)

    def test_mixed_batch_terminal_reset_cannot_transfer_other_environments(self):
        state = HandoffState(2, "cpu", .02, HandoffSettings(stable_duration_s=.02))
        state.complete_step(torch.tensor([.8, .8]), torch.tensor([True, True]),
                            torch.tensor([.2, .2]), torch.tensor([False, True]), torch.ones(2))
        self.assertEqual(state.phase.tolist(), [STUDENT_CONTROL, TEACHER_CONTROL])
        state.reset([1])
        self.assertEqual(state.phase.tolist(), [STUDENT_CONTROL, TEACHER_CONTROL])

    def test_handoff_preserves_physics_history_and_executed_action(self):
        env = environment_fixture(HandoffSettings(stable_duration_s=.02))
        env.reset_idx = Mock(side_effect=AssertionError("handoff must not reset"))
        before_history = env.obs_buf.clone()
        before_physics = [v.clone() for v in (env.root_states, env.dof_pos, env.box_states)]
        env.check_termination()
        info = env.extras["handoff"]
        self.assertTrue((info["executed_phase"] == TEACHER_CONTROL).all())
        self.assertTrue((info["post_gate_phase"] == STUDENT_CONTROL).all())
        env.compute_observations()
        env.reset_idx.assert_not_called()
        self.assertEqual(tuple(env.obs_buf.shape), (2, 738))
        self.assertEqual(tuple(env.privileged_obs_buf.shape), (2, 126))
        torch.testing.assert_close(env.obs_buf[:, :-123], before_history[:, 123:])
        torch.testing.assert_close(env.obs_buf[:, -44:-15], env.actions)
        self.assertEqual(env.obs_buf[:, -3:].abs().sum(), 0)
        for before, after in zip(before_physics, (env.root_states, env.dof_pos, env.box_states)):
            torch.testing.assert_close(before, after)

    def test_joint_mapping_is_validated_before_checkpoint_loading(self):
        path = ROOT / "legged_gym/legged_gym/scripts/play_mixed_handoff.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "actor_contract")
        namespace = {"Path": Path, "LEGGED_GYM_ROOT_DIR": str(ROOT / "legged_gym")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        with self.assertRaisesRegex(ValueError, "joint ordering"):
            namespace["actor_contract"](NS(dof_names=["wrong"] * 29))

    def test_command_ownership_and_teacher_semantics(self):
        env = environment_fixture()
        env.root_states[0, 0] = -2.  # Teacher is still approaching.
        env.handoff.phase[1] = STUDENT_CONTROL
        env.handoff.student_applied_command[1] = torch.tensor([-.2, .3, -.4])
        actor, critic = env.compute_task_observations()
        torch.testing.assert_close(actor[0, -3:], torch.tensor([.85, 0., 0.]))
        torch.testing.assert_close(actor[1, -3:], torch.tensor([-.2, .3, -.4]))
        torch.testing.assert_close(env.commands[:, :3], critic[:, -3:])
        env.root_states[0, 0] = .6
        actor, _ = env.compute_task_observations()
        self.assertEqual(actor[0, -3:].abs().sum(), 0)
        env.root_states[0, 0] = -2.
        actor, _ = env.compute_task_observations()
        self.assertEqual(actor[0, -3:].abs().sum(), 0)  # Reached latch stays set.

    def test_unilateral_contact_and_force_on_wrong_body_cannot_switch(self):
        env = environment_fixture(HandoffSettings(stable_duration_s=.02))
        env.contact_forces[0, 8] = 0
        env.rigid_body_states[1, 8, 0] = 5.
        env.check_termination()
        self.assertTrue((env.handoff.phase == TEACHER_CONTROL).all())

    def test_terminal_diagnostics_survive_reset_and_timeout_failure_overlap(self):
        env = environment_fixture()
        env.episode_length_buf[:] = 3001
        env.root_states[0, 2] = .1
        env.check_termination()
        info = env.extras["handoff"]
        self.assertTrue(info["timeout"].all())
        self.assertEqual(info["failure"].tolist(), [True, False])
        self.assertTrue(info["handoff_failure"].all())
        height = info["box_height"].clone()
        env.box_states[:] = 0  # Simulate the subsequent automatic reset.
        torch.testing.assert_close(info["box_height"], height)

    def test_first_student_action_diagnostics(self):
        env = environment_fixture(HandoffSettings(stable_duration_s=.02))
        env.check_termination()
        teacher_action = env.actions.clone()
        env.executed_phase = env.handoff.phase.clone()
        env.actions[:] = -.2
        env.check_termination()
        info = env.extras["handoff"]
        self.assertTrue(info["first_student_step"].all())
        torch.testing.assert_close(info["teacher_last_action"], teacher_action)
        torch.testing.assert_close(info["student_first_action"], env.actions)
        torch.testing.assert_close(info["action_discontinuity"], env.actions - teacher_action)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = str(Path(self.folder.name) / "actor.pt")
        self.contract = {"dof_names": [f"joint_{i}" for i in range(29)], "activation": "elu"}
        actor = nn.Sequential(nn.Linear(738, 8), nn.ELU(), nn.Linear(8, 29))
        self.checkpoint = {"model_state_dict": {"actor." + k: v for k, v in actor.state_dict().items()},
                           "scope1_contract": self.contract}

    def tearDown(self):
        self.folder.cleanup()

    def load(self, **kwargs):
        torch.save(self.checkpoint, self.path)
        return load_actor(self.path, self.contract, **kwargs)

    def test_actor_only_frozen_eval_and_actual_layer_dimensions(self):
        self.checkpoint["model_state_dict"]["critic.0.weight"] = torch.ones(2, 126)
        self.checkpoint["amp_state_dict"] = {"unused": True}
        for _ in range(2):  # Both policies use the same frozen loader.
            actor, spec = self.load()
            self.assertFalse(actor.training)
            self.assertTrue(all(not p.requires_grad for p in actor.parameters()))
            self.assertEqual(spec["dims"], [738, 8, 29])
            with torch.no_grad():
                self.assertEqual(tuple(actor(torch.zeros(2, 738)).shape), (2, 29))

    def test_reject_dimensions_parameter_names_and_joint_order(self):
        for mutation, message in (
            (lambda c: c["model_state_dict"].update({"actor.0.weight": torch.zeros(8, 737)}), "dimensions"),
            (lambda c: c["model_state_dict"].pop("actor.2.bias"), "parameter names/shapes"),
            (lambda c: c.update(scope1_contract={**self.contract, "dof_names": list(reversed(self.contract["dof_names"]))}), "dof_names"),
        ):
            original = {**self.checkpoint, "model_state_dict": dict(self.checkpoint["model_state_dict"])}
            mutation(self.checkpoint)
            with self.assertRaisesRegex(ValueError, message):
                self.load()
            self.checkpoint = original

    def test_missing_metadata_requires_explicit_assumption(self):
        self.checkpoint.pop("scope1_contract")
        with self.assertRaisesRegex(ValueError, "missing scope1_contract"):
            self.load()
        with self.assertWarns(UserWarning):
            _, spec = self.load(allow_legacy_contract=True)
        self.assertEqual(spec["semantics"], "source_assumption")

    def test_output_contract_nonfinite_weights_and_adapter_rejected(self):
        self.checkpoint["model_state_dict"]["actor.2.weight"] = torch.zeros(28, 8)
        with self.assertRaisesRegex(ValueError, "output=29"):
            self.load()
        self.checkpoint["model_state_dict"]["actor.2.weight"] = torch.full((29, 8), float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.load()
        self.checkpoint["handoff_contract"] = {}
        with self.assertRaisesRegex(ValueError, "action-adapter"):
            self.load()


if __name__ == "__main__":
    unittest.main()
