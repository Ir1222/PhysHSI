"""Behavior-preserving runner comparison against the audited Git baseline."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from test_mixed_handoff_training import (
    ROOT, Expert, TensorEnvironment, actor_critic, configuration,
    MixedHandoffOnPolicyRunner,
)


class RefactorTests(unittest.TestCase):
    def test_event_export_is_sparse_and_opt_in(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            path = root / "teacher.pt"
            torch.save(dict(model_state_dict=actor_critic().state_dict()), path)
            with patch("torch.utils.tensorboard.SummaryWriter"):
                runner = MixedHandoffOnPolicyRunner(TensorEnvironment(), configuration(False, 4), path,
                                                    log_dir=root / "log")
            runner.reset()
            runner.collect_step()
            self.assertFalse((root / "log/handoff_events.jsonl").exists())
            runner.detailed_diagnostics = True
            for _ in range(3):
                runner.collect_step()
            events = [json.loads(line) for line in (root / "log/handoff_events.jsonl").read_text().splitlines()]
            self.assertEqual([row["env_id"] for row in events], [0, 1])
            self.assertTrue(all(row["handoff_trigger"] for row in events))
            self.assertIn("student/linear_tracking_error", events[0])
            self.assertEqual(events[0]["student/termination_reason"], 0)
            runner.writer.close()

    def test_gated_amp_preserves_actor_critic_contract_and_updates_fresh_student(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / "teacher.pt"
            torch.manual_seed(7)
            torch.save(dict(model_state_dict=actor_critic().state_dict()), path)
            runner = MixedHandoffOnPolicyRunner(
                TensorEnvironment(), configuration(True, 36), path, expert=Expert())
            self.assertEqual(runner.alg.actor_critic.num_actor_obs, 738)
            self.assertEqual(runner.alg.actor_critic.num_critic_obs, 126)
            self.assertEqual(runner.alg.actor_critic.num_actions, 29)
            self.assertEqual(runner.alg.amp.trunk[0].in_features, 541)
            self.assertEqual(runner.storage.data["amp_observations"].shape[-1], 541)
            before = copy.deepcopy(runner.alg.actor_critic.state_dict())
            runner.reset()
            for _ in range(36):
                runner.collect_step()
            runner.storage.compute_returns(runner.alg.gamma, runner.alg.lam)
            metrics = runner.alg.update(runner.storage)
            self.assertEqual(metrics["updated"], 1)
            self.assertTrue(any(not torch.equal(value, before[key])
                                for key, value in runner.alg.actor_critic.state_dict().items()))
            runner.assert_teacher_frozen()

    def test_registry_dispatch_preserves_original_runner(self):
        from test_mixed_handoff import classes_from_source
        namespace = dict(Tuple=tuple, VecEnv=object, LeggedRobotCfg=object, LeggedRobotCfgPPO=object,
                         HIMOnPolicyRunner=Mock(), class_to_dict=lambda cfg: {},
                         update_cfg_from_args=lambda a, b, c: (a, b))
        classes_from_source(ROOT / "legged_gym/legged_gym/utils/task_registry.py", ["TaskRegistry"], namespace)
        registry = namespace["TaskRegistry"]()
        args = NS(rl_device="cpu")
        cfg = NS(runner=NS(resume=False))
        env = object()
        original, _ = registry.make_alg_runner(env, args=args, train_cfg=cfg, log_root=None)
        namespace["HIMOnPolicyRunner"].assert_called_once_with(env, {}, None, device="cpu")
        cfg.runner_class_name = "MixedHandoffOnPolicyRunner"
        registry._make_mixed_handoff_runner = Mock()
        mixed, _ = registry.make_alg_runner(env, args=args, train_cfg=cfg, log_root=None)
        self.assertIs(mixed, registry._make_mixed_handoff_runner.return_value)
        self.assertIsNot(mixed, original)
        registry._make_mixed_handoff_runner.assert_called_once_with(env, cfg, args, None)

    def test_native_train_uses_collection_limit_without_random_episode_clock(self):
        source = ROOT / "legged_gym/legged_gym/scripts/train.py"
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "train")
        registry = Mock()
        cfg = NS(runner_class_name="MixedHandoffOnPolicyRunner",
                 runner=NS(max_iterations=2, max_optimizer_updates=1, collect_only=True))
        registry.make_env.return_value = (object(), object())
        runner = Mock()
        registry.make_alg_runner.return_value = (runner, cfg)
        ns = dict(task_registry=registry)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
        ns["train"](NS(task="carrybox_mixed_handoff"))
        runner.learn.assert_called_once_with(2, optimize=False, max_optimizer_updates=1)


if __name__ == "__main__":
    unittest.main()
