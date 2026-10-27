"""Behavior-preserving runner comparison against the audited Git baseline."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from test_mixed_handoff_training import (
    ROOT, CONTRACT, Expert, TensorEnvironment, actor_critic, configuration,
    MixedHandoffOnPolicyRunner,
)


class RefactorTests(unittest.TestCase):
    def test_event_export_is_sparse_and_verbose_snapshots_are_opt_in(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            path = root / "teacher.pt"
            torch.save(dict(model_state_dict=actor_critic().state_dict(), scope1_contract=CONTRACT), path)
            with patch("torch.utils.tensorboard.SummaryWriter"):
                runner = MixedHandoffOnPolicyRunner(TensorEnvironment(), configuration(False, 4), path,
                                                    CONTRACT, log_dir=root / "log")
            runner.reset()
            for _ in range(4):
                runner.collect_step()
            events = [json.loads(line) for line in (root / "log/handoff_events.jsonl").read_text().splitlines()]
            self.assertEqual([row["env_id"] for row in events], [0, 1])
            self.assertTrue(all(row["handoff_trigger"] for row in events))
            self.assertNotIn("student/linear_error", events[0])
            self.assertEqual(events[0]["student/termination_reason"], 0)
            runner.writer.close()

    def test_baseline_rollout_gae_ppo_amp_and_weights_are_identical(self):
        source = subprocess.check_output([
            "git", "show", "6beaf95:rsl_rl/rsl_rl/runners/mixed_handoff_on_policy_runner.py"], cwd=ROOT).decode()
        baseline = {"__name__": "baseline_runner"}
        exec(compile(source, "baseline_runner.py", "exec"), baseline)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / "teacher.pt"
            torch.manual_seed(7)
            torch.save(dict(model_state_dict=actor_critic().state_dict(), scope1_contract=CONTRACT), path)
            results = []
            for cls in (baseline["MixedHandoffOnPolicyRunner"], MixedHandoffOnPolicyRunner):
                torch.manual_seed(123)
                runner = cls(TensorEnvironment(), configuration(True, 36), path, CONTRACT, expert=Expert())
                runner.reset()
                samples = [{k: v.clone() for k, v in runner.collect_step().items()} for _ in range(36)]
                rollout = copy.deepcopy(runner.storage.data)
                runner.storage.compute_returns(runner.alg.gamma, runner.alg.lam)
                returns = runner.storage.data["returns"].clone()
                metrics = runner.alg.update(runner.storage)
                runner.assert_teacher_frozen()
                results.append((samples, rollout, returns, metrics, copy.deepcopy(runner.alg.actor_critic.state_dict()),
                                copy.deepcopy(runner.alg.amp.state_dict())))
            def compare(a, b):
                if isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for k in a:
                        compare(a[k], b[k])
                elif isinstance(a, (tuple, list)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        compare(x, y)
                elif isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                else:
                    self.assertEqual(a, b)
            compare(*results)

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
