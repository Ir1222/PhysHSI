"""Bounded CPU tensor benchmark; never represents PhysX or CUDA throughput."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from unittest.mock import patch

import torch

from test_mixed_handoff_training import (
    actor_critic, configuration, CONTRACT, Expert, MixedHandoffOnPolicyRunner,
    TensorEnvironment,
)
from handoff_regression_support import regression_environment, apply_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", action="store_true", help="Use audited pre-refactor runner from Git")
    args = parser.parse_args()
    torch.set_num_threads(1)
    Runner = MixedHandoffOnPolicyRunner
    if args.baseline:
        namespace = {"__name__": "baseline_runner"}
        source = subprocess.check_output(["git", "show", "6beaf95:rsl_rl/rsl_rl/runners/mixed_handoff_on_policy_runner.py"]).decode()
        exec(compile(source, "baseline_runner.py", "exec"), namespace)
        Runner = namespace["MixedHandoffOnPolicyRunner"]
    results = dict(device="cpu", torch=torch.__version__, seed=7, num_envs=2,
                   rollout_length=100, repeats=3, physics="synthetic tensor fixture")
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        checkpoint = Path(folder) / "teacher.pt"
        torch.manual_seed(7)
        torch.save(dict(model_state_dict=actor_critic().state_dict(), scope1_contract=CONTRACT), checkpoint)
        for amp in (False, True):
            timings = []
            for repeat in range(4):  # warmup then three measured identical seeds
                torch.manual_seed(7)
                runner = Runner(TensorEnvironment(), configuration(amp, 100),
                                                    checkpoint, CONTRACT, expert=Expert() if amp else None)
                with patch.object(runner, "log") as log:
                    runner.learn(1)
                    metrics = log.call_args.args[0]
                if repeat:
                    timings.append(metrics)
            results["amp" if amp else "ppo"] = {
                key: statistics.median(m[key] for m in timings)
                for key in ("Perf/total_fps", "Perf/collection_time", "Perf/learning_time",
                            "student_valid_ratio", "amp_valid_fraction")}
            results["storage_bytes"] = sum(t.numel()*t.element_size() for t in runner.storage.data.values())
        env = regression_environment()
        times = []
        for repeat in range(4):
            start = time.perf_counter()
            for step in range(100):
                apply_sample(env, step % 7)
                env.check_termination()
                env.compute_reward()
            if repeat:
                times.append(time.perf_counter()-start)
        results["environment_hooks_100_steps_s"] = statistics.median(times)
        # Profile CPU scalar extraction as an observable proxy, not CUDA syncs.
        torch.manual_seed(7)
        runner = Runner(TensorEnvironment(), configuration(False, 100), checkpoint, CONTRACT)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            runner.learn(1, optimize=False)
        results["cpu_scalar_extractions_collection_iteration"] = sum(
            e.count for e in profiler.key_averages() if e.key == "aten::_local_scalar_dense")
    Path(args.output).write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
