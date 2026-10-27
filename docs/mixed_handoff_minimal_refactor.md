# Minimal mixed-handoff refactor validation

Read [the pre-edit comparison](mixed_handoff_baseline_comparison.md) first.
Changes are deliberately split: A preserves rewards and training numerics;
B changes the Student objective and deletes its obsolete dependencies.

## Commit A

`task_registry.py:make_alg_runner` dispatches explicitly to the existing mixed
runner. Its task factory builds the independent CarryWith expert and records
the complete environment config for strict resume. `envs/__init__.py` now
registers training cfg/PPO cfg; `play_mixed_handoff.py` explicitly constructs
the original Viewer cfg. `train.py` uses the masked runner's collection/update
limits without randomizing the episode clock. The dedicated debug script keeps
RSI ablations and delegates runner creation to the same registry.

`mixed_handoff_on_policy_runner.py` accumulates scalar counts and sum/square
statistics on device. Per-metric reduction still uses float32 and the running
accumulator uses float64 to reproduce Python-double accumulation. Statistics
and distributions transfer at iteration boundaries. Sparse event IDs index
only selected rows/fields before transfer. Set
`runner.detailed_handoff_events=True` in config for full physical snapshots.
Control-flow checks and sparse event transfers still synchronize: this change
does not claim zero GPU synchronization. CUDA timing boundaries are explicit;
`Perf/gpu_peak_memory_bytes` records per-iteration peak allocated memory on the
RL device (CPU reports zero; use an external profiler for multi-device totals).

No reward/state formula, physics, ActorCritic, AMP, PPO or storage modification
is included in A. The next actor/critic buffers are necessary for smoothness
and terminal bootstrapping; no storage removal was justified.

Validation: 93 CPU tests passed. The original 31-term frozen Scope 2 reward
fixture passes. `test_minimal_refactor.py` evaluates the pre-edit Runner from
Git and the new Runner with identical seeds: all rollout fields, per-step
samples, returns, PPO/AMP losses, Student weights and discriminator weights
match **exactly**, including asynchronous failures/timeouts and rollout end.
Additional tests cover native dispatch, unchanged single-policy dispatch,
training clock, and sparse event export. Compileall and diff whitespace checks
pass. No Isaac Gym or external pretrained checkpoint is available here.

### Measured CPU fixture benchmark

Python 3.12.8 / torch 2.6.0+cpu, one torch thread, seed 7, 2 synthetic environments,
100-step rollouts, one warmup plus median of 3 trials. Real networks/Adam/AMP;
synthetic simulator observations. CPU scheduling noise is material; these are
bounded measurements, **not measured PhysX/GPU speed improvements**.

| Metric | Baseline | A |
|---|---:|---:|
| PPO FPS | 791.98 | 720.62 |
| PPO collection / learning, s | .19747 / .05353 | .22151 / .05603 |
| PPO + AMP FPS | 497.55 | 594.27 |
| PPO + AMP collection / learning, s | .29644 / .09587 | .25372 / .08712 |
| CPU scalar extractions / collection-only iteration | 4145 | 2039 |
| Student-valid / all samples | .81 | .81 |
| AMP-valid / Student samples | .339506 | .339506 |
| Rollout bytes (2×100) | 1,890,000 | 1,890,000 |
| GPU peak memory / GPU→CPU sync frequency | unavailable | unavailable |

Run `python tests/benchmark_mixed_handoff.py --output results.json` and add
`--baseline` to evaluate the audited old Runner. Run serially. Raw local
outputs are in ignored `logs/refactor_validation/`; no long training was run.

## Ubuntu verification commands

From the repository root in the existing Isaac Gym Python environment:

```bash
python -m unittest discover -s tests -v

# Small collection, then PPO-only, then PPO+AMP. Original external Teacher
# is source-audited legacy data: opt in explicitly, never bypass silently.
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --allow_legacy_contract --num_envs 4 --seed 1 --steps_per_env 100 --max_iterations 2 --collect_only --no_amp --run_name collect_smoke
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --allow_legacy_contract --num_envs 4 --seed 1 --max_iterations 5 --max_optimizer_updates 5 --no_amp --run_name ppo_smoke
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --allow_legacy_contract --num_envs 4 --seed 1 --max_iterations 5 --max_optimizer_updates 5 --run_name amp_smoke
```

AMP/optimizer counts must actually be nonzero before calling the corresponding
smoke successful. Do not bypass the gate or weaken failures if counts are zero.
For before/after GPU benchmarking, use the same Teacher, environment count,
seed, rollout, PhysX settings and bounded commands on baseline/A/B checkouts;
record `Perf/collection_time`, `Perf/learning_time`, `Perf/total_fps`, validity
fractions and `Perf/gpu_peak_memory_bytes` from training_metrics.jsonl.
Count actual GPU synchronization using the target machine's profiler; CPU
scalar-extraction counts above are not a GPU measurement.

Full Approach → physical handoff → Student Viewer, from repository root:

```bash
TEACHER=legged_gym/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt
# Set STUDENT to the actual model_5.pt produced by the amp_smoke run.
STUDENT=legged_gym/logs/mixed_handoff_student/REPLACE_WITH_ACTUAL_RUN/model_5.pt
python legged_gym/legged_gym/scripts/play_mixed_handoff.py --teacher_checkpoint "$TEACHER" --student_checkpoint "$STUDENT" --allow_legacy_contract --mode mixed --num_envs 1 --steps 1000 --csv logs/minimal_viewer.csv
```

Ordinary native entrypoint (uses config defaults, currently bounded to 20
collection iterations / at most 5 nonempty updates):

```bash
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless
```

The default external checkpoint path is in `MixedHandoffCfgPPO.runner`.
Strict semantic checks remain enabled. If that legacy checkpoint has no
`scope1_contract`, use the explicit `--allow_legacy_contract` compatibility
flag after checking the audited joint/feature/action conventions in
`play_mixed_handoff.py:actor_contract`. Never infer semantic compatibility
from shapes alone. A new Student checkpoint saved by this path embeds the
contract and loads with the existing Viewer Actor loader. Full resume still
requires the same Teacher hash, AMP source, reward/environment and optimizer
settings; fresh Actor initialization is separate from full resume.
