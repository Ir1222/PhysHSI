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
only selected rows/fields before transfer in explicit debug mode. Set
`--detailed_handoff_events` for physical snapshots and TensorBoard histograms.
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

## Commit B: minimal Student objective

The exact Student scales are now:

```python
class scales:
    carry_lin_vel_tracking = 3.0
    carry_yaw_vel_tracking = 2.5
    carry_bilateral_contact = 0.5
    lin_vel_z = -1.0
    ang_vel_xy = -0.05
    orientation = -1.0
    dof_acc = -1e-7
    action_rate = -0.03
    torques = -1e-4
    dof_vel = -2e-4
    dof_pos_limits = -5.0
    dof_vel_limits = -1e-3
    torque_limits = -0.03
```

`carrybox_mixed_handoff_config.py` retains only tracking sigmas (.25/.25)
outside this list. `_prepare_student_reward_function` copies scales and applies
control dt once. The seven joint/actuator regularizers are bound directly to
the original Policy 1 implementations. Three small stability helpers preserve
the existing explicit body-frame references; tests compare them exactly to
the original PhysHSI cached-velocity/projected-gravity formulas. Gravity uses
the existing simulator tensor, with no per-step constant tensor construction.

Deleted active terms and implementations:

```text
carry_arm_range, carry_hip_posture, carry_foot_heading, carry_feet_width,
carry_knee_width, carry_waist_reference, carry_torso_pelvis_alignment,
carry_hand_box_surface, carry_hand_slip, carry_relative_velocity,
carry_relative_position, carry_box_tilt, zero_command_stillness,
feet_air_time, feet_slip, feet_stumble, collision, feet_contact_forces
```

Deleted dependencies: arm/hip/waist joint lists and index buffers, foot/knee
reward body lists, posture targets/ranges/deadzones/softness, hand-face shaping,
relative-position/velocity tolerances, previous hand/box-relative positions,
derivative-valid history, raw-contact history, posture/quaternion helper
functions, and all unused Student reward cache fields. Inherited simulation
state used by Teacher observations/rewards/termination remains intact.

Normal `_update_student_carry_state` now computes tracking errors, filtered
hand-box contact, grasp timer, drop/separation and enabled terminal flags.
The physical gate already computes palm-to-rotated-box proximity and rotated
box bottom; contact is passed directly into the Student update. Bilateral
reward and grasp loss share its one-frame OR filter. **Default numerics are
unchanged**: force >1N and box-surface distance <.08m. The duplicate Student
force-threshold setting is removed: an explicit nondefault gate force override
now intentionally also controls Student contact. This correction is tested
with a 3N threshold and 2N forces; neither gate nor positive contact reward
accepts those contacts. The independent .3s complete-grasp-loss grace remains.

Drop height .15m, separation 1m, native failure rules, timeout boundary and
optional tilt termination remain unchanged. Disabled tilt is not computed for
normal training; explicit tilt debugging/enabled termination still computes it.
No URDF, action/torque clipping, PD gain, simulator constraint or Teacher source
file was changed.

Production reward aggregation has separate Teacher/Student lists and episode
sums, with ownership from `executed_phase` and `executed_command`. Full per-term
raw/weighted clones and physical action snapshots require
`env_cfg.diagnostics.detailed=True`; Viewer `--csv` explicitly enables it.
For detailed event-file export also set `runner.detailed_handoff_events=True`.
Normal TensorBoard receives tracking/contact/failure/validity metrics and
`episode/student/mean_step_reward/<term>` on resets, using Student-step
denominators; removed posture metrics are not calculated for logging.

The resulting environment data flow is:

```text
PickUp-only native RSI → frozen Teacher action
  → physics / native state refresh
  → physical gate geometry + minimal Student contact/tracking/failure update
  → native physical failure + independent Student failures / timeout
  → executed-policy-owned reward (Teacher still owns gate-triggering action)
  → native AMP observation before reset → ordinary episode reset if needed
  → next Actor observation with applied rate-limited command
  → Student-only PPO / segment-bounded GAE / 10-frame-valid 540D AMP
```

### Numerical and training verification

The original frozen JSON fixture is unchanged. New tests compare all 13 retained
raw terms and their contributions against that data, then reconstruct the new
total from **only** the retained contributions. Across seven two-environment
samples: raw retained max error = 0; weighted retained max error =
3.725290298461914e-9; Teacher raw/weighted max errors = 0. Ownership, contact
filter, tracking frames, terminal reasons and Teacher episode sums also match.
Normal/debug reward totals, contact, failure flags and episode sums agree exactly.

All 94 CPU tests passed. The CPU suite additionally runs real minimal reward computation into two bounded
PPO-only and PPO+AMP updates, checks a frozen Teacher, and reloads the saved
Student Actor through the existing Viewer loader. These use synthetic physics
and synthetic Teacher weights; they do not establish external model_17000.pt
loading or physical Viewer stability. Five tests for deleted shaping behavior
were removed; minimal reward/production-path tests replace them. The A Runner
rollout/GAE/PPO/AMP exact numerical regression is retained.

Measured B CPU fixture results (same benchmark settings as A):

| Metric | Baseline | A | B |
|---|---:|---:|---:|
| PPO FPS | 791.98 | 720.62 | 647.67 |
| PPO collection / learning, s | .19747 / .05353 | .22151 / .05603 | .25419 / .06470 |
| PPO + AMP FPS | 497.55 | 594.27 | 542.33 |
| PPO + AMP collection / learning, s | .29644 / .09587 | .25372 / .08712 | .26920 / .10824 |
| Actual environment hooks, 100 steps, s | 1.13938 | 1.21913 | .49828 |
| CPU scalar extractions / collection iteration | 4145 | 2039 | 2039 |
| Student-valid / all samples | .81 | .81 | .81 |
| AMP-valid / Student samples | .339506 | .339506 | .339506 |
| Rollout bytes | 1,890,000 | 1,890,000 | 1,890,000 |

Runner timings use synthetic observations/rewards to isolate collection/update
infrastructure; the separate environment-hooks timing runs actual environment
reward/termination code. All timings are individual medians, so FPS need not
equal the reciprocal of the sum of median times. B's hook measurement uses
normal diagnostics, while baseline/A always executed their full diagnostics.
This is the intended production-default comparison. Timing variance prevents
a claim of overall training acceleration. GPU FPS, memory, synchronization and
physical Student/AMP-valid fractions still require Ubuntu measurements.

Full resume from a pre-B checkpoint rejects the changed Student reward config;
use a fresh run or explicit Actor/std initialization for a different objective.
The Teacher checkpoint and its native pickup/hold task remain independent.
AMP supplies the motion prior; reduced shaping may change contact/stability
and survival. Short synthetic tests do not predict trained-policy quality.

## Ubuntu verification commands

From the repository root in the existing Isaac Gym Python environment:

```bash
python -m unittest discover -s tests -v

# Small collection, then PPO-only, then PPO+AMP. Original external Teacher
# is source-audited legacy data: opt in explicitly, never bypass silently.
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --num_envs 4 --seed 1 --steps_per_env 100 --max_iterations 2 --collect_only --no_amp --run_name collect_smoke
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --num_envs 4 --seed 1 --max_iterations 5 --max_optimizer_updates 5 --no_amp --run_name ppo_smoke
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --num_envs 4 --seed 1 --max_iterations 5 --max_optimizer_updates 5 --run_name amp_smoke
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
python legged_gym/legged_gym/scripts/play_mixed_handoff.py --teacher_checkpoint "$TEACHER" --student_checkpoint "$STUDENT" --mode mixed --num_envs 1 --steps 1000 --csv logs/minimal_viewer.csv
```

Ordinary native entrypoint (4096 environments, 20 s episodes, 100 rollout
steps/environment, 20000 collection iterations, save every 500 iterations;
no extra optimizer-update limit):

```bash
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless
```

The default external checkpoint path is in `MixedHandoffCfgPPO.runner`.
Native PhysHSI `model_state_dict` checkpoints load directly without extra
metadata. Actor loading validates the exact ELU network, keys/shapes and finite
weights. Teacher is frozen and evaluated without gradients. Fresh Student
initialization copies Actor/std only; Critic, Adam and 540D AMP are initialized
independently. The Viewer reads both new and old Student checkpoints.
Full resume still requires matching network, algorithm/optimizer configuration,
AMP configuration, Teacher hash, expert identity, environment/reward settings,
runtime timestep and joint order. New checkpoints store `environment_signature`;
old optional `scope1_contract` runtime fields are accepted. Simulator state is
not restored; resume starts new Teacher episodes with empty Student AMP history.

## Production cleanup and metric scopes

Training defaults now match native CarryBox scale and duration. PickUp-only RSI,
0.1 s training gate, zero forced Stand, Mixed Commands and slew limits retain
their previous values. Viewer defaults remain 1 environment, full Approach,
60 s episodes, 0.3 s gate and 0.5 s Stand. Domain randomization, termination,
all 13 Student reward terms/coefficients, PPO loss/Adam/adaptive KL behavior,
Student-only GAE, timeout bootstrap and ten-frame 540D AMP validity are unchanged.
The original `carrybox` task still dispatches to `HIMOnPolicyRunner`.

| Scalar | Scope and denominator |
|---|---|
| `student_valid_ratio` | Current rollout Student transitions / all rollout transitions |
| `student_step_fraction` | Cumulative Student steps / cumulative environment steps; restored on resume |
| `episode_student_fraction` | Mean of Student / (Teacher + Student) steps for episodes ending in this rollout; includes Teacher-only episodes as zero |
| `episode_student_fraction_count` | Number of completed episodes used for that mean; zero means no samples |
| `amp_valid_fraction` | Current rollout AMP-valid / Student-valid steps |
| `cumulative_amp_valid_fraction` | Cumulative AMP-valid steps / cumulative Student steps |
| `handoff_success_rate` | Cumulative successful gates / (successful gates + Teacher episodes ending before handoff) |
| `student/segment_duration_s` | Mean duration of Student segments ending in this rollout, failures and timeouts together |
| `student/completed_survival_s` | Mean observed physical-failure Student duration |
| `student/censored_timeout_s` | Mean timeout-censored Student duration; failure takes precedence on overlap |
| `student/*/count` | Sample count for each of the three duration metrics |
| `student_incomplete_segments`, `student_incomplete_mean_frames` | Current active Student segments; excluded from ended-segment/survival means |
| `kl` | Existing PPO mean minibatch KL used by adaptive scheduling, emitted only on actual updates |
| `learning_rate` | Actual current optimizer learning rate after adaptive scheduling |

Episode counters survive rollout boundaries and clear only for the environments
that reset. Duration distributions report `median` (p50) and `p90` under
`distribution/student/...`; the old `performance/student/...` means remain.
Absent duration means and zero counts indicate no completed observations, not
zero survival. A zero Student denominator yields ratio zero with explicit
`student_valid_count`/`student_steps` available. Quantiles are descriptive
ended-segment summaries, not a censoring-corrected survival estimator.

All scalars go to TensorBoard and `training_metrics.jsonl` every iteration.
Terminal output stays on its 100-iteration/final cadence and includes real KL,
learning rate and Student ratios. Detailed event files, physical snapshots and
TensorBoard histograms require `--detailed_handoff_events`. Production skips
step-level event CPU export. Exact frozen-weight comparison runs at completion
(and every debug iteration); frozen mode/gradient checks still run every iteration.
The runner reuses PPO's counts, avoids a duplicate AMP sample materialization,
removes redundant clones and skips duplicate final checkpoint writes.

From the repository root in the project's Ubuntu Isaac Gym environment:

```bash
# Short, explicitly bounded PPO + AMP smoke; configured Teacher and expert must exist.
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --num_envs 16 --seed 1 --steps_per_env 100 --max_iterations 30 --max_optimizer_updates 3 --run_name cleanup_smoke

# Formal training: no additional optimizer-update limit.
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless

# Full resume: replace the path with the checkpoint produced by a matching run.
python legged_gym/legged_gym/scripts/train.py --task carrybox_mixed_handoff --headless --resume --resume_path /absolute/path/to/model_500.pt
```

The smoke's collection bound can be reached without three updates if too few
episodes pass the unchanged physical gate. Check nonzero PPO/AMP counts rather
than weakening gate or termination conditions. CPU tests use synthetic physics
fixtures and actual networks, losses, optimizer, expert data and TensorBoard;
they do not establish Isaac Gym/CUDA training throughput or success rates.

Validation on 2026-10-27: `python -m unittest discover -s tests -v` passed all
99 CPU tests (the unchanged pre-edit baseline passed 94). Coverage includes
native checkpoint loading, frozen Teacher and fresh Student Actor/std/critic/Adam,
exact rollout/GAE/PPO/AMP/weight agreement with the historical Runner, unlimited
updates beyond five, native registry dispatch/defaults/CLI overrides, asynchronous
episode ratios including zero-Student episodes, survival censoring, actual
TensorBoard/JSON round trips, Viewer loading and old/new full resume checks.
`git diff --check` passed. Original CarryBox source/config, ActorCritic, PPO,
Storage, AMP discriminator and 540D projection files have no changes.
This Windows host has PyTorch 2.6.0+cpu, no CUDA/Isaac Gym, and no configured
`model_17000.pt`; checkpoint regressions use native-format tensor fixtures.
Ubuntu GPU smoke/formal training and the real external Teacher remain unverified
by this cleanup. The previously supplied Ubuntu training results are not a new
validation run of this commit.
