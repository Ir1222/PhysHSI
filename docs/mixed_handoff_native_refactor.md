# Native PhysHSI organization, with Scope 2 behavior preserved

This is a file/method/buffer refactor of Scope 2, with baseline commit
`6295c21694c793ff7f4f2f445f1331eba2313754`. It introduces no training or
reward/geometry/termination tuning. Original `carrybox.py`, `carrybox_config.py`,
the handoff state machine, Actor loader and Viewer CLI remain unchanged.

## Files

Modified:

- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py`: environment-owned
  Student buffers, update/reset methods, individual unweighted reward methods,
  and independent reward preparation.
- `legged_gym/legged_gym/envs/__init__.py`: import task configuration from its
  dedicated module; keep the independent task registration.
- `tests/test_mixed_handoff.py`, `tests/test_student_carry.py`: use real nested
  configurations and environment methods instead of the removed helper object.
- `docs/mixed_handoff_scope2.md`: point configuration and validation instructions
  to the current native structure.

Created:

- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py`.
- `tests/handoff_regression_support.py`: deterministic physical input fixtures.
- `tests/fixtures/mixed_handoff_scope2.json`: frozen outputs/configuration of
  the baseline implementation, containing data rather than a second implementation.
- `tests/test_handoff_native_regression.py`: before/after numerical comparisons,
  initialization order, coefficient scaling and configuration checks.
- `docs/mixed_handoff_native_refactor.md`: this report.

Removed: `legged_gym/legged_gym/student_carry.py`. No runtime or test import of
that module remains.

## Configuration and execution

The task now follows the usual sibling-file layout:

```text
legged_gym/legged_gym/envs/g1/
  carrybox.py                         # unchanged
  carrybox_config.py                  # unchanged
  carrybox_mixed_handoff.py
  carrybox_mixed_handoff_config.py
```

`MixedHandoffCfg(G1Cfg)` preserves inherited Teacher `rewards.scales`.
Student parameters and the explicit `class scales` are in `student_rewards`;
drop/separation/grasp/tilt settings are in `student_termination`. All 41 former
configuration entries, including all 31 coefficients, are compared against the
baseline values without tolerances. The complete unchanged reward definitions
and table are in [Scope 2](mixed_handoff_scope2.md#exact-student-reward-table).

`_init_buffers()` first invokes the original buffer initialization, then creates
handoff and Student physical/history buffers. `_update_student_carry_state()`
samples physical quantities once in the existing pre-reward hook. Individual
`_reward_carry_*()` methods return one unweighted `[num_envs]` tensor using these
buffers. First-Student derivative validity, per-environment contact filters,
grasp-loss timing and ordinary-reset behavior are unchanged.

`_prepare_reward_function()` preserves the inherited Teacher preparation, then
calls `_prepare_student_reward_function()`. Student preparation copies nested
scales, removes zero coefficients, applies `dt` once, resolves reward methods and
stores `student_reward_scales`, `student_reward_names`,
`student_reward_functions` and `student_episode_sums`. Repeated preparation does
not mutate config values or double-scale the coefficients.

The seven Student-specific body/foot regularizers resolve to
`_reward_student_<name>` through an explicit preparation mapping. None overrides
the identically named Teacher regularizer. Other matching regularizers use the
original inherited methods. Student/Teacher lists and sums remain independent.

`compute_reward()` masks both lists using `executed_phase`; the gate-triggering
step stays Teacher-owned. Rewards use the same executed command, clipping and
Teacher termination-after-clipping order. Totals are still composed as
`teacher_total + student_total`. CSV names, terminal measurements and Student-step
episode denominators are retained. The native helper's alphabetical reward
registration may change CSV column order; consumers should use field names.

## Before/after numerical results

The frozen fixture was evaluated from the original baseline Git sources, with
the real repository quaternion functions. Tests consume only its saved numerical
outputs. Seven deterministic two-environment samples exercise staggered gates,
first Student steps, temporal motion, contact loss, joint/body regularization,
drop/separation and simultaneous timeout. Episode reset statistics are also
compared. These are synthetic tensor fixtures, not Viewer recordings.

Every enabled term and CSV measurement is compared, with the following maximum
absolute differences on CPU float32:

| Compared outputs | Count | Maximum difference |
|---|---:|---:|
| Teacher raw terms | 15 per sample | 0 |
| Teacher weighted terms and sums | 15 per sample | 0 |
| Student raw terms | all 31 per sample | 0 |
| Tracking/contact/posture/derivative metrics and termination flags | all saved fields | 0 |
| Student weighted terms | all 31 per sample | 3.73e-9 |
| Reward totals and means | all saved fields | 2.99e-8 |
| Per-environment episode sums | all saved terms | 3.73e-9 |

The tiny weighted/total differences are float32 operation-order effects from
preparing `scale*dt` once and native registration order. Raw rewards, physical
metrics and Teacher terms are checked for exact equality. Remaining floating
outputs use `atol=5e-8`, `rtol=1e-7`; flags and counts retain exact values.

All existing Scope 1/2 tests remain. Added checks cover initialization order,
Teacher config equality, every migrated configuration value, disabled terms,
repeated preparation, `dt` exactly once, and non-overriding Student helpers.

```bash
python -m unittest discover -s tests -v
python -m compileall -q legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py \
  legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py tests
git diff --check
```

No geometry mismatch correction or tolerance adjustment is included, following
the user's clarification. Existing hand-surface tests and values remain. Hard
box-tilt termination stays disabled, and the previous soft posture/reference
calibration limitations remain as documented in Scope 2.

Isaac Gym is unavailable on this development machine. The user reported that
Scope 1 and Scope 2 ran in the Viewer, but **the refactored Viewer regression has
not been executed here**. CPU comparisons do not establish physical stability.

## Ubuntu Viewer regression

Use the existing environment and checkpoint variable:

```bash
git pull --ff-only origin codex/minimal-carrybox-loco-pickup
TEACHER=/home/han/pyx/PhysHSI-carrybox/legged_gym/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt
python legged_gym/legged_gym/scripts/play_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --student_checkpoint "$TEACHER" \
  --allow_legacy_contract --mode stand --num_envs 1 --steps 1000 \
  --csv logs/handoff_native_stand.csv
```

The same fixed/mixed CLI modes and rate limits remain available. This remains
inference validation only; no Scope 3 or PPO/AMP training was added.
