# Scope 3 training efficiency candidate

Implementation baseline: `6b3f1cb`, branch `codex/minimal-carrybox-loco-pickup`.

## Effective defaults

| Setting | Training candidate D | Standard Viewer |
|---|---|---|
| Environments | 16 | 1 |
| Reset | Native reference reset, PickUp only | Native full Approach |
| Maximum episode | 20 seconds | 60 seconds |
| RSI window | contact_index ± 0.2 seconds, clipped per clip | Disabled |
| Physical gate duration | 0.1 seconds | 0.3 seconds |
| Forced Student Stand | 0 seconds | 0.5 seconds |
| Command slew limits | (0.8, 0.6, 1.2) per second | Same |
| Command resampling | 4–6 seconds | Same |
| PPO rollout | 100 steps per environment | N/A |

`--episode_length_s` defaults to `None`, so only an explicit flag overrides the configuration. Startup prints the effective environment, box/reset configuration, RSI settings, gate and seed. The original reward coefficients, failure rules, actor/critic/action dimensions (738/126/29), six Actor frames, ten AMP frames and CarryWith-only 540D expert discriminator remain unchanged.

## Native RSI and consistency

The training box configuration selects `reset_mode="random"`, `skill=["pickUp"]`, `skill_init_prob=[1.0]`. The existing `_reset_actors()` dispatch still selects RSI. A small environment override samples PickUp IDs and times, calls the native `get_motion_state()`, and fills the native `_reset_ref_*` dictionaries. The existing `_reset_boxes()`, `_reset_task()` and `_reset_env_tensors()` consume these same IDs/times and handle box poses, symmetric box orientation, support placement, task targets and simulator writes. No CarryWith RSI, attachment, handoff teleport or alternate environment is introduced.

The original RSI implementation overwrites root reference velocity with random velocity and can perturb reference joint positions. Those perturbations are skipped for this training-only reset to retain a feasible contact pose. Root and joint velocities use their reference values. The original box reset zeros object velocity; the training override restores finite-difference linear/angular velocities from the same native object reference, including its ground-height correction. This is confined to enabled training PickUp RSI. The clips have no measured box quaternion; native root-heading reconstruction and box yaw symmetry remain. Physical consistency must still be checked in PhysX.

The original platform rule is retained: reference times before contact_index use the support, and times at/after contact use its default position. This boundary deserves visual inspection. Episode reset retains native history and task initialization and clears Student state/AMP validity. Every episode begins in Teacher control. The extra zero-action physics step in native `BaseTask.reset()` refreshes observations but cannot advance the training gate. Handoff never resets Actor histories or physical state.

At 60 motion frames/second, inspected data give:

| Clip | Frames | Last valid frame | Contact frame | Contact time | Default window, frames |
|---|---:|---:|---:|---:|---|
| pickup1.pt | 192 | 191 | 107 | 1.7833 s | 95–119 |
| pickup2.pt | 178 | 177 | 100 | 1.6667 s | 88–112 |
| pickup3.pt | 129 | 128 | 42 | 0.7000 s | 30–54 |

`--pickup_rsi_window full` calls the original full-clip PickUp time sampler. `--pickup_contact_window_s START END` accepts seconds relative to each clip's own contact_index. For example, `-0.8 -0.1` emphasizes pre-contact, `-0.2 0.2` initial contact/lift transition, `0.0 0.5` lift, and `0.5 1.0` later lift/early hold. These are experiment windows, not validated phase labels; inspect the actual motion and physical continuation. Bounds are clipped to `[0, clip_length-1]`; no common normalized phase or fixed clip frame ID is assumed.

## Gate-to-Student timing

At the end of the gate-triggering Teacher action, `complete_step()` changes ownership for the next action. If forced Stand is zero, it immediately samples the ordinary mixed target and takes one slew-limited command increment from zero. At dt=0.02 the maximum first increment is `(0.016, 0.012, 0.024)`. The next Actor observation contains this applied command; its first actual Student action is PPO-valid. Reward reads `executed_command`, captured before physics, so preparing the next target cannot change the preceding Teacher reward. Subsequent commands advance once per action. There is no extra command scheduling delay with zero Stand, and no action blending.

Random Stand remains in the existing mixed distribution. Standard Viewer and named A/B retain their original nonzero forced Stand semantics. Gate A requires 15 consecutive valid 50Hz steps and Gate B requires five. Height, force, palm geometry and clearance thresholds and terminal exclusion are unchanged. Student fall, drop, separation and grasp loss retain their terminations; immediate Student failure is not classified as Teacher handoff failure.

Student-only PPO/GAE and timeout bootstrap implementations are unchanged. AMP frames 1–9 are invalid; frame 10 is valid, including a terminal tenth frame. Reset clears validity. A Student failure before ten frames contributes real PPO failures without fake AMP samples. Existing Actor checkpoints remain loadable, and full resume still validates settings; changing the RSI window requires fresh initialization rather than silently resuming a different environment.

Native episode_length_buf includes the one initialization refresh step. Handoff time uses this native clock; runner Teacher-step/warm-up statistics count actual collected Teacher actions. They can differ by one dt for an initial reset. Failure terminations still take precedence over timeout bootstrap; the original timeout boundary (`episode_length_buf > max_episode_length`) is retained.

## Diagnostics and interpretation

Each iteration records full metrics in `training_metrics.jsonl` and TensorBoard. The terminal uses a compact PhysHSI training panel every 100 collection iterations and at completion. `handoff_events.jsonl` preserves complete per-environment vectors at gates, first Student actions and terminations. It records box height/clearance, both contacts/forces/distances, full base and box linear/angular velocities, tilt, last Teacher action, first actual Student action and its discontinuity. First-action/discontinuity placeholders are omitted before an actual Student action exists. Gate-specific averages use gate rows; discontinuity statistics use first-Student rows.

Metrics include cumulative Teacher/Student/AMP step counts, rollout Student-valid and AMP-valid counts/fractions, actual PPO updates/optimizer steps, completed-episode Teacher steps/warm-up, success/failure rates, zero-Student episode fraction, segment lengths, fraction below ten frames, and fraction of Student segments reaching AMP validity. Handoff success/failure rates use observed gates and pre-gate episode endings; unfinished Teacher episodes are counted separately.

Handoff times, warm-up and Student segment/survival lengths have min/p10/median/p90/max statistics and TensorBoard histograms. `student/completed_survival_s` describes observed physical failure endpoints; `student/censored_timeout_s` is reported separately. Active Student segments are identified as incomplete, with their count and current mean frame count, and are excluded from completed survival statistics. A missing completed-survival metric means no observed failures in that iteration, not zero survival. These summaries are per collection iteration; they are not a censoring-corrected lifetime estimator.

Termination bit flags preserve simultaneous causes: 1 box drop, 2 separation, 4 grasp loss, 8 enabled tilt, 16 inherited physical failure, 32 timeout. Teacher physical failures and episode timeouts are also reported separately. Keep the same checkpoint, seed and physical configuration for A/B/C/D comparisons. Prefer useful samples and feasible optimizer updates with meaningful survival over gate frequency alone.

## Ubuntu validation commands

Run in the existing Isaac Gym Python environment. These commands use direct arguments and do not set DISPLAY, library paths, network addresses or checkpoint-path variables. The working directory is:

```bash
cd /home/han/pyx/PhysHSI-carrybox/legged_gym
```

CPU unit tests (Isaac Gym is not needed):

```bash
python -m unittest discover -s ../tests -p 'test_*.py' -v
```

PickUp RSI state inspection, with Viewer and printed clip/time/root/joints/box/platform/phase data; this also checks the frozen Teacher's physical continuation. Two collection iterations, no optimization:

```bash
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --num_envs 1 --seed 1 --ablation D --inspect_pickup_rsi --collect_only --no_amp --max_iterations 2 --log_dir logs/efficiency_inspect
```

Frozen Teacher/handoff bounded collection, four environments and twenty iterations:

```bash
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --num_envs 4 --seed 1 --ablation D --collect_only --no_amp --headless --max_iterations 20 --log_dir logs/efficiency_collect
```

Controlled collection-only ablations, including 540D AMP validity/reward collection but no optimization:

```bash
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --seed 1 --ablation A --collect_only --headless --max_iterations 20 --log_dir logs/efficiency_A
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --seed 1 --ablation B --collect_only --headless --max_iterations 20 --log_dir logs/efficiency_B
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --seed 1 --ablation C --collect_only --headless --max_iterations 20 --log_dir logs/efficiency_C
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --seed 1 --ablation D --collect_only --headless --max_iterations 20 --log_dir logs/efficiency_D
```

A = full Approach/20s/0.3s gate/0.5s Stand; B = PickUp RSI/20s/0.3s/0.5s; C = PickUp RSI/20s/0.3s/0s; D = PickUp RSI/20s/0.1s/0s. `--ablation historical` retains the original full Approach/60s/0.3s/0.5s reference, whose raw fractions should not be treated as a controlled 20s comparison. Explicit CLI flags override the selected profile. `--no_pickup_rsi` selects full Approach; `--reset_mode default` also disables the training RSI extension. `--reset_mode random` and `hybrid` retain native dispatch, and `--pickup_rsi` enables the training sampler with PickUp-only selection. `--steps_per_env`, `--num_envs`, `--seed`, `--episode_length_s`, `--stable_duration_s`, `--student_initial_stand_duration_s`, `--no_amp`, and `--collect_only` remain independent controls.

Short Student PPO-only smoke (500 collected control steps per environment, at most five updates):

```bash
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --seed 1 --ablation D --no_amp --headless --max_iterations 5 --max_optimizer_updates 5 --log_dir logs/efficiency_ppo
```

Short Student PPO + 540D AMP smoke, same limits:

```bash
python legged_gym/scripts/train_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --allow_legacy_contract --seed 1 --ablation D --headless --max_iterations 5 --max_optimizer_updates 5 --log_dir logs/efficiency_amp
```

Both smoke runs finish at collection iteration 5 and save `model_5.pt`. Check that `optimizer_updates > 0`, `optimizer_steps > 0`, Student-valid count is nonzero, and AMP-valid count is nonzero for an actual discriminator update. Zero counts are diagnostic results, not grounds to bypass the gate or disable physical failure. Student initialization defaults to the same `model_17000.pt` Teacher checkpoint in all commands; `--student_init_checkpoint` can explicitly select a compatible separate initialization.

Full Approach Viewer using the saved Student, standard gate/Stand defaults and mixed commands:

```bash
python legged_gym/scripts/play_mixed_handoff.py --teacher_checkpoint logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt --student_checkpoint logs/efficiency_amp/model_5.pt --allow_legacy_contract --num_envs 1 --seed 1 --mode mixed --steps 3000 --csv logs/efficiency_viewer.csv
```

For full Approach evaluation with candidate gate/command timing, append `--stable_duration_s 0.1 --student_initial_stand_duration_s 0.0` to that Viewer command. `--mode stand` and `--mode fixed --fixed_command 0.8 0 0` remain controlled evaluation options. The Viewer never defaults to RSI. For a visible PPO smoke, omit `--headless`; no address/environment setup command is needed here.

## Validation and remaining risks

Local validation on 2026-10-27: **77 tests passed, zero failures/errors (14.551 seconds)** with CPU PyTorch 2.6.0. Tests execute native sampling/reset functions against all three real PickUp clips, exact prior reward regression fixtures, actual CPU PPO/Adam/AMP, masks, GAE, bootstrapping and checkpoint paths. All eight documented training commands and the Viewer command passed CLI parsing/configuration checks using the existing CPU parser fixtures. Python compilation and `git diff --check` passed. The Windows sandbox required directing temporary test files into the ignored workspace experiments directory; the first attempt using its system temporary directory failed on filesystem permissions.

The Windows host has no Isaac Gym. Ubuntu physical collection, ablations, Teacher continuation, Viewer and GPU optimization have **not been executed locally**. No training-efficiency improvement or physically reliable gate is claimed from unit tests.

The shorter gate can reduce grasp security. Contact-window states may need PhysX settling; missing measured box rotation, the native platform removal boundary and motion/robot collision geometry can affect continuation. Untrained Student failures are legitimate RL samples. If segments are usually below ten frames, PPO can update while AMP cannot; diagnose the physical snapshots and failure distributions before changing the window or gate. This task does not require a stable pretrained Student and does not suppress early termination.

## Files changed

- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py`: dedicated training defaults and RSI settings.
- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py`: native training reset override, consistent reference velocities, initialization gate exclusion and vector diagnostics.
- `legged_gym/legged_gym/mixed_handoff.py`: immediate first command at zero-Stand handoff.
- `legged_gym/legged_gym/pickup_rsi.py`: contact-aware time bounds, sampling and object reference velocities.
- `legged_gym/legged_gym/scripts/train_mixed_handoff.py`: independent overrides, ablations, reset inspection and startup logging.
- `rsl_rl/rsl_rl/runners/mixed_handoff_on_policy_runner.py`: efficiency/survival diagnostics, sparse physical events and compatible RSI resume validation; no PPO/AMP algorithm changes.
- `tests/test_pickup_rsi_efficiency.py`: real-clip/reset/command/diagnostic/compatibility tests.
- `tests/test_mixed_handoff_training.py`: updated training-default expectations; Viewer/reward/rollout checks retained.
- `tests/test_handoff_native_regression.py`: exclude only four new additive diagnostic fields when comparing the unchanged Scope 2 golden snapshot.
- `docs/mixed_handoff_efficiency.md`: configuration, timing, validation commands and limits.

The original `carrybox.py`, `carrybox_config.py`, `train.py`, `play_mixed_handoff.py`, MotionLib implementations, Student rewards, PPO/storage/masked-AMP implementations and frozen regression fixture are unchanged.

## Terminal logger

`MixedHandoffCfgPPO.runner` sets `terminal_log_interval = 100` and `save_interval = 500`. No logging-frequency CLI option is required or provided. Terminal cadence uses the absolute collection iteration, independently of PPO updates and checkpoint saves. The last completed iteration always prints a final panel, including an early stop at the optimizer-update limit. A final iteration on a regular logging boundary prints one panel.

The approximately 80-column panel follows `HIMOnPolicyRunner.log()` with a hash border, centered iteration title, aligned labels and a dashed totals separator. It includes FPS, collection/learning time, value/surrogate losses, action noise, learning rate, Student task/AMP rewards, Student/AMP validity, handoff rate, completed Student survival, actual PPO updates, total steps, elapsed time and ETA. Reward contributions and their standard deviations remain in detailed logs. Missing samples display `N/A`: in particular, AMP reward uses only AMP-valid frames, skipped PPO updates have no losses, and unfinished or timeout-censored Student segments do not supply completed survival times. Sample summaries describe the current collection iteration.

Collection and learning use `time.perf_counter()` with CUDA synchronization at timing boundaries on the runner/environment devices. FPS follows native PhysHSI semantics: collected transitions divided by collection plus learning time. Total time is elapsed wall time for the current `learn()` invocation; ETA extrapolates the remaining collection-iteration budget from this invocation, so resumed iteration counts do not distort it. A final panel has zero ETA, including an optimizer-limit stop. Lifetime timesteps and actual update counts use the existing resumed counters. Checkpoint format and loading rules are unchanged; older runner configurations default to the 100-iteration logging interval.

Every collection iteration still writes all TensorBoard scalars and available histograms, flushes the writer and appends the complete scalar dictionary to `training_metrics.jsonl`. The sparse physical event file remains unchanged. Existing training and Viewer commands above need no additional flags.

Logger validation: **89 Scope 1/2/3 and logger tests passed (18.672 seconds)** on CPU. Added coverage checks panel width/fields, `N/A`, 100/200/final cadence, early optimizer-limit completion, independent save/TensorBoard frequencies, preserved event/metrics files, deterministic timing/FPS, CUDA synchronization calls and legacy resume. With the same seed, logging intervals 1 and 100 produce exactly equal Student weights and identical PPO update/optimizer-step counts. CUDA synchronization is mock-tested here; actual Isaac Gym/GPU execution remains unexecuted on this host.
