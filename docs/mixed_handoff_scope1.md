# Scope 1: frozen pickup Teacher to mixed-velocity Student

This is an inference integration, with no training or policy improvement claim.
The existing `carrybox` task, `play.py`, checkpoint files, robot/box assets, PD
gains, action scale, rewards, motion references and datasets are unchanged.

## Files

- `legged_gym/legged_gym/envs/__init__.py`: independent task registration.
- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py`: subclass/configuration,
  physical gate, command observation ownership, reset-safe diagnostics.
- `legged_gym/legged_gym/mixed_handoff.py`: small simulator-independent two-phase
  state, command sampler/rate limiter and frozen Actor loader.
- `legged_gym/legged_gym/scripts/play_mixed_handoff.py`: inference/viewer CLI.
- `tests/test_mixed_handoff.py`: CPU tests using tensors and real environment hooks.
- `docs/mixed_handoff_scope1.md`: execution and validation notes.

Implementation stays on `codex/minimal-carrybox-loco-pickup`, as requested.
The supplied task's older working directory and proposed new branch do not
override that instruction. Reference branches were read with `git show` from
`D:\AAAProject\0806\PhysHSI`; neither checkout was switched or merged.
The old handoff's Actor extraction was adapted without its three phases,
action adapter, critic context, reward masks, PPO or imitation dependencies.
The velocity branch's weighted dead-zone sampling and mixed-command rejection
semantics were reused locally without importing its locomotion environment.

## Execution and ownership

The task inherits normal default episode initialization, so Teacher must
approach and physically pick up the box. The normal `env.reset()` settling
step precedes the viewer's logged policy loop. Episodes default to 60 seconds;
physical failures and horizon truncations are recorded separately, including
simultaneous failure and timeout.

Only `TEACHER_CONTROL=0` and `STUDENT_CONTROL=1` exist. The viewer evaluates only
the owning Actor for each environment, under `torch.no_grad()`. Both Actors are
in eval mode with all gradients disabled. No optimizer, critic or discriminator
is instantiated by the viewer. Inherited motion-library/AMP observation buffers
remain environment dependencies; no new AMP data or training is introduced.

The Teacher retains the original task observation logic: `[0.85,0,0]` until
the existing reached-box latch sets, then `[0,0,0]`. During Student control,
only the final three entries of the 15D task feature are replaced with the
Student applied command. `env.commands[:,:3]` receives the same value, making
it available to future tracking rewards. No heading transform can replace it.

After physics and inherited failure checks, the gate requires COM height
strictly above 0.72 m and simultaneous left/right physical contact for 0.3 s,
accumulated using control `dt` (normally 0.02 s). Invalid samples clear the
accumulator; terminal samples cannot switch ownership.

Two configurable validity checks supplement the initial gate:

- Net hand contact forces alone do not identify the collision partner. Require
  force >1 N on each rubber hand AND its palm point within 0.08 m of the box
  volume in box coordinates. Distance never substitutes for contact force.
- A raised support could satisfy an absolute-height condition without a lift.
  Require the rotated box's lowest corner >0.01 m above the original support
  top or ground, whichever is higher.

These are `--contact_force`, `--hand_box_margin`, and `--min_clearance`. No
speed or tilt thresholds were added. Hand bodies use explicit environment-domain
indices into the physical state/contact tensors; inherited Actor features keep
their original construction. The box is the existing symmetric primitive, so
its root center is its COM. Box tilt is measured in radians.

The action on the gate-triggering step remains Teacher-owned in diagnostics.
The post-step phase becomes Student and the next observation contains a zero
Student command. No simulator setter, reset, teleport, action blending, or
extra observation-history update occurs at handoff. Six historical 123D frames
remain intact; the newest frame contains the clipped action actually executed.
Thus Teacher frames legitimately remain in the first Student's 738D input.
Only ordinary episode resets clear history/actions and the handoff state.
Critic observations retain 126 dimensions.

The first Student action always sees `[0,0,0]`, even with zero configured stand
duration. By default Student executes at least 0.5 s of zero-command actions.
Then a target is sampled and the *next* applied command moves by at most
`[0.8,0.6,1.2] * dt` per axis. This bound applies on every later command change.
There is no action interpolation or separate ownership transition phase.

Mixed targets use stand/vx/vy/yaw/mixed probabilities `.1/.1/.1/.1/.6`, ranges
`[-.6,1.2]/[-.5,.5]/[-.7,.7]`, and independent per-environment 4–6 s resampling.
Axis specialists sample uniformly outside the ±0.1 dead zone, weighted by
interval length. Mixed samples are rejected if both XY norm and absolute yaw
are below 0.1 (up to four attempts, matching the reference).

## Checkpoint compatibility

Supply trusted native PhysHSI checkpoints directly. The loader reads only
`checkpoint['model_state_dict']` Actor weights, requires the exact
`738 -> 512 -> 256 -> 256 -> 29` ELU network, validates state_dict keys/shapes
and finite weights, then returns a frozen eval-mode Actor. It does not load
the Critic, optimizer or AMP discriminator for inference. Old action-adapter
checkpoints remain unsupported. Optional `scope1_contract` metadata is ignored
by Actor loading; no metadata injection or compatibility flag is needed.

Simulator joint order is checked against `resources/config/joint_id.txt` during
environment initialization. Full Student resume separately checks network,
PPO/AMP configuration, runtime timestep/joints, environment settings,
Teacher hash and CarryWith expert identity. See the current baseline instructions
in [mixed_handoff_minimal_refactor.md](mixed_handoff_minimal_refactor.md).

## Ubuntu viewer commands

Run in the repository root with the project's Isaac Gym environment activated
and packages installed as for the original `play.py`. Set the Student path to
the audited mixed velocity Actor checkpoint; no fixed-0.8 A5/B2 is selected.

```bash
TEACHER=/home/han/pyx/PhysHSI-carrybox/legged_gym/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt
STUDENT=/absolute/path/to/mixed_velocity/model_19999.pt
PLAYER=(python legged_gym/legged_gym/scripts/play_mixed_handoff.py
  --teacher_checkpoint "$TEACHER" --student_checkpoint "$STUDENT"
  --num_envs 1 --seed 1 --steps 3000
  --episode_length_s 60 --rl_device cuda:0)

# A: Student standing carry throughout.
"${PLAYER[@]}" --mode stand --csv logs/handoff_stand.csv

# B: Ramp to an isolated command after 0.5 s Student stand.
"${PLAYER[@]}" --mode fixed --fixed_command 0.5 0 0 --csv logs/handoff_forward.csv
"${PLAYER[@]}" --mode fixed --fixed_command -0.3 0 0 --csv logs/handoff_backward.csv
"${PLAYER[@]}" --mode fixed --fixed_command 0 0.3 0 --csv logs/handoff_lateral.csv
"${PLAYER[@]}" --mode fixed --fixed_command 0 0 0.4 --csv logs/handoff_yaw.csv

# C: Prescribed mixed distribution and continuous command ramp.
"${PLAYER[@]}" --mode mixed --csv logs/handoff_mixed.csv
```

Additional tuning example: `--stable_duration_s 0.3 --handoff_height 0.72
--student_initial_stand_duration_s 0.5 --command_rates 0.8 0.6 1.2`.
Use `--headless` for non-viewer inference; no training is launched.

CSV records every control step and environment. Arrays are JSON values inside
CSV fields. `executed_phase`, `executed_command` and `target_command` describe
the action just executed; `post_gate_phase`, `next_target_command` and
`next_applied_command` describe the next step before any episode reset.
Reset flags identify an ordinary robot AND box reset following that terminal
sample; the next episode starts in Teacher control. `reset_count` is the count
before that reset. Physical diagnostics are cloned before reset destroys them.
`handoff_time=-1` means no handoff yet. `handoff_failure` means an episode ended
before handoff, including a pickup timeout; `failure` describes a physical
termination. `teacher_warmup_duration` freezes at successful handoff time.
`action_discontinuity` is first Student minus last Teacher action and is valid
only when `action_discontinuity_valid` is true. Event rows print on handoff,
first Student action and reset; use `--csv` for complete traces.

## Validation and remaining risks

```bash
python -m unittest discover -s tests -v
python -m compileall -q legged_gym/legged_gym/mixed_handoff.py \
  legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py \
  legged_gym/legged_gym/scripts/play_mixed_handoff.py tests/test_mixed_handoff.py
git diff --check
```

CPU coverage: checkpoint shape/name/joint-order rejection, source-assumption
opt-in, frozen eval Actors, sustained bilateral gate/reset on invalid contact,
terminal gate veto, no handoff reset or physical-state mutation, 738D/126D
history/action continuity, original Teacher commands and Student ownership,
first zero command, initial stand, repeated rate bounds, mixed sampling/dead
zones/4–6 s interval, asynchronous environment state, terminal diagnostics and
simultaneous physical failure/timeout.

Isaac Gym is unavailable on this Windows machine. Real checkpoint compatibility,
PhysX execution, Actor performance, grasp stability and viewer modes A/B/C
remain **unexecuted**. Tensor fixtures exercise actual environment methods but
do not validate Gym API calls, rotations or mesh collision behavior. First run
Mode A with real checkpoints, inspect gate contacts/clearance and both boundary
actions, then isolated Mode B directions and Mode C. Confirm the box stays held
and all command changes satisfy the CSV rate bounds.

Net forces plus proximity are still an attribution heuristic rather than
pairwise collision reporting. Tune the exposed hand margin against real hand
meshes if necessary. A hard Actor switch can produce a large action jump; this
scope measures it and intentionally does not add an action adapter. A locomotion
Student may fail on standing carry or the Teacher's history/state distribution;
the velocity reference also used a different box-size distribution. Existing
Teacher physical termination rules remain; Student box/contact loss is logged
but adds no new termination or reward design. Any later training, AMP,
tracking/contact rewards or policy robustness work belongs to a separate scope.
