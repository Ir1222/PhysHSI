# Scope 2: Student carry rewards and failure diagnostics

This extends only `carrybox_mixed_handoff`. For the subsequent behavior-preserving
file/method migration and numerical comparison, see [native refactor](mixed_handoff_native_refactor.md). It defines rewards for diagnostics
and later training; neither Actor is updated. Scope 1's checkpoint loader,
ownership state, physical gate, command distribution/ramp, viewer CLI, PD,
assets, and 738D/126D/29D contracts are unchanged. No PPO, AMP, storage, rollout
mask, network, or dataset changes are included.

## Changed files and local sources

- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py`: separate reward
  composition, Student-only failure checks, per-step and episode diagnostics.
- `legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py`: native nested
  Student reward scales, tolerances and termination settings. Mathematical helpers
  and individual reward methods live in the environment; `student_carry.py` was
  removed by the behavior-preserving native refactor.
- `tests/test_mixed_handoff.py`: extend tensor fixtures for Scope 2 while keeping
  the Scope 1 tests and CLI regression cases.
- `tests/test_student_carry.py`: reward, frame, history and termination tests.
- `docs/mixed_handoff_scope2.md`: this guide and complete reward table.

Reference code was read with `git show velocity-tracking:<path>` in the local
`D:\AAAProject\0806\PhysHSI` repository. Sources: `carrybox_locomotion.py` and
`carrybox_locomotion_config.py`, plus the reference `carrybox.py` and
`carrybox_config.py` for tracking frames/sigmas. No branch was merged or fetched.
The original current `carrybox.py` and `carrybox_config.py` remain unchanged.

## Ownership, command and reward composition

All masks use `executed_phase`, captured before physics. The gate-triggering
step receives the original Teacher reward, even though its resulting phase
is Student. The first Student reward follows an actually executed Student action.
Teacher terms and their original `episode_sums` are zero on Student transitions;
Student terms and sums are zero on Teacher transitions. Each applicable Student
regularization appears once, with no inherited Approach/PickUp/Hold additions.

Tracking/stillness/air-time terms use `executed_command`, never the resampled
target or next applied command. Linear velocity is world velocity of the
explicit pelvis reference (`cfg.control.upper_body_link`, currently `pelvis`)
rotated by `calc_heading_quat(pelvis_quat)` inverse. This is heading-only XY;
pelvis pitch/roll cannot mix vertical motion into tracking. Yaw rate is that
body's world-Z angular velocity. XYZW conventions and the repository's actual
quaternion utilities are used. Existing Actor observation construction is intact.

All coefficients below are multiplied by control `dt`, just like PhysHSI's
reward preparation. The inherited `only_positive_rewards` flag (default false)
clips each owned total at zero if enabled. The Teacher's optional termination
term is still added after clipping. There is no new Student terminal reward.
Returned rewards have shape `[num_envs]`; malformed/non-finite rewards fail
explicitly. `student_task_reward` is the weighted sum of the first 16 table
rows, before clipping; `student_total_reward` includes all regularization and
the configured clipping. Individual contributions are recorded before clipping.

## Exact Student reward table

Below, `C(e)=1/(1+sum(e²))`, `D(e,d,s)=exp(-mean((max(abs(e)-d,0)/s)²))`,
and `I(x,l,u,s)=exp(-(range_violation(x,l,u)/s)²)`.
`range_violation=max(l-x,0)+max(x-u,0)`.
Source function names refer to the local reference unless marked **current**.

| Term | Coefficient | Definition / source |
|---|---:|---|
| carry_lin_vel_tracking | 3.0 | `exp(-sum((executed_xy-measured_yaw_xy)²)/0.25)`; `_reward_carry_lin_vel_tracking` in reference `carrybox.py` |
| carry_yaw_vel_tracking | 2.5 | `exp(-(executed_yaw-world_z_rate)²/0.25)`; `_reward_carry_yaw_vel_tracking` in reference `carrybox.py` |
| carry_bilateral_contact | 0.5 | both filtered box-approximate contacts; `_reward_carry_bilateral_contact` |
| carry_hand_box_surface | 1.5 | `C(surface_excess/0.04)`; `_reward_carry_hand_box_surface` |
| carry_hand_slip | 0.5 | `C(max(slip-0.35,0)/0.35)*derivative_valid`; `_reward_carry_hand_slip` |
| carry_relative_velocity | 0.5 | `C(max(abs(relative_velocity)-[.35,.45,.45],0)/[.35,.45,.45])*derivative_valid`; `_reward_carry_relative_velocity` |
| carry_relative_position | 0.75 | `C(range_violation(torso_local_box,[.18,-.20,-.15],[.53,.20,.25])/[.1,.1,.1])`; `_reward_carry_relative_position` |
| carry_arm_range | 0.2 | `C(14-joint range_violation/0.35)`; `_reward_carry_arm_range` |
| carry_box_tilt | -0.5 | sum of squared box-local gravity XY components; `_reward_carry_box_tilt` |
| carry_hip_posture | 0.8 | `D(hip-default_hip,[.08,.10,.08,.10],[.15,.15,.15,.15])`; `_reward_carry_hip_posture` |
| carry_foot_heading | 0.35 | `D(wrapped_foot_minus_pelvis_projected_heading,.12,.20)`; `_reward_carry_foot_heading` |
| carry_feet_width | 0.30 | `I(abs(heading_local_left_minus_right_foot_y),.09,.27,.08)`; `_reward_carry_feet_width` |
| carry_knee_width | 0.20 | `I(abs(heading_local_left_minus_right_knee_y),.14,.25,.06)`; `_reward_carry_knee_width` |
| carry_waist_reference | 0.4 | `D(waist-[-.041372441,.034433149,.249833077],[.30,.07,.12],[.20,.10,.12])`; `_reward_carry_waist_reference` |
| carry_torso_pelvis_alignment | 0.2 | `D(SO(3)_log(inv(reference)*inv(pelvis)*torso),[.12,.12,.30],[.12,.12,.20])`; `_reward_carry_torso_pelvis_alignment` |
| zero_command_stillness | 0.2 | `exp(-4*sum(measured_yaw_xy²)-2*world_z_rate²)` for nonmoving command; `_reward_zero_command_stillness` |
| lin_vel_z | -1.0 | pelvis full-local vertical velocity squared; inherited `_reward_lin_vel_z` formula |
| ang_vel_xy | -0.05 | sum of squared pelvis full-local XY angular velocity; inherited `_reward_ang_vel_xy` formula |
| orientation | -1.0 | sum of squared pelvis-local gravity XY; inherited `_reward_orientation` formula |
| feet_air_time | 0.2 | `sum((feet_air_time-.5)*first_contacts)*moving`; locomotion `_reward_feet_air_time` |
| feet_slip | -0.2 | sum world-XY foot speed when foot Z force >1 N; inherited `_reward_feet_slip` formula |
| feet_stumble | -1.0 | any foot XY force norm >3*abs(Z force); inherited `_reward_feet_stumble` formula |
| collision | -0.5 | **current** `_reward_collision`, penalized-body contact count >0.1 N |
| feet_contact_forces | -0.001 | sum force norm exceeding existing `max_contact_force=350 N`; inherited formula |
| dof_acc | -1e-7 | **current** `_reward_dof_acc` |
| action_rate | -0.03 | **current** `_reward_action_rate` |
| torques | -1e-4 | **current** `_reward_torques`, squared `torques/p_gains` |
| dof_vel | -2e-4 | **current** `_reward_dof_vel` |
| dof_pos_limits | -5.0 | **current** `_reward_dof_pos_limits` |
| dof_vel_limits | -1e-3 | **current** `_reward_dof_vel_limits` |
| torque_limits | -0.03 | **current** `_reward_torque_limits` |

Both tracking sigmas are verified as `.25` in reference `carrybox_config.py`,
with units `(m/s)²` and `(rad/s)²`. Full arm bounds and all numeric tolerances
are in `MixedHandoffCfg.student_rewards`, copied without retuning. The torso reference XYZW is
`[.020052686,.124780931,-.023215466,.991709963]` and is normalized before use.
Hip order is left roll/yaw then right roll/yaw; waist order is yaw/roll/pitch.
These and arm ordering are validated; indices are resolved by joint names.
Feet/knees/pelvis/torso/palms/hands use explicit environment-domain body indices.

Hand face coordinates use the actual randomized half extents. Left is box +Y,
right is box -Y: side excess is `max(abs(sign*hand_y-half_y)-.03,0)`;
face-edge excess is `max(abs(hand_xz)-half_xz-.01,0)`. Slip is the norm of the
finite-difference box-local hand X/Z components, excluding box-Y face-normal
motion. Relative box velocity is the finite difference of torso-local box
position, including rotation of that frame, exactly as in the reference.

## Contacts, temporal validity and reset

Raw hand contact is rubber-hand force norm >1 N. Approximate box contact also
requires Scope 1's box proximity margin. Separate raw and box-approximate states
are each filtered by OR with their own preceding sample, independently per
environment. The bilateral reward and grasp-loss timer use filtered approximate
box contact. This intentionally tightens the reference's net-force-only input;
the scalar reward definition and one-frame OR filter are unchanged. It remains
an approximation, not pairwise collision identification or a grasp attachment.
One lost hand removes the bilateral reward after filtering; complete loss of
both contacts advances the grasp timer.

Derivative validity is false on Teacher transitions, the first actual Student
transition and the first Student transition after an episode reset. Those
derivative measurements and rewards are zero for the invalid sample. Physical
samples are cached once per control step; later valid Student steps use adjacent
real samples. Ordinary reset clears derivative/contact histories, grasp duration
and Student counters for only the reset environments. Handoff itself clears none
of the robot/box state or observation/action history. No PPO/AMP masks are added.

## Student terminations and calibration

New conditions apply only to executed Student steps, leaving Teacher and the
gate-triggering step untouched. Existing robot failure checks and horizon
handling are preserved. `student/termination_reason` is an OR-able bit mask:

| Bit | Condition | Default |
|---:|---|---|
| 1 | rotated box bottom above ground below `box_drop_height` | <0.15 m |
| 2 | pelvis-to-box XY distance above `robot_box_max_distance` | >1.0 m |
| 4 | complete filtered approximate grasp loss | >0.30 s |
| 8 | box local +Z tilt from world +Z above threshold, **only if enabled** | >70°, hard termination disabled |
| 16 | inherited robot physical failure | unchanged |
| 32 | ordinary episode horizon truncation | unchanged |

The drop calculation uses Scope 1's rotation-aware lowest corner and ground
origin, instead of Carry12's upright `COM_z-half_size_z` approximation. It is
not the support clearance used by the gate. The grasp grace period plus contact
filter prevents a single missed sample or empty initialization from terminating
the first Student step. Configurable values live under `MixedHandoffCfg.student_rewards` and
`MixedHandoffCfg.student_termination`; `terminate_box_tilt=False` is the default.

The supplied successful Scope 1 log has tilt `1.7434 rad` (99.89°), exceeding
70° (1.2217 rad). Thus its *logged tilt* fails the old tilt threshold, and hard
tilt termination stays disabled. The XYZW geometric interpretation is tested:
the angle is between the box's local +Z and world +Z, invariant to quaternion
sign. It is not an angle to the nearest equivalent face of a symmetric box.
The retained tilt penalty is `sin(tilt)²`: like the reference it vanishes at
180°, a known limitation, not a newly calibrated stability objective.

Reported clearance .8563 m is well above the gate's .01 m; reported COM height
1.0635 m and positive support clearance are compatible with the .15 m drop
threshold. Base speed .3751 m/s and box angular speed 2.3193 rad/s have no new
hard limits. Separation, palm coordinates, hip/waist joints and torso pose were
not supplied, so compliance with those thresholds/ranges **cannot be verified**
from that log. Gate-time diagnostics now expose every relevant violation.

Reference waist/torso constants were calibrated offline on CarryWith, not this
Teacher checkpoint. Even the current nominal zero waist pitch lies below the
old waist-pitch dead-zone interval `[.129833,.369833]`; that is a nominal-pose
comparison, not a measurement of Teacher Hold. No claim is made that the actual
Hold pose lies inside all arm, hand-face or torso-box workspace bounds. All are
soft rewards only, configurable and unchanged from the reference. Inspect
`hand_side_error`, `arm_range_violation`, `box_position_violation`, `waist_error`
and `torso_alignment_error` at handoff before any later tuning/training.

## Logging and validation

All Scope 1 CSV fields remain. Additional `student/*` physical measurements
are available even on the Teacher/gate step for guardrail calibration; this
does not add Student rewards to Teacher totals. `teacher_reward_raw/*`,
`teacher_reward_weighted/*`, `student_reward_raw/*` and
`student_reward_weighted/*` are ownership-masked and distinct.
`student_reward_before_clip`, `student_task_reward`, `student_total_reward`,
`teacher_total_reward`, `total_reward`, `episode_student_steps` and
`episode_student_mean_reward` are cloned before automatic reset. JSON array
fields remain usable through the existing CSV viewer writer.

The existing `extras['episode']` gets `student/mean_step_reward/<term>` and
`.../total`: per-episode contribution sum divided by actual Student steps,
then averaged only over completed episodes that executed Student steps.
These are per-control-step (dt-scaled), not per-second values.
Episodes with no Student execution are counted separately. Handoff success,
early failure and actual Student-step counts remain distinguishable in CSV.
Old episode entries are cleared at the next step instead of replayed.

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python -m compileall -q legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff_config.py \
  legged_gym/legged_gym/envs/g1/carrybox_mixed_handoff.py tests
git diff --check
```

Tests cover exact Teacher composition through the gate; asynchronous ownership;
first Student semantics; clipping/termination ordering; no double regularization;
executed versus next commands; forward/lateral/yaw/tilted-pelvis frames;
contact filtering/attribution and face geometry; temporal validity and reset;
feasible posture intervals/joint order; Student-only drop/separation/grasp
conditions; default diagnostic-only 100° tilt; separate timeout/failure and
pre-reset measurements; Student-only episode denominators; and Scope 1 history,
gate, command ramp, frozen Actor and CLI regressions.

Scope 1's successful Isaac Gym Viewer run with `model_17000.pt` in both roles
was reported by the user. **Scope 2 Isaac Gym runtime has not been executed**
on this Windows development machine, which has no Isaac Gym. The unit tests
use actual repository quaternion math, not identity rotation mocks for Student
math, but cannot certify real contact behavior or hardware stability.

## Short Ubuntu reward-diagnostics Viewer run

```bash
git pull --ff-only origin codex/minimal-carrybox-loco-pickup
TEACHER=/home/han/pyx/PhysHSI-carrybox/legged_gym/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt

# Same checkpoint in both roles: regression/smoke test, not learned locomotion.
python legged_gym/legged_gym/scripts/play_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --student_checkpoint "$TEACHER" \
  --mode stand --num_envs 1 --steps 1000 \
  --csv logs/handoff_scope2_stand.csv

# With an audited mixed-velocity Actor:
STUDENT=/absolute/path/to/mixed_velocity/model_19999.pt
python legged_gym/legged_gym/scripts/play_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --student_checkpoint "$STUDENT" \
  --mode mixed --num_envs 1 --steps 1500 \
  --csv logs/handoff_scope2_mixed.csv
```

Start with stand. Check that the gate step has only Teacher reward, first Student
derivatives are invalid/zero, contacts persist, and tilt exceedance alone does
not reset. Inspect termination bits and hand/waist/workspace violations. For an
isolated command, substitute `--mode fixed --fixed_command 0.3 0 0` in the same
command. Rewards alone demonstrate task-definition consistency, not locomotion
learning. Scope 3 remains outside this change.
