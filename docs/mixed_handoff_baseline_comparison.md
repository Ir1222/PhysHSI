# Baseline comparison before implementation

Audited on 2026-10-27, before editing implementation. MixedHandoff and Policy 1:
`6beaf95a541993e0c8ae1f20dc51d1261b2a2132` (local HEAD equals tracked
`origin/codex/minimal-carrybox-loco-pickup`). Official PhysHSI:
`upstream/main` = `07bf22bdaf9c23c1d40366334a744c1816b6d42b`.
Carry12: downloaded the five requested files plus inherited `carrybox.py` from
[velocity-tracking](https://github.com/Ir1222/Carry12/tree/velocity-tracking).
This is a branch snapshot, not a claim about every historical Carry12 version.

Paths below are relative to each repository. `E` = `legged_gym/legged_gym`,
`R` = `rsl_rl/rsl_rl`; MH = MixedHandoff.

| Component | Official PhysHSI | Current Policy 1 | Carry12 | Current MH | Final decision |
|---|---|---|---|---|---|
| Network / action std | `R/modules/actor_critic.py:ActorCritic`: Sequential ELU, directly learned std | Same class/configured 512/256/256 actor; pretrained std | Same ActorCritic; `carry_locomotion_on_policy_runner.py:load_init_policy` copies actor/std | `mixed_handoff_on_policy_runner.py:initialize_student` copies actor/std, independent critic; frozen `load_actor` Teacher | Preserve 738→512→256→256→29, critic, and std semantics |
| Observations / frames | `E/envs/g1/carrybox.py:compute_observations`: six 123D actor frames, 126D critic; original task tail is local goal position | `compute_task_observations`: tail becomes approach/hold velocity command; no success sentinel | inherited CarryBox observation path plus applied `carry_policy_commands` | `carrybox_mixed_handoff.py:step/compute_task_observations` snapshots executed command; Student gets next applied command | Preserve all feature ordering/noise/history; tracking uses heading-only XY velocity and world-Z angular velocity |
| AMP input / discriminator | `carrybox_config.py:amp`, `R/modules/amp.py:AMP`: 10×60D, 512/256 ReLU trunk + scalar head | Same 600D AMP/network | `carry_locomotion_ppo.py` has no AMP optimizer | `R/modules/masked_amp.py:project_amp/StudentAMPHistory`: 10×54D drops each frame's six velocity features; native AMP class | Preserve 540D mask/history and native discriminator; never restore 600D Student AMP |
| Expert sampling | `E/envs/motionlib`, `HIMOnPolicyRunner.__init__`: environment MotionLib | loco/pickUp/hold motion YAML | CarryWith RSI reference; PPO-only learning | `motionlib_carrywith.py:CarryWithMotionLib/validate_carrywith_config`: independent CarryWith-only expert | Retain independent library and identity checks; Teacher dataset unchanged |
| Reset / RSI | `carrybox.py:_reset_actors/_reset_ref_state_init/_reset_boxes`: hybrid RSI; randomizes reference root velocity | Working PickUp/hold semantics | `carrybox_locomotion.py:_reset_actors/_reset_task`: CarryWith reset | MH overrides reference sampling and box velocity only for training PickUp RSI; clears phase/history on episode reset | Keep extension: copying original random velocity at contact is not physically equivalent; no physical/history reset at handoff |
| Post-physics | `carrybox.py:post_physics_step`: refresh → state/callback → termination → reward → AMP → terminal critic → reset → observations | Same lifecycle | adds carry/contact/lower-body state hooks | MH `check_termination` computes gate geometry and universal Student cache before rewards/reset | A retains calculations; B removes posture/derivative-only cache, shares physical contact geometry |
| Rewards | `_prepare_reward_function/compute_reward`: nonzero methods, dt once, positive clipping then termination | seven original actuator/joint regularizers + pickup/hold tasks | tracking scales 3/2.5 plus numerous posture terms | MH separate registrations, 31 Student terms, many cloned raw/weighted diagnostic tensors | A leaves old numerics intact; B exact 13-term Student list, Teacher methods/scales untouched, debug-only detailed reward snapshots |
| Termination | `carrybox.py:check_termination`: physical failures/timeouts; optional box rules | Same | adds drop/separation/grasp/tilt | MH physical failures distinguished from timeout; Student drop/separation/grasp; optional tilt disabled | Preserve thresholds, grace/filter and simultaneous reason bits; no reward for failures |
| PPO / GAE | `him_ppo.py:process_env_step/compute_returns/update`, `him_rollout_storage.py:compute_returns`: all single-policy samples | Same | `carry_locomotion_ppo.py`: PPO + smoothness, no discriminator | `mixed_handoff_ppo.py:ppo_loss/update`, MH storage: Student-only masked trace, terminal critic timeout bootstrap, failure precedence | Retain MH implementation and numerical tests. Native single-policy GAE cannot encode ownership boundaries |
| Runner / storage | `HIMOnPolicyRunner.learn`; native storage infers next obs by shifting and drops last transition in minibatches | Same | dedicated locomotion runner/storage | explicit next actor/critic observations, values, Student/AMP masks; retains final transition | Keep required fields: next actor/critic needed for smoothness; terminal critic needed for timeout. No unproven memory deduplication |
| CLI / registry | `E/scripts/train.py`, `utils/task_registry.py`: task registration and HIM runner | Native carrybox interface | `train_carry_locomotion.py`: task runner | MH registered with Viewer cfg + single-policy PPO cfg; only dedicated train script selects correct runner | Register training cfg/PPO cfg; explicit task runner dispatch; shared constructor and loop; preserve carrybox branch |
| Logs / checkpoints | native terminal panel/TensorBoard/save/load in HIM runner | Same | locomotion panel + init-policy loading | MH per-step metric `.item()`, full diagnostic CPU copies on events; strict Teacher/expert/settings checkpoint checks | GPU reductions until iteration end, sparse selected event rows/fields, detailed snapshots opt-in; keep checks and native panel 100/save 500 |

## Mechanism and dependency decisions

| Additional mechanism and source | Classification | Decision / dependencies |
|---|---|---|
| `E/mixed_handoff.py:HandoffState.complete_step`, MH `step` | Essential for two-policy training | Keep physical gate, next-action ownership, executed phase/command, slew limiting, mixed resampling; gate step is Teacher-owned |
| MH `_reset_ref_state_init/_reset_boxes`, `E/pickup_rsi.py` | Essential for two-policy training | Keep PickUp-only contact-window RSI and coherent reference velocities; no warmup/blending/attachment |
| MH `_prepare_student_reward_function` | Reusable from original PhysHSI | Keep independent method registration/dt scaling and Student episode sums |
| MH `_update_student_carry_state` tracking/contact/drop/separation/grasp | Essential | Keep heading XY/world yaw tracking, force + box proximity, one-frame contact filter and grace timer |
| Same function arm/hip/foot/knee/waist/torso references and hand-slip/relative-pose derivatives | Redundant after B; potential training-performance problem | Delete with removed rewards, including index lists, constants and history buffers. They are still required in A |
| MH `check_termination` rotated box bottom and palm proximity | Essential | Keep: clearance gates and rotated-bottom drop detection require real geometry |
| MH `compute_reward` full raw/weighted clones | Debug-only; potential training-performance problem | B keeps only optional detailed snapshots; production logs retain compact active-term contributions/episode sums |
| MH runner `collect_step/learn` scalar metric conversion and full event rows | Potential training-performance problem | A batch GPU statistics and sparse events; retain meaningful distributions and ownership/validity metrics |
| MH runner frozen Teacher snapshot and semantic/expert checks | Essential | Keep; checked per iteration or load, not reward shaping |
| MH storage next observations, next critic, next values, masks | Essential | They serve smoothness, terminal bootstrap and ownership. Keep explicit storage, including final rollout step |
| Dedicated train script environment ablations | Debug-only | Retain useful overrides; route construction through registry so there is one loop |

## Baseline verification and limits

Before implementation: 89 CPU tests passed (Python 3.12.8, torch 2.6.0+cpu).
Tests use actual tensor math, ActorCritic, AMP, PPO/Adam and synthetic physics
fixtures. This Windows machine has no CUDA/Isaac Gym; the external model_17000.pt
is not bundled. Physical pickup/handoff, original external checkpoint loading,
GPU speed/memory and Viewer stability remain Ubuntu validation requirements.

Read the requested official `carrybox`, config, train, registry, HIM PPO/runner/
storage, ActorCritic and AMP sources; compared Policy 1 diffs and Carry12's five
requested files plus inherited base. Original code is the architectural reference,
not a reason to remove two-policy masks or restore native GAE's single-policy trace.
