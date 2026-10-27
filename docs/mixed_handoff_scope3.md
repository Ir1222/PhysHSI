# Scope 3: Student-only PPO and 540D masked AMP

Implemented against `9db724080e0bdb9044c05bbbeaa353fea9be2d03` on
`codex/minimal-carrybox-loco-pickup`. Validation date: 2026-10-26.
Dataset integration update: 2026-10-27. The three original CarryWith binaries
are now tracked at the default YAML paths and verified without a motion-root override.
No production training was launched. Isaac Gym is unavailable on this Windows
machine; physical smoke tests below remain to be run on Ubuntu.

## Files and integration

New training components:

| File | Purpose |
| --- | --- |
| `rsl_rl/rsl_rl/storage/mixed_handoff_rollout_storage.py` | Explicit ownership, terminal data, GAE and valid-only minibatches |
| `rsl_rl/rsl_rl/algorithms/mixed_handoff_ppo.py` | Student PPO, independent AMP sampling, Adam and gradient checks |
| `rsl_rl/rsl_rl/runners/mixed_handoff_on_policy_runner.py` | Frozen Teacher routing, collection, logging, saving and resume |
| `rsl_rl/rsl_rl/modules/masked_amp.py` | Shared feature projection and consecutive Student-frame counter |
| `legged_gym/legged_gym/envs/motionlib/motionlib_carrywith.py` | Separate CarryWith expert source using existing MotionLib preprocessing |
| `legged_gym/resources/config/carrybox_carrywith_only.yaml` | Three clips, weights 5:2:5 |
| `legged_gym/legged_gym/scripts/train_mixed_handoff.py` | Explicit dedicated runner entrypoint |
| `tests/test_mixed_handoff_training.py` | Deterministic numerical tests |
| `tests/audit_carrywith_amp.py` | Optional CPU audit of real motion files against URDF FK |

Existing changes are confined to the mixed-handoff config, two lazy runner/algorithm
package exports, a lazy GitPython import in the code-snapshot utility, and small
MotionLib hooks. MotionLib now permits zero pickup clips, closes its mapping file,
and exposes file resolution for the separate expert root. Its original expert
sampler and normal pickup behavior are unchanged. Lazy exports retain the original
public names and avoid importing the optional Muon optimizer in CPU tests.

The original `carrybox` environment/config, HIMPPO, HIM runner, task registry,
`train.py`, mixed-handoff environment, gate and Viewer script are unchanged.
The new entrypoint passes a training profile to the registered mixed environment
and instantiates `MixedHandoffOnPolicyRunner` explicitly. It does not depend on
the registry's hard-coded HIM runner dispatch.

## Architecture and initialization

For each environment, pre-step ownership selects either the deterministic frozen
Teacher Actor or the Student's stochastic Normal policy. The runner cross-checks
ownership against the environment's pre-reset `handoff.executed_phase` diagnostic.
The handoff-triggering action remains a Teacher action. No reset, RSI, blending,
imitation loss or extra observation is introduced at handoff.

The intended external Teacher checkpoint is:

```text
/home/han/pyx/PhysHSI-carrybox/legged_gym/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt
```

Teacher loading uses the existing audited `load_actor()` path: semantic
`scope1_contract` validation, ELU architecture, `eval()`, `requires_grad_(False)`
and `torch.no_grad()` during inference. Legacy checkpoints require the existing
explicit `--allow_legacy_contract` source-audit opt-in. Teacher weights are hashed,
excluded from Adam, and compared exactly against a snapshot after every collection
iteration, including iterations without optimization.

Fresh Student initialization defaults to this same Teacher checkpoint, or uses
`--student_init_checkpoint`. Only its Actor and compatible positive finite
`model_state_dict.std` are copied. The required Actor is exactly
`738 -> 512 -> 256 -> 256 -> 29`, ELU, six history frames. ActorCritic creates a
fresh 126D Critic. AMP(540), optional 540D normalizer and Adam are fresh. No Teacher
Critic, optimizer or 600D discriminator is restored. Parameter storage is separate.

Action std is the existing ActorCritic's directly optimized 29-element Parameter,
not log std. Missing or differently shaped source std leaves configured
`init_noise_std=1.0`; an otherwise compatible invalid std is rejected. Std is
updated through PPO/entropy gradients, clamped to at least `1e-4` after Adam,
and saved/resumed in `model_state_dict.std`. Initialization provenance is saved.

## Ownership, returns and optimization

Fixed rollout tensors retain observations, next observations, actions, sampled
Student log probabilities/means/std, Critic values, rewards, done/timeout/failure
flags, next Critic observations/values, next ownership, Student-valid, AMP-valid
and 540D AMP histories. Teacher action rows are zero placeholders; Student is
never asked to supply log probabilities for Teacher-executed actions.

Only `student_valid` indices enter PPO minibatches. Teacher rewards, actions,
values and observation placeholders cannot affect losses, entropy, adaptive KL,
advantage normalization or smoothing. All valid samples, including the final
rollout timestep and minibatch remainders, are used.

For each valid transition, GAE uses `delta = r + gamma * next_value - value`.
Next value is permitted only for a continuing Student state or a pure timeout.
Physical failure takes precedence over a simultaneous timeout. The actual
pre-reset terminal Critic observation returned by the environment supplies timeout
bootstrap; the reset Teacher observation never supplies it. Recursive GAE carries
only through continuing Student transitions in the same episode. At a rollout
boundary, the stored next Student Critic value supplies bootstrap and the
uncollected future GAE tail is zero.

Advantage mean/std use valid samples only, with population std and `1e-8`
stabilization. A single valid sample retains its raw finite advantage. Empty
rollouts skip Adam entirely and continue collection. `collection_iterations`,
`optimizer_updates` (nonempty PPO rollout updates) and `optimizer_steps` (Adam
minibatch steps) are logged separately.

PPO preserves clipped surrogate/value losses, entropy, adaptive KL, Adam and the
original policy/value observation-interpolation smoothness objectives. Smoothness
uses only nonterminal same-phase Student pairs, including valid pairs at the
rollout edge. Combined finite gradients are clipped at the configured norm.
The default learning rate is `2e-4`, following the audited local Carry12
`velocity-tracking` branch; epochs/minibatches, gamma, lambda and KL settings
follow the native configuration. Reward coefficients and handoff thresholds are
unchanged.

## AMP contract and expert data

Every original 60D frame is ordered as follows:

| Original indices | Features | 54D result |
| --- | --- | --- |
| 0 | Base height above the lower ankle | 0 |
| 1..29 | All DOF positions, canonical joint order | 1..29 |
| 30..44 | Left/right palm, left/right ankle, head positions | 30..44 |
| 45..47 | Box position | 45..47 |
| 48..53 | Base linear/angular velocities | Removed |
| 54..59 | Heading-removed root rotation, tangent/normal representation | 48..53 |

The shared helper reshapes `[N,600] -> [N,10,60]`, selects
`range(48) + range(54,60)` within **each** frame, then returns `[N,540]`.
Both policy and expert call this helper. Chronological oldest-to-newest ordering
is preserved. No upper-body joints, end-effectors, box position or rotation
features are removed; no velocity feature enters the discriminator.

Expert preprocessing reuses MotionLib's joint mapping, inverse pelvis quaternion
for root-relative end-effectors/box offset, radial box XY clipping at the same
`thresh_robot2object` (and zero Z outside that radius), base height and
heading-removed rotation functions. The current control observation frame is
pelvis. The legacy fifth `keyframe_head_joint` slot was numerically checked against
the current `mid360_link` position using FK; labels alone are not taken as proof.
The original simulator's observation/cache timing is preserved and still requires
the physical smoke test.

The sampler accepts only `carryWith`, exactly `carrywith1/2/3.pt`; it validates
paths, positive finite weights, tensor contracts, joint mapping and window length.
It preserves YAML weights (default 5:2:5), samples clips first, then samples
uniformly over valid windows in that clip. At 60 fps and control dt 0.02 s,
adjacent expert history frames are 1.2 source frames apart, covering 0.18 s.
Interpolation is confined to each clip, including its final-frame clamp.
No temporal jitter, alternate skill or artificial RSI state is substituted.

The three original binaries are now included in this branch at:

```text
legged_gym/resources/dataset/dataset_carrybox/carryWith/carrywith1.pt
legged_gym/resources/dataset/dataset_carrybox/carryWith/carrywith2.pt
legged_gym/resources/dataset/dataset_carrybox/carryWith/carrywith3.pt
```

The local `velocity-tracking` branch's YAML and PPO configuration were read
without checking out or merging that branch. The migrated files match the
previously audited SHA256 identities below. The YAML's `../dataset/...` paths
already resolve relative to `legged_gym/resources/config/`, so the default
training config works from either checkout with no absolute Windows paths or
`--amp_motion_root`. Directory case is exactly `carryWith`, including on Ubuntu.
`--amp_motion_root` remains an optional override. Missing files still fail before
PhysX allocation with resolved paths. Teacher continues to use its original
`carrybox_loco_pickup.yaml` and synthesized Hold clips.

## AMP validity and rewards

A persistent counter tracks consecutive **post-physics Student-owned** frames
per environment. Teacher steps clear it; the first nine Student frames are
AMP-invalid and the tenth is valid. It persists across collection iterations.
A valid terminal frame can be used, then done clears the counter for the next
episode. Old simulator history bytes cannot become valid until ten fresh Student
frames have replaced them.

Task reward is the unchanged environment reward, already scaled by control dt.
The original runner's discriminator reward scale is retained explicitly:

```text
amp_reward = 0.5 * max(1 - 0.25 * (D(history) - 1)^2, 0)
AMP-invalid Student: combined = task_reward
AMP-valid Student:   combined = 0.75 * task_reward + 0.25 * amp_reward
AMP disabled:        combined = task_reward
```

Blending happens **after** task dt scaling; AMP reward receives no extra dt
factor, matching the native runner baseline. Both coefficients are configurable
in the dedicated config. Teacher receives no Student AMP reward. The existing
AMP least-squares discriminator and expert gradient penalty are reused.

Discriminator policy samples are selected independently from AMP-valid history
rows, irrespective of PPO minibatch membership. Zero valid AMP rows skip
discriminator loss/statistics updates while Student PPO continues. Optional
normalization is 540D and receives raw valid policy and CarryWith expert data
before normalization. Default normalization is disabled.

TensorBoard and console JSON report counts/ratios, handoff outcomes/time,
optimizer counters, losses, entropy, KL, LR, valid advantage statistics,
Actor/Critic/AMP gradient norms, discriminator output moments/penalty, reward
components and all existing Student physical diagnostics. `performance/*`
Student means exclude Teacher rows; reward std is also reported. Per-term raw
and weighted reward contributions preserve the Scope 2 calibration diagnostics.

## Checkpoints

`mixed_handoff_student_v1` stores ActorCritic/std, discriminator, Adam, optional
normalizer moments, collection/update/Adam counters, configuration, Teacher
SHA256/path, initialization provenance, expert SHA256/weights/timing, the 738/126/29
and 540/54/10 training contract, and the original `scope1_contract`.
`play_mixed_handoff.py` can load its Actor directly; no conversion is required.

Full resume rejects Teacher checkpoints, incompatible contracts, changed Teacher
or expert identity, and changed PPO/AMP/environment semantics. Dataset location
may move if content hashes match; environment count and episode length may be
configured for the new run. Simulator state is not checkpointed: resumed training
starts a normal Teacher pickup episode with cleared AMP validity.

## Executed CPU validation

```bash
python -m unittest discover -s tests -v
python tests/audit_carrywith_amp.py
```

All **66 tests passed** (42 previous Scope 1/2 tests plus 24 training/dataset tests). They
exercise real ActorCritic, AMP, Adam and MotionLib on CPU; simulator-dependent
imports are adapted only for the tensor fixtures. Cases include async ownership,
poisoned Teacher placeholders, empty rollouts, hand-calculated GAE, pure timeout
and simultaneous failure, valid-only normalization, smoothness boundaries,
history warmup/reset, expert probabilities/clip boundaries, missing files,
normalizer isolation, CLI, resume and the existing Viewer Actor loader. The
dataset integration test loads the tracked 256/194/323-frame clips using the
default YAML and verifies finite 540D output without an external data override.

A seeded synthetic valid batch (5 Student rows, 5 AMP rows, two Adam minibatches,
LR `2e-4`) produced these maximum absolute parameter changes:

| Component | Change |
| --- | ---: |
| Student Actor | 0.000400263816 |
| Student action std | 0.000398427248 |
| AMP discriminator | 0.000400274992 |
| Frozen Teacher | **0 exactly** |

Actor/Critic/AMP gradient norms were finite and nonzero. The fresh AMP gradient
norm before clipping was much larger than the Actor norm in this synthetic
batch (about 1460 vs 19); the physical short run must inspect their balance.
These are synthetic optimizer checks, not results from the external Teacher
checkpoint or physical learning measurements.

Real-file audit sampled 12,000 finite 540D histories and checked all 773 frames
against URDF FK: 256/194/323 frames respectively. Maximum ankle-position error
was below `2.4e-7 m`; base-height error below `3.5e-8 m`. Legacy hand/head FK
residuals reached 2.823/4.194/3.819 mm by clip and were retained, not recalibrated.
The audited file SHA256 identities are:

```text
carrywith1.pt da7e7cb38bcc2324a9d6aeac347dfb94d567f624a4d0c8cb86f22c860bffafb9
carrywith2.pt a35d464857092be005533adbfbe5b4e77ae637f8616210ddeb342f7fdd886ff0
carrywith3.pt 1f0868733cee5eafeb85f72fd2975c521f8d6d085f7961541c1f55f505c55d29
```

## Ubuntu short smoke commands (not executed here)

Run inside the existing working Isaac Gym environment with this branch installed.
Pull the branch to obtain the three tracked files, then audit the default data
before starting PhysX. All commands are bounded; even zero-Student collection
terminates at `--max_iterations` and reports `updated=0` instead of claiming learning.

```bash
cd /home/han/pyx/PhysHSI-carrybox
git switch codex/minimal-carrybox-loco-pickup
git pull --ff-only origin codex/minimal-carrybox-loco-pickup
export PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl:${PYTHONPATH:-}"
TEACHER="$PWD/legged_gym/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt"
RUN="$PWD/legged_gym/logs/mixed_handoff_student/scope3_smoke_c"
test -f "$TEACHER"
python tests/audit_carrywith_amp.py

# A: Frozen Teacher / real handoff collection, no optimization, no expert needed.
python legged_gym/legged_gym/scripts/train_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --allow_legacy_contract \
  --num_envs 4 --steps_per_env 100 --max_iterations 30 \
  --headless --seed 1 --rl_device cuda:0 --collect_only --no_amp \
  --log_dir legged_gym/logs/mixed_handoff_student/scope3_smoke_a

# B: Student PPO only; stop after one nonempty PPO rollout update.
python legged_gym/legged_gym/scripts/train_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --student_init_checkpoint "$TEACHER" \
  --allow_legacy_contract --num_envs 4 --steps_per_env 100 \
  --max_iterations 30 --max_optimizer_updates 1 --headless \
  --seed 1 --rl_device cuda:0 --no_amp \
  --log_dir legged_gym/logs/mixed_handoff_student/scope3_smoke_b

# C / fresh training: Student PPO + masked CarryWith AMP, at most three updates.
python legged_gym/legged_gym/scripts/train_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --student_init_checkpoint "$TEACHER" \
  --allow_legacy_contract \
  --num_envs 4 --steps_per_env 100 --max_iterations 30 \
  --max_optimizer_updates 3 --headless --seed 1 --rl_device cuda:0 \
  --log_dir "$RUN"

# Find the final saved collection checkpoint, then resume for one further update.
STUDENT=$(python -c 'import pathlib,sys; print(max(pathlib.Path(sys.argv[1]).glob("model_*.pt"), key=lambda p:int(p.stem.split("_")[-1])))' "$RUN")
python legged_gym/legged_gym/scripts/train_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --allow_legacy_contract \
  --num_envs 4 --steps_per_env 100 \
  --max_iterations 30 --max_optimizer_updates 1 --headless --seed 1 \
  --rl_device cuda:0 --resume --resume_path "$STUDENT" \
  --log_dir "${RUN}_resume"

# Existing Viewer: load the saved Student Actor directly.
python legged_gym/legged_gym/scripts/play_mixed_handoff.py \
  --teacher_checkpoint "$TEACHER" --student_checkpoint "$STUDENT" \
  --allow_legacy_contract --num_envs 1 --steps 1500 --seed 1 \
  --rl_device cuda:0 --csv "${RUN}/viewer.csv"

tensorboard --logdir legged_gym/logs/mixed_handoff_student
```

For C, require nonzero `student_valid_count`, `amp_valid_count`, finite nonzero
Actor/Critic/AMP gradients, increasing optimizer counters and a saved checkpoint.
If the gate never succeeds, counts remain zero and this does not pass a learning
smoke test. Inspect JSON/TensorBoard `performance/student_reward_weighted/*`,
contact ratio, slip, relative box motion, tilt and termination breakdown.

## Remaining validation before scale-up

The Linux Teacher checkpoint and real PhysX rollout have not been executed here.
Run A/B/C, resume and Viewer checks before increasing scale. The default expert
paths now contain the tracked binaries after pulling this branch. Short controlled updates
validate infrastructure; they do not establish stable carrying performance.

Scope 2's known hand-face convention mismatch, box-tilt and arm-posture penalties
remain unchanged. They can penalize the real Teacher Hold pose and obstruct
Student learning. Their numerical contributions will be reported by the new
training logs; physical contribution values cannot be claimed from these CPU
fixtures. Reward/contact calibration should be a separate change after smoke
diagnostics. No additional domain randomization or curriculum was enabled.
The training profile uses 16 environments, 60 s episodes and `test=False`, keeps
default pre-pickup resets, and inherits existing box/robot randomization with
disturbance, delay and pushes disabled, exactly as in the mixed Viewer profile.
