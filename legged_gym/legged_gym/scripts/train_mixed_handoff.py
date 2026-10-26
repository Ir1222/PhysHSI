"""Short, explicit Student PPO + 540D AMP training; original train.py is untouched."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

import isaacgym  # Must precede torch on the Ubuntu Isaac Gym machine.
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.g1.carrybox_mixed_handoff_config import MixedHandoffTrainingCfg, MixedHandoffCfgPPO
from legged_gym.envs.motionlib.motionlib_carrywith import CarryWithMotionLib, validate_carrywith_config
from legged_gym.scripts.play_mixed_handoff import actor_contract
from legged_gym.utils import task_registry, class_to_dict
from rsl_rl.runners.mixed_handoff_on_policy_runner import MixedHandoffOnPolicyRunner


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_init_checkpoint", help="Fresh Actor/std initialization; defaults to Teacher")
    parser.add_argument("--amp_expert_config")
    parser.add_argument("--amp_motion_root", help="Directory containing the three original carrywith*.pt files")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps_per_env", type=int, default=100)
    parser.add_argument("--max_iterations", type=int, default=20, help="Collection iterations, including Teacher warmup")
    parser.add_argument("--max_optimizer_updates", type=int, default=5, help="Stop after this many nonempty rollout updates")
    parser.add_argument("--episode_length_s", type=float, default=60.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path")
    parser.add_argument("--allow_legacy_contract", action="store_true",
                        help="Same audited legacy Actor semantics opt-in as play_mixed_handoff.py")
    parser.add_argument("--no_amp", action="store_true", help="Diagnostic task-reward-only PPO")
    parser.add_argument("--collect_only", action="store_true", help="Teacher/Student rollout, no optimization")
    parser.add_argument("--log_dir")
    custom, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = gymutil.parse_arguments(description=__doc__)
    finally:
        sys.argv = original
    vars(args).update(vars(custom))
    args.sim_device, args.play_dataset = args.rl_device, False
    if args.physics_engine != gymapi.SIM_PHYSX:
        parser.error("Mixed handoff requires PhysX")
    if min(args.num_envs, args.steps_per_env, args.max_iterations, args.max_optimizer_updates) < 1 or args.episode_length_s <= 0:
        parser.error("Environment counts, lengths and update limits must be positive")
    if args.resume != bool(args.resume_path) or (args.resume and args.student_init_checkpoint):
        parser.error("Use --resume --resume_path together; fresh --student_init_checkpoint cannot be combined with resume")
    if not Path(args.teacher_checkpoint).is_file():
        parser.error("Missing external frozen Teacher checkpoint: " + args.teacher_checkpoint)
    return args


def train(args):
    env_cfg, train_cfg = MixedHandoffTrainingCfg(), MixedHandoffCfgPPO()
    env_cfg.env.num_envs, env_cfg.env.episode_length_s = args.num_envs, args.episode_length_s
    train_cfg.seed = env_cfg.seed = args.seed
    train_cfg.runner.num_steps_per_env = args.steps_per_env
    train_cfg.runner.max_iterations = args.max_iterations
    train_cfg.runner.max_optimizer_updates = args.max_optimizer_updates
    train_cfg.amp.enabled = not args.no_amp
    expert_config = args.amp_expert_config or train_cfg.amp.expert_config.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    train_cfg.amp.expert_config = str(Path(expert_config).resolve())
    if train_cfg.amp.enabled:
        validate_carrywith_config(expert_config, args.amp_motion_root)  # Fail before allocating PhysX.
    env, _ = task_registry.make_env("carrybox_mixed_handoff", args=args, env_cfg=env_cfg)
    try:
        expert = None
        if train_cfg.amp.enabled:
            expert = CarryWithMotionLib(expert_config,
                                       env_cfg.dataset.joint_mapping_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
                                       env.dof_names, env_cfg.dataset.frame_rate, args.rl_device,
                                       env.dt, env_cfg.rewards.thresh_robot2object, args.amp_motion_root)
        cfg = class_to_dict(train_cfg)
        cfg["environment"] = class_to_dict(env_cfg)
        log_dir = args.log_dir or str(Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name /
                                     datetime.now().strftime("%b%d_%H-%M-%S_scope3"))
        # Explicit isolated runner selection; registry's original HIM dispatch is unchanged.
        runner = MixedHandoffOnPolicyRunner(env, cfg, args.teacher_checkpoint, actor_contract(env),
                                            args.student_init_checkpoint, expert, log_dir, args.rl_device,
                                            args.allow_legacy_contract, initialize=not args.resume)
        if args.resume:
            runner.load(args.resume_path)
        runner.learn(args.max_iterations, optimize=not args.collect_only,
                     max_optimizer_updates=args.max_optimizer_updates)
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    train(get_args())
