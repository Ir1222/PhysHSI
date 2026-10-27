"""Compatibility/debug CLI using the native task registry and shared training runner."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import isaacgym  # Must precede torch on the Ubuntu Isaac Gym machine.
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.g1.carrybox_mixed_handoff_config import MixedHandoffCfg, MixedHandoffTrainingCfg, MixedHandoffCfgPPO
from legged_gym.envs.motionlib.motionlib_carrywith import validate_carrywith_config
from legged_gym.utils import task_registry, class_to_dict


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_init_checkpoint", help="Fresh Actor/std initialization; defaults to Teacher")
    parser.add_argument("--amp_expert_config")
    parser.add_argument("--amp_motion_root", help="Directory containing the three original carrywith*.pt files")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps_per_env", type=int, default=100)
    parser.add_argument("--max_iterations", type=int, default=20000, help="Collection iterations, including Teacher warmup")
    parser.add_argument("--max_optimizer_updates", type=int, default=None, help="Optional nonempty rollout update limit")
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--ablation", choices=["A", "B", "C", "D", "historical"],
                        help="Controlled 20s A/B/C/D or historical 60s full-Approach baseline")
    parser.add_argument("--reset_mode", choices=["default", "random", "hybrid"])
    pickup = parser.add_mutually_exclusive_group()
    pickup.add_argument("--pickup_rsi", dest="pickup_rsi", action="store_true")
    pickup.add_argument("--no_pickup_rsi", dest="pickup_rsi", action="store_false")
    parser.set_defaults(pickup_rsi=None)
    parser.add_argument("--pickup_rsi_window", choices=["full", "contact"])
    parser.add_argument("--pickup_contact_window_s", type=float, nargs=2, metavar=("START", "END"),
                        help="Seconds relative to each clip's contact_index; clipped to clip bounds")
    parser.add_argument("--inspect_pickup_rsi", action="store_true", help="Print sampled reset clip/time and physical states (opt-in Viewer without --headless)")
    parser.add_argument("--stable_duration_s", type=float)
    parser.add_argument("--student_initial_stand_duration_s", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path")
    parser.add_argument("--detailed_handoff_events", action="store_true", help="Opt in to detailed diagnostics and event export")
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
    if (min(args.num_envs, args.steps_per_env, args.max_iterations) < 1
            or (args.max_optimizer_updates is not None and args.max_optimizer_updates < 1)
            or (args.episode_length_s is not None and not 0 < args.episode_length_s < float("inf"))):
        parser.error("Environment counts, lengths and update limits must be positive")
    if args.resume != bool(args.resume_path) or (args.resume and args.student_init_checkpoint):
        parser.error("Use --resume --resume_path together; fresh --student_init_checkpoint cannot be combined with resume")
    if not Path(args.teacher_checkpoint).is_file():
        parser.error("Missing external frozen Teacher checkpoint: " + args.teacher_checkpoint)
    return args


def configure_environment(env_cfg, args):
    """Apply a named ablation, then explicit overrides to the native config."""
    if args.ablation:
        pickup = args.ablation in ("B", "C", "D")
        env_cfg.pickup_rsi.enabled = pickup
        env_cfg.asset.box.reset_mode = "random" if pickup else "default"
        env_cfg.env.episode_length_s = 60. if args.ablation == "historical" else 20.
        env_cfg.handoff.stable_duration_s = .1 if args.ablation == "D" else .3
        env_cfg.handoff.student_initial_stand_duration_s = 0. if args.ablation in ("C", "D") else .5
    if args.pickup_rsi is not None:
        env_cfg.pickup_rsi.enabled = args.pickup_rsi
        env_cfg.asset.box.reset_mode = "random" if args.pickup_rsi else "default"
    if args.reset_mode is not None:
        env_cfg.asset.box.reset_mode = args.reset_mode
        if args.reset_mode == "default":
            env_cfg.pickup_rsi.enabled = False
    if args.pickup_rsi_window is not None:
        env_cfg.pickup_rsi.window = args.pickup_rsi_window
    if args.pickup_contact_window_s is not None:
        start, end = args.pickup_contact_window_s
        if not (-float("inf") < start <= end < float("inf")):
            raise ValueError("PickUp contact window must be finite and ordered")
        env_cfg.pickup_rsi.contact_window_s = (start, end)
    for name in ("stable_duration_s", "student_initial_stand_duration_s"):
        if getattr(args, name) is not None:
            value = getattr(args, name)
            if not 0 <= value < float("inf") or (name == "stable_duration_s" and value == 0):
                raise ValueError("Invalid handoff duration: " + name)
            setattr(env_cfg.handoff, name, value)
    if args.episode_length_s is not None:
        env_cfg.env.episode_length_s = args.episode_length_s
    if env_cfg.pickup_rsi.enabled:
        env_cfg.asset.box.skill = ["pickUp"]
        env_cfg.asset.box.skill_init_prob = [1.0]
    else:
        env_cfg.asset.box.skill = list(MixedHandoffCfg.asset.box.skill)
        env_cfg.asset.box.skill_init_prob = list(MixedHandoffCfg.asset.box.skill_init_prob)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.pickup_rsi.log_resets = args.inspect_pickup_rsi


def train(args):
    env_cfg, train_cfg = MixedHandoffTrainingCfg(), MixedHandoffCfgPPO()
    configure_environment(env_cfg, args)
    print(json.dumps(dict(event="effective_training_environment", env=class_to_dict(env_cfg.env),
                          box=class_to_dict(env_cfg.asset.box), pickup_rsi=class_to_dict(env_cfg.pickup_rsi),
                          handoff=class_to_dict(env_cfg.handoff), seed=args.seed)), flush=True)
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
        # The compatibility CLI keeps ablations; runner/expert construction is
        # shared with train.py through the native registry.
        for name in ("experiment_name", "run_name", "load_run", "checkpoint"):
            if not hasattr(args, name):
                setattr(args, name, None)
        runner, train_cfg = task_registry.make_alg_runner(
            env, args=args, train_cfg=train_cfg)
        runner.learn(args.max_iterations, optimize=not args.collect_only,
                     max_optimizer_updates=args.max_optimizer_updates)
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    train(get_args())
