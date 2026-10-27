"""Frozen Teacher -> physical gate -> Student stand -> smooth SE(2) commands."""
import argparse
import csv
import json
import sys
from pathlib import Path

import isaacgym  # Must precede torch on the Isaac Gym training machine.
from isaacgym import gymapi, gymutil
import torch

from legged_gym.envs import *  # Register the independent task.
from legged_gym.utils import task_registry
from legged_gym.mixed_handoff import load_actor, STUDENT_CONTROL


def get_args():
    parameters = [
        {"name": "--teacher_checkpoint", "type": str, "required": True},
        {"name": "--student_checkpoint", "type": str, "required": True},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--headless", "action": "store_true"},
        {"name": "--steps", "type": int, "default": 3000},
        {"name": "--episode_length_s", "type": float, "default": 60.},
        {"name": "--mode", "choices": ["stand", "fixed", "mixed"], "default": "mixed"},
        {"name": "--fixed_command", "type": float, "nargs": 3, "default": [0., 0., 0.]},
        {"name": "--handoff_height", "type": float, "default": .72},
        {"name": "--stable_duration_s", "type": float, "default": .3},
        {"name": "--student_initial_stand_duration_s", "type": float, "default": .5},
        {"name": "--command_rates", "type": float, "nargs": 3, "default": [.8, .6, 1.2]},
        {"name": "--contact_force", "type": float, "default": 1.},
        {"name": "--hand_box_margin", "type": float, "default": .08},
        {"name": "--min_clearance", "type": float, "default": .01},
        {"name": "--csv", "type": str, "help": "Optional per-control-step, per-environment diagnostics"},
    ]
    # Isaac Gym's custom-parameter adapter only supports name/type/default/
    # action/help. Parse our CLI with argparse so choices, required and nargs
    # retain their normal semantics; forward only Gym's own flags to gymutil.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for parameter in parameters:
        parser.add_argument(parameter["name"], **{k: v for k, v in parameter.items() if k != "name"})
    custom_args, gym_args = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *gym_args]
        args = gymutil.parse_arguments(description=__doc__)
    finally:
        sys.argv = original_argv
    vars(args).update(vars(custom_args))
    args.sim_device = args.rl_device
    args.play_dataset = False
    if args.physics_engine != gymapi.SIM_PHYSX:
        raise ValueError("Mixed handoff requires PhysX")
    if args.num_envs < 1 or args.steps < 1 or not args.episode_length_s > 0:
        raise ValueError("num_envs, steps and episode_length_s must be positive")
    return args


def play(args):
    from legged_gym.envs.g1.carrybox_mixed_handoff_config import MixedHandoffCfg
    cfg = MixedHandoffCfg()  # Viewer retains full Approach and its original gate.
    cfg.diagnostics.detailed = bool(args.csv)
    cfg.env.episode_length_s = args.episode_length_s
    cfg.handoff.height = args.handoff_height
    for name in ("stable_duration_s", "student_initial_stand_duration_s", "mode",
                 "fixed_command", "contact_force", "hand_box_margin", "min_clearance"):
        setattr(cfg.handoff, name, getattr(args, name))
    cfg.handoff.rates = args.command_rates
    env, _ = task_registry.make_env("carrybox_mixed_handoff", args=args, env_cfg=cfg)
    stream = None
    try:
        teacher, teacher_spec = load_actor(args.teacher_checkpoint, env.device)
        student, student_spec = load_actor(args.student_checkpoint, env.device)
        print(json.dumps({"event": "actor_checkpoint", "teacher": teacher_spec,
                          "student": student_spec,
                          "teacher_checkpoint": args.teacher_checkpoint,
                          "student_checkpoint": args.student_checkpoint,
                          "seed": args.seed, "settings": vars(env.handoff.settings)}))
        if args.csv:
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            stream = open(args.csv, "w", newline="", encoding="utf-8")
        writer = None
        with torch.no_grad():
            obs, _ = env.reset()  # Normal default initialization, before Teacher pickup.
            for step in range(args.steps):
                student_ids = env.handoff.phase == STUDENT_CONTROL
                actions = torch.empty_like(env.actions)
                if (~student_ids).any():
                    actions[~student_ids] = teacher(obs[~student_ids])
                if student_ids.any():
                    actions[student_ids] = student(obs[student_ids])
                obs, _, _, _, infos, _, _, _ = env.step(actions)
                diagnostics = infos["handoff"]
                values = {k: v.cpu().tolist() for k, v in diagnostics.items()}
                for env_id in range(env.num_envs):
                    row = {"step": step, "env_id": env_id,
                           **{k: v[env_id] for k, v in values.items()}}
                    if stream:
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                                         for k, v in row.items()})
                    if row["handoff_trigger"] or row["first_student_step"] or row["robot_reset"]:
                        print(json.dumps(row), flush=True)
    finally:
        if stream:
            stream.close()
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    play(get_args())
