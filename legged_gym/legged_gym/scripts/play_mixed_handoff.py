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
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import task_registry, class_to_dict
from legged_gym.mixed_handoff import load_actor, STUDENT_CONTROL


def get_args():
    parameters = [
        {"name": "--teacher_checkpoint", "type": str, "required": True},
        {"name": "--student_checkpoint", "type": str, "required": True},
        {"name": "--allow_legacy_contract", "action": "store_true",
         "help": "Explicitly assume source-audited joint/observation/control semantics for old weights"},
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


def actor_contract(env):
    """JSON-compatible semantics shared by both Actors; not inferred from weights."""
    mapping = Path(LEGGED_GYM_ROOT_DIR) / "resources/config/joint_id.txt"
    canonical = [line.split()[1] for line in mapping.read_text().splitlines() if line.strip()]
    if list(env.dof_names) != canonical or len(canonical) != 29:
        raise ValueError("Simulator joint ordering differs from the audited G1 29-DOF contract")
    return {
        "actor_obs": 738, "critic_obs": 126, "actions": 29, "history_frames": 6,
        "dof_names": list(env.dof_names), "activation": "elu",
        "proprio_features": ["base_ang_vel*0.25", "projected_gravity",
                             "dof_pos-default_dof_pos", "dof_vel*0.05",
                             "end_effectors_hands_feet_head_local", "executed_actions"],
        "task_features": ["box_pos_local", "box_rotation_tan_norm_local", "box_size",
                          "raw_vx_vy_yaw_rate"],
        "observation_frame": env.cfg.control.upper_body_link,
        "observation_scales": class_to_dict(env.cfg.normalization.obs_scales),
        "action_history": "last_executed_clipped_action",
        "dt": env.dt, "control": class_to_dict(env.cfg.control),
        "default_joint_angles": dict(env.cfg.init_state.default_joint_angles),
        "clip_actions": env.cfg.normalization.clip_actions,
        "clip_observations": env.cfg.normalization.clip_observations,
        "observation_normalization": "none",
    }


def play(args):
    from legged_gym.envs.g1.carrybox_mixed_handoff_config import MixedHandoffCfg
    cfg = MixedHandoffCfg()  # Viewer retains full Approach and its original gate.
    cfg.env.episode_length_s = args.episode_length_s
    cfg.handoff.height = args.handoff_height
    for name in ("stable_duration_s", "student_initial_stand_duration_s", "mode",
                 "fixed_command", "contact_force", "hand_box_margin", "min_clearance"):
        setattr(cfg.handoff, name, getattr(args, name))
    cfg.handoff.rates = args.command_rates
    env, _ = task_registry.make_env("carrybox_mixed_handoff", args=args, env_cfg=cfg)
    stream = None
    try:
        contract = actor_contract(env)
        teacher, teacher_spec = load_actor(args.teacher_checkpoint, contract, env.device,
                                           args.allow_legacy_contract)
        student, student_spec = load_actor(args.student_checkpoint, contract, env.device,
                                           args.allow_legacy_contract)
        print(json.dumps({"event": "actor_contract", "teacher": teacher_spec,
                          "student": student_spec, "contract": contract,
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
