"""Single-frame whole-body Pose AMP projection, conditioning, and reward helpers."""
import math

import torch


# Preserve every posture feature from the original PhysHSI 60D frame while
# excluding only base linear and angular velocity (indices 48:54).
KEEP_INDICES = tuple(range(48)) + tuple(range(54, 60))


def project_pose_amp(observations):
    """Project only the latest chronological 60D frame to the 54D pose."""
    if observations.shape[-1] < 60 or observations.shape[-1] % 60:
        raise ValueError("Pose AMP expects one or more chronological 60D PhysHSI frames")
    current_frame = observations[..., -60:]
    return current_frame[..., list(KEEP_INDICES)]


def classify_pose_mode(executed_command, linear_threshold, yaw_threshold):
    """Return True for Stand; yaw-only, lateral, and linear commands are Moving."""
    if executed_command.shape[-1] != 3:
        raise ValueError("Pose AMP mode classification expects [vx, vy, yaw] commands")
    if (not math.isfinite(linear_threshold) or linear_threshold < 0
            or not math.isfinite(yaw_threshold) or yaw_threshold < 0):
        raise ValueError("Pose AMP Stand thresholds must be finite and nonnegative")
    return ((executed_command[..., :2].norm(dim=-1) <= linear_threshold)
            & (executed_command[..., 2].abs() <= yaw_threshold))


def compute_pose_tracking_gate(linear_error, yaw_error, linear_sigma, yaw_sigma, gate_floor):
    """Compute beta + (1-beta) exp(-e_lin/sigma_lin) exp(-e_yaw/sigma_yaw)."""
    if linear_error.shape != yaw_error.shape:
        raise ValueError("Linear and yaw tracking errors must have identical shapes")
    if (not torch.isfinite(linear_error).all() or not torch.isfinite(yaw_error).all()
            or (linear_error < 0).any() or (yaw_error < 0).any()):
        raise ValueError("Tracking errors must be finite and nonnegative")
    if (not math.isfinite(linear_sigma) or linear_sigma <= 0
            or not math.isfinite(yaw_sigma) or yaw_sigma <= 0):
        raise ValueError("Tracking sigmas must be finite and positive")
    if not math.isfinite(gate_floor) or not 0 <= gate_floor <= 1:
        raise ValueError("Pose AMP gate_floor must be in [0, 1]")
    quality = torch.exp(-linear_error / linear_sigma) * torch.exp(-yaw_error / yaw_sigma)
    return gate_floor + (1. - gate_floor) * quality


def add_gated_pose_reward(task_reward, pose_raw, gate, pose_valid, dt, style_weight):
    """Add the dt-scaled Pose AMP bonus without attenuating task reward."""
    if (task_reward.ndim != 1 or not
            (task_reward.shape == pose_raw.shape == gate.shape == pose_valid.shape)):
        raise ValueError("Pose AMP tensors must have identical one-dimensional shapes")
    if (not math.isfinite(dt) or dt <= 0
            or not math.isfinite(style_weight) or style_weight < 0):
        raise ValueError("Pose AMP dt must be positive and style_weight nonnegative")
    bonus = torch.where(pose_valid, dt * style_weight * gate * pose_raw, 0.)
    return task_reward + bonus, bonus


def condition_pose_observation(pose_observation, stand_mode):
    """Append one Stand(1)/Moving(0) condition to the 54D pose."""
    if pose_observation.shape[-1] != 54 or stand_mode.shape != pose_observation.shape[:-1]:
        raise ValueError("Pose AMP conditioning expects 54D pose and one aligned mode label")
    return torch.cat((pose_observation, stand_mode.to(pose_observation.dtype).unsqueeze(-1)), dim=-1)


def pose_amp_valid(student, dones):
    """Every executed Student step, including a terminal step, owns its pose."""
    if student.shape != dones.shape or student.dtype != torch.bool or dones.dtype != torch.bool:
        raise ValueError("Pose AMP ownership masks must be aligned boolean tensors")
    return student.clone()
