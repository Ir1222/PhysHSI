"""Whole-body AMP projection, mode conditioning and Student history ownership."""
import math

import torch


KEEP_INDICES = tuple(range(48)) + tuple(range(54, 60))


def project_amp(observations):
    if observations.shape[-1] != 600:
        raise ValueError("Expected ten chronological 60D PhysHSI AMP frames")
    frames = observations.reshape(*observations.shape[:-1], 10, 60)
    return frames[..., list(KEEP_INDICES)].flatten(-2)


def classify_amp_mode(executed_command, linear_threshold, yaw_threshold):
    """Return True for Stand using all three applied command components."""
    if executed_command.shape[-1] != 3:
        raise ValueError("AMP mode classification expects [vx, vy, yaw] commands")
    if (not math.isfinite(linear_threshold) or linear_threshold < 0
            or not math.isfinite(yaw_threshold) or yaw_threshold < 0):
        raise ValueError("AMP Stand thresholds must be finite and nonnegative")
    return ((executed_command[..., :2].norm(dim=-1) <= linear_threshold)
            & (executed_command[..., 2].abs() <= yaw_threshold))


def compute_tracking_gate(linear_error, yaw_error, linear_sigma, yaw_sigma, gate_floor):
    """Reuse squared Student tracking errors to produce a bounded style gate."""
    if linear_error.shape != yaw_error.shape:
        raise ValueError("Linear and yaw tracking errors must have identical shapes")
    if (not torch.isfinite(linear_error).all() or not torch.isfinite(yaw_error).all()
            or (linear_error < 0).any() or (yaw_error < 0).any()):
        raise ValueError("Tracking errors must be finite and nonnegative")
    if (not math.isfinite(linear_sigma) or linear_sigma <= 0
            or not math.isfinite(yaw_sigma) or yaw_sigma <= 0):
        raise ValueError("Tracking sigmas must be finite and positive")
    if not math.isfinite(gate_floor) or not 0 <= gate_floor <= 1:
        raise ValueError("AMP gate_floor must be in [0, 1]")
    quality = torch.exp(-linear_error / linear_sigma) * torch.exp(-yaw_error / yaw_sigma)
    return gate_floor + (1. - gate_floor) * quality


def add_gated_amp_reward(task_reward, amp_raw, gate, amp_valid, dt, style_weight):
    """Add dt-scaled AMP without attenuating the task reward."""
    if (task_reward.ndim != 1 or not
            (task_reward.shape == amp_raw.shape == gate.shape == amp_valid.shape)):
        raise ValueError("Gated AMP tensors must have identical one-dimensional shapes")
    if (not math.isfinite(dt) or dt <= 0
            or not math.isfinite(style_weight) or style_weight < 0):
        raise ValueError("AMP dt must be positive and style_weight nonnegative")
    bonus = torch.where(amp_valid, dt * style_weight * gate * amp_raw, 0.)
    return task_reward + bonus, bonus


def condition_amp_observation(motion_observation, stand_mode):
    """Append one discriminator condition while preserving the 540D motion vector."""
    if motion_observation.shape[-1] != 540 or stand_mode.shape != motion_observation.shape[:-1]:
        raise ValueError("AMP conditioning expects 540D motion and one aligned mode label")
    return torch.cat((motion_observation, stand_mode.to(motion_observation.dtype).unsqueeze(-1)), dim=-1)


class StudentAMPHistory:
    def __init__(self, num_envs, device):
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.mode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.mode_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def step(self, student, dones, stand_mode=None):
        # The returned history is post-physics, BEFORE reset. A terminal tenth
        # Student frame is valid; the next episode starts from zero.
        if stand_mode is None:
            stand_mode = torch.zeros_like(student)
        if not (student.shape == dones.shape == stand_mode.shape == self.count.shape):
            raise ValueError("Student AMP history masks must match the environment batch")
        changed = self.mode_initialized & (stand_mode != self.mode)
        continuing = student & ~changed
        self.count = torch.where(continuing, self.count + 1, torch.where(student, 1, 0))
        self.mode.copy_(torch.where(student, stand_mode, self.mode))
        self.mode_initialized.copy_(student)
        valid = student & (self.count >= 10)
        self.count[dones] = 0
        self.mode_initialized[dones] = False
        return valid
