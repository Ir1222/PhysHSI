"""Training-only PickUp sampling; times use native MotionLib frame units."""
import math

import torch


def pickup_window_bounds(motionlib, motion_ids, window="contact", contact_window_s=(-.2, .2)):
    end = motionlib.motion_len["pickUp"][motion_ids].float() - 1
    if window == "full":
        return torch.zeros_like(end), end
    if window != "contact" or len(contact_window_s) != 2:
        raise ValueError("PickUp RSI window must be full or contact with two second offsets")
    low, high = contact_window_s
    if not all(math.isfinite(v) for v in (low, high)) or low > high:
        raise ValueError("PickUp contact window must be finite and ordered")
    contact = motionlib.contact_index[motion_ids].float()
    if ((contact < 0) | (contact > end)).any():
        raise ValueError("PickUp contact index outside clip")
    return ((contact + low * motionlib.fps).clamp_min(0).minimum(end),
            (contact + high * motionlib.fps).clamp_min(0).minimum(end))


def sample_pickup_time(motionlib, motion_ids, settings):
    if settings.window == "full":
        return motionlib.sample_time_rsi("pickUp", motion_ids)
    low, high = pickup_window_bounds(motionlib, motion_ids, settings.window, settings.contact_window_s)
    return low + torch.rand(motion_ids.shape, device=motion_ids.device) * (high - low)


def pickup_box_velocity(motionlib, motion_ids, times):
    """Differentiate the same native, ground-corrected object reference.

    Old PickUp clips have no measured object quaternion; get_obj_motion_state
    supplies its native heading fallback. Use shortest-arc world angular velocity.
    """
    end = motionlib.motion_len["pickUp"][motion_ids].float() - 1
    before, after = (times - .5).clamp_min(0), (times + .5).minimum(end)
    p0, q0, _, _ = motionlib.get_obj_motion_state("pickUp", motion_ids, before)
    p1, q1, _, _ = motionlib.get_obj_motion_state("pickUp", motion_ids, after)
    duration = ((after - before) / motionlib.fps).clamp_min(1e-8)[:, None]
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
    # q1 * conjugate(q0), in xyzw order.
    vector = q0[:, 3:4] * q1[:, :3] - q1[:, 3:4] * q0[:, :3] - torch.cross(q1[:, :3], q0[:, :3], dim=-1)
    scalar = (q0 * q1).sum(-1, keepdim=True)
    norm = vector.norm(dim=-1, keepdim=True)
    angular = vector * (2 * torch.atan2(norm, scalar) / norm.clamp_min(1e-8)) / duration
    return (p1 - p0) / duration, angular
