"""Inference-only handoff state and Actor loading; no Isaac Gym dependency."""
import math
import re
from dataclasses import dataclass

import torch
from torch import nn

TEACHER_CONTROL, STUDENT_CONTROL = range(2)


@dataclass
class HandoffSettings:
    height: float = 0.72
    stable_duration_s: float = 0.3
    student_initial_stand_duration_s: float = 0.5
    rates: tuple = (0.8, 0.6, 1.2)
    mode: str = "mixed"
    fixed_command: tuple = (0.0, 0.0, 0.0)
    contact_force: float = 1.0
    hand_box_margin: float = 0.08
    min_clearance: float = 0.01

    def __post_init__(self):
        numbers = (self.height, self.stable_duration_s,
                   self.student_initial_stand_duration_s, self.contact_force,
                   self.hand_box_margin, self.min_clearance, *self.rates,
                   *self.fixed_command)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("Handoff settings must be finite")
        if (self.stable_duration_s <= 0 or self.student_initial_stand_duration_s < 0
                or len(self.rates) != 3 or min(self.rates) <= 0
                or len(self.fixed_command) != 3 or self.contact_force <= 0
                or self.hand_box_margin <= 0 or self.min_clearance < 0
                or self.mode not in ("stand", "fixed", "mixed")):
            raise ValueError("Invalid handoff duration, rates, contact thresholds or mode")


def sample_commands(count, device):
    """velocity-tracking semantics: stand / axis specialists / moving mixed."""
    ranges = ((-0.6, 1.2), (-0.5, 0.5), (-0.7, 0.7))
    modes = torch.multinomial(torch.tensor([.1, .1, .1, .1, .6], device=device),
                              count, replacement=True)
    command = torch.zeros(count, 3, device=device)
    for axis, (low, high) in enumerate(ranges):
        ids = (modes == axis + 1).nonzero().flatten()
        # Uniform over the union [low,-.1] U [.1,high], weighted by length.
        u = torch.rand(len(ids), device=device) * (high - low - .2)
        command[ids, axis] = torch.where(u < -.1 - low, low + u, low + u + .2)
    remaining = (modes == 4).nonzero().flatten()
    for _ in range(4):
        if not len(remaining):
            break
        for axis, (low, high) in enumerate(ranges):
            command[remaining, axis] = low + (high - low) * torch.rand(len(remaining), device=device)
        moving = ((command[remaining, :2].norm(dim=-1) >= .1)
                  | (command[remaining, 2].abs() >= .1))
        remaining = remaining[~moving]
    if len(remaining):
        raise RuntimeError("Could not sample moving mixed commands in four retries")
    return command


class HandoffState:
    def __init__(self, count, device, dt, settings):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Control dt must be positive and finite")
        self.dt, self.settings = dt, settings
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.stable_time = torch.zeros(count, device=device)
        self.student_time = torch.zeros_like(self.stable_time)
        self.handoff_time = torch.full_like(self.stable_time, -1.)
        self.resample_time = torch.zeros_like(self.stable_time)
        self.student_target_command = torch.zeros(count, 3, device=device)
        self.student_applied_command = torch.zeros_like(self.student_target_command)

    def reset(self, ids):
        self.phase[ids] = TEACHER_CONTROL
        for tensor in (self.stable_time, self.student_time, self.resample_time,
                       self.student_target_command, self.student_applied_command):
            tensor[ids] = 0
        self.handoff_time[ids] = -1

    def complete_step(self, height, bilateral, clearance, done, episode_time):
        """Evaluate the executed step; prepare ownership/command for the NEXT one."""
        student = self.phase == STUDENT_CONTROL
        valid = ((height > self.settings.height) & bilateral
                 & (clearance > self.settings.min_clearance) & ~done)
        self.stable_time = torch.where(valid & ~student, self.stable_time + self.dt, 0.)
        switched = ~student & (self.stable_time + 1e-7 >= self.settings.stable_duration_s)
        self.phase[switched] = STUDENT_CONTROL
        self.handoff_time[switched] = episode_time[switched]
        self.student_time[student] += self.dt
        self.resample_time[student] -= self.dt
        # With no forced stand, schedule at the gate, for the next Actor obs.
        # The just-executed action/reward still belongs to Teacher.
        command_student = student | (switched & (self.settings.student_initial_stand_duration_s == 0))
        active = (command_student & ~done & (self.student_time + 1e-7
                  >= self.settings.student_initial_stand_duration_s))
        due = (active & (self.resample_time <= 0)).nonzero().flatten()
        if len(due):
            if self.settings.mode == "mixed":
                self.student_target_command[due] = sample_commands(len(due), self.phase.device)
            elif self.settings.mode == "fixed":
                self.student_target_command[due] = self.student_target_command.new_tensor(self.settings.fixed_command)
            self.resample_time[due] = 4 + 2 * torch.rand(len(due), device=self.phase.device)
        bound = self.student_applied_command.new_tensor(self.settings.rates) * self.dt
        delta = (self.student_target_command - self.student_applied_command).clamp(-bound, bound)
        self.student_applied_command[active] += delta[active]
        return switched


def load_actor(path, device="cpu"):
    """Load the native PhysHSI Actor only; optional old metadata is ignored."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Isaac Gym installations with older torch
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: expected training checkpoint dictionary")
    if "handoff_contract" in checkpoint:
        raise ValueError(f"{path}: legacy handoff/action-adapter checkpoint is unsupported")
    state = checkpoint.get("model_state_dict", {})
    if not isinstance(state, dict):
        raise ValueError(f"{path}: model_state_dict must be a dictionary")
    actor_state = {k[6:]: v for k, v in state.items() if k.startswith("actor.")}
    ids = sorted(int(k.split(".")[0]) for k in actor_state
                 if re.fullmatch(r"\d+\.weight", k))
    if not ids or ids != list(range(0, 2 * len(ids), 2)):
        raise ValueError(f"{path}: expected Sequential ELU actor keys actor.0/2/4/...weight")
    layers, dims = [], []
    for i, index in enumerate(ids):
        weight = actor_state[f"{index}.weight"]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or (dims and dims[-1] != weight.shape[1]):
            raise ValueError(f"{path}: inconsistent Actor layer {index}")
        if not dims:
            dims.append(weight.shape[1])
        dims.append(weight.shape[0])
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if i + 1 < len(ids):
            layers.append(nn.ELU())
    if dims != [738, 512, 256, 256, 29]:
        raise ValueError(f"{path}: Actor dimensions {dims}; required 738->512->256->256->29 ELU (output=29)")
    actor = nn.Sequential(*layers)
    try:
        actor.load_state_dict(actor_state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{path}: incompatible Actor parameter names/shapes: {error}") from error
    if not all(torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError(f"{path}: non-finite Actor weights")
    actor.to(device).eval().requires_grad_(False)
    return actor, {"dims": dims, "activation": "elu"}
