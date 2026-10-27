"""Continuous handoff with separate Teacher and Student carry objectives."""

import math
import json

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .carrybox import LeggedRobot
from .carrybox_mixed_handoff_config import MixedHandoffCfg
from legged_gym.mixed_handoff import HandoffSettings, HandoffState, STUDENT_CONTROL
from legged_gym.pickup_rsi import sample_pickup_time, pickup_box_velocity
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.torch_utils import calc_heading_quat, quat_conjugate, quat_mul


def range_violation(value, lower, upper):
    return (lower - value).clamp_min(0.0) + (value - upper).clamp_min(0.0)


def constraint_reward(excess):
    return 1.0 / (1.0 + excess.flatten(1).square().sum(-1))


def deadzone_gaussian_reward(error, deadzone, softness):
    excess = (error.abs() - deadzone).clamp_min(0.0)
    return torch.exp(-(excess / softness).square().mean(-1))


def interval_gaussian_reward(value, lower, upper, softness):
    return torch.exp(-(range_violation(value, lower, upper) / softness).square())


def projected_heading(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotation_vector(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[..., 3:4] < 0.0, -q, q)
    vector = q[..., :3]
    norm = vector.norm(dim=-1, keepdim=True)
    half = torch.atan2(norm, q[..., 3:4].clamp(0.0, 1.0))
    return vector * torch.where(norm > 1e-07, 2 * half / norm.clamp_min(1e-12), 2.0)


class MixedHandoff(LeggedRobot):
    def reset(self):
        # BaseTask.reset performs one zero-action physics step to refresh obs.
        # In training it cannot count as an actual frozen-Teacher action.
        self._initializing_training_observations = hasattr(self.cfg, "pickup_rsi")
        try:
            return super().reset()
        finally:
            self._initializing_training_observations = False

    def _reset_ref_state_init(self, env_ids):
        settings = getattr(self.cfg, "pickup_rsi", None)
        if settings is None or not settings.enabled:
            return super()._reset_ref_state_init(env_ids)
        if list(self.skill) != ["pickUp"] or list(self.box_cfg.skill_init_prob) != [1.0]:
            raise ValueError("Training PickUp RSI requires pickUp-only skill/probability")
        # Retain the native reset bookkeeping consumed by _reset_boxes,
        # _reset_task and _reset_env_tensors. Only sampling and reference
        # velocities differ; do not perturb a contact pose with joint noise.
        ids = self.motionlib.sample_motions("pickUp", len(env_ids))
        times = sample_pickup_time(self.motionlib, ids, settings)
        pos, rot, lin, ang, joints, velocities, _ = self.motionlib.get_motion_state("pickUp", ids, times)
        self.root_states[env_ids, :3] = pos + self.env_origins[env_ids]
        self.root_states[env_ids, 3:7] = rot
        self.root_states[env_ids, 7:10] = lin
        self.root_states[env_ids, 10:13] = ang
        self.dof_pos[env_ids] = joints
        self.dof_vel[env_ids] = velocities
        self._reset_ref_env_ids["pickUp"] = env_ids
        self._reset_ref_motion_ids["pickUp"] = ids
        self._reset_ref_motion_times["pickUp"] = times

    def _reset_boxes(self, env_ids):
        super()._reset_boxes(env_ids)
        settings = getattr(self.cfg, "pickup_rsi", None)
        if settings is not None and settings.enabled and "pickUp" in self._reset_ref_env_ids:
            ids = self._reset_ref_env_ids["pickUp"]
            linear, angular = pickup_box_velocity(self.motionlib, self._reset_ref_motion_ids["pickUp"],
                                                  self._reset_ref_motion_times["pickUp"])
            self.box_states[ids, 7:10] = linear
            self.box_states[ids, 10:13] = angular

    def _init_buffers(self):
        super()._init_buffers()
        if (self.num_obs, self.num_privileged_obs, self.num_actions) != (738, 126, 29):
            raise ValueError("Mixed handoff requires the PhysHSI 738/126/29 contract")
        settings = HandoffSettings(
            **{k: getattr(self.cfg.handoff, k) for k in HandoffSettings.__dataclass_fields__}
        )
        self.handoff = HandoffState(self.num_envs, self.device, self.dt, settings)
        self.teacher_last_action = torch.zeros_like(self.actions)
        self.student_first_action = torch.zeros_like(self.actions)
        self.reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # The tensors include platform, target platform and box. Resolve explicit
        # environment-domain indices rather than assuming an actor-handle offset.
        from isaacgym import gymapi

        def index(name):
            value = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.actor_handles[0], name, gymapi.DOMAIN_ENV
            )
            if value < 0:
                raise ValueError("Missing handoff rigid body: " + name)
            return value

        self.torso_index = index("torso_link")
        self.gate_hand_indices = [index(side + "_rubber_hand") for side in ("left", "right")]
        self.gate_palm_indices = [index(side + "_palm_link") for side in ("left", "right")]
        self.student_body_indices = dict(
            pelvis=index(self.cfg.control.upper_body_link),
            torso=self.torso_index,
            palms=self.gate_palm_indices,
            hands=self.gate_hand_indices,
            feet=[index(name) for name in self.cfg.student_rewards.carry_foot_links],
            knees=[index(name) for name in self.cfg.student_rewards.carry_knee_links],
        )
        self._init_student_carry_buffers()

    def _init_student_carry_buffers(self):
        c = self.cfg.student_rewards
        for field in (
            "carry_lin_vel_sigma",
            "carry_yaw_vel_sigma",
            "carry_hand_surface_violation_scale",
            "carry_hand_slip_violation_scale",
            "carry_arm_violation_scale",
            "carry_foot_heading_softness",
            "carry_feet_width_softness",
            "carry_knee_width_softness",
            "hand_contact_threshold",
        ):
            if not math.isfinite(getattr(c, field)) or getattr(c, field) <= 0:
                raise ValueError("Student carry parameter must be positive and finite: " + field)
        for field in ("robot_box_max_distance", "grasp_loss_grace_s"):
            value = getattr(self.cfg.student_termination, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Student carry parameter must be positive and finite: " + field)
        for field in (
            "carry_box_position_violation_scale",
            "carry_box_velocity_violation_scale",
            "carry_hip_posture_softness",
            "carry_waist_reference_softness",
            "carry_torso_pelvis_alignment_softness",
        ):
            value = self.dof_pos.new_tensor(getattr(c, field))
            if not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError("Student carry softness must be positive and finite: " + field)
        for field in ("carry_hip_joint_names", "carry_waist_joint_names", "carry_arm_joint_names"):
            expected = getattr(MixedHandoffCfg.student_rewards, field)
            if (
                tuple(getattr(c, field)) != tuple(expected)
                or len(self.dof_names) != 29
                or len(set(self.dof_names)) != 29
            ):
                raise ValueError("Student posture joint order mismatch: " + field)
        self.carry_arm_indices = [self.dof_names.index(n) for n in c.carry_arm_joint_names]
        self.carry_hip_indices = [self.dof_names.index(n) for n in c.carry_hip_joint_names]
        self.carry_waist_indices = [self.dof_names.index(n) for n in c.carry_waist_joint_names]
        if self.default_dof_pos.shape != (1, 29):
            raise ValueError("Student default_dof_pos must have shape [1,29]")
        self.carry_hip_target = self.default_dof_pos[0, self.carry_hip_indices].clone()
        self.previous_hand_box_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.previous_box_relative_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.student_history_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_hand_contacts = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self.last_box_contacts = torch.zeros_like(self.last_hand_contacts)
        self.grasp_loss_time = torch.zeros(self.num_envs, device=self.device)
        self.student_steps = torch.zeros(self.num_envs, device=self.device)
        self.student_total_sum = torch.zeros_like(self.student_steps)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self._prepare_student_reward_function()

    def _prepare_student_reward_function(self):
        # Copy the config, remove disabled terms, and apply dt exactly once,
        # independently of the inherited Teacher reward list.
        self.student_reward_scales = class_to_dict(self.cfg.student_rewards.scales)
        for name in list(self.student_reward_scales):
            scale = self.student_reward_scales[name]
            if not math.isfinite(scale):
                raise ValueError("Student carry reward coefficients must be finite")
            if scale == 0:
                self.student_reward_scales.pop(name)
            else:
                self.student_reward_scales[name] *= self.dt
        self.student_reward_names = []
        self.student_reward_functions = []
        # These use explicit Student body references / mixed commands and must
        # never shadow the inherited Teacher regularization implementations.
        student_specific = (
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "feet_air_time",
            "feet_slip",
            "feet_stumble",
            "feet_contact_forces",
        )
        for name in self.student_reward_scales:
            prefix = "_reward_student_" if name in student_specific else "_reward_"
            self.student_reward_names.append(name)
            self.student_reward_functions.append(getattr(self, prefix + name))
        self.student_episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self.student_reward_scales
        }

    def _reset_student_carry_buffers(self, env_ids):
        for value in (
            self.previous_hand_box_pos,
            self.previous_box_relative_pos,
            self.student_history_valid,
            self.last_hand_contacts,
            self.last_box_contacts,
            self.grasp_loss_time,
            self.student_steps,
            self.student_total_sum,
            *self.student_episode_sums.values(),
        ):
            value[env_ids] = 0

    def step(self, actions):
        self.executed_phase = self.handoff.phase.clone()
        self.executed_command = self.commands[:, :3].clone()
        self.executed_target = self.handoff.student_target_command.clone()
        return super().step(actions)

    def compute_task_observations(self):
        actor, critic = super().compute_task_observations()
        if hasattr(self, "handoff"):
            student = self.handoff.phase == STUDENT_CONTROL
            actor[student, -3:] = self.handoff.student_applied_command[student]
            critic[student, -3:] = self.handoff.student_applied_command[student]
        # Teacher gets exactly the original approach/hold features. Student's
        # applied command is also the command available to future rewards.
        self.commands[:, :3] = critic[:, -3:]
        return actor, critic

    def check_termination(self):
        # Evaluate physical failure separately so simultaneous timeout + failure
        # is still recorded as failure. No changes to inherited physical rules.
        horizon = self.max_episode_length
        try:
            self.max_episode_length = float("inf")
            super().check_termination()
        finally:
            self.max_episode_length = horizon
        failure = self.reset_buf.clone().bool()
        self.time_out_buf = self.episode_length_buf > horizon
        self.reset_buf = failure | self.time_out_buf
        state = self.handoff
        settings = state.settings
        forces = self.contact_forces[:, self.gate_hand_indices].norm(dim=-1)
        offset = (
            self.rigid_body_states[:, self.gate_palm_indices, :3] - self.box_states[:, None, :3]
        )
        q = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        hand_local = quat_rotate_inverse(q, offset.reshape(-1, 3)).reshape(-1, 2, 3)
        surface_distance = (
            (hand_local.abs() - self._box_size[:, None] / 2).clamp_min(0).norm(dim=-1)
        )
        contact = (forces > settings.contact_force) & (surface_distance < settings.hand_box_margin)
        # Rotated box's lowest corner, above the original support top (or floor).
        x, y, z, w = self.box_states[:, 3:7].unbind(-1)
        vertical_row = torch.stack(
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1
        )
        bottom = self.box_states[:, 2] - (vertical_row.abs() * self._box_size / 2).sum(-1)
        support = torch.maximum(
            self.env_origins[:, 2], self.platform_pos[:, 2] + self._platform_height / 2
        )
        clearance = bottom - support
        student = self.executed_phase == STUDENT_CONTROL
        self.hand_box_surface_distance = surface_distance
        self.box_bottom = bottom
        self._update_student_carry_state()
        inherited_failure = failure.clone()
        failure |= self.student_carry_failure
        self.reset_buf |= self.student_carry_failure
        # Bit flags retain simultaneous causes: 1 drop, 2 separation, 4 grasp,
        # 8 enabled tilt, 16 inherited physical failure, 32 horizon timeout.
        reason = self.student_carry_reason | (16 * (student & inherited_failure).long())
        reason |= 32 * (student & self.time_out_buf).long()
        first = (self.executed_phase == STUDENT_CONTROL) & (state.student_time == 0)
        teacher = self.executed_phase != STUDENT_CONTROL
        self.teacher_last_action[teacher] = self.actions[teacher]
        self.student_first_action[first] = self.actions[first]
        switched = state.complete_step(
            self.box_states[:, 2],
            contact.all(-1),
            clearance,
            self.reset_buf | getattr(self, "_initializing_training_observations", False),
            self.episode_length_buf * self.dt,
        )
        torso = self.rigid_body_states[:, self.torso_index]
        delta = self.student_first_action - self.teacher_last_action
        base_linear = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
        base_angular = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 10:13])
        # Clone BEFORE reset_idx; every tensor has one row per environment.
        measurements = {
            "executed_phase": self.executed_phase,
            "executed_command": self.executed_command,
            "target_command": self.executed_target,
            "post_gate_phase": state.phase,
            "next_target_command": state.student_target_command,
            "next_applied_command": state.student_applied_command,
            "episode_time": self.episode_length_buf * self.dt,
            "handoff_time": state.handoff_time,
            "teacher_warmup_duration": torch.where(
                state.handoff_time >= 0, state.handoff_time, self.episode_length_buf * self.dt
            ),
            "handoff_success": state.handoff_time >= 0,
            "handoff_failure": self.reset_buf & (state.handoff_time < 0),
            "handoff_trigger": switched,
            "first_student_step": first,
            "student_time": state.student_time,
            "student_initial_stand_duration": torch.full_like(
                state.student_time, settings.student_initial_stand_duration_s
            ),
            "box_height": self.box_states[:, 2],
            "box_clearance": clearance,
            "hand_contact": contact,
            "hand_force": forces,
            "hand_surface_distance": surface_distance,
            "base_velocity": torch.cat((base_linear[:, :2], base_angular[:, 2:3]), -1),
            "base_linear_velocity": base_linear,
            "base_angular_velocity": base_angular,
            "box_linear_velocity": self.box_states[:, 7:10],
            "box_angular_velocity": self.box_states[:, 10:13],
            "base_xy_speed": self.root_states[:, 7:9].norm(dim=-1),
            "box_angular_speed": self.box_states[:, 10:13].norm(dim=-1),
            "box_torso_relative_velocity": self.box_states[:, 7:10] - torso[:, 7:10],
            "box_tilt": torch.acos(vertical_row[:, 2].clamp(-1, 1)),
            "robot_reset": self.reset_buf,
            "box_reset": self.reset_buf,
            "failure": failure,
            "timeout": self.time_out_buf,
            "reset_count": self.reset_count,
            "teacher_last_action": self.teacher_last_action,
            "student_first_action": self.student_first_action,
            "action_discontinuity_valid": state.student_time > 0,
            "action_discontinuity": delta,
            "action_discontinuity_l2": delta.norm(dim=-1),
        }
        self.extras["handoff"] = {k: v.detach().clone() for k, v in measurements.items()}
        self.extras["handoff"].update(
            {"student/" + k: v.detach().clone() for k, v in self.student_carry_metrics.items()}
        )
        self.extras["handoff"]["student/termination_reason"] = reason.detach().clone()

    def compute_reward(self):
        # Gate evaluation may already have advanced handoff.phase. Reward
        # ownership and commands always belong to the action actually executed.
        student = self.executed_phase == STUDENT_CONTROL
        teacher = ~student
        diagnostics = self.extras["handoff"]
        teacher_total = torch.zeros_like(self.rew_buf)
        student_total = torch.zeros_like(self.rew_buf)
        task_total = torch.zeros_like(self.rew_buf)
        for name, function in zip(self.reward_names, self.reward_functions):
            raw = torch.where(teacher, function(), 0.0)
            contribution = raw * self.reward_scales[name]  # inherited dt scaling
            teacher_total += contribution
            self.episode_sums[name] += contribution
            diagnostics["teacher_reward_raw/" + name] = raw.detach().clone()
            diagnostics["teacher_reward_weighted/" + name] = contribution.detach().clone()
        for name, function in zip(self.student_reward_names, self.student_reward_functions):
            raw = function()
            if raw.shape != self.rew_buf.shape:
                raise ValueError("Student reward must have shape [num_envs]: " + name)
            raw = torch.where(student, raw, 0.0)
            contribution = raw * self.student_reward_scales[name]
            student_total += contribution
            if name.startswith("carry_") or name == "zero_command_stillness":
                task_total += contribution
            self.student_episode_sums[name] += contribution
            diagnostics["student_reward_raw/" + name] = raw.detach().clone()
            diagnostics["student_reward_weighted/" + name] = contribution.detach().clone()
        diagnostics["student_reward_before_clip"] = student_total.detach().clone()
        if self.cfg.rewards.only_positive_rewards:
            teacher_total.clamp_min_(0.0)
            student_total.clamp_min_(0.0)
        # Preserve inherited termination-after-positive-clipping semantics.
        if "termination" in self.reward_scales:
            raw = torch.where(teacher, self._reward_termination(), 0.0)
            contribution = raw * self.reward_scales["termination"]
            teacher_total += contribution
            self.episode_sums["termination"] += contribution
            diagnostics["teacher_reward_raw/termination"] = raw.detach().clone()
            diagnostics["teacher_reward_weighted/termination"] = contribution.detach().clone()
        self.rew_buf[:] = teacher_total + student_total
        if not torch.isfinite(self.rew_buf).all():
            raise ValueError("Non-finite phase-owned reward")
        self.student_steps += student
        self.student_total_sum += student_total
        diagnostics.update(
            teacher_total_reward=teacher_total.detach().clone(),
            student_task_reward=task_total.detach().clone(),
            student_total_reward=student_total.detach().clone(),
            total_reward=self.rew_buf.detach().clone(),
            episode_student_steps=self.student_steps.detach().clone(),
            episode_student_mean_reward=(self.student_total_sum / self.student_steps.clamp_min(1))
            .detach()
            .clone(),
        )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.extras.pop("episode", None)

    def reset_idx(self, env_ids):
        summary = {}
        if len(env_ids) and hasattr(self, "student_steps"):
            steps = self.student_steps[env_ids]
            reached = steps > 0
            summary = {
                "student/steps": steps.mean(),
                "student/episodes_without_student": (~reached).float().sum(),
                "student/episodes_with_student": reached.float().sum(),
            }
            # Average per-episode, per-Student-step contributions; episodes with
            # no Student steps are counted separately and excluded from means.
            denominator = reached.sum().clamp_min(1)
            for name, sums in {
                "total": self.student_total_sum,
                **self.student_episode_sums,
            }.items():
                summary["student/mean_step_reward/" + name] = (
                    sums[env_ids] / steps.clamp_min(1) * reached
                ).sum() / denominator
        super().reset_idx(env_ids)
        if summary:
            self.extras["episode"].update(summary)
        if len(env_ids) and hasattr(self, "handoff"):
            self._reset_student_carry_buffers(env_ids)
            self.handoff.reset(env_ids)
            self.teacher_last_action[env_ids] = 0
            self.student_first_action[env_ids] = 0
            self.reset_count[env_ids] += 1
            # Ordinary episode resets only; never called on a handoff.
            self.obs_buf[env_ids] = 0
            self.actions[env_ids] = 0
            settings = getattr(self.cfg, "pickup_rsi", None)
            if settings is not None and settings.enabled and settings.log_resets:
                ids = self._reset_ref_env_ids.get("pickUp", env_ids[:0])
                print(json.dumps(dict(event="pickup_rsi_reset", env_ids=ids.cpu().tolist(),
                                      motion_ids=self._reset_ref_motion_ids.get("pickUp", ids).cpu().tolist(),
                                      motion_frames=self._reset_ref_motion_times.get("pickUp", ids).cpu().tolist(),
                                      root=self.root_states[ids].cpu().tolist(),
                                      dof_pos=self.dof_pos[ids].cpu().tolist(),
                                      dof_vel=self.dof_vel[ids].cpu().tolist(),
                                      box=self.box_states[ids].cpu().tolist(),
                                      platform=self.platform_pos[ids].cpu().tolist(),
                                      phase=self.handoff.phase[ids].cpu().tolist())), flush=True)

    def _update_student_carry_state(self):
        """Sample once after physics, before termination/rewards/reset. Never moves physics."""
        student = self.executed_phase == STUDENT_CONTROL
        surface_distance = self.hand_box_surface_distance
        box_bottom = self.box_bottom
        c, b, t = (self.cfg.student_rewards, self.student_body_indices, self.dof_pos.new_tensor)
        pelvis = self.rigid_body_states[:, b["pelvis"]]
        torso = self.rigid_body_states[:, b["torso"]]
        box = self.box_states
        velocity = quat_rotate_inverse(calc_heading_quat(pelvis[:, 3:7]), pelvis[:, 7:10])
        yaw_rate = pelvis[:, 12]
        linear_error = (self.executed_command[:, :2] - velocity[:, :2]).square().sum(-1)
        yaw_error = (self.executed_command[:, 2] - yaw_rate).square()
        moving = (self.executed_command[:, :2].norm(dim=-1) > 0.05) | (
            self.executed_command[:, 2].abs() > 0.05
        )
        offset = self.rigid_body_states[:, b["palms"], :3] - box[:, None, :3]
        q = box[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        hand = quat_rotate_inverse(q, offset.reshape(-1, 3)).reshape(-1, 2, 3)
        relative_pos = quat_rotate_inverse(torso[:, 3:7], box[:, :3] - torso[:, :3])
        valid = student & self.student_history_valid
        slip = ((hand - self.previous_hand_box_pos) / self.dt)[..., [0, 2]].norm(dim=-1)
        slip = torch.where(valid[:, None], slip, 0.0)
        relative_vel = torch.where(
            valid[:, None], (relative_pos - self.previous_box_relative_pos) / self.dt, 0.0
        )
        self.previous_hand_box_pos.copy_(hand)
        self.previous_box_relative_pos.copy_(relative_pos)
        self.student_history_valid.copy_(student)
        raw = self.contact_forces[:, b["hands"]].norm(dim=-1) > c.hand_contact_threshold
        approximate = raw & (surface_distance < self.handoff.settings.hand_box_margin)
        filtered_raw = raw | self.last_hand_contacts
        filtered_box = approximate | self.last_box_contacts
        self.last_hand_contacts.copy_(raw)
        self.last_box_contacts.copy_(approximate)
        self.grasp_loss_time = torch.where(
            student & ~filtered_box.any(-1), self.grasp_loss_time + self.dt, 0.0
        )
        half = self._box_size[:, None, :] / 2
        side_error = (t([1.0, -1.0]) * hand[..., 1] - half[..., 1]).abs()
        side = (side_error - c.carry_hand_side_tolerance).clamp_min(0.0)
        face = (hand[..., [0, 2]].abs() - half[..., [0, 2]] - c.carry_hand_face_margin).clamp_min(
            0.0
        )
        surface = torch.cat((side[..., None], face), -1)
        position_excess = range_violation(
            relative_pos,
            t(c.carry_box_relative_position_lower),
            t(c.carry_box_relative_position_upper),
        )
        velocity_excess = (
            relative_vel.abs() - t(c.carry_box_relative_velocity_tolerance)
        ).clamp_min(0.0)
        arm_excess = range_violation(
            self.dof_pos[:, self.carry_arm_indices],
            t(c.carry_arm_range_lower),
            t(c.carry_arm_range_upper),
        )
        hip_error = self.dof_pos[:, self.carry_hip_indices] - self.carry_hip_target
        feet = self.rigid_body_states[:, b["feet"]]
        knees = self.rigid_body_states[:, b["knees"]]
        heading = calc_heading_quat(pelvis[:, 3:7])
        feet_width = quat_rotate_inverse(heading, feet[:, 0, :3] - feet[:, 1, :3])[:, 1].abs()
        knee_width = quat_rotate_inverse(heading, knees[:, 0, :3] - knees[:, 1, :3])[:, 1].abs()
        foot_error = projected_heading(feet[:, :, 3:7]) - projected_heading(pelvis[:, 3:7])[:, None]
        foot_error = torch.atan2(foot_error.sin(), foot_error.cos())
        waist_error = self.dof_pos[:, self.carry_waist_indices] - t(c.carry_waist_reference_target)
        relative_q = quat_mul(quat_conjugate(pelvis[:, 3:7]), torso[:, 3:7])
        reference = t(c.carry_torso_pelvis_reference_quat)
        reference = (reference / reference.norm()).expand_as(relative_q)
        alignment_error = rotation_vector(quat_mul(quat_conjugate(reference), relative_q))
        box_up = quat_rotate_inverse(box[:, 3:7], t([0.0, 0.0, 1.0]).expand(self.num_envs, -1))
        tilt = torch.acos(box_up[:, 2].clamp(-1, 1))
        drop = box_bottom - self.env_origins[:, 2] < self.cfg.student_termination.box_drop_height
        separation = (box[:, :2] - pelvis[:, :2]).norm(
            dim=-1
        ) > self.cfg.student_termination.robot_box_max_distance
        grasp = self.grasp_loss_time > self.cfg.student_termination.grasp_loss_grace_s
        tilt_exceeded = tilt > math.radians(self.cfg.student_termination.box_tilt_termination_deg)
        self.student_carry_failure = student & (
            drop
            | separation
            | grasp
            | tilt_exceeded & self.cfg.student_termination.terminate_box_tilt
        )
        self.student_carry_reason = student.long() * (
            drop.long()
            + 2 * separation.long()
            + 4 * grasp.long()
            + 8 * (tilt_exceeded & self.cfg.student_termination.terminate_box_tilt).long()
        )
        self.student_carry_metrics = dict(
            measured_velocity_yaw=velocity[:, :2],
            measured_yaw_rate_world=yaw_rate,
            linear_tracking_error=linear_error,
            yaw_tracking_error=yaw_error,
            raw_hand_contact=raw,
            filtered_raw_hand_contact=filtered_raw,
            box_contact_approximate=approximate,
            filtered_box_contact=filtered_box,
            bilateral_contact=filtered_box.all(-1),
            derivative_valid=valid,
            hand_box_position=hand,
            hand_surface_violation=surface,
            hand_side_error=side_error,
            hand_slip=slip,
            box_relative_position=relative_pos,
            box_relative_velocity=relative_vel,
            box_position_violation=position_excess,
            arm_range_violation=arm_excess,
            box_tilt=tilt,
            box_angular_velocity=box[:, 10:13],
            hip_posture_error=hip_error,
            feet_width=feet_width,
            knee_width=knee_width,
            foot_heading_error=foot_error,
            waist_error=waist_error,
            torso_alignment_error=alignment_error,
            grasp_loss_duration=self.grasp_loss_time,
            proposed_box_drop=drop,
            proposed_separation=separation,
            box_tilt_exceeded=tilt_exceeded,
            moving_command=moving,
            termination_reason=self.student_carry_reason,
        )
        self.base_lin_vel_yaw = velocity
        self.base_yaw_rate_world = yaw_rate
        self.student_linear_tracking_error = linear_error
        self.student_yaw_tracking_error = yaw_error
        self.hand_contact_filt = filtered_box
        self.hand_surface_violation = surface
        self.hand_slip = slip
        self.student_derivative_valid = valid
        self.box_velocity_excess = velocity_excess
        self.box_position_excess = position_excess
        self.carry_arm_excess = arm_excess
        self.carry_hip_error = hip_error
        self.carry_foot_heading_error = foot_error
        self.carry_feet_width = feet_width
        self.carry_knee_width = knee_width
        self.carry_waist_error = waist_error
        self.carry_torso_alignment_error = alignment_error
        self.box_up = box_up
        self.student_command_moving = moving
        self.hand_box_pos = hand
        self.box_relative_pos = relative_pos
        self.box_relative_velocity = relative_vel

    def _reward_carry_lin_vel_tracking(self):
        return torch.exp(
            -self.student_linear_tracking_error / self.cfg.student_rewards.carry_lin_vel_sigma
        )

    def _reward_carry_yaw_vel_tracking(self):
        return torch.exp(
            -self.student_yaw_tracking_error / self.cfg.student_rewards.carry_yaw_vel_sigma
        )

    def _reward_carry_bilateral_contact(self):
        return self.hand_contact_filt.all(-1).float()

    def _reward_carry_hand_box_surface(self):
        return constraint_reward(
            self.hand_surface_violation
            / self.cfg.student_rewards.carry_hand_surface_violation_scale
        )

    def _reward_carry_hand_slip(self):
        return (
            constraint_reward(
                (self.hand_slip - self.cfg.student_rewards.carry_hand_slip_tolerance).clamp_min(0.0)
                / self.cfg.student_rewards.carry_hand_slip_violation_scale
            )
            * self.student_derivative_valid
        )

    def _reward_carry_relative_velocity(self):
        return (
            constraint_reward(
                self.box_velocity_excess
                / self.dof_pos.new_tensor(
                    self.cfg.student_rewards.carry_box_velocity_violation_scale
                )
            )
            * self.student_derivative_valid
        )

    def _reward_carry_relative_position(self):
        return constraint_reward(
            self.box_position_excess
            / self.dof_pos.new_tensor(self.cfg.student_rewards.carry_box_position_violation_scale)
        )

    def _reward_carry_arm_range(self):
        return constraint_reward(
            self.carry_arm_excess / self.cfg.student_rewards.carry_arm_violation_scale
        )

    def _reward_carry_box_tilt(self):
        return self.box_up[:, :2].square().sum(-1)

    def _reward_carry_hip_posture(self):
        return deadzone_gaussian_reward(
            self.carry_hip_error,
            self.dof_pos.new_tensor(self.cfg.student_rewards.carry_hip_posture_deadzone),
            self.dof_pos.new_tensor(self.cfg.student_rewards.carry_hip_posture_softness),
        )

    def _reward_carry_foot_heading(self):
        return deadzone_gaussian_reward(
            self.carry_foot_heading_error,
            self.cfg.student_rewards.carry_foot_heading_deadzone,
            self.cfg.student_rewards.carry_foot_heading_softness,
        )

    def _reward_carry_feet_width(self):
        return interval_gaussian_reward(
            self.carry_feet_width,
            *self.cfg.student_rewards.carry_feet_width_range,
            self.cfg.student_rewards.carry_feet_width_softness,
        )

    def _reward_carry_knee_width(self):
        return interval_gaussian_reward(
            self.carry_knee_width,
            *self.cfg.student_rewards.carry_knee_width_range,
            self.cfg.student_rewards.carry_knee_width_softness,
        )

    def _reward_carry_waist_reference(self):
        return deadzone_gaussian_reward(
            self.carry_waist_error,
            self.dof_pos.new_tensor(self.cfg.student_rewards.carry_waist_reference_deadzone),
            self.dof_pos.new_tensor(self.cfg.student_rewards.carry_waist_reference_softness),
        )

    def _reward_carry_torso_pelvis_alignment(self):
        return deadzone_gaussian_reward(
            self.carry_torso_alignment_error,
            self.dof_pos.new_tensor(self.cfg.student_rewards.carry_torso_pelvis_alignment_deadzone),
            self.dof_pos.new_tensor(self.cfg.student_rewards.carry_torso_pelvis_alignment_softness),
        )

    def _reward_zero_command_stillness(self):
        return (
            torch.exp(
                -4 * self.base_lin_vel_yaw[:, :2].square().sum(-1)
                - 2 * self.base_yaw_rate_world.square()
            )
            * ~self.student_command_moving
        )

    def _reward_student_lin_vel_z(self):
        return quat_rotate_inverse(
            self.rigid_body_states[:, self.student_body_indices["pelvis"]][:, 3:7],
            self.rigid_body_states[:, self.student_body_indices["pelvis"]][:, 7:10],
        )[:, 2].square()

    def _reward_student_ang_vel_xy(self):
        return (
            quat_rotate_inverse(
                self.rigid_body_states[:, self.student_body_indices["pelvis"]][:, 3:7],
                self.rigid_body_states[:, self.student_body_indices["pelvis"]][:, 10:13],
            )[:, :2]
            .square()
            .sum(-1)
        )

    def _reward_student_orientation(self):
        return (
            quat_rotate_inverse(
                self.rigid_body_states[:, self.student_body_indices["pelvis"]][:, 3:7],
                self.dof_pos.new_tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1),
            )[:, :2]
            .square()
            .sum(-1)
        )

    def _reward_student_feet_air_time(self):
        return ((self.feet_air_time - 0.5) * self.first_contacts).sum(
            -1
        ) * self.student_command_moving

    def _reward_student_feet_slip(self):
        return (
            self.rigid_body_states[:, self.student_body_indices["feet"]][:, :, 7:9].norm(dim=-1)
            * (self.contact_forces[:, self.student_body_indices["feet"], 2] > 1.0)
        ).sum(-1)

    def _reward_student_feet_stumble(self):
        return (
            (
                self.contact_forces[:, self.student_body_indices["feet"], :2].norm(dim=-1)
                > 3 * self.contact_forces[:, self.student_body_indices["feet"], 2].abs()
            )
            .any(-1)
            .float()
        )

    def _reward_student_feet_contact_forces(self):
        return (
            (
                self.contact_forces[:, self.student_body_indices["feet"]].norm(dim=-1)
                - self.cfg.rewards.max_contact_force
            )
            .clamp_min(0.0)
            .sum(-1)
        )
