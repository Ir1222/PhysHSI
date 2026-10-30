"""Continuous handoff with separate Teacher and Student carry objectives."""

import math
import json

import torch
import torch.nn.functional as F
from isaacgym.torch_utils import quat_rotate_inverse

from .carrybox import LeggedRobot
from legged_gym.mixed_handoff import HandoffSettings, HandoffState, STUDENT_CONTROL
from legged_gym.pickup_rsi import sample_pickup_time, pickup_box_velocity
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.torch_utils import calc_heading_quat


GAIT_REWARD_NAMES = (
    "gait_foot_slip",
    "gait_swing_clearance",
    "gait_foot_scuff",
    "gait_air_time",
)


class VelTrackPoseAMP(LeggedRobot):
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
        # This is a simulator invariant, independent of checkpoint metadata.
        from pathlib import Path
        from legged_gym import LEGGED_GYM_ROOT_DIR
        mapping = Path(LEGGED_GYM_ROOT_DIR) / "resources/config/joint_id.txt"
        canonical = [line.split()[1] for line in mapping.read_text().splitlines() if line.strip()]
        if list(self.dof_names) != canonical or len(canonical) != 29:
            raise ValueError("Simulator joint ordering differs from the G1 29-DOF mapping")
        settings = HandoffSettings(
            **{k: getattr(self.cfg.handoff, k) for k in HandoffSettings.__dataclass_fields__}
        )
        self.handoff = HandoffState(self.num_envs, self.device, self.dt, settings)
        self.detailed_diagnostics = self.cfg.diagnostics.detailed
        if self.detailed_diagnostics:
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

        if self.detailed_diagnostics:
            self.torso_index = index("torso_link")
        self.gate_hand_indices = [index(side + "_rubber_hand") for side in ("left", "right")]
        self.gate_palm_indices = [index(side + "_palm_link") for side in ("left", "right")]
        self.student_body_indices = dict(pelvis=index(self.cfg.control.upper_body_link))
        self._init_gait_buffers()
        self._init_student_carry_buffers()

    def _init_gait_buffers(self):
        from isaacgym import gymapi
        from legged_gym import LEGGED_GYM_ROOT_DIR
        from legged_gym.utils.gait_geometry import build_gait_geometry

        if self.cfg.terrain.mesh_type != "plane":
            raise ValueError("Gait sole clearance requires the configured plane terrain")
        if not torch.allclose(self.env_origins[:, 2], torch.zeros_like(self.env_origins[:, 2]),
                              atol=1e-6, rtol=0):
            raise ValueError("Gait sole clearance requires the current world-Z=0 plane geometry")
        gait = self.cfg.gait
        for name in (
            "contact_threshold_N",
            "moving_threshold",
            "horizontal_speed_scale_m_s",
            "target_height_m",
            "scuff_height_m",
            "air_time_range_s",
            "geometry_fit_tolerance_m",
            "geometry_validation_tolerance_m",
        ):
            value = getattr(gait, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Gait parameter must be positive and finite: " + name)
        if (not math.isfinite(gait.slip_tolerance_m_s) or gait.slip_tolerance_m_s < 0
                or not math.isfinite(gait.air_time_min_s) or gait.air_time_min_s < 0):
            raise ValueError("Gait tolerances must be nonnegative and finite")
        if (isinstance(gait.geometry_max_points, bool)
                or not isinstance(gait.geometry_max_points, int)
                or not 4 <= gait.geometry_max_points <= 256):
            raise ValueError("gait.geometry_max_points must be an integer in [4, 256]")

        body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        indices = []
        for name in foot_names:
            if name not in body_names:
                raise ValueError("Missing gait rigid body: " + name)
            local_index = body_names.index(name)
            resolved = [
                self.gym.get_actor_rigid_body_index(env, actor, local_index, gymapi.DOMAIN_ENV)
                for env, actor in zip(self.envs, self.actor_handles)
            ]
            if len(set(resolved)) != 1 or not 0 <= resolved[0] < self.contact_forces.shape[1]:
                raise ValueError("Invalid environment-domain gait body: " + name)
            indices.append(resolved[0])
        if indices[0] == indices[1]:
            raise ValueError("Left and right gait bodies must be distinct")
        self.gait_feet_indices = torch.tensor(indices, device=self.device, dtype=torch.long)

        urdf = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.gait_geometry = build_gait_geometry(
            urdf,
            foot_names,
            gait.geometry_max_points,
            gait.geometry_fit_tolerance_m,
            gait.geometry_validation_tolerance_m,
        )
        bodies = self.gait_geometry["bodies"]
        count = max(bodies[name]["point_count"] for name in foot_names)
        points = [
            bodies[name]["points"] + [bodies[name]["points"][0]] * (count - bodies[name]["point_count"])
            for name in foot_names
        ]
        self.gait_sole_points = self.dof_pos.new_tensor(points)
        self.gait_geometry["body_indices_domain_env"] = dict(zip(foot_names, indices))

        contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.gait_contact_raw = contact.clone()
        self.gait_previous_raw = contact.clone()
        self.gait_contact_filtered = contact.clone()
        self.gait_touchdown = contact.clone()
        self.gait_swing = contact.clone()
        self.gait_stance_seen = contact.clone()
        self.gait_air_time = self.dof_pos.new_zeros(self.num_envs, 2)
        self.gait_preceding_air_time = self.gait_air_time.clone()
        self.gait_horizontal_velocity = self.dof_pos.new_zeros(self.num_envs, 2, 2)
        self.gait_horizontal_speed = self.gait_air_time.clone()
        self.gait_clearance = self.gait_air_time.clone()
        self.gait_previous_owner = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_history_valid = self.gait_previous_owner.clone()

    def _reset_gait_history(self, env_ids):
        for name in (
            "contact_raw",
            "previous_raw",
            "contact_filtered",
            "touchdown",
            "swing",
            "stance_seen",
            "air_time",
            "preceding_air_time",
            "horizontal_velocity",
            "horizontal_speed",
            "clearance",
            "previous_owner",
            "history_valid",
        ):
            getattr(self, "gait_" + name)[env_ids] = 0

    def _gait_moving(self):
        threshold = self.cfg.gait.moving_threshold
        student = self.executed_phase == STUDENT_CONTROL
        command = self.executed_command
        return student & ((command[:, :2].norm(dim=-1) > threshold)
                          | (command[:, 2].abs() > threshold))

    def _update_gait_state(self):
        student = self.executed_phase == STUDENT_CONTROL
        changed = ~self.gait_history_valid | (self.gait_previous_owner != student)
        self._reset_gait_history(changed)
        history = ~changed
        body = self.rigid_body_states[:, self.gait_feet_indices]
        raw = self.contact_forces[:, self.gait_feet_indices].norm(dim=-1) > self.cfg.gait.contact_threshold_N
        filtered = raw | self.gait_previous_raw
        moving = self._gait_moving()
        self.gait_touchdown.copy_(filtered & ~self.gait_contact_filtered & history[:, None]
                                  & self.gait_stance_seen & (self.gait_air_time > 0))
        self.gait_preceding_air_time.copy_(self.gait_air_time)
        self.gait_contact_raw.copy_(raw)
        self.gait_contact_filtered.copy_(filtered)
        self.gait_previous_raw.copy_(raw)
        self.gait_swing.copy_(~filtered)
        self.gait_stance_seen &= moving[:, None]
        self.gait_stance_seen |= filtered & moving[:, None]
        self.gait_air_time.copy_(torch.where(
            ~filtered & moving[:, None], self.gait_air_time + self.dt, 0.0
        ))
        self.gait_horizontal_velocity.copy_(body[:, :, 7:9])
        self.gait_horizontal_speed.copy_(self.gait_horizontal_velocity.norm(dim=-1))

        points = self.gait_sole_points[None].expand(self.num_envs, -1, -1, -1)
        quaternion = F.normalize(body[:, :, None, 3:7], dim=-1)
        xyz = quaternion[..., :3].expand_as(points)
        tangent = 2 * torch.cross(xyz, points, dim=-1)
        rotated = points + quaternion[..., 3:4] * tangent + torch.cross(xyz, tangent, dim=-1)
        # The configured plane is world Z=0; environment origins have zero Z.
        self.gait_clearance.copy_(body[:, :, 2] + rotated[..., 2].amin(dim=-1))
        self.gait_previous_owner.copy_(student)
        self.gait_history_valid[:] = True

    def _gait_physical_diagnostics(self):
        moving = self._gait_moving()[:, None]
        nan = torch.full_like(self.gait_horizontal_speed, float("nan"))
        stance = moving & self.gait_contact_filtered
        swing = moving & self.gait_swing
        swing_motion = swing & (self.gait_horizontal_speed > self.cfg.gait.slip_tolerance_m_s)
        return {
            "student/stance_slip_m_s": torch.where(stance, self.gait_horizontal_speed, nan),
            "student/swing_clearance_m": torch.where(swing, self.gait_clearance, nan),
            "student/low_clearance_swing_fraction": torch.where(
                swing_motion,
                (self.gait_clearance < self.cfg.gait.scuff_height_m).float(),
                nan,
            ),
        }

    def _init_student_carry_buffers(self):
        self.detailed_diagnostics = self.cfg.diagnostics.detailed
        for field in ("carry_lin_vel_sigma", "carry_yaw_vel_sigma"):
            value = getattr(self.cfg.student_rewards, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Student carry parameter must be positive and finite: " + field)
        for field in ("robot_box_max_distance", "grasp_loss_grace_s"):
            value = getattr(self.cfg.student_termination, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Student carry parameter must be positive and finite: " + field)
        self.last_box_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
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
        student_specific = ("lin_vel_z", "ang_vel_xy", "orientation")
        for name in self.student_reward_scales:
            prefix = "_reward_student_" if name in student_specific else "_reward_"
            self.student_reward_names.append(name)
            self.student_reward_functions.append(getattr(self, prefix + name))
        if len(self.student_reward_names) != len(set(self.student_reward_names)):
            raise ValueError("Duplicate Student reward name")
        overlap = set(GAIT_REWARD_NAMES) & set(self.reward_names)
        if overlap:
            raise ValueError("Gait reward is already registered by the inherited task: "
                             + ", ".join(sorted(overlap)))
        self.student_episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self.student_reward_scales
        }

    def _reset_student_carry_buffers(self, env_ids):
        for value in (
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
        if self.detailed_diagnostics:
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
        self._update_student_carry_state(contact)
        inherited_failure = failure.clone()
        failure |= self.student_carry_failure
        self.reset_buf |= self.student_carry_failure
        # Bit flags retain simultaneous causes: 1 drop, 2 separation, 4 grasp,
        # 8 enabled tilt, 16 inherited physical failure, 32 horizon timeout.
        reason = self.student_carry_reason | (16 * (student & inherited_failure).long())
        reason |= 32 * (student & self.time_out_buf).long()
        first = (self.executed_phase == STUDENT_CONTROL) & (state.student_time == 0)
        teacher = self.executed_phase != STUDENT_CONTROL
        if self.detailed_diagnostics:
            self.teacher_last_action[teacher] = self.actions[teacher]
            self.student_first_action[first] = self.actions[first]
        switched = state.complete_step(
            self.box_states[:, 2],
            contact.all(-1),
            clearance,
            self.reset_buf | getattr(self, "_initializing_training_observations", False),
            self.episode_length_buf * self.dt,
        )
        # Snapshot required fields before reset mutates phase/episode buffers.
        measurements = dict(
            executed_phase=self.executed_phase,
            executed_command=self.executed_command,
            handoff_time=state.handoff_time,
            handoff_failure=self.reset_buf & (state.handoff_time < 0),
            handoff_trigger=switched,
            first_student_step=first,
            failure=failure,
            timeout=self.time_out_buf,
            robot_reset=self.reset_buf,
            box_height=self.box_states[:, 2],
            box_clearance=clearance,
            hand_contact=contact,
            hand_force=forces,
            hand_surface_distance=surface_distance,
        )
        if self.detailed_diagnostics:
            torso = self.rigid_body_states[:, self.torso_index]
            delta = self.student_first_action - self.teacher_last_action
            base_linear = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
            base_angular = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 10:13])
            measurements.update(
                target_command=self.executed_target,
                post_gate_phase=state.phase,
                next_target_command=state.student_target_command,
                next_applied_command=state.student_applied_command,
                episode_time=self.episode_length_buf * self.dt,
                teacher_warmup_duration=torch.where(state.handoff_time >= 0, state.handoff_time,
                                                    self.episode_length_buf * self.dt),
                handoff_success=state.handoff_time >= 0,
                student_time=state.student_time,
                student_initial_stand_duration=torch.full_like(state.student_time, settings.student_initial_stand_duration_s),
                base_velocity=torch.cat((base_linear[:, :2], base_angular[:, 2:3]), -1),
                base_linear_velocity=base_linear,
                base_angular_velocity=base_angular,
                box_linear_velocity=self.box_states[:, 7:10],
                box_angular_velocity=self.box_states[:, 10:13],
                base_xy_speed=self.root_states[:, 7:9].norm(dim=-1),
                box_angular_speed=self.box_states[:, 10:13].norm(dim=-1),
                box_torso_relative_velocity=self.box_states[:, 7:10] - torso[:, 7:10],
                box_tilt=torch.acos(vertical_row[:, 2].clamp(-1, 1)),
                box_reset=self.reset_buf,
                reset_count=self.reset_count,
                teacher_last_action=self.teacher_last_action,
                student_first_action=self.student_first_action,
                action_discontinuity_valid=state.student_time > 0,
                action_discontinuity=delta,
                action_discontinuity_l2=delta.norm(dim=-1),
            )
        self.extras["handoff"] = {k: v.detach().clone() for k, v in measurements.items()}
        self.extras["handoff"].update(
            {"student/" + k: v.detach().clone() for k, v in self.student_carry_metrics.items()}
        )
        self.extras["handoff"]["student/termination_reason"] = reason.detach().clone()

    def compute_reward(self):
        # Gate evaluation may already have advanced handoff.phase. Reward
        # ownership and commands always belong to the action actually executed.
        self._update_gait_state()
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
            if self.detailed_diagnostics:
                diagnostics["teacher_reward_raw/" + name] = raw.detach().clone()
                diagnostics["teacher_reward_weighted/" + name] = contribution.detach().clone()
        for name, function in zip(self.student_reward_names, self.student_reward_functions):
            raw = function()
            if raw.shape != self.rew_buf.shape:
                raise ValueError("Student reward must have shape [num_envs]: " + name)
            raw = torch.where(student, raw, 0.0)
            contribution = raw * self.student_reward_scales[name]
            student_total += contribution
            if name.startswith("carry_"):
                task_total += contribution
            self.student_episode_sums[name] += contribution
            if self.detailed_diagnostics:
                diagnostics["student_reward_raw/" + name] = raw.detach().clone()
                diagnostics["student_reward_weighted/" + name] = contribution.detach().clone()
            if name in GAIT_REWARD_NAMES:
                diagnostics["student/" + name] = raw.detach().clone()
        diagnostics.update({name: value.detach() for name, value in
                            self._gait_physical_diagnostics().items()})
        if self.detailed_diagnostics:
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
            if self.detailed_diagnostics:
                diagnostics["teacher_reward_raw/termination"] = raw.detach().clone()
                diagnostics["teacher_reward_weighted/termination"] = contribution.detach().clone()
        self.rew_buf[:] = teacher_total + student_total
        if not torch.isfinite(self.rew_buf).all():
            raise ValueError("Non-finite phase-owned reward")
        self.student_steps += student
        self.student_total_sum += student_total
        diagnostics.update(student_task_reward=task_total.detach(),
                           student_total_reward=student_total.detach())
        if self.detailed_diagnostics:
            diagnostics.update(
                teacher_total_reward=teacher_total.detach(), total_reward=self.rew_buf.detach().clone(),
                episode_student_steps=self.student_steps.detach().clone(),
                episode_student_mean_reward=(self.student_total_sum / self.student_steps.clamp_min(1)).detach(),
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
            self._reset_gait_history(env_ids)
            self.handoff.reset(env_ids)
            if self.detailed_diagnostics:
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

    def _update_student_carry_state(self, hand_contact=None):
        """Shared tracking/contact/termination state, sampled before reset."""
        student = self.executed_phase == STUDENT_CONTROL
        pelvis = self.rigid_body_states[:, self.student_body_indices["pelvis"]]
        velocity = quat_rotate_inverse(calc_heading_quat(pelvis[:, 3:7]), pelvis[:, 7:10])
        self.student_linear_tracking_error = (self.executed_command[:, :2] - velocity[:, :2]).square().sum(-1)
        self.student_yaw_tracking_error = (self.executed_command[:, 2] - pelvis[:, 12]).square()
        # Same box-proximity predicate as the physical gate; keep the existing
        # one-frame OR filter and independent complete-grasp-loss grace period.
        if hand_contact is None or self.detailed_diagnostics:
            raw = self.contact_forces[:, self.gate_hand_indices].norm(dim=-1) > self.handoff.settings.contact_force
        if hand_contact is None:
            hand_contact = raw & (self.hand_box_surface_distance < self.handoff.settings.hand_box_margin)
        self.hand_contact_filt = hand_contact | self.last_box_contacts
        self.last_box_contacts.copy_(hand_contact)
        self.grasp_loss_time = torch.where(
            student & ~self.hand_contact_filt.any(-1), self.grasp_loss_time + self.dt, 0.0)
        termination = self.cfg.student_termination
        drop = self.box_bottom - self.env_origins[:, 2] < termination.box_drop_height
        separation = (self.box_states[:, :2] - pelvis[:, :2]).norm(dim=-1) > termination.robot_box_max_distance
        grasp = self.grasp_loss_time > termination.grasp_loss_grace_s
        tilt_failure = torch.zeros_like(student)
        if termination.terminate_box_tilt or self.detailed_diagnostics:
            q = self.box_states[:, 3:7]
            tilt = torch.acos((1 - 2 * (q[:, 0].square() + q[:, 1].square())).clamp(-1, 1))
            tilt_exceeded = tilt > math.radians(termination.box_tilt_termination_deg)
            tilt_failure = tilt_exceeded & termination.terminate_box_tilt
        self.student_carry_failure = student & (drop | separation | grasp | tilt_failure)
        self.student_carry_reason = student.long() * (
            drop.long() + 2 * separation.long() + 4 * grasp.long() + 8 * tilt_failure.long())
        self.student_carry_metrics = dict(
            measured_velocity_yaw=velocity[:, :2], measured_yaw_rate_world=pelvis[:, 12],
            linear_tracking_error=self.student_linear_tracking_error,
            yaw_tracking_error=self.student_yaw_tracking_error,
            bilateral_contact=self.hand_contact_filt.all(-1),
            grasp_loss_duration=self.grasp_loss_time,
            proposed_box_drop=drop, proposed_separation=separation,
            termination_reason=self.student_carry_reason,
        )
        if self.detailed_diagnostics:
            self.student_carry_metrics.update(raw_hand_contact=raw,
                box_contact_approximate=hand_contact, filtered_box_contact=self.hand_contact_filt,
                box_tilt=tilt, box_tilt_exceeded=tilt_exceeded)

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

    def _reward_gait_foot_slip(self):
        gait = self.cfg.gait
        slip = ((self.gait_horizontal_speed - gait.slip_tolerance_m_s)
                / gait.horizontal_speed_scale_m_s).clamp(0, 1)
        return (slip * self.gait_contact_filtered).mean(-1) * self._gait_moving()

    def _reward_gait_swing_clearance(self):
        gait = self.cfg.gait
        motion = ((self.gait_horizontal_speed - gait.slip_tolerance_m_s)
                  / gait.horizontal_speed_scale_m_s).clamp(0, 1)
        clearance = (self.gait_clearance / gait.target_height_m).clamp(0, 1)
        return (self.gait_swing * motion * clearance).mean(-1) * self._gait_moving()

    def _reward_gait_foot_scuff(self):
        gait = self.cfg.gait
        motion = (self.gait_horizontal_speed / gait.horizontal_speed_scale_m_s).clamp(0, 1)
        low = torch.exp(-self.gait_clearance.clamp_min(0) / gait.scuff_height_m)
        return (self.gait_swing * motion * low).mean(-1) * self._gait_moving()

    def _reward_gait_air_time(self):
        gait = self.cfg.gait
        score = ((self.gait_preceding_air_time - gait.air_time_min_s)
                 / gait.air_time_range_s).clamp(0, 1)
        return (score * self.gait_touchdown).sum(-1).clamp_max(1) * self._gait_moving()

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
                self.gravity_vec,
            )[:, :2]
            .square()
            .sum(-1)
        )
