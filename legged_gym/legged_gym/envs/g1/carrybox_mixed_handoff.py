"""Continuous handoff with separate Teacher and Student carry objectives."""
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .carrybox import LeggedRobot
from .carrybox_config import G1Cfg
from legged_gym.mixed_handoff import HandoffSettings, HandoffState, STUDENT_CONTROL
from legged_gym.student_carry import StudentCarry, StudentCarryCfg


class MixedHandoffCfg(G1Cfg):
    class env(G1Cfg.env):
        num_envs = 1
        episode_length_s = 60
        test = True

    class asset(G1Cfg.asset):
        class box(G1Cfg.asset.box):
            reset_mode = "default"

    # Match play.py's inference switches; no control gains or geometry changes.
    class domain_rand(G1Cfg.domain_rand):
        disturbance = False
        delay = False
        push_robots = False

    class handoff:
        height = 0.72
        stable_duration_s = 0.3
        student_initial_stand_duration_s = 0.5
        rates = (0.8, 0.6, 1.2)
        mode = "mixed"
        fixed_command = (0., 0., 0.)
        contact_force = 1.0
        hand_box_margin = 0.08
        min_clearance = 0.01

    class student(StudentCarryCfg):
        pass


class MixedHandoff(LeggedRobot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.num_obs, self.num_privileged_obs, self.num_actions) != (738, 126, 29):
            raise ValueError("Mixed handoff requires the PhysHSI 738/126/29 contract")
        settings = HandoffSettings(**{k: getattr(self.cfg.handoff, k)
                                      for k in HandoffSettings.__dataclass_fields__})
        self.handoff = HandoffState(self.num_envs, self.device, self.dt, settings)
        self.teacher_last_action = torch.zeros_like(self.actions)
        self.student_first_action = torch.zeros_like(self.actions)
        self.reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # The tensors include platform, target platform and box. Resolve explicit
        # environment-domain indices rather than assuming an actor-handle offset.
        from isaacgym import gymapi
        def index(name):
            value = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.actor_handles[0], name, gymapi.DOMAIN_ENV)
            if value < 0:
                raise ValueError("Missing handoff rigid body: " + name)
            return value
        self.torso_index = index("torso_link")
        self.gate_hand_indices = [index(side + "_rubber_hand") for side in ("left", "right")]
        self.gate_palm_indices = [index(side + "_palm_link") for side in ("left", "right")]
        self.student_carry = StudentCarry(self, self.cfg.student, dict(
            pelvis=index(self.cfg.control.upper_body_link), torso=self.torso_index,
            palms=self.gate_palm_indices, hands=self.gate_hand_indices,
            feet=[index(name) for name in self.cfg.student.carry_foot_links],
            knees=[index(name) for name in self.cfg.student.carry_knee_links],
        ))

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
        offset = self.rigid_body_states[:, self.gate_palm_indices, :3] - self.box_states[:, None, :3]
        q = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        hand_local = quat_rotate_inverse(q, offset.reshape(-1, 3)).reshape(-1, 2, 3)
        surface_distance = (hand_local.abs() - self._box_size[:, None] / 2).clamp_min(0).norm(dim=-1)
        contact = (forces > settings.contact_force) & (surface_distance < settings.hand_box_margin)
        # Rotated box's lowest corner, above the original support top (or floor).
        x, y, z, w = self.box_states[:, 3:7].unbind(-1)
        vertical_row = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1)
        bottom = self.box_states[:, 2] - (vertical_row.abs() * self._box_size / 2).sum(-1)
        support = torch.maximum(self.env_origins[:, 2], self.platform_pos[:, 2] + self._platform_height / 2)
        clearance = bottom - support
        student = self.executed_phase == STUDENT_CONTROL
        self.student_carry.update(self, student, surface_distance, bottom)
        inherited_failure = failure.clone()
        failure |= self.student_carry.failure
        self.reset_buf |= self.student_carry.failure
        # Bit flags retain simultaneous causes: 1 drop, 2 separation, 4 grasp,
        # 8 enabled tilt, 16 inherited physical failure, 32 horizon timeout.
        reason = self.student_carry.reason | (16 * (student & inherited_failure).long())
        reason |= 32 * (student & self.time_out_buf).long()
        first = (self.executed_phase == STUDENT_CONTROL) & (state.student_time == 0)
        teacher = self.executed_phase != STUDENT_CONTROL
        self.teacher_last_action[teacher] = self.actions[teacher]
        self.student_first_action[first] = self.actions[first]
        switched = state.complete_step(self.box_states[:, 2], contact.all(-1), clearance,
                                       self.reset_buf, self.episode_length_buf * self.dt)
        torso = self.rigid_body_states[:, self.torso_index]
        delta = self.student_first_action - self.teacher_last_action
        base_linear = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
        base_angular = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 10:13])
        # Clone BEFORE reset_idx; every tensor has one row per environment.
        measurements = {
            "executed_phase": self.executed_phase, "executed_command": self.executed_command,
            "target_command": self.executed_target, "post_gate_phase": state.phase,
            "next_target_command": state.student_target_command,
            "next_applied_command": state.student_applied_command,
            "episode_time": self.episode_length_buf * self.dt,
            "handoff_time": state.handoff_time, "teacher_warmup_duration": torch.where(
                state.handoff_time >= 0, state.handoff_time, self.episode_length_buf * self.dt),
            "handoff_success": state.handoff_time >= 0,
            "handoff_failure": self.reset_buf & (state.handoff_time < 0),
            "handoff_trigger": switched, "first_student_step": first,
            "student_time": state.student_time,
            "student_initial_stand_duration": torch.full_like(state.student_time, settings.student_initial_stand_duration_s),
            "box_height": self.box_states[:, 2], "box_clearance": clearance,
            "hand_contact": contact, "hand_force": forces,
            "hand_surface_distance": surface_distance,
            "base_velocity": torch.cat((base_linear[:, :2], base_angular[:, 2:3]), -1),
            "base_xy_speed": self.root_states[:, 7:9].norm(dim=-1),
            "box_angular_speed": self.box_states[:, 10:13].norm(dim=-1),
            "box_torso_relative_velocity": self.box_states[:, 7:10] - torso[:, 7:10],
            "box_tilt": torch.acos(vertical_row[:, 2].clamp(-1, 1)),
            "robot_reset": self.reset_buf, "box_reset": self.reset_buf,
            "failure": failure, "timeout": self.time_out_buf,
            "reset_count": self.reset_count,
            "teacher_last_action": self.teacher_last_action,
            "student_first_action": self.student_first_action,
            "action_discontinuity_valid": state.student_time > 0,
            "action_discontinuity": delta,
            "action_discontinuity_l2": delta.norm(dim=-1),
        }
        self.extras["handoff"] = {k: v.detach().clone() for k, v in measurements.items()}
        self.extras["handoff"].update({"student/" + k: v.detach().clone()
                                       for k, v in self.student_carry.metrics.items()})
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
            raw = torch.where(teacher, function(), 0.)
            contribution = raw * self.reward_scales[name]  # inherited dt scaling
            teacher_total += contribution
            self.episode_sums[name] += contribution
            diagnostics["teacher_reward_raw/" + name] = raw.detach().clone()
            diagnostics["teacher_reward_weighted/" + name] = contribution.detach().clone()
        carry = self.student_carry
        for name, scale in self.cfg.student.scales.items():
            raw = (carry.terms[name] if name in carry.terms
                   else getattr(self, "_reward_" + name)())
            if raw.shape != self.rew_buf.shape:
                raise ValueError("Student reward must have shape [num_envs]: " + name)
            raw = torch.where(student, raw, 0.)
            contribution = raw * scale * self.dt
            student_total += contribution
            if name.startswith("carry_") or name == "zero_command_stillness":
                task_total += contribution
            carry.episode_sums[name] += contribution
            diagnostics["student_reward_raw/" + name] = raw.detach().clone()
            diagnostics["student_reward_weighted/" + name] = contribution.detach().clone()
        diagnostics["student_reward_before_clip"] = student_total.detach().clone()
        if self.cfg.rewards.only_positive_rewards:
            teacher_total.clamp_min_(0.)
            student_total.clamp_min_(0.)
        # Preserve inherited termination-after-positive-clipping semantics.
        if "termination" in self.reward_scales:
            raw = torch.where(teacher, self._reward_termination(), 0.)
            contribution = raw * self.reward_scales["termination"]
            teacher_total += contribution
            self.episode_sums["termination"] += contribution
            diagnostics["teacher_reward_raw/termination"] = raw.detach().clone()
            diagnostics["teacher_reward_weighted/termination"] = contribution.detach().clone()
        self.rew_buf[:] = teacher_total + student_total
        if not torch.isfinite(self.rew_buf).all():
            raise ValueError("Non-finite phase-owned reward")
        carry.steps += student
        carry.total_sum += student_total
        diagnostics.update(
            teacher_total_reward=teacher_total.detach().clone(),
            student_task_reward=task_total.detach().clone(),
            student_total_reward=student_total.detach().clone(),
            total_reward=self.rew_buf.detach().clone(),
            episode_student_steps=carry.steps.detach().clone(),
            episode_student_mean_reward=(carry.total_sum / carry.steps.clamp_min(1)).detach().clone(),
        )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.extras.pop("episode", None)

    def reset_idx(self, env_ids):
        summary = {}
        if len(env_ids) and hasattr(self, "student_carry"):
            carry = self.student_carry
            steps = carry.steps[env_ids]
            reached = steps > 0
            summary = {"student/steps": steps.mean(),
                       "student/episodes_without_student": (~reached).float().sum(),
                       "student/episodes_with_student": reached.float().sum()}
            # Average per-episode, per-Student-step contributions; episodes with
            # no Student steps are counted separately and excluded from means.
            denominator = reached.sum().clamp_min(1)
            for name, sums in {"total": carry.total_sum, **carry.episode_sums}.items():
                summary["student/mean_step_reward/" + name] = (
                    sums[env_ids] / steps.clamp_min(1) * reached).sum() / denominator
        super().reset_idx(env_ids)
        if summary:
            self.extras["episode"].update(summary)
        if len(env_ids) and hasattr(self, "handoff"):
            self.student_carry.reset(env_ids)
            self.handoff.reset(env_ids)
            self.teacher_last_action[env_ids] = 0
            self.student_first_action[env_ids] = 0
            self.reset_count[env_ids] += 1
            # Ordinary episode resets only; never called on a handoff.
            self.obs_buf[env_ids] = 0
            self.actions[env_ids] = 0
