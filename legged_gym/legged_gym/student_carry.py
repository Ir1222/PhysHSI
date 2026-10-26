"""Small Carry12 reward subset for the existing PhysHSI handoff task."""
import math

import torch

from legged_gym.utils.torch_utils import (
    calc_heading_quat, quat_conjugate, quat_mul, quat_rotate_inverse,
)


class StudentCarryCfg:
    # Coefficients before control-dt scaling. Teacher configuration is separate.
    scales = dict(
        carry_lin_vel_tracking=3., carry_yaw_vel_tracking=2.5,
        carry_bilateral_contact=.5, carry_hand_box_surface=1.5,
        carry_hand_slip=.5, carry_relative_velocity=.5,
        carry_relative_position=.75, carry_arm_range=.2, carry_box_tilt=-.5,
        carry_hip_posture=.8, carry_foot_heading=.35, carry_feet_width=.30,
        carry_knee_width=.20, carry_waist_reference=.4,
        carry_torso_pelvis_alignment=.2, zero_command_stillness=.2,
        lin_vel_z=-1., ang_vel_xy=-.05, orientation=-1.,
        feet_air_time=.2, feet_slip=-.2, feet_stumble=-1.,
        collision=-.5, feet_contact_forces=-.001,
        dof_acc=-1e-7, action_rate=-.03, torques=-1e-4, dof_vel=-2e-4,
        dof_pos_limits=-5., dof_vel_limits=-1e-3, torque_limits=-.03,
    )
    carry_lin_vel_sigma = .25  # (m/s)^2, denominator of squared error
    carry_yaw_vel_sigma = .25  # (rad/s)^2
    hand_contact_threshold = 1.  # N; box proximity also required
    carry_hand_side_tolerance = .03
    carry_hand_face_margin = .01
    carry_hand_surface_violation_scale = .04
    carry_hand_slip_tolerance = .35
    carry_hand_slip_violation_scale = .35
    carry_arm_joint_names = tuple(side + "_" + joint + "_joint"
        for side in ("left", "right") for joint in
        ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
         "wrist_roll", "wrist_pitch", "wrist_yaw"))
    carry_arm_range_lower = [-.90, -.25, -.85, -.25, -1.10, -.60, -.60,
                             -1.10, -1., -.55, -.25, -.55, -.60, -.60]
    carry_arm_range_upper = [.55, .80, .70, 1.15, .55, .60, .60,
                             .25, .25, 1., 1.40, 1.10, .60, .60]
    carry_arm_violation_scale = .35
    carry_box_relative_position_lower = [.18, -.20, -.15]
    carry_box_relative_position_upper = [.53, .20, .25]
    carry_box_position_violation_scale = [.10, .10, .10]
    carry_box_relative_velocity_tolerance = [.35, .45, .45]
    carry_box_velocity_violation_scale = [.35, .45, .45]
    carry_hip_joint_names = ("left_hip_roll_joint", "left_hip_yaw_joint",
                             "right_hip_roll_joint", "right_hip_yaw_joint")
    carry_hip_posture_deadzone = [.08, .10, .08, .10]
    carry_hip_posture_softness = [.15, .15, .15, .15]
    carry_foot_heading_deadzone = .12
    carry_foot_links = ("left_ankle_pitch_link", "right_ankle_pitch_link")
    carry_knee_links = ("left_knee_link", "right_knee_link")
    carry_foot_heading_softness = .20
    carry_feet_width_range = [.09, .27]
    carry_feet_width_softness = .08
    carry_knee_width_range = [.14, .25]
    carry_knee_width_softness = .06
    carry_waist_joint_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    carry_waist_reference_target = [-.041372441, .034433149, .249833077]
    carry_waist_reference_deadzone = [.30, .07, .12]
    carry_waist_reference_softness = [.20, .10, .12]
    carry_torso_pelvis_reference_quat = [.020052686, .124780931, -.023215466, .991709963]
    carry_torso_pelvis_alignment_deadzone = [.12, .12, .30]
    carry_torso_pelvis_alignment_softness = [.12, .12, .20]
    box_drop_height = .15  # box bottom above ground, m (not COM or support clearance)
    robot_box_max_distance = 1.  # XY distance, m
    grasp_loss_grace_s = .30
    box_tilt_termination_deg = 70.
    terminate_box_tilt = False  # Observed Teacher handoff tilt is about 100 degrees.


def range_violation(value, lower, upper):
    return (lower - value).clamp_min(0.) + (value - upper).clamp_min(0.)


def constraint_reward(excess):
    return 1. / (1. + excess.flatten(1).square().sum(-1))


def deadzone_gaussian_reward(error, deadzone, softness):
    excess = (error.abs() - deadzone).clamp_min(0.)
    return torch.exp(-(excess / softness).square().mean(-1))


def interval_gaussian_reward(value, lower, upper, softness):
    return torch.exp(-(range_violation(value, lower, upper) / softness).square())


def projected_heading(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rotation_vector(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[..., 3:4] < 0., -q, q)
    vector = q[..., :3]
    norm = vector.norm(dim=-1, keepdim=True)
    half = torch.atan2(norm, q[..., 3:4].clamp(0., 1.))
    return vector * torch.where(norm > 1e-7, 2*half / norm.clamp_min(1e-12), 2.)


class StudentCarry:
    def __init__(self, env, cfg, bodies):
        self.cfg, self.bodies = cfg, bodies
        self.tensor = env.dof_pos.new_tensor
        for field in ("carry_lin_vel_sigma", "carry_yaw_vel_sigma",
                      "carry_hand_surface_violation_scale", "carry_hand_slip_violation_scale",
                      "carry_arm_violation_scale", "carry_foot_heading_softness",
                      "carry_feet_width_softness", "carry_knee_width_softness",
                      "robot_box_max_distance", "grasp_loss_grace_s", "hand_contact_threshold"):
            if not math.isfinite(getattr(cfg, field)) or getattr(cfg, field) <= 0:
                raise ValueError("Student carry parameter must be positive and finite: " + field)
        for field in ("carry_box_position_violation_scale", "carry_box_velocity_violation_scale",
                      "carry_hip_posture_softness", "carry_waist_reference_softness",
                      "carry_torso_pelvis_alignment_softness"):
            value = self.tensor(getattr(cfg, field))
            if not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError("Student carry softness must be positive and finite: " + field)
        if not all(math.isfinite(value) for value in cfg.scales.values()):
            raise ValueError("Student carry reward coefficients must be finite")
        for field, expected in (
            ("carry_hip_joint_names", StudentCarryCfg.carry_hip_joint_names),
            ("carry_waist_joint_names", StudentCarryCfg.carry_waist_joint_names),
            ("carry_arm_joint_names", StudentCarryCfg.carry_arm_joint_names),
        ):
            names = tuple(getattr(cfg, field))
            if names != expected or len(env.dof_names) != 29 or len(set(env.dof_names)) != 29:
                raise ValueError("Student posture joint order mismatch: " + field)
        self.arm_ids = [env.dof_names.index(n) for n in cfg.carry_arm_joint_names]
        self.hip_ids = [env.dof_names.index(n) for n in cfg.carry_hip_joint_names]
        self.waist_ids = [env.dof_names.index(n) for n in cfg.carry_waist_joint_names]
        if env.default_dof_pos.shape != (1, 29):
            raise ValueError("Student default_dof_pos must have shape [1,29]")
        self.hip_target = env.default_dof_pos[0, self.hip_ids].clone()
        self.previous_hand = env.dof_pos.new_zeros(env.num_envs, 2, 3)
        self.previous_box = env.dof_pos.new_zeros(env.num_envs, 3)
        self.history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_raw_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.last_box_contact = torch.zeros_like(self.last_raw_contact)
        self.grasp_loss_time = env.dof_pos.new_zeros(env.num_envs)
        self.steps = env.dof_pos.new_zeros(env.num_envs)
        self.total_sum = torch.zeros_like(self.steps)
        self.episode_sums = {name: torch.zeros_like(self.steps) for name in cfg.scales}

    def reset(self, ids):
        for value in (self.previous_hand, self.previous_box, self.history_valid,
                      self.last_raw_contact, self.last_box_contact,
                      self.grasp_loss_time, self.steps, self.total_sum,
                      *self.episode_sums.values()):
            value[ids] = 0

    def update(self, env, student, surface_distance, box_bottom):
        """Sample once after physics, before termination/rewards/reset. Never moves physics."""
        c, b, t = self.cfg, self.bodies, self.tensor
        pelvis = env.rigid_body_states[:, b["pelvis"]]
        torso = env.rigid_body_states[:, b["torso"]]
        box = env.box_states
        velocity = quat_rotate_inverse(calc_heading_quat(pelvis[:, 3:7]), pelvis[:, 7:10])
        yaw_rate = pelvis[:, 12]
        linear_error = (env.executed_command[:, :2] - velocity[:, :2]).square().sum(-1)
        yaw_error = (env.executed_command[:, 2] - yaw_rate).square()
        moving = ((env.executed_command[:, :2].norm(dim=-1) > .05)
                  | (env.executed_command[:, 2].abs() > .05))
        offset = env.rigid_body_states[:, b["palms"], :3] - box[:, None, :3]
        q = box[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        hand = quat_rotate_inverse(q, offset.reshape(-1, 3)).reshape(-1, 2, 3)
        relative_pos = quat_rotate_inverse(torso[:, 3:7], box[:, :3] - torso[:, :3])
        valid = student & self.history_valid
        slip = ((hand - self.previous_hand) / env.dt)[..., [0, 2]].norm(dim=-1)
        slip = torch.where(valid[:, None], slip, 0.)
        relative_vel = torch.where(valid[:, None], (relative_pos - self.previous_box) / env.dt, 0.)
        self.previous_hand.copy_(hand)
        self.previous_box.copy_(relative_pos)
        self.history_valid.copy_(student)
        raw = env.contact_forces[:, b["hands"]].norm(dim=-1) > c.hand_contact_threshold
        approximate = raw & (surface_distance < env.handoff.settings.hand_box_margin)
        filtered_raw = raw | self.last_raw_contact
        filtered_box = approximate | self.last_box_contact
        self.last_raw_contact.copy_(raw)
        self.last_box_contact.copy_(approximate)
        self.grasp_loss_time = torch.where(student & ~filtered_box.any(-1),
                                           self.grasp_loss_time + env.dt, 0.)
        half = env._box_size[:, None, :] / 2
        side_error = (t([1., -1.]) * hand[..., 1] - half[..., 1]).abs()
        side = (side_error - c.carry_hand_side_tolerance).clamp_min(0.)
        face = (hand[..., [0, 2]].abs() - half[..., [0, 2]] - c.carry_hand_face_margin).clamp_min(0.)
        surface = torch.cat((side[..., None], face), -1)
        position_excess = range_violation(relative_pos, t(c.carry_box_relative_position_lower),
                                          t(c.carry_box_relative_position_upper))
        velocity_excess = (relative_vel.abs() - t(c.carry_box_relative_velocity_tolerance)).clamp_min(0.)
        arm_excess = range_violation(env.dof_pos[:, self.arm_ids], t(c.carry_arm_range_lower),
                                     t(c.carry_arm_range_upper))
        hip_error = env.dof_pos[:, self.hip_ids] - self.hip_target
        feet = env.rigid_body_states[:, b["feet"]]
        knees = env.rigid_body_states[:, b["knees"]]
        heading = calc_heading_quat(pelvis[:, 3:7])
        feet_width = quat_rotate_inverse(heading, feet[:, 0, :3] - feet[:, 1, :3])[:, 1].abs()
        knee_width = quat_rotate_inverse(heading, knees[:, 0, :3] - knees[:, 1, :3])[:, 1].abs()
        foot_error = projected_heading(feet[:, :, 3:7]) - projected_heading(pelvis[:, 3:7])[:, None]
        foot_error = torch.atan2(foot_error.sin(), foot_error.cos())
        waist_error = env.dof_pos[:, self.waist_ids] - t(c.carry_waist_reference_target)
        relative_q = quat_mul(quat_conjugate(pelvis[:, 3:7]), torso[:, 3:7])
        reference = t(c.carry_torso_pelvis_reference_quat)
        reference = (reference / reference.norm()).expand_as(relative_q)
        alignment_error = rotation_vector(quat_mul(quat_conjugate(reference), relative_q))
        box_up = quat_rotate_inverse(box[:, 3:7], t([0., 0., 1.]).expand(env.num_envs, -1))
        tilt = torch.acos(box_up[:, 2].clamp(-1, 1))
        raw_terms = {
            "carry_lin_vel_tracking": torch.exp(-linear_error / c.carry_lin_vel_sigma),
            "carry_yaw_vel_tracking": torch.exp(-yaw_error / c.carry_yaw_vel_sigma),
            "carry_bilateral_contact": filtered_box.all(-1).float(),
            "carry_hand_box_surface": constraint_reward(surface / c.carry_hand_surface_violation_scale),
            "carry_hand_slip": constraint_reward((slip-c.carry_hand_slip_tolerance).clamp_min(0.)
                                                  / c.carry_hand_slip_violation_scale) * valid,
            "carry_relative_velocity": constraint_reward(velocity_excess / t(c.carry_box_velocity_violation_scale)) * valid,
            "carry_relative_position": constraint_reward(position_excess / t(c.carry_box_position_violation_scale)),
            "carry_arm_range": constraint_reward(arm_excess / c.carry_arm_violation_scale),
            "carry_box_tilt": box_up[:, :2].square().sum(-1),
            "carry_hip_posture": deadzone_gaussian_reward(hip_error, t(c.carry_hip_posture_deadzone), t(c.carry_hip_posture_softness)),
            "carry_foot_heading": deadzone_gaussian_reward(foot_error, c.carry_foot_heading_deadzone, c.carry_foot_heading_softness),
            "carry_feet_width": interval_gaussian_reward(feet_width, *c.carry_feet_width_range, c.carry_feet_width_softness),
            "carry_knee_width": interval_gaussian_reward(knee_width, *c.carry_knee_width_range, c.carry_knee_width_softness),
            "carry_waist_reference": deadzone_gaussian_reward(waist_error, t(c.carry_waist_reference_deadzone), t(c.carry_waist_reference_softness)),
            "carry_torso_pelvis_alignment": deadzone_gaussian_reward(alignment_error, t(c.carry_torso_pelvis_alignment_deadzone), t(c.carry_torso_pelvis_alignment_softness)),
            "zero_command_stillness": torch.exp(-4*velocity[:, :2].square().sum(-1)-2*yaw_rate.square()) * ~moving,
            "lin_vel_z": quat_rotate_inverse(pelvis[:, 3:7], pelvis[:, 7:10])[:, 2].square(),
            "ang_vel_xy": quat_rotate_inverse(pelvis[:, 3:7], pelvis[:, 10:13])[:, :2].square().sum(-1),
            "orientation": quat_rotate_inverse(pelvis[:, 3:7], t([0., 0., -1.]).expand(env.num_envs, -1))[:, :2].square().sum(-1),
            "feet_air_time": ((env.feet_air_time-.5)*env.first_contacts).sum(-1) * moving,
            "feet_slip": (feet[:, :, 7:9].norm(dim=-1) * (env.contact_forces[:, b["feet"], 2] > 1.)).sum(-1),
            "feet_stumble": (env.contact_forces[:, b["feet"], :2].norm(dim=-1) > 3*env.contact_forces[:, b["feet"], 2].abs()).any(-1).float(),
            "feet_contact_forces": (env.contact_forces[:, b["feet"]].norm(dim=-1)-env.cfg.rewards.max_contact_force).clamp_min(0.).sum(-1),
        }
        drop = box_bottom - env.env_origins[:, 2] < c.box_drop_height
        separation = (box[:, :2] - pelvis[:, :2]).norm(dim=-1) > c.robot_box_max_distance
        grasp = self.grasp_loss_time > c.grasp_loss_grace_s
        tilt_exceeded = tilt > math.radians(c.box_tilt_termination_deg)
        self.failure = student & (drop | separation | grasp | (tilt_exceeded & c.terminate_box_tilt))
        self.reason = student.long() * (drop.long() + 2*separation.long() + 4*grasp.long()
                                       + 8*(tilt_exceeded & c.terminate_box_tilt).long())
        self.terms = raw_terms
        self.metrics = dict(
            measured_velocity_yaw=velocity[:, :2], measured_yaw_rate_world=yaw_rate,
            linear_tracking_error=linear_error, yaw_tracking_error=yaw_error,
            raw_hand_contact=raw, filtered_raw_hand_contact=filtered_raw,
            box_contact_approximate=approximate, filtered_box_contact=filtered_box,
            bilateral_contact=filtered_box.all(-1), derivative_valid=valid,
            hand_box_position=hand, hand_surface_violation=surface,
            hand_side_error=side_error, hand_slip=slip,
            box_relative_position=relative_pos, box_relative_velocity=relative_vel,
            box_position_violation=position_excess, arm_range_violation=arm_excess,
            box_tilt=tilt, box_angular_velocity=box[:, 10:13], hip_posture_error=hip_error,
            feet_width=feet_width, knee_width=knee_width, foot_heading_error=foot_error,
            waist_error=waist_error, torso_alignment_error=alignment_error,
            grasp_loss_duration=self.grasp_loss_time, proposed_box_drop=drop,
            proposed_separation=separation, box_tilt_exceeded=tilt_exceeded,
            moving_command=moving, termination_reason=self.reason,
        )
