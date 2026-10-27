"""Configuration for the continuous frozen-Teacher / Student task."""

from .carrybox_config import G1Cfg, G1CfgPPO


class MixedHandoffCfg(G1Cfg):

    class env(G1Cfg.env):
        num_envs = 1
        episode_length_s = 60
        test = True

    class asset(G1Cfg.asset):

        class box(G1Cfg.asset.box):
            reset_mode = "default"

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
        fixed_command = (0.0, 0.0, 0.0)
        contact_force = 1.0
        hand_box_margin = 0.08
        min_clearance = 0.01

    class student_rewards:

        class scales:
            carry_lin_vel_tracking = 3.0
            carry_yaw_vel_tracking = 2.5
            carry_bilateral_contact = 0.5
            carry_hand_box_surface = 1.5
            carry_hand_slip = 0.5
            carry_relative_velocity = 0.5
            carry_relative_position = 0.75
            carry_arm_range = 0.2
            carry_box_tilt = -0.5
            carry_hip_posture = 0.8
            carry_foot_heading = 0.35
            carry_feet_width = 0.3
            carry_knee_width = 0.2
            carry_waist_reference = 0.4
            carry_torso_pelvis_alignment = 0.2
            zero_command_stillness = 0.2
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            feet_air_time = 0.2
            feet_slip = -0.2
            feet_stumble = -1.0
            collision = -0.5
            feet_contact_forces = -0.001
            dof_acc = -1e-07
            action_rate = -0.03
            torques = -0.0001
            dof_vel = -0.0002
            dof_pos_limits = -5.0
            dof_vel_limits = -0.001
            torque_limits = -0.03

        carry_lin_vel_sigma = 0.25  # (m/s)^2
        carry_yaw_vel_sigma = 0.25  # (rad/s)^2
        hand_contact_threshold = 1.0  # N; box proximity is also required
        carry_hand_side_tolerance = 0.03
        carry_hand_face_margin = 0.01
        carry_hand_surface_violation_scale = 0.04
        carry_hand_slip_tolerance = 0.35
        carry_hand_slip_violation_scale = 0.35
        carry_arm_joint_names = (
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        )
        carry_arm_range_lower = [
            -0.9,
            -0.25,
            -0.85,
            -0.25,
            -1.1,
            -0.6,
            -0.6,
            -1.1,
            -1.0,
            -0.55,
            -0.25,
            -0.55,
            -0.6,
            -0.6,
        ]
        carry_arm_range_upper = [
            0.55,
            0.8,
            0.7,
            1.15,
            0.55,
            0.6,
            0.6,
            0.25,
            0.25,
            1.0,
            1.4,
            1.1,
            0.6,
            0.6,
        ]
        carry_arm_violation_scale = 0.35
        carry_box_relative_position_lower = [0.18, -0.2, -0.15]
        carry_box_relative_position_upper = [0.53, 0.2, 0.25]
        carry_box_position_violation_scale = [0.1, 0.1, 0.1]
        carry_box_relative_velocity_tolerance = [0.35, 0.45, 0.45]
        carry_box_velocity_violation_scale = [0.35, 0.45, 0.45]
        carry_hip_joint_names = (
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        )
        carry_hip_posture_deadzone = [0.08, 0.1, 0.08, 0.1]
        carry_hip_posture_softness = [0.15, 0.15, 0.15, 0.15]
        carry_foot_heading_deadzone = 0.12
        carry_foot_links = ("left_ankle_pitch_link", "right_ankle_pitch_link")
        carry_knee_links = ("left_knee_link", "right_knee_link")
        carry_foot_heading_softness = 0.2
        carry_feet_width_range = [0.09, 0.27]
        carry_feet_width_softness = 0.08
        carry_knee_width_range = [0.14, 0.25]
        carry_knee_width_softness = 0.06
        carry_waist_joint_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        carry_waist_reference_target = [-0.041372441, 0.034433149, 0.249833077]
        carry_waist_reference_deadzone = [0.3, 0.07, 0.12]
        carry_waist_reference_softness = [0.2, 0.1, 0.12]
        carry_torso_pelvis_reference_quat = [0.020052686, 0.124780931, -0.023215466, 0.991709963]
        carry_torso_pelvis_alignment_deadzone = [0.12, 0.12, 0.3]
        carry_torso_pelvis_alignment_softness = [0.12, 0.12, 0.2]

    class student_termination:
        box_drop_height = 0.15  # Rotated box bottom above ground, m
        robot_box_max_distance = 1.0  # Pelvis-to-box XY distance, m
        grasp_loss_grace_s = 0.3  # s of complete filtered contact loss
        box_tilt_termination_deg = 70.0
        terminate_box_tilt = False  # Existing successful handoff can exceed 70 degrees.


class MixedHandoffTrainingCfg(MixedHandoffCfg):
    """Candidate D; Viewer and original carrybox retain full Approach resets."""
    class env(MixedHandoffCfg.env):
        num_envs = 16
        episode_length_s = 20.0
        test = False

    class asset(MixedHandoffCfg.asset):
        class box(MixedHandoffCfg.asset.box):
            reset_mode = "random"
            skill = ["pickUp"]
            skill_init_prob = [1.0]

    class pickup_rsi:
        enabled = True
        window = "contact"  # "full" delegates to the original full-clip sampler.
        contact_window_s = (-0.2, 0.2)
        log_resets = False

    class handoff(MixedHandoffCfg.handoff):
        stable_duration_s = 0.1
        student_initial_stand_duration_s = 0.0


class MixedHandoffCfgPPO(G1CfgPPO):
    runner_class_name = "MixedHandoffOnPolicyRunner"

    class algorithm(G1CfgPPO.algorithm):
        learning_rate = 2e-4  # Carry12 velocity-tracking baseline
        value_smoothness_coef = .1
        smoothness_upper_bound = 1.
        smoothness_lower_bound = .1
        amp_batch_size = 256

    class runner(G1CfgPPO.runner):
        algorithm_class_name = "MixedHandoffPPO"
        num_steps_per_env = 100
        max_iterations = 20  # collection iterations, including Teacher warmup
        max_optimizer_updates = 5
        terminal_log_interval = 100
        save_interval = 500
        experiment_name = "mixed_handoff_student"
        run_name = "scope3"
        teacher_checkpoint = "{LEGGED_GYM_ROOT_DIR}/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt"
        student_init_checkpoint = None  # Same pretrained Actor/std as Teacher.
        allow_legacy_contract = False
        collect_only = False
        detailed_handoff_events = False

    class amp:
        enabled = True
        num_obs = 540
        num_one_step_obs = 54
        window_length = 10
        amp_coef = .25
        reward_scale = .5  # same discriminator reward scale as HIMOnPolicyRunner
        use_normalizer = False
        expert_config = "{LEGGED_GYM_ROOT_DIR}/resources/config/carrybox_carrywith_only.yaml"
