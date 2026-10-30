"""Configuration for the continuous frozen-Teacher / Student task."""

from .carrybox_config import G1Cfg, G1CfgPPO


class VelTrackPoseAMPCfg(G1Cfg):

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

    class diagnostics:
        detailed = False  # Opt in for Viewer CSV / detailed reward debugging.

    class gait:
        contact_threshold_N = 1.0
        moving_threshold = 0.1
        slip_tolerance_m_s = 0.05
        horizontal_speed_scale_m_s = 0.30
        target_height_m = 0.04
        scuff_height_m = 0.02
        air_time_min_s = 0.06
        air_time_range_s = 0.20
        geometry_max_points = 128
        geometry_fit_tolerance_m = 0.0005
        geometry_validation_tolerance_m = 0.001

    class student_rewards:
        class scales:
            carry_lin_vel_tracking = 2.5 #3.0
            carry_yaw_vel_tracking = 3.0
            carry_bilateral_contact = 0.5
            gait_foot_slip = -0.05
            gait_swing_clearance = 0.08
            gait_foot_scuff = -0.05
            gait_air_time = 0.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -1e-7
            action_rate = -0.03
            torques = -1e-4
            dof_vel = -2e-4
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            torque_limits = -0.03

        carry_lin_vel_sigma = 0.25  # (m/s)^2, pelvis heading frame
        carry_yaw_vel_sigma = 0.25  # (rad/s)^2, world-Z yaw rate

    class student_termination:
        box_drop_height = 0.15  # Rotated box bottom above ground, m
        robot_box_max_distance = 1.0  # Pelvis-to-box XY distance, m
        grasp_loss_grace_s = 0.3  # s of complete filtered contact loss
        box_tilt_termination_deg = 70.0
        terminate_box_tilt = False  # Existing successful handoff can exceed 70 degrees.


class VelTrackPoseAMPTrainingCfg(VelTrackPoseAMPCfg):
    """Candidate D; Viewer and original carrybox retain full Approach resets."""
    class env(VelTrackPoseAMPCfg.env):
        num_envs = G1Cfg.env.num_envs
        episode_length_s = 30.0
        test = False

    class asset(VelTrackPoseAMPCfg.asset):
        class box(VelTrackPoseAMPCfg.asset.box):
            reset_mode = "random"
            skill = ["pickUp"]
            skill_init_prob = [1.0]

    class pickup_rsi:
        enabled = True
        window = "contact"  # "full" delegates to the original full-clip sampler.
        contact_window_s = (-0.2, 0.2)
        log_resets = False

    class handoff(VelTrackPoseAMPCfg.handoff):
        stable_duration_s = 0.1
        student_initial_stand_duration_s = 0.0


class VelTrackPoseAMPCfgPPO(G1CfgPPO):
    runner_class_name = "VelTrackPoseAMPOnPolicyRunner"

    class algorithm(G1CfgPPO.algorithm):
        learning_rate = 2e-4  # Carry12 velocity-tracking baseline
        value_smoothness_coef = .1
        smoothness_upper_bound = 1.
        smoothness_lower_bound = .1
        amp_batch_size = 256

    class runner(G1CfgPPO.runner):
        algorithm_class_name = "MixedHandoffPPO"
        num_steps_per_env = 100
        max_iterations = G1CfgPPO.runner.max_iterations
        max_optimizer_updates = None  # No extra limit on nonempty rollout updates.
        terminal_log_interval = 100
        save_interval = 500
        experiment_name = "vel_track_pose_amp"
        run_name = "pose_amp_v1"
        teacher_checkpoint = "{LEGGED_GYM_ROOT_DIR}/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt"
        student_init_checkpoint = None  # Same pretrained Actor/std as Teacher.
        collect_only = False
        detailed_handoff_events = False

    class amp:
        enabled = True
        num_obs = 54
        num_one_step_obs = 54
        window_length = 1
        discriminator_num_obs = 55  # 54D pose + one Stand(1)/Moving(0) condition.
        pose_style_weight = 2.0
        pose_gate_floor = 0.5
        stand_linear_threshold = 0.05
        stand_yaw_threshold = 0.05
        use_normalizer = False
        expert_config = "{LEGGED_GYM_ROOT_DIR}/resources/config/carrybox_hold_carrywith_amp.yaml"
