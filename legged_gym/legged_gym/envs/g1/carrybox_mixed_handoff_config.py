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

    class diagnostics:
        detailed = False  # Opt in for Viewer CSV / detailed reward debugging.

    class student_rewards:
        class scales:
            carry_lin_vel_tracking = 3.0
            carry_yaw_vel_tracking = 2.5
            carry_bilateral_contact = 0.5
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


class MixedHandoffTrainingCfg(MixedHandoffCfg):
    """Candidate D; Viewer and original carrybox retain full Approach resets."""
    class env(MixedHandoffCfg.env):
        num_envs = G1Cfg.env.num_envs
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
        max_iterations = G1CfgPPO.runner.max_iterations
        max_optimizer_updates = None  # No extra limit on nonempty rollout updates.
        terminal_log_interval = 100
        save_interval = 500
        experiment_name = "mixed_handoff_student"
        run_name = "scope3"
        teacher_checkpoint = "{LEGGED_GYM_ROOT_DIR}/logs/minimal_carrybox/Oct25_19-58-13_loco_pickup_hold_resume/model_17000.pt"
        student_init_checkpoint = None  # Same pretrained Actor/std as Teacher.
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
