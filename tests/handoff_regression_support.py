"""Deterministic inputs shared by the frozen Scope 2 snapshot and native tests."""
import torch

from test_student_carry import fixture, quat
from test_mixed_handoff import MATH


def regression_environment():
    env = fixture()
    env.handoff.settings.stable_duration_s = .04
    # Include every enabled original Teacher reward, not just the two terms in
    # the small ownership fixture. These remain the actual inherited methods.
    scales = dict(dof_acc=-1e-7, action_rate=-.03, torques=-1e-4, dof_vel=-2e-4,
                  dof_pos_limits=-5., dof_vel_limits=-1e-3, torque_limits=-.03,
                  walk_task=1., carryup_task=1., pickup_contact=.3,
                  pickup_lift_progress=1., pickup_success=1., hold_contact=.5,
                  hold_base_still=.4, hold_box_still=.6)
    env.reward_names = sorted(scales)
    env.reward_functions = [getattr(env, "_reward_" + name) for name in env.reward_names]
    env.reward_scales = {name: scales[name]*env.dt for name in env.reward_names}
    env.episode_sums = {name: torch.zeros(2) for name in env.reward_names}
    for name, value in dict(robot2object_pos=0., robot2object_vel=1., start_heading=.5,
                            hand_pos=.7, box_height=2., target_box_height=.72,
                            thresh_object2goal=.05).items():
        setattr(env.cfg.rewards, name, value)
    env.hand_pos_indices = env.hand_colli_indices = [7, 8]
    env.box_height_at_reset = torch.full((2,), .3)
    env.object2goal_dist_xy = torch.ones(2)
    env.base_quat = env.root_states[:, 3:7]
    env.forward_vec = torch.tensor([[1., 0., 0.]]).repeat(2, 1)
    env._reward_walk_task.__func__.__globals__.update(
        quat_apply=MATH.quat_apply, normalize=MATH.normalize,
        wrap_to_pi=lambda x: torch.atan2(x.sin(), x.cos()))
    return env


def apply_sample(env, step):
    """Asynchronous gate -> first Student -> motion -> losses -> terminal sample."""
    env.executed_phase = env.handoff.phase.clone()
    env.executed_command = torch.tensor([[.1*step, -.04*step, .03*step],
                                          [-.02*step, .06*step, -.05*step]])
    env.executed_target = env.executed_command.clone() + .3
    env.episode_length_buf[:] = step + 1
    env.actions[:] = torch.linspace(-.2, .3, 29) + step*.015
    env.dof_pos[:] = torch.linspace(-.3, .4, 29) + step*.01
    env.dof_vel[:] = torch.linspace(-.2, .2, 29) * (step+1)
    env.torques = env.dof_pos * 5
    env.computed_torques = env.torques * 1.1
    env.rigid_body_states[:, 0, 3:7] = quat(.05*step, -.03*step, .08*step)
    env.rigid_body_states[:, 6, 3:7] = quat(.02*step, .2, -.02*step)
    env.rigid_body_states[:, 0, 7:10] = torch.tensor([.08*step, -.03*step, .02])
    env.rigid_body_states[:, 0, 10:13] = torch.tensor([.1, -.2, .04*step])
    env.box_states[:, 0] = .6 + .002*step
    env.box_states[:, 3:7] = quat(.02*step)
    env.box_states[:, 7:13] = torch.tensor([.1, .02, -.01, .1, -.2, .3]) * (step+1)
    env.rigid_body_states[:, 7, :3] = env.box_states[:, :3] + torch.tensor([.01, .15, -.01])
    env.rigid_body_states[:, 8, :3] = env.box_states[:, :3] + torch.tensor([-.01, -.15, .01])
    env.rigid_body_states[:, 1, 1] = .11
    env.rigid_body_states[:, 2, 1] = -.10
    env.rigid_body_states[:, 3, 1] = .105
    env.rigid_body_states[:, 4, 1] = -.095
    env.rigid_body_states[:, [1, 2], 7] = .02*step
    env.feet_air_time[:] = .6 + .01*step
    env.first_contacts[:] = step % 2 == 0
    env.contact_forces[:, [1, 2], 2] = 20.
    env.contact_forces[:, [7, 8], 0] = 2.
    if step < 2:
        env.contact_forces[1, 8] = 0.  # Second environment hands over later.
    if step >= 4:
        env.contact_forces[0, [7, 8]] = 0.
    if step == 6:
        env.box_states[0, 2] = .2
        env.box_states[1, 0] = 2.
        env.episode_length_buf[:] = 3001
    env.robot2object_dir = env.box_states[:, :2] - env.root_states[:, :2]
    env.robot2object_dist = env.robot2object_dir.norm(dim=-1)


def snapshot(env, raw_student, metrics, student_steps, student_total, student_sums):
    return dict(
        diagnostics={k: v.detach().tolist() for k, v in env.extras["handoff"].items()},
        raw_student={k: v.detach().tolist() for k, v in raw_student.items()},
        metrics={k: v.detach().tolist() for k, v in metrics.items()},
        teacher_sums={k: v.detach().tolist() for k, v in env.episode_sums.items()},
        student_sums={k: v.detach().tolist() for k, v in student_sums.items()},
        student_steps=student_steps.tolist(), student_total=student_total.tolist(),
    )
