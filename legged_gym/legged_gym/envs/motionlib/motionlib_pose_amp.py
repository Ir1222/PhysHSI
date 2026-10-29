"""Mode-conditioned single-frame HOLD/CarryWith Pose AMP expert sampling."""
import hashlib
from pathlib import Path

import torch

from .motionlib_carrybox import MotionLib
from .motionlib_carrywith import validate_carrywith_config
from rsl_rl.modules.masked_pose_amp import condition_pose_observation, project_pose_amp


class PoseAMPMotionLib(MotionLib):
    """Sample independent valid reference poses with the original preprocessing."""
    def __init__(self, motion_file, mapping_file, dof_names, fps, device,
                 control_dt, thresh_robot2object, motion_root=None):
        self.config_path, self.entries, self.files = validate_carrywith_config(
            motion_file, motion_root)
        if fps != 60:
            raise ValueError("HOLD/CarryWith experts are audited at exactly 60 FPS")
        if control_dt <= 0:
            raise ValueError("Expert control_dt must be positive")
        self.control_dt = control_dt
        self.motion_root = Path(motion_root).resolve() if motion_root else None
        self._resolved_by_name = {file.name: file for file in self.files}
        super().__init__(str(self.config_path), mapping_file, dof_names, fps, device,
                         window_length=1, ratio_random_range=(1., 1.),
                         thresh_robot2object=thresh_robot2object)
        source = "Pose-HOLD+CarryWith-mode-conditioned" if "hold" in self.skills else "Pose-CarryWith-only"
        self.identity = dict(
            source=source,
            config=str(self.config_path),
            files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in self.files],
            weights={skill: self.motion_weights[skill].tolist() for skill in self.skills},
            fps=fps,
            control_dt=control_dt,
            pose_features=54,
            pose_history=1,
            condition="stand=1,moving=0",
        )

    def resolve_motion_file(self, directory, filename):
        return str(self._resolved_by_name[Path(filename).name])

    def process_motions(self, dof_names):
        if len(dof_names) != 29 or self.mapping != {name: i for i, name in enumerate(dof_names)}:
            raise ValueError("HOLD/CarryWith requires the audited 29-joint mapping order")
        for skill in self.skills:
            for entry, clip in zip(self.entries[skill], self.motion_data[skill]):
                path = self._resolved_by_name[Path(entry["file"]).name]
                n = len(clip["base_height"])
                if n < 1:
                    raise ValueError(f"{path}: expert clip must contain at least one pose")
                if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                       for value in clip.values()):
                    raise ValueError(f"{path}: expert clip contains non-finite values")
                contracts = (("base_height", ()), ("base_position", (3,)), ("base_quat", (4,)),
                             ("joint_position", (29,)), ("joint_velocity", (29,)),
                             ("box_pos_local", (3,)))
                for key, tail in contracts:
                    if (key not in clip or clip[key].shape != (n, *tail)
                            or not torch.isfinite(clip[key]).all()):
                        raise ValueError(f"{path}: invalid {key} feature contract")
                links = clip.get("link_position")
                if (links is None or links.ndim != 3 or links.shape[0] != n
                        or links.shape[1] < 5 or links.shape[2] != 3
                        or not torch.isfinite(links).all()):
                    raise ValueError(f"{path}: invalid hands/feet/head link positions")
                for key in ("base_quat", "box_quat"):
                    if key in clip:
                        quat = clip[key]
                        if quat.shape != (n, 4) or not torch.isfinite(quat).all():
                            raise ValueError(f"{path}: invalid {key} feature contract")
                        if not torch.allclose(quat.norm(dim=-1), torch.ones(n), atol=1e-3, rtol=1e-3):
                            raise ValueError(f"{path}: {key} must contain unit quaternions")
        super().process_motions(dof_names)

    def _sample_pose(self, skill, batch_size):
        ids = torch.multinomial(self.motion_weights[skill], batch_size, replacement=True)
        length = self.motion_len[skill][ids]
        # Draw a valid continuous reference-frame coordinate. For a one-frame
        # clip, floor == ceil == 0; otherwise interpolation stays in bounds.
        coordinate = torch.rand(batch_size, device=self.device) * (length - 1).clamp_min(0)
        floor = coordinate.floor().long()
        ceil = torch.minimum(floor + 1, length - 1)
        offsets = self.motion_start_ids[skill][ids]
        blend = (coordinate - floor).unsqueeze(-1)

        def sample(tensor):
            return tensor[offsets + floor] * (1 - blend) + tensor[offsets + ceil] * blend

        raw_pose = torch.cat([sample(t) for t in (
            self.motion_base_height,
            self.motion_dof_pos,
            self.motion_end_effector_pos,
            self.motion_box_pos,
            self.motion_base_lin_vel,
            self.motion_base_ang_vel,
            self.motion_base_rot,
        )], dim=-1)
        return project_pose_amp(raw_pose)

    def get_expert_obs(self, batch_size, stand_mode=None):
        if stand_mode is None:
            stand_mode = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        else:
            stand_mode = torch.as_tensor(stand_mode, device=self.device).bool().reshape(-1)
            if len(stand_mode) != batch_size:
                raise ValueError("Expert Stand/Moving labels must match batch_size")
        pose = torch.empty(batch_size, 54, device=self.device)
        for mode, skill in ((False, "carryWith"), (True, "hold")):
            selected = stand_mode == mode
            if not selected.any():
                continue
            # CarryWith-only remains an explicit ablation. Its Stand-labelled
            # rows keep the discriminator condition even when the source falls back.
            source = skill if skill in self.skills else "carryWith"
            pose[selected] = self._sample_pose(source, int(selected.sum()))
        return condition_pose_observation(pose, stand_mode)
