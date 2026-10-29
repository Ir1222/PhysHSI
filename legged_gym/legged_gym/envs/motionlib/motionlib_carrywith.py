"""Mode-conditioned HOLD/CarryWith sampling with original PhysHSI preprocessing."""
import hashlib
from pathlib import Path

import torch
import yaml

from .motionlib_carrybox import MotionLib
from rsl_rl.modules.masked_amp import condition_amp_observation, project_amp


EXPECTED_FILES = {
    "carryWith": ("carrywith1.pt", "carrywith2.pt", "carrywith3.pt"),
    "hold": ("hold_carrywith1.pt", "hold_carrywith2.pt", "hold_pickup2.pt", "hold_pickup3.pt"),
}


def _resolve_file(config_path, entry, skill, motion_root):
    if motion_root is None:
        return (config_path.parent / entry["file"]).resolve()
    root = Path(motion_root).resolve()
    subdir = "hold_synthetic" if skill == "hold" else "carryWith"
    nested = root / subdir / Path(entry["file"]).name
    return (nested if nested.is_file() else root / Path(entry["file"]).name).resolve()


def validate_carrywith_config(path, motion_root=None):
    """Validate either the legacy CarryWith ablation or HOLD+CarryWith config."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    motions = config.get("motions", {})
    if set(motions) not in ({"carryWith"}, {"carryWith", "hold"}):
        raise ValueError(f"{path}: expert motions must contain carryWith and optional hold only")
    files = []
    for skill in ("carryWith", "hold"):
        if skill not in motions:
            continue
        entries = motions[skill]
        expected = EXPECTED_FILES[skill]
        names = tuple(Path(entry["file"]).name for entry in entries)
        if len(entries) != len(expected) or set(names) != set(expected):
            raise ValueError(f"{path}: {skill} expects exactly {', '.join(expected)}")
        weights = torch.tensor([entry["weight"] for entry in entries], dtype=torch.float)
        if not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError(f"{path}: {skill} weights must be finite and positive")
        if skill == "carryWith" and names != expected:
            raise ValueError(f"{path}: CarryWith order must preserve weights 5:2:5")
        if skill == "carryWith" and not torch.equal(weights, torch.tensor([5., 2., 5.])):
            raise ValueError(f"{path}: CarryWith weights must remain 5:2:5")
        files.extend(_resolve_file(path, entry, skill, motion_root) for entry in entries)
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Missing AMP expert files:\n" + "\n".join(missing) +
                                "\nUse the tracked dataset paths or --amp_motion_root DIR.")
    return path, motions, files


class CarryWithMotionLib(MotionLib):
    """Backward-compatible name for the mode-conditioned carry expert library."""
    def __init__(self, motion_file, mapping_file, dof_names, fps, device,
                 control_dt, thresh_robot2object, motion_root=None):
        self.config_path, self.entries, self.files = validate_carrywith_config(motion_file, motion_root)
        if fps != 60:
            raise ValueError("HOLD/CarryWith experts are audited at exactly 60 FPS")
        if control_dt <= 0:
            raise ValueError("Expert control_dt must be positive")
        self.control_dt = control_dt
        self.motion_root = Path(motion_root).resolve() if motion_root else None
        self._resolved_by_name = {file.name: file for file in self.files}
        super().__init__(str(self.config_path), mapping_file, dof_names, fps, device,
                         window_length=10, ratio_random_range=(1., 1.),
                         thresh_robot2object=thresh_robot2object)
        self.ratio = fps * control_dt
        source = "HOLD+CarryWith-mode-conditioned" if "hold" in self.skills else "CarryWith-only"
        self.identity = dict(source=source, config=str(self.config_path),
                             files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                    for p in self.files],
                             weights={skill: self.motion_weights[skill].tolist() for skill in self.skills},
                             fps=fps, control_dt=control_dt, condition="stand=1,moving=0")

    def resolve_motion_file(self, directory, filename):
        return str(self._resolved_by_name[Path(filename).name])

    def process_motions(self, dof_names):
        if len(dof_names) != 29 or self.mapping != {name: i for i, name in enumerate(dof_names)}:
            raise ValueError("HOLD/CarryWith requires the audited 29-joint mapping order")
        for skill in self.skills:
            entries = self.entries[skill]
            for entry, clip in zip(entries, self.motion_data[skill]):
                path = self._resolved_by_name[Path(entry["file"]).name]
                n = len(clip["base_height"])
                if n - 1 < 9 * self.fps * self.control_dt:
                    raise ValueError(f"{path}: expert clip too short for ten control-timestep frames")
                if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                       for value in clip.values()):
                    raise ValueError(f"{path}: expert clip contains non-finite values")
                contracts = (("base_height", ()), ("base_position", (3,)), ("base_quat", (4,)),
                             ("joint_position", (29,)), ("joint_velocity", (29,)),
                             ("box_pos_local", (3,)))
                for key, tail in contracts:
                    if key not in clip or clip[key].shape != (n, *tail) or not torch.isfinite(clip[key]).all():
                        raise ValueError(f"{path}: invalid {key} feature contract")
                links = clip.get("link_position")
                if (links is None or links.ndim != 3 or links.shape[0] != n
                        or links.shape[1] < 5 or links.shape[2] != 3 or not torch.isfinite(links).all()):
                    raise ValueError(f"{path}: invalid hands/feet/head link positions")
                for key in ("base_quat", "box_quat"):
                    if key in clip:
                        quat = clip[key]
                        if quat.shape != (n, 4) or not torch.isfinite(quat).all():
                            raise ValueError(f"{path}: invalid {key} feature contract")
                        if not torch.allclose(quat.norm(dim=-1), torch.ones(n), atol=1e-3, rtol=1e-3):
                            raise ValueError(f"{path}: {key} must contain unit quaternions")
        super().process_motions(dof_names)

    def _sample_motion(self, skill, batch_size):
        ids = torch.multinomial(self.motion_weights[skill], batch_size, replacement=True)
        length = self.motion_len[skill][ids]
        span = 9 * self.ratio
        start = torch.rand(batch_size, device=self.device) * (length - 1 - span)
        times = start[:, None] + torch.arange(10, device=self.device)[None, :] * self.ratio
        floor = times.floor().long()
        ceil = torch.minimum(floor + 1, length[:, None] - 1)
        offsets = self.motion_start_ids[skill][ids, None]
        blend = (times - floor).unsqueeze(-1)

        def sample(tensor):
            return tensor[offsets + floor] * (1-blend) + tensor[offsets + ceil] * blend

        frames = torch.cat([sample(t) for t in (
            self.motion_base_height, self.motion_dof_pos, self.motion_end_effector_pos,
            self.motion_box_pos, self.motion_base_lin_vel, self.motion_base_ang_vel,
            self.motion_base_rot)], dim=-1)
        return project_amp(frames.flatten(1))

    def get_expert_obs(self, batch_size, stand_mode=None):
        if stand_mode is None:
            stand_mode = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        else:
            stand_mode = torch.as_tensor(stand_mode, device=self.device).bool().reshape(-1)
            if len(stand_mode) != batch_size:
                raise ValueError("Expert Stand/Moving labels must match batch_size")
        motion = torch.empty(batch_size, 540, device=self.device)
        for mode, skill in ((False, "carryWith"), (True, "hold")):
            selected = stand_mode == mode
            if not selected.any():
                continue
            # CarryWith-only remains an explicit ablation: Stand-labelled expert
            # rows draw from CarryWith while retaining the discriminator label.
            source = skill if skill in self.skills else "carryWith"
            motion[selected] = self._sample_motion(source, int(selected.sum()))
        return condition_amp_observation(motion, stand_mode)
