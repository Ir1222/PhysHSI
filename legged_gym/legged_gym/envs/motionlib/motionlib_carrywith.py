"""CarryWith-only sampling using the original PhysHSI feature preprocessing."""
import hashlib
from pathlib import Path

import torch
import yaml

from .motionlib_carrybox import MotionLib
from rsl_rl.modules.masked_amp import project_amp


def validate_carrywith_config(path, motion_root=None):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if set(config.get("motions", {})) != {"carryWith"}:
        raise ValueError(f"{path}: expert motions must contain ONLY carryWith")
    entries = config["motions"]["carryWith"]
    expected = {f"carrywith{i}.pt" for i in (1, 2, 3)}
    if len(entries) != 3 or {Path(e["file"]).name for e in entries} != expected:
        raise ValueError(f"{path}: expected exactly carrywith1.pt, carrywith2.pt, carrywith3.pt")
    files = [(Path(motion_root) / Path(e["file"]).name if motion_root else path.parent / e["file"]).resolve()
             for e in entries]
    weights = torch.tensor([e["weight"] for e in entries], dtype=torch.float)
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError(f"{path}: CarryWith weights must be finite and positive")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError("Missing CarryWith expert files:\n" + "\n".join(missing) +
                                "\nCopy the original CarryWith clips to these paths, or use --amp_motion_root DIR.")
    return path, entries, files


class CarryWithMotionLib(MotionLib):
    def __init__(self, motion_file, mapping_file, dof_names, fps, device,
                 control_dt, thresh_robot2object, motion_root=None):
        self.config_path, self.entries, self.files = validate_carrywith_config(motion_file, motion_root)
        if fps <= 0 or control_dt <= 0:
            raise ValueError("Expert fps and control_dt must be positive")
        self.control_dt = control_dt
        self.motion_root = Path(motion_root).resolve() if motion_root else None
        super().__init__(str(self.config_path), mapping_file, dof_names, fps, device,
                         window_length=10, ratio_random_range=(1., 1.),
                         thresh_robot2object=thresh_robot2object)
        self.ratio = fps * control_dt
        self.identity = dict(source="CarryWith-only", config=str(self.config_path),
                             files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.files],
                             weights=self.motion_weights_tot.tolist(), fps=fps, control_dt=control_dt)

    def resolve_motion_file(self, directory, filename):
        return str(self.motion_root / Path(filename).name) if self.motion_root else super().resolve_motion_file(directory, filename)

    def process_motions(self, dof_names):
        if len(dof_names) != 29 or self.mapping != {name: i for i, name in enumerate(dof_names)}:
            raise ValueError("CarryWith requires the audited 29-joint mapping order")
        for path, clip in zip(self.files, self.motion_data["carryWith"]):
            n = len(clip["base_height"])
            if n - 1 < 9 * self.fps * self.control_dt:
                raise ValueError(f"{path}: CarryWith clip too short for ten control-timestep frames")
            for key, tail in (("base_position", (3,)), ("base_quat", (4,)),
                              ("joint_position", (29,)), ("joint_velocity", (29,)), ("box_pos_local", (3,))):
                if clip[key].shape != (n, *tail) or not torch.isfinite(clip[key]).all():
                    raise ValueError(f"{path}: invalid {key} feature contract")
            if clip["link_position"].ndim != 3 or clip["link_position"].shape[0] != n or clip["link_position"].shape[1] < 5 or clip["link_position"].shape[2] != 3:
                raise ValueError(f"{path}: expected hands/feet/head link positions")
            if not all(torch.isfinite(clip[k]).all() for k in ("base_height", "link_position")):
                raise ValueError(f"{path}: non-finite height/end-effector features")
        super().process_motions(dof_names)

    def get_expert_obs(self, batch_size):
        ids = torch.multinomial(self.motion_weights_tot, batch_size, replacement=True)
        length = self.motion_len_tot[ids]
        span = 9 * self.ratio
        start = torch.rand(batch_size, device=self.device) * (length - 1 - span)
        times = start[:, None] + torch.arange(10, device=self.device)[None, :] * self.ratio
        floor = times.floor().long()
        ceil = torch.minimum(floor + 1, length[:, None] - 1)
        offsets = self.motion_start_ids_tot[ids, None]
        blend = (times - floor).unsqueeze(-1)

        def sample(tensor):
            return tensor[offsets + floor] * (1-blend) + tensor[offsets + ceil] * blend

        frames = torch.cat([sample(t) for t in (
            self.motion_base_height, self.motion_dof_pos, self.motion_end_effector_pos,
            self.motion_box_pos, self.motion_base_lin_vel, self.motion_base_ang_vel,
            self.motion_base_rot)], dim=-1)
        return project_amp(frames.flatten(1))
