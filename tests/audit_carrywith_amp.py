"""Optional CPU audit of real legacy CarryWith clips against the current URDF.

Run from the repository root to audit the tracked CarryWith dataset. An optional
--motion_root DIR selects another copy. No Isaac Gym is imported.
The unit-test import adapter loads the real MotionLib with the real quaternion
utilities while bypassing the simulator-dependent environment package initializer.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from test_mixed_handoff_training import CarryWithMotionLib, ROOT


def audit(motion_root=None, samples=12000):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    mapping = ROOT / "legged_gym/resources/config/joint_id.txt"
    names = [line.split()[1] for line in mapping.read_text().splitlines()]
    sampler = CarryWithMotionLib(ROOT / "legged_gym/resources/config/carrybox_carrywith_only.yaml",
                                str(mapping), names, 60, "cpu", .02, .7, motion_root)
    joints = ET.parse(ROOT / "legged_gym/resources/robots/g1/urdf/g1_29dof.urdf").getroot().findall("joint")
    links = ["left_palm_link", "right_palm_link", "left_ankle_pitch_link", "right_ankle_pitch_link", "mid360_link"]
    clips = []
    for path, data in zip(sampler.files, sampler.motion_data["carryWith"]):
        errors, height_errors = [], []
        for frame in range(len(data["base_height"])):
            transforms, pending = {"pelvis": np.eye(4)}, list(joints)
            while pending:
                count = len(pending)
                for joint in list(pending):
                    parent = joint.find("parent").get("link")
                    if parent not in transforms:
                        continue
                    origin = joint.find("origin")
                    attributes = {} if origin is None else origin.attrib
                    matrix = np.eye(4)
                    matrix[:3, 3] = np.fromstring(attributes.get("xyz", "0 0 0"), sep=" ")
                    matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(attributes.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                    if joint.get("type") in ("revolute", "continuous"):
                        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                        angle = float(data["joint_position"][frame, names.index(joint.get("name"))])
                        rotation = np.eye(4)
                        rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
                        matrix = matrix @ rotation
                    transforms[joint.find("child").get("link")] = transforms[parent] @ matrix
                    pending.remove(joint)
                if len(pending) == count:
                    raise ValueError("Unresolved URDF hierarchy")
            expected = np.array([transforms[link][:3, 3] for link in links])
            base = Rotation.from_quat(data["base_quat"][frame].numpy())
            local = base.inv().apply(data["link_position"][frame, :5].numpy())
            errors.append(np.linalg.norm(local-expected, axis=1))
            # Same max(root_z - left/right ankle_z) definition as the environment.
            expected_height = -np.min(base.apply(expected[2:4])[:, 2])
            height_errors.append(abs(float(data["base_height"][frame])-expected_height))
        max_error = np.max(errors, axis=0)
        if max_error.max() > .005 or max(height_errors) > 1e-5:
            raise ValueError(f"{path}: legacy joint/end-effector/height semantics fail FK audit")
        clips.append(dict(file=path.name, frames=len(errors), end_effector_max_error_m=max_error.tolist(),
                          base_height_max_error_m=max(height_errors)))
    observations = sampler.get_expert_obs(samples)
    if observations.shape != (samples, 540) or not torch.isfinite(observations).all():
        raise ValueError("Invalid real expert samples")
    return dict(identity=sampler.identity, clips=clips, samples=samples,
                sample_shape=list(observations.shape), finite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_root", help="Override the tracked dataset; defaults to paths in carrybox_carrywith_only.yaml")
    parser.add_argument("--samples", type=int, default=12000)
    args = parser.parse_args()
    print(json.dumps(audit(args.motion_root, args.samples), indent=2))
