"""Read and compress foot collision geometry for sole-clearance rewards."""

import hashlib
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


def _mesh_vertices(path):
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        triangles = np.dtype([
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ])
        vertices = np.frombuffer(data, dtype=triangles, count=count, offset=84)["vertices"]
        vertices = vertices.reshape(-1, 3)
    else:
        vertices = np.array([
            [float(value) for value in line.split()[1:]]
            for line in data.decode("ascii").splitlines()
            if line.strip().startswith("vertex ")
        ])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid STL: " + str(path))
    return np.unique(vertices, axis=0).astype(float)


def _collision_vertices(urdf, body_name):
    urdf = Path(urdf)
    link = ET.parse(str(urdf)).find("link[@name='" + body_name + "']")
    if link is None:
        raise ValueError("Missing gait collision body in URDF: " + body_name)
    groups, meshes = [], []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            raise ValueError("Expected mesh collision for gait body: " + body_name)
        path = (urdf.parent / mesh.get("filename")).resolve()
        vertices = _mesh_vertices(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        roll, pitch, yaw = np.fromstring(
            origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "
        )
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        groups.append(vertices @ rotation.T + xyz)
        meshes.append(path)
    if not groups:
        raise ValueError("Gait body has no active collision geometry: " + body_name)
    return np.concatenate(groups), meshes


def build_gait_geometry(urdf, body_names, max_points=128, fit_tolerance=0.0005,
                        validation_tolerance=0.001):
    """Compress collision support vertices and validate on independent directions."""
    if not 4 <= max_points <= 256:
        raise ValueError("geometry max_points must be in [4, 256]")

    def directions(count):
        index = np.arange(count) + 0.5
        z = 1 - 2 * index / count
        phi = index * np.pi * (3 - np.sqrt(5))
        radius = np.sqrt(1 - z * z)
        return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), -1)

    result = {}
    fit_directions = directions(2048)
    for name in body_names:
        vertices, meshes = _collision_vertices(urdf, name)
        support = fit_directions @ vertices.T
        exact = support.min(-1)
        indices = [int(vertices[:, 2].argmin())]
        estimate = support[:, indices[0]].copy()
        while (estimate - exact).max() > fit_tolerance and len(indices) < max_points:
            direction = int((estimate - exact).argmax())
            index = int(support[direction].argmin())
            indices.append(index)
            estimate = np.minimum(estimate, support[:, index])
        points = vertices[indices]
        errors = []
        for batch in np.array_split(directions(8192), 32):
            errors.extend(((batch @ points.T).min(-1) - (batch @ vertices.T).min(-1)).tolist())
        errors = np.maximum(errors, 0)
        error = {
            "max_m": float(errors.max()),
            "mean_m": float(errors.mean()),
            "p95_m": float(np.quantile(errors, 0.95)),
            "validation_directions": 8192,
        }
        if error["max_m"] > validation_tolerance:
            raise ValueError("Gait geometry validation failed: " + name + " " + repr(error))
        result[name] = {
            "points": points.tolist(),
            "full_vertex_count": len(vertices),
            "point_count": len(points),
            "error": error,
            "mesh_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in meshes],
        }
    urdf = Path(urdf)
    return {
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "method": "greedy collision support vertices; independent sphere validation",
        "bodies": result,
    }
