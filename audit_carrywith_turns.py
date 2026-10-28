"""Audit sustained CarryWith turning vs short-term yaw jitter (Windows-compatible).

Run from the PhysHSI repository root:
    python audit_carrywith_turns.py
Or:
    python audit_carrywith_turns.py --dir 'D:\\...\\carryWith'

Requires: torch, numpy, scipy. Only open .pt files from trusted sources.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def load_clip(path):
    # Local project files are trusted; torch.load can deserialize arbitrary objects.
    data = torch.load(path, map_location="cpu", weights_only=False)
    quat = np.asarray(data["base_quat"], dtype=np.float64)
    position = np.asarray(data["base_position"], dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4 or position.shape != (len(quat), 3):
        raise ValueError(f"Invalid shape in {path}")
    # Original PhysHSI stores xyzw quaternions.
    yaw = np.unwrap(Rotation.from_quat(quat).as_euler("xyz", degrees=False)[:, 2])
    return yaw, position


def metrics(yaw, position, fps, window_s, yaw_threshold, stand_speed):
    k = round(fps * window_s)
    if len(yaw) <= k:
        return None
    duration = k / fps
    # Difference over a *whole* window; back-and-forth heading jitter cancels.
    turning_rate = (yaw[k:] - yaw[:-k]) / duration
    xy_speed = np.linalg.norm(position[k:, :2] - position[:-k, :2], axis=1) / duration
    turning = np.abs(turning_rate) > yaw_threshold
    standing = xy_speed < stand_speed
    return dict(
        turn=float(turning.mean()),
        left=float((turning_rate > yaw_threshold).mean()),
        right=float((turning_rate < -yaw_threshold).mean()),
        turn_in_place=float((turning & standing).mean()),
        turn_while_moving=float((turning & ~standing).mean()),
        p90=float(np.quantile(np.abs(turning_rate), .9)),
        mean_abs=float(np.abs(turning_rate).mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("legged_gym/resources/dataset/dataset_carrybox/carryWith"))
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--yaw-threshold", type=float, default=.3)
    parser.add_argument("--stand-speed", type=float, default=.15)
    args = parser.parse_args()
    paths = [args.dir / f"carrywith{i}.pt" for i in (1, 2, 3)]
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        parser.error("Missing files: " + ", ".join(missing))
    weights = np.asarray([5., 2., 5.])
    weights /= weights.sum()
    durations = [.25, .5, 1.]
    by_window = {s: [] for s in durations}
    for path, weight in zip(paths, weights):
        yaw, pos = load_clip(path)
        net = np.degrees(yaw[-1] - yaw[0])
        gross = np.degrees(np.abs(np.diff(yaw)).sum())
        print(f"\n{path.name}: {len(yaw) / args.fps:.2f}s, net={net:+.1f} deg, gross={gross:.1f} deg, weight={weight:.1%}")
        for s in durations:
            data = metrics(yaw, pos, args.fps, s, args.yaw_threshold, args.stand_speed)
            if data is None:
                continue
            by_window[s].append(data)
            print(f"  {s:>4.2f}s: sustained turn={data['turn']:5.1%}  L={data['left']:5.1%} R={data['right']:5.1%}"
                  f"  in-place={data['turn_in_place']:5.1%} moving={data['turn_while_moving']:5.1%}"
                  f"  mean|w|={data['mean_abs']:.3f} p90={data['p90']:.3f} rad/s")
    print("\n=== Clip-weighted aggregate (AMP sampling weights 5:2:5) ===")
    for s in durations:
        rows = by_window[s]
        if len(rows) != len(paths):
            continue
        weighted = {key: sum(w * row[key] for w, row in zip(weights, rows)) for key in rows[0]}
        print(f"{s:.2f}s: turn={weighted['turn']:.1%}, left={weighted['left']:.1%},"
              f" right={weighted['right']:.1%}, in-place={weighted['turn_in_place']:.1%},"
              f" moving={weighted['turn_while_moving']:.1%}")
    print("NOTE: A turning window is not proof of physically plausible turning gait;")
    print("      in-place means only low root XY displacement (not a validated planted-foot pivot).")


if __name__ == "__main__":
    main()
