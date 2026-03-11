#!/usr/bin/env python3
"""
Convert a poses JSON file to a NumPy array using the right-handed display-world
convention used by reconstruct_3d.py when logging to Rerun.

Coordinate convention (output):
    Right-handed world — +X forward, +Y left, +Z up.
    This is Unreal's left-handed world (+X forward, +Y right, +Z up) with
    the Y axis negated, which matches Rerun's RIGHT_HAND_Z_UP view coordinates.

Output format:
    A (N, 4, 4) float64 array of camera-to-world SE3 matrices:

        [ R  | t ]
        [----+---]
        [ 0  | 1 ]

    where R (3×3) satisfies v_world = R @ v_camera_local, and t (3,) is the
    camera position in cm.  The camera-local axes are:
        +X  forward
        +Y  world-LEFT  (visual right = −local Y)
        +Z  up

Usage:
    ./export_poses.py poses.json                   # → poses.npy
    ./export_poses.py poses.json -o cam_poses.npy
    ./export_poses.py poses.json --stride 9        # every 9th pose (dataset step)
"""

import json
import argparse
import numpy as np
from pathlib import Path


def quat_to_rotation_matrix(q: dict) -> np.ndarray:
    """
    Unreal quaternion {x,y,z,w} → 3×3 rotation matrix R in Unreal world space
    (v_world = R @ v_camera_local).
    """
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def ue_to_display_world(poses: list[dict]) -> np.ndarray:
    """
    Convert a list of Unreal poses to (N, 4, 4) SE3 matrices in the
    right-handed display world.

    Conversion applied per pose:
        position:  negate Y   (left-handed +Y-right  →  right-handed +Y-left)
        rotation:  H @ R @ H  (H = diag(1,−1,1) conjugation, keeps det = +1)
    """
    N = len(poses)
    out = np.eye(4, dtype=np.float64)[None].repeat(N, axis=0)  # (N, 4, 4)

    for i, p in enumerate(poses):
        t = p['translation']
        pos = np.array([t['x'], -t['y'], t['z']], dtype=np.float64)

        R = quat_to_rotation_matrix(p['rotation'])
        R[1, :] *= -1   # H @ R @ H  conjugation:
        R[:, 1] *= -1   #   negate row 1 then col 1 (det stays +1)

        out[i, :3, :3] = R
        out[i, :3,  3] = pos

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Export poses JSON to a NumPy SE3 array (right-handed display world)')
    parser.add_argument('poses', type=str,
                        help='Path to poses JSON (from parse_camera_poses.py)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .npy file (default: <poses_stem>.npy)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Keep every Nth pose (default: 1 = all poses)')

    args = parser.parse_args()

    with open(args.poses) as f:
        data = json.load(f)
    all_poses = data['transforms']
    print(f"Loaded {len(all_poses)} poses from {args.poses}")

    poses = all_poses[::args.stride]
    if args.stride > 1:
        print(f"Keeping every {args.stride}th pose → {len(poses)} poses")

    transforms = ue_to_display_world(poses)
    print(f"Output shape: {transforms.shape}  dtype: {transforms.dtype}")

    output = args.output or Path(args.poses).with_suffix('.npy')
    np.save(output, transforms)
    print(f"Saved → {output}")

    # Sanity check
    dets = np.linalg.det(transforms[:, :3, :3])
    pos = transforms[:, :3, 3]
    print(f"\nSanity check:")
    print(f"  det(R) range:  [{dets.min():.6f}, {dets.max():.6f}]  (should be ≈ 1.0)")
    print(f"  X range (cm):  [{pos[:, 0].min():.0f}, {pos[:, 0].max():.0f}]")
    print(f"  Y range (cm):  [{pos[:, 1].min():.0f}, {pos[:, 1].max():.0f}]")
    print(f"  Z range (cm):  [{pos[:, 2].min():.0f}, {pos[:, 2].max():.0f}]")


if __name__ == "__main__":
    main()
