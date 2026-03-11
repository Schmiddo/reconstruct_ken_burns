#!/usr/bin/env python3
"""
Map rendered dataset frames to their corresponding poses from a .sav recording.

The renderer subsamples the recorded pose stream at a fixed stride: frame N
(1-based) was rendered from pose (N-1)*step (0-based).  This script
auto-detects that stride from the ratio of total poses to total frames, or
accepts a manual override via --step.  The result is written as a JSON mapping
file consumed by reconstruct_3d.py.

By default the stride is rounded to the nearest integer and poses are snapped.
With --interpolate the exact fractional stride is used and each frame's pose is
SLERP'd between the two bracketing recorded poses.  This removes the drift that
accumulates when the true stride (e.g. 9.0057) is not a whole number.

Usage:
    ./match_poses.py poses.json --scene western-flying
    ./match_poses.py poses.json --scene western-flying --step 9
    ./match_poses.py poses.json --scene western-flying --interpolate
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_poses(filepath: str) -> list[dict]:
    with open(filepath) as f:
        data = json.load(f)
    return data['transforms']


def load_dataset_metadata(data_dir: str, scene: str) -> dict[int, dict]:
    """Load intSample and fltFov from all per-frame meta.json files."""
    rgb_dir = Path(data_dir) / scene
    metadata = {}
    for meta_file in sorted(rgb_dir.glob("*-meta.json")):
        frame_idx = int(meta_file.stem.split('-')[0])
        with open(meta_file) as f:
            metadata[frame_idx] = json.load(f)
    return metadata


def detect_step(num_poses: int, num_frames: int) -> int:
    """Detect the subsampling step from the ratio."""
    ratio = num_poses / num_frames
    step = int(round(ratio))
    # Sanity-check: step*num_frames should be close to num_poses
    assert abs(step * num_frames - num_poses) <= step, \
        f"Unexpected ratio {ratio:.3f}: {num_poses} poses / {num_frames} frames"
    return step


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between unit quaternions q0 and q1.
    Both are [x, y, z, w] arrays.  t=0 returns q0, t=1 returns q1.
    """
    dot = float(np.dot(q0, q1))
    # Choose the shorter arc
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    # Fall back to linear interpolation for nearly-parallel quaternions
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    theta  = theta0 * t
    sin_theta0 = np.sin(theta0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    return s0 * q0 + s1 * q1


def interpolate_pose(poses: list[dict], t: float) -> tuple[dict, dict]:
    """
    Return (position, rotation) for fractional pose index t, SLERP'd between
    the two bracketing recorded poses.
    """
    i0 = min(int(t), len(poses) - 2)
    i1 = i0 + 1
    frac = t - i0

    p0, p1 = poses[i0], poses[i1]

    # Linear interpolation for translation
    position = {
        'x': p0['translation']['x'] + frac * (p1['translation']['x'] - p0['translation']['x']),
        'y': p0['translation']['y'] + frac * (p1['translation']['y'] - p0['translation']['y']),
        'z': p0['translation']['z'] + frac * (p1['translation']['z'] - p0['translation']['z']),
    }

    # SLERP for rotation
    r0 = p0['rotation']
    r1 = p1['rotation']
    q0 = np.array([r0['x'], r0['y'], r0['z'], r0['w']], dtype=np.float64)
    q1 = np.array([r1['x'], r1['y'], r1['z'], r1['w']], dtype=np.float64)
    q = slerp(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1), frac)
    rotation = {'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2]), 'w': float(q[3])}

    return position, rotation


def build_mapping(poses: list[dict],
                  metadata: dict[int, dict],
                  step: int) -> list[dict]:
    """
    Build the frame→pose mapping by snapping to the nearest integer pose index.

    Frame index is 1-based (matches filenames like 00001-*.png).
    Pose index is 0-based into the poses list: pose_idx = (frame_idx - 1) * step.
    """
    mapping = []
    for frame_idx in sorted(metadata.keys()):
        pose_idx = (frame_idx - 1) * step
        if pose_idx >= len(poses):
            continue
        pose = poses[pose_idx]
        meta = metadata[frame_idx]
        mapping.append({
            'frame_index': frame_idx,
            'pose_index':  pose_idx,
            'int_sample':  meta.get('intSample'),
            'fov_deg':     meta.get('fltFov', 90.0),
            'position':    pose['translation'],
            'rotation':    pose['rotation'],
        })
    return mapping


def build_mapping_interpolated(poses: list[dict],
                               metadata: dict[int, dict],
                               exact_step: float) -> list[dict]:
    """
    Build the frame→pose mapping using SLERP at the fractional pose index.

    Uses the exact (non-rounded) step so accumulated drift across the sequence
    is eliminated.  The stored pose_index is the floor of the fractional index.
    """
    mapping = []
    for frame_idx in sorted(metadata.keys()):
        t = (frame_idx - 1) * exact_step
        i0 = min(int(t), len(poses) - 2)
        if i0 >= len(poses):
            continue
        meta = metadata[frame_idx]
        position, rotation = interpolate_pose(poses, t)
        mapping.append({
            'frame_index': frame_idx,
            'pose_index':  i0,          # floor; fractional part was SLERP'd
            'pose_t':      round(t, 6), # exact fractional index (for reference)
            'int_sample':  meta.get('intSample'),
            'fov_deg':     meta.get('fltFov', 90.0),
            'position':    position,
            'rotation':    rotation,
        })
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description='Match dataset frames to extracted poses, output mapping JSON')
    parser.add_argument('poses', type=str,
                        help='Path to extracted poses JSON (from parse_camera_poses.py)')
    parser.add_argument('--data', '-d', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--scene', '-s', type=str, default='western-flying',
                        help='Scene name (subdirectory under --data)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output mapping JSON (default: <scene>-mapping.json)')
    parser.add_argument('--step', type=int, default=None,
                        help='Override auto-detected subsampling step')
    parser.add_argument('--interpolate', action='store_true',
                        help='SLERP between poses using the exact fractional step '
                             'instead of snapping to the nearest integer index')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress analysis output')

    args = parser.parse_args()

    poses = load_poses(args.poses)
    print(f"Loaded {len(poses)} poses from {args.poses}")

    rgb_dir = Path(args.data) / args.scene
    if not rgb_dir.exists():
        raise SystemExit(f"Scene directory not found: {rgb_dir}")

    metadata = load_dataset_metadata(args.data, args.scene)
    num_frames = len(metadata)
    print(f"Found {num_frames} frames in {rgb_dir}")

    exact_step = len(poses) / num_frames

    if args.step is not None:
        step = args.step
        print(f"Using manual subsampling step: {step} (exact ratio: {exact_step:.5f})")
    else:
        step = detect_step(len(poses), num_frames)
        print(f"Detected subsampling step: {step} (exact ratio: {exact_step:.5f})")

    if args.interpolate:
        print(f"Mode: SLERP interpolation at fractional step {exact_step:.5f}")
        mapping = build_mapping_interpolated(poses, metadata, exact_step)
    else:
        print(f"Mode: integer snap (step={step})")
        mapping = build_mapping(poses, metadata, step)

    fovs = [m['fov_deg'] for m in mapping]
    positions = np.array([[m['position']['x'], m['position']['y'], m['position']['z']]
                          for m in mapping])
    consec_dist = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    if not args.quiet:
        print(f"\nFOV range:  {min(fovs):.1f}° – {max(fovs):.1f}°")
        print(f"Positions:  X [{positions[:,0].min():.0f}, {positions[:,0].max():.0f}]  "
              f"Y [{positions[:,1].min():.0f}, {positions[:,1].max():.0f}]  "
              f"Z [{positions[:,2].min():.0f}, {positions[:,2].max():.0f}]  (cm)")
        print(f"Frame dist: avg {consec_dist.mean():.1f} cm, "
              f"min {consec_dist.min():.1f}, max {consec_dist.max():.1f}")

        print("\nFirst 5 frames:")
        for m in mapping[:5]:
            p = m['position']
            t_str = f"  t={m['pose_t']:.3f}" if 'pose_t' in m else ''
            print(f"  frame {m['frame_index']:5d} → pose {m['pose_index']:5d}{t_str}  "
                  f"({p['x']:8.1f}, {p['y']:8.1f}, {p['z']:7.1f})  fov={m['fov_deg']:.1f}°")

    output_data = {
        'scene':       args.scene,
        'poses_file':  args.poses,
        'num_poses':   len(poses),
        'num_frames':  num_frames,
        'step':        step,
        'exact_step':  exact_step,
        'interpolated': args.interpolate,
        'frames':      mapping,
    }

    output = args.output or f"{args.scene}-mapping.json"
    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nMapping saved → {output}")


if __name__ == "__main__":
    main()
