#!/usr/bin/env python3
"""
Map rendered dataset frames to their corresponding poses from a .sav recording.

The renderer subsamples the recorded pose stream at a fixed stride: frame N
(1-based) was rendered from pose (N-1)*step (0-based).  This script
auto-detects that stride from the ratio of total poses to total frames, or
accepts a manual override via --step.  The result is written as a JSON mapping
file consumed by reconstruct_3d.py.

Usage:
    ./match_poses.py poses.json --scene western-flying
    ./match_poses.py poses.json --scene western-flying --step 9
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


def build_mapping(poses: list[dict],
                  metadata: dict[int, dict],
                  step: int) -> list[dict]:
    """
    Build the canonical frame→pose mapping.

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
            'frame_index': frame_idx,         # 1-based, matches filename
            'pose_index': pose_idx,            # 0-based index into poses array
            'int_sample': meta.get('intSample'),
            'fov_deg': meta.get('fltFov', 90.0),
            'position': pose['translation'],   # {x, y, z} in cm
            'rotation': pose['rotation'],      # {x, y, z, w} quaternion
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

    if args.step is not None:
        step = args.step
        print(f"Using manual subsampling step: every {step}th pose")
    else:
        step = detect_step(len(poses), num_frames)
        print(f"Detected subsampling step: every {step}th pose")

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
            print(f"  frame {m['frame_index']:5d} → pose {m['pose_index']:5d}  "
                  f"({p['x']:8.1f}, {p['y']:8.1f}, {p['z']:7.1f})  fov={m['fov_deg']:.1f}°")

    output_data = {
        'scene': args.scene,
        'poses_file': args.poses,
        'num_poses': len(poses),
        'num_frames': num_frames,
        'step': step,
        'frames': mapping,
    }

    output = args.output or f"{args.scene}-mapping.json"
    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nMapping saved → {output}")


if __name__ == "__main__":
    main()
