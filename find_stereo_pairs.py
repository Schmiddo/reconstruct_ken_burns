#!/usr/bin/env python3
"""
Find inter-frame stereo pairs and coverage-maximising frame subsets from a pose file.

Each dataset frame has 4 intra-frame views (tl, tr, bl, br) with a 40-unit
baseline.  This tool additionally identifies inter-frame pairs and subsets
of frames whose spatial distribution is best for reconstruction.

Usage:
    ./find_stereo_pairs.py poses.json
    ./find_stereo_pairs.py poses.json --coverage --num-frames 50 --output frames.json
    ./find_stereo_pairs.py poses.json --min-baseline 100 --max-baseline 500
"""

import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class StereoPair:
    """A pair of frames suitable for stereo reconstruction."""
    frame1: int
    frame2: int
    distance: float  # Distance between camera positions
    angle_diff: float  # Angle between view directions (degrees)
    score: float  # Overall quality score


def load_poses(filepath: str) -> list[dict]:
    """Load poses from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data['transforms']


def load_dataset_metadata(data_dir: str, scene: str) -> dict[int, dict]:
    """Load metadata for all frames."""
    rgb_dir = Path(data_dir) / scene
    metadata = {}

    for meta_file in sorted(rgb_dir.glob("*-meta.json")):
        frame_idx = int(meta_file.stem.split('-')[0])
        with open(meta_file) as f:
            meta = json.load(f)
        metadata[frame_idx] = meta

    return metadata


def quaternion_to_forward(q: dict) -> np.ndarray:
    """Convert quaternion to forward direction vector."""
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    # Unreal Engine: +X is forward
    fx = 1 - 2 * (y*y + z*z)
    fy = 2 * (x*y + w*z)
    fz = 2 * (x*z - w*y)
    return np.array([fx, fy, fz])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in degrees."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def find_stereo_pairs(poses: list[dict],
                      min_baseline: float = 50.0,
                      max_baseline: float = 300.0,
                      max_angle: float = 30.0,
                      frame_indices: list[int] = None) -> list[StereoPair]:
    """
    Find pairs of frames suitable for stereo reconstruction.

    Args:
        poses: List of camera poses
        min_baseline: Minimum distance between cameras (cm)
        max_baseline: Maximum distance between cameras (cm)
        max_angle: Maximum angle between view directions (degrees)
        frame_indices: Optional list of frame indices to consider

    Returns:
        List of StereoPair objects sorted by score
    """
    if frame_indices is None:
        frame_indices = list(range(len(poses)))

    # Extract positions and forward vectors
    positions = {}
    forwards = {}

    for idx in frame_indices:
        if idx >= len(poses):
            continue
        pose = poses[idx]
        positions[idx] = np.array([
            pose['translation']['x'],
            pose['translation']['y'],
            pose['translation']['z']
        ])
        forwards[idx] = quaternion_to_forward(pose['rotation'])

    pairs = []
    indices = sorted(positions.keys())

    for i, idx1 in enumerate(indices):
        for idx2 in indices[i+1:]:
            # Compute distance
            distance = np.linalg.norm(positions[idx1] - positions[idx2])

            if distance < min_baseline or distance > max_baseline:
                continue

            # Compute angle difference
            angle = angle_between_vectors(forwards[idx1], forwards[idx2])

            if angle > max_angle:
                continue

            # Compute quality score
            # Prefer: moderate baseline, similar view direction
            # Score formula: higher is better
            baseline_score = 1.0 - abs(distance - (min_baseline + max_baseline) / 2) / (max_baseline - min_baseline)
            angle_score = 1.0 - angle / max_angle
            score = 0.5 * baseline_score + 0.5 * angle_score

            pairs.append(StereoPair(
                frame1=idx1,
                frame2=idx2,
                distance=distance,
                angle_diff=angle,
                score=score
            ))

    # Sort by score (descending)
    pairs.sort(key=lambda p: p.score, reverse=True)

    return pairs


def find_reconstruction_frames(poses: list[dict],
                               num_frames: int = 50,
                               min_distance: float = 100.0) -> list[int]:
    """
    Select frames for reconstruction using a coverage-based approach.

    Greedily selects frames that maximize spatial coverage.
    """
    positions = np.array([
        [p['translation']['x'], p['translation']['y'], p['translation']['z']]
        for p in poses
    ])

    n = len(positions)
    selected = [0]  # Start with first frame

    while len(selected) < num_frames:
        best_idx = -1
        best_min_dist = -1

        for i in range(n):
            if i in selected:
                continue

            # Minimum distance to any selected frame
            min_dist = min(np.linalg.norm(positions[i] - positions[j])
                          for j in selected)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx < 0 or best_min_dist < min_distance:
            break

        selected.append(best_idx)

    return sorted(selected)


def analyze_intra_frame_stereo(baseline: float = 40.0) -> dict:
    """
    Analyze the stereo setup within each frame (4 views per frame).

    Each frame has tl, tr, bl, br views with the specified baseline.
    """
    # View positions relative to center (assuming square arrangement)
    half_baseline = baseline / 2
    views = {
        'tl': np.array([-half_baseline, half_baseline, 0]),  # top-left
        'tr': np.array([half_baseline, half_baseline, 0]),   # top-right
        'bl': np.array([-half_baseline, -half_baseline, 0]), # bottom-left
        'br': np.array([half_baseline, -half_baseline, 0])   # bottom-right
    }

    # Compute baselines between view pairs
    pairs = {}
    for name1, pos1 in views.items():
        for name2, pos2 in views.items():
            if name1 < name2:
                pair_name = f"{name1}-{name2}"
                dist = np.linalg.norm(pos1 - pos2)
                pairs[pair_name] = dist

    return {
        'baseline': baseline,
        'view_positions': {k: v.tolist() for k, v in views.items()},
        'pair_baselines': pairs,
        'horizontal_baseline': baseline,  # tl-tr or bl-br
        'vertical_baseline': baseline,    # tl-bl or tr-br
        'diagonal_baseline': baseline * np.sqrt(2)  # tl-br or tr-bl
    }


def main():
    parser = argparse.ArgumentParser(
        description='Find stereo pairs for 3D reconstruction')
    parser.add_argument('poses', type=str,
                       help='Path to extracted poses JSON')
    parser.add_argument('--data', '-d', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--scene', '-s', type=str, default='western-flying',
                       help='Scene name')
    parser.add_argument('--min-baseline', type=float, default=50.0,
                       help='Minimum baseline between frames (cm)')
    parser.add_argument('--max-baseline', type=float, default=300.0,
                       help='Maximum baseline between frames (cm)')
    parser.add_argument('--max-angle', type=float, default=30.0,
                       help='Maximum angle between view directions (degrees)')
    parser.add_argument('--num-pairs', '-n', type=int, default=20,
                       help='Number of stereo pairs to output')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Find coverage-maximizing frames instead of pairs')
    parser.add_argument('--num-frames', type=int, default=50,
                       help='Number of frames for coverage mode')
    parser.add_argument('--intra-frame-baseline', type=float, default=40.0,
                       help='Baseline between views within a frame')

    args = parser.parse_args()

    # Load poses
    poses = load_poses(args.poses)
    print(f"Loaded {len(poses)} poses")

    # Analyze intra-frame stereo setup
    intra_stereo = analyze_intra_frame_stereo(args.intra_frame_baseline)
    print("\n" + "=" * 60)
    print("INTRA-FRAME STEREO (4 views per frame)")
    print("=" * 60)
    print(f"Baseline: {intra_stereo['baseline']} units")
    print(f"Horizontal pairs (tl-tr, bl-br): {intra_stereo['horizontal_baseline']:.1f}")
    print(f"Vertical pairs (tl-bl, tr-br): {intra_stereo['vertical_baseline']:.1f}")
    print(f"Diagonal pairs (tl-br, tr-bl): {intra_stereo['diagonal_baseline']:.1f}")

    if args.coverage:
        # Coverage mode: find well-distributed frames
        print("\n" + "=" * 60)
        print(f"COVERAGE-MAXIMIZING FRAMES (max {args.num_frames})")
        print("=" * 60)

        frames = find_reconstruction_frames(
            poses,
            num_frames=args.num_frames,
            min_distance=args.min_baseline
        )

        print(f"Selected {len(frames)} frames with min {args.min_baseline}cm spacing:")
        for i, idx in enumerate(frames[:20]):
            pos = poses[idx]['translation']
            print(f"  {i+1:3d}. Pose {idx:4d}: ({pos['x']:8.1f}, {pos['y']:8.1f}, {pos['z']:7.1f})")
        if len(frames) > 20:
            print(f"  ... and {len(frames) - 20} more")

        if args.output:
            output_data = {
                'mode': 'coverage',
                'num_frames': len(frames),
                'min_distance': args.min_baseline,
                'frame_indices': frames,
                'frames': [{
                    'index': idx,
                    'position': poses[idx]['translation'],
                    'rotation': poses[idx]['rotation']
                } for idx in frames]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to {args.output}")

    else:
        # Stereo pairs mode
        print("\n" + "=" * 60)
        print(f"INTER-FRAME STEREO PAIRS")
        print("=" * 60)
        print(f"Baseline range: {args.min_baseline} - {args.max_baseline} cm")
        print(f"Max angle difference: {args.max_angle} degrees")

        pairs = find_stereo_pairs(
            poses,
            min_baseline=args.min_baseline,
            max_baseline=args.max_baseline,
            max_angle=args.max_angle
        )

        print(f"\nFound {len(pairs)} valid stereo pairs")
        print(f"\nTop {min(args.num_pairs, len(pairs))} pairs by quality:")

        for i, pair in enumerate(pairs[:args.num_pairs]):
            pos1 = poses[pair.frame1]['translation']
            pos2 = poses[pair.frame2]['translation']
            print(f"  {i+1:3d}. Pose {pair.frame1:4d} <-> {pair.frame2:4d}: "
                  f"baseline={pair.distance:6.1f}cm, angle={pair.angle_diff:5.1f}°, "
                  f"score={pair.score:.3f}")

        if args.output:
            output_data = {
                'mode': 'stereo_pairs',
                'params': {
                    'min_baseline': args.min_baseline,
                    'max_baseline': args.max_baseline,
                    'max_angle': args.max_angle
                },
                'total_pairs': len(pairs),
                'pairs': [{
                    'frame1': p.frame1,
                    'frame2': p.frame2,
                    'distance': p.distance,
                    'angle_diff': p.angle_diff,
                    'score': p.score
                } for p in pairs[:args.num_pairs * 2]]  # Save more than displayed
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
