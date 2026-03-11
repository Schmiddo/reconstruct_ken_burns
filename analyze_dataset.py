#!/usr/bin/env python3
"""
Analyse a Ken Burns dataset scene: trajectory statistics, spatial coverage,
FOV range, available depth/normal data, and keyframe selection for reconstruction.

Usage:
    ./analyze_dataset.py --poses poses.json --scene western-flying
    ./analyze_dataset.py --scene western-flying            # dataset stats only
    ./analyze_dataset.py --poses poses.json --keyframes    # print keyframe list
"""

import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetFrame:
    """A single frame from the dataset."""
    index: int  # 1-based index from filename
    sample: int  # intSample from meta.json
    fov: float   # fltFov from meta.json (always present; defaults to 90.0)
    has_rgb: bool
    has_depth: bool
    has_normal: bool


@dataclass
class PoseInfo:
    """Camera pose with additional analysis info."""
    index: int
    position: np.ndarray
    rotation: np.ndarray  # quaternion (x, y, z, w)
    fov: float | None = None


def load_poses(filepath: str) -> list[dict]:
    """Load poses from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data['transforms']


def load_dataset_metadata(data_dir: str, scene: str) -> list[DatasetFrame]:
    """Load metadata for all frames in a dataset scene."""
    rgb_dir = Path(data_dir) / scene
    depth_dir = Path(data_dir) / f"{scene}-depth"
    normal_dir = Path(data_dir) / f"{scene}-normal"

    frames = []

    # Find all meta.json files
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    meta_files = sorted(rgb_dir.glob("*-meta.json"))

    for meta_file in meta_files:
        # Parse frame index from filename (e.g., "00001-meta.json" -> 1)
        frame_idx = int(meta_file.stem.split('-')[0])

        # Load metadata
        with open(meta_file) as f:
            meta = json.load(f)

        # Check what data exists
        prefix = f"{frame_idx:05d}"
        has_rgb = (rgb_dir / f"{prefix}-tl-image.png").exists()
        has_depth = (depth_dir / f"{prefix}-tl-depth.exr").exists() if depth_dir.exists() else False
        has_normal = (normal_dir / f"{prefix}-tl-normal.exr").exists() if normal_dir.exists() else False

        frames.append(DatasetFrame(
            index=frame_idx,
            sample=meta.get('intSample', frame_idx),
            fov=meta.get('fltFov', 90.0),
            has_rgb=has_rgb,
            has_depth=has_depth,
            has_normal=has_normal
        ))

    return frames


def poses_to_numpy(poses: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert poses list to numpy arrays."""
    positions = np.array([
        [p['translation']['x'], p['translation']['y'], p['translation']['z']]
        for p in poses
    ])
    rotations = np.array([
        [p['rotation']['x'], p['rotation']['y'], p['rotation']['z'], p['rotation']['w']]
        for p in poses
    ])
    return positions, rotations


def compute_pose_distances(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between all poses."""
    # Use broadcasting to compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return distances


def compute_consecutive_distances(positions: np.ndarray) -> np.ndarray:
    """Compute distances between consecutive poses."""
    diff = np.diff(positions, axis=0)
    return np.linalg.norm(diff, axis=1)


def analyze_trajectory_coverage(positions: np.ndarray,
                                grid_size: float = 100.0) -> dict:
    """Analyze spatial coverage of the trajectory."""
    # Compute bounding box
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    extent = max_pos - min_pos

    # Divide into grid cells and count occupancy
    grid_dims = np.ceil(extent / grid_size).astype(int)
    grid_dims = np.maximum(grid_dims, 1)

    # Assign each position to a cell
    cell_indices = ((positions - min_pos) / grid_size).astype(int)
    cell_indices = np.minimum(cell_indices, grid_dims - 1)

    # Count unique cells
    unique_cells = set(map(tuple, cell_indices))
    total_cells = np.prod(grid_dims)

    return {
        'bounding_box': {
            'min': min_pos.tolist(),
            'max': max_pos.tolist(),
            'extent': extent.tolist()
        },
        'grid_size': grid_size,
        'grid_dims': grid_dims.tolist(),
        'occupied_cells': len(unique_cells),
        'total_cells': int(total_cells),
        'coverage_ratio': len(unique_cells) / total_cells
    }


def find_keyframes(positions: np.ndarray,
                   min_distance: float = 50.0,
                   max_frames: int = 100) -> list[int]:
    """
    Select keyframes for 3D reconstruction based on spatial distribution.

    Uses a greedy algorithm to select frames that are well-distributed.
    """
    n = len(positions)
    if n == 0:
        return []

    selected = [0]  # Start with first frame

    while len(selected) < max_frames:
        # Find the frame furthest from all selected frames
        best_idx = -1
        best_min_dist = -1

        for i in range(n):
            if i in selected:
                continue

            # Compute minimum distance to any selected frame
            min_dist = min(np.linalg.norm(positions[i] - positions[j])
                          for j in selected)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx < 0 or best_min_dist < min_distance:
            break

        selected.append(best_idx)

    return sorted(selected)


def analyze_view_overlap(positions: np.ndarray,
                        rotations: np.ndarray,
                        fov_degrees: float = 90.0) -> dict:
    """Analyze potential view overlap between frames."""
    n = len(positions)

    # Convert quaternions to forward vectors
    def quat_to_forward(q):
        x, y, z, w = q
        # Forward direction after rotation (Unreal: +X is forward)
        fx = 1 - 2 * (y*y + z*z)
        fy = 2 * (x*y + w*z)
        fz = 2 * (x*z - w*y)
        return np.array([fx, fy, fz])

    forwards = np.array([quat_to_forward(r) for r in rotations])

    # Compute view direction similarity (dot product)
    # Higher = more similar viewing direction
    view_similarity = forwards @ forwards.T

    # Compute distances
    distances = compute_pose_distances(positions)

    # Good stereo pairs: similar view direction, moderate distance
    # (not too close, not too far)
    good_baseline_min = 20.0  # cm
    good_baseline_max = 200.0  # cm

    good_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            sim = view_similarity[i, j]

            if good_baseline_min < dist < good_baseline_max and sim > 0.8:
                good_pairs.append((i, j, dist, sim))

    return {
        'total_frames': n,
        'good_stereo_pairs': len(good_pairs),
        'avg_view_similarity': float(np.mean(view_similarity)),
        'sample_pairs': good_pairs[:10] if good_pairs else []
    }


def print_summary(poses: list[dict],
                  dataset_frames: list[DatasetFrame] | None = None):
    """Print analysis summary."""
    positions, rotations = poses_to_numpy(poses)

    print("=" * 60)
    print("TRAJECTORY ANALYSIS")
    print("=" * 60)

    print(f"\nTotal poses: {len(poses)}")

    # Position statistics
    print(f"\nPosition ranges (cm):")
    print(f"  X: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f}")
    print(f"  Y: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f}")
    print(f"  Z: {positions[:, 2].min():.1f} to {positions[:, 2].max():.1f}")

    # Consecutive distances
    consec_dist = compute_consecutive_distances(positions)
    print(f"\nConsecutive frame distances (cm):")
    print(f"  Min: {consec_dist.min():.2f}")
    print(f"  Max: {consec_dist.max():.2f}")
    print(f"  Mean: {consec_dist.mean():.2f}")
    print(f"  Median: {np.median(consec_dist):.2f}")

    # Coverage analysis
    coverage = analyze_trajectory_coverage(positions)
    print(f"\nSpatial coverage (100cm grid):")
    print(f"  Bounding box: {coverage['bounding_box']['extent']}")
    print(f"  Occupied cells: {coverage['occupied_cells']}/{coverage['total_cells']}")
    print(f"  Coverage ratio: {coverage['coverage_ratio']:.1%}")

    # View overlap
    overlap = analyze_view_overlap(positions, rotations)
    print(f"\nView overlap analysis:")
    print(f"  Good stereo pairs (20-200cm, similar direction): {overlap['good_stereo_pairs']}")

    # Keyframe selection
    keyframes = find_keyframes(positions, min_distance=100.0, max_frames=50)
    print(f"\nSuggested keyframes (min 100cm apart): {len(keyframes)} frames")
    print(f"  Indices: {keyframes[:10]}..." if len(keyframes) > 10 else f"  Indices: {keyframes}")

    if dataset_frames:
        print(f"\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        print(f"\nDataset frames: {len(dataset_frames)}")
        print(f"Pose to frame ratio: {len(poses) / len(dataset_frames):.2f}")

        fovs = [f.fov for f in dataset_frames]
        print(f"\nFOV range: {min(fovs):.1f} to {max(fovs):.1f} degrees")

        has_depth = sum(1 for f in dataset_frames if f.has_depth)
        has_normal = sum(1 for f in dataset_frames if f.has_normal)
        print(f"Frames with depth: {has_depth}")
        print(f"Frames with normals: {has_normal}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Ken Burns dataset for 3D reconstruction')
    parser.add_argument('--poses', '-p', type=str,
                       help='Path to extracted poses JSON')
    parser.add_argument('--data', '-d', type=str,
                       default='data',
                       help='Path to dataset directory')
    parser.add_argument('--scene', '-s', type=str,
                       default='western-flying',
                       help='Scene name (e.g., western-flying)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for analysis results')
    parser.add_argument('--keyframes', '-k', action='store_true',
                       help='Output keyframe indices for reconstruction')
    parser.add_argument('--min-distance', type=float, default=100.0,
                       help='Minimum distance between keyframes (cm)')
    parser.add_argument('--max-keyframes', type=int, default=100,
                       help='Maximum number of keyframes to select')

    args = parser.parse_args()

    # Load poses if provided
    poses = None
    if args.poses:
        poses = load_poses(args.poses)
        print(f"Loaded {len(poses)} poses from {args.poses}")

    # Load dataset metadata
    dataset_frames = None
    try:
        dataset_frames = load_dataset_metadata(args.data, args.scene)
        print(f"Loaded {len(dataset_frames)} dataset frames from {args.scene}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    if poses:
        print_summary(poses, dataset_frames)

        if args.keyframes:
            positions, _ = poses_to_numpy(poses)
            keyframes = find_keyframes(
                positions,
                min_distance=args.min_distance,
                max_frames=args.max_keyframes
            )
            print(f"\n" + "=" * 60)
            print(f"KEYFRAMES ({len(keyframes)} selected)")
            print("=" * 60)
            for i, kf in enumerate(keyframes):
                p = poses[kf]['translation']
                print(f"  {i+1:3d}. Frame {kf:4d}: ({p['x']:8.1f}, {p['y']:8.1f}, {p['z']:7.1f})")

        if args.output:
            positions, rotations = poses_to_numpy(poses)
            results = {
                'total_poses': len(poses),
                'coverage': analyze_trajectory_coverage(positions),
                'view_overlap': analyze_view_overlap(positions, rotations),
                'keyframes': find_keyframes(positions, args.min_distance, args.max_keyframes)
            }
            if dataset_frames:
                results['dataset'] = {
                    'frames': len(dataset_frames),
                    'fov_range': [min(f.fov for f in dataset_frames),
                                  max(f.fov for f in dataset_frames)]
                }

            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif dataset_frames:
        print(f"\nDataset frames: {len(dataset_frames)}")
        fovs = [f.fov for f in dataset_frames]
        print(f"FOV range: {min(fovs):.1f} to {max(fovs):.1f}")


if __name__ == "__main__":
    main()
