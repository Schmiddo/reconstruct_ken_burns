#!/usr/bin/env python3
"""
Visualise camera trajectories from pose JSON files using matplotlib.

Usage:
    ./visualize_poses.py poses.json
    ./visualize_poses.py poses.json --subsample 10 --save trajectory.png
    ./visualize_poses.py a.json b.json --compare
    ./visualize_poses.py poses.json --time-plot
"""

import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # registers the '3d' projection


def quaternion_to_direction(qx, qy, qz, qw):
    """
    Convert a quaternion to a forward direction vector.
    In Unreal Engine, forward is +X axis.
    """
    # Rotate the forward vector (1, 0, 0) by the quaternion
    # Using quaternion rotation formula: v' = q * v * q^-1
    # Simplified for unit vector (1, 0, 0):
    fx = 1 - 2 * (qy * qy + qz * qz)
    fy = 2 * (qx * qy + qw * qz)
    fz = 2 * (qx * qz - qw * qy)
    return np.array([fx, fy, fz])


def load_poses(filepath):
    """Load poses from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data['transforms']


def visualize_trajectory(poses, title="Camera Trajectory", subsample=1, show_orientation=True):
    """
    Visualize camera trajectory in 3D.

    Args:
        poses: List of transform dictionaries
        title: Plot title
        subsample: Only plot every Nth pose (for performance)
        show_orientation: Whether to show camera orientation arrows
    """
    # Extract positions
    positions = np.array([
        [p['translation']['x'], p['translation']['y'], p['translation']['z']]
        for p in poses
    ])

    # Subsample for performance
    positions = positions[::subsample]
    poses_subsampled = poses[::subsample]

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create color gradient based on time (frame index)
    n_points = len(positions)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', alpha=0.3, linewidth=0.5)

    # Plot points with color gradient
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=np.arange(n_points), cmap='viridis', s=2, alpha=0.6)

    # Mark start and end points
    ax.scatter(*positions[0], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(*positions[-1], color='red', s=100, marker='s', label='End', zorder=5)

    # Show orientation arrows (subsampled more heavily)
    if show_orientation:
        arrow_subsample = max(1, len(poses_subsampled) // 50)  # Show ~50 arrows max
        arrow_length = np.ptp(positions) * 0.02  # 2% of total range

        for i in range(0, len(poses_subsampled), arrow_subsample):
            p = poses_subsampled[i]
            pos = positions[i]
            rot = p['rotation']
            direction = quaternion_to_direction(rot['x'], rot['y'], rot['z'], rot['w'])

            ax.quiver(pos[0], pos[1], pos[2],
                     direction[0] * arrow_length,
                     direction[1] * arrow_length,
                     direction[2] * arrow_length,
                     color='red', alpha=0.5, arrow_length_ratio=0.3)

    # Labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'{title}\n({len(poses)} frames, showing every {subsample})')

    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Frame')

    ax.legend()

    # Equal aspect ratio
    max_range = np.ptp(positions, axis=0).max() / 2
    mid = positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    return fig, ax


def visualize_comparison(poses1, poses2, label1="Trajectory 1", label2="Trajectory 2", subsample=1):
    """
    Visualize two trajectories overlaid for comparison.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for poses, label, color in [(poses1, label1, 'blue'), (poses2, label2, 'orange')]:
        positions = np.array([
            [p['translation']['x'], p['translation']['y'], p['translation']['z']]
            for p in poses
        ])[::subsample]

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                color=color, alpha=0.6, linewidth=1, label=label)
        ax.scatter(*positions[0], color=color, s=100, marker='o', zorder=5)
        ax.scatter(*positions[-1], color=color, s=100, marker='s', zorder=5)

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('Trajectory Comparison')
    ax.legend()

    return fig, ax


def plot_position_over_time(poses, title="Position over Time"):
    """Plot X, Y, Z positions over time as 2D line plots."""
    positions = np.array([
        [p['translation']['x'], p['translation']['y'], p['translation']['z']]
        for p in poses
    ])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    frames = np.arange(len(positions))

    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']

    for ax, label, color, i in zip(axes, labels, colors, range(3)):
        ax.plot(frames, positions[:, i], color=color, linewidth=0.5)
        ax.set_ylabel(f'{label} (cm)')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame')
    axes[0].set_title(title)

    plt.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description='Visualize camera poses from JSON files')
    parser.add_argument('files', nargs='+', help='JSON pose file(s) to visualize')
    parser.add_argument('--subsample', '-s', type=int, default=1,
                       help='Subsample factor (plot every Nth frame)')
    parser.add_argument('--no-orientation', action='store_true',
                       help='Hide camera orientation arrows')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple trajectories in one plot')
    parser.add_argument('--time-plot', action='store_true',
                       help='Show position vs time plot')
    parser.add_argument('--save', '-o', type=str,
                       help='Save figure to file instead of displaying')

    args = parser.parse_args()

    if args.compare and len(args.files) >= 2:
        # Compare mode: overlay trajectories
        poses_list = [load_poses(f) for f in args.files]
        labels = [f.replace('-poses.json', '').replace('.json', '') for f in args.files]

        fig, ax = visualize_comparison(poses_list[0], poses_list[1],
                                       labels[0], labels[1],
                                       subsample=args.subsample)
    else:
        # Single or multiple separate plots
        for filepath in args.files:
            poses = load_poses(filepath)
            title = filepath.replace('-poses.json', '').replace('.json', '')

            print(f"Loaded {len(poses)} poses from {filepath}")

            # 3D trajectory
            fig, ax = visualize_trajectory(
                poses,
                title=title,
                subsample=args.subsample,
                show_orientation=not args.no_orientation
            )

            if args.time_plot:
                plot_position_over_time(poses, title=f"{title} - Position vs Time")

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
