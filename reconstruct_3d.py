#!/usr/bin/env python3
"""
3D reconstruction from depth maps + camera poses, visualised with Rerun.

Reads the frame→pose mapping produced by match_poses.py, back-projects each
pixel into world space using the EXR depth maps and FOV, and streams the result
into Rerun frame-by-frame as it is processed.

Usage:
    ./reconstruct_3d.py <scene>-mapping.json
    ./reconstruct_3d.py <scene>-mapping.json --frames 100 --subsample 4
    ./reconstruct_3d.py <scene>-mapping.json --rrd out.rrd --no-viewer
    ./reconstruct_3d.py <scene>-mapping.json --frame-list 1,50,200 --all-views
"""

import json
import argparse
import numpy as np
from pathlib import Path

import OpenEXR
import Imath
import cv2
import rerun as rr
import rerun.blueprint as rrb


VIEWS = ['tl', 'tr', 'bl', 'br']


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mapping(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_depth_exr(path: Path) -> np.ndarray:
    """Return a float32 H×W depth array (in scene units / cm)."""
    f = OpenEXR.InputFile(str(path))
    dw = f.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    raw = f.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, dtype=np.float32).reshape(H, W)


def load_rgb(path: Path) -> np.ndarray:
    """Return a uint8 H×W×3 RGB array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def quat_to_rotation_matrix(q: dict) -> np.ndarray:
    """
    Unreal quaternion {x,y,z,w} → 3×3 rotation matrix R where
    v_world = R @ v_camera_local.

    Unreal axes: +X forward, +Y right, +Z up (left-handed world).
    """
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# Maps display-world camera axes to Rerun/OpenCV camera axes.
#
# After the H@R@H conversion the camera lives in a right-handed world where
# local +X = forward, local +Y = world LEFT (not right!), local +Z = world up.
# Rerun uses the OpenCV convention: +X = image-right, +Y = image-down, +Z = forward.
#
#   Rerun  +X (image right) = display-world  −Y  (camera local −Y = visual right)
#   Rerun  +Y (image down)  = display-world  −Z  (camera local −Z = visual down)
#   Rerun  +Z (forward)     = display-world  +X  (camera local +X = forward)
_UE_TO_RR_CAM = np.array([
    [0., -1.,  0.],
    [0.,  0., -1.],
    [1.,  0.,  0.],
], dtype=np.float64)

# Signed baseline offsets in camera-local (right, up) for each view.
# tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right.
VIEW_OFFSETS: dict[str, tuple[float, float]] = {
    'tl': (-0.5, +0.5),
    'tr': (+0.5, +0.5),
    'bl': (-0.5, -0.5),
    'br': (+0.5, -0.5),
}


def view_camera_pos(center_pos: np.ndarray, R: np.ndarray,
                    view: str, hbaseline: float, vbaseline: float) -> np.ndarray:
    """World-space position of one view's camera, offset from the rig centre."""
    hr, vr = VIEW_OFFSETS[view]
    # In the right-handed display world, camera local +Y points world-LEFT,
    # so visual right is −local-Y = R @ [0, −1, 0].
    right_world = -(R @ [0., 1., 0.])
    up_world    =   R @ [0., 0., 1.]
    return center_pos + hr * hbaseline * right_world + vr * vbaseline * up_world


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Numerically stable rotation matrix → unit quaternion (xyzw order).
    Uses Shepperd's method.  Input must have det ≈ +1.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def backproject(depth: np.ndarray,
                rgb: np.ndarray,
                cam_pos: np.ndarray,
                R: np.ndarray,
                fov_deg: float,
                max_depth: float,
                subsample: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project valid depth pixels into world-space 3-D points.

    cam_pos : world-space camera position in the right-handed display world
              (already offset for this view)
    R       : 3×3 rotation matrix in the right-handed display world
              (post H@R@H conversion; v_world = R @ v_cam_local)

    Depth convention: linear scene depth along camera +X (forward),
    NOT radial distance.

    fov_deg : full horizontal opening angle in degrees (Unreal fltFov).
              Focal length = (W/2) / tan(fov_deg/2).

    Returns
    -------
    pts : (N, 3) float32   world-space XYZ in cm
    col : (N, 3) uint8     RGB colours
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    # fov_deg is the full opening angle, so half-angle goes into tan.
    f = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)

    # World-space camera axes in the right-handed display world.
    # local +Y is world-LEFT, so visual right is −local-Y.
    forward = R @ [1.,  0., 0.]
    right   = R @ [0., -1., 0.]
    up      = R @ [0.,  0., 1.]

    # Subsampled pixel grid
    ui = np.arange(0, W, subsample, dtype=np.int32)
    vi = np.arange(0, H, subsample, dtype=np.int32)
    uu, vv = np.meshgrid(ui, vi)
    ui, vi = uu.ravel(), vv.ravel()

    d = depth[vi, ui].astype(np.float64)
    mask = (d > 0) & (d < max_depth)
    d, ui, vi = d[mask], ui[mask], vi[mask]

    du = (ui - cx) / f   # normalised right offset
    dv = (vi - cy) / f   # normalised down  offset

    # P = cam_pos + d*(forward + du*right - dv*up)
    pts = (cam_pos[None, :]
           + d[:, None] * (forward[None, :]
                           + du[:, None] * right[None, :]
                           - dv[:, None] * up[None, :]))
    col = rgb[vi, ui]
    return pts.astype(np.float32), col


# ---------------------------------------------------------------------------
# PLY output
# ---------------------------------------------------------------------------

def save_ply(path: str, pts: np.ndarray, col: np.ndarray) -> None:
    n = len(pts)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                      ('r','u1'),('g','u1'),('b','u1')])
    data = np.empty(n, dtype=dtype)
    data['x'] = pts[:, 0]; data['y'] = pts[:, 1]; data['z'] = pts[:, 2]
    data['r'] = col[:, 0]; data['g'] = col[:, 1]; data['b'] = col[:, 2]
    with open(path, 'wb') as fh:
        fh.write(header.encode())
        data.tofile(fh)
    print(f"Saved {n:,} points → {path}")


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

def log_camera(cam_pos: np.ndarray, R_world_ue: np.ndarray,
               entity: str, fov_deg: float,
               rgb: np.ndarray | None, arrow_length: float) -> None:
    """
    Log a camera as a forward-pointing arrow.
    If rgb is provided, also log a Pinhole + image so the image projects into
    the 3-D view correctly.
    """
    # Camera forward in the right-handed display world: local +X.
    forward_world = R_world_ue @ [1., 0., 0.]

    rr.log(entity, rr.Arrows3D(
        origins=[cam_pos],
        vectors=[forward_world * arrow_length],
        colors=[[255, 200, 50]],
        radii=arrow_length * 0.02,
    ), static=True)

    if rgb is not None:
        H, W = rgb.shape[:2]
        f = float((W / 2.0) / np.tan(np.radians(fov_deg) / 2.0))
        R_world_rr = R_world_ue @ _UE_TO_RR_CAM.T
        q_xyzw = rotation_matrix_to_quat_xyzw(R_world_rr)
        pinhole = f"{entity}/pinhole"
        rr.log(pinhole, rr.Transform3D(
            translation=cam_pos.tolist(),
            rotation=rr.Quaternion(xyzw=q_xyzw),
        ), static=True)
        rr.log(pinhole, rr.Pinhole(focal_length=[f, f], width=W, height=H), static=True)
        rr.log(f"{pinhole}/image", rr.Image(rgb), static=True)


def setup_rerun(scene: str) -> None:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Reconstruction", origin="world"),
            rrb.Spatial2DView(name="Camera image", origin="world/frames"),
            column_shares=[3, 1],
        ),
        # collapse_panels=False keeps the entity tree and blueprint panels visible
    )
    rr.init(f"kenburns/{scene}", spawn=True, default_blueprint=blueprint)
    # Right-handed, Z-up: matches our display world (+X forward, +Y left, +Z up).
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='3-D reconstruction viewer (Rerun)')
    parser.add_argument('mapping',
                        help='frame_pose_mapping.json from match_poses.py')
    parser.add_argument('--data', '-d', default='data',
                        help='Dataset root directory')
    parser.add_argument('--scene', '-s', default='western-flying')
    parser.add_argument('--view', default='tl', choices=VIEWS,
                        help='Which of the 4 camera views to use (default: tl)')
    parser.add_argument('--all-views', action='store_true',
                        help='Use all 4 views (overrides --view)')
    parser.add_argument('--frames', '-n', type=int, default=30,
                        help='Number of evenly-spaced frames to load')
    parser.add_argument('--subsample', type=int, default=4,
                        help='Take every Nth pixel per axis for the point cloud')
    parser.add_argument('--max-depth', type=float, default=50_000.0,
                        help='Depth threshold for sky/invalid pixels (cm)')
    parser.add_argument('--hbaseline', type=float, default=40.0,
                        help='Horizontal baseline between left/right view pairs (cm)')
    parser.add_argument('--vbaseline', type=float, default=40.0,
                        help='Vertical baseline between top/bottom view pairs (cm)')
    parser.add_argument('--arrow-length', type=float, default=500.0,
                        help='Length of the camera-orientation arrow in cm (default: 500)')
    parser.add_argument('--log-images', action='store_true',
                        help='Also log RGB images to Rerun (increases data size)')
    parser.add_argument('--output', '-o', default=None,
                        help='Also write a binary PLY file')
    parser.add_argument('--rrd', default=None,
                        help='Save Rerun recording to this .rrd file')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Do not spawn the Rerun viewer (useful with --rrd)')
    parser.add_argument('--frame-list',
                        help='Comma-separated frame indices to use instead of --frames')
    parser.add_argument('--poses', default=None,
                        help='Full poses JSON (from parse_camera_poses.py) — draws the '
                             'complete camera path and sparse orientation arrows')
    parser.add_argument('--pose-arrows', type=int, default=200,
                        help='Number of evenly-spaced arrows to draw along the full path '
                             '(default: 200)')
    args = parser.parse_args()

    mapping_data = load_mapping(args.mapping)
    all_frames = mapping_data['frames']
    scene = mapping_data.get('scene', args.scene)
    print(f"Mapping: {len(all_frames)} frames  scene={scene}")

    # Select frames
    if args.frame_list:
        wanted = {int(x) for x in args.frame_list.split(',')}
        frames = [f for f in all_frames if f['frame_index'] in wanted]
    else:
        step = max(1, len(all_frames) // args.frames)
        frames = all_frames[::step][:args.frames]

    views = VIEWS if args.all_views else [args.view]
    print(f"Frames: {len(frames)}, views: {views}, subsample: {args.subsample}×{args.subsample}")

    rgb_dir   = Path(args.data) / scene
    depth_dir = Path(args.data) / f"{scene}-depth"

    # Initialise Rerun
    if not args.no_viewer:
        setup_rerun(scene)
    else:
        rr.init(f"kenburns/{scene}")
    if args.rrd:
        rr.save(args.rrd)
        print(f"Recording → {args.rrd}")

    # Process frames — all logged as static, entity paths grouped by frame index
    ply_pts, ply_col = [], []
    path_positions: list[list[float]] = []

    for frame in frames:
        fidx = frame['frame_index']

        # Read intrinsics directly from the per-frame meta.json.
        # fltFov is the full horizontal opening angle in degrees.
        meta_path = rgb_dir / f"{fidx:05d}-meta.json"
        with open(meta_path) as fh:
            meta = json.load(fh)
        fov = meta['fltFov']

        R = quat_to_rotation_matrix(frame['rotation'])
        center = np.array([frame['position']['x'],
                           frame['position']['y'],
                           frame['position']['z']], dtype=np.float64)
        # Convert Unreal left-handed world (+Y right) → right-handed (+Y left).
        # The correct rotation conversion is the conjugation H @ R @ H where
        # H = diag(1,-1,1): negate row 1 AND col 1 (the double-negation on R[1,1]
        # cancels out, leaving it unchanged).  This keeps det(R)=+1.
        center[1] *= -1
        R[1, :] *= -1
        R[:, 1] *= -1
        path_positions.append(center.tolist())

        frame_prefix = f"world/frames/{fidx:05d}"

        # One arrow per frame at the rig centre
        forward_world = R @ [1., 0., 0.]
        rr.log(f"{frame_prefix}/camera", rr.Arrows3D(
            origins=[center],
            vectors=[forward_world * args.arrow_length],
            colors=[[255, 200, 50]],
            radii=args.arrow_length * 0.02,
        ), static=True)

        for view in views:
            fname      = f"{fidx:05d}-{view}"
            depth_path = depth_dir / f"{fname}-depth.exr"
            rgb_path   = rgb_dir   / f"{fname}-image.png"

            if not depth_path.exists() or not rgb_path.exists():
                print(f"  [skip] {fname}")
                continue

            depth = load_depth_exr(depth_path)
            rgb   = load_rgb(rgb_path)

            cam_pos = view_camera_pos(center, R, view,
                                      args.hbaseline, args.vbaseline)

            pts, col = backproject(depth, rgb,
                                   cam_pos=cam_pos,
                                   R=R,
                                   fov_deg=fov,
                                   max_depth=args.max_depth,
                                   subsample=args.subsample)

            rr.log(f"{frame_prefix}/points/{view}",
                   rr.Points3D(pts, colors=col, radii=2.0), static=True)

            if args.log_images:
                log_camera(cam_pos, R,
                           entity=f"{frame_prefix}/cameras/{view}",
                           fov_deg=fov,
                           rgb=rgb,
                           arrow_length=args.arrow_length)

            print(f"  frame {fidx:5d} [{view}]: {len(pts):>7,} pts  "
                  f"pos=({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f})")

            if args.output:
                ply_pts.append(pts)
                ply_col.append(col)

    # Log camera path once, after all frames are processed
    if len(path_positions) > 1:
        rr.log("world/camera_path", rr.LineStrips3D(
            [np.array(path_positions, dtype=np.float32)],
            colors=[[255, 80, 80]], radii=5.0,
        ), static=True)

    if args.output and ply_pts:
        save_ply(args.output, np.concatenate(ply_pts), np.concatenate(ply_col))

    if args.poses:
        with open(args.poses) as fh:
            all_poses = json.load(fh)['transforms']
        print(f"Logging full pose path ({len(all_poses)} poses) …")

        all_pos = np.array([[p['translation']['x'],
                            -p['translation']['y'],   # negate Y: left-handed → right-handed
                             p['translation']['z']] for p in all_poses],
                           dtype=np.float32)

        # Log line + arrows on a "pose" timeline so the path unrolls when scrubbing.
        # We update at each arrow step to keep data volume bounded.
        arrow_step = max(1, len(all_poses) // args.pose_arrows)
        arrow_origins: list[np.ndarray] = []
        arrow_vectors: list[np.ndarray] = []
        next_arrow = 0  # pose index of the next arrow

        for pose_idx in range(len(all_poses)):
            rr.set_time("pose", sequence=pose_idx)

            # Growing line: update at every arrow step and at the last pose
            is_arrow = (pose_idx == next_arrow)
            is_last  = (pose_idx == len(all_poses) - 1)
            if is_arrow or is_last:
                rr.log("world/full_path", rr.LineStrips3D(
                    [all_pos[:pose_idx + 1]],
                    colors=[[120, 180, 255]], radii=3.0,
                ))

            if is_arrow:
                p = all_poses[pose_idx]
                arrow_origins.append(np.array([p['translation']['x'],
                                              -p['translation']['y'],   # negate Y
                                               p['translation']['z']],
                                              dtype=np.float32))
                R_p = quat_to_rotation_matrix(p['rotation'])
                R_p[1, :] *= -1   # H @ R @ H conjugation
                R_p[:, 1] *= -1
                fwd = R_p @ [1., 0., 0.]
                arrow_vectors.append((fwd * args.arrow_length).astype(np.float32))
                rr.log("world/full_path_arrows", rr.Arrows3D(
                    origins=np.array(arrow_origins),
                    vectors=np.array(arrow_vectors),
                    colors=[[120, 180, 255]],
                    radii=args.arrow_length * 0.015,
                ))
                next_arrow += arrow_step

        print(f"  {len(arrow_origins)} arrows, {len(all_poses)} timeline steps")


if __name__ == "__main__":
    main()
