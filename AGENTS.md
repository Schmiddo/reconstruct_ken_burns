# Agent Context: reveng_kenburns

Reverse-engineering tool-chain for extracting and visualising camera poses from
an Unreal Engine 4.19 application ("Ken Burns" flythrough renderer).

---

## Pipeline overview

```
recording.sav  ──parse_camera_poses.py──►  <scene>-poses.json
                                                    │
                                     ┌──────────────┤
                                     ▼              ▼
                            match_poses.py    export_poses.py
                                     │              │
                                     ▼              ▼
                         <scene>-mapping.json   poses.npy
                                     │
                                     ▼
                            reconstruct_3d.py  (→ Rerun viewer / .rrd / .ply)
```

Supporting / analysis scripts: `visualize_poses.py`, `analyze_dataset.py`,
`find_stereo_pairs.py`.

---

## File formats

### `<scene>-poses.json`  (output of `parse_camera_poses.py`)
```json
{
  "file": "western-flying.sav",
  "transform_count": 6344,
  "transforms": [
    {
      "rotation":    {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
      "translation": {"x": -282.03, "y": -511.72, "z": 79.98},
      "scale":       {"x": 1.0, "y": 1.0, "z": 1.0}
    }
  ]
}
```
Key: `transforms[i].translation` and `transforms[i].rotation`.
All values are in **Unreal world space** (left-handed, cm).

### `<scene>-mapping.json`  (output of `match_poses.py`)
```json
{
  "scene": "western-flying",
  "poses_file": "western-flying-poses.json",
  "num_poses": 6344,
  "num_frames": 705,
  "step": 9,
  "frames": [
    {
      "frame_index": 1,
      "pose_index":  0,
      "int_sample":  1,
      "fov_deg":     90.0,
      "position":    {"x": -282.03, "y": -511.72, "z": 79.98},
      "rotation":    {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
  ]
}
```
Key differences from poses JSON: field is `position` (not `translation`), and
`fov_deg` is included.  `frame_index` is 1-based and matches filenames.

### `poses.npy`  (output of `export_poses.py`)
Shape `(N, 4, 4)` float64.  Camera-to-world SE3 matrices in the **right-handed
display world** (after coordinate conversion — see below).

---

## Coordinate systems — critical details

There are **three** coordinate frames in play. Getting these wrong produces
mirrored positions and/or backwards rotations.

### 1. Unreal world (raw data)
- Left-handed: **+X forward, +Y right, +Z up**
- Quaternion formula is standard but operates in a left-handed sense
- Units: centimetres

### 2. Right-handed display world (used internally and in Rerun)
- **+X forward, +Y left, +Z up** (Y is negated relative to Unreal)
- Matches Rerun's `RIGHT_HAND_Z_UP` view coordinate setting
- Conversion from Unreal world:
  ```python
  center[1] *= -1          # negate Y in position
  R[1, :] *= -1            # H @ R @ H conjugation, step 1
  R[:, 1] *= -1            # H @ R @ H conjugation, step 2  (det stays +1)
  ```
  **Do not** use a one-sided flip (`R[1,:]*=-1` only) — that gives det = −1
  (a rotoreflection, not a rotation), breaking all derived directions.

- After conversion, camera-local axes are:
  - `+X` → forward
  - `+Y` → world **LEFT** (not right!)
  - `+Z` → up
  - Visual right  = `R @ [0, -1, 0]`  (negative local Y)
  - Visual up     = `R @ [0,  0, 1]`

### 3. Rerun / OpenCV camera frame
- **+X image-right, +Y image-down, +Z forward**
- Mapping from display-world camera frame:
  ```python
  _UE_TO_RR_CAM = np.array([
      [0., -1.,  0.],   # Rerun +X = display-world −Y  (visual right)
      [0.,  0., -1.],   # Rerun +Y = display-world −Z  (visual down)
      [1.,  0.,  0.],   # Rerun +Z = display-world +X  (forward)
  ])
  # world-to-Rerun-camera rotation:
  R_world_rr = R_world_ue @ _UE_TO_RR_CAM.T
  ```
  Used only in `log_camera()` when logging a `rr.Pinhole` + `rr.Transform3D`.

---

## Camera model

- Depth maps store **linear scene depth along the camera's forward (+X) axis**,
  NOT radial distance.
- FOV in `fltFov` is the **full horizontal opening angle** (degrees).
- Focal length: `f = (W/2) / tan(fov_rad/2)`.  Image is square (512×512) so
  `fx = fy = f`.
- Back-projection formula:
  ```
  P = cam_pos + d * (forward + du*right - dv*up)
  ```
  where `du = (u - cx)/f`, `dv = (v - cy)/f`, and:
  - `forward = R @ [1, 0, 0]`
  - `right   = R @ [0, -1, 0]`   ← note the minus sign (see coordinate note)
  - `up      = R @ [0, 0, 1]`

---

## Dataset facts (western-flying)

| Property | Value |
|---|---|
| Poses in .sav | 6 344 |
| Rendered frames | 705 |
| Subsampling step | 9 (every 9th pose used) |
| Mapping formula | `pose_idx = (frame_idx - 1) * step` |
| Frame index | 1-based (matches filenames `00001-*.png`) |
| Depth sentinel | ~65 280 cm (sky); filter with `max_depth=50 000` |

### Camera rig (4 views per frame)
```
  tl ──40 cm── tr
  │             │
 40 cm         40 cm
  │             │
  bl ──40 cm── br
```
Offsets from rig centre in (right, up): `tl=(−0.5,+0.5)`, `tr=(+0.5,+0.5)`,
`bl=(−0.5,−0.5)`, `br=(+0.5,−0.5)`, scaled by the baseline (default 40 cm).

---

## Data directory layout

```
data/
  <scene>/
    {frame:05d}-{view}-image.png   # 512×512 uint8 RGB; views: tl tr bl br
    {frame:05d}-meta.json          # {"intSample": N, "fltFov": 90.0}
  <scene>-depth/
    {frame:05d}-{view}-depth.exr   # float32 single-channel 'Y', cm
  <scene>-normal/
    {frame:05d}-{view}-normal.exr  # not yet used
```

---

## Common pitfalls

- **One-sided Y flip**: `R[1,:]*=-1` without `R[:,1]*=-1` gives det = −1.
  Always apply both lines (H@R@H).
- **`right` axis sign**: In the display world, visual right is **−local Y**, so
  use `R @ [0, -1, 0]`, never `R @ [0, 1, 0]`.
- **Depth is not radial**: Using `sqrt(x²+y²+d²)` as depth is wrong.
- **FOV is full angle**: Half-angle goes into `tan()`.
- **Frame index is 1-based**: `frame_idx=1` → `pose_idx=0`.
- **Mapping uses `position`, not `translation`**: The mapping JSON field name
  differs from the poses JSON field name.
- **`rr.Pinhole` focal length**: Pass as a list `[f, f]`, not a scalar.
- **open3d unavailable**: No cp314 wheel; use rerun-sdk instead.

---

## Environment

- Python 3.14 in `.venv`; install packages with `uv pip install`
- Installed: `numpy`, `matplotlib`, `opencv-python-headless`, `openexr`,
  `openexr-python`, `rerun-sdk==0.30.1`
- `rr.set_time("pose", sequence=n)` for timeline scrubbing
- `static=True` on `rr.log(...)` for permanent (non-timeline) entities
