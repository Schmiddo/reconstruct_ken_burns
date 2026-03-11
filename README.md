# Ken Burns Camera Pose Extraction

Reverse engineering the save file format used to store camera poses from an Unreal Engine 4 application as used in the [Ken Burns](https://github.com/sniklaus/3d-ken-burns) dataset, trying to figure out the poses used for rendering; see also [this issue](https://github.com/sniklaus/3d-ken-burns/issues/71).

To easily run the code, use [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just), link `data` to your copy of the dataset, and place the `.sav` file in the root of this repo.
Then run eg `just reconstruct western-flying` and you should see a rerun visualization of the scene.
The blue line indicates the poses of the savegame, the red line connects the poses used for the reconstruction.

The rest of this readme is AI-generated and quite verbose, but might still be helpful so I'm leaving it as-is.

## Overview

The `.sav` files contain camera transform history data captured from an Unreal Engine 4.19 application. The goal is to extract the camera poses (position and rotation) for each frame.

## File Format

### GVAS Format

The files use the **GVAS** (Game Versioned Archive Save) format, which is the standard save file format for Unreal Engine 4/5 games.

**Header** (736 bytes):
```
Offset  Content
0x0000  "GVAS" magic bytes
0x0004  Save game version: 2
0x0008  Package version: 516
0x000C  Engine version: 4.19.2 (changelist 4033788)
0x0016  Engine string: "++UE4+Release-4.19"
0x002D  Custom version count: 3
0x0031  Custom version GUIDs (3 × 20 bytes)
0x006D  Save class name (empty in these files)
```

**Note**: Package version 516 is not supported by the standard `gvas` Rust crate or `gvas2json` tool, requiring a custom parser.

### Data Structure

```
FirstPersonSave_C
└── transformHistory: ArrayProperty<StructProperty>
    └── [0..N] Transform
        ├── Rotation: StructProperty<Quat>
        │   └── X, Y, Z, W (4 × float32, little-endian)
        ├── Translation: StructProperty<Vector>
        │   └── X, Y, Z (3 × float32, little-endian)
        └── Scale3D: StructProperty<Vector>
            └── X, Y, Z (3 × float32, always 1.0, 1.0, 1.0)
```

### Property Serialization

Each property in GVAS follows this pattern:

```
[4 bytes]  Property name length (uint32)
[N bytes]  Property name (null-terminated string)
[4 bytes]  Property type length (uint32)
[N bytes]  Property type (e.g., "StructProperty\0")
[8 bytes]  Property value size (uint64)
[4 bytes]  Struct type name length (uint32)
[N bytes]  Struct type name (e.g., "Quat\0", "Vector\0")
[17 bytes] GUID + flags (all zeros for these files)
[N bytes]  Actual data values
```

### Transform Byte Layout

Each transform is exactly **253 bytes**, but only **40 bytes (15.8%)** is actual data. The rest is repeated GVAS metadata:

| Component | Header Overhead | Data | Total |
|-----------|-----------------|------|-------|
| Rotation (Quat) | 66 bytes | 16 bytes | 82 bytes |
| Translation (Vector) | 71 bytes | 12 bytes | 83 bytes |
| Scale3D (Vector) | 67 bytes | 12 bytes | 79 bytes |
| None marker | 9 bytes | - | 9 bytes |
| **Total** | **213 bytes (84.2%)** | **40 bytes (15.8%)** | **253 bytes** |

Each transform ends with a `"None\0"` marker, followed immediately by the next transform's `"Rotation"` property.

## What's NOT in the Files

The save files contain **only raw camera transforms** with no additional metadata:

- No timestamps or frame indices
- No frame rate or delta time between frames
- No FOV, focal length, or other camera parameters
- No camera or sequence names
- No scene or level information
- No keyframe markers or animation curves

To correlate poses with rendered images, external information about the capture frame rate is needed.

## Coordinate System

Unreal Engine uses a left-handed coordinate system:
- **X**: Forward
- **Y**: Right
- **Z**: Up

Units are in centimeters by default.

## Tools in This Repository

| Tool | Description |
|------|-------------|
| `parse_camera_poses.py` | Parse GVAS `.sav` files → poses JSON |
| `match_poses.py` | Map dataset frames to poses → mapping JSON + SE3 NPY array |
| `reconstruct_3d.py` | Back-project depth maps → Rerun 3-D viewer / PLY |

### Usage

```bash
# 1. Parse .sav → poses JSON
uv run parse-camera-poses western-flying.sav western-flying-poses.json

# 2. Match frames to poses → mapping JSON + SE3 pose array
uv run match-poses western-flying-poses.json --scene western-flying
# outputs: western-flying-mapping.json, western-flying-poses.npy

# Optional: use SLERP interpolation for sub-integer pose stride
uv run match-poses western-flying-poses.json --scene western-flying --interpolate

# 3. Visualise 3-D reconstruction in Rerun
uv run reconstruct-3d western-flying-mapping.json --frames 50 --all-views

# Also show the full pose path (all poses, unrollable on a timeline)
uv run reconstruct-3d western-flying-mapping.json --poses western-flying-poses.json

# Save to .rrd without spawning the viewer
uv run reconstruct-3d western-flying-mapping.json --rrd out.rrd --no-viewer
```

### Setup

```bash
uv sync          # creates .venv and installs all dependencies
```

After that, scripts are available via `uv run`:

```bash
uv run parse-camera-poses recording.sav poses.json
uv run match-poses poses.json --scene western-flying
uv run reconstruct-3d mapping.json --all-views
```

Or use the justfile recipes which call `uv run` automatically:

```bash
just extract-poses western-flying
just reconstruct western-flying
```

### Output Formats

**`<scene>-mapping.json`** (from `match_poses.py`):
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
      "pose_index": 0,
      "int_sample": 1,
      "fov_deg": 90.0,
      "position": {"x": -282.03, "y": -511.72, "z": 79.98},
      "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }
  ]
}
```

**`<scene>-poses.npy`** (from `match_poses.py`):
Shape `(num_frames, 4, 4)` float64. Camera-to-world SE3 matrices in the right-handed
display world (+X forward, +Y left, +Z up — Unreal's left-handed world with Y negated).

**`<scene>-poses.json`** (from `parse_camera_poses.py`):
```json
{
  "file": "western-flying.sav",
  "transform_count": 6344,
  "transforms": [
    {
      "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 },
      "translation": { "x": -282.03, "y": -511.72, "z": 79.98 },
      "scale": { "x": 1.0, "y": 1.0, "z": 1.0 }
    }
  ]
}
```

## Dataset Structure

The rendered dataset (in `data/`) uses a subset of the extracted poses:

| Property | Value |
|----------|-------|
| Poses in .sav | 6,344 |
| Frames in dataset | 705 |
| Subsampling ratio | Every 9th pose |
| Views per frame | 4 (tl, tr, bl, br) |
| Intra-frame baseline | 40 units |
| Image resolution | 512 × 512 |
| FOV range | 60° - 90° |

**Per-frame data:**
- 4 RGB images: `{frame:05d}-{view}-image.png` (tl, tr, bl, br)
- 4 depth maps: `{frame:05d}-{view}-depth.exr`
- 4 normal maps: `{frame:05d}-{view}-normal.exr`
- 1 metadata file: `{frame:05d}-meta.json` with `intSample` and `fltFov`

**View arrangement** (baseline = 40):
```
    tl ----40---- tr
    |              |
   40            40
    |              |
    bl ----40---- br

Diagonal baseline: 56.6 (40 × √2)
```

## References

- [gvas crate](https://github.com/localcc/gvas/) - Rust GVAS parser (doesn't support version 516)
- [UE4 Save Game Format](https://docs.unrealengine.com/4.27/en-US/InteractiveExperiences/SaveGame/) - Official documentation
- [GVAS file format](https://github.com/13xforever/gvas-converter) - Community documentation
