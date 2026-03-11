# Ken Burns Camera Pose Extraction

Reverse engineering the save file format used to store camera poses from an Unreal Engine 4 application.

## Overview

The `.sav` files (`western-flying.sav`, `western-walking.sav`) contain camera transform history data captured from an Unreal Engine 4.19 application. The goal is to extract the camera poses (position and rotation) for each frame.

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

### Example: First Transform

Located starting around offset `0x390`:

| Field | Offset | Raw Bytes | Value |
|-------|--------|-----------|-------|
| Rotation X | 0x3DB | `00 00 00 00` | 0.0 |
| Rotation Y | 0x3DF | `00 00 00 00` | 0.0 |
| Rotation Z | 0x3E3 | `00 00 00 00` | 0.0 |
| Rotation W | 0x3E7 | `00 00 80 3F` | 1.0 |
| Translation X | 0x432 | `45 03 8D C3` | -282.02 |
| Translation Y | 0x436 | `A1 DB FF C3` | -511.72 |
| Translation Z | 0x43A | `15 F5 9F 42` | 79.98 |
| Scale3D X | 0x481 | `00 00 80 3F` | 1.0 |
| Scale3D Y | 0x485 | `00 00 80 3F` | 1.0 |
| Scale3D Z | 0x489 | `00 00 80 3F` | 1.0 |

The rotation `(0, 0, 0, 1)` is an identity quaternion (no rotation).

### Transform Byte Layout

Each transform is exactly **253 bytes**, but only **40 bytes (15.8%)** is actual data. The rest is repeated GVAS metadata:

| Component | Header Overhead | Data | Total |
|-----------|-----------------|------|-------|
| Rotation (Quat) | 66 bytes | 16 bytes | 82 bytes |
| Translation (Vector) | 71 bytes | 12 bytes | 83 bytes |
| Scale3D (Vector) | 67 bytes | 12 bytes | 79 bytes |
| None marker | 9 bytes | - | 9 bytes |
| **Total** | **213 bytes (84.2%)** | **40 bytes (15.8%)** | **253 bytes** |

**Per-property overhead breakdown:**
```
Property name:      4 + len(name) + 1      (length-prefixed string)
"StructProperty":   4 + 14 + 1 = 19 bytes  (repeated for every property)
Size field:         8 bytes                (uint64)
Struct type:        4 + len(type) + 1      (e.g., "Quat", "Vector")
GUID + flags:       17 bytes               (all zeros, but still stored)
```

**Storage efficiency:**
| Metric | Value |
|--------|-------|
| Overhead per transform | 213 bytes |
| Actual data per transform | 40 bytes |
| Total file size (6,344 transforms) | 1.6 MB |
| Wasted on metadata | 1.29 MB |
| Actual camera data | 248 KB |

A minimal binary format would be ~6.5× smaller (just transform count + raw floats).

Each transform ends with a `"None\0"` marker, followed immediately by the next transform's `"Rotation"` property.

## File Statistics

| File | Size | Transforms | Position Range (X, Y, Z) |
|------|------|------------|--------------------------|
| western-flying.sav | 1,605,966 bytes | 6,344 | (-3116, -6789, 64) to (2263, 1839, 1074) |
| western-walking.sav | 1,602,677 bytes | 6,331 | (-4111, -5933, 61) to (2799, 1908, 488) |

The "flying" trajectory has higher Z values (up to 1074 cm) while "walking" stays closer to ground level (max 488 cm).

Both files were created with identical engine versions and contain only the `transformHistory` array.

## What's NOT in the Files

The save files contain **only raw camera transforms** with no additional metadata:

- No timestamps or frame indices
- No frame rate or delta time between frames
- No FOV, focal length, or other camera parameters
- No camera or sequence names
- No scene or level information
- No keyframe markers or animation curves

To correlate poses with rendered images, external information about the capture frame rate would be needed.

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
| `match_poses.py` | Map dataset frames to poses → mapping JSON |
| `export_poses.py` | Convert poses / mapping → NumPy SE3 array (`.npy`) |
| `reconstruct_3d.py` | Back-project depth maps → Rerun 3-D viewer / PLY |
| `visualize_poses.py` | Matplotlib 3-D trajectory visualiser |
| `analyze_dataset.py` | Trajectory coverage and view-overlap analysis |
| `find_stereo_pairs.py` | Find frames with good inter-frame stereo baselines |

### Usage

```bash
# 1. Parse .sav → poses JSON
./parse_camera_poses.py western-flying.sav western-flying-poses.json

# 2a. Match frames to poses → mapping JSON
./match_poses.py western-flying-poses.json --scene western-flying
# output: western-flying-mapping.json

# 2b. Export poses as NumPy SE3 array (all poses)
./export_poses.py western-flying-poses.json
# Export only the dataset-mapped subset (every 9th pose)
./export_poses.py western-flying-poses.json --stride 9

# 3. Visualise 3-D reconstruction in Rerun
./reconstruct_3d.py western-flying-mapping.json --frames 50 --all-views

# Also show the full pose path (all 6344 poses, unrollable on a timeline)
./reconstruct_3d.py western-flying-mapping.json --poses western-flying-poses.json

# Save to .rrd without spawning the viewer
./reconstruct_3d.py western-flying-mapping.json --rrd out.rrd --no-viewer

# Visualize trajectory in Matplotlib
./visualize_poses.py western-flying-poses.json
./visualize_poses.py western-flying-poses.json western-walking-poses.json --compare
./visualize_poses.py western-flying-poses.json --subsample 10 --save trajectory.png
```

### Dependencies

```bash
uv pip install numpy matplotlib opencv-python-headless openexr openexr-python rerun-sdk
```

### Output Format

```json
{
  "file": "western-flying.sav",
  "transform_count": 6344,
  "transforms": [
    {
      "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 },
      "translation": { "x": -282.03, "y": -511.72, "z": 79.98 },
      "scale": { "x": 1.0, "y": 1.0, "z": 1.0 }
    },
    ...
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

### Analysis Tools Usage

```bash
# Analyze trajectory and dataset
./analyze_dataset.py -p western-flying-poses.json -d data -s western-flying

# Find frames with good stereo baselines
./find_stereo_pairs.py western-flying-poses.json --coverage --num-frames 50

# Find specific stereo pairs
./find_stereo_pairs.py western-flying-poses.json --min-baseline 100 --max-baseline 200
```

## References

- [gvas crate](https://github.com/localcc/gvas/) - Rust GVAS parser (doesn't support version 516)
- [UE4 Save Game Format](https://docs.unrealengine.com/4.27/en-US/InteractiveExperiences/SaveGame/) - Official documentation
- [GVAS file format](https://github.com/13xforever/gvas-converter) - Community documentation
