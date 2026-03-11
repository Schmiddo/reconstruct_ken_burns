#!/usr/bin/env python3
"""
Parse Unreal Engine GVAS save files to extract the ``transformHistory`` array.

Each entry in the array is an Unreal Engine ``Transform`` struct containing:
- ``rotation``    — quaternion {x, y, z, w} in Unreal world space
- ``translation`` — position in cm in Unreal world space
- ``scale``       — uniform scale (typically 1, 1, 1)

Coordinate system: Unreal left-handed world — +X forward, +Y right, +Z up.
Quaternions follow the standard formula but operate in a left-handed sense.

Output JSON format::

    {
      "file": "recording.sav",
      "transform_count": N,
      "transforms": [
        {"rotation": {"x":…,"y":…,"z":…,"w":…},
         "translation": {"x":…,"y":…,"z":…},
         "scale": {"x":…,"y":…,"z":…}},
        …
      ]
    }

Usage:
    ./parse_camera_poses.py recording.sav poses.json
    ./parse_camera_poses.py recording.sav          # prints JSON to stdout
"""

import struct
import sys
import json
import argparse
from dataclasses import dataclass, asdict


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Transform:
    rotation: Quaternion
    translation: Vector3
    scale: Vector3


class GVASParser:
    """Parser for GVAS save files containing transform history."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_bytes(self, count: int) -> bytes:
        """Read a specific number of bytes and advance offset."""
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result

    def read_uint32(self) -> int:
        """Read a little-endian uint32."""
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_uint64(self) -> int:
        """Read a little-endian uint64."""
        return struct.unpack('<Q', self.read_bytes(8))[0]

    def read_float(self) -> float:
        """Read a little-endian float32."""
        return struct.unpack('<f', self.read_bytes(4))[0]

    def read_string(self) -> str:
        """Read a length-prefixed null-terminated string."""
        length = self.read_uint32()
        if length == 0:
            return ""
        raw = self.read_bytes(length)
        # Remove null terminator
        return raw[:-1].decode('utf-8', errors='replace')

    def peek_string(self) -> str:
        """Peek at the next string without advancing offset."""
        saved_offset = self.offset
        result = self.read_string()
        self.offset = saved_offset
        return result

    def skip_guid(self):
        """Skip a 16-byte GUID plus 1-byte flag (17 bytes total)."""
        self.offset += 17

    def read_quaternion(self) -> Quaternion:
        """Read a quaternion (4 floats)."""
        return Quaternion(
            x=self.read_float(),
            y=self.read_float(),
            z=self.read_float(),
            w=self.read_float()
        )

    def read_vector3(self) -> Vector3:
        """Read a 3D vector (3 floats)."""
        return Vector3(
            x=self.read_float(),
            y=self.read_float(),
            z=self.read_float()
        )

    def find_sequence(self, sequence: bytes, start: int = 0) -> int:
        """Find a byte sequence in the data, returning offset or -1."""
        return self.data.find(sequence, start)

    def parse_header(self) -> dict:
        """Parse the GVAS file header."""
        self.offset = 0

        # Magic bytes "GVAS"
        magic = self.read_bytes(4)
        if magic != b'GVAS':
            raise ValueError(f"Invalid magic bytes: {magic}")

        # Save game version
        save_version = self.read_uint32()

        # Package version
        package_version = self.read_uint32()

        # Engine version (major.minor.patch.changelist)
        engine_major = struct.unpack('<H', self.read_bytes(2))[0]
        engine_minor = struct.unpack('<H', self.read_bytes(2))[0]
        engine_patch = struct.unpack('<H', self.read_bytes(2))[0]
        engine_changelist = self.read_uint32()

        # Engine version string
        engine_version = self.read_string()

        # Custom version count and entries
        custom_version_count = self.read_uint32()
        custom_versions = []
        for _ in range(custom_version_count):
            guid = self.read_bytes(16)
            version = self.read_uint32()
            custom_versions.append({'guid': guid.hex(), 'version': version})

        # Save game class name
        save_class = self.read_string()

        return {
            'magic': magic.decode('utf-8'),
            'save_version': save_version,
            'package_version': package_version,
            'engine_version': engine_version,
            'custom_version_count': custom_version_count,
            'save_class': save_class,
            'header_end_offset': self.offset
        }

    def parse_struct_property_header(self) -> tuple[str, str, int, str]:
        """
        Parse a StructProperty header.

        Returns: (property_name, property_type, size, struct_type)
        """
        prop_name = self.read_string()
        prop_type = self.read_string()
        size = self.read_uint64()
        struct_type = self.read_string()
        self.skip_guid()
        return prop_name, prop_type, size, struct_type

    def parse_array_property_header(self) -> tuple[str, int, str]:
        """
        Parse an ArrayProperty header.

        Returns: (property_name, size, element_type)
        """
        prop_name = self.read_string()
        prop_type = self.read_string()
        if prop_type != "ArrayProperty":
            raise ValueError(f"Expected ArrayProperty, got {prop_type}")
        size = self.read_uint64()
        element_type = self.read_string()
        return prop_name, size, element_type

    def find_transform_history(self) -> int:
        """Find the transformHistory array and return its element count."""
        # Search for "transformHistory" followed by "ArrayProperty"
        search = b'transformHistory\x00'
        pos = self.find_sequence(search)
        if pos == -1:
            raise ValueError("Could not find transformHistory property")

        # Position just before the property name length
        self.offset = pos - 4

        # Parse array property header
        prop_name, size, element_type = self.parse_array_property_header()

        if element_type != "StructProperty":
            raise ValueError(f"Expected StructProperty elements, got {element_type}")

        # Read array metadata: count and struct info
        # Skip 1 byte (unknown flag)
        self.offset += 1

        # Read inner struct GUID
        self.read_bytes(16)

        # Read element count
        element_count = self.read_uint32()

        # Read inner struct property info (transformHistory again as struct name)
        inner_name = self.read_string()
        inner_type = self.read_string()
        inner_size = self.read_uint64()
        struct_name = self.read_string()

        # Skip struct GUID
        self.skip_guid()

        return element_count

    def parse_transform(self) -> Transform | None:
        """Parse a single Transform struct."""
        # Check if we're at a Rotation property
        next_str = self.peek_string()
        if next_str != "Rotation":
            return None

        # Parse Rotation (Quat)
        rot_name, rot_type, rot_size, rot_struct = self.parse_struct_property_header()
        if rot_struct != "Quat":
            raise ValueError(f"Expected Quat struct, got {rot_struct}")
        rotation = self.read_quaternion()

        # Parse Translation (Vector)
        trans_name, trans_type, trans_size, trans_struct = self.parse_struct_property_header()
        if trans_struct != "Vector":
            raise ValueError(f"Expected Vector struct, got {trans_struct}")
        translation = self.read_vector3()

        # Parse Scale3D (Vector)
        scale_name, scale_type, scale_size, scale_struct = self.parse_struct_property_header()
        if scale_struct != "Vector":
            raise ValueError(f"Expected Vector struct, got {scale_struct}")
        scale = self.read_vector3()

        # Read "None" marker indicating end of struct
        none_marker = self.read_string()
        if none_marker != "None":
            raise ValueError(f"Expected 'None' marker, got '{none_marker}'")

        return Transform(rotation=rotation, translation=translation, scale=scale)

    def parse_transforms(self, quiet: bool = False) -> list[Transform]:
        """Parse all transforms from the file."""
        def log(*msg):
            if not quiet:
                print(*msg, file=sys.stderr)

        # Parse header
        header = self.parse_header()
        log(f"GVAS Header:")
        log(f"  Engine: {header['engine_version']}")
        log(f"  Save class: {header['save_class']}")
        log(f"  Package version: {header['package_version']}")

        # Find and parse transform history
        element_count = self.find_transform_history()
        log(f"  Transform count: {element_count}")

        # Parse each transform
        transforms = []
        for i in range(element_count):
            transform = self.parse_transform()
            if transform is None:
                print(f"Warning: Failed to parse transform {i} at offset 0x{self.offset:x}", file=sys.stderr)
                break
            transforms.append(transform)

            # Progress indicator
            if (i + 1) % 1000 == 0:
                log(f"  Parsed {i + 1}/{element_count} transforms...")

        return transforms


def transforms_to_dict(transforms: list[Transform]) -> list[dict]:
    """Convert transforms to a JSON-serializable format."""
    result = []
    for t in transforms:
        result.append({
            'rotation': {
                'x': t.rotation.x,
                'y': t.rotation.y,
                'z': t.rotation.z,
                'w': t.rotation.w
            },
            'translation': {
                'x': t.translation.x,
                'y': t.translation.y,
                'z': t.translation.z
            },
            'scale': {
                'x': t.scale.x,
                'y': t.scale.y,
                'z': t.scale.z
            }
        })
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Parse Unreal Engine GVAS save files to extract camera pose history')
    parser.add_argument('input', type=str,
                        help='Path to .sav file')
    parser.add_argument('output', type=str, nargs='?', default=None,
                        help='Output JSON file (default: print to stdout)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress and summary output')

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    # Read input file
    with open(input_path, 'rb') as f:
        data = f.read()

    def log(*msg):
        if not args.quiet:
            print(*msg, file=sys.stderr)

    log(f"Parsing {input_path} ({len(data)} bytes)...")

    # Parse transforms
    gvas = GVASParser(data)
    transforms = gvas.parse_transforms(quiet=args.quiet)

    log(f"Successfully parsed {len(transforms)} transforms")

    # Convert to JSON
    output = {
        'file': input_path,
        'transform_count': len(transforms),
        'transforms': transforms_to_dict(transforms)
    }

    # Output
    json_str = json.dumps(output, indent=2)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_str)
        log(f"Output written to {output_path}")
    else:
        print(json_str)

    # Print summary statistics
    if transforms and not args.quiet:
        print(f"\nTransform Statistics:", file=sys.stderr)
        xs = [t.translation.x for t in transforms]
        ys = [t.translation.y for t in transforms]
        zs = [t.translation.z for t in transforms]
        print(f"  Translation X: {min(xs):.2f} to {max(xs):.2f}", file=sys.stderr)
        print(f"  Translation Y: {min(ys):.2f} to {max(ys):.2f}", file=sys.stderr)
        print(f"  Translation Z: {min(zs):.2f} to {max(zs):.2f}", file=sys.stderr)

        print(f"\nFirst 3 transforms:", file=sys.stderr)
        for i, t in enumerate(transforms[:3]):
            print(f"  [{i}] pos=({t.translation.x:.2f}, {t.translation.y:.2f}, {t.translation.z:.2f}) "
                  f"rot=({t.rotation.x:.4f}, {t.rotation.y:.4f}, {t.rotation.z:.4f}, {t.rotation.w:.4f})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
