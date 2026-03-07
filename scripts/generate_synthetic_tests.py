#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""
Generate synthetic JXL test images to exercise low-coverage code paths.

This script creates small test images with specific characteristics and encodes
them using cjxl with various options to maximize code coverage.

Usage: python3 scripts/generate_synthetic_tests.py
"""

import os
import subprocess
import struct
import sys
import tempfile
import zlib
from pathlib import Path

# Find cjxl: check CJXL_PATH env var first, then common locations
CJXL_PATHS = [
    Path(__file__).parent.parent.parent / "jxl-efforts/libjxl/build/tools/cjxl",
    Path.home() / "work/jxl-efforts/libjxl/build/tools/cjxl",
    Path.home() / "work/libjxl/build/tools/cjxl",
]

def find_cjxl():
    env_path = os.environ.get("CJXL_PATH")
    if env_path:
        return env_path
    for p in CJXL_PATHS:
        if p.exists():
            return str(p)
    # Try PATH
    try:
        subprocess.run(["cjxl", "--version"], capture_output=True, check=True)
        return "cjxl"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def create_png(width, height, bit_depth=8, color_type=2, pixels=None):
    """Create a minimal PNG file in memory.

    color_type: 0=grayscale, 2=RGB, 4=grayscale+alpha, 6=RGBA
    """
    def png_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    png += png_chunk(b'IHDR', ihdr_data)

    # Generate pixel data if not provided
    if pixels is None:
        channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
        bytes_per_sample = 2 if bit_depth == 16 else 1
        row_bytes = width * channels * bytes_per_sample

        raw_data = b''
        for y in range(height):
            raw_data += b'\x00'  # filter byte
            for x in range(width):
                for c in range(channels):
                    # Create gradient pattern
                    val = ((x + y * 17 + c * 37) * 255 // max(width + height, 1)) % 256
                    if bit_depth == 16:
                        raw_data += struct.pack(">H", val * 257)
                    else:
                        raw_data += bytes([val])
    else:
        raw_data = pixels

    # IDAT (compressed pixel data)
    compressed = zlib.compress(raw_data, 9)
    png += png_chunk(b'IDAT', compressed)

    # IEND
    png += png_chunk(b'IEND', b'')

    return png


def create_indexed_png(width, height, num_colors=16):
    """Create an indexed/paletted PNG."""
    def png_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    png = b'\x89PNG\r\n\x1a\n'

    # IHDR - color_type 3 = indexed
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    png += png_chunk(b'IHDR', ihdr_data)

    # PLTE - palette
    palette = b''
    for i in range(num_colors):
        r = (i * 17) % 256
        g = (i * 37 + 100) % 256
        b = (i * 59 + 50) % 256
        palette += bytes([r, g, b])
    png += png_chunk(b'PLTE', palette)

    # IDAT
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # filter byte
        for x in range(width):
            raw_data += bytes([(x + y) % num_colors])

    compressed = zlib.compress(raw_data, 9)
    png += png_chunk(b'IDAT', compressed)
    png += png_chunk(b'IEND', b'')

    return png


def encode_jxl(cjxl, png_data, output_path, options):
    """Encode PNG data to JXL with given options."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(png_data)
        png_path = f.name

    try:
        cmd = [cjxl, png_path, str(output_path)] + options
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARN: {' '.join(options)} failed: {result.stderr[:100]}")
            return False
        return True
    finally:
        os.unlink(png_path)


def main():
    cjxl = find_cjxl()
    if not cjxl:
        print("ERROR: cjxl not found. Please build libjxl first.")
        sys.exit(1)

    print(f"Using cjxl: {cjxl}")

    # Output directory
    out_dir = Path(__file__).parent.parent / "jxl/resources/test/synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # === Color Encoding Coverage (headers/color_encoding.rs) ===
    print("\n=== Generating color encoding tests ===")

    # Small 8x8 RGB image for color space tests
    rgb_png = create_png(8, 8, bit_depth=8, color_type=2)

    color_tests = [
        ("colorspace_srgb", ["-d", "0", "-e", "3"]),  # Lossless sRGB
        # Different transfer functions via quality settings
        ("tf_srgb_q90", ["-q", "90"]),
    ]

    for name, opts in color_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, rgb_png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Bit Depth Coverage (headers/bit_depth.rs) ===
    print("\n=== Generating bit depth tests ===")

    bit_depth_tests = [
        ("synth_8bit", create_png(8, 8, bit_depth=8, color_type=2), ["-d", "0", "-e", "3"]),
        ("synth_16bit", create_png(8, 8, bit_depth=16, color_type=2), ["-d", "0", "-e", "3"]),
        ("synth_8bit_gray", create_png(8, 8, bit_depth=8, color_type=0), ["-d", "0", "-e", "3"]),
        ("synth_16bit_gray", create_png(8, 8, bit_depth=16, color_type=0), ["-d", "0", "-e", "3"]),
    ]

    for name, png, opts in bit_depth_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Palette Coverage (frame/modular/transforms/palette.rs) ===
    print("\n=== Generating palette tests ===")

    # Indexed PNG for palette mode
    indexed_png = create_indexed_png(16, 16, num_colors=16)

    palette_tests = [
        ("palette_indexed", indexed_png, ["-d", "0", "-e", "3"]),
        ("palette_small", create_indexed_png(8, 8, num_colors=4), ["-d", "0", "-e", "3"]),
    ]

    for name, png, opts in palette_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # Force palette transform with modular mode on RGB
    rgb_small = create_png(16, 16, bit_depth=8, color_type=2)
    palette_force_tests = [
        ("palette_forced", ["-d", "0", "-e", "3", "-m", "1", "--modular_palette_colors=256"]),
        ("palette_lossy", ["-d", "1", "-e", "3", "-m", "1", "--modular_lossy_palette"]),
    ]

    for name, opts in palette_force_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, rgb_small, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Container Coverage (container/parse.rs) ===
    print("\n=== Generating container tests ===")

    container_tests = [
        ("container_bare", ["-d", "0", "-e", "3"]),  # Default (usually no container for small)
        ("container_forced_on", ["-d", "0", "-e", "3", "-x", "container=1"]),
    ]

    for name, opts in container_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, rgb_png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Modular Predictor Coverage ===
    print("\n=== Generating modular predictor tests ===")

    # Different predictor modes (--modular_predictor or -P)
    predictor_tests = [
        ("predictor_zero", ["-d", "0", "-e", "3", "-m", "1", "-P", "0"]),
        ("predictor_left", ["-d", "0", "-e", "3", "-m", "1", "-P", "1"]),
        ("predictor_top", ["-d", "0", "-e", "3", "-m", "1", "-P", "2"]),
        ("predictor_select", ["-d", "0", "-e", "3", "-m", "1", "-P", "4"]),
        ("predictor_gradient", ["-d", "0", "-e", "3", "-m", "1", "-P", "5"]),
        ("predictor_weighted", ["-d", "0", "-e", "3", "-m", "1", "-P", "6"]),
    ]

    for name, opts in predictor_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, rgb_png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Alpha Channel Coverage ===
    print("\n=== Generating alpha channel tests ===")

    rgba_png = create_png(8, 8, bit_depth=8, color_type=6)  # RGBA
    gray_alpha_png = create_png(8, 8, bit_depth=8, color_type=4)  # Gray+Alpha

    alpha_tests = [
        ("alpha_rgba", rgba_png, ["-d", "0", "-e", "3"]),
        ("alpha_gray", gray_alpha_png, ["-d", "0", "-e", "3"]),
        ("alpha_16bit", create_png(8, 8, bit_depth=16, color_type=6), ["-d", "0", "-e", "3"]),
    ]

    for name, png, opts in alpha_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Group Size Coverage ===
    print("\n=== Generating group size tests ===")

    # Larger image to test group boundaries
    large_png = create_png(64, 64, bit_depth=8, color_type=2)

    group_tests = [
        ("group_128", ["-d", "0", "-e", "3", "-g", "0"]),  # 128x128
        ("group_256", ["-d", "0", "-e", "3", "-g", "1"]),  # 256x256
        ("group_512", ["-d", "0", "-e", "3", "-g", "2"]),  # 512x512
        ("group_1024", ["-d", "0", "-e", "3", "-g", "3"]), # 1024x1024
    ]

    for name, opts in group_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, large_png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === VarDCT Specific (for comparison) ===
    print("\n=== Generating VarDCT tests ===")

    vardct_tests = [
        ("vardct_q50", ["-q", "50"]),
        ("vardct_q90", ["-q", "90"]),
        ("vardct_effort1", ["-q", "75", "-e", "1"]),
        ("vardct_effort9", ["-q", "75", "-e", "9"]),
    ]

    for name, opts in vardct_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, rgb_png, out, opts):
            generated.append(name)
            print(f"  {name}")

    # === Squeeze Transform Coverage ===
    print("\n=== Generating squeeze tests ===")

    squeeze_tests = [
        ("squeeze_default", ["-d", "0", "-e", "3", "-m", "1"]),
        # Squeeze is automatic for larger images in modular mode
    ]

    larger_png = create_png(32, 32, bit_depth=8, color_type=2)
    for name, opts in squeeze_tests:
        out = out_dir / f"{name}.jxl"
        if encode_jxl(cjxl, larger_png, out, opts):
            generated.append(name)
            print(f"  {name}")

    print(f"\n=== Generated {len(generated)} test files in {out_dir} ===")
    print("\nFiles:")
    for name in sorted(generated):
        jxl_path = out_dir / f"{name}.jxl"
        size = jxl_path.stat().st_size if jxl_path.exists() else 0
        print(f"  {name}.jxl ({size} bytes)")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
