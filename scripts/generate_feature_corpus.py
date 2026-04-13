#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""
Generate a comprehensive JXL feature corpus exercising every spec feature.

Creates test images with "strange tint" color patterns and encodes them with
every combination of: effort levels, quality/distance, lossy/lossless,
splines, patches, dots, noise, EPF, gaborish, progressive modes, resampling,
squeeze, palette, predictors, RCT color transforms, group sizes, container
modes, faster_decoding levels, alpha, HDR color spaces, and more.

Usage:
    python3 scripts/generate_feature_corpus.py [--output-dir DIR]

Requires cjxl from libjxl to be available.
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


CJXL_PATHS = [
    Path(__file__).parent.parent.parent / "jxl-efforts/libjxl/build/tools/cjxl",
    Path.home() / "work/jxl-efforts/libjxl/build/tools/cjxl",
    Path.home() / "work/libjxl/build/tools/cjxl",
]

DJXL_PATHS = [
    Path(__file__).parent.parent.parent / "jxl-efforts/libjxl/build/tools/djxl",
    Path.home() / "work/jxl-efforts/libjxl/build/tools/djxl",
    Path.home() / "work/libjxl/build/tools/djxl",
]

TESTDATA = Path.home() / "work/jxl-efforts/libjxl/testdata"

# Interesting crop regions from flower.png (2268x1512).
# Each is (x, y, w, h, description).
CROP_REGIONS = [
    (800, 500, 256, 256, "petals_bokeh"),       # Petals + bokeh + bud — primary
    (1500, 800, 256, 256, "striped_petals"),     # Striped petal veins
    (1000, 300, 256, 256, "petal_veins"),        # Sharp veins + color transitions + green
    (1100, 600, 128, 128, "center_detail"),      # Tight crop, maximum detail density
]


def find_tool(env_var, paths, name):
    """Find a tool by env var, known paths, or PATH."""
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
    for p in paths:
        if p.exists():
            return str(p)
    try:
        subprocess.run([name, "--version"], capture_output=True, check=True)
        return name
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def find_cjxl():
    return find_tool("CJXL_PATH", CJXL_PATHS, "cjxl")


def find_djxl():
    return find_tool("DJXL_PATH", DJXL_PATHS, "djxl")


def find_oxipng():
    """Find oxipng for aggressive PNG compression."""
    try:
        subprocess.run(["oxipng", "--version"], capture_output=True, check=True)
        return "oxipng"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# PNG / PPM creation helpers
# ---------------------------------------------------------------------------

def png_chunk(chunk_type, data):
    chunk = chunk_type + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)


def create_png(width, height, bit_depth=8, color_type=2, pixel_fn=None):
    """Create a PNG. pixel_fn(x, y, c) -> value in [0, max_val]."""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    max_val = (1 << bit_depth) - 1
    bps = 2 if bit_depth == 16 else 1

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter none
        for x in range(width):
            for c in range(channels):
                if pixel_fn:
                    v = pixel_fn(x, y, c) & max_val
                else:
                    v = ((x * 37 + y * 17 + c * 59) * max_val // max(width + height, 1)) % (max_val + 1)
                if bps == 2:
                    raw += struct.pack(">H", v)
                else:
                    raw.append(v)

    png = b'\x89PNG\r\n\x1a\n'
    png += png_chunk(b'IHDR', struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
    png += png_chunk(b'IDAT', zlib.compress(bytes(raw), 9))
    png += png_chunk(b'IEND', b'')
    return png


def create_ppm(width, height, maxval=255, pixel_fn=None):
    """Create a P6 PPM. pixel_fn(x, y, c) -> value in [0, maxval]."""
    bps = 2 if maxval > 255 else 1
    header = f"P6\n{width} {height}\n{maxval}\n".encode()
    data = bytearray()
    for y in range(height):
        for x in range(width):
            for c in range(3):
                if pixel_fn:
                    v = min(max(pixel_fn(x, y, c), 0), maxval)
                else:
                    v = ((x * 37 + y * 17 + c * 59) * maxval // max(width + height, 1)) % (maxval + 1)
                if bps == 2:
                    data += struct.pack(">H", v)
                else:
                    data.append(v)
    return header + bytes(data)


def create_pgm(width, height, maxval=255, pixel_fn=None):
    """Create a P5 PGM (grayscale)."""
    bps = 2 if maxval > 255 else 1
    header = f"P5\n{width} {height}\n{maxval}\n".encode()
    data = bytearray()
    for y in range(height):
        for x in range(width):
            if pixel_fn:
                v = min(max(pixel_fn(x, y, 0), 0), maxval)
            else:
                v = ((x * 37 + y * 17) * maxval // max(width + height, 1)) % (maxval + 1)
            if bps == 2:
                data += struct.pack(">H", v)
            else:
                data.append(v)
    return header + bytes(data)


def create_pam(width, height, channels=4, maxval=255, pixel_fn=None):
    """Create a PAM (RGBA or other multi-channel)."""
    bps = 2 if maxval > 255 else 1
    if channels == 4:
        tupltype = "RGB_ALPHA"
    elif channels == 2:
        tupltype = "GRAYSCALE_ALPHA"
    elif channels == 3:
        tupltype = "RGB"
    else:
        tupltype = "GRAYSCALE"

    header = (f"P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {channels}\n"
              f"MAXVAL {maxval}\nTUPLTYPE {tupltype}\nENDHDR\n").encode()
    data = bytearray()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                if pixel_fn:
                    v = min(max(pixel_fn(x, y, c), 0), maxval)
                else:
                    v = ((x * 37 + y * 17 + c * 59) * maxval // max(width + height, 1)) % (maxval + 1)
                if bps == 2:
                    data += struct.pack(">H", v)
                else:
                    data.append(v)
    return header + bytes(data)


def create_indexed_png(width, height, num_colors=16):
    """Create an indexed/paletted PNG (color_type=3)."""
    png = b'\x89PNG\r\n\x1a\n'
    png += png_chunk(b'IHDR', struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))

    palette = bytearray()
    for i in range(num_colors):
        palette += bytes([(i * 67 + 30) % 256, (i * 137 + 100) % 256, (i * 211 + 50) % 256])
    png += png_chunk(b'PLTE', bytes(palette))

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.append((x + y * 3) % num_colors)
    png += png_chunk(b'IDAT', zlib.compress(bytes(raw), 9))
    png += png_chunk(b'IEND', b'')
    return png


# ---------------------------------------------------------------------------
# Pixel generators for "strange tint" and feature-triggering images
# ---------------------------------------------------------------------------

def tint_magenta(x, y, c):
    """Strong magenta tint — high R, low G, high B."""
    base = ((x * 37 + y * 17) % 256)
    if c == 0: return min(base + 140, 255)
    if c == 1: return max(base - 100, 0)
    if c == 2: return min(base + 120, 255)
    return 200  # alpha

def tint_cyan_poison(x, y, c):
    """Cyan-green toxic tint."""
    base = ((x * 23 + y * 41) % 256)
    if c == 0: return max(base // 4, 0)
    if c == 1: return min(base + 80, 255)
    if c == 2: return min(base + 60, 255)
    return 255

def tint_sepia_inverted(x, y, c):
    """Inverted sepia — blues and cyans dominate."""
    lum = ((x * 13 + y * 29) % 256)
    if c == 0: return max(255 - lum - 40, 0)
    if c == 1: return max(255 - lum - 10, 0)
    if c == 2: return min(255 - lum + 30, 255)
    return 255

def tint_neon_stripes(x, y, c):
    """Alternating neon stripes — sharp transitions for EPF/gaborish testing."""
    stripe = (y // 4) % 5
    patterns = [
        (255, 0, 128),   # hot pink
        (0, 255, 64),    # neon green
        (64, 0, 255),    # electric purple
        (255, 255, 0),   # yellow
        (0, 200, 255),   # sky blue
    ]
    r, g, b = patterns[stripe]
    noise = ((x * 7 + y * 3) % 30) - 15
    vals = [r, g, b, 255]
    return max(0, min(255, vals[c] + noise))

def tint_sunset_gradient(x, y, c, w=128, h=128):
    """Sunset gradient — warm to cool diagonal."""
    t = (x + y) / (w + h)
    if c == 0: return int(255 * (1.0 - t * 0.3))
    if c == 1: return int(128 * (1.0 - t))
    if c == 2: return int(255 * t)
    return 255

def tint_radioactive(x, y, c):
    """Radioactive green-yellow with dark patches."""
    r = int(128 + 127 * math.sin(x * 0.3 + y * 0.1))
    g = int(200 + 55 * math.cos(x * 0.1 + y * 0.2))
    b = int(30 + 25 * math.sin(x * 0.5))
    vals = [r, g, b, 255]
    return max(0, min(255, vals[c]))

def tint_deep_ocean(x, y, c):
    """Deep ocean blues with bioluminescent spots."""
    base_b = int(80 + 100 * math.sin(y * 0.05))
    spot = 1 if ((x % 17 < 2) and (y % 13 < 2)) else 0
    if c == 0: return 10 + spot * 200
    if c == 1: return 30 + spot * 255 + int(20 * math.sin(y * 0.1))
    if c == 2: return max(0, min(255, base_b + spot * 100))
    return 255

def tint_infrared(x, y, c):
    """Infrared false-color — reds and warm tones with cold blue shadows."""
    lum = int(128 + 127 * math.sin(x * 0.07 + y * 0.05))
    if c == 0: return min(255, lum + 80)
    if c == 1: return lum // 3
    if c == 2: return max(0, 200 - lum)
    return 255


def spline_image(x, y, c, w=256, h=256):
    """Thin anti-aliased curved lines on dark background — triggers spline detection."""
    bg = 15
    # Several curves at different positions
    curves = [
        lambda xx: int(h * 0.3 + 40 * math.sin(xx * 2 * math.pi / w)),
        lambda xx: int(h * 0.5 + 50 * math.sin(xx * 3 * math.pi / w + 1.0)),
        lambda xx: int(h * 0.7 + 30 * math.cos(xx * 4 * math.pi / w + 2.0)),
        lambda xx: int(h * 0.2 + 60 * math.sin(xx * 1.5 * math.pi / w - 0.5)),
    ]
    colors = [
        (255, 80, 200),   # pink line
        (80, 255, 120),   # green line
        (100, 150, 255),  # blue line
        (255, 220, 50),   # yellow line
    ]

    best_dist = 999
    best_idx = 0
    for i, curve in enumerate(curves):
        cy = curve(x)
        dist = abs(y - cy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_dist <= 1:
        # On the line
        return colors[best_idx][c] if c < 3 else 255
    elif best_dist <= 2:
        # Anti-alias fringe
        frac = 1.0 - (best_dist - 1.0)
        fg = colors[best_idx][c] if c < 3 else 255
        return int(bg + (fg - bg) * frac)
    else:
        return bg if c < 3 else 255


def patches_image(x, y, c, w=256, h=256):
    """Image with repeated small patches — triggers patch detection."""
    # 4x4 pixel patches repeated in a grid with slight variation
    patch_w, patch_h = 4, 4
    px, py = x % 32, y % 32  # tile period
    # A few distinct patches at specific tile positions
    tile_x, tile_y = (x // 32) % 4, (y // 32) % 4
    base_colors = [
        [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)],
        [(200, 50, 200), (50, 200, 200), (180, 120, 60), (60, 120, 180)],
        [(255, 128, 0), (0, 128, 255), (128, 255, 0), (128, 0, 255)],
        [(220, 220, 220), (30, 30, 30), (128, 128, 128), (200, 100, 150)],
    ]
    color = base_colors[tile_y % 4][tile_x % 4]
    noise = ((px * 3 + py * 7) % 20) - 10
    if c < 3:
        return max(0, min(255, color[c] + noise))
    return 255


def dots_image(x, y, c, w=256, h=256):
    """Sparse bright dots on dark background — triggers dots feature."""
    # Halftone-like dot pattern
    dot_spacing = 8
    cx = (x % dot_spacing) - dot_spacing // 2
    cy = (y % dot_spacing) - dot_spacing // 2
    dist_sq = cx * cx + cy * cy
    if dist_sq <= 2:
        # Dot center — bright colored
        idx = ((x // dot_spacing) + (y // dot_spacing) * 3) % 6
        dot_colors = [(255, 60, 60), (60, 255, 60), (60, 60, 255),
                      (255, 255, 60), (255, 60, 255), (60, 255, 255)]
        return dot_colors[idx][c] if c < 3 else 255
    return 20 if c < 3 else 255


def checkerboard_16bit(x, y, c, w=64, h=64):
    """16-bit checkerboard with large value swings."""
    check = ((x // 8) + (y // 8)) % 2
    if check:
        return [60000, 30000, 50000, 65535][c]
    else:
        return [5000, 40000, 10000, 65535][c]


def gradient_smooth(x, y, c, w=128, h=128):
    """Smooth gradient — good baseline for quality comparison."""
    t = x / max(w - 1, 1)
    s = y / max(h - 1, 1)
    if c == 0: return int(255 * t)
    if c == 1: return int(255 * s)
    if c == 2: return int(255 * (1.0 - t) * s)
    return 255


def high_frequency(x, y, c, w=128, h=128):
    """High-frequency noise-like pattern — stresses entropy coder."""
    v = ((x * 127 + y * 251 + c * 63) * 1103515245 + 12345) & 0xFF
    return v


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

def encode_jxl(cjxl, input_data, input_ext, output_path, options, quiet=True):
    """Write input_data to temp file, encode with cjxl, return success."""
    with tempfile.NamedTemporaryFile(suffix=input_ext, delete=False) as f:
        f.write(input_data)
        tmp = f.name
    try:
        cmd = [cjxl, tmp, str(output_path)] + options
        if quiet:
            cmd.append("--quiet")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, result.stderr[:200]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test matrix definitions
# ---------------------------------------------------------------------------

EFFORTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DISTANCES_LOSSY = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0]
DISTANCES_LOSSLESS = [0.0]
QUALITIES_SAMPLE = [10, 30, 50, 68, 75, 85, 90, 95, 100]

PREDICTORS = list(range(16))  # 0..15
RCT_COLORSPACES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 20, 30, 41]
GROUP_SIZES = [0, 1, 2, 3]
EPF_LEVELS = [0, 1, 2, 3]
RESAMPLING = [1, 2, 4, 8]
FASTER_DECODING = [0, 1, 2, 3, 4]

COLOR_SPACES = ["sRGB", "DisplayP3", "Rec2100PQ", "Rec2100HLG"]
INTENSITY_TARGETS = [100, 255, 1000, 4000, 10000]


def build_image_suite():
    """Return dict of name -> (data, extension, description)."""
    images = {}

    # --- Strange tint images (128x128 RGB) ---
    tint_fns = [
        ("magenta", tint_magenta),
        ("cyan_poison", tint_cyan_poison),
        ("sepia_inv", tint_sepia_inverted),
        ("neon_stripes", tint_neon_stripes),
        ("radioactive", tint_radioactive),
        ("deep_ocean", tint_deep_ocean),
        ("infrared", tint_infrared),
    ]
    for name, fn in tint_fns:
        images[f"tint_{name}_128"] = (
            create_ppm(128, 128, pixel_fn=fn),
            ".ppm", f"Strange tint: {name}"
        )

    # Sunset needs w/h params
    images["tint_sunset_128"] = (
        create_ppm(128, 128, pixel_fn=lambda x, y, c: tint_sunset_gradient(x, y, c, 128, 128)),
        ".ppm", "Strange tint: sunset gradient"
    )

    # --- Feature-triggering images ---
    images["splines_256"] = (
        create_ppm(256, 256, pixel_fn=lambda x, y, c: spline_image(x, y, c, 256, 256)),
        ".ppm", "Thin curves for spline detection"
    )
    images["patches_256"] = (
        create_ppm(256, 256, pixel_fn=lambda x, y, c: patches_image(x, y, c, 256, 256)),
        ".ppm", "Repeated patches for patch detection"
    )
    images["dots_256"] = (
        create_ppm(256, 256, pixel_fn=lambda x, y, c: dots_image(x, y, c, 256, 256)),
        ".ppm", "Dot pattern for dots feature"
    )

    # --- Standard test images ---
    images["gradient_128"] = (
        create_ppm(128, 128, pixel_fn=lambda x, y, c: gradient_smooth(x, y, c, 128, 128)),
        ".ppm", "Smooth gradient"
    )
    images["highfreq_128"] = (
        create_ppm(128, 128, pixel_fn=lambda x, y, c: high_frequency(x, y, c, 128, 128)),
        ".ppm", "High-frequency pattern"
    )

    # --- Different pixel formats ---
    images["gray_64"] = (
        create_pgm(64, 64),
        ".pgm", "8-bit grayscale"
    )
    images["gray_16bit_64"] = (
        create_pgm(64, 64, maxval=65535, pixel_fn=lambda x, y, c: ((x * 37 + y * 17) * 65535 // 128) % 65536),
        ".pgm", "16-bit grayscale"
    )
    images["rgba_128"] = (
        create_pam(128, 128, channels=4, pixel_fn=tint_magenta),
        ".pam", "RGBA with magenta tint"
    )
    images["grayalpha_64"] = (
        create_pam(64, 64, channels=2, pixel_fn=lambda x, y, c: ((x + y) * 2) % 256 if c == 0 else (255 - x * 2) % 256),
        ".pam", "Grayscale + alpha"
    )
    images["rgb16_128"] = (
        create_ppm(128, 128, maxval=65535,
                   pixel_fn=lambda x, y, c: checkerboard_16bit(x, y, c, 128, 128)),
        ".ppm", "16-bit RGB checkerboard"
    )
    images["indexed_32"] = (
        create_indexed_png(32, 32, num_colors=8),
        ".png", "8-color indexed PNG"
    )
    images["indexed_256"] = (
        create_indexed_png(64, 64, num_colors=256),
        ".png", "256-color indexed PNG"
    )

    # --- Size variants for group boundary testing ---
    images["tiny_8x8"] = (
        create_ppm(8, 8, pixel_fn=tint_radioactive),
        ".ppm", "Tiny 8x8"
    )
    images["small_33x17"] = (
        create_ppm(33, 17, pixel_fn=tint_cyan_poison),
        ".ppm", "Odd dimensions 33x17"
    )
    images["medium_257x129"] = (
        create_ppm(257, 129, pixel_fn=tint_neon_stripes),
        ".ppm", "Non-power-of-2 257x129"
    )
    images["wide_512x16"] = (
        create_ppm(512, 16, pixel_fn=tint_infrared),
        ".ppm", "Wide strip 512x16"
    )
    images["tall_16x512"] = (
        create_ppm(16, 512, pixel_fn=tint_deep_ocean),
        ".ppm", "Tall strip 16x512"
    )
    images["large_513x513"] = (
        create_ppm(513, 513, pixel_fn=lambda x, y, c: gradient_smooth(x, y, c, 513, 513)),
        ".ppm", "Large 513x513 (>1 group)"
    )

    return images


def generate_effort_quality_matrix(cjxl, images, out_dir, stats):
    """Every effort × representative distances, lossy and lossless."""
    section = out_dir / "effort_quality"
    section.mkdir(exist_ok=True)

    # Use a subset of images for the full matrix (it's a cross product)
    matrix_images = ["tint_magenta_128", "gradient_128", "splines_256", "gray_64"]

    distances = [0.0, 0.5, 1.0, 2.0, 5.0, 15.0]

    for img_name in matrix_images:
        if img_name not in images:
            continue
        data, ext, desc = images[img_name]
        for effort in EFFORTS:
            for dist in distances:
                d_str = f"d{dist:.1f}".replace(".", "_")
                name = f"{img_name}_e{effort}_{d_str}"
                opts = ["-e", str(effort), "-d", str(dist)]
                if dist == 0.0:
                    opts += ["-m", "1"]  # lossless needs modular
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, data, ext, path, opts)
                stats["total"] += 1
                if ok:
                    stats["ok"] += 1
                else:
                    stats["fail"] += 1
                    stats["errors"].append((name, err))
    print(f"  effort×quality matrix: {stats['ok']} ok, {stats['fail']} fail")


def generate_quality_sweep(cjxl, images, out_dir, stats):
    """Quality parameter sweep at fixed effort."""
    section = out_dir / "quality_sweep"
    section.mkdir(exist_ok=True)

    img_name = "tint_neon_stripes_128"
    data, ext, _ = images[img_name]
    effort = 7

    for q in QUALITIES_SAMPLE:
        name = f"{img_name}_e{effort}_q{q}"
        if q == 100:
            opts = ["-e", str(effort), "-d", "0.0"]
        else:
            opts = ["-e", str(effort), "-q", str(q)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok:
            stats["ok"] += 1
        else:
            stats["fail"] += 1
            stats["errors"].append((name, err))


def generate_modular_features(cjxl, images, out_dir, stats):
    """Modular mode: predictors, RCT, palette, squeeze, group sizes."""
    section = out_dir / "modular"
    section.mkdir(exist_ok=True)

    data, ext, _ = images["tint_sunset_128"]

    # All 16 predictors, lossless
    for pred in PREDICTORS:
        name = f"predictor_{pred}"
        opts = ["-d", "0", "-e", "7", "-m", "1", "-P", str(pred)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # RCT color transforms
    for rct in RCT_COLORSPACES:
        name = f"rct_{rct}"
        opts = ["-d", "0", "-e", "7", "-m", "1", "-C", str(rct)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Group sizes
    large_data, large_ext, _ = images["large_513x513"]
    for gs in GROUP_SIZES:
        name = f"group_size_{gs}"
        opts = ["-d", "0", "-e", "7", "-m", "1", "-g", str(gs)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, large_data, large_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Squeeze variants
    for squeeze in [0, 1]:
        name = f"squeeze_{squeeze}"
        opts = ["-d", "0", "-e", "7", "-m", "1", "-R", str(squeeze)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Lossy squeeze
    for squeeze in [0, 1]:
        for dist in [1.0, 4.0]:
            d_str = f"d{dist:.0f}"
            name = f"squeeze_{squeeze}_lossy_{d_str}"
            opts = ["-d", str(dist), "-e", "7", "-m", "1", "-R", str(squeeze)]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, data, ext, path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Palette variants
    for name_suffix, opts_extra in [
        ("palette_forced", ["--modular_palette_colors=256"]),
        ("palette_lossy", ["--modular_lossy_palette", "--modular_palette_colors=0"]),
        ("palette_4colors", ["--modular_palette_colors=4"]),
        ("palette_off", ["--modular_palette_colors=0"]),
    ]:
        name = name_suffix
        opts = ["-d", "0", "-e", "7", "-m", "1"] + opts_extra
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Indexed PNG palette
    for idx_name in ["indexed_32", "indexed_256"]:
        idx_data, idx_ext, _ = images[idx_name]
        name = f"{idx_name}_modular"
        opts = ["-d", "0", "-e", "7", "-m", "1"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, idx_data, idx_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # MA tree iterations
    for iters in [0, 25, 50, 100]:
        name = f"ma_iterations_{iters}"
        opts = ["-d", "0", "-e", "7", "-m", "1", "-I", str(iters)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Previous channels property
    for npc in [0, 1, 3, 7, 11]:
        name = f"prev_channels_{npc}"
        opts = ["-d", "0", "-e", "7", "-m", "1", "-E", str(npc)]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  modular features: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_vardct_features(cjxl, images, out_dir, stats):
    """VarDCT mode: EPF, gaborish, noise, patches, dots, resampling."""
    section = out_dir / "vardct"
    section.mkdir(exist_ok=True)

    base_data, base_ext, _ = images["tint_magenta_128"]

    # EPF levels
    for epf in EPF_LEVELS:
        name = f"epf_{epf}"
        opts = ["-d", "1.0", "-e", "7", f"--epf={epf}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Gaborish
    for gab in [0, 1]:
        name = f"gaborish_{gab}"
        opts = ["-d", "1.0", "-e", "7", f"--gaborish={gab}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Noise
    for noise in [0, 1]:
        name = f"noise_{noise}"
        opts = ["-d", "1.0", "-e", "7", f"--noise={noise}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Photon noise ISO levels
    for iso in [100, 400, 800, 1600, 3200, 6400, 12800]:
        name = f"photon_noise_iso{iso}"
        opts = ["-d", "1.0", "-e", "7", f"--photon_noise_iso={iso}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Patches
    patch_data, patch_ext, _ = images["patches_256"]
    for patches in [0, 1]:
        name = f"patches_{patches}"
        opts = ["-d", "1.0", "-e", "7", f"--patches={patches}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, patch_data, patch_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Dots
    dots_data, dots_ext, _ = images["dots_256"]
    for dot in [0, 1]:
        name = f"dots_{dot}"
        opts = ["-d", "1.0", "-e", "7", f"--dots={dot}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, dots_data, dots_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Resampling
    for rs in RESAMPLING:
        name = f"resampling_{rs}x"
        opts = ["-d", "2.0", "-e", "7", f"--resampling={rs}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # EC resampling with alpha
    alpha_data, alpha_ext, _ = images["rgba_128"]
    for rs in [2, 4, 8]:
        name = f"ec_resampling_{rs}x"
        opts = ["-d", "2.0", "-e", "7", f"--ec_resampling={rs}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, alpha_data, alpha_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Already downsampled + upsampling modes
    for rs in [2, 4]:
        for up_mode in [-1, 0, 1]:
            name = f"already_ds_{rs}x_up{up_mode}"
            opts = ["-d", "2.0", "-e", "7",
                    f"--resampling={rs}", "--already_downsampled",
                    f"--upsampling_mode={up_mode}"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Faster decoding levels
    for fd in FASTER_DECODING:
        name = f"faster_decoding_{fd}"
        opts = ["-d", "1.0", "-e", "7", f"--faster_decoding={fd}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # EPF × noise × gaborish × patches combinations
    for epf in [0, 2, 3]:
        for noise in [0, 1]:
            for gab in [0, 1]:
                for patches in [0, 1]:
                    name = f"combo_epf{epf}_n{noise}_g{gab}_p{patches}"
                    opts = ["-d", "1.0", "-e", "7",
                            f"--epf={epf}", f"--noise={noise}",
                            f"--gaborish={gab}", f"--patches={patches}"]
                    path = section / f"{name}.jxl"
                    ok, err = encode_jxl(cjxl, base_data, base_ext, path, opts)
                    stats["total"] += 1
                    if ok: stats["ok"] += 1
                    else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  vardct features: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_spline_tests(cjxl, images, out_dir, stats):
    """Spline-triggering images at various efforts and distances."""
    section = out_dir / "splines"
    section.mkdir(exist_ok=True)

    spline_data, spline_ext, _ = images["splines_256"]

    # Splines across efforts — higher effort = more likely to use splines
    for effort in EFFORTS:
        for dist in [0.5, 1.0, 2.0, 5.0]:
            d_str = f"d{dist:.1f}".replace(".", "_")
            name = f"splines_e{effort}_{d_str}"
            opts = ["-d", str(dist), "-e", str(effort)]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, spline_data, spline_ext, path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Also use the real libjxl splines test image if available
    splines_png = TESTDATA / "jxl" / "splines.png"
    if splines_png.exists():
        real_data = splines_png.read_bytes()
        for effort in [3, 7, 9]:
            for dist in [1.0, 3.0]:
                d_str = f"d{dist:.1f}".replace(".", "_")
                name = f"real_splines_e{effort}_{d_str}"
                opts = ["-d", str(dist), "-e", str(effort)]
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, real_data, ".png", path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Dots on splines reference
    dots_splines = TESTDATA / "dots" / "grayscale_patches_on_splines.pfm"
    if dots_splines.exists():
        ds_data = dots_splines.read_bytes()
        for effort in [5, 7, 9]:
            name = f"dots_on_splines_e{effort}"
            opts = ["-d", "1.0", "-e", str(effort)]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, ds_data, ".pfm", path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  spline tests: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_progressive_tests(cjxl, images, out_dir, stats):
    """Progressive decode modes."""
    section = out_dir / "progressive"
    section.mkdir(exist_ok=True)

    data, ext, _ = images["tint_cyan_poison_128"]

    modes = [
        ("basic", ["--progressive"]),
        ("ac", ["--progressive_ac"]),
        ("qac", ["--qprogressive_ac"]),
        ("dc0", ["--progressive_dc=0"]),
        ("dc1", ["--progressive_dc=1"]),
        ("dc2", ["--progressive_dc=2"]),
        ("group_order_center", ["--group_order=1"]),
        ("group_order_center_custom", ["--group_order=1", "--center_x=10", "--center_y=10"]),
    ]

    for mode_name, mode_opts in modes:
        for dist in [0.0, 1.0, 3.0]:
            d_str = f"d{dist:.1f}".replace(".", "_")
            name = f"prog_{mode_name}_{d_str}"
            base_opts = ["-d", str(dist), "-e", "7"]
            if dist == 0.0:
                base_opts += ["-m", "1"]
            opts = base_opts + mode_opts
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, data, ext, path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Combined progressive modes
    combos = [
        ("ac_dc1", ["--progressive_ac", "--progressive_dc=1"]),
        ("qac_dc2", ["--qprogressive_ac", "--progressive_dc=2"]),
        ("full_progressive", ["--progressive", "--progressive_dc=2"]),
    ]
    for mode_name, mode_opts in combos:
        name = f"prog_combo_{mode_name}"
        opts = ["-d", "1.0", "-e", "7"] + mode_opts
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  progressive: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_color_tests(cjxl, images, out_dir, stats):
    """Color spaces, intensity targets, HDR."""
    section = out_dir / "color"
    section.mkdir(exist_ok=True)

    data, ext, _ = images["tint_infrared_128"]

    # Color space declarations
    for cs in COLOR_SPACES:
        name = f"colorspace_{cs}"
        opts = ["-d", "1.0", "-e", "7", "-x", f"color_space={cs}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Full enumerated color encodings
    color_encs = [
        "RGB_D65_SRG_Per_SRG",
        "RGB_D65_SRG_Rel_SRG",
        "RGB_D65_SRG_Abs_SRG",
        "RGB_D65_202_Rel_PeQ",
        "RGB_D65_202_Rel_HLG",
        "RGB_D65_DCI_Per_SRG",
        "RGB_D65_Lin_Per_Lin",
    ]
    for enc in color_encs:
        safe_name = enc.replace("_", "").lower()[:30]
        name = f"enc_{safe_name}"
        opts = ["-d", "1.0", "-e", "7", "-x", f"color_space={enc}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Intensity targets
    for nits in INTENSITY_TARGETS:
        name = f"intensity_{nits}nits"
        opts = ["-d", "1.0", "-e", "7", f"--intensity_target={nits}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # HDR room image if available
    hdr_room = TESTDATA / "jxl" / "hdr_room.png"
    if hdr_room.exists():
        hdr_data = hdr_room.read_bytes()
        for cs in ["Rec2100PQ", "Rec2100HLG"]:
            for dist in [1.0, 3.0]:
                d_str = f"d{dist:.0f}"
                name = f"hdr_room_{cs}_{d_str}"
                opts = ["-d", str(dist), "-e", "7", "-x", f"color_space={cs}"]
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, hdr_data, ".png", path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  color tests: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_alpha_tests(cjxl, images, out_dir, stats):
    """Alpha channel variants."""
    section = out_dir / "alpha"
    section.mkdir(exist_ok=True)

    rgba_data, rgba_ext, _ = images["rgba_128"]
    ga_data, ga_ext, _ = images["grayalpha_64"]

    # Alpha distance combinations
    for alpha_dist in [0.0, 0.5, 1.0, 3.0]:
        for img_dist in [0.0, 1.0, 3.0]:
            ad = f"ad{alpha_dist:.1f}".replace(".", "_")
            dd = f"d{img_dist:.1f}".replace(".", "_")
            name = f"rgba_{dd}_{ad}"
            opts = ["-d", str(img_dist), "-a", str(alpha_dist), "-e", "7"]
            if img_dist == 0.0:
                opts += ["-m", "1"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, rgba_data, rgba_ext, path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Grayscale + alpha
    name = "gray_alpha_lossless"
    opts = ["-d", "0", "-e", "7", "-m", "1"]
    path = section / f"{name}.jxl"
    ok, err = encode_jxl(cjxl, ga_data, ga_ext, path, opts)
    stats["total"] += 1
    if ok: stats["ok"] += 1
    else: stats["fail"] += 1; stats["errors"].append((name, err))

    name = "gray_alpha_lossy"
    opts = ["-d", "1.0", "-e", "7"]
    path = section / f"{name}.jxl"
    ok, err = encode_jxl(cjxl, ga_data, ga_ext, path, opts)
    stats["total"] += 1
    if ok: stats["ok"] += 1
    else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Premultiply alpha
    for pm in [-1, 0, 1]:
        name = f"premultiply_{pm}"
        opts = ["-d", "1.0", "-e", "7", f"--premultiply={pm}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, rgba_data, rgba_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Keep invisible
    for ki in [0, 1]:
        name = f"keep_invisible_{ki}"
        opts = ["-d", "1.0", "-e", "7", f"--keep_invisible={ki}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, rgba_data, rgba_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  alpha tests: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_container_tests(cjxl, images, out_dir, stats):
    """Container format, codestream level, brotli, etc."""
    section = out_dir / "container"
    section.mkdir(exist_ok=True)

    data, ext, _ = images["tint_radioactive_128"]

    tests = [
        ("bare_codestream", ["-d", "1.0", "-e", "7", "--container=0"]),
        ("forced_container", ["-d", "1.0", "-e", "7", "--container=1"]),
        ("container_brotli_0", ["-d", "1.0", "-e", "7", "--container=1", "--compress_boxes=0"]),
        ("container_brotli_1", ["-d", "1.0", "-e", "7", "--container=1", "--compress_boxes=1"]),
        ("brotli_effort_1", ["-d", "1.0", "-e", "7", "--container=1", "--brotli_effort=1"]),
        ("brotli_effort_11", ["-d", "1.0", "-e", "7", "--container=1", "--brotli_effort=11"]),
        ("codestream_level_5", ["-d", "1.0", "-e", "7", "--codestream_level=5"]),
    ]

    for name, opts in tests:
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  container tests: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_bitdepth_tests(cjxl, images, out_dir, stats):
    """Bit depth and pixel format variants."""
    section = out_dir / "bitdepth"
    section.mkdir(exist_ok=True)

    # Override bit depth on 8-bit input
    data, ext, _ = images["tint_magenta_128"]
    for bd in [8, 10, 12, 16]:
        name = f"override_bd{bd}_lossy"
        opts = ["-d", "1.0", "-e", "7", f"--override_bitdepth={bd}"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Native 16-bit images
    data16, ext16, _ = images["rgb16_128"]
    for dist in [0.0, 1.0, 3.0]:
        d_str = f"d{dist:.1f}".replace(".", "_")
        name = f"native_16bit_{d_str}"
        opts = ["-d", str(dist), "-e", "7"]
        if dist == 0.0:
            opts += ["-m", "1"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data16, ext16, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # 16-bit grayscale
    g16_data, g16_ext, _ = images["gray_16bit_64"]
    name = "gray_16bit_lossless"
    opts = ["-d", "0", "-e", "7", "-m", "1"]
    path = section / f"{name}.jxl"
    ok, err = encode_jxl(cjxl, g16_data, g16_ext, path, opts)
    stats["total"] += 1
    if ok: stats["ok"] += 1
    else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Use libjxl testdata multi-depth images if available
    flower_dir = TESTDATA / "jxl" / "flower"
    if flower_dir.exists():
        for depth in [1, 2, 4, 8, 12, 16]:
            pgm = flower_dir / f"flower_small.g.depth{depth}.pgm"
            if pgm.exists():
                pgm_data = pgm.read_bytes()
                name = f"flower_gray_depth{depth}"
                opts = ["-d", "0", "-e", "5", "-m", "1"]
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, pgm_data, ".pgm", path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

        for depth in [2, 4, 8, 12, 16]:
            ppm = flower_dir / f"flower_small.rgb.depth{depth}.ppm"
            if ppm.exists():
                ppm_data = ppm.read_bytes()
                name = f"flower_rgb_depth{depth}"
                opts = ["-d", "0", "-e", "5", "-m", "1"]
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, ppm_data, ".ppm", path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

        for depth in [2, 4, 8, 12, 16]:
            pam = flower_dir / f"flower_small.rgba.depth{depth}.pam"
            if pam.exists():
                pam_data = pam.read_bytes()
                name = f"flower_rgba_depth{depth}"
                opts = ["-d", "0", "-e", "5", "-m", "1"]
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, pam_data, ".pam", path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  bitdepth tests: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_dimension_tests(cjxl, images, out_dir, stats):
    """Unusual dimensions and sizes."""
    section = out_dir / "dimensions"
    section.mkdir(exist_ok=True)

    dim_images = ["tiny_8x8", "small_33x17", "medium_257x129",
                  "wide_512x16", "tall_16x512", "large_513x513"]

    for img_name in dim_images:
        data, ext, _ = images[img_name]
        # Lossless
        name = f"{img_name}_lossless"
        opts = ["-d", "0", "-e", "7", "-m", "1"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

        # Lossy
        name = f"{img_name}_lossy"
        opts = ["-d", "1.0", "-e", "7"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, data, ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # 1x1 pixel edge case
    one_px = create_ppm(1, 1, pixel_fn=lambda x, y, c: [255, 0, 128][c])
    for mode in [("lossless", ["-d", "0", "-m", "1", "-e", "3"]),
                 ("lossy", ["-d", "1.0", "-e", "3"])]:
        name = f"1x1_{mode[0]}"
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, one_px, ".ppm", path, mode[1])
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # 1xN and Nx1
    one_col = create_ppm(1, 64, pixel_fn=tint_magenta)
    name = "1x64_lossless"
    path = section / f"{name}.jxl"
    ok, err = encode_jxl(cjxl, one_col, ".ppm", path, ["-d", "0", "-m", "1", "-e", "5"])
    stats["total"] += 1
    if ok: stats["ok"] += 1
    else: stats["fail"] += 1; stats["errors"].append((name, err))

    one_row = create_ppm(64, 1, pixel_fn=tint_cyan_poison)
    name = "64x1_lossless"
    path = section / f"{name}.jxl"
    ok, err = encode_jxl(cjxl, one_row, ".ppm", path, ["-d", "0", "-m", "1", "-e", "5"])
    stats["total"] += 1
    if ok: stats["ok"] += 1
    else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  dimension tests: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_cross_feature_combos(cjxl, images, out_dir, stats):
    """Cross-product of interesting feature combinations."""
    section = out_dir / "combos"
    section.mkdir(exist_ok=True)

    data, ext, _ = images["tint_sepia_inv_128"]

    # Resampling × EPF × progressive
    for rs in [2, 4]:
        for epf in [0, 3]:
            for prog in [False, True]:
                prog_str = "prog" if prog else "noprog"
                name = f"rs{rs}_epf{epf}_{prog_str}"
                opts = ["-d", "2.0", "-e", "7",
                        f"--resampling={rs}", f"--epf={epf}"]
                if prog:
                    opts.append("--progressive")
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, data, ext, path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Modular lossless with every predictor + squeeze
    for pred in [0, 6, 14, 15]:
        for squeeze in [0, 1]:
            for gs in [0, 3]:
                name = f"mod_p{pred}_sq{squeeze}_gs{gs}"
                opts = ["-d", "0", "-e", "7", "-m", "1",
                        f"-P", str(pred), f"-R", str(squeeze), f"-g", str(gs)]
                path = section / f"{name}.jxl"
                ok, err = encode_jxl(cjxl, data, ext, path, opts)
                stats["total"] += 1
                if ok: stats["ok"] += 1
                else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Noise + dots + patches at different qualities
    patch_data, patch_ext, _ = images["patches_256"]
    for dist in [0.5, 1.0, 4.0]:
        d_str = f"d{dist:.1f}".replace(".", "_")
        name = f"noise_dots_patches_{d_str}"
        opts = ["-d", str(dist), "-e", "9",
                "--noise=1", "--dots=1", "--patches=1"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, patch_data, patch_ext, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Alpha + resampling + progressive
    alpha_data, alpha_ext, _ = images["rgba_128"]
    for rs in [2, 4]:
        for ec_rs in [2, 4]:
            name = f"alpha_rs{rs}_ecrs{ec_rs}_prog"
            opts = ["-d", "2.0", "-e", "7",
                    f"--resampling={rs}", f"--ec_resampling={ec_rs}",
                    "--progressive"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, alpha_data, alpha_ext, path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Strange tints with extreme settings
    tint_images = ["tint_magenta_128", "tint_deep_ocean_128", "tint_radioactive_128"]
    for tint in tint_images:
        tdata, text, _ = images[tint]
        short = tint.replace("tint_", "").replace("_128", "")
        # Extreme distance
        name = f"{short}_d25"
        opts = ["-d", "25.0", "-e", "7"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, tdata, text, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

        # High noise + extreme quality
        name = f"{short}_d25_noise_iso12800"
        opts = ["-d", "25.0", "-e", "7", "--photon_noise_iso=12800"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, tdata, text, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

        # Disable perceptual optimizations
        name = f"{short}_no_perceptual"
        opts = ["-d", "1.0", "-e", "7", "--disable_perceptual_optimizations"]
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, tdata, text, path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  cross-feature combos: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def generate_animation_tests(cjxl, images, out_dir, stats):
    """Animation tests using libjxl testdata if available."""
    section = out_dir / "animation"
    section.mkdir(exist_ok=True)

    gif_path = TESTDATA / "jxl" / "traffic_light.gif"
    if not gif_path.exists():
        print("  animation: skipped (no traffic_light.gif)")
        return

    gif_data = gif_path.read_bytes()

    tests = [
        ("anim_lossless", ["-d", "0"]),
        ("anim_lossy_d1", ["-d", "1.0"]),
        ("anim_lossy_d3", ["-d", "3.0"]),
        ("anim_modular", ["-d", "0", "-m", "1"]),
        ("anim_progressive", ["-d", "1.0", "--progressive"]),
        ("anim_progressive_dc2", ["-d", "1.0", "--progressive_dc=2"]),
        ("anim_effort1", ["-d", "1.0", "-e", "1"]),
        ("anim_effort10", ["-d", "1.0", "-e", "10"]),
        ("anim_frame_indexed", ["-d", "1.0", "--frame_indexing=111"]),
        ("anim_container", ["-d", "1.0", "--container=1"]),
    ]

    for name, opts in tests:
        base_opts = ["-e", "7"] if "-e" not in " ".join(opts) else []
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, gif_data, ".gif", path, base_opts + opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Blending frames if available
    blending_dir = TESTDATA / "jxl" / "blending"
    if blending_dir.exists():
        for frame_png in sorted(blending_dir.glob("*.png"))[:4]:
            frame_data = frame_png.read_bytes()
            name = f"blend_{frame_png.stem}"
            opts = ["-d", "1.0", "-e", "5"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, frame_data, ".png", path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  animation: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


def crop_flower_regions(out_dir):
    """Crop interesting regions from flower.png using ImageMagick. Returns dict of name -> path."""
    flower = TESTDATA / "jxl" / "flower" / "flower.png"
    if not flower.exists():
        return {}

    crops = {}
    crop_dir = out_dir / "source_crops"
    crop_dir.mkdir(exist_ok=True)

    for x, y, w, h, desc in CROP_REGIONS:
        crop_path = crop_dir / f"{desc}_{w}x{h}.png"
        if crop_path.exists() and crop_path.stat().st_size > 0:
            crops[desc] = crop_path
            continue
        try:
            result = subprocess.run(
                ["convert", str(flower), "-crop", f"{w}x{h}+{x}+{y}", "+repage",
                 str(crop_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and crop_path.exists():
                crops[desc] = crop_path
            else:
                print(f"    WARN: crop {desc} failed: {result.stderr[:80]}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"    WARN: ImageMagick not available for cropping")
            break

    # Also crop from flower_alpha.png for alpha tests
    flower_alpha = TESTDATA / "jxl" / "flower" / "flower_alpha.png"
    if flower_alpha.exists():
        crop_path = crop_dir / "petals_alpha_256x256.png"
        if not crop_path.exists():
            try:
                subprocess.run(
                    ["convert", str(flower_alpha), "-crop", "256x256+800+500", "+repage",
                     str(crop_path)],
                    capture_output=True, timeout=30,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        if crop_path.exists():
            crops["petals_alpha"] = crop_path

    return crops


def generate_natural_image_sweep(cjxl, images, out_dir, stats):
    """Use cropped regions from flower for natural-image feature testing."""
    section = out_dir / "natural"
    section.mkdir(exist_ok=True)

    crops = crop_flower_regions(out_dir)
    if not crops:
        print("  natural image: skipped (no flower image or ImageMagick)")
        return

    print(f"  {len(crops)} natural image crops prepared")

    # Primary crop for comprehensive sweep
    primary_name = "petals_bokeh"
    if primary_name not in crops:
        primary_name = next(iter(crops))
    primary_path = crops[primary_name]
    primary_data = primary_path.read_bytes()

    # Full effort × distance sweep on primary crop
    for effort in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for dist in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0]:
            d_str = f"d{dist:.1f}".replace(".", "_")
            name = f"{primary_name}_e{effort}_{d_str}"
            opts = ["-d", str(dist), "-e", str(effort)]
            if dist == 0.0:
                opts += ["-m", "1"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, primary_data, ".png", path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # All crops at representative settings
    for crop_name, crop_path in crops.items():
        crop_data = crop_path.read_bytes()
        for dist in [0.0, 1.0, 3.0]:
            d_str = f"d{dist:.1f}".replace(".", "_")
            name = f"{crop_name}_{d_str}"
            opts = ["-d", str(dist), "-e", "7"]
            if dist == 0.0:
                opts += ["-m", "1"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, crop_data, ".png", path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Primary crop with every feature flag
    feature_flags = [
        ("noise", ["--noise=1"]),
        ("no_noise", ["--noise=0"]),
        ("patches", ["--patches=1"]),
        ("no_patches", ["--patches=0"]),
        ("dots", ["--dots=1"]),
        ("no_dots", ["--dots=0"]),
        ("gab", ["--gaborish=1"]),
        ("no_gab", ["--gaborish=0"]),
        ("epf0", ["--epf=0"]),
        ("epf3", ["--epf=3"]),
        ("noise_iso3200", ["--photon_noise_iso=3200"]),
        ("progressive", ["--progressive"]),
        ("progressive_dc2", ["--progressive_dc=2"]),
        ("resampling2", ["--resampling=2"]),
        ("resampling4", ["--resampling=4"]),
        ("faster1", ["--faster_decoding=1"]),
        ("faster3", ["--faster_decoding=3"]),
        ("modular_lossless", ["-d", "0", "-m", "1"]),
        ("modular_lossy", ["-d", "1.0", "-m", "1"]),
    ]
    for fname, extra_opts in feature_flags:
        name = f"{primary_name}_{fname}"
        base = ["-e", "7"]
        if not any(o.startswith("-d") for o in extra_opts):
            base += ["-d", "1.0"]
        opts = base + extra_opts
        path = section / f"{name}.jxl"
        ok, err = encode_jxl(cjxl, primary_data, ".png", path, opts)
        stats["total"] += 1
        if ok: stats["ok"] += 1
        else: stats["fail"] += 1; stats["errors"].append((name, err))

    # Alpha crop if available
    if "petals_alpha" in crops:
        alpha_data = crops["petals_alpha"].read_bytes()
        for dist in [0.0, 1.0, 3.0]:
            d_str = f"d{dist:.1f}".replace(".", "_")
            name = f"petals_alpha_{d_str}"
            opts = ["-d", str(dist), "-e", "7"]
            if dist == 0.0:
                opts += ["-m", "1"]
            path = section / f"{name}.jxl"
            ok, err = encode_jxl(cjxl, alpha_data, ".png", path, opts)
            stats["total"] += 1
            if ok: stats["ok"] += 1
            else: stats["fail"] += 1; stats["errors"].append((name, err))

    print(f"  natural image: {stats['ok']} ok, {stats['fail']} fail (cumulative)")


# ---------------------------------------------------------------------------
# Decode pass: djxl + oxipng
# ---------------------------------------------------------------------------

def decode_all_jxl(out_dir, oxipng_level="4"):
    """Decode every .jxl to .png using djxl, then compress with oxipng."""
    djxl = find_djxl()
    if not djxl:
        print("\nWARN: djxl not found, skipping decode pass")
        return

    oxipng = find_oxipng()
    if oxipng:
        print(f"\nDecode pass: djxl + oxipng -o {oxipng_level}")
    else:
        print("\nDecode pass: djxl only (oxipng not found)")

    jxl_files = sorted(out_dir.rglob("*.jxl"))
    total = len(jxl_files)
    decoded = 0
    optimized = 0
    skipped = 0
    failed = 0

    for i, jxl_path in enumerate(jxl_files):
        png_path = jxl_path.with_suffix(".png")

        # Skip if PNG already exists and is newer than JXL
        if png_path.exists() and png_path.stat().st_mtime >= jxl_path.stat().st_mtime:
            skipped += 1
            continue

        # Decode with djxl
        try:
            result = subprocess.run(
                [djxl, str(jxl_path), str(png_path), "--quiet"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                failed += 1
                if failed <= 5:
                    print(f"  WARN decode: {jxl_path.name}: {result.stderr[:60]}")
                continue
            decoded += 1
        except subprocess.TimeoutExpired:
            failed += 1
            continue

        # Optimize with oxipng
        if oxipng and png_path.exists():
            try:
                subprocess.run(
                    [oxipng, "-o", oxipng_level, "--quiet", str(png_path)],
                    capture_output=True, timeout=120,
                )
                optimized += 1
            except subprocess.TimeoutExpired:
                pass  # Keep the unoptimized PNG

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  [{i + 1}/{total}] decoded={decoded} optimized={optimized}"
                  f" skipped={skipped} failed={failed}")

    print(f"  Decode complete: {decoded} decoded, {optimized} optimized,"
          f" {skipped} skipped, {failed} failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate JXL feature corpus")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="/mnt/v/output/zenjxl-decoder/feature-corpus",
                        help="Output directory for generated JXL files")
    parser.add_argument("--section", "-s", type=str, default=None,
                        help="Only generate a specific section (effort_quality, modular, "
                             "vardct, splines, progressive, color, alpha, container, "
                             "bitdepth, dimensions, combos, animation, natural, quality_sweep)")
    parser.add_argument("--skip-decode", action="store_true",
                        help="Skip the djxl decode + oxipng pass")
    parser.add_argument("--decode-only", action="store_true",
                        help="Only run the decode pass (no encoding)")
    parser.add_argument("--oxipng-level", type=str, default="4",
                        help="oxipng optimization level (default: 4, max: max)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.decode_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        decode_all_jxl(out_dir, args.oxipng_level)
        return

    cjxl = find_cjxl()
    if not cjxl:
        print("ERROR: cjxl not found. Set CJXL_PATH or build libjxl.", file=sys.stderr)
        sys.exit(1)

    version = subprocess.run([cjxl, "--version"], capture_output=True, text=True)
    print(f"Using cjxl: {cjxl}")
    print(f"Version: {version.stdout.strip()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    print("\nBuilding input image suite...")
    images = build_image_suite()
    print(f"  {len(images)} input images prepared")

    stats = {"total": 0, "ok": 0, "fail": 0, "errors": []}

    sections = {
        "effort_quality": generate_effort_quality_matrix,
        "quality_sweep": generate_quality_sweep,
        "modular": generate_modular_features,
        "vardct": generate_vardct_features,
        "splines": generate_spline_tests,
        "progressive": generate_progressive_tests,
        "color": generate_color_tests,
        "alpha": generate_alpha_tests,
        "container": generate_container_tests,
        "bitdepth": generate_bitdepth_tests,
        "dimensions": generate_dimension_tests,
        "combos": generate_cross_feature_combos,
        "animation": generate_animation_tests,
        "natural": generate_natural_image_sweep,
    }

    if args.section:
        if args.section not in sections:
            print(f"Unknown section: {args.section}")
            print(f"Available: {', '.join(sections.keys())}")
            sys.exit(1)
        selected = {args.section: sections[args.section]}
    else:
        selected = sections

    for section_name, generator in selected.items():
        print(f"\n=== {section_name} ===")
        generator(cjxl, images, out_dir, stats)

    # Summary
    print(f"\n{'='*60}")
    print(f"TOTAL: {stats['total']} attempted, {stats['ok']} succeeded, {stats['fail']} failed")

    if stats["errors"]:
        print(f"\nFailed encodings ({len(stats['errors'])}):")
        for name, err in stats["errors"][:50]:
            print(f"  {name}: {err[:80]}")
        if len(stats["errors"]) > 50:
            print(f"  ... and {len(stats['errors']) - 50} more")

    # --- Decode pass: djxl → PNG, then oxipng ---
    if not args.skip_decode:
        decode_all_jxl(out_dir, args.oxipng_level)

    # Write manifest
    manifest = out_dir / "MANIFEST.txt"
    with open(manifest, "w") as f:
        f.write(f"# JXL Feature Corpus\n")
        f.write(f"# Generated with cjxl: {version.stdout.strip()}\n")
        f.write(f"# Total: {stats['ok']} JXL files\n\n")
        for jxl in sorted(out_dir.rglob("*.jxl")):
            rel = jxl.relative_to(out_dir)
            size = jxl.stat().st_size
            png = jxl.with_suffix(".png")
            png_size = png.stat().st_size if png.exists() else 0
            f.write(f"{rel}\t{size}\t{png_size}\n")

    print(f"\nManifest written to {manifest}")

    # Count by directory
    print("\nFiles per section:")
    for sub in sorted(out_dir.iterdir()):
        if sub.is_dir():
            jxl_count = len(list(sub.glob("*.jxl")))
            png_count = len(list(sub.glob("*.png")))
            total_jxl = sum(f.stat().st_size for f in sub.glob("*.jxl"))
            total_png = sum(f.stat().st_size for f in sub.glob("*.png"))
            print(f"  {sub.name}: {jxl_count} jxl ({total_jxl / 1024:.1f} KB)"
                  f" + {png_count} png ({total_png / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
