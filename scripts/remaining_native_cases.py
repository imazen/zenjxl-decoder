#!/usr/bin/env python3
# Copyright (c) Imazen LLC. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Capture the pinned libjxl metadata, geometry and F32 pixel controls."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def pack(fields):
    bits = 0
    offset = 0
    for value, width in fields:
        assert 0 <= value < 1 << width
        bits |= value << offset
        offset += width
    return bits.to_bytes((offset + 7) // 8, "little")


def enum(value):
    return [(value, 2)] if value < 2 else [(2, 2), (value - 2, 4)]


def source_color_cases(inputs):
    # The native 4x3 RGB8 Modular image's signature, dimensions and metadata
    # prefix. This producer fixes storage, extra channels and transform data.
    prefix = [
        (0x0AFF, 16), (0, 1), (0, 2), (2, 9), (3, 3),
        (0, 1), (0, 1), (0, 1), (0, 2), (1, 1), (0, 2), (0, 1),
    ]
    tail = [(0, 2), (1, 1)]  # extensions and default transform weights
    source = (inputs / "plain-srgb.jxl").read_bytes()
    header = pack(prefix + [(1, 1)] + tail)
    assert source.startswith(header)
    for name, color_space, transfer in (
        ("explicit-srgb", 0, 13), ("unknown-space", 3, 13),
        ("unknown-transfer", 0, 2), ("unknown-both", 3, 2),
    ):
        # Explicit no-ICC descriptor, D65, sRGB primaries, explicit transfer
        # enum and relative intent. The complete frame payload is unmodified.
        color = [(0, 1), (0, 1)] + enum(color_space) + enum(1) + enum(1)
        color += [(0, 1)] + enum(transfer) + enum(1)
        data = pack(prefix + color + tail) + source[len(header):]
        (inputs / f"{name}.jxl").write_bytes(data)


def observe(binary, input_path, prefix, coalescing=True):
    with prefix.with_suffix(".log").open("x") as log:
        result = subprocess.run(
            [binary, "observe", input_path, prefix, str(int(coalescing)), "1"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    inputs = args.output / "inputs"
    binary = args.binary.resolve(strict=True)
    subprocess.run([binary, "generate", inputs], check=True)
    source_color_cases(inputs)
    for path in sorted(inputs.glob("*.jxl")):
        expected = int(path.stem.startswith("unknown-"))
        assert observe(binary, path, args.output / path.stem) == expected, path.stem
        if expected:
            assert (args.output / f"{path.stem}.log").read_text().endswith("result 1 0\n")
    assert observe(binary, inputs / "cropped-add.jxl", args.output / "cropped-add-local", False) == 0
    assert (args.output / "explicit-srgb-0.f32").read_bytes() == (
        args.output / "plain-srgb-0.f32"
    ).read_bytes()
    local_log = (args.output / "cropped-add-local.log").read_text()
    assert "frame 1 1 2 -1 1 3 2 1\n" in local_log
    assert (args.output / "cropped-add-local-1.f32").stat().st_size == 3 * 2 * 3 * 4
    sprite = Path(__file__).resolve().parents[1] / "zenjxl-decoder/resources/test/animation_seek_sprite_sheets.jxl"
    with (args.output / "sprite.log").open("x") as log:
        subprocess.run([binary, "seek", sprite, args.output / "sprite", "9"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    for frame in range(9, 12):
        assert (args.output / f"sprite-{frame}.f32").read_bytes() == (
            args.output / f"sprite-seek-{frame - 9}.f32"
        ).read_bytes()
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    (args.output / "identity.json").write_text(json.dumps({
        "oracle_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "files": {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
