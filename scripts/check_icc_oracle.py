#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Compare caller-supplied valid ICC cases with an independent libjxl build."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        image.load()
        return image.mode, image.size, image.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("list", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("djxl", type=Path)
    args = parser.parse_args()
    cases = [Path(line) for line in args.list.read_text().splitlines()]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("empty or duplicate case list")
    args.out.mkdir(exist_ok=False)
    results = []
    for source in cases:
        name = source.name
        icc = args.out / Path(name).with_suffix(".icc")
        ppm = args.out / Path(name).with_suffix(".ppm")
        process = subprocess.run([str(args.djxl), str(source), str(ppm),
                                  "--icc_out=" + str(icc), "--quiet"], capture_output=True)
        if process.returncode:
            raise RuntimeError("libjxl rejected an explicitly valid ICC case")
        native_icc = (args.native / icc.name).read_bytes()
        reference_icc = icc.read_bytes()
        native_pixels = pixels(args.native / ppm.name)
        reference_pixels = pixels(ppm)
        if native_icc != reference_icc or native_pixels != reference_pixels:
            raise AssertionError("native and independent ICC/pixel outputs differ")
        results.append({"case": name, "input_sha256": sha(source.read_bytes()),
                        "icc_bytes": len(native_icc), "icc_sha256": sha(native_icc),
                        "pixel_sha256": sha(native_pixels[2]), "status": "ok"})
        print(json.dumps({"complete": len(results), "total": len(cases), "status": "ok"}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
