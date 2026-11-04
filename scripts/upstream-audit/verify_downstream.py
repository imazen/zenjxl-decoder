#!/usr/bin/env python3
"""Strict byte comparison of the cropped animation against libjxl v0.12.

Run under the shared-host heavy wrapper (see just downstream-oracle).
Outputs are preserved in a new caller-selected directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--djxl", default=os.environ.get("DJXL_PATH", "djxl"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
args.output.mkdir(parents=True, exist_ok=False)
version = subprocess.check_output([args.djxl, "--version"], stderr=subprocess.STDOUT, text=True)
if "v0.12.0 " not in version:
    raise SystemExit("Expected pinned djxl v0.12.0; got " + version)
print(version.strip(), flush=True)
env = os.environ.copy()
env["ZENJXL_CORRECTNESS_OUTPUT_DIR"] = str(args.output.resolve())
command = ["cargo", "test", "--locked", "-p", "zenjxl-decoder", "--all-features", "--lib",
           "cropped_animation_u8_pipelines_and_chunks_are_bit_exact", "--", "--test-threads=1"]
subprocess.run(command, cwd=root, env=env, check=True)
reference = args.output / "crop.apng"
oracle = [args.djxl, str(root / "zenjxl-decoder/resources/test/crop2x_small.jxl"),
          str(reference), "--num_threads=1", "--bits_per_sample=8"]
with (args.output / "djxl.log").open("xb") as log:
    subprocess.run(oracle, stdout=log, stderr=subprocess.STDOUT, check=True)
image = Image.open(reference)
assert image.n_frames == 2 and image.size == (64, 64)
results = []
for frame in range(image.n_frames):
    image.seek(frame)
    expected = image.convert("RGBA").tobytes()
    actual = (args.output / f"crop2x_small-{frame}.rgba8").read_bytes()
    assert len(actual) == len(expected)
    result = {"frame": frame, "different_bytes": sum(a != b for a, b in zip(actual, expected)),
              "max_abs_diff": max(abs(a-b) for a, b in zip(actual, expected))}
    results.append(result)
    print(json.dumps(result), flush=True)
metadata = {"decoder_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "decoder_diff": subprocess.check_output(["git", "diff", "--stat"], cwd=root, text=True),
            "version": version, "commands": [command, oracle], "results": results}
(args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
if any(r["different_bytes"] for r in results):
    raise SystemExit("Exact pixel comparison failed; retained all artifacts")
