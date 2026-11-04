#!/usr/bin/env python3
"""Strict RGBA byte comparison of four decoder regressions against libjxl v0.12.

Run under the shared-host heavy wrapper (see just downstream-oracle).
Outputs are preserved in a new caller-selected directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--djxl", default=os.environ.get("DJXL_PATH", "djxl"))
parser.add_argument("--decoder-revision", help="Explicit source revision for a source export without Git metadata")
parser.add_argument("--reference-directory", type=Path, help="Frozen v0.12 references with input/output SHA-256 provenance")
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
           "--", "cropped_animation_u8_pipelines_and_chunks_are_bit_exact",
           "downstream_fixture_u8_pipelines_and_chunks_are_bit_exact", "--test-threads=1"]
subprocess.run(command, cwd=root, env=env, check=True)
fixtures = [
    ("crop2x_small", "apng", (64, 64), 2),
    ("ec_dim_shift_no_upsampling", "png", (64, 48), 1),
    ("squeeze_2049x257_corner_patch", "png", (2049, 257), 1),
    ("ooo_jxlp_multiframe", "apng", (32, 32), 2),
]
results = []
commands = [command]
reference_records = []
frozen = None
if args.reference_directory:
    frozen = json.loads((args.reference_directory / "reference-provenance.json").read_text())
    if "v0.12.0 " not in frozen["version"]:
        raise SystemExit("Frozen reference must come from djxl v0.12.0")
for name, extension, size, frames in fixtures:
    reference = args.output / f"{name}.{extension}"
    oracle = [args.djxl, str(root / f"zenjxl-decoder/resources/test/{name}.jxl"),
              str(reference), "--num_threads=1", "--bits_per_sample=8"]
    input_hash = hashlib.sha256((root / f"zenjxl-decoder/resources/test/{name}.jxl").read_bytes()).hexdigest()
    if frozen is None:
        commands.append(oracle)
        with (args.output / f"{name}-djxl.log").open("xb") as log:
            subprocess.run(oracle, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        record = next(record for record in frozen["references"] if record["fixture"] == name)
        source = args.reference_directory / record["reference"]
        if record["input_sha256"] != input_hash:
            raise SystemExit(f"Frozen input hash mismatch: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["reference_sha256"]:
            raise SystemExit(f"Frozen reference hash mismatch: {name}")
        shutil.copyfile(source, reference)
    reference_records.append({"fixture": name, "input_sha256": input_hash,
                              "reference": reference.name,
                              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()})
    image = Image.open(reference)
    assert image.n_frames == frames and image.size == size
    for frame in range(frames):
        image.seek(frame)
        expected = image.convert("RGBA").tobytes()
        actual = (args.output / f"{name}-{frame}.rgba8").read_bytes()
        assert len(actual) == len(expected)
        result = {"fixture": name, "frame": frame, "bytes": len(actual),
                  "different_bytes": sum(a != b for a, b in zip(actual, expected)),
                  "max_abs_diff": max(abs(a-b) for a, b in zip(actual, expected))}
        results.append(result)
        print(json.dumps(result), flush=True)
(args.output / "reference-provenance.json").write_text(json.dumps(
    {"version": frozen["version"] if frozen else version, "references": reference_records}, indent=2) + "\n")
metadata = {"frozen_reference_directory": str(args.reference_directory) if frozen else None,
            "decoder_commit": args.decoder_revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "decoder_diff": None if args.decoder_revision else subprocess.check_output(["git", "diff", "--stat"], cwd=root, text=True),
            "version": version, "commands": commands, "results": results}
(args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
if any(r["different_bytes"] for r in results):
    raise SystemExit("Exact pixel comparison failed; retained all artifacts")
