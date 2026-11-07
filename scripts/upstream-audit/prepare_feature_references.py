#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Prepare PNG references required by codec-corpus unit controls.

Inputs come from codec-corpus. A fresh output tree links those immutable
inputs and keeps generated references and their provenance separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

CASES = [
    ("conformance", "cmyk_layers"),
    ("features", "multiple_layers_noise_spline"),
    ("edge-cases", "basic"),
    ("edge-cases", "3x3_srgb_lossless"),
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--corpus", type=Path, default=os.environ.get("CODEC_CORPUS_PATH"))
parser.add_argument("--output", type=Path)
parser.add_argument("--djxl", default=os.environ.get("DJXL_PATH", "djxl"))
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
if args.corpus is None:
    parser.error("CODEC_CORPUS_PATH is required; just feature-corpus prints the input cache root")
corpus = args.corpus.resolve()
if (corpus / "jxl").is_dir():
    corpus /= "jxl"
if not (corpus / "conformance").is_dir():
    parser.error(f"Not a codec-corpus JXL directory: {corpus}")
if args.check:
    missing = [str(corpus / "reference" / category / f"{name}.png")
               for category, name in CASES
               if not (corpus / "reference" / category / f"{name}.png").is_file()]
    if missing:
        raise SystemExit("Missing required PNG references:\n" + "\n".join(missing)
                         + "\nRun just feature-corpus-references with a fresh output directory first.")
    print("Required codec-corpus PNG references are present", flush=True)
    raise SystemExit(0)
if args.output is None:
    parser.error("--output must name a fresh directory")
version = subprocess.check_output([args.djxl, "--version"], stderr=subprocess.STDOUT, text=True)
if "v0.12.0 " not in version:
    raise SystemExit("Expected pinned djxl v0.12.0; got " + version)
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
jxl = output / "jxl"
jxl.mkdir()
for category in ["conformance", "features", "photographic", "edge-cases"]:
    source = corpus / category
    if not source.is_dir():
        raise SystemExit(f"Missing codec-corpus category: {source}")
    (jxl / category).symlink_to(source, target_is_directory=True)
records = []
for category, name in CASES:
    source = corpus / category / f"{name}.jxl"
    if not source.is_file():
        raise SystemExit(f"Missing required codec-corpus input: {source}")
    destination = jxl / "reference" / category / f"{name}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [args.djxl, str(source), str(destination), "--num_threads=1", "--bits_per_sample=8"]
    with (output / f"{name}-djxl.log").open("xb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    records.append({"input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "reference": str(destination), "reference_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "command": command})
(output / "reference-provenance.json").write_text(json.dumps({"version": version, "references": records}, indent=2) + "\n")
print(version.strip(), flush=True)
print(f"CODEC_CORPUS_PATH={output}", flush=True)
