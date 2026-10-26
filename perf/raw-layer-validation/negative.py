#!/usr/bin/env python3
"""Prove the independent gate rejects one changed bit in a qualified CMYK plane."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


captured, fixtures, oracle, output = map(Path, sys.argv[1:])
output.mkdir()
modified = output / "modified"
modified.mkdir()
name = "conformance_test_images_cmyk_layers.jxl"
original = captured / name / "layer-0-0.f32le"
before = hashlib.sha256(original.read_bytes()).hexdigest()
for directory in captured.iterdir():
    destination = modified / directory.name
    destination.mkdir()
    for source in directory.iterdir():
        target = destination / source.name
        if source == original:
            value = bytearray(source.read_bytes())
            value[0] ^= 1  # One mantissa bit; below the lossy float tolerance.
            target.write_bytes(value)
        else:
            target.symlink_to(source.resolve())
result_directory = output / "comparison"
result_directory.mkdir()
result = subprocess.run([
    sys.executable, str(Path(__file__).with_name("compare.py")),
    str(modified), str(fixtures), str(oracle), str(result_directory),
], capture_output=True, timeout=60, check=False)
(output / "stdout.txt").write_bytes(result.stdout)
(output / "stderr.txt").write_bytes(result.stderr)
assert result.returncode != 0, "altered exact sample was accepted"
report = json.loads((result_directory / "comparison.json").read_text())
assert report["failures"] == [{
    "fixture": "conformance_test_images/cmyk_layers.jxl", "layer": 0,
    "channel": 0, "expected": "bit-exact float samples", "different": 1,
}], report["failures"]
assert hashlib.sha256(original.read_bytes()).hexdigest() == before
print("Independent gate rejected exactly one changed CMYK mantissa bit; original artifacts unchanged.")
