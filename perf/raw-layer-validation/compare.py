#!/usr/bin/env python3
"""Compare the required native layer exports with independent libjxl output."""
import hashlib
import json
from array import array
import math
from pathlib import Path
import re
import subprocess
import sys

def main():
    captured, fixtures, oracle, output = map(Path, sys.argv[1:])
    source = Path(__file__).parents[2] / "zenjxl-decoder/src/tests/parity.rs"
    declaration = re.search(r"CONFORMANCE_THRESHOLD: f32 = ([^;]+);", source.read_text())
    assert declaration and declaration[1] == "1.0 / 256.0", "review changed native conformance gate"
    threshold = 1.0 / 256.0
    reports = []
    failures = []
    for name in ["multiple_layers_noise_spline.jxl", "conformance_test_images/cmyk_layers.jxl"]:
        key = name.replace("/", "_")
        native_directory = captured / key
        native = json.loads((native_directory / "layers.json").read_text())
        destination = output / key
        destination.mkdir()
        fixture = fixtures / name
        result = subprocess.run([str(oracle), str(fixture), str(destination)],
                                capture_output=True, timeout=30, check=False)
        (destination / "stdout.json").write_bytes(result.stdout)
        (destination / "stderr.txt").write_bytes(result.stderr)
        assert result.returncode == 0, (name, result.returncode, result.stderr.decode())
        independent = json.loads(result.stdout)
        assert len(native) == len(independent["layers"]), name
        entry = {"fixture": name, "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                 "libjxl": independent["libjxl"], "layers": []}
        for index, (actual, expected) in enumerate(zip(native, independent["layers"], strict=True)):
            assert actual["origin"] == expected["origin"], (name, index, "origin")
            assert actual["size"] == expected["size"], (name, index, "size")
            assert actual["planes"] == expected["planes"], (name, index, "channels")
            for field in ["blend", "extra_blends", "save_as_reference", "duration", "is_last"]:
                assert actual[field] == expected[field], (name, index, field, actual[field], expected[field])
            layer = {"index": index, "origin": actual["origin"], "size": actual["size"], "planes": []}
            for channel, filename in enumerate(actual["planes"]):
                raw_a = (native_directory / filename).read_bytes()
                raw_b = (destination / filename).read_bytes()
                a, b = array("f"), array("f")
                a.frombytes(raw_a)
                b.frombytes(raw_b)
                if sys.byteorder != "little":
                    a.byteswap()
                    b.byteswap()
                assert len(a) == len(b) == actual["size"][0] * actual["size"][1]
                max_error = 0.0
                outside = 0
                for offset, (left, right) in enumerate(zip(a, b, strict=True)):
                    assert math.isfinite(left) and math.isfinite(right)
                    error = abs(left - right)
                    max_error = max(max_error, error)
                    if error > threshold:
                        if outside == 0:
                            failures.append({"fixture": name, "layer": index, "channel": channel,
                                             "sample": offset, "native": left, "oracle": right})
                        outside += 1
                a_bits, b_bits = array("I"), array("I")
                a_bits.frombytes(raw_a)
                b_bits.frombytes(raw_b)
                assert a_bits.itemsize == a.itemsize == 4
                stats = {"channel": channel, "samples": len(a),
                         "different_float_bits": sum(left != right for left, right in zip(a_bits, b_bits, strict=True)),
                         "max_absolute_error": max_error, "over_threshold": outside}
                # Lossless CMYK and the alpha plane of this RGB fixture also
                # have exact independent evidence. Keep that stronger gate.
                exact = "cmyk" in name or channel == 3
                stats["required_bit_exact"] = exact
                if exact and stats["different_float_bits"]:
                    failures.append({"fixture": name, "layer": index, "channel": channel,
                                     "expected": "bit-exact float samples", "different": stats["different_float_bits"]})
                layer["planes"].append(stats)
            entry["layers"].append(layer)
        reports.append(entry)
    report = {"gate": "existing native F32 conformance threshold", "threshold": threshold,
              "fixtures": reports, "failures": failures,
              "limits": "Lossy RGB float differences use the existing conformance gate and are reported, not relabeled as exact agreement. CMYK/alpha samples and layer/blend metadata are exact."}
    path = output / "comparison.json"
    with path.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"report": str(path), "fixtures": len(reports), "failures": failures}))
    assert not failures, "independent local-layer samples exceeded the existing conformance gate"


if __name__ == "__main__":
    main()
