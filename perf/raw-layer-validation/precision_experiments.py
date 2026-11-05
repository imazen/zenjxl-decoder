#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Test arithmetic changes in isolated exports, retaining all pixel assertions.

Caller holds the heavy lock and invokes run-heavy. Supply a fresh output
folder and the independent v0.12 C reader built by raw-layer-oracle. The
source checkout is never modified. These are diagnostic variants, not options
for the production decoder.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time


def replace(source, name, old, new):
    path = source / name
    text = path.read_text()
    assert text.count(old) == 1, (name, old)
    path.write_text(text.replace(old, new))


def fused_math(source):
    replace(source, "zenjxl-decoder/src/features/noise.rs",
            "((hi - low) * frac_x + low).clamp(0.0, 1.0)",
            "(hi - low).mul_add(frac_x, low).clamp(0.0, 1.0)")
    for color, strength in [("red", "r"), ("green", "g")]:
        replace(source, "zenjxl-decoder/src/render/stages/noise.rs",
                f"noise_strength_{strength}\n                * (k_rgn_corr * addit_rnd_noise_{color} + k_rg_corr * addit_rnd_noise_correlated)",
                f"noise_strength_{strength}\n                * k_rgn_corr.mul_add(addit_rnd_noise_{color}, k_rg_corr * addit_rnd_noise_correlated)")
    replace(source, "zenjxl-decoder/src/render/stages/noise.rs",
            "let k_rgn_corr = 0.0078125;", "let k_rgn_corr: f32 = 0.0078125;")
    replace(source, "zenjxl-decoder/src/render/stages/noise.rs",
            "row[0][x] += ytox * rg_noise + red_noise - green_noise;",
            "row[0][x] += ytox.mul_add(rg_noise, red_noise - green_noise);")
    replace(source, "zenjxl-decoder/src/render/stages/noise.rs",
            "row[2][x] += ytob * rg_noise;",
            "row[2][x] = ytob.mul_add(rg_noise, row[2][x]);")
    replace(source, "zenjxl-decoder/src/util/fast_math.rs",
            "x4 * 0.06960438 + (x2 * -0.84087373 + 1.68179268)",
            "x4.mul_add(0.06960438, x2.mul_add(-0.84087373, 1.68179268))")
    replace(source, "zenjxl-decoder/src/util/fast_math.rs",
            "cosx_prescaling * cosx_prescaling - SQRT_2",
            "cosx_prescaling.mul_add(cosx_prescaling, -SQRT_2)")
    replace(source, "zenjxl-decoder/src/util/fast_math.rs",
            "cosx_scale1 * cosx_scale1 - 1.0", "cosx_scale1.mul_add(cosx_scale1, -1.0)")

    replace(source, "zenjxl-decoder/src/util/fast_math.rs",
            "D::F32Vec::splat(d, 1.0) - inv_denom5 * inv_denom5",
            "inv_denom5.neg_mul_add(inv_denom5, D::F32Vec::splat(d, 1.0))")


def spline_reduction(source):
    # The pinned AVX2 reference accumulates eight lanes, then combines 128-bit
    # blocks and reverses four/two lanes. Keep that order explicit across tiers.
    replace(source, "zenjxl-decoder/src/features/spline.rs", """zip(self.0, precomputed.0)
            .map(|(coeff, cos)| coeff * cos)
            .sum::<f32>()
            * SQRT_2""", """{
            let mut lanes = [0.0f32; 8];
            for (i, (coeff, cos)) in zip(self.0, precomputed.0).enumerate() {
                lanes[i % 8] = SQRT_2.mul_add(coeff * cos, lanes[i % 8]);
            }
            let sums: [f32; 4] = core::array::from_fn(|i| lanes[i] + lanes[i + 4]);
            (sums[0] + sums[3]) + (sums[1] + sums[2])
        }""")


def high_precision(source):
    # Use the existing option in both the public raw path and the independent
    # native pre-composition probe. Production defaults remain unchanged.
    replace(source, "zenjxl-decoder/src/tests/raw_layers.rs",
            "let mut input = bytes;\n    let options = JxlDecoderOptions::default()\n        .with_coalescing(false)",
            "let mut input = bytes;\n    let options = JxlDecoderOptions::default()\n        .with_high_precision(true)\n        .with_coalescing(false)")
    replace(source, "zenjxl-decoder/src/api/decoder.rs",
            "let mut options = JxlDecoderOptions::default();\n        // Correctness tests should not be constrained by memory limits.",
            "let mut options = JxlDecoderOptions::default().with_high_precision(capture_raw_layers);\n        // Correctness tests should not be constrained by memory limits.")


def spline_constants(source):
    path = source / "zenjxl-decoder/src/features/spline.rs"
    text = path.read_text()
    table = "const DCT_MULTIPLIERS: [f32; 32] = [\n" + "".join(
        f"    (std::f64::consts::PI / 32.0 * {i}.0) as f32,\n" for i in range(32)) + "];"
    text, count = re.subn(r"const DCT_MULTIPLIERS: \[f32; 32\] = \[.*?\n\];", lambda _: table, text, flags=re.S)
    assert count == 1, "spline multiplier declaration changed"
    path.write_text(text)


def spline_floor(source):
    replace(source, "zenjxl-decoder/src/features/spline.rs", """[0.01, color[0], color[1], color[2]]
            .iter()
            .map(|chan| (chan * intensity).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()""", """color.into_iter().fold(0.01f32, |maximum, channel| maximum.max((channel * intensity).abs()))""")


def reference_spacing(source):
    # Reference operation order, including recomputing the remaining distance:
    # https://github.com/libjxl/libjxl/blob/a7a9c787341cf703dede03c2009fa460cae5e5df/lib/jxl/splines.cc
    # The source is BSD-licensed, Copyright (c) the JPEG XL Project Authors.
    path = source / "zenjxl-decoder/src/features/spline.rs"
    text = path.read_text()
    start = text.index("fn for_each_equally_spaced_point<")
    end = text.index("\nconst DCT_MULTIPLIERS", start)
    function = """fn for_each_equally_spaced_point<F: FnMut(Point, f32)>(
    points: &[Point], desired_distance: f32, mut f: F,
) {
    if points.is_empty() { return; }
    let mut current = points[0];
    f(current, desired_distance);
    let mut next = 0;
    while next < points.len() {
        let mut previous = current;
        let mut distance_from_previous = 0.0;
        loop {
            if next == points.len() {
                f(previous, distance_from_previous);
                return;
            }
            let delta = points[next] - previous;
            let distance_to_next = (delta.x * delta.x + delta.y * delta.y).sqrt();
            if distance_from_previous + distance_to_next >= desired_distance {
                current = previous + delta * ((desired_distance - distance_from_previous) / distance_to_next);
                f(current, desired_distance);
                break;
            }
            distance_from_previous += distance_to_next;
            previous = points[next];
            next += 1;
        }
    }
}
"""
    path.write_text(text[:start] + function + text[end:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("oracle", type=Path)
    cases = ["unchanged", "fused-math", "fused-math-spline", "high-precision", "high-precision-fused-math", "high-precision-fused-math-spline", "high-precision-constants", "high-precision-constants-fused-math-spline-floor", "high-precision-reference-spacing", "high-precision-constants-fused-math-spline-floor-reference-spacing"]
    parser.add_argument("--case", action="append", choices=cases,
                        help="select explicit experiment cases; default runs all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir()
    rows = []
    for name in args.case or cases:
        case = args.output / name
        case.mkdir()
        source = case / "source"
        shutil.copytree(root, source, ignore=shutil.ignore_patterns(
            ".git", ".jj", "target", ".workongoing", ".zenbench"))
        if "fused-math" in name:
            fused_math(source)
        if "spline" in name:
            spline_reduction(source)
        if name.startswith("high-precision"):
            high_precision(source)
        if "constants" in name:
            spline_constants(source)
        if "-floor" in name:
            spline_floor(source)
        if name.endswith("reference-spacing"):
            reference_spacing(source)
        # Source switches sharing a Cargo target must be newer than the prior
        # compilation, including exports whose archive timestamps were retained.
        for path in source.rglob("*.rs"):
            path.touch()
        captured = case / "captured"
        captured.mkdir()
        comparison = case / "comparison"
        comparison.mkdir()
        env = os.environ | {"JXL_RAW_LAYER_ARTIFACTS": str(captured)}
        env.pop("ZENJXL_ALLOW_MISSING_CORPUS", None)
        command = ["cargo", "test", "--locked", "-p", "zenjxl-decoder",
                   "--all-features", "--lib",
                   "tests::raw_layers::raw_layer_output_uses_the_public_decoder_without_compositing",
                   "--", "--exact", "--test-threads=1"]
        for stage, command in [
            ("native exact export", command),
            ("independent comparison", ["python3", "perf/raw-layer-validation/compare.py",
                                       str(captured), str(source / "zenjxl-decoder/resources/test"),
                                       str(args.oracle.resolve()), str(comparison)]),
            ("existing numeric regressions", ["cargo", "test", "--locked", "-p", "zenjxl-decoder",
                                              "--all-features", "--lib", "--", "noise", "spline",
                                              "fast_math", "--test-threads=1"]),
        ]:
            print(name, stage, flush=True)
            log = case / (stage.replace(" ", "-") + ".log")
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=source, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    (args.output / "progress.json").write_text(json.dumps({
                        "case": name, "stage": stage, "pid": child.pid,
                        "last_check": time.time(), "completed": rows,
                    }))
                    time.sleep(2)
            assert child.returncode == 0, (name, stage, child.returncode, str(log))
            if stage == "native exact export":
                text = log.read_text()
                assert "running 1 test" in text and "1 passed; 0 failed" in text, str(log)
        report = json.loads((comparison / "comparison.json").read_text())
        fixtures = []
        for fixture in report["fixtures"]:
            planes = [p for layer in fixture["layers"] for p in layer["planes"]]
            fixtures.append({"fixture": fixture["fixture"], "sha256": fixture["sha256"],
                             "samples": sum(p["samples"] for p in planes),
                             "different_float_bits": sum(p["different_float_bits"] for p in planes),
                             "max_absolute_error": max(p["max_absolute_error"] for p in planes)})
        rows.append({"case": name, "fixtures": fixtures})
        (args.output / "results.json").write_text(json.dumps({
            "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
            "results": rows,
        }, indent=2) + "\n")
    (args.output / "progress.json").write_text(json.dumps({"stage": "complete", "pid": None, "completed": rows}))


if __name__ == "__main__":
    main()
