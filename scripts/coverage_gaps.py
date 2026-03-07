#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""
Analyze coverage gaps and suggest test files to improve coverage.

Usage: python3 scripts/coverage_gaps.py
"""

import json
import os
import subprocess
from pathlib import Path

# Map of low-coverage code areas to test files that exercise them
COVERAGE_HINTS = {
    "headers/color_encoding.rs": [
        "colorspace_DisplayP3",      # Custom primaries (P3)
        "colorspace_Rec2100HLG",     # HLG transfer function
        "colorspace_Rec2100PQ",      # PQ transfer function
        "hdr_hlg_test",              # HLG
        "hdr_pq_test",               # PQ
    ],
    "frame/modular/transforms/palette.rs": [
        "delta_palette",             # Palette transform
    ],
    "container/parse.rs": [
        "container_forced",          # Container format
        "compress_boxes_0",          # Compressed boxes
        "compress_boxes_1",
        "no_container",              # Bare codestream
    ],
    "headers/bit_depth.rs": [
        "bitdepth_10",
        "bitdepth_12",
        "bitdepth_16",
    ],
    "features/blending.rs": [
        "blendmodes",
        "blendmodes_5",
        "keep_invisible_0",
        "keep_invisible_1",
    ],
    "frame/modular/predict.rs": [
        "modular_predictor_0",
        "modular_predictor_1",
        "modular_predictor_2",
        "modular_predictor_5",
        "modular_predictor_6",
        "modular_predictor_14",
        "modular_predictor_15",
    ],
    "icc/tag.rs": [
        "custom_icc_profile",
        "with_icc",
    ],
}

def get_current_tests():
    """Get list of test files currently in resources."""
    test_dir = Path("jxl/resources/test")
    tests = set()
    for jxl in test_dir.rglob("*.jxl"):
        tests.add(jxl.stem)
    return tests

def get_codec_corpus_tests():
    """Get list of test files in codec-corpus."""
    corpus_dir = Path(os.path.expanduser("~/work/codec-eval/codec-corpus/jxl"))
    if not corpus_dir.exists():
        corpus_dir = Path(os.environ.get("CODEC_CORPUS_PATH", "")) / "jxl"

    tests = {}
    for category in ["conformance", "features", "edge-cases", "photographic"]:
        cat_dir = corpus_dir / category
        if cat_dir.exists():
            for jxl in cat_dir.glob("*.jxl"):
                tests[jxl.stem] = jxl
    return tests

def load_coverage():
    """Load coverage.json if it exists."""
    try:
        with open("coverage.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def analyze_gaps():
    """Analyze coverage gaps and suggest improvements."""
    coverage = load_coverage()
    current_tests = get_current_tests()
    corpus_tests = get_codec_corpus_tests()

    print("=" * 60)
    print("COVERAGE GAP ANALYSIS")
    print("=" * 60)

    if coverage:
        print("\n## Low Coverage Files (<75% lines)")
        print("-" * 60)
        for entry in coverage['data'][0]['files']:
            path = entry['filename']
            if '/src/' in path:
                path = path.split('/src/')[1]
            ln = entry['summary']['lines']
            pct = (ln['covered'] / ln['count'] * 100) if ln['count'] > 0 else 100
            if pct < 75 and 'test' not in path and 'cli' not in path:
                print(f"  {path:45} {pct:5.1f}%")

    print("\n## Suggested Test Files to Add")
    print("-" * 60)

    missing_tests = []
    for code_file, test_files in COVERAGE_HINTS.items():
        for test_name in test_files:
            if test_name not in current_tests and test_name in corpus_tests:
                missing_tests.append((code_file, test_name, corpus_tests[test_name]))

    if missing_tests:
        for code_file, test_name, path in missing_tests:
            print(f"  {test_name:40} -> {code_file}")

        print("\n## Commands to Copy Missing Test Files")
        print("-" * 60)
        dest = Path("jxl/resources/test/coverage_boost")
        print(f"mkdir -p {dest}")
        for _, test_name, src_path in missing_tests:
            print(f"cp {src_path} {dest}/")
    else:
        print("  All suggested test files already present!")

    print("\n## Test Files in Corpus Not Currently Used")
    print("-" * 60)
    unused = set(corpus_tests.keys()) - current_tests
    # Show first 20
    for name in sorted(unused)[:20]:
        print(f"  {name}")
    if len(unused) > 20:
        print(f"  ... and {len(unused) - 20} more")

    print(f"\n  Total: {len(current_tests)} used, {len(unused)} available in corpus")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    analyze_gaps()
