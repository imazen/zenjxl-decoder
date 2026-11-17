#!/usr/bin/env python3
# Copyright (c) Imazen LLC. BSD-3-Clause.
"""Locate upstream tests for manual assertion review; names do not prove coverage.

Scans ordinary #[test] functions and named test_all_instruction_sets! calls.
It does not expand macros, evaluate cfg, or establish assertion equivalence.
The upstream checkout/export must match --revision for the generated links.
"""

import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("upstream", type=Path)
parser.add_argument("--revision", default="f370be423933628eb7f783f1b61d4c765eca5f79")
parser.add_argument("--unmatched", action="store_true")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
names = {}
for crate in ("zenjxl-decoder", "zenjxl-decoder-simd", "zenjxl-decoder-macros"):
    for directory in ("src", "tests"):
        for path in (repo / crate / directory).rglob("*.rs"):
            for name in re.findall(r"\bfn\s+(\w+)\s*(?:<|\()", path.read_text()):
                names.setdefault(name, set()).add(str(path.relative_to(repo)))

ordinary = re.compile(r"#\[test\]\s*(?:#\[[^\n]*\]\s*)*fn\s+(\w+)\s*\(")
macro = re.compile(r"test_all_instruction_sets!\s*\(\s*(\w+)\s*\)")
cases = []
for crate in ("jxl", "jxl_cms", "jxl_simd", "jxl_transforms", "jxl-gainmap", "jxl-image-rs-integration"):
    root = args.upstream / crate
    if not root.is_dir():
        parser.error(f"missing upstream crate: {root}")
    for path in sorted(root.rglob("*.rs")):
        source = path.read_text()
        for kind, pattern in (("ordinary", ordinary), ("instruction_set_macro", macro)):
            for match in pattern.finditer(source):
                name = match.group(1)
                relative = path.relative_to(args.upstream)
                line = source.count("\n", 0, match.start()) + 1
                cases.append({
                    "kind": kind,
                    "upstream": f"https://github.com/libjxl/jxl-rs/blob/{args.revision}/{relative}#L{line}",
                    "name": name,
                    "same_name_locations": sorted(names.get(name, ())),
                })

if args.unmatched:
    print(json.dumps([case for case in cases if not case["same_name_locations"]], indent=2))
else:
    print(json.dumps({
        "revision_for_links": args.revision,
        "ordinary_functions": sum(case["kind"] == "ordinary" for case in cases),
        "named_instruction_set_macro_calls": sum(case["kind"] == "instruction_set_macro" for case in cases),
        "without_same_name": sum(not case["same_name_locations"] for case in cases),
        "meaning": "source navigation only; review renamed assertions and absent implementations manually",
    }, indent=2))
