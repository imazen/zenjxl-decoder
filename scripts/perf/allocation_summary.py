#!/usr/bin/env python3
"""Summarize issue-18 zenbench runs; cross-binary deltas are exploratory."""

import argparse
import csv
import json
from pathlib import Path
import sys


def decodes(path):
    run = json.loads(path.read_text())
    groups = {
        group["group_name"]: group
        for group in run["comparisons"]
        if [bench["name"] for bench in group["benchmarks"]] == ["decode"]
    }
    return run, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow([
        "threads", "fixture", "pixels", "before_mean_ns", "after_mean_ns",
        "latency_change_pct", "before_mpix_s", "after_mpix_s",
        "before_rounds", "after_rounds", "before_unreliable", "after_unreliable",
    ])
    for threads in [1, 8]:
        before, old = decodes(args.logs / f"base-before-t{threads}.json")
        after, new = decodes(args.logs / f"candidate-v3-t{threads}.json")
        if old.keys() != new.keys():
            raise ValueError("before/after decode workloads differ")
        for name, group in old.items():
            candidate = new[name]
            if group["throughput"] != candidate["throughput"]:
                raise ValueError(f"before/after pixel counts differ: {name}")
            pixels = group["throughput"]["Elements"]
            baseline_ns = group["benchmarks"][0]["summary"]["mean"]
            candidate_ns = candidate["benchmarks"][0]["summary"]["mean"]
            writer.writerow([
                threads, name, pixels, f"{baseline_ns:.3f}", f"{candidate_ns:.3f}",
                f"{100 * (candidate_ns / baseline_ns - 1):.3f}",
                f"{pixels * 1000 / baseline_ns:.3f}",
                f"{pixels * 1000 / candidate_ns:.3f}",
                group["completed_rounds"], candidate["completed_rounds"],
                before["unreliable"], after["unreliable"],
            ])


if __name__ == "__main__":
    main()
