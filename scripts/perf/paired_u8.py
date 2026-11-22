#!/usr/bin/env python3
# Copyright (c) Imazen LLC.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Interleaved in-memory RGB8/RGBA8 decode comparisons with an A/A control."""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

SPEED = re.compile(r"Decoded (\d+) pixels in [\d.]+ seconds: ([\d.]+) MP/s")
WORKERS = (1, 4, 8)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def readback(binary, file, workers, options, prefix, path, log):
    """Hash the full lossless PNG readback without storing decoded rasters."""
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as stream, os.fdopen(write_fd, "wb") as output:
        path.symlink_to(f"/dev/fd/{write_fd}")
        try:
            cmd = [*prefix, str(binary), str(file), str(path),
                   "--data-type", "u16", *options]
            print(json.dumps(cmd), file=log)
            with subprocess.Popen(
                cmd, pass_fds=(write_fd,), stdout=log, stderr=log
            ) as process:
                output.close()
                header = stream.read(8)
                sha = hashlib.sha256(header)
                while chunk := stream.read(1 << 16):
                    sha.update(chunk)
                if code := process.wait():
                    raise subprocess.CalledProcessError(code, cmd)
                if header != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"missing PNG readback: {file} at {workers} workers")
                return sha.hexdigest()
        finally:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--cpus", help="Comma-separated CPU numbers in allocation order")
    parser.add_argument("--allow-level10", action="store_true")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of interleaved blocks (default: 3)")
    parser.add_argument("--cold", action="store_true",
                        help="Run each measured decode in a fresh process, with no warmup")
    parser.add_argument("--per-sample", action="store_true",
                        help="Use a fresh process and one warmup for every warm sample")
    parser.add_argument("--verify-pixels", action="store_true",
                        help="Check exact 16-bit output at every worker count before timing")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    cpus = args.cpus.split(",") if args.cpus else []
    if cpus and (len(cpus) < max(WORKERS) or len(set(cpus)) != len(cpus)):
        parser.error("--cpus needs at least eight distinct CPU numbers")
    streams = {file.resolve(): digest(file) for file in args.files}
    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()
    if digest(baseline) == digest(candidate):
        parser.error("baseline and candidate binaries must differ")
    variants = [("baseline", baseline), ("candidate", candidate), ("aa", baseline)]
    args.output.mkdir()
    individual = args.cold or args.per_sample
    repetitions = 1 if individual else 5
    samples = 5 if individual else 1
    warmup = 0 if args.cold else 1
    manifest = {
        "binaries": {str(path): digest(path) for path in (baseline, candidate)},
        "streams": {str(path): sha for path, sha in streams.items()},
        "rounds": args.rounds,
        "repetitions": 5,
        "warmup": warmup,
        "cold_process": args.cold,
        "processes_per_block": samples,
        "workers": WORKERS,
        "cpus": cpus,
        "allow_level10": args.allow_level10,
        "verify_pixels": args.verify_pixels,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sample_output = args.output / "samples.tsv"
    sample_file = sample_output.open("x", newline="", buffering=1) if individual else nullcontext()
    with (
        sample_file as sample_log,
        (args.output / "results.tsv").open("x", newline="", buffering=1) as results,
        (args.output / "commands.log").open("x", buffering=1) as log,
    ):
        writer = csv.writer(results, delimiter="\t", lineterminator="\n")
        writer.writerow(["file", "stream_sha256", "workers", "round", "variant", "pixels", "mp_s"])
        options = ["--no-cms"]
        if args.allow_level10:
            options.append("--allow-level10")

        def affinity(workers):
            return ["taskset", "-c", ",".join(cpus[:workers])] if cpus else []

        def worker_args(workers):
            return ["--num-threads", str(workers if workers > 1 else 0)]

        if args.verify_pixels:
            with (args.output / "readbacks.tsv").open("x", newline="", buffering=1) as exact:
                parity = csv.writer(exact, delimiter="\t", lineterminator="\n")
                parity.writerow(["file", "workers", "baseline_png_sha256", "candidate_png_sha256"])
                for file in streams:
                    expected = None
                    for workers in WORKERS:
                        hashes = [
                            readback(binary, file, workers, [*options, *worker_args(workers)],
                                     affinity(workers), args.output / "readback.png", log)
                            for binary in (baseline, candidate)
                        ]
                        parity.writerow([file, workers, *hashes])
                        if expected is None:
                            expected = hashes[0]
                        if hashes != [expected, expected]:
                            raise ValueError(f"decoded output differs: {file} at {workers} workers")

        shape = {}
        sample_writer = None
        if sample_log is not None:
            sample_writer = csv.writer(sample_log, delimiter="\t", lineterminator="\n")
            sample_writer.writerow(["file", "workers", "round", "variant", "sample", "pixels", "mp_s"])
        for workers, file, round_id in itertools.product(WORKERS, streams, range(args.rounds)):
            first = round_id % len(variants)
            for name, binary in variants[first:] + variants[:first]:
                cmd = [*affinity(workers), str(binary), str(file), *options, "--data-type", "u8",
                       *worker_args(workers),
                       "--speedtest", "--num-reps", str(repetitions), "--warmup-reps", str(warmup)]
                rates = []
                for sample in range(1, samples + 1):
                    print(json.dumps(cmd), file=log)
                    run = subprocess.run(cmd, capture_output=True, text=True)
                    print(run.stdout, run.stderr, sep="", file=log)
                    run.check_returncode()
                    match = SPEED.search(run.stdout)
                    if not match or int(match[1]) % repetitions or float(match[2]) <= 0:
                        raise ValueError(f"missing or invalid timing: {name} {file}\n{run.stdout}")
                    pixels = int(match[1]) // repetitions
                    if pixels != shape.setdefault(file, pixels):
                        raise ValueError(f"decoded dimensions differ: {name} {file}")
                    rate = float(match[2])
                    rates.append(rate)
                    if sample_writer is not None:
                        sample_writer.writerow([file, workers, round_id + 1, name, sample, pixels, rate])
                # The block aggregate divides total pixels by total decode time.
                speed = len(rates) / sum(1 / rate for rate in rates)
                writer.writerow([file, streams[file], workers, round_id + 1, name, pixels, f"{speed:.6f}"])


if __name__ == "__main__":
    main()
