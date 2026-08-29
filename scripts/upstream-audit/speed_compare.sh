#!/bin/zsh
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Rough CLI --speedtest comparison, fork vs upstream jxl-rs, interleaved
# A/B/A/B per image (2 passes x 5 reps). This is a smoke-level measurement
# for "are we far behind upstream on this file?" -- NOT a zenbench paired
# measurement; do not put these numbers in source as constants.
#
# Usage:  scripts/upstream-audit/speed_compare.sh file1.jxl file2.jxl ... > speed.tsv
# Env:    FORK_CLI, UPSTREAM_CLI as in corpus_compare.sh.
#         THREADS   thread count for both CLIs (default 1). With 1 the fork
#                   runs its sequential path (parallel = false); upstream
#                   jxl_cli is pinned with RAYON_NUM_THREADS (it gained a
#                   --num_threads flag in #904, 2026-08-25, but the env var
#                   works on every upstream version so we keep using it).
# Note:   the fork CLI is measured with --no-cms so the lcms2 stage is not
#         timed (upstream jxl_cli has no CMS). When FORK_CLI is not set the
#         CLI is (re)built first: `cargo test --no-default-features` also
#         writes target/release/zenjxl-decoder-cli, WITHOUT the `threads`
#         feature, and a binary built that way silently ignores
#         --num-threads (measured once as a bogus 5x "regression").
set -u
HERE=${0:A:h}
ROOT=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/../..")
if [[ -z ${FORK_CLI:-} ]]; then
  (cd "$ROOT" && cargo build --release -q -p zenjxl-decoder-cli) || exit 1
fi
FORK=${FORK_CLI:-$ROOT/target/release/zenjxl-decoder-cli}
UP=${UPSTREAM_CLI:-$HOME/tmp/jxl-rs/target/release/jxl_cli}
T=${THREADS:-1}
printf "file\tthreads\tfork_MPs_p1\tup_MPs_p1\tfork_MPs_p2\tup_MPs_p2\n"
for f in "$@"; do
  b=$(basename "$f" .jxl)
  r=()
  for pass in 1 2; do
    r+=($(nice -n 19 "$FORK" --no-cms --num-threads $T --speedtest -n 5 --warmup-reps 1 "$f" 2>&1 | grep -o '[0-9.]* MP/s' | awk '{print $1}'))
    r+=($(RAYON_NUM_THREADS=$T nice -n 19 "$UP" --speedtest -n 5 --warmup-reps 1 "$f" 2>&1 | grep -o '[0-9.]* MP/s' | awk '{print $1}'))
  done
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$b" "$T" "$r[1]" "$r[2]" "$r[3]" "$r[4]"
done
