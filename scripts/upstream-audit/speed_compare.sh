#!/bin/zsh
# Rough single-thread CLI --speedtest comparison, fork vs upstream jxl-rs,
# interleaved A/B/A/B per image (2 passes x 5 reps). This is a smoke-level
# measurement for "are we far behind upstream on this file?" -- NOT a zenbench
# paired measurement; do not put these numbers in source as constants.
#
# Usage:  scripts/upstream-audit/speed_compare.sh file1.jxl file2.jxl ... > speed.tsv
# Env:    FORK_CLI, UPSTREAM_CLI as in corpus_compare.sh.
# Note:   upstream jxl_cli >= 0.6.0 is multi-threaded by default (rayon runner,
#         no --num-threads flag); RAYON_NUM_THREADS=1 pins it to one thread.
set -u
HERE=${0:A:h}
ROOT=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/../..")
FORK=${FORK_CLI:-$ROOT/target/release/zenjxl-decoder-cli}
UP=${UPSTREAM_CLI:-$HOME/tmp/jxl-rs/target/release/jxl_cli}
printf "file\tfork_MPs_p1\tup_MPs_p1\tfork_MPs_p2\tup_MPs_p2\n"
for f in "$@"; do
  b=$(basename "$f" .jxl)
  r=()
  for pass in 1 2; do
    r+=($(nice -n 19 "$FORK" --no-cms --num-threads 1 --speedtest -n 5 --warmup-reps 1 "$f" 2>&1 | grep -o '[0-9.]* MP/s' | awk '{print $1}'))
    r+=($(RAYON_NUM_THREADS=1 nice -n 19 "$UP" --speedtest -n 5 --warmup-reps 1 "$f" 2>&1 | grep -o '[0-9.]* MP/s' | awk '{print $1}'))
  done
  printf "%s\t%s\t%s\t%s\t%s\n" "$b" "$r[1]" "$r[2]" "$r[3]" "$r[4]"
done
