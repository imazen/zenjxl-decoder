#!/bin/zsh
# Decode every given .jxl with the fork CLI and an upstream jxl-rs CLI, both at
# 16-bit PNG output (u8 output is dithered upstream since jxl-rs 0.6.0, which
# would make every lossy comparison noisy), and compare pixels exactly.
#
# Usage:
#   scripts/upstream-audit/corpus_compare.sh zenjxl-decoder/resources/test/*.jxl > cmp.tsv
#
# Env:
#   FORK_CLI      path to zenjxl-decoder-cli   (default: target/release/zenjxl-decoder-cli)
#   UPSTREAM_CLI  path to upstream jxl_cli     (default: ~/tmp/jxl-rs/target/release/jxl_cli;
#                 build with: git clone https://github.com/libjxl/jxl-rs ~/tmp/jxl-rs &&
#                 cd ~/tmp/jxl-rs && nice -n 19 cargo build --release -j 8 -p jxl_cli)
#   OUT           scratch dir for the decoded PNGs (default: ~/tmp/corpus_cmp)
#   DT / BD       --data-type / --override-bitdepth passed to both CLIs (default u16 / 16)
#
# Output: TSV with file, fork exit code, upstream exit code, and the pngdiff summary
# ("differing pixels=0/N" means bit-identical).
set -u
HERE=${0:A:h}
ROOT=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/../..")
FORK=${FORK_CLI:-$ROOT/target/release/zenjxl-decoder-cli}
UP=${UPSTREAM_CLI:-$HOME/tmp/jxl-rs/target/release/jxl_cli}
OUT=${OUT:-$HOME/tmp/corpus_cmp}; mkdir -p "$OUT"
DT=${DT:-u16}; BD=${BD:-16}
printf "file\tfork_exit\tup_exit\tcompare\n"
for f in "$@"; do
  b=$(basename "$f" .jxl)
  timeout 120 nice -n 19 "$FORK" --num-threads 1 --data-type $DT --override-bitdepth $BD "$f" "$OUT/$b.fork.png" > "$OUT/$b.fork.log" 2>&1; fe=$?
  timeout 120 env RAYON_NUM_THREADS=1 nice -n 19 "$UP" --data-type $DT --override-bitdepth $BD "$f" "$OUT/$b.up.png" > "$OUT/$b.up.log" 2>&1; ue=$?
  if [[ $fe == 0 && $ue == 0 ]]; then
    cmp=$(FULL16=1 python3 "$HERE/pngdiff.py" "$OUT/$b.fork.png" "$OUT/$b.up.png" 2>&1 | tail -2 | tr '\n' ' ')
  else
    cmp="fork: $(grep -v '^$' "$OUT/$b.fork.log" | head -1 | cut -c1-60) | up: $(grep -v '^$' "$OUT/$b.up.log" | head -1 | cut -c1-60)"
  fi
  printf "%s\t%s\t%s\t%s\n" "$b" "$fe" "$ue" "$cmp"
done
