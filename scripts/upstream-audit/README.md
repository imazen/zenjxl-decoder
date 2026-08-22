# Upstream (libjxl/jxl-rs) audit tools

Small, dependency-free helpers used for the periodic "what did upstream change
and does our output still match" audit. See `docs/UPSTREAM_SYNC.md` for the
procedure and the current ledger.

| file | purpose |
|---|---|
| `corpus_compare.sh` | decode each fixture with the fork CLI and an upstream `jxl_cli` at 16-bit, report exact pixel equality per file |
| `speed_compare.sh`  | rough interleaved single-thread `--speedtest` A/B (smoke-level; not zenbench) |
| `pngdiff.py`        | stdlib-only PNG pixel diff (8/16-bit, gray/RGB/+alpha); `FULL16=1` compares full 16-bit values |
| `groupdiff.py`      | per-256px-group map of where two decodes differ (for thread-count / tile bugs) |

The upstream CLI is built from a throwaway clone (never inside this repo):

```sh
git clone https://github.com/libjxl/jxl-rs ~/tmp/jxl-rs
(cd ~/tmp/jxl-rs && nice -n 19 cargo build --release -j 8 -p jxl_cli)
cargo build --release -p zenjxl-decoder-cli
scripts/upstream-audit/corpus_compare.sh zenjxl-decoder/resources/test/*.jxl > ~/tmp/cmp.tsv
grep -vc 'differing pixels=0/' ~/tmp/cmp.tsv   # anything but the header line is a divergence
```
