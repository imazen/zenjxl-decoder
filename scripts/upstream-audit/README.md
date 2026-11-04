# Upstream (libjxl/jxl-rs) audit tools

Small helpers used for the periodic "what did upstream change
and does our output still match" audit. See `docs/UPSTREAM_SYNC.md` for the
procedure and the current ledger.

| file | purpose |
|---|---|
| `corpus_compare.sh` | decode each fixture with the fork CLI and an upstream `jxl_cli` at 16-bit, report exact pixel equality per file |
| `speed_compare.sh`  | rough interleaved single-thread `--speedtest` A/B (smoke-level; not zenbench) |
| `pngdiff.py`        | stdlib-only PNG pixel diff (8/16-bit, gray/RGB/+alpha); `FULL16=1` compares full 16-bit values |
| `prepare_feature_references.py` | prepare the two required codec-corpus diagnostic PNGs with pinned v0.12; input cache remains unchanged |
| `verify_downstream.py` | strict v0.12 decoder byte comparison; requires Pillow and a caller-provided fresh output directory |
| `groupdiff.py`      | per-256px-group map of where two decodes differ (for thread-count / tile bugs) |

The upstream CLI is built from a throwaway clone (never inside this repo):

```sh
git clone https://github.com/libjxl/jxl-rs ~/tmp/jxl-rs
(cd ~/tmp/jxl-rs && nice -n 19 cargo build --release -j 8 -p jxl_cli)
cargo build --release -p zenjxl-decoder-cli
scripts/upstream-audit/corpus_compare.sh zenjxl-decoder/resources/test/*.jxl > ~/tmp/cmp.tsv
grep -vc 'differing pixels=0/' ~/tmp/cmp.tsv   # anything but the header line is a divergence
```

Run `DJXL_PATH=/path/to/v0.12/tools/djxl just downstream-oracle /path/to/new-output`
to compare cropped animation, extra-channel, squeeze and reordered-stream RGBA
bytes with the independent decoder. The command
requires exact equality, retains diagnostics on failure, and serializes its
resource-capped build/test/oracle run with the shared heavy-job lock. Honor any
active host reservation before invoking it. Output must name a new directory so
previous measurements are preserved.

The input cache fetched by `just feature-corpus` does not contain generated PNG
references. Before the full local gate, run
`CODEC_CORPUS_PATH=/path/to/input/cache DJXL_PATH=/path/to/v0.12/djxl just feature-corpus-references /path/to/fresh-output`,
then use the printed `CODEC_CORPUS_PATH` for `just downstream-integrated-check`.
The preflight fails on either missing reference before starting the long tests.
This prepares the two diagnostic references, not a full feature parity corpus.

For cross-host comparisons, `verify_downstream.py --reference-directory <capture>`
uses a frozen v0.12 reference capture with `reference-provenance.json`. Every
input and reference image must match its SHA-256 record. Fresh runs write this
manifest automatically. This preserves strict bytes when djxl itself has a
host-dependent floating-point rounding boundary; it adds no pixel tolerance.

The per-tier diagnostic `just downstream-tier-check` reports each differing
fixture. Set `ZENJXL_TIER_OUTPUT_DIR` to an existing empty directory to retain
each executed permutation's label and decoded bytes. Its existing float
envelope remains in place; passing this gate does not establish whole-decoder
bit identity across SIMD tiers.
