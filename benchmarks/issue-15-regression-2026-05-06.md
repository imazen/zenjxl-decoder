# Issue #15 — decoder bug sweep, 2026-05-06

Sweep of 7,329 `.jxl` files (curated subset of `/mnt/v/output/**`,
`~/work/codec-corpus/jxl`, `~/work/codec-eval/codec-corpus/jxl`,
`~/work/third-party/jxl-rs`) decoded by `zenjxl-decoder-cli` (v0.3.8,
release build), then cross-checked against `libjxl djxl 0.12.0`.

Full per-file pass/fail TSV: `/mnt/v/output/zenjxl-decoder-bug/sweep-2026-05-06.tsv`
(581 KB, kept in block storage).

## Results

- 7,329 files attempted; 152 failures total.
- **25 failures are genuine zenjxl-decoder bugs** (libjxl djxl accepts the
  bitstream; we reject or panic).
- 127 failures are bitstreams libjxl djxl also rejects — likely jxl-encoder
  bugs producing invalid output, not decoder bugs. Out of scope for this
  repo.

## Genuine decoder-bug breakdown

| count | error                                                | notes |
|-------|------------------------------------------------------|-------|
| 15 | `Invalid AC: N nonzeros after decoding block`           | Issue #15 family. N ∈ {1, 2, 8, 9, 15}. |
| 4  | `ANS stream checksum mismatch`                          | Same screen-content + low-distance trigger. |
| 3  | `Invalid AC: nonzeros N is too large for 1 8x8 blocks`  | N ∈ {112, 144}. Different code path, likely same root cause. |
| 3  | (panic) `index out of bounds` in `low_memory_pipeline/mod.rs:409` | `spot.jxl` from libjxl conformance corpus. Default decode config; the existing in-repo `conformance::spot` test passes because it uses `render_spot_colors: false`. Separate from issue #15. |

The 22 non-panic failures all live in the issue-15 corpus at
`/mnt/v/fuzzes/zenjxl-decoder/regression/issue-15/`, paired with djxl
reference PNGs (`<name>.ref.png`) for pixel-parity checks once decode
is fixed.

## Upstream-inheritance check

Latest `libjxl/jxl-rs` (v0.4.3, commit `e7405e0`) was tested against the
46 AC-nonzeros-after files: **all 46 fail identically in upstream**.
Bug is wholly inherited from upstream jxl-rs's AC token reader.

## Trigger pattern

All 22 non-panic genuine bugs:
- Source: `jxl-encoder` (our encoder) output via `cjxl-rs-latest` and the
  earlier `akfcrc022` repro set. None are from libjxl C output.
- Content: screen-content (gb82-sc: terminal, gui, codec_wiki, gmessage,
  graph, imac, windows, windows9) and the akfcrc022 sample.
- Distance: predominantly d ≤ 2.0; nothing above d 3.0 reproduces.
- Effort: e9 reliably triggers (akfcrc022 set was sweep-limited).

The trigger surface — high effort + low distance + screen content —
matches the `enhanced_clustering_vardct` / `optimize_uint_configs_vardct`
hypothesis from the original issue.

## Regression tests

`zenjxl-decoder/src/tests/regression_known_bugs.rs` walks the corpus
directory (env `ZENJXL_REGRESSION_CORPUS` or
`/mnt/v/fuzzes/zenjxl-decoder/regression/issue-15`) and decodes every
`.jxl` it finds. Currently fails on all 22 files; turns green when the
upstream-inherited AC-reader bug is fixed.

## Companion files in this directory

- `cross-decoder-2026-05-06.tsv` — 46 AC-nonzeros-after files cross-tested
  against zenjxl-decoder, jxl-rs upstream v0.4.3, and libjxl djxl 0.12.0.
- `all-fails-vs-djxl-2026-05-06.tsv` — every one of the 152 sweep failures
  cross-tested against libjxl djxl 0.12.0.
