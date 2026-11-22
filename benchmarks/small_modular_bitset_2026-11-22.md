# Small Modular decision trees on i270

Candidate `f79e1057f219e8253a9c4cf6e8d08296519de857` uses a mask per
split interval to choose the leaf. It reuses Zen's property and predictor
calculations. The candidate requires 2–8 properties and at most 64
leaves, with thresholds inside the fixed lookup range. Existing simpler
tree paths retain precedence. The experiment adapts the approach in
[jxl-rs b48d1c6fc](https://github.com/libjxl/jxl-rs/commit/b48d1c6fc).

| Stream | Median paired speedup, 1 / 4 / 8 workers | p95 latency ratio, 1 / 4 / 8 workers |
|---|---:|---:|
| Scanned text, tier 0 | 1.123 / 1.124 / 1.131 | 0.813 / 0.889 / 0.888 |
| Aliased plot, tier 0 | 1.106 / 1.032 / 0.995 | 0.903 / 0.934 / 0.969 |
| Mixed page, tier 0 | 1.014 / 1.026 / 1.025 | 0.994 / 0.963 / 0.923 |
| Textured photo, VarDCT d4 | 0.997 / 1.008 / 0.994 | 1.006 / 0.998 / 1.007 |
| Smooth photo, VarDCT d4.08, tier 1 | 1.006 / 1.036 / 1.015 | 0.991 / 0.983 / 1.015 |

The scan and plot together give median throughput ratios of 1.115×,
1.078× and 1.063× at one, four and eight workers. The largest control
p95 ratio in the repeat is 1.015×. The five native VarDCT profiles
show LF-group latency ratios of 0.949–0.995 at eight workers. This
experiment misses the 15% LF setup target. Separate CPU samples hit
the mask selection instructions on the scan, plot, mixed page and
textured photo.

Both binaries rendered identical 16-bit output on 10 streams at all
three worker counts, including the fast lossless controls. The 22
focused Modular tests pass. The added tree oracle checks thresholds,
out-of-range values and repeated splits against branch traversal.
The comparison baseline is `88ce5198e3649b490875eda30b6f7eda6d5e1271`.
The timed endpoint is allocating RGB8 from in-memory input with CMS off.

[Block results](small_modular_bitset_blocks_2026-11-22.tsv) use three
interleaved blocks of five decodes. [Individual latency results](small_modular_bitset_latency_2026-11-22.tsv)
use five blocks of five processes per arm, each warming once.
[LF timings](small_modular_bitset_phase_2026-11-22.tsv) exclude three
warmups per process and use the existing phase diagnostic. All runs
include A/A. Rust 1.99.1 release with locked dependencies and
`--no-default-features --features all-simd,threads` built both binaries.
The build, tests and initial capture used 1.50 GiB peak RSS over
317 seconds. Sampling and the latency repeat used 0.20 GiB over
357 seconds. Full samples and commands are retained on i270 under
`/mnt/v/output/zenjxl-decoder/small-bitset-trees-2026-11-22`.

## Budgeted implementation on main

The [landed decoder](https://github.com/imazen/zenjxl-decoder/commit/45ac2482179e097b0b587519f241ae343b84d667)
charges the lookup tables and construction scratch to the decoder's
memory tracker. The flat path handles an unavailable reservation. The
scratch reservation drops when construction finishes. The table stays
charged until the channel ends.

The final implementation reproduces all 21 readbacks, including the
odd RGBA8 and gray-alpha fixtures. The Modular and restrictive-limit
checks pass, including the oracle's partial-reservation rollback.
Strict release library Clippy passes. The scan measures 1.115×,
1.115× and 1.125× at one, four and eight workers. The plot is 1.106×
at one worker. Its eight-worker result varies across the two builds.

The first 15-sample control has a 1.146× p95 ratio on textured VarDCT
at eight workers. The 45-sample repeat gives 0.986× with A/A at
0.995×. That repeat has a 1.108× p95 ratio on one-worker gray-alpha.
The 105-sample alpha repeat gives 1.030× there with A/A at 1.046×.
All p95 increases are within 10% in the final controls.

[Final speed and parity run](small_modular_bitset_budgeted_2026-11-22.tsv),
[45-sample controls](small_modular_bitset_controls_2026-11-22.tsv) and
[105-sample alpha controls](small_modular_bitset_alpha_2026-11-22.tsv).
The run-heavy peak RSS values are 1.52, 0.49 and 0.02 GiB respectively.
Artifacts and commands are in
`/mnt/v/output/zenjxl-decoder/small-bitset-budgeted-2026-11-22` on i270.
