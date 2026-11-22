# Weighted prediction in VarDCT LF groups

A same-stream CPU profile of decoder `6000956dc06e36b04e3b0c4ee9a527e9d6acd913`
on i270 attributes 4.39% and 4.28% of self samples to
`WpOnlyLookup<false>::decode_channel` on native d1 images 1007 and 1619.
Their HF metadata loops account for 0.29% and 0.26%. On the tier-1 d0.98
stream of 1007, Gradient accounts for 2.80% and the HF metadata loop
for 0.83%. Each sampled decode batch runs at eight workers on the same
RGB8, CMS-disabled endpoint. Profiling all three batches takes 9 s and
0.24 GiB peak RSS. Raw stacks and stream hashes are under
`/mnt/v/output/zenjxl-decoder/lf-profile-main-2026-11-22`.

## Constant coefficients

Candidate `b9c561108d7c0c15d86bca6d54c3307aa6061b3b` selects a
specialized row kernel once per channel when all seven predictor
coefficients and four weight parameters match the format defaults.
The differential test checks reconstructed rows, properties and error
histories. All eight frozen streams match their own baseline
across worker counts in the exact 16-bit PNG readback checks.

Eight-worker paired LF-group latency ratios for the two profiled
ordinary d1 streams are 0.994 and 0.972. Their whole-decode speedups
are 0.999× and 0.982×. The largest LF-group reduction over all five
VarDCT streams is 3.53%, on native 1019 d4. The optimization misses
both the 15% LF-group reduction and 5% decode throughput targets.
The small Modular RGBA control has a 1.306× p95 latency ratio at four
workers, with A/A at 1.053×. The candidate is archived under
`rejected/lf-default-wp-coefficients`.

[Decode results](lf_wp_default_2026-11-22.tsv) use three interleaved
blocks of five per-sample processes with one warmup each.
[Phase results](lf_wp_default_phase_2026-11-22.tsv) use three
interleaved blocks of eight decodes after three warmups and record
the existing LF-group wall timer, which includes HF metadata
decoding. Both comparisons use the identical input files, rotate
the baseline, candidate and A/A arms, and record RGB8 or RGBA8 with
CMS disabled. The release build uses Rust 1.99.1 with
`--no-default-features --features all-simd,threads`.
The successful build, tests and benchmark took 325 s and 4.26 GiB
peak RSS. Commands, binary and stream hashes, exact readbacks and raw
timings are under `/mnt/v/output/zenjxl-decoder/lf-wp-default-2026-11-22`
on i270.

## Exact lookup for small prediction errors

Candidate `3af4135af4a8aaf6390f1cdbe7c7090529af6826` adds two
256-entry tables to the default-coefficient kernel. All four errors
must fit the lookup. The generic arithmetic handles larger errors
and custom coefficients. Three focused tests pass, including the
full table range and large-error fallbacks. All eight streams match
baseline in the same exact readbacks at 1, 4 and 8 workers.

On native d1 photos 1007 and 1619, the paired LF-group reductions
are 12.62% and 10.70%. Whole-image decode speedups at eight workers
are 1.042× and 1.025×. The tier-1 d0.98 control, whose profile spends
more time in Gradient prediction, has a 1.014× LF latency ratio.
This change misses the 15% phase and 5% full-image gates. The
one-worker p95 on 1619 is 1.173×, with A/A at 0.991×. The small
RGBA control's eight-worker throughput is 0.931× with A/A at
0.929×. The measured candidate is archived under
`rejected/lf-default-wp-weight-table`.

The [decode](lf_wp_table_2026-11-22.tsv) and
[phase](lf_wp_table_phase_2026-11-22.tsv) runs use the same streams,
baseline, interleaving and readbacks as the coefficient experiment.
The passing build, tests and benchmark took 308 s and 4.34 GiB peak
RSS. Raw data and commands are under
`/mnt/v/output/zenjxl-decoder/lf-wp-weight-table-2026-11-22` on i270.

## Render-stage lock reference

Upstream [jxl-rs 3b67215d8](https://github.com/libjxl/jxl-rs/commit/3b67215d8)
replaces per-row RwLock acquisition in several stages with a snapshot
cached per thread. At `6000956d`, Zen's corresponding EPF, Modular XYB,
noise, patch and spline stages read frame or stage-owned data directly.
The current render code takes mutexes around group contexts and pools
of group buffers. These acquisitions occur per group, outside the row
kernels. The upstream change therefore has no corresponding per-row
lock to replace in the current Zen pipeline.

The eight-worker profiles above attribute 0.07% and 0.16% of CPU
samples to the contended standard mutex entry point on native d1
images 1007 and 1619. This profiling event measures on-CPU samples.
It provides no estimate of blocked time in the group scheduler.
