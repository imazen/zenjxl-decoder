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
