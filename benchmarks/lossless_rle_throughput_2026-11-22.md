# Whole-system Modular lossless throughput on i270

The baseline uses encoder `f574a3172b5009c9914e600903fb3602d74b60c6` and
encodes fixed Gradient with the existing
`--fast-decode -g 0` and decodes with zenjxl-decoder
`c266f4c7f2b602b66d5d5f678917dffd645a119f`. The candidate chooses the
padded coded size among shared and local prefix forms, with per-channel
histograms and distance-1 RLE available. It uses encoder
`7fb131664f54c8f1f15c759c5e1829310e4b0f5c` and experimental decoder
`085d9d122dd78540920d7b87503a6eeddf532c48`. The latter combines fused
plain prefix and fused RLE tokens with the row-span optimization.
[Decoder-only row-span evidence](rle_gradient_spans_2026-11-22.md).

| Input | Candidate bytes / baseline bytes | Paired speedup at 1 / 8 workers | Candidate MP/s at 1 / 8 workers |
|---|---:|---:|---:|
| Textured photo, native | 87.14% | 1.228 / 1.180 | 103.30 / 435.63 |
| Textured photo, 1024 px | 88.06% | 1.243 / 1.203 | 99.73 / 524.89 |
| Smooth photo, native | 72.28% | 1.534 / 1.372 | 148.62 / 600.16 |
| Smooth photo, 1024 px | 81.42% | 1.391 / 1.271 | 134.92 / 614.93 |
| Terminal | 17.35% | 3.027 / 1.995 | 314.65 / 977.70 |
| Aliased plot | 11.67% | 3.121 / 1.990 | 331.18 / 1018.23 |
| Mixed web page | 36.50% | 2.675 / 1.687 | 268.49 / 916.50 |
| Scanned text | 96.40% | 1.182 / 1.184 | 117.38 / 532.16 |

All eight files shrink and the terminal is under half the baseline
bytes. The native photo median is 125.96 / 517.89 MP/s, with 1.381× /
1.276× paired speedup. The four photo cells, counting the crops,
give 119.11 / 562.52 MP/s and 1.317× / 1.237×. The native eight-worker
absolute target and the pooled one-worker absolute target remain
open. The scan is another distinct weak case.

All 24 end-to-end PNG16 comparisons are exact across source, option
and worker setting. The frozen encoder-profile files also passed
independent jxl-rs and djxl v0.12 readbacks in the encoder qualification
study. The timing uses a fresh allocating RGB8 endpoint with CMS off,
one warmup per process, and five interleaved blocks of five samples
at each worker count. Stock A/A uses the same codestream and binary
to stock. The eight-worker smooth native A/A p95 ratio reaches 1.112×.
The eight-worker plot A/A is 0.844×. Both values stay in the data.

The two harness controls verify same-binary alternate-stream and
same-stream different-binary comparisons, with three exact readbacks
each. The run used 0.11 GiB peak RSS over 105 seconds.
[Per-cell bytes, hashes, medians and p95](lossless_rle_throughput_2026-11-22.tsv).
Full samples, commands and build identities are in
`/mnt/v/output/zenjxl-decoder/lossless-system-spans-2026-11-22` on i270.

## 16-group batch repeat

The same source pairs and options, with
[16 recycled groups per worker](https://github.com/imazen/zenjxl-decoder/commit/01f35c77411422dbc8da0da2f6343f40dd07a15d),
give native photo medians of 125.82 / 540.53 MP/s and paired speedups
of 1.380× / 1.341× at one and eight workers. The textured photo reaches
476.32 MP/s at eight workers and the smooth photo reaches 604.73.
Across both native and cropped sizes, photo medians are 118.84 /
549.28 MP/s. The files have the same hashes and byte counts as the
first comparison. All 15 readbacks are exact. The scan retains a
1.185× one-worker ratio and 1.164× at eight workers. The largest A/A
p95 increase in this repeat is 9.2%, on the four-worker smooth photo.

[Paired batch repeat](lossless_rle_batch16_2026-11-22.tsv). It uses
the same interleaving, warmup and endpoints as the original system
comparison and peaks at 0.12 GiB RSS over 95 seconds.
[Same-stream batch results and standalone RSS](fused_prefix_2026-11-22.md#modular-recycling-batch-size).

## Companion replay on current main

The shared/local encoder policy tries frame-wide plain Gradient prefix
tables, then frame-wide RLE tables. It accepts a shared form within
both the stock rate and 5% of the actual local-table size. The same
eight files decode faster with the experimental decoder on current
main in the claimed warm whole-system comparison. All eight are
smaller than the stock 128-pixel fast-decode streams.
[Encoder policy and i270 evidence](https://github.com/imazen/jxl-encoder/blob/3987ff45a47b5b28c5f5016ba78ec24af6053872/benchmarks/faster_decode_imazen26_lossless_2026-11-22.md#shared-and-local-prefix-selection).

The [companion experiment](https://github.com/imazen/zenjxl-decoder/commit/68bd721c5467f671e88142d4cd9a01fe58457793)
combines per-channel fused prefix reads, the row-bounded RLE kernel on
main, fused RLE tokens and 16 recycled groups per worker. The baseline
binary uses main's row-bounded RLE code. Rust 1.99.1, AVX2, and the
allocating RGB8 endpoint with CMS off are fixed. All timings use one
warmup per process and three interleaved blocks of five measured
decodes at each worker count.

| Comparison | 1 worker | 4 workers | 8 workers |
|---|---:|---:|---:|
| Same eight profile streams | 1.116× | 1.151× | 1.043× |
| Ten ordinary stream controls | 1.010× | 1.006× | 1.019× |
| Whole-system photo throughput | 129.54 MP/s | 399.62 MP/s | 569.35 MP/s |
| Whole-system photo speedup | 1.440× | 1.375× | 1.299× |
| Whole-system graphics and scan speedup | 3.025× | 2.402× | 1.752× |

The warm photo medians clear the absolute throughput targets on
this cohort, including the two native originals alone at 132.32 /
563.88 MP/s. The eight-worker median improvement on the same eight
files falls short of the 5% decoder-only gate. This bundle remains an
experiment. All 78 paired readbacks at three worker counts agree exactly
in both RGB8/RGBA8 and PNG16. Controls include the ordinary Modular
encoder, VarDCT, stock fast-decode files, odd alpha and RGB16 HDR.
Scoped formatting, strict Clippy and all 23 focused Modular tests
pass. Peak RSS from `run-heavy` was 1.51 GiB for the 115-second build and
tests, 0.23 GiB for the 259-second timing run and 0.20 GiB for the
9-second RGB8/RGBA8 output check.

The targeted nine-block repeats cover 14 cells with p95 outliers or a
control median below 0.98×. In the repeat, every ordinary-control p95
and A/A p95 is within the 10% band. The former 0.979× median on the
eight-worker native VarDCT comparison repeats at 1.011×. The
eight-worker smooth photo crop still has an A/A p95 ratio of 1.102×
on the same-file comparison, with a candidate/stock ratio of 0.994×.
The repeat used 0.20 GiB peak RSS over 36 seconds.

[Per-cell throughput and p95](lossless_rle_current_companion_2026-11-22.tsv)
and [targeted p95 repeats](lossless_rle_current_companion_tail_2026-11-22.tsv).
The full evidence is in `/mnt/v/output/zenjxl-decoder/shared-prefix-companion-2026-11-22`
on i270.
