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
