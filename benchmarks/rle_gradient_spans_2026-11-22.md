# Row-bounded Modular RLE spans on i270

The [landed decoder](https://github.com/imazen/zenjxl-decoder/commit/c266f4c7f2b602b66d5d5f678917dffd645a119f)
takes a prefix-coded distance-1 run as one span, stopping at the row
boundary. It keeps the pending repeat count for the next row or
channel. A zero-residual span copies the preceding row when the left
and top-left samples agree. For the first row, the corresponding
zero-residual loop fills with the previous sample.

| Input | Paired speedup at 1 / 4 / 8 workers | Candidate p95 ratio at 1 / 4 / 8 workers |
|---|---:|---:|
| 1007 textured photo | 1.026 / 1.020 / 1.049 | 0.977 / 0.964 / 0.964 |
| 1019 smooth photo | 1.248 / 1.167 / 1.165 | 0.803 / 0.878 / 0.893 |
| 8457 terminal | 2.066 / 1.618 / 1.397 | 0.487 / 0.634 / 0.777 |
| 7026 aliased plot | 2.161 / 1.737 / 1.444 | 0.438 / 0.575 / 1.008 |
| Mixed page, tier 0 | 1.002 / 0.998 / 1.002 | 0.864 / 0.996 / 1.001 |
| Textured photo, VarDCT d4 | 1.011 / 1.007 / 0.994 | 0.999 / 1.021 / 0.979 |
| Odd RGBA | 1.193 / 1.070 / 1.072 | 0.825 / 0.882 / 0.943 |
| Odd gray-alpha | 1.100 / 1.046 / 0.979 | 0.821 / 0.991 / 1.002 |

These are decoder-only comparisons on identical codestreams against
`45ac2482179e097b0b587519f241ae343b84d667`. All 24 PNG16 readbacks match.
The 22 focused Modular tests pass, including the RLE readback with
libjxl v0.12 files. Strict release library Clippy passes. Allocate a
fresh in-memory RGB8 output for each timing with CMS off. Use 25
individually recorded warm samples per arm and worker count, A/A
controls and the same pinned P cores for both binaries. The plot's
eight-worker A/A p95 was 1.166×. The build, checks and timing took
239 seconds with 1.52 GiB peak RSS.

The two native photos on current main measure 112.59 and 467.08 MP/s
at one and eight workers, respectively, taking the median of their
per-image throughputs. The native throughput targets remain open.

[Per-stream timings and stream hashes](rle_gradient_spans_main_2026-11-22.tsv).
Raw samples, commands and both binaries are in
`/mnt/v/output/zenjxl-decoder/rle-spans-main-2026-11-22` on i270.
