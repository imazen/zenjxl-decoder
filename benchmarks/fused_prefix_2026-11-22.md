# Fused prefix lookups on current decoder main

Decoder baseline `e58a4838725704e37d90f2acd1d11dcccd7ab98a`, i270 AVX2 with
1, 4 and 8 pinned P cores, Rust 1.99.1 release, locked dependencies,
`--no-default-features --features all-simd,threads`. The same streams go
to both binaries. The timed endpoint is warm, allocating, in-memory u8
output with CMS disabled. Per cell, take the median of three paired
throughput ratios. Each block uses five measured decodes and a baseline
repeat as the A/A control.

| Comparison | Seven SDR fast streams, 1 / 4 / 8 workers | 16-bit fast stream, 1 / 4 / 8 | Seven controls, 1 / 4 / 8 |
|---|---|---|---|
| Fused lookup inside shared loop vs main | 0.923 / 0.907 / 0.915 | 0.967 / 1.036 / 0.981 | 0.999 / 1.004 / 0.995 |
| Dedicated one-symbol loop vs main | 1.139 / 1.096 / 1.084 | 1.140 / 1.100 / 1.104 | 1.001 / 0.997 / 1.003 |
| Dedicated two-symbol loop vs one-symbol loop | 1.083 / 1.075 / 1.114 | 0.901 / 0.907 / 0.959 | 1.001 / 0.998 / 0.991 |
| Dedicated two-symbol loop vs main | 1.233 / 1.186 / 1.179 | 1.026 / 1.040 / 1.071 | 0.999 / 1.006 / 1.001 |

The dedicated one-symbol reader fuses prefix, extra bits and signed
conversion, and retains the existing register-resident cursor for misses.
The dedicated two-symbol reader uses a wider table and packs a second
residual when both fit. Both use the existing Gradient recurrence and
restrict the fast path to a single Gradient leaf and one prefix histogram
without LZ77.

The two-symbol arm reaches median **126.91 MP/s at 1 worker** and
**609.63 MP/s at 8 workers** across the seven SDR streams in the direct
comparison with main. It misses the 1.25x relative target on these streams.
The 16-bit input favors the one-symbol loop. The largest control change
in the direct comparison is 0.891x (8-worker VarDCT d4), with worst
control A/A deviation of 5.7% at 8 workers. The worst SDR A/A deviation in the direct comparison is 8.3% at 8
workers. The per-stream results retain these individual cells.
The shared-loop candidate is rejected on these fixtures. The dedicated
candidates need a broader per-feature replay and a confirmed control
verdict. Group interleaving is a separate experiment.

All three direct comparisons reproduced identical 16-bit PNG readback
hashes for all 15 streams at all three worker counts. The two-symbol build
passed the five focused Modular decoder tests, including exact parity
against its independent reader path. The dataset is a pilot with selected
CLIC photos, a CID22 photo, screenshots, HDR, and Modular and VarDCT
controls. Source-sample checks and the independent-decoder gates belong to
profile qualification.

[Per-stream paired results](fused_prefix_2026-11-22.tsv). Full block
records, verified stream copies, commands, binary hashes and retained
patches are under `/mnt/v/output/zenjxl-decoder/fused-prefix-2026-11-22` on
the measured machine. Reproduce the comparison with
`just paired-u8 OUTPUT BASELINE CANDIDATE FILE... --cpus 2,3,0,1,4,5,6,7 --allow-level10 --verify-pixels`
using the stream hashes in the TSV. The shared-loop run precedes the
readback check. Run-heavy peak RSS in table order was 0.17, 0.18, 0.17 and
0.18 GiB. The CLI records block aggregates, so this pilot cannot settle
per-decode p95 or buffer reuse. Historical starting points are
[PR 61](https://github.com/imazen/zenjxl-decoder/pull/61) commits
`0441e81b` and `31ad13d6`.
