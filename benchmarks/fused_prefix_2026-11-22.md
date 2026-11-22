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

## Per-channel histogram lookup

The [multi-context experiment](https://github.com/imazen/zenjxl-decoder/commit/c06340c393e10546404c208fb0e84e03601c7fa0)
builds up to four charged lookup tables per tree on demand. Each channel
uses the table and fallback reader for its selected cluster. LZ77 and
singleton codes retain their existing paths. The
[matched-source control](https://github.com/imazen/zenjxl-decoder/commit/6b31390fd961977511f3ab73c72d8d7839815040)
restricts table construction to single-context streams.

| Stream | Paired throughput, 1 / 4 / 8 workers | p95 latency, 1 / 4 / 8 workers |
|---|---:|---:|
| Textured 1007, 1024 px | 1.093× / 1.072× / 1.076× | 0.908× / 0.946× / 0.879× |
| Textured 1007, native | 1.051× / 1.037× / 0.977× | 0.951× / 0.991× / 0.989× |
| Smooth 1019, native | 1.012× / 1.001× / 1.028× | 0.984× / 0.990× / 1.027× |
| Terminal 8457, native | 1.005× / 1.008× / 0.959× | 1.002× / 0.984× / 1.023× |

The 1024 px textured crop gains at all three worker counts. The native
textured stream loses 2.3% median throughput at eight workers. The native
terminal loses 4.1% at eight workers, and the single-context shared-prefix
control loses 2.6% with A/A at 0.961×. The native throughput and general
adoption targets remain open.

Single-worker `perf` sampling finds `decode_residual`, used to construct
the tables, at 16.05% of sampled userspace time on the textured crop and
11.07% on its native original. On the smooth native stream, the distance-1
RLE decoder takes 63.58% of sampled userspace time and table construction
takes 2.66%.

The initial comparison against the earlier two-symbol build includes
11 unchanged input streams and exact readbacks at one, four and eight
workers. The isolated six-stream comparison adds 18 exact readback pairs
and 25 individual decodes per arm and worker count. Both use 1.99.1 release
builds and i270 P-core pinning. The initial build and run used 0.91 GiB
peak RSS over 104 seconds. The isolated test used 0.90 GiB over 159 seconds.
The separate profiler used 0.10 GiB over nine seconds.

[Isolated samples summary](fused_channel_prefix_2026-11-22.tsv).
Raw timings, both binaries, stream hashes and perf samples are in
`/mnt/v/output/zenjxl-decoder/fused-channel-contexts-2026-11-22` on i270.
