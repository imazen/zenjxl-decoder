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

### Building two-symbol entries from the first-residual table

The [table-construction experiment](https://github.com/imazen/zenjxl-decoder/commit/7413bf0782e24a80c6e8131e0a4f1814c813d43f)
decodes the first residual for each window once. It builds each pair by
indexing the first-residual entry for the remaining bits and checking
that its length fits. Both builds use the same multi-context decoder
and the same six streams.

| Stream | Paired throughput at 1 / 4 / 8 workers | p95 latency at 1 / 4 / 8 workers |
|---|---:|---:|
| Textured 1007, native | 1.088× / 1.063× / 1.065× | 0.917× / 0.940× / 0.934× |
| Textured 1007, 1024 px | 1.131× / 1.100× / 1.041× | 0.877× / 0.909× / 1.175× |
| Smooth 1019, native | 1.017× / 1.027× / 0.992× | 0.980× / 0.953× / 1.015× |

The native photo medians are 104.86 and 433.11 MP/s at one and eight
workers. The shared-prefix control has a four-worker paired ratio of
0.977×. The textured crop's eight-worker p95 is 1.175× and its A/A
p95 is 1.220×. The experiment remains short of the native throughput
and control gates.

All 18 decoder and worker-count comparisons match exactly. The build
and timing run used 0.90 GiB peak RSS over 157 seconds. A separate
single-worker profile used 0.10 GiB over eight seconds and retained
raw CPU samples from both builds on both textured files.
[Paired per-stream results](fused_channel_pair_build_2026-11-22.tsv).

## Interleaving two Modular groups on i270

Candidate `2d15f44e08916571f8cf5bafe3457c2849731a4e` alternates prefix
reads for eligible Gradient channels from two independent groups.
The comparator `7413bf0782e24a80c6e8131e0a4f1814c813d43f` already has
the same prefix table, table builder and two-symbol reader. Both were
built with Rust 1.99.1 and the release flags above. The paired run
reproduced the same 16-bit readback hashes on all nine streams in
both binaries at 1, 4 and 8 workers.

| Stream | Paired speedup, 1 / 4 / 8 | Candidate p95 latency ratio, 1 / 4 / 8 |
|---|---:|---:|
| Textured photo, shared prefix | 1.070 / 1.050 / 0.992 | 0.935 / 0.883 / 1.008 |
| Smooth photo, shared prefix | 1.087 / 0.988 / 0.964 | 0.928 / 1.051 / 0.963 |
| Textured photo, per-channel | 1.013 / 0.950 / 0.923 | 0.986 / 1.079 / 1.013 |
| Smooth photo, per-channel | 1.001 / 0.955 / 0.943 | 0.996 / 1.130 / 1.052 |
| Terminal, per-channel | 1.002 / 1.004 / 0.902 | 1.015 / 1.000 / 1.004 |
| Mixed page, tier 0 | 0.999 / 0.973 / 0.988 | 1.001 / 1.017 / 1.022 |

The shared-prefix one-worker improvement repeats across both images.
The faster per-channel streams gain at most 1.3% at one worker and
lose 5.7–9.8% at eight. Their median throughput is 105.53 MP/s at one
worker and 406.89 MP/s at eight across the two photos. Their throughput
targets remain open. The smooth per-channel four-worker p95 exceeds
1.10× with a same-cell A/A of 1.038×. Some eight-worker tail controls
also drift. Do not use the candidate as a general optimization.
The shared-prefix one-worker result applies to that stream shape.

Separate CPU sampling reaches the paired kernel in both stream shapes.
The kernel accounts for 74.39% and 65.50% of user CPU samples at one and
eight workers on the textured shared-prefix file, and 32.48% and 29.60%
on its per-channel counterpart. These sample shares establish that the
paths ran. They give no direct estimate of the scheduling cost.

The [block run](fused_prefix_interleaving_blocks_2026-11-22.tsv) uses
three interleaved blocks of five decodes. The
[individual run](fused_prefix_interleaving_latency_2026-11-22.tsv) uses
five blocks of five fresh processes per arm, each warming once, with
the same allocating RGB8 endpoint and CPU affinity. It uses the nearest
rank p95 from 25 recorded latencies per cell. Both include A/A.
Run-heavy peak RSS was 0.92 GiB for the build and block run, and
0.13 GiB for the individual run and for the separate profile. Full
binary and stream hashes, exact readbacks, commands, raw timings and
named profile captures are under
`/mnt/v/output/zenjxl-decoder/interleaved-prefix-groups-2026-11-22`
on i270. The first profile invocation reused a filename through shell
expansion. The four separately named captures in that directory
supply the recorded sample shares.

## Refill on demand

Candidate `c22ba6afadcca090fd53301667f80014531e948a` refills the
register-resident cursor when fewer than 12 bits remain. The archived
comparison `7413bf0782e24a80c6e8131e0a4f1814c813d43f` attempts a
refill at every fused lookup. Both have the same decoder, table and
two-symbol prediction loop apart from that branch. The zqi prefix
microbenchmark motivated examining refill frequency.

The two native shared-prefix photos slow to 0.947× and 0.925× at
one worker, with the matching A/A estimates within 0.81% of baseline.
The textured per-channel native stream and 1024 px crop reach 0.968×
and 0.955×. All ten streams reproduce exact 16-bit readbacks at all
three worker counts. Eight-worker A/A variation reaches 32.5% on one
crop. The one-worker result rejects the conditional refill change.

[Per-stream results](fused_prefix_refill_2026-11-22.tsv) use three
interleaved blocks of five decodes on the same i270 P cores and
allocating RGB8 endpoint as the other ablations. The build and
run took 92 seconds with 0.91 GiB peak RSS. Full timing, commands,
binary and stream identities and readbacks are under
`/mnt/v/output/zenjxl-decoder/fused-prefix-refill-2026-11-22`.
