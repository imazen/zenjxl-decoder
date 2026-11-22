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

## Gradient arithmetic and prefix-table entry size

Two experiments start at `7413bf0782e24a80c6e8131e0a4f1814c813d43f`.

- [Wrapping i32 Gradient](https://github.com/imazen/zenjxl-decoder/commit/6cfe2fbb14086cb4acb699313e238bbee85dae74)
  clamps the top-left sample before the wrapping sum. It passes the
  wide-arithmetic boundary oracle and 33 readback comparisons. It
  slows both native per-channel photos by 1.7–2.5% at one worker
  and the terminal and plot by 6.2–6.7%.
  [Paired results](fused_prefix_gradient_i32_2026-11-22.tsv).
- [32-bit entries](https://github.com/imazen/zenjxl-decoder/commit/555c36eb00e32124cfd598a6f7c205b7a829547d)
  pack two signed 12-bit residuals while retaining the 12-bit input
  window. Larger residuals use the regular prefix cursor. All 36
  readbacks, including the wide-channel control, match. The native
  per-channel photos improve by 0.2–1.0% at one worker and show mixed
  eight-worker results.
  [Paired results](fused_prefix_packed32_2026-11-22.tsv).

Each uses three blocks of five warm allocating RGB8 decodes with A/A on
i270 at one, four and eight workers. Peak RSS was 1.49 and 1.50 GiB,
respectively, for each build, tests and measurements. Raw files are in
`gradient-i32-2026-11-22` and `packed-prefix-2026-11-22` under the
decoder's output directory.

## RLE spans and fused RLE tokens

The [row-span experiment](https://github.com/imazen/zenjxl-decoder/commit/92f3c617dd49e95b05d2d1d186d2dfbb7d2c9ee2)
changes the RLE path only, against the same fused-prefix comparator.
The two native profile photos reach 120.16 / 514.22 MP/s at one and
eight workers. All 33 worker-count and decoder comparisons are exact.
The span change qualifies separately on [decoder main](rle_gradient_spans_2026-11-22.md).
[Span ablation](fused_prefix_rle_spans_2026-11-22.tsv).

The [fused token experiment](https://github.com/imazen/zenjxl-decoder/commit/085d9d122dd78540920d7b87503a6eeddf532c48)
adds a short lookup for literals and run lengths on top of spans.
It initializes at most one tracked table per eligible cluster. Relative
to the span build, it improves the textured and smooth native photos
by 4.5% and 5.7% at one worker. Their eight-worker changes are +4.3%
and −5.0% respectively, with A/A block spread up to 8.6%. All 33
readbacks are exact. The [whole-system study](lossless_rle_throughput_2026-11-22.md)
measures the encoder profile and fused decoder together.
[Token ablation](fused_prefix_rle_tokens_2026-11-22.tsv).

These two sets use the same block and worker protocol. Peak RSS was
1.49 and 1.50 GiB for their builds and runs. Full timings and commands
are in `rle-spans-2026-11-22` and `fused-rle-tokens-2026-11-22` under
the decoder's output directory.

## Modular recycling batch size

The prior fused-token comparator recycles after eight groups per worker.
The [16-group experiment](https://github.com/imazen/zenjxl-decoder/commit/01f35c77411422dbc8da0da2f6343f40dd07a15d)
reaches 488.49 / 595.87 MP/s on the textured and smooth native photos
at eight workers. The [32-group experiment](https://github.com/imazen/zenjxl-decoder/commit/6d066a4ae42c1e8094511f943719515081311478)
reaches 476.92 / 580.03 MP/s. Each reproduces all 33 readbacks. Several
eight-worker A/A controls drift, especially on small graphics.

Peak process RSS for the textured and smooth photos rises from
127420 / 114528 KiB to 140956 / 127924 KiB with 16 groups. The 32-group
run measures 168204 / 153620 KiB against its same-run baseline
127556 / 113828 KiB. GNU time records these RSS values on separate
same-stream runs. The larger batches remain experimental.

[16-group timings](fused_prefix_batch16_2026-11-22.tsv) and
[32-group timings](fused_prefix_batch32_2026-11-22.tsv). The build and
benchmark peak RSS values are 0.90 and 0.91 GiB. Raw timings, commands
and RSS checks are under `rle-sched-16-2026-11-22` and
`rle-sched-32-2026-11-22` on i270.

## Independent steps on the shared/local profile

The i270 replay holds the eight selected imazen-26 files fixed and
adds one decoder change at a time. The baseline is the landed
row-span decoder. The files cover shared plain prefix, shared RLE
and group-local tables. Each row uses 15 separately timed RGB8
decodes per arm at each worker count, warm in memory, with CMS off
and an interleaved A/A control. All 96 PNG16 comparisons agree.

| Added change | Matching stream forms | Paired median, 1 / 4 / 8 workers |
|---|---|---:|
| Fused one-residual lookup | Shared plain | 1.294× / 1.217× / 1.297× |
| Two residuals per lookup | Shared plain | 1.068× / 1.062× / 1.037× |
| Fused RLE tokens | Shared RLE | 1.120× / 1.068× / 0.996× |
| 16 recycled groups per worker | All eight streams | 1.000× / 1.057× / 0.996× |

The one-residual lookup accounts for most of the shared plain-prefix
gain. The incremental two-residual, RLE-token and larger-batch results
fall short of the independent eight-worker target on these files.
The scheduling experiment varies batch size. The earlier
[group-interleaving experiment](#interleaving-two-modular-groups-on-i270)
instead alternates the entropy readers within a worker.

[Per-cell step results, bytes and p95](fused_prefix_independent_steps_2026-11-22.tsv).
Build receipts, exact output hashes, commands and raw timings are under
`/mnt/v/output/zenjxl-decoder/independent-prefix-2026-11-22` on i270.
The isolated builds used 0.91 GiB peak RSS over 92 seconds. The timed
comparisons used 0.12 GiB over 212 seconds.

### Compact lookup and ordinary stream controls

The proposed production change limits the lookup to channels of at least
4,096 samples that use a shared tree. It packs the first residual and its
bit length into a 32-bit entry. A table consumes 16 KiB of the existing
memory budget. Ineligible histograms, singleton codes and long residuals
use the established decoder paths.

The [compact experiment](https://github.com/imazen/zenjxl-decoder/commit/a25ed33bfe9dce54e5f1951d5832bd864e044552)
reaches 1.282× / 1.200× / 1.139× on the shared plain streams. Its long
eight-worker repeat on RGBA RLE reaches 0.958× throughput and 1.124×
p95 latency. Moving the new path behind an out-of-line call
[recovers RGBA](https://github.com/imazen/zenjxl-decoder/commit/4d3e13f9d335c3923b701d9034cc16ab793a111f).
In the nine-block repeat the RGBA control reaches 1.007× throughput and
0.993× p95. The same source still regresses on other valid files.

| Out-of-line path | 1 worker | 4 workers | 8 workers |
|---|---:|---:|---:|
| Shared plain, eight-file profile replay subset | 1.288× | 1.237× | 1.226× |
| Ten ordinary controls, warm median | 0.999× | 1.002× | 1.019× |

Both revisions match all 54 warm pairs exactly in PNG16 and RGB8 or
RGBA8 at one, four and eight workers. Scoped formatting, strict Clippy
and 23 focused Modular tests pass. The longer out-of-line repeats find
the native smooth local-table file at 0.929× eight-worker throughput
and 1.143× p95, with A/A p95 1.077×. The local-table plot reaches
0.943× throughput. The shared-RLE mixed page p95 remains 1.132×.
These regressions leave both revisions on experiment branches. Warm
and cold measurements use three interleaved blocks of five decodes per
arm. The diagnosed tails use nine or 21 blocks as recorded in the run.

[Per-cell attribution and control effects](fused_prefix_compact_controls_2026-11-22.tsv)
and [p95 repeats](fused_prefix_compact_tails_2026-11-22.tsv).
The compact test build peaked at 1.51 GiB over 99 seconds and its
timing run at 0.23 GiB over 220 seconds. The full out-of-line build
and comparison used 1.51 GiB peak RSS over 295 seconds. Its targeted
repeats used 0.07 GiB over 47 seconds. Exact logs and binary hashes
remain in the same i270 run directory.

## Frame-owned lookup for shared Gradient prefix streams

The [frame-owned experiment](https://github.com/imazen/zenjxl-decoder/commit/e8af08b565ef57a78d70007ae7583a91a85ce399)
caches up to four one-residual tables on the frame-global MA tree.
Eligible trees use prefix coding without LZ77, Gradient leaves with unit
multipliers and zero offsets, and splits on channel or stream. The table
uses the regular cursor for residuals outside its bit window. The
group-local tree layout and channel loop retain their existing code.

| Identical streams | 1 worker | 4 workers | 8 workers |
|---|---:|---:|---:|
| Seven photo streams, median paired speedup | 1.142× | 1.124× | 1.111× |
| Ten ordinary controls, median paired speedup | 0.997× | 1.001× | 1.002× |
| Detailed 12 MP photo, shared plain prefix | 1.289× | 1.259× | 1.219× |
| Smooth 12 MP photo, local RLE | 0.995× | 1.005× | 1.024× |

All 51 warm pairs match in PNG16 at their respective worker counts.
The focused Modular suite passes all 23 tests, including checks that both
fused and regular cursor paths run and return identical frames. Scoped
formatting and strict library Clippy pass. The largest control p95 ratios
are 1.199× for a 64 px photo at one worker, 1.118× for a plot at four and
1.127× for an odd RGBA control at eight. The revision remains experimental
while those tails and the caller-buffer endpoint are evaluated.

The reference is `c266f4c7` and the release binary is `2fd15d5a`.
The experiment bookmark adds test-only controls. These are three blocks
of five interleaved warm allocating RGB8/RGBA8 decodes on i270, with A/A.
The release build, library Clippy and paired capture used 0.91 GiB peak
RSS over 231 seconds under `run-heavy`.
[Per-cell timing, p95 and stream hashes](fused_prefix_frame_owned_2026-11-22.tsv).

### Frame-owned confirmation and integration

The initial tail repeats use nine blocks of five and bring every
flagged p95 at or below 1.095×. An additional 21-block odd RGBA repeat
measures 1.036× paired throughput and 0.824× p95 at eight workers,
with A/A at 0.985× throughput and 1.034× p95.

The caller-owned decode output confirms the textured 256 px group
file gains at 1/4/8 workers with both decoder recreation and rewind.
The two RLE photographs show smaller changes. Each pair matches RGB8
output across all endpoint variants, and matches the reference binary
on the independent PNG8 and PNG16 readbacks.

| Identical 12 MP textured stream, 256 px groups | 1 worker | 4 workers | 8 workers |
|---|---:|---:|---:|
| Reuse caller's output buffer | 1.322× | 1.239× | 1.146× |
| Reuse output and rewind decoder | 1.320× | 1.217× | 1.160× |
| Final production CLI, warm allocating | 1.295× | 1.207× | 1.123× |

On fresh processes the textured native 128 px group file reaches
1.274× / 1.192× / 1.172× at 1/4/8 workers. The smooth crop's
four-worker p95 is 0.979× on the nine-block repeat with the final
production binary. The final four- and eight-worker plot controls
had initial p95 outliers. They repeat at 0.995× and 0.980× p95
over 21 blocks. The plot's eight-worker throughput ratio is
0.979× in that repeat.

Release format and strict Clippy pass, including the profiling feature.
The two-core library run passed 995 tests. Its 103 failures all require
the external feature corpus. Rerunning those cases with the local
`CODEC_CORPUS_PATH` passes all 103, with the existing ignores retained.
The fixture test runs on two cores to bound its internal worker count.
Both initial 8 GiB wrapper runs were stopped by the memory cgroup.
The two-core run and encoder replay used 5.06 GiB peak RSS under
the 12 GiB wrapper. No source expectations changed.

[Warm and cold confirmations](fused_prefix_frame_owned_confirmation_2026-11-22.tsv).
[Caller-buffer output and rewind results](fused_prefix_frame_owned_reuse_2026-11-22.tsv).
Raw samples, release binary hashes, source commits and build commands
are retained in the same i270 run directory.

## Frame-owned RLE tables on the native photo holdout

This variant uses the historical 10-bit literal/run lookup in the shared
fixed-Gradient path. For each eligible frame it lazily builds at most four
4 KiB tables under the existing memory tracker. Run state and reconstruction
continue to use row-bounded spans. The generic loop remains available for
local tables, longer tokens and short input windows. The underlying lookup
comes from the earlier [fused token experiment](https://github.com/imazen/zenjxl-decoder/commit/085d9d122dd78540920d7b87503a6eeddf532c48).

On i270, the same-stream baseline has the boundary scheduling change
in `ff36edc0`. It has the same decode source as `866472df`, the parent
of the measured candidate `1d56a4dba582f3793c92622bcc0a1ce9a3f2fbdd`.
Both binaries use Rust 1.99.1 and locked release dependencies with
`all-simd,threads`. Each endpoint is allocating RGB8 or RGBA8 with
CMS off, pinned P cores, three interleaved blocks of five decodes,
and baseline A/A.

| Same profile file | Cold speedup at 1 / 4 / 8 workers | Warm speedup at 1 / 4 / 8 workers |
|---|---:|---:|
| 1009 cityscape, shared RLE | 1.203× / 1.143× / 1.116× | 1.215× / 1.163× / 1.149× |
| 1427 macro, shared RLE | 1.165× / 1.116× / 1.078× | 1.174× / 1.148× / 1.027× |
| 1447 glacier, regular RLE | 1.001× / 0.985× / 1.017× | 1.001× / 1.005× / 0.932× |

The nine-block warm glacier repeat gives 0.999× / 1.014× / 1.018×.
The original 8-worker A/A was 0.946×. Its four-worker warm p95
latency ratio of 1.110 moves to 0.992 in the repeat. The stock
prefix and VarDCT controls keep at least 0.985× median throughput
when rounded to three decimals, and at most 1.052× p95 latency.
Odd RGBA improves in all measured cells. A sampled one-worker profile
identifies the fused RLE body on the cityscape, macro and forced
128-pixel glacier. The production glacier uses the non-fused RLE body.

The whole-system replays use the new decoder on both sides and stock
`--fast-decode -g 0` on the baseline. The three forced 128-pixel RLE
streams are from the encoder's retained width ablation.

| Photo cohort endpoint | Median MP/s, 1 / 4 / 8 workers | Paired median speedup, 1 / 4 / 8 workers |
|---|---:|---:|
| Production 256 px, warm | 133.22 / 389.79 / 660.40 | 1.183× / 1.205× / 1.195× |
| Forced 128 px, warm | 143.76 / 378.32 / 549.48 | 1.282× / 1.236× / 1.214× |
| Forced 128 px, cold | 137.45 / 331.55 / 511.34 | 1.274× / 1.193× / 1.157× |

The forced 128-pixel glacier reaches 165.43 / 739.18 MP/s warm. Its
stream is 25.45% smaller than stock. The colder eight-worker cohort
remains below 550 MP/s. The forced-width results characterize the
retained stream choices. The public encoder entry point is a separate check.

All 36 same-stream and stock-to-profile PNG16 comparisons agree across
binary and worker count. Strict library Clippy and the focused Modular
and progressive suites pass. The progressive suite retains one existing
ignore. The warm build and timing run takes 166 s at 0.91 GiB peak RSS.
The cold controls, warm width run and CPU profiles finish before the
test build fills the disk, at 0.40 GiB peak RSS. After clearing inactive
Cargo dev artifacts, the test and cold width replay takes 102 s at
1.50 GiB peak RSS.

[Per-cell timings, p95 and A/A](fused_rle_shared_holdout_2026-11-22.tsv).
Raw samples, exact readbacks, command and hash manifests, profiles and
logs are under `/mnt/v/output/zenjxl-decoder/fused-rle-shared-holdout-2026-11-22`.

## Twelve-bit RLE lookup width

The [candidate](https://github.com/imazen/zenjxl-decoder/commit/c5e24c3e4003b039c5ecb15ce8e15ad41fe2549e)
widens the shared fixed-Gradient RLE lookup from 10 to 12 lookahead bits.
The memory-tracked table grows from 4 to 16 KiB per eligible cluster.
The wider lookup can fuse longer literal and repeat tokens. Both builds
use current baseline `edc2abcc`, Rust 1.99.1 and the same locked release
dependencies with `all-simd,threads`.

The four independent photo streams retain their previous widths.
The controls cover the 128 px stock prefix stream, scanned text, odd
RGBA and native d1 VarDCT. Both same-file endpoints decode in memory
with CMS off at 1/4/8 pinned i270 workers. Three interleaved blocks of
five include baseline A/A. The cold process measures its first decode.
The warm process measures after one warmup. All 48 same-file PNG16
readbacks agree across binaries and worker counts.

| Comparison | Median MP/s, 1 / 4 / 8 workers | Median paired speedup, 1 / 4 / 8 workers |
|---|---:|---:|
| Wider vs original, cold photos | 133.62 / 363.68 / 528.00 | 1.011× / 1.007× / 1.002× |
| Wider vs original, warm photos | 139.07 / 403.55 / 581.11 | 1.009× / 0.990× / 1.009× |
| Wider build on both streams, cold profile vs stock | 133.52 / 366.01 / 533.45 | 1.253× / 1.333× / 1.243× |

The stock-to-profile comparison checks all 12 PNG16 pairs exactly.
It meets the 1.25× one-worker bar in this run. The 550 MP/s cold
eight-worker target remains open. The same-file gain falls short of
the 5% decoder landing gate, so the source is archived on
`rejected/rle-twelve-bit-window`.

Class p95 ratios stay within 1.022×. The per-file cold p95 ratios
include 1.214× on 1606 at one worker, 1.125× on 1237 at eight and
1.511× on the odd RGBA control at eight. Those A/A p95 ratios
are 0.993×, 1.004× and 1.134× respectively.

Strict library Clippy and the focused RLE reader differential pass.
The two earlier unit-test builds fail for lack of disk space. The
completed test follows a package-scoped cleanup of inactive Cargo
outputs. The baseline and candidate timing binaries retain their
original hashes.

[Per-file timings](fused_rle_twelve_bit_window_2026-11-22.tsv)
and [class summaries](fused_rle_twelve_bit_window_classes_2026-11-22.tsv).
The baseline build uses 1.99 GiB peak RSS over 52 s. The timing run
uses 0.23 GiB over 193 s and the passing test uses 4.32 GiB over
98 s under `run-heavy`. Exact hashes, raw timed samples, readbacks
and command logs are under `rle-window-i270-2026-11-22` in the i270
decoder run directory.

## Keep Gradient prediction in 32 bits

The [candidate](https://github.com/imazen/zenjxl-decoder/commit/3f08ae0c35f242343dba34ddc52cff30cbc1288a)
reconstructs the fixed-Gradient prefix/RLE path with a wrapping i32
intermediate. When top-left is between left and top, the unclamped
prediction also lies between those values, so it fits in i32. If
top-left is outside their range, the clamp selects one of the endpoints.
An arithmetic test checks all 729 triples from nine values, including
the signed extremes and their neighbors, against the ordinary widened
predictor.

The build uses the locked `all-simd,threads` closure and Rust 1.99.1.
Its base `0ee1d786` has the same decoder and dependency sources as the
retained binary from `edc2abcc`. The four independent photo streams and
the stock prefix, scan, odd RGBA and d1 VarDCT controls use RGB8 or
RGBA8 on i270, with CMS off, 1/4/8 pinned workers, and three interleaved
blocks of five per arm with A/A. Strict library Clippy and all 48
same-file PNG16 readbacks pass.

| Allocating photo endpoint | Candidate MP/s, 1 / 4 / 8 workers | Median paired speedup, 1 / 4 / 8 workers |
|---|---:|---:|
| Cold | 128.59 / 359.28 / 528.11 | 0.974× / 0.989× / 0.998× |
| Warm | 133.57 / 394.23 / 585.31 | 0.974× / 0.984× / 0.972× |

The one-worker slowdown appears on all four photos at both endpoints.
Class p95 ratios stay within 1.076×. The individual results include
a warm eight-worker p95 ratio of 1.187× on 1606 and 1.295× on the
scan, with A/A at 1.040× and 1.005× respectively. The decoder
throughput gate fails. Source is archived on `rejected/rle-gradient-i32`.

Both the arithmetic and the existing reader differential tests pass.
The build and timing wrapper uses 1.98 GiB peak RSS over 211 s.
The focused release tests take 99 s at 4.29 GiB peak RSS.
[Per-file timing](fused_rle_gradient_i32_2026-11-22.tsv) and
[class summaries](fused_rle_gradient_i32_classes_2026-11-22.tsv).
The raw samples, pixel and binary hashes, and commands are under
`gradient-i32-i270-2026-11-22` on i270.
