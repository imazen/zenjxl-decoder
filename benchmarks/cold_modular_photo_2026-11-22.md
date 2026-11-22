# First lossless photo decode on i270

The [production endpoint comparison](photo_profile_endpoints_2026-11-22.md)
measured 516 MP/s at eight workers on the three selected native Modular
throughput streams in a fresh process. The first decode is an in-memory
RGB8 decode and includes output allocation and prefaulting.

## Phase and page-fault attribution

The phase diagnostic uses the same decoder library `6000956d` and the
frozen stock and profile streams for 1007, 1019 and 1069. Each of three
blocks starts five fresh processes per stream. Each process decodes the
same file twice, with the existing `JXL_PHASE_TIMING` diagnostics enabled,
CMS off and eight pinned P cores. The table reports medians of the
individual first-minus-second differences in milliseconds.

| Profile stream | Total | Outside section and frame timers | Frame total | Group decode P2 |
|---|---:|---:|---:|---:|
| 1007 textured | +6.65 | +5.52 | +0.44 | +1.67 |
| 1019 smooth | +1.76 | +3.42 | −2.06 | −0.11 |
| 1069 night | +0.61 | +1.76 | −1.30 | +0.39 |

The residual includes output allocation and prefaulting, header processing,
and other time outside the two existing timers. The CLI's constant-zero
`buf_alloc` counter leaves allocation and prefaulting pooled in the residual.

On the smooth profile, the single-decode process generates 859 sampled
page faults with `perf record -e page-faults -c 32`. The Rayon prefault
closure accounts for 31.66% of those samples. The RLE decoder and its
`memmove` caller stack account for 22.24% and 26.54%. The profile
measures fault events across the command lifetime, including loading the
in-memory input. The phase and fault capture wrappers take 5 s each,
with respective peak RSS of 0.16 and 0.14 GiB.

[Phase medians](cold_modular_photo_phases_2026-11-22.tsv).

## Output prefault ablation

Candidate `902a853a67ef` bypasses explicit output prefaulting.
The ablation compares identical bytes at 1, 4 and 8 workers using
eight-worker-affinity prefixes on i270. Each endpoint uses three
interleaved blocks of five individual samples with A/A. Cold processes
decode once. Warm processes decode once after one untimed warmup.
All 21 PNG16 readbacks match exactly.

The eight-worker cold photo median speedup is 1.016×. The smooth
photo reaches 506.64 MP/s from a 511.73 MP/s baseline. The eight-worker
warm photo median speedup is 1.014×. At eight workers the warm d1
VarDCT control measures 0.963× with A/A at 1.017×. The candidate
misses the throughput target and is archived under
`rejected/cold-output-prefault-disabled`. The build, exact checks and
two timing endpoints take 212 s at 0.91 GiB peak RSS.

[Cold](cold_modular_no_prefault_2026-11-22.tsv) and
[warm](warm_modular_no_prefault_2026-11-22.tsv) data include p95 and A/A.

The raw timing samples, stream and binary hashes, fault profiles and
commands are under `/mnt/v/output/zenjxl-decoder/cold-modular-photo-2026-11-22`
on i270.

## Modular recycling batch sizes

The current implementation recycles buffers after eight groups per
worker in eligible Modular frames. Compare four and sixteen groups
against that baseline at both endpoints. The eligibility check,
bitstream and RGB8 output format are fixed. Each candidate uses
three interleaved blocks of five samples with A/A at 1, 4 and 8
workers. The exact comparisons pass for 21 cells with sixteen and
24 with four. The four-group panel adds a native 128-pixel
Modular stock stream.

| Experimental batch | 8-worker cold photo median | Paired cold speedup | 8-worker warm photo median | Paired warm speedup |
|---|---:|---:|---:|---:|
| 16 groups per worker | 390.89 MP/s | 0.796× | 484.65 MP/s | 0.860× |
| 4 groups per worker | 576.70 MP/s | 1.126× | 617.31 MP/s | 1.112× |

GNU time records first-decode process peaks. On the smooth
photo, sixteen groups increase RSS from 115,752 to 168,032 KiB
and minor faults from 27,814 to 40,804. In the separate
four-group run RSS falls from 116,168 to 90,608 KiB and faults
from 27,799 to 21,528. Each memory cell is one fresh process.

A follow-up phase capture interleaves 15 fresh processes per
photo and arm. On 1019, skipping prefault changes the time
outside the frame and section timers from 7.16 to 5.59 ms,
while P3b render increases from 2.66 to 3.89 ms. Four groups
reduce P2 from 11.09 to 10.01 ms and the P3a store step from
1.71 to 1.07 ms. Sixteen groups increase both phases to 13.55
and 2.72 ms. These are diagnostic medians. The timing tables
measure throughput with diagnostics disabled.

Four groups slow the 128-pixel stock stream to 0.824× at eight
workers cold. The unconditional four-group and sixteen-group
candidates are archived as `rejected/cold-modular-batch4-unconditional`
and `rejected/cold-modular-batch16`. Restricting the policy to the 256-pixel geometry restores the 128-pixel
control. The ordinary lossless photo and document still slow to 0.970× and
0.957× cold at eight workers. The size-only candidate is archived under
`rejected/cold-modular-batch-geometry-only`. Its cold run completed with
30 exact readbacks. The partial warm run is retained separately.

[Per-stream batch timing](cold_modular_batch_ablations_2026-11-22.tsv).
Additional summaries and raw memory counts are in `batch-summary.tsv`,
`batch-memory.tsv` and `compare-phases.tsv` in the run directory.

## Four groups with a coding-structure guard

The decoder uses four groups per worker in eligible 256-pixel Modular
frames whose shared tree uses prefix coding or which have local trees
without a shared tree. The shared ANS and other group-size cases use the
eight-group schedule. The three throughput photos move from 64 to 32
groups per batch at eight workers. The 128-pixel stock stream and both
ordinary lossless controls retain batches of 64. The existing checks
for noise, border dependencies and group count still control recycling.

The same-stream cold photo median at eight workers rises from 499.44 to
575.12 MP/s, a paired 1.147×. Warm reaches 587.61 MP/s with a 1.025× paired
gain. On the smooth photo, first-decode process RSS falls from 115,776
to 90,732 KiB and minor faults fall from 27,815 to 21,539. Class p95
ratios for the three photos are 0.918× cold and 0.983× warm.

Three whole-system stream pairs compare stock `--fast-decode -g 0`
with the throughput encodes from jxl-encoder `25ca0c65`. Cold throughput
of the new profile at 1 / 4 / 8 workers is
146.24 / 391.51 / 591.43 MP/s. The paired speedups are
1.369× / 1.415× / 1.321×. With warm caller-owned output, throughput is
165.28 / 515.75 / 651.39 MP/s and speedups are
1.398× / 1.509× / 1.249×. At eight workers the reused-output A/A median is
0.898×. The reused-output controls compare the two streams on the same
new decoder and check each reused result against its allocating decode.

The [scoped allocation results](cold_modular_batch_scoped_2026-11-22.tsv)
cover ten streams at three worker counts and both endpoints.
The [control replays](cold_modular_batch_control_repeats_2026-11-22.tsv)
retain the initial outliers and longer samples in the run directory.
The 75-sample warm stock control reaches 1.003× median speedup with
1.044× p95, after a 0.933× median in the 45-sample replay. The mixed-page
cold four-worker p95 moves from 1.178× in the 45-sample replay to 0.927×.
The extended RGBA controls also clear the 2% median and 10% p95 limits.

The [whole-system cold results](cold_modular_batch_system_cold_2026-11-22.tsv)
retain both stream hashes. All 30 same-file and nine whole-system
worker-wise PNG16 comparisons are exact. Strict library Clippy, 60 Modular
tests and 17 progressive tests pass with the local codec corpus configured.
One pre-existing progressive ignore remains. The first test invocation
lacked `CODEC_CORPUS_PATH`; the complete replay runs those tests with the
corpus present. The build and main run peak at 1.50 GiB RSS, the corpus and
endpoint replay at 0.51 GiB, and the extended controls at 0.21 GiB.

The exact binary build uses the production candidate's code at the revision
recorded in `coding-scope.commit` in the run directory. Phase marks,
per-class normalized percentiles, release commands, raw timing samples,
maximum RSS and warm reuse data remain there alongside the bitstream hashes.

## Recycling at exactly 16 groups per worker

At the old cutoff, a 256-pixel prefix-coded Modular frame with exactly
16 groups per worker bypasses recycling. The 1447 glacier throughput
stream has 128 groups. At eight workers it previously decoded all 128
in one batch. It now uses four batches of 32. The equality case uses
the existing smaller-batch eligibility check. Shared ANS trees, other
group widths, noise and neighbor-dependent rendering keep their
existing scheduling conditions.

The same-stream builds start from decoder main
`7f545adfe487ab5e433aed9617f5e6c9297110a9` with the same locked
dependencies, Rust 1.99.1 and `all-simd,threads`. The tested candidate
source is `3bd0cfb5767f41156faa57cedeba776f82f7b363a4`. The i270
benchmark pins P cores and decodes to the same in-memory RGB8 or RGBA8
output with CMS off. Baseline, candidate and A/A rotate across three
blocks of five. Cold uses fresh processes. Warm uses one warmup per
process. All 18 PNG16 readback pairs agree across decoders and 1/4/8
workers.

| Eight-worker 1447 throughput stream | Baseline MP/s | Candidate MP/s | Paired speedup | p95 latency ratio |
|---|---:|---:|---:|---:|
| Cold allocating | 361.49 | 511.98 | 1.439× | 0.494 |
| Warm allocating | 499.95 | 649.30 | 1.306× | 0.840 |

A separate phase trace uses one first decode per binary. Group decode
falls from 15.29 to 8.16 ms and the final group phase from 4.28 to 0.05 ms.
Named group phases total 24.26 ms before and 10.43 ms after. The timing
table measures with phase diagnostics disabled. Builds, exact readbacks,
timing and strict library Clippy took 193 s at 0.90 GiB peak RSS.

The initial control tails include 1.110× cold p95 on the odd RGBA
photo at four workers and 1.142× cold A/A p95 on VarDCT at eight.
The eight-worker city throughput stream initially measures 0.962×
warm with A/A at 0.894×. In the nine-block replays, these three
candidate ratios are 1.000× p95, 1.004× p95 and 1.006× throughput,
respectively. The odd RGBA warm eight-worker A/A p95 is 1.138× in
the repeat, while the candidate p95 is 1.083×. After the repeats,
all control medians are at least 0.982× and all candidate control
p95 ratios are at most 1.084×.

[Same-stream timings and control repeats](cold_modular_batch_boundary_2026-11-22.tsv).
The six stream hashes, exact readbacks, phase logs, commands and
raw samples are in
`/mnt/v/output/zenjxl-decoder/cold-prefix-boundary-2026-11-22/current-main`.

### Profile comparison on three further native photographs

Decode the frozen stock and throughput encodes for imazen-26 originals
1009, 1427 and 1447 with the boundary candidate on both sides. The
stock uses `--fast-decode -g 0`. The throughput streams preserve exact
samples and save 17.70% median bytes. The encoder study has independent
jxl-rs and djxl v0.12 checks, encoder identities, and the balanced
profile comparison.
[Encoder results](https://github.com/imazen/jxl-encoder/blob/873e5c9/benchmarks/faster_decode_imazen26_lossless_2026-11-22.md#three-additional-native-lossless-photo-controls).

| RGB8 endpoint | Median MP/s, 1 / 4 / 8 workers | Paired median speedup, 1 / 4 / 8 workers |
|---|---:|---:|
| Cold allocating | 126.09 / 351.36 / 507.11 | 1.172× / 1.191× / 1.153× |
| Warm allocating | 130.45 / 388.68 / 564.26 | 1.166× / 1.159× / 1.142× |
| Warm caller-owned | 141.65 / 458.78 / 629.86 | 1.181× / 1.243× / 1.326× |

The eight-worker allocating cold endpoint still falls below 550 MP/s.
The one-worker relative speedup falls below 1.25× at every endpoint.
The cityscape is the principal one-worker outlier, reaching 0.922×
warm. At one worker on that source, `perf` assigns 74.61% of stock
self samples to `decode_fused_prefix` and 77.63% of profile self
samples to `rle_gradient::decode`. Integer-to-RGB8 conversion accounts
for 3.71–3.98% in the stock, profile and forced plain-prefix captures.

Both allocating endpoints use three blocks of five, with all nine
stock-to-profile PNG16 pairs matching exactly at each endpoint.
The caller-owned harness checks byte equality between the profile and
stock outputs and between its reused and allocating decodes.
The extended controls and all three endpoints took 307 s
at 0.51 GiB peak RSS. The sampling scout took 15 s at 0.13 GiB peak RSS.

[Per-original timings, p95 and A/A](cold_modular_batch_boundary_system_2026-11-22.tsv).
Raw timing and exact readback output are in the same benchmark run
directory as the boundary replay. Raw CPU profiles are under
`/mnt/v/output/zenjxl-decoder/render-wait-i270-2026-11-22`.

## Render scratch and border preparation on both widths

The panel uses the six retained production encodes of imazen-26 photos
1009, 1427 and 1447 at 128 and 256 pixels per group. Stock lossless,
ordinary VarDCT and an odd RGBA stream are controls. The baseline binary
at `8496962d` has the same decoder source as main `8bbdb170`. Both
candidates build with its locked release dependencies and runtime SIMD
dispatch. Decode uses RGB8 or RGBA8, CMS off, and 1, 4 or 8 pinned i270
workers. Cold and warm each use three interleaved blocks of five with A/A.
Warm runs decode once before measurement. Before timing, all 27
same-file PNG16 comparisons pass for each candidate.

One variant retains owned render row scratch between batches. Each batch
still uses its own read view of the pipeline. The other variant uses the
existing no-copy preparation path for recyclable Modular batches, whose
eligibility already requires zero border dependencies. It processes
readiness markers without creating empty border images. Both variants
start independently from the baseline.

Both experiments miss the independent 5% throughput gate.

| Variant | 256 px cold / warm | 128 px cold / warm |
|---|---:|---:|
| Reuse scratch across batches | 0.994× / 1.000× | 1.012× / 1.016× |
| Skip empty borders | 1.003× / 0.943× | 1.006× / 0.967× |

The ratios are eight-worker paired medians, with A/A medians of
0.994–1.016× across these cells. In separate five-repeat first-decode
traces, skipping empty borders reduces prepare time by 0.36–0.58 ms
for the 128 px streams. Render time rises from 4.44 to 4.89 ms on
1009 and from 4.37 to 4.74 ms on 1427. The 256 px macro's cold
p95 under scratch reuse is 1.175× with A/A 0.936×. The stock
eight-worker control under the border change is 0.957× cold.
The per-file table retains these observations.

[Scratch source](https://github.com/imazen/zenjxl-decoder/tree/rejected/modular-batch-render-context-reuse)
and [border source](https://github.com/imazen/zenjxl-decoder/tree/rejected/modular-borderless-batch-preparation)
are archived. The builds, exact checks and timing take 223 s and
224 s respectively, each at 0.91 GiB peak RSS.
[Paired timings and p95](cold_modular_batch_preparation_2026-11-22.tsv)
and [phase medians](cold_modular_batch_preparation_phases_2026-11-22.tsv).
Full samples, hashes, binaries and commands are in
`/mnt/v/output/zenjxl-decoder/cold-compact-allocation-2026-11-22` on i270.

### Use both group widths for cold throughput

This follow-up selects between the two shipped lossless options on the
three photos used for the width tuning. Count the groups the image
would have at 256 px. Choose the 128 px stream when that count is at
most 128, the eight-worker boundary of 16 groups per worker. Otherwise
choose the 256 px stream. The cityscape and macro have 192 groups and
the glacier has 128. The lookup uses dimensions only. The threshold
was chosen after examining this panel, so the rule needs a fresh
holdout before it can select a default.

Both sides use the same `8496962d` decoder and the encoder outputs
verified at `7fca6d6b`. The stock comparison uses `--fast-decode -g 0`.
Every chosen file is smaller than stock. The median saving is 17.70%.
All nine PNG16 readbacks match in each pass at 1, 4 and 8 workers.
The allocating RGB8 path runs once per fresh process with CMS off.

| Interleaved blocks of five | Median MP/s, 1 / 4 / 8 workers | Paired speedup, 1 / 4 / 8 workers |
|---|---:|---:|
| 3 | 147.20 / 389.78 / 550.20 | 1.364× / 1.324× / 1.258× |
| 9 | 147.15 / 389.44 / 562.59 | 1.364× / 1.327× / 1.273× |

The eight-worker A/A medians are 0.998× and 1.012×. A/A p95 stays
within 1.030× for every file and worker count. The cityscape remains the
slowest chosen stream, at 483.92 MP/s in the longer eight-worker run.
The two replays take 26 s and 72 s, both at 0.12 GiB peak RSS.

[Per-file timings and p95](cold_modular_width_routing_2026-11-22.tsv).
The `geometry-cold` and `geometry-repeat-cold` directories in the
same i270 run directory retain raw timings, selected stream names,
exact readbacks and input and binary hashes.

## Cold phase attribution on the independent width panel

The same decoder source as `5113c242` decodes each of the four pinned
photo profiles and the 1606 stock stream twice in each fresh process,
with CMS off and one or eight pinned i270 workers. The diagnostic build
is `edc2abcc`. At each worker count, the three interleaved blocks of five
first/second pairs give 15 observations per file. Frame and section
times are subtracted from CLI total on each decode before calculating
the median first-minus-second difference.

| Profile | 8-worker first-minus-second total | Outside frame and section timers | Group decode P2 gap |
|---|---:|---:|---:|
| 1012 g128 | +2.24 ms | +1.26 ms | +0.48 ms |
| 1606 g128 | +3.16 ms | +2.11 ms | +0.45 ms |
| 1237 g256 | +3.53 ms | +4.68 ms | −0.29 ms |
| 1619 g256 | +0.78 ms | +3.77 ms | −0.45 ms |

The residual includes allocation, output prefaulting and other work
outside the frame and section timers. The CLI still reports a
constant-zero separate allocation field. The individual phase medians
can differ from the sum of the median phase differences. The
diagnostic capture uses 0.13 GiB peak RSS over 14 s.

The [prefault experiment](https://github.com/imazen/zenjxl-decoder/commit/96a3e380518e1649c45633887aa42ed87040cdf0)
groups the same 4 KiB page touches into 256 KiB Rayon work items.
This changes neither the pages touched nor the existing threshold and
eligibility. Rebuild against the same locked dependencies, Rust 1.99.1,
and release features `all-simd,threads`. Compare the same profile
streams and four controls using 1/4/8 workers, RGB8 or RGBA8 and three
interleaved blocks of five with A/A. The timed cold decodes run once
per process. A warmup decode precedes each warm measurement. Strict
library Clippy and 48 exact PNG16 pairs pass.

| i270 photo endpoint | Paired speedup at 1 / 4 / 8 workers | Candidate MP/s at 1 / 4 / 8 workers |
|---|---:|---:|
| Cold allocating | 1.000× / 0.999× / 0.997× | 132.14 / 364.70 / 525.84 |
| Warm allocating | 1.001× / 0.990× / 1.015× | 137.26 / 397.29 / 607.81 |

The photo medians miss the independent 5% decoder improvement gate.
All class p95 latency ratios stay within 1.078×. Two control tails
exceed 1.10× individually. Scanned text is 1.194× with A/A at 1.367×
at four workers cold. Odd RGBA is 1.375× with A/A at 1.168× at eight
workers warm. The experiment is archived on
`rejected/cold-prefault-256k-spans`. Build, exact checks and timings
use 2.00 GiB peak RSS over 210 s under `run-heavy`.

[Phase diagnostics](cold_modular_independent_phases_2026-11-22.tsv),
[paired timing](cold_modular_prefault_256k_2026-11-22.tsv) and
[class summaries](cold_modular_prefault_256k_classes_2026-11-22.tsv).
The raw timings, readbacks, commands and hashes are under
`independent-cold-phases-2026-11-22` on i270.

## Inline Modular channel lists

The [candidate](https://github.com/imazen/zenjxl-decoder/commit/b604803c9c0022955faef8748a7fe6165301a89d)
stores up to four borrow guards and channel references in `SmallVec`
inline storage. Larger channel lists can spill to the heap. The local
transform path keeps its existing vector and calls the same entropy
decoder over a slice. The path without local transforms passes the
inline slice directly.

One complete first decode of 1606 g128 at one worker under heaptrack
makes 12,813 allocation calls on baseline and 10,909 on the
candidate. The reported peak heap is 38.92 MB for both. This is a
whole-process diagnostic, including startup and input acquisition.

Baseline `edc2abcc` has the same decoder and dependency sources as
the candidate's parent `3e2818ad`. Use the same four independent photo
streams and RGB8 or RGBA8 controls for stock prefix, scanned text,
odd alpha and native d1 VarDCT. Each allocating endpoint uses i270 at
1/4/8 pinned workers and three interleaved blocks of five with A/A.
Cold runs decode once per process and warm runs decode after one
warmup. Strict library Clippy and all 48 frozen-stream PNG16 pairs pass.

| Endpoint | Photo throughput ratio at 1 / 4 / 8 workers | Candidate MP/s at 1 / 4 / 8 workers |
|---|---:|---:|
| Cold allocating | 0.994× / 1.012× / 1.003× | 131.25 / 363.34 / 524.56 |
| Warm allocating | 0.993× / 0.996× / 0.990× | 136.35 / 396.89 / 588.56 |

Neither endpoint reaches the independent decoder gate. All candidate
class p95 ratios stay within 1.057×. The 1237 and 1619 individual
cold eight-worker p95 ratios are 1.193× and 1.182×, with A/A at 1.155×
and 0.964×. The experiment is archived on
`rejected/inline-modular-channel-lists`.

The six transform limit tests pass. The build and timing wrapper
takes 210 s at 1.98 GiB peak RSS. Heaptrack capture and release test
build use 4.34 GiB over 100 s. The initial borrow-lifetime compile
failure and the zero-match test filter are retained in the run logs.
The completed transform tests use the compiled binary and match all
six cases. [Per-file timings](cold_modular_inline_channels_2026-11-22.tsv)
and [class p95](cold_modular_inline_channels_classes_2026-11-22.tsv).
Raw timing, heaptrack, sources and exact hashes are under
`inline-modular-channels-2026-11-22` on i270.

## Three groups per worker under the scoped rule

The [candidate](https://github.com/imazen/zenjxl-decoder/commit/4c6892dc26d9a2b4564701e333777aaf52feaf21)
uses three groups per worker in the existing 256-pixel prefix or
local-tree recycling rule. The baseline uses four. The 128-pixel
streams and stock prefix, scan, RGBA and VarDCT remain controls.
Baseline binary `edc2abcc` shares decode and dependency sources with
the experiment parent `9f5ea645`.

Both allocating endpoints use i270, RGB8 or RGBA8, CMS disabled,
1/4/8 pinned workers, and three interleaved blocks of five with
same-file A/A. All 48 same-file PNG16 pairs and 12 stock-to-profile
pairs match. Strict library Clippy passes.

| Relevant profile | Cold four-worker speedup | Warm four-worker speedup | Cold eight-worker speedup | Warm eight-worker speedup |
|---|---:|---:|---:|---:|
| 1237 g256 | 0.864× | 0.793× | 0.977× | 0.893× |
| 1619 g256 | 0.863× | 0.798× | 0.994× | 1.001× |

A separate phase replay at four and eight workers uses 15 first
decodes per file and arm. At four workers, each file goes from
12 batches of 16 to 16 batches of 12. The timed render phase
increases from 4.29 to 8.84 ms for 1237 and 4.32 to 8.44 ms
for 1619. Group decode increases from 20.58 to 21.91 ms and
17.68 to 18.79 ms respectively. At eight workers the render phase
also increases, by 0.39 and 0.30 ms. These phase measurements
use the existing diagnostics. Standalone throughput timings keep
the diagnostics disabled.

The cold system replay with the candidate on both arms reaches
1.234× stock at one worker and 524.37 MP/s at eight on the four
photos. The experiment misses the speed goals. The four-worker
class p95 ratios are 1.175× cold and 1.235× warm. The unchanged
scan control's cold four-worker per-file p95 reaches 1.493× with
A/A at 1.134×. The source is archived on
`rejected/scoped-modular-batch-three`.

[Per-file timings](cold_modular_batch_three_2026-11-22.tsv),
[class p95](cold_modular_batch_three_classes_2026-11-22.tsv) and
[phase summaries](cold_modular_batch_three_phases_2026-11-22.tsv).
The build, exact decode and timing wrapper uses 1.98 GiB peak RSS
over 243 s. Phase capture uses 0.09 GiB over 4 s. Commands, binary
and stream hashes, readbacks and raw samples are under
`modular-batch3-i270-2026-11-22` on i270.

## Six groups per worker under the scoped rule

The [candidate](https://github.com/imazen/zenjxl-decoder/commit/d12de6c313467d1a12cc2b0275d8c62de947b86c)
increases the four-group batch to six under the same 256-pixel
recycling rule. It isolates batch width from the encoder's group size
and entropy choices. The retained baseline `edc2abcc` matches the
decoder source and dependency lock at the candidate parent
`1fc1f412`. The i270 runs use the same eight files, P cores,
allocating RGB8 or RGBA8 and three interleaved blocks of five
with A/A. Strict Clippy, 48 same-file readbacks and 12
stock-to-profile readbacks pass.

| Relevant photo | Cold four-worker speedup | Warm four-worker speedup | Cold eight-worker speedup | Warm eight-worker speedup |
|---|---:|---:|---:|---:|
| 1237 g256 | 0.995× | 1.008× | 0.968× | 0.976× |
| 1619 g256 | 0.988× | 1.000× | 0.937× | 0.946× |

The phase capture uses 15 first decodes per file and arm. At four
workers, the number of batches falls from 12 to eight and render
time decreases from 4.44 to 4.02 ms on 1237 and 4.37 to 3.97 ms
on 1619. Total time changes from 35.25 to 35.44 ms and 31.60 to
31.32 ms. At eight workers, six batches fall to four while the
group decode, store and render medians all increase on both photos.
The independent whole-system replay measures 1.237× stock at one
worker and 494.60 MP/s at eight, below the cold targets.

Photo class p95 ratios stay within 1.091×. The 1619 cold
eight-worker p95 reaches 1.127× with A/A at 1.012×. Candidate
control class p95 ratios stay within 1.033×. The experiment is
archived on `rejected/scoped-modular-batch-six`. The matched
two-photo evidence favors four groups over three or six for
the current scoped recycling policy.

[Per-file timing](cold_modular_batch_six_2026-11-22.tsv),
[class p95](cold_modular_batch_six_classes_2026-11-22.tsv), and
[phase summaries](cold_modular_batch_six_phases_2026-11-22.tsv).
Build and validation use 1.99 GiB peak RSS over 242 s. Phase capture
uses 0.11 GiB over 5 s. Exact commands, timings and hashes are under
`modular-batch6-i270-2026-11-22` on i270.

## CPU cycles and short RLE copies

The decoder binary from `edc2abcc` has the same decoder and CLI source
and lock as `19d6449d`. Sample user cycles on the pinned i270 P cores
with `perf record --event cycles:u --count 1250000`. Each cell runs
15 fresh processes with one allocating RGB8 decode and CMS off.
The shares below are sampled self cycles across the process,
including input loading. The libc columns include every caller.

| Stream at 8 workers | Fused RLE Gradient | Plain fused prefix | memmove | memset | RCT | i32 to u8 |
|---|---:|---:|---:|---:|---:|---:|
| 1012 g128 | 50.39% | <0.10% | 9.29% | 3.54% | 6.79% | 6.40% |
| 1237 g256 | 67.98% | <0.10% | 7.09% | 2.86% | 6.08% | 4.61% |
| 1619 g256 | 65.34% | <0.10% | 7.99% | 3.25% | 6.51% | 4.09% |
| 1606 stock | <0.10% | 67.58% | 3.89% | 2.32% | 5.35% | 4.41% |

On the two g256 streams the contended mutex symbol accounts for
0.14% and 0.24% of sampled self cycles. These samples give no
measurement of off-CPU waiting.

The [short-copy candidate](https://github.com/imazen/zenjxl-decoder/commit/d08794ec99874af74950a184bad6a4f785e46fb5)
uses fixed-size copies for two, three and four samples when a fused
RLE run has zero residual and predicts the top row. Other spans use
the existing slice copy. The fused symbol is 144 bytes larger in the
candidate executable and the non-fused symbol has the same size.
The regular-reader differential unit test, strict library Clippy,
48 same-file PNG16 pairs and 12 stock-to-profile pairs pass.

The comparison uses the same four photo profiles, four controls,
allocating RGB8/RGBA8 output, and three blocks of five with A/A at
1, 4 and 8 workers. The eight-worker photo median is 0.992× cold
(A/A 1.010×) and 0.996× warm (A/A 1.006×). The whole-system cold
replay reaches 1.230× stock at one worker and 532.27 MP/s at eight.
The photo class p95 ratio stays within 1.074×. The 1619 g256 cold
eight-worker p95 is 1.176× with A/A at 1.005×. The candidate misses
the throughput gate and is archived on
`rejected/modular-short-rle-top-copy`.

The completed profile wrapper uses 0.09 GiB peak RSS over 7 s.
The earlier stack-symbolization attempt was stopped after the first
recording, which is retained and used for the self-cycle report.
The build, exact checks and replay take 361 s at 4.30 GiB peak RSS.
Raw profiles, timings, hashes and commands are in
`modular-cold-cycles-i270-2026-11-22` and
`modular-short-rle-copy-i270-2026-11-22` on i270.

[Cycle attribution](cold_modular_cycles_2026-11-22.tsv),
[per-file timings and p95](cold_modular_short_rle_copy_2026-11-22.tsv),
and [class comparisons](cold_modular_short_rle_copy_classes_2026-11-22.tsv).

## Signed residuals in the RLE cursor

The [candidate](https://github.com/imazen/zenjxl-decoder/commit/3b863f1c6f6980eb2331ae2cd39c1089eab73397)
packs signed literals into the existing 10-bit fused lookup and retains
the last residual in signed form in the RLE cursor. Repeat lengths stay
unsigned. Wider literals use the general token decode. Cursor handoffs
convert the last residual to and from the general reader's unsigned
representation. This also changes the non-fused cursor, so the replay
adds the regular RLE stream of glacier photo 1447 as a fifth control.

The four target streams, pinned i270 cores, Rust 1.99.1 release scope,
RGB8/RGBA8 output and cold, warm and A/A schedule match the short-copy
experiment above. Four RLE tests, including full-width packed-symbol
round trips and exact generic-decoder comparisons, pass with strict
library Clippy. All 54 same-file PNG16 pairs and 12 stock-to-profile
pairs pass across worker counts.

| Comparison | Median paired speedup, 1 / 4 / 8 workers | Candidate MP/s at 8 workers |
|---|---:|---:|
| Same-file photos, cold | 1.011× / 1.007× / 1.010× | 538.32 |
| Same-file photos, warm | 1.010× / 1.005× / 0.998× | 622.16 |
| Cold profiles vs stock on the candidate | 1.266× / 1.341× / 1.240× | 535.79 |

The 256 px profiles gain 2.1–2.4% at one worker cold. The non-fused
1447 control slows 5.0% cold and 4.0% warm at one worker, with A/A
ratios of 0.999× and 1.002×. Its warm four-worker p95 reaches 1.104×
with A/A at 1.058×. The scanned-text cold eight-worker p95 is 1.280×,
with A/A at 1.348×. Candidate class p95 ratios stay within 1.037×.
The candidate falls short of the independent decoder speedup and
550 MP/s cold targets. Source is archived on
`rejected/signed-modular-rle-cursor`.

In a separate capture on the same 256 px streams and binaries,
three alternating blocks of five fresh processes per arm count
retired user instructions, branches and branch misses on the
pinned P cores. The candidate uses 2.1–2.2% fewer instructions
and 9.9–10.6% more branches at one and eight workers. Branch misses
increase by 0.4–1.1%. The counts cover the entire process.

The build and qualification use 4.31 GiB peak RSS over 357 s.
The counter capture uses 0.09 GiB over 8 s. Raw timings, exact
readbacks, hashes and commands are in
`modular-signed-rle-i270-2026-11-22` on i270.
[Per-file timing and p95](cold_modular_signed_rle_2026-11-22.tsv),
[class results](cold_modular_signed_rle_classes_2026-11-22.tsv)
and [hardware counters](cold_modular_signed_rle_instructions_2026-11-22.tsv).
