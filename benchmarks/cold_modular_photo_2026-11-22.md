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
