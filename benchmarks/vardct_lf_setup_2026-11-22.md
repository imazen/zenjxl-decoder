# VarDCT LF and HF metadata setup on i270

The baseline is the decoder code in
[dda06cbe](https://github.com/imazen/zenjxl-decoder/commit/dda06cbe104184544a5e4cf9dcb13901fdb01342).
All builds use Rust 1.99.1, locked release dependencies and
`--no-default-features --features all-simd,threads`. The same-file benchmark
uses the retained imazen-26 encodes and an allocating RGB8 or RGBA8
endpoint with CMS disabled. Baseline, candidate and A/A use three
interleaved blocks of five processes with one warmup decode per process.
The harness pins the same P cores at 1, 4 and 8 workers. It compares
PNG16 hashes across binaries and worker counts before timing.
The separate eight-worker phase measurements use three interleaved
blocks of eight timed decodes after three warmups per process.

## Where the baseline spends time

The [phase inventory](vardct_lf_setup_baseline_2026-11-22.tsv) records
1- and 8-worker medians from seven timed decodes after two warmups. At eight workers, the three
ordinary d1 photos spend 4.96–5.89 ms in LF groups, which is 14.1–16.3%
of decode wall time. LF-group timing includes the parallel LF and HF
metadata decode, overlapped HF-global parsing and copy-back. Copy-back
takes 0.20–0.21 ms, and the concurrent HF-global parser takes 0.24–0.25 ms.

A CPU-clock profile of 1007 at ordinary d1 attributes 4.24% of total
self samples to `WpOnlyLookup<false>::decode_channel` and 0.52% to
`decode_hf_metadata_into_rects`. On the smooth 1019 d4 control the
same functions account for 2.42% and 0.17%. The tier-1 d4 stream of
1019 has 0.17% in HF metadata and no samples attributed to the
weighted-only channel decoder. The scout and profiles took 20 s
with 0.24 GiB peak RSS.

## Exact weight normalization

Candidate `3d97e182306654827cfdd55ce6376fb0f7b1c32b` clamps the normalization shift to zero.
The original low-error branch already uses the same lookup and
unshifted product, so the change preserves the arithmetic for every
wire weight. The focused test checks both sides of the normalization
and reciprocal-table boundaries for all sixteen wire weights. Both
predictor tests and all 33 exact PNG16 comparisons pass.

On the three ordinary d1 photos, the median paired LF time ratio is
0.943 and the median eight-worker decoder speedup is 1.006× (A/A 1.015×).
The native screenshot and mixed-page controls run at 0.924× and
0.968× baseline at eight workers. The screenshot p95 latency ratio is
1.137 (A/A 1.046). The experiment misses the stage and throughput
targets and the controls fail. Source is on
`rejected/lf-wp-unified-shift`. The passing run took 458 s with
1.50 GiB peak RSS. An earlier build in the shared Cargo cache failed
because its extracted `cc` dependency lacked modules. The measured
build uses the retained project Cargo home.

## AVX2-tier compile of the existing weighted loop

Candidate `da447f0951f31d78bb74745bbf27e90e32f1b594` dispatches once per weighted-only channel
through the existing AVX2 feature and token. It uses BMI2 shifts and
leading-zero counts. In the generated code both dispatched and
generic kernels call the residual-reader closure for every sample.
The baseline inlines the residual read and calls the slow reader
only when necessary.

The ordinary d1 photos have a median paired LF-group latency ratio
of 1.051 (A/A 1.006) and an eight-worker decode speedup of 0.990×
(A/A 0.979×). Photo 1619 has a p95 latency ratio of 1.128 (A/A 1.046).
All 33 PNG16 comparisons and the predictor control pass. This
comparison uses a 256-pixel screenshot and 1024-pixel mixed-page
control to exercise those paths. The first normalization run uses
the native versions of those two files. The archived source is
`rejected/lf-wp-avx2-outlined-reader`. The run took 314 s with
1.50 GiB peak RSS.

## Inlined AVX2 residual reader

Candidate `a13f5116a64af827bf54d0f58b6b822dece38f71` replaces the residual closure with a typed
reader whose hot method is explicitly inlined. The AVX2 machine code
retains BMI2 shifts and leading-zero counts and removes the call to
the ordinary residual closure. The ordinary d1 photos still have a
median LF-group latency ratio of 1.034 (A/A 0.997). Eight-worker decode
speedup is 1.003× (A/A 1.004×). All 33 exact PNG16 checks and the
predictor control pass. The mixed-page crop and screenshot control
have p95 latency ratios of 1.114 and 1.122 at one and four workers
respectively (A/A 0.985 and 0.975). The result is archived as
`rejected/lf-wp-avx2-inlined-reader`. The run took 314 s with
1.52 GiB peak RSS.


## Weighted predictor row windows

Candidate `e7d211539b97755e8861f1102617ff1687cd524e` handles the two row edges explicitly and
uses adjacent top-row samples and the carried left pixel for the
interior. The output, tree properties and both error histories match
the per-pixel decoder for default and custom weights through odd and
even rows at widths 1, 2, 3, 4, 17, 255 and 256. The differential test
and all 24 same-file PNG16 comparisons pass.

The ordinary d1 photos have a median paired LF-group latency ratio of
1.063 (A/A 1.010). Their median eight-worker speedup is 0.985× (A/A
1.010×). The generated code calls the shared pixel step from both row
loops. The small RGBA control has a one-worker p95 latency ratio of
1.459 (A/A 1.016). The source is archived on `rejected/lf-wp-row-windows`.
The run took 250 s with 1.50 GiB peak RSS.

All four candidates miss the 15% LF phase reduction and 5% eight-worker
decode speedup targets on the three ordinary d1 photos.

[Paired decode results](vardct_lf_setup_variants_2026-11-22.tsv)
retain every cell and p95 ratio. [Phase results](vardct_lf_setup_phase_2026-11-22.tsv)
retain the LF-group timing for each file and variant. Exact commands,
binary and stream hashes, raw timings, readbacks, profiles, and the
generated-code inspection are under
`/mnt/v/output/zenjxl-decoder/lf-hf-setup-2026-11-22` on i270.

## Neighbor-render waits

Eight-worker RGB8 output on the three native photo streams reaches
71,232–121,356 `sched_yield` calls per process across 15 decodes,
including three warmups. These are medians of three runs. The same
one-worker measurements record zero yields. In separate stack captures
with decoder `8496962d`, 57.3–60.2% of recorded yield stacks come from
the neighbor-render wait in `decode_groups_parallel`. The remaining
recorded stacks come from Rayon's worker wait. This count measures
call origin. It gives no duration estimate for a yield or a row lock.

The [worker-retirement experiment](https://github.com/imazen/zenjxl-decoder/commit/78938076f269410b93571000b709db966e848899)
lets a worker exit when the decode and ready-render queues are empty.
Workers that still own a decode publish their input and drain ready
renders. All 18 PNG16 outputs agree with the baseline across
binary and 1/4/8 workers. The warm allocating timings use three
interleaved blocks of five and A/A.

| VarDCT stream | Eight-worker yields, baseline → retirement | Eight-worker paired speedup |
|---|---:|---:|
| 1009 ordinary d1 | 106,910 → 21,160 | 0.931× |
| 1427 tier 1 d1 | 107,708 → 19,167 | 0.979× |
| 1447 tier 1 d3.98 | 60,260 → 18,010 | 0.936× |

Each syscall pair is a single process per arm with three warmups and
12 timed decodes. The warm speedup uses the separate interleaved
timings. The five VarDCT cells have median 0.972× and 0.964× at four
and eight workers, with A/A at 1.004× and 1.002×. The speedup gate fails.
The source is archived on `rejected/retire-vardct-render-workers`.

The stack capture uses 0.24 GiB peak RSS over 5 s. The build, readback,
timing and syscall comparison use 0.89 GiB over 150 s.
[Per-file timings and A/A](vardct_render_worker_retirement_2026-11-22.tsv).
Raw stacks, syscall counts, binary and stream hashes, exact readbacks
and measurements are under
`/mnt/v/output/zenjxl-decoder/render-wait-i270-2026-11-22`.
