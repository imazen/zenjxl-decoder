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
and other time outside the two existing timers. The CLI's `buf_alloc`
counter is a constant zero in this build, so these timings do not separate
allocation and prefaulting from the rest of that residual.

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
