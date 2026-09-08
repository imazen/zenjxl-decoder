# Can chunked `process()` calls substitute for in-library cancellation?

Measured 2026-09-08 on an Apple M4 Pro (12 threads, macOS 25.5), against
**upstream libjxl/jxl-rs `main` @ `9e6b9f9` (v0.7.1)** and this fork at
`17dc3030`.

> **Correction (same day):** the "chunking buys nothing" floor documented in
> Result 1 below is **encoder-dependent, not universal**. It reproduces on this
> repo's older `resources/test` fixtures, which place `HfGlobal` last. On
> `cjxl 0.12.0` output, chunking does buy granularity — at a 1.9-7.3x throughput
> cost. See `cancellation-real-photo-sweep-2026-09-08.md` for the real-photo
> sweep at 12 / 20 / 108 MP. Results 2, 3 and 4 below are unaffected.

## Question

libjxl/jxl-rs#897 proposed cooperative cancellation. The maintainer's response:

> I don't see why jxl-rs should implement this: the same result can be achieved
> outside the library by feeding the data incrementally to `.process()`, and
> interrupting the decoding process upon request between calls to process().

The mechanism is real — `JxlBitstreamInput` is caller-supplied, `process()`
returns `NeedsMoreInput`, and `jxl_cli` already uses exactly this shape via
`with_capped_size` / `--render-interval`. The question measured here is whether
it delivers *the same result*, i.e. a bounded cancellation latency.

## Method

`dev/upstream_chunked_cancel_probe.rs` drives the upstream public API with an
input that hands out at most `N` bytes per `process()` call, and records the
duration of every individual `process()` call. The largest such duration is the
best achievable cancellation latency for a caller that can only interrupt
*between* calls. Both an honest `available_bytes()` (full remaining length) and
a capped one were tried; the results are within noise of each other.

`dev/zenjxl_cancel_latency_probe.rs` measures this fork's `with_stop()` handle:
fire a cancel from another thread at a fraction of the decode and time how long
the decode call takes to return afterwards.

## Result 1 — feeding smaller chunks stops helping almost immediately

`portrait_4k_q90.jxl` (5.6 MB, 4096x6137, standard `cjxl` output), rayon over
12 threads:

| bytes fed per `process()` | total decode | largest single call | calls |
|---|---|---|---|
| whole file at once | 70.6 ms | 70.0 ms | 3 |
| 1 MiB | 102.7 ms | 60.6 ms | 8 |
| 64 KiB | 107.9 ms | 57.8 ms | 88 |
| 4 KiB | 109.0 ms | 57.9 ms | 1376 |
| 512 B | 117.3 ms | 62.4 ms | 10993 |

Cutting the chunk by 2000x moves the latency floor by nothing. Single-threaded,
the same file at 4 KiB chunks spends **487 ms of its 538 ms in one call**.

Cause, confirmed by instrumenting `process_sections`: `cjxl` writes `HfGlobal`
as the *last* section in file order, and `process_sections` early-returns at
`if !self.section_state.hf_global_done` until it arrives. All 384 HF groups
then decode and render in a single `decode_and_render_hf_groups` batch
(traced: `n=384, 57.2 ms`) on the final call, no matter how the input was fed.

Across the whole `resources/test` corpus (33 files whose one-shot decode takes
>= 5 ms) at 4 KiB chunks, the largest single call is a median **61 %** of the
whole-file decode time (min 20 %, max 132 % — over 100 % because chunking adds
its own overhead), while total time inflates by a median 1.4x and up to 4.0x
(`clic_landscape_d1` 16.6 -> 66.3 ms, `green_queen_modular_e3` 4.4 -> 16.8 ms).
Part of that cost is structural: `available_bytes()` doubles as the parallelism
hint, so starving the input starves the thread pool. Eight of the 33 got no
extra interruption points at all — still 3 `process()` calls at 4 KiB.

## Result 2 — when bytes are few and pixels are many, there is nothing to chunk

`cjxl --resampling=8` on a solid image. Input size is decoupled from decode work,
so the caller cannot manufacture interruption points at all:

| file | input | output | `process()` calls | largest call (12 threads) |
|---|---|---|---|---|
| `r8_2000.jxl` | 146 B | 2000x2000 | 3 | 20.2 ms |
| `r8_4000.jxl` | 310 B | 4000x4000 | 3 | 82.5 ms |
| `r8_8000.jxl`\* | 411 B | 8000x8000 | 3 | 51.9 ms |
| `r8_12000.jxl` | 495 B | 12000x12000 | 3 | 104.2 ms |

Generated with `cjxl solid.png out.jxl -d 1 -e 7 --resampling=8` on a
solid-colour PNG of the given size (\* the 8000 entry was encoded from a
separately generated source and lands off the trend; the point is the count of
`process()` calls, which is 3 in every row).

`r8_12000.jxl` single-threaded: **721 ms in one uninterruptible call from 495
bytes of input**, and it scales with pixel count, not file size. A 4 KiB chunk
cap is already larger than the entire file.

## Result 3 — the caller's `JxlParallelRunner` is not a usable hook either

`JxlParallelRunner::run` returns `Result<()>`, so in principle a runner could
abort. In practice `run_ordered`'s default calls the caller's `run` **once per
parallel region** with `num = max_threads`, and each task then loops over the
group indices internally without re-entering the caller. Measured per decode:

- `portrait_4k_q90.jxl` (384 HF groups): **2** `run()` invocations, 18 task closures.
- `r8_12000.jxl`: **1** `run()` invocation, 12 task closures.

And when `max_threads <= 1`, `run_ordered` calls `fun(i)` directly and never
enters the caller's runner at all. `jxl::error::Error` also has no cancellation
variant (126 variants, none for abort), so such a runner would have to hijack an
unrelated error.

## Result 4 — in-library checks land where chunking cannot

This fork's `JxlDecoderOptions::with_stop()`, cancel fired from another thread
mid-decode; latency is measured from the `cancel()` call to the decode returning:

| file | full decode | cancel at 10 % | at 25 % | at 50 % |
|---|---|---|---|---|
| `portrait_4k_q90.jxl` | 85.4 ms | 2.4 ms | 1.1 ms | 1.0 ms |
| `r8_12000.jxl` (495 B) | 177.1 ms | 25.5 ms | 19.1 ms | 35.9 ms |

Not apples-to-apples — different codebase — but it bounds what per-group checks
buy: ~1-2 ms on the photo where chunked input floors out at 58 ms (487 ms
single-threaded), and 19-36 ms on the bomb where chunked input cannot interrupt
at all. The remaining coarseness on the bomb case is this fork's own check
placement, not a limit of the approach.

## Conclusion

Chunked input gives coarse, best-effort cancellation between frames and between
*section-completion batches*. It does not give a latency bound: the indivisible
unit is one frame's pixel decode, whose wall time is a function of pixel count
and is unbounded with respect to how little input the caller hands over. The
maintainer's mechanism works; the claim that it achieves "the same result" does
not hold on `main` as measured.

Reproduce with `dev/upstream_chunked_cancel_probe.rs` (path-dep on an upstream
checkout) and `dev/zenjxl_cancel_latency_probe.rs`.
