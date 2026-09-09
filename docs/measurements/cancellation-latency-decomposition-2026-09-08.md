# Where cancellation latency actually comes from — and a correction

Measured 2026-09-08, Apple M4 Pro, macOS 25.5, `zenjxl-decoder` at `5271be3`.

## The claim being corrected

An earlier note in this series reported that this fork's worst-case
cancellation latency on 108 MP VarDCT (40.3 ms for `-d 1`, 42.3 ms for `-d 3`,
against 6-12 ms everywhere else) was caused by **`check_cancelled()` being too
sparse on that path**, and proposed adding per-group checks. **That diagnosis
was wrong.** The checks are not sparse. No code change was warranted and none
was made.

## What the measurement actually shows

Instrumenting *every* cancellation-check site — both
`DecoderState::check_cancelled()` and the `stop.check()` calls inside the rayon
closures in `frame/render.rs` and `frame/decode.rs`, which do **not** go through
`check_cancelled` — and recording the longest wall-clock span in which no thread
checked at all:

| file | longest span with no check |
|---|---|
| `p108mp_d1` | ~3 ms |
| `p108mp_d3` | 3.4 ms (ending in `decode_lf_global`) |
| `p108mp_lossless` | 3.1 ms (ending in `decode_groups_parallel`) |

Phase timings for the same 108 MP `-d 1` decode (`JXL_PHASE_TIMING=1`) show
where the work is, and both hot phases check per item:

```
1692 groups (1 batch) | P1: 0.02ms | setup: 0.00ms | P2: 98.16ms | collect: 0.05ms
                      | P3a-store: 0.13ms | P3a-prep: 0.16ms | P3b: 102.73ms | P3c: 0.27ms
```

So ~3 ms is the real check interval, not 40 ms.

## The residual is parallel drain, not check density

Cancel fired at 41 points across the decode, latency measured from `cancel()` to
the call returning, at four thread counts:

| `RAYON_NUM_THREADS` | full decode | median | p95 | **max** |
|---|---|---|---|---|
| 1 | 1960.5 ms | 1.1 ms | 5.1 ms | **12.0 ms** |
| 2 | 1048.3 ms | 1.1 ms | 6.8 ms | **26.3 ms** |
| 4 | 589.9 ms | 1.3 ms | 8.0 ms | **23.8 ms** |
| 12 | 277.9 ms | 1.5 ms | 8.7 ms | **21.3 ms** |

Worst-case latency *rises* with thread count while the decode gets 7x faster.
That is the cost of `try_for_each` short-circuit semantics: once one worker
returns `Err`, rayon still has to let every in-flight group task finish and join
before the call can return. More threads means more in-flight work to drain.
Adding checks cannot shorten that; only smaller work items or a different
parallel primitive could.

Note also that the 40.3 / 42.3 ms figures in the earlier note were single
samples. Repeated here, the 12-thread max is 21.3 ms.

## Conclusion

Median cancellation latency is 1.1-1.5 ms and p95 is 5-9 ms at every thread
count. The tail is bounded by in-flight parallel work, which is a deliberate
throughput trade, not a defect. Left as-is.

Method: temporary instrumentation in `frame/mod.rs` and the `stop.check()` call
sites (reverted after measuring); `~/tmp/cancel-probe2` with `DENSE=1` for the
latency sweep.
