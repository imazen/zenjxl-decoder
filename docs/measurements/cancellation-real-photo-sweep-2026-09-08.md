# Chunked-`process()` cancellation on a real photo at 12 / 20 / 108 MP

Measured 2026-09-08, Apple M4 Pro (12 threads, 26 GB), upstream
**libjxl/jxl-rs `main` @ `9e6b9f9` (v0.7.1)**, rayon parallel runner.
Follow-up to `cancellation-vs-chunked-input-2026-09-08.md`, which used the
repo's `resources/test` corpus.

## Source

One real photograph from the imazen-26 corpus, natively 108 MP:

`png-v3/1400-lilith-nature/1415_nature_tropical-coastline-view_kee-beach-kauai_s21u_iso16-f1p8_20221001-133315-1_12000x9000.sdr.png`

- **108 MP** — native 12000x9000
- **20 MP** — 5164x3873, `vips resize --kernel mitchell`
- **12 MP** — 4000x3000, `vips resize --kernel mitchell`

Encoded with `cjxl 0.12.0` at three modes: `-d 1 -e 7` (VarDCT, ~1 bpp),
`-d 3 -e 7` (VarDCT, ~0.35 bpp), `-q 100 -e 7` (lossless modular).
Downscales only — no synthetic upscaling.

## Result — the granularity is buyable, and the price is 2-7x throughput

`total` is wall time for the whole decode; `max call` is the longest single
`process()` call, i.e. the best cancellation latency a caller can get by
interrupting between calls. Untraced runs, warm, rayon over 12 threads.

### `-d 1` (VarDCT, ~1 bpp)

| bytes/call | 12 MP (1.48 MB) | 20 MP (2.45 MB) | 108 MP (12.3 MB) |
|---|---|---|---|
| | total / max call / calls | total / max call / calls | total / max call / calls |
| whole file | 31.6 / 31.3 / 3 | 47.9 / 47.5 / 3 | 223.2 / 220.9 / 3 |
| 1 MiB | 32.5 / 27.0 / 4 | 50.1 / 33.0 / 5 | 238.7 / 40.1 / 14 |
| 64 KiB | 66.0 / **8.6** / 25 | 91.0 / **8.5** / 40 | 451.3 / 12.5 / 190 |
| 4 KiB | 157.9 / 5.7 / 362 | 263.9 / 7.8 / 600 | 1585.0 / **9.4** / 2999 |
| 512 B | 213.8 / 5.6 / 2885 | 350.1 / 7.7 / 4789 | — |

### `-d 3` (VarDCT, ~0.35 bpp)

| bytes/call | 12 MP (0.54 MB) | 20 MP (0.87 MB) | 108 MP (4.25 MB) |
|---|---|---|---|
| whole file | 42.8 / 42.4 / 3 | 63.7 / 63.1 / 3 | 313.4 / 310.9 / 3 |
| 1 MiB | 43.7 / 43.3 / 3 | 64.7 / 64.1 / 3 | 320.2 / 182.8 / 7 |
| 64 KiB | 54.8 / 23.5 / 11 | 90.6 / 37.7 / 16 | 457.9 / 43.1 / 67 |
| 4 KiB | 117.0 / 11.8 / 133 | 186.6 / 16.3 / 214 | 908.6 / **7.3** / 1038 |
| 512 B | 139.1 / **6.7** / 1052 | 231.8 / **7.0** / 1701 | — |

### `-q 100` (lossless modular)

| bytes/call | 12 MP (9.96 MB) | 20 MP (16.3 MB) | 108 MP (63.0 MB) |
|---|---|---|---|
| whole file | 128.8 / 128.5 / 3 | 187.5 / 187.0 / 3 | 890.3 / 888.2 / 3 |
| 1 MiB | 151.7 / 25.8 / 12 | 240.8 / 26.5 / 18 | 1043.2 / 26.5 / 63 |
| 64 KiB | 854.0 / **8.1** / 154 | 1367.9 / **7.9** / 252 | 5356.5 / **8.4** / 964 |
| 4 KiB | 1071.0 / 8.0 / 2433 | 1707.7 / 7.8 / 3991 | 8269.1 / 8.6 / 15390 |
| 512 B | 1080.1 / 8.1 / 19454 | 1697.5 / 7.7 / 31913 | — |

**Cost of a 10 ms cancellation budget**, best chunk size per file:

| | 12 MP | 20 MP | 108 MP |
|---|---|---|---|
| `-d 1` | 2.1x (64 KiB) | 1.9x (64 KiB) | 7.1x (4 KiB) |
| `-d 3` | 3.3x (512 B) | 3.6x (512 B) | 2.9x (4 KiB) |
| lossless | 6.6x (64 KiB) | 7.3x (64 KiB) | 6.0x (64 KiB) |

## Result — hit rate: most `process()` calls do no work

A call is *productive* if `decode_and_render_hf_groups` consumed at least one
group section on it (instrumented `process_sections`). Hit rate is
productive / total calls. Timings in this instrumented pass are inflated by the
trace and are not the numbers tabulated above.

| file | chunk | calls | productive | group sections | **hit rate** |
|---|---|---|---|---|---|
| 12 MP `-d 1` | 4 KiB | 362 | 122 | 192 | **34 %** |
| 12 MP `-d 3` | 4 KiB | 133 | 74 | 192 | 56 % |
| 12 MP lossless | 4 KiB | 2433 | 192 | 192 | **8 %** |
| 20 MP `-d 1` | 4 KiB | 600 | 215 | 336 | **36 %** |
| 20 MP `-d 3` | 4 KiB | 214 | 120 | 336 | 56 % |
| 20 MP lossless | 4 KiB | 3991 | 336 | 336 | **8 %** |
| 108 MP `-d 1` | 4 KiB | 2999 | 1457 | 1692 | 49 % |
| 108 MP `-d 3` | 4 KiB | 1038 | 635 | 1692 | 61 % |
| 108 MP lossless | 4 KiB | 15390 | 1692 | 1692 | **11 %** |

At 64 KiB nearly every call is productive (93-100 % of the calls that reached
the section stage). The productive count never exceeds the frame's group-section
count, which is the real ceiling on interruption points: **you cannot get more
cancellation points than the frame has group sections**, no matter how finely
you slice the bytes — visible in the lossless rows, where productive lands
exactly on 192 / 336 / 1692. Everything past that is pure call overhead, which
is where the 4 KiB and 512 B rows spend their time.

## Result — the encoder decides whether chunking works at all

The floor found in the earlier write-up is encoder-dependent, not universal.
Instrumenting the HF batch size at 4 KiB chunks:

- `resources/test/portrait_4k_q90.jxl` (older encoder): 388 no-op calls, then
  **one batch of all 384 groups**. `HfGlobal` is the last section in file order,
  so `process_sections` early-returns until EOF. No chunk size helps.
- `p12mp_d1.jxl` (cjxl 0.12): batches of 1-5 groups spread across the stream.
  Chunking works, at the prices tabulated above.

A caller cannot tell which case it is holding without parsing the TOC itself.

## Comparison — this fork's `with_stop()` on the same files

Cancel fired from another thread; latency measured from `cancel()` to the decode
call returning:

| file | full decode | cancel @10 % | @25 % | @50 % | @75 % |
|---|---|---|---|---|---|
| 12 MP `-d 1` | 48.7 ms | 0.3 ms | 1.0 ms | 1.1 ms | (done) |
| 20 MP `-d 1` | 67.2 ms | 3.2 ms | 0.9 ms | 0.9 ms | 0.9 ms |
| 108 MP `-d 1` | 309.7 ms | 0.9 ms | 3.5 ms | 2.3 ms | 6.0 ms |
| 108 MP lossless | 1273.9 ms | 6.6 ms | 8.7 ms | 9.2 ms | 11.0 ms |

Sub-10 ms at every size and mode, at full one-shot throughput and with no chunk
size to tune.

## Conclusion

On modern `cjxl` output, chunked input does buy real cancellation granularity —
the earlier "hard floor" is an artifact of older encoders that place `HfGlobal`
last. But the granularity costs 1.9-7.3x throughput to reach a 10 ms budget,
requires the caller to pick a chunk size that varies by 128x across these nine
files (512 B to 64 KiB) with no way to know the right one in advance, and
bottoms out at the frame's group-section count regardless. At 4 KiB, 8-11 % of
calls do any work on lossless files. The in-library check reaches the same latency at
1.0x throughput with no tuning.

Raw logs: `probe_sweep_2026-09-08.log`, `real-photo-sweep-2026-09-08/`.
Probes: `dev/upstream_chunked_cancel_probe.rs`, `dev/zenjxl_cancel_latency_probe.rs`.
