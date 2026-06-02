# Uninit output-buffer optimization — measurement (#758 decision)

**Date:** 2026-06-01
**Machine:** AMD Ryzen 9 7950X (16C/32T), 128 GB RAM, Linux (WSL2)
**Code under test:** `zenjxl-decoder` @ `1933c3d5` (main), `--features threads,allow-unsafe`, **no** `target-cpu=native`
**Harness:** `benches/decode_bench.rs` via zenbench, paired/interleaved A/B in one process
**Scaffold:** a runtime `FORCE_ZERO_INIT` toggle was added to `try_allocate`/`prefault_parallel` + a paired bench, run, then **reverted** (not committed). Toggle lets zenbench interleave the two variants so common-mode noise cancels.

## What was compared

- **`uninit`** (current): `new_uninit` skips zeroing (`Vec::set_len`, allow-unsafe) + `prefault_parallel` touches one byte per 4 KB page across rayon threads.
- **`zeroed`** (models upstream #758): `vec![0u8; n]` — calloc / lazy zero-pages (the same thing upstream's safe `Image::new` does), **no** prefault. `allow-unsafe` stays on in both so the `cast_row` path is identical; only buffer-init differs.

Swept **thread count** (`RAYON_NUM_THREADS` 1/4/8/16) × **image size** (0.26–25 MP).

## Result — uninit speedup vs zeroed (+ = uninit faster, − = uninit slower)

| image | output px | 1T | 4T | 8T | 16T |
|---|---:|---:|---:|---:|---:|
| portrait_4k_q75 | ~25 MP | **−6.4%** | +? (noisy) | **+16.5%** | **+28%** |
| city_4k_q75 | ~10 MP | **−4.3%** | noisy | **+14%** | **+20%** |
| cafe_web_q80 | ~2 MP | −3 to −8% | +4% | +8% | **+14%** |
| green_queen_vardct (small) | ~0.26 MP | ~0/−4% | noisy | ~0 | **−38%** |

Bold = tight paired CI (reliable). Raw throughputs (Mpix/s), 16T: portrait 127→176, city 115→148, cafe_web 78.5→91.3, small 53.7→41.9.

## Reading

1. **The optimization is regime-dependent, not a free win.**
   - **Win** only with a *large image + many threads*: +14–16 % @8T and +20–28 % @16T on the 4K images, +14 % @16T on web. This corroborates the old "+34 % @16T (193→259 MP/s)" note — but that note recorded *only* this favorable case.
   - **Loss** elsewhere: **−4 to −8 % at 1T on every size**, and **−38 % on the small image at 16T**.

2. **The culprit is `prefault_parallel`, not skip-zeroing.** At 1T the parallel prefault degenerates to a serial extra full-buffer pass (pure overhead on top of the render, which faults the same pages anyway) → the 1T regression. On a tiny buffer at 16T, 16 threads coordinating to fault ~0.26 MP is overhead-bound → −38 %. `vec![0; n]` is lazy-zero (calloc), so skip-zeroing alone is ~neutral at 1T; the regression tracks the prefault pass. (Not separately isolated here — a 3-way run would confirm, but the 1T sign + the small-image-at-16T sign both point at prefault.)

3. **Memory:** uninit and zeroed allocate identical bytes and fault identical pages; peak RSS is unchanged. This is purely a throughput tradeoff.

## Caveats

- The 4K decodes only yield **4 rounds** each (slow), so *absolute* numbers are thin; the **paired deltas** are reliable because both variants are interleaved under identical conditions and the trend is **monotonic and consistent across all four thread counts**.
- **4T is noisy** (the `zeroed` base saw ±155 ms scheduling variance on portrait) — treat it as the crossover/transition region between the 1T loss and the 8T win.

## Implication for the #758 decision

This fork is **web-focused** (per CLAUDE.md), where the common patterns are (a) small-to-medium images and (b) one single-threaded decode per concurrent request. The optimization is a **net loss in both** of those patterns and a **soundness liability** (the lone `unsafe` under `allow-unsafe`). It wins only for "one large image decoded with a big thread pool" (desktop/batch), where it's +20–28 % at 16T.

**Options (decision is the user's):**
- **A. Remove (adopt #758):** simplest, restores soundness, *improves* the common web cases; gives up only the large-image-many-threads win.
- **B. Gate `prefault_parallel`** on `output_bytes ≥ ~4 MB && rayon::current_num_threads() ≥ ~8` (keep `new_uninit`): keeps the high-T-large win, removes the 1T and small-image regressions. Still carries the `unsafe`.
- **C. Keep `new_uninit` (skip-zero), drop `prefault_parallel` unconditionally:** likely removes the regressions while keeping any skip-zero benefit — needs the 3-way confirm.

## Raw data
Per-thread zenbench logs: `/tmp/uninit_paired_T{1,4,8,16}.log` (ephemeral). Key paired lines transcribed in the table above. Git: `1933c3d5a3830ff30f69e156759bf3b2c1d7db50`.
