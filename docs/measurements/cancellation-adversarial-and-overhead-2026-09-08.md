# TOC order, adversarial layouts, and the 10 %-overhead granularity limit

Measured 2026-09-08, Apple M4 Pro (12 threads), upstream **libjxl/jxl-rs `main`
@ `9e6b9f9` (v0.7.1)**, rayon runner. Third instalment; see
`cancellation-vs-chunked-input-2026-09-08.md` and
`cancellation-real-photo-sweep-2026-09-08.md`. Same source photo
(imazen-26 `1415_nature_tropical-coastline-view_kee-beach-kauai`, 12000x9000,
Mitchell-downscaled to 20 MP / 12 MP).

## 1. Does `jxl-encoder` put `HfGlobal` last? No.

Section emission order is `[dc_global, dc_groups.., ac_global, ac_groups..]` —
`ac_global` (`HfGlobal`) is pushed **before** every AC group
(`jxl-encoder/src/vardct/encoder.rs:6577`, `:6603`, `:6617`, `:6644`). The
center-first reordering permutes only the AC groups: it builds an identity
prefix of `2 + num_dc_groups` entries and offsets the AC permutation past it
(`:6678-6686`), so `HfGlobal` keeps its position. The single-group path
(`:6422-6511`) concatenates everything into one section, where ordering is moot.

Confirmed empirically — `jxl-encoder` output chunks like `cjxl 0.12` output,
not like the HfGlobal-last fixtures:

| file | one-shot | max call @1 MiB | @64 KiB | @4 KiB |
|---|---|---|---|---|
| 12 MP `-d 1` | 31.4 ms | 26.2 | 8.3 | **5.0** |
| 20 MP `-d 1` | 49.4 ms | 32.7 | 7.6 | **7.4** |
| 12 MP lossless | 144.5 ms | 33.6 | 8.7 | **9.5** |
| 20 MP lossless | 253.6 ms | 36.8 | 8.9 | 21.9 |
| 108 MP lossless | 2238.1 ms | 48.0 | 9.7 | **9.0** |

(108 MP lossy was not measured: `jxl-encoder` refuses it at the default 4 GB
budget — "estimated peak working set 14956428800 bytes ... exceeds memory budget
cap".)

## 2. Adversarial layouts — and you don't have to be adversarial

The HfGlobal-last layout that defeats chunking entirely is not exotic and needs
no crafting: **stock `cjxl 0.12.0 --output_mode=2`** (out-of-order `jxlp`
streaming output, a documented flag) produces it at any size.

| file | input | output | one-shot | max call @4 KiB | calls | best @<=10 % overhead |
|---|---|---|---|---|---|---|
| `m12mp_stream` (`-d 1 --output_mode=2`) | 1.48 MB | 4000x3000 | 34.1 ms | **32.7 ms** | 364 | 32.7 ms (**1.0x**) |
| `adv108_d1_stream` (`-d 1 --output_mode=2`) | 12.3 MB | 12000x9000 | 235.4 ms | **226.4 ms** | 3004 | 230.0 ms (**1.0x**) |
| `--output_mode=1` (seeking) 12 MP | 1.48 MB | 4000x3000 | 32.2 ms | 25.3 ms | 362 | — |
| `--buffering=2` 12 MP (control) | 1.48 MB | 4000x3000 | 31.2 ms | 5.7 ms | 362 | — |

3004 `process()` calls, and the largest is still 96 % of the whole decode. No
chunk size in the 32 KiB - 16 MiB sweep improves on it.

The other axis is decoupling bytes from work, where there is nothing left to
chunk:

| file | input | output | calls @4 KiB | max call (12 threads) | single-threaded |
|---|---|---|---|---|---|
| `adv108_ups` (`--resampling=8`) | 242 KB | 12000x9000 | 61 | 39.8 ms | — |
| `r8_12000` (solid, `--resampling=8`) | **495 B** | 12000x12000 | **3** | 104 ms | **721 ms** |
| `adv67_single` (1024x1024 x8) | **502 B** | 8192x8192 (1 section) | **3** | 51.9 ms | — |

A 4 KiB chunk cap is already larger than the whole file. Combining both axes
(HfGlobal-last **and** large) is the worst case: a 108 MP lossless
`--output_mode=2` file would be ~890 ms in a single uninterruptible call. That
one could not be measured — see the bug below.

### Upstream bug found while building these

`jxl-rs main @ 9e6b9f9` fails with **"Source file truncated"** on valid files
from stock `cjxl 0.12.0 -q 100 -e 7 --output_mode=2`. `djxl 0.12.0` decodes them
correctly. Bisected on square crops of the same photo:

| size | jxl-rs | djxl |
|---|---|---|
| 512x512 | OK | OK |
| 1024x1024 | OK | OK |
| 2048x2048 | OK | OK |
| **3072x3072** | **FAIL** | OK |
| 4096x4096 | FAIL | OK |
| 4000x3000, 5164x3873, 12000x9000 | FAIL | OK |

Lossy `--output_mode=2` decodes fine at every size tested, including 108 MP, so
this is specific to the modular/lossless path. All these files carry
out-of-order `jxlp` boxes whose "last" flag is not on the physically final box
(indices run `0,1,3,4,35..42,82..89,...` with two boxes appended after the
flagged one). Minimal repro: 3072x3072 crop, 4.1 MB.

## 3. How granular can chunking get within a 10 % throughput budget?

Chunk sizes swept 32 KiB - 16 MiB, median of 3 measured runs each, picking the
chunk with the smallest max call whose total stays within 1.10x the one-shot
decode:

| file | one-shot | best chunk | **max call** | vs one-shot | actual overhead |
|---|---|---|---|---|---|
| 12 MP `-d 1` | 31.5 ms | 256 KiB | **15.6 ms** | 2.0x | 8.9 % |
| 12 MP `-d 3` | 39.3 ms | 256 KiB | **28.6 ms** | 1.4x | 2.5 % |
| 12 MP lossless | 119.4 ms | 2 MiB | **36.0 ms** | 3.3x | 7.0 % |
| 20 MP `-d 1` | 48.5 ms | 512 KiB | **17.1 ms** | 2.8x | 6.0 % |
| 20 MP `-d 3` | 66.1 ms | 256 KiB | **44.8 ms** | 1.5x | 8.3 % |
| 20 MP lossless | 190.8 ms | 4 MiB | **65.6 ms** | 2.9x | 3.6 % |
| 108 MP `-d 1` | 239.4 ms | 1 MiB | **43.5 ms** | 5.3x | 5.4 % |
| 108 MP `-d 3` | 332.2 ms | 512 KiB | **143.3 ms** | 2.3x | 4.9 % |
| 108 MP lossless | 934.4 ms | 2 MiB | **51.1 ms** | 18.1x | 9.8 % |
| 12 MP `--output_mode=2` | 34.1 ms | (any) | **32.7 ms** | 1.0x | — |
| 108 MP `--output_mode=2` | 235.4 ms | (any) | **230.0 ms** | 1.0x | — |

**Answer: 16-143 ms on well-behaved files, and the whole decode on streaming-TOC
files.** Never below ~15 ms even at 12 MP. The best chunk size varies 16x across
the nine well-behaved files (256 KiB to 4 MiB) with no rule the caller can
derive — 20 MP lossless wants 4 MiB, 108 MP lossless wants 2 MiB, 108 MP `-d 3`
wants 512 KiB — and the wrong pick is expensive in both directions.

### The in-library check, for scale

Cost of carrying a live (never-firing) `enough::Stop` token through this fork's
decoder, median of 7 paired runs:

| file | no stop | with stop | overhead |
|---|---|---|---|
| 12 MP `-d 1` | 32.8 ms | 32.5 ms | **-0.8 %** |
| 20 MP `-d 1` | 51.6 ms | 51.8 ms | **+0.3 %** |
| 108 MP `-d 1` | 241.4 ms | 243.2 ms | **+0.7 %** |
| 108 MP lossless | 1441.7 ms | 1444.2 ms | **+0.2 %** |

Within noise, and it delivers 0.3-11.0 ms cancellation latency (previous
write-up) rather than 16-143 ms — with no chunk size to tune and no dependence
on how the encoder laid out the TOC.

## Conclusion

- `jxl-encoder` does **not** produce the pathological layout; its output is
  chunkable.
- The pathological layout is one stock `cjxl` flag away (`--output_mode=2`), so
  a caller relying on chunked cancellation is one encoder setting from having
  none — and an attacker can also simply decouple bytes from pixels.
- Inside a 10 % throughput budget, chunking buys 16-143 ms of cancellation
  latency on cooperative files and nothing at all on hostile ones. The
  in-library check buys 0.3-11 ms at 0 % on both.

Raw logs: `adversarial-2026-09-08/`.
Probes: `dev/upstream_chunked_cancel_probe.rs`, `dev/zenjxl_cancel_latency_probe.rs`.
