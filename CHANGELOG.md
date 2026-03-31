# Changelog

All notable changes to this project will be documented in this file.

This project is a fork of [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs). The changelog covers changes made in this fork.

## [0.3.4] - 2026-03-30

### Fixed

- **ICC amplification DoS** -- A crafted 19-byte JXL codestream could claim a huge ICC profile, causing the entropy decode loop to iterate hundreds of millions of times. Added a per-symbol progress check that detects degenerate streams producing output without consuming input (>1024:1 amplification ratio). Works correctly in both one-shot and incremental decode modes.

### Dependencies

- Updated all 55 dependencies to latest compatible versions, including `time` 0.3.46→0.3.47 (fixes GHSA-r6v5-fh4h-64xc stack exhaustion DoS), `archmage` 0.9.5→0.9.15, `zerocopy` 0.8.27→0.8.48.

## [0.3.3] - 2026-03-30

Ports 6 upstream bugfixes from libjxl/jxl-rs (March 2026) and 5 performance optimizations from PR #705. Yanks broken 0.3.2 release.

### Fixed (upstream ports)

- **vsqueeze grid boundary** (PR #731) -- Grid-based processing used wrong row when `has_tail=false` but `in_next_avg` exists, corrupting squeeze output.
- **hsqueeze grid boundary** (PR #735) -- Single-pixel-width shortcut looped over residual height instead of output height.
- **Stage pruning with shift** (PR #725, fuzzer-found) -- Pruning render pipeline stages with non-zero shift corrupted downstream channel dimensions.
- **EC upsampling validation** (PR #741) -- `check()` tested raw `ec_upsampling` instead of the effective value after `dim_shift`, allowing invalid configurations.
- **Mixed-upsampling patches** (PR #742) -- Patches with EC upsampling differing from color upsampling were silently accepted instead of rejected.
- **LF preview alpha overflow + BGR order** (PR #740) -- `1u8 << 8` overflowed to 0 (should be 255); BGR/BGRA output formats got RGB channel order.

### Performance

- **BitReader section padding** -- Append 8 zero bytes to section buffers so `refill()` always takes the fast 8-byte path, eliminating `refill_slow()` calls.
- **Property used-mask** -- Skip unused property computation per pixel. Trees typically split on 2-4 of 16 properties; the rest are now skipped.
- **HybridUint fast path** -- When `msb_in_token == 0`, simplify entropy decoding to `(1 << nbits) | bits`.
- **Inline annotation audit** -- 14 hot-path functions upgraded to `#[inline(always)]` across BitReader, ANS, Huffman, LZ77, and modular predict.
- **Blending SmallVec** -- Replace per-row `Vec` heap allocations with stack-based `SmallVec<[_; 8]>`.

Combined effect: **+4% to +16%** across VarDCT and modular images (single-threaded).

### Yanked

- **0.3.2** -- Broken release: `BitReader::new_padded` was changed to return `Result`, causing 478 of 1277 tests to fail (`SectionTooShort` on valid files).

## [0.3.1] - 2026-03-30

### Fixed

- **OOM from crafted JXL codestream headers** -- A 26-byte JXL header could request a 4.2GB allocation. Three fixes:
  - `Size::check()` now rejects `width * height > 2^30` during header parsing, before any pixel buffer allocation.
  - `alloc_zeroed_fallible` uses `try_reserve` instead of `vec![0u8; n]`, returning an error instead of aborting on allocation failure.
  - Default `max_pixels` lowered from 2^30 to 2^28 (256 megapixels).

## [0.3.0] - 2026-03-06

Initial public release of the zenjxl-decoder fork.

### Added

- **Resource limits** — `JxlDecoderLimits` API caps pixels, memory, ICC size, tree size, and more. `LimitExceeded` errors include the resource name, actual value, and limit.
- **Memory tracking** — `max_memory_bytes` budget with atomic, lock-free tracking across threads.
- **Fallible allocation** — All significant allocations return `TryReserveError` instead of panicking.
- **Cooperative cancellation** — `enough::Stop` trait integration lets any thread cancel or timeout decoding.
- **Parallel decoding** — Rayon-based parallel group decode and render via the `threads` feature.
- **CMS-based CMYK→RGB** — ICC profile-based CMYK conversion via optional `moxcms` backend (`cms` feature).
- **JPEG reconstruction** — Lossless JPEG reconstruction from JXL containers (`jpeg` feature).
- **`allow-unsafe` feature** — Opt-in `unsafe` fast paths in the main crate; safe fallbacks used by default.
- **`#![forbid(unsafe_code)]`** by default in the main `jxl` crate (without `allow-unsafe`).

### Fixed

- sRGB transfer function applied by default for XYB-encoded images (was outputting linear).
- RCT overflow panic via wrapping arithmetic on edge-case pixel values.
- Extra channel format slot allocation for all extra channels, not just the first.
- Progressive AC validation: `last_pass` must be strictly increasing.
- Extra channel bit depth: use the channel's own `bit_depth` for modular-to-f32 conversion.
- Noise seeding: separate RNG seeds per subregion for upsampled frames.
- CMYK blending order: blend in CMYK space, then CMS-convert to RGB.
