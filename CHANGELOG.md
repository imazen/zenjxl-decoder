# Changelog

All notable changes to this project will be documented in this file.

This project is a fork of [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs). The changelog covers changes made in this fork.

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal -- batch them. -->

## [0.3.8] - 2026-04-17

### Added

- **`basic_info()` embedded-profile guard** -- Ported from jxl-rs #745; hides `basic_info()` until the embedded ICC/color-profile box is parsed, preventing callers from observing partial image-info state. Adds integration tests using `cmyk_layers.jxl` and `basic.jxl` (fa4400f, 470a6f4).
- **Shared `apply_decoder_options()` helper** -- Routes both the primary `DecoderState` creation and the preview-recovery recreation through one helper so the two sites cannot drift (5bb4632).
- **Chunked drip-decode animation stress test** -- Feeds `animation_newtons_cradle.jxl` through `JxlDecoderInner::process` in 1 KiB chunks and asserts no error or panic at any boundary, mirroring the Chrome integration repro from jxl-rs #743 (5bb4632).
- **EC-upsampling-after-dim_shift regression tests** -- Ports the test harness from jxl-rs #741 (negative, positive, and real-file cases) so future refactors cannot silently regress the existing `check()` validation (4c7cbcc, e96e434).

### Changed

- **Decoder options preserved across preview-frame recovery** -- When a JXL file contains a preview frame, `sections::handle_frame_finalized` previously recreated `DecoderState` for the main frame while copying only four of nine fields. `high_precision`, `premultiply_output`, `parallel`, `memory_tracker`, and `embedded_color_profile` silently reverted to defaults. Ported as the independent subset of upstream jxl-rs #743 that does not depend on the animation seek API in #678 (5bb4632, 3e369a1).
- **Blanket `#![allow(dead_code, unused_imports)]` removed from `lib.rs`** -- Dead code is now surfaced and handled individually instead of suppressed crate-wide; removed 9 unused imports and added targeted `#[cfg(test)]` / `#[allow(dead_code)]` with comments where items are intentionally kept (c84147f, b83639d, PR #13).
- **Clippy passes under `-D warnings` for both `--all-features` and `--no-default-features`** -- Added `#[cfg(feature = ...)]` gates and targeted `#[allow(dead_code)]` on thread-only and jpeg-only items; removed unused imports; suppressed `field_reassign_with_default` where the `#[non_exhaustive]` struct-literal rewrite is not available to external callers (265dc07, b1b8e39, a385d59, cd6d5fa).

### Fixed

- **Memory budget bypass on preview-bearing files** -- A caller asking for a restrictive `max_memory_bytes` saw the budget enforced on the preview frame but silently dropped to unlimited for the main frame; `parallel=false` was flipped back on; CMYK ICC detection misfired (5bb4632).
- **ClusterFuzzLite build paths** -- `build.sh` referenced the upstream `jxl-rs/jxl/` subdirectory layout; corrected to the fork's root workspace and `zenjxl-decoder/resources/test/` location (1a49043).
- **Benchmark test image paths** -- Updated from the upstream `jxl/resources/test/` prefix to `zenjxl-decoder/resources/test/` (3f18f86).
- **Missing copyright headers** -- Added headers to fuzz targets, the nightly fuzz workflow, and the fuzz script, caught by the `source_checks` CI job (e91149d).

### Dependencies

- Bumped `rand` 0.10.0 -> 0.10.1 (ee04a49).

### Docs

- README now lists `wasm128` under the all-SIMD row and per-ISA row (already wired in `Cargo.toml` and the `-simd` crate) (5055dae).
- Added crate-level rustdoc to `zenjxl-decoder/src/lib.rs` describing the fork, entry points, SIMD dispatch, safety posture, and feature flags, with credit to upstream libjxl/jxl-rs and libjxl under BSD-3-Clause (5055dae).
- Backfilled `[0.3.6]` and `[0.3.7]` CHANGELOG sections from git log (5055dae).

## [0.3.7] - 2026-04-10

### Fixed

- **i686 address space exhaustion in test suite** -- Test suite ran out of 32-bit virtual address space under parallel execution (d4b1167).
- **32-bit memory limit** -- Raised the default 32-bit memory limit to 2 GB so correctness tests fit on i686 (ce5b5e0).
- **Large-image tests gated on 64-bit** -- Tests that require >2 GB address space are now excluded from 32-bit targets (ee65a6f).
- **slow_probe_regression timing** -- Raised threshold from 5 ms to 10 ms to stabilise CI against loaded runners (4ea9ba5).
- **Memory limit disabled in correctness tests** -- Correctness tests no longer trip the default cap on large conformance images (727f00c).

## [0.3.6] - 2026-04-10

### Added

- **cargo-fuzz infrastructure** -- Three fuzz targets and a JXL format dictionary for continuous fuzzing (d9cfa74).
- **Nightly fuzz workflow** -- 60-second fuzz run on every push, 5-minute run nightly (8086be0).
- **BitReader panic regression seeds** -- Captured regression seeds for the `BitReader::new_padded` panic fixed in 0.3.5 (c5460e2).
- **Minimized OOM regression seed** -- 781-byte seed reproducing the crafted-header OOM fixed in 0.3.1 (91cc64d).

### Changed

- **Default `max_memory_bytes` lowered to 4 GB** -- Prevents OOM from crafted inputs in default configuration; raise explicitly via `JxlDecoderLimits` for large images (b1693bf).
- **Clippy runs once on Ubuntu** -- Removed redundant per-platform clippy jobs from CI; other platforms still run tests (c2026cd).

### Fixed

- **MemoryGuard::forget() leak** -- `MemoryGuard::forget()` leaked 32 bytes per tracked image allocation; the guard now releases its accounting slot on drop as well as on explicit forget (14e7739).
- **OOM-test clippy lint** -- Resolved `field_reassign_with_default` on the non-exhaustive options struct in the OOM regression test (8686bea).

## [0.3.5] - 2026-04-01

### Fixed

- **Huffman alphabet ratio overflow** -- Increased `ALPHABET_BITS_RATIO` from 32 to 256 to prevent false rejections of valid streams.
- **Shift overflow in property mask** -- Prevent shift overflow in `compute_used_property_mask` for large property indices.
- **Section padding for non-section buffers** -- Add `SECTION_PADDING` to non-section buffer allocation to prevent out-of-bounds reads during BitReader refill.
- **ANS alias map validation** -- Replace `assert!` in `build_alias_map` with proper error returns for malformed streams.
- **Flat tree child_id bounds checking** -- Validate child_id references in flat trees to prevent out-of-bounds access.
- **HybridUint nbits overflow** -- Track `nbits>=32` overflow in `ErrorState` for deferred reporting instead of silent corruption.
- **Memory tracker threading** -- Thread `MemoryTracker` to local modular tree decoding for accurate accounting.

### Performance

- **HybridUint OR-accumulator** -- Use OR-accumulator for overflow detection, reducing branches in the hot path.

### Dependencies

- Updated moxcms to 0.8.1 (from crates.io, with `extended_range` + `options` features).
- Updated wasm-bindgen 0.2.117, js-sys/web-sys 0.3.94.
- Updated archmage 0.9.16, zenbench 0.1.3, libc 0.2.184.

### CI

- Added full CI matrix with i686 cross-compilation, macOS Intel, windows-11-arm.
- Reduced i686 test parallelism to 2 threads for address space constraints.

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
