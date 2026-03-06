# Changelog

All notable changes to this project will be documented in this file.

This project is a fork of [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs). The changelog covers changes made in this fork.

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
