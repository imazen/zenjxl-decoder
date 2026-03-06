# zenjxl-decoder

A JPEG XL decoder in safe Rust. Fork of [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs) with security hardening, bug fixes, and parallel decoding.

Maintained by [Lilith River](https://github.com/lilith) at [Imazen](https://github.com/imazen).

```toml
[dependencies]
zenjxl-decoder = "0.3"
```

The Rust lib name is `jxl`, so you `use jxl::...` in code.

If you find a conformant JXL file that decodes incorrectly (or not at all), [open an issue](https://github.com/imazen/zenjxl-decoder/issues/new).

## What's different from upstream

This fork fixes several decoding bugs, adds resource limits and cooperative cancellation, and wires up parallel decoding via rayon. All changes are BSD-3-Clause licensed under the Google CLA, intended for upstream contribution.

### Bug Fixes

| Bug | Description |
|-----|-------------|
| sRGB Transfer Function | Apply sRGB TF by default for XYB-encoded images (was outputting linear) |
| RCT Overflow Panic | Wrapping arithmetic to prevent panic on edge-case pixel values |
| Extra Channel Format Slots | Allocate format slots for all extra channels, not just first |
| Progressive AC Validation | Fix inverted `last_pass` validation (must be strictly increasing) |
| Extra Channel Bit Depth | Use extra channel's own `bit_depth` for modular-to-f32 conversion |
| Noise Seeding (upsampling > 1) | Separate RNG seeds per subregion for upsampled frames |
| CMYK Blending Order | Blend in CMYK space, then CMS-convert to RGB |

### New Features

| Feature | Description |
|---------|-------------|
| CMS-based CMYK to RGB | ICC profile-based CMYK conversion via optional `moxcms` backend |
| Cooperative cancellation | `enough::Stop` trait — cancel or timeout decoding from any thread |
| Resource limits | `JxlDecoderLimits` API — cap pixels, memory, ICC size, tree size, etc. |
| Memory tracking | `max_memory_bytes` budget with atomic, lock-free tracking |
| Fallible allocation | All significant allocations return `TryReserveError` instead of panicking |
| Parallel decoding | Rayon-based parallel group decode and render (opt-in via `threads` feature) |

## Usage

### Basic decode

```rust
use jxl::api::{JxlDecoder, JxlDecoderOptions};

let data = std::fs::read("image.jxl")?;
let mut decoder = JxlDecoder::new(&data, JxlDecoderOptions::default());
// ... process frames
```

### Resource limits

```rust
use jxl::api::{JxlDecoderLimits, JxlDecoderOptions};

// For untrusted input
let options = JxlDecoderOptions {
    limits: JxlDecoderLimits::restrictive(),
    ..Default::default()
};
```

| Limit | Default | Restrictive |
|-------|---------|-------------|
| `max_pixels` | 2^30 (~1B) | 100M |
| `max_extra_channels` | 256 | 16 |
| `max_icc_size` | 256 MB | 1 MB |
| `max_tree_size` | 4M nodes | 1M nodes |
| `max_patches` | (derived) | 64K |
| `max_spline_points` | 1M | 64K |
| `max_reference_frames` | 4 | 2 |
| `max_memory_bytes` | None | 1 GB |

All limits return `Error::LimitExceeded { resource, actual, limit }` when exceeded.

### Cancellation

The decoder accepts any [`enough::Stop`](https://docs.rs/enough) implementation:

```rust
use almost_enough::Stopper;
use std::sync::Arc;

let stop = Arc::new(Stopper::new());
let stop_clone = Arc::clone(&stop);

// Cancel from another thread
std::thread::spawn(move || {
    std::thread::sleep(std::time::Duration::from_secs(5));
    stop_clone.cancel();
});

let options = JxlDecoderOptions {
    stop,
    ..Default::default()
};
```

Cancellation checks run inside parallel closures too, so a cancel request is noticed mid-batch.

### Parallel decoding

Enable the `threads` feature (on by default in `jxl_cli`):

```toml
[dependencies]
zenjxl-decoder = { version = "0.3", features = ["threads"] }
```

Set `parallel: false` on `JxlDecoderOptions` to force single-threaded mode at runtime.

## Features

| Feature | Description |
|---------|-------------|
| `threads` | Rayon-based parallel group decode and render |
| `cms` | moxcms CMS backend for ICC profile transforms |
| `jpeg` | JPEG reconstruction from JXL containers |
| `all-simd` | All SIMD backends (SSE4.2, AVX2, AVX-512, NEON) |
| `sse42` / `avx` / `avx512` / `neon` | Individual SIMD targets |
| `allow-unsafe` | Enable `unsafe` fast paths in the main crate (safe fallbacks used otherwise) |

## Conformance

Against [libjxl/conformance](https://github.com/libjxl/conformance) test suite (Level 5): **17/23 passing** (74%).

```bash
cargo test --features cms conformance -- --ignored --nocapture
```

| Status | Tests |
|--------|-------|
| Pass | alpha_nonpremultiplied, alpha_triangles, bench_oriented_brg_5, bicycles, blendmodes_5, cafe_5, delta_palette, grayscale_5, grayscale_jpeg_5, grayscale_public_university, lz77_flower, noise_5, opsin_inverse_5, patches_5, patches_lossless, sunset_logo, upsampling_5 |
| Skip | animation_icos4d_5, animation_newtons_cradle, animation_spline_5 (animation not yet supported) |
| Fail | bike_5, progressive_5 (out-of-gamut/HDR values), spot (6-channel output not yet supported) |

Against codec-corpus (184 JXL files with djxl reference output): **184/184 passing** (100%).

## Upstream Contributions

Bug fixes reported to [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs):

| # | Type | Status | Description |
|---|------|--------|-------------|
| [#607](https://github.com/libjxl/jxl-rs/pull/607) | PR | Merged | ICC tag parsing overflow fix |
| [#637](https://github.com/libjxl/jxl-rs/pull/637) | PR | Merged | Extra channel bit_depth fix (from our [#604](https://github.com/libjxl/jxl-rs/pull/604)) |
| [#632](https://github.com/libjxl/jxl-rs/pull/632) | PR | Merged | RCT wrapping arithmetic (from our [#603](https://github.com/libjxl/jxl-rs/pull/603)) |
| [#609](https://github.com/libjxl/jxl-rs/issues/609) | Issue | Fixed | Progressive AC last_pass validation bug |
| [#610](https://github.com/libjxl/jxl-rs/issues/610) | Issue | Fixed | Noise seeding wrong when upsampling > 1 |
| [#660](https://github.com/libjxl/jxl-rs/pull/660) | PR | Open | Conformance test infrastructure |
| [#602](https://github.com/libjxl/jxl-rs/pull/602) | PR | Closed | Integer overflow handling (ICC part merged as #607; rest declined) |

## License

BSD-3-Clause. See [LICENSE](LICENSE).
