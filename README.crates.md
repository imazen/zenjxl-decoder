<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenjxl-decoder

A pure-Rust **JPEG XL decoder** (ISO/IEC 18181), built for decoding untrusted bytes in a server. `#![forbid(unsafe_code)]` by default, with multi-architecture SIMD (SSE4.2/AVX2/AVX-512/NEON/WASM128) dispatched at runtime through the safe [`archmage`](https://crates.io/crates/archmage) layer.

`zenjxl-decoder` is a fork of the upstream [`libjxl/jxl-rs`](https://github.com/libjxl/jxl-rs) reference decoder, which in turn tracks the C++ [`libjxl`](https://github.com/libjxl/libjxl) implementation. Upstream remains the source of truth for codec behaviour; this fork adds what a production service needs on top — enforced resource limits, cooperative cancellation, parallel decode, fallible allocation, and a handful of bug fixes that have been contributed back upstream. Maintained by [Lilith River](https://github.com/lilith) at [Imazen](https://github.com/imazen).

The Rust lib name is `zenjxl_decoder`, so you `use zenjxl_decoder::...` in code. Found a conformant JXL file that decodes incorrectly (or not at all)? [Open an issue](https://github.com/imazen/zenjxl-decoder/issues/new).

## Quick start

```toml
[dependencies]
zenjxl-decoder = "0.3.10"
```

```rust
use zenjxl_decoder::decode;

let data = std::fs::read("image.jxl")?;
let image = decode(&data)?; // one-shot: bytes -> JxlImage

let width: usize = image.width;
let height: usize = image.height;
let pixels: &[u8] = &image.data; // interleaved, row-major, tightly packed
let channels: usize = image.channels; // 4 = RGBA, 2 = GrayAlpha

assert_eq!(image.data.len(), width * height * channels);
```

`decode` is the one-liner. It returns a [`JxlImage`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/struct.JxlImage.html) whose public fields hold everything you need: `width`, `height`, `channels`, the pixel `data: Vec<u8>`, `is_grayscale`, the input/output color profiles, plus `exif`, `xmp`, and an HDR `gain_map` when present.

Need only the dimensions? [`read_header(&data)`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/fn.read_header.html) parses the header (and ICC profile) without decoding pixels and returns a `JxlImageInfo`; the size is `header.info.size`.

**Output format.** `decode` / `decode_with` always emit **8-bit interleaved** pixels with an alpha channel: `RGBA8` (4 channels) for color images, or `GrayAlpha8` (2 channels) when `image.is_grayscale` is `true`. Images with no source alpha get an opaque `255` alpha. Alpha is **straight (not premultiplied)** by default — set `premultiply_output` on the options for premultiplied output. Pixels are in the output color profile (`image.output_profile`); for XYB / wide-gamut sources this is sRGB with the sRGB transfer function applied, so HDR / float values are range-mapped into u8, not returned as floats. Only the first frame is decoded; for animation use the streaming [`JxlDecoder`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/struct.JxlDecoder.html) API. 8-bit output is **blue-noise dithered** like libjxl's `djxl` (a fixed, position-indexed pattern below half a code, so lossless 8-bit content is untouched and streamed and one-shot decodes agree); `JxlDecoderOptions::with_dither_u8(false)` gives plain round-to-nearest.

## Decoding untrusted input

`decode()` already enforces the default limits (≈256 MP, 4 GB memory on 64-bit / 2 GB on 32-bit, plus capped channel / tree / spline / patch counts). Allocations derived from header fields are checked *before* they happen, so a malicious file is rejected with an `Err`, not OOM'd or panicked.

For a web frontend, tighten further with the `restrictive()` preset and pass a cancellation handle so a slow decode can be aborted. `JxlDecoderOptions` is `#[non_exhaustive]`, so build it from `default()` and chain the `with_*` setters (a struct literal won't compile from another crate):

```rust
use std::sync::Arc;
use zenjxl_decoder::{decode_with, api::{JxlDecoderOptions, JxlDecoderLimits}};
use almost_enough::Stopper; // separate crate: `cargo add almost-enough`

let stop = Arc::new(Stopper::new()); // flip from another thread / a timeout to cancel

let options = JxlDecoderOptions::default()
    .with_limits(JxlDecoderLimits::restrictive()) // 120 MP, 1 GB, tight counts
    .with_stop(stop.clone())                       // cooperative cancellation
    .with_reject_progressive(true);                // optionally refuse progressive content

// On another thread / timer:  stop.cancel();
let image = decode_with(&data, options)?;
```

### Resource limits

| Limit | Default | Restrictive |
|-------|---------|-------------|
| `max_pixels` | 2^28 (~256M) | 120M |
| `max_extra_channels` | 256 | 16 |
| `max_icc_size` | 256 MB | 16 MB |
| `max_tree_size` | 4M nodes | 1M nodes |
| `max_patches` | (derived from image size) | 64K |
| `max_spline_points` | 1M | 64K |
| `max_reference_frames` | 4 | 2 |
| `max_memory_bytes` | 4 GB (2 GB on 32-bit) | 1 GB |

`JxlDecoderLimits` is also `#[non_exhaustive]`: start from `default()`, `restrictive()`, or `unlimited()` (every field `None` — **trusted input only**) and override individual fields with the `with_*` setters. Exceeding any limit returns `Error::LimitExceeded { resource, actual, limit }` — never a panic. The `resource` string is one of `pixels`, `memory_bytes`, `icc_size`, `icc_amplification`, `extra_channels`, `reference_frames`.

### Cancellation

The `stop` field is an `Arc<dyn `[`enough::Stop`](https://docs.rs/enough)`>`. It defaults to `Arc::new(enough::Unstoppable)` — the no-op handle, checked periodically at no cost — which never cancels. The simplest way to *actually* cancel or time out is the ready-made [`Stopper`](https://crates.io/crates/almost-enough) shown above (`cargo add almost-enough`; both crates are at `0.4`). A cancelled decode surfaces `Error::Cancelled`, and the checks run inside the parallel closures too, so a cancel is noticed mid-batch.

Prefer not to add a dependency? `Stop` and `StopReason` are re-exported from `zenjxl_decoder::api`, so you can implement the trait over an `AtomicBool` with no direct `enough` / `almost-enough` dependency:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use zenjxl_decoder::api::{Stop, StopReason, JxlDecoderOptions};

struct CancelFlag(AtomicBool);

impl Stop for CancelFlag {
    fn check(&self) -> Result<(), StopReason> {
        if self.0.load(Ordering::Relaxed) {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

let flag = Arc::new(CancelFlag(AtomicBool::new(false)));
// ... set `flag.0.store(true, Ordering::Relaxed)` from another thread to cancel.
let options = JxlDecoderOptions::default().with_stop(flag);
```

### Errors

`decode` / `decode_with` return `zenjxl_decoder::api::Result<JxlImage>` = `Result<JxlImage, At<Error>>`. The error is a [`whereat::At<Error>`](https://docs.rs/whereat) — a `file:line` source-location wrapper around the crate's [`Error`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/enum.Error.html). Call `.error()` on it to borrow the inner `Error`. `Error` is `#[non_exhaustive]` (~130 variants), so a server usually wants the coarse [`ErrorClass`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/enum.ErrorClass.html) from `Error::kind()` rather than matching every variant:

```rust
use zenjxl_decoder::{decode_with, api::{ErrorClass, JxlDecoderLimits, JxlDecoderOptions}};

let options = JxlDecoderOptions::default().with_limits(JxlDecoderLimits::restrictive());
match decode_with(&data, options) {
    Ok(image) => { /* image.width, image.height, image.data … */ }
    Err(e) => match e.error().kind() {           // `e` is At<Error>; `.error()` borrows the Error
        ErrorClass::LimitExceeded => {}           // oversized / hostile input -> HTTP 413
        ErrorClass::Cancelled => {}               // a Stop token cancelled or timed out -> 499/408
        ErrorClass::InvalidBitstream => {}        // malformed / truncated input -> 400
        ErrorClass::Unsupported => {}             // valid but unsupported feature -> 415
        _ => {}                                   // OutOfMemory / Io / OutputConfiguration / Internal
    },
}
```

Each `ErrorClass` variant documents its typical HTTP status. To distinguish individual causes, match on `e.error()` directly — every variant of `Error` is `#[non_exhaustive]`, so keep a wildcard arm.

### Parallel decoding

The `threads` feature (on by default in the bundled `zenjxl-decoder-cli`) parallelizes group decode and render across rayon's global pool; size it with `RAYON_NUM_THREADS`.

```toml
[dependencies]
zenjxl-decoder = { version = "0.3.10", features = ["threads"] }
```

To force single-threaded mode at runtime, use `JxlDecoderOptions::default().with_parallel(false)`.

## Other entry points

- [`decode_with(bytes, options)`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/fn.decode_with.html) — full control over limits, cancellation, parallel mode, CMS, and premultiplied alpha.
- [`read_header(bytes)`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/fn.read_header.html) / `read_header_with(bytes, limits)` — metadata only, no pixel decode.
- `reconstruct_jpeg(bytes)` / `reconstruct_jpeg_with(..)` — losslessly reconstruct the original JPEG from a JXL that was transcoded from one (requires the `jpeg` feature). Returns `Ok(None)` when the file carries no reconstruction data.
- The streaming [`JxlDecoder`](https://docs.rs/zenjxl-decoder/latest/zenjxl_decoder/api/struct.JxlDecoder.html) typestate API decodes frame-by-frame for animation, and seeks: `scanned_frames()` lists every visible frame with a `seek_target`, `start_new_frame(target)` jumps to it (feed the file from `target.decode_start_file_offset`), and `JxlDecoderOptions::scan_frames_only` builds the seek table from frame headers alone.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `all-simd` | ✅ | All SIMD backends, dispatched at runtime via [`archmage`](https://crates.io/crates/archmage) |
| `sse42` / `avx` / `avx512` / `neon` / `wasm128` | — | Individual SIMD targets |
| `threads` | — | Rayon-based parallel group decode and render |
| `cms` | — | [`moxcms`](https://crates.io/crates/moxcms) ICC-profile color transforms (required for CMYK input) |
| `jpeg` | — | Lossless JPEG reconstruction from JXL containers (pulls in `brotli`) |
| `allow-unsafe` | — | Opt into a small set of `unsafe` fast paths guarded by safe fallbacks (the default build is `#![forbid(unsafe_code)]`) |

## What's different from upstream

This fork builds on `libjxl/jxl-rs` and adds the pieces a production service needs. Every change is BSD-3-Clause under the Google CLA and intended for upstream contribution.

### New features

| Feature | Description |
|---------|-------------|
| Resource limits | `JxlDecoderLimits` API — cap pixels, memory, ICC size, tree size, channels, patches, splines, reference frames |
| Memory tracking | `max_memory_bytes` budget with atomic, lock-free accounting |
| Fallible allocation | Significant allocations return an error instead of aborting |
| Cooperative cancellation | `enough::Stop` — cancel or time out a decode from any thread |
| Error classification | `Error::kind() -> ErrorClass` for HTTP-style bucketing without matching ~130 variants |
| Parallel decoding | Rayon-based parallel group decode and render (opt-in via `threads`) |
| CMS-based CMYK→RGB | ICC profile-based CMYK conversion via the optional `moxcms` backend |

### Bug fixes (reported and contributed back upstream)

| Area | Fix |
|------|-----|
| sRGB transfer function | Apply the sRGB TF by default for XYB-encoded images (was outputting linear) |
| RCT overflow | Wrapping arithmetic to avoid a panic on edge-case pixel values |
| Extra-channel format slots | Allocate format slots for all extra channels, not just the first |
| Progressive AC validation | Fix inverted `last_pass` validation (must be strictly increasing) |
| Extra-channel bit depth | Use each extra channel's own `bit_depth` for modular-to-f32 conversion |
| Noise seeding (upsampling > 1) | Separate RNG seeds per subregion for upsampled frames |
| CMYK blending order | Blend in CMYK space, then CMS-convert to RGB |

## Conformance

Against the [libjxl/conformance](https://github.com/libjxl/conformance) suite (Level 5): **17/23 passing** (as of December 2025).

```bash
cargo test --features cms conformance -- --ignored --nocapture
```

| Status | Tests |
|--------|-------|
| Pass | alpha_nonpremultiplied, alpha_triangles, bench_oriented_brg_5, bicycles, blendmodes_5, cafe_5, delta_palette, grayscale_5, grayscale_jpeg_5, grayscale_public_university, lz77_flower, noise_5, opsin_inverse_5, patches_5, patches_lossless, sunset_logo, upsampling_5 |
| Skip | animation_icos4d_5, animation_newtons_cradle, animation_spline_5 (animation *rendering* not yet supported; animation metadata is parsed — see `JxlBasicInfo::animation`, reachable from `read_header`'s `JxlImageInfo` or a decoded `JxlImage`) |
| Fail | bike_5, progressive_5 (out-of-gamut / HDR values), spot (6-channel output not yet supported) |

Against [codec-corpus](https://github.com/imazen/codec-corpus) (184 JXL files with djxl reference output): **184/184 passing** (as of December 2025).

## Sister project

[jxl-encoder](https://github.com/imazen/jxl-encoder) — a multithreaded JPEG XL encoder in Rust, also by Imazen.

## License

BSD-3-Clause, matching upstream `jxl-rs`. See [LICENSE](https://github.com/imazen/zenjxl-decoder/blob/main/LICENSE).

This is a fork of [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs) (BSD-3-Clause). We'd rather contribute back than maintain a parallel codebase, and are happy to release these improvements under the original BSD-3-Clause license if upstream takes over maintaining them. Open an issue or reach out.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · **zenjxl-decoder** · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
