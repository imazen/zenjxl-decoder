# zenjxl-decoder ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjxl-decoder/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenjxl-decoder?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenjxl-decoder?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenjxl-decoder) ![docs.rs](https://img.shields.io/docsrs/zenjxl-decoder?style=flat-square) ![License](https://img.shields.io/crates/l/zenjxl-decoder?style=flat-square)

A pure-Rust **JPEG XL decoder** (ISO/IEC 18181), built for decoding untrusted bytes in a server.

`zenjxl-decoder` is a fork of the upstream [`libjxl/jxl-rs`](https://github.com/libjxl/jxl-rs) reference decoder. Upstream remains the source of truth for codec behaviour; this fork adds the things a production service needs on top: enforced resource limits, cooperative cancellation, parallel decode, and `#![forbid(unsafe_code)]` by default.

## Quick start

```rust
// Decode the first frame to interleaved 8-bit pixels.
let bytes = std::fs::read("photo.jxl")?;
let image = zenjxl_decoder::decode(&bytes)?;

// `data` is row-major, tightly packed, RGBA (4 ch) or GrayAlpha (2 ch).
assert_eq!(image.data.len(), image.width * image.height * image.channels);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`decode` returns a [`JxlImage`] with `width`, `height`, `channels`, the pixel `data: Vec<u8>`, plus color profiles, EXIF, and an HDR `gain_map` when present. Alpha is always emitted (opaque where the source has none).

Need just the dimensions? `read_header(&bytes)` parses the header without decoding pixels.

## Decoding untrusted input (the server path)

**The default already enforces limits** — `decode()` uses [`JxlDecoderLimits::default()`] (≈256 MP, 4 GB memory on 64-bit / 2 GB on 32-bit, capped channel/tree/spline/patch counts). Allocation derived from header fields is checked *before* it happens, so a malicious file is rejected, not OOM'd.

For a web frontend, tighten further with `restrictive()`, and pass a cancellation handle so a slow decode can be aborted:

`JxlDecoderOptions` is `#[non_exhaustive]`, so you configure it by mutating fields on a `default()` value (not a struct literal):

```rust
use std::sync::Arc;
use zenjxl_decoder::{decode_with, api::{JxlDecoderOptions, JxlDecoderLimits}};
use almost_enough::Stopper;

let stop = Arc::new(Stopper::new());     // flip this from another thread / a timeout to cancel

let mut options = JxlDecoderOptions::default();
options.limits = JxlDecoderLimits::restrictive(); // 120 MP, 1 GB, tight counts — for untrusted web content
options.stop = stop.clone();                       // cooperative cancellation; default is `Unstoppable` (no-op)
options.reject_progressive = true;                 // optionally refuse progressive content

// On another thread / timer:  stop.cancel();
let image = decode_with(&bytes, options)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Limit presets:

| preset | max pixels | max memory | use for |
|--------|-----------|-----------|---------|
| `JxlDecoderLimits::default()` | ~256 MP | 4 GB (2 GB on 32-bit) | general decoding |
| `JxlDecoderLimits::restrictive()` | 120 MP | 1 GB | untrusted web content |
| `JxlDecoderLimits::unlimited()` | none | none | **trusted input only** |

Every field is a `pub Option<_>` you can set individually (`max_pixels`, `max_memory_bytes`, `max_extra_channels`, `max_tree_size`, `max_patches`, `max_spline_points`, `max_reference_frames`, `max_icc_size`). `None` means "no limit" for that field. When a limit trips, decode returns an `Err` you can inspect — it is never a panic.

Cancellation uses the [`enough`](https://crates.io/crates/enough) `Stop` trait. The default `stop` is `Arc::new(enough::Unstoppable)`, which the decoder checks periodically at no cost. To actually cancel, hand it an [`almost_enough::Stopper`](https://crates.io/crates/almost-enough) (or any `Stop`) and flip it from another thread or a timeout.

## Other entry points

- `decode_with(bytes, options)` — full control (limits, cancellation, parallel, CMS, premultiplied alpha).
- `read_header(bytes)` / `read_header_with(bytes, limits)` — metadata only, no pixel decode.
- `reconstruct_jpeg(bytes)` / `reconstruct_jpeg_with(..)` — losslessly reconstruct the original JPEG from a JXL that was transcoded from one (requires the `jpeg` feature).
- The streaming [`JxlDecoder`] type (in `api`) decodes frame-by-frame for animation.

## Features

| feature | default | effect |
|---------|---------|--------|
| `all-simd` | ✅ | runtime SIMD dispatch (SSE4.2/AVX2/AVX-512/NEON/WASM128) via [`archmage`](https://crates.io/crates/archmage); per-ISA: `sse42`, `avx`, `avx512`, `neon`, `wasm128` |
| `threads` | — | parallel group decode/render on the rayon global pool (`RAYON_NUM_THREADS` to size it) |
| `cms` | — | ICC-aware color conversion via [`moxcms`](https://crates.io/crates/moxcms) |
| `jpeg` | — | `reconstruct_jpeg*` (pulls in `brotli`) |

## Safety

The crate is `#![forbid(unsafe_code)]`; all SIMD is expressed through the safe [`archmage`](https://crates.io/crates/archmage) dispatch layer. There is no `unsafe` in the decode path.

## License

BSD-3-Clause, matching upstream `jxl-rs`. See [LICENSE](LICENSE).
