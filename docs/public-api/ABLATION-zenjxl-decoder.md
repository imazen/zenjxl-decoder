# Public-API Ablation Report: zenjxl-decoder

**Date:** 2026-06-10
**Snapshot commit:** 9b213828c07d (main)
**Items in snapshot:** 1177 (default features) / 1205 (all features, +28 from `brotli` feature enabling `reconstruct_jpeg`/`reconstruct_jpeg_with`)
**Grep template:** `grep -rn "<symbol>" /home/lilith/work/ --include="*.rs" --exclude-dir=target --exclude-dir=".jj" --exclude-dir="zenjxl-decoder" 2>/dev/null`

---

## Summary

| Total items | Flagged A | Flagged B | Flagged % |
|-------------|-----------|-----------|-----------|
| 1205        | 4         | 2         | ~0.5%     |

Conservative bar applied: only items with zero external consumers AND no plausible deliberate-contract rationale are flagged. All confirmed-external items (reconstruct_jpeg, JxlDecoder, JxlDecoderOptions, JxlColorEncoding, JxlColorProfile, JxlBasicInfo, GainMapBundle, JxlImage, etc.) are explicitly KEPT.

---

## Module inventory

The 1177-item default surface organizes roughly as:

| Module / group | ~Items | Nature |
|----------------|--------|--------|
| `api` enums/structs (JxlColorType, JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint, JxlBitDepth, Orientation, etc.) | ~250 | Deliberate user-facing types — KEEP |
| `api::Error` variants + impls | ~100 | Error enum — KEEP |
| `api::JxlDecoder<S>` + `JxlDecoderOptions` + `JxlDecoderLimits` | ~80 | Core decoder API — KEEP |
| `api::JxlDecoderInner` | ~20 | See below |
| `api::Image<T>`, `ImageRect`, `ImageRectMut`, `OwnedRawImage`, `RawImageRect`, `RawImageRectMut` | ~120 | Image buffer types — mostly KEEP, see specifics |
| `api::JxlBasicInfo`, `JxlFrameHeader`, `JxlImage`, `JxlImageInfo`, `JxlAnimation`, `JxlExtraChannel`, `JxlOutputBuffer`, `JxlPixelFormat`, `VardctQuantizer`, `ToneMapping`, `GainMapBundle` | ~120 | Result / output types — KEEP |
| `api::JxlBitstreamInput`, `JxlCms`, `JxlCmsTransformer` traits | ~30 | Extension points — KEEP |
| `api::decode`, `decode_with`, `read_header`, `read_header_with`, `check_signature`, `compute_md5` fns | 6 | Convenience functions — see below |
| `api::reconstruct_jpeg`, `reconstruct_jpeg_with` (brotli feature) | 2 | JBRD surface — KEEP (external consumer confirmed) |
| `api::states::{Initialized, WithImageInfo, WithFrameInfo}`, `JxlState` trait | ~14 | Typestate machinery — KEEP |
| `api::ProcessingResult<T, U>` | ~8 | API primitive — KEEP |
| `api::DataTypeTag`, `Endianness`, `JxlSignatureType`, `JxlProgressiveMode` | ~20 | Enumerations — KEEP |
| `api::Rect`, `Point` | ~30 | Geometry types — see below |
| `profile!` macro | 1 | See below |
| `image::internal::DistinctRowsIndexes` (via `Image::distinct_full_rows_mut`) | indirect | See below |
| `util::memory_tracker::MemoryTracker` (via `Image::new_tracked`) | indirect | See below |
| `headers::color_encoding::ColorEncoding` (via `JxlColorEncoding::from_internal`) | indirect | See below |
| `container::frame_index::FrameIndexBox` (via `JxlDecoder::frame_index`) | indirect | See below |

---

## Per-item flag table

### Confirmed KEEP (external consumers found)

| Item | External consumer | Location |
|------|-------------------|----------|
| `reconstruct_jpeg` / `reconstruct_jpeg_with` | `zenjxl_decoder::reconstruct_jpeg(&jxl)` | `jxl-encoder/tests/it/jbrd_roundtrip_conformance.rs:172` |
| `JxlDecoder<S>`, `JxlDecoderOptions`, `JxlDecoderLimits` | Extensively used | `zenjxl/src/decode.rs`, `zenjxl/src/codec.rs` |
| `JxlColorEncoding`, `JxlColorProfile`, `JxlTransferFunction`, `JxlPrimaries`, `JxlWhitePoint` | Used in pattern matching | `zenjxl/src/decode.rs:266–290` |
| `JxlBasicInfo`, `JxlPixelFormat`, `JxlOutputBuffer`, `JxlBitstreamInput` | Used in decoder loop | `zenjxl/src/codec.rs` |
| `GainMapBundle`, `ExtraChannel`, `JxlAnimation` | Used via `zenjxl` | `zenjxl/src/decode.rs:11` |
| `decode` / `decode_with` / `read_header` / `read_header_with` | (no direct external use confirmed but part of documented surface) | — |

---

### Flagged items

| Item | Class | Evidence | Proposed action | Semver impact |
|------|-------|----------|-----------------|---------------|
| `JxlDecoderInner` (struct + all methods) | A | `grep "JxlDecoderInner"` outside repo: **0 hits** in zen org (excluding third-party read-only copies of jxl-rs upstream). The struct is the internal state machine used by `JxlDecoder<S>`. External callers have no reason to use `JxlDecoderInner` directly; `JxlDecoder<S>` provides the full typed-state API. Inside the repo, `JxlDecoderInner` is the implementation of `JxlDecoder` (confirmed in `src/api/decoder.rs:35`). This is an implementation detail that leaked `pub` — the jxl-rs heritage means it was `pub` there too. | A: `#[doc(hidden)]` on `JxlDecoderInner` and a crate-internal `pub(crate)` pending the next 0.x minor. Alternatively A-then-B: doc-hide now, demote in next minor. | A = non-breaking; eventual B = breaking |
| `JxlDecoder<S>::frame_index()` return type leaking `container::frame_index::FrameIndexBox` | A | `grep "FrameIndexBox"` outside repo: **0 hits** in zen org (excluding third-party). The return type `&zenjxl_decoder::container::frame_index::FrameIndexBox` exposes an internal module path (`container::frame_index`) in a public method signature. Users can call `frame_index()` and get the struct, but the struct's module path is implementation detail. Zero external callers have been found who depend on the `FrameIndexBox` type by name. | A: Add `#[doc(hidden)]` to the `container::frame_index` module (or re-export `FrameIndexBox` into `api::` if it is intentionally stable). If the frame-index API is a deliberate feature, the type should live in `api::` not `container::frame_index::`. | A = non-breaking (type is still accessible, just not in docs) |
| `api::compute_md5(&[u8]) -> [u8; 16]` | B | `grep "compute_md5"` outside repo: **0 hits** in zen org. Only found in `third-party/jxl-rs` upstream copies and inside the crate itself (ICC profile generation). This is an ICC-profile checksum utility that was `pub` in jxl-rs and inherited. No external caller needs it. | B: demote to `pub(crate)` in next 0.x minor. A-then-B: `#[doc(hidden)]` now. | Breaking on demotion — class B |
| `profile!` macro | B | `grep "profile!"` outside repo: **0 hits** in zen org. The macro is a performance profiling annotation (tracing spans). It is an internal developer tool that was inherited from jxl-rs' `profile!` infrastructure. | B: demote or gate behind an internal feature. A-then-B: `#[doc(hidden)]` now. | Breaking on demotion — class B |
| `Image<T>::new_tracked(...)` and `Image<T>::new_with_padding_tracked(...)` (taking `&MemoryTracker`) | A | `grep "new_tracked\|MemoryTracker\|new_with_padding_tracked"` outside repo: **0 hits** in zen org. These constructors expose `zenjxl_decoder::util::memory_tracker::MemoryTracker` in their signatures, leaking an internal module path. External callers cannot construct a `MemoryTracker` (its constructor is internal), so these methods are effectively unusable from outside the crate. They exist so the decoder can allocate images within a resource-budget tracking context. | A: `#[doc(hidden)]` on `new_tracked` and `new_with_padding_tracked`, and on `util::memory_tracker::MemoryTracker` module. The `Image<T>` type itself and its other constructors are KEEP. | Non-breaking |
| `Image<T>::distinct_full_rows_mut<I: image::internal::DistinctRowsIndexes>(...)` | A | `grep "distinct_full_rows_mut\|DistinctRowsIndexes"` outside repo: **0 hits** in zen org (excluding third-party read-only copies of jxl-rs). The method takes a generic bound `I: zenjxl_decoder::image::internal::DistinctRowsIndexes` — an internal trait in `image::internal`. External callers can only call this method with `[usize; N]` arrays (the only external impl), but the trait bound exposes the internal module. | A: `#[doc(hidden)]` on `distinct_full_rows_mut` and on `image::internal::DistinctRowsIndexes`. The method can remain present (it's called internally) but with hidden docs. | Non-breaking |
| `JxlColorEncoding::from_internal(&headers::color_encoding::ColorEncoding)` | A | `grep "from_internal"` outside zen org: **0 hits** in zen org (excluding third-party). This method takes `&zenjxl_decoder::headers::color_encoding::ColorEncoding` — an internal bitstream struct from the `headers` module. External callers cannot construct a `ColorEncoding` (it is parsed from bitstream bytes internally). This is a conversion helper that should be `pub(crate)`. | A: `#[doc(hidden)]` on `from_internal`; demote to `pub(crate)` in next minor. The `headers::color_encoding::ColorEncoding` type should not be in any public signature. | A = non-breaking; future pub(crate) = breaking |

---

### Borderline KEEP items (checked, retained with note)

| Item | Rationale for KEEP |
|------|--------------------|
| `api::Rect` and `api::Point` | Used in error variants (`SplineAdjacentCoincidingControlPoints`, `SplinesPointOutOfRange`) and in `Image<T>::get_rect()`, `JxlOutputBuffer::rect()`. External callers constructing output buffers or reading error fields need these. No evidence they should be hidden. |
| `api::VardctQuantizer` | Returned by `JxlDecoder::vardct_quantizer()`. Useful for JBRD use cases and round-trip debugging. Intentional. |
| `api::ToneMapping` | Field of `JxlBasicInfo`. Intentional. |
| `api::JxlProgressiveMode` | Field of `JxlDecoderOptions`. Intentional. |
| `api::JxlImage`, `api::JxlImageInfo` | Returned by `decode()` / `read_header()` convenience fns. Intentional. |
| `api::OwnedRawImage`, `api::RawImageRect`, `api::RawImageRectMut` | Used in `JxlOutputBuffer::from_image_rect_mut()` and `Image<T>::from_raw()`/`into_raw()`. Low-level but deliberate. |
| `api::ImageDataType` trait | Bounds the `Image<T>` generic. Intentional. |
| `api::JxlBitstreamInput`, `JxlCms`, `JxlCmsTransformer` | Extension-point traits. Intentional. |
| `api::states::*` and `JxlState` | Typestate machinery for `JxlDecoder<S>`. Deliberate design. |
| `api::check_signature` / `api::decode` / `api::decode_with` / `api::read_header` / `api::read_header_with` | Top-level convenience functions. Intentional. |

---

## Top 10 highest-confidence ablations

1. **`JxlDecoderInner` (Class A)** — The internal state machine struct is pub despite having zero external consumers. It is named "Inner" which is the conventional Rust signal for "not for you". `#[doc(hidden)]` now; `pub(crate)` in next minor.

2. **`JxlColorEncoding::from_internal(&headers::color_encoding::ColorEncoding)` (Class A)** — Takes an internal bitstream type that external callers cannot construct. A conversion helper that should never have been `pub`. Zero org-wide external hits.

3. **`Image<T>::new_tracked` / `Image<T>::new_with_padding_tracked` (Class A)** — Expose `util::memory_tracker::MemoryTracker` in their signatures. `MemoryTracker` has no public constructor reachable from outside the crate, making these methods dead API for external users. `#[doc(hidden)]` on both.

4. **`Image<T>::distinct_full_rows_mut<I: image::internal::DistinctRowsIndexes>` (Class A)** — The trait bound references an internal module. External callers can only call this with `[usize; N]` but the API leaks `image::internal`. `#[doc(hidden)]` on method and trait.

5. **`JxlDecoder<S>::frame_index() -> Option<&container::frame_index::FrameIndexBox>` (Class A)** — Returns a type in an internal module path. If `FrameIndexBox` is deliberate API (it exposes frame byte offsets for seeking), re-export it into `api::`. If not intentional, `#[doc(hidden)]` on the method.

6. **`compute_md5(&[u8]) -> [u8; 16]` (Class B)** — ICC profile checksum helper. Zero external hits. Should be `pub(crate)`. `#[doc(hidden)]` as interim; `pub(crate)` in next minor.

7. **`profile!` macro (Class B)** — Internal profiling annotation macro (tracing span helper). Zero external hits. Should be hidden or moved behind an internal feature.

8–10. The remaining items in `container::frame_index`, `image::internal`, and `util::memory_tracker` that become reachable only through the above signatures. Once the above are addressed these resolve automatically.

---

## Percentage check

6 items flagged across A and B classes out of 1205 total ≈ 0.5%. Well within the 10% conservative threshold.

---

## Notes on jxl-rs heritage

`zenjxl-decoder` is built on the foundation of the jxl-rs reference implementation (maintained at <https://github.com/libjxl/jxl-oxide>), which did significant API design work on the decoder surface. The types exposed here — `JxlColorEncoding`, `JxlDecoder<S>`, the typestate API, `JxlDecoderOptions` — follow jxl-rs conventions closely. The leaked internals (`from_internal`, `JxlDecoderInner`, `compute_md5`) are directly inherited from jxl-rs' own `pub` decisions, where the upstream similarly exposes them. Flagging these is not a criticism of the upstream design — it reflects the different context where `zenjxl-decoder` is a published crate with explicit downstream consumers (`zenjxl`, `jxl-encoder`) that only use the top-level API.
