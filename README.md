# JPEG XL in Rust

This is a work-in-progress reimplementation of a JPEG XL decoder in Rust, aiming to be conforming, safe, and fast.

We strive to decode all conformant JPEG XL bitstreams correctly. If you find an image that can be decoded with the reference
implementation `djxl` (from [`libjxl`](https://github.com/libjxl/libjxl)) but is decoded incorrectly or not at all by `jxl-rs`,
please report it by [opening an issue](https://github.com/libjxl/jxl-rs/issues/new).

For more information, including contributing instructions, refer to the [libjxl repository](https://github.com/libjxl/libjxl).

---

## Development Branch Changes

This branch (`develop`) contains improvements being prepared for upstream contribution to [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs).

### Goals

1. **Zero Panic** - Eliminate all potential panics from untrusted input; all errors should be recoverable
2. **Fallible Allocation** - Support environments where allocation can fail gracefully
3. **JXL Parity** - Pixel-perfect output matching the reference `djxl` implementation
4. **Performance Parity** - Match or exceed libjxl decode performance

### License

All changes in this branch are released under the same BSD-style license as jxl-rs and are contributed under the Google CLA. These changes are intended to be upstreamed to the official libjxl/jxl-rs repository.

### Bug Fixes

| Bug | File(s) | Description |
|-----|---------|-------------|
| sRGB Transfer Function | `frame/render.rs` | Apply sRGB transfer function by default for XYB-encoded images (was outputting linear) |
| RCT Overflow Panic | `frame/modular/transforms/rct.rs` | Use wrapping arithmetic to prevent panic on edge-case pixel values |
| Extra Channel Format Slots | `frame/render.rs` | Allocate format slots for all extra channels, not just first |
| Linear Gamma Detection | `tests/parity.rs` | Detect gAMA=100000 in reference PNGs to compare linear values correctly |
| Grayscale ICC Detection | `tests/parity.rs`, `api/color.rs` | Detect grayscale from pixel format, not output color profile |
| Progressive AC Validation | `headers/frame_header.rs` | Fix inverted `last_pass` validation (must be strictly increasing) |
| Extra Channel Bit Depth | `frame/modular/decode/channel.rs` | Use extra channel's own `bit_depth` for modular-to-f32 conversion |
| Noise Seeding (upsampling > 1) | `frame/decode.rs` | Iterate upsampling subdivisions with separate RNG seeds per subregion |
| CMYK Blending Order | `frame/render.rs` | Defer CMS conversion for CMYK images with blending (blend in CMYK space, then convert to RGB) |

### New Features

| Feature | File(s) | Description |
|---------|---------|-------------|
| CMS-based CMYK→RGB | `render/stages/cms_cmyk.rs`, `api/moxcms_wrapper.rs` | Convert CMYK to RGB using embedded ICC profiles via optional `moxcms` integration |
| BlackChannelStage | `render/stages/black.rs` | Simple CMYK K-channel application for images without ICC profiles |
| UnpremultiplyAlphaStage | `render/stages/unpremultiply_alpha.rs` | Pipeline stage for alpha unpremultiplication |

### Security Features: Resource Limits & Fallible Allocation

This branch adds comprehensive protection against "JXL bombs" - maliciously crafted files designed to exhaust memory or CPU. These features support the **Zero Panic** and **Fallible Allocation** goals.

#### JxlDecoderLimits API

The `JxlDecoderLimits` struct provides configurable limits for all decoder resource consumption:

```rust
use jxl::api::{JxlDecoderLimits, JxlDecoderOptions};

// For untrusted web content - restrictive limits
let options = JxlDecoderOptions {
    limits: JxlDecoderLimits::restrictive(),
    ..Default::default()
};

// For trusted local files - default limits
let options = JxlDecoderOptions::default();

// For maximum compatibility (use with caution)
let options = JxlDecoderOptions {
    limits: JxlDecoderLimits::unlimited(),
    ..Default::default()
};
```

| Limit | Default | Restrictive | Description |
|-------|---------|-------------|-------------|
| `max_pixels` | 2^30 (~1B) | 100M | Maximum total pixels (width × height) |
| `max_extra_channels` | 256 | 16 | Maximum extra channels (alpha, depth, etc.) |
| `max_icc_size` | 256 MB | 1 MB | Maximum ICC profile size |
| `max_tree_size` | 4M nodes | 1M nodes | Maximum modular tree nodes |
| `max_patches` | (derived) | 64K | Maximum patch count |
| `max_spline_points` | 1M | 64K | Maximum spline control points |
| `max_reference_frames` | 4 | 2 | Maximum stored reference frames |
| `max_memory_bytes` | None | 1 GB | Total memory budget (see below) |

All limits return `Error::LimitExceeded` with details when exceeded:

```rust
Error::LimitExceeded {
    resource: "pixels",      // Which limit was hit
    actual: 2_000_000_000,   // Attempted value
    limit: 1_073_741_824,    // Configured limit
}
```

#### Memory Tracking (`max_memory_bytes`)

When `max_memory_bytes` is set, the decoder tracks all significant allocations through a thread-safe `MemoryTracker`:

```rust
// Limit decoder to 512 MB total memory
let options = JxlDecoderOptions {
    limits: JxlDecoderLimits {
        max_memory_bytes: Some(512 * 1024 * 1024),
        ..Default::default()
    },
    ..Default::default()
};
```

**How it works:**
1. The `MemoryTracker` uses atomic operations for lock-free tracking
2. Clone is cheap (Arc-based) - all decoder components share the same counter
3. Allocations are checked before allocation, rolling back on failure
4. `MemoryGuard` RAII type ensures memory is released even on errors

**Implementation details (`util/memory_tracker.rs`):**
- `try_allocate(bytes)` - Attempt allocation, returns `Error::LimitExceeded` if over budget
- `release(bytes)` - Return memory to budget (called on deallocation)
- `would_exceed(bytes)` - Check without allocating (for early rejection)
- `MemoryGuard` - RAII guard that releases memory on drop

#### Fallible Allocation Utilities

The `util/vec_helpers.rs` module provides fallible allocation for OOM-resistant decoding:

```rust
use jxl::util::vec_helpers::{NewWithCapacity, TryVecExt, AlignedVec};

// Fallible Vec creation (won't panic on OOM)
let vec: Vec<u8> = Vec::new_with_capacity(1024)?;

// Fallible element-filled vector
let zeros = Vec::<f32>::try_from_elem(0.0, 1024)?;

// SIMD-aligned allocation (64-byte for AVX-512)
let aligned: AlignedVec<f32> = aligned_vec_with_capacity(1024)?;
let zeroed: AlignedVec<f32> = aligned_vec_zeroed(1024)?;
```

**Key types:**
- `NewWithCapacity` trait - Fallible `Vec`/`String` allocation via `try_reserve`
- `TryVecExt` trait - Additional fallible operations (`try_from_elem`, `try_extend_from_slice`)
- `AlignedVec<T>` - 64-byte aligned vector for optimal SIMD performance
- All return `TryReserveError` on allocation failure instead of panicking

#### Cancellation Support

The decoder accepts any `enough::Stop` implementation for cooperative cancellation:

```rust
use almost_enough::Stopper;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

let stop = Arc::new(Stopper::new());
let stop_clone = Arc::clone(&stop);

// Cancel from another thread after timeout
thread::spawn(move || {
    thread::sleep(Duration::from_secs(5));
    stop_clone.cancel();
});

// Decoder checks stop at safe points and returns Error::Cancelled
let options = JxlDecoderOptions {
    stop: stop,
    ..Default::default()
};
```

**API:**
- `cancel()` - Signal cancellation (atomic, thread-safe)
- `is_cancelled()` - Check status
- `stop.check()` - Returns `Error::Cancelled` if stopped

#### Files

| File | Description |
|------|-------------|
| `api/options.rs` | `JxlDecoderLimits`, `stop: Arc<dyn Stop>`, limit presets |
| `util/memory_tracker.rs` | `MemoryTracker`, `MemoryGuard` for budget tracking |
| `util/vec_helpers.rs` | `NewWithCapacity`, `TryVecExt`, `AlignedVec` |
| `error.rs` | `Error::LimitExceeded`, `Error::Cancelled` variants |

### Test Infrastructure

| Test Suite | File | Coverage |
|------------|------|----------|
| Parity Testing | `tests/parity.rs` | Infrastructure for pixel-exact comparison against `djxl` reference output |
| Codec-Corpus Tests | `tests/codec_corpus.rs` | 184 test images from codec-corpus with reference comparison |
| Conformance Tests | `tests/conformance.rs` | Official libjxl/conformance test suite (auto-fetches on first run) |
| Decoder API Tests | `tests/decode_api.rs` | Tests ported from libjxl `decode_test.cc` |
| Feature Tests | `tests/feature_tests.rs` | Comprehensive tests for all JXL encoder options |
| Entropy Tests | `tests/entropy.rs` | ANS and Huffman codec edge cases |
| Streaming Tests | `tests/streaming.rs` | Chunked/progressive decoding scenarios |

### Parity Test Results

Against codec-corpus (184 JXL files with djxl reference output):
- **184/184 passing** (100%)

All test images decode with pixel-exact or near-exact parity, including:
- Single-layer CMYK with ICC profiles
- Multi-layer CMYK with alpha blending (blending in CMYK space, then CMS conversion)
- All modular/VarDCT encoding modes
- Animation and layer compositing

See [PARITY_INVESTIGATION.md](PARITY_INVESTIGATION.md) for detailed bug investigation notes.

### Official Conformance Tests

Against [libjxl/conformance](https://github.com/libjxl/conformance) test suite (Level 5):
- **17/23 passing** (74%)

```bash
# Run conformance tests (auto-fetches test data on first run)
cargo test --features cms conformance -- --ignored --nocapture
```

| Status | Tests |
|--------|-------|
| ✅ Pass | alpha_nonpremultiplied, alpha_triangles, bench_oriented_brg_5, bicycles, blendmodes_5, cafe_5, delta_palette, grayscale_5, grayscale_jpeg_5, grayscale_public_university, lz77_flower, noise_5, opsin_inverse_5, patches_5, patches_lossless, sunset_logo, upsampling_5 |
| ⏭️ Skip | animation_icos4d_5, animation_newtons_cradle, animation_spline_5 (animation not yet supported) |
| ❌ Fail | bike_5, progressive_5 (out-of-gamut/HDR decode), spot (6-channel output) |

#### TODOs to Reach Full Conformance

1. **Animation frame iteration** - Add API to decode individual frames for animation tests
   - Current API composites all frames into final output
   - Need `next_frame()` / `is_last_frame()` methods on decoder

2. **Spot color output** - Support outputting all channels including spot colors
   - `spot` test expects 6 channels (RGB + 2 spot colors)
   - Currently only outputting 4 channels (RGBA)

3. **Investigate remaining decode bugs** - Debug out-of-gamut/HDR value differences
   - `bike_5` has 13 pixels with out-of-gamut (negative) values differing from reference
   - `progressive_5` has many pixels with HDR (> 1.0) values differing from reference
   - May be related to XYB-to-RGB matrix or opsin inverse handling of extreme values

### Full Changelog (vs upstream)

**84 files changed, 8813 insertions(+), 664 deletions(-)**

#### New Files

| File | Description |
|------|-------------|
| **Security & Allocation** | |
| `jxl/src/util/memory_tracker.rs` | Thread-safe memory budget tracking with `MemoryTracker` and `MemoryGuard` |
| `jxl/src/util/profiling.rs` | Optional profiling instrumentation |
| **Color Management** | |
| `jxl/src/api/moxcms_wrapper.rs` | Optional moxcms CMS integration for ICC profile transforms |
| `jxl/src/render/stages/black.rs` | CMYK K-channel application stage |
| `jxl/src/render/stages/cms_cmyk.rs` | CMS-based CMYK→RGB conversion using embedded ICC |
| `jxl/src/render/stages/unpremultiply_alpha.rs` | Alpha unpremultiplication pipeline stage |
| **Test Infrastructure** | |
| `jxl/src/tests/codec_corpus.rs` | 184-image corpus parity tests with djxl reference |
| `jxl/src/tests/conformance.rs` | Official libjxl/conformance test suite (1143 lines) |
| `jxl/src/tests/coverage_boost.rs` | Additional tests for code coverage gaps |
| `jxl/src/tests/decode_api.rs` | API tests ported from libjxl decode_test.cc |
| `jxl/src/tests/entropy.rs` | ANS and Huffman codec edge case tests |
| `jxl/src/tests/feature_tests.rs` | Comprehensive JXL encoder option tests |
| `jxl/src/tests/parity.rs` | Parity test infrastructure and helpers |
| `jxl/src/tests/streaming.rs` | Chunked/progressive decoding tests |
| `jxl/src/tests/synthetic.rs` | Synthetic test image generation |
| **Scripts & Docs** | |
| `scripts/coverage.sh` | Local code coverage analysis |
| `scripts/coverage_gaps.py` | Coverage gap detection and reporting |
| `scripts/generate_synthetic_tests.py` | Test image generator script |
| `PARITY_INVESTIGATION.md` | Detailed bug investigation notes |
| `TEST_GAP_ANALYSIS.md` | Test coverage analysis document |

#### Modified Files

| File | Changes |
|------|---------|
| **Security & Limits** | |
| `jxl/src/api/options.rs` | `JxlDecoderLimits` API, `stop: Arc<dyn Stop>`, limit presets |
| `jxl/src/api/decoder.rs` | Wire limits and stop handle to decoder state |
| `jxl/src/api/inner/codestream_parser/non_section.rs` | Check limits during header parsing, initialize memory tracker |
| `jxl/src/api/inner/codestream_parser/sections.rs` | Check cancellation during section decoding |
| `jxl/src/error.rs` | Add `Error::LimitExceeded`, `Error::Cancelled`, `From<StopReason>` |
| `jxl/src/frame/mod.rs` | Store limits, stop handle, memory tracker in `DecoderState` |
| `jxl/src/frame/decode.rs` | Noise seeding fix; check limits during decode |
| `jxl/src/frame/group.rs` | Use fallible allocation for group buffers |
| `jxl/src/frame/render.rs` | sRGB transfer function, extra channel slots, CMYK blending order |
| `jxl/src/features/patches.rs` | Check `max_patches` limit |
| `jxl/src/features/spline.rs` | Check `max_spline_points` limit |
| `jxl/src/headers/size.rs` | Saturating arithmetic for size calculations |
| `jxl/src/icc/mod.rs` | Check `max_icc_size` limit |
| `jxl/src/icc/tag.rs` | Overflow-safe tag parsing |
| `jxl/src/image/typed.rs` | Use fallible allocation for image buffers |
| `jxl/src/render/builder.rs` | Memory tracking for render pipeline |
| `jxl/src/render/stages/chroma_upsample.rs` | Fallible allocation for upsample buffers |
| `jxl/src/util/vec_helpers.rs` | `NewWithCapacity`, `TryVecExt`, `AlignedVec` utilities |
| `jxl/src/util/mod.rs` | Export new modules |
| **Bug Fixes** | |
| `jxl/src/api/color.rs` | Grayscale ICC detection fix |
| `jxl/src/frame/modular/decode/channel.rs` | Extra channel bit_depth fix |
| `jxl/src/frame/modular/transforms/rct.rs` | Wrapping arithmetic to prevent panic |
| `jxl/src/frame/quant_weights.rs` | Minor fix |
| `jxl/src/headers/frame_header.rs` | Progressive AC last_pass validation fix |
| `jxl_simd/src/*.rs` | SIMD improvements across all platforms |

#### Commit History

```
da0c8cc merge: integrate security features from fix/parity-tests-and-bugfixes
cd4d96a docs: update README with full changelog vs upstream
5095a2a fix: improve conformance test ICC handling, 16/23 -> 17/23 passing
7959f04 docs: add conformance test status and TODOs to README
3b42220 feat: auto-fetch conformance test data on first run
26cdc9e feat: add official JPEG XL conformance test infrastructure
b8cff3c fix: CMYK blending order - blend in CMYK space then convert to RGB
375c5ef chore: upgrade moxcms from 0.5 to 0.7

Security & Allocation Features:
ae7ce8f security: implement max_memory_bytes tracking and fallible allocations
362f109 security: wire up max_patches limit and add aligned allocation utilities
0c16e7a test: add limit enforcement and cancellation tests
1bd9b2f feat: enforce security limits and add cancellation support
4b3b659 feat: add cancellation support and memory budget limit
0b7ebec security: add configurable JxlDecoderLimits API
325414b security: use checked arithmetic instead of saturating for limit calculations
a6ae2ad security: fix ICC tag and render builder overflow vulnerabilities
08d36ce security: fix integer overflow vulnerabilities in patches, splines, size

Earlier commits:
7eb846f fix: correct noise generation seeding for upsampled frames
590adb8 feat: add CMS-based CMYK to RGB conversion stage
c4ee8f1 feat: add BlackChannelStage for CMYK to RGB conversion
908170c fix: use extra channel's own bit_depth for modular-to-f32 conversion
4bad28f feat: add optional moxcms CMS integration
16ff26e feat: add UnpremultiplyAlphaStage for future alpha handling
f594118 fix: correct last_pass validation to require strictly increasing
b87a384 fix: detect grayscale from default pixel format, not output profile
9cfce3f fix: detect linear gamma from reference PNG gAMA chunk
6d49aee fix: extra channel format slots allocation
dfa374f fix: apply sRGB transfer function for XYB-encoded images
5d8e1ff fix: RCT transform overflow panic with wrapping arithmetic
... (earlier commits establishing test infrastructure)
```

### Upstream PRs

The following PRs have been submitted to upstream jxl-rs:

| PR | Status | Description |
|----|--------|-------------|
| [#602](https://github.com/libjxl/jxl-rs/pull/602) | Open | Integer overflow handling (saturating arithmetic, u64 returns) |
| [#603](https://github.com/libjxl/jxl-rs/pull/603) | Open | Patches: use saturating arithmetic for position calculations |
| [#604](https://github.com/libjxl/jxl-rs/pull/604) | Approved | Extra channel bit_depth fix |
| [#607](https://github.com/libjxl/jxl-rs/pull/607) | Open | ICC tag parsing overflow fix |
| [#609](https://github.com/libjxl/jxl-rs/issues/609) | Issue | Progressive AC last_pass validation bug |
| [#610](https://github.com/libjxl/jxl-rs/issues/610) | Issue | Noise seeding wrong when upsampling > 1 |