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

### New Features

| Feature | File(s) | Description |
|---------|---------|-------------|
| CMS-based CMYK→RGB | `render/stages/cms_cmyk.rs`, `api/moxcms_wrapper.rs` | Convert CMYK to RGB using embedded ICC profiles via optional `moxcms` integration |
| BlackChannelStage | `render/stages/black.rs` | Simple CMYK K-channel application for images without ICC profiles |
| UnpremultiplyAlphaStage | `render/stages/unpremultiply_alpha.rs` | Pipeline stage for alpha unpremultiplication |

### Test Infrastructure

| Test Suite | File | Coverage |
|------------|------|----------|
| Parity Testing | `tests/parity.rs` | Infrastructure for pixel-exact comparison against `djxl` reference output |
| Codec-Corpus Tests | `tests/codec_corpus.rs` | 184 test images from codec-corpus with reference comparison |
| Decoder API Tests | `tests/decode_api.rs` | Tests ported from libjxl `decode_test.cc` |
| Feature Tests | `tests/feature_tests.rs` | Comprehensive tests for all JXL encoder options |
| Entropy Tests | `tests/entropy.rs` | ANS and Huffman codec edge cases |
| Streaming Tests | `tests/streaming.rs` | Chunked/progressive decoding scenarios |

### Parity Test Results

Against codec-corpus (184 JXL files with djxl reference output):
- **183/184 passing** (99.5%)
- **1 remaining**: `cmyk_layers` (requires full ICC-based CMS for CMYK→RGB conversion)

See [PARITY_INVESTIGATION.md](PARITY_INVESTIGATION.md) for detailed bug investigation notes.

### Files Changed

```
39 files changed, 5783 insertions(+), 649 deletions(-)

New files:
  jxl/src/api/moxcms_wrapper.rs        - Optional CMS integration
  jxl/src/render/stages/black.rs       - CMYK K-channel stage
  jxl/src/render/stages/cms_cmyk.rs    - CMS-based CMYK conversion
  jxl/src/render/stages/unpremultiply_alpha.rs
  jxl/src/tests/codec_corpus.rs        - Corpus parity tests
  jxl/src/tests/decode_api.rs          - API tests from libjxl
  jxl/src/tests/entropy.rs             - Entropy coding tests
  jxl/src/tests/feature_tests.rs       - Feature coverage tests
  jxl/src/tests/parity.rs              - Parity test infrastructure
  jxl/src/tests/streaming.rs           - Streaming decode tests
  PARITY_INVESTIGATION.md              - Bug investigation notes
  TEST_GAP_ANALYSIS.md                 - Test coverage analysis
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