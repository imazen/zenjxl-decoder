# Proposed Contributions to jxl-rs

**Date**: 2025-12-29
**Contributor**: Lilith River (lilith@imazen.io)
**Fork**: https://github.com/lilith/jxl-rs
**Branch**: [`fix/parity-tests-and-bugfixes`](https://github.com/lilith/jxl-rs/tree/fix/parity-tests-and-bugfixes)

## Overview

I built parity testing infrastructure against libjxl/djxl which uncovered several decoding bugs. I also did a security audit and found integer overflow vulnerabilities. This document summarizes fixes I'd like to contribute.

---

## 1. Bug Fixes

All discovered via parity testing against djxl (183/184 tests pass after fixes, up from ~40%).

### 1.1 sRGB Transfer Function Default
[**`7c5432e`** - fix: apply sRGB transfer function by default for XYB decoding](https://github.com/lilith/jxl-rs/commit/7c5432e)

`xyb_output_linear` defaulted to `true`, causing XYB images to skip the sRGB transfer function. Mid-tones appeared ~2x too dark.

**Question**: Is linear output the intended default? Should it require explicit opt-in?

### 1.2 RCT Overflow Panic
[**`d2f7ac3`** - fix: use wrapping arithmetic for RCT transforms to prevent overflow panic](https://github.com/lilith/jxl-rs/commit/d2f7ac3)

RCT transforms panicked on overflow in debug builds. Lossless PFM images crashed.

**Fix**: Added `wrapping_add`/`wrapping_sub` to I32SimdVec trait (touches jxl_simd for all platforms).

### 1.3 Extra Channel Bit Depth
[**`b68ccf1`** - fix: use extra channel's own bit_depth for modular-to-f32 conversion](https://github.com/lilith/jxl-rs/commit/b68ccf1)

Extra channels used the image's `bit_depth` instead of their own `bit_depth()`. Alpha values were completely wrong for mixed bit-depth images.

### 1.4 Noise Generation for Upsampled Frames
[**`7eb846f`** - fix: correct noise generation seeding for upsampled frames](https://github.com/lilith/jxl-rs/commit/7eb846f)

Noise seeding was incorrect for upsampled frames. libjxl seeds each upsampling subregion independently and shares ONE RNG across all 3 channels per subregion. jxl-rs used a single seed for the entire upsampled area.

### 1.5 Progressive Pass Validation
[**`b8cdc60`** - fix: correct last_pass validation to require strictly increasing](https://github.com/lilith/jxl-rs/commit/b8cdc60)

`last_pass` validation required strictly increasing values, but the check was inverted.

### 1.6 Grayscale Detection from ICC Profiles
[**`0195bb4`** - fix: detect grayscale from default pixel format, not output color profile](https://github.com/lilith/jxl-rs/commit/0195bb4)

Grayscale detection failed for ICC profiles. Fixed by using `decoder.current_pixel_format().color_type` instead of checking `output_color_profile()`.

### 1.7 Extra Channel Format Slots
[**`584fe69`** - fix: allocate extra channel format slots for all extra channels](https://github.com/lilith/jxl-rs/commit/584fe69)

Decoder required format slots for ALL extra channels, not just alpha. Images with spot colors or CMYK crashed.

---

## 2. Security Hardening

### 2.1 Integer Overflow Vulnerabilities

[**`08d36ce`** - security: fix integer overflow vulnerabilities in patches, splines, size](https://github.com/lilith/jxl-rs/commit/08d36ce)
[**`a6ae2ad`** - security: fix ICC tag and render builder overflow vulnerabilities](https://github.com/lilith/jxl-rs/commit/a6ae2ad)
[**`325414b`** - security: use checked arithmetic instead of saturating for limit calculations](https://github.com/lilith/jxl-rs/commit/325414b)

Found integer overflows in:
- Patch coordinate calculations (could cause out-of-bounds access)
- Spline rendering loop bounds
- Image size calculations
- ICC tag offset/size parsing
- Render builder allocation sizing

All fixed with checked arithmetic and proper error propagation.

**Question**: Would you like CVE-style advisories? Are there downstream users to notify?

### 2.2 Decoder Limits API

[**`0b7ebec`** - security: add configurable JxlDecoderLimits API](https://github.com/lilith/jxl-rs/commit/0b7ebec)
[**`4b3b659`** - feat: add CancellationToken and memory budget limit](https://github.com/lilith/jxl-rs/commit/4b3b659)
[**`1bd9b2f`** - feat: enforce security limits and add cancellation support](https://github.com/lilith/jxl-rs/commit/1bd9b2f)
[**`ae7ce8f`** - security: implement max_memory_bytes tracking and fallible allocations](https://github.com/lilith/jxl-rs/commit/ae7ce8f)
[**`362f109`** - security: wire up max_patches limit and add aligned allocation utilities](https://github.com/lilith/jxl-rs/commit/362f109)

Added configurable limits for DoS protection:

```rust
let limits = JxlDecoderLimits::builder()
    .max_image_pixels(100_000_000)
    .max_memory_bytes(512 * 1024 * 1024)
    .max_patches(10_000)
    .cancellation_token(token)
    .build();

decoder.set_limits(limits);
```

**Question**: Is this API surface acceptable? Should it be public or internal-only?

---

## Questions

1. **Bug fixes**: Accept as-is, or need changes?
2. **Security fixes**: Need CVEs/advisories for downstream notification?
3. **Limits API**: Public API acceptable? Different design preferred?
4. **Process**: One PR or split by category?

---

## Parity Testing (How Bugs Were Found)

The test infrastructure compares pixel output against djxl reference using [codec-corpus](https://github.com/imazen/codec-corpus). After all fixes above, **183/184 tests pass (99.5%)**.

The only remaining failure is `cmyk_layers` which requires ICC-based CMS - different CMS libraries (moxcms vs skcms) produce legitimately different results.

**Question**: Want the test infrastructure included, or just the fixes it found?

---

## Other Experiments (Not Proposing)

| Branch | What | Finding |
|--------|------|---------|
| [`experiment/parallel-decode`](https://github.com/lilith/jxl-rs/tree/experiment/parallel-decode) | Parallel VarDCT | 1.35-1.8× for 4K+, but only 37% parallelizable |
| [`experiment/unsafe-simd-entropy`](https://github.com/lilith/jxl-rs/tree/experiment/unsafe-simd-entropy) | SIMD entropy | No improvement (ANS is inherently sequential) |
