# jxl-rs Parity Investigation

Started: 2025-12-27
Last Updated: 2025-12-27

## Overview

Investigating and fixing pixel parity issues between jxl-rs and libjxl (djxl reference).

---

## Completed Fixes

### 1. sRGB Transfer Function Bug (FIXED)

**Commit**: `a3cdf7a`

**Symptom**: VarDCT (lossy) images had mid-tones ~2x too dark
- basic.jxl: jxl-rs output `[255, 6, 5]`, reference `[255, 42, 37]`
- R channel correct, G/B channels ~7x too low

**Root Cause**: `xyb_output_linear` defaulted to `true` in `JxlDecoderOptions`, causing the sRGB transfer function (FromLinearStage) to be skipped.

**Fix**: Changed default to `false` in `jxl/src/api/options.rs:55`

**Impact**:
- basic.jxl now matches exactly: `[255, 42, 37]`
- ~56 tests now pass (up from ~40)

### 2. Grayscale Channel Handling (FIXED)

**Commit**: `5975c45` (previous session)

**Symptom**: Grayscale images were being compared as RGB vs 1-channel

**Fix**: Test infrastructure now properly handles grayscale output

### 3. RCT Overflow Crash (FIXED)

**Commit**: `46536b6`

**Symptom**: `lossless_pfm` crashed with "attempt to add with overflow" at `rct.rs:28`

**Root Cause**: Scalar i32 panics on overflow in debug mode, but RCT transform expects wrapping arithmetic for modular integer data.

**Fix**:
- Added `wrapping_add`/`wrapping_sub` methods to `I32SimdVec` trait
- Implemented for all SIMD types (SSE42, AVX, AVX512, NEON) and scalar
- Updated `rct.rs` to use wrapping operations

**Impact**: `lossless_pfm` now PASSES

### 4. Extra Channel Format Slots (FIXED)

**Commit**: `88ae4b6`

**Symptom**: `cmyk_layers`, `spot`, `large_header` crashed with assertion failure at `render.rs:372` - extra channel format count mismatch

**Root Cause**: Test only allocated extra channel slots for alpha, but decoder requires a slot for EVERY extra channel.

**Fix**: Allocate `None` slots for all extra channels based on `basic_info.extra_channels.len()`

**Impact**:
- `spot`: PASSES
- `large_header`: PASSES
- `cmyk_layers`: No longer crashes (now pixel mismatch - CMYK color space issue)

### 5. Linear Gamma Detection in Reference PNGs (FIXED)

**Commit**: `44d4128`

**Symptom**: max_error=74 for grayscale, progressive, patches tests

**Root Cause**: Some djxl-generated reference PNGs have gAMA=100000 (linear gamma, gamma=1.0) instead of sRGB gamma (gAMA=45455, gamma≈0.45). This happens for XYB-encoded images with embedded ICC profiles. Our decoder was outputting sRGB values while reference was in linear colorspace.

**Investigation**:
- Reference grayscale.png: pixel values 0-3 (linear)
- jxl-rs sRGB output: pixel values 15-28 (sRGB)
- jxl-rs linear output: pixel values 0-3 (matches!)

**Fix**:
- Added `png_has_linear_gamma()` to detect gAMA=100000 in reference PNGs
- When detected, decode with `xyb_output_linear=true` to match reference

**Impact**:
- grayscale, grayscale_5: PASS
- progressive, progressive_5: PASS
- patches, patches_5: PASS
- 177/184 tests pass (up from 171)

### 6. Grayscale ICC Profile Detection (FIXED)

**Commit**: (pending)

**Symptom**: grayscale_jpeg, with_icc showed "Channel count mismatch" - decoder output 3-4 channels, reference expected 1-2.

**Root Cause**: Grayscale detection only worked for `JxlColorProfile::Simple(GrayscaleColorSpace)`, not ICC profiles. When an ICC profile was embedded, the test didn't detect it as grayscale and requested RGB output.

**Fix**: Use `decoder.current_pixel_format().color_type` instead of checking `output_color_profile()`. The decoder already sets the correct default format based on `color_encoding.color_space` from the file header.

**Impact**:
- grayscale_jpeg, grayscale_jpeg_5: PASS
- with_icc: PASS
- 180/184 tests pass (up from 177)

### 7. Progressive AC Validation (FIXED)

**Commit**: (pending)

**Symptom**: progressive_ac failed with "PassesLastPassNonIncreasing" decode error.

**Root Cause**: The validation logic for `Passes::last_pass` was inverted. libjxl requires `last_pass` values to be **strictly increasing** (`last_pass[i] > last_pass[i-1]`), but jxl-rs was checking for **strictly decreasing** (`lp >= last_lp` rejects increasing values).

**Fix**: Changed validation from `if lp >= last_lp` to `if lp <= prev_lp` in frame_header.rs.

**Impact**:
- progressive_ac: PASS
- 181/184 tests pass (up from 180)

### 8. Extra Channel Bit Depth Bug (FIXED)

**Commit**: `908170c`

**Symptom**: alpha_premultiplied had max_error=239, alpha values were completely wrong (255 instead of 127-128)

**Root Cause**: Extra channels (including alpha) were using `metadata.bit_depth` instead of each extra channel's own `bit_depth` in the ConvertModularToF32Stage. This caused incorrect scaling during i32→f32 conversion.

**Fix**:
- Added `bit_depth()` getter to `ExtraChannelInfo`
- Changed render.rs to use `metadata.extra_channel_info[i - 3].bit_depth()` for extra channels

**Impact**:
- alpha_premultiplied: PASS
- 182/184 tests pass (up from 181)

---

## Remaining Issues (2 failures)

### 1. CMYK Color Space (max_error=224)

| File | Error |
|------|-------|
| cmyk_layers | max_error=224, error_count=324100/1048576 |

**Investigation (2025-12-27)**:
- Added `BlackChannelStage` to apply K channel to CMY (simple CMYK→RGB: R=C*K, G=M*K, B=Y*K)
- Stage runs BEFORE BlendingStage to ensure layers are composited with correct RGB values
- However, this simple formula doesn't match djxl's output

**Root Cause**: CMYK to RGB conversion in libjxl requires **ICC profile-based CMS** (lcms2/skcms).
The cmyk_layers.jxl file has an embedded CMYK ICC profile. djxl uses its CMS to convert
CMYK colors to RGB accurately. Our simple K multiplication is insufficient.

**Fix Needed**: Enable moxcms CMS integration for CMYK ICC profile conversion. The CMS needs to:
1. Detect CMYK color space from ICC profile
2. Create CMYK→sRGB transform using the embedded profile
3. Apply transform during render pipeline

### 2. Multi-layer Noise/Spline (max_error=66-70)

| File | Error |
|------|-------|
| multiple_layers_noise_spline | max_error=66-70, error_count=~3.5M/9.4M (37%) |

**Investigation (2025-12-27)**:
- Individual noise tests PASS: noise, noise_5, 8x8_noise, noise_0, noise_1, photon_noise_*
- Individual spline tests PASS: animation_spline, splines, spline_on_first_frame
- Only the COMBINATION of layers + noise + splines fails
- Error pattern: primarily in B (blue) channel, values differ by 0-16 (noise-like)
- R channel (red) matches almost exactly; G channel (green) is 0 in both

**Detailed Debug Output (pixel comparison)**:
- Pixel (0,0): ref=[124, 0, 0, 255] jxl-rs=[124, 0, 11, 255] → B differs by 11
- Pattern repeats: B channel has noise differences while R/G match

**RNG Investigation**:
- Checked Xorshift128Plus initialization against libjxl
- jxl-rs 4-seed constructor uses: s0[i] = split_mix_64(s0[i-1]), s1[i] = split_mix_64(s1[i-1])
- libjxl uses: s0[i] = SplitMix64(s1[i-1]), s1[i] = SplitMix64(s0[i])
- ATTEMPTED FIX: Changed to match libjxl → BROKE all other noise tests
- REVERTED: Original jxl-rs initialization is correct for passing noise tests

**Conclusion**: RNG is NOT the issue. The issue is specific to the multi-layer + noise + splines
combination.

**Libjxl Source Analysis**:
- libjxl noise seeding: `x0 = (gx * upsampling + ix) * group_dim`, `y0 = (gy * upsampling + iy) * group_dim`
- jxl-rs noise seeding: `x0 = gx * upsampling * group_dim`, `y0 = gy * upsampling * group_dim`
- No frame origin offset in either (noise uses local frame coordinates)
- Blending stage does NOT access noise channels - they're regular channels by then

**Next Investigation Steps**:
1. Print visible_frame_index and nonvisible_frame_index for each frame in multi-layer decode
2. Check if multiple frames have noise enabled independently
3. Compare x0, y0 coordinates between jxl-rs and djxl for multi-layer images
4. Investigate the `ix`, `iy` loop in libjxl noise generation - may indicate subgroup iteration

---

## Test Results Summary

### Before Fix (from CHANGES_WHEN_YOU_WERE_AWAY.md)
- Passed: ~66 (36%)
- Failed: ~114
- Crashes: 4
- Total: 184

### After sRGB Fix (partial run - 69 files)
- Passed: 56 (81%)
- Failed: 9
- Crashes: 4

### After RCT + Extra Channel Fixes (full run - 184 files)
- Passed: 171 (93%)
- Failed: 13
- Crashes: 0

### After Linear Gamma Detection Fix (full run - 184 files)
- Passed: 177 (96%)
- Failed: 7
- Crashes: 0

### After Grayscale ICC Fix (full run - 184 files)
- Passed: 180 (98%)
- Failed: 4
- Crashes: 0

### After Progressive AC Validation Fix (full run - 184 files)
- Passed: 181 (98.4%)
- Failed: 3
- Crashes: 0

### After Extra Channel Bit Depth Fix (full run - 184 files)
- Passed: 182 (98.9%)
- Failed: 2
- Crashes: 0

---

## Technical Notes

### XYB to RGB Pipeline

1. **XybStage**: XYB → linear RGB (correct)
   - Formula: `linear = ((gamma - cbrt(bias))^3 + bias) * intensity_scale`
   - Unit tests pass for sRGB primaries

2. **FromLinearStage**: linear RGB → sRGB (was being skipped!)
   - Applies sRGB transfer function: `sRGB = 1.055 * linear^(1/2.4) - 0.055`

3. **ConvertF32ToU8Stage**: float [0,1] → u8 [0,255]

### Key Files

- `jxl/src/api/options.rs` - Decoder options (fixed here)
- `jxl/src/frame/render.rs` - Pipeline construction
- `jxl/src/render/stages/xyb.rs` - XYB conversion
- `jxl/src/render/stages/from_linear.rs` - Transfer functions
- `jxl/src/frame/group.rs` - VarDCT dequantization
- `jxl/src/frame/modular/transforms/rct.rs` - RCT (overflow bug)

---

## Debug Commands

```bash
# Run single parity test
CODEC_CORPUS_PATH=/path/to/codec-corpus cargo test -p jxl test_parity_basic -- --nocapture

# Run all parity tests
CODEC_CORPUS_PATH=/path/to/codec-corpus cargo test -p jxl test_all_codec_corpus_parity -- --ignored --nocapture

# Debug with backtrace
RUST_BACKTRACE=1 CODEC_CORPUS_PATH=/path/to/codec-corpus cargo test -p jxl test_parity_basic -- --nocapture
```

---

## Session Log

### 2025-12-27

1. Investigated VarDCT max_error=74 pattern
2. Verified reference data was correct (djxl outputs `[255, 42, 37]`)
3. Traced through XYB stage - found conversion was mathematically correct
4. Discovered linear values were being output without sRGB TF
5. Found `xyb_output_linear: true` default was the culprit
6. Fixed by changing default to `false`
7. Verified fix: basic.jxl now matches exactly
8. Partial parity test: 56/69 pass (81%)
9. Fixed RCT overflow crash by adding wrapping arithmetic to I32SimdVec
10. Fixed extra channel format assertion by allocating slots for all extra channels
11. Full parity test: 171/184 pass (93%), 0 crashes
12. Investigated max_error=74 pattern in grayscale, progressive, patches
13. Discovered reference PNGs have gAMA=100000 (linear gamma)
14. Added png_has_linear_gamma() to detect linear references
15. When detected, decode with xyb_output_linear=true
16. Full parity test: 177/184 pass (96%), 0 crashes
17. Fixed grayscale ICC detection by using default pixel format's color_type
18. Full parity test: 180/184 pass (98%), 0 crashes
19. Fixed progressive_ac by correcting last_pass validation (strictly increasing, not decreasing)
20. Full parity test: 181/184 pass (98.4%), 0 crashes
21. Created UnpremultiplyAlphaStage (for future use)
22. Tested unpremultiply on alpha_premultiplied - made things worse (djxl doesn't unpremultiply)
23. Remaining 3 failures need deeper investigation (alpha, CMYK, noise/splines)
24. Added moxcms as optional CMS dependency with "cms" feature flag
25. Implemented JxlCms trait wrapper for moxcms (MoxCms struct)
26. Investigated alpha_premultiplied - found alpha values completely wrong (255 instead of 127)
27. Root cause is NOT fill_opaque_alpha or pipeline config - actual alpha data values are wrong
28. Found bug: extra channels using `metadata.bit_depth` instead of their own `bit_depth`
29. Fixed by using `metadata.extra_channel_info[i - 3].bit_depth()` in render.rs
30. Full parity test: 182/184 pass (98.9%), 0 crashes
31. Remaining 2 failures: cmyk_layers (CMYK color space), multiple_layers_noise_spline (noise/splines)
32. Investigated cmyk_layers - has Black (K) extra channel + Alpha
33. Created BlackChannelStage to apply K to CMY (simple CMYK→RGB formula: R=C*K)
34. Added stage to pipeline BEFORE BlendingStage for correct layer compositing
35. Simple K multiplication doesn't match djxl - requires ICC profile-based CMS
36. Investigated multiple_layers_noise_spline - individual noise/spline tests all pass
37. Only the combination of layers + noise + splines fails - likely pipeline ordering issue
38. Created debug test for multiple_layers_noise_spline to compare pixel values
39. Found error pattern: B channel has noise-like differences (0-16), R/G channels match
40. Investigated Xorshift128Plus RNG initialization - compared with libjxl
41. jxl-rs uses different expansion loop than libjxl (s0[i] from s0[i-1] vs s1[i-1])
42. Attempted to match libjxl initialization - BROKE all other noise tests
43. Reverted RNG change - original implementation is correct for passing tests
44. Conclusion: RNG is NOT the issue for multiple_layers_noise_spline
45. Issue is specific to multi-layer + noise + splines combination, needs further investigation
