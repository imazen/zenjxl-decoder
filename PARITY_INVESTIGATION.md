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

---

## Remaining Issues (13 failures)

### 1. Progressive Mode Errors

| File | Error |
|------|-------|
| progressive | max_error=74 |
| progressive_5 | max_error=74 |

**Hypothesis**: Progressive AC decoding may have coefficient ordering issues

### 3. Patches Mode Errors

| File | Error |
|------|-------|
| patches | max_error=74 |
| patches_5 | max_error=74 |

### 4. Grayscale VarDCT Errors

| File | Error |
|------|-------|
| grayscale | max_error=74 |
| grayscale_5 | max_error=74 |
| grayscale_jpeg | Channel count mismatch (3 vs 1) |
| grayscale_jpeg_5 | Channel count mismatch (3 vs 1) |

**Note**: The max_error=74 for grayscale may be a separate issue from the RGB fix

### 5. Alpha Premultiplication

| File | Error |
|------|-------|
| alpha_premultiplied | max_error=239 |

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

### After All Session Fixes (full run - 184 files)
- Passed: 171 (93%)
- Failed: 13
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
