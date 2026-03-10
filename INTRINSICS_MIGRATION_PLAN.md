# Plan: Eliminate jxl_simd, Switch to archmage + magetypes

> **SUPERSEDED (2026-03-10):** Phase 1 of this plan (making jxl_simd safe) was completed
> via a different approach — jxl_simd was rewritten internally to use archmage intrinsics
> and magetypes vector types, with `#[arcane]` proc macros for dispatch. The crate is now
> `#![forbid(unsafe_code)]` with zero unsafe anywhere. The remaining phases (eliminating
> jxl_simd entirely and replacing it with direct archmage/magetypes/linear-srgb/garb usage
> in the jxl crate) have not been started. The plan below may still be useful as a roadmap
> for that future work, but the API mappings and phase structure should be re-evaluated.

## Goal

Remove the `jxl_simd` crate entirely. Replace all SIMD dispatch with `archmage` macros (`#[autoversion]`, `incant!`) and `magetypes` vector types (`f32x8<T>`, `i32x8<T>`, etc.). Use existing ecosystem crates (`linear-srgb`, `garb`, `zenpixels-convert`) instead of reimplementing SIMD kernels.

## Current State

- **jxl_simd**: ~9000 lines, 6 backends (scalar, sse42, avx, avx512, neon, wasm128)
- **38 files** in `jxl/src/` import from jxl_simd
- **42 unique SIMD methods** used across the codebase
- **~20 dispatch points** via `simd_function!` macro
- jxl_simd already uses archmage internally for its AVX/AVX512/NEON backends

## Available Ecosystem Crates

These crates are authored by the same team, use archmage, and are `#![forbid(unsafe_code)]`:

| Crate | Version | Provides | Replaces |
|-------|---------|----------|----------|
| **archmage** | 0.9.3 | Token dispatch, `#[autoversion]`, `incant!`, safe intrinsics | jxl_simd dispatch macros |
| **magetypes** | 0.9.3 | f32x8, i32x8, etc. with operators, transcendentals | jxl_simd F32SimdVec/I32SimdVec traits |
| **linear-srgb** | 0.6.2 | sRGB/PQ/HLG/BT.709 transfer functions, rational poly, LUT tables | `color/tf.rs`, `util/rational_poly.rs`, `stages/from_linear.rs`, `stages/to_linear.rs` |
| **garb** | 0.2.0 | RGBA↔BGRA swizzle, RGB↔RGBA expand/strip, channel interleaving | `render/simd_utils.rs` interleave/deinterleave |
| **zenpixels-convert** | 0.1.0 | Format negotiation, depth conversion, alpha handling | `stages/convert.rs` depth/format logic |

## Target Architectures

| Target | Token | Width | Notes |
|--------|-------|-------|-------|
| x86-64 v3 (AVX2+FMA) | `X64V3Token` | 256-bit | Primary desktop/server |
| x86-64 v4 (AVX-512) | `X64V4Token` | 512-bit | Feature-gated |
| ARM v2 (NEON+) | `Arm64V2Token` | 128-bit | All modern ARM |
| WASM SIMD128 | `Wasm128Token` | 128-bit | Compile-time only |
| Scalar fallback | `ScalarToken` | 1 lane | Always available |

We drop SSE4.2 (`X64V2Token`) as a separate backend. AVX2 is baseline x86 now, and the scalar fallback covers older CPUs. Simplifies from 6 backends to 5.

## API Gap Analysis

### Covered by ecosystem crates (no jxl code needed)

| jxl_simd usage | Crate | API |
|---|---|---|
| `eval_rational_poly_simd()` for sRGB TF | **linear-srgb** | `x8::srgb_to_linear_v3()`, `x8::linear_to_srgb_v3()` (token-gated `#[rite]`, inlines into caller) |
| PQ transfer function | **linear-srgb** | `tf::pq_to_linear()`, `tf::linear_to_pq()` (requires `transfer` feature) |
| HLG transfer function | **linear-srgb** | `tf::hlg_to_linear()`, `tf::linear_to_hlg()` |
| BT.709 transfer function | **linear-srgb** | `tf::bt709_to_linear()`, `tf::linear_to_bt709()` |
| `store_interleaved_2/3/4` for output | **garb** | `rgb_to_rgba()`, `rgba_to_bgra_inplace()`, strided variants |
| `fast_log2f`, `fast_pow2f` | **magetypes** | `f32x8::log2_lowp()`, `f32x8::exp2_lowp()` (~1% error, same as current) |
| `log2_midp`, `exp2_midp` | **magetypes** | `f32x8::log2_midp()`, `f32x8::exp2_midp()` (~3 ULP) |
| Rational polynomial evaluation | **linear-srgb** | Internal, or reuse pattern from magetypes transcendentals |
| u8↔f32 LUT-based sRGB | **linear-srgb** | `srgb_u8_to_linear()`, `linear_to_srgb_u8()` (const LUT, zero math) |

### Available in magetypes (no wrapper needed)

- `splat`, `load`, `store`, `from_slice`, `from_array`, `to_array`
- Arithmetic: `+`, `-`, `*`, `/`, `mul_add`, `mul_sub`
- Comparisons: `simd_gt`, `simd_lt`, `simd_ge`, `simd_le`, `simd_eq`
- `blend(mask, if_true, if_false)` (replaces `if_then_else_f32`)
- `abs`, `sqrt`, `floor`, `ceil`, `round`, `min`, `max`, `clamp`
- `bitcast_f32_to_i32`, `bitcast_i32_to_f32`, `convert_f32_to_i32`, `convert_i32_to_f32`
- `interleave_lo`, `interleave_hi`, `interleave_4ch`, `deinterleave_4ch`
- `from_u8`, `to_u8` (f32x4/f32x8 to/from u8)
- Shift operations: `shl_const::<N>()`, `shr_arithmetic_const::<N>()`
- Bitwise: `&`, `|`, `^`, `!`
- Reductions: `reduce_add`, `reduce_min`, `reduce_max`

### Needs thin inline helpers (in jxl, not a separate module)

These are trivial one-liners, not worth a dedicated module:

| jxl_simd method | Inline replacement |
|---|---|
| `neg_mul_add(a, b)` = `-(self*a) + b` | `b - self * a` or `(-self).mul_add(a, b)` |
| `copysign(sign_source)` | Bitwise: `(self & !SIGN_BIT) \| (sign & SIGN_BIT)` |
| `andnot(other)` | `!self & other` |
| `lt_zero()` / `eq_zero()` | `self.simd_lt(zero)` / `self.simd_eq(zero)` |
| `maskz_i32(mask)` | `blend(mask, self, zero)` |
| `round_store_u8(dest)` | `self.clamp(0, 255).to_u8()` + store |
| `round_store_u16(dest)` | `self.round().clamp(0, 65535)` + convert + store |
| `load_f16_bits` / `store_f16_bits` | Bit manipulation (shift + mask), or `half` crate |
| `mul_wide_take_high(scalar)` | Widen to i64, multiply, shift right 32 |
| `prepare_table_bf16_8` / `table_lookup_bf16_8` | `vpermps` / equivalent via `incant!` |

### Dispatch pattern replacement

**Current** (`simd_function!` macro):
```rust
simd_function!(
    gaborish_dispatch,
    d: D,
    fn gaborish_process(stage: &GaborishStage, xsize: usize, ...) {
        let w0 = D::F32Vec::splat(d, stage.weight0);
        // ... vectorized loop using D::F32Vec
    }
);
// Called as: gaborish_dispatch(stage, xsize, ...);
```

**New** (`#[autoversion]` for auto-vectorized code):
```rust
#[autoversion]
fn gaborish_process(_token: SimdToken, stage: &GaborishStage, xsize: usize, ...) {
    // Scalar-style loop — LLVM auto-vectorizes per target feature level
    for x in 0..xsize {
        let sum = center[x+1] * w0
            + (top[x+1] + left[x+1] + bottom[x+1] + right[x+1]) * w1
            + (top[x] + top[x+2] + bottom[x] + bottom[x+2]) * w2;
        out[x] = sum;
    }
}
```

**New** (`incant!` for hand-tuned code):
```rust
#[rite]
fn epf0_v3(_token: X64V3Token, ...) { /* AVX2 with masking */ }
#[rite]
fn epf0_neon(_token: NeonToken, ...) { /* NEON variant */ }
fn epf0_scalar(_token: ScalarToken, ...) { /* scalar fallback */ }

// At call site:
incant!(epf0(args), [v3, neon, wasm128, scalar]);
```

## Migration Phases

### Phase 1: Add ecosystem deps, migrate transfer functions (5 files)

Add `linear-srgb`, `garb`, `magetypes`, and `archmage` as direct deps of the `jxl` crate.

**Replace entirely with linear-srgb calls:**
1. `util/rational_poly.rs` — delete, use linear-srgb's rational poly or magetypes transcendentals
2. `util/fast_math.rs` — replace `fast_log2f`/`fast_pow2f` with `f32x8::log2_lowp()`/`f32x8::exp2_lowp()` from magetypes
3. `color/tf.rs` — replace with `linear_srgb::tf::*` functions (sRGB, PQ, HLG, BT.709)
4. `render/stages/from_linear.rs` — use `linear_srgb::tokens::x8::linear_to_srgb_v3()` etc.
5. `render/stages/to_linear.rs` — use `linear_srgb::tokens::x8::srgb_to_linear_v3()` etc.

**Validation**: Conformance tests must still pass — linear-srgb uses the same rational polynomial coefficients as libjxl.

### Phase 2: Migrate simple stages with `#[autoversion]` (6 files)

Stages that only use basic F32SimdVec operations (load, store, arithmetic, FMA). Write plain scalar loops and let LLVM auto-vectorize.

**Files** (simplest first):
1. `render/stages/gaborish.rs` — 3x3 convolution, only load/store/arithmetic/FMA
2. `render/stages/premultiply_alpha.rs` — multiply RGB by alpha
3. `render/stages/unpremultiply_alpha.rs` — divide RGB by alpha
4. `render/stages/ycbcr.rs` — color space conversion
5. `render/stages/chroma_upsample.rs` — bilinear upsampling
6. `render/stages/upsample.rs` — general upsampling

**Validation**: `test_stage_consistency` + concurrency tests after each file.

### Phase 3: Migrate masking-heavy stages (6 files)

Stages that use comparisons, conditional selects, and copysign. Use `#[autoversion]` where LLVM handles ternary patterns well, `incant!` for complex masking.

**Files**:
1. `render/stages/xyb.rs` — XYB to RGB, uses gt/if_then_else/copysign
2. `render/stages/convert.rs` — type conversion, uses round_store_u8/u16
3. `render/stages/noise.rs` — noise synthesis, uses masking
4. `render/stages/epf/common.rs` — EPF shared code
5. `render/stages/epf/epf0.rs` — EPF step 0
6. `render/stages/epf/epf1.rs` + `epf2.rs` — EPF steps 1-2

### Phase 4: Migrate transforms (14 files)

The IDCT transforms are the performance-critical core. They use only load/store/arithmetic/FMA on f32, making them ideal `#[autoversion]` candidates.

**Files**: `transforms/idct{2,4,8,16,32}.rs`, `transforms/idct2d.rs`, `transforms/idct_large.rs`, `transforms/reinterpreting_dct{2,4,8,16,32}.rs`, `transforms/reinterpreting_dct2d.rs`, `transforms/transform.rs`

**Pattern**: LLVM is excellent at auto-vectorizing FMA-heavy DSP loops. Verify with `cargo asm`.

**Risk**: 2D transpose operations use architecture-specific shuffles. May need `incant!` variants for the transpose kernel only.

### Phase 5: Migrate modular/group operations (3 files)

1. `frame/group.rs` — dequantization (I32SimdVec)
2. `frame/modular/transforms/rct.rs` — reversible color transform (I32SimdVec)
3. `frame/modular/transforms/squeeze.rs` — modular squeeze (I32SimdVec, U32SimdVec)

**Pattern**: `#[autoversion]` for the integer scalar loops. magetypes `i32x8<T>` / `u32x8<T>` if explicit SIMD needed.

### Phase 6: Migrate save/output + interleave utils (2 files)

1. `render/low_memory_pipeline/save/identity.rs` — U8SimdVec, U16SimdVec, interleaving
2. `render/simd_utils.rs` — interleave/deinterleave dispatch helpers

**Pattern**: Use `garb` for channel swizzling where it fits. For custom interleave patterns (2-channel, 3-channel f32), use magetypes `interleave_lo`/`interleave_hi` or `#[autoversion]` scalar loops.

### Phase 7: Remove jxl_simd

1. Remove `jxl_simd` from workspace members in root `Cargo.toml`
2. Remove `jxl_simd` dependency from `jxl/Cargo.toml`
3. Delete `jxl_simd/` directory
4. Remove feature forwarding (`sse42`, `avx`, `avx512`, `neon`, `wasm128`) — replace with archmage feature names
5. Update CI workflows
6. Replace `test_all_instruction_sets!` with per-token test helpers or `#[autoversion]` tests
7. Replace `bench_all_instruction_sets!` similarly

## Key Decisions

### `#[autoversion]` vs `incant!`

- **`#[autoversion]`**: Best for regular loop bodies. LLVM auto-vectorizes scalar code compiled with appropriate `target_feature`. Zero manual intrinsics. Works for ~80% of our code (IDCT, convolutions, FMA chains, simple arithmetic, basic masking).

- **`incant!`**: Required when auto-vectorization fails or produces suboptimal code. Needed for: transpose kernels, table lookups, bf16 table lookups.

**Default to `#[autoversion]`**. Only drop to `incant!` when benchmarks show it matters.

### Width-generic code

For `#[autoversion]` code, width doesn't matter — just write scalar loops. The compiler picks the best vector width per target.

For `incant!` code, write the function generic over `T: SimdTypes` and monomorphize per token.

### Scalar fallback

`#[autoversion]` generates a `_scalar` variant automatically. `incant!` requires an explicit `_scalar` function. Both use archmage's `ScalarToken`.

### Transfer functions

Don't reimplement. `linear-srgb` provides SIMD-accelerated, `#![forbid(unsafe_code)]`, token-gated `#[rite]` functions that inline directly into our pipeline stages. Same rational polynomial approach as libjxl, verified to the same precision.

## Benchmarking Strategy

1. Before each phase, run full decode benchmarks, save results under `benchmarks/`
2. After each phase, re-run and compare
3. Key images: `bike_web_q85.jxl` (multi-group VarDCT), `city_4k_q75.jxl` (large), `3x3_srgb_lossless.jxl` (modular)
4. If any regression >5%, investigate with `cargo asm` before proceeding
5. Commit benchmark results with git hash + command used

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `#[autoversion]` slower than hand-written intrinsics | Medium | Medium | Benchmark each phase, fall back to `incant!` |
| Transpose kernel can't auto-vectorize | High | Low | Keep as `incant!` with per-arch variants |
| bf16 table lookup has no magetypes equivalent | High | Low | Implement via `incant!` with arch intrinsics |
| linear-srgb precision differs from current impl | Low | High | Conformance tests catch immediately |
| Build times increase | Low | Low | Fewer backends (5 vs 6) may offset monomorphization cost |

## Dependency Changes

### Added to `jxl/Cargo.toml`
```toml
archmage = { version = "0.9", features = ["macros"] }
magetypes = "0.9"
linear-srgb = { version = "0.6", features = ["transfer"] }
garb = "0.2"
```

### Removed from `jxl/Cargo.toml`
```toml
jxl_simd = { package = "zenjxl-decoder-simd", path = "../jxl_simd", version = "=0.3.0" }
```

### Removed from workspace
```toml
# members: remove "jxl_simd"
```

## Phase Summary

| Phase | Files | What | Key crate |
|-------|-------|------|-----------|
| 1. Transfer functions | 5 | Delete reimplementations, use linear-srgb | linear-srgb, magetypes |
| 2. Simple stages | 6 | `#[autoversion]` scalar loops | archmage |
| 3. Masking stages | 6 | `#[autoversion]` or `incant!` | archmage, magetypes |
| 4. Transforms | 14 | `#[autoversion]` FMA loops | archmage |
| 5. Modular/group | 3 | `#[autoversion]` integer loops | archmage, magetypes |
| 6. Save/output | 2 | garb + magetypes interleave | garb, magetypes |
| 7. Remove jxl_simd | cleanup | Delete crate, update CI | — |
| **Total** | **36 modified, jxl_simd deleted** | | |
