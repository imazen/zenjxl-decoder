# Plan: Eliminate jxl_simd, Switch to archmage + magetypes

## Goal

Remove the `jxl_simd` crate entirely. Replace all SIMD dispatch with `archmage` macros (`#[autoversion]`, `incant!`) and `magetypes` vector types (`f32x8<T>`, `i32x8<T>`, etc.).

## Current State

- **jxl_simd**: ~9000 lines, 6 backends (scalar, sse42, avx, avx512, neon, wasm128)
- **38 files** in `jxl/src/` import from jxl_simd
- **42 unique SIMD methods** used across the codebase
- **~20 dispatch points** via `simd_function!` macro
- jxl_simd already uses archmage internally for its AVX/AVX512/NEON backends

## Target Architectures

| Target | Token | Width | Notes |
|--------|-------|-------|-------|
| x86-64 v3 (AVX2+FMA) | `X64V3Token` | 256-bit | Primary desktop/server |
| x86-64 v4 (AVX-512) | `X64V4Token` | 512-bit | Feature-gated |
| ARM v2 (NEON+) | `Arm64V2Token` | 128-bit | All modern ARM |
| WASM SIMD128 | `Wasm128Token` | 128-bit | Compile-time only |
| Scalar fallback | `ScalarToken` | 1 lane | Always available |

Note: We drop SSE4.2 (`X64V2Token`) as a separate backend. AVX2 is baseline x86 now, and the scalar fallback covers older CPUs. Simplifies from 6 backends to 5.

## API Gap Analysis

### Available in magetypes (no work needed)
- `splat`, `load`, `store`, `from_slice`, `from_array`, `to_array`
- Arithmetic: `+`, `-`, `*`, `/`, `mul_add`, `mul_sub`
- Comparisons: `simd_gt`, `simd_lt`, `simd_ge`, `simd_le`, `simd_eq`
- `blend(mask, if_true, if_false)` (replaces `if_then_else_f32`)
- `abs`, `sqrt`, `floor`, `ceil`, `round`, `min`, `max`, `clamp`
- `bitcast_f32_to_i32`, `bitcast_i32_to_f32`, `convert_f32_to_i32`, `convert_i32_to_f32`
- `interleave_lo`, `interleave_hi`, `interleave_4ch`, `deinterleave_4ch`
- `from_u8`, `to_u8` (f32x4/f32x8 to/from u8)
- Transcendentals: `log2_midp`, `exp2_midp`, `pow_midp`, `cbrt_midp`
- Shift operations: `shl_const::<N>()`, `shr_arithmetic_const::<N>()`
- Bitwise: `&`, `|`, `^`, `!`
- Reductions: `reduce_add`, `reduce_min`, `reduce_max`

### Needs thin wrappers in jxl (build in a `simd_compat` module)

| jxl_simd method | Replacement strategy |
|---|---|
| `neg_mul_add(a, b)` = `-(self*a) + b` | `b - self * a` or `(-self).mul_add(a, b)` |
| `copysign(sign_source)` | Bitwise: `(self & !SIGN_BIT) \| (sign & SIGN_BIT)` |
| `andnot(other)` | `!self & other` (bitwise NOT then AND) |
| `lt_zero()` / `eq_zero()` | `self.simd_lt(zero)` / `self.simd_eq(zero)` |
| `maskz_i32(mask)` | `blend(mask, self, zero)` |
| `shl!(val, N)` / `shr!(val, N)` | `val.shl_const::<N>()` / `val.shr_arithmetic_const::<N>()` |
| `load_deinterleaved_2/3(slice)` | Manual: load contiguous then shuffle, or scalar loop |
| `store_interleaved_2/3(a, b, out)` | Manual: `interleave` + store, or scalar loop |
| `round_store_u8(dest)` | `self.round().clamp(0, 255).to_u8()` + store |
| `round_store_u16(dest)` | `self.round().clamp(0, 65535)` + convert + store |
| `load_f16_bits` / `store_f16_bits` | Use `half` crate or manual bit manipulation |
| `mul_wide_take_high(scalar)` | `((self as i64) * (scalar as i64)) >> 32` via widen+shift |
| `prepare_table_bf16_8` / `table_lookup_bf16_8` | Keep as arch-specific helper or use `vpermps`/equivalent |

### Dispatch pattern replacement

**Current** (`simd_function!` macro):
```rust
simd_function!(
    my_dispatch,
    d: D,
    fn my_function(args...) { /* uses D::F32Vec */ }
);
// Called as: my_dispatch(args...);
```

**New** (`#[autoversion]` for auto-vectorized code):
```rust
#[autoversion]
fn my_function(_token: SimdToken, args...) {
    // Write scalar-style loop, compiler auto-vectorizes
    for i in 0..n {
        out[i] = a[i] * b[i] + c[i];
    }
}
```

**New** (`incant!` for hand-tuned intrinsics):
```rust
// Define per-architecture variants
#[rite]
fn my_function_v3(_token: X64V3Token, args...) { /* AVX2 intrinsics */ }
#[rite]
fn my_function_neon(_token: NeonToken, args...) { /* NEON intrinsics */ }
fn my_function_scalar(_token: ScalarToken, args...) { /* scalar fallback */ }

// Dispatch at call site
incant!(my_function(args), [v3, neon, wasm128, scalar]);
```

## Migration Phases

### Phase 1: Create simd_compat bridge module (in jxl crate)

Add `jxl/src/simd_compat.rs` that provides thin wrapper functions for operations magetypes lacks natively. This module uses `archmage` and `magetypes` directly.

Implement:
- `neg_mul_add(a, b, c)` -> `c - a * b`
- `copysign(value, sign)` -> bitwise
- `round_store_u8(f32_vec, dest)` -> clamp + convert + store
- `round_store_u16(f32_vec, dest)` -> clamp + convert + store
- `load_deinterleaved_2/3/4` -> loads + shuffles
- `store_interleaved_2/3/4` -> shuffles + stores
- `load_f16_bits` / `store_f16_bits` -> half-float bit manipulation
- `mul_wide_take_high` -> widen + shift
- `prepare_table_bf16_8` / `table_lookup_bf16_8` -> permute instructions

### Phase 2: Migrate simple stages (6 files)

Start with stages that only use basic F32SimdVec operations (load, store, arithmetic, FMA). These can use `#[autoversion]` directly — the compiler will auto-vectorize them.

**Files** (simplest first):
1. `render/stages/gaborish.rs` — 3x3 convolution, only load/store/arithmetic/FMA
2. `render/stages/premultiply_alpha.rs` — multiply RGB by alpha
3. `render/stages/unpremultiply_alpha.rs` — divide RGB by alpha
4. `render/stages/ycbcr.rs` — color space conversion
5. `render/stages/chroma_upsample.rs` — bilinear upsampling
6. `render/stages/upsample.rs` — general upsampling

**Pattern**: Replace `simd_function!` with `#[autoversion]`, remove `D::F32Vec` type parameter, use plain `f32` scalar loops. The compiler auto-vectorizes.

**Validation**: Run `test_stage_consistency` tests + concurrency tests after each file.

### Phase 3: Migrate arithmetic-heavy stages (8 files)

Stages that use comparisons, masking, and transfer functions. Need `blend()` for conditional selects.

**Files**:
1. `render/stages/xyb.rs` — XYB to RGB, uses gt/if_then_else/copysign
2. `render/stages/convert.rs` — type conversion, uses round_store_u8/u16
3. `render/stages/from_linear.rs` — linear to sRGB, uses rational_poly + masking
4. `render/stages/to_linear.rs` — sRGB to linear, uses rational_poly + masking
5. `render/stages/noise.rs` — noise synthesis, uses masking
6. `render/stages/epf/common.rs` — EPF shared code
7. `render/stages/epf/epf0.rs` — EPF step 0
8. `render/stages/epf/epf1.rs` — EPF step 1
9. `render/stages/epf/epf2.rs` — EPF step 2

**Pattern**: Use `incant!` with magetypes for comparison-heavy code. Define `_v3`, `_neon`, `_wasm128`, `_scalar` variants. Or use `#[autoversion]` where LLVM handles masking well.

### Phase 4: Migrate math utilities (3 files)

These are leaf functions called by the stages.

1. `util/fast_math.rs` — fast_pow2f, fast_log2f (bitcast-heavy)
2. `util/rational_poly.rs` — polynomial evaluation (FMA chains)
3. `color/tf.rs` — transfer function evaluation

**Pattern**: `#[autoversion]` for polynomial chains. Bitcast operations need magetypes `bitcast_f32_to_i32` / `bitcast_i32_to_f32`.

### Phase 5: Migrate transforms (14 files)

The IDCT transforms are the performance-critical core. These use only load/store/arithmetic/FMA on f32, making them good `#[autoversion]` candidates.

**Files**:
1. `transforms/idct2.rs`
2. `transforms/idct4.rs`
3. `transforms/idct8.rs`
4. `transforms/idct16.rs`
5. `transforms/idct32.rs`
6. `transforms/idct2d.rs`
7. `transforms/idct_large.rs`
8. `transforms/reinterpreting_dct2.rs`
9. `transforms/reinterpreting_dct4.rs`
10. `transforms/reinterpreting_dct8.rs`
11. `transforms/reinterpreting_dct16.rs`
12. `transforms/reinterpreting_dct32.rs`
13. `transforms/reinterpreting_dct2d.rs`
14. `transforms/transform.rs` (dispatch)

**Pattern**: These work on rows of f32 with known stride. `#[autoversion]` should produce excellent code — LLVM is very good at auto-vectorizing FMA-heavy DSP loops. Test with `cargo asm` to verify.

**Risk**: The 2D transpose operations currently use architecture-specific shuffles. May need `incant!` variants for the transpose kernel.

### Phase 6: Migrate modular/group operations (3 files)

1. `frame/group.rs` — dequantization (I32SimdVec)
2. `frame/modular/transforms/rct.rs` — reversible color transform (I32SimdVec)
3. `frame/modular/transforms/squeeze.rs` — modular squeeze (I32SimdVec, U32SimdVec)

**Pattern**: These use integer SIMD. `#[autoversion]` for the scalar loops, or `i32x8<T>` / `u32x8<T>` from magetypes.

### Phase 7: Migrate save/output (1 file)

1. `render/low_memory_pipeline/save/identity.rs` — U8SimdVec, U16SimdVec, interleaving

**Pattern**: Uses `round_store_u8`, `round_store_u16`, `store_interleaved_2/3/4`. Use the `simd_compat` helpers from Phase 1.

### Phase 8: Migrate dispatch infrastructure

1. `render/simd_utils.rs` — interleave/deinterleave dispatch helpers
2. Replace `simd_function!` macro usage everywhere
3. Replace `test_all_instruction_sets!` with per-token test helpers
4. Replace `bench_all_instruction_sets!` with per-token benchmark helpers

### Phase 9: Remove jxl_simd

1. Remove `jxl_simd` from workspace members
2. Remove `jxl_simd` dependency from `jxl/Cargo.toml`
3. Remove `jxl_simd/` directory
4. Remove feature forwarding (`sse42`, `avx`, `avx512`, `neon`, `wasm128`)
5. Add direct `archmage` + `magetypes` deps to `jxl/Cargo.toml`
6. Update CI to test with `--features avx` → direct feature names

## Key Decisions

### `#[autoversion]` vs `incant!`

- **`#[autoversion]`**: Best for regular loop bodies. LLVM auto-vectorizes scalar code compiled with `target_feature(enable="avx2,fma")`. Zero manual intrinsics. Works for ~70% of our code (IDCT, convolutions, FMA chains, simple arithmetic).

- **`incant!`**: Required when auto-vectorization fails or produces suboptimal code. Needed for: transpose kernels, table lookups, complex masking patterns, interleave/deinterleave, bf16 table lookups.

**Default to `#[autoversion]`**. Only drop to `incant!` when benchmarks show it matters.

### Width-generic code

jxl_simd's `SimdDescriptor` trait made code generic over SIMD width. magetypes' `SimdTypes` trait does the same thing — `T::F32` maps to the right-width f32 vector for token `T`.

For `#[autoversion]` code, width doesn't matter — just write scalar loops.

For `incant!` code, write the function generic over `T: SimdTypes` and monomorphize per token.

### Scalar fallback

archmage's `ScalarToken` + magetypes' scalar types provide a complete fallback path. `#[autoversion]` generates a `_scalar` variant automatically. `incant!` requires an explicit `_scalar` function.

## Benchmarking Strategy

1. Before each phase, run the full decode benchmark suite and save results
2. After each phase, re-run and compare
3. Key images: `bike_web_q85.jxl` (multi-group VarDCT), `city_4k_q75.jxl` (large), `3x3_srgb_lossless.jxl` (modular)
4. If any regression >5%, investigate with `cargo asm` before proceeding
5. Commit benchmark results under `benchmarks/`

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `#[autoversion]` produces slower code than hand-written intrinsics | Medium | Medium | Benchmark each phase, fall back to `incant!` |
| Transpose kernel can't auto-vectorize | High | Low | Keep as `incant!` with per-arch variants |
| bf16 table lookup has no magetypes equivalent | High | Low | Implement via raw intrinsics in `incant!` variant |
| Build times increase (more monomorphization) | Low | Low | `#[autoversion]` may actually reduce binary size vs 6 separate backends |
| Semantic differences in rounding/precision | Low | High | Conformance tests catch this immediately |

## File Count Summary

| Phase | Files | Complexity |
|-------|-------|-----------|
| 1. simd_compat bridge | 1 new | Medium |
| 2. Simple stages | 6 | Low |
| 3. Complex stages | 9 | Medium |
| 4. Math utilities | 3 | Medium |
| 5. Transforms | 14 | Medium-High |
| 6. Modular/group | 3 | Medium |
| 7. Save/output | 1 | Medium |
| 8. Dispatch infra | 2 | Low |
| 9. Remove jxl_simd | cleanup | Low |
| **Total** | **38 + 1 new** | |
