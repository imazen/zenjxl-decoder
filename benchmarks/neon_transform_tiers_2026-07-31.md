# zenjxl-decoder: first SIMD-tier measurement of the inverse transforms (aarch64)

Apple M4 Pro, aarch64. zenjxl-decoder @ `216094f`.
`cargo bench -p zenjxl-decoder --features __bench_kernels --bench kernel_tiers`
Wall time 134.0s, 30 rounds per arm.

**Verdict: all 13 kernels healthy. No regressions. Nothing needed changing.**

## Relation to `neon_tier_isolation_2026-07-28.meta`

That file measured this crate END-TO-END on ARM and established the split:
VarDCT gains **1.71-1.83x** from NEON, modular gains 1.00-1.06x, and the
modular flatness was audited to a structural cause (serial ANS/Huffman + a
horizontally-chained modular predictor) rather than missing kernels.

This file measures one level down: the 13 individual inverse-DCT kernels that
make up the VarDCT win. Those two questions are different. An end-to-end 1.83x
is entirely compatible with one of the 13 shapes being slower than its own
scalar fallback and dragging the rest — the aggregate cannot separate them.
This bench separates them, and the answer is that none of them is.

## Why this was measured

`zenjxl-decoder-simd` exists for nothing but SIMD — 159 `#[arcane]` sites —
and had **no benchmark of any kind**. `decode_bench` and `tf_bench` measure
whole decodes, which structurally cannot reveal a single kernel that is slower
than its own scalar fallback: one bad transform is a rounding error inside a
full decode. That is exactly how eight such regressions hid in other zen crates
during the 2026-07 aarch64 sweep (zenquant 0.58x, zenav1-svt inverse transforms
0.59x, zenwebp idct4x4 0.74x, and five more) — every one invisible to the
crate's existing end-to-end benchmarks.

## Method

This bench does NOT toggle `dangerously_disable_token_process_wide`, unlike the
other zen tier benches. It doesn't need to: every transform is generic over
`D: SimdDescriptor`, so instantiating the same source with `NeonDescriptor` and
`ScalarDescriptor` compiles two real tiers. That is a stronger comparison than
token toggling — no dispatch machinery inside the timed region at all — and it
needs no `testable_dispatch` feature.

Buffers are rebuilt in `with_input` (untimed). These transforms are in-place, so
a clone inside the timed closure would charge every arm for an allocation; that
mistake manufactured a false 0.92x "regression" in zenresize on 2026-07-30.

## Results

| kernel | neon ns | scalar ns | ratio |
|---|---|---|---|
| idct2d_2_2 | 17.8 | 17.9 | 1.01x |
| idct2d_4_4 | 29.4 | 34.2 | 1.16x |
| idct2d_4_8 | 23.2 | 36.3 | 1.56x |
| idct2d_8_4 | 21.3 | 41.0 | 1.92x |
| idct2d_8_8 | 38.1 | 77.4 | 2.03x |
| idct2d_8_16 | 58.8 | 74.7 | 1.27x |
| idct2d_8_32 | 91.5 | 188.0 | 2.05x |
| idct2d_16_8 | 44.5 | 93.2 | 2.09x |
| idct2d_16_16 | 82.4 | 180.2 | 2.19x |
| idct2d_16_32 | 208.0 | 378.0 | 1.82x |
| idct2d_32_8 | 86.6 | 217.7 | 2.51x |
| idct2d_32_16 | 185.0 | 440.0 | 2.38x |
| idct2d_32_32 | 418.0 | 972.0 | 2.33x |

**`idct2d_2_2`'s 1.01x is correct, not a finding.** It ignores its descriptor
and forces `ScalarDescriptor` internally (`idct2d.rs:345`) because 2x2 is too
small to vectorize, so both arms run identical code by construction. Predicted
before the run; the measurement confirms it.

Remember NEON is BASELINE on aarch64 — the "scalar" arm is autovectorized by
LLVM too. These 1.2x-2.5x figures are the hand-written vector path beating an
already-vectorized portable one, which is a real margin, not the usual
SIMD-vs-nothing comparison.

## Secondary finding: the `#[inline(always)]` hazard is not firing

The 159 primitives (`splat`, `load`, `add`, ...) are `#[inline(always)]`
single-intrinsic wrappers that must fuse into the caller's `#[arcane]` region.
`#[inline(always)]` is a *hint* LLVM drops when a caller exceeds its cost
threshold — zensim measured a **5.3x whole-extraction regression** when exactly
that happened (`feature_v2.rs`, the POOL_SIMD note: every vector operator became
a call to a non-inlined `core::arch` shim).

Checked by symbol audit on the linked `zenjxl-decoder-cli` binary (3920 text
symbols): **zero SIMD primitives survived as standalone symbols.** The only 12
`zenjxl_decoder_simd` symbols are `SimdDescriptor::call` instantiations — the
`#[arcane]` entry points, which are supposed to be real functions — plus the
IDCT bodies inside them. That is the healthy shape.

Instrument caveat worth recording: the same audit run against
`libzenjxl_decoder.rlib` reported 0 symbols *total*, i.e. a clean-looking null
result produced by `nm` not reading the archive rather than by the code being
clean. Always confirm the total symbol count is nonzero before believing a
"nothing found" symbol audit.

## What this does NOT cover

- **x86_64.** Only the aarch64 tier was measured; this host has no x86 CPU. The
  non-inlining hazard matters *more* there (NEON is baseline on aarch64, so a
  primitive that escapes its target-feature region still compiles to NEON; an
  AVX2 primitive that escapes does not stay AVX2). The sse42/avx/avx512 backends
  are unmeasured.
- The other transform families (`reinterpreting_dct2d`, the `idct_large`
  64/128/256 paths — the latter two are `#[cfg(test)]`-gated today).
