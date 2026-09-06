# JXL decoder ARM audit, 2026-09-06

Coverage: 13 inverse-transform shapes and four whole-image fixtures. Modular
decoding has no measured NEON benefit in these two fixtures; its serial decode
path still needs a call-stack profile before choosing an optimization. Other
fixtures and decoder kernels are not covered by this measurement.

Host: Apple M4 Pro, macOS, Rust 1.98.0 / LLVM 22, four build/Rayon/OMP threads,
`nice -n 19`, runtime dispatch without `target-cpu=native`. Decoder source is
`2187821e`; the whole-image paired benchmark addition is `f2c2a61b`.

All 12 vector-capable inverse-transform shapes favored NEON over their scalar
descriptor. The remaining 2×2 shape explicitly uses `ScalarDescriptor` in
`src/transforms/idct2d.rs`, so both arms execute scalar (19.5 versus 19.6 ns).
Examples: 8×8 51.2 versus 93.8 ns; 16×16 147 versus 324 ns; 32×32 970 versus
2259 ns. Full results and confidence intervals: [kernel log](jxl-decoder-tiers.log).

| Whole-image fixture | NEON mean | Forced scalar mean |
|---|---:|---:|
| cafe_web_q80.jxl | 40.26 ms | 55.91 ms |
| portrait_4k_q75.jxl | 456.23 ms | 1094 ms |
| green_queen_modular_e3.jxl | 17.17 ms | 17.23 ms |
| gray_alpha_lossless.jxl | 279.55 µs | 282.33 µs |

The portrait run has only seven rounds and a drift warning. Both modular
comparisons have confidence intervals crossing zero. Fixture names are
preserved verbatim; they are not a claim about other image dimensions or modes.
See [whole-image log](jxl-decoder-full-tiers.log).

The [assembly excerpt](jxl-inline-transform.asm) shows vector arithmetic,
including `fmla.4s`, inlined into zenbench's measured transform body. Buffer
allocation/free and timer calls surround that body; this is not evidence of
an allocation-free whole decoder. No out-of-line SIMD primitive calls occur
in the timed portion of this excerpt.

The existing `cross_tier_determinism` integration gate passed both tests;
see [validation log](jxl-decoder-parity.log). That gate's existing contract
requires exact integer-fixture output and allows its pre-existing bounded
floating-point tier differences. It is not universal pixel-identity evidence.
Benchmark clippy with `-D warnings` and scoped formatting also passed.

Reproduce with `just arm-decode-tiers-macos` and
`just arm-kernel-tiers-macos`. Results describe these fixtures and kernels,
not a codec-wide speedup or a completed modular optimization.
