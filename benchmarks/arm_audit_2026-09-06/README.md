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


## Modular decode profile

A five-second native sample of 2000 repeated Green Queen modular decodes captured 3697 main-thread samples. The collapsed top-of-stack table attributes 3558 samples to `decode_modular_channel_impl<WpOnlyLookupConfig420>`; the hot interior call is `channel.rs:155`. This combines weighted prediction, entropy decoding and predictor-error updates after inlining, so the sample does not separate their individual costs.

The complete run decoded 515964000 pixels in 32.38 s wall / 31.63 s user time, with maximum RSS 18055168 bytes from `time -l`. It is one 438x589 image repeated, not a size-scaling result. The profiler’s own footprint display is not used as the memory measurement. Captures and provenance are in `modular_profile.pointer.md`.

The weighted predictor keeps four errors per position in a flat `Vec<u32>` and forms a checked four-element slice on each lookup. The next codegen experiment is to store `Vec<[u32; 4]>` directly, preserving the layout and arithmetic while reducing each lookup to a position bound check. Adjacent pixel decoding depends on prior decoded symbols and predictor state, so a row-wide SIMD conversion is not assumed from this profile.

## Weighted-error array experiment (rejected)

Experiment `a8a46e2e` replaced the private flattened `Vec<u32>` with `Vec<[u32; 4]>`, preserving the position-major bytes and all arithmetic. The two position accessors became direct fixed-array accesses. LLVM's hot weighted specialization changed from 2816 to 2850 assembly lines; two bounds-panic call sites remained and slice-failure call sites went from two to one. Fewer range conversions did not produce a timing win, so the experiment is preserved locally but excluded from main.

| Modular fixture | Original mean | Array experiment mean |
|---|---:|---:|
| Green Queen (normal decode group) | 16.30 ms | 16.65 ms |
| issue648 palette | 102.31 ms | 105.58 ms |
| grayscale patches | 6.86 ms | 7.09 ms |
| small grayscale patches | 690.18 us | 705.47 us |
| small grayscale patches with ICC | 678.10 us | 686.86 us |
| gray alpha | 181.00 us | 183.62 us |
| 3x3 RGB | 14.54 us | 13.29 us |
| 3x3 RGBA | 14.45 us | 14.58 us |

These are separate builds, not an interleaved before/after confidence interval. Tiny cases are noisy. Green Queen's independently paired NEON/scalar arms remain a tie in each build (17.48/17.53 ms original, 17.74/17.80 ms experiment). This rejects the array hypothesis on these samples; it does not establish that the modular loop cannot be improved.

Full [before](jxl-modular-array-before.log) and [after](jxl-modular-array-after.log) logs use the old benchmark grouping. **Ignore their `modular` throughput and cross-fixture percentage columns:** the last image's pixel count overwrote the whole group's throughput, and different fixtures are not comparable implementations. The benchmark now creates a separate group for each image so pixel counts remain attached to the correct workload.

Experiment correctness: [37 modular tests](jxl-array-modular-corpus-tests.log) and [two cross-tier tests](jxl-array-cross-tier-tests.log) passed. The first corpus invocation failed because its lookup omitted the local directory; the rerun used `CODEC_CORPUS_PATH=/Users/lilith/work/zen/codec-corpus`. The feature helper only prints decode errors, so its captured output was also inspected: all selected features decoded successfully. Golden weighted-predictor and palette hashes have exact assertions. The existing cross-tier float checks retain their documented tolerance; this is not an exact float-parity claim.

Reproduce the benchmark with `just arm-modular-macos`. Rust 1.98/LLVM 22, M4 Pro, nice -n19, four build/Rayon/OMP threads, no target-cpu=native, same flags for both builds.

The corrected per-fixture grouping was run on both 3x3 fixtures ([log](jxl-decode-throughput-group-check.log)); each now reports its own nine-pixel throughput and no comparison to another image. Strict Clippy for `decode_bench` passed ([log](jxl-decode-bench-clippy.log)).
