//! Per-kernel SIMD-tier comparison for the JPEG XL inverse transforms.
//!
//! Before this bench the crate had NO benchmark of its SIMD layer at all,
//! despite `zenjxl-decoder-simd` existing for nothing else — 159 `#[arcane]`
//! sites and zero measurements. `decode_bench`/`tf_bench` measure whole
//! decodes, which cannot reveal an individual kernel that is SLOWER than its
//! own scalar fallback: one bad transform is a rounding error inside a full
//! decode, and that is exactly how eight such regressions hid in other zen
//! crates during the 2026-07 aarch64 sweep.
//!
//! **How the tiers are selected here.** Unlike the other zen tier benches,
//! this one does NOT toggle `dangerously_disable_token_process_wide`. It does
//! not need to: every transform is generic over `D: SimdDescriptor`, so
//! instantiating the SAME source with `NeonDescriptor` and `ScalarDescriptor`
//! compiles two real tiers and compares them directly. That is a stronger
//! comparison than token toggling (no dispatch machinery in the timed region
//! at all) and it needs no `testable_dispatch` feature.
//!
//! **Reading the numbers — aarch64 only.** NEON is BASELINE on aarch64, so
//! the "scalar" arm is autovectorized by LLVM too. A ratio near 1.00x means
//! both arms compiled to equivalent work, NOT that SIMD is missing. Below
//! 1.00x is the finding: the hand-written vector path losing to the portable
//! one.
//!
//! **`idct2d_2_2` is expected to read ~1.00x.** It ignores its descriptor and
//! forces `ScalarDescriptor` internally (idct2d.rs) — 2x2 is too small to
//! vectorize. Both arms run identical code by construction. That is a
//! deliberate design decision, not a regression.
//!
//! Run: `cargo bench -p zenjxl-decoder --features __bench_kernels --bench kernel_tiers`

// `NeonDescriptor` only exists on aarch64. The `[[bench]]` is registered for
// every target (Cargo cannot gate a bench by architecture), so the body is
// gated here and other architectures get an empty `main` instead of a
// compile error under `cargo clippy --all-targets --all-features`.
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use zenbench::prelude::*;
    use zenjxl_decoder::__bench_kernels as k;

    use jxl_simd::SimdDescriptor;

    /// Deterministic coefficient block. Values are in the range a real dequantized
    /// DCT block occupies — a flat or all-zero buffer would let both arms skip
    /// work and report a meaningless ratio.
    fn coeffs(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// One `idct2d_{w}_{h}` variant, both tiers, on its own buffer.
    ///
    /// The buffer is rebuilt in `with_input` (untimed) rather than cloned inside
    /// the timed closure: these transforms are in-place, so a timed clone would
    /// charge every arm for an allocation. That mistake produced a false 0.92x
    /// "regression" in zenresize on 2026-07-30 and is worth not repeating.
    macro_rules! idct_bench {
        ($suite:expr, $name:literal, $fn:ident, $len:expr) => {
            $suite.compare($name, |g| {
                g.throughput(Throughput::Elements($len as u64));
                g.bench("neon", |b| {
                    let d = jxl_simd::NeonDescriptor::new().expect("aarch64 baseline");
                    b.with_input(|| coeffs($len, 7)).run(move |mut buf| {
                        k::$fn(d, &mut buf);
                        buf
                    })
                });
                g.bench("scalar", |b| {
                    let d = jxl_simd::ScalarDescriptor::new().expect("always available");
                    b.with_input(|| coeffs($len, 7)).run(move |mut buf| {
                        k::$fn(d, &mut buf);
                        buf
                    })
                });
            });
        };
    }

    pub fn bench_kernels(suite: &mut Suite) {
        if jxl_simd::NeonDescriptor::new().is_none() {
            eprintln!("[kernel_tiers] NEON descriptor unavailable — aarch64 only. Skipping.");
            return;
        }
        eprintln!("[kernel_tiers] comparing NeonDescriptor vs ScalarDescriptor per kernel");

        // All 13 shipped 2D inverse DCT block shapes. JPEG XL picks among these
        // per variable-block, so every one of them is a production path — there is
        // no "rare" shape to skip.
        idct_bench!(suite, "idct2d_2_2", idct2d_2_2, 2 * 2);
        idct_bench!(suite, "idct2d_4_4", idct2d_4_4, 4 * 4);
        idct_bench!(suite, "idct2d_4_8", idct2d_4_8, 4 * 8);
        idct_bench!(suite, "idct2d_8_4", idct2d_8_4, 8 * 4);
        idct_bench!(suite, "idct2d_8_8", idct2d_8_8, 8 * 8);
        idct_bench!(suite, "idct2d_8_16", idct2d_8_16, 8 * 16);
        idct_bench!(suite, "idct2d_8_32", idct2d_8_32, 8 * 32);
        idct_bench!(suite, "idct2d_16_8", idct2d_16_8, 16 * 8);
        idct_bench!(suite, "idct2d_16_16", idct2d_16_16, 16 * 16);
        idct_bench!(suite, "idct2d_16_32", idct2d_16_32, 16 * 32);
        idct_bench!(suite, "idct2d_32_8", idct2d_32_8, 32 * 8);
        idct_bench!(suite, "idct2d_32_16", idct2d_32_16, 32 * 16);
        idct_bench!(suite, "idct2d_32_32", idct2d_32_32, 32 * 32);
    }
}

#[cfg(target_arch = "aarch64")]
zenbench::main!(aarch64::bench_kernels);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!(
        "[kernel_tiers] aarch64-only bench (NeonDescriptor vs ScalarDescriptor); nothing to run here."
    );
}
