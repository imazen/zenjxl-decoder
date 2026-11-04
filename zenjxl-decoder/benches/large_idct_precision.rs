//! Paired large-IDCT accuracy-repair cost; input setup is outside timing.
use jxl_simd::SimdDescriptor;
use zenbench::prelude::*;
use zenjxl_decoder::__bench_kernels as candidate;

#[path = "../src/transforms/idct32.rs"]
mod idct32;

#[allow(dead_code, clippy::excessive_precision)]
mod baseline {
    use super::idct32::idct_32;
    mod large {
        include!("support/large_idct_f32_baseline.rs");
    }
    pub use large::*;
}

fn coefficients(size: usize) -> Vec<f32> {
    let mut seed = 7u32;
    (0..size)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

fn measure<D: SimdDescriptor + 'static>(suite: &mut Suite, d: D, tier: &str) {
    macro_rules! compare {
        ($kernel:ident, $size:expr) => {
            suite.compare(format!("{tier}/{}", stringify!($kernel)), move |group| {
                group.throughput(Throughput::Elements($size as u64));
                group.bench("f32-baseline", move |b| {
                    b.with_input(|| coefficients($size)).run(move |mut block| {
                        baseline::$kernel(d, &mut block);
                        block
                    })
                });
                group.bench("candidate", move |b| {
                    b.with_input(|| coefficients($size)).run(move |mut block| {
                        candidate::$kernel(d, &mut block);
                        block
                    })
                });
            });
        };
    }
    compare!(idct2d_64_64, 64 * 64);
    compare!(idct2d_128_128, 128 * 128);
    compare!(idct2d_256_256, 256 * 256);
}

fn bench_large(suite: &mut Suite) {
    measure(suite, jxl_simd::ScalarDescriptor {}, "scalar");
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if let Some(d) = jxl_simd::AvxDescriptor::new() {
        measure(suite, d, "avx2");
    }
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    if let Some(d) = jxl_simd::NeonDescriptor::new() {
        measure(suite, d, "neon");
    }
}
zenbench::main!(bench_large);
