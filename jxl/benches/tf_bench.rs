// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Benchmark comparing different linear→sRGB transfer function implementations.
//!
//! Run with: cargo bench -p zenjxl-decoder --bench tf_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use jxl_simd::{F32SimdVec, SimdDescriptor, SimdMask};
use std::hint::black_box;

// ============================================================================
// Approach 1: Current jxl-rs rational polynomial (SIMD)
// ============================================================================

#[inline(always)]
fn eval_rational_poly_simd<D: SimdDescriptor, const P: usize, const Q: usize>(
    d: D,
    x: D::F32Vec,
    p: [f32; P],
    q: [f32; Q],
) -> D::F32Vec {
    let mut yp = D::F32Vec::splat(d, p[P - 1]);
    for i in (0..P - 1).rev() {
        yp = yp.mul_add(x, D::F32Vec::splat(d, p[i]));
    }
    let mut yq = D::F32Vec::splat(d, q[Q - 1]);
    for i in (0..Q - 1).rev() {
        yq = yq.mul_add(x, D::F32Vec::splat(d, q[i]));
    }
    yp / yq
}

#[inline(always)]
fn linear_to_srgb_simd<D: SimdDescriptor>(d: D, samples: &mut [f32]) {
    #[allow(clippy::excessive_precision)]
    const P: [f32; 5] = [
        -5.135152395e-4,
        5.287254571e-3,
        3.903842876e-1,
        1.474205315,
        7.352629620e-1,
    ];
    #[allow(clippy::excessive_precision)]
    const Q: [f32; 5] = [
        1.004519624e-2,
        3.036675394e-1,
        1.340816930,
        9.258482155e-1,
        2.424867759e-2,
    ];
    for vec in samples.chunks_exact_mut(D::F32Vec::LEN) {
        let x = D::F32Vec::load(d, vec);
        let a = x.abs();
        D::F32Vec::splat(d, 0.0031308)
            .gt(a)
            .if_then_else_f32(
                a * D::F32Vec::splat(d, 12.92),
                eval_rational_poly_simd(d, a.sqrt(), P, Q),
            )
            .copysign(x)
            .store(vec);
    }
}

// f32 to u8 conversion (current approach - separate stage)
#[inline(always)]
fn f32_to_u8_simd<D: SimdDescriptor>(d: D, input: &[f32], output: &mut [u8]) {
    let zero = D::F32Vec::splat(d, 0.0);
    let one = D::F32Vec::splat(d, 1.0);
    let scale = D::F32Vec::splat(d, 255.0);
    for (in_chunk, out_chunk) in input
        .chunks_exact(D::F32Vec::LEN)
        .zip(output.chunks_exact_mut(D::F32Vec::LEN))
    {
        let val = D::F32Vec::load(d, in_chunk);
        let clamped = val.max(zero).min(one);
        let scaled = clamped * scale;
        scaled.round_store_u8(out_chunk);
    }
}

// ============================================================================
// Approach 2: fast-srgb8 LUT (104-entry, scalar)
// ============================================================================

#[allow(clippy::excessive_precision)]
const TO_SRGB8_TABLE: [u32; 104] = [
    0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d, 0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
    0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a, 0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
    0x010e0033, 0x01280033, 0x01410033, 0x015b0033, 0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
    0x01dc0067, 0x020f0067, 0x02430067, 0x02760067, 0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
    0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce, 0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
    0x06970158, 0x07420142, 0x07e30130, 0x087b0120, 0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
    0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180, 0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
    0x11070264, 0x1238023e, 0x1357021d, 0x14660201, 0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
    0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad, 0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
    0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392, 0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
    0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5, 0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
    0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d, 0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
    0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f, 0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
];

const MINV_BITS: u32 = 0x39000000; // 2^(-13)

/// fast-srgb8 approach: linear f32 → sRGB u8 directly via 104-entry LUT
#[inline]
fn fast_srgb8_f32_to_u8(f: f32) -> u8 {
    let maxv = f32::from_bits(0x3f7fffff); // 1.0 - epsilon
    let minv = f32::from_bits(MINV_BITS);
    let mut input = f;
    if !(input > minv) {
        input = minv;
    }
    if input > maxv {
        input = maxv;
    }
    let fu = input.to_bits();
    let i = ((fu - MINV_BITS) >> 20) as usize;
    let entry = TO_SRGB8_TABLE[i];
    let bias = (entry >> 16) << 9;
    let scale = entry & 0xffff;
    let t = (fu >> 12) & 0xff;
    ((bias + scale * t) >> 16) as u8
}

fn fast_srgb8_batch(input: &[f32], output: &mut [u8]) {
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = fast_srgb8_f32_to_u8(*inp);
    }
}

// ============================================================================
// Approach 3: LUT-4096 with interpolation (linear-srgb style)
// ============================================================================

// Generate the 4096-entry encode table at compile time
const fn generate_encode_table_4096() -> [f32; 4096] {
    let mut table = [0.0f32; 4096];
    let mut i = 0;
    while i < 4096 {
        let linear = i as f64 / 4095.0;
        let srgb = if linear <= 0.0031308 {
            linear * 12.92
        } else {
            // Use a manual pow approximation since powf is not const
            // For the table, we'll compute at runtime instead
            0.0 // placeholder
        };
        table[i] = srgb as f32;
        i += 1;
    }
    table
}

// Runtime-initialized LUT for linear→sRGB f32
fn make_encode_table_4096() -> Vec<f32> {
    (0..4096)
        .map(|i| {
            let linear = i as f64 / 4095.0;
            let srgb = if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            srgb as f32
        })
        .collect()
}

/// LUT-4096 with linear interpolation: linear f32 → sRGB f32
#[inline]
fn lut_linear_to_srgb_f32(linear: f32, table: &[f32; 4096]) -> f32 {
    let x = linear.clamp(0.0, 1.0);
    let scaled = x * 4095.0;
    let lower = scaled as usize;
    let upper = (lower + 1).min(4095);
    let frac = scaled - lower as f32;
    table[lower] + frac * (table[upper] - table[lower])
}

fn lut_4096_batch(input: &[f32], output: &mut [u8], table: &[f32; 4096]) {
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let srgb = lut_linear_to_srgb_f32(*inp, table);
        *out = (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

// ============================================================================
// Approach 4: LUT-65536 direct (linear f32 → sRGB u8, no interpolation)
// ============================================================================

fn make_direct_lut_65536() -> Vec<u8> {
    (0..65536u32)
        .map(|i| {
            let linear = i as f64 / 65535.0;
            let srgb = if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Direct 64K LUT: linear f32 → sRGB u8 (quantize to 16-bit index)
#[inline]
fn direct_lut_linear_to_srgb_u8(linear: f32, table: &[u8; 65536]) -> u8 {
    let idx = (linear.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize;
    table[idx]
}

fn direct_lut_65k_batch(input: &[f32], output: &mut [u8], table: &[u8; 65536]) {
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = direct_lut_linear_to_srgb_u8(*inp, table);
    }
}

// ============================================================================
// Approach 5: Fused rational polynomial + u8 (single SIMD pass)
// ============================================================================

jxl_simd::simd_function!(
    fused_linear_to_srgb_u8_dispatch,
    d: D,
    fn fused_linear_to_srgb_u8(input: &[f32], output: &mut [u8]) {
        #[allow(clippy::excessive_precision)]
        const P: [f32; 5] = [
            -5.135152395e-4,
            5.287254571e-3,
            3.903842876e-1,
            1.474205315,
            7.352629620e-1,
        ];
        #[allow(clippy::excessive_precision)]
        const Q: [f32; 5] = [
            1.004519624e-2,
            3.036675394e-1,
            1.340816930,
            9.258482155e-1,
            2.424867759e-2,
        ];

        let zero = D::F32Vec::splat(d, 0.0);
        let one = D::F32Vec::splat(d, 1.0);
        let scale = D::F32Vec::splat(d, 255.0);
        let threshold = D::F32Vec::splat(d, 0.0031308);
        let linear_scale = D::F32Vec::splat(d, 12.92);

        for (in_chunk, out_chunk) in input
            .chunks_exact(D::F32Vec::LEN)
            .zip(output.chunks_exact_mut(D::F32Vec::LEN))
        {
            let x = D::F32Vec::load(d, in_chunk);
            // Clamp to [0, 1] first (skip sign handling for u8 output)
            let a = x.max(zero).min(one);

            // Apply sRGB TF
            let srgb = threshold
                .gt(a)
                .if_then_else_f32(
                    a * linear_scale,
                    eval_rational_poly_simd(d, a.sqrt(), P, Q),
                );

            // Scale to [0, 255] and convert to u8
            let scaled = srgb * scale;
            scaled.round_store_u8(out_chunk);
        }
    }
);

// ============================================================================
// Approach 6: Scalar powf (reference/naive)
// ============================================================================

fn naive_linear_to_srgb_u8(input: &[f32], output: &mut [u8]) {
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let a = inp.clamp(0.0, 1.0);
        let srgb = if a <= 0.0031308 {
            a * 12.92
        } else {
            a.powf(1.0 / 2.4).mul_add(1.055, -0.055)
        };
        *out = (srgb * 255.0 + 0.5) as u8;
    }
}

// ============================================================================
// Test data generation
// ============================================================================

fn generate_test_data(n: usize) -> Vec<f32> {
    // Generate representative linear RGB values
    // Most pixel values in decoded images cluster in [0, 1] with some near-zero and near-one
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        // Mix of linear ramp and gamma-adjusted distribution
        let v = if i % 3 == 0 {
            t * t // quadratic (more darks)
        } else if i % 3 == 1 {
            t // linear ramp
        } else {
            t.sqrt() // sqrt (more lights)
        };
        data.push(v);
    }
    data
}

// ============================================================================
// Accuracy test
// ============================================================================

fn check_accuracy(name: &str, input: &[f32], output: &[u8], reference: &[u8]) {
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    let mut count_wrong = 0usize;
    for (i, (&actual, &expected)) in output.iter().zip(reference.iter()).enumerate() {
        let diff = actual.abs_diff(expected);
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff as u64;
        if diff > 0 {
            count_wrong += 1;
        }
        if diff > 1 && i < 10 {
            eprintln!(
                "  {name}: input={:.6} expected={expected} got={actual} diff={diff}",
                input[i]
            );
        }
    }
    let n = output.len();
    let avg_diff = sum_diff as f64 / n as f64;
    eprintln!(
        "{name}: max_u8_err={max_diff}, avg_err={avg_diff:.4}, wrong={count_wrong}/{n} ({:.1}%)",
        count_wrong as f64 / n as f64 * 100.0
    );
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_tf(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536, 262144];

    for &size in &sizes {
        let mut group = c.benchmark_group(format!("linear_to_srgb_u8/{size}"));
        let input = generate_test_data(size);
        let mut output = vec![0u8; size];

        // Reference output using naive powf
        let mut reference = vec![0u8; size];
        naive_linear_to_srgb_u8(&input, &mut reference);

        // 1. Current approach: rational poly SIMD (two stages)
        // Note: The real pipeline applies TF in-place then reads back for u8 conversion.
        // We benchmark both stages but pre-copy to avoid measuring memcpy overhead.
        group.bench_function("rational_poly_simd_2stage", |b| {
            let mut srgb_f32 = input.clone();
            b.iter(|| {
                // In the real pipeline, the data is already in the buffer; we simulate
                // by resetting each iteration since linear_to_srgb is in-place.
                srgb_f32.copy_from_slice(&input);
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if let Some(d) = jxl_simd::AvxDescriptor::new() {
                    linear_to_srgb_simd(d, &mut srgb_f32);
                    f32_to_u8_simd(d, &srgb_f32, &mut output);
                    return;
                }
                let d = jxl_simd::ScalarDescriptor::new().unwrap();
                linear_to_srgb_simd(d, &mut srgb_f32);
                f32_to_u8_simd(d, &srgb_f32, &mut output);
            });
        });

        // 1b. Just the rational poly SIMD TF (no u8 conversion, no copy)
        group.bench_function("rational_poly_simd_tf_only", |b| {
            let mut srgb_f32 = input.clone();
            linear_to_srgb_simd(jxl_simd::ScalarDescriptor::new().unwrap(), &mut srgb_f32);
            // Already in sRGB, so re-applying TF is wrong for accuracy but fine for perf measurement
            b.iter(|| {
                // We need fresh linear data each time since TF is in-place
                let src = black_box(&input);
                srgb_f32.copy_from_slice(src);
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if let Some(d) = jxl_simd::AvxDescriptor::new() {
                    linear_to_srgb_simd(d, black_box(&mut srgb_f32));
                    return;
                }
                linear_to_srgb_simd(
                    jxl_simd::ScalarDescriptor::new().unwrap(),
                    black_box(&mut srgb_f32),
                );
            });
        });

        // Check accuracy of approach 1
        {
            let mut srgb_f32 = input.clone();
            let d = jxl_simd::ScalarDescriptor::new().unwrap();
            linear_to_srgb_simd(d, &mut srgb_f32);
            f32_to_u8_simd(d, &srgb_f32, &mut output);
            if size == 65536 {
                check_accuracy("rational_poly_simd_2stage", &input, &output, &reference);
            }
        }

        // 2. fast-srgb8 LUT (scalar)
        group.bench_function("fast_srgb8_lut104", |b| {
            b.iter(|| {
                fast_srgb8_batch(black_box(&input), &mut output);
            });
        });
        if size == 65536 {
            fast_srgb8_batch(&input, &mut output);
            check_accuracy("fast_srgb8_lut104", &input, &output, &reference);
        }

        // 3. LUT-4096 with interpolation
        let table_4096_vec = make_encode_table_4096();
        let table_4096: &[f32; 4096] = table_4096_vec.as_slice().try_into().unwrap();
        group.bench_function("lut4096_interp", |b| {
            b.iter(|| {
                lut_4096_batch(black_box(&input), &mut output, table_4096);
            });
        });
        if size == 65536 {
            lut_4096_batch(&input, &mut output, table_4096);
            check_accuracy("lut4096_interp", &input, &output, &reference);
        }

        // 4. Direct 64K LUT
        let table_65k_vec = make_direct_lut_65536();
        let table_65k: &[u8; 65536] = table_65k_vec.as_slice().try_into().unwrap();
        group.bench_function("direct_lut_65k", |b| {
            b.iter(|| {
                direct_lut_65k_batch(black_box(&input), &mut output, table_65k);
            });
        });
        if size == 65536 {
            direct_lut_65k_batch(&input, &mut output, table_65k);
            check_accuracy("direct_lut_65k", &input, &output, &reference);
        }

        // 5. Fused rational poly + u8 (single SIMD pass)
        group.bench_function("fused_poly_u8_simd", |b| {
            b.iter(|| {
                fused_linear_to_srgb_u8_dispatch(black_box(&input), &mut output);
            });
        });
        if size == 65536 {
            fused_linear_to_srgb_u8_dispatch(&input, &mut output);
            check_accuracy("fused_poly_u8_simd", &input, &output, &reference);
        }

        // 6. Naive powf (reference)
        group.bench_function("naive_powf", |b| {
            b.iter(|| {
                naive_linear_to_srgb_u8(black_box(&input), &mut output);
            });
        });

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(100);
    targets = bench_tf
);
criterion_main!(benches);
