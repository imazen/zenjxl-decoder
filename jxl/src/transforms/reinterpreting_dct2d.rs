// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;

/// Transpose an NxM matrix stored in row-major order into MxN, in place via temp buffer.
#[inline(always)]
fn transpose<const N: usize, const M: usize>(data: &mut [f32]) {
    debug_assert!(data.len() >= N * M);
    let mut tmp = vec![0.0f32; N * M];
    for i in 0..N {
        for j in 0..M {
            tmp[j * N + i] = data[i * M + j];
        }
    }
    data[..N * M].copy_from_slice(&tmp);
}

/// Transpose a square NxN matrix in-place.
#[inline(always)]
fn transpose_square<const N: usize>(data: &mut [f32]) {
    debug_assert!(data.len() >= N * N);
    for i in 0..N {
        for j in i + 1..N {
            data.swap(i * N + j, j * N + i);
        }
    }
}

/// Scatter NxM data into output buffer with stride M*8.
#[inline(always)]
fn scatter<const N: usize, const M: usize>(data: &[f32], output: &mut [f32]) {
    let out_stride = M * 8;
    for y in 0..N {
        for x in 0..M {
            output[y * out_stride + x] = data[y * M + x];
        }
    }
}

// ===== 1D-only cases =====

#[inline(always)]
pub fn reinterpreting_dct2d_1_2(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 2);
    apply_reinterpreting_dct_2(data, 0, 1);
    scatter::<1, 2>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_2_1(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 2);
    apply_reinterpreting_dct_2(data, 0, 1);
    scatter::<1, 2>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_1_4(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 4);
    apply_reinterpreting_dct_4(data, 0, 1);
    scatter::<1, 4>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_1(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 4);
    apply_reinterpreting_dct_4(data, 0, 1);
    scatter::<1, 4>(data, output);
}

// ===== Square 2D cases =====

#[inline(always)]
pub fn reinterpreting_dct2d_2_2(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 4);
    // Column-wise 2-point reinterpreting DCT
    for col in 0..2 { apply_reinterpreting_dct_2(data, col, 2); }
    transpose_square::<2>(data);
    for col in 0..2 { apply_reinterpreting_dct_2(data, col, 2); }
    scatter::<2, 2>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_4(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 16);
    for col in 0..4 { apply_reinterpreting_dct_4(data, col, 4); }
    transpose_square::<4>(data);
    for col in 0..4 { apply_reinterpreting_dct_4(data, col, 4); }
    scatter::<4, 4>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_8_8(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 64);
    for col in 0..8 { apply_reinterpreting_dct_8(data, col, 8); }
    transpose_square::<8>(data);
    for col in 0..8 { apply_reinterpreting_dct_8(data, col, 8); }
    scatter::<8, 8>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_16_16(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 256);
    for col in 0..16 { apply_reinterpreting_dct_16(data, col, 16); }
    transpose_square::<16>(data);
    for col in 0..16 { apply_reinterpreting_dct_16(data, col, 16); }
    scatter::<16, 16>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_32_32(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 1024);
    for col in 0..32 { apply_reinterpreting_dct_32(data, col, 32); }
    transpose_square::<32>(data);
    for col in 0..32 { apply_reinterpreting_dct_32(data, col, 32); }
    scatter::<32, 32>(data, output);
}

// ===== Rectangular 2D cases (wide: cols > rows) =====

#[inline(always)]
pub fn reinterpreting_dct2d_2_4(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 8);
    // Column-wise 2-point DCT on 2×4 matrix
    for col in 0..4 { apply_reinterpreting_dct_2(data, col, 4); }
    transpose::<2, 4>(data);
    // Column-wise 4-point DCT on 4×2 matrix
    for col in 0..2 { apply_reinterpreting_dct_4(data, col, 2); }
    transpose::<4, 2>(data);
    scatter::<2, 4>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_4_8(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 32);
    for col in 0..8 { apply_reinterpreting_dct_4(data, col, 8); }
    transpose::<4, 8>(data);
    for col in 0..4 { apply_reinterpreting_dct_8(data, col, 4); }
    transpose::<8, 4>(data);
    scatter::<4, 8>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_8_16(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 128);
    for col in 0..16 { apply_reinterpreting_dct_8(data, col, 16); }
    transpose::<8, 16>(data);
    for col in 0..8 { apply_reinterpreting_dct_16(data, col, 8); }
    transpose::<16, 8>(data);
    scatter::<8, 16>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_16_32(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 512);
    for col in 0..32 { apply_reinterpreting_dct_16(data, col, 32); }
    transpose::<16, 32>(data);
    for col in 0..16 { apply_reinterpreting_dct_32(data, col, 16); }
    transpose::<32, 16>(data);
    scatter::<16, 32>(data, output);
}

// ===== Rectangular 2D cases (tall: rows > cols) =====
// For tall matrices, output is in C×R (transposed) layout.
// Algorithm: R-point column DCT → transpose R×C→C×R → C-point column DCT → scatter C×R.

#[inline(always)]
pub fn reinterpreting_dct2d_4_2(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 8);
    for col in 0..2 { apply_reinterpreting_dct_4(data, col, 2); }
    transpose::<4, 2>(data);
    for col in 0..4 { apply_reinterpreting_dct_2(data, col, 4); }
    // Data is now 2×4 (C×R). Scatter in that layout.
    scatter::<2, 4>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_8_4(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 32);
    for col in 0..4 { apply_reinterpreting_dct_8(data, col, 4); }
    transpose::<8, 4>(data);
    for col in 0..8 { apply_reinterpreting_dct_4(data, col, 8); }
    scatter::<4, 8>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_16_8(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 128);
    for col in 0..8 { apply_reinterpreting_dct_16(data, col, 8); }
    transpose::<16, 8>(data);
    for col in 0..16 { apply_reinterpreting_dct_8(data, col, 16); }
    scatter::<8, 16>(data, output);
}

#[inline(always)]
pub fn reinterpreting_dct2d_32_16(data: &mut [f32], output: &mut [f32]) {
    assert_eq!(data.len(), 512);
    for col in 0..16 { apply_reinterpreting_dct_32(data, col, 16); }
    transpose::<32, 16>(data);
    for col in 0..32 { apply_reinterpreting_dct_16(data, col, 32); }
    scatter::<16, 32>(data, output);
}
