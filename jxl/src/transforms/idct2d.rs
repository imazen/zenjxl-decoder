// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;

/// Transpose an NxM matrix stored in row-major order into an MxN temp buffer, then copy back.
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

/// Apply a 1D IDCT of size N to all columns of an N×M matrix.
/// `apply_fn` is called for each column: apply_fn(data, col, stride=M).
#[inline(always)]
fn idct_columns<const M: usize>(data: &mut [f32], apply_fn: fn(&mut [f32], usize, usize)) {
    for col in 0..M {
        apply_fn(data, col, M);
    }
}

// ===== Square 2D IDCTs =====

#[inline(always)]
pub fn idct2d_2_2(data: &mut [f32]) {
    assert_eq!(data.len(), 4);
    idct_columns::<2>(data, apply_idct_2);
    transpose_square::<2>(data);
    idct_columns::<2>(data, apply_idct_2);
}

#[inline(always)]
pub fn idct2d_4_4(data: &mut [f32]) {
    assert_eq!(data.len(), 16);
    idct_columns::<4>(data, apply_idct_4);
    transpose_square::<4>(data);
    idct_columns::<4>(data, apply_idct_4);
}

#[inline(always)]
pub fn idct2d_8_8(data: &mut [f32]) {
    assert_eq!(data.len(), 64);
    idct_columns::<8>(data, apply_idct_8);
    transpose_square::<8>(data);
    idct_columns::<8>(data, apply_idct_8);
}

#[inline(always)]
pub fn idct2d_16_16(data: &mut [f32]) {
    assert_eq!(data.len(), 256);
    idct_columns::<16>(data, apply_idct_16);
    transpose_square::<16>(data);
    idct_columns::<16>(data, apply_idct_16);
}

#[inline(always)]
pub fn idct2d_32_32(data: &mut [f32]) {
    assert_eq!(data.len(), 1024);
    idct_columns::<32>(data, apply_idct_32);
    transpose_square::<32>(data);
    idct_columns::<32>(data, apply_idct_32);
}

// ===== Rectangular 2D IDCTs (rows < cols → "wide") =====

#[inline(always)]
pub fn idct2d_4_8(data: &mut [f32]) {
    assert_eq!(data.len(), 32);
    // 4-point IDCT on 4 rows × 8 cols: column-wise (stride=8)
    for col in 0..8 { apply_idct_4(data, col, 8); }
    // Transpose 4×8 → 8×4
    transpose::<4, 8>(data);
    // 8-point IDCT on 8 rows × 4 cols: column-wise (stride=4)
    for col in 0..4 { apply_idct_8(data, col, 4); }
    // Transpose 8×4 → 4×8
    transpose::<8, 4>(data);
}

#[inline(always)]
pub fn idct2d_8_16(data: &mut [f32]) {
    assert_eq!(data.len(), 128);
    for col in 0..16 { apply_idct_8(data, col, 16); }
    transpose::<8, 16>(data);
    for col in 0..8 { apply_idct_16(data, col, 8); }
    transpose::<16, 8>(data);
}

#[inline(always)]
pub fn idct2d_8_32(data: &mut [f32]) {
    assert_eq!(data.len(), 256);
    for col in 0..32 { apply_idct_8(data, col, 32); }
    transpose::<8, 32>(data);
    for col in 0..8 { apply_idct_32(data, col, 8); }
    transpose::<32, 8>(data);
}

#[inline(always)]
pub fn idct2d_16_32(data: &mut [f32]) {
    assert_eq!(data.len(), 512);
    for col in 0..32 { apply_idct_16(data, col, 32); }
    transpose::<16, 32>(data);
    for col in 0..16 { apply_idct_32(data, col, 16); }
    transpose::<32, 16>(data);
}

// ===== Rectangular 2D IDCTs (rows > cols → "tall") =====
// Input is stored in C×R (transposed) layout per JPEG XL convention.
// Algorithm: C-point column IDCT on C×R → transpose C×R→R×C → R-point column IDCT.

#[inline(always)]
pub fn idct2d_8_4(data: &mut [f32]) {
    assert_eq!(data.len(), 32);
    // Input is 4×8 (C×R). Column-wise 4-point IDCT (stride 8).
    for col in 0..8 { apply_idct_4(data, col, 8); }
    transpose::<4, 8>(data);
    // Now 8×4. Column-wise 8-point IDCT (stride 4).
    for col in 0..4 { apply_idct_8(data, col, 4); }
}

#[inline(always)]
pub fn idct2d_16_8(data: &mut [f32]) {
    assert_eq!(data.len(), 128);
    for col in 0..16 { apply_idct_8(data, col, 16); }
    transpose::<8, 16>(data);
    for col in 0..8 { apply_idct_16(data, col, 8); }
}

#[inline(always)]
pub fn idct2d_32_8(data: &mut [f32]) {
    assert_eq!(data.len(), 256);
    for col in 0..32 { apply_idct_8(data, col, 32); }
    transpose::<8, 32>(data);
    for col in 0..8 { apply_idct_32(data, col, 8); }
}

#[inline(always)]
pub fn idct2d_32_16(data: &mut [f32]) {
    assert_eq!(data.len(), 512);
    for col in 0..32 { apply_idct_16(data, col, 32); }
    transpose::<16, 32>(data);
    for col in 0..16 { apply_idct_32(data, col, 16); }
}
