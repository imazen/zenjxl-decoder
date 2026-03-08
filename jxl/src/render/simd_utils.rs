// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD utilities for interleaving and deinterleaving channel data.
//!
//! These functions assume that input buffers are padded to at least the SIMD
//! vector length (up to 16 elements), as is standard in the render pipeline.

use archmage::prelude::*;

#[autoversion]
fn interleave_2_inner(
    _token: SimdToken,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    len: usize,
) {
    for i in 0..len {
        out[i * 2] = a[i];
        out[i * 2 + 1] = b[i];
    }
}

/// Interleave 2 planar channels into packed format.
/// Buffers must be padded to SIMD vector length.
pub fn interleave_2_dispatch(a: &[f32], b: &[f32], out: &mut [f32]) {
    interleave_2_inner(a, b, out, a.len().min(b.len()));
}

#[autoversion]
fn deinterleave_2_inner(
    _token: SimdToken,
    input: &[f32],
    a: &mut [f32],
    b: &mut [f32],
    len: usize,
) {
    for i in 0..len {
        a[i] = input[i * 2];
        b[i] = input[i * 2 + 1];
    }
}

/// Deinterleave packed format into 2 planar channels.
/// Buffers must be padded to SIMD vector length.
pub fn deinterleave_2_dispatch(input: &[f32], a: &mut [f32], b: &mut [f32]) {
    deinterleave_2_inner(input, a, b, a.len().min(b.len()));
}

#[autoversion]
fn interleave_3_inner(
    _token: SimdToken,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    out: &mut [f32],
    len: usize,
) {
    for i in 0..len {
        out[i * 3] = a[i];
        out[i * 3 + 1] = b[i];
        out[i * 3 + 2] = c[i];
    }
}

/// Interleave 3 planar channels into packed RGB format.
/// Buffers must be padded to SIMD vector length.
pub fn interleave_3_dispatch(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    interleave_3_inner(a, b, c, out, a.len().min(b.len()).min(c.len()));
}

#[autoversion]
fn deinterleave_3_inner(
    _token: SimdToken,
    input: &[f32],
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    len: usize,
) {
    for i in 0..len {
        a[i] = input[i * 3];
        b[i] = input[i * 3 + 1];
        c[i] = input[i * 3 + 2];
    }
}

/// Deinterleave packed RGB format into 3 planar channels.
/// Buffers must be padded to SIMD vector length.
pub fn deinterleave_3_dispatch(input: &[f32], a: &mut [f32], b: &mut [f32], c: &mut [f32]) {
    deinterleave_3_inner(input, a, b, c, a.len().min(b.len()).min(c.len()));
}

#[autoversion]
fn interleave_4_inner(
    _token: SimdToken,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    e: &[f32],
    out: &mut [f32],
    len: usize,
) {
    for i in 0..len {
        out[i * 4] = a[i];
        out[i * 4 + 1] = b[i];
        out[i * 4 + 2] = c[i];
        out[i * 4 + 3] = e[i];
    }
}

/// Interleave 4 planar channels into packed RGBA format.
/// Buffers must be padded to SIMD vector length.
pub fn interleave_4_dispatch(a: &[f32], b: &[f32], c: &[f32], e: &[f32], out: &mut [f32]) {
    interleave_4_inner(a, b, c, e, out, a.len().min(b.len()).min(c.len()).min(e.len()));
}

#[autoversion]
fn deinterleave_4_inner(
    _token: SimdToken,
    input: &[f32],
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    e: &mut [f32],
    len: usize,
) {
    for i in 0..len {
        a[i] = input[i * 4];
        b[i] = input[i * 4 + 1];
        c[i] = input[i * 4 + 2];
        e[i] = input[i * 4 + 3];
    }
}

/// Deinterleave packed RGBA format into 4 planar channels.
/// Buffers must be padded to SIMD vector length.
pub fn deinterleave_4_dispatch(
    input: &[f32],
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    e: &mut [f32],
) {
    deinterleave_4_inner(input, a, b, c, e, a.len().min(b.len()).min(c.len()).min(e.len()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleave_deinterleave_2_roundtrip() {
        // Use 16 elements to ensure SIMD alignment for all backends
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let b = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0,
        ];

        let mut packed = vec![0.0; 32];
        interleave_2_dispatch(&a, &b, &mut packed);

        // Check interleaved format
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 10.0);
        assert_eq!(packed[2], 2.0);
        assert_eq!(packed[3], 20.0);

        // Deinterleave back
        let mut a_out = vec![0.0; 16];
        let mut b_out = vec![0.0; 16];
        deinterleave_2_dispatch(&packed, &mut a_out, &mut b_out);

        assert_eq!(a_out, a);
        assert_eq!(b_out, b);
    }

    #[test]
    fn test_interleave_deinterleave_3_roundtrip() {
        // Use 16 elements to ensure SIMD alignment for all backends
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=16).map(|x| x as f32 * 10.0).collect();
        let c: Vec<f32> = (1..=16).map(|x| x as f32 * 100.0).collect();

        let mut packed = vec![0.0; 48];
        interleave_3_dispatch(&a, &b, &c, &mut packed);

        // Check interleaved format
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 10.0);
        assert_eq!(packed[2], 100.0);
        assert_eq!(packed[3], 2.0);
        assert_eq!(packed[4], 20.0);
        assert_eq!(packed[5], 200.0);

        // Deinterleave back
        let mut a_out = vec![0.0; 16];
        let mut b_out = vec![0.0; 16];
        let mut c_out = vec![0.0; 16];
        deinterleave_3_dispatch(&packed, &mut a_out, &mut b_out, &mut c_out);

        assert_eq!(a_out, a);
        assert_eq!(b_out, b);
        assert_eq!(c_out, c);
    }

    #[test]
    fn test_interleave_deinterleave_4_roundtrip() {
        // Use 16 elements to ensure SIMD alignment for all backends
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=16).map(|x| x as f32 * 10.0).collect();
        let c: Vec<f32> = (1..=16).map(|x| x as f32 * 100.0).collect();
        let d: Vec<f32> = (1..=16).map(|x| x as f32 * 1000.0).collect();

        let mut packed = vec![0.0; 64];
        interleave_4_dispatch(&a, &b, &c, &d, &mut packed);

        // Check interleaved format
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 10.0);
        assert_eq!(packed[2], 100.0);
        assert_eq!(packed[3], 1000.0);
        assert_eq!(packed[4], 2.0);
        assert_eq!(packed[5], 20.0);

        // Deinterleave back
        let mut a_out = vec![0.0; 16];
        let mut b_out = vec![0.0; 16];
        let mut c_out = vec![0.0; 16];
        let mut d_out = vec![0.0; 16];
        deinterleave_4_dispatch(&packed, &mut a_out, &mut b_out, &mut c_out, &mut d_out);

        assert_eq!(a_out, a);
        assert_eq!(b_out, b);
        assert_eq!(c_out, c);
        assert_eq!(d_out, d);
    }
}
