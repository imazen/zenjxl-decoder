// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Pixel-exact gates for the single-leaf modular fast paths.
//!
//! The fixtures are one 96x64 crop of a CLIC 2025 training photo, encoded
//! lossless by jxl-encoder's `fast_decode_lossless` example with tree learning
//! off, so each channel prunes to one MA leaf: `Top` (North) or `Gradient`,
//! each with ANS and with prefix codes. The expected FNV-1a 64 hash covers the
//! RGBA u8 output and was computed from libjxl v0.12 `djxl` output, which
//! also matches the source PNG byte for byte.

const EXPECTED_RGBA_FNV1A64: u64 = 0xa5a3_a32f_1541_c0a7;

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn check(data: &[u8]) {
    let image = zenjxl_decoder::decode(data).expect("decode");
    assert_eq!((image.width, image.height), (96, 64));
    assert_eq!(fnv1a64(&image.data), EXPECTED_RGBA_FNV1A64);
}

#[test]
fn single_top_leaf_prefix() {
    check(include_bytes!("testdata/single-leaf/top_prefix.jxl"));
}

#[test]
fn single_top_leaf_ans() {
    check(include_bytes!("testdata/single-leaf/top_ans.jxl"));
}

#[test]
fn single_gradient_leaf_prefix() {
    check(include_bytes!("testdata/single-leaf/grad_prefix.jxl"));
}

#[test]
fn single_gradient_leaf_ans() {
    check(include_bytes!("testdata/single-leaf/grad_ans.jxl"));
}
