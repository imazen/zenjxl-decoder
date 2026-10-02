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
//!
//! The multi-group fixtures (jxl-encoder `with_fast_decode`, effort 4) cover
//! reading two groups together: a 520x24 RGB strip has three groups (one
//! pair plus a leftover), and a 270x260 gray crop has four groups whose edge
//! groups differ in width and height. Their hashes cover the output of
//! `decode`/`decode_with` (RGBA, or gray+alpha) and were computed from
//! `djxl` output that matches the source PNG.

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

const STRIP_3_GROUPS_FNV1A64: u64 = 0x82ee_22db_6956_9a7d;
const GRAY_4_GROUPS_FNV1A64: u64 = 0x3b35_7bb9_5fd3_2006;

fn check_multigroup(data: &[u8], size: (usize, usize), expected: u64, parallel: bool) {
    let mut options = zenjxl_decoder::api::JxlDecoderOptions::default();
    options.parallel = parallel;
    let image = zenjxl_decoder::decode_with(data, options).expect("decode");
    assert_eq!((image.width, image.height), size);
    assert_eq!(fnv1a64(&image.data), expected);
}

#[test]
fn paired_groups_sequential() {
    check_multigroup(
        include_bytes!("testdata/single-leaf/grad_prefix_3groups.jxl"),
        (520, 24),
        STRIP_3_GROUPS_FNV1A64,
        false,
    );
    check_multigroup(
        include_bytes!("testdata/single-leaf/grad_prefix_gray_4groups.jxl"),
        (270, 260),
        GRAY_4_GROUPS_FNV1A64,
        false,
    );
}

/// With one rayon thread, four groups are enough for the parallel path to
/// read two groups per task.
#[cfg(feature = "threads")]
#[test]
fn paired_groups_parallel() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        check_multigroup(
            include_bytes!("testdata/single-leaf/grad_prefix_3groups.jxl"),
            (520, 24),
            STRIP_3_GROUPS_FNV1A64,
            true,
        );
        check_multigroup(
            include_bytes!("testdata/single-leaf/grad_prefix_gray_4groups.jxl"),
            (270, 260),
            GRAY_4_GROUPS_FNV1A64,
            true,
        );
    });
}
