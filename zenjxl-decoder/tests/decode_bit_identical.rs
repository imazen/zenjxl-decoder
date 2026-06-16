// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Bit-identical decode regression gate for the VarDCT scratch-buffer reuse in
//! issue #40.
//!
//! The fix (reusing the per-pass `num_nzeros` maps and replacing the per-block
//! `Vec` scratch in the AFV transform with stack arrays) is a pure allocation
//! lifetime change: it must not alter a single output pixel. Buffer reuse that
//! fails to fully reset state would silently corrupt the decode — exactly the
//! class of bug this asserts against.
//!
//! The reference hashes below were captured on the clean `main` commit
//! (ac1d907, the parent of this change) by decoding each fixture through the
//! public `zenjxl_decoder::decode` API. If a future change alters decoded
//! output for these fixtures, this test fails loudly rather than shipping a
//! pixel regression. Update the constants ONLY after independently confirming
//! the output change is intended.

use std::path::Path;

/// FNV-1a/64 over the full interleaved RGBA byte buffer. A whole-buffer content
/// hash: any pixel difference (even one bit) changes it.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

struct Expected {
    file: &'static str,
    width: usize,
    height: usize,
    channels: usize,
    len: usize,
    hash: u64,
}

/// Decode `file` and assert its dimensions, channel count, byte length, and a
/// stable content hash all match the clean-`main` reference.
fn assert_decode_matches(e: &Expected) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources/test")
        .join(e.file);
    let data = std::fs::read(&path)
        .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()));

    let image = zenjxl_decoder::decode(&data)
        .unwrap_or_else(|err| panic!("decode failed for {}: {err:?}", e.file));

    assert_eq!(image.width, e.width, "{}: width changed", e.file);
    assert_eq!(image.height, e.height, "{}: height changed", e.file);
    assert_eq!(
        image.channels, e.channels,
        "{}: channel count changed",
        e.file
    );
    assert_eq!(
        image.data.len(),
        e.len,
        "{}: output byte length changed",
        e.file
    );
    assert_eq!(
        fnv1a_64(&image.data),
        e.hash,
        "{}: decoded pixels differ from the clean-main reference \
         (0x{:016x} != expected 0x{:016x}) -- a buffer-reuse change has corrupted output",
        e.file,
        fnv1a_64(&image.data),
        e.hash,
    );
}

/// `bike_web_q85.jxl` is the 5.24 MP VarDCT photo used by the allocation
/// profiler in `examples/heaptrack_decode.rs`; it exercises the AFV transform
/// and `num_nzeros` paths touched by the issue #40 fix.
#[test]
fn bike_web_q85_decode_is_bit_identical() {
    assert_decode_matches(&Expected {
        file: "bike_web_q85.jxl",
        width: 2048,
        height: 2560,
        channels: 4,
        len: 20_971_520,
        hash: 0x14af_5870_ea86_5a92,
    });
}

/// A second VarDCT fixture at a different quality, so the gate covers more than
/// one coefficient distribution / block-type mix.
#[test]
fn bike_web_q75_decode_is_bit_identical() {
    assert_decode_matches(&Expected {
        file: "bike_web_q75.jxl",
        width: 2048,
        height: 2560,
        channels: 4,
        len: 20_971_520,
        hash: 0x9714_2587_6e6a_59e3,
    });
}
