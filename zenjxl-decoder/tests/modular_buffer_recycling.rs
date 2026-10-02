// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Recycled modular group buffers must not change output (upstream #812).
//!
//! `issue865_large_toc.jxl` is a 5249x5377 modular frame with 1806 groups.
//! It is decoded three ways:
//! - sequentially, where each two-group chunk is rendered right after
//!   decoding and its buffers return to the recycle pool;
//! - in parallel on one- and four-thread pools, which decode in batches of
//!   eight groups per thread and recycle buffers between batches.
//!
//! Each must give the pixels of libjxl v0.12 `djxl`: the expected hash is
//! the FNV-1a of `djxl`'s output as RGBA u8 (computed 2026-10-02).

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

const DATA: &[u8] = include_bytes!("testdata/jxlrs-865/issue865_large_toc.jxl");

fn decode_hash(parallel: bool) -> u64 {
    let mut options = zenjxl_decoder::api::JxlDecoderOptions::default();
    options.parallel = parallel;
    let image = zenjxl_decoder::decode_with(DATA, options).expect("decode");
    assert_eq!((image.width, image.height), (5249, 5377));
    fnv1a64(&image.data)
}

#[test]
fn sequential_recycling_keeps_output() {
    assert_eq!(decode_hash(false), EXPECTED);
}

#[cfg(feature = "threads")]
#[test]
fn parallel_recycling_keeps_output() {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(
            pool.install(|| decode_hash(true)),
            EXPECTED,
            "{threads} threads"
        );
    }
}

const EXPECTED: u64 = 11_630_157_713_389_196_120;
