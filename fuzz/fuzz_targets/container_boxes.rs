// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjxl_decoder::api::{JxlDecoderLimits, JxlDecoderOptions};

/// A known-good bare codestream. The container layout is what this target
/// fuzzes, so the payload is fixed and always valid.
const CODESTREAM: &[u8] = include_bytes!("../../zenjxl-decoder/resources/test/3x3_srgb_lossy.jxl");

fn options() -> JxlDecoderOptions {
    let mut limits = JxlDecoderLimits::restrictive();
    limits.max_pixels = Some(1_000_000);
    limits.max_memory_bytes = Some(32 * 1024 * 1024);
    let mut options = JxlDecoderOptions::default();
    options.limits = limits;
    options.parallel = false;
    options
}

fn push_box(out: &mut Vec<u8>, ty: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(ty);
    out.extend_from_slice(payload);
}

fn push_jxlp(out: &mut Vec<u8>, index: u32, last: bool, payload: &[u8]) {
    let mut body = (index | if last { 0x8000_0000 } else { 0 })
        .to_be_bytes()
        .to_vec();
    body.extend_from_slice(payload);
    push_box(out, b"jxlp", &body);
}

// Byte-for-byte splitting and reordering of a valid codestream across `jxlp`
// boxes must not change what it decodes to. Raw-byte fuzzing essentially never
// produces a well-formed out-of-order container, so the recipe is taken from
// the fuzzer input and the payload is fixed: `data` chooses how many boxes to
// use, where to cut, which boxes are empty, and in what order they appear in
// the file.
//
// Empty boxes and out-of-order delivery are exactly what `cjxl --output_mode=2`
// emits, and both were sources of decode failures (see `CHANGELOG.md`, the
// out-of-order `jxlp` entries, and libjxl/jxl-rs#956).
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    // 1..=32 payload-carrying boxes.
    let num_cuts = (data[0] % 32) as usize;
    let empty_mask = u16::from_le_bytes([data[1], data[2]]);
    let rotate = data[3] as usize;
    let recipe = &data[4..];

    // Cut the codestream into `num_cuts + 1` pieces at fuzzer-chosen offsets.
    let mut cuts: Vec<usize> = recipe
        .iter()
        .take(num_cuts)
        .map(|b| (*b as usize) * CODESTREAM.len() / 256)
        .collect();
    cuts.sort_unstable();
    let mut pieces: Vec<&[u8]> = Vec::with_capacity(cuts.len() + 1);
    let mut prev = 0;
    for c in cuts {
        pieces.push(&CODESTREAM[prev..c]);
        prev = c;
    }
    pieces.push(&CODESTREAM[prev..]);

    // Interleave empty boxes: they carry an index but no payload, so they must
    // advance the index space without ever being read as end of input.
    let mut payloads: Vec<&[u8]> = Vec::with_capacity(pieces.len() + 16);
    for (i, p) in pieces.into_iter().enumerate() {
        if i < 16 && (empty_mask >> i) & 1 == 1 {
            payloads.push(&[]);
        }
        payloads.push(p);
    }

    // Emit the boxes in a rotated order so later indices precede earlier ones,
    // which is the out-of-order case; rotate == 0 keeps them in order.
    let n = payloads.len();
    let mut file = Vec::new();
    push_box(&mut file, b"JXL ", &[0x0d, 0x0a, 0x87, 0x0a]);
    // ftyp minor version 1: out-of-order `jxlp` is only legal there.
    push_box(&mut file, b"ftyp", b"jxl \x00\x00\x00\x01jxl ");
    let start = if n == 0 { 0 } else { rotate % n };
    for k in 0..n {
        let idx = (start + k) % n;
        push_jxlp(&mut file, idx as u32, idx == n - 1, payloads[idx]);
    }

    let container = zenjxl_decoder::decode_with(&file, options());
    let bare = zenjxl_decoder::decode_with(CODESTREAM, options());

    // The container must not change the pixels. A container that cannot be
    // decoded at all is a finding too, since every layout built here is legal.
    match (container, bare) {
        (Ok(a), Ok(b)) => {
            assert_eq!(
                (a.width, a.height, a.channels),
                (b.width, b.height, b.channels),
                "container changed image geometry"
            );
            assert!(a.data == b.data, "container changed pixels");
        }
        (Err(e), Ok(_)) => panic!("legal container layout failed to decode: {e:?}"),
        // The bare codestream is a fixture; if it stops decoding that is a
        // separate bug and the unit tests will catch it.
        (_, Err(_)) => {}
    }
});
