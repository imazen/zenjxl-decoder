// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Ports of the upstream jxl-rs tests that live on `jxl/tests/testdata/`
//! fixtures, synced 2026-09-09. The fixtures are in `tests/testdata/`, which
//! the `resources/test/` sweep does not scan, so each one needs a test here.
//!
//! These assert *upstream's* expectations, not this fork's current behaviour.
//! Where the fork's public API differs (upstream's `request_aux_boxes` /
//! `aux_boxes()` / `trailing_box()` streaming surface has no analogue here;
//! EXIF arrives on `JxlImage::exif` instead), the port keeps the value being
//! asserted and changes only how it is reached.
//!
//! Known failures are tracked in `docs/UPSTREAM_SYNC.md`; see the "testdata
//! sync" section. Do not weaken an assertion to make one pass.
//!
//! `invalid_animated_ooo_jxlp.jxl` is *not* here: this fork decodes it, and
//! the decode is byte-identical to `djxl 0.12.0`, so it lives in
//! `resources/test/` where the automatic sweeps cover it. Upstream rejects it
//! with `InvalidBox`, which is stricter than the reference implementation --
//! see `docs/UPSTREAM_SYNC.md`.
//!
//! `zero_length_skippable_box.jxl` is exercised in
//! `api::inner::box_parser`'s own test module instead, where upstream keeps
//! it: the fixture holds no `jxlc`/`jxlp` at all, so there is nothing to
//! decode and the assertion is that draining the parser terminates.
//!
//! Tests that expect an error use [`decode_with`], not the `decode` helper in
//! `api::decoder::tests`: that helper `unwrap()`s the `process` result
//! internally, so it panics where upstream's `decode_internal` returns `Err`,
//! and a test built on it could not tell a graceful rejection from a crash.

use crate::api::decoder::tests::{compare_frames, decode};
use crate::api::{JxlDecoderOptions, decode_with};

fn testdata(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Upstream `test_fuzzer_context_map_num_histograms_overflow`: must not panic
/// in either pipeline; an error is fine.
#[test]
fn fuzzer_context_map_num_histograms_overflow_does_not_panic() {
    let data = testdata("context_map_num_histograms_overflow.jxl");
    let _ = decode_with(&data, JxlDecoderOptions::default());
}

/// Upstream `test_fuzzer_modular_palette_empty_meta_channel`: must fail, not
/// panic and not decode.
#[test]
fn fuzzer_modular_palette_empty_meta_channel_errors() {
    let data = testdata("modular_palette_empty_meta_channel.jxl");
    assert!(decode_with(&data, JxlDecoderOptions::default()).is_err());
}

/// Upstream `test_modular_rle_fast_path`: the LZ77 RLE fast path must produce
/// the same pixels as the same image encoded without LZ77.
#[test]
fn modular_rle_fast_path_matches_without_lz77() {
    let data = testdata("modular_rle_fast_path.jxl");
    let (_, frames) = decode(&data, usize::MAX, false, false, None).unwrap();
    let no_lz77 = testdata("modular_rle_fast_path_without_lz77.jxl");
    let (_, no_lz77_frames) = decode(&no_lz77, usize::MAX, false, false, None).unwrap();
    compare_frames(
        std::path::Path::new("modular_rle_fast_path.jxl"),
        0,
        &frames[0],
        &no_lz77_frames[0],
    )
    .unwrap();
}

/// Upstream `test_fuzzer_vardct_grayscale_unused_channel`: a grayscale
/// non-XYB VarDCT frame has no stage consuming colour channels 1 and 2;
/// asking the pipeline for their scratch buffers used to panic.
#[test]
fn fuzzer_vardct_grayscale_unused_channel() {
    let data = testdata("vardct_grayscale_unused_channel.jxl");
    let (_, frames) = decode(&data, usize::MAX, false, false, None).unwrap();
    let (_, simple_frames) = decode(&data, usize::MAX, true, false, None).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), 1);
    assert_eq!(frames[0][0].size(), (1, 1));
    compare_frames(
        std::path::Path::new("vardct_grayscale_unused_channel.jxl"),
        0,
        &frames[0],
        &simple_frames[0],
    )
    .unwrap();
    // Streaming input with flushing exercises the low-memory pipeline.
    decode(&data, 1, false, true, None).unwrap();
}

/// Upstream `aux_box_before_codestream` / `aux_box_trailing_finite` /
/// `aux_box_trailing_infinite`: the `Exif` box is found with the expected
/// payload, whether it precedes the codestream, follows it with a finite box
/// size, or follows it with size 0 (runs to end of file).
///
/// Upstream reaches this through `request_aux_boxes` + `aux_boxes()` /
/// `trailing_box()` and asserts the *raw* box length (170 plain, 120 brotli-
/// compressed). This fork exposes the processed payload as `JxlImage::exif`
/// with the 4-byte TIFF header offset stripped, so the equivalent assertion is
/// that every one of these files yields the same 166-byte EXIF payload.
#[test]
fn exif_box_payload_sizes() {
    for name in [
        "exif.jxl",
        "exif_trailing_finite.jxl",
        "exif_trailing_infinite.jxl",
    ] {
        let data = testdata(name);
        let img = decode_with(&data, JxlDecoderOptions::default())
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e:?}"));
        let exif = img
            .exif
            .unwrap_or_else(|| panic!("{name}: no Exif box was captured"));
        assert_eq!(exif.len(), 166, "{name}: Exif payload size");
    }
}

/// The brotli-compressed (`brob`) halves of the same three files. `brob`
/// decompression needs the `jpeg` feature, which is what pulls in brotli;
/// without it those boxes are skipped by design.
#[cfg(feature = "jpeg")]
#[test]
fn exif_brob_box_payload_sizes() {
    for name in [
        "exif_brob.jxl",
        "exif_brob_trailing_finite.jxl",
        "exif_brob_trailing_infinite.jxl",
    ] {
        let data = testdata(name);
        let img = decode_with(&data, JxlDecoderOptions::default())
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e:?}"));
        let exif = img
            .exif
            .unwrap_or_else(|| panic!("{name}: no Exif box was captured"));
        assert_eq!(exif.len(), 166, "{name}: decompressed Exif payload size");
    }
}
