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
//! `zero_length_skippable_box.jxl` is exercised in
//! `api::inner::box_parser`'s own test module instead, where upstream keeps
//! it: the fixture holds no `jxlc`/`jxlp` at all, so there is nothing to
//! decode and the assertion is that draining the parser terminates.
//!
//! Tests that expect an error use [`decode_with`], not the `decode` helper in
//! `api::decoder::tests`: that helper `unwrap()`s the `process` result
//! internally, so it panics where upstream's `decode_internal` returns `Err`,
//! and a test built on it could not tell a graceful rejection from a crash.

use std::time::Duration;

use crate::api::decoder::tests::{compare_frames, decode};
use crate::api::{JxlDecoderOptions, decode_with};
use crate::error::Error;

fn testdata(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Runs `f` on a worker thread and fails if it does not finish in time.
///
/// A decoder that spins forever would otherwise hang the whole test binary
/// instead of reporting a failure, and CI would time out with no useful
/// output. The spinning thread is leaked deliberately — the process is about
/// to abort on the panic anyway.
fn with_deadline<T: Send + 'static>(
    secs: u64,
    what: &str,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(v) => v,
        Err(_) => panic!("{what}: still running after {secs}s (upstream expects it to return)"),
    }
}

/// Upstream `ooo_jxlp_with_trailing_bytes_does_not_hang`: an out-of-order
/// `jxlp` stream with trailing container bytes used to spin forever when
/// header parsing needed more codestream than the boxes provided. Upstream
/// asserts a single `process()` call returns `NeedsMoreInput`.
#[test]
fn ooo_jxlp_with_trailing_bytes_does_not_hang() {
    let data = testdata("ooo_jxlp_with_trailing_bytes.jxl");
    let res = with_deadline(20, "ooo_jxlp_with_trailing_bytes", move || {
        decode_with(&data, JxlDecoderOptions::default()).map(|_| ())
    });
    // Upstream stops with "needs more input"; reaching a decoded image or any
    // other error is a divergence, but the point of the fixture is that the
    // call returns at all.
    assert!(
        matches!(res, Err(ref e) if matches!(e.error(), Error::OutOfBounds(_))),
        "expected an out-of-input result, got {res:?}"
    );
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

/// Upstream `test_fuzzer_patches_ec_upsampling_dim_shift`: patches with mixed
/// upsampling are rejected with a specific error.
#[test]
fn fuzzer_patches_ec_upsampling_dim_shift_rejected() {
    let data = testdata("patches_ec_upsampling_dim_shift.jxl");
    let result = decode_with(&data, JxlDecoderOptions::default());
    assert!(
        matches!(result, Err(ref e) if matches!(e.error(), Error::PatchesUnsupportedMixedUpsampling(..))),
        "expected a mixed upsampling error, got {:?}",
        result.map(|_| "a decoded image")
    );
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
/// `aux_box_trailing_infinite`: the `Exif` box is found and has the expected
/// payload size, whether it precedes the codestream, follows it with a finite
/// box size, or follows it with a to-end-of-file size.
///
/// Upstream reaches this through `request_aux_boxes` + `aux_boxes()` /
/// `trailing_box()`; this fork exposes it as `JxlImage::exif`, so the port
/// asserts the same payload sizes through that. Upstream's sizes are 170 for
/// the plain `Exif` boxes and 120 for the brotli-compressed `brob` ones;
/// this fork strips the 4-byte TIFF header offset, hence the -4.
#[test]
fn exif_box_payload_sizes() {
    for (name, upstream_size) in [
        ("exif.jxl", 170usize),
        ("exif_brob.jxl", 120),
        ("exif_trailing_finite.jxl", 170),
        ("exif_brob_trailing_finite.jxl", 120),
        ("exif_trailing_infinite.jxl", 170),
        ("exif_brob_trailing_infinite.jxl", 120),
    ] {
        let data = testdata(name);
        let img = decode_with(&data, JxlDecoderOptions::default())
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e:?}"));
        let exif = img
            .exif
            .unwrap_or_else(|| panic!("{name}: no Exif box was captured"));
        assert_eq!(
            exif.len(),
            upstream_size - 4,
            "{name}: Exif payload size (upstream {upstream_size} minus the 4-byte TIFF offset)"
        );
    }
}

/// Upstream `decode_ooo_jxlp_invalid_animated_container`: out-of-order `jxlp`
/// boxes require every frame to start in a box that has all logically-earlier
/// boxes physically before it and all later ones after it. This file does not,
/// and upstream rejects it with `InvalidBox`.
#[test]
fn invalid_animated_ooo_jxlp_is_rejected() {
    let data = testdata("invalid_animated_ooo_jxlp.jxl");
    let res = decode_with(&data, JxlDecoderOptions::default());
    assert!(
        matches!(res, Err(ref e) if matches!(e.error(), Error::InvalidBox)),
        "expected rejection due to a frame starting in a non-valid checkpoint box, got {:?}",
        res.map(|_| "a decoded image")
    );
}
