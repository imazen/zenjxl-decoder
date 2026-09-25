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

/// Decodes `data` in `chunk_size` pieces, flushing pixels whenever the decoder
/// stops for input, and treats running out of input as the end of a partial
/// image rather than an error. Mirrors upstream jxl-rs `decode_internal` with
/// `do_flush: true, allow_partial: true`: the file is truncated on purpose,
/// and the point is that flushing what exists never panics.
///
/// Returns how many times it flushed, so callers can prove the partial-render
/// path actually ran rather than the file bailing out before the frame body.
fn decode_partial_with_flush(data: &[u8], chunk_size: usize) -> crate::error::Result<usize> {
    let mut flushes = 0usize;
    use crate::api::{
        JxlDataFormat, JxlDecoder, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    };
    use crate::image::{Image, Rect};

    let mut options = JxlDecoderOptions::default();
    options.limits.max_memory_bytes = None;
    let mut remaining = data;
    let mut fed = &remaining[0..0];
    // Hands the decoder the next chunk on top of whatever it has not consumed.
    macro_rules! feed {
        () => {{
            fed = &remaining[..(fed.len().saturating_add(chunk_size)).min(remaining.len())];
        }};
    }

    let mut dec = JxlDecoder::<states::Initialized>::new(options);
    let mut info = loop {
        feed!();
        let before = fed.len();
        let r = dec.process(&mut fed)?;
        remaining = &remaining[(before - fed.len())..];
        match r {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if remaining.is_empty() {
                    return Ok(flushes); // truncated before the image header: nothing to flush
                }
                dec = fallback;
            }
        }
    };

    let (w, h) = info.basic_info().size;
    let fmt = info.current_pixel_format().clone();
    info.set_pixel_format(JxlPixelFormat {
        color_type: fmt.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: fmt
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    });
    let fmt = info.current_pixel_format().clone();
    let mut buffers = vec![Image::<f32>::new((
        w * fmt.color_type.samples_per_pixel(),
        h,
    ))?];
    for ecf in fmt.extra_channel_format.iter().flatten() {
        let _ = ecf;
        buffers.push(Image::<f32>::new((w, h))?);
    }
    let mut out: Vec<_> = buffers
        .iter_mut()
        .map(|b| {
            let size = b.size();
            JxlOutputBuffer::from_image_rect_mut(
                b.get_rect_mut(Rect {
                    origin: (0, 0),
                    size,
                })
                .into_raw(),
            )
        })
        .collect();

    loop {
        let mut frame = loop {
            feed!();
            let before = fed.len();
            let r = info.process(&mut fed)?;
            remaining = &remaining[(before - fed.len())..];
            match r {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if remaining.is_empty() {
                        return Ok(flushes);
                    }
                    info = fallback;
                }
            }
        };
        info = loop {
            feed!();
            let before = fed.len();
            let r = frame.process(&mut fed, &mut out)?;
            remaining = &remaining[(before - fed.len())..];
            match r {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                    // Flush whatever has arrived, including at the very end of
                    // a truncated file -- that final flush is the one the
                    // upstream bugs panicked in.
                    fallback.flush_pixels(&mut out)?;
                    flushes += 1;
                    if remaining.is_empty() {
                        return Ok(flushes);
                    }
                    frame = fallback;
                }
            }
        };
        if !info.has_more_frames() {
            return Ok(flushes);
        }
    }
}

/// Upstream `flush_truncated_squeeze_missing_tiles` (jxl-rs `82b981a`,
/// chromium issue 562761172): flushing a truncated image used to panic when a
/// smooth-squeeze upsample step read a tile whose channel had not been
/// decoded yet.
///
/// The panicking code is upstream's partial LF-global render, which this fork
/// does not have: progressive preview is recorded as N/A in
/// `docs/UPSTREAM_SYNC.md` ("the fork kept the pre-March flush design"). Here
/// the decoder stops for input at the frame header and flushes nothing, so the
/// upstream regression cannot occur. The test pins what does apply -- chunked,
/// flushing decode of this truncated file returns without panicking -- and
/// deliberately does not require a flush, which would assert a feature the
/// fork chose not to implement.
#[test]
fn flush_truncated_squeeze_missing_tiles() {
    let data = testdata("truncated_squeeze_flush_missing_tiles.jxl");
    for chunk_size in [64, 256, usize::MAX] {
        decode_partial_with_flush(&data, chunk_size)
            .unwrap_or_else(|e| panic!("chunk_size {chunk_size}: {e:?}"));
    }
}

/// Upstream `flush_truncated_squeeze_missing_avg` (jxl-rs `fce6e28`): the same
/// partial decode when the squeeze residuals are present but the averages are
/// not. Same N/A reasoning as above: upstream completes the frame header here
/// and renders partial LF-global; this fork waits for the section (167 more
/// bytes) and so never reaches the code that upstream fixed.
#[test]
fn flush_truncated_squeeze_missing_avg() {
    let data = testdata("truncated_squeeze_missing_avg.jxl");
    for chunk_size in [64, 256, usize::MAX] {
        decode_partial_with_flush(&data, chunk_size)
            .unwrap_or_else(|e| panic!("chunk_size {chunk_size}: {e:?}"));
    }
}
