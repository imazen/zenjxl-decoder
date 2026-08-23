// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Blue-noise dithering of 8-bit output (libjxl `stage_write.cc`
//! `MakeUnsigned`, ported to jxl-rs in #841).
//!
//! `decode()` always emits u8, so this is what every consumer sees. The
//! dither is a fixed 32x32 table indexed by absolute pixel position and
//! channel, added in output-code units before rounding, so:
//! - it is deterministic and position-stable (a streamed decode equals a
//!   one-shot decode),
//! - it changes a lossy sample by at most one code relative to plain
//!   rounding,
//! - it never changes an exact 8-bit value (|d| < 0.5), so lossless 8-bit
//!   content is untouched,
//! - with the same table and indexing as libjxl, `djxl`'s u8 output is
//!   reproduced bit-for-bit wherever the float pipelines agree.

use crate::api::{
    Endianness, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, ProcessingResult, states,
};
use crate::image::{Image, Rect};
use crate::util::test::fixture_bytes;

fn testdata(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// RGB samples of the public u8 API (alpha dropped).
fn decode_rgb8(data: &[u8], options: JxlDecoderOptions) -> (usize, usize, Vec<u8>) {
    let img = crate::decode_with(data, options).unwrap();
    assert_eq!(img.channels, 4);
    let rgb = img
        .data
        .chunks_exact(4)
        .flat_map(|px| px[..3].iter().copied())
        .collect();
    (img.width, img.height, rgb)
}

/// RGB samples as native-endian u16 through the streaming API (u16 output
/// is never dithered).
fn decode_rgb16(data: &[u8]) -> (usize, usize, Vec<u16>) {
    let mut input = data;
    let mut options = JxlDecoderOptions::default();
    options.limits.max_memory_bytes = None;
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder = loop {
        match decoder.process(&mut input).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let (width, height) = decoder.basic_info().size;
    let num_ec = decoder.basic_info().extra_channels.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U16 {
            endianness: Endianness::native(),
            bit_depth: 16,
        }),
        extra_channel_format: vec![None; num_ec],
    });
    let mut decoder = loop {
        match decoder.process(&mut input).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let mut out = Image::<u16>::new((width * 3, height)).unwrap();
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        out.get_rect_mut(Rect {
            origin: (0, 0),
            size: (width * 3, height),
        })
        .into_raw(),
    )];
    loop {
        match decoder.process(&mut input, &mut buffers).unwrap() {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    }
    let mut samples = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        samples.extend_from_slice(out.row(y));
    }
    (width, height, samples)
}

/// Plain (undithered) u8 value of a 16-bit sample.
fn round16_to_8(v: u16) -> u8 {
    ((v as u32 * 255 + 32767) / 65535) as u8
}

/// `with_preview.jxl` (64x64 lossy) must reproduce djxl 0.12.0's RGB8 output
/// byte-for-byte; upstream jxl-rs 0.6 does. Before dithering the fork
/// differed in 3085 of 12288 samples.
#[test]
fn u8_output_matches_djxl_dithered_reference_64x64() {
    let reference = testdata("dither/with_preview_djxl012_rgb8.raw");
    let (w, h, rgb) = decode_rgb8(
        &fixture_bytes("with_preview.jxl"),
        JxlDecoderOptions::default(),
    );
    assert_eq!((w, h), (64, 64));
    assert_eq!(rgb.len(), reference.len());
    let diffs = rgb.iter().zip(&reference).filter(|(a, b)| a != b).count();
    assert_eq!(
        diffs,
        0,
        "{diffs} of {} samples differ from djxl",
        rgb.len()
    );
}

/// Same on the 3x3 lossy fixture (values from djxl 0.12.0 / jxl-rs 0.6).
#[test]
fn u8_output_matches_djxl_3x3_lossy() {
    let (_, _, rgb) = decode_rgb8(
        &fixture_bytes("3x3_srgb_lossy.jxl"),
        JxlDecoderOptions::default(),
    );
    assert_eq!(
        rgb,
        [
            255, 0, 14, 0, 255, 16, 0, 0, 255, 128, 63, 64, 60, 128, 63, 61, 57, 129, 255, 255,
            255, 129, 130, 129, 0, 0, 0
        ]
    );
}

/// Exact 8-bit values are never moved by the dither (|d| < 0.5).
#[test]
fn lossless_8bit_u8_output_is_unchanged_by_dither() {
    for name in [
        "3x3_srgb_lossless.jxl",
        "gray_alpha_lossless.jxl",
        "squeeze_edge.jxl",
        "green_queen_modular_e3.jxl",
    ] {
        let data = fixture_bytes(name);
        let (_, _, rgb16) = decode_rgb16(&data);
        let img = crate::decode(&data).unwrap();
        let samples_per_px = img.channels;
        let color = if img.is_grayscale { 1 } else { 3 };
        let rgb8: Vec<u8> = img
            .data
            .chunks_exact(samples_per_px)
            .flat_map(|px| px[..color].iter().copied())
            .collect();
        // grayscale: the u16 decode above requested RGB, so compare against R.
        let want: Vec<u8> = rgb16
            .chunks_exact(3)
            .flat_map(|px| px[..color].iter().map(|&v| round16_to_8(v)))
            .collect();
        assert_eq!(rgb8, want, "{name}: dither changed an exact 8-bit sample");
        // and those are genuinely exact 8-bit values
        assert!(
            rgb16.iter().all(|&v| v % 257 == 0),
            "{name}: source is not 8-bit exact"
        );
    }
}

/// The dither table is indexed by absolute image position, so a streamed
/// decode (groups rendered in batches, possibly re-rendered) must give the
/// same u8 pixels as a one-shot decode.
#[test]
fn u8_dither_is_position_stable_across_chunked_decodes() {
    for name in [
        "with_preview.jxl",
        "green_queen_vardct_e3.jxl",
        "oddsize_ups.jxl",
    ] {
        let data = fixture_bytes(name);
        let (w, h, a) = decode_rgb8(&data, JxlDecoderOptions::default());
        let mut input = data.as_slice();
        // stream in 97-byte chunks through the typed API
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut chunk = &input[..0];
        macro_rules! advance {
            ($dec:ident $(, $buf:expr)?) => {
                loop {
                    chunk = &input[..(chunk.len() + 97).min(input.len())];
                    let before = chunk.len();
                    let r = $dec.process(&mut chunk $(, $buf)?).unwrap();
                    input = &input[(before - chunk.len())..];
                    match r {
                        ProcessingResult::Complete { result } => break result,
                        ProcessingResult::NeedsMoreInput { fallback, .. } => $dec = fallback,
                    }
                }
            };
        }
        let mut dec = advance!(decoder);
        let num_ec = dec.basic_info().extra_channels.len();
        dec.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: vec![None; num_ec],
        });
        let mut dec = advance!(dec);
        let mut out = Image::<u8>::new((w * 3, h)).unwrap();
        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            out.get_rect_mut(Rect {
                origin: (0, 0),
                size: (w * 3, h),
            })
            .into_raw(),
        )];
        let _ = advance!(dec, &mut buffers);
        let mut b = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            b.extend_from_slice(out.row(y));
        }
        let diffs = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(
            diffs, 0,
            "{name}: streamed u8 decode differs from one-shot in {diffs} samples"
        );
    }
}

/// Dithering moves a lossy sample by at most one code relative to plain
/// rounding of the 16-bit output.
#[test]
fn u8_is_within_one_code_of_rounded_u16() {
    for name in [
        "green_queen_vardct_e3.jxl",
        "with_preview.jxl",
        "cafe_web_q80.jxl",
    ] {
        let data = fixture_bytes(name);
        let (_, _, rgb8) = decode_rgb8(&data, JxlDecoderOptions::default());
        let (_, _, rgb16) = decode_rgb16(&data);
        assert_eq!(rgb8.len(), rgb16.len());
        let mut moved = 0usize;
        for (&a, &b) in rgb8.iter().zip(&rgb16) {
            let plain = round16_to_8(b);
            let d = (a as i32 - plain as i32).abs();
            assert!(d <= 1, "{name}: u8 {a} vs plain {plain} (u16 {b})");
            moved += (d == 1) as usize;
        }
        // A lossy photo has a large fraction of non-exact samples; the dither
        // must actually be doing something.
        assert!(
            moved > rgb8.len() / 20,
            "{name}: only {moved} samples dithered"
        );
    }
}

/// `with_dither_u8(false)` restores plain rounding.
#[test]
fn dither_can_be_disabled() {
    let data = fixture_bytes("with_preview.jxl");
    let (_, _, rgb8) = decode_rgb8(&data, JxlDecoderOptions::default().with_dither_u8(false));
    let (_, _, rgb16) = decode_rgb16(&data);
    let want: Vec<u8> = rgb16.iter().map(|&v| round16_to_8(v)).collect();
    assert_eq!(rgb8, want);
}
