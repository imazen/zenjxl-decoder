// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests for JPEG reconstruction from JXL containers with JBRD boxes.

use crate::api::{
    JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult, states,
};
use crate::image::{Image, Rect};

/// Helper: decode a JXL file and return the reconstructed JPEG bytes (if any).
fn decode_jpeg_reconstruction(jxl_data: &[u8]) -> Option<Vec<u8>> {
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = jxl_data;

    // Process until we have image info
    let mut decoder = loop {
        match decoder.process(&mut input).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    panic!("Unexpected end of input during header");
                }
                decoder = fallback;
            }
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;

    // Set RGB f32 format
    let format = JxlPixelFormat {
        color_type: crate::api::JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(format);

    // Process until frame info
    let mut decoder = loop {
        match decoder.process(&mut input).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    panic!("Unexpected end of input during frame header");
                }
                decoder = fallback;
            }
        }
    };

    // Prepare output buffer
    let mut color_buffer = Image::<f32>::new((width * 3, height)).unwrap();
    let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
        color_buffer
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * 3, height),
            })
            .into_raw(),
    )];

    // Decode frame
    let mut decoder = loop {
        match decoder.process(&mut input, &mut buffers).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    panic!("Unexpected end of input during frame decode");
                }
                decoder = fallback;
            }
        }
    };

    // Get reconstructed JPEG bytes
    decoder.take_jpeg_reconstruction()
}

/// Helper: assert byte-exact JPEG reconstruction match.
fn assert_jpeg_match(reconstructed: &[u8], reference: &[u8], label: &str) {
    assert!(
        reconstructed.len() >= 2,
        "{label}: Reconstructed JPEG too short: {} bytes",
        reconstructed.len()
    );
    assert_eq!(
        &reconstructed[..2],
        &[0xFF, 0xD8],
        "{label}: Should start with JPEG SOI marker"
    );

    if reconstructed != reference {
        // Find first byte difference for diagnostics
        let min_len = reconstructed.len().min(reference.len());
        for i in 0..min_len {
            if reconstructed[i] != reference[i] {
                let start = i.saturating_sub(4);
                let end = (i + 8).min(min_len);
                eprintln!(
                    "{label}: First difference at byte {i} (0x{i:04x}): \
                     got 0x{:02x}, expected 0x{:02x}",
                    reconstructed[i], reference[i]
                );
                eprintln!(
                    "  got: {:02x?}",
                    &reconstructed[start..end.min(reconstructed.len())]
                );
                eprintln!(
                    "  ref: {:02x?}",
                    &reference[start..end.min(reference.len())]
                );
                break;
            }
        }
    }

    assert_eq!(
        reconstructed.len(),
        reference.len(),
        "{label}: length mismatch: got {} expected {}",
        reconstructed.len(),
        reference.len()
    );
    assert_eq!(reconstructed, reference, "{label}: byte-exact match failed");
}

/// 3x3 JPEG with 4:2:0 subsampling (Y=2x2, Cb=Cr=1x1).
#[test]
fn test_jpeg_reconstruction_3x3() {
    let jxl_data = std::fs::read("resources/test/3x3_jpeg_recompression.jxl").unwrap();
    let reference = std::fs::read("resources/test/3x3_jpeg_recompression_reference.jpg").unwrap();
    let reconstructed =
        decode_jpeg_reconstruction(&jxl_data).expect("JPEG reconstruction should succeed for 3x3");
    assert_jpeg_match(&reconstructed, &reference, "3x3 (4:2:0)");
}

/// 16x16 JPEG with 4:2:0 subsampling.
#[test]
fn test_jpeg_reconstruction_16x16() {
    let jxl_data = std::fs::read("resources/test/test_16x16_jpeg_recompression.jxl").unwrap();
    let reference = std::fs::read("resources/test/test_16x16.jpg").unwrap();
    let reconstructed = decode_jpeg_reconstruction(&jxl_data)
        .expect("JPEG reconstruction should succeed for 16x16");
    assert_jpeg_match(&reconstructed, &reference, "16x16 (4:2:0)");
}

/// 8x8 JPEG with 4:4:4 subsampling (all components 1x1).
#[test]
fn test_jpeg_reconstruction_8x8_444() {
    let jxl_data = std::fs::read("resources/test/test_8x8_444_jpeg_recompression.jxl").unwrap();
    let reference = std::fs::read("resources/test/test_8x8_444.jpg").unwrap();
    let reconstructed = decode_jpeg_reconstruction(&jxl_data)
        .expect("JPEG reconstruction should succeed for 8x8 4:4:4");
    assert_jpeg_match(&reconstructed, &reference, "8x8 (4:4:4)");
}

/// 64x64 JPEG with 4:2:0 subsampling — multi-MCU image.
#[test]
fn test_jpeg_reconstruction_64x64_420() {
    let jxl_data = std::fs::read("resources/test/test_64x64_420_jpeg_recompression.jxl").unwrap();
    let reference = std::fs::read("resources/test/test_64x64_420.jpg").unwrap();
    let reconstructed = decode_jpeg_reconstruction(&jxl_data)
        .expect("JPEG reconstruction should succeed for 64x64 4:2:0");
    assert_jpeg_match(&reconstructed, &reference, "64x64 (4:2:0)");
}

/// 128x128 JPEG with 4:4:4 — encoded by libjxl cjxl v0.12.0.
#[test]
fn test_jpeg_reconstruction_libjxl_128x128_444() {
    let jxl_data = std::fs::read("resources/test/test_128x128_444_libjxl.jxl").unwrap();
    let reference = std::fs::read("resources/test/test_128x128_444_libjxl.jpg").unwrap();
    let reconstructed = decode_jpeg_reconstruction(&jxl_data)
        .expect("JPEG reconstruction should succeed for libjxl 128x128 4:4:4");
    assert_jpeg_match(&reconstructed, &reference, "128x128 libjxl (4:4:4)");
}

/// 128x128 JPEG with 4:2:0 — encoded by libjxl cjxl v0.12.0.
#[test]
fn test_jpeg_reconstruction_libjxl_128x128_420() {
    let jxl_data = std::fs::read("resources/test/test_128x128_420_libjxl.jxl").unwrap();
    let reference = std::fs::read("resources/test/test_128x128_420_libjxl.jpg").unwrap();
    let reconstructed = decode_jpeg_reconstruction(&jxl_data)
        .expect("JPEG reconstruction should succeed for libjxl 128x128 4:2:0");
    assert_jpeg_match(&reconstructed, &reference, "128x128 libjxl (4:2:0)");
}

/// A regular JXL file without jbrd box should not produce JPEG reconstruction.
#[test]
fn test_no_jpeg_reconstruction_for_non_jpeg_jxl() {
    let jxl_data = std::fs::read("resources/test/basic.jxl").unwrap();
    let reconstructed = decode_jpeg_reconstruction(&jxl_data);
    assert!(
        reconstructed.is_none(),
        "Non-JPEG JXL should not produce JPEG reconstruction"
    );
}
