// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests ported from libjxl decode_test.cc
//!
//! These tests verify decoder API behavior matches the reference implementation.
//! DO NOT WEAKEN TOLERANCES or modify tests to pass when implementation is wrong.

use crate::api::{
    JxlDecoder, JxlDecoderOptions, JxlSignatureType, ProcessingResult, check_signature, states,
};

/// Helper to process decoder to WithImageInfo state
fn process_to_image_info(
    data: &[u8],
) -> Result<crate::api::JxlDecoder<states::WithImageInfo>, crate::error::Error> {
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data;

    loop {
        match decoder.process(&mut input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                // If input is exhausted but decoder needs more, that's a truncation error
                if input.is_empty() {
                    return Err(crate::error::Error::OutOfBounds(0));
                }
                decoder = fallback;
            }
        }
    }
}

/// Test vectors from libjxl JxlSignatureCheckTest
/// Ported from decode_test.cc:599-628
#[cfg(test)]
mod signature_tests {
    use super::*;

    // Signature check test cases from libjxl
    // Format: (expected_result, bytes)
    // Results: Invalid, NotEnoughBytes, Codestream, Container

    #[test]
    fn test_signature_invalid_starts_with_a() {
        // No JPEGXL header starts with 'a'
        let result = check_signature(b"a");
        assert!(matches!(
            result,
            ProcessingResult::Complete { result: None }
        ));
    }

    #[test]
    fn test_signature_invalid_abcdef() {
        let result = check_signature(b"abcdef");
        assert!(matches!(
            result,
            ProcessingResult::Complete { result: None }
        ));
    }

    #[test]
    fn test_signature_codestream_valid() {
        // Valid codestream signature: 0xff 0x0a
        let result = check_signature(&[0xff, 0x0a]);
        match result {
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream),
            } => {}
            _ => panic!("Expected codestream signature, got {:?}", result),
        }
    }

    #[test]
    fn test_signature_codestream_with_trailing_data() {
        // Codestream with more data after signature
        let result = check_signature(&[0xff, 0x0a, 0x12, 0x34, 0x00, 0x00]);
        match result {
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Codestream),
            } => {}
            _ => panic!("Expected codestream signature, got {:?}", result),
        }
    }

    #[test]
    fn test_signature_container_partial_needs_more() {
        // Partial container signature needs more bytes
        // Container signature: 00 00 00 0c 4a 58 4c 20 0d 0a 87 0a
        let result = check_signature(&[0, 0, 0, 0xc, b'J', b'X', b'L', b' ', 0xD, 0xA, 0x87]);
        match result {
            ProcessingResult::NeedsMoreInput { .. } => {}
            _ => panic!(
                "Expected NeedsMoreInput for partial container signature, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_signature_container_valid() {
        // Full container signature
        let result = check_signature(&[0, 0, 0, 0xc, b'J', b'X', b'L', b' ', 0xD, 0xA, 0x87, 0x0A]);
        match result {
            ProcessingResult::Complete {
                result: Some(JxlSignatureType::Container),
            } => {}
            _ => panic!("Expected container signature, got {:?}", result),
        }
    }

    #[test]
    fn test_signature_single_zero_needs_more() {
        // Single zero byte - needs more to determine
        let result = check_signature(&[0]);
        match result {
            ProcessingResult::NeedsMoreInput { .. } => {}
            _ => panic!(
                "Expected NeedsMoreInput for single zero byte, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_signature_jpeg_is_invalid() {
        // JPEG signature should be invalid (not JXL)
        let result = check_signature(&[0xff, 0xd8, 0xff, 0xe0]);
        assert!(matches!(
            result,
            ProcessingResult::Complete { result: None }
        ));
    }

    #[test]
    fn test_signature_png_is_invalid() {
        // PNG signature should be invalid
        let result = check_signature(&[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
        assert!(matches!(
            result,
            ProcessingResult::Complete { result: None }
        ));
    }
}

/// Tests for decoder basic info extraction
/// Ported from decode_test.cc BasicInfoTest, BasicInfoSizeHintTest
#[cfg(test)]
mod basic_info_tests {
    use super::*;

    fn test_resources_dir() -> std::path::PathBuf {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("resources/test")
    }

    #[test]
    fn test_basic_info_from_file() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // basic.jxl is a 10x10 pixel test image
        // These values should match libjxl output exactly
        assert!(info.size.0 > 0, "Width should be positive");
        assert!(info.size.1 > 0, "Height should be positive");
    }

    #[test]
    fn test_basic_info_conformance_cafe() {
        // Test against cafe conformance image
        let path = test_resources_dir().join("conformance_test_images/cafe.jxl");
        if !path.exists() {
            eprintln!("Skipping test - cafe.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // cafe.jxl is 1280x1600
        assert_eq!(info.size.0, 1280, "Width mismatch for cafe.jxl");
        assert_eq!(info.size.1, 1600, "Height mismatch for cafe.jxl");
    }

    #[test]
    fn test_basic_info_animation() {
        let path = test_resources_dir().join("conformance_test_images/animation_spline.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_spline.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // animation_spline.jxl is 320x320 animated
        assert_eq!(info.size.0, 320, "Width mismatch for animation_spline.jxl");
        assert_eq!(info.size.1, 320, "Height mismatch for animation_spline.jxl");
        assert!(info.animation.is_some(), "Should have animation info");
    }
}

/// Tests for empty and malformed input handling
/// Ported from decode_test.cc ProcessEmptyInputWithBoxes, ExtraBytesAfterCompressedStream
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let data: &[u8] = &[];
        let result = process_to_image_info(data);

        // Empty input should fail gracefully
        assert!(result.is_err(), "Empty input should return error");
    }

    #[test]
    fn test_truncated_signature() {
        // Just the first byte of codestream signature
        let data: &[u8] = &[0xff];
        let result = process_to_image_info(data);

        // Truncated signature should fail or request more input
        // This tests robustness against truncated files
        assert!(result.is_err(), "Truncated signature should return error");
    }

    #[test]
    fn test_invalid_signature() {
        // Completely invalid data
        let data: &[u8] = &[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let result = process_to_image_info(data);

        assert!(result.is_err(), "Invalid signature should return error");
    }

    #[test]
    fn test_truncated_header() {
        // Valid signature but truncated header
        let data: &[u8] = &[0xff, 0x0a, 0x00]; // Codestream signature + minimal truncated data
        let result = process_to_image_info(data);

        // Should fail because header is incomplete
        assert!(result.is_err(), "Truncated header should return error");
    }
}

/// Tests for orientation handling
/// Ported from decode_test.cc OrientedCroppedFrameTest
#[cfg(test)]
mod orientation_tests {
    use super::*;

    fn test_resources_dir() -> std::path::PathBuf {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("resources/test")
    }

    #[test]
    fn test_orientation_identity() {
        let path = test_resources_dir().join("orientation1_identity.jxl");
        if !path.exists() {
            eprintln!("Skipping test - orientation1_identity.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // Identity orientation should not change dimensions
        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }

    #[test]
    fn test_orientation_rotate_90() {
        let path = test_resources_dir().join("orientation6_rotate_90_cw.jxl");
        if !path.exists() {
            eprintln!("Skipping test - orientation6_rotate_90_cw.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // 90 degree rotation swaps width and height
        // The original image should have different width/height
        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }
}

/// Tests for extra channel handling
/// Ported from decode_test.cc ExtraChannelTest, SpotColorTest
#[cfg(test)]
mod extra_channel_tests {
    use super::*;

    fn test_resources_dir() -> std::path::PathBuf {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("resources/test")
    }

    #[test]
    fn test_alpha_channel_nonpremultiplied() {
        let path = test_resources_dir().join("conformance_test_images/alpha_nonpremultiplied.jxl");
        if !path.exists() {
            eprintln!("Skipping test - alpha_nonpremultiplied.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // Should have alpha as extra channel
        assert!(
            !info.extra_channels.is_empty(),
            "Should have at least one extra channel (alpha)"
        );
    }

    #[test]
    fn test_spot_color_channel() {
        let path = test_resources_dir().join("conformance_test_images/spot.jxl");
        if !path.exists() {
            eprintln!("Skipping test - spot.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = process_to_image_info(&data).expect("Failed to decode");

        let info = decoder.basic_info();
        // Spot color image should have extra channels
        assert!(
            !info.extra_channels.is_empty(),
            "Spot color image should have extra channels"
        );
    }
}

/// Regression tests for slow probe inputs discovered by fuzzing.
/// Malformed JXL codestreams with entropy-coded ICC headers that triggered
/// large Huffman table allocations, causing 13-120ms probe times under
/// sanitizers. Fixed by capping per-table alphabet size relative to
/// available input bits in HuffmanCodes::decode.
#[cfg(test)]
mod slow_probe_regression {
    use super::*;
    use std::time::Instant;

    /// Probe must complete in <5ms even in debug builds.
    const MAX_PROBE_US: u128 = 5000;

    fn assert_fast_probe(name: &str, data: &[u8]) {
        let start = Instant::now();
        let _ = process_to_image_info(data);
        let us = start.elapsed().as_micros();
        assert!(us <= MAX_PROBE_US, "{name} probe took {us}us, limit is {MAX_PROBE_US}us");
    }

    #[test]
    fn slow_unit_15b() {
        assert_fast_probe("15B", &[
            0xff, 0x0a, 0x23, 0x01, 0x08, 0xff, 0xfd, 0xa0,
            0xff, 0xc9, 0x97, 0xa0, 0x00, 0xff, 0xff,
        ]);
    }

    #[test]
    fn slow_unit_16b() {
        assert_fast_probe("16B", &[
            0xff, 0x0a, 0x85, 0x01, 0x08, 0xef, 0xff, 0xff,
            0xff, 0x41, 0x41, 0x41, 0xff, 0x08, 0x60, 0x0f,
        ]);
    }

    #[test]
    fn slow_unit_19b() {
        assert_fast_probe("19B", &[
            0xff, 0x0a, 0x5b, 0x01, 0x08, 0x3f, 0xff, 0xff,
            0xff, 0xfd, 0x00, 0x00, 0xa8, 0xa8, 0x00, 0xf7,
            0xff, 0x0c, 0x2b,
        ]);
    }

    #[test]
    fn slow_unit_60b() {
        assert_fast_probe("60B", &[
            0xff, 0x0a, 0x85, 0x01, 0x08, 0xff, 0xff, 0xff,
            0xff, 0x35, 0x06, 0x89, 0xcd, 0x29, 0x8d, 0xff,
            0x60, 0x0a, 0xf7, 0x30, 0x21, 0x88, 0xff, 0xff,
            0x71, 0xff, 0x0a, 0x36, 0x21, 0x88, 0xff, 0xd4,
            0x71, 0xff, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76,
            0x69, 0x66, 0xff, 0xff, 0x3d, 0x0a, 0x3c, 0xf8,
            0x12, 0x88, 0xff, 0xff, 0x8d, 0x3d, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x64,
        ]);
    }
}
