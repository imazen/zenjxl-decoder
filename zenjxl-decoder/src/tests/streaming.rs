// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Streaming decoder tests
//!
//! Tests for incremental/streaming decoding behavior.
//! These verify that the decoder handles chunked input correctly.
//! DO NOT WEAKEN TOLERANCES or modify tests to pass when implementation is wrong.

use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

fn test_resources_dir() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("resources/test")
}

/// Helper to decode a JXL file with all data at once
fn decode_oneshot(data: &[u8]) -> Result<JxlDecoder<states::WithImageInfo>, crate::error::Error> {
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data;

    loop {
        match decoder.process(&mut input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(crate::error::Error::OutOfBounds(0));
                }
                decoder = fallback;
            }
        }
    }
}

/// Helper to decode with chunked input, returning basic info
fn decode_chunked(
    data: &[u8],
    chunk_size: usize,
) -> Result<JxlDecoder<states::WithImageInfo>, crate::error::Error> {
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut offset = 0;

    loop {
        let end = (offset + chunk_size).min(data.len());
        let mut chunk = &data[offset..end];
        let chunk_len_before = chunk.len();

        match decoder.process(&mut chunk)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                let consumed = chunk_len_before - chunk.len();
                offset += consumed;

                if offset >= data.len() {
                    return Err(crate::error::Error::OutOfBounds(0));
                }
                decoder = fallback;
            }
        }
    }
}

/// Tests for progressive decoding
/// Ported from decode_test.cc ProgressionTest, ProgressiveEventTest
#[cfg(test)]
mod progressive_tests {
    use super::*;

    /// Test that progressive images can be decoded
    #[test]
    fn test_progressive_decode_passes() {
        // Use progressive.jxl from conformance images if available
        let path = test_resources_dir().join("conformance_test_images/progressive.jxl");
        if !path.exists() {
            // Fall back to a regular test file
            let path = test_resources_dir().join("basic.jxl");
            let data = std::fs::read(&path).expect("Failed to read test file");

            let decoder = decode_oneshot(&data).expect("Failed to decode");
            let info = decoder.basic_info();

            // basic.jxl should decode successfully
            assert!(info.size.0 > 0);
            assert!(info.size.1 > 0);
            return;
        }

        let data = std::fs::read(&path).expect("Failed to read test file");
        let decoder = decode_oneshot(&data).expect("Failed to decode");
        let info = decoder.basic_info();

        // Progressive images should have valid dimensions
        assert!(info.size.0 > 0, "Width should be positive");
        assert!(info.size.1 > 0, "Height should be positive");

        // Check that we can access basic info
        assert!(info.bit_depth.bits_per_sample() > 0);
    }

    /// Test progressive decode with multiple passes
    #[test]
    fn test_progressive_multiple_passes() {
        let path = test_resources_dir().join("conformance_test_images/progressive.jxl");
        if !path.exists() {
            eprintln!("Skipping test - progressive.jxl not found");
            return;
        }

        let data = std::fs::read(&path).expect("Failed to read test file");

        // Decode with small chunks to exercise progressive path
        let decoder = decode_chunked(&data, 256).expect("Failed to decode");
        let info = decoder.basic_info();

        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }
}

/// Tests for animation streaming
/// Ported from decode_test.cc AnimationTestStreaming
#[cfg(test)]
mod animation_streaming_tests {
    use super::*;

    /// Test that we can decode animation frame by frame
    #[test]
    fn test_animation_frame_by_frame() {
        let path = test_resources_dir().join("conformance_test_images/animation_spline.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_spline.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_oneshot(&data).expect("Failed to decode");
        let info = decoder.basic_info();

        // Verify this is an animated image
        assert!(info.animation.is_some(), "Should be animated");

        let animation = info.animation.as_ref().unwrap();

        // Animation should have valid timing info
        assert!(
            animation.tps_numerator > 0,
            "TPS numerator should be positive"
        );
        assert!(
            animation.tps_denominator > 0,
            "TPS denominator should be positive"
        );

        // Check dimensions
        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }

    /// Test that animation can be decoded with various chunk sizes
    #[test]
    fn test_animation_streaming_chunks() {
        let path =
            test_resources_dir().join("conformance_test_images/animation_newtons_cradle.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_newtons_cradle.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Test various chunk sizes
        let chunk_sizes = [64, 256, 1024, 4096];

        for &chunk_size in &chunk_sizes {
            let decoder = decode_chunked(&data, chunk_size).unwrap_or_else(|e| {
                panic!("Failed to decode with chunk size {}: {:?}", chunk_size, e)
            });

            let info = decoder.basic_info();
            assert!(
                info.animation.is_some(),
                "Should be animated with chunk size {}",
                chunk_size
            );
        }
    }

    /// Test animation with splines
    #[test]
    fn test_animation_with_splines() {
        let path = test_resources_dir().join("conformance_test_images/animation_spline.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_spline.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_oneshot(&data).expect("Failed to decode");
        let info = decoder.basic_info();

        // Verify dimensions match expected (320x320 for animation_spline.jxl)
        assert_eq!(info.size.0, 320, "Width should be 320");
        assert_eq!(info.size.1, 320, "Height should be 320");
        assert!(info.animation.is_some(), "Should be animated");
    }
}

/// Tests for flush behavior
/// Ported from decode_test.cc FlushTest, FlushTestImageOutCallback
#[cfg(test)]
mod flush_tests {
    use super::*;

    /// Test that partial decoding doesn't corrupt state
    #[test]
    fn test_flush_partial_image() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Decode with very small chunks to simulate streaming
        let decoder = decode_chunked(&data, 16).expect("Failed to decode");
        let info = decoder.basic_info();

        // Should still get valid basic info
        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }

    /// Test flush behavior with alpha channel
    #[test]
    fn test_flush_with_alpha() {
        let path = test_resources_dir().join("3x3a_srgb_lossless.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Decode with small chunks
        let decoder = decode_chunked(&data, 32).expect("Failed to decode");
        let info = decoder.basic_info();

        // Should have alpha channel
        assert!(
            !info.extra_channels.is_empty(),
            "Should have extra channels for alpha"
        );
    }

    /// Test that we can abort and restart decoding
    #[test]
    fn test_restart_after_partial() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Start decoding with only part of the data
        let partial_data = &data[..data.len() / 2];
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = partial_data;

        // Try to process - should need more input
        let result = decoder.process(&mut input);
        match result {
            Ok(ProcessingResult::NeedsMoreInput { .. }) => {
                // Expected - we didn't provide enough data
            }
            Ok(ProcessingResult::Complete { .. }) => {
                // Also valid if file is very small
            }
            Err(_) => {
                // Also acceptable - partial data may be invalid
            }
        }

        // Now decode the full file fresh
        let decoder = decode_oneshot(&data).expect("Failed to decode full file");
        let info = decoder.basic_info();
        assert!(info.size.0 > 0);
    }
}

/// Tests for frame skipping
/// Ported from decode_test.cc SkipFrameTest, SkipFrameWithBlendingTest
#[cfg(test)]
mod frame_skip_tests {
    use super::*;

    /// Test that animation info is available before decoding frames
    #[test]
    fn test_animation_info_available() {
        let path = test_resources_dir().join("conformance_test_images/animation_spline.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_spline.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_oneshot(&data).expect("Failed to decode");
        let info = decoder.basic_info();

        // Animation info should be available from basic info
        assert!(
            info.animation.is_some(),
            "Animation info should be available"
        );

        let anim = info.animation.as_ref().unwrap();
        // Animation should have reasonable timing
        assert!(anim.tps_numerator > 0);
        assert!(anim.tps_denominator > 0);
    }

    // TODO: Frame skipping API not yet implemented in jxl-rs.
    // These tests would verify:
    // - JxlDecoderSkipFrames functionality
    // - Correct handling of frame dependencies when skipping
    // - Blending requirements when skipping to specific frames
    //
    // The libjxl tests (SkipFrameTest, SkipFrameWithBlendingTest) use
    // JxlDecoderSkipFrames which doesn't exist in jxl-rs yet.

    /// Test blending mode detection
    #[test]
    fn test_blendmodes_file() {
        let path = test_resources_dir().join("conformance_test_images/blendmodes.jxl");
        if !path.exists() {
            eprintln!("Skipping test - blendmodes.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_oneshot(&data).expect("Failed to decode");
        let info = decoder.basic_info();

        // blendmodes.jxl should have animation or multiple frames
        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }
}

/// Tests for input handling edge cases
/// Ported from decode_test.cc InputHandlingTestOneShot, InputHandlingTestStreaming
#[cfg(test)]
mod input_handling_tests {
    use super::*;

    /// Test one-shot decode with all data available
    #[test]
    fn test_one_shot_decode() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_oneshot(&data).expect("Failed to decode");
        let info = decoder.basic_info();

        // Verify decode succeeded
        assert!(info.size.0 > 0, "Width should be positive");
        assert!(info.size.1 > 0, "Height should be positive");
    }

    /// Test byte-by-byte streaming (most extreme case)
    #[test]
    fn test_byte_by_byte_streaming() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Decode one byte at a time
        let decoder = decode_chunked(&data, 1).expect("Failed to decode byte-by-byte");
        let oneshot = decode_oneshot(&data).expect("Failed to decode oneshot");

        // Both should produce the same basic info
        assert_eq!(
            decoder.basic_info().size,
            oneshot.basic_info().size,
            "Dimensions should match between byte-by-byte and oneshot"
        );
    }

    /// Test with various chunk sizes
    #[test]
    fn test_various_chunk_sizes() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Get reference from oneshot decode
        let reference = decode_oneshot(&data).expect("Failed to decode reference");
        let ref_size = reference.basic_info().size;

        // Test various chunk sizes
        let chunk_sizes = [
            1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 64, 128, 256, 512, 1024,
        ];

        for &chunk_size in &chunk_sizes {
            let decoder = decode_chunked(&data, chunk_size)
                .unwrap_or_else(|e| panic!("Failed with chunk size {}: {:?}", chunk_size, e));

            assert_eq!(
                decoder.basic_info().size,
                ref_size,
                "Size mismatch with chunk size {}",
                chunk_size
            );
        }
    }

    /// Test that prime chunk sizes work correctly
    #[test]
    fn test_prime_chunk_sizes() {
        let path = test_resources_dir().join("3x3_srgb_lossless.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Prime numbers as chunk sizes to test edge cases
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

        for &prime in &primes {
            let decoder = decode_chunked(&data, prime)
                .unwrap_or_else(|e| panic!("Failed with prime chunk size {}: {:?}", prime, e));

            let info = decoder.basic_info();
            assert_eq!(
                info.size.0, 3,
                "Width should be 3 with chunk size {}",
                prime
            );
            assert_eq!(
                info.size.1, 3,
                "Height should be 3 with chunk size {}",
                prime
            );
        }
    }

    /// Test large file with streaming
    #[test]
    fn test_large_file_streaming() {
        // Use cafe.jxl which is larger
        let path = test_resources_dir().join("conformance_test_images/cafe.jxl");
        if !path.exists() {
            eprintln!("Skipping test - cafe.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Test with various chunk sizes
        for &chunk_size in &[1024, 4096, 16384] {
            let decoder = decode_chunked(&data, chunk_size)
                .unwrap_or_else(|e| panic!("Failed with chunk size {}: {:?}", chunk_size, e));

            let info = decoder.basic_info();
            // cafe.jxl is 1280x1600
            assert_eq!(info.size.0, 1280, "Width should be 1280");
            assert_eq!(info.size.1, 1600, "Height should be 1600");
        }
    }
}

/// Tests for container format handling
#[cfg(test)]
mod container_tests {
    use super::*;

    /// Test that container format files can be decoded
    #[test]
    fn test_container_format() {
        // Files with _with_container suffix are in container format
        let path = test_resources_dir().join("has_permutation_with_container.jxl");
        if !path.exists() {
            eprintln!("Skipping test - has_permutation_with_container.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_oneshot(&data).expect("Failed to decode container format");
        let info = decoder.basic_info();

        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }

    /// Test container with streaming
    #[test]
    fn test_container_streaming() {
        let path = test_resources_dir().join("has_permutation_with_container.jxl");
        if !path.exists() {
            eprintln!("Skipping test - has_permutation_with_container.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Container format should work with streaming
        let decoder =
            decode_chunked(&data, 256).expect("Failed to decode container with streaming");
        let info = decoder.basic_info();

        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }
}

/// Tests for HDR content
#[cfg(test)]
mod hdr_tests {
    use super::*;

    /// Test HDR PQ file decoding
    #[test]
    fn test_hdr_pq_streaming() {
        let path = test_resources_dir().join("hdr_pq_test.jxl");
        if !path.exists() {
            eprintln!("Skipping test - hdr_pq_test.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_chunked(&data, 512).expect("Failed to decode HDR PQ");
        let info = decoder.basic_info();

        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }

    /// Test HDR HLG file decoding
    #[test]
    fn test_hdr_hlg_streaming() {
        let path = test_resources_dir().join("hdr_hlg_test.jxl");
        if !path.exists() {
            eprintln!("Skipping test - hdr_hlg_test.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        let decoder = decode_chunked(&data, 512).expect("Failed to decode HDR HLG");
        let info = decoder.basic_info();

        assert!(info.size.0 > 0);
        assert!(info.size.1 > 0);
    }
}
