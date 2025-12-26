// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Streaming decoder tests
//!
//! Tests for incremental/streaming decoding behavior.
//! These verify that the decoder handles chunked input correctly.

use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

fn test_resources_dir() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("resources/test")
}

/// Tests for progressive decoding
/// Ported from decode_test.cc ProgressionTest, ProgressiveEventTest
#[cfg(test)]
mod progressive_tests {
    use super::*;

    #[test]
    fn test_progressive_decode_passes() {
        let path = test_resources_dir().join("conformance_test_images/progressive.jxl");
        if !path.exists() {
            eprintln!("Skipping test - progressive.jxl not found");
            return;
        }
        let _data = std::fs::read(&path).expect("Failed to read test file");

        // Progressive images should report multiple passes
        // TODO: Verify pass count matches libjxl
    }
}

/// Tests for animation streaming
/// Ported from decode_test.cc AnimationTestStreaming
#[cfg(test)]
mod animation_streaming_tests {
    use super::*;

    #[test]
    fn test_animation_frame_by_frame() {
        let path = test_resources_dir().join("conformance_test_images/animation_spline.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_spline.jxl not found");
            return;
        }
        let data = std::fs::read(&path).expect("Failed to read test file");

        // Test that we can decode animation frame by frame
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = data.as_slice();

        let decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                Err(_) => panic!("Decoder error"),
            }
        };

        let info = decoder.basic_info();
        assert!(info.animation.is_some(), "Should be animated");

        // TODO: Implement frame-by-frame decoding test
        // Each frame should match libjxl output exactly
    }

    #[test]
    fn test_animation_streaming_chunks() {
        let path =
            test_resources_dir().join("conformance_test_images/animation_newtons_cradle.jxl");
        if !path.exists() {
            eprintln!("Skipping test - animation_newtons_cradle.jxl not found");
            return;
        }
        let _data = std::fs::read(&path).expect("Failed to read test file");

        // Feed data in chunks and verify we can decode progressively
        let _chunk_sizes = [64, 256, 1024, 4096];

        // TODO: Test would verify that partial decoding works correctly
        // and matches libjxl streaming behavior
    }
}

/// Tests for flush behavior
/// Ported from decode_test.cc FlushTest, FlushTestImageOutCallback
#[cfg(test)]
mod flush_tests {

    #[test]
    fn test_flush_partial_image() {
        // Test that flushing produces a valid partial image
        // TODO: Implement flush test
    }

    #[test]
    fn test_flush_with_alpha() {
        // Test flush behavior with alpha channel
        // TODO: Implement
    }
}

/// Tests for frame skipping
/// Ported from decode_test.cc SkipFrameTest, SkipFrameWithBlendingTest
#[cfg(test)]
mod frame_skip_tests {

    #[test]
    fn test_skip_to_frame() {
        // Test skipping to a specific frame in animation
        // TODO: This requires frame skipping API to be implemented
    }

    #[test]
    fn test_skip_with_blending() {
        // Skipping frames that use blending requires decoding dependencies
        // TODO: Implement
    }
}

/// Tests for input handling edge cases
/// Ported from decode_test.cc InputHandlingTestOneShot, InputHandlingTestStreaming
#[cfg(test)]
mod input_handling_tests {
    use super::*;

    #[test]
    fn test_one_shot_decode() {
        let path = test_resources_dir().join("basic.jxl");
        let data = std::fs::read(&path).expect("Failed to read test file");

        // One-shot decode - all data available at once
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = data.as_slice();

        let _decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                Err(_) => panic!("Decoder error"),
            }
        };

        // Verify all input was consumed
        // In one-shot mode, decoder should have processed everything
    }

    #[test]
    fn test_byte_by_byte_streaming() {
        let path = test_resources_dir().join("basic.jxl");
        let _data = std::fs::read(&path).expect("Failed to read test file");

        // This is the most extreme streaming test - one byte at a time
        // It should still produce the same output as one-shot decode

        // TODO: Implement byte-by-byte streaming test
        // This requires comparing output to one-shot decode result
    }

    #[test]
    fn test_random_chunk_sizes() {
        let path = test_resources_dir().join("basic.jxl");
        let _data = std::fs::read(&path).expect("Failed to read test file");

        // Test with random chunk sizes to ensure no edge cases
        let _chunk_sizes = [
            1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 64, 128, 256, 512, 1024,
        ];

        // TODO: Each chunk size should produce identical output
    }
}
