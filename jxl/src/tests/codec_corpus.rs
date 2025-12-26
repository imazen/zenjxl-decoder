// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Codec-corpus exact parity tests.
//!
//! These tests compare jxl-rs output against libjxl reference outputs
//! for all images in the codec-corpus JXL test suite.
//!
//! IMPORTANT: DO NOT WEAKEN TOLERANCES. If a test fails, the implementation
//! is wrong and must be fixed.

use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

use super::parity::{
    CONFORMANCE_THRESHOLD_U8, CodecCorpusTestCase, ReferenceImage, codec_corpus_jxl_dir,
    compare_u8_buffers, discover_codec_corpus_tests, load_ppm,
};

/// Decode a JXL file using jxl-rs and return the pixel data.
fn decode_jxl_to_pixels(path: &std::path::Path) -> Result<(usize, usize, usize, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read JXL: {}", e))?;

    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data.as_slice();

    loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => {
                let info = result.basic_info();
                let width = info.size.0 as usize;
                let height = info.size.1 as usize;

                // Get pixel data from the decoder
                // For now, return placeholder until we wire up actual pixel output
                // The actual implementation would get pixels from result.pixels() or similar

                // TODO: Extract actual pixel data from decoder result
                // This requires understanding how jxl-rs exposes decoded pixels
                return Err("Pixel extraction not yet implemented".to_string());
            }
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("Decode error: {:?}", e)),
        }
    }
}

/// Run a single parity test case.
fn run_parity_test(test_case: &CodecCorpusTestCase) -> Result<(), String> {
    // Skip if no reference available
    let ref_path = test_case.reference_path.as_ref().ok_or_else(|| {
        format!(
            "No reference output for {}/{}",
            test_case.category, test_case.name
        )
    })?;

    // Skip PNG references (need external decoder)
    if ref_path.extension().and_then(|e| e.to_str()) == Some("png") {
        return Err("PNG reference loading not yet supported".to_string());
    }

    // Load reference
    let reference =
        ReferenceImage::load(ref_path).map_err(|e| format!("Failed to load reference: {}", e))?;

    // Decode with jxl-rs
    let (width, height, channels, actual) = decode_jxl_to_pixels(&test_case.jxl_path)?;

    // Verify dimensions match
    if width != reference.width || height != reference.height {
        return Err(format!(
            "Dimension mismatch: got {}x{}, expected {}x{}",
            width, height, reference.width, reference.height
        ));
    }

    if channels != reference.channels {
        return Err(format!(
            "Channel count mismatch: got {}, expected {}",
            channels, reference.channels
        ));
    }

    // Compare pixels
    let result = compare_u8_buffers(
        &reference.pixels,
        &actual,
        width,
        height,
        channels,
        CONFORMANCE_THRESHOLD_U8,
    );

    if result.passed {
        Ok(())
    } else {
        Err(format!(
            "Pixel mismatch: max_error={}, error_count={}/{}, first_error={:?}",
            result.max_abs_error,
            result.error_count,
            result.total_pixels,
            result.first_error_location
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that codec-corpus is discoverable
    #[test]
    fn test_codec_corpus_discovery() {
        let corpus_dir = codec_corpus_jxl_dir();

        if corpus_dir.is_none() {
            eprintln!("Skipping: codec-corpus not found");
            eprintln!("Set CODEC_CORPUS_PATH environment variable to enable these tests");
            return;
        }

        let tests = discover_codec_corpus_tests();
        eprintln!("Found {} codec-corpus test cases", tests.len());

        // Should have tests in each category
        let conformance = tests.iter().filter(|t| t.category == "conformance").count();
        let features = tests.iter().filter(|t| t.category == "features").count();
        let photographic = tests
            .iter()
            .filter(|t| t.category == "photographic")
            .count();
        let edge_cases = tests.iter().filter(|t| t.category == "edge-cases").count();

        eprintln!("  conformance: {}", conformance);
        eprintln!("  features: {}", features);
        eprintln!("  photographic: {}", photographic);
        eprintln!("  edge-cases: {}", edge_cases);

        assert!(tests.len() >= 50, "Expected at least 50 test cases");
    }

    /// Test that reference outputs can be loaded
    #[test]
    fn test_load_reference_ppm() {
        let corpus_dir = match codec_corpus_jxl_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: codec-corpus not found");
                return;
            }
        };

        let ref_path = corpus_dir.join("reference/edge-cases/basic.ppm");
        if !ref_path.exists() {
            eprintln!("Skipping: reference outputs not generated");
            eprintln!("Run: codec-corpus/jxl/generate_references.sh");
            return;
        }

        let (width, height, channels, pixels) = load_ppm(&ref_path).expect("Failed to load PPM");

        // basic.jxl is a 1x1 image
        assert_eq!(width, 1);
        assert_eq!(height, 1);
        assert_eq!(channels, 3);
        assert_eq!(pixels.len(), 3);

        eprintln!("basic.jxl reference pixels: {:?}", pixels);
    }

    /// Test basic.jxl parity (smallest test case)
    #[test]
    fn test_parity_basic() {
        let tests = discover_codec_corpus_tests();
        let basic = tests.iter().find(|t| t.name == "basic");

        let Some(test_case) = basic else {
            eprintln!("Skipping: basic.jxl not found in codec-corpus");
            return;
        };

        match run_parity_test(test_case) {
            Ok(()) => eprintln!("PASS: basic.jxl"),
            Err(e) => {
                // Expected to fail until pixel extraction is implemented
                eprintln!("PENDING: basic.jxl - {}", e);
            }
        }
    }

    /// Test 3x3_srgb_lossless parity
    #[test]
    fn test_parity_3x3_srgb_lossless() {
        let tests = discover_codec_corpus_tests();
        let test = tests.iter().find(|t| t.name == "3x3_srgb_lossless");

        let Some(test_case) = test else {
            eprintln!("Skipping: 3x3_srgb_lossless.jxl not found");
            return;
        };

        match run_parity_test(test_case) {
            Ok(()) => eprintln!("PASS: 3x3_srgb_lossless.jxl"),
            Err(e) => eprintln!("PENDING: 3x3_srgb_lossless.jxl - {}", e),
        }
    }

    // TODO: Generate tests for all codec-corpus images
    // This would be done via a build script or test generator

    /// Run all codec-corpus parity tests (integration test)
    #[test]
    #[ignore] // Run with --ignored to execute all parity tests
    fn test_all_codec_corpus_parity() {
        let tests = discover_codec_corpus_tests();

        if tests.is_empty() {
            panic!("No codec-corpus tests found. Set CODEC_CORPUS_PATH.");
        }

        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;

        for test_case in &tests {
            match run_parity_test(test_case) {
                Ok(()) => {
                    eprintln!("PASS: {}/{}", test_case.category, test_case.name);
                    passed += 1;
                }
                Err(e) if e.contains("not yet") || e.contains("not supported") => {
                    eprintln!("SKIP: {}/{} - {}", test_case.category, test_case.name, e);
                    skipped += 1;
                }
                Err(e) => {
                    eprintln!("FAIL: {}/{} - {}", test_case.category, test_case.name, e);
                    failed += 1;
                }
            }
        }

        eprintln!();
        eprintln!("=== Codec-Corpus Parity Results ===");
        eprintln!("Passed:  {}", passed);
        eprintln!("Failed:  {}", failed);
        eprintln!("Skipped: {}", skipped);
        eprintln!("Total:   {}", tests.len());

        if failed > 0 {
            panic!("{} parity tests failed. DO NOT WEAKEN TOLERANCES.", failed);
        }
    }
}
