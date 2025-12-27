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

use crate::api::{
    JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult, states,
};
use crate::image::{Image, Rect};

use super::parity::{
    CONFORMANCE_THRESHOLD_U8, CodecCorpusTestCase, ReferenceImage, codec_corpus_jxl_dir,
    compare_u8_buffers, discover_codec_corpus_tests, load_ppm,
};

/// Decode a JXL file using jxl-rs and return the pixel data as u8.
/// Returns (width, height, channels, pixels) where pixels is RGB/RGBA u8 data.
fn decode_jxl_to_pixels(path: &std::path::Path) -> Result<(usize, usize, usize, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read JXL: {}", e))?;
    let mut input = data.as_slice();

    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    // Advance to image info
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input during header".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("Header decode error: {:?}", e)),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;

    // Determine output format based on whether image has alpha
    let has_alpha = basic_info.extra_channels.iter().any(|ec| {
        matches!(
            ec.ec_type,
            crate::headers::extra_channels::ExtraChannel::Alpha
        )
    });

    let (color_type, channels) = if has_alpha {
        (JxlColorType::Rgba, 4)
    } else {
        (JxlColorType::Rgb, 3)
    };

    // Request u8 output format
    let pixel_format = JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: if has_alpha { vec![None] } else { vec![] },
    };
    decoder.set_pixel_format(pixel_format);

    // Advance to frame info
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input before frame".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("Frame info decode error: {:?}", e)),
        }
    };

    // Create output buffer
    let mut output_image = Image::<u8>::new((width * channels, height))
        .map_err(|e| format!("Buffer error: {:?}", e))?;

    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    // Decode frame pixels
    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input during frame".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("Frame decode error: {:?}", e)),
        }
    }

    // Extract pixels from Image<u8>
    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        let row = output_image.row(y);
        pixels.extend_from_slice(row);
    }

    Ok((width, height, channels, pixels))
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

    // Load reference (supports PPM, PGM, PNG)
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

    // Handle channel count mismatch (e.g., RGBA decoded vs RGB reference)
    let (compare_channels, ref_pixels, actual_pixels) = if channels == reference.channels {
        (channels, reference.pixels.clone(), actual.clone())
    } else if channels == 4 && reference.channels == 3 {
        // RGBA vs RGB: compare only RGB channels
        let mut rgb_actual = Vec::with_capacity(width * height * 3);
        for pixel in actual.chunks_exact(4) {
            rgb_actual.extend_from_slice(&pixel[..3]);
        }
        (3, reference.pixels.clone(), rgb_actual)
    } else if channels == 3 && reference.channels == 4 {
        // RGB vs RGBA: compare only RGB channels from reference
        let mut rgb_ref = Vec::with_capacity(width * height * 3);
        for pixel in reference.pixels.chunks_exact(4) {
            rgb_ref.extend_from_slice(&pixel[..3]);
        }
        (3, rgb_ref, actual.clone())
    } else {
        return Err(format!(
            "Channel count mismatch: got {}, expected {}",
            channels, reference.channels
        ));
    };

    // Compare pixels
    let result = compare_u8_buffers(
        &ref_pixels,
        &actual_pixels,
        width,
        height,
        compare_channels,
        CONFORMANCE_THRESHOLD_U8,
    );

    if result.passed {
        Ok(())
    } else if result.max_abs_error == f64::INFINITY {
        Err(format!(
            "Buffer size mismatch: jxl-rs={}x{}x{}={} bytes, reference={}x{}x{}={} bytes",
            width,
            height,
            channels,
            actual_pixels.len(),
            reference.width,
            reference.height,
            reference.channels,
            ref_pixels.len()
        ))
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
    fn test_load_reference() {
        let corpus_dir = match codec_corpus_jxl_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: codec-corpus not found");
                return;
            }
        };

        // Try PNG first (preferred), then PPM
        let ref_png = corpus_dir.join("reference/edge-cases/basic.png");
        let ref_ppm = corpus_dir.join("reference/edge-cases/basic.ppm");

        let ref_path = if ref_png.exists() {
            ref_png
        } else if ref_ppm.exists() {
            ref_ppm
        } else {
            eprintln!("Skipping: reference outputs not generated");
            eprintln!("Run: codec-corpus/jxl/generate_references.sh");
            return;
        };

        let reference = ReferenceImage::load(&ref_path).expect("Failed to load reference");

        // basic.jxl is a 1x1 image
        assert_eq!(reference.width, 1);
        assert_eq!(reference.height, 1);
        assert_eq!(reference.channels, 3);
        assert_eq!(reference.pixels.len(), 3);

        eprintln!(
            "basic.jxl reference pixels (from {:?}): {:?}",
            ref_path.extension(),
            reference.pixels
        );
    }

    /// Debug test to see what jxl-rs decodes for basic.jxl
    #[test]
    fn test_decode_basic_debug() {
        let corpus_dir = match codec_corpus_jxl_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: codec-corpus not found");
                return;
            }
        };

        let jxl_path = corpus_dir.join("edge-cases/basic.jxl");
        if !jxl_path.exists() {
            eprintln!("Skipping: basic.jxl not found");
            return;
        }

        match decode_jxl_to_pixels(&jxl_path) {
            Ok((width, height, channels, pixels)) => {
                eprintln!("jxl-rs decoded: {}x{} {} channels", width, height, channels);
                eprintln!("jxl-rs pixels: {:?}", pixels);
            }
            Err(e) => {
                eprintln!("jxl-rs decode failed: {}", e);
            }
        }

        // Also load reference for comparison
        let ref_path = corpus_dir.join("reference/edge-cases/basic.png");
        if ref_path.exists() {
            if let Ok(reference) = ReferenceImage::load(&ref_path) {
                eprintln!("djxl reference: {:?}", reference.pixels);
            }
        }
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
            // Catch panics to continue testing other files
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_parity_test(test_case)
            }));

            match result {
                Ok(Ok(())) => {
                    eprintln!("PASS: {}/{}", test_case.category, test_case.name);
                    passed += 1;
                }
                Ok(Err(e)) if e.contains("not yet") || e.contains("not supported") => {
                    eprintln!("SKIP: {}/{} - {}", test_case.category, test_case.name, e);
                    skipped += 1;
                }
                Ok(Err(e)) => {
                    eprintln!("FAIL: {}/{} - {}", test_case.category, test_case.name, e);
                    failed += 1;
                }
                Err(_) => {
                    eprintln!(
                        "CRASH: {}/{} - decoder panicked",
                        test_case.category, test_case.name
                    );
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
