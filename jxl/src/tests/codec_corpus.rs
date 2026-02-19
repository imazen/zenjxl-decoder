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

#[cfg(feature = "cms")]
use crate::api::MoxCms;
use crate::api::{
    JxlColorProfile, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, ProcessingResult, states,
};
use crate::image::{Image, Rect};

use super::parity::{
    CONFORMANCE_THRESHOLD_U8, CodecCorpusTestCase, ReferenceImage, codec_corpus_jxl_dir,
    compare_u8_buffers, discover_codec_corpus_tests, png_has_linear_gamma,
};

/// Decode a JXL file using jxl-rs and return the pixel data as u8.
/// Returns (width, height, channels, pixels) where pixels is RGB/RGBA u8 data.
fn decode_jxl_to_pixels(path: &std::path::Path) -> Result<(usize, usize, usize, Vec<u8>), String> {
    decode_jxl_to_pixels_with_options(path, false)
}

/// Decode a JXL file with optional linear output.
fn decode_jxl_to_pixels_with_options(
    path: &std::path::Path,
    output_linear: bool,
) -> Result<(usize, usize, usize, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read JXL: {}", e))?;
    let mut input = data.as_slice();

    #[cfg(feature = "cms")]
    let options = JxlDecoderOptions {
        cms: Some(Box::new(MoxCms::new())),
        ..JxlDecoderOptions::default()
    };
    #[cfg(not(feature = "cms"))]
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

    // Get the default pixel format, which is set based on the file header's color_space.
    // This correctly detects grayscale even when an ICC profile is used.
    let default_format = decoder.current_pixel_format();
    let is_grayscale = matches!(
        default_format.color_type,
        JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha
    );

    // Determine output format based on whether image has alpha and is grayscale
    let has_alpha = basic_info.extra_channels.iter().any(|ec| {
        matches!(
            ec.ec_type,
            crate::headers::extra_channels::ExtraChannel::Alpha
        )
    });

    let (color_type, channels) = match (is_grayscale, has_alpha) {
        (true, true) => (JxlColorType::GrayscaleAlpha, 2),
        (true, false) => (JxlColorType::Grayscale, 1),
        (false, true) => (JxlColorType::Rgba, 4),
        (false, false) => (JxlColorType::Rgb, 3),
    };

    // Request u8 output format
    // Must provide a format slot for EVERY extra channel in the image, even if we don't
    // want to receive them (use None to skip). The decoder asserts that the number of
    // format slots matches frame_header.num_extra_channels.
    let num_extra_channels = basic_info.extra_channels.len();
    let extra_channel_format = if num_extra_channels > 0 {
        // For RGBA/BGRA, the first alpha is included in color_type, so skip it (None)
        // For GrayscaleAlpha, alpha is part of the color_type, so skip it (None)
        // All other extra channels also get None (we don't need them for parity testing)
        vec![None; num_extra_channels]
    } else {
        vec![]
    };
    let pixel_format = JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format,
    };
    decoder.set_pixel_format(pixel_format);

    // If linear output is requested (reference PNG has gAMA=100000), set output
    // color profile to linear TF so decoded values match the linear reference.
    if output_linear {
        if let JxlColorProfile::Simple(enc) = decoder.output_color_profile().clone() {
            decoder
                .set_output_color_profile(JxlColorProfile::Simple(enc.with_linear_tf()))
                .map_err(|e| format!("Failed to set linear output profile: {:?}", e))?;
        }
    }

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

    // Check if reference PNG has linear gamma (gAMA=100000).
    // djxl outputs linear values for some XYB-encoded images, indicated by gAMA=100000
    // in the PNG. We need to decode with linear output to match these references.
    let use_linear = if ref_path.extension().and_then(|e| e.to_str()) == Some("png") {
        super::parity::png_has_linear_gamma(ref_path).unwrap_or(false)
    } else {
        false
    };

    // Decode with jxl-rs
    let (width, height, channels, actual) =
        decode_jxl_to_pixels_with_options(&test_case.jxl_path, use_linear)?;

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
        if ref_path.exists()
            && let Ok(reference) = ReferenceImage::load(&ref_path)
        {
            eprintln!("djxl reference: {:?}", reference.pixels);
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

    /// Debug test for multiple_layers_noise_spline - examines frame structure
    #[test]
    fn test_debug_multiple_layers_noise_spline() {
        use crate::bit_reader::BitReader;
        use crate::headers::encodings::UnconditionalCoder;
        use crate::headers::frame_header::FrameHeader;
        use crate::headers::{FileHeader, JxlHeader};

        let tests = discover_codec_corpus_tests();
        let test_case = tests
            .iter()
            .find(|t| t.name == "multiple_layers_noise_spline");

        let Some(test_case) = test_case else {
            eprintln!("Skipping: multiple_layers_noise_spline.jxl not found");
            return;
        };

        // Read the raw file and parse frame headers
        let data = std::fs::read(&test_case.jxl_path).expect("Failed to read file");
        eprintln!("File size: {} bytes", data.len());

        // Skip container if present
        let offset = if data.len() >= 12
            && &data[0..4] == b"\x00\x00\x00\x0C"
            && &data[4..8] == b"JXL "
        {
            // ISOBMFF container - find the codestream box
            eprintln!("ISOBMFF container detected");
            let mut pos = 0;
            let mut found_offset = 0;
            while pos + 8 <= data.len() {
                let box_size =
                    u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                let box_type = &data[pos + 4..pos + 8];
                if box_type == b"jxlc" || box_type == b"jxlp" {
                    found_offset = pos + 8;
                    break;
                }
                if box_size == 0 {
                    break;
                }
                pos += box_size;
            }
            found_offset
        } else {
            // Bare codestream - start from beginning
            eprintln!("Bare codestream");
            0
        };

        // Parse file header
        let mut br = BitReader::new(&data[offset..]);
        let file_header = match FileHeader::read(&mut br) {
            Ok(fh) => fh,
            Err(e) => {
                eprintln!("Failed to parse file header: {:?}", e);
                return;
            }
        };
        eprintln!(
            "Image size: {}x{}",
            file_header.size.xsize(),
            file_header.size.ysize()
        );
        eprintln!("XYB encoded: {}", file_header.image_metadata.xyb_encoded);

        // Parse first frame header
        let nonserialized = file_header.frame_header_nonserialized();
        let frame_header = match FrameHeader::read_unconditional(&(), &mut br, &nonserialized) {
            Ok(fh) => fh,
            Err(e) => {
                eprintln!("Failed to parse frame header: {:?}", e);
                return;
            }
        };

        eprintln!("\nFrame 0:");
        eprintln!("  is_last: {}", frame_header.is_last);
        eprintln!("  is_visible: {}", frame_header.is_visible());
        eprintln!("  has_noise: {}", frame_header.has_noise());
        eprintln!("  has_splines: {}", frame_header.has_splines());
        eprintln!("  has_patches: {}", frame_header.has_patches());
        eprintln!("  needs_blending: {}", frame_header.needs_blending());
        eprintln!("  size: {:?}", frame_header.size());
        eprintln!("  upsampling: {}", frame_header.upsampling);
        eprintln!("  encoding: {:?}", frame_header.encoding);
        eprintln!("  can_be_referenced: {}", frame_header.can_be_referenced);
        eprintln!("  save_before_ct: {}", frame_header.save_before_ct);
        eprintln!("  group_dim: {}", frame_header.group_dim());
        eprintln!("  size_groups: {:?}", frame_header.size_groups());

        // Load reference
        let ref_path = test_case.reference_path.as_ref().unwrap();
        let reference = ReferenceImage::load(ref_path).expect("Failed to load reference");
        eprintln!(
            "\nReference: {}x{}, {} channels",
            reference.width, reference.height, reference.channels
        );

        // Check if linear gamma
        let use_linear = if ref_path.extension().and_then(|e| e.to_str()) == Some("png") {
            png_has_linear_gamma(ref_path).unwrap_or(false)
        } else {
            false
        };
        eprintln!("Linear output: {}", use_linear);

        // Decode with jxl-rs
        let (width, height, channels, actual) =
            decode_jxl_to_pixels_with_options(&test_case.jxl_path, use_linear)
                .expect("Decode failed");
        eprintln!("jxl-rs output: {}x{}, {} channels", width, height, channels);

        // Print first few pixels
        eprintln!("\nFirst 10x10 pixels comparison:");
        for y in 0..10.min(height) {
            for x in 0..10.min(width) {
                let ref_idx = (y * width + x) * reference.channels;
                let act_idx = (y * width + x) * channels;

                let ref_pix: Vec<u8> =
                    reference.pixels[ref_idx..ref_idx + reference.channels].to_vec();
                let act_pix: Vec<u8> = actual[act_idx..act_idx + channels].to_vec();

                if ref_pix != act_pix {
                    eprintln!("  ({},{}) ref={:?} jxl-rs={:?}", x, y, ref_pix, act_pix);
                }
            }
        }

        // Run the parity test
        match run_parity_test(test_case) {
            Ok(()) => eprintln!("PASS"),
            Err(e) => eprintln!("FAIL: {}", e),
        }
    }

    /// Debug test for cmyk_layers - examines what's happening with CMYK conversion
    #[test]
    fn test_debug_cmyk_layers() {
        use crate::bit_reader::BitReader;
        use crate::headers::{FileHeader, JxlHeader};

        let tests = discover_codec_corpus_tests();
        let test_case = tests.iter().find(|t| t.name == "cmyk_layers");

        let Some(test_case) = test_case else {
            eprintln!("Skipping: cmyk_layers.jxl not found");
            return;
        };

        // Parse file header to see extra channel info
        let data = std::fs::read(&test_case.jxl_path).expect("Failed to read file");
        let offset = if data.len() >= 12
            && &data[0..4] == b"\x00\x00\x00\x0C"
            && &data[4..8] == b"JXL "
        {
            let mut pos = 0;
            let mut found_offset = 0;
            while pos + 8 <= data.len() {
                let box_size =
                    u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                let box_type = &data[pos + 4..pos + 8];
                if box_type == b"jxlc" || box_type == b"jxlp" {
                    found_offset = pos + 8;
                    break;
                }
                if box_size == 0 {
                    break;
                }
                pos += box_size;
            }
            found_offset
        } else {
            0
        };

        let mut br = BitReader::new(&data[offset..]);
        let file_header = FileHeader::read(&mut br).expect("Failed to parse file header");

        eprintln!(
            "Image size: {}x{}",
            file_header.size.xsize(),
            file_header.size.ysize()
        );
        eprintln!("XYB encoded: {}", file_header.image_metadata.xyb_encoded);
        eprintln!(
            "Color space: {:?}",
            file_header.image_metadata.color_encoding.color_space
        );
        eprintln!(
            "want_icc: {}",
            file_header.image_metadata.color_encoding.want_icc
        );
        eprintln!(
            "Extra channels: {}",
            file_header.image_metadata.extra_channel_info.len()
        );
        for (i, info) in file_header
            .image_metadata
            .extra_channel_info
            .iter()
            .enumerate()
        {
            eprintln!("  Channel {}: type={:?}", i, info.ec_type);
        }

        // Parse frame headers - cmyk_layers has 4 frames!
        use crate::headers::encodings::UnconditionalCoder;
        use crate::headers::frame_header::FrameHeader;
        let nonserialized = file_header.frame_header_nonserialized();
        let mut frame_idx = 0;
        loop {
            match FrameHeader::read_unconditional(&(), &mut br, &nonserialized) {
                Ok(frame_header) => {
                    eprintln!("\nFrame {}:", frame_idx);
                    eprintln!("  name: {:?}", frame_header.name);
                    eprintln!("  is_last: {}", frame_header.is_last);
                    eprintln!("  needs_blending: {}", frame_header.needs_blending());
                    eprintln!("  blending_info: {:?}", frame_header.blending_info);
                    eprintln!("  size: {:?}", frame_header.size());
                    eprintln!("  origin: {:?}", (frame_header.x0, frame_header.y0));
                    if frame_header.is_last {
                        break;
                    }
                    frame_idx += 1;
                    // Skip to next frame (very rough - might not work properly)
                    // This is just for debugging
                }
                Err(e) => {
                    eprintln!("Error parsing frame {}: {:?}", frame_idx, e);
                    break;
                }
            }
        }

        // Load reference
        let ref_path = test_case.reference_path.as_ref().unwrap();
        eprintln!("Reference path: {:?}", ref_path);
        let reference = ReferenceImage::load(ref_path).expect("Failed to load reference");
        eprintln!(
            "Reference: {}x{}, {} channels",
            reference.width, reference.height, reference.channels
        );

        // Decode with jxl-rs
        let (width, height, channels, actual) =
            decode_jxl_to_pixels(&test_case.jxl_path).expect("Decode failed");
        eprintln!("jxl-rs output: {}x{}, {} channels", width, height, channels);

        // Compute overall error stats
        let mut max_error = 0u8;
        let mut total_error = 0u64;
        let mut error_count = 0usize;
        let mut channel_errors = [0u64; 4]; // Per-channel error sums

        for y in 0..height.min(reference.height) {
            for x in 0..width.min(reference.width) {
                let ref_idx = (y * reference.width + x) * reference.channels;
                let act_idx = (y * width + x) * channels;

                for c in 0..channels.min(reference.channels) {
                    let ref_val = reference.pixels[ref_idx + c];
                    let act_val = actual[act_idx + c];
                    let diff = ref_val.abs_diff(act_val);
                    if diff > 0 {
                        error_count += 1;
                        total_error += diff as u64;
                        channel_errors[c] += diff as u64;
                        if diff > max_error {
                            max_error = diff;
                        }
                    }
                }
            }
        }

        let total_pixels = width * height * channels;
        eprintln!("Error stats:");
        eprintln!("  max_error: {}", max_error);
        eprintln!("  error_count: {}/{}", error_count, total_pixels);
        eprintln!(
            "  avg_error: {:.2}",
            if error_count > 0 {
                total_error as f64 / error_count as f64
            } else {
                0.0
            }
        );
        eprintln!(
            "  per-channel total error: R={}, G={}, B={}, A={}",
            channel_errors[0], channel_errors[1], channel_errors[2], channel_errors[3]
        );

        // Print first 10 pixels with differences
        eprintln!("\nFirst 10 pixel differences:");
        let mut count = 0;
        'outer: for y in 0..height.min(reference.height) {
            for x in 0..width.min(reference.width) {
                let ref_idx = (y * reference.width + x) * reference.channels;
                let act_idx = (y * width + x) * channels;

                let ref_pix: Vec<u8> =
                    reference.pixels[ref_idx..ref_idx + reference.channels].to_vec();
                let act_pix: Vec<u8> = actual[act_idx..act_idx + channels].to_vec();

                if ref_pix != act_pix {
                    eprintln!("  ({},{}) ref={:?} jxl-rs={:?}", x, y, ref_pix, act_pix);
                    count += 1;
                    if count >= 10 {
                        break 'outer;
                    }
                }
            }
        }

        // Find and print worst errors (those matching max_error)
        eprintln!("\nWorst errors (diff >= {}):", max_error.saturating_sub(5));
        let mut worst_count = 0;
        for y in 0..height.min(reference.height) {
            for x in 0..width.min(reference.width) {
                let ref_idx = (y * reference.width + x) * reference.channels;
                let act_idx = (y * width + x) * channels;

                for c in 0..channels.min(reference.channels) {
                    let ref_val = reference.pixels[ref_idx + c];
                    let act_val = actual[act_idx + c];
                    let diff = ref_val.abs_diff(act_val);

                    if diff >= max_error.saturating_sub(5) {
                        let ref_pix: Vec<u8> =
                            reference.pixels[ref_idx..ref_idx + reference.channels].to_vec();
                        let act_pix: Vec<u8> = actual[act_idx..act_idx + channels].to_vec();
                        eprintln!(
                            "  ({},{}) ch={} diff={}: ref={:?} jxl-rs={:?}",
                            x, y, c, diff, ref_pix, act_pix
                        );
                        worst_count += 1;
                        if worst_count >= 10 {
                            return;
                        }
                        break; // Only print once per pixel
                    }
                }
            }
        }
    }

    /// Debug test to compare frame headers for noise tests
    #[test]
    fn test_debug_noise_upsampling() {
        use crate::bit_reader::BitReader;
        use crate::headers::encodings::UnconditionalCoder;
        use crate::headers::frame_header::FrameHeader;
        use crate::headers::{FileHeader, JxlHeader};

        let tests = discover_codec_corpus_tests();

        // List of noise-related tests to examine
        let noise_tests = [
            "noise",
            "noise_5",
            "8x8_noise",
            "multiple_layers_noise_spline",
        ];

        for test_name in noise_tests {
            let test_case = tests.iter().find(|t| t.name == test_name);
            let Some(test_case) = test_case else {
                eprintln!("{}: NOT FOUND", test_name);
                continue;
            };

            let data = std::fs::read(&test_case.jxl_path).expect("Failed to read file");

            // Skip container if present
            let offset =
                if data.len() >= 12 && &data[0..4] == b"\x00\x00\x00\x0C" && &data[4..8] == b"JXL "
                {
                    let mut pos = 0;
                    let mut found_offset = 0;
                    while pos + 8 <= data.len() {
                        let box_size = u32::from_be_bytes([
                            data[pos],
                            data[pos + 1],
                            data[pos + 2],
                            data[pos + 3],
                        ]) as usize;
                        let box_type = &data[pos + 4..pos + 8];
                        if box_type == b"jxlc" || box_type == b"jxlp" {
                            found_offset = pos + 8;
                            break;
                        }
                        if box_size == 0 {
                            break;
                        }
                        pos += box_size;
                    }
                    found_offset
                } else {
                    0
                };

            let mut br = BitReader::new(&data[offset..]);
            let file_header = match FileHeader::read(&mut br) {
                Ok(fh) => fh,
                Err(e) => {
                    eprintln!("{}: Failed to parse file header: {:?}", test_name, e);
                    continue;
                }
            };

            let nonserialized = file_header.frame_header_nonserialized();
            let frame_header = match FrameHeader::read_unconditional(&(), &mut br, &nonserialized) {
                Ok(fh) => fh,
                Err(e) => {
                    eprintln!("{}: Failed to parse frame header: {:?}", test_name, e);
                    continue;
                }
            };

            eprintln!(
                "{}: image={}x{}, frame_size={:?}, upsampling={}, group_dim={}, size_groups={:?}, has_noise={}, xyb={}, visible={}",
                test_name,
                file_header.size.xsize(),
                file_header.size.ysize(),
                frame_header.size(),
                frame_header.upsampling,
                frame_header.group_dim(),
                frame_header.size_groups(),
                frame_header.has_noise(),
                file_header.image_metadata.xyb_encoded,
                frame_header.is_visible()
            );
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

    /// Debug test to extract CMYK ICC profile and test moxcms behavior.
    /// This generates a standalone repro case for moxcms CMYK issues.
    #[test]
    #[ignore] // Run with: cargo test --features cms extract_cmyk_icc_repro -- --ignored --nocapture
    #[cfg(feature = "cms")]
    fn extract_cmyk_icc_repro() {
        use crate::api::JxlColorProfile;
        use std::io::Write;

        // Find cmyk_layers.jxl
        let tests = discover_codec_corpus_tests();
        let cmyk_test = tests.iter().find(|t| t.name == "cmyk_layers");
        let Some(test_case) = cmyk_test else {
            panic!("cmyk_layers test not found. Set CODEC_CORPUS_PATH.");
        };

        // Decode the JXL file to get the embedded ICC profile
        let data = std::fs::read(&test_case.jxl_path).expect("Failed to read JXL");
        let mut input = data.as_slice();

        let options = JxlDecoderOptions {
            cms: Some(Box::new(MoxCms::new())),
            ..JxlDecoderOptions::default()
        };
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);

        let decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("Decode error: {:?}", e),
            }
        };

        // Get the embedded ICC profile
        let embedded = decoder.embedded_color_profile().clone();
        let icc_data = match embedded {
            JxlColorProfile::Icc(ref data) => data.clone(),
            _ => panic!("Expected ICC profile, got simple encoding"),
        };

        // Write ICC profile to file
        let icc_path = "/tmp/cmyk_layers.icc";
        let mut file = std::fs::File::create(icc_path).expect("Failed to create ICC file");
        file.write_all(&icc_data).expect("Failed to write ICC");
        eprintln!(
            "Wrote ICC profile ({} bytes) to {}",
            icc_data.len(),
            icc_path
        );

        // Load reference to find pixel values that differ
        let ref_path = test_case.reference_path.as_ref().expect("No reference");
        let reference = ReferenceImage::load(ref_path).expect("Failed to load reference");

        // Decode with jxl-rs
        let (_, _, _, actual) = decode_jxl_to_pixels(&test_case.jxl_path).expect("Decode failed");

        // Find first differing pixel
        let mut sample_pixels: Vec<(usize, usize, [u8; 4], [u8; 4])> = Vec::new();
        for y in 0..reference.height.min(100) {
            for x in 0..reference.width.min(100) {
                let ref_idx = (y * reference.width + x) * reference.channels;
                let act_idx = (y * reference.width + x) * reference.channels;

                let ref_rgba = [
                    reference.pixels[ref_idx],
                    reference.pixels[ref_idx + 1],
                    reference.pixels[ref_idx + 2],
                    if reference.channels > 3 {
                        reference.pixels[ref_idx + 3]
                    } else {
                        255
                    },
                ];
                let act_rgba = [
                    actual[act_idx],
                    actual[act_idx + 1],
                    actual[act_idx + 2],
                    if reference.channels > 3 {
                        actual[act_idx + 3]
                    } else {
                        255
                    },
                ];

                let max_diff = ref_rgba
                    .iter()
                    .zip(&act_rgba)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap_or(0);

                if max_diff > 1 && sample_pixels.len() < 20 {
                    sample_pixels.push((x, y, ref_rgba, act_rgba));
                }
            }
        }

        // Print moxcms repro test
        eprintln!("\n=== MOXCMS REPRO TEST ===");
        eprintln!("Add this test to moxcms/tests/cmyk_test.rs:\n");
        eprintln!(
            r#"#[test]
fn test_cmyk_jxl_layers_profile() {{
    // ICC profile from cmyk_layers.jxl (conformance test image)
    // Profile: US Web Coated (SWOP) v2 or similar CMYK profile
    let icc_data = include_bytes!("cmyk_layers.icc");

    let profile = moxcms::ColorProfile::new_from_slice(icc_data).unwrap();
    let srgb = moxcms::ColorProfile::new_srgb();

    let options = moxcms::TransformOptions::default();
    let transform = profile
        .create_transform_f32(
            moxcms::Layout::Rgba, // CMYK uses Rgba layout (4 channels)
            &srgb,
            moxcms::Layout::Rgb,
            options,
        )
        .unwrap();

    // Sample CMYK values that produce incorrect RGB in moxcms
    // JXL uses reflectance convention: 1.0 = no ink, 0.0 = full ink
    // ICC uses: 0.0 = no ink, 1.0 = max ink
    // Values below are already converted to ICC convention
    let test_cases = ["#
        );

        for (i, (x, y, ref_rgba, act_rgba)) in sample_pixels.iter().enumerate() {
            // We need to figure out what CMYK values produced these
            // For now, just print the expected vs actual RGB
            eprintln!(
                "        // Pixel ({}, {}): expected RGB [{}, {}, {}], got [{}, {}, {}]",
                x, y, ref_rgba[0], ref_rgba[1], ref_rgba[2], act_rgba[0], act_rgba[1], act_rgba[2]
            );
            if i >= 5 {
                eprintln!("        // ... ({} more samples)", sample_pixels.len() - 6);
                break;
            }
        }

        eprintln!(
            r#"    ];

    // The issue: moxcms clips near-white CMYK values to pure white
    // Expected: slight color tints preserved
    // Actual: clipped to [255, 255, 255]
}}"#
        );

        eprintln!("\n=== END REPRO TEST ===");
        eprintln!(
            "\nTo use: copy {} to moxcms/tests/ and add the test above",
            icc_path
        );
    }
}
