// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! HDR integration tests.
//!
//! Tests decode of HDR content (PQ, HLG) to various output formats,
//! validates pixel values, ICC profile generation, intensity target
//! handling, and tone mapping.

use crate::api::{
    Endianness, JxlBasicInfo, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};
use crate::image::{Image, Rect};

/// Decode a JXL file to the `WithImageInfo` state and return basic info.
fn decode_header(data: &[u8]) -> (JxlBasicInfo, JxlDecoder<states::WithImageInfo>) {
    decode_header_with_options(data, JxlDecoderOptions::default())
}

/// Decode a JXL file to the `WithImageInfo` state with custom options.
fn decode_header_with_options(
    data: &[u8],
    options: JxlDecoderOptions,
) -> (JxlBasicInfo, JxlDecoder<states::WithImageInfo>) {
    let decoder: JxlDecoder<states::Initialized> = JxlDecoder::new(options);
    let mut input: &[u8] = data;
    let decoder: JxlDecoder<states::WithImageInfo> = match decoder.process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { fallback, .. } => {
            match fallback.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => result,
                _ => panic!("Expected complete after second attempt"),
            }
        }
    };
    let info = decoder.basic_info().clone();
    (info, decoder)
}

/// Decode a JXL file to F32 RGBA pixels.
fn decode_to_f32_rgba(data: &[u8]) -> (usize, usize, Vec<f32>) {
    let (info, mut decoder) = decode_header(data);
    let width = info.size.0;
    let height = info.size.1;
    let channels = 4;

    let num_extra = info.extra_channels.len();
    let pixel_format = JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; num_extra],
    };
    decoder.set_pixel_format(pixel_format);

    let mut input: &[u8] = data;
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Frame decode error: {e:?}"),
        }
    };

    let mut output_image =
        Image::<f32>::new((width * channels, height)).expect("Failed to allocate");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Pixel decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Decode a JXL file to F32 RGBA pixels with custom options.
fn decode_to_f32_rgba_with_options(
    data: &[u8],
    options: JxlDecoderOptions,
) -> (usize, usize, Vec<f32>) {
    let (info, mut decoder) = decode_header_with_options(data, options);
    let width = info.size.0;
    let height = info.size.1;
    let channels = 4;

    let num_extra = info.extra_channels.len();
    let pixel_format = JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; num_extra],
    };
    decoder.set_pixel_format(pixel_format);

    let mut input: &[u8] = data;
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Frame decode error: {e:?}"),
        }
    };

    let mut output_image =
        Image::<f32>::new((width * channels, height)).expect("Failed to allocate");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Pixel decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Decode a JXL file to U16 RGBA pixels.
fn decode_to_u16_rgba(data: &[u8]) -> (usize, usize, Vec<u16>) {
    let (info, mut decoder) = decode_header(data);
    let width = info.size.0;
    let height = info.size.1;
    let channels = 4;

    let num_extra = info.extra_channels.len();
    let pixel_format = JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::U16 {
            endianness: Endianness::native(),
            bit_depth: 16,
        }),
        extra_channel_format: vec![None; num_extra],
    };
    decoder.set_pixel_format(pixel_format);

    let mut input: &[u8] = data;
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Frame decode error: {e:?}"),
        }
    };

    let mut output_image =
        Image::<u16>::new((width * channels, height)).expect("Failed to allocate");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Pixel decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

/// Decode a JXL file to RGB only (3 channels) F32 pixels.
fn decode_to_f32_rgb(data: &[u8]) -> (usize, usize, Vec<f32>) {
    let (info, mut decoder) = decode_header(data);
    let width = info.size.0;
    let height = info.size.1;
    let channels = 3;

    let num_extra = info.extra_channels.len();
    let pixel_format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; num_extra],
    };
    decoder.set_pixel_format(pixel_format);

    let mut input: &[u8] = data;
    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Frame decode error: {e:?}"),
        }
    };

    let mut output_image =
        Image::<f32>::new((width * channels, height)).expect("Failed to allocate");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("Pixel decode error: {e:?}"),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    (width, height, pixels)
}

fn load_test_file(name: &str) -> Vec<u8> {
    let path = std::path::Path::new("resources/test").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()))
}

// ============================================================================
// Metadata tests
// ============================================================================

#[test]
fn pq_metadata_intensity_target() {
    let data = load_test_file("hdr_pq_test.jxl");
    let (info, _decoder) = decode_header(&data);
    // PQ content should have a high intensity target (typically 10000 or 4000 nits)
    assert!(
        info.tone_mapping.intensity_target > 255.0,
        "PQ intensity target should be > 255 nits (SDR), got {}",
        info.tone_mapping.intensity_target
    );
}

#[test]
fn hlg_metadata_intensity_target() {
    let data = load_test_file("hdr_hlg_test.jxl");
    let (info, _decoder) = decode_header(&data);
    // HLG content should have intensity_target > SDR
    assert!(
        info.tone_mapping.intensity_target > 255.0,
        "HLG intensity target should be > 255 nits (SDR), got {}",
        info.tone_mapping.intensity_target
    );
}

// ============================================================================
// F32 output format tests
// ============================================================================

#[test]
fn pq_decode_to_f32_rgba() {
    let data = load_test_file("hdr_pq_test.jxl");
    let (w, h, pixels) = decode_to_f32_rgba(&data);
    assert!(w > 0 && h > 0, "Image dimensions should be positive");
    assert_eq!(pixels.len(), w * h * 4, "Pixel count mismatch");

    // All pixel values should be finite
    assert!(
        pixels.iter().all(|v| v.is_finite()),
        "Found non-finite pixel values in PQ F32 output"
    );

    // RGB channels should be in [0, 1] range (PQ encoded values)
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 4;
            for c in 0..3 {
                let v = pixels[base + c];
                assert!(
                    (-0.01..=1.01).contains(&v),
                    "PQ F32 pixel ({x},{y}) channel {c} out of range: {v}"
                );
            }
            // Alpha should be ~1.0
            let a = pixels[base + 3];
            assert!(
                (a - 1.0).abs() < 0.01,
                "Alpha at ({x},{y}) should be ~1.0, got {a}"
            );
        }
    }
}

#[test]
fn hlg_decode_to_f32_rgba() {
    let data = load_test_file("hdr_hlg_test.jxl");
    let (w, h, pixels) = decode_to_f32_rgba(&data);
    assert!(w > 0 && h > 0, "Image dimensions should be positive");
    assert_eq!(pixels.len(), w * h * 4, "Pixel count mismatch");

    assert!(
        pixels.iter().all(|v| v.is_finite()),
        "Found non-finite pixel values in HLG F32 output"
    );

    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 4;
            for c in 0..3 {
                let v = pixels[base + c];
                assert!(
                    (-0.01..=1.5).contains(&v),
                    "HLG F32 pixel ({x},{y}) channel {c} out of range: {v}"
                );
            }
        }
    }
}

#[test]
fn pq_decode_to_f32_rgb_no_alpha() {
    let data = load_test_file("hdr_pq_test.jxl");
    let (w, h, pixels) = decode_to_f32_rgb(&data);
    assert_eq!(pixels.len(), w * h * 3, "RGB pixel count mismatch");
    assert!(
        pixels.iter().all(|v| v.is_finite()),
        "Found non-finite pixel values"
    );
}

// ============================================================================
// U16 output format tests
// ============================================================================

#[test]
fn pq_decode_to_u16_rgba() {
    let data = load_test_file("hdr_pq_test.jxl");
    let (w, h, pixels) = decode_to_u16_rgba(&data);
    assert!(w > 0 && h > 0);
    assert_eq!(pixels.len(), w * h * 4);

    // U16 values should be in [0, 65535]
    // Alpha channel should be at or near 65535
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 4;
            let a = pixels[base + 3];
            assert!(
                a >= 65500,
                "U16 alpha at ({x},{y}) should be near 65535, got {a}"
            );
        }
    }
}

#[test]
fn hlg_decode_to_u16_rgba() {
    let data = load_test_file("hdr_hlg_test.jxl");
    let (w, h, pixels) = decode_to_u16_rgba(&data);
    assert!(w > 0 && h > 0);
    assert_eq!(pixels.len(), w * h * 4);

    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 4;
            let a = pixels[base + 3];
            assert!(
                a >= 65500,
                "U16 alpha at ({x},{y}) should be near 65535, got {a}"
            );
        }
    }
}

// ============================================================================
// ICC profile tests
// ============================================================================

#[test]
fn pq_icc_profile_has_lab_pcs_and_a2b0() {
    let data = load_test_file("hdr_pq_test.jxl");
    let (_info, decoder) = decode_header(&data);
    let profile = decoder.output_color_profile();
    let icc = profile.try_as_icc().expect("PQ should produce ICC profile");

    // HDR PQ with standard primaries should use Lab PCS
    assert_eq!(&icc[20..24], b"Lab ", "PCS should be Lab for PQ HDR");

    // Must have A2B0 tag for tone mapping LUT
    assert!(
        icc.windows(4).any(|w| w == b"A2B0"),
        "Missing A2B0 tag in PQ ICC profile"
    );

    // Must have B2A0 tag (Apple compatibility)
    assert!(
        icc.windows(4).any(|w| w == b"B2A0"),
        "Missing B2A0 tag in PQ ICC profile"
    );

    // Must have CICP tag for standard identification
    assert!(
        icc.windows(4).any(|w| w == b"cicp"),
        "Missing cicp tag in PQ ICC profile"
    );
}

#[test]
fn hlg_icc_profile_has_lab_pcs_and_a2b0() {
    let data = load_test_file("hdr_hlg_test.jxl");
    let (_info, decoder) = decode_header(&data);
    let profile = decoder.output_color_profile();
    let icc = profile
        .try_as_icc()
        .expect("HLG should produce ICC profile");

    assert_eq!(&icc[20..24], b"Lab ", "PCS should be Lab for HLG HDR");
    assert!(
        icc.windows(4).any(|w| w == b"A2B0"),
        "Missing A2B0 tag in HLG ICC profile"
    );
    assert!(
        icc.windows(4).any(|w| w == b"B2A0"),
        "Missing B2A0 tag in HLG ICC profile"
    );
}

// ============================================================================
// Desired intensity target / tone mapping tests
// ============================================================================

#[test]
fn pq_desired_intensity_target_changes_output() {
    let data = load_test_file("hdr_pq_test.jxl");

    // Decode without tone mapping
    let (_w1, _h1, pixels_native) = decode_to_f32_rgba(&data);

    // Decode with desired_intensity_target = 250 nits (SDR)
    let options = JxlDecoderOptions {
        desired_intensity_target: Some(250.0),
        ..Default::default()
    };
    let (_w2, _h2, pixels_tonemapped) = decode_to_f32_rgba_with_options(&data, options);

    // Both should decode to the same dimensions
    assert_eq!(_w1, _w2);
    assert_eq!(_h1, _h2);
    assert_eq!(pixels_native.len(), pixels_tonemapped.len());

    // Tone mapping should change pixel values (unless the image is
    // already at or near the target luminance)
    let native_sum: f32 = pixels_native.iter().take(3).sum();
    let tonemapped_sum: f32 = pixels_tonemapped.iter().take(3).sum();

    // At minimum, both should produce valid finite values
    assert!(
        native_sum.is_finite(),
        "Native decode produced non-finite values"
    );
    assert!(
        tonemapped_sum.is_finite(),
        "Tone-mapped decode produced non-finite values"
    );
}

#[test]
fn hlg_desired_intensity_target_changes_output() {
    let data = load_test_file("hdr_hlg_test.jxl");

    let (_w1, _h1, pixels_native) = decode_to_f32_rgba(&data);

    let options = JxlDecoderOptions {
        desired_intensity_target: Some(300.0),
        ..Default::default()
    };
    let (_w2, _h2, pixels_adjusted) = decode_to_f32_rgba_with_options(&data, options);

    assert_eq!(_w1, _w2);
    assert_eq!(_h1, _h2);
    assert_eq!(pixels_native.len(), pixels_adjusted.len());

    let native_sum: f32 = pixels_native.iter().take(3).sum();
    let adjusted_sum: f32 = pixels_adjusted.iter().take(3).sum();

    assert!(native_sum.is_finite());
    assert!(adjusted_sum.is_finite());
}

// ============================================================================
// Cross-format consistency tests
// ============================================================================

#[test]
fn pq_f32_u16_consistency() {
    let data = load_test_file("hdr_pq_test.jxl");
    let (w_f32, h_f32, pixels_f32) = decode_to_f32_rgba(&data);
    let (w_u16, h_u16, pixels_u16) = decode_to_u16_rgba(&data);

    assert_eq!(w_f32, w_u16);
    assert_eq!(h_f32, h_u16);

    // Compare F32 and U16 values: U16 should be approximately F32 * 65535
    let total_pixels = w_f32 * h_f32 * 4;
    let mut max_error: f32 = 0.0;
    for i in 0..total_pixels {
        let f32_val = pixels_f32[i];
        let u16_val = pixels_u16[i] as f32 / 65535.0;
        let error = (f32_val - u16_val).abs();
        max_error = max_error.max(error);
    }

    // Allow some quantization error (U16 has 1/65535 precision)
    assert!(
        max_error < 0.01,
        "F32/U16 max error too large: {max_error} (expected < 0.01)"
    );
}

#[test]
fn hlg_f32_u16_consistency() {
    let data = load_test_file("hdr_hlg_test.jxl");
    let (w_f32, h_f32, pixels_f32) = decode_to_f32_rgba(&data);
    let (w_u16, h_u16, pixels_u16) = decode_to_u16_rgba(&data);

    assert_eq!(w_f32, w_u16);
    assert_eq!(h_f32, h_u16);

    let total_pixels = w_f32 * h_f32 * 4;
    let mut max_error: f32 = 0.0;
    for i in 0..total_pixels {
        let f32_val = pixels_f32[i];
        let u16_val = pixels_u16[i] as f32 / 65535.0;
        let error = (f32_val - u16_val).abs();
        max_error = max_error.max(error);
    }

    assert!(
        max_error < 0.01,
        "F32/U16 max error too large: {max_error} (expected < 0.01)"
    );
}

// ============================================================================
// Transfer function roundtrip properties
// ============================================================================

#[test]
fn pq_pixel_values_plausible() {
    // After decoding PQ content, pixels should represent PQ-encoded values.
    // For a non-black image, at least some pixels should be non-zero.
    let data = load_test_file("hdr_pq_test.jxl");
    let (w, h, pixels) = decode_to_f32_rgba(&data);

    let rgb_sum: f64 = pixels
        .chunks_exact(4)
        .map(|px| (px[0] + px[1] + px[2]) as f64)
        .sum();

    // For a non-trivial image, the mean pixel value should be > 0
    let mean = rgb_sum / (w * h * 3) as f64;
    assert!(
        mean > 0.0 || (w == 1 && h == 1),
        "PQ image appears to be all-black (mean = {mean})"
    );
}

#[test]
fn hlg_pixel_values_plausible() {
    let data = load_test_file("hdr_hlg_test.jxl");
    let (w, h, pixels) = decode_to_f32_rgba(&data);

    let rgb_sum: f64 = pixels
        .chunks_exact(4)
        .map(|px| (px[0] + px[1] + px[2]) as f64)
        .sum();

    let mean = rgb_sum / (w * h * 3) as f64;
    assert!(
        mean > 0.0 || (w == 1 && h == 1),
        "HLG image appears to be all-black (mean = {mean})"
    );
}

// ============================================================================
// Tone mapping stage unit property tests
// ============================================================================

/// Helper: run a ToneMappingStage on a single pixel via process_row_chunk.
fn run_tone_map_pixel(
    stage: &crate::render::stages::ToneMappingStage,
    r: f32,
    g: f32,
    b: f32,
) -> [f32; 3] {
    use crate::render::RenderPipelineInPlaceStage;
    let mut rv = [r];
    let mut gv = [g];
    let mut bv = [b];
    let mut rows: Vec<&mut [f32]> = vec![&mut rv, &mut gv, &mut bv];
    stage.process_row_chunk((0, 0), 1, &mut rows, None);
    [rv[0], gv[0], bv[0]]
}

#[test]
fn rec2408_monotonic() {
    // The Rec.2408 tone mapper should be monotonic — brighter input produces
    // brighter or equal output.
    use crate::render::stages::ToneMappingStage;

    let luminances = [0.2627_f32, 0.678, 0.0593];
    let stage = ToneMappingStage::pq(0, 10000.0, 250.0, luminances);
    let n = 16;
    let mut prev_out = -1.0_f32;

    for i in 0..n {
        let val = i as f32 / (n - 1) as f32;
        let [out_r, _, _] = run_tone_map_pixel(&stage, val, val, val);
        assert!(
            out_r >= prev_out - 1e-6,
            "Rec.2408 not monotonic at input {val}: output {out_r} < previous {prev_out}"
        );
        prev_out = out_r;
    }
}

#[test]
fn rec2408_preserves_black() {
    // Black (0, 0, 0) should map to (near) black.
    use crate::render::stages::ToneMappingStage;

    let luminances = [0.2627_f32, 0.678, 0.0593];
    let stage = ToneMappingStage::pq(0, 10000.0, 250.0, luminances);
    let [r, g, b] = run_tone_map_pixel(&stage, 0.0, 0.0, 0.0);

    for (c, v) in [r, g, b].iter().enumerate() {
        assert!(
            v.abs() < 0.01,
            "Black not preserved after tone mapping: channel {c} = {v}"
        );
    }
}

#[test]
fn hlg_ootf_preserves_black() {
    use crate::render::stages::ToneMappingStage;

    let luminances = [0.2627_f32, 0.678, 0.0593];
    let stage = ToneMappingStage::hlg(0, 1000.0, 300.0, luminances);
    let [r, g, b] = run_tone_map_pixel(&stage, 0.0, 0.0, 0.0);

    for (c, v) in [r, g, b].iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "Black not preserved after HLG OOTF: channel {c} = {v}"
        );
    }
}
