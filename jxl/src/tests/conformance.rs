// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Official JPEG XL conformance tests.
//!
//! These tests compare jxl-rs output against the official libjxl conformance
//! test suite reference images (NPY format).
//!
//! To run these tests:
//! 1. Clone the conformance repo: `git clone https://github.com/libjxl/conformance`
//! 2. Download reference data: `cd conformance && bash scripts/download_and_symlink_using_curl.sh`
//! 3. Run tests: `CONFORMANCE_PATH=/path/to/conformance cargo test --features cms conformance -- --ignored`
//!
//! IMPORTANT: DO NOT WEAKEN TOLERANCES. If a test fails, the implementation
//! is wrong and must be fixed.

#[cfg(feature = "cms")]
use crate::api::MoxCms;
use crate::api::{
    JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult, states,
};

use std::path::{Path, PathBuf};

/// Get the conformance test directory from environment variable.
fn conformance_dir() -> Option<PathBuf> {
    std::env::var("CONFORMANCE_PATH").ok().map(PathBuf::from)
}

/// Metadata for a conformance test frame.
#[derive(Debug, Clone)]
struct FrameInfo {
    #[allow(dead_code)]
    name: String,
    rms_error: f32,
    peak_error: f32,
}

/// Metadata for a conformance test case.
#[derive(Debug)]
struct ConformanceTestCase {
    name: String,
    path: PathBuf,
    frames: Vec<FrameInfo>,
    #[allow(dead_code)]
    intensity_target: f32,
    #[allow(dead_code)]
    extra_channel_types: Vec<String>,
    #[allow(dead_code)]
    bits_per_sample: Vec<u32>,
}

impl ConformanceTestCase {
    fn input_jxl(&self) -> PathBuf {
        self.path.join("input.jxl")
    }

    fn reference_npy(&self) -> PathBuf {
        self.path.join("reference_image.npy")
    }
}

/// Parse test.json for a conformance test case.
fn parse_test_json(path: &Path) -> Option<ConformanceTestCase> {
    let test_json_path = path.join("test.json");
    let content = std::fs::read_to_string(&test_json_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let frames: Vec<FrameInfo> = json
        .get("frames")?
        .as_array()?
        .iter()
        .map(|f| FrameInfo {
            name: f
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            rms_error: f.get("rms_error").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            peak_error: f.get("peak_error").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        })
        .collect();

    let intensity_target = json
        .get("intensity_target")
        .and_then(|v| v.as_f64())
        .unwrap_or(255.0) as f32;

    let extra_channel_types: Vec<String> = json
        .get("extra_channel_type")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let bits_per_sample: Vec<u32> = json
        .get("bits_per_sample")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();

    Some(ConformanceTestCase {
        name: path.file_name()?.to_str()?.to_string(),
        path: path.to_path_buf(),
        frames,
        intensity_target,
        extra_channel_types,
        bits_per_sample,
    })
}

/// Discover all conformance test cases.
fn discover_conformance_tests() -> Vec<ConformanceTestCase> {
    let Some(conformance_dir) = conformance_dir() else {
        return vec![];
    };

    let testcases_dir = conformance_dir.join("testcases");
    if !testcases_dir.exists() {
        return vec![];
    }

    // Read main_level5.txt to get the list of tests
    let corpus_path = testcases_dir.join("main_level5.txt");
    let test_names: Vec<String> = if corpus_path.exists() {
        std::fs::read_to_string(&corpus_path)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|s| s.trim().to_string())
            .collect()
    } else {
        // Fall back to listing directories
        std::fs::read_dir(&testcases_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default()
    };

    test_names
        .into_iter()
        .filter_map(|name| {
            let test_path = testcases_dir.join(&name);
            if test_path.exists() {
                parse_test_json(&test_path)
            } else {
                None
            }
        })
        .collect()
}

/// Simple NPY file reader for f32 arrays.
/// NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
fn read_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read NPY: {}", e))?;

    // Check magic number
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic number".to_string());
    }

    let major_version = data[6];
    let _minor_version = data[7];

    // Parse header length
    let header_len = if major_version == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };

    let header_start = if major_version == 1 { 10 } else { 12 };
    let header_end = header_start + header_len;
    let header =
        std::str::from_utf8(&data[header_start..header_end]).map_err(|_| "Invalid NPY header")?;

    // Parse shape from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1024, 1024, 4), }"
    let shape_start = header.find("'shape': (").ok_or("No shape in NPY header")?;
    let shape_str_start = shape_start + "'shape': (".len();
    let shape_str_end = header[shape_str_start..]
        .find(')')
        .ok_or("Invalid shape in NPY header")?
        + shape_str_start;
    let shape_str = &header[shape_str_start..shape_str_end];

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Verify dtype is f32 (little-endian)
    if !header.contains("'<f4'") && !header.contains("'float32'") {
        return Err(format!("Unsupported NPY dtype (expected f32): {}", header));
    }

    // Read data
    let data_start = header_end;
    let num_elements: usize = shape.iter().product();
    let expected_bytes = num_elements * 4;

    if data.len() < data_start + expected_bytes {
        return Err(format!(
            "NPY file too short: {} < {}",
            data.len(),
            data_start + expected_bytes
        ));
    }

    let pixels: Vec<f32> = data[data_start..data_start + expected_bytes]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((shape, pixels))
}

/// Decode a JXL file to f32 pixels for conformance testing.
/// Currently only supports single-frame images. Animation support would require
/// API changes to iterate frames.
fn decode_jxl_to_f32(path: &Path) -> Result<(usize, usize, usize, usize, Vec<f32>), String> {
    use crate::image::{Image, Rect};

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

    // Check for animation - we only support single-frame for now
    if basic_info.animation.is_some() {
        return Err("Animation not yet supported in conformance tests".to_string());
    }

    // Determine format based on color space and alpha
    let default_format = decoder.current_pixel_format();
    let is_grayscale = matches!(
        default_format.color_type,
        JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha
    );

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

    // Request f32 output
    let num_extra_channels = basic_info.extra_channels.len();
    let extra_channel_format = vec![None; num_extra_channels];

    let format = JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format,
    };

    decoder.set_pixel_format(format);

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

    // Create output buffer - use Image<f32> for proper alignment
    let mut output_image = Image::<f32>::new((width * channels, height))
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

    // Extract pixels from Image<f32>
    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        let row = output_image.row(y);
        pixels.extend_from_slice(row);
    }

    // Return as single frame (1, height, width, channels)
    Ok((1, height, width, channels, pixels))
}

/// Compare decoded output against NPY reference using conformance thresholds.
fn compare_conformance(
    test: &ConformanceTestCase,
    decoded_shape: (usize, usize, usize, usize), // (frames, height, width, channels)
    decoded: &[f32],
    ref_shape: &[usize],
    reference: &[f32],
) -> Result<(), String> {
    let (dec_frames, dec_height, dec_width, dec_channels) = decoded_shape;

    // Reference shape is (frames, height, width, channels)
    if ref_shape.len() != 4 {
        return Err(format!("Invalid reference shape: {:?}", ref_shape));
    }
    let (ref_frames, ref_height, ref_width, ref_channels) =
        (ref_shape[0], ref_shape[1], ref_shape[2], ref_shape[3]);

    // Allow decoded to have more frames (e.g., last frame repeated)
    if dec_frames < ref_frames {
        return Err(format!(
            "Frame count mismatch: decoded {} < reference {}",
            dec_frames, ref_frames
        ));
    }

    if dec_height != ref_height || dec_width != ref_width {
        return Err(format!(
            "Size mismatch: decoded {}x{} vs reference {}x{}",
            dec_width, dec_height, ref_width, ref_height
        ));
    }

    // Allow decoded RGBA when reference is RGB (if alpha is all 1.0)
    let compare_channels = if dec_channels == 4 && ref_channels == 3 {
        3 // Compare only RGB
    } else if dec_channels != ref_channels {
        return Err(format!(
            "Channel count mismatch: decoded {} vs reference {}",
            dec_channels, ref_channels
        ));
    } else {
        dec_channels
    };

    let frame_size_dec = dec_height * dec_width * dec_channels;
    let frame_size_ref = ref_height * ref_width * ref_channels;

    for frame_idx in 0..ref_frames {
        let frame_info = test
            .frames
            .get(frame_idx)
            .ok_or_else(|| format!("No frame info for frame {}", frame_idx))?;

        let dec_frame_start = frame_idx * frame_size_dec;
        let ref_frame_start = frame_idx * frame_size_ref;

        let mut max_error: f32 = 0.0;
        let mut sum_sq_errors: Vec<f64> = vec![0.0; compare_channels];
        let pixel_count = dec_height * dec_width;

        for y in 0..dec_height {
            for x in 0..dec_width {
                let dec_idx = dec_frame_start + (y * dec_width + x) * dec_channels;
                let ref_idx = ref_frame_start + (y * ref_width + x) * ref_channels;

                for c in 0..compare_channels {
                    let dec_val = decoded[dec_idx + c];
                    let ref_val = reference[ref_idx + c];
                    let error = (dec_val - ref_val).abs();

                    max_error = max_error.max(error);
                    sum_sq_errors[c] += (error as f64) * (error as f64);
                }
            }
        }

        // Compute per-channel RMSE and take max
        let max_rmse: f32 = sum_sq_errors
            .iter()
            .map(|&sum| (sum / pixel_count as f64).sqrt() as f32)
            .fold(0.0f32, f32::max);

        // Check against thresholds
        if max_error > frame_info.peak_error {
            return Err(format!(
                "Frame {}: peak error {} > threshold {}",
                frame_idx, max_error, frame_info.peak_error
            ));
        }

        if max_rmse > frame_info.rms_error {
            return Err(format!(
                "Frame {}: RMSE {} > threshold {}",
                frame_idx, max_rmse, frame_info.rms_error
            ));
        }

        eprintln!(
            "  Frame {}: peak_error={:.6} (limit {:.6}), rmse={:.6} (limit {:.6})",
            frame_idx, max_error, frame_info.peak_error, max_rmse, frame_info.rms_error
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// List all available conformance tests.
    #[test]
    #[ignore] // Run with: CONFORMANCE_PATH=/path/to/conformance cargo test list_conformance_tests -- --ignored --nocapture
    fn list_conformance_tests() {
        let tests = discover_conformance_tests();
        if tests.is_empty() {
            eprintln!("No conformance tests found. Set CONFORMANCE_PATH environment variable.");
            eprintln!("Example: CONFORMANCE_PATH=~/work/conformance cargo test ...");
            return;
        }

        eprintln!("Found {} conformance tests:", tests.len());
        for test in &tests {
            eprintln!("  {} ({} frames)", test.name, test.frames.len());
        }
    }

    /// Run a single conformance test by name.
    fn run_conformance_test(name: &str) -> Result<(), String> {
        let tests = discover_conformance_tests();
        let test = tests
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Test '{}' not found", name))?;

        let input_path = test.input_jxl();
        let ref_path = test.reference_npy();

        if !input_path.exists() {
            return Err(format!("Input file not found: {:?}", input_path));
        }
        if !ref_path.exists() {
            return Err(format!(
                "Reference NPY not found: {:?}. Run download_and_symlink_using_curl.sh",
                ref_path
            ));
        }

        eprintln!("Testing: {}", name);

        // Load reference
        let (ref_shape, reference) = read_npy_f32(&ref_path)?;
        eprintln!("  Reference shape: {:?}", ref_shape);

        // Decode JXL
        let (frames, height, width, channels, decoded) = decode_jxl_to_f32(&input_path)?;
        eprintln!(
            "  Decoded shape: ({}, {}, {}, {})",
            frames, height, width, channels
        );

        // Compare
        compare_conformance(
            test,
            (frames, height, width, channels),
            &decoded,
            &ref_shape,
            &reference,
        )?;

        eprintln!("  PASS");
        Ok(())
    }

    /// Run all conformance tests.
    #[test]
    #[ignore] // Run with: CONFORMANCE_PATH=/path/to/conformance cargo test --features cms run_all_conformance -- --ignored --nocapture
    fn run_all_conformance() {
        let tests = discover_conformance_tests();
        if tests.is_empty() {
            eprintln!("No conformance tests found. Set CONFORMANCE_PATH.");
            eprintln!("1. git clone https://github.com/libjxl/conformance");
            eprintln!("2. cd conformance && bash scripts/download_and_symlink_using_curl.sh");
            eprintln!(
                "3. CONFORMANCE_PATH=/path/to/conformance cargo test --features cms run_all_conformance -- --ignored --nocapture"
            );
            return;
        }

        eprintln!("Running {} conformance tests...\n", tests.len());

        let mut passed = 0;
        let mut failed = 0;
        let mut failures: Vec<(String, String)> = Vec::new();

        for test in &tests {
            match run_conformance_test(&test.name) {
                Ok(()) => passed += 1,
                Err(e) => {
                    failed += 1;
                    failures.push((test.name.clone(), e));
                }
            }
            eprintln!();
        }

        eprintln!("\n=== Results ===");
        eprintln!("Passed: {}/{}", passed, passed + failed);
        eprintln!("Failed: {}/{}", failed, passed + failed);

        if !failures.is_empty() {
            eprintln!("\nFailures:");
            for (name, error) in &failures {
                eprintln!("  {}: {}", name, error);
            }
            panic!("{} conformance tests failed", failed);
        }
    }

    // Individual test functions for each conformance test.
    // These allow running specific tests and seeing which ones pass in CI.

    macro_rules! conformance_test {
        ($name:ident) => {
            #[test]
            #[ignore]
            fn $name() {
                if conformance_dir().is_none() {
                    eprintln!("Skipped: CONFORMANCE_PATH not set");
                    return;
                }
                run_conformance_test(stringify!($name)).unwrap();
            }
        };
    }

    // Level 5 conformance tests
    conformance_test!(alpha_nonpremultiplied);
    conformance_test!(alpha_triangles);
    conformance_test!(animation_icos4d_5);
    conformance_test!(animation_newtons_cradle);
    conformance_test!(animation_spline_5);
    conformance_test!(bench_oriented_brg_5);
    conformance_test!(bicycles);
    conformance_test!(bike_5);
    conformance_test!(blendmodes_5);
    conformance_test!(cafe_5);
    conformance_test!(delta_palette);
    conformance_test!(grayscale_5);
    conformance_test!(grayscale_jpeg_5);
    conformance_test!(grayscale_public_university);
    conformance_test!(lz77_flower);
    conformance_test!(noise_5);
    conformance_test!(opsin_inverse_5);
    conformance_test!(patches_5);
    conformance_test!(patches_lossless);
    conformance_test!(progressive_5);
}
