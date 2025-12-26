// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Parity test infrastructure for comparing jxl-rs output against libjxl reference.
//!
//! IMPORTANT: These utilities use EXACT tolerances derived from the JPEG XL spec.
//! DO NOT WEAKEN TOLERANCES to make tests pass. If a test fails, the implementation
//! is wrong and must be fixed.

use std::path::Path;

/// Maximum allowed absolute error for pixel values in [0, 1] range.
/// This is the conformance threshold from the JPEG XL specification.
pub const CONFORMANCE_THRESHOLD: f32 = 1.0 / 256.0;

/// Maximum allowed absolute error for 8-bit integer pixels.
/// Equivalent to CONFORMANCE_THRESHOLD scaled to [0, 255].
pub const CONFORMANCE_THRESHOLD_U8: u8 = 1;

/// Maximum allowed absolute error for 16-bit integer pixels.
/// Equivalent to CONFORMANCE_THRESHOLD scaled to [0, 65535].
pub const CONFORMANCE_THRESHOLD_U16: u16 = 256;

/// Result of a parity comparison
#[derive(Debug)]
pub struct ParityResult {
    pub passed: bool,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub error_count: usize,
    pub total_pixels: usize,
    pub first_error_location: Option<(usize, usize, usize)>, // (x, y, channel)
}

impl ParityResult {
    pub fn assert_passed(&self) {
        if !self.passed {
            panic!(
                "Parity test FAILED:\n\
                 - Max absolute error: {}\n\
                 - Max relative error: {}\n\
                 - Error count: {} / {} pixels\n\
                 - First error at: {:?}\n\
                 \n\
                 DO NOT WEAKEN TOLERANCES. Fix the implementation.",
                self.max_abs_error,
                self.max_rel_error,
                self.error_count,
                self.total_pixels,
                self.first_error_location,
            );
        }
    }
}

/// Compare two f32 pixel buffers with conformance threshold.
///
/// # Arguments
/// * `reference` - Expected output from libjxl
/// * `actual` - Output from jxl-rs
/// * `width` - Image width
/// * `height` - Image height
/// * `channels` - Number of channels
/// * `threshold` - Maximum allowed absolute error
///
/// # Returns
/// ParityResult with comparison statistics
pub fn compare_f32_buffers(
    reference: &[f32],
    actual: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    threshold: f32,
) -> ParityResult {
    assert_eq!(reference.len(), actual.len());
    assert_eq!(reference.len(), width * height * channels);

    let mut max_abs_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut error_count: usize = 0;
    let mut first_error_location: Option<(usize, usize, usize)> = None;

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let idx = (y * width + x) * channels + c;
                let r = reference[idx] as f64;
                let a = actual[idx] as f64;

                let abs_error = (r - a).abs();
                let rel_error = if r.abs() > 1e-10 {
                    abs_error / r.abs()
                } else {
                    abs_error
                };

                max_abs_error = max_abs_error.max(abs_error);
                max_rel_error = max_rel_error.max(rel_error);

                if abs_error > threshold as f64 {
                    error_count += 1;
                    if first_error_location.is_none() {
                        first_error_location = Some((x, y, c));
                    }
                }
            }
        }
    }

    ParityResult {
        passed: error_count == 0,
        max_abs_error,
        max_rel_error,
        error_count,
        total_pixels: width * height * channels,
        first_error_location,
    }
}

/// Compare two u8 pixel buffers with conformance threshold.
pub fn compare_u8_buffers(
    reference: &[u8],
    actual: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    threshold: u8,
) -> ParityResult {
    assert_eq!(reference.len(), actual.len());
    assert_eq!(reference.len(), width * height * channels);

    let mut max_abs_error: f64 = 0.0;
    let mut error_count: usize = 0;
    let mut first_error_location: Option<(usize, usize, usize)> = None;

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let idx = (y * width + x) * channels + c;
                let r = reference[idx];
                let a = actual[idx];

                let abs_error = (r as i16 - a as i16).unsigned_abs() as u8;
                max_abs_error = max_abs_error.max(abs_error as f64);

                if abs_error > threshold {
                    error_count += 1;
                    if first_error_location.is_none() {
                        first_error_location = Some((x, y, c));
                    }
                }
            }
        }
    }

    ParityResult {
        passed: error_count == 0,
        max_abs_error,
        max_rel_error: max_abs_error / 255.0,
        error_count,
        total_pixels: width * height * channels,
        first_error_location,
    }
}

/// Compare two u16 pixel buffers with conformance threshold.
pub fn compare_u16_buffers(
    reference: &[u16],
    actual: &[u16],
    width: usize,
    height: usize,
    channels: usize,
    threshold: u16,
) -> ParityResult {
    assert_eq!(reference.len(), actual.len());
    assert_eq!(reference.len(), width * height * channels);

    let mut max_abs_error: f64 = 0.0;
    let mut error_count: usize = 0;
    let mut first_error_location: Option<(usize, usize, usize)> = None;

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let idx = (y * width + x) * channels + c;
                let r = reference[idx];
                let a = actual[idx];

                let abs_error = (r as i32 - a as i32).unsigned_abs() as u16;
                max_abs_error = max_abs_error.max(abs_error as f64);

                if abs_error > threshold {
                    error_count += 1;
                    if first_error_location.is_none() {
                        first_error_location = Some((x, y, c));
                    }
                }
            }
        }
    }

    ParityResult {
        passed: error_count == 0,
        max_abs_error,
        max_rel_error: max_abs_error / 65535.0,
        error_count,
        total_pixels: width * height * channels,
        first_error_location,
    }
}

/// Load reference data from a binary file.
/// Format: little-endian f32 values, row-major order.
pub fn load_reference_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    assert_eq!(
        bytes.len() % 4,
        0,
        "Reference file size must be multiple of 4"
    );

    let mut result = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(result)
}

/// Load reference data from a binary file (u8 format).
pub fn load_reference_u8(path: &Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Load reference data from a binary file (u16 format).
/// Format: little-endian u16 values.
pub fn load_reference_u16(path: &Path) -> std::io::Result<Vec<u16>> {
    let bytes = std::fs::read(path)?;
    assert_eq!(
        bytes.len() % 2,
        0,
        "Reference file size must be multiple of 2"
    );

    let mut result = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        result.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(result)
}

/// Get path to reference data directory.
pub fn reference_data_dir() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("src/tests/reference_data")
}

/// Get path to test resources directory (existing test images).
pub fn test_resources_dir() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("resources/test")
}

/// Get path to codec-corpus JXL directory.
/// Returns None if codec-corpus is not found.
pub fn codec_corpus_jxl_dir() -> Option<std::path::PathBuf> {
    // Try relative path from jxl-rs workspace
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Try common relative locations
    let candidates = [
        manifest_dir.join("../../codec-eval/codec-corpus/jxl"),
        manifest_dir.join("../../../codec-eval/codec-corpus/jxl"),
        std::path::PathBuf::from("/home/lilith/work/codec-eval/codec-corpus/jxl"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Some(candidate.clone());
        }
    }

    // Try CODEC_CORPUS_PATH environment variable
    if let Ok(path) = std::env::var("CODEC_CORPUS_PATH") {
        let p = std::path::PathBuf::from(path).join("jxl");
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Parse a PPM (P6 binary) file into pixel data.
/// Returns (width, height, channels, pixels) where pixels is RGB u8 data.
pub fn load_ppm(path: &Path) -> std::io::Result<(usize, usize, usize, Vec<u8>)> {
    use std::io::{BufRead, BufReader, Read};

    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic number
    let mut magic = String::new();
    reader.read_line(&mut magic)?;
    let magic = magic.trim();

    if magic != "P6" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected P6 PPM, got {}", magic),
        ));
    }

    // Skip comments and read dimensions
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if !trimmed.starts_with('#') && !trimmed.is_empty() {
            break;
        }
    }

    let dims: Vec<usize> = line
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let (width, height) = (dims[0], dims[1]);

    // Read max value
    line.clear();
    reader.read_line(&mut line)?;
    let max_val: u16 = line.trim().parse().unwrap();

    let channels = 3; // PPM is always RGB

    // Read pixel data
    let pixel_count = width * height * channels;
    let pixels = if max_val <= 255 {
        let mut buf = vec![0u8; pixel_count];
        reader.read_exact(&mut buf)?;
        buf
    } else {
        // 16-bit PPM, convert to 8-bit
        let mut buf16 = vec![0u8; pixel_count * 2];
        reader.read_exact(&mut buf16)?;
        buf16
            .chunks_exact(2)
            .map(|c| (u16::from_be_bytes([c[0], c[1]]) >> 8) as u8)
            .collect()
    };

    // Verify we got all pixels
    if pixels.len() != pixel_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Expected {} pixels, got {}", pixel_count, pixels.len()),
        ));
    }

    Ok((width, height, channels, pixels))
}

/// Parse a PGM (P5 binary) file into pixel data.
/// Returns (width, height, 1, pixels) where pixels is grayscale u8 data.
pub fn load_pgm(path: &Path) -> std::io::Result<(usize, usize, usize, Vec<u8>)> {
    use std::io::{BufRead, BufReader, Read};

    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic number
    let mut magic = String::new();
    reader.read_line(&mut magic)?;
    let magic = magic.trim();

    if magic != "P5" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Expected P5 PGM, got {}", magic),
        ));
    }

    // Skip comments and read dimensions
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if !trimmed.starts_with('#') && !trimmed.is_empty() {
            break;
        }
    }

    let dims: Vec<usize> = line
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let (width, height) = (dims[0], dims[1]);

    // Read max value
    line.clear();
    reader.read_line(&mut line)?;
    let _max_val: u16 = line.trim().parse().unwrap();

    let channels = 1;
    let pixel_count = width * height;

    let mut pixels = vec![0u8; pixel_count];
    reader.read_exact(&mut pixels)?;

    Ok((width, height, channels, pixels))
}

/// Reference image data loaded from PPM/PNG
#[derive(Debug)]
pub struct ReferenceImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
}

impl ReferenceImage {
    /// Load reference image from PPM or PNG file
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext.to_lowercase().as_str() {
            "ppm" => {
                let (width, height, channels, pixels) = load_ppm(path)?;
                Ok(Self {
                    width,
                    height,
                    channels,
                    pixels,
                })
            }
            "pgm" => {
                let (width, height, channels, pixels) = load_pgm(path)?;
                Ok(Self {
                    width,
                    height,
                    channels,
                    pixels,
                })
            }
            "png" => {
                // For PNG, we'd need a PNG decoder - skip for now
                // In practice, use PPM references where possible
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "PNG reference loading requires external decoder",
                ))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported reference format: {}", ext),
            )),
        }
    }
}

/// Codec-corpus test case
#[derive(Debug)]
pub struct CodecCorpusTestCase {
    pub name: String,
    pub category: String,
    pub jxl_path: std::path::PathBuf,
    pub reference_path: Option<std::path::PathBuf>,
}

/// Discover all JXL test cases in codec-corpus
pub fn discover_codec_corpus_tests() -> Vec<CodecCorpusTestCase> {
    let mut tests = Vec::new();

    let Some(corpus_dir) = codec_corpus_jxl_dir() else {
        return tests;
    };

    for category in &["conformance", "features", "photographic", "edge-cases"] {
        let cat_dir = corpus_dir.join(category);
        if !cat_dir.exists() {
            continue;
        }

        if let Ok(entries) = std::fs::read_dir(&cat_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("jxl") {
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();

                    // Look for reference output
                    let ref_dir = corpus_dir.join("reference").join(category);
                    let ref_ppm = ref_dir.join(format!("{}.ppm", name));
                    let ref_png = ref_dir.join(format!("{}.png", name));

                    let reference_path = if ref_ppm.exists() {
                        Some(ref_ppm)
                    } else if ref_png.exists() {
                        Some(ref_png)
                    } else {
                        None
                    };

                    tests.push(CodecCorpusTestCase {
                        name,
                        category: category.to_string(),
                        jxl_path: path,
                        reference_path,
                    });
                }
            }
        }
    }

    // Sort by category and name for consistent ordering
    tests.sort_by(|a, b| (&a.category, &a.name).cmp(&(&b.category, &b.name)));
    tests
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_f32_exact_match() {
        let reference = vec![0.0, 0.5, 1.0, 0.25];
        let actual = vec![0.0, 0.5, 1.0, 0.25];
        let result = compare_f32_buffers(&reference, &actual, 2, 2, 1, CONFORMANCE_THRESHOLD);
        assert!(result.passed);
        assert_eq!(result.error_count, 0);
    }

    #[test]
    fn test_compare_f32_within_threshold() {
        let reference = vec![0.0, 0.5, 1.0, 0.25];
        let actual = vec![0.001, 0.501, 1.001, 0.251]; // Within 1/256
        let result = compare_f32_buffers(&reference, &actual, 2, 2, 1, CONFORMANCE_THRESHOLD);
        assert!(result.passed);
    }

    #[test]
    fn test_compare_f32_exceeds_threshold() {
        let reference = vec![0.0, 0.5, 1.0, 0.25];
        let actual = vec![0.0, 0.5, 1.0, 0.30]; // Last pixel exceeds threshold
        let result = compare_f32_buffers(&reference, &actual, 2, 2, 1, CONFORMANCE_THRESHOLD);
        assert!(!result.passed);
        assert_eq!(result.error_count, 1);
        assert_eq!(result.first_error_location, Some((1, 1, 0)));
    }

    #[test]
    fn test_compare_u8_exact_match() {
        let reference = vec![0, 128, 255, 64];
        let actual = vec![0, 128, 255, 64];
        let result = compare_u8_buffers(&reference, &actual, 2, 2, 1, CONFORMANCE_THRESHOLD_U8);
        assert!(result.passed);
    }

    #[test]
    fn test_compare_u8_within_threshold() {
        let reference = vec![0, 128, 255, 64];
        let actual = vec![1, 127, 254, 65]; // Within 1
        let result = compare_u8_buffers(&reference, &actual, 2, 2, 1, CONFORMANCE_THRESHOLD_U8);
        assert!(result.passed);
    }

    #[test]
    fn test_compare_u8_exceeds_threshold() {
        let reference = vec![0, 128, 255, 64];
        let actual = vec![0, 128, 255, 67]; // Last pixel differs by 3
        let result = compare_u8_buffers(&reference, &actual, 2, 2, 1, CONFORMANCE_THRESHOLD_U8);
        assert!(!result.passed);
        assert_eq!(result.error_count, 1);
    }
}
