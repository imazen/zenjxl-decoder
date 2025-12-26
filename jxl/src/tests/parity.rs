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
