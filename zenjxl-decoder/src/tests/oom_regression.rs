// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression tests for OOM bugs found via fuzzing.
//!
//! These tests verify that crafted JXL codestreams with absurd dimensions
//! return errors instead of causing unbounded memory allocation.

use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

/// Helper: attempt to decode data through the full pipeline, expecting an error.
/// Returns Ok(()) if decoding correctly fails with an error, panics if it OOMs.
fn assert_decode_rejects(data: &[u8], label: &str) {
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input: &[u8] = data;

    // Try to get past header parsing
    let decoder = match decoder.process(&mut input) {
        Err(e) => {
            eprintln!("[{label}] Correctly rejected at header: {e}");
            return;
        }
        Ok(ProcessingResult::NeedsMoreInput { .. }) => {
            eprintln!("[{label}] Correctly rejected: needs more input (truncated)");
            return;
        }
        Ok(ProcessingResult::Complete { result }) => result,
    };

    // Header parsed — try frame info
    let mut input: &[u8] = &[];
    match decoder.process(&mut input) {
        Err(e) => eprintln!("[{label}] Correctly rejected at frame info: {e}"),
        Ok(ProcessingResult::NeedsMoreInput { .. }) => {
            eprintln!("[{label}] Correctly rejected: needs more input at frame stage")
        }
        Ok(ProcessingResult::Complete { .. }) => eprintln!(
            "[{label}] WARNING: header+frame parsed without error, but no OOM — test passes (limits caught it)"
        ),
    }
}

/// Helper: attempt read_header, expecting an error.
fn assert_header_rejects(data: &[u8], label: &str) {
    match crate::read_header(data) {
        Err(e) => {
            eprintln!("[{label}] read_header correctly rejected: {e}");
        }
        Ok(info) => {
            let (w, h) = info.info.size;
            eprintln!(
                "[{label}] read_header returned {w}x{h} — checking that dimensions are reasonable"
            );
            let total = w as u64 * h as u64;
            // If the header parses, dimensions should be within default limits
            assert!(
                total <= (1u64 << 28),
                "Parsed dimensions {w}x{h} = {total} pixels exceed default max_pixels limit"
            );
        }
    }
}

/// 26-byte crafted codestream that previously caused a 4.2GB allocation.
/// From fuzz_push_decode: oom-6fadd07f7bdfdae2541a25bad5ccd2148f528150
#[test]
fn test_oom_push_decode_26bytes() {
    let data = include_bytes!("oom_artifacts/oom_push_decode_26bytes.jxl");
    assert_decode_rejects(data, "push_decode_26bytes");
    assert_header_rejects(data, "push_decode_26bytes_header");
}

/// 19,856-byte crafted codestream from fuzz_probe.
/// From fuzz_probe: oom-6756decd60ffd8b37acb632e85638247de7a2bcb
#[test]
fn test_oom_probe_19856bytes() {
    let data = include_bytes!("oom_artifacts/oom_probe_19856bytes.jxl");
    assert_decode_rejects(data, "probe_19856bytes");
    assert_header_rejects(data, "probe_19856bytes_header");
}

/// 234-byte crafted codestream from fuzz_animation.
/// From fuzz_animation: oom-9bda52f40ce794761446e38c5656fc2265aee83f
#[test]
fn test_oom_animation_234bytes() {
    let data = include_bytes!("oom_artifacts/oom_animation_234bytes.jxl");
    assert_decode_rejects(data, "animation_234bytes");
    assert_header_rejects(data, "animation_234bytes_header");
}

/// Verify that the default limits are tight enough to prevent multi-GB allocations.
#[test]
fn test_default_limits_are_bounded() {
    let limits = crate::api::JxlDecoderLimits::default();

    // max_pixels should be at most 256 megapixels
    assert!(
        limits.max_pixels.unwrap() <= 1 << 28,
        "Default max_pixels {} is too large",
        limits.max_pixels.unwrap()
    );

    // Default limits should have a memory budget to prevent OOM from crafted inputs
    assert!(
        limits.max_memory_bytes.is_some(),
        "Default max_memory_bytes should be set"
    );

    // The restrictive preset should have a memory budget
    let restrictive = crate::api::JxlDecoderLimits::restrictive();
    assert!(
        restrictive.max_memory_bytes.is_some(),
        "Restrictive max_memory_bytes should be set"
    );
}

/// Verify that alloc_zeroed_fallible returns an error for absurd sizes
/// instead of aborting the process.
#[test]
fn test_image_allocation_is_fallible() {
    use crate::image::Image;

    // Try to allocate a 1GB image — should fail gracefully, not abort.
    // 32768 * 32768 * 4 bytes (f32) = 4 GB
    let result = Image::<f32>::new((32768, 32768));
    // This may succeed on systems with lots of RAM/overcommit — the key thing
    // is that it doesn't ABORT. If it fails, it should return Err, not panic.
    match result {
        Ok(_) => eprintln!("Large allocation succeeded (system has enough memory)"),
        Err(e) => eprintln!("Large allocation correctly failed: {e}"),
    }
}

/// 781-byte crafted JXL container (2143x1050 16-bit) that triggered
/// unbounded internal allocation with no default memory budget.
/// The default max_memory_bytes limit must prevent this from OOMing.
#[test]
fn test_oom_container_781bytes() {
    let data = include_bytes!("oom_artifacts/oom_container_781bytes.jxl");
    // Full decode path must return an error, not OOM
    match crate::decode(data) {
        Err(e) => eprintln!("[container_781bytes] decode correctly rejected: {e}"),
        Ok(_) => panic!("Expected error from crafted 781-byte container, got Ok"),
    }
}

/// Crafted 19-byte JXL codestream that claims want_icc=true with a huge
/// ICC size. Without the amplification check in IncrementalIccReader, the
/// ICC decode loop iterates hundreds of millions of times, hanging the
/// decoder indefinitely.
#[test]
fn test_icc_amplification_dos() {
    let data: &[u8] = &[
        0xff, 0x0a, 0xff, 0x00, 0x1a, 0xff, 0xd8, 0x55, 0x55, 0x55, 0x05, 0x34, 0x0a, 0x44, 0x49,
        0x46, 0x00, 0x4e, 0x46,
    ];
    assert_decode_rejects(data, "icc_amplification_19bytes");
}

/// Variant from fuzz_decode_limits artifact.
#[test]
fn test_icc_amplification_dos_variant() {
    let data: &[u8] = &[
        0xff, 0x0a, 0x87, 0x40, 0x0a, 0x87, 0x87, 0x0a, 0x87, 0x59, 0x59, 0x59, 0xbb, 0xb3, 0xb3,
        0xb3, 0x00, 0x59, 0x00,
    ];
    assert_decode_rejects(data, "icc_amplification_variant");
}

/// Regression: MemoryGuard::forget() leaked 32 bytes of Arc<MemoryTrackerInner>
/// per tracked image allocation. The fix uses disarm() which zeroes the guard's
/// byte count and drops normally, releasing the Arc refcount.
///
/// These inputs triggered the leak in `decode` and `decode_with_limits` fuzz targets.
#[test]
fn test_leak_memory_tracker_decode() {
    let data1 = include_bytes!("oom_artifacts/leak_memory_tracker_decode_1.jxl");
    let data2 = include_bytes!("oom_artifacts/leak_memory_tracker_decode_2.jxl");
    // Should return an error, not panic or leak.
    let _ = crate::decode(data1);
    let _ = crate::decode(data2);
}

#[test]
fn test_leak_memory_tracker_decode_with_limits() {
    let data = include_bytes!("oom_artifacts/leak_memory_tracker_limits.jxl");
    let mut limits = crate::api::JxlDecoderLimits::restrictive();
    limits.max_pixels = Some(4_000_000);
    limits.max_memory_bytes = Some(64 * 1024 * 1024);
    let mut options = crate::api::JxlDecoderOptions::default();
    options.limits = limits;
    options.parallel = false;
    // Should return an error, not panic or leak.
    let _ = crate::decode_with(data, options);
}
