// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests that `JxlDecoderOptions::reject_progressive` gates progressive content
//! during frame decode.
//!
//! The gate fires at the first non-preview frame header parse — before any of
//! the frame's passes are decoded. A frame is "progressive" when it is
//! multi-pass (`num_passes > 1`) or an `LFFrame`. Patch/blend dictionary frames
//! (`ReferenceOnly`) and `SkipProgressive` frames must NOT trip the gate: real
//! fixtures legitimately begin with a `ReferenceOnly` frame.
//!
//! Fixture frame structure (verified against the codestream parser):
//! - `progressive_ac.jxl`: `RegularFrame`, `num_passes = 3` → progressive.
//! - `basic.jxl`, `3x3_srgb_lossless.jxl`: `RegularFrame`, `num_passes = 1` →
//!   not progressive.
//! - `grayscale_patches_var_dct.jxl`: first frame is `ReferenceOnly` → not
//!   progressive (must decode even with the gate on).

use crate::api::{JxlDecoderOptions, decode_with};
use crate::error::Error;

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources/test")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()))
}

/// A multi-pass (`num_passes = 3`) frame must be rejected as soon as its header
/// is parsed when `reject_progressive` is set.
#[test]
fn progressive_multipass_rejected_when_gate_set() {
    let data = fixture("progressive_ac.jxl");
    // `JxlImage` is not `Debug`, so match on the result rather than `expect_err`.
    match decode_with(
        &data,
        JxlDecoderOptions {
            reject_progressive: true,
            ..Default::default()
        },
    ) {
        Ok(_) => {
            panic!("progressive (num_passes=3) frame must be rejected when reject_progressive=true")
        }
        Err(Error::ProgressiveRejected) => {}
        Err(other) => panic!("expected Error::ProgressiveRejected, got {other:?}"),
    }
}

/// The same progressive fixture must decode normally with the gate off (the
/// default), proving the gate is the only thing rejecting it.
#[test]
fn progressive_multipass_decodes_when_gate_clear() {
    let data = fixture("progressive_ac.jxl");
    let img = decode_with(
        &data,
        JxlDecoderOptions {
            reject_progressive: false,
            ..Default::default()
        },
    )
    .expect("progressive fixture must decode when reject_progressive=false");
    assert!(
        img.width > 0 && img.height > 0,
        "decoded progressive image must have non-zero dimensions"
    );
}

/// `reject_progressive` defaults to `false`, so a default-options decode of a
/// progressive fixture succeeds.
#[test]
fn progressive_multipass_decodes_with_default_options() {
    let data = fixture("progressive_ac.jxl");
    decode_with(&data, JxlDecoderOptions::default())
        .expect("progressive fixture must decode with default options (gate off by default)");
}

/// A single-pass (`num_passes = 1`) `RegularFrame` is not progressive and must
/// decode regardless of the gate.
#[test]
fn non_progressive_decodes_with_gate_set() {
    for name in ["basic.jxl", "3x3_srgb_lossless.jxl"] {
        let data = fixture(name);
        let img = decode_with(
            &data,
            JxlDecoderOptions {
                reject_progressive: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("non-progressive fixture {name} must decode with gate on: {e:?}")
        });
        assert!(
            img.width > 0 && img.height > 0,
            "{name}: decoded image must have non-zero dimensions"
        );
    }
}

/// A fixture whose first frame is `ReferenceOnly` (a patch/blend dictionary
/// frame) must NOT trip the gate even when `reject_progressive` is set.
#[test]
fn reference_only_first_frame_not_rejected() {
    let data = fixture("grayscale_patches_var_dct.jxl");
    let img = decode_with(
        &data,
        JxlDecoderOptions {
            reject_progressive: true,
            ..Default::default()
        },
    )
    .expect("ReferenceOnly-first-frame fixture must decode when reject_progressive=true");
    assert!(
        img.width > 0 && img.height > 0,
        "decoded patch image must have non-zero dimensions"
    );
}
