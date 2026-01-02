// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests using synthetic JXL images generated with cjxl.
//!
//! These small images are designed to exercise specific code paths
//! that may have low coverage from other test suites.

use crate::api::{states, JxlDecoder, JxlDecoderOptions, ProcessingResult};

fn decode_synthetic(name: &str) -> Result<(usize, usize), String> {
    let path = format!(
        "{}/resources/test/synthetic/{}.jxl",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let data = std::fs::read(&path).map_err(|e| format!("Failed to read {}: {}", path, e))?;

    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data.as_slice();

    loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => {
                let info = result.basic_info();
                return Ok((info.size.0, info.size.1));
            }
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("{:?}", e)),
        }
    }
}

macro_rules! synthetic_test {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            match decode_synthetic($file) {
                Ok((w, h)) => {
                    eprintln!("PASS: {} decoded {}x{}", $file, w, h);
                }
                Err(e) => panic!("FAIL: {} - {}", $file, e),
            }
        }
    };
}

// === Color Encoding Tests ===
synthetic_test!(test_colorspace_srgb, "colorspace_srgb");
synthetic_test!(test_tf_srgb_q90, "tf_srgb_q90");

// === Bit Depth Tests ===
synthetic_test!(test_synth_8bit, "synth_8bit");
synthetic_test!(test_synth_16bit, "synth_16bit");
synthetic_test!(test_synth_8bit_gray, "synth_8bit_gray");
synthetic_test!(test_synth_16bit_gray, "synth_16bit_gray");

// === Palette Tests ===
synthetic_test!(test_palette_indexed, "palette_indexed");
synthetic_test!(test_palette_small, "palette_small");
synthetic_test!(test_palette_forced, "palette_forced");
synthetic_test!(test_palette_lossy, "palette_lossy");

// === Container Tests ===
synthetic_test!(test_container_bare, "container_bare");
synthetic_test!(test_container_forced_on, "container_forced_on");

// === Modular Predictor Tests ===
synthetic_test!(test_predictor_zero, "predictor_zero");
synthetic_test!(test_predictor_left, "predictor_left");
synthetic_test!(test_predictor_top, "predictor_top");
synthetic_test!(test_predictor_select, "predictor_select");
synthetic_test!(test_predictor_gradient, "predictor_gradient");
synthetic_test!(test_predictor_weighted, "predictor_weighted");

// === Alpha Channel Tests ===
synthetic_test!(test_alpha_rgba, "alpha_rgba");
synthetic_test!(test_alpha_gray, "alpha_gray");
synthetic_test!(test_alpha_16bit, "alpha_16bit");

// === Group Size Tests ===
synthetic_test!(test_group_128, "group_128");
synthetic_test!(test_group_256, "group_256");
synthetic_test!(test_group_512, "group_512");
synthetic_test!(test_group_1024, "group_1024");

// === VarDCT Tests ===
synthetic_test!(test_vardct_q50, "vardct_q50");
synthetic_test!(test_vardct_q90, "vardct_q90");
synthetic_test!(test_vardct_effort1, "vardct_effort1");
synthetic_test!(test_vardct_effort9, "vardct_effort9");

// === Squeeze Transform Tests ===
synthetic_test!(test_squeeze_default, "squeeze_default");
