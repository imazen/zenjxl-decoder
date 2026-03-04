// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests specifically targeting low-coverage code paths.
//!
//! These tests use codec-corpus files that exercise specific features.
//! They are skipped when CODEC_CORPUS_PATH is not set.

use crate::api::{JxlDecoder, JxlDecoderOptions, states};
use std::path::PathBuf;

fn codec_corpus_path() -> Option<PathBuf> {
    // Try environment variable first
    if let Ok(path) = std::env::var("CODEC_CORPUS_PATH") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // Try common relative locations
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../codec-eval/codec-corpus"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../codec-eval/codec-corpus"),
    ];

    for p in candidates {
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn get_test_file(category: &str, name: &str) -> Option<PathBuf> {
    let corpus = codec_corpus_path()?;
    let path = corpus
        .join("jxl")
        .join(category)
        .join(format!("{}.jxl", name));
    if path.exists() { Some(path) } else { None }
}

fn decode_file(path: &PathBuf) -> Result<(usize, usize), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data.as_slice();

    loop {
        match decoder.process(&mut input) {
            Ok(crate::api::ProcessingResult::Complete { result }) => {
                let info = result.basic_info();
                return Ok((info.size.0, info.size.1));
            }
            Ok(crate::api::ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("{:?}", e)),
        }
    }
}

macro_rules! coverage_test {
    ($name:ident, $category:literal, $file:literal) => {
        #[test]
        fn $name() {
            let Some(path) = get_test_file($category, $file) else {
                eprintln!("Skipping {}: codec-corpus not available", $file);
                return;
            };
            match decode_file(&path) {
                Ok((w, h)) => eprintln!("PASS: {} decoded {}x{}", $file, w, h),
                Err(e) => panic!("FAIL: {} - {}", $file, e),
            }
        }
    };
}

// Color encoding coverage (headers/color_encoding.rs)
coverage_test!(
    test_colorspace_display_p3,
    "features",
    "colorspace_DisplayP3"
);
coverage_test!(
    test_colorspace_rec2100_hlg,
    "features",
    "colorspace_Rec2100HLG"
);
coverage_test!(
    test_colorspace_rec2100_pq,
    "features",
    "colorspace_Rec2100PQ"
);

// Container parsing coverage (container/parse.rs)
coverage_test!(test_container_forced, "features", "container_forced");
coverage_test!(test_compress_boxes_0, "features", "compress_boxes_0");
coverage_test!(test_compress_boxes_1, "features", "compress_boxes_1");
coverage_test!(test_no_container, "features", "no_container");

// Bit depth coverage (headers/bit_depth.rs)
coverage_test!(test_bitdepth_10, "features", "bitdepth_10");
coverage_test!(test_bitdepth_12, "features", "bitdepth_12");
coverage_test!(test_bitdepth_16, "features", "bitdepth_16");
coverage_test!(test_bitdepth_8, "features", "bitdepth_8");

// Blending coverage (features/blending.rs)
coverage_test!(test_keep_invisible_0, "features", "keep_invisible_0");
coverage_test!(test_keep_invisible_1, "features", "keep_invisible_1");

// Modular predictor coverage (frame/modular/predict.rs)
coverage_test!(test_modular_predictor_0, "features", "modular_predictor_0");
coverage_test!(test_modular_predictor_1, "features", "modular_predictor_1");
coverage_test!(test_modular_predictor_2, "features", "modular_predictor_2");
coverage_test!(test_modular_predictor_5, "features", "modular_predictor_5");
coverage_test!(test_modular_predictor_6, "features", "modular_predictor_6");
coverage_test!(
    test_modular_predictor_14,
    "features",
    "modular_predictor_14"
);
coverage_test!(
    test_modular_predictor_15,
    "features",
    "modular_predictor_15"
);

// ICC coverage (icc/tag.rs)
coverage_test!(test_custom_icc_profile, "features", "custom_icc_profile");

// Modular colorspace coverage
coverage_test!(
    test_modular_colorspace_0,
    "features",
    "modular_colorspace_0"
);
coverage_test!(
    test_modular_colorspace_1,
    "features",
    "modular_colorspace_1"
);
coverage_test!(
    test_modular_colorspace_6,
    "features",
    "modular_colorspace_6"
);

// Modular group size coverage
coverage_test!(
    test_modular_group_size_0,
    "features",
    "modular_group_size_0"
);
coverage_test!(
    test_modular_group_size_1,
    "features",
    "modular_group_size_1"
);
coverage_test!(
    test_modular_group_size_2,
    "features",
    "modular_group_size_2"
);
coverage_test!(
    test_modular_group_size_3,
    "features",
    "modular_group_size_3"
);

// Extra channels resampling
coverage_test!(test_ec_resampling_1x, "features", "ec_resampling_1x");
coverage_test!(test_ec_resampling_2x, "features", "ec_resampling_2x");
coverage_test!(test_ec_resampling_4x, "features", "ec_resampling_4x");

// Resampling coverage
coverage_test!(test_resampling_1x, "features", "resampling_1x");
coverage_test!(test_resampling_2x, "features", "resampling_2x");
coverage_test!(test_resampling_4x, "features", "resampling_4x");
coverage_test!(test_resampling_8x, "features", "resampling_8x");

// EPF levels
coverage_test!(test_epf_level_0, "features", "epf_level_0");
coverage_test!(test_epf_level_1, "features", "epf_level_1");
coverage_test!(test_epf_level_2, "features", "epf_level_2");
coverage_test!(test_epf_level_3, "features", "epf_level_3");

// Gaborish
coverage_test!(test_gaborish_0, "features", "gaborish_0");
coverage_test!(test_gaborish_1, "features", "gaborish_1");

// Animation variants
coverage_test!(test_animation_modular, "features", "animation_modular");
coverage_test!(
    test_animation_progressive,
    "features",
    "animation_progressive"
);
coverage_test!(
    test_animation_with_patches,
    "features",
    "animation_with_patches"
);
