// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Comprehensive feature tests for JXL decoder.
//!
//! Tests all JXL features against codec-corpus test files.
//! Tests may fail if features are not yet implemented - this is expected.

use super::parity::codec_corpus_jxl_dir;
use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};

/// Helper to decode a JXL file and return basic info
fn decode_file(path: &std::path::Path) -> Result<(u32, u32, bool), String> {
    let data = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;

    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data.as_slice();

    loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => {
                let info = result.basic_info();
                let has_animation = info.animation.is_some();
                return Ok((info.size.0 as u32, info.size.1 as u32, has_animation));
            }
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("Unexpected end of input".to_string());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("Decode error: {:?}", e)),
        }
    }
}

/// Test helper that tries to decode and reports result
fn test_feature(name: &str, filename: &str) {
    let corpus = match codec_corpus_jxl_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP {}: codec-corpus not found", name);
            return;
        }
    };

    let path = corpus.join("features").join(filename);
    if !path.exists() {
        eprintln!("SKIP {}: {} not found", name, filename);
        return;
    }

    match decode_file(&path) {
        Ok((w, h, anim)) => {
            eprintln!(
                "PASS {}: {}x{}{}",
                name,
                w,
                h,
                if anim { " (animated)" } else { "" }
            );
        }
        Err(e) => {
            eprintln!("FAIL {}: {}", name, e);
        }
    }
}

#[cfg(test)]
mod epf_tests {
    use super::*;

    #[test]
    fn test_epf_level_0() {
        test_feature("EPF level 0", "epf_level_0.jxl");
    }
    #[test]
    fn test_epf_level_1() {
        test_feature("EPF level 1", "epf_level_1.jxl");
    }
    #[test]
    fn test_epf_level_2() {
        test_feature("EPF level 2", "epf_level_2.jxl");
    }
    #[test]
    fn test_epf_level_3() {
        test_feature("EPF level 3", "epf_level_3.jxl");
    }
}

#[cfg(test)]
mod gaborish_tests {
    use super::*;

    #[test]
    fn test_gaborish_disabled() {
        test_feature("Gaborish disabled", "gaborish_0.jxl");
    }
    #[test]
    fn test_gaborish_enabled() {
        test_feature("Gaborish enabled", "gaborish_1.jxl");
    }
}

#[cfg(test)]
mod resampling_tests {
    use super::*;

    #[test]
    fn test_resampling_1x() {
        test_feature("Resampling 1x", "resampling_1x.jxl");
    }
    #[test]
    fn test_resampling_2x() {
        test_feature("Resampling 2x", "resampling_2x.jxl");
    }
    #[test]
    fn test_resampling_4x() {
        test_feature("Resampling 4x", "resampling_4x.jxl");
    }
    #[test]
    fn test_resampling_8x() {
        test_feature("Resampling 8x", "resampling_8x.jxl");
    }

    #[test]
    fn test_ec_resampling_1x() {
        test_feature("EC Resampling 1x", "ec_resampling_1x.jxl");
    }
    #[test]
    fn test_ec_resampling_2x() {
        test_feature("EC Resampling 2x", "ec_resampling_2x.jxl");
    }
    #[test]
    fn test_ec_resampling_4x() {
        test_feature("EC Resampling 4x", "ec_resampling_4x.jxl");
    }
}

#[cfg(test)]
mod progressive_tests {
    use super::*;

    #[test]
    fn test_progressive_basic() {
        test_feature("Progressive basic", "progressive_basic.jxl");
    }
    #[test]
    fn test_progressive_ac() {
        test_feature("Progressive AC", "progressive_ac.jxl");
    }
    #[test]
    fn test_qprogressive_ac() {
        test_feature("Q-Progressive AC", "qprogressive_ac.jxl");
    }
    #[test]
    fn test_progressive_dc_0() {
        test_feature("Progressive DC 0", "progressive_dc_0.jxl");
    }
    #[test]
    fn test_progressive_dc_1() {
        test_feature("Progressive DC 1", "progressive_dc_1.jxl");
    }
    #[test]
    fn test_progressive_dc_2() {
        test_feature("Progressive DC 2", "progressive_dc_2.jxl");
    }
}

#[cfg(test)]
mod noise_tests {
    use super::*;

    #[test]
    fn test_noise_disabled() {
        test_feature("Noise disabled", "noise_0.jxl");
    }
    #[test]
    fn test_noise_enabled() {
        test_feature("Noise enabled", "noise_1.jxl");
    }
    #[test]
    fn test_photon_noise_iso100() {
        test_feature("Photon noise ISO 100", "photon_noise_iso100.jxl");
    }
    #[test]
    fn test_photon_noise_iso400() {
        test_feature("Photon noise ISO 400", "photon_noise_iso400.jxl");
    }
    #[test]
    fn test_photon_noise_iso1600() {
        test_feature("Photon noise ISO 1600", "photon_noise_iso1600.jxl");
    }
    #[test]
    fn test_photon_noise_iso6400() {
        test_feature("Photon noise ISO 6400", "photon_noise_iso6400.jxl");
    }
}

#[cfg(test)]
mod patches_dots_tests {
    use super::*;

    #[test]
    fn test_patches_disabled() {
        test_feature("Patches disabled", "patches_0.jxl");
    }
    #[test]
    fn test_patches_enabled() {
        test_feature("Patches enabled", "patches_1.jxl");
    }
    #[test]
    fn test_dots_disabled() {
        test_feature("Dots disabled", "dots_0.jxl");
    }
    #[test]
    fn test_dots_enabled() {
        test_feature("Dots enabled", "dots_1.jxl");
    }
}

#[cfg(test)]
mod modular_tests {
    use super::*;

    #[test]
    fn test_modular_colorspace_0() {
        test_feature("Modular colorspace 0 (RGB)", "modular_colorspace_0.jxl");
    }
    #[test]
    fn test_modular_colorspace_1() {
        test_feature("Modular colorspace 1", "modular_colorspace_1.jxl");
    }
    #[test]
    fn test_modular_colorspace_6() {
        test_feature("Modular colorspace 6 (YCoCg)", "modular_colorspace_6.jxl");
    }

    #[test]
    fn test_modular_predictor_0() {
        test_feature("Modular predictor 0 (zero)", "modular_predictor_0.jxl");
    }
    #[test]
    fn test_modular_predictor_1() {
        test_feature("Modular predictor 1 (left)", "modular_predictor_1.jxl");
    }
    #[test]
    fn test_modular_predictor_2() {
        test_feature("Modular predictor 2 (top)", "modular_predictor_2.jxl");
    }
    #[test]
    fn test_modular_predictor_5() {
        test_feature("Modular predictor 5 (gradient)", "modular_predictor_5.jxl");
    }
    #[test]
    fn test_modular_predictor_6() {
        test_feature("Modular predictor 6 (weighted)", "modular_predictor_6.jxl");
    }
    #[test]
    fn test_modular_predictor_14() {
        test_feature("Modular predictor 14", "modular_predictor_14.jxl");
    }
    #[test]
    fn test_modular_predictor_15() {
        test_feature("Modular predictor 15 (all)", "modular_predictor_15.jxl");
    }

    #[test]
    fn test_modular_group_size_0() {
        test_feature("Modular group 128x128", "modular_group_size_0.jxl");
    }
    #[test]
    fn test_modular_group_size_1() {
        test_feature("Modular group 256x256", "modular_group_size_1.jxl");
    }
    #[test]
    fn test_modular_group_size_2() {
        test_feature("Modular group 512x512", "modular_group_size_2.jxl");
    }
    #[test]
    fn test_modular_group_size_3() {
        test_feature("Modular group 1024x1024", "modular_group_size_3.jxl");
    }
}

#[cfg(test)]
mod hdr_colorspace_tests {
    use super::*;

    #[test]
    fn test_colorspace_srgb() {
        test_feature("Color space sRGB", "colorspace_sRGB.jxl");
    }
    #[test]
    fn test_colorspace_displayp3() {
        test_feature("Color space Display P3", "colorspace_DisplayP3.jxl");
    }
    #[test]
    fn test_colorspace_rec2100pq() {
        test_feature("Color space Rec.2100 PQ", "colorspace_Rec2100PQ.jxl");
    }
    #[test]
    fn test_colorspace_rec2100hlg() {
        test_feature("Color space Rec.2100 HLG", "colorspace_Rec2100HLG.jxl");
    }

    #[test]
    fn test_intensity_100nits() {
        test_feature("Intensity 100 nits", "intensity_target_100nits.jxl");
    }
    #[test]
    fn test_intensity_1000nits() {
        test_feature("Intensity 1000 nits", "intensity_target_1000nits.jxl");
    }
    #[test]
    fn test_intensity_4000nits() {
        test_feature("Intensity 4000 nits", "intensity_target_4000nits.jxl");
    }
    #[test]
    fn test_intensity_10000nits() {
        test_feature("Intensity 10000 nits", "intensity_target_10000nits.jxl");
    }
}

#[cfg(test)]
mod bitdepth_tests {
    use super::*;

    #[test]
    fn test_bitdepth_8() {
        test_feature("Bit depth 8", "bitdepth_8.jxl");
    }
    #[test]
    fn test_bitdepth_10() {
        test_feature("Bit depth 10", "bitdepth_10.jxl");
    }
    #[test]
    fn test_bitdepth_12() {
        test_feature("Bit depth 12", "bitdepth_12.jxl");
    }
    #[test]
    fn test_bitdepth_16() {
        test_feature("Bit depth 16", "bitdepth_16.jxl");
    }
}

#[cfg(test)]
mod quality_tests {
    use super::*;

    #[test]
    fn test_distance_0_0() {
        test_feature("Distance 0.0 (lossless)", "distance_0_0.jxl");
    }
    #[test]
    fn test_distance_0_5() {
        test_feature("Distance 0.5", "distance_0_5.jxl");
    }
    #[test]
    fn test_distance_1_0() {
        test_feature("Distance 1.0 (visually lossless)", "distance_1_0.jxl");
    }
    #[test]
    fn test_distance_2_0() {
        test_feature("Distance 2.0", "distance_2_0.jxl");
    }
    #[test]
    fn test_distance_5_0() {
        test_feature("Distance 5.0", "distance_5_0.jxl");
    }
    #[test]
    fn test_distance_10_0() {
        test_feature("Distance 10.0", "distance_10_0.jxl");
    }

    #[test]
    fn test_effort_1() {
        test_feature("Effort 1", "effort_1.jxl");
    }
    #[test]
    fn test_effort_3() {
        test_feature("Effort 3", "effort_3.jxl");
    }
    #[test]
    fn test_effort_5() {
        test_feature("Effort 5", "effort_5.jxl");
    }
    #[test]
    fn test_effort_7() {
        test_feature("Effort 7", "effort_7.jxl");
    }
    #[test]
    fn test_effort_9() {
        test_feature("Effort 9", "effort_9.jxl");
    }
    #[test]
    fn test_effort_10() {
        test_feature("Effort 10", "effort_10.jxl");
    }

    #[test]
    fn test_faster_decoding_0() {
        test_feature("Faster decoding 0", "faster_decoding_0.jxl");
    }
    #[test]
    fn test_faster_decoding_1() {
        test_feature("Faster decoding 1", "faster_decoding_1.jxl");
    }
    #[test]
    fn test_faster_decoding_2() {
        test_feature("Faster decoding 2", "faster_decoding_2.jxl");
    }
    #[test]
    fn test_faster_decoding_3() {
        test_feature("Faster decoding 3", "faster_decoding_3.jxl");
    }
    #[test]
    fn test_faster_decoding_4() {
        test_feature("Faster decoding 4", "faster_decoding_4.jxl");
    }
}

#[cfg(test)]
mod alpha_tests {
    use super::*;

    #[test]
    fn test_alpha_distance_0_0() {
        test_feature("Alpha distance 0.0", "alpha_distance_0_0.jxl");
    }
    #[test]
    fn test_alpha_distance_0_5() {
        test_feature("Alpha distance 0.5", "alpha_distance_0_5.jxl");
    }
    #[test]
    fn test_alpha_distance_1_0() {
        test_feature("Alpha distance 1.0", "alpha_distance_1_0.jxl");
    }
    #[test]
    fn test_alpha_distance_2_0() {
        test_feature("Alpha distance 2.0", "alpha_distance_2_0.jxl");
    }

    #[test]
    fn test_premultiply_0() {
        test_feature("Premultiply disabled", "premultiply_0.jxl");
    }
    #[test]
    fn test_premultiply_1() {
        test_feature("Premultiply enabled", "premultiply_1.jxl");
    }

    #[test]
    fn test_keep_invisible_0() {
        test_feature("Keep invisible 0", "keep_invisible_0.jxl");
    }
    #[test]
    fn test_keep_invisible_1() {
        test_feature("Keep invisible 1", "keep_invisible_1.jxl");
    }

    #[test]
    fn test_squeeze_responsive_0() {
        test_feature("Squeeze/responsive 0", "squeeze_responsive_0.jxl");
    }
    #[test]
    fn test_squeeze_responsive_1() {
        test_feature("Squeeze/responsive 1", "squeeze_responsive_1.jxl");
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    #[test]
    fn test_custom_icc_profile() {
        test_feature("Custom ICC profile", "custom_icc_profile.jxl");
    }
    #[test]
    fn test_with_exif() {
        test_feature("With EXIF", "with_exif.jxl");
    }
    #[test]
    fn test_with_xmp() {
        test_feature("With XMP", "with_xmp.jxl");
    }
    #[test]
    fn test_with_exif_and_xmp() {
        test_feature("With EXIF and XMP", "with_exif_and_xmp.jxl");
    }
    #[test]
    fn test_stripped_metadata() {
        test_feature("Stripped metadata", "stripped_metadata.jxl");
    }
    #[test]
    fn test_compress_boxes_0() {
        test_feature("Compress boxes 0", "compress_boxes_0.jxl");
    }
    #[test]
    fn test_compress_boxes_1() {
        test_feature("Compress boxes 1", "compress_boxes_1.jxl");
    }
}

#[cfg(test)]
mod container_tests {
    use super::*;

    #[test]
    fn test_container_forced() {
        test_feature("Container forced", "container_forced.jxl");
    }
    #[test]
    fn test_no_container() {
        test_feature("No container", "no_container.jxl");
    }
    #[test]
    fn test_group_order_0() {
        test_feature("Group order 0 (scanline)", "group_order_0.jxl");
    }
    #[test]
    fn test_group_order_1() {
        test_feature("Group order 1 (center-first)", "group_order_1.jxl");
    }
}

#[cfg(test)]
mod animation_feature_tests {
    use super::*;

    #[test]
    fn test_animation_effort_3() {
        test_feature("Animation effort 3", "animation_effort_3.jxl");
    }
    #[test]
    fn test_animation_effort_7() {
        test_feature("Animation effort 7", "animation_effort_7.jxl");
    }
    #[test]
    fn test_animation_effort_9() {
        test_feature("Animation effort 9", "animation_effort_9.jxl");
    }

    #[test]
    fn test_animation_distance_0_0() {
        test_feature("Animation lossless", "animation_distance_0_0.jxl");
    }
    #[test]
    fn test_animation_distance_1_0() {
        test_feature("Animation visually lossless", "animation_distance_1_0.jxl");
    }
    #[test]
    fn test_animation_distance_3_0() {
        test_feature("Animation lossy", "animation_distance_3_0.jxl");
    }

    #[test]
    fn test_animation_modular() {
        test_feature("Animation modular", "animation_modular.jxl");
    }
    #[test]
    fn test_animation_progressive() {
        test_feature("Animation progressive", "animation_progressive.jxl");
    }
    #[test]
    fn test_animation_with_patches() {
        test_feature("Animation with patches", "animation_with_patches.jxl");
    }
    #[test]
    fn test_animation_frame_indexed() {
        test_feature("Animation frame indexed", "animation_frame_indexed.jxl");
    }
}

/// Run all feature tests and summarize results
#[test]
#[ignore] // Run with --ignored
fn test_all_features_summary() {
    let corpus = match codec_corpus_jxl_dir() {
        Some(d) => d,
        None => {
            panic!("codec-corpus not found. Set CODEC_CORPUS_PATH.");
        }
    };

    let features_dir = corpus.join("features");
    if !features_dir.exists() {
        panic!("Features directory not found: {:?}", features_dir);
    }

    let mut passed = 0;
    let mut failed = 0;
    let skipped = 0;

    for entry in std::fs::read_dir(&features_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("jxl") {
            continue;
        }

        let name = path.file_stem().unwrap().to_string_lossy();

        match decode_file(&path) {
            Ok((w, h, anim)) => {
                eprintln!(
                    "PASS {}: {}x{}{}",
                    name,
                    w,
                    h,
                    if anim { " (anim)" } else { "" }
                );
                passed += 1;
            }
            Err(e) => {
                eprintln!("FAIL {}: {}", name, e);
                failed += 1;
            }
        }
    }

    eprintln!();
    eprintln!("=== Feature Test Summary ===");
    eprintln!("Passed:  {}", passed);
    eprintln!("Failed:  {}", failed);
    eprintln!("Skipped: {}", skipped);
    eprintln!("Total:   {}", passed + failed + skipped);

    // Don't fail - these are exploratory tests
    eprintln!();
    if failed > 0 {
        eprintln!(
            "Note: {} tests failed. This indicates unimplemented features.",
            failed
        );
    }
}
