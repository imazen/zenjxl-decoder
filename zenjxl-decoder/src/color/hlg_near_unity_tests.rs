// Copyright (c) Imazen LLC. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Positive-luminance vectors from unmodified libjxl v0.12 HlgOOTF_Base.
use super::*;
#[test]
fn hlg_near_unity_ootf_matches_native_vectors() {
    for (index, line) in include_str!("hlg_ootf_v012.tsv").lines().enumerate() {
        let v: Vec<f32> = line.split('\t').map(|x| x.parse().unwrap()).collect();
        assert_eq!(v.len(), 10);
        for precise in [false, true] {
            let (mut r, mut g, mut b) = (v[4], v[5], v[6]);
            let rows = [
                std::slice::from_mut(&mut r),
                std::slice::from_mut(&mut g),
                std::slice::from_mut(&mut b),
            ];
            if precise {
                hlg_ootf_inner_precise(v[0] as f64, [v[1], v[2], v[3]], rows);
            } else {
                hlg_ootf_inner(v[0], [v[1], v[2], v[3]], rows);
            }
            for (channel, (observed, expected)) in [r, g, b].into_iter().zip(&v[7..]).enumerate() {
                assert!(
                    observed.is_finite() && (observed - expected).abs() <= 0.000002,
                    "native HLG OOTF row {index}, precise {precise}, channel {channel}: {observed} vs {expected}, exponent {}",
                    v[0]
                );
            }
        }
    }
}
#[test]
fn hlg_near_unity_deadband_includes_both_exact_endpoints() {
    for exponent in [-0.01_f32, 0.0, 0.01] {
        for precise in [false, true] {
            let (mut r, mut g, mut b) = (0.125_f32, 0.25_f32, 0.5_f32);
            let rows = [
                std::slice::from_mut(&mut r),
                std::slice::from_mut(&mut g),
                std::slice::from_mut(&mut b),
            ];
            if precise {
                hlg_ootf_inner_precise(exponent as f64, [0.2126, 0.7152, 0.0722], rows);
            } else {
                hlg_ootf_inner(exponent, [0.2126, 0.7152, 0.0722], rows);
            }
            assert_eq!([r, g, b], [0.125, 0.25, 0.5]);
        }
    }
}

fn hlg_fixture() -> Vec<u8> {
    include_str!("hlg_near_unity_255.jxl.hex")
        .split_whitespace()
        .flat_map(|line| {
            line.as_bytes()
                .chunks_exact(2)
                .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
        })
        .collect()
}

#[test]
fn hlg_near_unity_complete_file_matches_native_original_signal() {
    use crate::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    for padding in [0, 5] {
        let data = hlg_fixture();
        let mut input = data.as_slice();
        let mut options = JxlDecoderOptions::default();
        options.limits.max_codestream_level = crate::api::JxlCodestreamLevel::Level10;
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let ProcessingResult::Complete { result: mut dwi } = decoder.process(&mut input).unwrap()
        else {
            panic!("fixture header incomplete")
        };
        assert_eq!(dwi.basic_info().size, (9, 3));
        assert_eq!(dwi.basic_info().tone_mapping.intensity_target, 255.0);
        let profile = dwi.embedded_color_profile().clone();
        let cpf = dwi.current_pixel_format().clone();
        assert_eq!(cpf.color_type.samples_per_pixel(), 3);
        assert!(cpf.extra_channel_format.is_empty());
        dwi.set_pixel_format(JxlPixelFormat {
            color_type: cpf.color_type,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        });
        dwi.set_output_color_profile(profile).unwrap();
        let stride = 9 * 3 + padding;
        let mut pixels = vec![f32::NAN; stride * 3];
        let ProcessingResult::Complete { result: dfi } = dwi.process(&mut input).unwrap() else {
            panic!("fixture frame header incomplete")
        };
        let buffer = bytemuck::cast_slice_mut(&mut pixels);
        let mut output = [if padding == 0 {
            JxlOutputBuffer::new(buffer, 3, 9 * 3 * 4)
        } else {
            JxlOutputBuffer::new_with_stride(buffer, 3, 9 * 3 * 4, stride * 4)
        }];
        let ProcessingResult::Complete { result: dwi } =
            dfi.process(&mut input, &mut output).unwrap()
        else {
            panic!("fixture pixels incomplete")
        };
        assert!(!dwi.has_more_frames());
        let expected = [
            f32::from_bits(0x3dffe91a),
            f32::from_bits(0x3e7ffd50),
            f32::from_bits(0x3efff7a5),
        ];
        for (row_index, row) in pixels.chunks_exact(stride).enumerate() {
            for (i, &value) in row[..27].iter().enumerate() {
                assert!(
                    value.is_finite() && (value - expected[i % 3]).abs() <= 0.0001,
                    "complete native HLG row {row_index}, sample {i}: {value} vs {}",
                    expected[i % 3]
                );
            }
            assert!(
                row[27..].iter().all(|v| v.is_nan()),
                "output row padding must stay untouched"
            );
        }
    }
}
