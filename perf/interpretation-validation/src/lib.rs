// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![forbid(unsafe_code)]
//! Standalone native-encoder/native-decoder qualification with libjxl inspection.
#![cfg(test)]

use jxl_encoder::{
    AnimationFrame, AnimationParams, ImageMetadata, LosslessConfig, PixelLayout, api::ExtraChannel,
};
use std::{fs::OpenOptions, io::Write, path::PathBuf, process::Command};
use zenjxl_decoder::api::{
    JxlBitDepth, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, ProcessingResult,
    states,
};

fn complete<T, U>(value: ProcessingResult<T, U>) -> T {
    match value {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => {
            panic!("complete native fixture requested more input")
        }
    }
}

fn inspect(name: &str, encoded: &[u8]) -> String {
    inspect_mode(name, encoded, false)
}

fn inspect_mode(name: &str, encoded: &[u8], full: bool) -> String {
    let root = PathBuf::from(
        std::env::var_os("JXL_INTERPRETATION_ARTIFACTS")
            .expect("persistent artifact directory required"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!("{name}.jxl"));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap()
        .write_all(encoded)
        .unwrap();
    let tool = std::env::var_os("JXL_INTERPRETATION_ORACLE").expect("libjxl oracle required");
    let result = Command::new("timeout")
        .args(["--kill-after=5s", "30s"])
        .arg(tool)
        .arg(&path)
        .args(if full { vec!["full"] } else { Vec::new() })
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8(result.stdout).unwrap();
    eprintln!("{name}: {text}");
    text
}

#[test]
fn native_subsampled_extra_channels_match_independent_pixels_on_odd_canvas() {
    let (width, height) = (17usize, 13usize);
    let color: Vec<u8> = (0..width * height * 3)
        .map(|i| (i * 19 + 7) as u8)
        .collect();
    for shift in [1, 2, 3] {
        let count = width.div_ceil(1 << shift) * height.div_ceil(1 << shift);
        let values: Vec<u8> = (0..count).map(|i| (i * 37 + 11) as u8).collect();
        for role in 0..7 {
            let extra = match role {
                0 => ExtraChannel::from_alpha_buf(&values, false),
                1 => ExtraChannel::depth(&values),
                2 => ExtraChannel::spot_color(&values, [0.25, 0.5, 0.75, 1.0]),
                3 => ExtraChannel::selection_mask(&values),
                4 => ExtraChannel::black(&values),
                5 => ExtraChannel::cfa(&values, 2),
                6 => ExtraChannel::thermal(&values),
                _ => unreachable!(),
            }
            .with_dim_shift(shift);
            let extras = [extra];
            let encoded = LosslessConfig::new()
                .with_effort(3)
                .encode_request(
                    width.try_into().unwrap(),
                    height.try_into().unwrap(),
                    PixelLayout::Rgb8,
                )
                .with_extra_channels(&extras)
                .encode(&color)
                .unwrap();
            let independent = inspect_mode(&format!("shift-{shift}-role-{role}"), &encoded, true);
            assert_eq!(
                &rows(&independent, "extra ")[0][..5],
                &[0, role, 8, 0, shift]
            );
            let (mut d, mut input) = decoder(&encoded);
            assert_eq!(d.basic_info().extra_channels[0].dim_shift, shift);
            let mut format = d.current_pixel_format().clone();
            assert_eq!(format.color_type.samples_per_pixel(), 3);
            format.color_data_format = Some(JxlDataFormat::U8 { bit_depth: 8 });
            format.extra_channel_format = vec![Some(JxlDataFormat::U8 { bit_depth: 8 })];
            d.set_pixel_format(format);
            let frame = complete(d.process(&mut input).unwrap());
            let mut rgb = vec![0xa5; width * height * 3];
            let mut auxiliary = vec![0xa5; width * height];
            let mut buffers = [
                JxlOutputBuffer::new(&mut rgb, height, width * 3),
                JxlOutputBuffer::new(&mut auxiliary, height, width),
            ];
            let _ = complete(frame.process(&mut input, &mut buffers).unwrap());
            let planes = rows(&independent, "pixels ");
            assert_eq!(planes.len(), 2);
            assert_eq!(planes[0][0], 0);
            assert_eq!(planes[1][0], 1);
            // Spot color can contribute to composited color; black can alter
            // its color interpretation. The independent reader checks those.
            if role != 2 && role != 4 {
                assert_eq!(rgb, color);
            }
            // libjxl's U8 stage adds ordered dithering; this decoder's optional
            // U8 stage uses blue noise. Compare the independent float samples
            // under the requested no-dither round-to-even conversion instead.
            let quantized = |samples: &[u32]| {
                samples
                    .iter()
                    .map(|&bits| {
                        let sample = f32::from_bits(bits);
                        assert!(sample.is_finite());
                        (sample * 255.0).clamp(0.0, 255.0).round_ties_even() as u8
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(rgb, quantized(&planes[0][1..]));
            assert_eq!(auxiliary, quantized(&planes[1][1..]));
        }
    }
}

fn rows(text: &str, prefix: &str) -> Vec<Vec<u32>> {
    text.lines()
        .filter_map(|l| l.strip_prefix(prefix))
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect()
}

fn decoder(bytes: &[u8]) -> (JxlDecoder<states::WithImageInfo>, &[u8]) {
    let mut input = bytes;
    let d = complete(
        JxlDecoder::new(
            JxlDecoderOptions::default()
                .with_adjust_orientation(false)
                .with_dither_u8(false),
        )
        .process(&mut input)
        .unwrap(),
    );
    (d, input)
}

#[test]
fn native_extra_roles_preserve_source_precision_and_rendering_parameters() {
    let color: Vec<u8> = (0..8 * 6 * 3).map(|v| (v * 37 + 11) as u8).collect();
    let samples: Vec<u8> = (0..8 * 6).map(|v| (v * 29 + 7) as u8).collect();
    let spot = [0.25, 0.5, 0.75, 1.0];
    for role in 0..7 {
        let extras = [match role {
            0 => ExtraChannel::from_alpha_buf(&samples, false),
            1 => ExtraChannel::depth(&samples),
            2 => ExtraChannel::spot_color(&samples, spot),
            3 => ExtraChannel::selection_mask(&samples),
            4 => ExtraChannel::black(&samples),
            5 => ExtraChannel::cfa(&samples, 2),
            6 => ExtraChannel::thermal(&samples),
            _ => unreachable!(),
        }];
        let metadata = ImageMetadata::new()
            .with_intrinsic_size(13, 9)
            .with_intensity_target(1000.0)
            .with_min_nits(0.125)
            .with_relative_to_max_display(true)
            .with_linear_below(0.25);
        let encoded = LosslessConfig::new()
            .with_effort(3)
            .encode_request(8, 6, PixelLayout::Rgb8)
            .with_extra_channels(&extras)
            .with_metadata(&metadata)
            .encode(&color)
            .unwrap();
        let independent = inspect(&format!("role-{role}"), &encoded);
        let (mut d, mut input) = decoder(&encoded);
        assert_eq!(d.basic_info().extra_channels.len(), 1);
        assert_eq!(d.basic_info().extra_channels[0].ec_type as u32, role);
        let detail = d.extra_channel_details(0).unwrap();
        assert!(d.extra_channel_details(1).is_none());
        assert_eq!(detail.bit_depth, JxlBitDepth::Int { bits_per_sample: 8 });
        assert_eq!(detail.spot_color, (role == 2).then_some(spot));
        assert_eq!(detail.cfa_channel, (role == 5).then_some(2));
        let observed = rows(&independent, "extra ");
        assert_eq!(&observed[0][..4], &[0, role, 8, 0]);
        if role == 2 {
            assert_eq!(&observed[0][5..9], &spot.map(f32::to_bits));
        }
        if role == 5 {
            assert_eq!(observed[0][9], 2);
        }
        let info = d.basic_info();
        assert_eq!(info.intrinsic_size, Some((13, 9)));
        assert_eq!(rows(&independent, "display "), vec![vec![13, 9]]);
        let tone = &info.tone_mapping;
        let expected = vec![
            1000.0f32.to_bits(),
            0.125f32.to_bits(),
            1,
            0.25f32.to_bits(),
        ];
        assert_eq!(
            vec![
                tone.intensity_target.to_bits(),
                tone.min_nits.to_bits(),
                u32::from(tone.relative_to_max_display),
                tone.linear_below.to_bits()
            ],
            expected
        );
        assert_eq!(rows(&independent, "tone "), vec![expected]);
        // Read every native extra plane, even when it is not a display channel.
        let mut format = d.current_pixel_format().clone();
        format.color_data_format = Some(JxlDataFormat::U8 { bit_depth: 8 });
        format.extra_channel_format = vec![Some(JxlDataFormat::U8 { bit_depth: 8 })];
        d.set_pixel_format(format);
        let f = complete(d.process(&mut input).unwrap());
        assert_eq!(f.frame_timecode(), None);
        let mut rgb = vec![0; 8 * 6 * 3];
        let mut auxiliary = vec![0; 8 * 6];
        let mut buffers = [
            JxlOutputBuffer::new(&mut rgb, 6, 8 * 3),
            JxlOutputBuffer::new(&mut auxiliary, 6, 8),
        ];
        let d = complete(f.process(&mut input, &mut buffers).unwrap());
        assert_eq!(auxiliary, samples);
        assert_eq!(d.extra_channel_details(0), Some(detail));
    }
}

#[test]
fn selected_timecodes_keep_present_zero_and_survive_reverse_seeks() {
    let pixels: Vec<Vec<u8>> = (0..3)
        .map(|f| (0..8 * 6 * 3).map(|i| (i * 31 + f * 43) as u8).collect())
        .collect();
    for present in [false, true] {
        let codes = [0x01020304, 0, 0x09080706];
        let frames: Vec<_> = pixels
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let f = AnimationFrame::new(p, [3, 7, 2][i]).with_name(format!("frame-{i}"));
                if present {
                    f.with_timecode(codes[i])
                } else {
                    f
                }
            })
            .collect();
        let params = AnimationParams {
            tps_numerator: 30000,
            tps_denominator: 1001,
            num_loops: 4,
            ..Default::default()
        };
        let encoded = LosslessConfig::new()
            .with_effort(3)
            .with_auto_delta_frames(false)
            .encode_animation(8, 6, PixelLayout::Rgb8, &params, &frames)
            .unwrap();
        let independent = inspect(&format!("timecodes-{present}"), &encoded);
        assert_eq!(
            rows(&independent, "timecodes "),
            vec![vec![u32::from(present)]]
        );
        assert_eq!(
            rows(&independent, "frame ")
                .iter()
                .map(|r| r[1])
                .collect::<Vec<_>>(),
            if present { codes.to_vec() } else { vec![0; 3] }
        );
        let (mut d, mut input) = decoder(&encoded);
        while d.has_more_frames() {
            let f = complete(d.process(&mut input).unwrap());
            d = complete(f.skip_frame(&mut input).unwrap());
        }
        let discovered = d.scanned_frames().to_vec();
        assert_eq!(discovered.len(), 3);
        for i in [2, 0, 1, 2] {
            let (mut selected, _) = decoder(&encoded);
            let target = discovered[i].seek_target.unwrap();
            selected.start_new_frame(target);
            let mut input = &encoded[target.decode_start_file_offset as usize..];
            let f = complete(selected.process(&mut input).unwrap());
            assert_eq!(f.frame_timecode(), present.then_some(codes[i]));
            assert_eq!(f.frame_header().name, format!("frame-{i}"));
        }
    }
}

#[test]
fn native_float_alpha_retains_integer_vs_float_and_exponent_precision() {
    for (layout, bits, exponent, samples) in [
        (
            PixelLayout::RgbaLinearF16,
            16,
            5,
            [0x3400u16, 0x3800, 0x3a00, 0x3c00]
                .repeat(8 * 6)
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            PixelLayout::RgbaLinearF32,
            32,
            8,
            [0.25f32, 0.5, 0.75, 1.0]
                .repeat(8 * 6)
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>(),
        ),
    ] {
        let encoded = LosslessConfig::new()
            .with_effort(3)
            .encode(&samples, 8, 6, layout)
            .unwrap();
        let independent = inspect(&format!("float-alpha-{bits}"), &encoded);
        let (decoder, _) = decoder(&encoded);
        let expected = JxlBitDepth::Float {
            bits_per_sample: bits,
            exponent_bits_per_sample: exponent,
        };
        assert_eq!(decoder.basic_info().bit_depth, expected);
        let extra = decoder.extra_channel_details(0).unwrap();
        assert_eq!(extra.bit_depth, expected);
        assert_eq!(extra.spot_color, None);
        assert_eq!(extra.cfa_channel, None);
        assert_eq!(
            &rows(&independent, "extra ")[0][..4],
            &[0, 0, bits, exponent]
        );
    }
}
