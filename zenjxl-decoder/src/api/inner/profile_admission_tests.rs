// Copyright (c) Imazen LLC. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Admission controls; the CMS observer does not produce converted pixels.
#[test]
fn hlg_near_unity_non_xyb_profile_change_requires_cms() {
    use crate::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    let data: Vec<u8> = include_str!("../../color/hlg_forward_255.jxl.hex")
        .split_whitespace()
        .flat_map(|line| {
            line.as_bytes()
                .chunks_exact(2)
                .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
        })
        .collect();
    let mut options = JxlDecoderOptions::default();
    options.limits.max_codestream_level = crate::api::JxlCodestreamLevel::Level10;
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data.as_slice();
    let ProcessingResult::Complete { result: mut dwi } = decoder.process(&mut input).unwrap()
    else {
        panic!("forward source header incomplete")
    };
    assert_eq!(dwi.basic_info().size, (9, 3));
    assert_eq!(dwi.basic_info().tone_mapping.intensity_target, 255.0);
    let original = dwi.embedded_color_profile().clone();
    let prior_output = dwi.output_color_profile().clone();
    let icc_error = dwi
        .set_output_color_profile(crate::api::JxlColorProfile::Icc(vec![]))
        .expect_err("ICC output without CMS must refuse");
    assert!(matches!(
        icc_error.error(),
        crate::error::Error::ICCOutputNoCMS
    ));
    assert_eq!(*dwi.output_color_profile(), prior_output);
    let profile = original.with_linear_tf().unwrap();
    let error = dwi
        .set_output_color_profile(profile)
        .expect_err("non-XYB profile change without CMS must refuse");
    assert!(matches!(
        error.error(),
        crate::error::Error::NonXybOutputNoCMS
    ));
    // Rejection must leave the supported original-profile decode available.
    let format = dwi.current_pixel_format().clone();
    assert_eq!(format.color_type.samples_per_pixel(), 3);
    assert!(format.extra_channel_format.is_empty());
    dwi.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    });
    dwi.set_output_color_profile(original).unwrap();
    let ProcessingResult::Complete { result: dfi } = dwi.process(&mut input).unwrap() else {
        panic!("forward frame header incomplete")
    };
    let mut pixels = vec![f32::NAN; 9 * 3 * 3];
    let mut output = [JxlOutputBuffer::new(
        bytemuck::cast_slice_mut(&mut pixels),
        3,
        9 * 3 * 4,
    )];
    let ProcessingResult::Complete { result: dwi } = dfi.process(&mut input, &mut output).unwrap()
    else {
        panic!("forward pixels incomplete")
    };
    assert!(!dwi.has_more_frames());
    for (i, &value) in pixels.iter().enumerate() {
        assert_eq!(
            value,
            [0.125_f32, 0.25, 0.5][i % 3],
            "supported lossless source sample {i}"
        );
    }
}

#[test]
fn hlg_near_unity_profile_admission_with_cms() {
    use crate::api::{
        JxlCms, JxlCmsTransformer, JxlColorProfile, JxlDecoder, JxlDecoderOptions,
        ProcessingResult, states,
    };
    // This checks admission only. Any attempt to perform a color conversion
    // through this observer is a test failure, not substitute CMS output.
    struct AdmissionOnlyCms;
    impl JxlCms for AdmissionOnlyCms {
        fn initialize_transforms(
            &self,
            _: usize,
            _: usize,
            _: JxlColorProfile,
            _: JxlColorProfile,
            _: f32,
        ) -> crate::error::Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send + Sync>>)> {
            panic!("admission control must not run a CMS transform")
        }
    }
    let data: Vec<u8> = include_str!("../../color/hlg_forward_255.jxl.hex")
        .split_whitespace()
        .flat_map(|line| {
            line.as_bytes()
                .chunks_exact(2)
                .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
        })
        .collect();
    let mut options = JxlDecoderOptions::default().with_cms(Box::new(AdmissionOnlyCms));
    options.limits.max_codestream_level = crate::api::JxlCodestreamLevel::Level10;
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data.as_slice();
    let ProcessingResult::Complete {
        result: mut decoder,
    } = decoder.process(&mut input).unwrap()
    else {
        panic!("CMS admission source header incomplete")
    };
    let requested = decoder.embedded_color_profile().with_linear_tf().unwrap();
    decoder.set_output_color_profile(requested.clone()).unwrap();
    assert_eq!(*decoder.output_color_profile(), requested);
}
