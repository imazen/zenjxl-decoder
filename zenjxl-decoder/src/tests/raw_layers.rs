// Copyright (c) Imazen LLC.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Native local-plane probe and qualified public raw-layer output.

use std::{cell::RefCell, io::Write, rc::Rc};

use crate::{
    api::decoder::tests::{decode, decode_with_layer_capture},
    api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    },
    frame::Frame,
    headers::frame_header::{BlendingInfo, FrameType},
    image::Image,
};

fn advance<S, T>(
    mut state: S,
    input: &mut &[u8],
    chunk: usize,
    mut step: impl FnMut(S, &mut &[u8]) -> crate::error::Result<ProcessingResult<T, S>>,
) -> crate::error::Result<T> {
    let mut pending = 0usize;
    for _ in 0..input.len().saturating_mul(2).saturating_add(256) {
        pending = pending.saturating_add(chunk).min(input.len());
        let mut offered = &input[..pending];
        let result = step(state, &mut offered)?;
        let consumed = pending - offered.len();
        *input = &input[consumed..];
        pending -= consumed;
        match result {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(
                    consumed != 0 || pending < input.len(),
                    "complete input stalled"
                );
                state = fallback;
            }
        }
    }
    panic!("bounded input driver exhausted");
}

fn native_layers(bytes: &[u8], chunk: usize, simple: bool) -> Vec<Layer> {
    let mut input = bytes;
    let options = JxlDecoderOptions::default()
        .with_coalescing(false)
        .with_adjust_orientation(false)
        .with_render_spot_colors(false);
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder = advance(decoder, &mut input, chunk, |d, i| d.process(i)).unwrap();
    decoder.set_use_simple_pipeline(simple);
    let extra_count = decoder.basic_info().extra_channels.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); extra_count],
    });
    if decoder.embedded_color_profile().is_cmyk() {
        assert_eq!(
            decoder.output_color_profile(),
            decoder.embedded_color_profile(),
            "CMYK interpretation must be available before the frame header"
        );
    }
    let mut layers = Vec::new();
    while decoder.has_more_frames() {
        let frame = advance(decoder, &mut input, chunk, |d, i| d.process(i)).unwrap();
        let info = frame.frame_layer_info();
        assert_eq!(frame.frame_header().size, info.size);
        let (width, height) = info.size;
        let padding = if chunk == usize::MAX { 0 } else { 7 };
        let sentinel = f32::from_bits(0x7fc00001);
        let mut planes: Vec<_> = (0..=extra_count)
            .map(|channel| {
                let active = width * if channel == 0 { 3 } else { 1 };
                (
                    active,
                    active + padding,
                    vec![sentinel; (active + padding) * height],
                )
            })
            .collect();
        let mut outputs: Vec<_> = planes
            .iter_mut()
            .map(|(active, stride, data)| {
                JxlOutputBuffer::new_with_stride(
                    bytemuck::cast_slice_mut(data),
                    height,
                    *active * 4,
                    *stride * 4,
                )
            })
            .collect();
        decoder = advance(frame, &mut input, chunk, |d, i| d.process(i, &mut outputs)).unwrap();
        drop(outputs);
        for (active, stride, data) in &planes {
            for row in data.chunks_exact(*stride) {
                assert!(row[..*active].iter().all(|p| p.is_finite()));
                assert!(
                    row[*active..]
                        .iter()
                        .all(|p| p.to_bits() == sentinel.to_bits()),
                    "row padding changed"
                );
            }
        }
        let mut samples: Vec<Vec<u32>> = (0..3)
            .map(|channel| {
                planes[0]
                    .2
                    .chunks_exact(planes[0].1)
                    .flat_map(|row| {
                        row[..planes[0].0]
                            .chunks_exact(3)
                            .map(move |p| p[channel].to_bits())
                    })
                    .collect()
            })
            .collect();
        samples.extend(planes.iter().skip(1).map(|(active, stride, data)| {
            data.chunks_exact(*stride)
                .flat_map(|row| row[..*active].iter().map(|p| p.to_bits()))
                .collect()
        }));
        let to_json = |value: &crate::api::JxlLayerBlend| {
            serde_json::json!({
                "mode": match value.mode {
                    crate::api::JxlLayerBlendMode::Replace => 0,
                    crate::api::JxlLayerBlendMode::Add => 1,
                    crate::api::JxlLayerBlendMode::Blend => 2,
                    crate::api::JxlLayerBlendMode::AlphaWeightedAdd => 3,
                    crate::api::JxlLayerBlendMode::Multiply => 4,
                },
                "source": value.source, "alpha": value.alpha_channel, "clamp": u32::from(value.clamp),
            })
        };
        layers.push(Layer {
            origin: info.origin,
            size: info.size,
            planes: samples,
            composition: serde_json::json!({
                "blend": to_json(&info.color_blend),
                "extra_blends": info.extra_channel_blends.iter().map(to_json).collect::<Vec<_>>(),
                "save_as_reference": info.saved_reference.unwrap_or(0),
                "duration": info.duration_ticks, "is_last": u32::from(info.is_last),
            }),
        });
    }
    layers
}

#[test]
fn raw_layer_output_uses_the_public_decoder_without_compositing() {
    for name in [
        "multiple_layers_noise_spline.jxl",
        "conformance_test_images/cmyk_layers.jxl",
    ] {
        let bytes = crate::util::test::fixture_bytes(name);
        let captured = Rc::new(RefCell::new(Vec::new()));
        let target = captured.clone();
        decode_with_layer_capture(
            &bytes,
            usize::MAX,
            false,
            false,
            Some(Box::new(move |frame, _| {
                if let Some(layer) = layer(frame) {
                    target.borrow_mut().push(layer);
                }
                Ok(())
            })),
            true,
        )
        .unwrap();
        let expected = Rc::try_unwrap(captured).unwrap().into_inner();
        for simple in [false, true] {
            for chunk in [usize::MAX, 16_384] {
                let actual = native_layers(&bytes, chunk, simple);
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(
                        (actual.origin, actual.size),
                        (expected.origin, expected.size)
                    );
                    assert_eq!(actual.composition, expected.composition);
                    assert_eq!(actual.planes.len(), expected.planes.len());
                    for (actual, expected) in actual.planes.iter().zip(&expected.planes) {
                        exact_samples(actual, expected);
                    }
                }
                if !simple && chunk == usize::MAX {
                    export(name, &actual);
                }
            }
        }
    }
}

#[test]
fn raw_layer_output_refuses_unqualified_requests() {
    use crate::error::{Error, ErrorClass};
    for bad in 0..4 {
        let mut options = JxlDecoderOptions::default()
            .with_coalescing(false)
            .with_adjust_orientation(false);
        match bad {
            0 => options.adjust_orientation = true,
            1 => options.premultiply_output = true,
            2 => options.skip_preview = false,
            3 => options.desired_intensity_target = Some(80.0),
            _ => unreachable!(),
        }
        let mut input: &[u8] = &[];
        let error = JxlDecoder::new(options)
            .process(&mut input)
            .err()
            .expect("ignored transform request");
        assert!(matches!(error.error(), Error::UncoalescedOutputUnsupported));
        assert_eq!(error.error().kind(), ErrorClass::Unsupported);
    }
    for bad in 0..3 {
        let name = if bad == 2 {
            "conformance_test_images/cmyk_layers.jxl"
        } else {
            "basic.jxl"
        };
        let bytes = crate::util::test::fixture_bytes(name);
        let mut input = bytes.as_slice();
        let options = JxlDecoderOptions::default()
            .with_coalescing(false)
            .with_adjust_orientation(false);
        let decoder = JxlDecoder::new(options);
        let mut decoder = advance(decoder, &mut input, usize::MAX, |d, i| d.process(i)).unwrap();
        let extras = decoder.basic_info().extra_channels.len();
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(if bad == 0 {
                JxlDataFormat::U8 { bit_depth: 8 }
            } else {
                JxlDataFormat::f32()
            }),
            extra_channel_format: vec![None; extras + usize::from(bad == 1)],
        });
        let error = decoder
            .process(&mut input)
            .err()
            .expect("ignored output requirement");
        assert!(matches!(
            error.error(),
            Error::UncoalescedOutputRequirement(_)
        ));
        assert_eq!(error.error().kind(), ErrorClass::Unsupported);
    }
}

#[test]
fn raw_layer_output_refusals_preserve_sequential_decode() {
    use crate::error::Error;
    let bytes = crate::util::test::fixture_bytes("basic.jxl");
    let mut scan_input = bytes.as_slice();
    let scanner = JxlDecoder::new(JxlDecoderOptions::default().with_scan_frames_only(true));
    let scanner = advance(scanner, &mut scan_input, usize::MAX, |d, i| d.process(i)).unwrap();
    let scanner = advance(scanner, &mut scan_input, usize::MAX, |d, i| d.process(i)).unwrap();
    let target = scanner.scanned_frames()[0].seek_target.unwrap();

    let mut input = bytes.as_slice();
    let options = JxlDecoderOptions::default()
        .with_coalescing(false)
        .with_adjust_orientation(false);
    let decoder = JxlDecoder::new(options);
    let mut decoder = advance(decoder, &mut input, usize::MAX, |d, i| d.process(i)).unwrap();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; decoder.basic_info().extra_channels.len()],
    });
    let error = decoder.try_start_new_frame(target).unwrap_err();
    assert!(matches!(
        error.error(),
        Error::UncoalescedOutputRequirement(_)
    ));
    assert!(decoder.scanned_frames().is_empty());
    let mut frame = advance(decoder, &mut input, usize::MAX, |d, i| d.process(i)).unwrap();
    let (width, height) = frame.frame_header().size;
    let sentinel = f32::from_bits(0x7fc00001);
    let mut data = vec![sentinel; width * height * 3];
    {
        let mut output = [JxlOutputBuffer::new(
            bytemuck::cast_slice_mut(&mut data),
            height,
            width * 12,
        )];
        let error = frame.flush_pixels(&mut output).unwrap_err();
        assert!(matches!(
            error.error(),
            Error::UncoalescedOutputRequirement(_)
        ));
    }
    assert!(data.iter().all(|p| p.to_bits() == sentinel.to_bits()));
    let mut output = [JxlOutputBuffer::new(
        bytemuck::cast_slice_mut(&mut data),
        height,
        width * 12,
    )];
    let decoder = advance(frame, &mut input, usize::MAX, |d, i| {
        d.process(i, &mut output)
    })
    .unwrap();
    assert!(!decoder.has_more_frames());
    drop(output);
    let expected = native_layers(&bytes, usize::MAX, false);
    assert_eq!(expected.len(), 1);
    for (index, pixel) in data.chunks_exact(3).enumerate() {
        for (channel, sample) in pixel.iter().enumerate() {
            assert_eq!(sample.to_bits(), expected[0].planes[channel][index]);
        }
    }
}

#[test]
fn raw_layer_output_rechecks_low_level_format_before_writing() {
    use crate::{api::JxlDecoderInner, error::Error};
    let bytes = crate::util::test::fixture_bytes("basic.jxl");
    let mut input = bytes.as_slice();
    let mut decoder = JxlDecoderInner::new(
        JxlDecoderOptions::default()
            .with_coalescing(false)
            .with_adjust_orientation(false),
    );
    assert!(matches!(
        decoder.process(&mut input, None).unwrap(),
        ProcessingResult::Complete { .. }
    ));
    let format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; decoder.basic_info().unwrap().extra_channels.len()],
    };
    decoder.set_pixel_format(format.clone());
    assert!(matches!(
        decoder.process(&mut input, None).unwrap(),
        ProcessingResult::Complete { .. }
    ));
    let (width, height) = decoder.frame_header().unwrap().size;
    let mut data = vec![f32::NAN; width * height * 3];
    let mut output = [JxlOutputBuffer::new(
        bytemuck::cast_slice_mut(&mut data),
        height,
        width * 12,
    )];
    decoder.set_pixel_format(JxlPixelFormat {
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        ..format.clone()
    });
    let remaining = input.len();
    let error = decoder.process(&mut input, Some(&mut output)).unwrap_err();
    assert!(matches!(
        error.error(),
        Error::UncoalescedOutputRequirement(_)
    ));
    assert_eq!(input.len(), remaining);
    drop(output);
    assert!(data.iter().all(|p| p.to_bits() == f32::NAN.to_bits()));
    decoder.set_pixel_format(format);
    let mut output = [JxlOutputBuffer::new(
        bytemuck::cast_slice_mut(&mut data),
        height,
        width * 12,
    )];
    assert!(matches!(
        decoder.process(&mut input, Some(&mut output)).unwrap(),
        ProcessingResult::Complete { .. }
    ));
    drop(output);
    assert!(data.iter().all(|p| p.is_finite()));
}

#[derive(Debug, PartialEq, Eq)]
struct Layer {
    origin: (i32, i32),
    size: (usize, usize),
    planes: Vec<Vec<u32>>,
    composition: serde_json::Value,
}

fn bits(plane: &Image<f32>) -> Vec<u32> {
    (0..plane.size().1)
        .flat_map(|y| plane.row(y).iter().map(|sample| sample.to_bits()))
        .collect()
}

fn exact_samples(actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(mismatch.is_none(), "first differing sample: {mismatch:?}");
}

fn layer(frame: &Frame) -> Option<Layer> {
    let header = frame.header();
    if header.frame_type != FrameType::RegularFrame {
        assert!(frame.raw_layer_data.is_none());
        return None;
    }
    let planes = frame
        .raw_layer_data
        .as_ref()
        .expect("regular layer not captured");
    let size = header.size_upsampled();
    for plane in planes {
        assert_eq!(plane.size(), size, "capture clipped a layer to the canvas");
        for y in 0..size.1 {
            assert!(
                plane.row(y).iter().all(|sample| sample.is_finite()),
                "unwritten or nonfinite layer sample at row {y}"
            );
        }
    }
    Some(Layer {
        origin: (header.x0, header.y0),
        size,
        planes: planes.iter().map(bits).collect(),
        composition: serde_json::json!({
            "blend": blend(&header.blending_info),
            "extra_blends": header.ec_blending_info.iter().map(blend).collect::<Vec<_>>(),
            "save_as_reference": header.save_as_reference,
            "duration": header.duration,
            "is_last": u32::from(header.is_last),
        }),
    })
}

fn blend(info: &BlendingInfo) -> serde_json::Value {
    serde_json::json!({
        "mode": info.mode as u32, "source": info.source,
        "alpha": info.alpha_channel, "clamp": u32::from(info.clamp),
    })
}

// Optional artifact export changes no assertions. The independent oracle gate
// explicitly supplies this directory, then requires and compares every file.
fn export(name: &str, layers: &[Layer]) {
    if let Some(root) = std::env::var_os("JXL_RAW_LAYER_ARTIFACTS") {
        let root = std::path::Path::new(&root).join(name.replace('/', "_"));
        std::fs::create_dir(&root).unwrap();
        let mut metadata = Vec::new();
        for (index, layer) in layers.iter().enumerate() {
            let mut files = Vec::new();
            for (channel, samples) in layer.planes.iter().enumerate() {
                let name = format!("layer-{index}-{channel}.f32le");
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join(&name))
                    .unwrap();
                let mut file = std::io::BufWriter::new(file);
                for sample in samples {
                    file.write_all(&sample.to_le_bytes()).unwrap();
                }
                file.flush().unwrap();
                files.push(name);
            }
            let mut item = layer.composition.clone();
            let item = item.as_object_mut().unwrap();
            item.insert("origin".into(), serde_json::json!(layer.origin));
            item.insert("size".into(), serde_json::json!(layer.size));
            item.insert("planes".into(), serde_json::json!(files));
            metadata.push(serde_json::Value::Object(item.clone()));
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("layers.json"))
            .unwrap();
        serde_json::to_writer_pretty(file, &metadata).unwrap();
    }
}

#[test]
fn raw_layer_probe_keeps_reference_pixels_and_all_local_samples() {
    for name in [
        "multiple_layers_noise_spline.jxl",
        "conformance_test_images/cmyk_layers.jxl",
    ] {
        let bytes = crate::util::test::fixture_bytes(name);
        let (_, baseline) = decode(&bytes, usize::MAX, false, false, None).unwrap();
        let mut expected = None;
        for simple in [false, true] {
            for chunk in [usize::MAX, 16_384] {
                let captured = Rc::new(RefCell::new(Vec::new()));
                let target = captured.clone();
                let (_, output) = decode_with_layer_capture(
                    &bytes,
                    chunk,
                    simple,
                    false,
                    Some(Box::new(move |frame, _| {
                        if let Some(layer) = layer(frame) {
                            target.borrow_mut().push(layer);
                        }
                        Ok(())
                    })),
                    true,
                )
                .unwrap();
                assert_eq!(output.len(), baseline.len());
                for (actual, reference) in output.iter().zip(&baseline) {
                    assert_eq!(actual.len(), reference.len());
                    for (actual, reference) in actual.iter().zip(reference) {
                        exact_samples(&bits(actual), &bits(reference));
                    }
                }
                let captured = Rc::try_unwrap(captured).unwrap().into_inner();
                assert!(!captured.is_empty());
                if let Some(expected) = &expected {
                    let expected: &Vec<Layer> = expected;
                    assert_eq!(captured.len(), expected.len());
                    for (actual, expected) in captured.iter().zip(expected) {
                        assert_eq!(
                            (actual.origin, actual.size),
                            (expected.origin, expected.size)
                        );
                        assert_eq!(actual.planes.len(), expected.planes.len());
                        assert_eq!(actual.composition, expected.composition);
                        for (actual, expected) in actual.planes.iter().zip(&expected.planes) {
                            exact_samples(actual, expected);
                        }
                    }
                } else {
                    expected = Some(captured);
                }
            }
        }
        let expected = expected.unwrap();
        if name == "multiple_layers_noise_spline.jxl" {
            assert_eq!(
                expected
                    .iter()
                    .map(|l| (l.origin, l.size))
                    .collect::<Vec<_>>(),
                vec![
                    ((0, 0), (2048, 1152)),
                    ((512, 164), (1024, 576)),
                    ((512, 640), (1024, 576)),
                    ((512, 164), (1024, 576)),
                    ((1860, 950), (1024, 576)),
                ]
            );
        }
        export(name, &expected);
    }
}
