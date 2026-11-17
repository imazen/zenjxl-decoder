// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Typestate adaptation of upstream's partial-input and progressive callbacks.

use crate::{
    api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    },
    error::Result,
    image::{Image, Rect},
};

#[derive(Default)]
pub(super) struct PartialFrames {
    pub frames: Vec<Vec<Image<f32>>>,
    pub flushes: usize,
}

fn output_buffers(buffers: &mut [Image<f32>]) -> Vec<JxlOutputBuffer<'_>> {
    buffers
        .iter_mut()
        .map(|image| {
            let size = image.size();
            JxlOutputBuffer::from_image_rect_mut(
                image
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size,
                    })
                    .into_raw(),
            )
        })
        .collect()
}

pub(super) fn decode_partial(
    data: &[u8],
    chunk: usize,
    parallel: bool,
    mut on_flush: impl FnMut(usize, usize, &[Image<f32>]),
) -> Result<PartialFrames> {
    assert!(chunk > 0);
    let mut result = PartialFrames::default();
    let mut options = JxlDecoderOptions::default();
    options.limits.max_memory_bytes = None;
    options.limits.max_codestream_level = crate::api::JxlCodestreamLevel::Level10;
    options.parallel = parallel;
    let mut remaining = data;
    let mut fed = &data[..0];
    macro_rules! feed {
        () => {
            fed = &remaining[..fed.len().saturating_add(chunk).min(remaining.len())];
        };
    }
    macro_rules! process {
        ($decoder:ident $(, $buffers:expr)?) => {{
            feed!();
            let before = fed.len();
            let event = $decoder.process(&mut fed $(, $buffers)?)?;
            remaining = &remaining[before - fed.len()..];
            event
        }};
    }
    // Even if the decoder retains an unconsumed suffix, no more input can be
    // supplied once that suffix is the entire remaining prefix.
    macro_rules! exhausted {
        () => {
            fed.len() == remaining.len()
        };
    }

    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut info = loop {
        match process!(decoder) {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if exhausted!() {
                    return Ok(result);
                }
                decoder = fallback;
            }
        }
    };
    let (width, height) = info.basic_info().size;
    let format = info.current_pixel_format().clone();
    info.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: format
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    });
    let format = info.current_pixel_format().clone();
    loop {
        let mut buffers = vec![Image::new_with_value(
            (width * format.color_type.samples_per_pixel(), height),
            f32::NAN,
        )?];
        for _ in format.extra_channel_format.iter().flatten() {
            buffers.push(Image::new_with_value((width, height), f32::NAN)?);
        }
        let mut frame = loop {
            match process!(info) {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                    let changed = fallback.flush_pixels(&mut output_buffers(&mut buffers))?;
                    if changed {
                        on_flush(data.len() - remaining.len(), result.frames.len(), &buffers);
                    }
                    if exhausted!() {
                        result.frames.push(buffers);
                        return Ok(result);
                    }
                    info = fallback;
                }
            }
        };
        info = loop {
            match process!(frame, &mut output_buffers(&mut buffers)) {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                    let changed = fallback.flush_pixels(&mut output_buffers(&mut buffers))?;
                    result.flushes += 1;
                    if changed {
                        on_flush(data.len() - remaining.len(), result.frames.len(), &buffers);
                    }
                    if exhausted!() {
                        result.frames.push(buffers);
                        return Ok(result);
                    }
                    frame = fallback;
                }
            }
        };
        result.frames.push(buffers);
        if !info.has_more_frames() {
            return Ok(result);
        }
    }
}

fn has_pixels(frames: &[Vec<Image<f32>>]) -> bool {
    frames
        .iter()
        .flatten()
        .any(|image| (0..image.size().1).any(|y| image.row(y).iter().any(|value| !value.is_nan())))
}

fn compare_prefix_frames(
    context: &str,
    reference: &[Vec<Image<f32>>],
    candidate: &[Vec<Image<f32>>],
) {
    if reference.is_empty() || candidate.is_empty() {
        assert!(
            !has_pixels(reference) && !has_pixels(candidate),
            "only one decode recovered pixels"
        );
        return;
    }
    assert_eq!(reference.len(), candidate.len());
    for (left, right) in reference.iter().zip(candidate) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.size(), right.size());
            for y in 0..left.size().1 {
                for (&a, &b) in left.row(y).iter().zip(right.row(y)) {
                    if a.is_nan() && b.is_nan() {
                        continue;
                    }
                    assert!(
                        !a.is_nan() && !b.is_nan() && (a == b || (a - b).abs() <= 5e-3),
                        "{context}: upstream prefix bound: {a} vs {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn upstream_partial_prefix_sweep() {
    let fixtures = crate::util::test::all_jxl_fixtures();
    assert!(!fixtures.is_empty());
    for path in fixtures {
        let data = std::fs::read(&path).unwrap();
        assert!(!data.is_empty(), "{}", path.display());
        arbtest::arbtest(|u| {
            let prefix = u.int_in_range(1..=data.len())?;
            let chunk = u.int_in_range(1..=4096)?;
            let context = format!("{} prefix={prefix} chunk={chunk}", path.display());
            let data = &data[..prefix];
            let reference =
                decode_partial(data, usize::MAX, false, |_, _, _| {}).unwrap_or_default();
            let candidate = decode_partial(data, chunk, false, |_, _, _| {}).unwrap_or_default();
            compare_prefix_frames(&context, &reference.frames, &candidate.frames);
            #[cfg(feature = "threads")]
            {
                let candidate = decode_partial(data, chunk, true, |_, _, _| {}).unwrap_or_default();
                compare_prefix_frames(&context, &reference.frames, &candidate.frames);
            }
            Ok(())
        });
    }
}

#[test]
fn upstream_bicycle_prefix_regression() {
    let data = crate::util::test::fixture_bytes("bicycles_web_q85.jxl");
    let data = &data[..29_726];
    let reference = decode_partial(data, usize::MAX, false, |_, _, _| {}).unwrap();
    let candidate = decode_partial(data, 2675, false, |_, _, _| {}).unwrap();
    compare_prefix_frames("sequential", &reference.frames, &candidate.frames);
    #[cfg(feature = "threads")]
    {
        let candidate = decode_partial(data, 2675, true, |_, _, _| {}).unwrap();
        compare_prefix_frames("parallel", &reference.frames, &candidate.frames);
    }
}

#[cfg(feature = "threads")]
#[test]
fn upstream_parallel_progressive_flushes() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let mut total_flushes = 0;
        for name in [
            "basic.jxl",
            "progressive_ac.jxl",
            "squeeze_empty_residual.jxl",
            "multiple_layers_noise_spline.jxl",
        ] {
            let data = crate::util::test::fixture_bytes(name);
            let chunk = (data.len() / 8).max(1024);
            let capture = |bytes, frame, buffers: &[Image<f32>]| {
                let samples = buffers
                    .iter()
                    .map(|image| {
                        (
                            image.size(),
                            (0..image.size().1)
                                .flat_map(|y| image.row(y).iter())
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                (bytes, frame, samples)
            };
            let mut sequential = Vec::new();
            let complete = decode_partial(&data, chunk, false, |bytes, frame, buffers| {
                sequential.push(capture(bytes, frame, buffers));
            })
            .unwrap();
            assert!(has_pixels(&complete.frames), "{name}: no final pixels");
            let mut count = 0;
            let parallel = decode_partial(&data, chunk, true, |bytes, frame, buffers| {
                let actual = capture(bytes, frame, buffers);
                let expected = sequential.get(count).expect("extra parallel flush");
                assert_eq!(
                    (expected.0, expected.1),
                    (bytes, frame),
                    "{name}: flush {count} timing"
                );
                assert!(
                    expected.2 == actual.2,
                    "{name}: flush {count} pixels at byte {bytes}"
                );
                count += 1;
            })
            .unwrap();
            assert_eq!(count, sequential.len(), "{name}: missing parallel flushes");
            total_flushes += count;
            assert_eq!(complete.frames.len(), parallel.frames.len());
            for (index, (a, b)) in complete.frames.iter().zip(&parallel.frames).enumerate() {
                crate::api::decoder::tests::compare_frames(std::path::Path::new(name), index, a, b)
                    .unwrap();
            }
        }
        assert!(
            total_flushes > 0,
            "no intermediate rendering in the fixture set"
        );
    });
}
