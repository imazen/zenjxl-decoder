// Copyright (c) Imazen LLC.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Exercise callback contracts through the real decoder, including both renderers.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    api::{
        JxlCms, JxlCmsTransformer, JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDataFormat,
        JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
    },
    error::{Error, Result},
};

const REFUSAL: &str = "application CMS refused this transform";
const CMYK: &str = "conformance_test_images/cmyk_layers.jxl";

#[derive(Clone, Copy, Debug)]
enum Behavior {
    Refuse,
    RefuseAfterOne,
    RefuseInit,
    Copy,
    Empty,
    Extra,
    Channels(usize),
}

#[derive(Default)]
struct Observed {
    calls: AtomicUsize,
    error: Mutex<Option<String>>,
    destinations: Mutex<Vec<JxlColorProfile>>,
}

struct TestCms {
    behavior: Behavior,
    observed: Arc<Observed>,
}
struct TestTransform {
    behavior: Behavior,
    observed: Arc<Observed>,
    input_channels: usize,
    max_pixels: usize,
}

impl JxlCms for TestCms {
    fn initialize_transforms(
        &self,
        n: usize,
        max_pixels: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send + Sync>>)> {
        self.observed
            .destinations
            .lock()
            .unwrap()
            .push(output.clone());
        if matches!(self.behavior, Behavior::RefuseInit) {
            let error = whereat::at!(Error::CmsError(REFUSAL.into()));
            *self.observed.error.lock().unwrap() = Some(format!("{error:?}"));
            return Err(error);
        }
        let count = match self.behavior {
            Behavior::Empty => 0,
            Behavior::Extra => n + 1,
            _ => n,
        };
        let channels = match self.behavior {
            Behavior::Channels(n) => n,
            _ => output.channels(),
        };
        Ok((
            channels,
            (0..count)
                .map(|_| {
                    Box::new(TestTransform {
                        behavior: self.behavior,
                        observed: Arc::clone(&self.observed),
                        input_channels: input.channels(),
                        max_pixels,
                    }) as Box<dyn JxlCmsTransformer + Send + Sync>
                })
                .collect(),
        ))
    }
}

impl JxlCmsTransformer for TestTransform {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        assert!(
            input.len() / self.input_channels <= self.max_pixels,
            "decoder exceeded the admitted callback chunk size"
        );
        let prior = self.observed.calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, Behavior::Refuse)
            || matches!(self.behavior, Behavior::RefuseAfterOne) && prior > 0
        {
            let error = whereat::at!(Error::CmsError(REFUSAL.into()));
            *self.observed.error.lock().unwrap() = Some(format!("{error:?}"));
            return Err(error);
        }
        let pixels = input.len() / self.input_channels;
        let output_channels = output.len() / pixels;
        for (src, dst) in input
            .chunks_exact(self.input_channels)
            .zip(output.chunks_exact_mut(output_channels))
        {
            dst.copy_from_slice(&src[..output_channels]);
        }
        Ok(())
    }
    fn do_transform_inplace(&mut self, data: &mut [f32]) -> Result<()> {
        let copy = data.to_vec();
        self.do_transform(&copy, data)
    }
}

fn complete<S, T>(
    mut state: S,
    input: &mut &[u8],
    mut step: impl FnMut(S, &mut &[u8]) -> Result<ProcessingResult<T, S>>,
) -> Result<T> {
    for _ in 0..256 {
        let before = input.len();
        match step(state, input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(input.len() < before, "complete fixture stalled");
                state = fallback;
            }
        }
    }
    panic!("bounded test driver exhausted");
}

fn first_image(
    name: &str,
    simple: bool,
    color_type: JxlColorType,
    cms: Option<TestCms>,
    target: Option<JxlColorProfile>,
) -> Result<Vec<f32>> {
    first_image_with_mode(name, simple, false, color_type, cms, target)
}

fn first_image_with_mode(
    name: &str,
    simple: bool,
    parallel: bool,
    color_type: JxlColorType,
    cms: Option<TestCms>,
    target: Option<JxlColorProfile>,
) -> Result<Vec<f32>> {
    let bytes = crate::util::test::fixture_bytes(name);
    let mut input = bytes.as_slice();
    let mut options = JxlDecoderOptions::default()
        .with_adjust_orientation(false)
        .with_render_spot_colors(false)
        .with_parallel(parallel);
    if let Some(cms) = cms {
        options = options.with_cms(Box::new(cms));
    }
    let mut decoder = complete(JxlDecoder::new(options), &mut input, |d, i| d.process(i))?;
    decoder.set_use_simple_pipeline(simple);
    decoder.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; decoder.basic_info().extra_channels.len()],
    });
    if let Some(target) = target {
        decoder.set_output_color_profile(target)?;
    }
    let frame = complete(decoder, &mut input, |d, i| d.process(i))?;
    let (width, height) = frame.frame_header().size;
    let active = width * color_type.samples_per_pixel();
    let stride = active + 7;
    let sentinel = f32::from_bits(0x7fc00001);
    let mut data = vec![sentinel; stride * height];
    let mut buffers = [JxlOutputBuffer::new_with_stride(
        bytemuck::cast_slice_mut(&mut data),
        height,
        active * 4,
        stride * 4,
    )];
    complete(frame, &mut input, |d, i| d.process(i, &mut buffers))?;
    drop(buffers);
    assert!(data.chunks_exact(stride).all(|r| {
        r[active..]
            .iter()
            .all(|v| v.to_bits() == sentinel.to_bits())
    }));
    Ok(data
        .chunks_exact(stride)
        .flat_map(|r| r[..active].iter().copied())
        .collect())
}

fn assert_callback_refusal(name: &str, color: JxlColorType) {
    for simple in [true, false] {
        let observed = Arc::<Observed>::default();
        let cms = TestCms {
            behavior: Behavior::Refuse,
            observed: Arc::clone(&observed),
        };
        let target = JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(
            color == JxlColorType::Grayscale,
        ));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            first_image(name, simple, color, Some(cms), Some(target))
        }));
        assert!(
            observed.calls.load(Ordering::Relaxed) > 0,
            "CMS never called: {name}, simple={simple}"
        );
        let result = result.expect("callback Err must not unwind through the decoder");
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("callback refusal must not substitute successful pixels"),
        };
        assert!(
            matches!(error.error(), Error::CmsError(message) if message == REFUSAL),
            "{error:?}"
        );
        assert_eq!(
            Some(format!("{error:?}")),
            *observed.error.lock().unwrap(),
            "original whereat context changed"
        );
    }
}

#[test]
fn cms_contract_rgb_refusal_is_returned() {
    assert_callback_refusal("3x3_srgb_lossless.jxl", JxlColorType::Rgb);
}

#[test]
fn cms_contract_gray_refusal_is_returned() {
    assert_callback_refusal("gray_alpha_lossless.jxl", JxlColorType::Grayscale);
}

#[test]
fn cms_contract_cmyk_refusal_is_returned() {
    assert_callback_refusal(CMYK, JxlColorType::Rgb);
}

#[test]
fn cms_contract_cmyk_receives_native_polarity_and_requested_destination() {
    for simple in [true, false] {
        let native = first_image(CMYK, simple, JxlColorType::Cmyk, None, None).unwrap();
        let observed = Arc::<Observed>::default();
        let cms = TestCms {
            behavior: Behavior::Copy,
            observed: Arc::clone(&observed),
        };
        let target = JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(false));
        let output = first_image(
            CMYK,
            simple,
            JxlColorType::Rgb,
            Some(cms),
            Some(target.clone()),
        )
        .unwrap();
        assert!(
            observed.calls.load(Ordering::Relaxed) > 0,
            "CMS never called"
        );
        assert_eq!(output.len() / 3, native.len() / 4);
        for (pixel, (got, raw)) in output
            .chunks_exact(3)
            .zip(native.chunks_exact(4))
            .enumerate()
        {
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                raw[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "callback polarity differs at pixel {pixel}, simple={simple}"
            );
        }
        assert!(
            observed
                .destinations
                .lock()
                .unwrap()
                .iter()
                .all(|p| *p == target),
            "CMS received a different output profile"
        );
    }
}

#[test]
fn cms_contract_invalid_initialization_refuses_before_transform() {
    for name in ["3x3_srgb_lossless.jxl", CMYK] {
        for simple in [true, false] {
            for behavior in [
                Behavior::Empty,
                Behavior::Extra,
                Behavior::Channels(0),
                Behavior::Channels(1),
                Behavior::Channels(4),
            ] {
                let observed = Arc::<Observed>::default();
                let cms = TestCms {
                    behavior,
                    observed: Arc::clone(&observed),
                };
                let target = JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(false));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    first_image(name, simple, JxlColorType::Rgb, Some(cms), Some(target))
                }));
                assert!(
                    result.is_ok(),
                    "invalid CMS initialization panicked: {name}, {behavior:?}"
                );
                assert!(
                    result.unwrap().is_err(),
                    "invalid CMS initialization succeeded: {name}, {behavior:?}"
                );
                assert_eq!(
                    observed.calls.load(Ordering::Relaxed),
                    0,
                    "invalid setup reached pixel transform"
                );
            }
        }
    }
}

#[test]
fn cms_contract_late_refusal_ends_decode() {
    for simple in [true, false] {
        for name in ["3x3_srgb_lossless.jxl", "gray_alpha_lossless.jxl", CMYK] {
            let color = if name.starts_with("gray") {
                JxlColorType::Grayscale
            } else {
                JxlColorType::Rgb
            };
            let observed = Arc::<Observed>::default();
            let cms = TestCms {
                behavior: Behavior::RefuseAfterOne,
                observed: Arc::clone(&observed),
            };
            let target = JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(
                color == JxlColorType::Grayscale,
            ));
            let result = first_image(name, simple, color, Some(cms), Some(target));
            assert!(
                observed.calls.load(Ordering::Relaxed) >= 2,
                "no successful chunk before refusal"
            );
            let Err(error) = result else {
                panic!("late refusal returned a completed image")
            };
            assert_eq!(Some(format!("{error:?}")), *observed.error.lock().unwrap());
        }
    }
}

#[cfg(feature = "threads")]
#[test]
fn cms_contract_parallel_refusal_preserves_original_error() {
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let observed = Arc::<Observed>::default();
            let cms = TestCms {
                behavior: Behavior::RefuseAfterOne,
                observed: Arc::clone(&observed),
            };
            let target = JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(false));
            let result = first_image_with_mode(
                CMYK,
                false,
                true,
                JxlColorType::Rgb,
                Some(cms),
                Some(target),
            );
            assert!(observed.calls.load(Ordering::Relaxed) >= 2);
            let Err(error) = result else {
                panic!("parallel refusal returned a completed image")
            };
            assert_eq!(Some(format!("{error:?}")), *observed.error.lock().unwrap());
        });
    }
}

#[test]
fn cms_contract_initialization_error_keeps_its_origin() {
    for name in ["3x3_srgb_lossless.jxl", CMYK] {
        for simple in [true, false] {
            let observed = Arc::<Observed>::default();
            let cms = TestCms {
                behavior: Behavior::RefuseInit,
                observed: Arc::clone(&observed),
            };
            let target = JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(false));
            let result = first_image(name, simple, JxlColorType::Rgb, Some(cms), Some(target));
            let Err(error) = result else {
                panic!("failed initialization returned pixels")
            };
            assert_eq!(observed.calls.load(Ordering::Relaxed), 0);
            assert_eq!(Some(format!("{error:?}")), *observed.error.lock().unwrap());
        }
    }
}
