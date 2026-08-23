// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Concurrency correctness tests.
//!
//! These tests decode images both single-threaded and multi-threaded,
//! then compare pixel output for exact equality. Any difference indicates
//! a data race or thread-safety bug in the parallel decode path.

#[cfg(all(test, feature = "threads"))]
mod tests {
    use crate::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use crate::image::{Image, Rect};

    /// Decode a JXL file to u8 pixels with the given parallel setting.
    fn decode_with_parallel(
        data: &[u8],
        parallel: bool,
    ) -> Result<(usize, usize, usize, Vec<u8>), String> {
        decode_with_parallel_chunked(data, parallel, usize::MAX)
    }

    /// Like [`decode_with_parallel`], but hands the decoder at most
    /// `chunk` bytes per `process` call, so a parallel decode goes through
    /// the incremental path (groups arrive in several batches; cross-batch
    /// borders and the final re-render are exercised).
    fn decode_with_parallel_chunked(
        data: &[u8],
        parallel: bool,
        chunk: usize,
    ) -> Result<(usize, usize, usize, Vec<u8>), String> {
        // `input` is the not-yet-consumed remainder; `window` is what the
        // decoder may see in the current call.
        let mut input = data;
        macro_rules! window {
            () => {
                &input[..chunk.min(input.len())]
            };
        }
        macro_rules! consumed {
            ($w:expr) => {{
                let left = $w.len();
                let before = chunk.min(input.len());
                input = &input[before - left..];
            }};
        }

        #[cfg(feature = "cms")]
        let options = JxlDecoderOptions {
            cms: Some(Box::new(crate::api::MoxCms::new())),
            parallel,
            ..JxlDecoderOptions::default()
        };
        #[cfg(not(feature = "cms"))]
        let options = JxlDecoderOptions {
            parallel,
            ..JxlDecoderOptions::default()
        };

        let mut decoder = JxlDecoder::<states::Initialized>::new(options);

        // Advance to image info
        let mut decoder = loop {
            let mut w = window!();
            let r = decoder.process(&mut w);
            consumed!(w);
            match r {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        return Err("Unexpected end of input during header".into());
                    }
                    decoder = fallback;
                }
                Err(e) => return Err(format!("Header decode error: {:?}", e)),
            }
        };

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;

        let default_format = decoder.current_pixel_format();
        let is_grayscale = matches!(
            default_format.color_type,
            JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha
        );

        let has_alpha = basic_info.extra_channels.iter().any(|ec| {
            matches!(
                ec.ec_type,
                crate::headers::extra_channels::ExtraChannel::Alpha
            )
        });

        let (color_type, channels) = match (is_grayscale, has_alpha) {
            (true, true) => (JxlColorType::GrayscaleAlpha, 2),
            (true, false) => (JxlColorType::Grayscale, 1),
            (false, true) => (JxlColorType::Rgba, 4),
            (false, false) => (JxlColorType::Rgb, 3),
        };

        let num_extra_channels = basic_info.extra_channels.len();
        let extra_channel_format = vec![None; num_extra_channels];
        let pixel_format = JxlPixelFormat {
            color_type,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format,
        };
        decoder.set_pixel_format(pixel_format);

        // Advance to frame info
        let mut decoder = loop {
            let mut w = window!();
            let r = decoder.process(&mut w);
            consumed!(w);
            match r {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        return Err("Unexpected end of input before frame".into());
                    }
                    decoder = fallback;
                }
                Err(e) => return Err(format!("Frame info decode error: {:?}", e)),
            }
        };

        // Create output buffer
        let mut output_image = Image::<u8>::new((width * channels, height))
            .map_err(|e| format!("Buffer error: {:?}", e))?;

        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * channels, height),
                })
                .into_raw(),
        )];

        // Decode frame pixels
        loop {
            let mut w = window!();
            let r = decoder.process(&mut w, &mut buffers);
            consumed!(w);
            match r {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        return Err("Unexpected end of input during frame".into());
                    }
                    decoder = fallback;
                }
                Err(e) => return Err(format!("Frame decode error: {:?}", e)),
            }
        }

        let mut pixels = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            pixels.extend_from_slice(output_image.row(y));
        }

        Ok((width, height, channels, pixels))
    }

    /// Compare serial and parallel decode outputs for a test image.
    /// Asserts pixel-exact equality.
    fn assert_serial_parallel_parity(name: &str, data: &[u8]) {
        let (w1, h1, c1, serial) =
            decode_with_parallel(data, false).unwrap_or_else(|e| panic!("{name} serial: {e}"));
        let (w2, h2, c2, parallel) =
            decode_with_parallel(data, true).unwrap_or_else(|e| panic!("{name} parallel: {e}"));

        assert_eq!((w1, h1, c1), (w2, h2, c2), "{name}: dimension mismatch");
        assert_eq!(
            serial.len(),
            parallel.len(),
            "{name}: buffer length mismatch"
        );

        if serial != parallel {
            // Find first differing pixel for diagnostics
            let mut first_diff = None;
            for (i, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
                if s != p {
                    let pixel_idx = i / c1;
                    let channel = i % c1;
                    let x = pixel_idx % w1;
                    let y = pixel_idx / w1;
                    first_diff = Some((x, y, channel, *s, *p));
                    break;
                }
            }

            let diff_count = serial
                .iter()
                .zip(parallel.iter())
                .filter(|(s, p)| s != p)
                .count();

            let (x, y, ch, sv, pv) = first_diff.unwrap();
            panic!(
                "{name}: serial/parallel mismatch! {diff_count} differing values. \
                 First at ({x},{y}) ch={ch}: serial={sv}, parallel={pv}"
            );
        }
    }

    fn test_image(name: &str) -> Vec<u8> {
        let path = format!("{}/resources/test/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"))
    }

    // -- Single-group images (baseline: parallel path should be no-op) --

    #[test]
    fn serial_parallel_parity_basic() {
        assert_serial_parallel_parity("basic", &test_image("basic.jxl"));
    }

    #[test]
    fn serial_parallel_parity_3x3_lossless() {
        assert_serial_parallel_parity("3x3_lossless", &test_image("3x3_srgb_lossless.jxl"));
    }

    #[test]
    fn serial_parallel_parity_3x3_lossy() {
        assert_serial_parallel_parity("3x3_lossy", &test_image("3x3_srgb_lossy.jxl"));
    }

    // -- Multi-group images (these actually exercise the parallel path) --

    #[test]
    fn serial_parallel_parity_bike_q75() {
        assert_serial_parallel_parity("bike_q75", &test_image("bike_web_q75.jxl"));
    }

    #[test]
    fn serial_parallel_parity_bike_q85() {
        assert_serial_parallel_parity("bike_q85", &test_image("bike_web_q85.jxl"));
    }

    #[test]
    fn serial_parallel_parity_cafe() {
        assert_serial_parallel_parity("cafe", &test_image("cafe_web_q80.jxl"));
    }

    #[test]
    fn serial_parallel_parity_bicycles() {
        assert_serial_parallel_parity("bicycles", &test_image("bicycles_web_q85.jxl"));
    }

    // -- Images with noise/splines (complex rendering) --

    #[test]
    fn serial_parallel_parity_noise() {
        assert_serial_parallel_parity("noise", &test_image("8x8_noise.jxl"));
    }

    #[test]
    fn serial_parallel_parity_noise_spline() {
        assert_serial_parallel_parity(
            "noise_spline",
            &test_image("multiple_layers_noise_spline.jxl"),
        );
    }

    // -- Modular images --

    #[test]
    fn serial_parallel_parity_grayscale_modular() {
        assert_serial_parallel_parity(
            "grayscale_modular",
            &test_image("grayscale_patches_modular.jxl"),
        );
    }

    #[test]
    fn serial_parallel_parity_grayscale_var_dct() {
        assert_serial_parallel_parity(
            "grayscale_var_dct",
            &test_image("grayscale_patches_var_dct.jxl"),
        );
    }

    // -- Multi-frame / layered images --

    #[test]
    fn serial_parallel_parity_spline_first_frame() {
        assert_serial_parallel_parity(
            "spline_first_frame",
            &test_image("spline_on_first_frame.jxl"),
        );
    }

    // -- Large multi-group images (4K) --

    #[test]
    fn serial_parallel_parity_city_4k() {
        assert_serial_parallel_parity("city_4k", &test_image("city_4k_q75.jxl"));
    }

    #[test]
    fn serial_parallel_parity_forest_4k() {
        assert_serial_parallel_parity("forest_4k", &test_image("forest_4k_q90.jxl"));
    }

    // -- Images with extra channels --

    #[test]
    fn serial_parallel_parity_extra_channels() {
        assert_serial_parallel_parity("extra_channels", &test_image("extra_channels.jxl"));
    }

    #[test]
    fn serial_parallel_parity_3x3a_lossless() {
        assert_serial_parallel_parity("3x3a_lossless", &test_image("3x3a_srgb_lossless.jxl"));
    }

    // -- Progressive images --

    #[test]
    fn serial_parallel_parity_progressive() {
        assert_serial_parallel_parity("progressive", &test_image("progressive_ac.jxl"));
    }

    // -- Orientation images --

    #[test]
    fn serial_parallel_parity_orientation_rotate() {
        assert_serial_parallel_parity(
            "orientation_rotate",
            &test_image("orientation6_rotate_90_cw.jxl"),
        );
    }

    // -- With ICC profile --

    #[test]
    fn serial_parallel_parity_with_icc() {
        assert_serial_parallel_parity("with_icc", &test_image("with_icc.jxl"));
    }

    // -- Stress test: decode same image many times in parallel threads --

    #[test]
    fn parallel_determinism_repeated() {
        let data = test_image("bike_web_q85.jxl");

        // Decode once as reference (serial)
        let (w, h, c, reference) = decode_with_parallel(&data, false).unwrap();

        // Decode 8 times in parallel
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let data = data.clone();
                std::thread::spawn(move || decode_with_parallel(&data, true))
            })
            .collect();

        for (i, handle) in handles.into_iter().enumerate() {
            let (w2, h2, c2, pixels) = handle
                .join()
                .unwrap()
                .unwrap_or_else(|e| panic!("Thread {i}: {e}"));
            assert_eq!((w, h, c), (w2, h2, c2), "Thread {i}: dimension mismatch");
            assert_eq!(
                reference, pixels,
                "Thread {i}: pixel mismatch vs serial reference"
            );
        }
    }
    /// Fixtures with the features whose parallel rendering has cross-group
    /// state: EPF/gaborish borders, chroma subsampling with several LF
    /// groups, noise, patches, splines, upsampling, blending / extra
    /// channels, orientation, spot colours, and the modular transforms.
    /// The 4K fixtures are left to the default-pool parity tests above.
    fn thread_sweep_corpus() -> Vec<(&'static str, Vec<u8>)> {
        [
            "multiple_lf_420.jxl",
            "test_600x600_422_libjxl.jxl",
            "bike_web_q85.jxl",
            "cafe_web_q80.jxl",
            "green_queen_vardct_e3.jxl",
            "green_queen_modular_e3.jxl",
            "issue772_blendbug.jxl",
            "oddsize_ups.jxl",
            "squeeze_edge.jxl",
            "squeeze_alpha.jxl",
            "upsampled_alpha.jxl",
            "splines.jxl",
            "8x8_noise.jxl",
            "multiple_layers_noise_spline.jxl",
            "grayscale_patches_var_dct.jxl",
            "grayscale_patches_modular.jxl",
            "conformance_test_images/patches.jxl",
            "conformance_test_images/upsampling.jxl",
            "conformance_test_images/noise.jxl",
            "conformance_test_images/spot.jxl",
            "conformance_test_images/alpha_triangles.jxl",
            "conformance_test_images/cmyk_layers.jxl",
            "conformance_test_images/bench_oriented_brg.jxl",
            "conformance_test_images/grayscale_public_university.jxl",
            "conformance_test_images/lossless_pfm.jxl",
        ]
        .into_iter()
        .map(|name| (name, test_image(name)))
        .collect()
    }

    /// Parallel decode must not depend on the rayon pool size: each size
    /// changes how groups are split into leaves (and so which thread renders
    /// which group, how many render contexts exist, ...).
    #[test]
    fn serial_parallel_parity_across_thread_counts_corpus() {
        let pools: Vec<_> = [1usize, 2, 3, 5, 8]
            .into_iter()
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap()
            })
            .collect();
        for (name, data) in &thread_sweep_corpus() {
            let (w, h, c, serial) =
                decode_with_parallel(data, false).unwrap_or_else(|e| panic!("{name} serial: {e}"));
            for pool in &pools {
                let (w2, h2, c2, parallel) = pool
                    .install(|| decode_with_parallel(data, true))
                    .unwrap_or_else(|e| panic!("{name} parallel: {e}"));
                assert_eq!((w, h, c), (w2, h2, c2), "{name}: dimension mismatch");
                let diff = serial.iter().zip(&parallel).filter(|(a, b)| a != b).count();
                assert_eq!(
                    diff,
                    0,
                    "{name}, {}-thread pool: {diff} of {} samples differ from the serial decode",
                    pool.current_num_threads(),
                    serial.len()
                );
            }
        }
    }

    /// Parallel decode with the input arriving in pieces takes the
    /// incremental path of `decode_groups_parallel`: groups are decoded in
    /// several batches, borders at batch boundaries are only partially
    /// ready, and a final full-readiness re-render corrects them. Every
    /// chunk size must give the one-shot serial pixels.
    #[test]
    fn chunked_parallel_decode_matches_serial() {
        for (name, data) in &thread_sweep_corpus() {
            let (w, h, c, serial) =
                decode_with_parallel(data, false).unwrap_or_else(|e| panic!("{name} serial: {e}"));
            for chunk in [777usize, 4096, 30_000] {
                if chunk >= data.len() {
                    continue;
                }
                let (w2, h2, c2, parallel) = decode_with_parallel_chunked(data, true, chunk)
                    .unwrap_or_else(|e| panic!("{name} parallel chunk {chunk}: {e}"));
                assert_eq!((w, h, c), (w2, h2, c2), "{name}: dimension mismatch");
                let diff = serial.iter().zip(&parallel).filter(|(a, b)| a != b).count();
                assert_eq!(
                    diff,
                    0,
                    "{name}, chunk {chunk}: {diff} of {} samples differ from the one-shot \
                     serial decode",
                    serial.len()
                );
            }
        }
    }

    /// Modular palette transforms are applied after all groups are decoded,
    /// one channel at a time across a whole group row (`AverageAll` /
    /// `Weighted` predictors) or group (the other predictors). The per-channel
    /// work is independent and runs on the rayon pool when `parallel` is set,
    /// so the result must not depend on the pool size.
    ///
    /// - `delta_palette.jxl`: implicit delta palette, `AverageAll` predictor
    ///   (full-row path), 555x751 = 3x3 groups;
    /// - the same bitstream with the predictor patched to `Weighted` (the
    ///   weighted-predictor full-row path with state carried across group
    ///   rows, see `modular_palette.rs`);
    /// - `issue648_palette0.jxl`, `sunset_logo.jxl`, `lz77_flower.jxl`:
    ///   regular (non-delta) palettes.
    #[test]
    fn serial_parallel_parity_palette_across_thread_counts() {
        let cases = [
            (
                "delta_palette",
                test_image("conformance_test_images/delta_palette.jxl"),
            ),
            (
                "delta_palette+weighted",
                crate::tests::modular_palette::delta_palette_with_weighted_predictor(),
            ),
            ("issue648_palette0", test_image("issue648_palette0.jxl")),
            (
                "sunset_logo",
                test_image("conformance_test_images/sunset_logo.jxl"),
            ),
            (
                "lz77_flower",
                test_image("conformance_test_images/lz77_flower.jxl"),
            ),
        ];
        for (name, data) in &cases {
            let (w, h, c, serial) = decode_with_parallel(data, false).unwrap();
            for threads in [1usize, 2, 3, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let (w2, h2, c2, parallel) =
                    pool.install(|| decode_with_parallel(data, true)).unwrap();
                assert_eq!((w, h, c), (w2, h2, c2), "{name}: dimension mismatch");
                let diff = serial.iter().zip(&parallel).filter(|(a, b)| a != b).count();
                assert_eq!(
                    diff,
                    0,
                    "{name}, {threads}-thread pool: {diff} of {} samples differ from the \
                     serial decode",
                    serial.len()
                );
            }
        }
    }

    /// Parallel decode with a CMS stage must not depend on the thread count.
    ///
    /// `CmsStage` hands one transformer to every render context. Contexts
    /// were created once per rayon *leaf* (`try_for_each_init` /
    /// `map_init`), and transformers were popped from a pool sized
    /// `current_num_threads() + 2` and never returned; with more leaves than
    /// that, the extra contexts had no transformer and `process_row_chunk`
    /// silently passed pixels through untransformed. A 4096x2519 ACEScg-tagged
    /// photo decoded with 2 threads had 157 of 160 groups wrong.
    ///
    /// The fixture is 520x280 (3x2 groups) with an ICC-only (ACEScg) profile,
    /// so the pipeline contains a real ICC -> sRGB CMS stage. Each pool size
    /// gets its own rayon pool so the leaf count is reproducible.
    #[cfg(feature = "cms")]
    #[test]
    fn serial_parallel_parity_icc_cms_across_thread_counts() {
        let path = format!(
            "{}/tests/testdata/parallel-cms/icc_aces_520x280_d5.jxl",
            env!("CARGO_MANIFEST_DIR")
        );
        let data = std::fs::read(&path).unwrap();
        let (w, h, c, serial) = decode_with_parallel(&data, false).unwrap();
        assert_eq!((w, h), (520, 280));
        for threads in [1usize, 2, 3, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (w2, h2, c2, parallel) =
                pool.install(|| decode_with_parallel(&data, true)).unwrap();
            assert_eq!((w, h, c), (w2, h2, c2));
            let diff = serial.iter().zip(&parallel).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff,
                0,
                "{threads}-thread pool: {diff} of {} samples differ from the serial decode \
                 (CMS transform skipped on some tiles)",
                serial.len()
            );
        }
    }
}
