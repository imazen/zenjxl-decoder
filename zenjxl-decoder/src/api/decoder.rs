// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};
#[cfg(test)]
use crate::frame::Frame;
use crate::{
    api::{JxlFrameHeader, VardctQuantizer},
    container::{frame_index::FrameIndexBox, gain_map::GainMapBundle},
    error::Result,
};
use states::*;
use std::marker::PhantomData;
#[cfg(test)]
use whereat::at;

pub mod states {
    pub trait JxlState {}
    pub struct Initialized;
    pub struct WithImageInfo;
    pub struct WithFrameInfo;
    impl JxlState for Initialized {}
    impl JxlState for WithImageInfo {}
    impl JxlState for WithFrameInfo {}
}

// Q: do we plan to add support for box decoding?
// If we do, one way is to take a callback &[u8; 4] -> Box<dyn Write>.

/// High level API using the typestate pattern to forbid invalid usage.
pub struct JxlDecoder<State: JxlState> {
    inner: Box<JxlDecoderInner>,
    _state: PhantomData<State>,
}

#[cfg(test)]
pub type FrameCallback = dyn FnMut(&Frame, usize) -> Result<()>;

impl<S: JxlState> JxlDecoder<S> {
    fn wrap_inner(inner: Box<JxlDecoderInner>) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Sets a callback that processes all frames by calling `callback(frame, frame_index)`.
    #[cfg(test)]
    pub fn set_frame_callback(&mut self, callback: Box<FrameCallback>) {
        self.inner.set_frame_callback(callback);
    }

    #[cfg(test)]
    pub fn decoded_frames(&self) -> usize {
        self.inner.decoded_frames()
    }

    /// Returns the reconstructed JPEG bytes if the file contained a JBRD box.
    /// Call after decoding a frame. Returns `None` if no JBRD box was present
    /// or the `jpeg` feature is not enabled.
    #[cfg(feature = "jpeg")]
    pub fn take_jpeg_reconstruction(&mut self) -> Option<Vec<u8>> {
        self.inner.take_jpeg_reconstruction()
    }

    /// Returns the parsed frame index box, if the file contained one.
    ///
    /// The frame index box (`jxli`) is an optional part of the JXL container
    /// format that provides a seek table for animated files, listing keyframe
    /// byte offsets, timestamps, and frame counts.
    pub fn frame_index(&self) -> Option<&FrameIndexBox> {
        self.inner.frame_index()
    }

    /// Returns the first regular VarDCT frame's quantizer (`global_scale`,
    /// `quant_lf`), if this is a lossy VarDCT image and that frame's `LfGlobal`
    /// section has been decoded.
    ///
    /// VarDCT quality is governed by `global_scale`. Returns `None` for Modular
    /// (lossless) images, or before the first regular frame has been reached —
    /// e.g. immediately after image info. To recover it from a header probe,
    /// advance one frame via `skip_frame`.
    pub fn vardct_quantizer(&self) -> Option<VardctQuantizer> {
        self.inner.vardct_quantizer()
    }

    /// Returns a reference to the parsed gain map bundle, if the file contained
    /// a `jhgm` box (ISO 21496-1 HDR gain map).
    ///
    /// The gain map codestream is a bare JXL codestream that can be decoded
    /// with the same decoder. The ISO 21496-1 metadata blob is stored as raw
    /// bytes for the caller to parse.
    ///
    /// Note: the `jhgm` box may appear after the codestream in the container.
    /// To capture trailing boxes, call `process` once more after the last
    /// frame has been decoded — it drains the remaining container boxes.
    pub fn gain_map(&self) -> Option<&GainMapBundle> {
        self.inner.gain_map()
    }

    /// Takes the parsed gain map bundle, if the file contained a `jhgm` box.
    /// After calling this, `gain_map()` will return `None`.
    pub fn take_gain_map(&mut self) -> Option<GainMapBundle> {
        self.inner.take_gain_map()
    }

    /// Returns the raw EXIF data from the `Exif` container box, if present.
    ///
    /// The 4-byte TIFF header offset prefix is stripped; this returns the raw
    /// EXIF/TIFF bytes starting with the byte-order marker (`II` or `MM`).
    /// Returns `None` for bare codestreams or files without an `Exif` box.
    ///
    /// Note: the `Exif` box may appear after the codestream in the container.
    /// To capture trailing boxes, call `process` once more after the last
    /// frame has been decoded — it drains the remaining container boxes.
    pub fn exif(&self) -> Option<&[u8]> {
        self.inner.exif()
    }

    /// Takes the EXIF data, leaving `None` in its place.
    pub fn take_exif(&mut self) -> Option<Vec<u8>> {
        self.inner.take_exif()
    }

    /// Returns the raw XMP data from the `xml ` container box, if present.
    ///
    /// Returns `None` for bare codestreams or files without an `xml ` box.
    ///
    /// Note: the `xml ` box may appear after the codestream in the container.
    /// To capture trailing boxes, call `process` once more after the last
    /// frame has been decoded — it drains the remaining container boxes.
    pub fn xmp(&self) -> Option<&[u8]> {
        self.inner.xmp()
    }

    /// Takes the XMP data, leaving `None` in its place.
    pub fn take_xmp(&mut self) -> Option<Vec<u8>> {
        self.inner.take_xmp()
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    pub fn rewind(mut self) -> JxlDecoder<Initialized> {
        self.inner.rewind();
        JxlDecoder::wrap_inner(self.inner)
    }

    fn map_inner_processing_result<SuccessState: JxlState>(
        self,
        inner_result: ProcessingResult<(), ()>,
    ) -> ProcessingResult<JxlDecoder<SuccessState>, Self> {
        match inner_result {
            ProcessingResult::Complete { .. } => ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            },
            ProcessingResult::NeedsMoreInput { size_hint, .. } => {
                ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                }
            }
        }
    }
}

impl JxlDecoder<Initialized> {
    pub fn new(options: JxlDecoderOptions) -> Self {
        Self::wrap_inner(Box::new(JxlDecoderInner::new(options)))
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}

impl JxlDecoder<WithImageInfo> {
    // TODO(veluca): once frame skipping is implemented properly, expose that in the API.

    /// Obtains the image's basic information.
    pub fn basic_info(&self) -> &JxlBasicInfo {
        self.inner.basic_info().unwrap()
    }

    /// Retrieves the file's color profile.
    pub fn embedded_color_profile(&self) -> &JxlColorProfile {
        self.inner.embedded_color_profile().unwrap()
    }

    /// Retrieves the current output color profile.
    pub fn output_color_profile(&self) -> &JxlColorProfile {
        self.inner.output_color_profile().unwrap()
    }

    /// Specifies the preferred color profile to be used for outputting data.
    /// Same semantics as JxlDecoderSetOutputColorProfile.
    pub fn set_output_color_profile(&mut self, profile: JxlColorProfile) -> Result<()> {
        self.inner.set_output_color_profile(profile)
    }

    /// Retrieves the current pixel format for output buffers.
    pub fn current_pixel_format(&self) -> &JxlPixelFormat {
        self.inner.current_pixel_format().unwrap()
    }

    /// Specifies pixel format for output buffers.
    ///
    /// Setting this may also change output color profile in some cases, if the profile was not set
    /// manually before.
    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        self.inner.set_pixel_format(pixel_format);
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithFrameInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    /// Draws all the pixels we have data for. This is useful for i.e. previewing LF frames.
    ///
    /// Returns `true` if any new pixels were written to `buffers` since the
    /// previous call to `flush_pixels`; `false` if nothing new was rendered.
    ///
    /// Note: see `process` for alignment requirements for the buffer data.
    pub fn flush_pixels(&mut self, buffers: &mut [JxlOutputBuffer<'_>]) -> Result<bool> {
        self.inner.flush_pixels(buffers)
    }

    pub fn has_more_frames(&self) -> bool {
        self.inner.has_more_frames()
    }

    #[cfg(test)]
    pub(crate) fn set_use_simple_pipeline(&mut self, u: bool) {
        self.inner.set_use_simple_pipeline(u);
    }
}

impl JxlDecoder<WithFrameInfo> {
    /// Skip the current frame.
    pub fn skip_frame(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    pub fn frame_header(&self) -> JxlFrameHeader {
        self.inner.frame_header().unwrap()
    }

    /// Number of passes we have full data for.
    pub fn num_completed_passes(&self) -> usize {
        self.inner.num_completed_passes().unwrap()
    }

    /// Draws all the pixels we have data for.
    ///
    /// Returns `true` if any new pixels were written to `buffers` since the
    /// previous call to `flush_pixels`; `false` if nothing new was rendered.
    ///
    /// Note: see `process` for alignment requirements for the buffer data.
    pub fn flush_pixels(&mut self, buffers: &mut [JxlOutputBuffer<'_>]) -> Result<bool> {
        self.inner.flush_pixels(buffers)
    }

    /// Guarantees to populate exactly the appropriate part of the buffers.
    /// Wants one buffer for each non-ignored pixel type, i.e. color channels and each extra channel.
    ///
    /// Note: the data in `buffers` should have alignment requirements that are compatible with the
    /// requested pixel format. This means that, if we are asking for 2-byte or 4-byte output (i.e.
    /// u16/f16 and f32 respectively), each row in the provided buffers must be aligned to 2 or 4
    /// bytes respectively. If that is not the case, the library may panic.
    pub fn process<In: JxlBitstreamInput>(
        mut self,
        input: &mut In,
        buffers: &mut [JxlOutputBuffer<'_>],
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, Some(buffers))?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::api::{JxlColorType, JxlDataFormat, JxlDecoderOptions};
    use crate::error::Error;
    use crate::image::{Image, Rect};
    use std::path::Path;

    #[test]
    fn decode_small_chunks() {
        arbtest::arbtest(|u| {
            decode(
                &std::fs::read(crate::util::test::fixture_path("green_queen_vardct_e3.jxl"))
                    .unwrap(),
                u.arbitrary::<u8>().unwrap() as usize + 1,
                false,
                false,
                None,
            )
            .unwrap();
            Ok(())
        });
    }

    /// Fully decode a (color-only) fixture and return the live decoder so the
    /// public `vardct_quantizer()` accessor can be exercised post-decode.
    fn quant_of(name: &str) -> Option<VardctQuantizer> {
        let data = crate::util::test::fixture_bytes(name);
        let mut input: &[u8] = &data;
        let mut options = JxlDecoderOptions::default();
        options.limits.max_memory_bytes = None;
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut dwi = match decoder.process(&mut input).unwrap() {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => panic!("need more input for header"),
        };
        let (w, h) = dwi.basic_info().size;
        let cpf = dwi.current_pixel_format().clone();
        assert!(
            cpf.extra_channel_format.iter().all(|e| e.is_none()),
            "fixture must be color-only for this test"
        );
        let fmt = JxlPixelFormat {
            color_type: cpf.color_type,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: cpf.extra_channel_format.iter().map(|_| None).collect(),
        };
        dwi.set_pixel_format(fmt.clone());
        let n = fmt.color_type.samples_per_pixel();
        loop {
            let mut img = Image::new_with_value((w * n, h), 0.0f32).unwrap();
            let mut bufs = vec![JxlOutputBuffer::from_image_rect_mut(
                img.get_rect_mut(Rect {
                    origin: (0, 0),
                    size: img.size(),
                })
                .into_raw(),
            )];
            // WithImageInfo -> WithFrameInfo parses the next frame's header/TOC
            // (no pixel buffers needed); the subsequent WithFrameInfo step
            // decodes the frame body and is where the quantizer gets stashed.
            let dfi = match dwi.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => result,
                ProcessingResult::NeedsMoreInput { .. } => panic!("need more input (frame info)"),
            };
            dwi = match dfi.process(&mut input, &mut bufs).unwrap() {
                ProcessingResult::Complete { result } => result,
                ProcessingResult::NeedsMoreInput { .. } => panic!("need more input (frame data)"),
            };
            if !dwi.has_more_frames() {
                break;
            }
        }
        dwi.vardct_quantizer()
    }

    #[test]
    fn vardct_quantizer_lossy_exposed() {
        let q =
            quant_of("green_queen_vardct_e3.jxl").expect("VarDCT image should expose a quantizer");
        assert!(q.global_scale >= 1, "global_scale must be >= 1");
        assert!(q.quant_lf >= 1, "quant_lf must be >= 1");
        assert!(q.inv_global_scale() > 0.0);
    }

    #[test]
    fn vardct_quantizer_none_for_lossless() {
        assert!(
            quant_of("3x3_srgb_lossless.jxl").is_none(),
            "lossless/modular image must have no VarDCT quantizer"
        );
    }

    #[allow(clippy::type_complexity)]
    pub fn decode(
        mut input: &[u8],
        chunk_size: usize,
        use_simple_pipeline: bool,
        do_flush: bool,
        callback: Option<Box<dyn FnMut(&Frame, usize) -> Result<()>>>,
    ) -> Result<(usize, Vec<Vec<Image<f32>>>)> {
        let mut options = JxlDecoderOptions::default();
        // Correctness tests should not be constrained by memory limits.
        // OOM/limit tests verify those separately.
        options.limits.max_memory_bytes = None;
        let mut initialized_decoder = JxlDecoder::<states::Initialized>::new(options);

        if let Some(callback) = callback {
            initialized_decoder.set_frame_callback(callback);
        }

        let mut chunk_input = &input[0..0];

        macro_rules! advance_decoder {
            ($decoder: ident $(, $extra_arg: expr)? $(; $flush_arg: expr)?) => {
                loop {
                    chunk_input =
                        &input[..(chunk_input.len().saturating_add(chunk_size)).min(input.len())];
                    let available_before = chunk_input.len();
                    let process_result = $decoder.process(&mut chunk_input $(, $extra_arg)?);
                    input = &input[(available_before - chunk_input.len())..];
                    match process_result.unwrap() {
                        ProcessingResult::Complete { result } => break result,
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            $(
                                let mut fallback = fallback;
                                if do_flush && !input.is_empty() {
                                    fallback.flush_pixels($flush_arg)?;
                                }
                            )?
                            if input.is_empty() {
                                panic!("Unexpected end of input");
                            }
                            $decoder = fallback;
                        }
                    }
                }
            };
        }

        // Process until we have image info
        let mut decoder_with_image_info = advance_decoder!(initialized_decoder);
        decoder_with_image_info.set_use_simple_pipeline(use_simple_pipeline);

        // Get basic info
        let basic_info = decoder_with_image_info.basic_info().clone();
        assert!(basic_info.bit_depth.bits_per_sample() > 0);

        // Get image dimensions (after upsampling, which is the actual output size)
        let (buffer_width, buffer_height) = basic_info.size;
        assert!(buffer_width > 0);
        assert!(buffer_height > 0);

        // Explicitly request F32 pixel format (test helper returns Image<f32>)
        let default_format = decoder_with_image_info.current_pixel_format();
        let requested_format = JxlPixelFormat {
            color_type: default_format.color_type,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: default_format
                .extra_channel_format
                .iter()
                .map(|_| Some(JxlDataFormat::f32()))
                .collect(),
        };
        decoder_with_image_info.set_pixel_format(requested_format);

        // Get the configured pixel format
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();

        let num_channels = pixel_format.color_type.samples_per_pixel();
        assert!(num_channels > 0);

        let mut frames = vec![];

        loop {
            // First channel is interleaved.
            let mut buffers = vec![Image::new_with_value(
                (buffer_width * num_channels, buffer_height),
                f32::NAN,
            )?];

            for ecf in pixel_format.extra_channel_format.iter() {
                if ecf.is_none() {
                    continue;
                }
                buffers.push(Image::new_with_value(
                    (buffer_width, buffer_height),
                    f32::NAN,
                )?);
            }

            let mut api_buffers: Vec<_> = buffers
                .iter_mut()
                .map(|b| {
                    JxlOutputBuffer::from_image_rect_mut(
                        b.get_rect_mut(Rect {
                            origin: (0, 0),
                            size: b.size(),
                        })
                        .into_raw(),
                    )
                })
                .collect();

            // Process until we have frame info
            let mut decoder_with_frame_info =
                advance_decoder!(decoder_with_image_info; &mut api_buffers);
            decoder_with_image_info =
                advance_decoder!(decoder_with_frame_info, &mut api_buffers; &mut api_buffers);

            // All pixels should have been overwritten, so they should no longer be NaNs.
            for buf in buffers.iter() {
                let (xs, ys) = buf.size();
                for y in 0..ys {
                    let row = buf.row(y);
                    for (x, v) in row.iter().enumerate() {
                        assert!(!v.is_nan(), "NaN at {x} {y} (image size {xs}x{ys})");
                    }
                }
            }

            frames.push(buffers);

            // Check if there are more frames
            if !decoder_with_image_info.has_more_frames() {
                let decoded_frames = decoder_with_image_info.decoded_frames();

                // Ensure we decoded at least one frame
                assert!(decoded_frames > 0, "No frames were decoded");

                return Ok((decoded_frames, frames));
            }
        }
    }

    fn decode_test_file(path: &Path) -> Result<()> {
        decode(
            &std::fs::read(path).map_err(|e| at!(Error::from(e)))?,
            usize::MAX,
            false,
            false,
            None,
        )?;
        Ok(())
    }

    /// Runs a check over every `.jxl` fixture, replacing the old
    /// `for_each_test_file!` proc macro (which enumerated `resources/test/` at
    /// compile time — impossible when the directory isn't packaged in the
    /// published crate, see #8). Fixtures are checked in parallel across scoped
    /// worker threads: the old macro emitted one `#[test]` per file, which cargo
    /// ran in parallel, so a single sequential loop would be much slower on the
    /// 4K fixtures. All failures are collected and reported together; asserts at
    /// least one fixture ran, so an unresolved corpus fails loudly rather than
    /// passing silently.
    fn run_fixture_sweep(label: &str, check: impl Fn(&Path) -> Result<()> + Sync) {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fixtures = crate::util::test::all_jxl_fixtures();
        let next = AtomicUsize::new(0);
        let ran = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::new());

        // Cap workers per sweep at 8. The four sweep `#[test]`s run concurrently
        // under cargo, and every worker holds a full decoded image while these
        // correctness tests disable memory limits, so uncapped `cores` workers
        // exploded peak memory on a many-core host (4 sweeps x 27 workers = 108
        // concurrent decodes). An absolute cap (not `cores / N`, which would
        // collapse to a slow sequential sweep on a 2-4 core CI runner) keeps both
        // ends sane: full parallelism on small hosts, bounded fan-out on large
        // ones. Wall-clock is bounded by the slowest single fixture regardless.
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let workers = cores.min(8).min(fixtures.len().max(1));

        let worker = || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(path) = fixtures.get(i) else { break };
                // Large images decode to 100+ MB; on 32-bit the address
                // space can't hold several in one process. Skip >1 MB
                // fixtures off 64-bit (matches the old macro's
                // `target_pointer_width = "64"` gate).
                #[cfg(not(target_pointer_width = "64"))]
                if std::fs::metadata(path)
                    .map(|m| m.len() > 1_000_000)
                    .unwrap_or(false)
                {
                    continue;
                }
                ran.fetch_add(1, Ordering::Relaxed);
                if let Err(e) = check(path) {
                    failures
                        .lock()
                        .unwrap()
                        .push(format!("{}: {e:?}", path.display()));
                }
            }
        };

        // wasm32-wasip1 has no threads (spawning traps): run the sweep inline.
        #[cfg(target_arch = "wasm32")]
        {
            let _ = workers;
            worker();
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(worker);
            }
        });

        let failures = failures.into_inner().unwrap();
        let ran = ran.load(Ordering::Relaxed);
        assert!(ran > 0, "{label}: no .jxl fixtures found");
        assert!(
            failures.is_empty(),
            "{label}: {} of {ran} fixtures failed:\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }

    #[test]
    fn decode_test_file_sweep() {
        run_fixture_sweep("decode_test_file", decode_test_file);
    }

    fn decode_test_file_chunks(path: &Path) -> Result<()> {
        decode(
            &std::fs::read(path).map_err(|e| at!(Error::from(e)))?,
            1,
            false,
            false,
            None,
        )?;
        Ok(())
    }

    #[test]
    fn decode_test_file_chunks_sweep() {
        run_fixture_sweep("decode_test_file_chunks", decode_test_file_chunks);
    }

    #[allow(dead_code)] // used by integration tests
    fn compare_frames(_path: &Path, fc: usize, f: &[Image<f32>], sf: &[Image<f32>]) -> Result<()> {
        assert_eq!(
            f.len(),
            sf.len(),
            "Frame {fc} has different channels counts",
        );
        for (c, (b, sb)) in f.iter().zip(sf.iter()).enumerate() {
            assert_eq!(
                b.size(),
                sb.size(),
                "Channel {c} in frame {fc} has different sizes",
            );
            let sz = b.size();
            for y in 0..sz.1 {
                for x in 0..sz.0 {
                    assert_eq!(
                        b.row(y)[x],
                        sb.row(y)[x],
                        "Pixels differ at position ({x}, {y}), channel {c}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Hash all pixel rows for memory-efficient comparison.
    fn hash_frames(frames: &[Vec<Image<f32>>]) -> Vec<Vec<Vec<u64>>> {
        use std::hash::{Hash, Hasher};
        frames
            .iter()
            .map(|channels| {
                channels
                    .iter()
                    .map(|img| {
                        let (_, ys) = img.size();
                        (0..ys)
                            .map(|y| {
                                let mut h = std::hash::DefaultHasher::new();
                                for &v in img.row(y) {
                                    v.to_bits().hash(&mut h);
                                }
                                h.finish()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn compare_pipelines(path: &Path) -> Result<()> {
        let file = std::fs::read(path).map_err(|e| at!(Error::from(e)))?;
        let reference_frames = decode(&file, usize::MAX, true, false, None)?.1;
        // Hash and drop reference pixels before second decode to halve peak
        // memory. Critical for 32-bit targets where two full 4K decoded
        // outputs + decoder state exceeds address space.
        let reference_hashes = hash_frames(&reference_frames);
        drop(reference_frames);
        let frames = decode(&file, usize::MAX, false, false, None)?.1;
        let frame_hashes = hash_frames(&frames);
        assert_eq!(
            reference_hashes,
            frame_hashes,
            "{}: pipeline outputs differ",
            path.display()
        );
        Ok(())
    }

    #[test]
    fn compare_pipelines_sweep() {
        run_fixture_sweep("compare_pipelines", compare_pipelines);
    }

    fn compare_incremental(path: &Path) -> Result<()> {
        let file = std::fs::read(path).unwrap();
        // One-shot decode — hash and drop before incremental decode.
        let (_, one_shot_frames) = decode(&file, usize::MAX, false, false, None)?;
        let reference_hashes = hash_frames(&one_shot_frames);
        drop(one_shot_frames);
        // Incremental decode with arbitrary flushes.
        let (_, frames) = decode(&file, 123, false, true, None)?;
        let frame_hashes = hash_frames(&frames);
        assert_eq!(
            reference_hashes,
            frame_hashes,
            "{}: incremental vs one-shot outputs differ",
            path.display()
        );

        Ok(())
    }

    #[test]
    fn compare_incremental_sweep() {
        run_fixture_sweep("compare_incremental", compare_incremental);
    }

    /// Like `compare_incremental`, with chunk sizes that deliver several
    /// groups per call: the parallel path then decodes in batches, and the
    /// sequential path sees calls that bring one or several new groups.
    /// 30000-byte chunks found two panics (a fragment narrower than its
    /// rectangle in the parallel render, and a flush of a not-yet-decoded
    /// neighbour group) that 123-byte chunks never reached.
    fn compare_incremental_large_chunks(path: &Path) -> Result<()> {
        let file = std::fs::read(path).unwrap();
        let (_, one_shot_frames) = decode(&file, usize::MAX, false, false, None)?;
        let reference_hashes = hash_frames(&one_shot_frames);
        drop(one_shot_frames);
        for chunk in [4096usize, 30_000] {
            if chunk >= file.len() {
                continue;
            }
            let (_, frames) = decode(&file, chunk, false, true, None)?;
            let frame_hashes = hash_frames(&frames);
            assert_eq!(
                reference_hashes,
                frame_hashes,
                "{}: incremental ({chunk}-byte chunks) vs one-shot outputs differ",
                path.display()
            );
        }
        Ok(())
    }

    #[test]
    fn compare_incremental_large_chunks_sweep() {
        run_fixture_sweep(
            "compare_incremental_large_chunks",
            compare_incremental_large_chunks,
        );
    }

    #[test]
    fn test_preview_size_none_for_regular_files() {
        let file = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        assert!(decoder.basic_info().preview_size.is_none());
    }

    #[test]
    fn test_preview_size_some_for_preview_files() {
        let file = std::fs::read(crate::util::test::fixture_path("with_preview.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        assert_eq!(decoder.basic_info().preview_size, Some((16, 16)));
    }

    #[test]
    fn test_num_completed_passes() {
        use crate::image::{Image, Rect};
        let file = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        // Process until we have image info
        let mut decoder_with_info = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        let info = decoder_with_info.basic_info().clone();
        let mut decoder_with_frame = loop {
            match decoder_with_info.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder_with_info = fallback;
                }
            }
        };
        // Before processing frame, passes should be 0
        assert_eq!(decoder_with_frame.num_completed_passes(), 0);
        // Process the frame
        let mut output = Image::<f32>::new((info.size.0 * 3, info.size.1)).unwrap();
        let rect = Rect {
            size: output.size(),
            origin: (0, 0),
        };
        let mut bufs = [JxlOutputBuffer::from_image_rect_mut(
            output.get_rect_mut(rect).into_raw(),
        )];
        loop {
            match decoder_with_frame.process(&mut input, &mut bufs).unwrap() {
                ProcessingResult::Complete { .. } => break,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder_with_frame = fallback,
            }
        }
    }

    #[test]
    fn test_set_pixel_format() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};

        let file = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let mut decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        // Check default pixel format
        let default_format = decoder.current_pixel_format().clone();
        assert_eq!(default_format.color_type, JxlColorType::Rgb);

        // Set a new pixel format
        let new_format = JxlPixelFormat {
            color_type: JxlColorType::Grayscale,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: vec![],
        };
        decoder.set_pixel_format(new_format.clone());

        // Verify it was set
        assert_eq!(decoder.current_pixel_format(), &new_format);
    }

    #[test]
    fn test_set_output_color_profile() {
        use crate::api::JxlColorProfile;

        let file = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let mut decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };

        // Get the embedded profile and set it as output (should work)
        let embedded = decoder.embedded_color_profile().clone();
        let result = decoder.set_output_color_profile(embedded);
        assert!(result.is_ok());

        // Setting an ICC profile without CMS should fail
        let icc_profile = JxlColorProfile::Icc(vec![0u8; 100]);
        let result = decoder.set_output_color_profile(icc_profile);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_output_tf_by_pixel_format() {
        use crate::api::{JxlColorEncoding, JxlTransferFunction};

        // Using test image with ICC profile to trigger default transfer function path
        let file = std::fs::read(crate::util::test::fixture_path("lossy_with_icc.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let mut decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };

        // Output data format will default to F32, so output color profile will be linear sRGB
        assert_eq!(
            *decoder.output_color_profile().transfer_function().unwrap(),
            JxlTransferFunction::Linear,
        );

        // Integer data format will set output color profile to sRGB
        decoder.set_pixel_format(JxlPixelFormat::rgba8(0));
        assert_eq!(
            *decoder.output_color_profile().transfer_function().unwrap(),
            JxlTransferFunction::SRGB,
        );

        decoder.set_pixel_format(JxlPixelFormat::rgba_f16(0));
        assert_eq!(
            *decoder.output_color_profile().transfer_function().unwrap(),
            JxlTransferFunction::Linear,
        );

        decoder.set_pixel_format(JxlPixelFormat::rgba16(0));
        assert_eq!(
            *decoder.output_color_profile().transfer_function().unwrap(),
            JxlTransferFunction::SRGB,
        );

        // Once output color profile is set by user, it will remain as is regardless of what pixel
        // format is set
        let profile = JxlColorProfile::Simple(JxlColorEncoding::srgb(false));
        decoder.set_output_color_profile(profile.clone()).unwrap();
        decoder.set_pixel_format(JxlPixelFormat::rgba_f16(0));
        assert!(decoder.output_color_profile() == &profile);
    }

    #[test]
    fn test_fill_opaque_alpha_both_pipelines() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};
        use crate::image::{Image, Rect};

        // Use basic.jxl which has no alpha channel
        let file = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();

        // Request RGBA format even though image has no alpha
        let rgba_format = JxlPixelFormat {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };

        // Test both pipelines (simple and low-memory)
        for use_simple in [true, false] {
            let options = JxlDecoderOptions::default();
            let decoder = JxlDecoder::<states::Initialized>::new(options);
            let mut input = file.as_slice();

            // Advance to image info
            macro_rules! advance_decoder {
                ($decoder:expr) => {
                    loop {
                        match $decoder.process(&mut input).unwrap() {
                            ProcessingResult::Complete { result } => break result,
                            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                                if input.is_empty() {
                                    panic!("Unexpected end of input");
                                }
                                $decoder = fallback;
                            }
                        }
                    }
                };
                ($decoder:expr, $buffers:expr) => {
                    loop {
                        match $decoder.process(&mut input, $buffers).unwrap() {
                            ProcessingResult::Complete { result } => break result,
                            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                                if input.is_empty() {
                                    panic!("Unexpected end of input");
                                }
                                $decoder = fallback;
                            }
                        }
                    }
                };
            }

            let mut decoder = decoder;
            let mut decoder = advance_decoder!(decoder);
            decoder.set_use_simple_pipeline(use_simple);

            // Set RGBA format
            decoder.set_pixel_format(rgba_format.clone());

            let basic_info = decoder.basic_info().clone();
            let (width, height) = basic_info.size;

            // Advance to frame info
            let mut decoder = advance_decoder!(decoder);

            // Prepare buffer for RGBA (4 channels interleaved)
            let mut color_buffer = Image::<f32>::new((width * 4, height)).unwrap();
            let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
                color_buffer
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size: (width * 4, height),
                    })
                    .into_raw(),
            )];

            // Decode frame
            let _decoder = advance_decoder!(decoder, &mut buffers);

            // Verify all alpha values are 1.0 (opaque)
            for y in 0..height {
                let row = color_buffer.row(y);
                for x in 0..width {
                    let alpha = row[x * 4 + 3];
                    assert_eq!(
                        alpha, 1.0,
                        "Alpha at ({},{}) should be 1.0, got {} (use_simple={})",
                        x, y, alpha, use_simple
                    );
                }
            }
        }
    }

    /// Test that premultiply_output=true produces premultiplied alpha output
    /// from a source with straight (non-premultiplied) alpha.
    #[test]
    fn test_premultiply_output_straight_alpha() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};

        // Use alpha_nonpremultiplied.jxl which has straight alpha (alpha_associated=false)
        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/alpha_nonpremultiplied.jxl",
        ))
        .unwrap();

        // Alpha is included in RGBA, so we set extra_channel_format to None
        // to indicate no separate buffer for the alpha extra channel
        let rgba_format = JxlPixelFormat {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![None],
        };

        // Test both pipelines
        for use_simple in [true, false] {
            let (straight_buffer, width, height) =
                decode_with_format::<f32>(&file, &rgba_format, use_simple, false).unwrap();
            let straight_buffer = &straight_buffer[0];
            let (premul_buffer, _, _) =
                decode_with_format::<f32>(&file, &rgba_format, use_simple, true).unwrap();
            let premul_buffer = &premul_buffer[0];

            // Verify premultiplied values: premul_rgb should equal straight_rgb * alpha
            let mut found_semitransparent = false;
            for y in 0..height {
                let straight_row = straight_buffer.row(y);
                let premul_row = premul_buffer.row(y);
                for x in 0..width {
                    let sr = straight_row[x * 4];
                    let sg = straight_row[x * 4 + 1];
                    let sb = straight_row[x * 4 + 2];
                    let sa = straight_row[x * 4 + 3];

                    let pr = premul_row[x * 4];
                    let pg = premul_row[x * 4 + 1];
                    let pb = premul_row[x * 4 + 2];
                    let pa = premul_row[x * 4 + 3];

                    // Alpha should be unchanged
                    assert!(
                        (sa - pa).abs() < 1e-5,
                        "Alpha mismatch at ({},{}): straight={}, premul={} (use_simple={})",
                        x,
                        y,
                        sa,
                        pa,
                        use_simple
                    );

                    // Check premultiplication: premul = straight * alpha
                    let expected_r = sr * sa;
                    let expected_g = sg * sa;
                    let expected_b = sb * sa;

                    // Allow 1% tolerance for precision differences between pipelines
                    let tol = 0.01;
                    assert!(
                        (expected_r - pr).abs() < tol,
                        "R mismatch at ({},{}): expected={}, got={} (use_simple={})",
                        x,
                        y,
                        expected_r,
                        pr,
                        use_simple
                    );
                    assert!(
                        (expected_g - pg).abs() < tol,
                        "G mismatch at ({},{}): expected={}, got={} (use_simple={})",
                        x,
                        y,
                        expected_g,
                        pg,
                        use_simple
                    );
                    assert!(
                        (expected_b - pb).abs() < tol,
                        "B mismatch at ({},{}): expected={}, got={} (use_simple={})",
                        x,
                        y,
                        expected_b,
                        pb,
                        use_simple
                    );

                    if sa > 0.01 && sa < 0.99 {
                        found_semitransparent = true;
                    }
                }
            }

            // Ensure the test image actually has some semi-transparent pixels
            assert!(
                found_semitransparent,
                "Test image should have semi-transparent pixels (use_simple={})",
                use_simple
            );
        }
    }

    /// Test that premultiply_output=true doesn't double-premultiply
    /// when the source already has premultiplied alpha (alpha_associated=true).
    #[test]
    fn test_premultiply_output_already_premultiplied() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};

        // Use alpha_premultiplied.jxl which has alpha_associated=true
        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/alpha_premultiplied.jxl",
        ))
        .unwrap();

        // Alpha is included in RGBA, so we set extra_channel_format to None
        let rgba_format = JxlPixelFormat {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![None],
        };

        // Test both pipelines
        for use_simple in [true, false] {
            let (without_flag_buffer, width, height) =
                decode_with_format::<f32>(&file, &rgba_format, use_simple, false).unwrap();
            let without_flag_buffer = &without_flag_buffer[0];
            let (with_flag_buffer, _, _) =
                decode_with_format::<f32>(&file, &rgba_format, use_simple, true).unwrap();
            let with_flag_buffer = &with_flag_buffer[0];

            // Both outputs should be identical since source is already premultiplied
            // and we shouldn't double-premultiply
            for y in 0..height {
                let without_row = without_flag_buffer.row(y);
                let with_row = with_flag_buffer.row(y);
                for x in 0..width {
                    for c in 0..4 {
                        let without_val = without_row[x * 4 + c];
                        let with_val = with_row[x * 4 + c];
                        assert!(
                            (without_val - with_val).abs() < 1e-5,
                            "Mismatch at ({},{}) channel {}: without_flag={}, with_flag={} (use_simple={})",
                            x,
                            y,
                            c,
                            without_val,
                            with_val,
                            use_simple
                        );
                    }
                }
            }
        }
    }

    /// Test that animations with reference frames work correctly.
    /// This exercises the buffer index calculation fix where reference frame
    /// save stages use indices beyond the API-provided buffer array.
    #[test]
    fn test_animation_with_reference_frames() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};
        use crate::image::{Image, Rect};

        // Use animation_spline.jxl which has multiple frames with references
        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/animation_spline.jxl",
        ))
        .unwrap();

        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();

        // Advance to image info
        let mut decoder = decoder;
        let mut decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder = fallback;
                }
            }
        };

        // Set RGB format with no extra channels
        let rgb_format = JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };
        decoder.set_pixel_format(rgb_format);

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;

        let mut frame_count = 0;

        // Decode all frames
        loop {
            // Advance to frame info
            let mut decoder_frame = loop {
                match decoder.process(&mut input).unwrap() {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        decoder = fallback;
                    }
                }
            };

            // Prepare buffer for RGB (3 channels interleaved)
            let mut color_buffer = Image::<f32>::new((width * 3, height)).unwrap();
            let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
                color_buffer
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size: (width * 3, height),
                    })
                    .into_raw(),
            )];

            // Decode frame - this should not panic even though reference frame
            // save stages target buffer indices beyond buffers.len()
            decoder = loop {
                match decoder_frame.process(&mut input, &mut buffers).unwrap() {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        decoder_frame = fallback;
                    }
                }
            };

            frame_count += 1;

            // Check if there are more frames
            if !decoder.has_more_frames() {
                break;
            }
        }

        // Verify we decoded multiple frames
        assert!(
            frame_count > 1,
            "Expected multiple frames in animation, got {}",
            frame_count
        );
    }

    #[test]
    fn test_skip_frame_then_decode_next() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};
        use crate::image::{Image, Rect};

        // Use animation_spline.jxl which has multiple frames
        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/animation_spline.jxl",
        ))
        .unwrap();

        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();

        // Advance to image info
        let mut decoder = decoder;
        let mut decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder = fallback;
                }
            }
        };

        // Set RGB format
        let rgb_format = JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };
        decoder.set_pixel_format(rgb_format);

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;

        // Advance to frame info for first frame
        let mut decoder_frame = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder = fallback;
                }
            }
        };

        // Skip the first frame (this is where the bug would leave stale frame state)
        let mut decoder = loop {
            match decoder_frame.skip_frame(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder_frame = fallback;
                }
            }
        };

        assert!(
            decoder.has_more_frames(),
            "Animation should have more frames"
        );

        // Advance to frame info for second frame
        // Without the fix, this would panic at assert!(self.frame.is_none())
        let mut decoder_frame = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder = fallback;
                }
            }
        };

        // Decode the second frame to verify everything works
        let mut color_buffer = Image::<f32>::new((width * 3, height)).unwrap();
        let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
            color_buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * 3, height),
                })
                .into_raw(),
        )];

        let decoder = loop {
            match decoder_frame.process(&mut input, &mut buffers).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder_frame = fallback;
                }
            }
        };

        // If we got here without panicking, the fix works
        // Optionally verify we can continue with more frames
        let _ = decoder.has_more_frames();
    }

    /// Test that u8 output matches f32 output within quantization tolerance.
    /// This test would catch bugs like the offset miscalculation in PR #586
    /// that caused black bars in u8 output.
    #[test]
    fn test_output_format_u8_matches_f32() {
        use crate::api::{JxlColorType, JxlDataFormat, JxlPixelFormat};

        // Use bicycles.jxl - a larger image that exercises offset calculations
        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/bicycles.jxl",
        ))
        .unwrap();

        // Test both RGB and BGRA to catch channel reordering bugs
        for (color_type, num_samples) in [(JxlColorType::Rgb, 3), (JxlColorType::Bgra, 4)] {
            let f32_format = JxlPixelFormat {
                color_type,
                color_data_format: Some(JxlDataFormat::f32()),
                extra_channel_format: vec![],
            };
            let u8_format = JxlPixelFormat {
                color_type,
                color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
                extra_channel_format: vec![],
            };

            // Test both pipelines
            for use_simple in [true, false] {
                let (f32_buffer, width, height) =
                    decode_with_format::<f32>(&file, &f32_format, use_simple, false).unwrap();
                let f32_buffer = &f32_buffer[0];
                let (u8_buffer, _, _) =
                    decode_with_format::<u8>(&file, &u8_format, use_simple, false).unwrap();
                let u8_buffer = &u8_buffer[0];

                // Compare values: u8 / 255.0 should match f32
                // Tolerance: quantization error of ±0.5/255 plus the blue-noise
                // dither of up to ±0.49219/255 applied before rounding
                // (render::stages::dither), i.e. (0.5 + 0.49219) / 255 = 0.00389,
                // plus a little f32 rounding slack.
                let tolerance = 0.004;
                let mut max_error: f32 = 0.0;

                for y in 0..height {
                    let f32_row = f32_buffer.row(y);
                    let u8_row = u8_buffer.row(y);
                    for x in 0..(width * num_samples) {
                        let f32_val = f32_row[x].clamp(0.0, 1.0);
                        let u8_val = u8_row[x] as f32 / 255.0;
                        let error = (f32_val - u8_val).abs();
                        max_error = max_error.max(error);
                        assert!(
                            error < tolerance,
                            "{:?} u8 mismatch at ({},{}): f32={}, u8={} (scaled={}), error={} (use_simple={})",
                            color_type,
                            x,
                            y,
                            f32_val,
                            u8_row[x],
                            u8_val,
                            error,
                            use_simple
                        );
                    }
                }
            }
        }
    }

    /// Test that u16 output matches f32 output within quantization tolerance.
    #[test]
    fn test_output_format_u16_matches_f32() {
        use crate::api::{Endianness, JxlColorType, JxlDataFormat, JxlPixelFormat};

        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/bicycles.jxl",
        ))
        .unwrap();

        // Test both RGB and BGRA
        for (color_type, num_samples) in [(JxlColorType::Rgb, 3), (JxlColorType::Bgra, 4)] {
            let f32_format = JxlPixelFormat {
                color_type,
                color_data_format: Some(JxlDataFormat::f32()),
                extra_channel_format: vec![],
            };
            let u16_format = JxlPixelFormat {
                color_type,
                color_data_format: Some(JxlDataFormat::U16 {
                    endianness: Endianness::native(),
                    bit_depth: 16,
                }),
                extra_channel_format: vec![],
            };

            for use_simple in [true, false] {
                let (f32_buffer, width, height) =
                    decode_with_format::<f32>(&file, &f32_format, use_simple, false).unwrap();
                let f32_buffer = &f32_buffer[0];
                let (u16_buffer, _, _) =
                    decode_with_format::<u16>(&file, &u16_format, use_simple, false).unwrap();
                let u16_buffer = &u16_buffer[0];

                // Tolerance: quantization error of ±0.5/65535 plus small rounding
                let tolerance = 0.0001;

                for y in 0..height {
                    let f32_row = f32_buffer.row(y);
                    let u16_row = u16_buffer.row(y);
                    for x in 0..(width * num_samples) {
                        let f32_val = f32_row[x].clamp(0.0, 1.0);
                        let u16_val = u16_row[x] as f32 / 65535.0;
                        let error = (f32_val - u16_val).abs();
                        assert!(
                            error < tolerance,
                            "{:?} u16 mismatch at ({},{}): f32={}, u16={} (scaled={}), error={} (use_simple={})",
                            color_type,
                            x,
                            y,
                            f32_val,
                            u16_row[x],
                            u16_val,
                            error,
                            use_simple
                        );
                    }
                }
            }
        }
    }

    /// Test that f16 output matches f32 output within f16 precision tolerance.
    #[test]
    fn test_output_format_f16_matches_f32() {
        use crate::api::{Endianness, JxlColorType, JxlDataFormat, JxlPixelFormat};
        use crate::util::f16;

        let file = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/bicycles.jxl",
        ))
        .unwrap();

        // Test both RGB and BGRA
        for (color_type, num_samples) in [(JxlColorType::Rgb, 3), (JxlColorType::Bgra, 4)] {
            let f32_format = JxlPixelFormat {
                color_type,
                color_data_format: Some(JxlDataFormat::f32()),
                extra_channel_format: vec![],
            };
            let f16_format = JxlPixelFormat {
                color_type,
                color_data_format: Some(JxlDataFormat::F16 {
                    endianness: Endianness::native(),
                }),
                extra_channel_format: vec![],
            };

            for use_simple in [true, false] {
                let (f32_buffer, width, height) =
                    decode_with_format::<f32>(&file, &f32_format, use_simple, false).unwrap();
                let f32_buffer = &f32_buffer[0];
                let (f16_buffer, _, _) =
                    decode_with_format::<f16>(&file, &f16_format, use_simple, false).unwrap();
                let f16_buffer = &f16_buffer[0];

                // f16 has about 3 decimal digits of precision
                // For values in [0,1], the relative error is about 0.001
                let tolerance = 0.002;

                for y in 0..height {
                    let f32_row = f32_buffer.row(y);
                    let f16_row = f16_buffer.row(y);
                    for x in 0..(width * num_samples) {
                        let f32_val = f32_row[x];
                        let f16_val = f16_row[x].to_f32();
                        let error = (f32_val - f16_val).abs();
                        assert!(
                            error < tolerance,
                            "{:?} f16 mismatch at ({},{}): f32={}, f16={}, error={} (use_simple={})",
                            color_type,
                            x,
                            y,
                            f32_val,
                            f16_val,
                            error,
                            use_simple
                        );
                    }
                }
            }
        }
    }

    /// Helper function to decode an image with a specific format.
    /// `flush_pixels` returns `true` only when new pixels were rendered since
    /// the previous call (upstream jxl-rs #755): a second back-to-back flush
    /// with no new input must report `false`, and a chunked decode of a
    /// multi-group image must report `true` at least once mid-stream.
    #[test]
    fn flush_pixels_reports_new_rendering() {
        const CHUNK: usize = 4096;
        let file = crate::util::test::fixture_bytes("bicycles_web_q85.jxl");
        let mut options = JxlDecoderOptions::default();
        options.limits.max_memory_bytes = None;
        let mut initialized = JxlDecoder::<states::Initialized>::new(options);

        let mut input: &[u8] = &file;
        let mut chunk_input = &input[0..0];

        macro_rules! feed {
            ($decoder:ident $(, $extra:expr)? ; $on_needs_more:expr) => {
                loop {
                    chunk_input =
                        &input[..(chunk_input.len().saturating_add(CHUNK)).min(input.len())];
                    let before = chunk_input.len();
                    let process_result = $decoder.process(&mut chunk_input $(, $extra)?);
                    input = &input[(before - chunk_input.len())..];
                    match process_result.unwrap() {
                        ProcessingResult::Complete { result } => break Some(result),
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            let mut fallback = fallback;
                            #[allow(clippy::redundant_closure_call)]
                            ($on_needs_more)(&mut fallback);
                            if input.is_empty() {
                                break None;
                            }
                            $decoder = fallback;
                        }
                    }
                }
            };
        }

        let mut with_image_info = feed!(initialized; |_f: &mut _| {}).unwrap();

        let num_extra_channels = with_image_info
            .current_pixel_format()
            .extra_channel_format
            .len();
        with_image_info.set_pixel_format(JxlPixelFormat::rgb8(num_extra_channels));
        let (width, height) = with_image_info.basic_info().size;
        let num_samples = with_image_info
            .current_pixel_format()
            .color_type
            .samples_per_pixel();
        let mut pixels = Image::<u8>::new((width * num_samples, height)).unwrap();

        // Nothing has been decoded yet: a flush now must report false.
        {
            let size = pixels.size();
            let mut bufs = [JxlOutputBuffer::from_image_rect_mut(
                pixels
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size,
                    })
                    .into_raw(),
            )];
            assert!(
                !with_image_info.flush_pixels(&mut bufs).unwrap(),
                "flush before any frame data must report false"
            );
        }

        let mut with_frame_info = feed!(with_image_info; |_f: &mut _| {}).unwrap();

        let mut saw_render_on_flush = false;
        let mut double_flushes = 0usize;
        {
            let size = pixels.size();
            let mut bufs = [JxlOutputBuffer::from_image_rect_mut(
                pixels
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size,
                    })
                    .into_raw(),
            )];
            let complete = feed!(with_frame_info, &mut bufs; |f: &mut JxlDecoder<
                WithFrameInfo,
            >| {
                let first = f.flush_pixels(&mut bufs).unwrap();
                let second = f.flush_pixels(&mut bufs).unwrap();
                assert!(!second, "second flush with no new data must report false");
                saw_render_on_flush |= first;
                double_flushes += 1;
            });
            assert!(complete.is_some(), "unexpected end of input");
        }
        assert!(double_flushes > 4, "expected a chunked multi-chunk decode");
        assert!(
            saw_render_on_flush,
            "expected at least one flush to report newly rendered pixels"
        );
    }

    /// Premultiplied RGBA output from a grayscale image remains gray.
    /// (upstream jxl-rs #903)
    #[test]
    fn test_premultiply_output_grayscale_as_rgba() {
        let file = crate::util::test::fixture_bytes("gray_alpha_lossless.jxl");
        let (buffers, width, height) =
            decode_with_format::<f32>(&file, &JxlPixelFormat::rgba_f32(1), false, true).unwrap();
        let rgba = &buffers[0];

        for y in 0..height {
            let row = rgba.row(y);
            for x in 0..width {
                assert_eq!(row[x * 4], row[x * 4 + 1], "R!=G at ({x},{y})");
                assert_eq!(row[x * 4 + 1], row[x * 4 + 2], "G!=B at ({x},{y})");
            }
        }
    }

    /// CMYK interleaved output matches the RGB color channels for C, M and Y,
    /// and the Black extra channel plane for K. (upstream jxl-rs #891)
    #[test]
    fn test_cmyk_pixel_format() {
        let file = crate::util::test::fixture_bytes("conformance_test_images/cmyk_layers.jxl");

        // cmyk_layers.jxl has two extra channels: Black (index 0) and Alpha
        // (index 1).
        let cmyk_format = JxlPixelFormat::cmyk8(2);
        let reference_format = JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: vec![Some(JxlDataFormat::U8 { bit_depth: 8 }), None],
        };

        for use_simple in [true, false] {
            let (cmyk_buffers, width, height) =
                decode_with_format::<u8>(&file, &cmyk_format, use_simple, false).unwrap();
            let (reference_buffers, _, _) =
                decode_with_format::<u8>(&file, &reference_format, use_simple, false).unwrap();
            let cmyk = &cmyk_buffers[0];
            let rgb = &reference_buffers[0];
            let black = &reference_buffers[1];

            for y in 0..height {
                let cmyk_row = cmyk.row(y);
                let rgb_row = rgb.row(y);
                let black_row = black.row(y);
                for x in 0..width {
                    for c in 0..3 {
                        assert_eq!(
                            cmyk_row[x * 4 + c],
                            rgb_row[x * 3 + c],
                            "CMY mismatch at ({x},{y}) channel {c} (use_simple={use_simple})"
                        );
                    }
                    assert_eq!(
                        cmyk_row[x * 4 + 3],
                        black_row[x],
                        "K mismatch at ({x},{y}) (use_simple={use_simple})"
                    );
                }
            }
        }
    }

    /// Requesting CMYK output for a non-CMYK image fails. (upstream jxl-rs #891)
    #[test]
    fn test_cmyk_pixel_format_requires_cmyk_image() {
        let file = crate::util::test::fixture_bytes("basic.jxl");
        let err = decode_with_format::<u8>(&file, &JxlPixelFormat::cmyk8(0), false, false)
            .unwrap_err();
        assert!(
            matches!(err.error(), Error::NotCmyk),
            "expected NotCmyk, got {err:?}"
        );
    }

    /// Helper to decode an image with a specific format, with buffers for the
    /// color channels (if requested) plus every requested extra channel plane.
    /// Returns the decoded buffers in process() buffer order.
    fn decode_with_format<T: crate::image::ImageDataType>(
        file: &[u8],
        pixel_format: &JxlPixelFormat,
        use_simple: bool,
        premultiply: bool,
    ) -> Result<(Vec<Image<T>>, usize, usize)> {
        let options = JxlDecoderOptions {
            premultiply_output: premultiply,
            ..Default::default()
        };
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file;

        // Advance to image info
        let mut decoder = loop {
            match decoder.process(&mut input)? {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    decoder = fallback;
                }
            }
        };
        decoder.set_use_simple_pipeline(use_simple);
        decoder.set_pixel_format(pixel_format.clone());

        let (width, height) = decoder.basic_info().size;
        let num_samples = pixel_format.color_type.samples_per_pixel();

        // Advance to frame info
        let mut decoder = loop {
            match decoder.process(&mut input)? {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    decoder = fallback;
                }
            }
        };

        let mut images = Vec::new();
        if pixel_format.color_data_format.is_some() {
            images.push(Image::<T>::new((width * num_samples, height)).unwrap());
        }
        for ec_format in &pixel_format.extra_channel_format {
            if ec_format.is_some() {
                images.push(Image::<T>::new((width, height)).unwrap());
            }
        }
        let mut buffers: Vec<JxlOutputBuffer> = images
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
            .collect();

        // Decode
        loop {
            match decoder.process(&mut input, &mut buffers)? {
                ProcessingResult::Complete { .. } => break,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    decoder = fallback;
                }
            }
        }
        drop(buffers);

        Ok((images, width, height))
    }

    /// Regression test for ClusterFuzz issue 5342436251336704
    /// Tests that malformed JXL files with overflow-inducing data don't panic
    // The test tolerates the helper's end-of-input panic through
    // `catch_unwind`, which needs unwinding; wasm32-wasip1 aborts on panic.
    #[cfg(panic = "unwind")]
    #[test]
    fn test_fuzzer_smallbuffer_overflow() {
        use std::panic;

        let data = include_bytes!("../../tests/testdata/fuzzer_smallbuffer_overflow.jxl");

        // The test passes if it doesn't panic with "attempt to add with overflow"
        // It's OK if it returns an error or panics with "Unexpected end of input"
        let result = panic::catch_unwind(|| {
            let _ = decode(data, 1024, false, false, None);
        });

        // If it panicked, make sure it wasn't an overflow panic
        if let Err(e) = result {
            let panic_msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            assert!(
                !panic_msg.contains("overflow"),
                "Unexpected overflow panic: {}",
                panic_msg
            );
        }
    }

    /// Helper to wrap a bare codestream in a JXL container with a jxli frame index box.
    fn wrap_with_frame_index(
        codestream: &[u8],
        tnum: u32,
        tden: u32,
        entries: &[(u64, u64, u64)], // (OFF_delta, T, F)
    ) -> Vec<u8> {
        use crate::util::test::build_frame_index_content;

        fn make_box(ty: &[u8; 4], content: &[u8]) -> Vec<u8> {
            let len = (8 + content.len()) as u32;
            let mut buf = Vec::new();
            buf.extend(len.to_be_bytes());
            buf.extend(ty);
            buf.extend(content);
            buf
        }

        let jxli_content = build_frame_index_content(tnum, tden, entries);

        // JXL signature box
        let sig = [
            0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
        ];
        // ftyp box
        let ftyp = make_box(b"ftyp", b"jxl \x00\x00\x00\x00jxl ");
        let jxli = make_box(b"jxli", &jxli_content);
        let jxlc = make_box(b"jxlc", codestream);

        let mut container = Vec::new();
        container.extend(&sig);
        container.extend(&ftyp);
        container.extend(&jxli);
        container.extend(&jxlc);
        container
    }

    #[test]
    fn test_frame_index_parsed_from_container() {
        // Read a bare animation codestream and wrap it in a container with a jxli box.
        let codestream = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/animation_icos4d_5.jxl",
        ))
        .unwrap();

        // Create synthetic frame index entries (delta offsets).
        // These are synthetic -- we don't know real frame offsets, but we can verify parsing.
        let entries = vec![
            (0u64, 100u64, 1u64), // Frame 0 at offset 0
            (500, 100, 1),        // Frame 1 at offset 500
            (600, 100, 1),        // Frame 2 at offset 1100
        ];

        let container = wrap_with_frame_index(&codestream, 1, 1000, &entries);

        // Decode with a large chunk size so the jxli box is fully consumed.
        let options = JxlDecoderOptions::default();
        let mut dec = JxlDecoder::<states::Initialized>::new(options);
        let mut input: &[u8] = &container;
        let dec = loop {
            match dec.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    dec = fallback;
                }
            }
        };

        // Check that frame index was parsed.
        let fi = dec.frame_index().expect("frame_index should be Some");
        assert_eq!(fi.num_frames(), 3);
        assert_eq!(fi.tnum, 1);
        assert_eq!(fi.tden.get(), 1000);
        // Verify absolute offsets (accumulated from deltas)
        assert_eq!(fi.entries[0].codestream_offset, 0);
        assert_eq!(fi.entries[1].codestream_offset, 500);
        assert_eq!(fi.entries[2].codestream_offset, 1100);
        assert_eq!(fi.entries[0].duration_ticks, 100);
        assert_eq!(fi.entries[2].frame_count, 1);
    }

    #[test]
    fn test_frame_index_none_for_bare_codestream() {
        // A bare codestream has no container, so no frame index.
        let data = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/animation_icos4d_5.jxl",
        ))
        .unwrap();
        let options = JxlDecoderOptions::default();
        let mut dec = JxlDecoder::<states::Initialized>::new(options);
        let mut input: &[u8] = &data;
        let dec = loop {
            match dec.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input");
                    }
                    dec = fallback;
                }
            }
        };
        assert!(dec.frame_index().is_none());
    }

    /// Regression test for Chromium ClusterFuzz issue 474401148.
    #[test]
    fn test_fuzzer_xyb_icc_no_panic() {
        use crate::api::ProcessingResult;

        #[rustfmt::skip]
        let data: &[u8] = &[
            0xff, 0x0a, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x25, 0x00,
        ];

        let opts = JxlDecoderOptions::default();
        let mut decoder = JxlDecoderInner::new(opts);
        let mut input = data;

        if let Ok(ProcessingResult::Complete { .. }) = decoder.process(&mut input, None)
            && let Some(profile) = decoder.output_color_profile()
        {
            let _ = profile.try_as_icc();
        }
    }

    #[test]
    fn test_pixel_limit_enforcement() {
        // Load a test image - green_queen is 256x256 = 65536 pixels
        let input =
            std::fs::read(crate::util::test::fixture_path("green_queen_vardct_e3.jxl")).unwrap();

        // Create options with a very restrictive pixel limit (smaller than the image)
        let mut options = JxlDecoderOptions::default();
        options.limits.max_pixels = Some(100); // Only 100 pixels allowed

        let decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input_slice = &input[..];

        // The decoder should fail when parsing the header with LimitExceeded error
        let result = decoder.process(&mut input_slice);
        match result {
            Err(err) => {
                assert!(
                    matches!(
                        err.error(),
                        Error::LimitExceeded {
                            resource: "pixels",
                            ..
                        }
                    ),
                    "Expected LimitExceeded for pixels, got {:?}",
                    err
                );
            }
            Ok(ProcessingResult::NeedsMoreInput { .. }) => {
                panic!("Expected error, got needs more input");
            }
            Ok(ProcessingResult::Complete { .. }) => {
                panic!("Expected error, got success");
            }
        }
    }

    #[test]
    fn test_restrictive_limits_preset() {
        // Verify the restrictive preset is reasonable
        let limits = crate::api::JxlDecoderLimits::restrictive();
        assert_eq!(limits.max_pixels, Some(120_000_000));
        assert_eq!(limits.max_extra_channels, Some(16));
        assert_eq!(limits.max_icc_size, Some(16 << 20));
        assert_eq!(limits.max_tree_size, Some(1 << 20));
        assert_eq!(limits.max_patches, Some(1 << 16));
        assert_eq!(limits.max_spline_points, Some(1 << 16));
        assert_eq!(limits.max_reference_frames, Some(2));
        assert_eq!(limits.max_memory_bytes, Some(1 << 30));
    }

    #[test]
    fn test_extra_channel_metadata() {
        let file = std::fs::read(crate::util::test::fixture_path("extra_channels.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        let info = decoder.basic_info();
        // extra_channels.jxl should have at least one extra channel
        assert!(
            !info.extra_channels.is_empty(),
            "expected at least one extra channel"
        );

        // Verify all new fields are populated
        for ec in &info.extra_channels {
            // bits_per_sample should be a reasonable value
            assert!(
                ec.bits_per_sample > 0 && ec.bits_per_sample <= 32,
                "unexpected bits_per_sample: {}",
                ec.bits_per_sample
            );
            // dim_shift should be <= 3
            assert!(ec.dim_shift <= 3, "unexpected dim_shift: {}", ec.dim_shift);
        }
    }

    #[test]
    fn test_extra_channel_alpha_with_new_fields() {
        use crate::headers::extra_channels::ExtraChannel;

        // 3x3a has alpha
        let file =
            std::fs::read(crate::util::test::fixture_path("3x3a_srgb_lossless.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        let info = decoder.basic_info();
        // Should have exactly one extra channel of type Alpha
        assert_eq!(info.extra_channels.len(), 1);
        let alpha = &info.extra_channels[0];
        assert_eq!(alpha.ec_type, ExtraChannel::Alpha);
        assert!(alpha.bits_per_sample > 0);
        // Default alpha channels typically have dim_shift 0 (full resolution)
        assert_eq!(alpha.dim_shift, 0);
    }

    #[test]
    fn test_preview_metadata_in_basic_info() {
        // with_preview.jxl has a preview; basic.jxl does not
        let file = std::fs::read(crate::util::test::fixture_path("with_preview.jxl")).unwrap();
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file.as_slice();
        let decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        let info = decoder.basic_info();
        let (pw, ph) = info.preview_size.expect("expected preview_size");
        assert!(pw > 0 && ph > 0, "preview dimensions should be positive");
    }

    #[test]
    fn test_stop_cancellation() {
        use almost_enough::Stopper;
        use enough::Stop;

        let stop = Stopper::new();
        assert!(!stop.should_stop());
        stop.cancel();
        assert!(stop.should_stop());
        // Verify it integrates with our error type
        let result: crate::error::Result<()> =
            stop.check().map_err(|e| at!(crate::error::Error::from(e)));
        assert!(matches!(
            result,
            Err(e) if matches!(e.error(), crate::error::Error::Cancelled)
        ));
    }

    /// Regression for the preview-frame recovery option-propagation bug that
    /// was fixed upstream in libjxl/jxl-rs #743 (commit f1514f1).
    ///
    /// When the input file carries a preview frame, the codestream parser
    /// decodes the preview with `process_without_output=true`, then discovers
    /// the main frame is a separate frame and recreates the [`DecoderState`]
    /// in `codestream_parser::sections::handle_frame_finalized`. Before the
    /// port, that recreation path dropped several fields (`high_precision`,
    /// `premultiply_output`, `parallel`, `memory_tracker`,
    /// `embedded_color_profile`) back to their constructor defaults, silently
    /// reverting options set by the caller.
    ///
    /// The fix centralizes option propagation through
    /// `non_section::apply_decoder_options` so both the primary creation path
    /// and the preview-recovery path populate the same fields.
    ///
    /// The test fully decodes `with_preview.jxl` with non-default options, so
    /// the preview frame finalize path runs and the recreation branch is
    /// taken, then asserts every recreated field carries the configured
    /// option value rather than the `DecoderState::new` default.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_preview_recovery_preserves_decoder_options() {
        let data = std::fs::read(crate::util::test::fixture_path("with_preview.jxl"))
            .expect("with_preview.jxl test fixture should exist");

        // Flip every option the recovery path used to drop to a non-default
        // value (`render_spot_colors=false`, `high_precision=true`,
        // `premultiply_output=true`, `parallel=false`, restrictive
        // `max_memory_bytes`) so a successful decode with the buggy code
        // would visibly carry the wrong field values. `JxlDecoderOptions`
        // is `#[non_exhaustive]`, so a struct literal with
        // `..Default::default()` is not allowed.
        let mut options = JxlDecoderOptions::default();
        options.high_precision = true;
        options.premultiply_output = true;
        options.parallel = false;
        options.render_spot_colors = false;
        // Generous enough to actually decode the tiny test file but still
        // a finite limit so memory_tracker.has_limit() is true.
        options.limits.max_memory_bytes = Some(64 * 1024 * 1024);

        let mut decoder = JxlDecoderInner::new(options);
        let mut input = data.as_slice();

        // 1. Process up to image info.
        match decoder.process(&mut input, None) {
            Ok(ProcessingResult::Complete { .. }) => {}
            other => panic!("expected image-info Complete, got {other:?}"),
        }
        assert!(decoder.basic_info().is_some());

        // 2. Process up to the (main) frame header. With the default
        //    `skip_preview=true`, the preview frame is fully decoded with
        //    `process_without_output=true`, then the recreate branch in
        //    `sections::handle_frame_finalized` runs and the decoder advances
        //    to the main frame. The main-frame `Frame::from_header_and_toc`
        //    consumes the recreated `DecoderState`, so by the time `process`
        //    returns here the recreated state lives inside the active Frame.
        match decoder.process(&mut input, None) {
            Ok(ProcessingResult::Complete { .. }) => {}
            other => panic!("expected frame-info Complete, got {other:?}"),
        }
        assert!(decoder.frame_header().is_some());

        // Inspect the recreated state (now inside the main-frame Frame)
        // BEFORE the main frame finalizes and drops it.
        let state = decoder
            .decoder_state_for_test()
            .expect("decoder_state must exist inside the active main frame");

        // Before the fix, all of the following assertions would fail when
        // the preview-recovery branch was taken: the recreated state reset
        // every knob below to its DecoderState::new() default.
        assert!(
            state.high_precision,
            "high_precision should survive preview-frame recovery"
        );
        assert!(
            state.premultiply_output,
            "premultiply_output should survive preview-frame recovery"
        );
        assert!(
            !state.parallel,
            "parallel=false should survive preview-frame recovery (was silently flipped back to DecoderState::new default)"
        );
        assert!(
            !state.render_spotcolors,
            "render_spotcolors=false should survive preview-frame recovery"
        );
        assert!(
            state.memory_tracker.has_limit(),
            "memory_tracker should carry the configured limit after preview-frame recovery, not revert to unlimited"
        );
        assert_eq!(
            state.memory_tracker.limit(),
            Some(64 * 1024 * 1024),
            "memory_tracker limit should equal configured max_memory_bytes"
        );
        assert!(
            state.embedded_color_profile.is_some(),
            "embedded_color_profile must be propagated so CMYK ICC and similar code paths work after preview recovery"
        );
        assert_eq!(
            state.limits.max_memory_bytes,
            Some(64 * 1024 * 1024),
            "limits.max_memory_bytes on DecoderState must match the configured options"
        );
    }

    /// Chunk-drip stress test mirroring the Chrome-integration repro from
    /// libjxl/jxl-rs #743. We don't have the seek API (upstream #678) yet, but
    /// we can still exercise the same box-parser and codestream-parser state
    /// machines by feeding an animation file to `flush_pixels` in 1 KiB chunks
    /// and asserting the decoder never errors or panics on any chunk boundary.
    #[test]
    fn test_chunked_drip_decode_animation_newtons_cradle() {
        let data = std::fs::read(crate::util::test::fixture_path(
            "conformance_test_images/animation_newtons_cradle.jxl",
        ))
        .expect("animation_newtons_cradle.jxl test fixture should exist");

        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoderInner::new(options);
        const CHUNK: usize = 1024;
        let mut fed = 0usize;

        while fed < data.len() {
            let end = (fed + CHUNK).min(data.len());
            let mut chunk = &data[fed..end];
            let before = chunk.len();
            match decoder.process(&mut chunk, None) {
                Ok(_) => {}
                Err(e) => panic!("decoder errored on chunk [{fed}..{end}]: {e:?}"),
            }
            let consumed = before - chunk.len();
            fed += consumed;
            if consumed == 0 {
                // No progress on this chunk — advance to feed more bytes.
                fed = end;
            }
        }
    }
}
