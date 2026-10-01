// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Incremental (Sans-I/O) JPEG XL decoder.

use std::sync::Arc;
use whereat::at;

use enough::Stop;

#[cfg(feature = "zencodec")]
use zencodec::{
    CodecError, ImageInfo, Orientation,
    decode::{IncrementalDecode, PushOutcome, PushStatus},
};
#[cfg(feature = "zencodec")]
use zenpixels::{PixelDescriptor, PixelSlice};

use crate::{
    api::{
        ExtraChannel, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
        JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
    },
    error::{Error, Result},
};

type At<E> = whereat::At<E>;

enum DecoderStage {
    Initialized(JxlDecoder<states::Initialized>),
    WithImageInfo(JxlDecoder<states::WithImageInfo>),
    WithFrameInfo(JxlDecoder<states::WithFrameInfo>),
    FrameComplete(JxlDecoder<states::WithImageInfo>),
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageId {
    Initialized,
    WithImageInfo,
    WithFrameInfo,
    FrameComplete,
    Complete,
}

struct ChosenFormat {
    color_type: JxlColorType,
    channels: usize,
    #[cfg(feature = "zencodec")]
    descriptor: PixelDescriptor,
}

/// Incremental (Sans-I/O) JPEG XL decoder.
///
/// Accepts compressed input bytes in arbitrary chunks (down to 1 byte) without
/// performing any I/O, and yields scanlines as soon as the image frame has
/// completed decoding.
pub struct JxlIncrementalDecoder {
    adjust_orientation: bool,
    stop: Option<Arc<dyn Stop + Send + Sync>>,
    #[cfg(feature = "zencodec")]
    preferred: Vec<PixelDescriptor>,
    #[cfg(feature = "zencodec")]
    descriptor: PixelDescriptor,
    #[cfg(feature = "zencodec")]
    info: Option<ImageInfo>,

    stage: Option<DecoderStage>,
    width: u32,
    height: u32,
    channels: usize,
    has_alpha: bool,
    pixel_data: Vec<u8>,
    current_row: u32,
    batch_rows: u32,

    output_ready: bool,
    is_complete: bool,
}

impl JxlIncrementalDecoder {
    /// Create a new incremental decoder with default options.
    pub fn new(options: JxlDecoderOptions) -> Self {
        let adjust_orientation = options.adjust_orientation;
        Self {
            adjust_orientation,
            stop: None,
            #[cfg(feature = "zencodec")]
            preferred: Vec::new(),
            #[cfg(feature = "zencodec")]
            descriptor: PixelDescriptor::RGBA8_SRGB,
            #[cfg(feature = "zencodec")]
            info: None,
            stage: Some(DecoderStage::Initialized(JxlDecoder::new(options))),
            width: 0,
            height: 0,
            channels: 4,
            has_alpha: false,
            pixel_data: Vec::new(),
            current_row: 0,
            batch_rows: 1,
            output_ready: false,
            is_complete: false,
        }
    }

    /// Set row batch size for `pull_batch` (defaults to 1 row).
    pub fn with_batch_rows(mut self, batch_rows: u32) -> Self {
        self.batch_rows = batch_rows.max(1);
        self
    }

    /// Set an optional cooperative cancellation token.
    pub fn with_stop(mut self, stop: Arc<dyn Stop + Send + Sync>) -> Self {
        self.stop = Some(stop);
        self
    }

    #[cfg(feature = "zencodec")]
    /// Create a new incremental decoder with preferred pixel descriptors.
    pub fn with_preferred(options: JxlDecoderOptions, preferred: &[PixelDescriptor]) -> Self {
        let mut dec = Self::new(options);
        dec.preferred = preferred.to_vec();
        dec
    }

    fn stage_id(&self) -> StageId {
        match self.stage.as_ref() {
            Some(DecoderStage::Initialized(_)) => StageId::Initialized,
            Some(DecoderStage::WithImageInfo(_)) => StageId::WithImageInfo,
            Some(DecoderStage::WithFrameInfo(_)) => StageId::WithFrameInfo,
            Some(DecoderStage::FrameComplete(_)) => StageId::FrameComplete,
            Some(DecoderStage::Complete) | None => StageId::Complete,
        }
    }

    fn choose_format(&self, is_grayscale: bool, has_alpha: bool) -> ChosenFormat {
        #[cfg(feature = "zencodec")]
        {
            // First pass: look for exact color match matching image's color space
            for desc in &self.preferred {
                if *desc == PixelDescriptor::RGBA8_SRGB && !is_grayscale {
                    return ChosenFormat {
                        color_type: JxlColorType::Rgba,
                        channels: 4,
                        descriptor: PixelDescriptor::RGBA8_SRGB,
                    };
                }
                if *desc == PixelDescriptor::RGB8_SRGB && !is_grayscale && !has_alpha {
                    return ChosenFormat {
                        color_type: JxlColorType::Rgb,
                        channels: 3,
                        descriptor: PixelDescriptor::RGB8_SRGB,
                    };
                }
                if *desc == PixelDescriptor::GRAY8_SRGB && is_grayscale && !has_alpha {
                    return ChosenFormat {
                        color_type: JxlColorType::Grayscale,
                        channels: 1,
                        descriptor: PixelDescriptor::GRAY8_SRGB,
                    };
                }
                if *desc == PixelDescriptor::GRAYA8_SRGB && is_grayscale {
                    return ChosenFormat {
                        color_type: JxlColorType::GrayscaleAlpha,
                        channels: 2,
                        descriptor: PixelDescriptor::GRAYA8_SRGB,
                    };
                }
            }

            // Second pass: cross-domain promotion (e.g. grayscale -> RGBA)
            for desc in &self.preferred {
                if *desc == PixelDescriptor::RGBA8_SRGB {
                    return ChosenFormat {
                        color_type: JxlColorType::Rgba,
                        channels: 4,
                        descriptor: PixelDescriptor::RGBA8_SRGB,
                    };
                }
                if *desc == PixelDescriptor::RGB8_SRGB && !has_alpha {
                    return ChosenFormat {
                        color_type: JxlColorType::Rgb,
                        channels: 3,
                        descriptor: PixelDescriptor::RGB8_SRGB,
                    };
                }
            }
        }

        if is_grayscale {
            if has_alpha {
                ChosenFormat {
                    color_type: JxlColorType::GrayscaleAlpha,
                    channels: 2,
                    #[cfg(feature = "zencodec")]
                    descriptor: PixelDescriptor::GRAYA8_SRGB,
                }
            } else {
                ChosenFormat {
                    color_type: JxlColorType::Grayscale,
                    channels: 1,
                    #[cfg(feature = "zencodec")]
                    descriptor: PixelDescriptor::GRAY8_SRGB,
                }
            }
        } else if has_alpha {
            ChosenFormat {
                color_type: JxlColorType::Rgba,
                channels: 4,
                #[cfg(feature = "zencodec")]
                descriptor: PixelDescriptor::RGBA8_SRGB,
            }
        } else {
            ChosenFormat {
                color_type: JxlColorType::Rgb,
                channels: 3,
                #[cfg(feature = "zencodec")]
                descriptor: PixelDescriptor::RGB8_SRGB,
            }
        }
    }

    /// Push a chunk of compressed bytes into the decoder.
    pub fn push_chunk_inner(
        &mut self,
        chunk: &[u8],
        is_eof: bool,
        stop: Option<&dyn Stop>,
    ) -> Result<usize> {
        if let Some(stop) = stop {
            stop.check().map_err(|e| at!(Error::from(e)))?;
        }
        if let Some(ref stop) = self.stop {
            let cancel: &dyn Stop = &**stop;
            cancel.check().map_err(|e| at!(Error::from(e)))?;
        }

        if self.is_complete {
            return Ok(0);
        }

        let mut remaining = chunk;

        loop {
            let before_len = remaining.len();
            let stage_before = self.stage_id();

            match self.stage.take().unwrap() {
                DecoderStage::Initialized(decoder) => {
                    match decoder.process(&mut remaining)? {
                        ProcessingResult::Complete { result } => {
                            let basic_info = result.basic_info().clone();
                            let (w, h) = basic_info.size;
                            self.width = w as u32;
                            self.height = h as u32;

                            let is_grayscale = result.current_pixel_format().color_type.is_grayscale();
                            let has_alpha = basic_info
                                .extra_channels
                                .iter()
                                .any(|ec| ec.ec_type == ExtraChannel::Alpha);
                            self.has_alpha = has_alpha;

                            let chosen = self.choose_format(is_grayscale, has_alpha);
                            self.channels = chosen.channels;
                            #[cfg(feature = "zencodec")]
                            {
                                self.descriptor = chosen.descriptor;
                            }

                            let main_alpha = basic_info
                                .extra_channels
                                .iter()
                                .position(|ec| ec.ec_type == ExtraChannel::Alpha);

                            let u8_format = JxlDataFormat::U8 { bit_depth: 8 };
                            let pixel_format = JxlPixelFormat {
                                color_type: chosen.color_type,
                                color_data_format: Some(u8_format),
                                extra_channel_format: basic_info
                                    .extra_channels
                                    .iter()
                                    .enumerate()
                                    .map(|(i, _)| {
                                        if Some(i) == main_alpha {
                                            None
                                        } else {
                                            Some(u8_format)
                                        }
                                    })
                                    .collect(),
                            };

                            let mut result = result;
                            result.set_pixel_format(pixel_format);

                            #[cfg(feature = "zencodec")]
                            {
                                let orientation = if self.adjust_orientation {
                                    Orientation::Identity
                                } else {
                                    Orientation::from_exif(basic_info.orientation as u8)
                                        .unwrap_or(Orientation::Identity)
                                };
                                let mut info = ImageInfo::new(self.width, self.height, zencodec::ImageFormat::Jxl)
                                    .with_alpha(has_alpha)
                                    .with_orientation(orientation);
                                if let Some(icc) = result.embedded_color_profile().try_as_icc() {
                                    info = info.with_icc_profile(icc.into_owned());
                                }
                                self.info = Some(info);
                            }

                            self.stage = Some(DecoderStage::WithImageInfo(result));
                        }
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            self.stage = Some(DecoderStage::Initialized(fallback));
                        }
                    }
                }
                DecoderStage::WithImageInfo(decoder) => {
                    match decoder.process(&mut remaining)? {
                        ProcessingResult::Complete { result } => {
                            let total_bytes = (self.width as usize)
                                .checked_mul(self.channels)
                                .and_then(|row| row.checked_mul(self.height as usize))
                                .ok_or_else(|| at!(Error::ArithmeticOverflow))?;

                            if self.pixel_data.len() != total_bytes {
                                self.pixel_data = vec![0u8; total_bytes];
                            }
                            self.stage = Some(DecoderStage::WithFrameInfo(result));
                        }
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            self.stage = Some(DecoderStage::WithImageInfo(fallback));
                        }
                    }
                }
                DecoderStage::WithFrameInfo(decoder) => {
                    let row_bytes = self.width as usize * self.channels;
                    let out_buf = JxlOutputBuffer::new(
                        &mut self.pixel_data,
                        self.height as usize,
                        row_bytes,
                    );
                    let mut bufs = [out_buf];
                    match decoder.process(&mut remaining, &mut bufs)? {
                        ProcessingResult::Complete { result } => {
                            self.output_ready = true;
                            self.stage = Some(DecoderStage::FrameComplete(result));
                        }
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            self.stage = Some(DecoderStage::WithFrameInfo(fallback));
                        }
                    }
                }
                DecoderStage::FrameComplete(decoder) => {
                    if !decoder.has_more_frames() && !remaining.is_empty() {
                        let decoder = decoder;
                        match decoder.process(&mut remaining)? {
                            ProcessingResult::Complete { .. } => {
                                self.stage = Some(DecoderStage::Complete);
                            }
                            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                                self.stage = Some(DecoderStage::FrameComplete(fallback));
                            }
                        }
                    } else {
                        self.stage = Some(DecoderStage::FrameComplete(decoder));
                        remaining = &[];
                        break;
                    }
                }
                DecoderStage::Complete => {
                    self.stage = Some(DecoderStage::Complete);
                    break;
                }
            }

            let consumed_this_step = before_len - remaining.len();
            if consumed_this_step == 0 && self.stage_id() == stage_before {
                break;
            }
            if remaining.is_empty() && self.stage_id() == stage_before {
                break;
            }
        }

        if is_eof && !self.output_ready {
            return Err(at!(Error::OutOfBounds(0)));
        }

        Ok(chunk.len() - remaining.len())
    }

    /// Whether pixel output is ready to be pulled via `pull_batch`.
    pub fn is_output_ready(&self) -> bool {
        self.output_ready
    }

    /// Whether decoding has fully completed.
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Width of the image in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the image in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Number of color channels in output.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Complete decoded pixel slice (available once output is ready).
    pub fn pixels(&self) -> Option<&[u8]> {
        if self.output_ready {
            Some(&self.pixel_data)
        } else {
            None
        }
    }
}

#[cfg(feature = "zencodec")]
impl IncrementalDecode for JxlIncrementalDecoder {
    type Error = At<CodecError>;

    fn push_chunk(
        &mut self,
        chunk: &[u8],
        is_eof: bool,
        stop: Option<&dyn Stop>,
    ) -> std::result::Result<PushOutcome, Self::Error> {
        let consumed = self.push_chunk_inner(chunk, is_eof, stop).map_err(CodecError::of)?;

        let status = if self.is_complete {
            PushStatus::Complete
        } else if self.output_ready {
            PushStatus::OutputReady
        } else {
            PushStatus::NeedMoreInput
        };

        Ok(PushOutcome::new(consumed, status))
    }

    fn pull_batch(&mut self) -> std::result::Result<Option<(u32, PixelSlice<'_>)>, Self::Error> {
        if let Some(ref stop) = self.stop {
            let cancel: &dyn Stop = &**stop;
            cancel.check().map_err(|e| at!(Error::from(e))).map_err(CodecError::of)?;
        }

        if !self.output_ready || self.is_complete {
            return Ok(None);
        }

        if self.current_row >= self.height {
            self.is_complete = true;
            return Ok(None);
        }

        let y = self.current_row;
        let rows_to_pull = self.batch_rows.min(self.height - self.current_row);
        let row_stride = self.width as usize * self.channels;
        let row_start = y as usize * row_stride;
        let row_len = rows_to_pull as usize * row_stride;
        let row_bytes = &self.pixel_data[row_start..row_start + row_len];

        self.current_row += rows_to_pull;
        if self.current_row >= self.height {
            self.is_complete = true;
        }

        let slice = PixelSlice::new(
            row_bytes,
            self.width,
            rows_to_pull,
            row_stride,
            self.descriptor,
        )
        .map_err(|e| at!(Error::from(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))))
        .map_err(CodecError::of)?;

        Ok(Some((y, slice)))
    }

    fn info(&self) -> Option<&ImageInfo> {
        self.info.as_ref()
    }

    fn is_complete(&self) -> bool {
        self.is_complete
    }
}
