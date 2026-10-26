// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    io::IoSliceMut,
    ops::{Deref, Range},
};
use whereat::at;

use crate::error::Result;

use crate::api::{JxlBitstreamInput, JxlDecoderInner, JxlOutputBuffer, ProcessingResult};

// General implementation strategy:
// - Anything that is not a section is read into a small buffer.
// - As soon as we know section sizes, data is read directly into sections.
// When the start of the populated range in `buf` goes past half of its length,
// the data in the buffer is moved back to the beginning.

pub(super) struct SmallBuffer {
    buf: Vec<u8>,
    range: Range<usize>,
}

impl SmallBuffer {
    pub(super) fn refill(
        &mut self,
        mut get_input: impl FnMut(&mut [IoSliceMut]) -> Result<usize, std::io::Error>,
        max: Option<usize>,
    ) -> Result<usize> {
        let mut total = 0;
        loop {
            if self.range.start >= self.buf.len() / 2 {
                let start = self.range.start;
                // `copy_within` handles a live range longer than the gap in
                // front of it, which an out-of-order `jxlp` splice
                // (`inject_bytes_front` sizes the buffer to the data) creates;
                // splitting at `start` panicked there.
                self.buf.copy_within(self.range.clone(), 0);
                self.range.start -= start;
                self.range.end -= start;
            }
            if self.range.len() >= self.buf.len() / 2 {
                break;
            }
            let stop = if let Some(max) = max {
                self.range
                    .end
                    .saturating_add(max.saturating_sub(total))
                    .min(self.buf.len())
            } else {
                self.buf.len()
            };
            let num = get_input(&mut [IoSliceMut::new(&mut self.buf[self.range.end..stop])])
                .map_err(|e| at!(crate::error::Error::from(e)))?;
            total += num;
            self.range.end += num;
            if num == 0 {
                break;
            }
        }
        Ok(total)
    }

    pub(super) fn take(&mut self, mut buffers: &mut [IoSliceMut]) -> usize {
        let mut num = 0;
        while !self.range.is_empty() {
            let Some((buf, rest)) = buffers.split_first_mut() else {
                break;
            };
            buffers = rest;
            let len = self.range.len().min(buf.len());
            // Only copy 'len' bytes, not the entire range, to avoid panic when buf is smaller than range
            buf[..len].copy_from_slice(&self.buf[self.range.start..self.range.start + len]);
            self.range.start += len;
            num += len;
        }
        num
    }

    pub(super) fn consume(&mut self, amount: usize) -> usize {
        let amount = amount.min(self.range.len());
        self.range.start += amount;
        amount
    }

    pub(super) fn new(initial_size: usize) -> Self {
        Self {
            buf: vec![0; initial_size],
            range: 0..0,
        }
    }

    pub(super) fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Prepend bytes so they are returned next by [`Self::take`] (and come
    /// first in the `Deref` view). Used to splice a buffered out-of-order
    /// `jxlp` payload in front of the container input.
    ///
    /// The backing buffer never shrinks here. If it were replaced by just the
    /// spliced bytes, a payload shorter than the next box header would leave
    /// a buffer that [`Self::refill`] can never fill up to a whole header, and
    /// the box parser would ask for more input forever without consuming any.
    pub(super) fn inject_bytes_front(&mut self, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }
        let capacity = self.buf.len();
        let mut combined = data;
        combined.extend_from_slice(&self.buf[self.range.clone()]);
        let live = combined.len();
        if live < capacity {
            combined.resize(capacity, 0);
        }
        self.buf = combined;
        self.range = 0..live;
    }

    pub(super) fn enlarge(&mut self) {
        // Note: we need a *4 here because doubling the buffer size might still not allow refill() to make progress.
        self.buf.resize(self.buf.len() * 4, 0);
    }

    pub(super) fn can_read_more(&self) -> bool {
        self.buf.len() > self.len() * 2 && self.range.end < self.buf.len()
    }
}

impl Deref for SmallBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.buf[self.range.clone()]
    }
}

impl JxlDecoderInner {
    /// Process more of the input file.
    /// This function will return when reaching the next decoding stage (i.e. finished decoding
    /// file/frame header, or finished decoding a frame).
    /// If called when decoding a frame with `None` for buffers, the frame will still be read,
    /// but pixel data will not be produced.
    #[inline(never)]
    pub fn process(
        &mut self,
        input: &mut dyn JxlBitstreamInput,
        buffers: Option<&mut [JxlOutputBuffer]>,
    ) -> Result<ProcessingResult<(), ()>> {
        if !self.options.coalescing
            && (self.options.adjust_orientation
                || !self.options.skip_preview
                || self.options.premultiply_output
                || self.options.desired_intensity_target.is_some()
                || self.options.cms.is_some())
        {
            return Err(whereat::at!(
                crate::error::Error::UncoalescedOutputUnsupported
            ));
        }
        self.validate_raw_output()?;
        let result = self.codestream_parser.process(
            &mut self.box_parser,
            input,
            &self.options,
            buffers,
            false,
        );
        // A complete last codestream box with frames still outstanding is a
        // damaged file, not a stream still arriving. Upstream jxl-rs 97e233d.
        let result = match result {
            Err(e)
                if matches!(e.error(), crate::error::Error::OutOfBounds(_))
                    && self.codestream_ended()
                    && !self.options.recover_partial_image =>
            {
                Err(whereat::at!(
                    crate::error::Error::UnexpectedCodestreamBoxEnd
                ))
            }
            other => other,
        };
        ProcessingResult::new(result)
    }

    // Validate on every processing entry, including low-level format changes
    // after a header. No caller buffer is touched by an unsupported request.
    fn validate_raw_output(&self) -> Result<()> {
        use crate::api::{JxlColorType, JxlDataFormat};
        use crate::error::Error;
        use crate::headers::extra_channels::ExtraChannel;
        if self.options.coalescing || self.options.scan_frames_only {
            return Ok(());
        }
        let Some(info) = self.basic_info() else {
            return Ok(());
        };
        let format = self.current_pixel_format().unwrap();
        let reject = |requirement| whereat::at!(Error::UncoalescedOutputRequirement(requirement));
        if format.extra_channel_format.len() != info.extra_channels.len() {
            return Err(reject("one format entry per source extra channel"));
        }
        if std::iter::once(&format.color_data_format)
            .chain(format.extra_channel_format.iter())
            .flatten()
            .any(|f| !matches!(f, JxlDataFormat::F32 { .. }))
        {
            return Err(reject("F32 output samples"));
        }
        if self.options.render_spot_colors
            && info
                .extra_channels
                .iter()
                .any(|e| e.ec_type == ExtraChannel::SpotColor)
        {
            return Err(reject("separate, unrendered spot channels"));
        }
        if format.color_data_format.is_some()
            && format.color_type != JxlColorType::Cmyk
            && info
                .extra_channels
                .iter()
                .enumerate()
                .any(|(index, extra)| {
                    extra.ec_type == ExtraChannel::Black
                        && format.extra_channel_format[index].is_none()
                })
        {
            return Err(reject("native CMYK components with the black channel"));
        }
        if self.codestream_parser.output_color_profile_set_by_user
            && self.output_color_profile() != self.embedded_color_profile()
        {
            return Err(reject("the source color profile"));
        }
        Ok(())
    }

    /// See [`JxlDecoder::codestream_ended`](crate::api::JxlDecoder::codestream_ended).
    pub(crate) fn codestream_ended(&self) -> bool {
        self.codestream_parser.has_more_frames && self.box_parser.codestream_ended()
    }

    /// Draws all the pixels we have data for. Returns `true` if any new pixels
    /// were written to `buffers` since the previous call to `flush_pixels`;
    /// returns `false` if no new rendering has happened, in which case the
    /// contents of `buffers` are unchanged from the caller's perspective.
    pub fn flush_pixels(&mut self, buffers: &mut [JxlOutputBuffer]) -> Result<bool> {
        if !self.options.coalescing {
            return Err(whereat::at!(
                crate::error::Error::UncoalescedOutputRequirement(
                    "completed layers; progressive raw-layer flush is not implemented"
                )
            ));
        }
        if self.codestream_parser.frame.is_none() {
            // No frame is being decoded yet (or the previous frame completed):
            // nothing can possibly be rendered. Returning early also keeps the
            // empty-input flush from running the header state machine, which
            // records header_needed_bytes/staging state for input that is not
            // there and derails the next real process() call.
            return Ok(false);
        }
        let mut input: &[u8] = &[];
        match self.codestream_parser.process(
            &mut self.box_parser,
            &mut input,
            &self.options,
            Some(buffers),
            true,
        ) {
            Ok(()) => Ok(self.codestream_parser.get_and_clear_pixels_dirty()),
            Err(e) if matches!(e.error(), crate::error::Error::OutOfBounds(_)) => {
                Ok(self.codestream_parser.get_and_clear_pixels_dirty())
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod small_buffer_tests {
    use super::SmallBuffer;

    /// A spliced buffer sized exactly to its data, consumed just past half,
    /// then refilled: the live range is longer than the gap in front of it.
    #[test]
    fn refill_after_splice_compacts_without_panicking() {
        let mut b = SmallBuffer::new(4);
        b.inject_bytes_front((0u8..27).collect());
        assert_eq!(b.consume(13), 13);
        let mut src: &[u8] = &[100, 101];
        b.refill(|bufs| std::io::Read::read_vectored(&mut src, bufs), None)
            .unwrap();
        assert_eq!(&b[..14], &(13u8..27).collect::<Vec<_>>()[..]);
    }
}
