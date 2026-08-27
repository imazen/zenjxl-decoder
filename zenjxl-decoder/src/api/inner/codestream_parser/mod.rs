// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    io::IoSliceMut,
};
use whereat::at;

use sections::SectionState;

#[cfg(test)]
use crate::api::FrameCallback;
use crate::{
    api::{
        JxlBasicInfo, JxlBitstreamInput, JxlColorEncoding, JxlColorProfile, JxlDataFormat,
        JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, VisibleFrameInfo,
        VisibleFrameSeekTarget,
        inner::{
            box_parser::{BoxParser, CodestreamBoxType},
            process::SmallBuffer,
        },
    },
    error::{Error, Result},
    frame::{DecoderState, Frame, Section},
    headers::{Animation, FileHeader, frame_header::FrameHeader, toc::IncrementalTocReader},
    icc::IncrementalIccReader,
};

mod non_section;
mod sections;

struct SectionBuffer {
    len: usize,
    data: Vec<u8>,
    section: Section,
}

/// Everything needed to restart decoding at the start of one frame's header:
/// where the header sits in the file, the container box state at that
/// position, and the per-decoder counters a sequential decode would have had
/// there (the noise RNG is seeded from the frame counters, so a seek must
/// reproduce them exactly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SeekPoint {
    /// Byte offset of the frame header in the input file. For a container
    /// whose `jxlp` boxes are out of order this is only the count of
    /// codestream bytes consumed before the frame (a stable, monotonic key
    /// for dependency tracking), see `seekable`.
    pub(crate) file_offset: u64,
    /// Codestream bytes left in the containing box at `file_offset`
    /// (`u64::MAX` for a bare codestream or an unbounded box).
    pub(crate) remaining_in_box: u64,
    /// Box parser bookkeeping at `file_offset`.
    pub(crate) box_type: CodestreamBoxType,
    /// Visible frames parsed before this frame (= the index of the next
    /// visible frame).
    pub(crate) visible_count_before: usize,
    /// `DecoderState::visible_frame_index` before this frame was parsed.
    pub(crate) visible_counter: usize,
    /// `DecoderState::nonvisible_frame_index` before this frame was parsed.
    pub(crate) nonvisible_counter: usize,
    /// `false` when `file_offset` is not a real file position (out-of-order
    /// `jxlp` boxes were spliced in before this frame), so it cannot be used
    /// as a seek target.
    pub(crate) seekable: bool,
}

/// Per-frame record kept for every non-preview frame parsed so far, keyed by
/// `SeekPoint::file_offset`. Never truncated: after a seek the same frames
/// re-record identical entries, and a seek target's dependency slots may
/// point at frames recorded during an earlier pass.
#[derive(Debug, Clone, Copy)]
struct FrameStartInfo {
    point: SeekPoint,
    /// Reference-slot dependency origins *before* this frame's own save, as
    /// `file_offset` keys into the same map.
    reference_slots: [Option<u64>; DecoderState::MAX_STORED_FRAMES],
    /// LF-slot dependency origins before this frame, likewise.
    lf_slots: [Option<u64>; DecoderState::NUM_LF_FRAMES],
}

/// Padding bytes appended to each section buffer so BitReader::refill()
/// can always take the fast 8-byte read path.
const SECTION_PADDING: usize = 8;

/// Grow `buf.data` so that bytes `[..readable]` can be received, without
/// ever allocating the TOC-declared length up front. Reaching `readable ==
/// buf.len` means the section will be complete, so the buffer takes its final
/// padded size.
fn grow_section_buffer(buf: &mut SectionBuffer, readable: usize) -> Result<()> {
    debug_assert!(readable <= buf.len);
    let target = if readable == buf.len {
        buf.len + SECTION_PADDING
    } else {
        readable
    };
    if buf.data.len() < target {
        let extra = target - buf.data.len();
        if target == buf.len + SECTION_PADDING {
            buf.data
                .try_reserve_exact(extra)
                .map_err(|e| at!(Error::from(e)))?;
        } else {
            buf.data
                .try_reserve(extra)
                .map_err(|e| at!(Error::from(e)))?;
        }
        buf.data.resize(target, 0);
    }
    Ok(())
}

pub(super) struct CodestreamParser {
    // TODO(veluca): this would probably be cleaner with some kind of state enum.
    pub(super) file_header: Option<FileHeader>,
    icc_parser: Option<IncrementalIccReader>,
    // These fields are populated once image information is available.
    pub(super) decoder_state: Option<DecoderState>,
    pub(super) basic_info: Option<JxlBasicInfo>,
    pub(super) animation: Option<Animation>,
    pub(super) embedded_color_profile: Option<JxlColorProfile>,
    pub(super) output_color_profile: Option<JxlColorProfile>,
    pub(super) pixel_format: Option<JxlPixelFormat>,
    xyb_encoded: bool,
    is_gray: bool,
    pub(super) output_color_profile_set_by_user: bool,

    // These fields are populated when starting to decode a frame, and cleared once
    // the frame is done.
    frame_header: Option<FrameHeader>,
    toc_parser: Option<IncrementalTocReader>,
    pub(super) frame: Option<Frame>,

    // Buffers.
    non_section_buf: SmallBuffer,
    non_section_bit_offset: u8,
    sections: VecDeque<SectionBuffer>,
    ready_section_data: usize,
    skip_sections: bool,
    // True when we need to process frames without copying them to output buffers, e.g. reference frames
    process_without_output: bool,
    // True once the preview frame has been processed (if there is one)
    preview_done: bool,
    // Saved file header for recreating decoder state after preview frame
    saved_file_header: Option<crate::headers::FileHeader>,
    /// The parsed image header, kept for the decoder's lifetime so a seek can
    /// build a fresh `DecoderState` even after the last frame consumed the
    /// live one.
    image_file_header: Option<FileHeader>,

    // --- Frame scanning / seeking (upstream jxl-rs #678) ---
    /// Visible frames discovered so far, sorted by `index`. Contiguous for a
    /// sequential pass; may have gaps after a seek jumped over frames that
    /// were never parsed.
    pub(super) scanned_frames: Vec<VisibleFrameInfo>,
    /// Index the next visible frame will get.
    visible_frame_index: usize,
    /// All non-preview frame starts parsed so far, keyed by file offset.
    frame_starts: BTreeMap<u64, FrameStartInfo>,
    /// For each reference slot, the file offset of the earliest frame that
    /// must be decoded to reconstruct the slot's current contents.
    reference_slot_decode_start: [Option<u64>; DecoderState::MAX_STORED_FRAMES],
    /// Same for the LF-frame slots.
    lf_slot_decode_start: [Option<u64>; DecoderState::NUM_LF_FRAMES],
    /// Seek point captured for the frame header currently being parsed.
    current_frame_start: Option<SeekPoint>,
    /// Visible frames still to be passed over internally after a seek before
    /// the requested frame is handed to the caller (upstream jxl-rs #702).
    pending_visible_skips: usize,
    /// The current frame is a visible frame being passed over after a seek:
    /// its sections may be skipped like `skip_frame` does.
    auto_skipping_visible: bool,

    section_state: SectionState,

    // Or only section if in single section special case.
    lf_global_section: Option<SectionBuffer>,
    lf_sections: Vec<SectionBuffer>,
    hf_global_section: Option<SectionBuffer>,
    // indexed by group, then by pass.
    hf_sections: Vec<Vec<Option<SectionBuffer>>>,
    // group indices that *might* have new renderable data.
    candidate_hf_sections: HashSet<usize>,

    /// Total length of the file, in bytes, once known: set when the last
    /// frame finishes decoding, as the count of input bytes actually consumed
    /// by the file (excluding bytes over-read into internal buffers). `None`
    /// until decoding is finished.
    pub(super) file_length: Option<u64>,

    /// Set when `decode_and_render_hf_groups` / `maybe_preview_lf_frame`
    /// actually write to the output buffers. Read and cleared by
    /// `flush_pixels` via [`Self::get_and_clear_pixels_dirty`].
    pixels_dirty: bool,

    pub(super) has_more_frames: bool,

    /// `(global_scale, quant_lf)` of the first regular VarDCT frame, captured
    /// when its `LfGlobal` is decoded and kept for the decoder's lifetime so a
    /// probe can recover the main image's encode quality. `None` for Modular
    /// (lossless) files or before the first regular frame is decoded.
    pub(super) first_vardct_quantizer: Option<(u32, u32)>,

    header_needed_bytes: Option<u64>,

    #[cfg(test)]
    pub frame_callback: Option<Box<FrameCallback>>,
    #[cfg(test)]
    pub decoded_frames: usize,

    /// JBRD box data for JPEG reconstruction.
    #[cfg(feature = "jpeg")]
    pub(super) jbrd_data: Option<Vec<u8>>,
    /// Reconstruction JpegData (built after the frame decodes if jbrd present).
    /// EXIF/XMP APPn placeholders are filled, and the bytes written, lazily in
    /// `take_jpeg_reconstruction` — after the trailing container boxes parse.
    #[cfg(feature = "jpeg")]
    pub(super) jpeg_recon: Option<crate::jpeg::data::JpegData>,
}

impl CodestreamParser {
    /// Once the last frame has finished decoding, computes the number of
    /// input bytes the file actually used: everything pulled from the input so
    /// far minus what still sits unconsumed in internal buffers (box buffer,
    /// buffered out-of-order `jxlp` payloads, the non-section buffer, and
    /// ready-but-unparsed section bytes).
    fn record_file_length(&mut self, box_parser: &BoxParser) {
        if !self.has_more_frames && self.file_length.is_none() {
            self.file_length = Some(
                box_parser
                    .total_file_read
                    .saturating_sub(box_parser.buffered_leftover())
                    .saturating_sub(self.non_section_buf.len() as u64)
                    .saturating_sub(self.ready_section_data as u64),
            );
        }
    }

    /// Returns whether any pixels were rendered since the last call, and
    /// clears the flag.
    pub(super) fn get_and_clear_pixels_dirty(&mut self) -> bool {
        let r = self.pixels_dirty;
        self.pixels_dirty = false;
        r
    }

    pub(super) fn new() -> Self {
        Self {
            file_header: None,
            icc_parser: None,
            decoder_state: None,
            basic_info: None,
            animation: None,
            embedded_color_profile: None,
            output_color_profile: None,
            pixel_format: None,
            xyb_encoded: false,
            is_gray: false,
            output_color_profile_set_by_user: false,
            frame_header: None,
            toc_parser: None,
            frame: None,
            first_vardct_quantizer: None,
            non_section_buf: SmallBuffer::new(4096),
            non_section_bit_offset: 0,
            sections: VecDeque::new(),
            ready_section_data: 0,
            skip_sections: false,
            process_without_output: false,
            preview_done: false,
            saved_file_header: None,
            image_file_header: None,
            scanned_frames: Vec::new(),
            visible_frame_index: 0,
            frame_starts: BTreeMap::new(),
            reference_slot_decode_start: [None; DecoderState::MAX_STORED_FRAMES],
            lf_slot_decode_start: [None; DecoderState::NUM_LF_FRAMES],
            current_frame_start: None,
            pending_visible_skips: 0,
            auto_skipping_visible: false,
            section_state: SectionState::new(0, 0),
            lf_global_section: None,
            lf_sections: vec![],
            hf_global_section: None,
            hf_sections: vec![],
            candidate_hf_sections: HashSet::new(),
            file_length: None,
            pixels_dirty: false,
            has_more_frames: true,
            header_needed_bytes: None,
            #[cfg(test)]
            frame_callback: None,
            #[cfg(test)]
            decoded_frames: 0,
            #[cfg(feature = "jpeg")]
            jbrd_data: None,
            #[cfg(feature = "jpeg")]
            jpeg_recon: None,
        }
    }

    fn has_visible_frame(&self) -> bool {
        if let Some(frame) = &self.frame {
            frame.header().is_visible()
        } else {
            false
        }
    }

    /// Returns the number of passes that are fully completed across all groups.
    pub(super) fn num_completed_passes(&self) -> usize {
        self.section_state.num_completed_passes()
    }

    #[cfg(test)]
    pub(crate) fn set_use_simple_pipeline(&mut self, u: bool) {
        self.decoder_state
            .as_mut()
            .unwrap()
            .set_use_simple_pipeline(u);
    }

    /// Rewinds for animation loop replay, keeping pixel_format setting.
    pub(super) fn rewind(&mut self) -> Option<JxlPixelFormat> {
        let pixel_format = self.pixel_format.take();
        *self = Self::new();
        self.pixel_format = pixel_format.clone();
        pixel_format
    }

    /// Captures the seek point for the frame header that is about to be
    /// parsed, if the box parser can already place it. The frame starts at
    /// codestream position "bytes handed to this parser" minus the bytes
    /// still unparsed in `non_section_buf` (the previous frame's over-read
    /// tail); the box parser maps that through the boxes it has entered, so
    /// it does not matter whether it has already read ahead into the next
    /// box while the tail of the previous one is still buffered here.
    /// Returns `false` when the frame starts exactly at the end of the last
    /// box the parser has entered (call again after `get_more_codestream`).
    fn try_capture_frame_start(&mut self, box_parser: &BoxParser) -> bool {
        let pos = box_parser
            .codestream_consumed
            .saturating_sub(self.non_section_buf.len() as u64);
        let Some((file_offset, remaining_in_box, box_type)) = box_parser.locate_codestream_pos(pos)
        else {
            return false;
        };
        let state = self
            .decoder_state
            .as_ref()
            .expect("frame start is captured only once image info exists");
        let seekable = !box_parser.reordered_jxlp;
        self.current_frame_start = Some(SeekPoint {
            // Without a real file layout the codestream position still
            // works as the monotonic dependency-tracking key.
            file_offset: if seekable { file_offset } else { pos },
            remaining_in_box,
            box_type,
            visible_count_before: self.visible_frame_index,
            visible_counter: state.visible_frame_index,
            nonvisible_counter: state.nonvisible_frame_index,
            seekable,
        });
        true
    }

    /// Records the frame that was just created from its header and TOC:
    /// remembers its seek point, resolves which earlier frame a seek must
    /// start from (through the reference / LF slots it reads), and appends a
    /// [`VisibleFrameInfo`] if the frame is visible. Preview frames are not
    /// recorded.
    fn record_frame_info(&mut self) {
        let Some(frame) = self.frame.as_ref() else {
            return;
        };
        let Some(start) = self.current_frame_start.take() else {
            return;
        };
        let header = frame.header();

        // Dependencies: blending reads exactly the slots named in the
        // header; patches may read any slot, so assume all of them.
        let mut used_reference_slots = [false; DecoderState::MAX_STORED_FRAMES];
        if header.needs_blending() {
            for info in header
                .ec_blending_info
                .iter()
                .chain(std::iter::once(&header.blending_info))
            {
                if let Some(slot) = used_reference_slots.get_mut(info.source as usize) {
                    *slot = true;
                }
            }
        }
        if header.has_patches() {
            used_reference_slots.fill(true);
        }
        let mut decode_start_offset = start.file_offset;
        for (slot, used) in used_reference_slots.iter().enumerate() {
            if *used && let Some(dep) = self.reference_slot_decode_start[slot] {
                decode_start_offset = decode_start_offset.min(dep);
            }
        }
        if header.has_lf_frame()
            && let Some(Some(dep)) = self.lf_slot_decode_start.get(header.lf_level as usize)
        {
            decode_start_offset = decode_start_offset.min(*dep);
        }

        let this = FrameStartInfo {
            point: start,
            reference_slots: self.reference_slot_decode_start,
            lf_slots: self.lf_slot_decode_start,
        };
        self.frame_starts.insert(start.file_offset, this);

        if header.is_visible() {
            let duration_ms = self
                .animation
                .as_ref()
                .filter(|anim| anim.tps_numerator > 0)
                .map_or(0.0, |anim| header.duration(anim));
            // The frame a seek has to start from: this one, or the earliest
            // origin of a slot it reads. Its entry is always present (it is
            // this frame or an earlier one) unless a `rewind` cleared the map
            // and a seek target restored slot origins from before the rewind.
            let decode_start = if decode_start_offset == start.file_offset {
                Some(this)
            } else {
                self.frame_starts.get(&decode_start_offset).copied()
            };
            let visible_frames_to_skip = decode_start.map(|ds| {
                self.visible_frame_index
                    .saturating_sub(ds.point.visible_count_before)
            });
            let seek_target =
                decode_start
                    .filter(|ds| ds.point.seekable)
                    .map(|ds| VisibleFrameSeekTarget {
                        decode_start_file_offset: ds.point.file_offset,
                        visible_frames_to_skip: visible_frames_to_skip.unwrap_or(0),
                        point: ds.point,
                        reference_slots: ds.reference_slots,
                        lf_slots: ds.lf_slots,
                    });
            let info = VisibleFrameInfo {
                index: self.visible_frame_index,
                duration_ms,
                duration_ticks: header.duration,
                file_offset: start.seekable.then_some(start.file_offset),
                is_last: header.is_last,
                is_keyframe: visible_frames_to_skip == Some(0),
                seek_target,
                name: header.name.clone(),
            };
            match self
                .scanned_frames
                .binary_search_by_key(&info.index, |f| f.index)
            {
                Ok(i) => self.scanned_frames[i] = info,
                Err(i) => self.scanned_frames.insert(i, info),
            }
            self.visible_frame_index += 1;
        }

        // This frame's own saves become the origin of whatever it stores.
        if header.can_be_referenced
            && let Some(slot) = self
                .reference_slot_decode_start
                .get_mut(header.save_as_reference as usize)
        {
            *slot = Some(decode_start_offset);
        }
        if header.lf_level != 0
            && let Some(slot) = self
                .lf_slot_decode_start
                .get_mut((header.lf_level - 1) as usize)
        {
            *slot = Some(decode_start_offset);
        }
    }

    /// Repositions the parser at `target` so the next `process` call parses
    /// frames from raw file input starting at
    /// `target.decode_start_file_offset`, passing over
    /// `target.visible_frames_to_skip` visible frames internally, and returns
    /// with the requested frame's header.
    ///
    /// Frame-level state is dropped. The decoder state is rebuilt fresh
    /// (empty reference and LF slots) with the frame counters a sequential
    /// decode would have had at the target, so noise seeds match; every slot
    /// the target and the frames after it read is re-filled by decoding
    /// forward from the dependency-resolved start. Image-level state (image
    /// header, basic info, color profiles, pixel format, `preview_done`) is
    /// kept. Dependency tracking resumes from the snapshot carried by the
    /// target, so `scanned_frames` stays consistent across seeks.
    pub(super) fn start_new_frame(
        &mut self,
        decode_options: &JxlDecoderOptions,
        target: &VisibleFrameSeekTarget,
    ) -> Result<()> {
        let file_header = self
            .image_file_header
            .clone()
            .ok_or_else(|| at!(Error::SeekBeforeImageInfo))?;
        let mut state = DecoderState::new(file_header);
        non_section::apply_decoder_options(
            &mut state,
            decode_options,
            &self.embedded_color_profile,
        );
        state.visible_frame_index = target.point.visible_counter;
        state.nonvisible_frame_index = target.point.nonvisible_counter;
        #[cfg(test)]
        if let Some(old) = self.decoder_state.as_ref() {
            state.use_simple_pipeline = old.use_simple_pipeline;
        }
        self.decoder_state = Some(state);

        self.frame_header = None;
        self.toc_parser = None;
        self.frame = None;
        self.non_section_buf = SmallBuffer::new(4096);
        self.non_section_bit_offset = 0;
        self.sections.clear();
        self.ready_section_data = 0;
        self.skip_sections = false;
        self.process_without_output = false;
        // Only non-preview frames are recorded, so the target is never the
        // preview frame; its header must be parsed with the main dimensions.
        self.preview_done = true;
        self.section_state = SectionState::new(0, 0);
        self.lf_global_section = None;
        self.lf_sections.clear();
        self.hf_global_section = None;
        self.hf_sections.clear();
        self.candidate_hf_sections.clear();
        self.file_length = None;
        self.pixels_dirty = false;
        self.has_more_frames = true;
        self.header_needed_bytes = None;

        self.visible_frame_index = target.point.visible_count_before;
        self.reference_slot_decode_start = target.reference_slots;
        self.lf_slot_decode_start = target.lf_slots;
        self.current_frame_start = None;
        self.pending_visible_skips = target.visible_frames_to_skip;
        self.auto_skipping_visible = false;
        Ok(())
    }

    /// Installs the decoder state handed back by `Frame::finalize` (shared by
    /// the decode and skip paths). `None` means the frame was `is_last`: for
    /// a skipped preview frame the main frame still follows, so the state is
    /// recreated from the saved image header with all options re-applied
    /// (libjxl/jxl-rs #743); otherwise the codestream is finished.
    pub(super) fn install_decoder_state_after_frame(
        &mut self,
        decoder_state: Option<DecoderState>,
        might_be_preview: bool,
        decode_options: &JxlDecoderOptions,
    ) {
        if let Some(state) = decoder_state {
            self.decoder_state = Some(state);
        } else if might_be_preview {
            if let Some(fh) = self.saved_file_header.take() {
                let mut new_state = DecoderState::new(fh);
                non_section::apply_decoder_options(
                    &mut new_state,
                    decode_options,
                    &self.embedded_color_profile,
                );
                self.decoder_state = Some(new_state);
            }
        } else {
            self.has_more_frames = false;
        }
    }

    pub(super) fn process(
        &mut self,
        box_parser: &mut BoxParser,
        input: &mut dyn JxlBitstreamInput,
        decode_options: &JxlDecoderOptions,
        mut output_buffers: Option<&mut [JxlOutputBuffer]>,
        do_flush: bool,
    ) -> Result<()> {
        if let Some(output_buffers) = &output_buffers {
            let px = self.pixel_format.as_ref().unwrap();
            let expected_len = std::iter::once(&px.color_data_format)
                .chain(px.extra_channel_format.iter())
                .filter(|x| x.is_some())
                .count();
            if output_buffers.len() != expected_len {
                return Err(at!(Error::WrongBufferCount(
                    output_buffers.len(),
                    expected_len
                )));
            }
        }
        // If we have sections to read, read into sections; otherwise, read into the local buffer.
        loop {
            if !self.sections.is_empty() {
                // Try to pick up JBRD data that may have arrived during box parsing
                #[cfg(feature = "jpeg")]
                if self.jbrd_data.is_none()
                    && let Some(data) = box_parser.take_jbrd_data()
                {
                    self.jbrd_data = Some(data);
                    if let Some(frame) = &mut self.frame {
                        frame.enable_jpeg_reconstruction();
                    }
                }

                let regular_frame = self.has_visible_frame();
                // Only skip sections if we don't need the frame data. Frames that can be
                // referenced must be decoded because they serve as sources for patches,
                // blending, or frame extension in subsequent frames.
                let can_be_referenced = self
                    .frame
                    .as_ref()
                    .is_some_and(|f| f.header().can_be_referenced);
                if decode_options.scan_frames_only
                    || (!can_be_referenced
                        && (self.auto_skipping_visible
                            || (!self.process_without_output && output_buffers.is_none())))
                {
                    self.skip_sections = true;
                }

                if !self.skip_sections {
                    // Read sections up to the end of the current box.
                    let mut available_codestream = match box_parser.get_more_codestream(input) {
                        Err(e) if matches!(e.error(), Error::OutOfBounds(_)) => 0,
                        Ok(c) => c as usize,
                        Err(e) => return Err(e),
                    };
                    // `get_more_codestream` reports the codestream bytes left in the
                    // current box (u64::MAX for a bare codestream), not what the input
                    // can deliver now. Bound it by the bytes actually obtainable in this
                    // call so section buffers are only grown for data that exists.
                    let obtainable = box_parser
                        .box_buffer
                        .len()
                        .saturating_add(input.available_bytes().map_err(|e| at!(Error::from(e)))?);
                    available_codestream = available_codestream.min(obtainable);
                    let mut section_buffers = vec![];
                    let mut ready = self.ready_section_data;
                    for buf in self.sections.iter_mut() {
                        if available_codestream == 0 {
                            break;
                        }
                        let len = buf.len;
                        if len > ready {
                            let readable = (available_codestream + ready).min(len);
                            // Section lengths come from the untrusted TOC (one entry can
                            // claim ~1 GB), so grow each buffer only as far as the bytes
                            // we are about to read, fallibly. Once the last byte of a
                            // section is in sight the buffer takes its final size,
                            // `len + SECTION_PADDING`: the zero padding lets
                            // BitReader::refill() always use the fast 8-byte path and is
                            // what BitReader::new_padded() checks for. (jxl-rs #856)
                            grow_section_buffer(buf, readable)?;
                            section_buffers.push(IoSliceMut::new(&mut buf.data[ready..readable]));
                            available_codestream -= readable - ready;
                        }
                        ready = ready.saturating_sub(len);
                    }
                    let ready_before = self.ready_section_data;
                    let mut buffers = &mut section_buffers[..];
                    loop {
                        let num = if !box_parser.box_buffer.is_empty() {
                            box_parser.box_buffer.take(buffers)
                        } else {
                            let num = input.read(buffers).map_err(|e| at!(Error::from(e)))?;
                            box_parser.total_file_read += num as u64;
                            num
                        };
                        self.ready_section_data += num;
                        box_parser.consume_codestream(num as u64);
                        IoSliceMut::advance_slices(&mut buffers, num);
                        if num == 0 || buffers.is_empty() {
                            break;
                        }
                    }
                    match self.process_sections(decode_options, &mut output_buffers, do_flush) {
                        Ok(None) => Ok(()),
                        Ok(Some(missing)) => Err(at!(Error::OutOfBounds(missing))),
                        Err(e) if matches!(e.error(), Error::OutOfBounds(_)) => {
                            Err(at!(Error::SectionTooShort))
                        }
                        Err(err) => Err(err),
                    }?;
                    self.record_file_length(box_parser);
                    // If no section data was read and sections are still pending,
                    // the input is truncated — return an error instead of looping
                    // forever waiting for data that will never arrive.
                    if !self.sections.is_empty()
                        && self.ready_section_data == ready_before
                        && input.available_bytes().unwrap_or(0) == 0
                        && box_parser.box_buffer.is_empty()
                    {
                        let total_needed: usize = self.sections.iter().map(|s| s.len).sum();
                        return Err(at!(Error::OutOfBounds(
                            total_needed.saturating_sub(self.ready_section_data),
                        )));
                    }
                } else {
                    let total_size = self.sections.iter().map(|x| x.len).sum::<usize>();
                    loop {
                        let to_skip = total_size - self.ready_section_data;
                        if to_skip == 0 {
                            break;
                        }
                        let available_codestream = box_parser.get_more_codestream(input)? as usize;
                        let to_skip = to_skip.min(available_codestream);
                        let skipped = if !box_parser.box_buffer.is_empty() {
                            box_parser.box_buffer.consume(to_skip)
                        } else {
                            let skipped = input.skip(to_skip).map_err(|e| at!(Error::from(e)))?;
                            box_parser.total_file_read += skipped as u64;
                            skipped
                        };
                        box_parser.consume_codestream(skipped as u64);
                        self.ready_section_data += skipped;
                        if skipped == 0 {
                            break;
                        }
                    }
                    if self.ready_section_data < total_size {
                        return Err(at!(Error::OutOfBounds(
                            total_size - self.ready_section_data
                        )));
                    } else {
                        self.sections.clear();
                        // Finalize the skipped frame, mirroring what process_sections does
                        let frame = self
                            .frame
                            .take()
                            .expect("frame must be set when skip_sections is true");
                        // A skipped preview frame (only reachable here with
                        // `scan_frames_only`) is `is_last` but the main frame
                        // still follows.
                        let might_be_preview = self.process_without_output
                            && self
                                .basic_info
                                .as_ref()
                                .is_some_and(|info| info.preview_size.is_some());
                        let decoder_state = frame.finalize()?;
                        self.install_decoder_state_after_frame(
                            decoder_state,
                            might_be_preview,
                            decode_options,
                        );
                        self.record_file_length(box_parser);
                        self.skip_sections = false;
                    }
                }
                if self.sections.is_empty() {
                    // Go back to parsing a new frame header, if any.
                    // Only return if this was a regular visible frame that was actually decoded
                    // (not a frame we were skipping like a preview frame)
                    let was_skipping = self.process_without_output;
                    self.process_without_output = false;
                    self.auto_skipping_visible = false;
                    if regular_frame && !was_skipping {
                        // JBRD reconstruction: the trailing Exif / XMP / brob
                        // metadata boxes follow the frame's codestream. When the
                        // codestream is large the section reads stop exactly at
                        // the frame end without over-reading the trailing boxes,
                        // so drive the box parser through them now — otherwise
                        // EXIF/XMP/ICC are silently dropped from the reconstructed
                        // APPn markers. `OutOfBounds` is the clean EOF after the
                        // last box; only fires when a reconstruction is pending,
                        // so non-JPEG decodes are unaffected.
                        #[cfg(feature = "jpeg")]
                        if self.jpeg_recon.is_some() {
                            match box_parser.get_more_codestream(input) {
                                Ok(_) => {}
                                Err(e) if matches!(e.error(), Error::OutOfBounds(_)) => {}
                                Err(e) => return Err(e),
                            }
                        }
                        return Ok(());
                    }
                    continue;
                }
            } else {
                // Trying to read a frame or a file header.
                assert!(self.frame.is_none());
                // Defensive (mirrors libjxl/jxl-rs#749): on normal input `has_more_frames`
                // is still set when we reach this header-reading branch, but a
                // malformed/edge input (e.g. skipping a final preview frame, or an empty
                // follow-up `process()` call) can re-enter here with it cleared. There is
                // no more image data to read, so return gracefully instead of panicking
                // on untrusted input.
                if !self.has_more_frames {
                    // The codestream is complete, but metadata boxes (`jhgm`
                    // gain map, `Exif`, `xml `, `brob`) may trail it in the
                    // container — jxl-encoder's `append_gain_map_bundle`
                    // writes `signature + ftyp + jxlc + jhgm` (#20). Drive
                    // the box parser through whatever input is available so
                    // they are captured. `OutOfBounds` is the clean EOF after
                    // the last box — or an incomplete trailing box, whose
                    // parsing resumes if `process` is called again with more
                    // input.
                    match box_parser.get_more_codestream(input) {
                        Ok(_) => {}
                        Err(e) if matches!(e.error(), Error::OutOfBounds(_)) => {}
                        Err(e) => return Err(e),
                    }
                    return Ok(());
                }

                // Record where the next frame header starts, for seeking. A
                // frame that begins exactly at a box boundary can only be
                // placed once `get_more_codestream` has entered the next box,
                // hence the second attempt inside the loop. Re-entries (the
                // header needed more input) recompute the same point because
                // nothing is consumed until the header parses.
                let mut capture_frame_start =
                    self.decoder_state.is_some() && self.frame_header.is_none();
                if capture_frame_start && self.try_capture_frame_start(box_parser) {
                    capture_frame_start = false;
                }

                // Loop to handle incremental parsing (e.g. large ICC profiles) that may need
                // multiple buffer refills to complete.
                loop {
                    let available_codestream = match box_parser.get_more_codestream(input) {
                        Err(e) if matches!(e.error(), Error::OutOfBounds(_)) => 0,
                        Ok(c) => c as usize,
                        Err(e) => return Err(e),
                    };
                    if capture_frame_start && self.try_capture_frame_start(box_parser) {
                        capture_frame_start = false;
                    }
                    let c = self.non_section_buf.refill(
                        |buf| {
                            if !box_parser.box_buffer.is_empty() {
                                Ok(box_parser.box_buffer.take(buf))
                            } else {
                                let num = input.read(buf)?;
                                box_parser.total_file_read += num as u64;
                                Ok(num)
                            }
                        },
                        Some(available_codestream),
                    )? as u64;
                    box_parser.consume_codestream(c);

                    // If we know that non-section parsing will require more bytes than what
                    // we added to the codestream, don't even try to parse non-section data.
                    if let Some(needed) = self.header_needed_bytes.as_mut() {
                        *needed = needed.saturating_sub(c);
                        if *needed > 0 {
                            if !self.non_section_buf.can_read_more() {
                                self.non_section_buf.enlarge();
                            }
                            // Check if input still has data - if so, refill and retry
                            if input.available_bytes().unwrap_or(0) > 0 {
                                continue;
                            } else {
                                return Err(at!(Error::OutOfBounds(*needed as usize)));
                            }
                        }
                    }

                    let range = self.non_section_buf.range();
                    match self.process_non_section(decode_options) {
                        Ok(()) => {
                            self.header_needed_bytes = None;
                            break;
                        }
                        Err(e) if matches!(e.error(), Error::OutOfBounds(_)) => {
                            let &Error::OutOfBounds(n) = e.error() else {
                                unreachable!()
                            };
                            let new_range = self.non_section_buf.range();
                            // If non-section parsing consumed no bytes, and the non-section buffer
                            // cannot accept more bytes, enlarge the buffer to allow to make progress.
                            if new_range == range && !self.non_section_buf.can_read_more() {
                                self.non_section_buf.enlarge();
                            }
                            self.header_needed_bytes = Some(n as u64);
                            // Check if input still has data - if so, refill and retry
                            if input.available_bytes().unwrap_or(0) > 0 {
                                continue;
                            } else {
                                return Err(at!(Error::OutOfBounds(n)));
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }

                if self.decoder_state.is_some() && self.frame_header.is_none() {
                    // Return to caller if we found image info.
                    return Ok(());
                }
                if self.frame.is_some() {
                    // Transfer JBRD data from box parser and enable JPEG reconstruction
                    #[cfg(feature = "jpeg")]
                    if self.jbrd_data.is_none()
                        && let Some(data) = box_parser.take_jbrd_data()
                    {
                        self.jbrd_data = Some(data);
                    }
                    #[cfg(feature = "jpeg")]
                    if self.jbrd_data.is_some()
                        && let Some(frame) = &mut self.frame
                    {
                        frame.enable_jpeg_reconstruction();
                    }

                    // Check if this is a preview frame that should be skipped
                    let is_preview_frame = !self.preview_done
                        && self
                            .basic_info
                            .as_ref()
                            .is_some_and(|info| info.preview_size.is_some());
                    if is_preview_frame {
                        self.preview_done = true;
                        if decode_options.skip_preview {
                            self.process_without_output = true;
                            continue;
                        }
                    } else {
                        // Frame scanning / seek table (never for the preview).
                        self.record_frame_info();
                    }

                    if self.has_visible_frame() {
                        if self.pending_visible_skips > 0 {
                            // A visible frame before the seek target: pass over
                            // it (decoding it without output if later frames
                            // can reference it) instead of returning it.
                            self.pending_visible_skips -= 1;
                            self.process_without_output = true;
                            self.auto_skipping_visible = true;
                            continue;
                        }
                        // Return to caller if we found visible frame info.
                        return Ok(());
                    } else {
                        self.process_without_output = true;
                        continue;
                    }
                }
            }
        }
    }

    pub(super) fn update_default_output_color_profile(&mut self) {
        // Only set default output_color_profile if not already configured by user
        if self.output_color_profile_set_by_user {
            return;
        }

        let embedded_color_profile = self.embedded_color_profile.as_ref().unwrap();
        let pixel_format = self.pixel_format.as_ref().unwrap();

        // Determine default output color profile following libjxl logic:
        // - For XYB: use embedded if can_output_to(), else:
        //   - if float samples are requested: linear sRGB,
        //   - else: sRGB
        // - For non-XYB: use embedded color profile, except CMYK (which can't be
        //   emitted as the RGB pixel format) which falls back to sRGB
        let output_color_profile = if self.xyb_encoded {
            // Use embedded if we can output to it, otherwise fall back to sRGB
            let base_encoding = if embedded_color_profile.can_output_to() {
                match &embedded_color_profile {
                    JxlColorProfile::Simple(enc) => enc.clone(),
                    JxlColorProfile::Icc(_) => {
                        unreachable!("can_output_to returns false for ICC")
                    }
                }
            } else {
                let data_format = pixel_format
                    .color_data_format
                    .unwrap_or(JxlDataFormat::U8 { bit_depth: 8 });
                let is_float = matches!(
                    data_format,
                    JxlDataFormat::F32 { .. } | JxlDataFormat::F16 { .. }
                );
                if is_float {
                    JxlColorEncoding::linear_srgb(self.is_gray)
                } else {
                    JxlColorEncoding::srgb(self.is_gray)
                }
            };

            JxlColorProfile::Simple(base_encoding)
        } else if embedded_color_profile.is_cmyk() {
            // A CMYK profile can't be emitted as the RGB pixel format directly:
            // keeping it as the output profile asks the CMS stage for a
            // CMYK -> CMYK transform, which fails. Convert to sRGB instead so the
            // CMS stage runs CMYK -> sRGB. (closes #37)
            let data_format = pixel_format
                .color_data_format
                .unwrap_or(JxlDataFormat::U8 { bit_depth: 8 });
            let is_float = matches!(
                data_format,
                JxlDataFormat::F32 { .. } | JxlDataFormat::F16 { .. }
            );
            JxlColorProfile::Simple(if is_float {
                JxlColorEncoding::linear_srgb(false)
            } else {
                JxlColorEncoding::srgb(false)
            })
        } else {
            embedded_color_profile.clone()
        };
        self.output_color_profile = Some(output_color_profile);
    }
}
