// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Where each part of a codestream sits, as bit and byte spans.
//!
//! Enabled with [`JxlDecoderOptions::with_record_layout`](crate::api::JxlDecoderOptions::with_record_layout)
//! and read with [`JxlDecoder::codestream_layout`](crate::api::JxlDecoder::codestream_layout).
//! It is a by-product of the normal header and TOC parse: nothing is decoded
//! twice and no pixel buffers are involved. All offsets are *codestream*
//! offsets (the `jxlc` / `jxlp` payloads concatenated, `jxlp` index words
//! excluded; for a bare codestream, file offsets). A walker that knows the
//! container boxes maps them onto the file.

use std::ops::Range;

/// A span of header bits that holds text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedSpan {
    /// Codestream bit range of the field (length prefix included).
    pub bits: Range<u64>,
    /// The decoded text.
    pub text: String,
}

/// Layout of one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameLayout {
    /// The frame is the preview frame (decoded before the main image).
    pub is_preview: bool,
    /// The frame is displayed (not a reference-only frame, and has a duration
    /// or is the last one).
    pub is_visible: bool,
    /// The frame can be a reference for a later frame (patches, blending, LF).
    pub can_be_referenced: bool,
    /// Frame header, bit range.
    pub header_bits: Range<u64>,
    /// The frame name, when it is not empty.
    pub name: Option<NamedSpan>,
    /// Extension fields of the frame header and its restoration filter that
    /// hold more than the empty selector.
    pub extensions: Vec<Range<u64>>,
    /// TOC, bit range (ends on a byte boundary).
    pub toc_bits: Range<u64>,
    /// Byte range of each section, in file order.
    pub sections: Vec<Range<u64>>,
}

/// Layout of a codestream, filled in as the decoder parses it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodestreamLayout {
    /// Image header (size, metadata, color encoding), bit range, signature
    /// excluded.
    pub image_header_bits: Option<Range<u64>>,
    /// Extra-channel names that are not empty, in channel order, with the
    /// channel index.
    pub extra_channel_names: Vec<(usize, NamedSpan)>,
    /// Extension fields of the image metadata that hold more than the empty
    /// selector.
    pub image_extensions: Vec<Range<u64>>,
    /// The embedded ICC stream (compressed form), bit range.
    pub icc_bits: Option<Range<u64>>,
    /// Frames whose header and TOC have been parsed, in codestream order.
    pub frames: Vec<FrameLayout>,
}
