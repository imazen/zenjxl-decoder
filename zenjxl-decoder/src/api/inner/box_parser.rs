// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::IoSliceMut;
use whereat::at;

use crate::container::frame_index::FrameIndexBox;
use crate::container::gain_map::GainMapBundle;
use crate::error::{Error, Result};
use crate::util::NewWithCapacity;

use crate::api::{
    JxlBitstreamInput, JxlSignatureType, check_signature_internal, inner::process::SmallBuffer,
};

#[derive(Clone)]
enum ParseState {
    SignatureNeeded,
    BoxNeeded,
    CodestreamBox(u64),
    SkippableBox(u64),
    /// Reading the first 8 payload bytes of `ftyp` (major brand + minor
    /// version), then skipping `skip_rest` bytes of compatible brands.
    FtypHead {
        head: [u8; 8],
        got: u8,
        skip_rest: u64,
    },
    /// Buffering an out-of-order `jxlp` box (payload accumulates in
    /// `OooJxlp::pending`) until its logical index comes up.
    BufferingOooJxlp {
        remaining: u64,
        idx: u32,
        last: bool,
    },
    /// The last codestream box has been consumed, nothing is buffered and the
    /// container has no more bytes: there is no further codestream.
    Exhausted,
    #[cfg(feature = "jpeg")]
    JbrdBox(u64),
    /// Buffering a jxli box: (remaining bytes, accumulated content).
    BufferingFrameIndex(u64, Vec<u8>),
    /// Buffering a jhgm box: (remaining bytes, accumulated content).
    BufferingGainMap(u64, Vec<u8>),
    /// Buffering an Exif box: (remaining bytes, accumulated content).
    BufferingExif(u64, Vec<u8>),
    /// Buffering an xml (XMP) box: (remaining bytes, accumulated content).
    BufferingXmp(u64, Vec<u8>),
    /// Buffering a brob (brotli-compressed) box: (remaining bytes, accumulated
    /// content). The content is `[4-byte inner box type][brotli payload]`; once
    /// complete it is decompressed and routed to exif/xmp by the inner type.
    #[cfg(feature = "jpeg")]
    BufferingBrob(u64, Vec<u8>),
}

enum CodestreamBoxType {
    None,
    Jxlc,
    Jxlp(u32),
    LastJxlp,
}

/// Out-of-order `jxlp` support (ISO/IEC 18181-2 with `ftyp` minor version 1;
/// `cjxl --output_mode 2` writes such files; libjxl decodes them).
///
/// A `jxlp` box whose index is ahead of the next expected one is buffered by
/// index and spliced in front of the container input the moment the
/// preceding index has been consumed. Upstream jxl-rs #752 / #777.
#[derive(Default)]
struct OooJxlp {
    /// `ftyp` minor version: 0 = `jxlp` boxes must be in order, 1 = out-of-order allowed.
    file_format_version: u32,
    ftyp_seen: bool,
    /// Payload of the box currently in `ParseState::BufferingOooJxlp`.
    pending: Vec<u8>,
    /// Complete out-of-order payloads keyed by logical index (`is_last` flag kept).
    buffered: std::collections::BTreeMap<u32, (Vec<u8>, bool)>,
}

impl OooJxlp {
    /// libjxl's `kNumBuffersLimit`: a file cannot make us hold more than this
    /// many boxes ahead of the one we are decoding.
    const MAX_BUFFERED_BOXES: usize = 1024;
    /// Grow the pending buffer in bounded steps; the declared box size is
    /// untrusted and is never allocated up front.
    const GROW_STEP: usize = 64 * 1024;
}

/// Counts the bytes pulled from the wrapped input, so the box parser can
/// account for every byte of the file it consumed (see `total_file_read`).
struct CountingInput<'a> {
    inner: &'a mut dyn JxlBitstreamInput,
    count: u64,
}

impl JxlBitstreamInput for CountingInput<'_> {
    fn available_bytes(&mut self) -> Result<usize, std::io::Error> {
        self.inner.available_bytes()
    }

    fn read(&mut self, bufs: &mut [std::io::IoSliceMut]) -> Result<usize, std::io::Error> {
        let num = self.inner.read(bufs)?;
        self.count += num as u64;
        Ok(num)
    }

    fn skip(&mut self, bytes: usize) -> Result<usize, std::io::Error> {
        let num = self.inner.skip(bytes)?;
        self.count += num as u64;
        Ok(num)
    }
}

pub(super) struct BoxParser {
    pub(super) box_buffer: SmallBuffer,
    /// Total bytes ever pulled from the caller's input (through
    /// `get_more_codestream` or counted explicitly by the codestream parser's
    /// direct section reads). Some of them may still sit unconsumed in
    /// [`Self::box_buffer`] or in buffered out-of-order `jxlp` payloads; see
    /// [`Self::buffered_leftover`].
    pub(super) total_file_read: u64,
    state: ParseState,
    box_type: CodestreamBoxType,
    #[cfg(feature = "jpeg")]
    jbrd_data: Option<Vec<u8>>,
    /// Parsed frame index box, if present in the file.
    pub(super) frame_index: Option<FrameIndexBox>,
    /// Parsed gain map bundle, if present in the file.
    pub(super) gain_map: Option<GainMapBundle>,
    /// Raw EXIF data from the `Exif` container box (without the 4-byte TIFF offset prefix).
    pub(super) exif: Option<Vec<u8>>,
    /// Raw XMP data from the `xml ` container box.
    pub(super) xmp: Option<Vec<u8>>,
    ooo_jxlp: OooJxlp,
}

impl BoxParser {
    pub(super) fn new() -> Self {
        BoxParser {
            box_buffer: SmallBuffer::new(128),
            total_file_read: 0,
            state: ParseState::SignatureNeeded,
            box_type: CodestreamBoxType::None,
            #[cfg(feature = "jpeg")]
            jbrd_data: None,
            frame_index: None,
            gain_map: None,
            exif: None,
            xmp: None,
            ooo_jxlp: OooJxlp::default(),
        }
    }

    fn next_expected_jxlp_index(&self) -> Option<u32> {
        match self.box_type {
            CodestreamBoxType::None => Some(0),
            CodestreamBoxType::Jxlp(i) => Some(i + 1),
            CodestreamBoxType::LastJxlp | CodestreamBoxType::Jxlc => None,
        }
    }

    /// If the next expected `jxlp` index was received out of order, splice
    /// its payload in front of the container input as the next codestream
    /// box.
    fn try_inject_next_buffered_jxlp(&mut self) {
        let Some(next) = self.next_expected_jxlp_index() else {
            return;
        };
        let Some((payload, is_last)) = self.ooo_jxlp.buffered.remove(&next) else {
            return;
        };
        let len = payload.len() as u64;
        self.box_buffer.inject_bytes_front(payload);
        self.box_type = if is_last {
            CodestreamBoxType::LastJxlp
        } else {
            CodestreamBoxType::Jxlp(next)
        };
        self.state = if len == 0 {
            ParseState::BoxNeeded
        } else {
            ParseState::CodestreamBox(len)
        };
    }

    /// Take the accumulated JBRD box data, if any was found.
    #[cfg(feature = "jpeg")]
    pub(super) fn take_jbrd_data(&mut self) -> Option<Vec<u8>> {
        self.jbrd_data.take()
    }

    // Reads input until the next byte of codestream is available.
    // This function might over-read bytes. Thus, the contents of self.box_buffer should always be
    // read after this function call.
    // Returns the number of codestream bytes that will be available to be read after this call,
    // including any bytes in self.box_buffer.
    // Might return `u64::MAX`, indicating that the rest of the file is codestream.
    pub(super) fn get_more_codestream(&mut self, input: &mut dyn JxlBitstreamInput) -> Result<u64> {
        let mut counting = CountingInput {
            inner: input,
            count: 0,
        };
        let result = self.get_more_codestream_impl(&mut counting);
        self.total_file_read += counting.count;
        result
    }

    /// Bytes read from the input that are still buffered (not yet part of any
    /// parsed structure): the box buffer plus buffered out-of-order `jxlp`
    /// payloads.
    pub(super) fn buffered_leftover(&self) -> u64 {
        self.box_buffer.len() as u64
            + self.ooo_jxlp.pending.len() as u64
            + self
                .ooo_jxlp
                .buffered
                .values()
                .map(|(data, _)| data.len() as u64)
                .sum::<u64>()
    }

    fn get_more_codestream_impl(&mut self, input: &mut dyn JxlBitstreamInput) -> Result<u64> {
        loop {
            match self.state.clone() {
                ParseState::SignatureNeeded => {
                    self.box_buffer.refill(|b| input.read(b), None)?;
                    match check_signature_internal(&self.box_buffer)? {
                        None => return Err(at!(Error::InvalidSignature)),
                        Some(JxlSignatureType::Codestream) => {
                            self.state = ParseState::CodestreamBox(u64::MAX);
                            return Ok(u64::MAX);
                        }
                        Some(JxlSignatureType::Container) => {
                            self.box_buffer
                                .consume(JxlSignatureType::Container.signature().len());
                            self.state = ParseState::BoxNeeded;
                        }
                    }
                }
                ParseState::Exhausted => {
                    return Ok(0);
                }
                ParseState::CodestreamBox(0) => {
                    // An empty codestream box (e.g. a `jxlp` with an index but no
                    // payload) contributes nothing; look for the next box.
                    self.state = ParseState::BoxNeeded;
                    self.try_inject_next_buffered_jxlp();
                }
                ParseState::FtypHead {
                    mut head,
                    mut got,
                    skip_rest,
                } => {
                    while got < 8 {
                        if self.box_buffer.is_empty() {
                            self.box_buffer.refill(|b| input.read(b), None)?;
                        }
                        if self.box_buffer.is_empty() {
                            // Keep the partial brand so a tiny next chunk can resume.
                            self.state = ParseState::FtypHead {
                                head,
                                got,
                                skip_rest,
                            };
                            return Err(at!(Error::OutOfBounds(8 - got as usize)));
                        }
                        head[got as usize] = self.box_buffer[0];
                        self.box_buffer.consume(1);
                        got += 1;
                    }
                    if &head[0..4] != b"jxl " {
                        return Err(at!(Error::InvalidBox));
                    }
                    let version = u32::from_be_bytes(head[4..8].try_into().unwrap());
                    if version > 1 {
                        return Err(at!(Error::InvalidBox));
                    }
                    self.ooo_jxlp.file_format_version = version;
                    self.ooo_jxlp.ftyp_seen = true;
                    self.state = if skip_rest == 0 {
                        ParseState::BoxNeeded
                    } else {
                        ParseState::SkippableBox(skip_rest)
                    };
                }
                ParseState::BufferingOooJxlp {
                    mut remaining,
                    idx,
                    last,
                } => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    let buf = &mut self.ooo_jxlp.pending;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.try_reserve(take).map_err(|e| at!(Error::from(e)))?;
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let step = num.min(OooJxlp::GROW_STEP);
                        let old_len = buf.len();
                        buf.try_reserve(step).map_err(|e| at!(Error::from(e)))?;
                        buf.resize(old_len + step, 0);
                        let read = input
                            .read(&mut [IoSliceMut::new(&mut buf[old_len..])])
                            .map_err(|e| at!(Error::from(e)))?;
                        buf.truncate(old_len + read);
                        if read == 0 {
                            self.state = ParseState::BufferingOooJxlp {
                                remaining,
                                idx,
                                last,
                            };
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        let payload = std::mem::take(&mut self.ooo_jxlp.pending);
                        self.ooo_jxlp.buffered.insert(idx, (payload, last));
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingOooJxlp {
                            remaining,
                            idx,
                            last,
                        };
                    }
                }
                ParseState::CodestreamBox(b) => {
                    return Ok(b);
                }
                ParseState::SkippableBox(mut s) => {
                    if s == 0 {
                        // Empty box (size == header size): nothing to skip.
                        self.state = ParseState::BoxNeeded;
                        continue;
                    }
                    let num = s.min(usize::MAX as u64) as usize;
                    let skipped = if !self.box_buffer.is_empty() {
                        self.box_buffer.consume(num)
                    } else {
                        input.skip(num).map_err(|e| at!(Error::from(e)))?
                    };
                    if skipped == 0 {
                        return Err(at!(Error::OutOfBounds(num)));
                    }
                    s -= skipped as u64;
                    if s == 0 {
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::SkippableBox(s);
                    }
                }
                #[cfg(feature = "jpeg")]
                ParseState::JbrdBox(mut remaining) => {
                    // Read jbrd box content into buffer
                    let num = remaining.min(usize::MAX as u64) as usize;
                    let jbrd = self.jbrd_data.get_or_insert_with(Vec::new);
                    if !self.box_buffer.is_empty() {
                        let avail = self.box_buffer.len().min(num);
                        jbrd.extend_from_slice(&self.box_buffer[..avail]);
                        self.box_buffer.consume(avail);
                        remaining -= avail as u64;
                    } else {
                        // Read from input using IoSliceMut
                        let chunk_size = num.min(65536);
                        let start = jbrd.len();
                        jbrd.resize(start + chunk_size, 0);
                        let read = input
                            .read(&mut [std::io::IoSliceMut::new(&mut jbrd[start..])])
                            .map_err(Error::from)?;
                        if read == 0 {
                            jbrd.truncate(start);
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        jbrd.truncate(start + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::JbrdBox(remaining);
                    }
                }
                ParseState::BufferingFrameIndex(mut remaining, mut buf) => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input
                            .read(&mut [IoSliceMut::new(&mut buf[old_len..])])
                            .map_err(|e| at!(Error::from(e)))?;
                        if read == 0 {
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        // Parse the buffered frame index box.
                        self.frame_index = Some(FrameIndexBox::parse(&buf)?);
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingFrameIndex(remaining, buf);
                    }
                }
                ParseState::BufferingGainMap(mut remaining, mut buf) => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input
                            .read(&mut [IoSliceMut::new(&mut buf[old_len..])])
                            .map_err(|e| at!(Error::from(e)))?;
                        if read == 0 {
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        // Parse the buffered gain map bundle.
                        self.gain_map = Some(GainMapBundle::parse(&buf)?);
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingGainMap(remaining, buf);
                    }
                }
                ParseState::BufferingExif(mut remaining, mut buf) => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input
                            .read(&mut [IoSliceMut::new(&mut buf[old_len..])])
                            .map_err(|e| at!(Error::from(e)))?;
                        if read == 0 {
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        // Exif box payload starts with a 4-byte TIFF header offset
                        // (big-endian u32). Strip it to return raw EXIF/TIFF data.
                        if buf.len() >= 4 {
                            self.exif = Some(buf[4..].to_vec());
                        }
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingExif(remaining, buf);
                    }
                }
                ParseState::BufferingXmp(mut remaining, mut buf) => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input
                            .read(&mut [IoSliceMut::new(&mut buf[old_len..])])
                            .map_err(|e| at!(Error::from(e)))?;
                        if read == 0 {
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        self.xmp = Some(buf);
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingXmp(remaining, buf);
                    }
                }
                #[cfg(feature = "jpeg")]
                ParseState::BufferingBrob(mut remaining, mut buf) => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input
                            .read(&mut [IoSliceMut::new(&mut buf[old_len..])])
                            .map_err(Error::from)?;
                        if read == 0 {
                            return Err(at!(Error::OutOfBounds(num)));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        // brob payload = [4-byte inner box type][brotli stream].
                        if buf.len() >= 4
                            && let Some(out) = brotli_decompress_box(&buf[4..])
                        {
                            match &buf[0..4] {
                                // Same handling as a raw Exif box: strip the
                                // 4-byte TIFF-header offset.
                                b"Exif" if out.len() >= 4 => self.exif = Some(out[4..].to_vec()),
                                b"xml " => self.xmp = Some(out),
                                _ => {}
                            }
                        }
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingBrob(remaining, buf);
                    }
                }
                ParseState::BoxNeeded => {
                    let read = self.box_buffer.refill(|b| input.read(b), None)?;
                    if self.box_buffer.is_empty()
                        && read == 0
                        && self.ooo_jxlp.ftyp_seen
                        && self.ooo_jxlp.buffered.is_empty()
                        && matches!(
                            self.box_type,
                            CodestreamBoxType::Jxlc | CodestreamBoxType::LastJxlp
                        )
                    {
                        self.state = ParseState::Exhausted;
                        return Ok(0);
                    }
                    let min_len = match &self.box_buffer[..] {
                        [0, 0, 0, 1, ..] => 16,
                        _ => 8,
                    };
                    if self.box_buffer.len() < min_len {
                        return Err(at!(Error::OutOfBounds(min_len - self.box_buffer.len())));
                    }
                    let ty: [_; 4] = self.box_buffer[4..8].try_into().unwrap();
                    let extra_len = if &ty == b"jxlp" { 4 } else { 0 };
                    if self.box_buffer.len() < min_len + extra_len {
                        return Err(at!(Error::OutOfBounds(
                            min_len + extra_len - self.box_buffer.len(),
                        )));
                    }
                    let box_len = match &self.box_buffer[..] {
                        [0, 0, 0, 1, ..] => {
                            u64::from_be_bytes(self.box_buffer[8..16].try_into().unwrap())
                        }
                        _ => u32::from_be_bytes(self.box_buffer[0..4].try_into().unwrap()) as u64,
                    };
                    // ISOBMFF: a box size of 0 means "extends to the end of the
                    // file" -- for any box type, as libjxl handles it (decode.cc);
                    // every consumer below treats u64::MAX as "until EOF". A box
                    // whose size equals its header size is a legal empty box
                    // (libjxl only rejects `box_size < header_size`); the fork used
                    // to reject both with InvalidBox. (jxl-rs #828)
                    let content_len = if box_len == 0 {
                        u64::MAX
                    } else {
                        if box_len < (min_len + extra_len) as u64 {
                            return Err(at!(Error::InvalidBox));
                        }
                        box_len - min_len as u64 - extra_len as u64
                    };
                    // ISO/IEC 18181-2: `ftyp` is the second box, exactly once
                    // (libjxl: "the second box must be the ftyp box").
                    if self.ooo_jxlp.ftyp_seen == (&ty == b"ftyp") {
                        return Err(at!(Error::InvalidBox));
                    }
                    match &ty {
                        b"ftyp" => {
                            // payload: major brand 'jxl ' + minor version, then
                            // compatible brands (skipped).
                            if content_len == u64::MAX || content_len < 8 {
                                return Err(at!(Error::InvalidBox));
                            }
                            self.state = ParseState::FtypHead {
                                head: [0; 8],
                                got: 0,
                                skip_rest: content_len - 8,
                            };
                        }
                        b"jxlc" => {
                            if matches!(
                                self.box_type,
                                CodestreamBoxType::Jxlp(..) | CodestreamBoxType::LastJxlp
                            ) {
                                return Err(at!(Error::InvalidBox));
                            }
                            self.box_type = CodestreamBoxType::Jxlc;
                            self.state = ParseState::CodestreamBox(content_len);
                        }
                        b"jxlp" => {
                            let index = u32::from_be_bytes(
                                self.box_buffer[min_len..min_len + 4].try_into().unwrap(),
                            );
                            let wanted_idx = match self.box_type {
                                CodestreamBoxType::Jxlc | CodestreamBoxType::LastJxlp => {
                                    return Err(at!(Error::InvalidBox));
                                }
                                CodestreamBoxType::None => 0,
                                CodestreamBoxType::Jxlp(i) => i + 1,
                            };
                            let last = index & 0x80000000 != 0;
                            let idx = index & 0x7fffffff;
                            if idx < wanted_idx {
                                return Err(at!(Error::InvalidBox));
                            }
                            if idx > wanted_idx {
                                // Out of order: only with ftyp minor version 1, each
                                // index once, a bounded number of boxes, and never
                                // an unbounded (size 0) box.
                                if self.ooo_jxlp.file_format_version < 1
                                    || self.ooo_jxlp.buffered.contains_key(&idx)
                                    || self.ooo_jxlp.buffered.len() >= OooJxlp::MAX_BUFFERED_BOXES
                                    || content_len == u64::MAX
                                {
                                    return Err(at!(Error::InvalidBox));
                                }
                                self.ooo_jxlp.pending.clear();
                                self.state = ParseState::BufferingOooJxlp {
                                    remaining: content_len,
                                    idx,
                                    last,
                                };
                                self.box_buffer.consume(min_len + extra_len);
                                continue;
                            }
                            self.box_type = if last {
                                CodestreamBoxType::LastJxlp
                            } else {
                                CodestreamBoxType::Jxlp(idx)
                            };
                            self.state = ParseState::CodestreamBox(content_len);
                        }
                        #[cfg(feature = "jpeg")]
                        b"jbrd" => {
                            self.state = ParseState::JbrdBox(content_len);
                        }
                        b"jhgm" => {
                            if content_len == u64::MAX {
                                return Err(at!(Error::InvalidBox));
                            }
                            // Reasonable size limit for a gain map bundle (256 MB).
                            // Gain maps contain a full JXL codestream so they can be large.
                            if content_len > 256 * 1024 * 1024 {
                                self.state = ParseState::SkippableBox(content_len);
                            } else {
                                // Use fallible alloc — capacity reserves up to
                                // 256 MB and a global allocator that can't
                                // satisfy the request must surface as a graceful
                                // error, never an `abort()` on the parser thread.
                                self.state = ParseState::BufferingGainMap(
                                    content_len,
                                    Vec::<u8>::new_with_capacity(content_len as usize)
                                        .map_err(|e| at!(Error::from(e)))?,
                                );
                            }
                        }
                        b"jxli" => {
                            if content_len == u64::MAX {
                                return Err(at!(Error::InvalidBox));
                            }
                            // Reasonable size limit for a frame index box (16 MB).
                            if content_len > 16 * 1024 * 1024 {
                                self.state = ParseState::SkippableBox(content_len);
                            } else {
                                self.state = ParseState::BufferingFrameIndex(
                                    content_len,
                                    Vec::<u8>::new_with_capacity(content_len as usize)
                                        .map_err(|e| at!(Error::from(e)))?,
                                );
                            }
                        }
                        b"Exif" => {
                            if content_len == u64::MAX {
                                return Err(at!(Error::InvalidBox));
                            }
                            // Reasonable size limit for EXIF data (16 MB).
                            if content_len > 16 * 1024 * 1024 {
                                self.state = ParseState::SkippableBox(content_len);
                            } else {
                                self.state = ParseState::BufferingExif(
                                    content_len,
                                    Vec::<u8>::new_with_capacity(content_len as usize)
                                        .map_err(|e| at!(Error::from(e)))?,
                                );
                            }
                        }
                        b"xml " => {
                            if content_len == u64::MAX {
                                return Err(at!(Error::InvalidBox));
                            }
                            // Reasonable size limit for XMP data (16 MB).
                            if content_len > 16 * 1024 * 1024 {
                                self.state = ParseState::SkippableBox(content_len);
                            } else {
                                self.state = ParseState::BufferingXmp(
                                    content_len,
                                    Vec::<u8>::new_with_capacity(content_len as usize)
                                        .map_err(|e| at!(Error::from(e)))?,
                                );
                            }
                        }
                        // Brotli-compressed metadata box. The encoder wraps EXIF
                        // / XMP in `brob` when brotli shrinks them; decompress
                        // and route by inner box type. Needs brotli (jpeg feat).
                        #[cfg(feature = "jpeg")]
                        b"brob" => {
                            if content_len == u64::MAX || content_len > 16 * 1024 * 1024 {
                                self.state = ParseState::SkippableBox(content_len);
                            } else {
                                self.state = ParseState::BufferingBrob(
                                    content_len,
                                    Vec::<u8>::new_with_capacity(content_len as usize)
                                        .map_err(Error::from)?,
                                );
                            }
                        }
                        _ => {
                            self.state = ParseState::SkippableBox(content_len);
                        }
                    }
                    self.box_buffer.consume(min_len + extra_len);
                }
            }
        }
    }

    pub(super) fn consume_codestream(&mut self, amount: u64) {
        if let ParseState::CodestreamBox(cb) = &mut self.state {
            *cb = cb.checked_sub(amount).unwrap();
            if *cb == 0 {
                self.state = ParseState::BoxNeeded;
                self.try_inject_next_buffered_jxlp();
            }
        } else if amount != 0 {
            unreachable!()
        }
    }
}

/// Brotli-decompress a `brob` box payload (the bytes after the 4-byte inner box
/// type). Returns `None` on malformed input. Only built with the `jpeg` feature
/// (which provides `brotli`); without it `brob` boxes are skipped.
#[cfg(feature = "jpeg")]
fn brotli_decompress_box(compressed: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::new();
    brotli::Decompressor::new(compressed, 4096)
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}
