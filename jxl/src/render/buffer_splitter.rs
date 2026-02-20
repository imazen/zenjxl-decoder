// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[cfg(feature = "threads")]
use crate::image::SharedOutputView;
use crate::{api::JxlOutputBuffer, headers::Orientation, image::Rect, util::ShiftRightCeil};

// Information for splitting the output buffers.
#[derive(Debug)]
pub(crate) struct SaveStageBufferInfo {
    pub(crate) downsample: (u8, u8),
    pub(crate) orientation: Orientation,
    pub(crate) byte_size: usize,
    pub(crate) after_extend: bool,
}

/// Data structure responsible for handing out access to portions of the output buffers.
pub struct BufferSplitter<'a, 'b>(&'a mut [Option<JxlOutputBuffer<'b>>]);

impl<'a, 'b> BufferSplitter<'a, 'b> {
    pub fn new(bufs: &'a mut [Option<JxlOutputBuffer<'b>>]) -> Self {
        Self(bufs)
    }

    pub(super) fn get_local_buffers(
        &mut self,
        save_buffer_info: &[Option<SaveStageBufferInfo>],
        rect: Rect,
        outside_current_frame: bool,
        frame_size: (usize, usize),
        full_image_size: (usize, usize),
        frame_origin: (isize, isize),
    ) -> Vec<Option<JxlOutputBuffer<'_>>> {
        let mut local_buffers = vec![];
        let buffers = &mut *self.0;
        local_buffers.reserve(buffers.len());
        for _ in 0..buffers.len() {
            local_buffers.push(None::<JxlOutputBuffer>);
        }
        let rect = if !outside_current_frame {
            rect.clip(frame_size)
        } else {
            rect
        };
        for (i, (info, buf)) in save_buffer_info.iter().zip(buffers.iter_mut()).enumerate() {
            let Some(bi) = info else {
                // We never write to this buffer.
                continue;
            };
            let Some(buf) = buf.as_mut() else {
                // The buffer to write into was not provided.
                continue;
            };
            if outside_current_frame && !bi.after_extend {
                // Before-extend stages do not write to rects outside the current frame.
                continue;
            }
            let mut channel_rect = rect.downsample(bi.downsample);
            if !outside_current_frame {
                let frame_size = (
                    frame_size.0.shrc(bi.downsample.0),
                    frame_size.1.shrc(bi.downsample.1),
                );
                channel_rect = channel_rect.clip(frame_size);
                if bi.after_extend {
                    // clip this rect to its visible area in the full image (in full image coordinates).
                    let origin = (
                        rect.origin.0 as isize + frame_origin.0,
                        rect.origin.1 as isize + frame_origin.1,
                    );
                    let end = (
                        origin.0 + rect.size.0 as isize,
                        origin.1 + rect.size.1 as isize,
                    );
                    let origin = (origin.0.max(0) as usize, origin.1.max(0) as usize);
                    let end = (
                        end.0.min(full_image_size.0 as isize).max(0) as usize,
                        end.1.min(full_image_size.1 as isize).max(0) as usize,
                    );
                    channel_rect = Rect {
                        origin,
                        size: (
                            end.0.saturating_sub(origin.0),
                            end.1.saturating_sub(origin.1),
                        ),
                    };
                }
            }
            if channel_rect.size.0 == 0 || channel_rect.size.1 == 0 {
                // Buffer would be empty anyway.
                continue;
            }
            let channel_rect = bi.orientation.display_rect(channel_rect, full_image_size);
            let channel_rect = channel_rect.to_byte_rect_sz(bi.byte_size);
            local_buffers[i] = Some(buf.rect(channel_rect));
        }
        local_buffers
    }

    pub fn get_full_buffers(&mut self) -> &mut [Option<JxlOutputBuffer<'b>>] {
        &mut *self.0
    }
}

/// Computes the channel rect for a given save buffer info and image rect.
/// Shared between BufferSplitter::get_local_buffers and SharedOutputBuffers::get_local_buffers.
#[cfg(feature = "threads")]
fn compute_channel_rect(
    bi: &SaveStageBufferInfo,
    rect: Rect,
    frame_size: (usize, usize),
    full_image_size: (usize, usize),
    frame_origin: (isize, isize),
) -> Option<Rect> {
    let mut channel_rect = rect.downsample(bi.downsample);
    let ds_frame_size = (
        frame_size.0.shrc(bi.downsample.0),
        frame_size.1.shrc(bi.downsample.1),
    );
    channel_rect = channel_rect.clip(ds_frame_size);
    if bi.after_extend {
        let origin = (
            rect.origin.0 as isize + frame_origin.0,
            rect.origin.1 as isize + frame_origin.1,
        );
        let end = (
            origin.0 + rect.size.0 as isize,
            origin.1 + rect.size.1 as isize,
        );
        let origin = (origin.0.max(0) as usize, origin.1.max(0) as usize);
        let end = (
            end.0.min(full_image_size.0 as isize).max(0) as usize,
            end.1.min(full_image_size.1 as isize).max(0) as usize,
        );
        channel_rect = Rect {
            origin,
            size: (
                end.0.saturating_sub(origin.0),
                end.1.saturating_sub(origin.1),
            ),
        };
    }
    if channel_rect.size.0 == 0 || channel_rect.size.1 == 0 {
        return None;
    }
    let channel_rect = bi.orientation.display_rect(channel_rect, full_image_size);
    Some(channel_rect.to_byte_rect_sz(bi.byte_size))
}

/// Thread-safe access to output buffers for parallel rendering.
/// Wraps shared views of the output buffers so multiple threads can
/// create non-overlapping sub-views concurrently.
///
/// # Safety
/// The backing memory of the output buffers must remain valid while this struct exists.
/// Callers must ensure all sub-views access non-overlapping regions.
#[cfg(feature = "threads")]
pub(crate) struct SharedOutputBuffers {
    views: Vec<Option<SharedOutputView>>,
}

#[cfg(feature = "threads")]
#[allow(unsafe_code)]
impl SharedOutputBuffers {
    /// Creates shared output buffers from the buffer splitter's full buffers.
    ///
    /// # Safety
    /// - The buffer splitter's backing data must outlive the returned SharedOutputBuffers
    /// - The buffer splitter's buffers must not be accessed while SharedOutputBuffers exists
    pub(crate) unsafe fn from_buffer_splitter(splitter: &mut BufferSplitter) -> Self {
        let buffers = splitter.get_full_buffers();
        Self {
            views: buffers
                .iter()
                // SAFETY: Guaranteed by the safety contract of from_buffer_splitter.
                .map(|buf| buf.as_ref().map(|b| unsafe { b.shared_view() }))
                .collect(),
        }
    }

    /// Creates local buffer sub-views for a given rect.
    /// Same logic as BufferSplitter::get_local_buffers but for parallel use.
    ///
    /// # Safety
    /// The rect must not overlap with any other rect being concurrently processed.
    pub(crate) unsafe fn get_local_buffers(
        &self,
        save_buffer_info: &[Option<SaveStageBufferInfo>],
        rect: Rect,
        frame_size: (usize, usize),
        full_image_size: (usize, usize),
        frame_origin: (isize, isize),
    ) -> Vec<Option<JxlOutputBuffer<'_>>> {
        let rect = rect.clip(frame_size);
        let mut local_buffers: Vec<Option<JxlOutputBuffer<'_>>> =
            Vec::with_capacity(self.views.len());
        for _ in 0..self.views.len() {
            local_buffers.push(None);
        }
        for (i, (info, view)) in save_buffer_info.iter().zip(self.views.iter()).enumerate() {
            let Some(bi) = info else {
                continue;
            };
            let Some(view) = view else {
                continue;
            };
            let Some(channel_rect) =
                compute_channel_rect(bi, rect, frame_size, full_image_size, frame_origin)
            else {
                continue;
            };
            // SAFETY: Caller ensures non-overlapping access across threads.
            local_buffers[i] = Some(unsafe { view.sub_view(channel_rect) });
        }
        local_buffers
    }
}
