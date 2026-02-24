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
/// Shared between BufferSplitter::get_local_buffers and ParallelOutputAccess::get_local_buffers.
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

/// Safe parallel access to output buffers for multi-threaded rendering.
///
/// Created via [`BufferSplitter::parallel_access`], which exclusively borrows
/// the buffer splitter — the compiler prevents any other access while this
/// guard exists. Internally uses raw pointers (like [`DisjointRowAccess`]) but
/// exposes a safe API. Debug builds track outstanding rects and assert no overlap.
///
/// [`DisjointRowAccess`]: crate::image::disjoint::DisjointRowAccess
#[cfg(feature = "threads")]
pub(crate) struct ParallelOutputAccess<'a> {
    views: Vec<Option<SharedOutputView>>,
    #[cfg(debug_assertions)]
    outstanding: std::sync::Mutex<Vec<(usize, Rect)>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

// SAFETY: All fields are Sync (SharedOutputView has manual Sync impl,
// Mutex is Sync, PhantomData is Sync). The raw pointers inside
// SharedOutputView are derived from the exclusively-borrowed BufferSplitter
// and only used to create non-overlapping sub-views.
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
unsafe impl Sync for ParallelOutputAccess<'_> {}

#[cfg(feature = "threads")]
#[allow(unsafe_code)]
impl ParallelOutputAccess<'_> {
    /// Creates local buffer sub-views for a given image rect.
    ///
    /// Each work item's rect must be non-overlapping with all other concurrent
    /// rects. This invariant is structurally guaranteed by `emit_work_items`,
    /// which partitions the image into non-overlapping regions via the serial
    /// `is_ready` assignment protocol.
    ///
    /// Debug builds verify non-overlap at runtime; release builds compile the
    /// check out (same pattern as `DisjointRowAccess`).
    pub(crate) fn get_local_buffers(
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
            #[cfg(debug_assertions)]
            {
                let mut outstanding = self.outstanding.lock().unwrap();
                for &(ch, ref existing) in outstanding.iter() {
                    if ch == i {
                        assert!(
                            !rects_overlap(existing, &channel_rect),
                            "ParallelOutputAccess: overlapping rects on channel {i}: \
                             existing {existing:?} vs new {channel_rect:?}"
                        );
                    }
                }
                outstanding.push((i, channel_rect));
            }
            // SAFETY: The work items are structurally non-overlapping (guaranteed
            // by emit_work_items' serial is_ready protocol). Debug builds verify
            // this above. The raw pointer is valid for the buffer splitter's
            // lifetime (enforced by the PhantomData lifetime on this struct).
            local_buffers[i] = Some(unsafe { view.sub_view(channel_rect) });
        }
        local_buffers
    }

    /// Clears the debug overlap tracker. Call after all parallel work items
    /// for a batch have completed (i.e., after `par_iter().try_for_each()`
    /// returns).
    #[cfg(debug_assertions)]
    pub(crate) fn clear_tracking(&self) {
        self.outstanding.lock().unwrap().clear();
    }

    /// No-op in release builds.
    #[cfg(not(debug_assertions))]
    pub(crate) fn clear_tracking(&self) {}
}

#[cfg(all(feature = "threads", debug_assertions))]
fn rects_overlap(a: &Rect, b: &Rect) -> bool {
    let a_x1 = a.origin.0 + a.size.0;
    let a_y1 = a.origin.1 + a.size.1;
    let b_x1 = b.origin.0 + b.size.0;
    let b_y1 = b.origin.1 + b.size.1;
    a.origin.0 < b_x1 && b.origin.0 < a_x1 && a.origin.1 < b_y1 && b.origin.1 < a_y1
}

#[cfg(feature = "threads")]
#[allow(unsafe_code)]
impl<'a, 'b> BufferSplitter<'a, 'b> {
    /// Creates a parallel output access guard that exclusively borrows this
    /// buffer splitter. The compiler prevents any other use of the splitter
    /// while the guard exists.
    pub(crate) fn parallel_access(&mut self) -> ParallelOutputAccess<'_> {
        let buffers = self.get_full_buffers();
        ParallelOutputAccess {
            views: buffers
                .iter()
                // SAFETY: The returned ParallelOutputAccess borrows `self` for
                // its entire lifetime, so the backing data remains valid and
                // no other code can access these buffers concurrently.
                .map(|buf| buf.as_ref().map(|b| unsafe { b.shared_view() }))
                .collect(),
            #[cfg(debug_assertions)]
            outstanding: std::sync::Mutex::new(Vec::new()),
            _marker: std::marker::PhantomData,
        }
    }
}
