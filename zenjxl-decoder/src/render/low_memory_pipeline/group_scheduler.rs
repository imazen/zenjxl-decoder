// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use crate::error::Result;
use crate::image::{OwnedRawImage, Rect};
use crate::render::LowMemoryRenderPipeline;
use crate::render::buffer_splitter::BufferSplitter;
use crate::render::internal::{ChannelInfo, RenderPipelineShared, Stage};
use crate::util::tracing_wrappers::*;

use super::{PipelineReadView, render_group, row_buffers::RowBuffer};

/// A renderable sub-rect of a group, produced by `prepare_group`.
/// The parallel render path collects these across multiple groups
/// and renders them concurrently.
pub(crate) struct RenderWorkItem {
    pub(crate) gx: usize,
    pub(crate) gy: usize,
    pub(crate) image_area: Rect,
}

/// Pure computation of renderable work items for a single group.
/// Image-space rectangle rendered by group `(gx, gy)` for the readiness
/// sub-range `xrange x yrange` (0..3 each: left/top border strip, centre,
/// right/bottom border strip), or `None` when the rectangle is empty.
///
/// Every coordinate is clamped to `input_size`: a last group column narrower
/// than the border (image width = k * group_dim + 1..border) would otherwise
/// make the *second-to-last* group's rectangle extend past the image, so its
/// stages ran over phantom columns without edge padding (the padding used to
/// be keyed on the group index, see render_group.rs) and, for a tiny last
/// group, `x1 - x0` underflowed into an absurd rectangle that sent
/// `mirror()` into an endless loop. Upstream jxl-rs #845 / #873.
fn ready_image_area(
    group_rect: Rect,
    (gx, gy): (usize, usize),
    group_count: (usize, usize),
    input_size: (usize, usize),
    border_size: (usize, usize),
    xrange: Range<u8>,
    yrange: Range<u8>,
) -> Option<Rect> {
    let y0 = match (gy == 0, yrange.start) {
        (true, 0) => group_rect.origin.1,
        (false, 0) => group_rect.origin.1.saturating_sub(border_size.1),
        (_, 1) => group_rect.origin.1 + border_size.1,
        // (_, 2)
        _ => group_rect.end().1.saturating_sub(border_size.1),
    };
    let x0 = match (gx == 0, xrange.start) {
        (true, 0) => group_rect.origin.0,
        (false, 0) => group_rect.origin.0.saturating_sub(border_size.0),
        (_, 1) => group_rect.origin.0 + border_size.0,
        // (_, 2)
        _ => group_rect.end().0.saturating_sub(border_size.0),
    };
    let y1 = match (gy + 1 == group_count.1, yrange.end) {
        (true, 3) => group_rect.end().1,
        (false, 3) => group_rect.end().1 + border_size.1,
        (_, 2) => group_rect.end().1.saturating_sub(border_size.1),
        // (_, 1)
        _ => group_rect.origin.1 + border_size.1,
    }
    .min(input_size.1);
    let x1 = match (gx + 1 == group_count.0, xrange.end) {
        (true, 3) => group_rect.end().0,
        (false, 3) => group_rect.end().0 + border_size.0,
        (_, 2) => group_rect.end().0.saturating_sub(border_size.0),
        // (_, 1)
        _ => group_rect.origin.0 + border_size.0,
    }
    .min(input_size.0);
    // `then` (lazy), not `then_some`: the argument of `then_some` is evaluated
    // before the condition is consulted, so an empty strip (`x1 <= x0`, e.g. a
    // 3-px last column inside a 7-px border) underflowed `x1 - x0` -- a panic
    // under overflow checks, a wrapped-then-discarded value in release.
    (x1 > x0 && y1 > y0).then(|| Rect {
        origin: (x0, y0),
        size: (x1 - x0, y1 - y0),
    })
}

/// Output rectangle of group `(gx, gy)` when every group is ready.
///
/// The readiness-mask rectangles of a group straddle its boundaries: the
/// strip `xrange == 0..1` is `[origin - border, origin + border)` and
/// `2..3` is `[end - border, end + border)`, so the strip between two
/// neighbours belongs to both of them. With all nine neighbours ready the
/// mask is all-true and `ready_image_area(.., 0..3, 0..3)` covers the
/// group *plus* every straddling strip, which rendered those strips twice
/// (~3% extra work for a 2-px border) and, worse, made the per-group output
/// rectangles overlap, so `Frame::decode_groups_parallel` could not hand
/// each group a disjoint fragment of the output and fell back to rendering
/// into temporary buffers plus a sequential copy of the whole output.
///
/// Instead each group owns the strips straddling its *left* and *top*
/// boundaries (`xrange`/`yrange` starting at 0) and leaves the ones at its
/// right and bottom to the next group (`..2`), except for the last column /
/// row which reaches the image edge (`..3`). These rectangles tile the
/// image exactly, and each one only reads `border` pixels past its end,
/// i.e. data that belongs to a neighbour that is present, or is padded
/// because the rectangle touches the image edge -- the same geometry as
/// the mask path, so no render-time padding rule changes.
fn full_readiness_area(
    group_rect: Rect,
    (gx, gy): (usize, usize),
    group_count: (usize, usize),
    input_size: (usize, usize),
    border_size: (usize, usize),
) -> Option<Rect> {
    let xrange = 0..if gx + 1 == group_count.0 { 3 } else { 2 };
    let yrange = 0..if gy + 1 == group_count.1 { 3 } else { 2 };
    ready_image_area(
        group_rect,
        (gx, gy),
        group_count,
        input_size,
        border_size,
        xrange,
        yrange,
    )
}

/// How `compute_work_items` is being called when every group is ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FullReadiness {
    /// Not every group is ready: use the readiness masks. (Only the
    /// parallel path asks for work items with partial readiness; the
    /// sequential path walks the mask itself in `render_with_new_group`.)
    #[cfg_attr(not(feature = "threads"), allow(dead_code))]
    No,
    /// Every group is ready. The payload says whether *every* group is
    /// rendered in this pass (`true`: one-shot decode, or the final
    /// re-render) or only the groups of the last incremental batch.
    Some(bool),
}

/// All `is_ready` flags must be set before calling. No mutation needed.
///
/// When `full_readiness` is true, skips readiness mask computation and
/// directly emits a single full-rect work item per group. This is the
/// common case for one-shot parallel decode where all groups are ready.
pub(super) fn compute_work_items(
    g: usize,
    is_ready: &[bool],
    shared: &RenderPipelineShared<RowBuffer>,
    border_size: (usize, usize),
    full_readiness: FullReadiness,
) -> Result<Vec<RenderWorkItem>> {
    let (gx, gy) = shared.group_position(g);
    let gsz = 1 << shared.log_group_size;
    let group_rect = Rect {
        size: (gsz, gsz),
        origin: (gsz * gx, gsz * gy),
    }
    .clip(shared.input_size);
    let group_count = shared.group_count;

    // Fast path: when all groups are ready, every group gets a single work
    // item. Skip readiness mask computation and foreach_ready_rect entirely.
    //
    // When every group is rendered in this pass the items tile the image
    // (see `full_readiness_area`). When only the groups of the last
    // incremental batch are rendered, a group must also produce the strips
    // straddling its right/bottom boundaries: the neighbour there may have
    // arrived in an earlier batch, when this group was not ready, so nobody
    // else rendered that strip (jxlrs-845 fixture streamed in small chunks
    // had an unwritten band at x = 254..258). That is the all-true mask
    // rectangle -- the group rectangle extended by the border on every
    // side, overlapping its neighbours' rectangles.
    match full_readiness {
        FullReadiness::Some(true) => {
            return Ok(full_readiness_area(
                group_rect,
                (gx, gy),
                group_count,
                shared.input_size,
                border_size,
            )
            .map(|image_area| RenderWorkItem { gx, gy, image_area })
            .into_iter()
            .collect());
        }
        FullReadiness::Some(false) => {
            return Ok(ready_image_area(
                group_rect,
                (gx, gy),
                group_count,
                shared.input_size,
                border_size,
                0..3,
                0..3,
            )
            .map(|image_area| RenderWorkItem { gx, gy, image_area })
            .into_iter()
            .collect());
        }
        FullReadiness::No => {}
    }

    let gxm1 = gx.saturating_sub(1);
    let gym1 = gy.saturating_sub(1);
    let gxp1 = (gx + 1).min(group_count.0 - 1);
    let gyp1 = (gy + 1).min(group_count.1 - 1);
    let gw = group_count.0;
    let mut ready_mask = [
        is_ready[gym1 * gw + gxm1],
        is_ready[gym1 * gw + gx],
        is_ready[gym1 * gw + gxp1],
        is_ready[gy * gw + gxm1],
        is_ready[gy * gw + gx],
        is_ready[gy * gw + gxp1],
        is_ready[gyp1 * gw + gxm1],
        is_ready[gyp1 * gw + gx],
        is_ready[gyp1 * gw + gxp1],
    ];
    ready_mask[0] &= ready_mask[1];
    ready_mask[0] &= ready_mask[3];
    ready_mask[2] &= ready_mask[1];
    ready_mask[2] &= ready_mask[5];
    ready_mask[6] &= ready_mask[3];
    ready_mask[6] &= ready_mask[7];
    ready_mask[8] &= ready_mask[5];
    ready_mask[8] &= ready_mask[7];

    let mut items = Vec::new();
    foreach_ready_rect(ready_mask, |xrange, yrange| {
        if let Some(image_area) = ready_image_area(
            group_rect,
            (gx, gy),
            group_count,
            shared.input_size,
            border_size,
            xrange,
            yrange,
        ) {
            items.push(RenderWorkItem { gx, gy, image_area });
        }
        Ok(())
    })?;
    Ok(items)
}

pub(super) struct InputBuffer {
    // One buffer per channel.
    pub(super) data: Vec<Option<OwnedRawImage>>,
    // Storage for left/right borders. Includes corners.
    pub(super) leftright: Vec<Option<OwnedRawImage>>,
    // Storage for top/bottom borders. Includes corners.
    pub(super) topbottom: Vec<Option<OwnedRawImage>>,
    // Number of ready channels in the current pass.
    pub(super) ready_channels: usize,
    pub(super) is_ready: bool,
    pub(super) num_completed_groups_3x3: usize,
}

impl InputBuffer {
    pub(super) fn has_buffer(&self, chan: usize) -> bool {
        self.data[chan].is_some()
    }

    pub(super) fn set_buffer(&mut self, chan: usize, buf: OwnedRawImage) {
        assert!(self.data[chan].is_none());
        self.data[chan] = Some(buf);
        self.ready_channels += 1;
    }

    pub(super) fn new(num_channels: usize) -> Self {
        let b = || (0..num_channels).map(|_| None).collect();
        Self {
            data: b(),
            leftright: b(),
            topbottom: b(),
            ready_channels: 0,
            is_ready: false,
            num_completed_groups_3x3: 0,
        }
    }
}

// Finds a small set of rectangles that cover all the "true" values in `ready_mask`,
// and calls `f` on each such rectangle.
fn foreach_ready_rect(
    ready_mask: [bool; 9],
    mut f: impl FnMut(Range<u8>, Range<u8>) -> Result<()>,
) -> Result<()> {
    // x range in middle row
    let xrange = (1 - ready_mask[3] as u8)..(2 + ready_mask[5] as u8);
    let can_extend_top = xrange.clone().all(|x| ready_mask[x as usize]);
    let can_extend_bottom = xrange.clone().all(|x| ready_mask[6 + x as usize]);
    let yrange = (1 - can_extend_top as u8)..(2 + can_extend_bottom as u8);
    f(xrange.clone(), yrange)?;

    if !can_extend_top {
        if ready_mask[1] {
            let xrange = (1 - ready_mask[0] as u8)..(2 + ready_mask[2] as u8);
            f(xrange, 0..1)?;
        } else {
            if ready_mask[0] {
                f(0..1, 0..1)?;
            }
            if ready_mask[2] {
                f(2..3, 0..1)?;
            }
        }
    } else {
        if ready_mask[0] && !xrange.contains(&0) {
            f(0..1, 0..1)?;
        }
        if ready_mask[2] && !xrange.contains(&2) {
            f(2..3, 0..1)?;
        }
    }

    if !can_extend_bottom {
        if ready_mask[7] {
            let xrange = (1 - ready_mask[6] as u8)..(2 + ready_mask[8] as u8);
            f(xrange, 2..3)?;
        } else {
            if ready_mask[6] {
                f(0..1, 2..3)?;
            }
            if ready_mask[8] {
                f(2..3, 2..3)?;
            }
        }
    } else {
        if ready_mask[6] && !xrange.contains(&0) {
            f(0..1, 2..3)?;
        }
        if ready_mask[8] && !xrange.contains(&2) {
            f(2..3, 2..3)?;
        }
    }

    Ok(())
}

impl LowMemoryRenderPipeline {
    pub(super) fn maybe_get_scratch_buffer(
        &mut self,
        channel: usize,
        kind: usize,
    ) -> Option<OwnedRawImage> {
        self.scratch_channel_buffers[channel * 3 + kind].pop()
    }

    pub(super) fn store_scratch_buffer(
        &mut self,
        channel: usize,
        kind: usize,
        image: OwnedRawImage,
    ) {
        // Group-sized buffers (kind 0) can be capped: in modular mode a
        // rendered tile is recycled here once and never taken back, so an
        // unbounded cache holds one tile per channel per group until the
        // frame ends (jxl-rs #812).
        if kind == 0
            && let Some(limit) = self.shared.group_scratch_buffers_limit
            && self.scratch_channel_buffers[channel * 3].len() >= limit
        {
            return;
        }
        self.scratch_channel_buffers[channel * 3 + kind].push(image)
    }

    /// Extracts border data from a group and computes renderable work items.
    /// Returns an empty vec if not all channels for the group are ready yet.
    pub(crate) fn prepare_group(&mut self, g: usize) -> Result<Vec<RenderWorkItem>> {
        let buf = &mut self.input_buffers[g];
        assert!(buf.ready_channels <= self.shared.num_used_channels());
        if buf.ready_channels != self.shared.num_used_channels() {
            return Ok(vec![]);
        }
        buf.ready_channels = 0;
        let (gx, gy) = self.shared.group_position(g);
        debug!("new data ready for group {gx},{gy}");

        let gsz = 1 << self.shared.log_group_size;
        let group_rect = Rect {
            size: (gsz, gsz),
            origin: (gsz * gx, gsz * gy),
        }
        .clip(self.shared.input_size);

        {
            for c in 0..self.shared.num_channels() {
                if !self.shared.channel_is_used[c] {
                    continue;
                }
                let (bx, by) = self.border_size;
                let (sx, sy) = self.input_buffers[g].data[c].as_ref().unwrap().byte_size();
                let ChannelInfo {
                    ty,
                    downsample: (dx, dy),
                } = self.shared.channel_info[0][c];
                let ty = ty.unwrap();
                let bx = bx >> dx;
                let by = by >> dy;
                let mut topbottom = if let Some(b) = self.input_buffers[g].topbottom[c].take() {
                    b
                } else if let Some(b) = self.maybe_get_scratch_buffer(c, 1) {
                    b
                } else {
                    let height = 4 * by;
                    let width = (1 << self.shared.log_group_size) * ty.size();
                    OwnedRawImage::new_zeroed_with_padding((width, height), (0, 0), (0, 0))?
                };
                let mut leftright = if let Some(b) = self.input_buffers[g].leftright[c].take() {
                    b
                } else if let Some(b) = self.maybe_get_scratch_buffer(c, 2) {
                    b
                } else {
                    let height = 1 << self.shared.log_group_size;
                    let width = 4 * bx * ty.size();
                    OwnedRawImage::new_zeroed_with_padding((width, height), (0, 0), (0, 0))?
                };
                let input = self.input_buffers[g].data[c].as_ref().unwrap();
                if by != 0 {
                    for y in 0..(2 * by).min(sy) {
                        topbottom.row_mut(y)[..sx].copy_from_slice(input.row(y));
                        topbottom.row_mut(4 * by - 1 - y)[..sx]
                            .copy_from_slice(input.row(sy - y - 1));
                    }
                }
                if bx != 0 {
                    let cs = (bx * 2 * ty.size()).min(sx);
                    for y in 0..sy {
                        let row_out = leftright.row_mut(y);
                        let row_in = input.row(y);
                        row_out[..cs].copy_from_slice(&row_in[..cs]);
                        row_out[4 * bx * ty.size() - cs..].copy_from_slice(&row_in[sx - cs..]);
                    }
                }
                self.input_buffers[g].leftright[c] = Some(leftright);
                self.input_buffers[g].topbottom[c] = Some(topbottom);
            }
        }
        self.input_buffers[g].is_ready = true;

        // Compute readiness mask from 3x3 neighborhood.
        let gxm1 = gx.saturating_sub(1);
        let gym1 = gy.saturating_sub(1);
        let gxp1 = (gx + 1).min(self.shared.group_count.0 - 1);
        let gyp1 = (gy + 1).min(self.shared.group_count.1 - 1);
        let gw = self.shared.group_count.0;
        let mut ready_mask = [
            self.input_buffers[gym1 * gw + gxm1].is_ready,
            self.input_buffers[gym1 * gw + gx].is_ready,
            self.input_buffers[gym1 * gw + gxp1].is_ready,
            self.input_buffers[gy * gw + gxm1].is_ready,
            self.input_buffers[gy * gw + gx].is_ready, // guaranteed true
            self.input_buffers[gy * gw + gxp1].is_ready,
            self.input_buffers[gyp1 * gw + gxm1].is_ready,
            self.input_buffers[gyp1 * gw + gx].is_ready,
            self.input_buffers[gyp1 * gw + gxp1].is_ready,
        ];
        // Corners require both adjacent sides to be ready.
        ready_mask[0] &= ready_mask[1];
        ready_mask[0] &= ready_mask[3];
        ready_mask[2] &= ready_mask[1];
        ready_mask[2] &= ready_mask[5];
        ready_mask[6] &= ready_mask[3];
        ready_mask[6] &= ready_mask[7];
        ready_mask[8] &= ready_mask[5];
        ready_mask[8] &= ready_mask[7];

        // Collect renderable sub-rects.
        let border_size = self.border_size;
        let group_count = self.shared.group_count;
        let mut items = Vec::new();
        foreach_ready_rect(ready_mask, |xrange, yrange| {
            if let Some(image_area) = ready_image_area(
                group_rect,
                (gx, gy),
                group_count,
                self.shared.input_size,
                border_size,
                xrange,
                yrange,
            ) {
                items.push(RenderWorkItem { gx, gy, image_area });
            }
            Ok(())
        })?;

        Ok(items)
    }

    /// Recycles data and border buffers after rendering a group.
    ///
    /// When `recycle_borders` is false, only center data buffers are recycled
    /// (border buffers are kept alive). This is used during adaptive batching
    /// where cross-batch `process_output` stores can re-ready groups whose
    /// neighbors' borders must remain valid.
    pub(crate) fn recycle_group_buffers(&mut self, g: usize, recycle_borders: bool) {
        let (gx, gy) = self.shared.group_position(g);

        // Recycle center data buffers.
        for c in 0..self.input_buffers[g].data.len() {
            if let Some(b) = std::mem::take(&mut self.input_buffers[g].data[c]) {
                self.store_scratch_buffer(c, 0, b);
            }
        }

        if !recycle_borders {
            return;
        }

        // Clear border buffers that will not be used again.
        // This is certainly the case if *all* the groups in the 3x3 group area around
        // the current group are complete.
        if self.shared.group_chan_complete[g].iter().all(|x| *x) {
            let gxm1 = gx.saturating_sub(1);
            let gym1 = gy.saturating_sub(1);
            let gxp1 = (gx + 1).min(self.shared.group_count.0 - 1);
            let gyp1 = (gy + 1).min(self.shared.group_count.1 - 1);
            let gw = self.shared.group_count.0;
            for g in [
                gym1 * gw + gxm1,
                gym1 * gw + gx,
                gym1 * gw + gxp1,
                gy * gw + gxm1,
                gy * gw + gx,
                gy * gw + gxp1,
                gyp1 * gw + gxm1,
                gyp1 * gw + gx,
                gyp1 * gw + gxp1,
            ] {
                self.input_buffers[g].num_completed_groups_3x3 += 1;
                if self.input_buffers[g].num_completed_groups_3x3 != 9 {
                    continue;
                }
                // Reset is_ready: once borders are recycled, this group must
                // not appear "ready" in any neighbor's readiness mask, or
                // rendering would try to read the recycled (None) buffers.
                self.input_buffers[g].is_ready = false;
                for c in 0..self.input_buffers[g].data.len() {
                    if let Some(b) = std::mem::take(&mut self.input_buffers[g].topbottom[c]) {
                        self.store_scratch_buffer(c, 1, b);
                    }
                    if let Some(b) = std::mem::take(&mut self.input_buffers[g].leftright[c]) {
                        self.store_scratch_buffer(c, 2, b);
                    }
                }
            }
        }
    }

    /// Recycles all remaining border buffers (topbottom + leftright) across all
    /// groups. Used after adaptive batching completes to free deferred borders.
    pub(crate) fn recycle_all_borders(&mut self) {
        for buf in &mut self.input_buffers {
            buf.is_ready = false;
            for c in 0..buf.data.len() {
                if let Some(b) = std::mem::take(&mut buf.topbottom[c]) {
                    self.scratch_channel_buffers[c * 3 + 1].push(b);
                }
                if let Some(b) = std::mem::take(&mut buf.leftright[c]) {
                    self.scratch_channel_buffers[c * 3 + 2].push(b);
                }
            }
        }
    }

    /// Sets is_ready and computes renderable work items for a group whose borders
    /// have already been extracted (via `extract_borders`).
    ///
    /// MUST be called serially, in group index order. The is_ready flag is set
    /// here (not in extract_borders) so that when computing the 3×3 readiness
    /// mask, only groups that have already been emitted are marked ready. This
    /// preserves the non-overlapping work-item guarantee: each border pixel is
    /// rendered by exactly one group.
    #[cfg(feature = "threads")]
    #[allow(dead_code)] // Threaded work-item emission for parallel group rendering
    pub(crate) fn emit_work_items(&mut self, g: usize) -> Result<Vec<RenderWorkItem>> {
        // Set is_ready NOW (before computing the mask) so that subsequent groups
        // see this group as ready. Earlier groups were already set in previous
        // iterations of this serial loop.
        self.input_buffers[g].is_ready = true;
        let (gx, gy) = self.shared.group_position(g);

        let gsz = 1 << self.shared.log_group_size;
        let group_rect = Rect {
            size: (gsz, gsz),
            origin: (gsz * gx, gsz * gy),
        }
        .clip(self.shared.input_size);

        // Compute readiness mask from 3x3 neighborhood.
        let gxm1 = gx.saturating_sub(1);
        let gym1 = gy.saturating_sub(1);
        let gxp1 = (gx + 1).min(self.shared.group_count.0 - 1);
        let gyp1 = (gy + 1).min(self.shared.group_count.1 - 1);
        let gw = self.shared.group_count.0;
        let mut ready_mask = [
            self.input_buffers[gym1 * gw + gxm1].is_ready,
            self.input_buffers[gym1 * gw + gx].is_ready,
            self.input_buffers[gym1 * gw + gxp1].is_ready,
            self.input_buffers[gy * gw + gxm1].is_ready,
            self.input_buffers[gy * gw + gx].is_ready, // guaranteed true
            self.input_buffers[gy * gw + gxp1].is_ready,
            self.input_buffers[gyp1 * gw + gxm1].is_ready,
            self.input_buffers[gyp1 * gw + gx].is_ready,
            self.input_buffers[gyp1 * gw + gxp1].is_ready,
        ];
        // Corners require both adjacent sides to be ready.
        ready_mask[0] &= ready_mask[1];
        ready_mask[0] &= ready_mask[3];
        ready_mask[2] &= ready_mask[1];
        ready_mask[2] &= ready_mask[5];
        ready_mask[6] &= ready_mask[3];
        ready_mask[6] &= ready_mask[7];
        ready_mask[8] &= ready_mask[5];
        ready_mask[8] &= ready_mask[7];

        let border_size = self.border_size;
        let group_count = self.shared.group_count;
        let mut items = Vec::new();
        foreach_ready_rect(ready_mask, |xrange, yrange| {
            if let Some(image_area) = ready_image_area(
                group_rect,
                (gx, gy),
                group_count,
                self.shared.input_size,
                border_size,
                xrange,
                yrange,
            ) {
                items.push(RenderWorkItem { gx, gy, image_area });
            }
            Ok(())
        })?;

        Ok(items)
    }

    /// Process a group: prepare borders, render all ready sub-rects, recycle buffers.
    pub(crate) fn render_with_new_group(
        &mut self,
        g: usize,
        buffer_splitter: &mut BufferSplitter,
    ) -> Result<()> {
        let items = self.prepare_group(g)?;
        if items.is_empty() {
            return Ok(());
        }

        let (origin, size) = if let Some(e) = self.shared.extend_stage_index {
            let Stage::Extend(e) = &self.shared.stages[e] else {
                unreachable!("extend stage is not an extend stage");
            };
            (e.frame_origin, e.image_size)
        } else {
            ((0, 0), self.shared.input_size)
        };

        {
            let view = PipelineReadView {
                shared: &self.shared,
                input_buffers: &self.input_buffers,
                stage_input_buffer_index: &self.stage_input_buffer_index,
                downsampling_for_stage: &self.downsampling_for_stage,
                stage_output_border_pixels: &self.stage_output_border_pixels,
                input_border_pixels: &self.input_border_pixels,
                border_size: self.border_size,
                opaque_alpha_buffers: &self.opaque_alpha_buffers,
                sorted_buffer_indices: &self.sorted_buffer_indices,
            };
            let ctx = &mut self.render_ctx;
            let save_buffer_info = &self.save_buffer_info;

            for item in &items {
                let mut local_buffers = buffer_splitter.get_local_buffers(
                    save_buffer_info,
                    item.image_area,
                    false,
                    view.shared.input_size,
                    size,
                    origin,
                );
                render_group::render(
                    ctx,
                    &view,
                    (item.gx, item.gy),
                    item.image_area,
                    &mut local_buffers,
                )?;
            }
        }

        self.recycle_group_buffers(g, !self.skip_border_recycling);

        Ok(())
    }
}

/// Extracts border data for a single group without computing work items.
///
/// This is the per-group workhorse for the parallel border extraction phase.
/// It resets ready_channels, copies border data, and sets is_ready = true.
/// Skips the scratch buffer pool (allocates fresh if needed) for thread safety.
///
/// When `skip_copy` is true, border data is not copied — the rendering phase
/// will read border pixels directly from neighbors' center data buffers.
/// This is only safe when ALL groups' center data remains alive through rendering
/// (i.e., unbatched parallel mode where Phase 3b completes before Phase 3c recycles).
///
/// After all groups have had borders extracted, call `emit_work_items` serially
/// to compute the readiness mask and work items (which reads neighbor is_ready).
#[cfg(feature = "threads")]
pub(super) fn extract_borders(
    buf: &mut InputBuffer,
    shared: &RenderPipelineShared<RowBuffer>,
    border_size: (usize, usize),
    skip_copy: bool,
) -> Result<bool> {
    // Use num_used_channels() — only channels with channel_is_used=true
    // receive data from process_output. Matching prepare_group's check.
    if buf.ready_channels != shared.num_used_channels() {
        return Ok(false);
    }
    buf.ready_channels = 0;

    if skip_copy {
        // Direct borders mode: rendering reads from center data directly.
        // No topbottom/leftright buffers needed.
        return Ok(true);
    }

    // Extract border data from the group's buffers.
    for c in 0..shared.num_channels() {
        if !shared.channel_is_used[c] {
            continue;
        }
        let (bx, by) = border_size;
        let (sx, sy) = buf.data[c].as_ref().unwrap().byte_size();
        let ChannelInfo {
            ty,
            downsample: (dx, dy),
        } = shared.channel_info[0][c];
        let ty = ty.unwrap();
        let bx = bx >> dx;
        let by = by >> dy;
        // Take existing border buffer or allocate fresh (skip scratch pool for
        // thread safety — no shared mutable pool access).
        let mut topbottom = if let Some(b) = buf.topbottom[c].take() {
            b
        } else {
            let height = 4 * by;
            let width = (1 << shared.log_group_size) * ty.size();
            OwnedRawImage::new_zeroed_with_padding((width, height), (0, 0), (0, 0))?
        };
        let mut leftright = if let Some(b) = buf.leftright[c].take() {
            b
        } else {
            let height = 1 << shared.log_group_size;
            let width = 4 * bx * ty.size();
            OwnedRawImage::new_zeroed_with_padding((width, height), (0, 0), (0, 0))?
        };
        let input = buf.data[c].as_ref().unwrap();
        if by != 0 {
            for y in 0..(2 * by).min(sy) {
                topbottom.row_mut(y)[..sx].copy_from_slice(input.row(y));
                topbottom.row_mut(4 * by - 1 - y)[..sx].copy_from_slice(input.row(sy - y - 1));
            }
        }
        if bx != 0 {
            let cs = (bx * 2 * ty.size()).min(sx);
            for y in 0..sy {
                let row_out = leftright.row_mut(y);
                let row_in = input.row(y);
                row_out[..cs].copy_from_slice(&row_in[..cs]);
                row_out[4 * bx * ty.size() - cs..].copy_from_slice(&row_in[sx - cs..]);
            }
        }
        buf.leftright[c] = Some(leftright);
        buf.topbottom[c] = Some(topbottom);
    }
    // NOTE: is_ready is NOT set here. It must be set serially during the
    // emit_work_items phase to preserve the non-overlapping work item guarantee.
    // See emit_work_items for details.

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foreach_ready_rect() {
        for i in 0..512 {
            let mut ready_mask = [false; 9];
            for j in 0..9 {
                if (i >> j) & 1 == 1 {
                    ready_mask[j] = true;
                }
            }
            if !ready_mask[4] {
                continue;
            }

            let mut covered = [false; 9];
            foreach_ready_rect(ready_mask, |xr, yr| {
                for y in yr {
                    for x in xr.clone() {
                        let idx = (y as usize) * 3 + (x as usize);
                        assert!(
                            ready_mask[idx],
                            "Covered not ready index {} in mask {:?} (x={}, y={})",
                            idx, ready_mask, x, y
                        );
                        assert!(
                            !covered[idx],
                            "Double coverage of index {} in mask {:?}",
                            idx, ready_mask
                        );
                        covered[idx] = true;
                    }
                }
                Ok(())
            })
            .unwrap();

            for j in 0..9 {
                if ready_mask[j] {
                    assert!(
                        covered[j],
                        "Failed to cover index {} in mask {:?}",
                        j, ready_mask
                    );
                }
            }
        }
    }

    /// With every group ready, the per-group output rectangles must tile the
    /// image exactly: no pixel rendered twice (overlapping writes force the
    /// parallel render into its copy-back fallback) and none missed. The
    /// all-true readiness-mask rectangle does *not* have this property -- it
    /// extends into every neighbour -- which is why the fast path stopped
    /// using it. Each rectangle may also only reach `border` pixels past its
    /// own edges, and only into a neighbour or past the image edge.
    #[test]
    fn full_readiness_areas_tile_the_image() {
        for &(w, h, log_group_size, border) in &[
            (2333usize, 2333usize, 8usize, 2usize),
            (515, 72, 8, 7),
            (520, 520, 8, 18),
            (256, 256, 8, 3),
            (257, 1, 8, 3),
            (1000, 700, 7, 8),
            (2333, 2333, 8, 0),
        ] {
            let gsz = 1 << log_group_size;
            let (gw, gh) = (w.div_ceil(gsz), h.div_ceil(gsz));
            let mut covered = vec![0u8; w * h];
            let mut old_overlaps = false;
            for gy in 0..gh {
                for gx in 0..gw {
                    let group_rect = Rect {
                        size: (gsz, gsz),
                        origin: (gsz * gx, gsz * gy),
                    }
                    .clip((w, h));
                    let area = full_readiness_area(
                        group_rect,
                        (gx, gy),
                        (gw, gh),
                        (w, h),
                        (border, border),
                    )
                    .unwrap();
                    assert!(area.size.0 > 0 && area.size.1 > 0);
                    for y in area.origin.1..area.end().1 {
                        for x in area.origin.0..area.end().0 {
                            covered[y * w + x] += 1;
                        }
                    }
                    // Reach: at most `border` past the group's own rectangle
                    // on the left/top (into the previous group), and never
                    // past the group's end on the right/bottom.
                    assert!(area.origin.0 + border >= group_rect.origin.0);
                    assert!(area.origin.1 + border >= group_rect.origin.1);
                    assert!(area.end().0 <= group_rect.end().0);
                    assert!(area.end().1 <= group_rect.end().1);
                    // The all-ready mask rectangle reaches into the neighbours.
                    let old = ready_image_area(
                        group_rect,
                        (gx, gy),
                        (gw, gh),
                        (w, h),
                        (border, border),
                        0..3,
                        0..3,
                    )
                    .unwrap();
                    old_overlaps |= old.size.0 * old.size.1 > area.size.0 * area.size.1;
                }
            }
            assert!(
                covered.iter().all(|&c| c == 1),
                "{w}x{h} log_group_size {log_group_size} border {border}: {} pixels missed, \
                 {} rendered twice",
                covered.iter().filter(|&&c| c == 0).count(),
                covered.iter().filter(|&&c| c > 1).count()
            );
            assert_eq!(old_overlaps, gw * gh > 1 && border > 0, "{w}x{h}");
        }
    }

    /// Upstream jxl-rs #873: an 8-px last group with an 18-px border in a
    /// 520-px image. The `(_, 1)` arms alone would put `x1`/`y1` at
    /// 512 + 18 = 530, past the image.
    #[test]
    fn ready_image_area_clips_tiny_edge_group() {
        let area = ready_image_area(
            Rect {
                origin: (512, 512),
                size: (8, 8),
            },
            (1, 1),
            (2, 2),
            (520, 520),
            (18, 18),
            0..1,
            0..1,
        );
        let area = area.unwrap();
        assert_eq!((area.origin, area.size), ((494, 494), (26, 26)));
    }

    /// The second-to-last group's right strip must stop at the image edge
    /// when the last column is narrower than the border (jxl-rs #845):
    /// 515-px image, groups 0..256, 256..512, 512..515, border 7.
    #[test]
    fn ready_image_area_second_to_last_group_stops_at_image_edge() {
        let area = ready_image_area(
            Rect {
                origin: (256, 0),
                size: (256, 72),
            },
            (1, 0),
            (3, 1),
            (515, 72),
            (7, 7),
            0..3,
            0..3,
        )
        .unwrap();
        assert_eq!(area.origin, (249, 0));
        assert_eq!(area.end(), (515, 72));
    }

    /// A tiny last group whose right strip starts past the (clamped) end is
    /// dropped instead of producing an underflowed width.
    #[test]
    fn ready_image_area_drops_empty_rects() {
        let area = ready_image_area(
            Rect {
                origin: (512, 0),
                size: (3, 72),
            },
            (2, 0),
            (3, 1),
            (515, 72),
            (7, 7),
            1..2,
            0..3,
        );
        assert!(
            area.is_none(),
            "centre strip of a 3-px group inside a 7-px border is empty"
        );
    }
}
