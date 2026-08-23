// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use crate::{
    api::JxlOutputBuffer,
    error::Result,
    image::{DataTypeTag, OwnedRawImage, Rect},
    render::{
        internal::{ChannelInfo, Stage},
        low_memory_pipeline::{helpers::get_distinct_indices, run_stage::ExtraInfo},
    },
    util::{ShiftRightCeil, SmallVec, mirror, tracing_wrappers::*},
};

use super::{GroupRenderContext, PipelineReadView, row_buffers::RowBuffer};

// Most images have at most 7 channels (RGBA + noise extra channels).
// 8 gives a bit extra leeway and makes the size a power of two.
pub(super) type ChannelVec<T> = SmallVec<[T; 8]>;

fn apply_x_padding(
    input_type: DataTypeTag,
    row: &mut [u8],
    to_pad: Range<isize>,
    valid_pixels: Range<isize>,
) {
    let x0_offset = RowBuffer::x0_byte_offset() as isize;
    let num_valid = valid_pixels.clone().count();
    let sz = input_type.size();
    match sz {
        1 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx) as usize;
                let to = (x0_offset + x) as usize;
                row[to] = row[from];
            }
        }
        2 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * 2) as usize;
                let to = (x0_offset + x * 2) as usize;
                row[to] = row[from];
                row[to + 1] = row[from + 1];
            }
        }
        4 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * 4) as usize;
                let to = (x0_offset + x * 4) as usize;
                row[to] = row[from];
                row[to + 1] = row[from + 1];
                row[to + 2] = row[from + 2];
                row[to + 3] = row[from + 3];
            }
        }
        _ => {
            // Generic fallback for any other element size (e.g. 8-byte F64).
            // The 1/2/4 arms above are fast paths; this replaces a previous
            // unimplemented!() that panicked. Not reachable via the public
            // output formats (U8/U16/F16/F32 -> 1/2/2/4 bytes), but kept
            // panic-free and correct for any size. (#9)
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * sz as isize) as usize;
                let to = (x0_offset + x * sz as isize) as usize;
                row.copy_within(from..from + sz, to);
            }
        }
    }
}

/// Copies the input rows of one channel into the first row buffer, for one
/// rendered rectangle.
///
/// Everything that does not depend on the row -- which neighbour buffers
/// are read, the byte offsets and lengths of the left/centre/right copies,
/// the padding past the image edge -- is computed once per rectangle in
/// [`BufferFiller::new`]; [`BufferFiller::fill`] then only does the copies
/// (jxl-rs f94cc26). Before, all of it was re-derived for every row of
/// every channel, which was the single largest piece of per-row overhead
/// in the pipeline.
struct BufferFiller<'a> {
    c: usize,
    ty: DataTypeTag,
    group_y0: usize,
    /// Added to `y - group_y0` for rows above the group: the row index in
    /// the top neighbour's buffer (its centre data with direct borders, the
    /// bottom half of its `topbottom` buffer otherwise).
    top_y_offset: usize,
    bot_y_offset: usize,
    /// Source buffers, `[row kind (top/centre/bottom)][left/centre/right]`.
    images: [Option<&'a OwnedRawImage>; 9],
    copy_byte_offset_initial: usize,
    /// Per row kind: byte offset of the left copy inside the left buffer's
    /// row (its width differs between centre data, `topbottom` and
    /// `leftright` buffers).
    src_byte_offset_left: [usize; 3],
    to_copy_left: usize,
    copy_start: usize,
    copy_end: usize,
    to_copy_right: usize,
    padding_range: Option<(isize, isize)>,
}

impl<'a> BufferFiller<'a> {
    /// `yrange` is the range of channel rows (in the channel's own
    /// resolution) that [`BufferFiller::fill`] will be called with; it
    /// decides which neighbour buffers are needed, and those must be present.
    fn new(
        view: &'a PipelineReadView,
        c: usize,
        (x0, xsize): (usize, usize),
        (gx, gy): (usize, usize),
        yrange: Range<usize>,
    ) -> Option<Self> {
        if !view.shared.channel_is_used[c] || yrange.is_empty() {
            return None;
        }
        let ChannelInfo {
            ty,
            downsample: (dx, dy),
        } = view.shared.channel_info[0][c];
        let ty = ty.expect("Channel info should be populated at this point");
        let group_ysize = 1 << (view.shared.log_group_size - dy as usize);
        let group_xsize = 1 << (view.shared.log_group_size - dx as usize);

        let (bx, by) = view.border_size;

        let group_y0 = gy * group_ysize;
        let group_x0 = gx << (view.shared.log_group_size - dx as usize);
        let group_x1 = group_x0 + group_xsize;
        let gw = view.shared.group_count.0;
        let gid = gy * gw + gx;

        // With direct borders (one-shot parallel decode, every group's centre
        // data stays alive), neighbour rows are read from the neighbours'
        // centre buffers; otherwise from the borders extracted into their
        // `topbottom` / `leftright` buffers.
        let direct_borders = view.input_buffers[gid].topbottom[c].is_none();
        let top_y_offset = if direct_borders {
            group_ysize
        } else {
            (by >> dy) * 4
        };
        let bot_y_offset = group_y0 + group_ysize;

        let has_top = yrange.start < group_y0;
        let has_center = yrange.start < bot_y_offset && yrange.end > group_y0;
        let has_bot = yrange.end > bot_y_offset;

        let copy_x0 = x0.saturating_sub(view.input_border_pixels[c].0);
        let copy_x1 =
            (x0 + xsize + view.input_border_pixels[c].0).min(view.shared.input_size.0.shrc(dx));
        debug_assert!(copy_x1 >= group_x0);

        let copy_byte_offset_initial = RowBuffer::x0_byte_offset() - (x0 - copy_x0) * ty.size();

        let has_left = copy_x0 < group_x0;
        let has_right = copy_x1 > group_x1;
        let to_copy_left = if has_left {
            (group_x0 - copy_x0) * ty.size()
        } else {
            0
        };

        let mut images: [Option<&'a OwnedRawImage>; 9] = [None; 9];
        let mut src_byte_offset_left = [0usize; 3];
        // (row kind, group row) pairs that will be read.
        let rows = [
            (0usize, has_top.then(|| gy - 1)),
            (1, has_center.then_some(gy)),
            (2, has_bot.then(|| gy + 1)),
        ];
        for (kind, igy) in rows {
            let Some(igy) = igy else { continue };
            let base_gid = igy * gw + gx;
            let is_topbottom = kind != 1;
            // Buffer of group `g` holding this row kind, and its width in
            // pixels (only needed for the left neighbour's offset).
            let buffer = |g: usize| -> (&'a OwnedRawImage, usize) {
                let b = &view.input_buffers[g];
                if direct_borders {
                    let buf = b.data[c].as_ref().unwrap();
                    (buf, buf.byte_size().0 / ty.size())
                } else if is_topbottom {
                    (b.topbottom[c].as_ref().unwrap(), group_xsize)
                } else {
                    (b.leftright[c].as_ref().unwrap(), 4 * (bx >> dx))
                }
            };
            if has_left {
                let (buf, xs) = buffer(base_gid - 1);
                images[kind * 3] = Some(buf);
                src_byte_offset_left[kind] = xs * ty.size() - to_copy_left;
            }
            images[kind * 3 + 1] = Some(if is_topbottom && !direct_borders {
                view.input_buffers[base_gid].topbottom[c].as_ref().unwrap()
            } else {
                view.input_buffers[base_gid].data[c].as_ref().unwrap()
            });
            if has_right {
                images[kind * 3 + 2] = Some(buffer(base_gid + 1).0);
            }
        }

        let copy_start = copy_x0.saturating_sub(group_x0) * ty.size();
        let copy_end = (copy_x1.min(group_x1) - group_x0) * ty.size();

        let (to_copy_right, padding_range) = if has_right {
            let next_group_xsize = view.shared.group_size(gid + 1).0.shrc(dx);
            let border_x = (copy_x1 - group_x1).min(next_group_xsize);
            let pad = if border_x + group_x1 < copy_x1 {
                let pad_from = (xsize + border_x) as isize;
                let pad_to = (xsize + copy_x1 - group_x1) as isize;
                Some((pad_from, pad_to))
            } else {
                None
            };
            (border_x * ty.size(), pad)
        } else {
            (0, None)
        };

        Some(Self {
            c,
            ty,
            group_y0,
            top_y_offset,
            bot_y_offset,
            images,
            copy_byte_offset_initial,
            src_byte_offset_left,
            to_copy_left,
            copy_start,
            copy_end,
            to_copy_right,
            padding_range,
        })
    }

    #[inline]
    fn fill(&self, ctx: &mut GroupRenderContext, y: usize) {
        let (kind, input_y) = if y < self.group_y0 {
            (0, y + self.top_y_offset - self.group_y0)
        } else if y >= self.bot_y_offset {
            (2, y - self.bot_y_offset)
        } else {
            (1, y - self.group_y0)
        };
        let base = kind * 3;
        let output_row = ctx.row_buffers[0][self.c].get_row_mut::<u8>(y);
        let mut copy_byte_offset = self.copy_byte_offset_initial;

        if let Some(left) = self.images[base] {
            let input_row = left.row(input_y);
            let src = self.src_byte_offset_left[kind];
            output_row[copy_byte_offset..copy_byte_offset + self.to_copy_left]
                .copy_from_slice(&input_row[src..src + self.to_copy_left]);
            copy_byte_offset += self.to_copy_left;
        }

        let center = self.images[base + 1].expect("row kind resolved in BufferFiller::new");
        let input_row = center.row(input_y);
        let to_copy = self.copy_end - self.copy_start;
        output_row[copy_byte_offset..copy_byte_offset + to_copy]
            .copy_from_slice(&input_row[self.copy_start..self.copy_end]);
        copy_byte_offset += to_copy;

        if let Some(right) = self.images[base + 2] {
            let input_row = right.row(input_y);
            output_row[copy_byte_offset..copy_byte_offset + self.to_copy_right]
                .copy_from_slice(&input_row[..self.to_copy_right]);
            if let Some((pad_from, pad_to)) = self.padding_range {
                apply_x_padding(self.ty, output_row, pad_from..pad_to, 0..pad_from);
            }
        }
    }
}

/// Channel row (in the channel's own resolution) that the render loop
/// below fills for virtual row `vy`, if any. Shared by the loop and by
/// the per-rectangle setup in [`BufferFiller::new`] so both agree exactly.
#[inline]
fn input_row_for_vy(
    view: &PipelineReadView,
    c: usize,
    vy: usize,
    y0: usize,
    num_extra_rows: usize,
) -> Option<usize> {
    let dy = view.shared.channel_info[0][c].downsample.1;
    let scaled_y_border = view.input_border_pixels[c].1 << dy;
    let stage_vy = vy as isize - num_extra_rows as isize + scaled_y_border as isize;
    if stage_vy % (1 << dy) != 0 {
        return None;
    }
    if stage_vy - (y0 as isize) < -(scaled_y_border as isize) {
        return None;
    }
    let y = stage_vy >> dy;
    // Do not produce rows in out-of-bounds areas.
    if y < 0 || y >= view.shared.input_size.1.shrc(dy) as isize {
        return None;
    }
    Some(y as usize)
}

// Renders *parts* of group's worth of data.
// In particular, renders the sub-rectangle given in `image_area`, where (1, 1) refers to
// the center of the group, and 0 and 2 include data from the neighbouring group (if any).
#[instrument(skip(ctx, view, buffers))]
pub(crate) fn render(
    ctx: &mut GroupRenderContext,
    view: &PipelineReadView,
    (gx, gy): (usize, usize),
    image_area: Rect,
    buffers: &mut [Option<JxlOutputBuffer>],
) -> Result<()> {
    let start_of_row = image_area.origin.0 == 0;
    let end_of_row = image_area.end().0 == view.shared.input_size.0;

    let Rect {
        origin: (x0, y0),
        size: (xsize, num_rows),
    } = image_area;

    let num_channels = view.shared.num_channels();
    let num_extra_rows = view.border_size.1;

    // This follows the same implementation strategy as the C++ code in libjxl.
    // We pretend that every stage has a vertical shift of 0, i.e. it is as tall
    // as the final image.
    // We call each such row a "virtual" row, because it may or may not correspond
    // to an actual row of the current processing stage; actual processing happens
    // when vy % (1<<vshift) == 0.

    let vy0 = y0.saturating_sub(num_extra_rows);
    let vy1 = image_area.end().1 + num_extra_rows;

    // Per-channel input copy plans. The channel rows the loop below fills
    // are a contiguous range; find its ends with the loop's own predicate.
    let fillers: ChannelVec<Option<BufferFiller>> = (0..num_channels)
        .map(|c| {
            let first = (vy0..vy1).find_map(|vy| input_row_for_vy(view, c, vy, y0, num_extra_rows));
            let last = (vy0..vy1)
                .rev()
                .find_map(|vy| input_row_for_vy(view, c, vy, y0, num_extra_rows));
            let yrange = match (first, last) {
                (Some(first), Some(last)) => first..last + 1,
                _ => 0..0,
            };
            let dx = view.shared.channel_info[0][c].downsample.0;
            BufferFiller::new(view, c, (x0 >> dx, xsize >> dx), (gx, gy), yrange)
        })
        .collect();

    for vy in vy0..vy1 {
        let mut current_origin = (0, 0);
        let mut current_size = view.shared.input_size;

        // Step 1: read input channels.
        for (c, filler) in fillers.iter().enumerate() {
            let Some(filler) = filler else {
                continue;
            };
            let Some(y) = input_row_for_vy(view, c, vy, y0, num_extra_rows) else {
                continue;
            };
            filler.fill(ctx, y);
        }
        // Step 2: go through stages one by one.
        for (i, stage) in view.shared.stages.iter().enumerate() {
            let (dx, dy) = view.downsampling_for_stage[i];
            // The logic below uses *virtual* y coordinates, so we need to convert the border
            // amount appropriately.
            let scaled_y_border = view.stage_output_border_pixels[i].1 << dy;
            // I knew the reason behind this formula at some point, but now I don't.
            let stage_vy = vy as isize - num_extra_rows as isize + scaled_y_border as isize;
            if stage_vy % (1 << dy) != 0 {
                continue;
            }
            if stage_vy - (y0 as isize) < -(scaled_y_border as isize) {
                continue;
            }
            let y = stage_vy >> dy;
            let shifted_ysize = view.shared.input_size.1.shrc(dy);
            // Do not produce rows in out-of-bounds areas.
            if y < 0 || y >= shifted_ysize as isize {
                continue;
            }
            let y = y as usize;

            let out_extra_x = view.stage_output_border_pixels[i].0;
            let shifted_xsize = xsize.shrc(dx);

            match stage {
                Stage::InPlace(s) => {
                    let mut buffers =
                        get_distinct_indices(&mut ctx.row_buffers, &view.sorted_buffer_indices[i]);
                    s.run_stage_on(
                        ExtraInfo {
                            xsize: shifted_xsize,
                            current_row: y,
                            group_x0: x0 >> dx,
                            out_extra_x,
                            start_of_row,
                            end_of_row,
                            image_height: shifted_ysize,
                        },
                        &mut buffers,
                        ctx.local_states[i].as_deref_mut(),
                    );
                }
                Stage::Save(s) => {
                    // Find buffers for channels that will be saved.
                    // Channel ordering is handled in stage_input_buffer_index construction.
                    let mut input_data: ChannelVec<_> = view.stage_input_buffer_index[i]
                        .iter()
                        .map(|(si, ci)| &ctx.row_buffers[*si][*ci])
                        .collect();
                    // Append opaque alpha buffer if fill_opaque_alpha is set
                    if let Some(ref alpha_buf) = view.opaque_alpha_buffers[i] {
                        input_data.push(alpha_buf);
                    }
                    s.save_lowmem(
                        &input_data,
                        &mut *buffers,
                        (xsize >> dx, num_rows >> dy),
                        y,
                        (x0 >> dx, y0 >> dy),
                        current_size,
                        current_origin,
                    )?;
                }
                Stage::Extend(s) => {
                    current_size = s.image_size;
                    current_origin = s.frame_origin;
                }
                Stage::InOut(s) => {
                    let borderx = s.border().0 as usize;
                    let bordery = s.border().1 as isize;
                    // Apply x padding where the rectangle being rendered touches
                    // the image edge. Keying this on the group index (gx == 0,
                    // gx + 1 == count) was wrong for the second-to-last group when
                    // its rectangle reached the right edge because the last column
                    // is narrower than the border (jxl-rs #845), and would be wrong
                    // under any scheduler that hands a group a rectangle it does
                    // not start (jxl-rs 43e2db6).
                    if start_of_row && borderx != 0 {
                        for (si, ci) in view.stage_input_buffer_index[i].iter() {
                            for iy in -bordery..=bordery {
                                let y = mirror(y as isize + iy, shifted_ysize);
                                apply_x_padding(
                                    s.input_type(),
                                    ctx.row_buffers[*si][*ci].get_row_mut::<u8>(y),
                                    -(borderx as isize)..0,
                                    // Either xsize is the actual size of the image, or it is
                                    // much larger than borderx, so this works out either way.
                                    0..shifted_xsize as isize,
                                );
                            }
                        }
                    }
                    if end_of_row && borderx != 0 {
                        for (si, ci) in view.stage_input_buffer_index[i].iter() {
                            for iy in -bordery..=bordery {
                                let y = mirror(y as isize + iy, shifted_ysize);
                                apply_x_padding(
                                    s.input_type(),
                                    ctx.row_buffers[*si][*ci].get_row_mut::<u8>(y),
                                    shifted_xsize as isize..(shifted_xsize + borderx) as isize,
                                    // borderx..0 is either data from the neighbouring group or
                                    // data that was filled in by the iteration above.
                                    -(borderx as isize)..shifted_xsize as isize,
                                );
                            }
                        }
                    }
                    let (inb, outb) = ctx.row_buffers.split_at_mut(i + 1);
                    // Prepare pointers to input and output buffers.
                    let input_data: ChannelVec<_> = view.stage_input_buffer_index[i]
                        .iter()
                        .map(|(si, ci)| &inb[*si][*ci])
                        .collect();
                    s.run_stage_on(
                        ExtraInfo {
                            xsize: shifted_xsize,
                            current_row: y,
                            group_x0: x0 >> dx,
                            out_extra_x,
                            start_of_row,
                            end_of_row,
                            image_height: shifted_ysize,
                        },
                        &input_data,
                        &mut outb[0][..],
                        ctx.local_states[i].as_deref_mut(),
                    );
                }
            }
        }
    }
    Ok(())
}

// Renders a chunk of data outside the current frame.
#[instrument(skip(ctx, view, buffers))]
pub(super) fn render_outside(
    ctx: &mut GroupRenderContext,
    view: &PipelineReadView,
    xrange: Range<usize>,
    yrange: Range<usize>,
    buffers: &mut [Option<JxlOutputBuffer>],
) -> Result<()> {
    let num_channels = view.shared.num_channels();
    let x0 = xrange.start;
    let y0 = yrange.start;
    let xsize = xrange.clone().count();
    let ysize = yrange.clone().count();
    // Significantly simplified version of render_group.
    for y in yrange.clone() {
        let extend = view.shared.extend_stage_index.unwrap();
        // Step 1: get padding from extend stage.
        for c in 0..num_channels {
            let (si, ci) = view.stage_input_buffer_index[extend][c];
            let buffer = &mut ctx.row_buffers[si][ci];
            let Stage::Extend(extend) = &view.shared.stages[extend] else {
                unreachable!("extend stage is not an extend stage");
            };
            let row = &mut buffer.get_row_mut(y)[RowBuffer::x0_offset::<f32>()..];
            extend.process_row_chunk((x0, y), xsize, c, row);
        }
        // Step 2: go through remaining stages one by one.
        for (i, stage) in view.shared.stages.iter().enumerate().skip(extend + 1) {
            assert_eq!(view.downsampling_for_stage[i], (0, 0));

            match stage {
                Stage::InPlace(s) => {
                    let mut buffers =
                        get_distinct_indices(&mut ctx.row_buffers, &view.sorted_buffer_indices[i]);
                    s.run_stage_on(
                        ExtraInfo {
                            xsize,
                            current_row: y,
                            group_x0: x0,
                            out_extra_x: 0,
                            start_of_row: false,
                            end_of_row: false,
                            image_height: view.shared.input_size.1,
                        },
                        &mut buffers,
                        ctx.local_states[i].as_deref_mut(),
                    );
                }
                Stage::Save(s) => {
                    // Find buffers for channels that will be saved.
                    // Channel ordering is handled in stage_input_buffer_index construction.
                    let mut input_data: ChannelVec<_> = view.stage_input_buffer_index[i]
                        .iter()
                        .map(|(si, ci)| &ctx.row_buffers[*si][*ci])
                        .collect();
                    // Append opaque alpha buffer if fill_opaque_alpha is set
                    if let Some(ref alpha_buf) = view.opaque_alpha_buffers[i] {
                        input_data.push(alpha_buf);
                    }
                    s.save_lowmem(
                        &input_data,
                        &mut *buffers,
                        (xsize, ysize),
                        y,
                        (x0, y0),
                        (xrange.end, yrange.end), // this is not true, but works out correctly.
                        (0, 0),
                    )?;
                }
                Stage::Extend(_) => {
                    unreachable!("duplicate extend stage");
                }
                Stage::InOut(s) => {
                    assert_eq!(s.border(), (0, 0));
                    let (inb, outb) = ctx.row_buffers.split_at_mut(i + 1);
                    // Prepare pointers to input and output buffers.
                    let input_data: ChannelVec<_> = view.stage_input_buffer_index[i]
                        .iter()
                        .map(|(si, ci)| &inb[*si][*ci])
                        .collect();
                    s.run_stage_on(
                        ExtraInfo {
                            xsize,
                            current_row: y,
                            group_x0: x0,
                            out_extra_x: 0,
                            start_of_row: false,
                            end_of_row: false,
                            image_height: view.shared.input_size.1,
                        },
                        &input_data,
                        &mut outb[0][..],
                        ctx.local_states[i].as_deref_mut(),
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod apply_x_padding_tests {
    use super::apply_x_padding;
    use crate::image::DataTypeTag;
    use crate::render::low_memory_pipeline::row_buffers::RowBuffer;
    use crate::util::mirror;

    /// 8-byte (F64) elements aren't reachable through the public output formats,
    /// but `apply_x_padding` must still mirror-pad any element size without
    /// panicking — it previously hit `unimplemented!()`. (#9)
    #[test]
    fn apply_x_padding_handles_eight_byte_element() {
        let x0 = RowBuffer::x0_byte_offset();
        let sz = DataTypeTag::F64.size();
        assert_eq!(sz, 8);

        let valid = 0isize..2;
        let num_valid = valid.clone().count();
        let pad = 2isize..4;

        // x0 padding + 4 pixels of `sz` bytes each.
        let mut row = vec![0u8; x0 + 4 * sz];
        for p in 0..2usize {
            for b in 0..sz {
                row[x0 + p * sz + b] = (p as u8) * 16 + b as u8 + 1;
            }
        }

        // Must not panic.
        apply_x_padding(DataTypeTag::F64, &mut row, pad.clone(), valid.clone());

        // Each padded pixel must equal its mirrored source pixel, byte for byte.
        for x in pad {
            let sx = mirror(x - valid.start, num_valid) as isize + valid.start;
            assert!(
                (0..num_valid as isize).contains(&sx),
                "mirror source in range"
            );
            for b in 0..sz {
                assert_eq!(
                    row[x0 + x as usize * sz + b],
                    row[x0 + sx as usize * sz + b],
                    "padded pixel {x} byte {b} must mirror valid pixel {sx}"
                );
            }
        }
    }
}
