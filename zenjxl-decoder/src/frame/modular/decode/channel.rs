// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::common::precompute_references;
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::{Histograms, PlainCursor, SymbolReader, unpack_signed},
    entropy_coding::fused_prefix::{FusedPrefixLut, Residuals},
    error::Result,
    frame::modular::{
        IMAGE_OFFSET, IMAGE_PADDING, ModularChannel, Tree,
        decode::{
            common::make_pixel,
            specialized_trees::{TreeSpecialCase, specialize_tree},
        },
        predict::{PredictionData, WeightedPredictorState, clamped_gradient},
        tree::{NUM_NONREF_PROPERTIES, PROPERTIES_PER_PREVCHAN, predict},
    },
    headers::modular::GroupHeader,
    image::Image,
    util::{IsolatedBuf, tracing_wrappers::*},
};
use whereat::at;

const SMALL_CHANNEL_THRESHOLD: usize = 64;

// General case decoder, for small buffers for which it's not worth trying to detect tree special cases.
#[inline(never)]
fn decode_modular_channel_small(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    let size = buffers[chan].data.size();
    let mut wp_state = WeightedPredictorState::new(&header.wp_header, size.0)?;
    let mut num_ref_props = tree
        .max_property_count()
        .saturating_sub(NUM_NONREF_PROPERTIES);
    // The precompute_references function stores 4 values per reference property (offset + 0,1,2,3)
    num_ref_props = num_ref_props.div_ceil(PROPERTIES_PER_PREVCHAN) * PROPERTIES_PER_PREVCHAN;
    let mut references = Image::<i32>::new((num_ref_props, size.0))?;
    let num_properties = NUM_NONREF_PROPERTIES + num_ref_props;

    const { assert!(IMAGE_OFFSET.1 == 2) };

    let mut property_buffer = IsolatedBuf::<i32>::new(num_properties);
    property_buffer[0] = chan as i32;
    property_buffer[1] = stream_id as i32;

    for y in 0..size.1 {
        precompute_references(buffers, chan, y, &mut references);
        // Reset non-static properties for each row (keep chan/stream_id in [0] and [1])
        property_buffer[2..].fill(0);
        let [row, row_top, row_toptop] =
            buffers[chan].data.distinct_full_rows_mut([y + 2, y + 1, y]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let row_top = &mut row_top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let row_toptop = &mut row_toptop[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        for x in 0..size.0 {
            let prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let prediction_result = predict(
                &tree.nodes,
                prediction_data,
                size.0,
                Some(&mut wp_state),
                x,
                y,
                &references,
                &mut property_buffer,
                u32::MAX, // cold path: compute all properties
            );
            let dec = reader.read_signed(&tree.histograms, br, prediction_result.context as usize);
            let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
            row[x] = val;
            wp_state.update_errors(val, (x, y), size.0);
        }
    }

    Ok(())
}

/// Where a channel decoder's symbols come from: the regular reader, or a
/// register-resident [`PlainCursor`] for streams without LZ77.
pub(super) trait SymbolSource {
    /// `SymbolReader::read_signed_clustered_inline`.
    fn read_signed(&mut self, histograms: &Histograms, cluster: usize) -> i32;
    /// `SymbolReader::read_signed_clustered_config_420`; requires
    /// `histograms.can_use_config_420_fast_path()`.
    fn read_signed_420(&mut self, histograms: &Histograms, cluster: usize) -> i32;
}

struct ReaderSource<'r, 'b, 'a> {
    reader: &'r mut SymbolReader,
    br: &'b mut BitReader<'a>,
}

impl SymbolSource for ReaderSource<'_, '_, '_> {
    #[inline(always)]
    fn read_signed(&mut self, histograms: &Histograms, cluster: usize) -> i32 {
        self.reader
            .read_signed_clustered_inline(histograms, self.br, cluster)
    }

    #[inline(always)]
    fn read_signed_420(&mut self, histograms: &Histograms, cluster: usize) -> i32 {
        self.reader
            .read_signed_clustered_config_420(histograms, self.br, cluster)
    }
}

/// Reads through a [`PlainCursor`]; near the end of the data, one symbol at
/// a time goes through the regular reader.
struct CursorSource<'r, 'b, 'a> {
    cursor: PlainCursor<'a>,
    reader: &'r mut SymbolReader,
    br: &'b mut BitReader<'a>,
}

/// The cursor goes in and out by value: a `&mut` to it would keep it in
/// memory for the whole channel loop.
#[cold]
#[inline(never)]
fn read_signed_slow<'a>(
    cursor: PlainCursor<'a>,
    reader: &mut SymbolReader,
    br: &mut BitReader<'a>,
    histograms: &Histograms,
    cluster: usize,
    config_420: bool,
) -> (i32, PlainCursor<'a>) {
    reader.finish_plain_cursor(cursor, br);
    let v = if config_420 {
        reader.read_signed_clustered_config_420(histograms, br, cluster)
    } else {
        reader.read_signed_clustered_inline(histograms, br, cluster)
    };
    (v, reader.plain_cursor(br).unwrap())
}

impl CursorSource<'_, '_, '_> {
    #[inline(always)]
    fn slow(&mut self, histograms: &Histograms, cluster: usize, config_420: bool) -> i32 {
        let (v, cursor) = read_signed_slow(
            self.cursor,
            self.reader,
            self.br,
            histograms,
            cluster,
            config_420,
        );
        self.cursor = cursor;
        v
    }
}

impl SymbolSource for CursorSource<'_, '_, '_> {
    #[inline(always)]
    fn read_signed(&mut self, histograms: &Histograms, cluster: usize) -> i32 {
        match histograms.read_unsigned_plain(&mut self.cursor, cluster) {
            Some(v) => unpack_signed(v),
            None => self.slow(histograms, cluster, false),
        }
    }

    #[inline(always)]
    fn read_signed_420(&mut self, histograms: &Histograms, cluster: usize) -> i32 {
        match histograms.read_unsigned_plain_420(&mut self.cursor, cluster) {
            Some(v) => unpack_signed(v),
            None => self.slow(histograms, cluster, true),
        }
    }
}

struct FusedPrefixSource<'r, 'b, 'a> {
    source: CursorSource<'r, 'b, 'a>,
    lut: &'r FusedPrefixLut,
    cluster: usize,
}

impl FusedPrefixSource<'_, '_, '_> {
    #[inline(always)]
    fn read(&mut self, histograms: &Histograms, allow_pair: bool) -> Residuals {
        match self.source.cursor.read_fused_signed(self.lut, allow_pair) {
            Some(residuals) => residuals,
            None => Residuals {
                first: self.source.read_signed(histograms, self.cluster),
                second: None,
            },
        }
    }
}

/// Single-Gradient reconstruction needs the previous row and the current
/// left value. Keep the entropy cursor in a by-value source for the channel.
#[inline(never)]
fn decode_fused_prefix<'r, 'b, 'a>(
    channel: &mut ModularChannel,
    source: FusedPrefixSource<'r, 'b, 'a>,
    histograms: &Histograms,
) -> CursorSource<'r, 'b, 'a> {
    let mut source = source;
    let (width, height) = channel.data.size();
    for y in 0..height {
        let [row, top] = channel.data.distinct_full_rows_mut([y + 2, y + 1]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + width];
        let mut x = 0;
        if y == 0 {
            let mut left = 0i32;
            while x < width {
                let residuals = source.read(histograms, x + 1 < width);
                left = left.wrapping_add(residuals.first);
                row[x] = left;
                x += 1;
                if let Some(second) = residuals.second {
                    left = left.wrapping_add(second);
                    row[x] = left;
                    x += 1;
                }
            }
            continue;
        }
        let top = &top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + width];
        let mut left = top[0];
        let mut top_left = top[0];
        while x < width {
            let residuals = source.read(histograms, x + 1 < width);
            let above = top[x];
            let guess = clamped_gradient(left.into(), above.into(), top_left.into());
            left = residuals.first.wrapping_add(guess as i32);
            row[x] = left;
            top_left = above;
            x += 1;
            if let Some(second) = residuals.second {
                let above = top[x];
                let guess = clamped_gradient(left.into(), above.into(), top_left.into());
                left = second.wrapping_add(guess as i32);
                row[x] = left;
                top_left = above;
                x += 1;
            }
        }
    }
    source.source
}

struct FusedRow<'a> {
    row: &'a mut [i32],
    top: &'a [i32],
    x: usize,
    left: i32,
    top_left: i32,
}

impl<'a> FusedRow<'a> {
    fn new(channel: &'a mut ModularChannel, y: usize) -> Self {
        let width = channel.data.size().0;
        let [row, top] = channel.data.distinct_full_rows_mut([y + 2, y + 1]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + width];
        let top = &top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + width];
        let initial = if y == 0 { 0 } else { top[0] };
        Self {
            row,
            top,
            x: 0,
            left: initial,
            top_left: initial,
        }
    }

    #[inline(always)]
    fn push<const FIRST: bool>(&mut self, residual: i32) {
        if FIRST {
            self.left = self.left.wrapping_add(residual);
        } else {
            let above = self.top[self.x];
            let guess = clamped_gradient(self.left.into(), above.into(), self.top_left.into());
            self.left = residual.wrapping_add(guess as i32);
            self.top_left = above;
        }
        self.row[self.x] = self.left;
        self.x += 1;
    }

    #[inline(always)]
    fn step<const FIRST: bool>(&mut self, source: &mut FusedPrefixSource, histograms: &Histograms) {
        let residuals = source.read(histograms, self.x + 1 < self.row.len());
        self.push::<FIRST>(residuals.first);
        if let Some(second) = residuals.second {
            self.push::<FIRST>(second);
        }
    }
}

#[inline(always)]
fn interleave_rows<const FIRST: bool>(
    mut a: FusedRow<'_>,
    source_a: &mut FusedPrefixSource,
    hist_a: &Histograms,
    mut b: FusedRow<'_>,
    source_b: &mut FusedPrefixSource,
    hist_b: &Histograms,
) {
    while a.x < a.row.len() && b.x < b.row.len() {
        a.step::<FIRST>(source_a, hist_a);
        b.step::<FIRST>(source_b, hist_b);
    }
    while a.x < a.row.len() {
        a.step::<FIRST>(source_a, hist_a);
    }
    while b.x < b.row.len() {
        b.step::<FIRST>(source_b, hist_b);
    }
}

// Own both cursors here so the hot loop can keep them in registers.
#[inline(never)]
fn decode_fused_prefix_pair(
    a: &mut ModularChannel,
    mut source_a: FusedPrefixSource,
    hist_a: &Histograms,
    b: &mut ModularChannel,
    mut source_b: FusedPrefixSource,
    hist_b: &Histograms,
) -> Result<()> {
    interleave_rows::<true>(
        FusedRow::new(a, 0),
        &mut source_a,
        hist_a,
        FusedRow::new(b, 0),
        &mut source_b,
        hist_b,
    );
    for y in 1..a.data.size().1 {
        interleave_rows::<false>(
            FusedRow::new(a, y),
            &mut source_a,
            hist_a,
            FusedRow::new(b, y),
            &mut source_b,
            hist_b,
        );
    }
    let a = source_a.source;
    let b = source_b.source;
    a.reader.finish_plain_cursor(a.cursor, a.br);
    b.reader.finish_plain_cursor(b.cursor, b.br);
    a.br.check_for_error().map_err(|e| at!(e))?;
    b.br.check_for_error().map_err(|e| at!(e))
}

pub(super) struct ChannelStream<'r, 'data> {
    pub id: usize,
    pub header: &'r GroupHeader,
    pub tree: &'r Tree,
    pub reader: &'r mut SymbolReader,
    pub bits: &'r mut BitReader<'data>,
}

impl<'r, 'data> ChannelStream<'r, 'data> {
    fn fused(
        self,
        chan: usize,
        size: (usize, usize),
    ) -> Result<Option<FusedPrefixSource<'r, 'r, 'data>>> {
        if size.0 <= IMAGE_PADDING.0 || size.0 * size.1 < 4096 || self.tree.histograms.uses_lz77() {
            return Ok(None);
        }
        let TreeSpecialCase::SingleGradientOnly(t) =
            specialize_tree(self.tree, chan, self.id, size.0, self.header)?
        else {
            return Ok(None);
        };
        let cluster = t.clustered_ctx();
        let Some(lut) = self.tree.fused_prefix(cluster) else {
            return Ok(None);
        };
        let Some(cursor) = self.reader.plain_cursor(self.bits) else {
            return Ok(None);
        };
        Ok(Some(FusedPrefixSource {
            source: CursorSource {
                cursor,
                reader: self.reader,
                br: self.bits,
            },
            lut,
            cluster,
        }))
    }
}

pub(super) fn try_decode_channel_pair(
    chan: usize,
    a: &mut ModularChannel,
    stream_a: ChannelStream<'_, '_>,
    b: &mut ModularChannel,
    stream_b: ChannelStream<'_, '_>,
) -> Result<bool> {
    let size_a = a.data.size();
    let size_b = b.data.size();
    if size_a.1 != size_b.1 {
        return Ok(false);
    }
    let hist_a = &stream_a.tree.histograms;
    let hist_b = &stream_b.tree.histograms;
    let (Some(source_a), Some(source_b)) =
        (stream_a.fused(chan, size_a)?, stream_b.fused(chan, size_b)?)
    else {
        return Ok(false);
    };
    debug_assert_eq!(a.data.offset(), IMAGE_OFFSET);
    debug_assert_eq!(b.data.offset(), IMAGE_OFFSET);
    decode_fused_prefix_pair(a, source_a, hist_a, b, source_b, hist_b)?;
    Ok(true)
}

pub(super) trait ModularChannelDecoder {
    const NEEDS_TOP: bool;
    const NEEDS_TOPTOP: bool;
    /// Whether to read through a [`PlainCursor`] when the stream allows it.
    /// On only for single-leaf trees: measured 2026-10-02 (M4 Pro), it made
    /// the fast-decode preset 34% faster single-threaded, but learned-tree
    /// lossless e7 and libjxl's gradient DC tree 10% and 3% slower on 12
    /// threads, with no single-threaded change.
    const USE_CURSOR: bool = false;
    fn init_row(&mut self, buffers: &mut [&mut ModularChannel], chan: usize, y: usize);
    fn decode_one<S: SymbolSource>(
        &mut self,
        prediction_data: PredictionData,
        pos: (usize, usize),
        xsize: usize,
        src: &mut S,
        histograms: &Histograms,
    ) -> i32;
}

#[inline(never)]
fn decode_modular_channel_impl<D: ModularChannelDecoder>(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    decoder: D,
    reader: &mut SymbolReader,
    br: &mut BitReader,
    histograms: &Histograms,
) -> Result<()> {
    #[cfg(test)]
    let use_cursor =
        D::USE_CURSOR && !tests::DISABLE_CURSOR.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    let use_cursor = D::USE_CURSOR;
    match reader.plain_cursor(br).filter(|_| use_cursor) {
        Some(cursor) => {
            #[cfg(test)]
            tests::CURSOR_USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let src = channel_loop(
                buffers,
                chan,
                decoder,
                CursorSource { cursor, reader, br },
                histograms,
            );
            src.reader.finish_plain_cursor(src.cursor, src.br);
        }
        None => {
            channel_loop(
                buffers,
                chan,
                decoder,
                ReaderSource { reader, br },
                histograms,
            );
        }
    }
    Ok(())
}

/// Takes and returns the symbol source by value so a cursor in it can live
/// in registers.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // Iterator chain (.enumerate().skip().take()) measured 1% overhead
fn channel_loop<D: ModularChannelDecoder, S: SymbolSource>(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    mut decoder: D,
    src: S,
    histograms: &Histograms,
) -> S {
    let mut src = src;
    let size = buffers[chan].data.size();
    debug_assert!(size.0 >= 4);
    // Wide one-row channels come here too (`082c50ed`): rows above y = 0
    // are read only through the padding rows, and `get_rows` handles y < 2.
    debug_assert!(size.1 >= 1);

    const { assert!(IMAGE_OFFSET.1 == 2) };

    // Let the compiler decide whether inlining in the borders is worth it.
    let do_decode_cold = |decoder: &mut D, prediction_data, pos, src: &mut S| {
        decoder.decode_one(prediction_data, pos, size.0, src, histograms)
    };

    for y in 0..size.1 {
        decoder.init_row(buffers, chan, y);
        let [row, row_top, row_toptop] =
            buffers[chan].data.distinct_full_rows_mut([y + 2, y + 1, y]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let row_top = &mut row_top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let row_toptop = &mut row_toptop[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let mut last = 0;
        let mut prediction_data = PredictionData::default();
        for x in 0..2 {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let val = do_decode_cold(&mut decoder, prediction_data, (x, y), &mut src);
            row[x] = val;
            last = val;
        }
        if y < 2 {
            for x in 2..size.0 - 2 {
                let prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
                let val = do_decode_cold(&mut decoder, prediction_data, (x, y), &mut src);
                row[x] = val;
            }
        } else {
            for x in 2..size.0 - 2 {
                prediction_data = prediction_data.update_for_interior_row(
                    row_top,
                    row_toptop,
                    x,
                    last,
                    D::NEEDS_TOP,
                    D::NEEDS_TOPTOP,
                );
                let val = decoder.decode_one(prediction_data, (x, y), size.0, &mut src, histograms);
                row[x] = val;
                last = val;
            }
        }
        for x in size.0 - 2..size.0 {
            prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let val = do_decode_cold(&mut decoder, prediction_data, (x, y), &mut src);
            row[x] = val;
        }
    }
    src
}

#[instrument(level = "debug", skip(buffers, reader, tree))]
pub(super) fn decode_modular_channel(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    decode_modular_channel_inner(buffers, chan, stream_id, header, tree, reader, br)?;
    // Stop at the channel that overran the section instead of decoding the
    // remaining channels from padding. Upstream jxl-rs b0ae00b.
    br.check_for_error().map_err(|e| at!(e))
}

fn decode_modular_channel_inner(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    crate::profile!(modular_decode);
    debug!("reading channel");
    let size = buffers[chan].data.size();
    // Short channels are fine for the specialised paths, which read rows
    // above y = 0 only through the padding (and `get_rows` handles y < 2);
    // VarDCT's AC-metadata channel is two rows of one sample per block, so
    // routing every channel with height <= 2 here sent 20-60K samples per
    // frame through the property-computing generic loop.
    if size.0 <= IMAGE_PADDING.0 || size.0 * size.1 <= SMALL_CHANNEL_THRESHOLD {
        return decode_modular_channel_small(buffers, chan, stream_id, header, tree, reader, br);
    }

    assert_eq!(buffers[chan].data.padding().1, IMAGE_PADDING.1);
    assert!(buffers[chan].data.padding().0 >= IMAGE_PADDING.0);
    assert_eq!(buffers[chan].data.offset(), IMAGE_OFFSET);

    // We now know the channel is wider than IMAGE_PADDING.0 and not tiny.

    let special_tree = specialize_tree(tree, chan, stream_id, size.0, header)?;
    match special_tree {
        TreeSpecialCase::NoTree(t) => {
            if let Some(v) = t.single_value_for_fill() {
                // Constant channel: no bits to read (jxl-rs #787).
                let img = &mut buffers[chan].data;
                for y in 0..size.1 {
                    img.row_mut(y).fill(v);
                }
                return Ok(());
            }
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::NoWp(t) => {
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::NoWp420(t) => {
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::WpOnlyConfig420(t) => {
            #[cfg(test)]
            if tests::DISABLE_WP_ROWS.load(std::sync::atomic::Ordering::Relaxed) {
                return decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms);
            }
            #[cfg(test)]
            tests::WP_ROWS_USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            t.decode_channel(buffers[chan], reader, br, &tree.histograms);
            Ok(())
        }
        TreeSpecialCase::WpOnly(t) => {
            #[cfg(test)]
            if tests::DISABLE_WP_ROWS.load(std::sync::atomic::Ordering::Relaxed) {
                return decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms);
            }
            #[cfg(test)]
            tests::WP_ROWS_USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            t.decode_channel(buffers[chan], reader, br, &tree.histograms);
            Ok(())
        }
        TreeSpecialCase::GradientLookupConfig420(t) => {
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::GradientLookup(t) => {
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::SingleGradientOnly(t) => {
            if super::rle_gradient::try_decode(
                buffers[chan],
                t.clustered_ctx(),
                reader,
                br,
                &tree.histograms,
            ) {
                return Ok(());
            }
            #[cfg(test)]
            let enable_fused = !tests::DISABLE_FUSED.load(std::sync::atomic::Ordering::Relaxed)
                && !tests::DISABLE_CURSOR.load(std::sync::atomic::Ordering::Relaxed);
            #[cfg(not(test))]
            let enable_fused = true;
            // Amortize table construction on smaller streams by keeping the
            // regular cursor for channels below one 64×64 block of samples.
            if enable_fused
                && size.0 * size.1 >= 4096
                && let Some(cursor) = reader.plain_cursor(br)
                && let Some(lut) = tree.fused_prefix(t.clustered_ctx())
            {
                #[cfg(test)]
                tests::FUSED_USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let fused = FusedPrefixSource {
                    source: CursorSource { cursor, reader, br },
                    lut,
                    cluster: t.clustered_ctx(),
                };
                let src = decode_fused_prefix(buffers[chan], fused, &tree.histograms);
                src.reader.finish_plain_cursor(src.cursor, src.br);
                return Ok(());
            }
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::General(t) => {
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
        TreeSpecialCase::General420(t) => {
            decode_modular_channel_impl(buffers, chan, t, reader, br, &tree.histograms)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test-only switch that sends every channel through the regular reader.
    pub(super) static DISABLE_CURSOR: AtomicBool = AtomicBool::new(false);
    /// Channels decoded through a `CursorSource`.
    pub(super) static CURSOR_USES: AtomicUsize = AtomicUsize::new(0);
    pub(super) static DISABLE_FUSED: AtomicBool = AtomicBool::new(false);
    pub(super) static FUSED_USES: AtomicUsize = AtomicUsize::new(0);
    static PREFIX_TEST_LOCK: Mutex<()> = Mutex::new(());
    /// Test-only switch that sends weighted-predictor-only channels through
    /// the per-sample `decode_one` path instead of `decode_row`.
    pub(super) static DISABLE_WP_ROWS: AtomicBool = AtomicBool::new(false);
    /// Channels decoded through `WpOnlyLookupConfig420::decode_channel`.
    pub(super) static WP_ROWS_USES: AtomicUsize = AtomicUsize::new(0);

    /// Decodes `data` with the switch off and on and compares the frames.
    fn fast_matches_generic(name: &str, data: &[u8], switch: &AtomicBool, uses: &AtomicUsize) {
        use crate::api::decoder::tests::decode;
        let before = uses.load(Ordering::Relaxed);
        let (_, fast) = decode(data, usize::MAX, false, false, None).unwrap();
        assert!(
            uses.load(Ordering::Relaxed) > before,
            "{name}: fast path not used"
        );
        switch.store(true, Ordering::Relaxed);
        let generic = decode(data, usize::MAX, false, false, None);
        switch.store(false, Ordering::Relaxed);
        let (_, generic) = generic.unwrap();
        assert_eq!(fast.len(), generic.len());
        for (f, g) in fast[0].iter().zip(&generic[0]) {
            assert_eq!(f.size(), g.size());
            for y in 0..f.size().1 {
                assert!(f.row(y) == g.row(y), "{name}: row {y} differs");
            }
        }
    }

    /// libjxl v0.12 files whose VarDCT DC uses the weighted-predictor-only
    /// tree: JPEG transcodes with 4/2/0 configs (photo crops, q40: 768x512
    /// 4:2:0 and 512x384 4:4:4) and `cjxl -d 3 -e 7` with other configs
    /// (512x384). `decode_row` must match the per-sample path.
    #[test]
    fn wp_rows_match_per_sample_path() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/wp-rows");
        for name in ["jpeg420.jxl", "jpeg444.jxl", "d3e7.jxl"] {
            let data = std::fs::read(dir.join(name)).unwrap();
            fast_matches_generic(name, &data, &DISABLE_WP_ROWS, &WP_ROWS_USES);
        }
    }

    #[test]
    fn fused_prefix_matches_plain_cursor() {
        let _lock = PREFIX_TEST_LOCK.lock().unwrap();
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/plain-cursor");
        for name in ["gradient_rgb8.jxl", "gradient_gray16.jxl"] {
            let data = std::fs::read(dir.join(name)).unwrap();
            fast_matches_generic(name, &data, &DISABLE_FUSED, &FUSED_USES);
        }
    }

    /// jxl-encoder single-leaf lossless files (prefix codes, no LZ77):
    /// Gradient leaves (`--fast-decode`, 200x96 RGB8 and 272x96 16-bit
    /// gray, two groups across) and a Zero leaf (`-P 0`, 104x36). Each section's
    /// last samples go through the regular reader. Output must match the
    /// regular reader's exactly.
    #[test]
    fn plain_cursor_matches_reader_path() {
        use crate::api::decoder::tests::decode;
        let _lock = PREFIX_TEST_LOCK.lock().unwrap();
        DISABLE_FUSED.store(true, Ordering::Relaxed);
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/plain-cursor");
        for name in ["gradient_rgb8.jxl", "gradient_gray16.jxl", "zero_rgb8.jxl"] {
            let data = std::fs::read(dir.join(name)).unwrap();
            let before = CURSOR_USES.load(Ordering::Relaxed);
            let (_, fast) = decode(&data, usize::MAX, false, false, None).unwrap();
            assert!(
                CURSOR_USES.load(Ordering::Relaxed) > before,
                "{name}: cursor not used"
            );
            DISABLE_CURSOR.store(true, Ordering::Relaxed);
            let generic = decode(&data, usize::MAX, false, false, None);
            DISABLE_CURSOR.store(false, Ordering::Relaxed);
            let (_, generic) = generic.unwrap();
            assert_eq!(fast.len(), generic.len());
            for (f, g) in fast[0].iter().zip(&generic[0]) {
                assert_eq!(f.size(), g.size());
                for y in 0..f.size().1 {
                    assert!(f.row(y) == g.row(y), "{name}: row {y} differs");
                }
            }
        }
        DISABLE_FUSED.store(false, Ordering::Relaxed);
    }
}
