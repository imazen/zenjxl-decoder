// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Single-gradient-leaf channels in prefix-coded streams whose LZ77 only
//! repeats the previous symbol: the shape libjxl's effort-1 lossless encoder
//! writes. Counterpart of upstream jxl-rs #797's `decode_fast_lossless`.
//!
//! Reconstruction is the same as `SingleGradientOnly::decode_one`. Edge rules
//! follow `PredictionData::get_rows`: row 0 predicts `left` (0 at x = 0), and
//! x = 0 on later rows predicts `row_top[0]`. Symbols come from an
//! [`RleCursor`] held in registers; near the end of the data the cursor is
//! returned and the regular reader takes over for that sample.

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::{Histograms, RleCode, RleCursor, SymbolReader, unpack_signed},
    entropy_coding::fused_prefix::FusedRleLut,
    frame::modular::{GlobalTree, IMAGE_OFFSET, ModularChannel},
};

/// Reads one sample with the regular reader. The cursor goes in and out by
/// value: a `&mut` to it would make it live in memory for the whole loop.
#[cold]
#[inline(never)]
fn read_slow<'b>(
    cursor: RleCursor<'b>,
    reader: &mut SymbolReader,
    br: &mut BitReader<'b>,
    histograms: &Histograms,
    cluster: usize,
) -> (i32, RleCursor<'b>) {
    reader.finish_rle_cursor(cursor, br);
    let v = reader.read_signed_clustered_inline(histograms, br, cluster);
    (v, reader.rle_cursor(histograms, br).unwrap())
}

struct Side<'a, 'b> {
    code: RleCode<'a>,
    cursor: RleCursor<'b>,
    reader: &'a mut SymbolReader,
    br: &'a mut BitReader<'b>,
    histograms: &'a Histograms,
    cluster: usize,
}

impl Side<'_, '_> {
    #[inline(always)]
    fn read_span(&mut self, limit: usize, fused: Option<&FusedRleLut>) -> (i32, usize) {
        match self.code.read_span(&mut self.cursor, limit, fused) {
            Some((v, count)) => (unpack_signed(v), count),
            None => {
                let (v, cursor) = read_slow(
                    self.cursor,
                    self.reader,
                    self.br,
                    self.histograms,
                    self.cluster,
                );
                self.cursor = cursor;
                (v, 1)
            }
        }
    }
}

/// Decodes `channel` if `reader` is in RLE state with prefix codes; returns
/// false, having read nothing, otherwise.
pub(super) fn try_decode(
    channel: &mut ModularChannel,
    cluster: usize,
    reader: &mut SymbolReader,
    br: &mut BitReader,
    histograms: &Histograms,
) -> bool {
    #[cfg(test)]
    if tests::DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let Some(cursor) = reader.rle_cursor(histograms, br) else {
        return false;
    };
    let Some(code) = histograms.rle_code(cluster) else {
        return false;
    };
    #[cfg(test)]
    tests::USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let side = decode::<false>(
        channel,
        Side {
            code,
            cursor,
            reader,
            br,
            histograms,
            cluster,
        },
        None,
    );
    side.reader.finish_rle_cursor(side.cursor, side.br);
    true
}
pub(super) fn try_decode_fused(
    channel: &mut ModularChannel,
    tree: &GlobalTree,
    channel_id: usize,
    stream_id: usize,
    reader: &mut SymbolReader,
    br: &mut BitReader<'_>,
) -> bool {
    #[cfg(test)]
    if tests::DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let histograms = &tree.tree.histograms;
    let Some(cursor) = reader.rle_cursor(histograms, br) else {
        return false;
    };
    let Some((cluster, fused)) = tree.rle_lut(channel_id, stream_id, &cursor) else {
        return false;
    };
    let Some(code) = histograms.rle_code(cluster) else {
        return false;
    };
    crate::profile!(modular_decode);
    #[cfg(test)]
    tests::USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let side = decode::<true>(
        channel,
        Side {
            code,
            cursor,
            reader,
            br,
            histograms,
            cluster,
        },
        Some(fused),
    );
    side.reader.finish_rle_cursor(side.cursor, side.br);
    true
}

/// Takes and returns the side by value so the cursor stays in registers;
/// behind a `&mut` into this non-inlined function it would be stored and
/// reloaded around every sample.
#[inline(never)]
fn decode<'a, 'b, const FUSED: bool>(
    channel: &mut ModularChannel,
    side: Side<'a, 'b>,
    fused: Option<&FusedRleLut>,
) -> Side<'a, 'b> {
    let fused = if FUSED { fused } else { None };
    const { assert!(IMAGE_OFFSET.1 == 2) };
    // A local copy, unlike the by-pointer argument, can be split into
    // registers.
    let mut side = side;
    let (w, h) = channel.data.size();
    for y in 0..h {
        let [row, row_top] = channel.data.distinct_full_rows_mut([y + 2, y + 1]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + w];
        if y == 0 {
            let mut last = 0i32;
            let mut x = 0;
            while x < w {
                let (residual, count) = side.read_span(w - x, fused);
                let end = x + count;
                if count == 1 {
                    last = last.wrapping_add(residual);
                    row[x] = last;
                } else if residual == 0 {
                    row[x..end].fill(last);
                } else {
                    for p in &mut row[x..end] {
                        last = last.wrapping_add(residual);
                        *p = last;
                    }
                }
                x = end;
            }
            continue;
        }
        let top = &row_top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + w];
        let mut left = top[0];
        let mut topleft = top[0];
        let mut x = 0;
        while x < w {
            let (residual, count) = side.read_span(w - x, fused);
            let end = x + count;
            if count == 1 {
                let t = top[x];
                left = gradient_i32(left, t, topleft).wrapping_add(residual);
                row[x] = left;
                topleft = t;
            } else if residual == 0 && left == topleft {
                // A zero residual with left == top-left predicts top, so
                // the same condition holds throughout this span.
                row[x..end].copy_from_slice(&top[x..end]);
                left = top[end - 1];
                topleft = left;
            } else {
                for (p, &t) in row[x..end].iter_mut().zip(&top[x..end]) {
                    left = gradient_i32(left, t, topleft).wrapping_add(residual);
                    *p = left;
                    topleft = t;
                }
            }
            x = end;
        }
    }
    side
}

#[inline(always)]
fn gradient_i32(left: i32, top: i32, top_left: i32) -> i32 {
    let min = left.min(top);
    let max = left.max(top);
    // With top_left in [min, max], the result also lies in [min, max].
    // Wrapping intermediates therefore give the exact final i32 value.
    let prediction = left.wrapping_add(top).wrapping_sub(top_left);
    let clamped_max = if top_left < min { max } else { prediction };
    if top_left > max { min } else { clamped_max }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test-only switch that sends these channels through the regular
    /// reader instead.
    pub(super) static DISABLE: AtomicBool = AtomicBool::new(false);
    /// Channels decoded here.
    pub(super) static USES: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn gradient_i32_matches_widened_gradient() {
        use crate::frame::modular::predict::clamped_gradient;

        let values = [
            i32::MIN,
            i32::MIN + 1,
            -2,
            -1,
            0,
            1,
            2,
            i32::MAX - 1,
            i32::MAX,
        ];
        for left in values {
            for top in values {
                for top_left in values {
                    assert_eq!(
                        super::gradient_i32(left, top, top_left),
                        clamped_gradient(left.into(), top.into(), top_left.into()) as i32
                    );
                }
            }
        }
    }

    /// libjxl v0.12 `cjxl -e 1 -d 0` files (272x96, two groups across;
    /// RGB8, 16-bit gray, RGBA8): RLE-only LZ77 with prefix codes and one
    /// Gradient leaf per channel. Every group's last samples fall back to
    /// the regular reader near the end of its section. Output must match
    /// the regular reader's exactly.
    #[test]
    fn rle_gradient_matches_reader_path() {
        use crate::api::decoder::tests::decode;
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/rle-gradient");
        for name in ["rgb8.jxl", "gray16.jxl", "rgba8.jxl"] {
            let data = std::fs::read(dir.join(name)).unwrap();
            let before = USES.load(Ordering::Relaxed);
            let (_, fast) = decode(&data, usize::MAX, false, false, None).unwrap();
            assert!(
                USES.load(Ordering::Relaxed) > before,
                "{name}: fast path not taken"
            );
            DISABLE.store(true, Ordering::Relaxed);
            let generic = decode(&data, usize::MAX, false, false, None);
            DISABLE.store(false, Ordering::Relaxed);
            let (_, generic) = generic.unwrap();
            assert_eq!(fast.len(), generic.len());
            for (f, g) in fast[0].iter().zip(&generic[0]) {
                assert_eq!(f.size(), g.size());
                for y in 0..f.size().1 {
                    assert!(f.row(y) == g.row(y), "{name}: row {y} differs");
                }
            }
        }
    }
}
