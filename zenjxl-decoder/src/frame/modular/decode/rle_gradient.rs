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
    entropy_coding::{
        decode::{Histograms, RleCode, RleCursor, SymbolReader, unpack_signed},
        fused_prefix::FusedRleLut,
    },
    frame::modular::{IMAGE_OFFSET, ModularChannel, Tree, predict::clamped_gradient},
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
    fused: Option<&'a FusedRleLut>,
    cursor: RleCursor<'b>,
    reader: &'a mut SymbolReader,
    br: &'a mut BitReader<'b>,
    histograms: &'a Histograms,
    cluster: usize,
}

impl Side<'_, '_> {
    #[inline(always)]
    fn read_span(&mut self, limit: usize) -> (i32, usize) {
        match self.code.read_span(&mut self.cursor, limit, self.fused) {
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
    tree: &Tree,
) -> bool {
    #[cfg(test)]
    if tests::DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let histograms = &tree.histograms;
    let Some(cursor) = reader.rle_cursor(histograms, br) else {
        return false;
    };
    let Some(code) = histograms.rle_code(cluster) else {
        return false;
    };
    let (width, height) = channel.data.size();
    let fused = if width.saturating_mul(height) >= 4096 {
        tree.fused_rle(cluster, &cursor)
    } else {
        None
    };
    #[cfg(test)]
    tests::USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let side = decode(
        channel,
        Side {
            code,
            fused,
            cursor,
            reader,
            br,
            histograms,
            cluster,
        },
    );
    side.reader.finish_rle_cursor(side.cursor, side.br);
    true
}
/// Takes and returns the side by value so the cursor stays in registers;
/// behind a `&mut` into this non-inlined function it would be stored and
/// reloaded around every sample.
#[inline(never)]
fn decode<'a, 'b>(channel: &mut ModularChannel, side: Side<'a, 'b>) -> Side<'a, 'b> {
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
                let (residual, count) = side.read_span(w - x);
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
            let (residual, count) = side.read_span(w - x);
            let end = x + count;
            if count == 1 {
                let t = top[x];
                left = (clamped_gradient(left.into(), t.into(), topleft.into()) as i32)
                    .wrapping_add(residual);
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
                    left = (clamped_gradient(left.into(), t.into(), topleft.into()) as i32)
                        .wrapping_add(residual);
                    *p = left;
                    topleft = t;
                }
            }
            x = end;
        }
    }
    side
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test-only switch that sends these channels through the regular
    /// reader instead.
    pub(super) static DISABLE: AtomicBool = AtomicBool::new(false);
    /// Channels decoded here.
    pub(super) static USES: AtomicUsize = AtomicUsize::new(0);

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
