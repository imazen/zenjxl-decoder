// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Single-gradient-leaf channels read through a fused prefix lookup
//! ([`FusedPrefixLut`]), one channel at a time or two channels from
//! independent bitstreams interleaved.
//!
//! Reconstruction is the same as `SingleGradientOnly::decode_one`. Edge rules
//! follow `PredictionData::get_rows`: row 0 predicts `left` (0 at x = 0), and
//! x = 0 on later rows predicts `row_top[0]`.
//!
//! Each lookup decodes one residual, or two when both codes fit in the
//! window. The bit cursor is a [`FastBits`] copy whose address never escapes,
//! so it stays in registers; it is synced to the `BitReader` only around the
//! cold fallback to the regular reader. Passing `&mut BitReader` through the
//! hot loop instead forced a store and reload of the cursor on every sample
//! (measured 2026-10-01).

use crate::{
    bit_reader::{BitReader, FastBits},
    entropy_coding::{
        decode::{Histograms, SymbolReader},
        fused_prefix::{FUSED_SIZE, FusedEntry, FusedPrefixLut, peek_table},
    },
    frame::modular::{IMAGE_OFFSET, ModularChannel},
};

#[cold]
#[inline(never)]
fn read_slow(
    reader: &mut SymbolReader,
    br: &mut BitReader,
    histograms: &Histograms,
    ctx: usize,
) -> i32 {
    reader.read_signed_clustered_inline(histograms, br, ctx)
}

/// Entropy state of one channel's bitstream.
pub(super) struct FusedSide<'a, 'b> {
    table: &'a [FusedEntry; FUSED_SIZE],
    fb: FastBits<'b>,
    reader: &'a mut SymbolReader,
    br: &'a mut BitReader<'b>,
    histograms: &'a Histograms,
    ctx: usize,
}

impl<'a, 'b> FusedSide<'a, 'b> {
    pub(super) fn new(
        lut: &'a FusedPrefixLut,
        reader: &'a mut SymbolReader,
        br: &'a mut BitReader<'b>,
        histograms: &'a Histograms,
        ctx: usize,
    ) -> Self {
        let fb = br.take_fast();
        Self {
            table: lut.table(),
            fb,
            reader,
            br,
            histograms,
            ctx,
        }
    }

    /// Writes the cursor back to the `BitReader`.
    #[inline(always)]
    fn finish(&mut self) {
        self.br.put_fast(self.fb);
    }

    #[inline(always)]
    fn slow(&mut self) -> i32 {
        self.br.put_fast(self.fb);
        let v = read_slow(self.reader, self.br, self.histograms, self.ctx);
        self.fb = self.br.take_fast();
        v
    }

    #[inline(always)]
    fn read_one(&mut self) -> i32 {
        match peek_table(self.table, &mut self.fb) {
            Some(e) => {
                self.fb.consume_buffered(e.len1());
                e.first()
            }
            None => self.slow(),
        }
    }
}

/// Reconstruction state of one row (y >= 1, x >= 1).
struct GradRow<'r> {
    row: &'r mut [i32],
    top: &'r [i32],
    x: usize,
    last: i32,
}

impl GradRow<'_> {
    #[inline(always)]
    fn done(&self) -> bool {
        self.x >= self.row.len()
    }

    #[inline(always)]
    fn push(&mut self, x: usize, r: i32) {
        let top = self.top[x];
        let topleft = self.top[x - 1];
        let last = self.last;
        let min = last.min(top);
        let max = last.max(top);
        let grad = last.wrapping_add(top).wrapping_sub(topleft);
        let grad_clamp_max = if topleft < min { max } else { grad };
        let pred = if topleft > max { min } else { grad_clamp_max };
        self.last = r.wrapping_add(pred);
        self.row[x] = self.last;
    }

    /// Decodes one lookup's worth: one residual, or two when both fit and
    /// the row has room.
    #[inline(always)]
    fn step(&mut self, side: &mut FusedSide) {
        let x = self.x;
        match peek_table(side.table, &mut side.fb) {
            Some(e) if e.has_two() && x + 1 < self.row.len() => {
                side.fb.consume_buffered(e.total_len());
                self.push(x, e.first());
                self.push(x + 1, e.second());
                self.x = x + 2;
            }
            Some(e) => {
                side.fb.consume_buffered(e.len1());
                self.push(x, e.first());
                self.x = x + 1;
            }
            None => {
                let v = side.slow();
                self.push(x, v);
                self.x = x + 1;
            }
        }
    }
}

/// Row 0: every sample predicts its left neighbour (0 at x = 0).
#[inline(always)]
fn first_row(row: &mut [i32], side: &mut FusedSide) {
    let mut last = 0i32;
    for o in row.iter_mut() {
        last = side.read_one().wrapping_add(last);
        *o = last;
    }
}

/// Starts row `y >= 1`: decodes x = 0, which predicts `row_top[0]`.
#[inline(always)]
fn start_row<'r>(row: &'r mut [i32], top: &'r [i32], side: &mut FusedSide) -> GradRow<'r> {
    let last = side.read_one().wrapping_add(top[0]);
    row[0] = last;
    // Same length as `row`, so `x < row.len()` also bounds `top[x]`.
    let top = &top[..row.len()];
    GradRow {
        row,
        top,
        x: 1,
        last,
    }
}

fn rows(channel: &mut ModularChannel, y: usize) -> (&mut [i32], &[i32]) {
    const { assert!(IMAGE_OFFSET.1 == 2) };
    let w = channel.data.size().0;
    let [row, row_top] = channel.data.distinct_full_rows_mut([y + 2, y + 1]);
    (
        &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + w],
        &row_top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + w],
    )
}

/// Decodes one channel. Sides are taken by value (and finished here) so that
/// their cursors stay in registers: behind a `&mut` into this non-inlined
/// function they were stored back to memory on every lookup.
#[inline(never)]
pub(super) fn decode_one(channel: &mut ModularChannel, side: FusedSide) {
    let mut side = side;
    let side = &mut side;
    let h = channel.data.size().1;
    for y in 0..h {
        let (row, top) = rows(channel, y);
        if y == 0 {
            first_row(row, side);
            continue;
        }
        let mut r = start_row(row, top, side);
        while !r.done() {
            r.step(side);
        }
    }
    side.finish();
}

/// Decodes two channels from independent bitstreams, alternating lookups
/// between them so their serial dependency chains overlap.
#[inline(never)]
pub(super) fn decode_two(
    ca: &mut ModularChannel,
    sa: FusedSide,
    cb: &mut ModularChannel,
    sb: FusedSide,
) {
    let (mut sa, mut sb) = (sa, sb);
    let (sa, sb) = (&mut sa, &mut sb);
    let (ha, hb) = (ca.data.size().1, cb.data.size().1);
    for y in 0..ha.max(hb) {
        if y >= ha {
            let (row, top) = rows(cb, y);
            let mut r = start_row(row, top, sb);
            while !r.done() {
                r.step(sb);
            }
            continue;
        }
        if y >= hb {
            let (row, top) = rows(ca, y);
            let mut r = start_row(row, top, sa);
            while !r.done() {
                r.step(sa);
            }
            continue;
        }
        let (row_a, top_a) = rows(ca, y);
        let (row_b, top_b) = rows(cb, y);
        if y == 0 {
            first_row(row_a, sa);
            first_row(row_b, sb);
            continue;
        }
        let mut ra = start_row(row_a, top_a, sa);
        let mut rb = start_row(row_b, top_b, sb);
        while !ra.done() && !rb.done() {
            ra.step(sa);
            rb.step(sb);
        }
        while !ra.done() {
            ra.step(sa);
        }
        while !rb.done() {
            rb.step(sb);
        }
    }
    sa.finish();
    sb.finish();
}
