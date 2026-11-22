// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Fuses a prefix code, hybrid integer extra bits, and signed unpacking.

use crate::{
    bit_reader::FastBits,
    entropy_coding::{decode::unpack_signed, huffman::Table, hybrid_uint::HybridUint},
    util::{MemoryGuard, MemoryTracker},
};

const WINDOW_BITS: u32 = 12;

// A smaller table resolves either one literal or one repeat-length token.
// Row span reconstruction still owns the repeat-count and predictor state.
const RLE_WINDOW_BITS: u32 = 10;
const RLE_TABLE_SIZE: usize = 1 << RLE_WINDOW_BITS;
// Four length bits, one repeat flag, then a signed literal or unsigned length.
const RLE_FLAG: u32 = 1 << LENGTH_BITS;
const RLE_PAYLOAD_SHIFT: u32 = LENGTH_BITS + 1;
const RLE_VALUE_MASK: u32 = u32::MAX >> RLE_PAYLOAD_SHIFT;

pub(super) enum RlePrefixToken {
    Literal(i32),
    Repeat(u32),
}

#[derive(Debug)]
pub(crate) struct FusedRleLut {
    entries: Box<[u32; RLE_TABLE_SIZE]>,
    _memory: MemoryGuard,
}

impl FusedRleLut {
    pub(super) fn build(
        table: &Table,
        literal: &HybridUint,
        repeat: &HybridUint,
        min_symbol: u32,
        tracker: &MemoryTracker,
    ) -> Option<Self> {
        let bytes = std::mem::size_of::<[u32; RLE_TABLE_SIZE]>() as u64;
        tracker.try_allocate(bytes).ok()?;
        let memory = MemoryGuard::new(tracker.clone(), bytes);
        let mut entries = Vec::new();
        entries.try_reserve_exact(RLE_TABLE_SIZE).ok()?;
        entries.resize(RLE_TABLE_SIZE, 0);
        for index in 0..RLE_TABLE_SIZE {
            if entries[index] != 0 {
                continue;
            }
            let (code_len, token) = table.lookup_window(index);
            if code_len > RLE_WINDOW_BITS {
                continue;
            }
            let (config, token, flag) = if let Some(length) = token.checked_sub(min_symbol) {
                (repeat, length, RLE_FLAG)
            } else {
                (literal, token, 0)
            };
            let Some((extra, value)) = config.decode_window(
                token,
                (index as u32) >> code_len,
                RLE_WINDOW_BITS - code_len,
            ) else {
                continue;
            };
            let length = code_len + extra;
            if length == 0 || value > RLE_VALUE_MASK {
                continue;
            }
            let payload = if flag == 0 {
                unpack_signed(value) as u32
            } else {
                value
            };
            let entry = (payload << RLE_PAYLOAD_SHIFT) | flag | length;
            // The token depends on the consumed low bits only.
            for slot in entries[index..].iter_mut().step_by(1 << length) {
                *slot = entry;
            }
        }
        Some(Self {
            entries: entries.into_boxed_slice().try_into().ok()?,
            _memory: memory,
        })
    }

    #[inline(always)]
    pub(super) fn read(&self, bits: &mut FastBits<'_>) -> Option<RlePrefixToken> {
        if !bits.ensure(RLE_WINDOW_BITS as usize) {
            return None;
        }
        let index = bits.peek_buffered(RLE_WINDOW_BITS as usize) as usize & (RLE_TABLE_SIZE - 1);
        let entry = self.entries[index];
        if entry == 0 {
            return None;
        }
        bits.consume_buffered((entry & LENGTH_MASK) as usize);
        Some(if entry & RLE_FLAG != 0 {
            RlePrefixToken::Repeat(entry >> RLE_PAYLOAD_SHIFT)
        } else {
            RlePrefixToken::Literal((entry as i32) >> RLE_PAYLOAD_SHIFT)
        })
    }
}

const TABLE_SIZE: usize = 1 << WINDOW_BITS;
const LENGTH_BITS: u32 = 4;
const LENGTH_MASK: u32 = (1 << LENGTH_BITS) - 1;

#[derive(Debug)]
pub(crate) struct FusedPrefixLut {
    entries: Box<[u32; TABLE_SIZE]>,
    _memory: MemoryGuard,
}

impl FusedPrefixLut {
    pub(super) fn build(table: &Table, uint: &HybridUint, tracker: &MemoryTracker) -> Option<Self> {
        let bytes = std::mem::size_of::<[u32; TABLE_SIZE]>() as u64;
        tracker.try_allocate(bytes).ok()?;
        let memory = MemoryGuard::new(tracker.clone(), bytes);
        let mut entries = Vec::new();
        entries.try_reserve_exact(TABLE_SIZE).ok()?;
        for window in 0..TABLE_SIZE {
            entries.push(decode_entry(table, uint, window).unwrap_or(0));
        }
        Some(Self {
            entries: entries.into_boxed_slice().try_into().ok()?,
            _memory: memory,
        })
    }

    #[inline(always)]
    pub(super) fn read(&self, bits: &mut FastBits<'_>) -> Option<i32> {
        if !bits.ensure(WINDOW_BITS as usize) {
            return None;
        }
        let index = bits.peek_buffered(WINDOW_BITS as usize) as usize & (TABLE_SIZE - 1);
        let entry = self.entries[index];
        let length = entry & LENGTH_MASK;
        if length == 0 {
            return None;
        }
        bits.consume_buffered(length as usize);
        Some((entry as i32) >> LENGTH_BITS)
    }
}

fn decode_entry(table: &Table, uint: &HybridUint, window: usize) -> Option<u32> {
    let (code_len, token) = table.lookup_window(window);
    let remaining = WINDOW_BITS.checked_sub(code_len)?;
    let (extra_len, value) = uint.decode_window(token, (window as u32) >> code_len, remaining)?;
    let length = code_len + extra_len;
    let residual = unpack_signed(value);
    // Reserve a zero length for misses. Residuals wider than 28 signed
    // bits also go through the regular cursor.
    (length != 0 && (residual << LENGTH_BITS) >> LENGTH_BITS == residual)
        .then_some(((residual as u32) << LENGTH_BITS) | length)
}
