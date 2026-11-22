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
const RLE_FLAG: u32 = 1 << 31;
const RLE_VALUE_MASK: u32 = (RLE_FLAG - 1) >> LENGTH_BITS;

pub(super) enum RlePrefixToken {
    Literal(u32),
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
            let entry = flag | (value << LENGTH_BITS) | length;
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
        bits.consume_buffered((entry & LENGTH_MASK as u32) as usize);
        let value = (entry >> LENGTH_BITS) & RLE_VALUE_MASK;
        Some(if entry & RLE_FLAG != 0 {
            RlePrefixToken::Repeat(value)
        } else {
            RlePrefixToken::Literal(value)
        })
    }
}

const TABLE_SIZE: usize = 1 << WINDOW_BITS;
const LENGTH_BITS: u32 = 4;
const LENGTH_MASK: u64 = (1 << LENGTH_BITS) - 1;
const FIRST_SHIFT: u32 = 16;
const SECOND_SHIFT: u32 = 40;
const RESIDUAL_MASK: u64 = (1 << 24) - 1;

#[derive(Clone, Copy)]
pub(crate) struct Residuals {
    pub first: i32,
    pub second: Option<i32>,
}

#[derive(Debug)]
pub(crate) struct FusedPrefixLut {
    entries: Box<[u64; TABLE_SIZE]>,
    _memory: MemoryGuard,
}

impl FusedPrefixLut {
    pub(super) fn build(table: &Table, uint: &HybridUint, tracker: &MemoryTracker) -> Option<Self> {
        let bytes = std::mem::size_of::<[u64; TABLE_SIZE]>() as u64;
        tracker.try_allocate(bytes).ok()?;
        let memory = MemoryGuard::new(tracker.clone(), bytes);
        let mut entries = Vec::new();
        entries.try_reserve_exact(TABLE_SIZE).ok()?;
        for window in 0..TABLE_SIZE as u32 {
            let Some((first_len, first)) = decode_residual(table, uint, window, WINDOW_BITS) else {
                entries.push(0);
                continue;
            };
            let entry = (u64::from(first_len) << LENGTH_BITS)
                | u64::from(first_len)
                | (first << FIRST_SHIFT);
            entries.push(entry);
        }
        // The remaining low bits already index a decoded first residual.
        // Its length determines whether it fits without the zero-filled high bits.
        for window in 0..TABLE_SIZE {
            let entry = entries[window];
            if entry == 0 {
                continue;
            }
            let first_len = (entry >> LENGTH_BITS) & LENGTH_MASK;
            let next = entries[window >> first_len];
            let second_len = (next >> LENGTH_BITS) & LENGTH_MASK;
            if second_len != 0 && first_len + second_len <= u64::from(WINDOW_BITS) {
                let second = (next >> FIRST_SHIFT) & RESIDUAL_MASK;
                entries[window] = (entry + second_len) | (second << SECOND_SHIFT);
            }
        }
        Some(Self {
            entries: entries.into_boxed_slice().try_into().ok()?,
            _memory: memory,
        })
    }

    #[inline(always)]
    pub(super) fn read(&self, bits: &mut FastBits<'_>, allow_pair: bool) -> Option<Residuals> {
        if !bits.ensure(WINDOW_BITS as usize) {
            return None;
        }
        let index = bits.peek_buffered(WINDOW_BITS as usize) as usize & (TABLE_SIZE - 1);
        let entry = self.entries[index];
        let first_len = (entry >> LENGTH_BITS) & LENGTH_MASK;
        if first_len == 0 {
            return None;
        }
        let total_len = entry & LENGTH_MASK;
        let first = (entry.wrapping_shl(24) as i64 >> SECOND_SHIFT) as i32;
        if allow_pair && total_len != first_len {
            bits.consume_buffered(total_len as usize);
            Some(Residuals {
                first,
                second: Some(((entry as i64) >> SECOND_SHIFT) as i32),
            })
        } else {
            bits.consume_buffered(first_len as usize);
            Some(Residuals {
                first,
                second: None,
            })
        }
    }
}

fn decode_residual(
    table: &Table,
    uint: &HybridUint,
    window: u32,
    available: u32,
) -> Option<(u32, u64)> {
    let (code_len, token) = table.lookup_window(window as usize);
    let remaining = available.checked_sub(code_len)?;
    let (extra_len, value) = uint.decode_window(token, window >> code_len, remaining)?;
    let length = code_len + extra_len;
    let residual = unpack_signed(value);
    // The two values have 24 signed bits each. Long residuals use the ordinary cursor.
    (length > 0 && (residual << 8) >> 8 == residual)
        .then_some((length, residual as u64 & RESIDUAL_MASK))
}
