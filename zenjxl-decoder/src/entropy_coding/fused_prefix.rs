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
        if !bits.ensure_lookup(WINDOW_BITS as usize) {
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
