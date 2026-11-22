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
