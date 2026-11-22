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

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
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
        let entry = self.peek_entry(bits);
        let available = bits.buffered_len();
        let length = (entry & LENGTH_MASK) as usize;
        // Every entry is independent of the peeked bits above its token
        // length. Subtracting one rejects both zero and incomplete lengths.
        let entry = if length.wrapping_sub(1) < available {
            entry
        } else {
            if available >= RLE_WINDOW_BITS as usize || !bits.ensure(RLE_WINDOW_BITS as usize) {
                return None;
            }
            let entry = self.peek_entry(bits);
            if entry == 0 {
                return None;
            }
            entry
        };
        bits.consume_buffered((entry & LENGTH_MASK) as usize);
        let value = (entry >> LENGTH_BITS) & RLE_VALUE_MASK;
        Some(if entry & RLE_FLAG != 0 {
            RlePrefixToken::Repeat(value)
        } else {
            RlePrefixToken::Literal(value)
        })
    }

    #[inline(always)]
    fn peek_entry(&self, bits: &FastBits<'_>) -> u32 {
        let index = bits.peek_buffered(RLE_WINDOW_BITS as usize) as usize & (RLE_TABLE_SIZE - 1);
        self.entries[index]
    }
}

const TABLE_SIZE: usize = 1 << WINDOW_BITS;
const LENGTH_BITS: u32 = 4;
const LENGTH_MASK: u32 = (1 << LENGTH_BITS) - 1;

#[cfg(test)]
#[test]
fn rle_lookup_consumes_only_complete_buffered_tokens() {
    use crate::bit_reader::BitReader;

    let token = |window: usize| {
        if window & 1 == 0 {
            (1, Some(RlePrefixToken::Literal(3)))
        } else if window & 3 == 1 {
            (2, Some(RlePrefixToken::Literal(4)))
        } else if window & 7 == 3 {
            (3, Some(RlePrefixToken::Repeat(9)))
        } else {
            (0, None)
        }
    };
    let entries = (0..RLE_TABLE_SIZE)
        .map(|window| match token(window) {
            (len, Some(RlePrefixToken::Literal(value))) => (value << LENGTH_BITS) | len,
            (len, Some(RlePrefixToken::Repeat(value))) => RLE_FLAG | (value << LENGTH_BITS) | len,
            (_, None) => 0,
        })
        .collect::<Vec<_>>();
    let lut = FusedRleLut {
        entries: entries.into_boxed_slice().try_into().unwrap(),
        _memory: MemoryGuard::new(MemoryTracker::default(), 0),
    };
    for window in 0..RLE_TABLE_SIZE {
        for available in 0..=RLE_WINDOW_BITS as usize {
            let start = 16 - available;
            let bytes = ((window as u32) << start).to_le_bytes();
            let mut reader = BitReader::new(&bytes[..2]);
            reader.read(start).unwrap();
            let mut bits = reader.take_fast();
            let (length, expected) = token(window);
            let expected = expected.filter(|_| length as usize <= available);
            assert_eq!(lut.read(&mut bits), expected, "{window}, {available}");
            reader.put_fast(bits);
            let consumed = if expected.is_some() { length as usize } else { 0 };
            assert_eq!(reader.total_bits_read(), start + consumed);
        }

        let bytes = (window as u64).to_le_bytes();
        let mut reader = BitReader::new(&bytes);
        let mut bits = reader.take_fast();
        let (length, expected) = token(window);
        assert_eq!(lut.read(&mut bits), expected, "{window}, refill");
        reader.put_fast(bits);
        assert_eq!(reader.total_bits_read(), length as usize);
    }
}

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
