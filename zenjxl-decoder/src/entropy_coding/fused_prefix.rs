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

// A smaller table resolves literals and repeat-length tokens.
// Row span reconstruction still owns the repeat-count and predictor state.
const RLE_WINDOW_BITS: u32 = 10;
const RLE_TABLE_SIZE: usize = 1 << RLE_WINDOW_BITS;
const RLE_FLAG: u32 = 1 << 31;
const RLE_VALUE_MASK: u32 = (RLE_FLAG - 1) >> LENGTH_BITS;

// Pair entries use a zero length tag and store the combined length, the
// first length, and two packed literals in the remaining 28 bits.
const RLE_PAIR_LENGTH_SHIFT: u32 = LENGTH_BITS;
const RLE_FIRST_LENGTH_SHIFT: u32 = 2 * LENGTH_BITS;
const RLE_FIRST_LITERAL_SHIFT: u32 = 3 * LENGTH_BITS;
const RLE_LITERAL_BITS: u32 = 10;
const RLE_SECOND_LITERAL_SHIFT: u32 = RLE_FIRST_LITERAL_SHIFT + RLE_LITERAL_BITS;
const RLE_LITERAL_MASK: u32 = (1 << RLE_LITERAL_BITS) - 1;

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(super) enum RlePrefixToken {
    Literal(u32),
    LiteralPair(u32, u32),
    Repeat(u32),
}

fn pack_rle_literal_pairs(entries: &mut [u32]) {
    // The second lookup drops the first token's low bits, so its index
    // precedes this one. Descending order keeps both source entries intact.
    for index in (0..entries.len()).rev() {
        let first = entries[index];
        if first == 0 || first & RLE_FLAG != 0 {
            continue;
        }
        let first_length = first & LENGTH_MASK;
        let first_value = first >> LENGTH_BITS;
        let second = entries[index >> first_length];
        if second == 0 || second & RLE_FLAG != 0 {
            continue;
        }
        let length = first_length + (second & LENGTH_MASK);
        let second_value = second >> LENGTH_BITS;
        if length <= RLE_WINDOW_BITS
            && first_value <= RLE_LITERAL_MASK
            && second_value <= RLE_LITERAL_MASK
        {
            entries[index] = (length << RLE_PAIR_LENGTH_SHIFT)
                | (first_length << RLE_FIRST_LENGTH_SHIFT)
                | (first_value << RLE_FIRST_LITERAL_SHIFT)
                | (second_value << RLE_SECOND_LITERAL_SHIFT);
        }
    }
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
        pack_rle_literal_pairs(&mut entries);
        Some(Self {
            entries: entries.into_boxed_slice().try_into().ok()?,
            _memory: memory,
        })
    }

    #[inline(always)]
    pub(super) fn read(&self, bits: &mut FastBits<'_>, allow_pair: bool) -> Option<RlePrefixToken> {
        if !bits.ensure(RLE_WINDOW_BITS as usize) {
            return None;
        }
        let index = bits.peek_buffered(RLE_WINDOW_BITS as usize) as usize & (RLE_TABLE_SIZE - 1);
        let entry = self.entries[index];
        if entry == 0 {
            return None;
        }
        let length = entry & LENGTH_MASK;
        if length == 0 {
            let first = (entry >> RLE_FIRST_LITERAL_SHIFT) & RLE_LITERAL_MASK;
            if allow_pair {
                let length = (entry >> RLE_PAIR_LENGTH_SHIFT) & LENGTH_MASK;
                bits.consume_buffered(length as usize);
                return Some(RlePrefixToken::LiteralPair(
                    first,
                    entry >> RLE_SECOND_LITERAL_SHIFT,
                ));
            }
            let length = (entry >> RLE_FIRST_LENGTH_SHIFT) & LENGTH_MASK;
            bits.consume_buffered(length as usize);
            return Some(RlePrefixToken::Literal(first));
        }
        bits.consume_buffered(length as usize);
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

#[cfg(test)]
#[test]
fn rle_literal_pair_windows_preserve_row_boundary() {
    use crate::bit_reader::BitReader;

    // Low-bit codes 0, 01, 011 and 111 cover both the repeat and
    // over-wide literal fallbacks.
    let token = |window: usize| {
        if window & 1 == 0 {
            (1, RlePrefixToken::Literal(3))
        } else if window & 3 == 1 {
            (2, RlePrefixToken::Literal(1023))
        } else if window & 7 == 3 {
            (3, RlePrefixToken::Repeat(9))
        } else {
            (3, RlePrefixToken::Literal(1024))
        }
    };
    let mut entries = (0..RLE_TABLE_SIZE)
        .map(|window| {
            let (length, token) = token(window);
            match token {
                RlePrefixToken::Literal(value) => (value << LENGTH_BITS) | length,
                RlePrefixToken::Repeat(value) => RLE_FLAG | (value << LENGTH_BITS) | length,
                RlePrefixToken::LiteralPair(..) => unreachable!(),
            }
        })
        .collect::<Vec<_>>();
    pack_rle_literal_pairs(&mut entries);
    let lut = FusedRleLut {
        entries: entries.into_boxed_slice().try_into().unwrap(),
        _memory: MemoryGuard::new(MemoryTracker::default(), 0),
    };
    for window in 0..RLE_TABLE_SIZE {
        let data = (window as u64).to_le_bytes();
        for allow_pair in [false, true] {
            let (first_length, first) = token(window);
            let (second_length, second) = token(window >> first_length);
            let (length, expected) = match (first, second) {
                (RlePrefixToken::Literal(first), RlePrefixToken::Literal(second))
                    if allow_pair && first <= 1023 && second <= 1023 =>
                {
                    (
                        first_length + second_length,
                        RlePrefixToken::LiteralPair(first, second),
                    )
                }
                (first, _) => (first_length, first),
            };
            let mut reader = BitReader::new(&data);
            let mut bits = reader.take_fast();
            assert_eq!(lut.read(&mut bits, allow_pair), Some(expected), "{window}");
            reader.put_fast(bits);
            let mut expected_reader = BitReader::new(&data);
            expected_reader.read(length as usize).unwrap();
            assert_eq!(
                reader.total_bits_read(),
                expected_reader.total_bits_read(),
                "{window}"
            );
            assert_eq!(
                reader.read(8).ok(),
                expected_reader.read(8).ok(),
                "{window}"
            );
        }
    }
}
