// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Fused prefix-code + hybrid-uint + signed-unpack lookup for one cluster.
//!
//! For a prefix-coded cluster without LZ77, the next `FUSED_BITS` bits of the
//! stream determine the prefix code, any hybrid-uint extra bits and so the
//! decoded residual, whenever their combined length fits in the window. The
//! table maps each window to that residual and its total bit length, so the
//! common case is one peek, one load and one consume instead of two table
//! reads, a split-token test and a second peek. Windows whose code and extra
//! bits do not fit store length 0 and fall back to the regular reader.

use std::fmt::Debug;

use crate::bit_reader::FastBits;
use crate::entropy_coding::decode::unpack_signed;
use crate::entropy_coding::huffman::HuffmanCodes;
use crate::entropy_coding::hybrid_uint::HybridUint;

pub const FUSED_BITS: usize = 12;
const FUSED_SIZE: usize = 1 << FUSED_BITS;

/// One table entry, packed into a `u64`:
///
/// | bits   | field                                                       |
/// |--------|-------------------------------------------------------------|
/// | 0..4   | `total_len`: bits for both residuals (one if `two` is 0); 0 = fallback |
/// | 4..8   | `len1`: bits for the first residual alone                   |
/// | 8      | `two`: a second residual also fits in the window            |
/// | 16..40 | first residual, signed                                      |
/// | 40..64 | second residual, signed (if `two`)                          |
///
/// Residuals decoded from at most `FUSED_BITS` bits are below 2^15 in
/// magnitude, well inside 24 signed bits.
#[derive(Clone, Copy)]
pub struct FusedEntry(u64);

impl FusedEntry {
    #[inline(always)]
    pub fn total_len(self) -> usize {
        (self.0 & 0xf) as usize
    }
    #[inline(always)]
    pub fn len1(self) -> usize {
        ((self.0 >> 4) & 0xf) as usize
    }
    #[inline(always)]
    pub fn has_two(self) -> bool {
        self.0 & 0x100 != 0
    }
    #[inline(always)]
    pub fn first(self) -> i32 {
        (((self.0 << 24) as i64) >> 40) as i32
    }
    #[inline(always)]
    pub fn second(self) -> i32 {
        ((self.0 as i64) >> 40) as i32
    }
}

pub struct FusedPrefixLut {
    /// Fixed-size so the masked index needs no bounds check.
    entries: Box<[FusedEntry; FUSED_SIZE]>,
}

impl Debug for FusedPrefixLut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FusedPrefixLut({} entries)", self.entries.len())
    }
}

/// `(len, residual)` for the code at the start of `window` if the code and
/// its extra bits fit in `avail` bits.
fn decode_one(
    codes: &HuffmanCodes,
    cluster: usize,
    uint: &HybridUint,
    window: u32,
    avail: u32,
) -> Option<(u32, i32)> {
    let (code_len, token) = codes.lookup_window(cluster, window);
    if code_len > avail {
        return None;
    }
    let nbits = uint.extra_bits(token);
    let total = code_len + nbits;
    if total == 0 || total > avail {
        return None;
    }
    let extra = (window >> code_len) & ((1u32 << nbits) - 1);
    Some((total, unpack_signed(uint.assemble(token, nbits, extra))))
}

impl FusedPrefixLut {
    /// Returns `None` if the table cannot be allocated; callers fall back to
    /// the regular reader.
    pub fn build(codes: &HuffmanCodes, cluster: usize, uint: &HybridUint) -> Option<Self> {
        let bits = FUSED_BITS as u32;
        let mut entries = Vec::new();
        entries.try_reserve_exact(FUSED_SIZE).ok()?;
        for window in 0..FUSED_SIZE as u32 {
            let Some((len1, v0)) = decode_one(codes, cluster, uint, window, bits) else {
                entries.push(FusedEntry(0));
                continue;
            };
            // Bits above the window are unknown here, so the second code is
            // decoded from a window whose high bits are zero and accepted
            // only if it ends inside the known bits.
            let rest = window >> len1;
            let second = decode_one(codes, cluster, uint, rest, bits - len1);
            let v0 = u64::from(v0 as u32 & 0xff_ffff) << 16;
            let entry = match second {
                Some((len2, v1)) => {
                    (u64::from(v1 as u32 & 0xff_ffff) << 40)
                        | v0
                        | 0x100
                        | (u64::from(len1) << 4)
                        | u64::from(len1 + len2)
                }
                None => v0 | (u64::from(len1) << 4) | u64::from(len1),
            };
            entries.push(FusedEntry(entry));
        }
        Some(Self {
            entries: entries.into_boxed_slice().try_into().ok()?,
        })
    }

    /// Entry for the next window, or `None` (nothing consumed) when the
    /// regular reader must be used: the code does not fit, or the data is
    /// too close to its end for the 8-byte refill.
    #[inline(always)]
    pub fn peek(&self, fb: &mut FastBits<'_>) -> Option<FusedEntry> {
        if !fb.ensure(FUSED_BITS) {
            return None;
        }
        let window = fb.peek_buffered(FUSED_BITS) as usize;
        let entry = self.entries[window & (FUSED_SIZE - 1)];
        (entry.total_len() != 0).then_some(entry)
    }
}
