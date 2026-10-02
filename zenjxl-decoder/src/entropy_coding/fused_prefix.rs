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
const LEN_BITS: u32 = 4;
const LEN_MASK: u32 = (1 << LEN_BITS) - 1;
const _: () = assert!(FUSED_BITS as u32 <= LEN_MASK);

pub struct FusedPrefixLut {
    /// `(residual << LEN_BITS) | total_len`; `total_len == 0` means fallback.
    /// Fixed-size so the masked index needs no bounds check.
    entries: Box<[u32; FUSED_SIZE]>,
}

impl Debug for FusedPrefixLut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FusedPrefixLut({} entries)", self.entries.len())
    }
}

impl FusedPrefixLut {
    /// Returns `None` if the table cannot be allocated; callers fall back to
    /// the regular reader.
    pub fn build(codes: &HuffmanCodes, cluster: usize, uint: &HybridUint) -> Option<Self> {
        let mut entries = Vec::new();
        entries.try_reserve_exact(FUSED_SIZE).ok()?;
        for window in 0..FUSED_SIZE as u32 {
            let (code_len, token) = codes.lookup_window(cluster, window);
            let entry = if code_len as usize > FUSED_BITS {
                0
            } else {
                let nbits = uint.extra_bits(token);
                let total = code_len + nbits;
                if total == 0 || total as usize > FUSED_BITS {
                    0
                } else {
                    let extra = (window >> code_len) & ((1u32 << nbits) - 1);
                    let residual = unpack_signed(uint.assemble(token, nbits, extra));
                    // |residual| < 2^FUSED_BITS, far inside the 28 value bits.
                    ((residual << LEN_BITS) as u32) | total
                }
            };
            entries.push(entry);
        }
        Some(Self {
            entries: entries.into_boxed_slice().try_into().ok()?,
        })
    }

    /// Decodes one residual from a register-resident cursor if its code and
    /// extra bits fit in the window. On `None` nothing has been consumed and
    /// the caller must use the regular reader (also near the end of data).
    #[inline(always)]
    pub fn read_signed_fast(&self, fb: &mut FastBits<'_>) -> Option<i32> {
        if !fb.ensure(FUSED_BITS) {
            return None;
        }
        let window = fb.peek_buffered(FUSED_BITS) as usize;
        let entry = self.entries[window & (FUSED_SIZE - 1)];
        let len = entry & LEN_MASK;
        if len == 0 {
            return None;
        }
        fb.consume_buffered(len as usize);
        Some((entry as i32) >> LEN_BITS)
    }
}
