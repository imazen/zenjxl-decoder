// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::{BitReader, FastBits};
use crate::error::Error;

use crate::util::CeilLog2;

#[derive(Debug)]
pub struct HybridUint {
    split_token: u32,
    split_exponent: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl HybridUint {
    pub(super) fn is_split_exponent_zero(&self) -> bool {
        self.split_exponent == 0
    }

    /// Tokens below this value encode themselves with no extra bits.
    pub fn split_token(&self) -> u32 {
        self.split_token
    }

    pub fn decode(log_alpha_size: usize, br: &mut BitReader) -> Result<HybridUint, Error> {
        let split_exponent = br.read((log_alpha_size + 1).ceil_log2())? as u32;
        let split_token = 1u32 << split_exponent;
        let msb_in_token;
        let lsb_in_token;
        if split_exponent != log_alpha_size as u32 {
            let nbits = (split_exponent + 1).ceil_log2() as usize;
            msb_in_token = br.read(nbits)? as u32;
            if msb_in_token > split_exponent {
                return Err(Error::InvalidUintConfig(split_exponent, msb_in_token, None));
            }
            let nbits = (split_exponent - msb_in_token + 1).ceil_log2() as usize;
            lsb_in_token = br.read(nbits)? as u32;
        } else {
            msb_in_token = 0;
            lsb_in_token = 0;
        }
        if lsb_in_token + msb_in_token > split_exponent {
            return Err(Error::InvalidUintConfig(
                split_exponent,
                msb_in_token,
                Some(lsb_in_token),
            ));
        }
        Ok(HybridUint {
            split_token,
            split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }

    /// Returns true if this config matches the 420 pattern (common in e3 images):
    /// split_exponent=4, msb_in_token=2, lsb_in_token=0
    #[inline(always)]
    pub fn is_config_420(&self) -> bool {
        self.split_exponent == 4
            && self.split_token == 16
            && self.msb_in_token == 2
            && self.lsb_in_token == 0
    }

    /// Specialized fast path for 420 config:
    /// split_exponent=4, msb_in_token=2, lsb_in_token=0
    ///
    /// `nbits_acc` accumulates raw nbits values via OR. If any call produces
    /// nbits >= 32 (invalid bitstream), bit 5+ will be set in the accumulator.
    /// Check `nbits_acc >= 32` after a batch of reads.
    #[inline(always)]
    pub fn read_config_420(symbol: u32, br: &mut BitReader, nbits_acc: &mut u32) -> u32 {
        if symbol < 16 {
            return symbol;
        }

        // Equivalent to: 2 + ((symbol - 16) >> 2)
        let nbits_raw = (symbol >> 2) - 2;
        *nbits_acc |= nbits_raw;
        let nbits = nbits_raw & 31;
        let bits = br.read_optimistic(nbits as usize) as u32;
        let hi = (symbol & 3) | 4;

        (hi << nbits) | bits
    }

    /// [`Self::read`] on a register-resident cursor; requires at least 31
    /// buffered bits.
    #[inline(always)]
    pub fn read_fast(&self, symbol: u32, fb: &mut FastBits<'_>, nbits_acc: &mut u32) -> u32 {
        if symbol < self.split_token {
            return symbol;
        }
        if self.msb_in_token == 0 && self.lsb_in_token == 0 {
            let nbits_raw = self.split_exponent + symbol - self.split_token;
            *nbits_acc |= nbits_raw;
            let nbits = nbits_raw & 31;
            let bits = fb.peek_buffered(nbits as usize) as u32;
            fb.consume_buffered(nbits as usize);
            return (1 << nbits) | bits;
        }
        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        let nbits_raw =
            self.split_exponent - bits_in_token + ((symbol - self.split_token) >> bits_in_token);
        *nbits_acc |= nbits_raw;
        let nbits = nbits_raw & 31;
        let low = symbol & ((1 << self.lsb_in_token) - 1);
        let symbol_nolow = symbol >> self.lsb_in_token;
        let bits = fb.peek_buffered(nbits as usize) as u32;
        fb.consume_buffered(nbits as usize);
        let hi = (symbol_nolow & ((1 << self.msb_in_token) - 1)) | (1 << self.msb_in_token);
        (((hi << nbits) | bits) << self.lsb_in_token) | low
    }

    /// Reads a hybrid uint value from the bitstream.
    ///
    /// `nbits_acc` accumulates raw nbits values via OR. If any call produces
    /// nbits >= 32 (invalid bitstream), bit 5+ will be set in the accumulator.
    /// Check `nbits_acc >= 32` after a batch of reads.
    #[inline(always)]
    pub fn read(&self, symbol: u32, br: &mut BitReader, nbits_acc: &mut u32) -> u32 {
        if symbol < self.split_token {
            return symbol;
        }
        // Fast path: when msb_in_token == 0 && lsb_in_token == 0, hi is always 1
        // and the low/shift computation is a no-op.
        if self.msb_in_token == 0 && self.lsb_in_token == 0 {
            let nbits_raw = self.split_exponent + symbol - self.split_token;
            *nbits_acc |= nbits_raw;
            let nbits = nbits_raw & 31;
            let bits = br.read_optimistic(nbits as usize) as u32;
            return (1 << nbits) | bits;
        }
        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        let nbits_raw =
            self.split_exponent - bits_in_token + ((symbol - self.split_token) >> bits_in_token);
        *nbits_acc |= nbits_raw;
        let nbits = nbits_raw & 31;
        let low = symbol & ((1 << self.lsb_in_token) - 1);
        let symbol_nolow = symbol >> self.lsb_in_token;
        let bits = br.read_optimistic(nbits as usize) as u32;
        let hi = (symbol_nolow & ((1 << self.msb_in_token) - 1)) | (1 << self.msb_in_token);
        (((hi << nbits) | bits) << self.lsb_in_token) | low
    }
}

impl HybridUint {
    /// Number of extra bits `read` consumes after `symbol` (unmasked, so an
    /// invalid config can report >= 32).
    pub fn extra_bits(&self, symbol: u32) -> u32 {
        if symbol < self.split_token {
            return 0;
        }
        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        self.split_exponent - bits_in_token + ((symbol - self.split_token) >> bits_in_token)
    }

    /// The value `read` returns for `symbol` followed by the `nbits` extra
    /// bits `bits`, where `nbits == self.extra_bits(symbol) < 32`.
    pub fn assemble(&self, symbol: u32, nbits: u32, bits: u32) -> u32 {
        if symbol < self.split_token {
            return symbol;
        }
        let low = symbol & ((1 << self.lsb_in_token) - 1);
        let symbol_nolow = symbol >> self.lsb_in_token;
        let hi = (symbol_nolow & ((1 << self.msb_in_token) - 1)) | (1 << self.msb_in_token);
        (((hi << nbits) | bits) << self.lsb_in_token) | low
    }
}

#[cfg(test)]
impl HybridUint {
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_token: 1 << split_exponent,
            split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }
}

#[cfg(test)]
mod test {
    /// `extra_bits` + `assemble` (used to build the fused prefix lookup)
    /// must reproduce `read` for every valid config and token.
    #[test]
    fn extra_bits_and_assemble_match_read() {
        use super::*;
        let data: Vec<u8> = (0..64u32)
            .map(|i| (i.wrapping_mul(0x9e) ^ 0x5a) as u8)
            .collect();
        for split_exponent in 0..=8u32 {
            for msb in 0..=split_exponent {
                for lsb in 0..=(split_exponent - msb) {
                    let cfg = HybridUint::new(split_exponent, msb, lsb);
                    for symbol in 0..256u32 {
                        let nbits = cfg.extra_bits(symbol);
                        if nbits > 24 {
                            continue;
                        }
                        let mut br = BitReader::new(&data);
                        let bits = br.peek(nbits as usize) as u32;
                        let mut acc = 0;
                        let expected = cfg.read(symbol, &mut br, &mut acc);
                        assert_eq!(br.total_bits_read(), nbits as usize);
                        assert_eq!(cfg.assemble(symbol, nbits, bits), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_hybrid_uint_decode_invalid() {
        use super::*;
        let mut br = BitReader::new(&[10, 75, 10, 75, 168, 139, 132, 255, 244]);
        br.skip_bits(1).unwrap();
        if let Ok(uint) = HybridUint::decode(15, &mut br) {
            let mut acc = 0u32;
            uint.read(1022, &mut br, &mut acc);
        }
    }
}
