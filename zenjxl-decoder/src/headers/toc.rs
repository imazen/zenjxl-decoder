// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_macros::UnconditionalCoder;
use whereat::at;

use crate::{
    bit_reader::BitReader,
    error::{Error, Result},
    headers::{encodings::*, frame_header::PermutationNonserialized},
};

use super::permutation::Permutation;

pub struct TocNonserialized {
    pub num_entries: u32,
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
#[nonserialized(TocNonserialized)]
pub struct Toc {
    #[default(false)]
    pub permuted: bool,

    // Here we don't use `condition(permuted)`, because `jump_to_byte_boundary` needs to be executed in both cases
    #[default(Permutation::default())]
    #[nonserialized(num_entries: nonserialized.num_entries, permuted: permuted)]
    pub permutation: Permutation,

    #[coder(u2S(Bits(10), Bits(14) + 1024, Bits(22) + 17408, Bits(30) + 4211712))]
    #[size_coder(explicit(nonserialized.num_entries))]
    pub entries: Vec<u32>,
}

#[derive(Debug)]
pub struct IncrementalTocReader {
    num_entries: u32,
    permuted: bool,
    permutation: Option<Permutation>,
    entries: Vec<u32>,
}

impl IncrementalTocReader {
    pub fn new(num_entries: u32, br: &mut BitReader) -> Result<Self> {
        let permuted = bool::read_unconditional(&(), br, &Empty {})?;
        let mut entries = Vec::new();
        // `num_entries` comes from the frame header before any TOC bytes are
        // read; reserve a bounded amount and grow as entries arrive. Upstream
        // jxl-rs d13a505.
        entries
            .try_reserve(num_entries.min(4096) as usize)
            .map_err(|e| at!(Error::from(e)))?;
        Ok(Self {
            num_entries,
            permuted,
            permutation: None,
            entries,
        })
    }

    #[allow(dead_code)] // Part of incremental TOC reader API
    pub fn num_read_entries(&self) -> u32 {
        self.entries.len() as u32
    }

    pub fn remaining_entries(&self) -> u32 {
        self.num_entries - self.entries.len() as u32
    }

    pub fn is_complete(&self) -> bool {
        self.permutation.is_some() && self.remaining_entries() == 0
    }

    pub fn read_step(&mut self, br: &mut BitReader) -> Result<()> {
        if self.permutation.is_none() {
            return self.read_permutation(br);
        }

        let entry_coder = U32Coder::Select(
            U32::Bits(10),
            U32::BitsOffset { n: 14, off: 1024 },
            U32::BitsOffset { n: 22, off: 17408 },
            U32::BitsOffset {
                n: 30,
                off: 4211712,
            },
        );
        let entry = u32::read_unconditional(&entry_coder, br, &Empty {})?;
        // Entries may have been read optimistically past the end of the
        // buffered input; surface that so the incremental parser retries with
        // more data (jxl-rs #811). Check BEFORE recording the entry: pushing
        // first left a garbage entry behind on retry, misaligning every entry
        // after it.
        br.check_for_error().map_err(|e| at!(e))?;
        self.entries
            .try_reserve(1)
            .map_err(|e| at!(Error::from(e)))?;
        self.entries.push(entry);
        Ok(())
    }

    fn read_permutation(&mut self, br: &mut BitReader) -> Result<()> {
        // A permuted TOC's permutation is entropy-coded. Wait until a lower
        // bound for the whole TOC is buffered (2 bits of permutation + 10 bits
        // of size per entry) before decoding it. Upstream jxl-rs 9763730.
        //
        // This is load-bearing, not just an optimisation: without it, the
        // conformance progressive images fed one byte at a time fail with
        // AlphabetTooLargeHuff raised exactly at the buffer boundary (16 of 16
        // bits read, so not an overrun). Root cause not yet identified; files
        // whose permutation needs more than this bound may still hit it.
        if self.permuted {
            const MIN_BITS_PER_ENTRY: usize = 2 + 10;
            let needed = (self.num_entries as usize).saturating_mul(MIN_BITS_PER_ENTRY);
            let available = br.total_bits_available();
            if needed > available {
                return Err(at!(Error::OutOfBounds((needed - available).div_ceil(8))));
            }
        }
        let permutation = Permutation::read_unconditional(
            &(),
            br,
            &PermutationNonserialized {
                num_entries: self.num_entries,
                permuted: self.permuted,
            },
        )?;
        // The permutation's entropy-coded symbols are read optimistically.
        // Check BEFORE storing it: storing first kept a permutation decoded
        // from zero-padding, so the retry skipped re-reading it and parsed the
        // entries from the wrong position.
        br.check_for_error().map_err(|e| at!(e))?;
        self.permutation = Some(permutation);
        Ok(())
    }

    pub fn finalize(self) -> Toc {
        assert!(self.is_complete());
        let permutation = self.permutation.unwrap();
        Toc {
            permuted: self.permuted,
            permutation,
            entries: self.entries,
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::ControlFlow;

    use test_log::test;

    use super::*;

    #[test]
    fn parse_arb() {
        arbtest::arbtest(|u| {
            // Not permuted
            let mut bytes = vec![0u8];
            let mut buf = 0u64;
            let mut buf_bits = 0u32;
            let mut num_entries = 0u32;

            u.arbitrary_loop(Some(1), Some(256), |u| {
                let selector = u.int_in_range(0..=3)?;
                let bits = match selector {
                    0 => 10,
                    1 => 14,
                    2 => 22,
                    _ => 30,
                };
                let val = u.int_in_range(0u64..=((1 << bits) - 1))?;
                let val = (val << 2) | selector as u64;

                buf |= val << buf_bits;
                buf_bits += bits + 2;
                if buf_bits >= 32 {
                    let bytes_to_add = (buf as u32).to_le_bytes();
                    bytes.extend_from_slice(&bytes_to_add);
                    buf >>= 32;
                    buf_bits -= 32;
                }

                num_entries += 1;
                Ok(ControlFlow::Continue(()))
            })?;
            if buf_bits > 0 {
                let bytes_to_add = (buf as u32).to_le_bytes();
                bytes.extend_from_slice(&bytes_to_add);
            }

            let mut br = BitReader::new(&bytes);
            let expected =
                Toc::read_unconditional(&(), &mut br, &TocNonserialized { num_entries }).unwrap();

            let mut br = BitReader::new(&bytes);
            let mut parser = IncrementalTocReader::new(num_entries, &mut br).unwrap();
            while !parser.is_complete() {
                parser.read_step(&mut br).unwrap();
            }
            let actual = parser.finalize();

            assert_eq!(actual, expected);
            Ok(())
        });
    }
}
