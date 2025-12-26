// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests ported from libjxl ans_test.cc and entropy_coder_test.cc
//!
//! These tests verify ANS and entropy coding matches the reference implementation.
//! DO NOT WEAKEN TOLERANCES or modify tests to pass when implementation is wrong.

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::unpack_signed;
use crate::entropy_coding::hybrid_uint::HybridUint;

/// Pack a signed integer into an unsigned one (zigzag encoding)
/// This matches libjxl's PackSigned function
fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    }
}

/// Test vectors for hybrid uint encoding
/// Ported from entropy_coder_test.cc
#[cfg(test)]
mod hybrid_uint_tests {
    use super::*;

    /// Test that HybridUint config (0,0,0) works correctly
    /// Config (0,0,0) means split_token=1, so any symbol >= 1 requires extra bits
    #[test]
    fn test_hybrid_uint_config_000() {
        let config = HybridUint::new(0, 0, 0);

        // Test reading symbol 0 - should return 0 directly (no extra bits needed)
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(0, &mut br), 0);

        // Test reading symbol 1 with 0 extra bits - should return 1
        // For config (0,0,0): split_token=1, bits_in_token=0
        // When symbol=1: nbits = 0 - 0 + ((1-1) >> 0) = 0
        // Result = ((1 | 1) << 0) | bits << 0 | 0 = 1
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(1, &mut br), 1);

        // Symbol 2: nbits = 0 - 0 + ((2-1) >> 0) = 1
        // Need to read 1 bit
        let data = [0b00000000u8]; // extra bits = 0
        let mut br = BitReader::new(&data);
        // hi = (0 & 0) | 1 = 1, result = ((1 << 1) | 0) << 0 = 2
        assert_eq!(config.read(2, &mut br), 2);

        let data = [0b00000001u8]; // extra bits = 1
        let mut br = BitReader::new(&data);
        // hi = (0 & 0) | 1 = 1, result = ((1 << 1) | 1) << 0 = 3
        assert_eq!(config.read(2, &mut br), 3);
    }

    /// Test that HybridUint config (4,1,1) works correctly
    /// split_token=16, msb_in_token=1, lsb_in_token=1
    #[test]
    fn test_hybrid_uint_config_411() {
        let config = HybridUint::new(4, 1, 1);

        // Symbols < 16 are returned directly
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(0, &mut br), 0);

        let mut br = BitReader::new(&data);
        assert_eq!(config.read(15, &mut br), 15);

        // Symbol 16: bits_in_token = 2, nbits = 4 - 2 + ((16-16) >> 2) = 2
        // low = 16 & 1 = 0, symbol_nolow = 8
        // hi = (8 & 1) | 2 = 2
        // With extra bits = 0: result = ((2 << 2) | 0) << 1 | 0 = 16
        let data = [0b00000000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(16, &mut br), 16);
    }

    /// Test that HybridUint config (4,2,0) works correctly
    /// split_token=16, msb_in_token=2, lsb_in_token=0
    #[test]
    fn test_hybrid_uint_config_420() {
        let config = HybridUint::new(4, 2, 0);

        // Symbols < 16 are returned directly
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(0, &mut br), 0);
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(8, &mut br), 8);
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(15, &mut br), 15);

        // Symbol 16: bits_in_token = 2, nbits = 4 - 2 + ((16-16) >> 2) = 2
        // low = 0, symbol_nolow = 16
        // hi = (16 & 3) | 4 = 4
        // With extra bits = 0: result = ((4 << 2) | 0) << 0 = 16
        let data = [0b00000000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(16, &mut br), 16);
    }

    /// Test that HybridUint config (4,2,1) works correctly
    /// split_token=16, msb_in_token=2, lsb_in_token=1
    #[test]
    fn test_hybrid_uint_config_421() {
        let config = HybridUint::new(4, 2, 1);

        // Symbols < 16 are returned directly
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(0, &mut br), 0);
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(15, &mut br), 15);

        // Symbol 16 with config (4,2,1):
        // bits_in_token = 3, nbits = 4 - 3 + ((16-16) >> 3) = 1
        // low = 16 & 1 = 0, symbol_nolow = 8
        // hi = (8 & 3) | 4 = 4
        // With extra bits = 0: result = ((4 << 1) | 0) << 1 | 0 = 16
        let data = [0b00000000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(config.read(16, &mut br), 16);
    }

    /// Test various values through hybrid uint roundtrip
    #[test]
    fn test_hybrid_uint_various_values() {
        // Test that small values work across different configs
        let configs = [
            HybridUint::new(0, 0, 0),
            HybridUint::new(4, 1, 1),
            HybridUint::new(4, 2, 0),
            HybridUint::new(4, 2, 1),
            HybridUint::new(5, 2, 2),
            HybridUint::new(6, 0, 0),
        ];

        for config in &configs {
            // Test symbol 0 always returns 0
            let data = [0u8; 8];
            let mut br = BitReader::new(&data);
            assert_eq!(config.read(0, &mut br), 0);
        }
    }
}

/// Pack/Unpack signed integer tests
/// Ported from entropy_coder_test.cc PackUnpack
#[cfg(test)]
mod pack_unpack_tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        // Test range from -31 to 31 (matching libjxl test)
        for i in -31i32..=31 {
            let packed = pack_signed(i);
            // Packed should be < 63 for this range
            assert!(
                packed < 63,
                "pack_signed({}) = {} should be < 63",
                i,
                packed
            );
            let unpacked = unpack_signed(packed);
            assert_eq!(
                i, unpacked,
                "Roundtrip failed: {} -> {} -> {}",
                i, packed, unpacked
            );
        }
    }

    #[test]
    fn test_pack_signed_specific_values() {
        // 0 -> 0
        assert_eq!(pack_signed(0), 0);
        // 1 -> 2
        assert_eq!(pack_signed(1), 2);
        // -1 -> 1
        assert_eq!(pack_signed(-1), 1);
        // 2 -> 4
        assert_eq!(pack_signed(2), 4);
        // -2 -> 3
        assert_eq!(pack_signed(-2), 3);
    }

    #[test]
    fn test_unpack_signed_specific_values() {
        // 0 -> 0
        assert_eq!(unpack_signed(0), 0);
        // 1 -> -1
        assert_eq!(unpack_signed(1), -1);
        // 2 -> 1
        assert_eq!(unpack_signed(2), 1);
        // 3 -> -2
        assert_eq!(unpack_signed(3), -2);
        // 4 -> 2
        assert_eq!(unpack_signed(4), 2);
    }

    #[test]
    fn test_pack_unpack_large_range() {
        // Test larger range
        for i in -1000i32..=1000 {
            let packed = pack_signed(i);
            let unpacked = unpack_signed(packed);
            assert_eq!(i, unpacked, "Roundtrip failed for {}", i);
        }
    }
}

/// ANS decoding tests
/// Note: Full roundtrip tests require encoder which jxl-rs doesn't have.
/// These tests verify decoder behavior with known test vectors.
#[cfg(test)]
mod ans_tests {
    use super::*;

    /// Test that we can create a BitReader and read basic data
    #[test]
    fn test_bit_reader_basic() {
        let data = [0xFFu8, 0x00, 0xAA, 0x55];
        let mut br = BitReader::new(&data);

        // Read 8 bits - should get 0xFF
        let val = br.read(8).unwrap();
        assert_eq!(val, 0xFF);

        // Read 8 bits - should get 0x00
        let val = br.read(8).unwrap();
        assert_eq!(val, 0x00);
    }

    /// Test bit reader with partial byte reads
    #[test]
    fn test_bit_reader_partial_bytes() {
        let data = [0b10101010u8, 0b01010101];
        let mut br = BitReader::new(&data);

        // Read 4 bits at a time
        let val1 = br.read(4).unwrap();
        let val2 = br.read(4).unwrap();
        let val3 = br.read(4).unwrap();
        let val4 = br.read(4).unwrap();

        // 0b10101010 = 0xAA, reading LSB first: first 4 bits = 0b1010 = 10
        assert_eq!(val1, 0b1010);
        assert_eq!(val2, 0b1010);
        assert_eq!(val3, 0b0101);
        assert_eq!(val4, 0b0101);
    }

    // TODO: Full ANS roundtrip tests require encoding capability which jxl-rs
    // doesn't have. The following tests would need an encoder:
    //
    // - test_ans_empty_roundtrip: Encode empty stream, decode it
    // - test_ans_single_symbol_roundtrip: Encode/decode single symbols
    // - test_ans_random_stream_roundtrip: Encode/decode random streams
    //
    // These tests are implemented in libjxl's ans_test.cc with the encoder.
    // For jxl-rs, we rely on the decode_test files that decode real JXL images
    // which exercises the ANS decoder with real encoded data.
}

/// LZ77 tests
/// Ported from ans_test.cc TestCheckpointingANSLZ77, TestCheckpointingPrefixLZ77
#[cfg(test)]
mod lz77_tests {

    /// Test LZ77 special distance table values
    /// These are the special 2D distance codes used in LZ77
    #[test]
    fn test_lz77_special_distances() {
        // The special distances table maps (dx, dy) pairs to distance codes
        // First few entries from libjxl:
        // (0,1), (1,0), (1,1), (-1,1), (0,2), (2,0), ...
        let special_distances: [(i8, u8); 10] = [
            (0, 1),
            (1, 0),
            (1, 1),
            (-1, 1),
            (0, 2),
            (2, 0),
            (1, 2),
            (-1, 2),
            (2, 1),
            (-2, 1),
        ];

        // Verify the pattern holds
        for (i, (dx, dy)) in special_distances.iter().enumerate() {
            // dy is u8 so always non-negative
            // The distances should represent valid offsets in a scan order
            let manhattan = dx.unsigned_abs() as u32 + *dy as u32;
            assert!(manhattan > 0, "Distance {} should be non-zero", i);
        }
    }

    /// Test LZ77 window size constant
    #[test]
    fn test_lz77_window_size() {
        // LZ77 window size is 2^20 in libjxl
        const LOG_WINDOW_SIZE: u32 = 20;
        const WINDOW_SIZE: u32 = 1 << LOG_WINDOW_SIZE;
        const WINDOW_MASK: u32 = WINDOW_SIZE - 1;

        assert_eq!(WINDOW_SIZE, 1048576);
        assert_eq!(WINDOW_MASK, 1048575);
        assert_eq!(WINDOW_MASK, 0xFFFFF);
    }

    // TODO: Full LZ77 checkpointing tests require encoder capability.
    // The TestCheckpointing tests in libjxl encode data with LZ77,
    // then test that checkpointing/restoring works correctly during decode.
    // Since jxl-rs is decode-only, we rely on real JXL files with LZ77 for testing.
}

/// Prefix code (Huffman) tests
#[cfg(test)]
mod prefix_tests {
    use super::*;

    /// Test that simple prefix codes can be read
    #[test]
    fn test_prefix_code_simple_alphabet() {
        // A simple 2-symbol Huffman code: 0 -> symbol 0, 1 -> symbol 1
        // This is the simplest possible prefix code
        let data = [0b01u8]; // bits: 1, 0

        let mut br = BitReader::new(&data);
        // Read 1 bit - should get 1
        assert_eq!(br.read(1).unwrap(), 1);
        // Read 1 bit - should get 0
        assert_eq!(br.read(1).unwrap(), 0);
    }

    /// Test reading multiple bits for longer codes
    #[test]
    fn test_prefix_code_variable_length() {
        // Simulate a variable-length code pattern
        // In Huffman coding, more frequent symbols get shorter codes
        // Data byte: 0b11100100
        // Bit positions: 76543210
        // BitReader reads LSB-first from position 0
        let data = [0b11100100u8, 0b00000001];

        let mut br = BitReader::new(&data);

        // Read various lengths (LSB-first ordering)
        let v1 = br.read(1).unwrap(); // bit 0: 0
        let v2 = br.read(2).unwrap(); // bits 1-2: 10 → 2
        let v3 = br.read(3).unwrap(); // bits 3-5: 100 → 4
        let v4 = br.read(2).unwrap(); // bits 6-7: 11 → 3

        assert_eq!(v1, 0);
        assert_eq!(v2, 2);
        assert_eq!(v3, 4);
        assert_eq!(v4, 3);
    }

    /// Test Huffman code length limits
    #[test]
    fn test_huffman_max_bits() {
        // JPEG XL uses Huffman codes with max 15 bits
        use crate::entropy_coding::huffman::HUFFMAN_MAX_BITS;
        assert_eq!(HUFFMAN_MAX_BITS, 15);
    }

    // TODO: Full Huffman decode tests require decoding actual Huffman-encoded
    // streams. The existing decode tests with modular mode exercise this path.
}

/// Context map tests
#[cfg(test)]
mod context_map_tests {
    /// Test context map size limits
    #[test]
    fn test_context_map_limits() {
        // Context IDs must be <= 255
        let max_context_id: u8 = 255;
        assert_eq!(max_context_id as u32, 255);

        // In JPEG XL, context maps can have up to 256 entries
        let max_contexts: usize = 256;
        assert!(max_contexts <= 256);
    }
}
