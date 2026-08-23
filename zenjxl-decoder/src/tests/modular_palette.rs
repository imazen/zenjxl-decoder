// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Modular palette transform regressions.
//!
//! No mainstream encoder emits a delta palette with the `Weighted`
//! predictor (libjxl's encoder hard-codes `Average4` for lossy palettes), so
//! the bitstream exercising that path is produced here by **patching** the
//! `predictor` field of the palette transform in the conformance
//! `delta_palette.jxl` from `Average4` (13) to `Weighted` (6). Parsing is
//! unaffected (only the inverse transform changes), so any spec-conformant
//! decoder must produce the same -- deterministic, if visually meaningless --
//! pixels. The expected hash was taken from libjxl `djxl` 0.12.0 and
//! cross-checked against jxl-rs 0.6 (`088ec7f`), which agree exactly.

use crate::{
    bit_reader::BitReader,
    container::ContainerParser,
    frame::{modular::Tree, quantizer::LfQuantFactors},
    headers::{
        FileHeader, JxlHeader,
        encodings::UnconditionalCoder,
        frame_header::{Encoding, FrameHeader},
        modular::{GroupHeader, TransformId},
        toc::{Toc, TocNonserialized},
    },
    util::MemoryTracker,
};

/// Predictor ids as serialised in the palette transform header.
const PREDICTOR_WEIGHTED: u32 = 6;
const PREDICTOR_AVERAGE4: u32 = 13;

/// Locate the 4 predictor bits of the last transform of the global modular
/// header (which must be a palette) and return the absolute *bit* offset of
/// that field inside `codestream`.
///
/// Mirrors the parse order of `Frame::decode_lf_global` for a modular frame
/// without patches / splines / noise: LF quant factors, optional global tree,
/// then the group header whose last field (for a palette transform) is
/// `predictor_id: Bits(4)`.
fn palette_predictor_bit_offset(codestream: &[u8]) -> (usize, u32) {
    let mut br = BitReader::new(codestream);
    let file_header = FileHeader::read(&mut br).unwrap();
    let frame_header =
        FrameHeader::read_unconditional(&(), &mut br, &file_header.frame_header_nonserialized())
            .unwrap();
    assert_eq!(frame_header.encoding, Encoding::Modular);
    assert!(
        !frame_header.has_patches() && !frame_header.has_splines() && !frame_header.has_noise(),
        "patcher only handles plain modular frames"
    );
    let toc = Toc::read_unconditional(
        &(),
        &mut br,
        &TocNonserialized {
            num_entries: frame_header.num_toc_entries() as u32,
        },
    )
    .unwrap();
    br.jump_to_byte_boundary().unwrap();
    let sec0_byte = br.total_bits_read() / 8;
    let sec0_len = toc.entries[0] as usize;
    let section = &codestream[sec0_byte..sec0_byte + sec0_len];

    let mut sbr = BitReader::new(section);
    LfQuantFactors::new(&mut sbr).unwrap();
    if sbr.read(1).unwrap() == 1 {
        Tree::read(&mut sbr, 1 << 22, &MemoryTracker::unlimited()).unwrap();
    }
    let header = GroupHeader::read(&mut sbr).unwrap();
    let end_bits = sbr.total_bits_read();
    let last = header
        .transforms
        .last()
        .expect("global modular header has no transforms");
    assert_eq!(
        last.id,
        TransformId::Palette,
        "last transform is not a palette"
    );
    // `delta_palette.jxl` has num_colors == num_deltas == 0: it uses the
    // *implicit* palette, where negative indices select the built-in delta
    // entries (libjxl `GetPaletteValue`), so `index < nb_deltas` is still the
    // delta condition. No explicit delta count is required.
    (sec0_byte * 8 + end_bits - 4, last.predictor_id)
}

fn read_bits_lsb(data: &[u8], bit_offset: usize, n: usize) -> u32 {
    (0..n).fold(0u32, |acc, i| {
        let b = bit_offset + i;
        acc | ((((data[b / 8] >> (b % 8)) & 1) as u32) << i)
    })
}

fn write_bits_lsb(data: &mut [u8], bit_offset: usize, n: usize, value: u32) {
    for i in 0..n {
        let b = bit_offset + i;
        let mask = 1u8 << (b % 8);
        if (value >> i) & 1 == 1 {
            data[b / 8] |= mask;
        } else {
            data[b / 8] &= !mask;
        }
    }
}

/// `delta_palette.jxl` with its palette predictor rewritten to `Weighted`.
fn delta_palette_with_weighted_predictor() -> Vec<u8> {
    let file = crate::util::test::fixture_bytes("conformance_test_images/delta_palette.jxl");
    let mut codestream = ContainerParser::collect_codestream(&file).unwrap();
    let (bit, predictor) = palette_predictor_bit_offset(&codestream);
    assert_eq!(predictor, PREDICTOR_AVERAGE4, "fixture changed?");
    assert_eq!(read_bits_lsb(&codestream, bit, 4), PREDICTOR_AVERAGE4);
    write_bits_lsb(&mut codestream, bit, 4, PREDICTOR_WEIGHTED);
    let (bit2, predictor2) = palette_predictor_bit_offset(&codestream);
    assert_eq!((bit2, predictor2), (bit, PREDICTOR_WEIGHTED));
    codestream
}

/// FNV-1a over the 8-bit RGB samples, row-major, as `djxl` writes them.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// Sanity check of the patcher itself: the *unpatched* fixture must still
/// decode to the conformance result (every pixel of `delta_palette.jxl` is
/// 8-bit exact, so a hash of the u8 output is a complete check).
#[test]
fn delta_palette_unpatched_hash() {
    let file = crate::util::test::fixture_bytes("conformance_test_images/delta_palette.jxl");
    let img = crate::decode(&file).unwrap();
    assert_eq!((img.width, img.height, img.channels), (555, 751, 4));
    let rgb = img
        .data
        .chunks_exact(4)
        .flat_map(|px| px[..3].iter().copied());
    // djxl 0.12.0 output, RGB8.
    assert_eq!(fnv1a(rgb), 0xfc236fafc4b975a7);
}

/// jxl-rs #791 ("Fix DeltaPalette with Weighted predictor").
///
/// Two bugs in the inverse palette with `nb_deltas > 0` and
/// `predictor == Weighted`:
/// 1. the weighted predictor was only evaluated for delta indices, but its
///    error state must be updated for **every** pixel (libjxl
///    `palette.cc` calls `PredictNoTreeWP` unconditionally), so every
///    delta pixel after a non-delta pixel used stale predictions;
/// 2. the weighted-predictor state carried across group rows saved only the
///    wrong half of the error row and none of the per-predictor error rows.
///
/// 555x751 with 256-px groups gives a 3x3 grid, so both the per-pixel and the
/// cross-group-row paths are exercised.
#[test]
fn delta_palette_weighted_predictor_matches_reference() {
    let codestream = delta_palette_with_weighted_predictor();
    let img = crate::decode(&codestream).unwrap();
    assert_eq!((img.width, img.height, img.channels), (555, 751, 4));
    let rgb = img
        .data
        .chunks_exact(4)
        .flat_map(|px| px[..3].iter().copied());
    // djxl 0.12.0 == jxl-rs 0.6 (088ec7f) output, RGB8.
    assert_eq!(fnv1a(rgb), 0x4927d8ae25cfcec3);
}
