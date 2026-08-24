// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression fixtures ported from upstream jxl-rs (see
//! `docs/UPSTREAM_SYNC.md`). The fixtures live in `resources/test/`, so the
//! generic sweeps in `api::decoder::tests` (one-shot, 1-byte chunks, simple
//! vs low-memory pipeline, incremental-with-flush vs one-shot) cover them
//! too; the tests here pin the *values* the upstream bug corrupted.

use crate::api::decoder::tests::decode;
use crate::util::test::fixture_bytes;

/// jxl-rs #728 (fixed in #739): the vertical squeeze step read the wrong
/// neighbour at the group boundary of a tall image (512x8240, many group
/// rows), turning a two-level image into intermediate values.
#[test]
fn issue728_vsqueeze_boundary_values_are_exact() {
    let (_, frames) = decode(
        &fixture_bytes("issue728_minimal.jxl"),
        usize::MAX,
        false,
        false,
        None,
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    let buf = &frames[0][0];
    let (xs, ys) = buf.size();
    assert_eq!((xs, ys), (512 * 3, 8240));
    for y in 0..ys {
        for (x, &v) in buf.row(y).iter().enumerate().take(xs) {
            assert!(
                v == 0.0 || v == 1.0,
                "pixel ({}, {y}) = {v}, expected 0.0 or 1.0",
                x / 3
            );
        }
    }
}

/// jxl-rs #734: the horizontal squeeze step at the last (odd) column of a
/// 257-px image; the whole image must stay solid blue.
#[test]
fn issue734_hsqueeze_odd_width_stays_solid_blue() {
    let (_, frames) = decode(
        &fixture_bytes("strategic_solid_blue.jxl"),
        usize::MAX,
        false,
        false,
        None,
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    let buf = &frames[0][0];
    let (xs, ys) = buf.size();
    assert_eq!((xs, ys), (257 * 3, 256));
    for y in 0..ys {
        let row = buf.row(y);
        for x in 0..257 {
            let px = (row[x * 3], row[x * 3 + 1], row[x * 3 + 2]);
            assert_eq!(px, (0.0, 0.0, 1.0), "pixel ({x}, {y})");
        }
    }
}

/// jxl-rs #772 (fixed in ebeed75): `AlphaWeightedAddBelow` blending copied
/// the whole foreground alpha row instead of the clipped `xsize` part when a
/// reference frame was missing, panicking / corrupting the last group
/// column. 750x1000 RGBA with a blended frame.
///
/// The reference is the RGBA8 output of jxl-rs 0.6 (`088ec7f`), which this
/// decoder reproduces bit-for-bit on macOS/NEON (FNV-1a 0x4878a5734c1fcfa0)
/// but not on every platform: other SIMD tiers, and the NEON tier on Windows
/// (a different libm), round the blended floats slightly differently and
/// move a few percent of the samples by one code under the dither (libjxl
/// `djxl` 0.12.0 differs from jxl-rs the same way). So the pinned reference
/// is the per-channel mean of each 50x50 block, which the corruption shifts
/// by tens of codes and platform noise by at most 0.5 (measured).
#[test]
fn issue772_clipped_blend_matches_jxl_rs() {
    const BLOCK: usize = 50;
    #[rustfmt::skip]
    const REFERENCE_BLOCK_MEANS: [u8; 15 * 20 * 4] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 2, 2, 0, 22, 3, 4, 0,
    22, 4, 4, 0, 22, 3, 4, 0, 22, 4, 4, 0, 22, 3, 4, 0, 22, 4, 4, 0,
    22, 3, 4, 0, 10, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 7, 7, 0, 91, 15, 15, 0,
    91, 15, 15, 0, 91, 15, 15, 0, 91, 15, 15, 0, 91, 15, 15, 0, 91, 15, 15, 0,
    91, 15, 15, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 7, 7, 0, 91, 15, 15, 0,
    91, 15, 15, 0, 91, 15, 15, 0, 91, 15, 15, 0, 91, 15, 15, 0, 91, 15, 15, 0,
    91, 15, 15, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 7, 7, 0, 91, 15, 15, 0,
    105, 17, 17, 0, 91, 15, 15, 0, 91, 15, 15, 0, 92, 15, 15, 0, 92, 15, 15, 0,
    102, 16, 16, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 8, 23, 51, 34, 28, 48, 108, 73, 45, 62, 135, 91,
    51, 63, 138, 93, 50, 55, 119, 78, 45, 34, 71, 44, 36, 8, 10, 3, 73, 12, 12, 0,
    114, 18, 18, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 20, 55, 126, 86, 29, 78, 182, 137, 29, 80, 182, 128, 29, 80, 182, 128,
    29, 80, 182, 128, 29, 80, 182, 128, 29, 78, 182, 134, 24, 66, 151, 104, 59, 15, 22, 8,
    91, 15, 15, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 12, 28, 18, 29, 80, 182, 128, 29, 71, 182, 160, 29, 71, 182, 162, 29, 71, 182, 160,
    29, 72, 182, 159, 29, 73, 182, 154, 29, 72, 182, 159, 29, 80, 182, 128, 67, 37, 74, 44,
    91, 15, 15, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 20, 55, 126, 86, 29, 77, 182, 139, 29, 76, 182, 142, 29, 76, 182, 144,
    29, 77, 182, 139, 29, 79, 182, 133, 29, 76, 182, 142, 24, 66, 151, 104, 72, 27, 34, 8,
    91, 15, 15, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 8, 23, 51, 34, 28, 48, 108, 73, 48, 65, 139, 91,
    45, 62, 137, 93, 42, 54, 117, 78, 34, 33, 69, 44, 24, 6, 8, 3, 73, 18, 18, 0,
    100, 22, 22, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 7, 7, 0, 107, 29, 29, 0,
    92, 16, 16, 0, 92, 16, 16, 0, 91, 15, 15, 0, 93, 16, 17, 0, 92, 16, 16, 0,
    107, 29, 29, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 7, 7, 0, 97, 20, 20, 0,
    105, 27, 27, 0, 109, 31, 31, 0, 107, 29, 29, 0, 109, 31, 31, 0, 98, 21, 21, 0,
    105, 28, 28, 0, 42, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 7, 7, 0, 91, 15, 15, 0,
    55, 120, 24, 80, 33, 179, 29, 125, 33, 179, 29, 125, 33, 179, 29, 125, 34, 180, 30, 125,
    33, 179, 29, 125, 32, 179, 29, 125, 31, 178, 29, 125, 12, 68, 11, 48, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 6, 6, 0, 88, 14, 14, 0,
    52, 122, 24, 82, 32, 182, 29, 129, 32, 182, 29, 161, 32, 182, 29, 151, 32, 182, 29, 158,
    32, 182, 29, 155, 32, 182, 29, 137, 32, 182, 29, 160, 12, 69, 11, 49, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    20, 117, 19, 82, 32, 182, 29, 129, 32, 182, 29, 152, 32, 182, 29, 157, 32, 182, 29, 155,
    32, 182, 29, 150, 32, 182, 29, 132, 32, 182, 29, 140, 12, 69, 11, 49, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18, 105, 17, 74, 29, 164, 26, 115, 29, 164, 26, 115, 29, 164, 26, 115, 29, 164, 26, 115,
    29, 164, 26, 115, 29, 164, 26, 115, 29, 164, 26, 115, 11, 62, 10, 44, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let img = crate::decode(&fixture_bytes("issue772_blendbug.jxl")).unwrap();
    assert_eq!((img.width, img.height, img.channels), (750, 1000, 4));
    for by in 0..img.height / BLOCK {
        for bx in 0..img.width / BLOCK {
            for c in 0..4 {
                let mut sum = 0u64;
                for y in by * BLOCK..(by + 1) * BLOCK {
                    let row = &img.data[y * img.width * 4..(y + 1) * img.width * 4];
                    for x in bx * BLOCK..(bx + 1) * BLOCK {
                        sum += row[x * 4 + c] as u64;
                    }
                }
                let mean = sum as f64 / (BLOCK * BLOCK) as f64;
                let reference = REFERENCE_BLOCK_MEANS[(by * 15 + bx) * 4 + c] as f64;
                let diff = (mean - reference).abs();
                assert!(
                    diff <= 1.0,
                    "block ({bx}, {by}) channel {c}: mean {mean:.2}, jxl-rs {reference}"
                );
            }
        }
    }
}

/// jxl-rs #875 (fix 365eb80, test 6401d6e): rendering of chroma-subsampled
/// frames read the LF image of the wrong LF group once there was more than
/// one (2333x2333 4:2:0 -> 2x2 LF groups), flattening the colour of the
/// other LF groups. The fixture has a saturated green rectangle at the start
/// of LF group (1, 0) and a red one at (0, 1); both must stay saturated.
/// This decoder's LF layout never had the bug; the test guards it.
#[test]
fn issue875_subsampled_frame_uses_the_right_lf_group() {
    let (_, mut frames) = decode(
        &fixture_bytes("multiple_lf_420.jxl"),
        usize::MAX,
        false,
        false,
        None,
    )
    .unwrap();
    let frame = frames.pop().unwrap();
    let [image]: [_; 1] = frame.try_into().unwrap();
    for (name, origin) in [("green", (2048 * 3, 0)), ("red", (0, 2048))] {
        let view = image.get_rect(crate::image::Rect {
            origin,
            size: (16 * 3, 16),
        });
        for y in 0..view.size().1 {
            for (x, px) in view.row(y).chunks(3).enumerate() {
                let [r, g, b] = px else { unreachable!() };
                let max = r.max(*g).max(*b);
                let min = r.min(*g).min(*b);
                assert!(
                    max - min > 0.5,
                    "{name} rect pixel ({x}, {y}) = ({r}, {g}, {b}) is not saturated"
                );
            }
        }
    }
}

/// Chromium issue 541318910 (jxl-rs e12b99b): flushing a frame that does not
/// support rendering before its last pass used to force an eager render of
/// an incomplete group (a squeeze with an empty residual channel).
#[test]
fn flush_without_partial_render_support() {
    let data = fixture_bytes("squeeze_empty_residual.jxl");
    for chunk_size in 1..=16 {
        decode(&data, chunk_size, false, true, None)
            .unwrap_or_else(|e| panic!("chunk size {chunk_size}: {e:?}"));
    }
}

/// Streaming a two-frame file (an LF frame plus a VarDCT frame with an
/// extra channel) in 30000-byte chunks with a flush after every chunk used
/// to panic in STEP 5 of `Frame::decode_and_render_hf_groups`: the flush
/// re-sent the extra channel of every group the modular dry run had
/// reported, and a transform with neighbour dependencies reports the 3x3
/// neighbourhood of a newly decoded group, including groups whose sections
/// had not arrived (no data to send). The streamed result must also equal
/// the one-shot decode.
#[cfg(target_pointer_width = "64")] // 24 MP x 2 decodes: too much for a 32-bit address space
#[test]
fn streamed_flush_of_undecoded_neighbour_groups() {
    let data = fixture_bytes("tirr_photo.jxl");
    let (_, one_shot) = decode(&data, usize::MAX, false, false, None).unwrap();
    let (_, streamed) = decode(&data, 30_000, false, true, None).unwrap();
    assert_eq!(one_shot.len(), streamed.len());
    for (fc, (a, b)) in one_shot.iter().zip(&streamed).enumerate() {
        assert_eq!(a.len(), b.len(), "frame {fc}");
        for (c, (ia, ib)) in a.iter().zip(b).enumerate() {
            assert_eq!(ia.size(), ib.size(), "frame {fc} channel {c}");
            for y in 0..ia.size().1 {
                assert!(
                    ia.row(y) == ib.row(y),
                    "frame {fc} channel {c} row {y}: streamed decode differs from one-shot"
                );
            }
        }
    }
}

/// jxl-rs #865 (fixed in 4413527): the incremental parser stopped refilling
/// its buffer after making progress inside a section, stalling on files
/// whose TOC alone is bigger than one refill (5249x5377 = 462 groups). The
/// fork's parser refills from the available input (see the huge-toc test in
/// `tests/section_buffer_alloc.rs`); this pins the file end-to-end: the
/// streamed decode must finish and match the one-shot decode exactly.
#[cfg(target_pointer_width = "64")] // 28 MP x 2 decodes
#[test]
fn issue865_large_toc_streams_and_matches_one_shot() {
    let data = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/testdata/jxlrs-865/issue865_large_toc.jxl"),
    )
    .unwrap();
    let (_, one_shot) = decode(&data, usize::MAX, false, false, None).unwrap();
    assert_eq!(one_shot.len(), 1);
    assert_eq!(one_shot[0][0].size(), (5249 * 3, 5377));
    let (_, streamed) = decode(&data, 123, false, true, None).unwrap();
    for (c, (ia, ib)) in one_shot[0].iter().zip(&streamed[0]).enumerate() {
        assert_eq!(ia.size(), ib.size(), "channel {c}");
        for y in 0..ia.size().1 {
            assert!(
                ia.row(y) == ib.row(y),
                "channel {c} row {y}: streamed decode differs from one-shot"
            );
        }
    }
}
