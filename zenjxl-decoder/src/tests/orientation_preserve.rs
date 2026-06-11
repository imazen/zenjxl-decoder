// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests that `JxlDecoderOptions::adjust_orientation` is load-bearing.
//!
//! "Correct" (`adjust_orientation = true`, the default) bakes the stored
//! orientation into the output pixels: the emitted image is upright, reported
//! at display dimensions, with a residual orientation of `Identity`.
//!
//! "Preserve" (`adjust_orientation = false`) skips the bake: pixels are emitted
//! in their stored (coded) orientation and dimensions, and the intrinsic
//! orientation is surfaced on the basic info so a later stage can bake it.
//!
//! The pixel-sacredness assertion is the core of this test: the two outputs
//! must differ by *exactly* the orientation transform and nothing else. We
//! verify that by mapping the Preserve (coded) buffer through the stored
//! orientation and asserting it is bit-for-bit equal to the Correct (baked)
//! buffer.

use crate::api::{JxlDecoderOptions, decode_with};
use crate::headers::Orientation;

/// Path to a git-tracked fixture whose codestream carries orientation 5
/// (Transpose). Transpose is a *transposing* orientation, so the display
/// dimensions are the coded dimensions with width/height swapped — this also
/// exercises the stored-vs-display dimension reporting.
fn transpose_fixture() -> Vec<u8> {
    // src/tests/ lives under CARGO_MANIFEST_DIR (the `zenjxl-decoder` crate);
    // the committed seed corpus is one level up at <repo>/fuzz/seed_corpus.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fuzz/seed_corpus/decode/orientation5_transpose.jxl");
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("missing committed fixture {}: {e}", path.display()))
}

/// Map a destination (display) pixel back to its source (coded) pixel for the
/// given stored orientation. This is the inverse of the save stage's
/// coded→display mapping: for destination pixel `(x, y)` in an image of display
/// size `disp`, return the coded pixel it was sourced from.
///
/// We use the decoder's own `Orientation::display_pixel`, which (by its
/// construction in `headers::image_metadata`) maps a *coded* coordinate to its
/// *display* coordinate given the coded size. For the orientations under test
/// here that mapping is an involution on coordinates when the size argument is
/// chosen consistently, but to stay fully general we build the forward coded→
/// display map and invert it explicitly.
fn coded_pixel_for_display(
    orientation: Orientation,
    dest: (usize, usize),
    coded: (usize, usize),
) -> (usize, usize) {
    // Build the forward map coded -> display once would be O(N); instead invert
    // analytically per orientation. `coded` is (w, h) in coded space.
    let (cw, ch) = coded;
    let (dx, dy) = dest;
    match orientation {
        Orientation::Identity => (dx, dy),
        Orientation::FlipHorizontal => (cw - 1 - dx, dy),
        Orientation::Rotate180 => (cw - 1 - dx, ch - 1 - dy),
        Orientation::FlipVertical => (dx, ch - 1 - dy),
        // Transposing orientations: display size is (ch, cw).
        Orientation::Transpose => (dy, dx),
        Orientation::Rotate90Cw => (dy, ch - 1 - dx),
        Orientation::AntiTranspose => (ch - 1 - dy, cw - 1 - dx),
        Orientation::Rotate90Ccw => (cw - 1 - dy, dx),
    }
}

#[test]
fn adjust_orientation_correct_bakes_and_reports_identity() {
    let data = transpose_fixture();

    let img = decode_with(
        &data,
        JxlDecoderOptions {
            adjust_orientation: true,
            ..Default::default()
        },
    )
    .expect("decode (Correct) failed");

    // Stored orientation is Transpose; the bake makes the residual Identity.
    assert_eq!(
        img.info.intrinsic_orientation,
        Orientation::Transpose,
        "fixture should carry a Transpose intrinsic orientation"
    );
    assert_eq!(
        img.info.orientation,
        Orientation::Identity,
        "Correct mode bakes orientation, so the residual must be Identity"
    );
    // Display size = coded size transposed.
    let (cw, ch) = img.info.coded_size;
    assert_eq!(
        img.info.size,
        (ch, cw),
        "Correct mode must report display (transposed) dimensions"
    );
    assert_eq!(
        (img.width, img.height),
        img.info.size,
        "emitted buffer dims must match reported size"
    );
}

#[test]
fn adjust_orientation_preserve_skips_bake_and_surfaces_orientation() {
    let data = transpose_fixture();

    let img = decode_with(
        &data,
        JxlDecoderOptions {
            adjust_orientation: false,
            ..Default::default()
        },
    )
    .expect("decode (Preserve) failed");

    // Intrinsic orientation is still surfaced...
    assert_eq!(
        img.info.intrinsic_orientation,
        Orientation::Transpose,
        "intrinsic orientation must always be the stored value"
    );
    // ...and in Preserve mode the residual equals it (nothing baked yet).
    assert_eq!(
        img.info.orientation,
        Orientation::Transpose,
        "Preserve mode must surface the stored orientation as the residual"
    );
    // Coded (stored) dimensions, NOT display dimensions.
    assert_eq!(
        img.info.size, img.info.coded_size,
        "Preserve mode must report stored (coded) dimensions"
    );
    assert_eq!(
        (img.width, img.height),
        img.info.coded_size,
        "emitted buffer dims must equal the coded dims in Preserve mode"
    );
}

/// Pixels are sacred: the Correct (baked) and Preserve (un-baked) outputs must
/// differ by EXACTLY the stored orientation transform — no resampling, no
/// off-by-one, no color change. We decode the same file both ways and assert
/// the un-baked buffer, remapped through the stored orientation, equals the
/// baked buffer bit-for-bit.
#[test]
fn correct_equals_preserve_under_exact_orientation_transform() {
    let data = transpose_fixture();

    let baked = decode_with(
        &data,
        JxlDecoderOptions {
            adjust_orientation: true,
            ..Default::default()
        },
    )
    .expect("decode (Correct) failed");

    let unbaked = decode_with(
        &data,
        JxlDecoderOptions {
            adjust_orientation: false,
            ..Default::default()
        },
    )
    .expect("decode (Preserve) failed");

    // Sanity: same channel layout / grayscale-ness, just transposed geometry.
    assert_eq!(baked.channels, unbaked.channels);
    assert_eq!(baked.is_grayscale, unbaked.is_grayscale);

    let orientation = unbaked.info.intrinsic_orientation;
    assert_ne!(
        orientation,
        Orientation::Identity,
        "this test only proves anything for a non-Identity orientation"
    );

    let coded = (unbaked.width, unbaked.height);
    let display = (baked.width, baked.height);
    let ch = baked.channels;

    // Geometry must be the orientation's coded->display size mapping.
    assert_eq!(
        display,
        Orientation::Transpose.map_size(coded),
        "display geometry must be the transposed coded geometry"
    );

    // Walk every destination (display/baked) pixel, find its source (coded/
    // unbaked) pixel, and compare every channel byte exactly.
    let baked_row = display.0 * ch;
    let coded_row = coded.0 * ch;
    let mut mismatches = 0usize;
    for dy in 0..display.1 {
        for dx in 0..display.0 {
            let (sx, sy) = coded_pixel_for_display(orientation, (dx, dy), coded);
            let b = &baked.data[dy * baked_row + dx * ch..][..ch];
            let u = &unbaked.data[sy * coded_row + sx * ch..][..ch];
            if b != u {
                mismatches += 1;
                if mismatches <= 4 {
                    eprintln!(
                        "mismatch at display ({dx},{dy}) <- coded ({sx},{sy}): \
                         baked {b:?} vs unbaked {u:?}"
                    );
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "baked output must equal un-baked output mapped through the stored \
         orientation, bit-for-bit ({mismatches} mismatched pixels)"
    );
}
