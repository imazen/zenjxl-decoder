// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression assertions from jxl-rs f370be423933628eb7f783f1b61d4c765eca5f79.

use crate::api::decode_with;
use crate::error::Error;

fn input(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/testdata")
            .join(name),
    )
    .unwrap()
}

/// Upstream 08bb971: adding the coded u32::MAX count must not overflow usize
/// on i686 before rejecting it. A native 64-bit run is only the control.
#[test]
fn vardct_num_nonzeros_overflow() {
    let result = decode_with(
        &input("vardct_num_nonzeros_overflow.jxl"),
        Default::default(),
    );
    assert!(
        matches!(&result, Err(e) if matches!(e.error(), Error::InvalidNumNonZeros(..))),
        "expected InvalidNumNonZeros, got {:?}",
        result.as_ref().err()
    );
}

/// Preserve upstream's chroma-separation assertions at the LF-group boundary.
#[test]
fn subsampled_chroma() {
    use crate::{api::decoder::tests::decode, image::Rect};
    let data = crate::util::test::fixture_bytes("multiple_lf_420.jxl");
    let (_, mut frames) = decode(&data, usize::MAX, false, false, None).unwrap();
    let frame = frames.pop().unwrap();
    let [image]: [_; 1] = frame.try_into().unwrap();
    for rect in [
        Rect {
            origin: (2048 * 3, 0),
            size: (16 * 3, 16),
        },
        Rect {
            origin: (0, 2048),
            size: (16 * 3, 16),
        },
    ] {
        let view = image.get_rect(rect);
        for y in 0..view.size().1 {
            for pixel in view.row(y).chunks_exact(3) {
                let [r, g, b]: [f32; 3] = pixel.try_into().unwrap();
                assert!(r.max(g).max(b) - r.min(g).min(b) > 0.5);
            }
        }
    }
}
