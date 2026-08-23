// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Render-pipeline group-border regressions (jxl-rs #845 / #873).
//!
//! `edge515x72_center_right_d2.jxl` is 515x72, VarDCT with EPF (border > 0),
//! encoded by `cjxl -d 2 -e 5 --group_order=1 --center_x=514 --center_y=0`:
//! the TOC is permuted so the **last** group column (3 px wide, narrower than
//! the render border) arrives before the second-to-last one. When the
//! second-to-last group was rendered with its right neighbour already ready,
//! its ready rectangle extended `border` pixels past the image edge and the
//! edge padding was not applied (it was keyed on the group index, not on the
//! rectangle touching the image edge), so the last `border` columns of those
//! rows were computed from stale data. A one-shot decode sorts groups by
//! index and never hit it; a streamed decode did.

use crate::api::decoder::tests::decode;

fn testdata(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn max_abs_diff(a: &[crate::image::Image<f32>], b: &[crate::image::Image<f32>]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut count = 0usize;
    for (ia, ib) in a.iter().zip(b) {
        assert_eq!(ia.size(), ib.size());
        for y in 0..ia.size().1 {
            for (&va, &vb) in ia.row(y).iter().zip(ib.row(y)) {
                let d = (va - vb).abs();
                if d > 0.0 {
                    count += 1;
                }
                max = max.max(d);
            }
        }
    }
    (max, count)
}

/// Streamed (permuted-TOC, right-edge-first) decode must equal the one-shot
/// decode, for every chunk size that lets the last column land first.
#[test]
fn permuted_toc_tiny_last_column_streamed_equals_oneshot() {
    let data = testdata("jxlrs-845/edge515x72_center_right_d2.jxl");
    let (_, oneshot) = decode(&data, usize::MAX, false, false, None).unwrap();
    assert_eq!(oneshot.len(), 1);
    for chunk in [1usize, 7, 64, 123, 512, 1024] {
        let (_, streamed) = decode(&data, chunk, false, false, None).unwrap();
        let (max, count) = max_abs_diff(&oneshot[0], &streamed[0]);
        assert_eq!(
            (max, count),
            (0.0, 0),
            "chunk {chunk}: streamed decode differs from one-shot in {count} samples (max {max})"
        );
    }
}

/// Same, with a flush after every chunk (the CLI's --allow-partial-files path).
#[test]
fn permuted_toc_tiny_last_column_streamed_with_flush_equals_oneshot() {
    let data = testdata("jxlrs-845/edge515x72_center_right_d2.jxl");
    let (_, oneshot) = decode(&data, usize::MAX, false, false, None).unwrap();
    for chunk in [64usize, 123, 1024] {
        let (_, streamed) = decode(&data, chunk, false, true, None).unwrap();
        let (max, count) = max_abs_diff(&oneshot[0], &streamed[0]);
        assert_eq!(
            (max, count),
            (0.0, 0),
            "chunk {chunk} + flush: differs in {count} samples (max {max})"
        );
    }
}

/// The simple (non-low-memory) pipeline is the reference for the low-memory
/// one on this file too.
#[test]
fn permuted_toc_tiny_last_column_pipelines_agree() {
    let data = testdata("jxlrs-845/edge515x72_center_right_d2.jxl");
    let (_, low_memory) = decode(&data, 123, false, false, None).unwrap();
    let (_, simple) = decode(&data, 123, true, false, None).unwrap();
    let (max, count) = max_abs_diff(&low_memory[0], &simple[0]);
    assert_eq!(
        (max, count),
        (0.0, 0),
        "pipelines differ in {count} samples (max {max})"
    );
}
