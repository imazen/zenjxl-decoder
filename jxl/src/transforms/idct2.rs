// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Apply 2-point IDCT to elements at `data[col]` and `data[col + stride]`.
#[inline(always)]
pub(super) fn apply_idct_2(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    data[col] = v0 + v1;
    data[col + stride] = v0 - v1;
}
