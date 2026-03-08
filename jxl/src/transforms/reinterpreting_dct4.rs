// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

/// Apply 4-point reinterpreting DCT to elements at `data[col + k*stride]` for k=0..4.
#[inline(always)]
pub(super) fn apply_reinterpreting_dct_4(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    let v2 = data[col + 2 * stride];
    let v3 = data[col + 3 * stride];

    let v4 = v0 + v3;
    let v5 = v1 + v2;
    let v6 = v4 + v5;
    let v7 = v4 - v5;
    let v8 = v0 - v3;
    let v9 = v1 - v2;
    let v10 = v8 * 0.5411961001461970;
    let v11 = v9 * 1.3065629648763764;
    let v12 = v10 + v11;
    let v13 = v10 - v11;
    let v14 = v12.mul_add(std::f32::consts::SQRT_2, v13);

    data[col] = v6 * 0.250000;
    data[col + stride] = v14 * 0.256440;
    data[col + 2 * stride] = v7 * 0.277234;
    data[col + 3 * stride] = v13 * 0.317640;
}
