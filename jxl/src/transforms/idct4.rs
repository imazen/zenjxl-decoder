// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Apply 4-point IDCT to elements at `data[col], data[col+s], data[col+2s], data[col+3s]`.
#[inline(always)]
pub(super) fn apply_idct_4(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    let v2 = data[col + 2 * stride];
    let v3 = data[col + 3 * stride];

    let v4 = v0 + v2;
    let v5 = v0 - v2;
    let v6 = v1 + v3;
    let v7 = v1 * std::f32::consts::SQRT_2;
    let v8 = v7 + v6;
    let v9 = v7 - v6;

    let r0 = v8.mul_add(0.5411961001461970, v4);
    let r3 = (-v8).mul_add(0.5411961001461970, v4);
    let r1 = v9.mul_add(1.3065629648763764, v5);
    let r2 = (-v9).mul_add(1.3065629648763764, v5);

    data[col] = r0;
    data[col + stride] = r1;
    data[col + 2 * stride] = r2;
    data[col + 3 * stride] = r3;
}
