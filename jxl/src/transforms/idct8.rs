// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Apply 8-point IDCT to elements at `data[col + k*stride]` for k=0..8.
#[inline(always)]
pub(super) fn apply_idct_8(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    let v2 = data[col + 2 * stride];
    let v3 = data[col + 3 * stride];
    let v4 = data[col + 4 * stride];
    let v5 = data[col + 5 * stride];
    let v6 = data[col + 6 * stride];
    let v7 = data[col + 7 * stride];

    let v8 = v0 + v4;
    let v9 = v0 - v4;
    let v10 = v2 + v6;
    let v11 = v2 * std::f32::consts::SQRT_2;
    let v12 = v11 + v10;
    let v13 = v11 - v10;
    let v14 = v12.mul_add(0.5411961001461970, v8);
    let v15 = (-v12).mul_add(0.5411961001461970, v8);
    let v16 = v13.mul_add(1.3065629648763764, v9);
    let v17 = (-v13).mul_add(1.3065629648763764, v9);

    let v18 = v1 + v3;
    let v19 = v3 + v5;
    let v20 = v5 + v7;
    let v21 = v1 * std::f32::consts::SQRT_2;
    let v22 = v21 + v19;
    let v23 = v21 - v19;
    let v24 = v18 + v20;
    let v25 = v18 * std::f32::consts::SQRT_2;
    let v26 = v25 + v24;
    let v27 = v25 - v24;
    let v28 = v26.mul_add(0.5411961001461970, v22);
    let v29 = (-v26).mul_add(0.5411961001461970, v22);
    let v30 = v27.mul_add(1.3065629648763764, v23);
    let v31 = (-v27).mul_add(1.3065629648763764, v23);

    let r0 = v28.mul_add(0.5097955791041592, v14);
    let r7 = (-v28).mul_add(0.5097955791041592, v14);
    let r1 = v30.mul_add(0.6013448869350453, v16);
    let r6 = (-v30).mul_add(0.6013448869350453, v16);
    let r2 = v31.mul_add(0.8999762231364156, v17);
    let r5 = (-v31).mul_add(0.8999762231364156, v17);
    let r3 = v29.mul_add(2.5629154477415055, v15);
    let r4 = (-v29).mul_add(2.5629154477415055, v15);

    data[col] = r0;
    data[col + stride] = r1;
    data[col + 2 * stride] = r2;
    data[col + 3 * stride] = r3;
    data[col + 4 * stride] = r4;
    data[col + 5 * stride] = r5;
    data[col + 6 * stride] = r6;
    data[col + 7 * stride] = r7;
}
