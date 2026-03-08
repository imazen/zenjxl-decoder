// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

/// Apply 8-point reinterpreting DCT to elements at `data[col + k*stride]` for k=0..8.
#[inline(always)]
pub(super) fn apply_reinterpreting_dct_8(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    let v2 = data[col + 2 * stride];
    let v3 = data[col + 3 * stride];
    let v4 = data[col + 4 * stride];
    let v5 = data[col + 5 * stride];
    let v6 = data[col + 6 * stride];
    let v7 = data[col + 7 * stride];

    let v8 = v0 + v7;
    let v9 = v1 + v6;
    let v10 = v2 + v5;
    let v11 = v3 + v4;
    let v12 = v8 + v11;
    let v13 = v9 + v10;
    let v14 = v12 + v13;
    let v15 = v12 - v13;
    let v16 = v8 - v11;
    let v17 = v9 - v10;
    let v18 = v16 * 0.5411961001461970;
    let v19 = v17 * 1.3065629648763764;
    let v20 = v18 + v19;
    let v21 = v18 - v19;
    let v22 = v20.mul_add(std::f32::consts::SQRT_2, v21);

    let v23 = v0 - v7;
    let v24 = v1 - v6;
    let v25 = v2 - v5;
    let v26 = v3 - v4;
    let v27 = v23 * 0.5097955791041592;
    let v28 = v24 * 0.6013448869350453;
    let v29 = v25 * 0.8999762231364156;
    let v30 = v26 * 2.5629154477415055;
    let v31 = v27 + v30;
    let v32 = v28 + v29;
    let v33 = v31 + v32;
    let v34 = v31 - v32;
    let v35 = v27 - v30;
    let v36 = v28 - v29;
    let v37 = v35 * 0.5411961001461970;
    let v38 = v36 * 1.3065629648763764;
    let v39 = v37 + v38;
    let v40 = v37 - v38;
    let v41 = v39.mul_add(std::f32::consts::SQRT_2, v40);
    let v42 = v33.mul_add(std::f32::consts::SQRT_2, v41);
    let v43 = v41 + v34;
    let v44 = v34 + v40;

    data[col] = v14 * 0.125000;
    data[col + stride] = v42 * 0.125794;
    data[col + 2 * stride] = v22 * 0.128220;
    data[col + 3 * stride] = v43 * 0.132413;
    data[col + 4 * stride] = v15 * 0.138617;
    data[col + 5 * stride] = v44 * 0.147222;
    data[col + 6 * stride] = v21 * 0.158820;
    data[col + 7 * stride] = v40 * 0.174311;
}
