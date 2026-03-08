// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

/// Apply 16-point IDCT to elements at `data[col + k*stride]` for k=0..16.
#[inline(always)]
pub(super) fn apply_idct_16(data: &mut [f32], col: usize, stride: usize) {
    let v0 = data[col];
    let v1 = data[col + stride];
    let v2 = data[col + 2 * stride];
    let v3 = data[col + 3 * stride];
    let v4 = data[col + 4 * stride];
    let v5 = data[col + 5 * stride];
    let v6 = data[col + 6 * stride];
    let v7 = data[col + 7 * stride];
    let v8 = data[col + 8 * stride];
    let v9 = data[col + 9 * stride];
    let v10 = data[col + 10 * stride];
    let v11 = data[col + 11 * stride];
    let v12 = data[col + 12 * stride];
    let v13 = data[col + 13 * stride];
    let v14 = data[col + 14 * stride];
    let v15 = data[col + 15 * stride];

    // Even-indexed sub-butterfly (8-point on indices 0,2,4,6,8,10,12,14)
    let v16 = v0 + v8;
    let v17 = v0 - v8;
    let v18 = v4 + v12;
    let v19 = v4 * std::f32::consts::SQRT_2;
    let v20 = v19 + v18;
    let v21 = v19 - v18;
    let v22 = v20.mul_add(0.5411961001461970, v16);
    let v23 = (-v20).mul_add(0.5411961001461970, v16);
    let v24 = v21.mul_add(1.3065629648763764, v17);
    let v25 = (-v21).mul_add(1.3065629648763764, v17);

    let v26 = v2 + v6;
    let v27 = v6 + v10;
    let v28 = v10 + v14;
    let v29 = v2 * std::f32::consts::SQRT_2;
    let v30 = v29 + v27;
    let v31 = v29 - v27;
    let v32 = v26 + v28;
    let v33 = v26 * std::f32::consts::SQRT_2;
    let v34 = v33 + v32;
    let v35 = v33 - v32;
    let v36 = v34.mul_add(0.5411961001461970, v30);
    let v37 = (-v34).mul_add(0.5411961001461970, v30);
    let v38 = v35.mul_add(1.3065629648763764, v31);
    let v39 = (-v35).mul_add(1.3065629648763764, v31);

    let v40 = v36.mul_add(0.5097955791041592, v22);
    let v41 = (-v36).mul_add(0.5097955791041592, v22);
    let v42 = v38.mul_add(0.6013448869350453, v24);
    let v43 = (-v38).mul_add(0.6013448869350453, v24);
    let v44 = v39.mul_add(0.8999762231364156, v25);
    let v45 = (-v39).mul_add(0.8999762231364156, v25);
    let v46 = v37.mul_add(2.5629154477415055, v23);
    let v47 = (-v37).mul_add(2.5629154477415055, v23);

    // Odd-indexed sub-butterfly (indices 1,3,5,7,9,11,13,15)
    let v48 = v1 + v3;
    let v49 = v3 + v5;
    let v50 = v5 + v7;
    let v51 = v7 + v9;
    let v52 = v9 + v11;
    let v53 = v11 + v13;
    let v54 = v13 + v15;
    let v55 = v1 * std::f32::consts::SQRT_2;
    let v56 = v55 + v51;
    let v57 = v55 - v51;
    let v58 = v49 + v53;
    let v59 = v49 * std::f32::consts::SQRT_2;
    let v60 = v59 + v58;
    let v61 = v59 - v58;
    let v62 = v60.mul_add(0.5411961001461970, v56);
    let v63 = (-v60).mul_add(0.5411961001461970, v56);
    let v64 = v61.mul_add(1.3065629648763764, v57);
    let v65 = (-v61).mul_add(1.3065629648763764, v57);

    let v66 = v48 + v50;
    let v67 = v50 + v52;
    let v68 = v52 + v54;
    let v69 = v48 * std::f32::consts::SQRT_2;
    let v70 = v69 + v67;
    let v71 = v69 - v67;
    let v72 = v66 + v68;
    let v73 = v66 * std::f32::consts::SQRT_2;
    let v74 = v73 + v72;
    let v75 = v73 - v72;
    let v76 = v74.mul_add(0.5411961001461970, v70);
    let v77 = (-v74).mul_add(0.5411961001461970, v70);
    let v78 = v75.mul_add(1.3065629648763764, v71);
    let v79 = (-v75).mul_add(1.3065629648763764, v71);

    let v80 = v76.mul_add(0.5097955791041592, v62);
    let v81 = (-v76).mul_add(0.5097955791041592, v62);
    let v82 = v78.mul_add(0.6013448869350453, v64);
    let v83 = (-v78).mul_add(0.6013448869350453, v64);
    let v84 = v79.mul_add(0.8999762231364156, v65);
    let v85 = (-v79).mul_add(0.8999762231364156, v65);
    let v86 = v77.mul_add(2.5629154477415055, v63);
    let v87 = (-v77).mul_add(2.5629154477415055, v63);

    // Combine even and odd
    let r0 = v80.mul_add(0.5024192861881557, v40);
    let r15 = (-v80).mul_add(0.5024192861881557, v40);
    let r1 = v82.mul_add(0.5224986149396889, v42);
    let r14 = (-v82).mul_add(0.5224986149396889, v42);
    let r2 = v84.mul_add(0.5669440348163577, v44);
    let r13 = (-v84).mul_add(0.5669440348163577, v44);
    let r3 = v86.mul_add(0.6468217833599901, v46);
    let r12 = (-v86).mul_add(0.6468217833599901, v46);
    let r4 = v87.mul_add(0.7881546234512502, v47);
    let r11 = (-v87).mul_add(0.7881546234512502, v47);
    let r5 = v85.mul_add(1.0606776859903471, v45);
    let r10 = (-v85).mul_add(1.0606776859903471, v45);
    let r6 = v83.mul_add(1.7224470982383342, v43);
    let r9 = (-v83).mul_add(1.7224470982383342, v43);
    let r7 = v81.mul_add(5.1011486186891553, v41);
    let r8 = (-v81).mul_add(5.1011486186891553, v41);

    data[col] = r0;
    data[col + stride] = r1;
    data[col + 2 * stride] = r2;
    data[col + 3 * stride] = r3;
    data[col + 4 * stride] = r4;
    data[col + 5 * stride] = r5;
    data[col + 6 * stride] = r6;
    data[col + 7 * stride] = r7;
    data[col + 8 * stride] = r8;
    data[col + 9 * stride] = r9;
    data[col + 10 * stride] = r10;
    data[col + 11 * stride] = r11;
    data[col + 12 * stride] = r12;
    data[col + 13 * stride] = r13;
    data[col + 14 * stride] = r14;
    data[col + 15 * stride] = r15;
}
