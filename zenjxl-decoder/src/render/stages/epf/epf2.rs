// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    BLOCK_DIM, MIN_SIGMA,
    features::epf::SigmaSource,
    render::{
        Channels, ChannelsMut, RenderPipelineInOutStage,
        stages::epf::common::{get_sigma, prepare_sad_mul_storage},
    },
};

use jxl_simd::{F32SimdVec, SimdMask, simd_function};

/// 3x3 plus-shaped kernel with 1 SAD per pixel. So this makes this filter a 3x3 filter.
pub struct Epf2Stage {
    /// Multiplier for sigma in pass 2
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: SigmaSource,
}

impl std::fmt::Display for Epf2Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EPF stage 2 with sigma scale: {}, border_sad_mul: {}",
            self.sigma_scale, self.border_sad_mul
        )
    }
}

impl Epf2Stage {
    pub fn new(
        sigma_scale: f32,
        border_sad_mul: f32,
        channel_scale: [f32; 3],
        sigma: SigmaSource,
    ) -> Self {
        Self {
            sigma,
            sigma_scale,
            channel_scale,
            border_sad_mul,
        }
    }
}

simd_function!(
epf2_process_row_chunk_dispatch,
d: D,
fn epf2_process_row_chunk(
    stage: &Epf2Stage,
    pos: (usize, usize),
    xsize: usize,
    input_rows: &Channels<f32>,
    output_rows: &mut ChannelsMut<f32>,
) {
    let (xpos, ypos) = pos;
    assert_eq!(input_rows.len(), 3, "Expected 3 channels, got {}", input_rows.len());
    let (input_x, input_y, input_b) = (&input_rows[0], &input_rows[1], &input_rows[2]);
    let (output_x, output_y, output_b) = output_rows.split_first_3_mut();

    // EPF2 has BORDER=1: 3 input rows per channel, read at columns
    // x..x + LEN + 2. All 9 rows are trimmed to one common length so that a
    // single bounds check per vector covers them, and each vector reads
    // through `LEN + 2`-sample windows at constant offsets.
    let min_in_len = 2 + xsize;
    let min_out_len = xsize;
    for ch in [input_x, input_y, input_b] {
        assert!(ch.len() >= 3);
    }
    let rows: [[&[f32]; 3]; 3] = [input_x, input_y, input_b].map(|ch| [ch[0], ch[1], ch[2]]);
    let common_len = rows.iter().flatten().map(|r| r.len()).min().unwrap();
    assert!(common_len >= min_in_len);
    let rows = rows.map(|ch| ch.map(|r| &r[..common_len]));
    assert!(output_x[0].len() >= min_out_len);
    assert!(output_y[0].len() >= min_out_len);
    assert!(output_b[0].len() >= min_out_len);

    let row_sigma = stage.sigma.row(ypos / BLOCK_DIM);

    const { assert!(D::F32Vec::LEN <= 16) };

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);
    let scale = stage.channel_scale.map(|s| D::F32Vec::splat(d, s));
    let len = D::F32Vec::LEN;

    for x in (0..xsize).step_by(len) {
        let sigma = get_sigma(d, x + xpos, row_sigma);
        let sad_mul = D::F32Vec::load_from(d, &sad_mul_storage, x % 8);
        let [wx, wy, wb] = rows.map(|ch| ch.map(|r| &r[x..x + len + 2]));

        let x_cc = D::F32Vec::load_from(d, wx[1], 1);
        let y_cc = D::F32Vec::load_from(d, wy[1], 1);
        let b_cc = D::F32Vec::load_from(d, wb[1], 1);

        let sigma_mask = D::F32Vec::splat(d, MIN_SIGMA).gt(sigma);
        if sigma_mask.all() {
            x_cc.store_at(output_x[0], x);
            y_cc.store_at(output_y[0], x);
            b_cc.store_at(output_b[0], x);
            continue;
        }

        let inv_sigma = sigma * sad_mul;

        let mut w_acc = D::F32Vec::splat(d, 1.0);
        let mut x_acc = x_cc;
        let mut y_acc = y_cc;
        let mut b_acc = b_cc;

        for (y_off, x_off) in [(0, 1), (1, 0), (1, 2), (2, 1)] {
            let (cx, cy, cb) = (
                D::F32Vec::load_from(d, wx[y_off], x_off),
                D::F32Vec::load_from(d, wy[y_off], x_off),
                D::F32Vec::load_from(d, wb[y_off], x_off),
            );
            let sad = (cx - x_cc).abs().mul_add(
                scale[0],
                (cy - y_cc).abs().mul_add(scale[1], (cb - b_cc).abs() * scale[2]),
            );
            let weight = sad
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            w_acc += weight;
            x_acc = weight.mul_add(cx, x_acc);
            y_acc = weight.mul_add(cy, y_acc);
            b_acc = weight.mul_add(cb, b_acc);
        }

        let inv_w = D::F32Vec::splat(d, 1.0) / w_acc;

        x_acc *= inv_w;
        y_acc *= inv_w;
        b_acc *= inv_w;
        x_acc = sigma_mask.if_then_else_f32(x_cc, x_acc);
        y_acc = sigma_mask.if_then_else_f32(y_cc, y_acc);
        b_acc = sigma_mask.if_then_else_f32(b_cc, b_acc);
        x_acc.store_at(output_x[0], x);
        y_acc.store_at(output_y[0], x);
        b_acc.store_at(output_b[0], x);
    }
});

impl RenderPipelineInOutStage for Epf2Stage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (1, 1);

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut (dyn std::any::Any + Send)>,
    ) {
        epf2_process_row_chunk_dispatch(self, (xpos, ypos), xsize, input_rows, output_rows);
    }
}
