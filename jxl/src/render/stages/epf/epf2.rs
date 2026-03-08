// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use archmage::prelude::*;

use crate::{
    BLOCK_DIM, MIN_SIGMA,
    features::epf::SigmaSource,
    render::{
        Channels, ChannelsMut, RenderPipelineInOutStage,
        stages::epf::common::{get_sigma_scalar, prepare_sad_mul_storage},
    },
};

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

#[autoversion]
fn epf2_process(
    _token: SimdToken,
    xpos: usize,
    ypos: usize,
    xsize: usize,
    sigma_scale: f32,
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    row_sigma: &[f32],
    sigma_constant: f32,
    sigma_is_constant: bool,
    // Input: 3 channels × 3 rows each (BORDER=1), center row is index 1
    ix0: &[f32], ix1: &[f32], ix2: &[f32],
    iy0: &[f32], iy1: &[f32], iy2: &[f32],
    ib0: &[f32], ib1: &[f32], ib2: &[f32],
    // Output: 3 channels × 1 row each
    ox: &mut [f32],
    oy: &mut [f32],
    ob: &mut [f32],
) {
    let sm = sigma_scale * 1.65;
    let bsm = sm * border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

    for x in 0..xsize {
        let abs_x = x + xpos;
        let sigma = if sigma_is_constant {
            sigma_constant
        } else {
            row_sigma[abs_x / BLOCK_DIM]
        };

        // Fast path: skip filtering if sigma is too small
        if sigma < MIN_SIGMA {
            ox[x] = ix1[1 + x];
            oy[x] = iy1[1 + x];
            ob[x] = ib1[1 + x];
            continue;
        }

        let sad_mul = sad_mul_storage[x % 8];
        let inv_sigma = sigma * sad_mul;

        let x_cc = ix1[1 + x];
        let y_cc = iy1[1 + x];
        let b_cc = ib1[1 + x];

        let mut w_acc = 1.0f32;
        let mut x_acc = x_cc;
        let mut y_acc = y_cc;
        let mut b_acc = b_cc;

        // Plus-shaped 4-neighbor kernel: (row, col) offsets into input
        for (y_off, x_off) in [(0usize, 1usize), (1, 0), (1, 2), (2, 1)] {
            let cx = [ix0, ix1, ix2][y_off][x_off + x];
            let cy = [iy0, iy1, iy2][y_off][x_off + x];
            let cb = [ib0, ib1, ib2][y_off][x_off + x];

            let sad = (cx - x_cc).abs().mul_add(
                channel_scale[0],
                (cy - y_cc).abs().mul_add(
                    channel_scale[1],
                    (cb - b_cc).abs() * channel_scale[2],
                ),
            );
            let weight = sad.mul_add(inv_sigma, 1.0).max(0.0);
            w_acc += weight;
            x_acc = weight.mul_add(cx, x_acc);
            y_acc = weight.mul_add(cy, y_acc);
            b_acc = weight.mul_add(cb, b_acc);
        }

        let inv_w = 1.0 / w_acc;
        ox[x] = x_acc * inv_w;
        oy[x] = y_acc * inv_w;
        ob[x] = b_acc * inv_w;
    }
}

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
        let row_sigma_data = self.sigma.row(ypos / BLOCK_DIM);
        let (row_sigma, sigma_constant, sigma_is_constant) = match row_sigma_data {
            crate::render::stages::epf::common::SigmaRow::Constant(c) => (&[] as &[f32], c, true),
            crate::render::stages::epf::common::SigmaRow::Variable(row) => (row, 0.0, false),
        };

        let (input_x, input_y, input_b) = (&input_rows[0], &input_rows[1], &input_rows[2]);
        let (output_x, output_y, output_b) = output_rows.split_first_3_mut();

        epf2_process(
            xpos, ypos, xsize,
            self.sigma_scale, self.border_sad_mul, self.channel_scale,
            row_sigma, sigma_constant, sigma_is_constant,
            input_x[0], input_x[1], input_x[2],
            input_y[0], input_y[1], input_y[2],
            input_b[0], input_b[1], input_b[2],
            output_x[0], output_y[0], output_b[0],
        );
    }
}
