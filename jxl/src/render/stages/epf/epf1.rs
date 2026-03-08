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

/// 3x3 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes this filter a 5x5 filter.
pub struct Epf1Stage {
    /// Multiplier for sigma in pass 1
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: SigmaSource,
}

impl std::fmt::Display for Epf1Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EPF stage 1 with sigma scale: {}, border_sad_mul: {}",
            self.sigma_scale, self.border_sad_mul
        )
    }
}

impl Epf1Stage {
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
#[allow(clippy::too_many_arguments)]
fn epf1_process(
    _token: SimdToken,
    rows: &[&[f32]],
    rpc: usize,
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    xpos: usize,
    xsize: usize,
    row_sigma: &[f32],
    sigma_constant: f32,
    sigma_is_constant: bool,
    sad_mul_storage: &[f32; 24],
    channel_scale: [f32; 3],
) {
    for x in 0..xsize {
        let abs_x = x + xpos;
        let sigma = if sigma_is_constant {
            sigma_constant
        } else {
            row_sigma[abs_x / BLOCK_DIM]
        };

        if sigma < MIN_SIGMA {
            out0[x] = rows[2][2 + x];
            out1[x] = rows[rpc + 2][2 + x];
            out2[x] = rows[2 * rpc + 2][2 + x];
            continue;
        }

        let sad_mul = sad_mul_storage[x % 8];
        let inv_sigma = sigma * sad_mul;

        // Compute SADs across all 3 channels
        let mut sads = [0.0f32; 4];
        for c in 0..3 {
            let b = c * rpc;
            let scale = channel_scale[c];
            let p20 = rows[b][2 + x];
            let p11 = rows[b + 1][1 + x];
            let p21 = rows[b + 1][2 + x];
            let p31 = rows[b + 1][3 + x];
            let p02 = rows[b + 2][x];
            let p12 = rows[b + 2][1 + x];
            let p22 = rows[b + 2][2 + x];
            let p32 = rows[b + 2][3 + x];
            let p42 = rows[b + 2][4 + x];
            let p13 = rows[b + 3][1 + x];
            let p23 = rows[b + 3][2 + x];
            let p33 = rows[b + 3][3 + x];
            let p24 = rows[b + 4][2 + x];

            let d20_21 = (p20 - p21).abs();
            let d11_21 = (p11 - p21).abs();
            let d22_21 = (p22 - p21).abs();
            let d31_21 = (p31 - p21).abs();
            let d02_12 = (p02 - p12).abs();
            let d11_12 = (p11 - p12).abs();
            let d12_22 = (p22 - p12).abs();
            let d31_32 = (p31 - p32).abs();
            let d22_32 = (p22 - p32).abs();
            let d42_32 = (p42 - p32).abs();
            let d13_12 = (p13 - p12).abs();
            let d22_23 = (p22 - p23).abs();
            let d13_23 = (p13 - p23).abs();
            let d33_23 = (p33 - p23).abs();
            let d33_32 = (p33 - p32).abs();
            let d24_23 = (p24 - p23).abs();

            sads[0] += (d20_21 + d11_12 + d22_21 + d31_32 + d22_23) * scale;
            sads[1] += (d11_21 + d02_12 + d12_22 + d22_32 + d13_23) * scale;
            sads[2] += (d31_21 + d12_22 + d22_32 + d42_32 + d33_23) * scale;
            sads[3] += (d22_21 + d13_12 + d22_23 + d33_32 + d24_23) * scale;
        }

        // Compute weights
        let mut w = 1.0f32;
        for sad in &mut sads {
            *sad = sad.mul_add(inv_sigma, 1.0).max(0.0);
            w += *sad;
        }
        let inv_w = 1.0 / w;

        // Compute output for each channel
        // Channel 0
        {
            let b = 0;
            let mut out = rows[b + 2][2 + x];
            out = rows[b + 3][2 + x].mul_add(sads[3], out);
            out = rows[b + 2][3 + x].mul_add(sads[2], out);
            out = rows[b + 2][1 + x].mul_add(sads[1], out);
            out = rows[b + 1][2 + x].mul_add(sads[0], out);
            out0[x] = out * inv_w;
        }
        // Channel 1
        {
            let b = rpc;
            let mut out = rows[b + 2][2 + x];
            out = rows[b + 3][2 + x].mul_add(sads[3], out);
            out = rows[b + 2][3 + x].mul_add(sads[2], out);
            out = rows[b + 2][1 + x].mul_add(sads[1], out);
            out = rows[b + 1][2 + x].mul_add(sads[0], out);
            out1[x] = out * inv_w;
        }
        // Channel 2
        {
            let b = 2 * rpc;
            let mut out = rows[b + 2][2 + x];
            out = rows[b + 3][2 + x].mul_add(sads[3], out);
            out = rows[b + 2][3 + x].mul_add(sads[2], out);
            out = rows[b + 2][1 + x].mul_add(sads[1], out);
            out = rows[b + 1][2 + x].mul_add(sads[0], out);
            out2[x] = out * inv_w;
        }
    }
}

impl RenderPipelineInOutStage for Epf1Stage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (2, 2);

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

        let sm = self.sigma_scale * 1.65;
        let bsm = sm * self.border_sad_mul;
        let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

        let rows = &input_rows.row_data;
        let rpc = input_rows.rows_per_channel;
        let (out0, out1, out2) = output_rows.split_first_3_mut();

        epf1_process(
            rows, rpc,
            out0[0], out1[0], out2[0],
            xpos, xsize,
            row_sigma, sigma_constant, sigma_is_constant,
            &sad_mul_storage,
            self.channel_scale,
        );
    }
}
