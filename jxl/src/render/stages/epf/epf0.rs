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
        stages::epf::common::prepare_sad_mul_storage,
    },
};

/// 5x5 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes this filter a 7x7 filter.
pub struct Epf0Stage {
    /// Multiplier for sigma in pass 0
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: SigmaSource,
}

impl std::fmt::Display for Epf0Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EPF stage 0 with sigma scale: {}, border_sad_mul: {}",
            self.sigma_scale, self.border_sad_mul
        )
    }
}

impl Epf0Stage {
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
fn epf0_process(
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
            out0[x] = rows[3][3 + x];
            out1[x] = rows[rpc + 3][3 + x];
            out2[x] = rows[2 * rpc + 3][3 + x];
            continue;
        }

        let sad_mul = sad_mul_storage[x % 8];
        let inv_sigma = sigma * sad_mul;

        // Compute SADs across all 3 channels
        let mut sads = [0.0f32; 12];
        for c in 0..3 {
            let b = c * rpc;
            let scale = channel_scale[c];

            // Load 26 pixels from 7 rows (BORDER=3, center at row 3, col 3+x)
            let p30 = rows[b][3 + x];
            let p21 = rows[b + 1][2 + x];
            let p31 = rows[b + 1][3 + x];
            let p41 = rows[b + 1][4 + x];
            let p12 = rows[b + 2][1 + x];
            let p22 = rows[b + 2][2 + x];
            let p32 = rows[b + 2][3 + x];
            let p42 = rows[b + 2][4 + x];
            let p52 = rows[b + 2][5 + x];
            let p03 = rows[b + 3][x];
            let p13 = rows[b + 3][1 + x];
            let p23 = rows[b + 3][2 + x];
            let p33 = rows[b + 3][3 + x];
            let p43 = rows[b + 3][4 + x];
            let p53 = rows[b + 3][5 + x];
            let p63 = rows[b + 3][6 + x];
            let p14 = rows[b + 4][1 + x];
            let p24 = rows[b + 4][2 + x];
            let p34 = rows[b + 4][3 + x];
            let p44 = rows[b + 4][4 + x];
            let p54 = rows[b + 4][5 + x];
            let p25 = rows[b + 5][2 + x];
            let p35 = rows[b + 5][3 + x];
            let p45 = rows[b + 5][4 + x];
            let p36 = rows[b + 6][3 + x];

            // Compute absolute differences
            let d32_30 = (p32 - p30).abs();
            let d32_21 = (p32 - p21).abs();
            let d32_31 = (p32 - p31).abs();
            let d32_41 = (p32 - p41).abs();
            let d32_12 = (p32 - p12).abs();
            let d32_22 = (p32 - p22).abs();
            let d32_42 = (p32 - p42).abs();
            let d32_52 = (p32 - p52).abs();
            let d32_23 = (p32 - p23).abs();
            let d32_34 = (p32 - p34).abs();
            let d32_43 = (p32 - p43).abs();
            let d32_33 = (p32 - p33).abs();
            let d23_21 = (p23 - p21).abs();
            let d23_12 = (p23 - p12).abs();
            let d23_22 = (p23 - p22).abs();
            let d23_03 = (p23 - p03).abs();
            let d23_13 = (p23 - p13).abs();
            let d23_33 = (p23 - p33).abs();
            let d23_43 = (p23 - p43).abs();
            let d23_14 = (p23 - p14).abs();
            let d23_24 = (p23 - p24).abs();
            let d23_34 = (p23 - p34).abs();
            let d23_25 = (p23 - p25).abs();
            let d33_31 = (p33 - p31).abs();
            let d33_22 = (p33 - p22).abs();
            let d33_42 = (p33 - p42).abs();
            let d33_13 = (p33 - p13).abs();
            let d33_43 = (p33 - p43).abs();
            let d33_53 = (p33 - p53).abs();
            let d33_24 = (p33 - p24).abs();
            let d33_34 = (p33 - p34).abs();
            let d33_44 = (p33 - p44).abs();
            let d33_35 = (p33 - p35).abs();
            let d43_41 = (p43 - p41).abs();
            let d43_42 = (p43 - p42).abs();
            let d43_52 = (p43 - p52).abs();
            let d43_53 = (p43 - p53).abs();
            let d43_63 = (p43 - p63).abs();
            let d43_34 = (p43 - p34).abs();
            let d43_44 = (p43 - p44).abs();
            let d43_54 = (p43 - p54).abs();
            let d43_45 = (p43 - p45).abs();
            let d34_14 = (p34 - p14).abs();
            let d34_24 = (p34 - p24).abs();
            let d34_44 = (p34 - p44).abs();
            let d34_54 = (p34 - p54).abs();
            let d34_25 = (p34 - p25).abs();
            let d34_35 = (p34 - p35).abs();
            let d34_45 = (p34 - p45).abs();
            let d34_36 = (p34 - p36).abs();

            // Accumulate 12 SADs
            sads[0] += (d32_30 + d23_21 + d33_31 + d43_41 + d32_34) * scale;
            sads[1] += (d32_21 + d23_12 + d33_22 + d32_43 + d23_34) * scale;
            sads[2] += (d32_31 + d23_22 + d32_33 + d43_42 + d33_34) * scale;
            sads[3] += (d32_41 + d32_23 + d33_42 + d43_52 + d43_34) * scale;
            sads[4] += (d32_12 + d23_03 + d33_13 + d23_43 + d34_14) * scale;
            sads[5] += (d32_22 + d23_13 + d23_33 + d33_43 + d34_24) * scale;
            sads[6] += (d32_42 + d23_33 + d33_43 + d43_53 + d34_44) * scale;
            sads[7] += (d32_52 + d23_43 + d33_53 + d43_63 + d34_54) * scale;
            sads[8] += (d32_23 + d23_14 + d33_24 + d43_34 + d34_25) * scale;
            sads[9] += (d32_33 + d23_24 + d33_34 + d43_44 + d34_35) * scale;
            sads[10] += (d32_43 + d23_34 + d33_44 + d43_54 + d34_45) * scale;
            sads[11] += (d32_34 + d23_25 + d33_35 + d43_45 + d34_36) * scale;
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
            let mut out = rows[b + 3][3 + x];
            out = rows[b + 5][3 + x].mul_add(sads[11], out);
            out = rows[b + 4][4 + x].mul_add(sads[10], out);
            out = rows[b + 4][3 + x].mul_add(sads[9], out);
            out = rows[b + 4][2 + x].mul_add(sads[8], out);
            out = rows[b + 3][5 + x].mul_add(sads[7], out);
            out = rows[b + 3][4 + x].mul_add(sads[6], out);
            out = rows[b + 3][2 + x].mul_add(sads[5], out);
            out = rows[b + 3][1 + x].mul_add(sads[4], out);
            out = rows[b + 2][4 + x].mul_add(sads[3], out);
            out = rows[b + 2][3 + x].mul_add(sads[2], out);
            out = rows[b + 2][2 + x].mul_add(sads[1], out);
            out = rows[b + 1][3 + x].mul_add(sads[0], out);
            out0[x] = out * inv_w;
        }
        // Channel 1
        {
            let b = rpc;
            let mut out = rows[b + 3][3 + x];
            out = rows[b + 5][3 + x].mul_add(sads[11], out);
            out = rows[b + 4][4 + x].mul_add(sads[10], out);
            out = rows[b + 4][3 + x].mul_add(sads[9], out);
            out = rows[b + 4][2 + x].mul_add(sads[8], out);
            out = rows[b + 3][5 + x].mul_add(sads[7], out);
            out = rows[b + 3][4 + x].mul_add(sads[6], out);
            out = rows[b + 3][2 + x].mul_add(sads[5], out);
            out = rows[b + 3][1 + x].mul_add(sads[4], out);
            out = rows[b + 2][4 + x].mul_add(sads[3], out);
            out = rows[b + 2][3 + x].mul_add(sads[2], out);
            out = rows[b + 2][2 + x].mul_add(sads[1], out);
            out = rows[b + 1][3 + x].mul_add(sads[0], out);
            out1[x] = out * inv_w;
        }
        // Channel 2
        {
            let b = 2 * rpc;
            let mut out = rows[b + 3][3 + x];
            out = rows[b + 5][3 + x].mul_add(sads[11], out);
            out = rows[b + 4][4 + x].mul_add(sads[10], out);
            out = rows[b + 4][3 + x].mul_add(sads[9], out);
            out = rows[b + 4][2 + x].mul_add(sads[8], out);
            out = rows[b + 3][5 + x].mul_add(sads[7], out);
            out = rows[b + 3][4 + x].mul_add(sads[6], out);
            out = rows[b + 3][2 + x].mul_add(sads[5], out);
            out = rows[b + 3][1 + x].mul_add(sads[4], out);
            out = rows[b + 2][4 + x].mul_add(sads[3], out);
            out = rows[b + 2][3 + x].mul_add(sads[2], out);
            out = rows[b + 2][2 + x].mul_add(sads[1], out);
            out = rows[b + 1][3 + x].mul_add(sads[0], out);
            out2[x] = out * inv_w;
        }
    }
}

impl RenderPipelineInOutStage for Epf0Stage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (3, 3);

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

        epf0_process(
            rows, rpc,
            out0[0], out1[0], out2[0],
            xpos, xsize,
            row_sigma, sigma_constant, sigma_is_constant,
            &sad_mul_storage,
            self.channel_scale,
        );
    }
}
