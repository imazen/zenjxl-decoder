// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use archmage::prelude::*;

use crate::render::{Channels, ChannelsMut, RenderPipelineInOutStage};

/// Apply Gabor-like filter to a channel.
#[derive(Debug)]
pub struct GaborishStage {
    channel: usize,
    weight0: f32,
    weight1: f32,
    weight2: f32,
}

impl GaborishStage {
    pub fn new(channel: usize, weight1: f32, weight2: f32) -> Self {
        let weight_total = 1.0 + weight1 * 4.0 + weight2 * 4.0;
        Self {
            channel,
            weight0: 1.0 / weight_total,
            weight1: weight1 / weight_total,
            weight2: weight2 / weight_total,
        }
    }
}

impl std::fmt::Display for GaborishStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gaborish filter for channel {}", self.channel)
    }
}

#[autoversion]
fn gaborish_process(
    _token: SimdToken,
    w0: f32,
    w1: f32,
    w2: f32,
    xsize: usize,
    row_top: &[f32],
    row_center: &[f32],
    row_bottom: &[f32],
    row_out: &mut [f32],
) {
    for x in 0..xsize {
        let p00 = row_top[x];
        let p01 = row_top[x + 1];
        let p02 = row_top[x + 2];
        let p10 = row_center[x];
        let p11 = row_center[x + 1];
        let p12 = row_center[x + 2];
        let p20 = row_bottom[x];
        let p21 = row_bottom[x + 1];
        let p22 = row_bottom[x + 2];

        let sum = p11 * w0
            + (p01 + p10 + p21 + p12) * w1
            + (p00 + p02 + p20 + p22) * w2;
        row_out[x] = sum;
    }
}

impl RenderPipelineInOutStage for GaborishStage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (1, 1);

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut (dyn std::any::Any + Send)>,
    ) {
        let row_out = &mut output_rows[0][0];
        let [row_top, row_center, row_bottom] = input_rows[0] else {
            unreachable!();
        };
        gaborish_process(
            self.weight0,
            self.weight1,
            self.weight2,
            xsize,
            row_top,
            row_center,
            row_bottom,
            row_out,
        );
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_abs_eq;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || GaborishStage::new(0, 0.115169525, 0.061248592),
            (500, 500),
            1,
        )
    }

    #[test]
    fn checkerboard() -> Result<()> {
        let mut image = Image::new((2, 2))?;
        image.row_mut(0).copy_from_slice(&[0.0, 1.0]);
        image.row_mut(1).copy_from_slice(&[1.0, 0.0]);

        let stage = GaborishStage::new(0, 0.115169525, 0.061248592);
        let output = make_and_run_simple_pipeline(stage, &[image], (2, 2), 0, 256)?;

        assert_all_almost_abs_eq(output[0].row(0), &[0.20686048, 0.7931395], 1e-6);
        assert_all_almost_abs_eq(output[0].row(1), &[0.7931395, 0.20686048], 1e-6);

        Ok(())
    }
}
