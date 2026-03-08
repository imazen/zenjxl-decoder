// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use archmage::prelude::*;

use crate::render::{Channels, ChannelsMut, RenderPipelineInOutStage};

pub struct HorizontalChromaUpsample {
    channel: usize,
}

impl HorizontalChromaUpsample {
    pub fn new(channel: usize) -> HorizontalChromaUpsample {
        HorizontalChromaUpsample { channel }
    }
}

impl std::fmt::Display for HorizontalChromaUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "chroma upsample of channel {}, horizontally",
            self.channel
        )
    }
}

#[autoversion]
fn hchroma_upsample(
    _token: SimdToken,
    input: &[f32],
    output: &mut [f32],
    xsize: usize,
) {
    // input has BORDER=1 padding, so input[x] is prev, input[x+1] is cur, input[x+2] is next
    for x in 0..xsize {
        let prev = input[x];
        let cur = input[x + 1];
        let next = input[x + 2];
        // left = 0.25 * prev + 0.75 * cur
        output[x * 2] = 0.25 * prev + 0.75 * cur;
        // right = 0.25 * next + 0.75 * cur
        output[x * 2 + 1] = 0.25 * next + 0.75 * cur;
    }
}

impl RenderPipelineInOutStage for HorizontalChromaUpsample {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (1, 0);
    const BORDER: (u8, u8) = (1, 0);

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
        crate::profile!(chroma_upsample);
        let input = &input_rows[0];
        let output = &mut output_rows[0];
        hchroma_upsample(input[0], output[0], xsize);
    }
}

pub struct VerticalChromaUpsample {
    channel: usize,
}

impl VerticalChromaUpsample {
    pub fn new(channel: usize) -> VerticalChromaUpsample {
        VerticalChromaUpsample { channel }
    }
}

impl std::fmt::Display for VerticalChromaUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "chroma upsample of channel {}, vertically", self.channel)
    }
}

#[autoversion]
fn vchroma_upsample(
    _token: SimdToken,
    input_prev: &[f32],
    input_cur: &[f32],
    input_next: &[f32],
    output_up: &mut [f32],
    output_down: &mut [f32],
    xsize: usize,
) {
    for x in 0..xsize {
        let prev = input_prev[x];
        let cur = input_cur[x];
        let next = input_next[x];
        output_up[x] = 0.25 * prev + 0.75 * cur;
        output_down[x] = 0.25 * next + 0.75 * cur;
    }
}

impl RenderPipelineInOutStage for VerticalChromaUpsample {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 1);
    const BORDER: (u8, u8) = (0, 1);

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
        crate::profile!(chroma_upsample);
        let input = &input_rows[0];
        let output = &mut output_rows[0];
        let (output_up, output_down) = output.split_at_mut(1);
        vchroma_upsample(
            input[0],
            input[1],
            input[2],
            output_up[0],
            output_down[0],
            xsize,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{error::Result, image::Image, render::test::make_and_run_simple_pipeline};
    use test_log::test;

    #[test]
    fn hchr_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || HorizontalChromaUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_hchr() -> Result<()> {
        let mut input = Image::new((3, 1))?;
        input.row_mut(0).copy_from_slice(&[1.0f32, 2.0, 4.0]);
        let stage = HorizontalChromaUpsample::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (6, 1), 0, 256)?;
        assert_eq!(output[0].row(0), [1.0, 1.25, 1.75, 2.5, 3.5, 4.0]);
        Ok(())
    }

    #[test]
    fn vchr_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || VerticalChromaUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_vchr() -> Result<()> {
        let mut input = Image::new((1, 3))?;
        input.row_mut(0)[0] = 1.0f32;
        input.row_mut(1)[0] = 2.0f32;
        input.row_mut(2)[0] = 4.0f32;
        let stage = VerticalChromaUpsample::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (1, 6), 0, 256)?;
        assert_eq!(output[0].row(0)[0], 1.0);
        assert_eq!(output[0].row(1)[0], 1.25);
        assert_eq!(output[0].row(2)[0], 1.75);
        assert_eq!(output[0].row(3)[0], 2.5);
        assert_eq!(output[0].row(4)[0], 3.5);
        assert_eq!(output[0].row(5)[0], 4.0);
        Ok(())
    }
}
