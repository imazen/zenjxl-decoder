// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use archmage::prelude::*;

use crate::api::{
    JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint, adapt_to_xyz_d50,
    primaries_to_xyz, primaries_to_xyz_d50,
};
use crate::error::Result;
use crate::headers::{FileHeader, OpsinInverseMatrix};
use crate::render::stages::from_linear;
use crate::render::{Channels, ChannelsMut, RenderPipelineInOutStage, RenderPipelineInPlaceStage};
use crate::util::{Matrix3x3, eval_rational_poly, inv_3x3_matrix, mul_3x3_matrix};

const SRGB_LUMINANCES: [f32; 3] = [0.2126, 0.7152, 0.0722];

#[derive(Clone)]
pub struct OutputColorInfo {
    // Luminance of each primary.
    pub luminances: [f32; 3],
    pub intensity_target: f32,
    pub opsin: OpsinInverseMatrix,
    pub tf: from_linear::TransferFunction,
}

#[cfg(test)]
impl Default for OutputColorInfo {
    fn default() -> Self {
        use crate::headers::encodings::Empty;
        Self {
            luminances: SRGB_LUMINANCES,
            intensity_target: 255.0,
            opsin: OpsinInverseMatrix::default(&Empty {}),
            tf: from_linear::TransferFunction::Srgb,
        }
    }
}

impl OutputColorInfo {
    fn opsin_matrix_to_matrix3x3(matrix: [f32; 9]) -> Matrix3x3<f64> {
        [
            [matrix[0] as f64, matrix[1] as f64, matrix[2] as f64],
            [matrix[3] as f64, matrix[4] as f64, matrix[5] as f64],
            [matrix[6] as f64, matrix[7] as f64, matrix[8] as f64],
        ]
    }

    fn matrix3x3_to_opsin_matrix(matrix: Matrix3x3<f64>) -> [f32; 9] {
        [
            matrix[0][0] as f32,
            matrix[0][1] as f32,
            matrix[0][2] as f32,
            matrix[1][0] as f32,
            matrix[1][1] as f32,
            matrix[1][2] as f32,
            matrix[2][0] as f32,
            matrix[2][1] as f32,
            matrix[2][2] as f32,
        ]
    }

    pub fn from_header(header: &FileHeader) -> Result<Self> {
        let srgb_output = OutputColorInfo {
            luminances: SRGB_LUMINANCES,
            intensity_target: header.image_metadata.tone_mapping.intensity_target,
            opsin: header.transform_data.opsin_inverse_matrix.clone(),
            tf: from_linear::TransferFunction::Srgb,
        };
        if header.image_metadata.color_encoding.want_icc {
            return Ok(srgb_output);
        }

        let tf;
        let mut inverse_matrix = Self::opsin_matrix_to_matrix3x3(
            header.transform_data.opsin_inverse_matrix.inverse_matrix,
        );
        let mut luminances = SRGB_LUMINANCES;
        let desired_colorspace =
            JxlColorEncoding::from_internal(&header.image_metadata.color_encoding)?;
        match &desired_colorspace {
            JxlColorEncoding::XYB { .. } => {
                return Ok(srgb_output);
            }
            JxlColorEncoding::RgbColorSpace {
                white_point,
                primaries,
                transfer_function,
                ..
            } => {
                tf = transfer_function;
                if *primaries != JxlPrimaries::SRGB || *white_point != JxlWhitePoint::D65 {
                    let [r, g, b] = JxlPrimaries::SRGB.to_xy_coords();
                    let w = JxlWhitePoint::D65.to_xy_coords();
                    let srgb_to_xyzd50 =
                        primaries_to_xyz_d50(r.0, r.1, g.0, g.1, b.0, b.1, w.0, w.1)?;
                    let [r, g, b] = primaries.to_xy_coords();
                    let w = white_point.to_xy_coords();
                    let original_to_xyz = primaries_to_xyz(r.0, r.1, g.0, g.1, b.0, b.1, w.0, w.1)?;
                    luminances = original_to_xyz[1].map(|lum| lum as f32);
                    let adapt_to_d50 = adapt_to_xyz_d50(w.0, w.1)?;
                    let original_to_xyzd50 = mul_3x3_matrix(&adapt_to_d50, &original_to_xyz);
                    let xyzd50_to_original = inv_3x3_matrix(&original_to_xyzd50)?;
                    let srgb_to_original = mul_3x3_matrix(&xyzd50_to_original, &srgb_to_xyzd50);
                    inverse_matrix = mul_3x3_matrix(&srgb_to_original, &inverse_matrix);
                }
            }

            JxlColorEncoding::GrayscaleColorSpace {
                transfer_function, ..
            } => {
                tf = transfer_function;
                let f64_luminances = luminances.map(|lum| lum as f64);
                let srgb_to_luminance: Matrix3x3<f64> =
                    [f64_luminances, f64_luminances, f64_luminances];
                inverse_matrix = mul_3x3_matrix(&srgb_to_luminance, &inverse_matrix);
            }
        }

        let mut opsin = header.transform_data.opsin_inverse_matrix.clone();
        opsin.inverse_matrix = Self::matrix3x3_to_opsin_matrix(inverse_matrix);
        let intensity_target = header.image_metadata.tone_mapping.intensity_target;
        let from_linear_tf = match tf {
            JxlTransferFunction::PQ => from_linear::TransferFunction::Pq { intensity_target },
            JxlTransferFunction::HLG => from_linear::TransferFunction::Hlg {
                intensity_target,
                luminance_rgb: luminances,
            },
            JxlTransferFunction::BT709 => from_linear::TransferFunction::Bt709,
            JxlTransferFunction::Linear => from_linear::TransferFunction::Gamma(1.0),
            JxlTransferFunction::SRGB => from_linear::TransferFunction::Srgb,
            JxlTransferFunction::DCI => from_linear::TransferFunction::Gamma(2.6_f32.recip()),
            JxlTransferFunction::Gamma(g) => from_linear::TransferFunction::Gamma(*g),
        };
        Ok(OutputColorInfo {
            luminances,
            intensity_target,
            opsin,
            tf: from_linear_tf,
        })
    }
}

/// Convert XYB to linear RGB with appropriate primaries, where 1.0 corresponds to `intensity_target` nits.
pub struct XybStage {
    first_channel: usize,
    output_color_info: OutputColorInfo,
}

impl XybStage {
    pub fn new(first_channel: usize, output_color_info: OutputColorInfo) -> Self {
        Self {
            first_channel,
            output_color_info,
        }
    }
}

impl std::fmt::Display for XybStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let channel = self.first_channel;
        write!(
            f,
            "XYB to linear for channel [{},{},{}]",
            channel,
            channel + 1,
            channel + 2
        )
    }
}

#[autoversion]
fn xyb_process(
    _token: SimdToken,
    mat: [f32; 9],
    bias_cbrt: [f32; 3],
    intensity_scale: f32,
    scaled_bias: [f32; 3],
    xsize: usize,
    row_x: &mut [f32],
    row_y: &mut [f32],
    row_b: &mut [f32],
) {
    for i in 0..xsize {
        let x = row_x[i];
        let y = row_y[i];
        let b = row_b[i];

        // Mix and apply bias
        let l = y + x - bias_cbrt[0];
        let m = y - x - bias_cbrt[1];
        let s = b - bias_cbrt[2];

        // Apply biased inverse gamma and scale (1.0 corresponds to `intensity_target` nits)
        let l = (l * l).mul_add(l * intensity_scale, scaled_bias[0]);
        let m = (m * m).mul_add(m * intensity_scale, scaled_bias[1]);
        let s = (s * s).mul_add(s * intensity_scale, scaled_bias[2]);

        // Apply opsin inverse matrix (linear LMS to linear sRGB)
        row_x[i] = mat[0].mul_add(l, mat[1].mul_add(m, mat[2] * s));
        row_y[i] = mat[3].mul_add(l, mat[4].mul_add(m, mat[5] * s));
        row_b[i] = mat[6].mul_add(l, mat[7].mul_add(m, mat[8] * s));
    }
}

fn xyb_process_prepare(opsin: &OpsinInverseMatrix, intensity_target: f32) -> ([f32; 9], [f32; 3], f32, [f32; 3]) {
    let bias = opsin.opsin_biases;
    let bias_cbrt = [bias[0].cbrt(), bias[1].cbrt(), bias[2].cbrt()];
    let intensity_scale = 255.0 / intensity_target;
    let scaled_bias = [
        bias[0] * intensity_scale,
        bias[1] * intensity_scale,
        bias[2] * intensity_scale,
    ];
    (opsin.inverse_matrix, bias_cbrt, intensity_scale, scaled_bias)
}

impl RenderPipelineInPlaceStage for XybStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut (dyn std::any::Any + Send)>,
    ) {
        let [row_x, row_y, row_b] = row else {
            panic!(
                "incorrect number of channels; expected 3, found {}",
                row.len()
            );
        };

        let (mat, bias_cbrt, intensity_scale, scaled_bias) =
            xyb_process_prepare(&self.output_color_info.opsin, self.output_color_info.intensity_target);
        xyb_process(mat, bias_cbrt, intensity_scale, scaled_bias, xsize, row_x, row_y, row_b);
    }
}

/// Fused XYB inverse + transfer function + u8 conversion in a single SIMD pass.
/// Supports sRGB and gamma transfer functions. Eliminates intermediate f32 buffer
/// writes between the three separate stages (XYB, TF, u8 conversion).
pub struct XybToU8Stage {
    first_channel: usize,
    output_color_info: OutputColorInfo,
    bit_depth: u8,
    tf: super::from_linear::TransferFunction,
}

impl XybToU8Stage {
    pub fn new(
        first_channel: usize,
        output_color_info: OutputColorInfo,
        bit_depth: u8,
        tf: super::from_linear::TransferFunction,
    ) -> Self {
        Self {
            first_channel,
            output_color_info,
            bit_depth,
            tf,
        }
    }
}

impl std::fmt::Display for XybToU8Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.first_channel;
        write!(
            f,
            "Fused XYB→{:?}→U{} for channels [{},{},{}]",
            self.tf,
            self.bit_depth,
            c,
            c + 1,
            c + 2,
        )
    }
}

/// XYB inverse, returning linear RGB for a single pixel.
#[inline(always)]
fn xyb_inverse_pixel(
    x: f32, y: f32, b: f32,
    mat: &[f32; 9],
    bias_cbrt: &[f32; 3],
    intensity_scale: f32,
    scaled_bias: &[f32; 3],
) -> (f32, f32, f32) {
    let l = y + x - bias_cbrt[0];
    let m = y - x - bias_cbrt[1];
    let s = b - bias_cbrt[2];
    let l = (l * l).mul_add(l * intensity_scale, scaled_bias[0]);
    let m = (m * m).mul_add(m * intensity_scale, scaled_bias[1]);
    let s = (s * s).mul_add(s * intensity_scale, scaled_bias[2]);
    let r = mat[0].mul_add(l, mat[1].mul_add(m, mat[2] * s));
    let g = mat[3].mul_add(l, mat[4].mul_add(m, mat[5] * s));
    let b = mat[6].mul_add(l, mat[7].mul_add(m, mat[8] * s));
    (r, g, b)
}

#[allow(clippy::excessive_precision)]
const SRGB_P: [f32; 5] = [
    -5.135152395e-4,
    5.287254571e-3,
    3.903842876e-1,
    1.474205315,
    7.352629620e-1,
];
#[allow(clippy::excessive_precision)]
const SRGB_Q: [f32; 5] = [
    1.004519624e-2,
    3.036675394e-1,
    1.340816930,
    9.258482155e-1,
    2.424867759e-2,
];

#[inline(always)]
fn srgb_to_u8_sample(linear: f32, max_val: f32) -> u8 {
    let a = linear.clamp(0.0, 1.0);
    let srgb = if a <= 0.0031308 {
        a * 12.92
    } else {
        eval_rational_poly(a.sqrt(), SRGB_P, SRGB_Q)
    };
    (srgb * max_val).round() as u8
}

#[autoversion]
fn xyb_to_srgb_u8_inner(
    _token: SimdToken,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_r: &mut [u8],
    out_g: &mut [u8],
    out_b: &mut [u8],
    mat: [f32; 9],
    bias_cbrt: [f32; 3],
    intensity_scale: f32,
    scaled_bias: [f32; 3],
    max_val: f32,
    xsize: usize,
) {
    for i in 0..xsize {
        let (r_lin, g_lin, b_lin) = xyb_inverse_pixel(
            in_x[i], in_y[i], in_b[i],
            &mat, &bias_cbrt, intensity_scale, &scaled_bias,
        );
        out_r[i] = srgb_to_u8_sample(r_lin, max_val);
        out_g[i] = srgb_to_u8_sample(g_lin, max_val);
        out_b[i] = srgb_to_u8_sample(b_lin, max_val);
    }
}

#[autoversion]
fn xyb_to_gamma_u8_inner(
    _token: SimdToken,
    in_x: &[f32],
    in_y: &[f32],
    in_b: &[f32],
    out_r: &mut [u8],
    out_g: &mut [u8],
    out_b: &mut [u8],
    mat: [f32; 9],
    bias_cbrt: [f32; 3],
    intensity_scale: f32,
    scaled_bias: [f32; 3],
    gamma: f32,
    max_val: f32,
    xsize: usize,
) {
    for i in 0..xsize {
        let (r_lin, g_lin, b_lin) = xyb_inverse_pixel(
            in_x[i], in_y[i], in_b[i],
            &mat, &bias_cbrt, intensity_scale, &scaled_bias,
        );
        // Gamma TF + u8 quantize: powf(abs(x), gamma) * copysign(1, x)
        let r_tf = crate::util::fast_powf(r_lin.abs(), gamma).copysign(r_lin);
        let g_tf = crate::util::fast_powf(g_lin.abs(), gamma).copysign(g_lin);
        let b_tf = crate::util::fast_powf(b_lin.abs(), gamma).copysign(b_lin);
        out_r[i] = (r_tf * max_val).round().clamp(0.0, max_val) as u8;
        out_g[i] = (g_tf * max_val).round().clamp(0.0, max_val) as u8;
        out_b[i] = (b_tf * max_val).round().clamp(0.0, max_val) as u8;
    }
}

impl RenderPipelineInOutStage for XybToU8Stage {
    type InputT = f32;
    type OutputT = u8;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<u8>,
        _state: Option<&mut (dyn std::any::Any + Send)>,
    ) {
        let max_val = ((1u32 << self.bit_depth) - 1) as f32;
        let (mat, bias_cbrt, intensity_scale, scaled_bias) =
            xyb_process_prepare(&self.output_color_info.opsin, self.output_color_info.intensity_target);

        let in_x = input_rows[0][0];
        let in_y = input_rows[1][0];
        let in_b = input_rows[2][0];
        let (out_r, out_g, out_b) = output_rows.split_first_3_mut();

        match self.tf {
            super::from_linear::TransferFunction::Srgb => {
                xyb_to_srgb_u8_inner(
                    in_x, in_y, in_b,
                    out_r[0], out_g[0], out_b[0],
                    mat, bias_cbrt, intensity_scale, scaled_bias,
                    max_val, xsize,
                );
            }
            super::from_linear::TransferFunction::Gamma(gamma) => {
                xyb_to_gamma_u8_inner(
                    in_x, in_y, in_b,
                    out_r[0], out_g[0], out_b[0],
                    mat, bias_cbrt, intensity_scale, scaled_bias,
                    gamma, max_val, xsize,
                );
            }
            _ => unreachable!("XybToU8Stage only supports Srgb and Gamma TFs"),
        }
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
            || XybStage::new(0, OutputColorInfo::default()),
            (500, 500),
            3,
        )
    }

    #[test]
    fn srgb_primaries() -> Result<()> {
        let mut input_x = Image::new((3, 1))?;
        let mut input_y = Image::new((3, 1))?;
        let mut input_b = Image::new((3, 1))?;
        input_x
            .row_mut(0)
            .copy_from_slice(&[0.028100073, -0.015386105, 0.0]);
        input_y
            .row_mut(0)
            .copy_from_slice(&[0.4881882, 0.71478134, 0.2781282]);
        input_b
            .row_mut(0)
            .copy_from_slice(&[0.471659, 0.43707693, 0.66613984]);

        let stage = XybStage::new(0, OutputColorInfo::default());
        let output =
            make_and_run_simple_pipeline(stage, &[input_x, input_y, input_b], (3, 1), 0, 256)?;

        assert_all_almost_abs_eq(output[0].row(0), &[1.0, 0.0, 0.0], 1e-6);
        assert_all_almost_abs_eq(output[1].row(0), &[0.0, 1.0, 0.0], 1e-6);
        assert_all_almost_abs_eq(output[2].row(0), &[0.0, 0.0, 1.0], 1e-6);

        Ok(())
    }

    #[test]
    fn fused_xyb_srgb_u8_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || {
                XybToU8Stage::new(
                    0,
                    OutputColorInfo::default(),
                    8,
                    super::super::from_linear::TransferFunction::Srgb,
                )
            },
            (500, 500),
            3,
        )
    }

    #[test]
    fn fused_xyb_gamma_u8_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || {
                XybToU8Stage::new(
                    0,
                    OutputColorInfo::default(),
                    8,
                    super::super::from_linear::TransferFunction::Gamma(0.454545),
                )
            },
            (500, 500),
            3,
        )
    }
}
