// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;
use std::fmt;

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum ColorSpace {
    RGB,
    Gray,
    XYB,
    Unknown,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ColorSpace::RGB => "RGB",
                ColorSpace::Gray => "Gra",
                ColorSpace::XYB => "XYB",
                ColorSpace::Unknown => "CS?",
            }
        )
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    DCI = 11,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Primaries {
    SRGB = 1,
    Custom = 2,
    BT2100 = 9,
    P3 = 11,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum TransferFunction {
    BT709 = 1,
    Unknown = 2,
    Linear = 8,
    SRGB = 13,
    PQ = 16,
    DCI = 17,
    HLG = 18,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative,
    Saturation,
    Absolute,
}

impl fmt::Display for RenderingIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RenderingIntent::Perceptual => "Per",
                RenderingIntent::Relative => "Rel",
                RenderingIntent::Saturation => "Sat",
                RenderingIntent::Absolute => "Abs",
            }
        )
    }
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct CustomXY {
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub x: i32,
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub y: i32,
}

impl CustomXY {
    /// Converts the stored scaled integer coordinates to f32 (x, y) values.
    pub fn as_f32_coords(&self) -> (f32, f32) {
        (self.x as f32 / 1_000_000.0, self.y as f32 / 1_000_000.0)
    }

    pub fn from_f32_coords(x: f32, y: f32) -> Self {
        Self {
            x: (x * 1_000_000.0).round() as i32,
            y: (y * 1_000_000.0).round() as i32,
        }
    }
}

pub struct CustomTransferFunctionNonserialized {
    color_space: ColorSpace,
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[nonserialized(CustomTransferFunctionNonserialized)]
#[validate]
pub struct CustomTransferFunction {
    #[condition(nonserialized.color_space != ColorSpace::XYB)]
    #[default(nonserialized.color_space == ColorSpace::XYB)]
    pub have_gamma: bool,
    #[condition(have_gamma && nonserialized.color_space != ColorSpace::XYB)]
    #[default(3333333)] // XYB gamma
    #[coder(Bits(24))]
    pub gamma: u32,
    #[condition(!have_gamma && nonserialized.color_space != ColorSpace::XYB)]
    #[default(TransferFunction::SRGB)]
    pub transfer_function: TransferFunction,
}

impl CustomTransferFunction {
    #[cfg(test)]
    pub fn empty() -> CustomTransferFunction {
        CustomTransferFunction {
            have_gamma: false,
            gamma: 0,
            transfer_function: TransferFunction::Unknown,
        }
    }
    pub fn gamma(&self) -> f32 {
        assert!(self.have_gamma);
        self.gamma as f32 * 0.0000001
    }

    pub fn check(&self, _: &CustomTransferFunctionNonserialized) -> Result<(), Error> {
        if self.have_gamma {
            let gamma = self.gamma();
            if gamma > 1.0 || gamma * 8192.0 < 1.0 {
                Err(Error::InvalidGamma(gamma))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[validate]
pub struct ColorEncoding {
    // all_default is never read.
    #[allow(dead_code)]
    #[all_default]
    all_default: bool,
    #[default(false)]
    pub want_icc: bool,
    #[default(ColorSpace::RGB)]
    pub color_space: ColorSpace,
    #[condition(!want_icc && color_space != ColorSpace::XYB)]
    #[default(WhitePoint::D65)]
    pub white_point: WhitePoint,
    // TODO(veluca): can this be merged in the enum?!
    #[condition(white_point == WhitePoint::Custom)]
    #[default(CustomXY::default(&field_nonserialized))]
    pub white: CustomXY,
    #[condition(!want_icc && color_space != ColorSpace::XYB && color_space != ColorSpace::Gray)]
    #[default(Primaries::SRGB)]
    pub primaries: Primaries,
    #[condition(primaries == Primaries::Custom)]
    #[default([CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized)])]
    pub custom_primaries: [CustomXY; 3],
    #[condition(!want_icc)]
    #[default(CustomTransferFunction::default(&field_nonserialized))]
    #[nonserialized(color_space)]
    pub tf: CustomTransferFunction,
    #[condition(!want_icc)]
    #[default(RenderingIntent::Relative)]
    pub rendering_intent: RenderingIntent,
}

impl ColorEncoding {
    pub fn check(&self, _: &Empty) -> Result<(), Error> {
        if self.color_space == ColorSpace::Unknown
            || (!self.want_icc
                && !self.tf.have_gamma
                && self.tf.transfer_function == TransferFunction::Unknown)
        {
            return Err(Error::InvalidColorEncoding);
        }
        if !self.want_icc
            && self.color_space == ColorSpace::XYB
            && self.rendering_intent != RenderingIntent::Perceptual
        {
            return Err(Error::InvalidRenderingIntent);
        }
        // A custom white point or custom primaries that cannot produce an ICC
        // profile (degenerate chromaticities, or matrices outside s15Fixed16)
        // are rejected at the header instead of later. `white` is only
        // meaningful when `white_point == Custom`, so the check goes through
        // `JxlColorEncoding`, which resolves the named white points.
        // Upstream jxl-rs 0f41860 + 96b0be3.
        if !self.want_icc
            && (self.white_point == WhitePoint::Custom || self.primaries == Primaries::Custom)
        {
            if self.primaries == Primaries::Custom
                && matches!(self.color_space, ColorSpace::RGB | ColorSpace::Unknown)
            {
                let [r, g, b] = &self.custom_primaries;
                // The determinant of the XYZ columns (x, y, 1-x-y) equals
                // this signed xy area. Use encoded millionth coordinates
                // exactly: 100 / 10^12 is the inclusive 1e-10 boundary.
                let det = (r.x as i128 - b.x as i128) * (g.y as i128 - b.y as i128)
                    - (g.x as i128 - b.x as i128) * (r.y as i128 - b.y as i128);
                if det.abs() < 100 {
                    return Err(Error::MatrixInversionFailed(det.abs() as f64 / 1e12));
                }
            }
            crate::api::JxlColorEncoding::from_internal(self)
                .map_err(|e| e.decompose().0)?
                .validate_icc_matrix()?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn xyb_wire_uses_implicit_gamma_and_perceptual_intent() {
        // all_default=false, want_icc=false, Enum(XYB)=2, intent=Perceptual.
        // Neither a transfer-function bit nor gamma bits are present.
        let bytes = [0x08, 0];
        let mut reader = BitReader::new(&bytes);
        let encoding = ColorEncoding::read_unconditional(&(), &mut reader, &Empty {}).unwrap();
        assert_eq!(reader.total_bits_read(), 10);
        assert_eq!(encoding.color_space, ColorSpace::XYB);
        assert!(encoding.tf.have_gamma);
        assert_eq!(encoding.tf.gamma, 3_333_333);
        assert!(matches!(
            crate::api::JxlColorEncoding::from_internal(&encoding).unwrap(),
            crate::api::JxlColorEncoding::XYB {
                rendering_intent: RenderingIntent::Perceptual
            }
        ));
        let mut reader = BitReader::new(&[0x08, 1]); // Relative intent is invalid for XYB
        assert!(matches!(
            ColorEncoding::read_unconditional(&(), &mut reader, &Empty {}),
            Err(Error::InvalidRenderingIntent)
        ));
    }

    #[test]
    fn review_color_gray_white_point_bounds() {
        let mut encoding = ColorEncoding::default(&Empty {});
        encoding.color_space = ColorSpace::Gray;
        encoding.white_point = WhitePoint::Custom;
        for (x, y) in [
            (-1, 500_000),
            (1_000_001, 500_000),
            (300_000, 0),
            (300_000, -1),
            (300_000, 1_000_001),
        ] {
            encoding.white = CustomXY { x, y };
            assert!(encoding.check(&Empty {}).is_err(), "white ({x}, {y})");
        }
        for (x, y) in [(0, 1_000_000), (1_000_000, 1_000_000), (312_700, 329_000)] {
            encoding.white = CustomXY { x, y };
            encoding.check(&Empty {}).unwrap();
        }
    }

    #[test]
    fn review_color_quantized_primary_determinant() {
        let mut encoding = ColorEncoding::default(&Empty {});
        encoding.white_point = WhitePoint::Custom;
        encoding.primaries = Primaries::Custom;
        encoding.white = CustomXY {
            x: 100_001,
            y: 100_001,
        };
        encoding.custom_primaries = [
            CustomXY {
                x: 100_000,
                y: 100_000,
            },
            CustomXY {
                x: 100_002,
                y: 100_000,
            },
            CustomXY {
                x: 100_000,
                y: 100_002,
            },
        ];
        // The unscaled XYZ-column determinant is 4e-12, below 1e-10.
        assert!(encoding.check(&Empty {}).is_err());
        encoding.white = CustomXY {
            x: 100_003,
            y: 100_003,
        };
        encoding.custom_primaries[1].x = 100_010;
        encoding.custom_primaries[2].y = 100_010;
        // Exactly 1e-10 in the encoded millionth coordinates is allowed.
        encoding.check(&Empty {}).unwrap();
        encoding.white_point = WhitePoint::D65;
        encoding.custom_primaries = [
            CustomXY { x: 1_000_000, y: 0 },
            CustomXY { x: 0, y: 1_000_000 },
            CustomXY { x: 0, y: 0 },
        ];
        // Imaginary XYZ primaries, including y=0, remain representable.
        encoding.check(&Empty {}).unwrap();
    }

    #[test]
    fn test_custom_primaries_with_standard_white_point() {
        let mut encoding = ColorEncoding::default(&Empty {});
        encoding.primaries = Primaries::Custom;
        encoding.white_point = WhitePoint::D65;
        // Set standard sRGB-like primaries
        encoding.custom_primaries = [
            CustomXY::from_f32_coords(0.64, 0.33),
            CustomXY::from_f32_coords(0.30, 0.60),
            CustomXY::from_f32_coords(0.15, 0.06),
        ];
        assert!(encoding.check(&Empty {}).is_ok());
    }

    #[test]
    fn test_custom_white_point_invalid() {
        let mut encoding = ColorEncoding::default(&Empty {});
        encoding.white_point = WhitePoint::Custom;
        encoding.white = CustomXY::from_f32_coords(0.3127, 0.0);
        assert!(encoding.check(&Empty {}).is_err());
    }

    #[test]
    fn test_degenerate_primaries_cannot_create_icc() {
        let mut encoding = ColorEncoding::default(&Empty {});
        encoding.white_point = WhitePoint::DCI;
        encoding.primaries = Primaries::Custom;
        encoding.custom_primaries = [
            CustomXY::from_f32_coords(0.333093, 0.28672),
            CustomXY::from_f32_coords(1.6e-5, 8e-6),
            CustomXY::from_f32_coords(0.0, 0.0),
        ];
        encoding.tf = CustomTransferFunction {
            have_gamma: false,
            transfer_function: TransferFunction::HLG,
            gamma: 0,
        };
        assert!(encoding.check(&Empty {}).is_err());
    }
}
