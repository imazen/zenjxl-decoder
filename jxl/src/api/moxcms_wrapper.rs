// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! moxcms-based CMS implementation for JPEG XL.
//!
//! This module provides a color management system implementation using the moxcms crate,
//! enabling ICC profile-based color transforms.

use crate::api::{JxlColorEncoding, JxlColorProfile, JxlPrimaries, JxlTransferFunction};
use crate::error::{Error, Result};

use super::color::{JxlCms, JxlCmsTransformer};

/// A CMS implementation using moxcms.
#[derive(Default, Clone)]
pub struct MoxCms;

impl MoxCms {
    pub fn new() -> Self {
        Self
    }
}

/// Wrapper around moxcms TransformExecutor to implement JxlCmsTransformer.
struct MoxCmsTransformer {
    transform: Box<dyn moxcms::TransformExecutor<f32> + Send + Sync>,
    input_channels: usize,
    output_channels: usize,
}

impl JxlCmsTransformer for MoxCmsTransformer {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.transform
            .transform(input, output)
            .map_err(|e| Error::CmsError(format!("moxcms transform error: {:?}", e)))
    }

    fn do_transform_inplace(&mut self, inout: &mut [f32]) -> Result<()> {
        if self.input_channels != self.output_channels {
            return Err(Error::CmsError(
                "in-place transform requires same input/output channel count".to_string(),
            ));
        }

        // moxcms doesn't support in-place transforms, so we need a temporary buffer.
        // For efficiency, we could pool these buffers, but for now just allocate.
        let input_copy = inout.to_vec();
        self.transform
            .transform(&input_copy, inout)
            .map_err(|e| Error::CmsError(format!("moxcms transform error: {:?}", e)))
    }
}

/// Convert a JxlColorProfile to a moxcms ColorProfile.
fn to_moxcms_profile(profile: &JxlColorProfile) -> Result<moxcms::ColorProfile> {
    match profile {
        JxlColorProfile::Icc(icc_data) => moxcms::ColorProfile::new_from_slice(icc_data)
            .map_err(|e| Error::CmsError(format!("moxcms ICC parse error: {:?}", e))),
        JxlColorProfile::Simple(encoding) => {
            // For simple encodings, generate an ICC profile and parse it
            if let Some(icc_data) = encoding.maybe_create_profile()? {
                moxcms::ColorProfile::new_from_slice(&icc_data)
                    .map_err(|e| Error::CmsError(format!("moxcms ICC parse error: {:?}", e)))
            } else {
                // Try to use built-in moxcms profiles for common color spaces
                match encoding {
                    JxlColorEncoding::RgbColorSpace {
                        primaries: JxlPrimaries::SRGB,
                        transfer_function: JxlTransferFunction::SRGB,
                        ..
                    } => Ok(moxcms::ColorProfile::new_srgb()),
                    JxlColorEncoding::RgbColorSpace {
                        primaries: JxlPrimaries::P3,
                        transfer_function: JxlTransferFunction::SRGB,
                        ..
                    } => Ok(moxcms::ColorProfile::new_display_p3()),
                    JxlColorEncoding::RgbColorSpace {
                        primaries: JxlPrimaries::BT2100,
                        transfer_function: JxlTransferFunction::PQ,
                        ..
                    } => Ok(moxcms::ColorProfile::new_bt2020_pq()),
                    JxlColorEncoding::RgbColorSpace {
                        primaries: JxlPrimaries::BT2100,
                        transfer_function: JxlTransferFunction::HLG,
                        ..
                    } => Ok(moxcms::ColorProfile::new_bt2020_hlg()),
                    JxlColorEncoding::GrayscaleColorSpace {
                        transfer_function: JxlTransferFunction::Gamma(gamma),
                        ..
                    } => Ok(moxcms::ColorProfile::new_gray_with_gamma(*gamma)),
                    _ => Err(Error::CmsError(
                        "Cannot create ICC profile for this color encoding".to_string(),
                    )),
                }
            }
        }
    }
}

/// Determine the moxcms Layout for a given profile.
fn get_layout(profile: &JxlColorProfile) -> moxcms::Layout {
    match profile {
        JxlColorProfile::Icc(_) => {
            // For ICC profiles, we assume RGB. A more complete implementation
            // would parse the ICC header to determine the color space.
            moxcms::Layout::Rgb
        }
        JxlColorProfile::Simple(encoding) => match encoding {
            JxlColorEncoding::RgbColorSpace { .. } | JxlColorEncoding::XYB { .. } => {
                moxcms::Layout::Rgb
            }
            JxlColorEncoding::GrayscaleColorSpace { .. } => moxcms::Layout::Gray,
        },
    }
}

/// Get number of channels for a layout.
fn layout_channels(layout: moxcms::Layout) -> usize {
    match layout {
        moxcms::Layout::Rgb => 3,
        moxcms::Layout::Rgba => 4,
        moxcms::Layout::Gray => 1,
        moxcms::Layout::GrayAlpha => 2,
    }
}

impl JxlCms for MoxCms {
    fn initialize_transforms(
        &self,
        n: usize,
        _max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _intensity_target: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer>>)> {
        let src_profile = to_moxcms_profile(&input)?;
        let dst_profile = to_moxcms_profile(&output)?;

        let src_layout = get_layout(&input);
        let dst_layout = get_layout(&output);

        let input_channels = layout_channels(src_layout);
        let output_channels = layout_channels(dst_layout);

        // Use default transform options (perceptual intent)
        let options = moxcms::TransformOptions::default();

        let mut transforms: Vec<Box<dyn JxlCmsTransformer>> = Vec::with_capacity(n);

        for _ in 0..n {
            let transform = src_profile
                .create_transform_f32(src_layout, &dst_profile, dst_layout, options)
                .map_err(|e| Error::CmsError(format!("moxcms create_transform error: {:?}", e)))?;

            transforms.push(Box::new(MoxCmsTransformer {
                transform,
                input_channels,
                output_channels,
            }));
        }

        Ok((output_channels, transforms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{JxlPrimaries, JxlTransferFunction, JxlWhitePoint};
    use crate::headers::color_encoding::RenderingIntent;

    #[test]
    fn test_srgb_identity_transform() -> Result<()> {
        let cms = MoxCms::new();

        let srgb = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent: RenderingIntent::Relative,
        });

        let (output_channels, mut transforms) =
            cms.initialize_transforms(1, 1024, srgb.clone(), srgb, 255.0)?;

        assert_eq!(output_channels, 3);
        assert_eq!(transforms.len(), 1);

        // Test a simple RGB value
        let input = [0.5f32, 0.3, 0.8];
        let mut output = [0.0f32; 3];

        transforms[0].do_transform(&input, &mut output)?;

        // sRGB to sRGB should be approximately identity
        assert!((output[0] - 0.5).abs() < 0.01);
        assert!((output[1] - 0.3).abs() < 0.01);
        assert!((output[2] - 0.8).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_srgb_to_display_p3() -> Result<()> {
        let cms = MoxCms::new();

        let srgb = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent: RenderingIntent::Relative,
        });

        let p3 = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::P3,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent: RenderingIntent::Relative,
        });

        let (output_channels, mut transforms) =
            cms.initialize_transforms(1, 1024, srgb, p3, 255.0)?;

        assert_eq!(output_channels, 3);
        assert_eq!(transforms.len(), 1);

        // Test sRGB red in Display P3: should be less saturated
        let srgb_red = [1.0f32, 0.0, 0.0];
        let mut p3_output = [0.0f32; 3];

        transforms[0].do_transform(&srgb_red, &mut p3_output)?;

        // sRGB red when expressed in P3 should have positive G and B
        // (because sRGB primaries fit inside P3 gamut)
        assert!(p3_output[0] > 0.8 && p3_output[0] < 1.0);
        assert!(p3_output[1] > 0.0); // Some green
        assert!(p3_output[2] > 0.0); // Some blue

        Ok(())
    }
}
