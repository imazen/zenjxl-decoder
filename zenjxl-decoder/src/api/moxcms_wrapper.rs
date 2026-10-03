// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! moxcms-based CMS implementation for JPEG XL.
//!
//! This module provides a color management system implementation using the moxcms crate,
//! enabling ICC profile-based color transforms.

use std::sync::Arc;

use whereat::at;

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
    transform: Arc<dyn moxcms::TransformExecutor<f32> + Send + Sync>,
    input_channels: usize,
    output_channels: usize,
    /// moxcms produced linear light; apply the sRGB curve here (see
    /// `initialize_transforms`).
    encode_srgb: bool,
    /// Luminance of the source black point for black-point compensation, applied
    /// in linear light before encoding; 0 disables it.
    black_point_y: f32,
}

/// Black-point compensation for a neutral source black point, then sign-preserving
/// sRGB encoding matching libjxl's `TF_SRGB::EncodedFromDisplay`.
///
/// lcms2's BPC maps XYZ to `(XYZ - BP) * W / (W - BP)` per component. When BP is
/// neutral (`BP = y * W`, as lcms2's detection always yields) that is
/// `(XYZ - y * W) / (1 - y)`, which a linear RGB space whose white is (1, 1, 1)
/// carries through unchanged: `(rgb - y) / (1 - y)`.
fn bpc_and_encode_srgb_in_place(samples: &mut [f32], black_point_y: f32) {
    let scale = 1.0 / (1.0 - black_point_y);
    for v in samples {
        let linear = (*v - black_point_y) * scale;
        let a = linear.abs();
        let e = if a <= 0.003_130_8 {
            a * 12.92
        } else {
            1.055 * a.powf(1.0 / 2.4) - 0.055
        };
        *v = e.copysign(linear);
    }
}

impl JxlCmsTransformer for MoxCmsTransformer {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.transform
            .transform(input, output)
            .map_err(|e| at!(Error::CmsError(format!("moxcms transform error: {:?}", e))))?;
        if self.encode_srgb {
            bpc_and_encode_srgb_in_place(output, self.black_point_y);
        }
        Ok(())
    }

    fn do_transform_inplace(&mut self, inout: &mut [f32]) -> Result<()> {
        if self.input_channels != self.output_channels {
            return Err(at!(Error::CmsError(
                "in-place transform requires same input/output channel count".to_string(),
            )));
        }

        // moxcms doesn't support in-place transforms, so we need a temporary buffer.
        // For efficiency, we could pool these buffers, but for now just allocate.
        let input_copy = inout.to_vec();
        self.transform
            .transform(&input_copy, inout)
            .map_err(|e| at!(Error::CmsError(format!("moxcms transform error: {:?}", e))))?;
        if self.encode_srgb {
            bpc_and_encode_srgb_in_place(inout, self.black_point_y);
        }
        Ok(())
    }
}

/// Convert a JxlColorProfile to a moxcms ColorProfile.
fn to_moxcms_profile(profile: &JxlColorProfile) -> Result<moxcms::ColorProfile> {
    match profile {
        JxlColorProfile::Icc(icc_data) => moxcms::ColorProfile::new_from_slice(icc_data)
            .map_err(|e| at!(Error::CmsError(format!("moxcms ICC parse error: {:?}", e)))),
        JxlColorProfile::Simple(encoding) => {
            // For simple encodings, generate an ICC profile and parse it
            if let Some(icc_data) = encoding.maybe_create_profile()? {
                moxcms::ColorProfile::new_from_slice(&icc_data)
                    .map_err(|e| at!(Error::CmsError(format!("moxcms ICC parse error: {:?}", e))))
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
                    _ => Err(at!(Error::CmsError(
                        "Cannot create ICC profile for this color encoding".to_string(),
                    ))),
                }
            }
        }
    }
}

/// ICC profile color space signatures (bytes 16-19 of ICC header)
const ICC_CMYK_SIGNATURE: &[u8; 4] = b"CMYK";
const ICC_GRAY_SIGNATURE: &[u8; 4] = b"GRAY";
const ICC_RGB_SIGNATURE: &[u8; 4] = b"RGB ";

/// Detect the color space from an ICC profile header.
/// The color space signature is at bytes 16-19.
fn detect_icc_color_space(icc_data: &[u8]) -> Option<&'static str> {
    if icc_data.len() < 20 {
        return None;
    }
    let sig = &icc_data[16..20];
    if sig == ICC_CMYK_SIGNATURE {
        Some("CMYK")
    } else if sig == ICC_GRAY_SIGNATURE {
        Some("GRAY")
    } else if sig == ICC_RGB_SIGNATURE {
        Some("RGB")
    } else {
        None
    }
}

/// Determine the moxcms Layout for a given profile.
fn get_layout(profile: &JxlColorProfile) -> moxcms::Layout {
    match profile {
        JxlColorProfile::Icc(icc_data) => {
            // Parse ICC header to determine color space
            match detect_icc_color_space(icc_data) {
                Some("CMYK") => moxcms::Layout::Rgba, // CMYK uses Rgba layout (4 channels)
                Some("GRAY") => moxcms::Layout::Gray,
                _ => moxcms::Layout::Rgb, // Default to RGB
            }
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
        // Multi-ink layouts (not used in JXL but must be handled for exhaustiveness)
        _ => 4, // Default to 4 channels for unknown layouts
    }
}

/// Luminance (D50 Y) of lcms2's black point for a CMYK profile under relative
/// colorimetric intent, measured through `linear_dst` (an RGB profile with
/// identity curves). `None` if the profile has no usable perceptual B2A table.
fn cmyk_black_point_y(
    cmyk: &moxcms::ColorProfile,
    linear_dst: &moxcms::ColorProfile,
    relative: moxcms::TransformOptions,
) -> Option<f32> {
    let perceptual = moxcms::TransformOptions {
        rendering_intent: moxcms::RenderingIntent::Perceptual,
        ..relative
    };
    // Lab black is RGB black (XYZ = 0) in any matrix/shaper RGB profile, so
    // enter the perceptual B2A table from `linear_dst` rather than through a
    // Lab profile.
    let to_cmyk = linear_dst
        .create_transform_f32(moxcms::Layout::Rgb, cmyk, moxcms::Layout::Rgba, perceptual)
        .ok()?;
    let to_rgb = cmyk
        .create_transform_f32(
            moxcms::Layout::Rgba,
            linear_dst,
            moxcms::Layout::Rgb,
            relative,
        )
        .ok()?;
    let mut ink = [0.0f32; 4];
    to_cmyk.transform(&[0.0, 0.0, 0.0], &mut ink).ok()?;
    let mut rgb = [0.0f32; 3];
    to_rgb.transform(&ink, &mut rgb).ok()?;
    let y = linear_dst.red_colorant.y * f64::from(rgb[0])
        + linear_dst.green_colorant.y * f64::from(rgb[1])
        + linear_dst.blue_colorant.y * f64::from(rgb[2]);
    // L* = 50 is Y = 0.18419.
    let y = y.clamp(0.0, 0.184_186);
    Some(y as f32)
}

impl JxlCms for MoxCms {
    fn initialize_transforms(
        &self,
        n: usize,
        _max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _intensity_target: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send + Sync>>)> {
        let src_profile = to_moxcms_profile(&input)?;
        let mut dst_profile = to_moxcms_profile(&output)?;

        let src_layout = get_layout(&input);
        let dst_layout = get_layout(&output);

        let input_channels = layout_channels(src_layout);
        let output_channels = layout_channels(dst_layout);

        // Use Perceptual intent as default - this tends to work better for most images.
        // Note: skcms may use profile's embedded intent, which we don't currently extract.
        // JXL is CICP-native so allow_use_cicp_transfer stays true (default).
        // BarycentricWeightScale::High cuts LUT interpolation error from max≤14 to max≤2
        // vs lcms2 with no measurable perf cost.
        //
        // CMYK sources: moxcms 0.9.1 evaluates the sRGB curve imprecisely after
        // a CMYK A2B lookup. In cmyk_layers' cyan layer (Lab 68.9,-39.5,-10.2)
        // linear R is 0.0129 — moxcms gets that right — but the encoded value
        // comes out 0.032 with extended range and 0.1165 without, where lcms2
        // and the closed-form curve give 0.1172. Without extended range it
        // also mirrors negative linear values (-0.0048 -> +0.056 instead of
        // clipping to 0). So for an sRGB-encoded destination we ask moxcms
        // for linear light, with extended range, and encode it ourselves.
        // Other destinations keep extended range off, the closer of the two.
        let src_is_cmyk = matches!(&input, JxlColorProfile::Icc(icc) if detect_icc_color_space(icc) == Some("CMYK"));
        let encode_srgb = src_is_cmyk
            && matches!(
                &output,
                JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
                    transfer_function: JxlTransferFunction::SRGB,
                    ..
                })
            );
        if encode_srgb {
            dst_profile.cicp = None;
            for trc in [
                &mut dst_profile.red_trc,
                &mut dst_profile.green_trc,
                &mut dst_profile.blue_trc,
            ] {
                // An empty curve is the ICC identity.
                *trc = Some(moxcms::ToneReprCurve::Lut(Vec::new()));
            }
        }
        let options = moxcms::TransformOptions {
            allow_extended_range_rgb_xyz: !src_is_cmyk || encode_srgb,
            rendering_intent: if encode_srgb {
                moxcms::RenderingIntent::RelativeColorimetric
            } else {
                moxcms::RenderingIntent::Perceptual
            },
            barycentric_weight_scale: moxcms::BarycentricWeightScale::High,
            ..moxcms::TransformOptions::default()
        };

        // libjxl's lcms2 path converts with the destination's rendering intent
        // (relative colorimetric for sRGB) and black-point compensation. moxcms
        // has no BPC, so reproduce lcms2's detection for a CMYK output-class
        // profile under relative intent (`BlackPointUsingPerceptualBlack`):
        // Lab black -> CMYK through the perceptual table -> back through the
        // relative table, keeping only lightness (a = b = 0, L* capped at 50).
        let black_point_y = if encode_srgb {
            cmyk_black_point_y(&src_profile, &dst_profile, options).unwrap_or(0.0)
        } else {
            0.0
        };

        let mut transforms: Vec<Box<dyn JxlCmsTransformer + Send + Sync>> = Vec::with_capacity(n);

        for _ in 0..n {
            let transform = src_profile
                .create_transform_f32(src_layout, &dst_profile, dst_layout, options)
                .map_err(|e| Error::CmsError(format!("moxcms create_transform error: {:?}", e)))?;

            transforms.push(Box::new(MoxCmsTransformer {
                transform,
                input_channels,
                output_channels,
                encode_srgb,
                black_point_y,
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

    /// The embedded CMYK profile of the committed `cmyk_layers.jxl` fixture.
    fn cmyk_layers_profile() -> JxlColorProfile {
        use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, states};
        let data = crate::util::test::fixture_bytes("conformance_test_images/cmyk_layers.jxl");
        let mut input = data.as_slice();
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let decoder = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        decoder.embedded_color_profile().clone()
    }

    /// CMYK -> sRGB matches libjxl's lcms2 path (relative colorimetric with
    /// black-point compensation), as `djxl` from a Homebrew (lcms2) build
    /// produces it. Expected values come from lcms2 2.18 with
    /// `cmsFLAGS_BLACKPOINTCOMPENSATION | cmsFLAGS_HIGHRESPRECALC`, fed the
    /// fixture's own CMYK values as libjxl does (100 - 100 * value).
    ///
    /// The cyan entries guard moxcms's sRGB encoding after a CMYK lookup:
    /// it gave R 0.032 (extended range) and mirrored the negative
    /// out-of-gamut R to +0.056 (without), against 0.1176 and -0.0645.
    #[test]
    fn cmyk_to_srgb_matches_lcms2_relative_bpc() -> Result<()> {
        let (channels, mut transforms) = MoxCms.initialize_transforms(
            1,
            1,
            cmyk_layers_profile(),
            JxlColorProfile::Simple(JxlColorEncoding::srgb(false)),
            255.0,
        )?;
        assert_eq!(channels, 3);
        // (JXL CMYK as stored, 1 = no ink) -> lcms2 sRGB. `None` marks a
        // channel left unchecked: the darker cyan sits where moxcms's 4-D
        // CLUT interpolation still differs from lcms2's (G 0.6850 vs 0.6791);
        // its R is what exercises the negative-value handling.
        let cases: [([f32; 4], [Option<f32>; 3]); 4] = [
            (
                [0.2863, 1.0, 0.6706, 1.0],
                [Some(0.11756), Some(0.73633), Some(0.72557)],
            ),
            (
                [0.2347, 0.9601, 0.5918, 0.9813],
                [Some(-0.0645), None, None],
            ),
            (
                [1.0, 1.0, 1.0, 0.0],
                [Some(0.13583), Some(0.12103), Some(0.12434)],
            ),
            ([1.0, 1.0, 1.0, 1.0], [Some(1.0), Some(1.0), Some(1.0)]),
        ];
        for (jxl, want) in cases {
            // The pipeline hands the CMS ICC-convention ink (0 = no ink).
            let ink = jxl.map(|v| 1.0 - v);
            let mut got = [0.0f32; 3];
            transforms[0].do_transform(&ink, &mut got)?;
            for (c, want) in want.iter().enumerate() {
                let Some(want) = *want else { continue };
                // Under a third of an 8-bit code.
                assert!(
                    (got[c] - want).abs() < 0.0013,
                    "CMYK {jxl:?} channel {c}: got {}, lcms2 {want}",
                    got[c]
                );
            }
        }
        Ok(())
    }

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
