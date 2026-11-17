// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Reproduce upstream's HDR CMS control through the separately bundled LCMS.
//! Default prints measurements; --check requires the unchanged upstream bounds.

use jxl::api::RenderingIntent;
use jxl::api::{
    JxlCms, JxlColorEncoding, JxlColorProfile, JxlPrimaries, JxlTransferFunction, JxlWhitePoint,
};
use zenjxl_decoder_cli::cms::Lcms2Cms;

fn probe(target: f32) -> Result<[f32; 9], Box<dyn std::error::Error>> {
    let profile = |transfer_function| {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::BT2100,
            transfer_function,
            rendering_intent: RenderingIntent::Relative,
        })
    };
    let (_, mut transforms) = Lcms2Cms.initialize_transforms(
        1,
        1024,
        profile(JxlTransferFunction::PQ),
        profile(JxlTransferFunction::Linear),
        target,
    )?;
    let pq_1000 = 0.7518271f32;
    let input = [
        pq_1000,
        pq_1000,
        pq_1000,
        0.0,
        0.0,
        151.0 / 255.0,
        0.0,
        0.0,
        pq_1000,
    ];
    let mut output = [0.0; 9];
    transforms[0].do_transform(&input, &mut output)?;
    Ok(output)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let check = match arguments.as_slice() {
        [] => false,
        [argument] if argument == "--check" => true,
        _ => return Err("usage: hdr_intensity_probe [--check]".into()),
    };
    let actual = probe(1000.0)?;
    println!("target=1000: {actual:?}");
    println!("target=10000: {:?}", probe(10000.0)?);
    if check {
        let expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.9072, 0.0697, 0.0697, 1.0];
        for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            if (actual - expected).abs() >= 0.01 {
                return Err(format!(
                    "upstream HDR bound: sample {index} = {actual}, expected {expected} within 0.01"
                )
                .into());
            }
        }
    }
    Ok(())
}
