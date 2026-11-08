// Copyright (c) Imazen LLC.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Compare the existing CMYK float regression samples through both CMS adapters.
use jxl::api::{
    JxlCms, JxlColorEncoding, JxlColorProfile, JxlDecoder, JxlDecoderOptions, MoxCms,
    ProcessingResult, states,
};
use zenjxl_decoder_cli::cms::Lcms2Cms;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("expected CMYK JXL fixture")?;
    let profile_output = args.next();
    if args.next().is_some() {
        return Err("expected fixture.jxl and optional fresh profile output".into());
    }
    let data = std::fs::read(path)?;
    let mut input = data.as_slice();
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let profile = loop {
        match decoder.process(&mut input)? {
            ProcessingResult::Complete { result } => break result.embedded_color_profile().clone(),
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    if let Some(path) = profile_output {
        use std::io::Write;
        let icc = profile.try_as_icc().ok_or("fixture has no ICC profile")?;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(icc.as_slice())?;
    }
    let output = JxlColorProfile::Simple(JxlColorEncoding::srgb(false));
    let cases = [
        [0.2863, 1.0, 0.6706, 1.0],
        [0.2347, 0.9601, 0.5918, 0.9813],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    for (name, cms) in [("moxcms", &MoxCms as &dyn JxlCms), ("lcms2", &Lcms2Cms)] {
        let (channels, mut transforms) =
            cms.initialize_transforms(1, cases.len(), profile.clone(), output.clone(), 255.0)?;
        assert_eq!(channels, 3);
        for sample in cases {
            let mut rgb = [0.0; 3];
            transforms[0].do_transform(&sample, &mut rgb)?;
            println!("{name}\t{sample:?}\t{rgb:?}");
        }
    }
    Ok(())
}
