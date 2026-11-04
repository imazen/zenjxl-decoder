//! Dump RGBA bytes using the CLI's LCMS backend for an exact oracle.
use std::{error::Error, path::PathBuf};
use zenjxl_decoder_cli::cms::Lcms2Cms;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let input = PathBuf::from(
        arguments
            .next()
            .ok_or("expected input.jxl and fresh output directory")?,
    );
    let output = PathBuf::from(arguments.next().ok_or("expected fresh output directory")?);
    if arguments.next().is_some() {
        return Err("expected exactly two arguments".into());
    }
    std::fs::create_dir(&output)?;
    let limits = jxl::api::JxlDecoderLimits::default()
        .with_max_codestream_level(jxl::api::JxlCodestreamLevel::Level10);
    let mut options = jxl::api::JxlDecoderOptions::default().with_limits(limits);
    options.cms = Some(Box::new(Lcms2Cms));
    let image = jxl::api::decode_with(&std::fs::read(input)?, options)?;
    std::fs::write(output.join("pixels.rgba8"), &image.data)?;
    std::fs::write(
        output.join("dimensions.txt"),
        format!("{} {} {}\n", image.width, image.height, image.channels),
    )?;
    if let Some(icc) = image.output_profile.try_as_icc() {
        std::fs::write(output.join("output.icc"), icc.as_slice())?;
    }
    Ok(())
}
