// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Compare complete RGB8 and RGBA8 still decoding with reusable output.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use color_eyre::eyre::{Result, bail, ensure};
use jxl::api::{
    ExtraChannel, JxlCodestreamLevel, JxlColorType, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, OwnedRawImage, ProcessingResult, Rect, states::Initialized,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = 3)]
    blocks: usize,
    #[arg(long)]
    allow_level10: bool,
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

struct Arm {
    name: &'static str,
    reuse_output: bool,
    rewind: bool,
    decoder: Option<JxlDecoder<Initialized>>,
    output: Option<OwnedRawImage>,
}

impl Arm {
    fn new(name: &'static str, reuse_output: bool, rewind: bool) -> Self {
        Self {
            name,
            reuse_output,
            rewind,
            decoder: None,
            output: None,
        }
    }

    fn decode(
        &mut self,
        mut input: &[u8],
        parallel: bool,
        allow_level10: bool,
    ) -> Result<Duration> {
        let start = Instant::now();
        let decoder = self.decoder.take().unwrap_or_else(|| {
            let mut options = JxlDecoderOptions::default();
            options.parallel = parallel;
            if allow_level10 {
                options.limits.max_codestream_level = JxlCodestreamLevel::Level10;
            }
            JxlDecoder::new(options)
        });
        let ProcessingResult::Complete {
            result: mut decoder,
        } = decoder.process(&mut input)?
        else {
            bail!("incomplete image header");
        };
        let info = decoder.basic_info();
        ensure!(info.animation.is_none(), "expected a still image");
        ensure!(
            matches!(
                decoder.current_pixel_format().color_type,
                JxlColorType::Rgb | JxlColorType::Rgba
            ),
            "expected RGB or RGBA input"
        );
        let num_extras = info.extra_channels.len();
        let (format, samples) = match info.extra_channels.as_slice() {
            [] => (JxlPixelFormat::rgb8(0), 3),
            [extra] if extra.ec_type == ExtraChannel::Alpha => {
                (JxlPixelFormat::rgba8(num_extras), 4)
            }
            _ => bail!("expected zero or one alpha channel"),
        };
        let size = (info.size.0 * samples, info.size.1);
        decoder.set_pixel_format(format);
        let mut output = match self.output.take() {
            Some(previous) if self.reuse_output && previous.byte_size() == size => previous,
            _ => {
                let output = OwnedRawImage::new_uninit(size)?;
                #[cfg(feature = "threads")]
                let output = {
                    let mut output = output;
                    output.prefault_parallel(parallel);
                    output
                };
                output
            }
        };
        let ProcessingResult::Complete { result: decoder } = decoder.process(&mut input)? else {
            bail!("incomplete frame header");
        };
        let rect = Rect {
            origin: (0, 0),
            size,
        };
        let mut buffers = [JxlOutputBuffer::from_image_rect_mut(
            output.get_rect_mut(rect),
        )];
        let ProcessingResult::Complete { result: decoder } =
            decoder.process(&mut input, &mut buffers)?
        else {
            bail!("incomplete frame data");
        };
        ensure!(!decoder.has_more_frames(), "expected exactly one frame");
        if self.rewind {
            self.decoder = Some(decoder.rewind());
        }
        black_box(&output);
        self.output = Some(output);
        Ok(start.elapsed())
    }
}

fn verify_outputs(arms: &[Arm; 4]) -> Result<()> {
    let base = arms[0].output.as_ref().unwrap();
    for arm in &arms[1..] {
        let pixels = arm.output.as_ref().unwrap();
        ensure!(base.byte_size() == pixels.byte_size(), "{} size", arm.name);
        for y in 0..base.byte_size().1 {
            ensure!(
                base.row(y) == pixels.row(y),
                "{} differs at row {y}",
                arm.name
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.workers >= 1 && args.blocks >= 3,
        "expected at least one worker and three blocks"
    );
    let parallel = args.workers > 1;
    #[cfg(feature = "threads")]
    if parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()?;
    }
    #[cfg(not(feature = "threads"))]
    ensure!(
        !parallel,
        "build with the threads feature for multiple workers"
    );

    println!("input\tworkers\tblock\tsample\tarm\tns");
    for path in args.inputs {
        let input = std::fs::read(&path)?;
        let mut arms = [
            Arm::new("allocate", false, false),
            Arm::new("aa", false, false),
            Arm::new("reuse-output", true, false),
            Arm::new("reuse-and-rewind", true, true),
        ];
        for arm in &mut arms {
            arm.decode(&input, parallel, args.allow_level10)?;
        }
        verify_outputs(&arms)?;
        for arm in &mut arms[2..] {
            arm.output.as_mut().unwrap().zero_fill();
        }
        for block in 0..args.blocks {
            for sample in 0..5 {
                for slot in 0..arms.len() {
                    let offset = if block % 2 == 0 {
                        slot
                    } else {
                        arms.len() - 1 - slot
                    };
                    let arm = &mut arms[(block + offset) % 4];
                    let elapsed = arm.decode(&input, parallel, args.allow_level10)?;
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        path.display(),
                        args.workers,
                        block + 1,
                        sample + 1,
                        arm.name,
                        elapsed.as_nanos()
                    );
                }
                if block == 0 && sample == 0 {
                    verify_outputs(&arms)?;
                }
            }
        }
        verify_outputs(&arms)?;
    }
    Ok(())
}
