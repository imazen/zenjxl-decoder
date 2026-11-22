// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Compare complete RGB8 and RGBA8 still decoding with reusable output.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::path::{Path, PathBuf};
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
    /// Compare these streams with the corresponding inputs using reused output.
    #[arg(long, num_args = 1..)]
    candidate_inputs: Option<Vec<PathBuf>>,
    /// Require identical pixels across alternate streams, in addition to reuse parity.
    #[arg(long, requires = "candidate_inputs")]
    exact_pairs: bool,
}

struct Arm<'a> {
    name: &'static str,
    input: &'a [u8],
    reuse_output: bool,
    rewind: bool,
    decoder: Option<JxlDecoder<Initialized>>,
    output: Option<OwnedRawImage>,
    shape: Option<(usize, usize, usize)>,
}

impl<'a> Arm<'a> {
    fn new(name: &'static str, input: &'a [u8], reuse_output: bool, rewind: bool) -> Self {
        Self {
            name,
            input,
            reuse_output,
            rewind,
            decoder: None,
            output: None,
            shape: None,
        }
    }

    fn decode(&mut self, parallel: bool, allow_level10: bool) -> Result<Duration> {
        let mut input = self.input;
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
        self.shape = Some((info.size.0, info.size.1, samples));
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

fn verify_output(expected: &Arm<'_>, arm: &Arm<'_>) -> Result<()> {
    ensure!(expected.shape == arm.shape, "{} output shape", arm.name);
    let expected = expected.output.as_ref().unwrap();
    let pixels = arm.output.as_ref().unwrap();
    for y in 0..expected.byte_size().1 {
        ensure!(
            expected.row(y) == pixels.row(y),
            "{} differs at row {y}",
            arm.name
        );
    }
    Ok(())
}

fn measure(
    args: &Args,
    path: &Path,
    arms: &mut [Arm<'_>],
    verify: impl Fn(&[Arm<'_>]) -> Result<()>,
) -> Result<()> {
    let parallel = args.workers > 1;
    for arm in arms.iter_mut() {
        arm.decode(parallel, args.allow_level10)?;
    }
    verify(arms)?;
    for arm in arms.iter_mut().filter(|arm| arm.reuse_output) {
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
                let index = (block + offset) % arms.len();
                let arm = &mut arms[index];
                let elapsed = arm.decode(parallel, args.allow_level10)?;
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
                verify(arms)?;
            }
        }
    }
    verify(arms)
}

fn compare_streams(args: &Args, path: &Path, base: &[u8], candidate: &[u8]) -> Result<()> {
    let mut base_reference = Arm::new("baseline reference", base, false, false);
    let mut candidate_reference = Arm::new("candidate reference", candidate, false, false);
    base_reference.decode(args.workers > 1, args.allow_level10)?;
    candidate_reference.decode(args.workers > 1, args.allow_level10)?;
    ensure!(
        base_reference.shape == candidate_reference.shape,
        "alternate streams must have the same dimensions and RGB8/RGBA8 layout"
    );
    if args.exact_pairs || base == candidate {
        verify_output(&base_reference, &candidate_reference)?;
    }
    let mut arms = [
        Arm::new("baseline", base, true, false),
        Arm::new("candidate", candidate, true, false),
        Arm::new("aa", base, true, false),
    ];
    measure(args, path, &mut arms, |arms| {
        verify_output(&base_reference, &arms[0])?;
        verify_output(&candidate_reference, &arms[1])?;
        verify_output(&base_reference, &arms[2])
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.workers >= 1 && args.blocks >= 3,
        "expected at least one worker and three blocks"
    );
    if let Some(candidates) = &args.candidate_inputs {
        ensure!(
            args.inputs.len() == candidates.len(),
            "expected one candidate input per baseline input"
        );
    }
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
    for (index, path) in args.inputs.iter().enumerate() {
        let input = std::fs::read(path)?;
        if let Some(candidates) = &args.candidate_inputs {
            compare_streams(&args, path, &input, &std::fs::read(&candidates[index])?)?;
            continue;
        }
        let mut arms = [
            Arm::new("allocate", &input, false, false),
            Arm::new("aa", &input, false, false),
            Arm::new("reuse-output", &input, true, false),
            Arm::new("reuse-and-rewind", &input, true, true),
        ];
        measure(&args, path, &mut arms, |arms| {
            for arm in &arms[1..] {
                verify_output(&arms[0], arm)?;
            }
            Ok(())
        })?;
    }
    Ok(())
}
