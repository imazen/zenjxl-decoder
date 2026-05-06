// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression corpus for known decoder bugs.
//!
//! Files in the corpus directory are spec-valid JXL bitstreams that libjxl djxl
//! accepts but zenjxl-decoder (and upstream jxl-rs) currently rejects. They were
//! found by issue #15 sweeps over jxl-encoder output. Each `<name>.jxl` may have
//! a paired `<name>.ref.png` produced by `djxl <name>.jxl <name>.ref.png` for
//! pixel-parity comparison once the decoder bug is fixed.
//!
//! The corpus lives in block storage (`/mnt/v/fuzzes/zenjxl-decoder/regression/`)
//! and is too large to commit; in CI, mount or copy it and point
//! `ZENJXL_REGRESSION_CORPUS` at the directory.

use crate::api::{
    JxlColorProfile, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, ProcessingResult, states,
};
#[cfg(feature = "cms")]
use crate::api::MoxCms;
use crate::image::{Image, Rect};

/// Resolve the regression-corpus root.
///
/// Order: `ZENJXL_REGRESSION_CORPUS` env var, then the canonical block-storage
/// path. Returns `None` when neither is reachable so CI without block storage
/// degrades to a no-op rather than a hard fail.
fn regression_corpus_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("ZENJXL_REGRESSION_CORPUS") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let canonical = std::path::PathBuf::from("/mnt/v/fuzzes/zenjxl-decoder/regression/issue-15");
    canonical.exists().then_some(canonical)
}

/// Decode a single JXL file via the public API. Returns the raw u8 pixel buffer
/// alongside (width, height, channels) so callers can pixel-compare.
fn decode_jxl(path: &std::path::Path) -> Result<(usize, usize, usize, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    let mut input = data.as_slice();

    #[cfg(feature = "cms")]
    let options = JxlDecoderOptions {
        cms: Some(Box::new(MoxCms::new())),
        ..JxlDecoderOptions::default()
    };
    #[cfg(not(feature = "cms"))]
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("unexpected EOF in header".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("header: {e:?}")),
        }
    };

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    let default_format = decoder.current_pixel_format();
    let is_grayscale = matches!(
        default_format.color_type,
        JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha
    );
    let has_alpha = basic_info.extra_channels.iter().any(|ec| {
        matches!(
            ec.ec_type,
            crate::headers::extra_channels::ExtraChannel::Alpha
        )
    });
    let (color_type, channels) = match (is_grayscale, has_alpha) {
        (true, true) => (JxlColorType::GrayscaleAlpha, 2),
        (true, false) => (JxlColorType::Grayscale, 1),
        (false, true) => (JxlColorType::Rgba, 4),
        (false, false) => (JxlColorType::Rgb, 3),
    };
    let extra_channel_format = vec![None; basic_info.extra_channels.len()];
    decoder.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format,
    });

    // Match djxl reference convention for linear-gamma PNGs.
    if let JxlColorProfile::Simple(enc) = decoder.output_color_profile().clone() {
        let _ = decoder.set_output_color_profile(JxlColorProfile::Simple(enc));
    }

    let mut decoder = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("unexpected EOF before frame".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("frame info: {e:?}")),
        }
    };

    let mut output_image = Image::<u8>::new((width * channels, height))
        .map_err(|e| format!("alloc: {e:?}"))?;
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];

    loop {
        match decoder.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return Err("unexpected EOF in frame".into());
                }
                decoder = fallback;
            }
            Err(e) => return Err(format!("frame: {e:?}")),
        }
    }

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    Ok((width, height, channels, pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_jxl_in_dir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("jxl") {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    }

    /// Decode every file in the regression corpus. Each file represents a
    /// previously-discovered decoder bug: zenjxl-decoder rejects it while
    /// libjxl djxl accepts it. As bugs are fixed, this test should turn green.
    /// New files added to the corpus pin the bug surface so it cannot regress.
    #[test]
    fn regression_corpus_decodes_clean() {
        let Some(dir) = regression_corpus_dir() else {
            eprintln!(
                "Skipping regression_corpus_decodes_clean: \
                 set ZENJXL_REGRESSION_CORPUS or mount the canonical path"
            );
            return;
        };
        let files = collect_jxl_in_dir(&dir);
        assert!(
            !files.is_empty(),
            "regression corpus at {dir:?} is empty — expected at least one .jxl"
        );

        let mut failures: Vec<(std::path::PathBuf, String)> = Vec::new();
        for f in &files {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_jxl(f)));
            match result {
                Ok(Ok(_)) => eprintln!("ok  {}", f.display()),
                Ok(Err(e)) => {
                    eprintln!("ERR {}: {e}", f.display());
                    failures.push((f.clone(), e));
                }
                Err(_) => {
                    eprintln!("PANIC {}", f.display());
                    failures.push((f.clone(), "decoder panicked".into()));
                }
            }
        }

        if !failures.is_empty() {
            eprintln!();
            eprintln!("=== regression-corpus decode failures: {} of {} ===",
                failures.len(), files.len());
            for (p, e) in &failures {
                eprintln!("  {} :: {}", p.display(), e);
            }
            panic!(
                "{} regression files still fail to decode (libjxl djxl accepts them)",
                failures.len()
            );
        }
    }

    /// Pixel-parity check: where `<name>.ref.png` exists, decoded output must
    /// match the libjxl-produced reference within the conformance threshold.
    /// Skipped per-file when the JXL still fails to decode (parent test will
    /// catch that), so this only kicks in once decode is fixed.
    #[test]
    fn regression_corpus_matches_djxl_reference() {
        use super::super::parity::{
            CONFORMANCE_THRESHOLD_U8, ReferenceImage, compare_u8_buffers, png_has_linear_gamma,
        };

        let Some(dir) = regression_corpus_dir() else {
            eprintln!(
                "Skipping regression_corpus_matches_djxl_reference: \
                 set ZENJXL_REGRESSION_CORPUS or mount the canonical path"
            );
            return;
        };
        let files = collect_jxl_in_dir(&dir);

        let mut compared = 0usize;
        let mut mismatches: Vec<(std::path::PathBuf, String)> = Vec::new();

        for jxl in &files {
            let ref_png = jxl.with_extension("ref.png");
            if !ref_png.exists() {
                continue;
            }
            let Ok((w, h, ch, actual)) = decode_jxl(jxl) else {
                // decode failure is reported by the sibling test
                continue;
            };
            let _linear = png_has_linear_gamma(&ref_png).unwrap_or(false);
            let reference = match ReferenceImage::load(&ref_png) {
                Ok(r) => r,
                Err(e) => {
                    mismatches.push((jxl.clone(), format!("ref load: {e}")));
                    continue;
                }
            };
            if w != reference.width || h != reference.height {
                mismatches.push((
                    jxl.clone(),
                    format!(
                        "dims {}x{} vs ref {}x{}",
                        w, h, reference.width, reference.height
                    ),
                ));
                continue;
            }
            // RGBA-decode vs RGB-reference: drop alpha for comparison.
            let (cmp_ch, ref_px, act_px) = if ch == reference.channels {
                (ch, reference.pixels.clone(), actual)
            } else if ch == 4 && reference.channels == 3 {
                let rgb = actual.chunks_exact(4).flat_map(|p| p[..3].to_vec()).collect();
                (3, reference.pixels.clone(), rgb)
            } else {
                mismatches.push((
                    jxl.clone(),
                    format!("ch {} vs ref {}", ch, reference.channels),
                ));
                continue;
            };
            let result = compare_u8_buffers(&ref_px, &act_px, w, h, cmp_ch, CONFORMANCE_THRESHOLD_U8);
            if !result.passed {
                mismatches.push((
                    jxl.clone(),
                    format!(
                        "max_err={} count={}/{}",
                        result.max_abs_error, result.error_count, result.total_pixels
                    ),
                ));
            }
            compared += 1;
        }

        eprintln!("compared {compared} ref-paired files");
        if !mismatches.is_empty() {
            for (p, e) in &mismatches {
                eprintln!("  MISMATCH {} :: {}", p.display(), e);
            }
            panic!("{} regression files mismatch djxl reference", mismatches.len());
        }
    }
}
