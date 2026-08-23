// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression tests for known decoder bugs.
//!
//! Each test pins a specific previously-broken bitstream so the bug
//! cannot silently regress. The full historical corpus lives in
//! [imazen/codec-corpus](https://github.com/imazen/codec-corpus) under
//! `jxl/conformance/` and is exercised by [`super::codec_corpus`] when
//! the corpus is reachable; this module only carries the smallest
//! reproducer for each distinct bug, in-tree, so the regression test
//! always runs without external setup.
//!
//! Add a new entry whenever a decoder bug is fixed:
//!
//! 1. Place the smallest reproducing `.jxl` (≤30 KB) in
//!    `tests/testdata/<issue-id>/`.
//! 2. Add a `#[test]` below that resolves the path with
//!    `testdata_dir().join("<issue-id>/<file>.jxl")` and asserts decode
//!    succeeds. Reference the issue in the doc comment.

#[cfg(feature = "cms")]
use crate::api::MoxCms;
use crate::api::{
    JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult, states,
};
use crate::image::{Image, Rect};

/// Path to the in-tree test-data directory, resolved from the crate manifest
/// at compile time. Avoids hard-coded absolute paths so the tests are
/// CI-portable and don't depend on any specific filesystem layout.
fn testdata_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata")
}

/// Decode a single JXL file via the public API and return its raw u8 pixel
/// buffer. Used by per-issue regression tests; we don't pixel-compare here
/// because reference PNGs would push test-data over the in-tree size budget
/// — for full pixel parity tests, the corpus in
/// [imazen/codec-corpus](https://github.com/imazen/codec-corpus) is
/// reference-paired and exercised by [`super::codec_corpus`].
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

    let mut output_image =
        Image::<u8>::new((width * channels, height)).map_err(|e| format!("alloc: {e:?}"))?;
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

    /// Issue #15: LZ77 distance-cluster after context_map padding.
    ///
    /// PR #671's context_map padding (in `Frame::decode`) appended 16 zero
    /// entries past the original `context_map.last()`, where the LZ77
    /// distance cluster lived. Subsequent `context_map.last()` reads in
    /// the AC reader returned a zero pad byte, routing LZ77 distance ANS
    /// reads through the wrong histogram and corrupting state.
    ///
    /// Fix: capture `lz_dist_cluster` in `Histograms` at decode time,
    /// before any external `resize`. Mirrors libjxl's
    /// `ANSCode::lz77.nonserialized_distance_context` (dec_ans.cc:362).
    ///
    /// Triggered by VarDCT bitstreams with `Optimal` or `Greedy` LZ77
    /// backward references — libjxl never emits these for VarDCT (only
    /// `RLE`), so libjxl's reference test corpora don't catch this.
    /// Smallest reproducer attached: `akfcrc022_e9_d3.0.jxl` (22 KB,
    /// produced by jxl-encoder at `-e 9 -d 3.0` on screen content).
    ///
    /// See:
    /// - <https://github.com/libjxl/jxl-rs/issues/765>
    /// - <https://github.com/libjxl/jxl-rs/pull/766>
    /// - <https://github.com/imazen/zenjxl-decoder/issues/15>
    /// jxl-rs #858 / libjxl conformance PR #48: a `Mul` blending frame in an
    /// image with **no** extra channels.
    ///
    /// The frame header serialises the `clamp` bit for `Mul` blending even
    /// when `num_extra_channels == 0` (libjxl `frame_header.cc`,
    /// `BlendingInfo::VisitFields`). The fork only read it when extra channels
    /// were present, so every field after it was shifted by one bit and the
    /// decode failed with `Source file truncated`.
    ///
    /// The fixture is the 32-byte codestream from
    /// <https://github.com/libjxl/conformance/pull/48>
    /// (`testcases/mul_no_extra_channels/input.jxl`): an 8x8 Modular image
    /// whose two frames multiply to a flat 0.25198 (16513/65535, as decoded
    /// by djxl 0.12 and jxl-rs 0.6).
    #[test]
    fn jxlrs_858_mul_blend_without_extra_channels() {
        let path = testdata_dir().join("jxlrs-858/mul_no_extra_channels.jxl");
        let data = std::fs::read(&path).unwrap();

        // Float pixels: the multiply must have happened (flat 0.25198, not the
        // 0.5-ish single-frame value and not garbage).
        let (_, frames) = crate::api::decoder::tests::decode(&data, usize::MAX, false, false, None)
            .unwrap_or_else(|e| panic!("decode of {} failed: {e:?}", path.display()));
        assert_eq!(frames.len(), 1, "expected exactly one visible frame");
        let image = &frames[0][0];
        assert_eq!(image.size(), (8 * 3, 8));
        let expected = 16513.0 / 65535.0;
        for y in 0..8 {
            for (x, &v) in image.row(y).iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1.5 / 65535.0,
                    "pixel ({x}, {y}) = {v}, expected {expected}"
                );
            }
        }

        // Public u8 API: 0.25198 * 255 = 64.25 -> 64, or 65 once output
        // dithering is applied. Alpha is synthesized opaque.
        let img = crate::decode(&data).unwrap();
        assert_eq!((img.width, img.height, img.channels), (8, 8, 4));
        for px in img.data.chunks_exact(4) {
            assert!(
                px[..3].iter().all(|&c| c == 64 || c == 65),
                "unexpected rgb {:?}",
                &px[..3]
            );
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn issue_15_lz77_distance_cluster_after_pad() {
        let path = testdata_dir().join("issue-15/akfcrc022_e9_d3.0.jxl");
        let (width, height, channels, pixels) = decode_jxl(&path)
            .unwrap_or_else(|e| panic!("decode of {} failed: {e}", path.display()));
        assert_eq!((width, height), (512, 512));
        assert!(
            channels == 3 || channels == 4,
            "unexpected channels: {channels}"
        );
        assert_eq!(pixels.len(), width * height * channels);
    }
}
