// Copyright (c) Imazen LLC.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Regression tests for XYB-encoded CMYK images (ISO/IEC 18181-1 Clauses E.1,
//! L.2.2, L.4) and `decode_with` convenience handling of CMYK images without a
//! CMS.
//!
//! Fixtures are 24×16 six-swatch CMYK images (3 columns × 2 rows of 8×8 blocks):
//!   top row:    White [0,0,0,0], Cyan [1,0,0,0], Magenta [0,1,0,0]
//!   bottom row: Yellow [0,0,1,0], Black [0,0,0,1], Mix [0.75,0.5,0.25,0.125]
//! in ink-coverage convention (JXL stores `1 - ink`).

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use zenjxl_decoder::api::{
    JxlCms, JxlCmsTransformer, JxlCodestreamLevel, JxlColorProfile, JxlColorType, JxlDataFormat,
    JxlDecoder, JxlDecoderLimits, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult,
};

/// 24×16 lossless (non-XYB) Level 10 CMYK image with embedded CMYK ICC profile.
const LOSSLESS_CMYK_SWATCHES_JXL: &[u8] = &[
    0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00, 0x14,
    0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78, 0x6c, 0x20,
    0x00, 0x00, 0x00, 0x09, 0x6a, 0x78, 0x6c, 0x6c, 0x0a, 0x00, 0x00, 0x01, 0x27, 0x6a, 0x78, 0x6c,
    0x63, 0xff, 0x0a, 0x03, 0xc1, 0x28, 0x00, 0xc2, 0xed, 0xc3, 0x00, 0x14, 0x0a, 0xa5, 0xc0, 0xc0,
    0x40, 0xe0, 0x48, 0x49, 0x90, 0x98, 0x07, 0x26, 0x01, 0x08, 0x67, 0xb2, 0x4b, 0xda, 0x6d, 0xe0,
    0xc7, 0x3a, 0x58, 0x09, 0xd5, 0x26, 0xbe, 0x5f, 0x4f, 0x09, 0x95, 0x41, 0x9a, 0x5b, 0x50, 0x18,
    0x0e, 0x01, 0xa5, 0x6b, 0x3f, 0x63, 0xda, 0xa9, 0x4d, 0x54, 0xcf, 0x77, 0x87, 0x0e, 0xe4, 0x2d,
    0x4e, 0x26, 0xd5, 0x8e, 0xed, 0x13, 0x26, 0x14, 0x8b, 0xc0, 0x4a, 0x4b, 0x06, 0x37, 0xe5, 0xc2,
    0x43, 0x26, 0x14, 0x00, 0x80, 0xf0, 0x0d, 0x28, 0x15, 0x22, 0xee, 0xaf, 0xaf, 0x01, 0x1c, 0xa1,
    0x10, 0x84, 0xdb, 0xc3, 0x22, 0x65, 0x92, 0xf8, 0x95, 0x0c, 0x70, 0xd2, 0x43, 0xd2, 0x06, 0x98,
    0x0a, 0xf4, 0xb8, 0x10, 0x93, 0x85, 0x60, 0x4c, 0xa4, 0x8d, 0x0a, 0xd4, 0x54, 0x77, 0xd6, 0xea,
    0x6e, 0xb0, 0xa4, 0x09, 0x2b, 0x85, 0x07, 0x68, 0x12, 0x27, 0x69, 0x02, 0x0e, 0x78, 0x36, 0xc7,
    0x07, 0x0c, 0x16, 0x1e, 0x23, 0x61, 0x4a, 0x10, 0xd6, 0xad, 0x04, 0xc0, 0xe7, 0x57, 0x1a, 0x33,
    0xa5, 0x00, 0x6a, 0x51, 0x16, 0xbf, 0xd7, 0x7f, 0x9e, 0x8a, 0x42, 0xfc, 0xf8, 0x3b, 0xc5, 0x0b,
    0x4f, 0xa0, 0x91, 0x3f, 0x50, 0x62, 0xe4, 0xb3, 0x71, 0xf8, 0x4e, 0x93, 0x30, 0x0d, 0x05, 0x47,
    0xbb, 0x73, 0x24, 0xfa, 0x3d, 0xbe, 0x78, 0x0f, 0x4f, 0x4a, 0xd8, 0x4c, 0xcd, 0xcc, 0xc2, 0x2d,
    0xac, 0x44, 0x8f, 0x69, 0xef, 0x1a, 0x17, 0xae, 0x44, 0xb3, 0xe1, 0x74, 0xd1, 0x8b, 0x6a, 0x99,
    0x95, 0xb1, 0x08, 0x08, 0x10, 0x00, 0xe0, 0x00, 0xcb, 0x2f, 0x15, 0x2a, 0x08, 0xa8, 0x91, 0x39,
    0x13, 0x63, 0x46, 0x5d, 0x20, 0x0a, 0x00, 0x2e, 0x99, 0x00, 0x80, 0x21, 0x39, 0xff, 0x3e, 0x2c,
    0x81, 0xdf, 0xff, 0x4f, 0xb2, 0x1f, 0xc0, 0x8f, 0x38, 0xd0, 0xbf, 0x0f, 0x12, 0xea, 0xf7, 0x2f,
    0xb1, 0xc3, 0x9c, 0xd6, 0x51, 0x08, 0xec, 0x6b, 0xbc, 0x9a, 0xf8, 0xf7, 0xd3, 0x17, 0x01, 0x03,
];

/// 24×16 lossy XYB-encoded Level 10 CMYK image with Black extra channel and
/// embedded CMYK ICC profile. Reference `djxl` (libjxl v0.12) swatch centers:
///   White [255,255,255], Cyan [5,255,255], Magenta [255,0,255],
///   Yellow [255,255,0], Black [0,0,0], Mix [129,188,225].
const XYB_CMYK_SWATCHES_JXL: &[u8] = &[
    0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00, 0x14,
    0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78, 0x6c, 0x20,
    0x00, 0x00, 0x00, 0x09, 0x6a, 0x78, 0x6c, 0x6c, 0x0a, 0x00, 0x00, 0x01, 0x45, 0x6a, 0x78, 0x6c,
    0x63, 0xff, 0x0a, 0x03, 0xc1, 0x28, 0x80, 0xc2, 0xed, 0xc3, 0x00, 0x14, 0x0a, 0xa5, 0xc0, 0xc0,
    0x40, 0xe0, 0x48, 0x49, 0x90, 0x98, 0x07, 0x26, 0x01, 0x08, 0x67, 0xb2, 0x4b, 0xda, 0x6d, 0xe0,
    0xc7, 0x3a, 0x58, 0x09, 0xd5, 0x26, 0xbe, 0x5f, 0x4f, 0x09, 0x95, 0x41, 0x9a, 0x5b, 0x50, 0x18,
    0x0e, 0x01, 0xa5, 0x6b, 0x3f, 0x63, 0xda, 0xa9, 0x4d, 0x54, 0xcf, 0x77, 0x87, 0x0e, 0xe4, 0x2d,
    0x4e, 0x26, 0xd5, 0x8e, 0xed, 0x13, 0x26, 0x14, 0x8b, 0xc0, 0x4a, 0x4b, 0x06, 0x37, 0xe5, 0xc2,
    0x43, 0x26, 0x14, 0x00, 0x80, 0xf0, 0x0d, 0x28, 0x15, 0x22, 0xee, 0xaf, 0xaf, 0x01, 0x1c, 0xa1,
    0x10, 0x84, 0xdb, 0xc3, 0x22, 0x65, 0x92, 0xf8, 0x95, 0x0c, 0x70, 0xd2, 0x43, 0xd2, 0x06, 0x98,
    0x0a, 0xf4, 0xb8, 0x10, 0x93, 0x85, 0x60, 0x4c, 0xa4, 0x8d, 0x0a, 0xd4, 0x54, 0x77, 0xd6, 0xea,
    0x6e, 0xb0, 0xa4, 0x09, 0x2b, 0x85, 0x07, 0x68, 0x12, 0x27, 0x69, 0x02, 0x0e, 0x78, 0x36, 0xc7,
    0x07, 0x0c, 0x16, 0x1e, 0x23, 0x61, 0x4a, 0x10, 0xd6, 0xad, 0x04, 0xc0, 0xe7, 0x57, 0x1a, 0x33,
    0xa5, 0x00, 0x6a, 0x51, 0x16, 0xbf, 0xd7, 0x7f, 0x9e, 0x8a, 0x42, 0xfc, 0xf8, 0x3b, 0xc5, 0x0b,
    0x4f, 0xa0, 0x91, 0x3f, 0x50, 0x62, 0xe4, 0xb3, 0x71, 0xf8, 0x4e, 0x93, 0x30, 0x0d, 0x05, 0x47,
    0xbb, 0x73, 0x24, 0xfa, 0x3d, 0xbe, 0x78, 0x0f, 0x4f, 0x4a, 0xd8, 0x4c, 0xcd, 0xcc, 0xc2, 0x2d,
    0xac, 0x44, 0x8f, 0x69, 0xef, 0x1a, 0x17, 0xae, 0x44, 0xb3, 0xe1, 0x74, 0xd1, 0x8b, 0x6a, 0x99,
    0x95, 0xb1, 0xe0, 0x1b, 0x4c, 0x80, 0x10, 0x00, 0x50, 0x01, 0x67, 0x13, 0x28, 0x01, 0x00, 0x40,
    0xa1, 0xb1, 0x4f, 0x6e, 0x5b, 0x55, 0x55, 0xff, 0x0f, 0xe0, 0xff, 0x01, 0x2c, 0x61, 0x6c, 0xec,
    0x89, 0xc1, 0x85, 0x84, 0x43, 0xc4, 0xdb, 0x2a, 0xc5, 0x21, 0xf2, 0x4b, 0x85, 0x8a, 0x28, 0x24,
    0x3c, 0x74, 0x03, 0x24, 0x04, 0xa0, 0xcf, 0xdf, 0x7f, 0xab, 0x1f, 0x10, 0xb0, 0x9f, 0xd6, 0x3c,
    0x46, 0xb2, 0xb0, 0x07, 0x3a, 0xf9, 0x16, 0x9c, 0x8f, 0x92, 0xb3, 0x80, 0x47, 0x32, 0xa1, 0x2d,
    0xa4, 0x18, 0x96, 0x8c, 0x34, 0x30, 0xd7, 0x0e, 0x40, 0x09, 0x14, 0x40, 0x25, 0x09,
];

fn level10_options() -> JxlDecoderOptions {
    JxlDecoderOptions::default()
        .with_dither_u8(false)
        .with_limits(
            JxlDecoderLimits::default().with_max_codestream_level(JxlCodestreamLevel::Level10),
        )
}

fn swatch_centers_rgb(data: &[u8], width: usize, stride_ch: usize) -> [[u8; 3]; 6] {
    let coords = [(4, 4), (12, 4), (20, 4), (4, 12), (12, 12), (20, 12)];
    coords.map(|(x, y)| {
        let o = (y * width + x) * stride_ch;
        [data[o], data[o + 1], data[o + 2]]
    })
}

fn assert_swatches_near(got: [[u8; 3]; 6], want: [[u8; 3]; 6], tol: u8, label: &str) {
    let names = ["White", "Cyan", "Magenta", "Yellow", "Black", "Mix"];
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for c in 0..3 {
            assert!(
                g[c].abs_diff(w[c]) <= tol,
                "{label} swatch {} ({}) ch {c}: got {g:?}, expected {w:?} (tol={tol})",
                i,
                names[i]
            );
        }
    }
}

fn decode_low_level_u8(
    jxl: &[u8],
    options: JxlDecoderOptions,
    color_type: JxlColorType,
) -> (usize, usize, Vec<u8>) {
    let mut input = jxl;
    let decoder = match JxlDecoder::new(options).process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("incomplete header"),
    };
    let (w, h) = decoder.basic_info().size;
    let n_ec = decoder.basic_info().extra_channels.len();
    let mut decoder = decoder;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; n_ec],
    });
    let frame = match decoder.process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("incomplete frame header"),
    };
    let ch = color_type.samples_per_pixel();
    let mut out = vec![0u8; w * h * ch];
    let mut bufs = [JxlOutputBuffer::new(&mut out, h, w * ch)];
    match frame.process(&mut input, &mut bufs).unwrap() {
        ProcessingResult::Complete { .. } => {}
        ProcessingResult::NeedsMoreInput { .. } => panic!("incomplete frame"),
    }
    (w, h, out)
}

fn decode_low_level_f32(
    jxl: &[u8],
    options: JxlDecoderOptions,
    color_type: JxlColorType,
) -> (usize, usize, Vec<f32>) {
    let mut input = jxl;
    let decoder = match JxlDecoder::new(options).process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("incomplete header"),
    };
    let (w, h) = decoder.basic_info().size;
    let n_ec = decoder.basic_info().extra_channels.len();
    let mut decoder = decoder;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![None; n_ec],
    });
    let frame = match decoder.process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("incomplete frame header"),
    };
    let ch = color_type.samples_per_pixel();
    let mut out = vec![0.0f32; w * h * ch];
    let mut bufs = [JxlOutputBuffer::new(
        bytemuck::cast_slice_mut(&mut out),
        h,
        w * ch * 4,
    )];
    match frame.process(&mut input, &mut bufs).unwrap() {
        ProcessingResult::Complete { .. } => {}
        ProcessingResult::NeedsMoreInput { .. } => panic!("incomplete frame"),
    }
    (w, h, out)
}

const DJXL_XYB_CMYK_SWATCHES: [[u8; 3]; 6] = [
    [255, 255, 255],
    [5, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [0, 0, 0],
    [129, 188, 225],
];

/// XYB-encoded CMYK images decode to sRGB via inverse XYB; the Black extra
/// channel must not be folded into the already-sRGB channels via
/// `BlackChannelStage` when no CMS is configured.
#[test]
fn xyb_cmyk_without_cms_does_not_apply_black_channel() {
    // Low-level API (RGB8, Black unrequested — previously hit BlackChannelStage
    // and turned Mix [129, 188, 225] into [113, 165, 198]).
    let (w, _, rgb) =
        decode_low_level_u8(XYB_CMYK_SWATCHES_JXL, level10_options(), JxlColorType::Rgb);
    assert_swatches_near(
        swatch_centers_rgb(&rgb, w, 3),
        DJXL_XYB_CMYK_SWATCHES,
        2,
        "xyb low-level no-cms",
    );

    // High-level convenience API.
    let img = zenjxl_decoder::decode_with(XYB_CMYK_SWATCHES_JXL, level10_options()).unwrap();
    assert_swatches_near(
        swatch_centers_rgb(&img.data, img.width, 4),
        DJXL_XYB_CMYK_SWATCHES,
        2,
        "xyb convenience no-cms",
    );
}

/// XYB-encoded CMYK images decoded with a CMS must not run a second CMYK->sRGB
/// ICC conversion on top of the XYB->sRGB output.
#[cfg(feature = "cms")]
#[test]
fn xyb_cmyk_with_moxcms_does_not_double_convert() {
    use zenjxl_decoder::api::MoxCms;

    // Previously turned Cyan [6, 255, 255] into [41, 255, 255] and Mix
    // [129, 188, 225] into [178, 210, 228].
    let (w, _, rgb) = decode_low_level_u8(
        XYB_CMYK_SWATCHES_JXL,
        level10_options().with_cms(Box::new(MoxCms)),
        JxlColorType::Rgb,
    );
    assert_swatches_near(
        swatch_centers_rgb(&rgb, w, 3),
        DJXL_XYB_CMYK_SWATCHES,
        2,
        "xyb low-level MoxCms",
    );

    let img = zenjxl_decoder::decode_with(
        XYB_CMYK_SWATCHES_JXL,
        level10_options().with_cms(Box::new(MoxCms)),
    )
    .unwrap();
    assert_swatches_near(
        swatch_centers_rgb(&img.data, img.width, 4),
        DJXL_XYB_CMYK_SWATCHES,
        2,
        "xyb convenience MoxCms",
    );
}

/// Spy CMS confirming that XYB-encoded CMYK images only ask the CMS for an
/// RGB (`linear_srgb` -> `srgb`) transform, never a 4-channel CMYK transform.
#[test]
fn xyb_cmyk_cms_receives_only_rgb_input() {
    struct SpyCms {
        in_channels: Arc<Mutex<Vec<usize>>>,
        calls: Arc<AtomicUsize>,
    }
    struct IdentityRgb;
    impl JxlCms for SpyCms {
        fn initialize_transforms(
            &self,
            n: usize,
            _max_pixels: usize,
            input: JxlColorProfile,
            output: JxlColorProfile,
            _intensity_target: f32,
        ) -> zenjxl_decoder::api::Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send + Sync>>)>
        {
            self.in_channels.lock().unwrap().push(input.channels());
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok((
                output.channels(),
                (0..n)
                    .map(|_| Box::new(IdentityRgb) as Box<dyn JxlCmsTransformer + Send + Sync>)
                    .collect(),
            ))
        }
    }
    impl JxlCmsTransformer for IdentityRgb {
        fn do_transform(
            &mut self,
            input: &[f32],
            output: &mut [f32],
        ) -> zenjxl_decoder::api::Result<()> {
            output.copy_from_slice(input);
            Ok(())
        }
        fn do_transform_inplace(&mut self, _inout: &mut [f32]) -> zenjxl_decoder::api::Result<()> {
            Ok(())
        }
    }

    let in_channels = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let opts = level10_options().with_cms(Box::new(SpyCms {
        in_channels: Arc::clone(&in_channels),
        calls: Arc::clone(&calls),
    }));
    let _ = decode_low_level_u8(XYB_CMYK_SWATCHES_JXL, opts, JxlColorType::Rgb);

    let recorded = in_channels.lock().unwrap();
    assert_eq!(
        *recorded,
        vec![3],
        "XYB CMYK must only initialize a 3-channel RGB CMS stage, never 4-channel CMYK; got {recorded:?}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

/// XYB-encoded CMYK decoded to f32 (default output profile: linear sRGB) must
/// emit linear sRGB directly without multiplying by K or applying CMYK ICC.
#[test]
fn xyb_cmyk_f32_linear_srgb_skips_cmyk_stages() {
    let (w, _, rgb_f32) =
        decode_low_level_f32(XYB_CMYK_SWATCHES_JXL, level10_options(), JxlColorType::Rgb);
    let mix_off = (12 * w + 20) * 3;
    let mix = [rgb_f32[mix_off], rgb_f32[mix_off + 1], rgb_f32[mix_off + 2]];
    // sRGB [129, 188, 225] in linear light is approximately [0.2195, 0.5029, 0.7529].
    let expected = [0.2195f32, 0.5029, 0.7529];
    for c in 0..3 {
        assert!(
            (mix[c] - expected[c]).abs() < 0.015,
            "xyb f32 mix ch {c}: got {mix:?}, expected ~{expected:?}"
        );
    }
}

/// Lossless (non-XYB) CMYK decoded via `decode_with` without a CMS must fold K
/// into CMY via `BlackChannelStage` (pure black ink -> [0, 0, 0], not white).
#[test]
fn lossless_cmyk_convenience_without_cms_folds_black() {
    let img = zenjxl_decoder::decode_with(LOSSLESS_CMYK_SWATCHES_JXL, level10_options()).unwrap();
    // Mix swatch is JXL CMY=[64/255, 128/255, 192/255] * K=(224/255) = [56, 112, 169].
    let want = [
        [255, 255, 255],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 0, 0],
        [56, 112, 169],
    ];
    assert_swatches_near(
        swatch_centers_rgb(&img.data, img.width, 4),
        want,
        1,
        "lossless convenience no-cms",
    );
}

/// Lossless (non-XYB) CMYK decoded via `decode_with` with a CMS still converts
/// through the embedded CMYK ICC profile.
#[cfg(feature = "cms")]
#[test]
fn lossless_cmyk_convenience_with_moxcms_applies_icc() {
    use zenjxl_decoder::api::MoxCms;

    let img = zenjxl_decoder::decode_with(
        LOSSLESS_CMYK_SWATCHES_JXL,
        level10_options().with_cms(Box::new(MoxCms)),
    )
    .unwrap();
    let want = [
        [255, 255, 255],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 0, 0],
        [129, 177, 212],
    ];
    assert_swatches_near(
        swatch_centers_rgb(&img.data, img.width, 4),
        want,
        2,
        "lossless convenience MoxCms",
    );
}

/// Requesting `JxlColorType::Cmyk` on a lossless CMYK image still returns the
/// raw native ink channels without folding K into CMY.
#[test]
fn lossless_cmyk_raw_output_preserves_native_inks() {
    let (w, _, cmyk) = decode_low_level_u8(
        LOSSLESS_CMYK_SWATCHES_JXL,
        level10_options(),
        JxlColorType::Cmyk,
    );
    // Black swatch center (12, 12): native JXL [255, 255, 255, 0] (0 = full K ink).
    let black_off = (12 * w + 12) * 4;
    assert_eq!(&cmyk[black_off..black_off + 4], &[255, 255, 255, 0]);
    // Mix swatch center (20, 12): native JXL [64, 128, 192, 224].
    let mix_off = (12 * w + 20) * 4;
    assert_eq!(&cmyk[mix_off..mix_off + 4], &[64, 128, 192, 224]);
}
