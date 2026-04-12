// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression tests for the new `JxlOutputBuffer` stride and uninit
//! constructors ported from upstream jxl-rs `e883140`
//! ("Add a new constructor to JxlOutputBuffer").
//!
//! These tests exercise the constructors against real JXL files decoded
//! through the public `JxlDecoder` API, covering packed/padded/uninit/1x1
//! and multichannel variants.

use crate::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};

fn test_resources_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test")
}

fn load(name: &str) -> Vec<u8> {
    let path = test_resources_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Drive the decoder to `WithImageInfo` state on the given bytes. If
/// `force_rgba8` is true, explicitly set an RGBA/U8 pixel format; otherwise
/// keep whatever default the decoder picked (needed for files with extra
/// channels where the default pixel format already lists them). Memory
/// limits are disabled for correctness tests.
fn decode_to_image_info(
    data: &[u8],
    force_rgba8: bool,
) -> crate::api::JxlDecoder<states::WithImageInfo> {
    let mut options = JxlDecoderOptions::default();
    options.limits.max_memory_bytes = None;
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = data;
    let mut decoder_wii = loop {
        match decoder.process(&mut input).expect("process failed") {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    panic!("unexpected end of input before image info");
                }
                decoder = fallback;
            }
        }
    };
    if force_rgba8 {
        decoder_wii.set_pixel_format(JxlPixelFormat::rgba8(0));
    }
    decoder_wii
}

/// Bytes per pixel (primary color plane) for the decoder's currently
/// configured pixel format.
fn color_bytes_per_pixel(decoder: &crate::api::JxlDecoder<states::WithImageInfo>) -> usize {
    let pf = decoder.current_pixel_format();
    pf.color_type.samples_per_pixel() * pf.color_data_format.as_ref().unwrap().bytes_per_sample()
}

/// Decode one frame into the provided buffer slice(s), returning when the
/// decoder transitions back to `WithImageInfo`. Panics on decode error so
/// tests fail loudly.
fn decode_one_frame(
    mut decoder_wii: crate::api::JxlDecoder<states::WithImageInfo>,
    mut input: &[u8],
    buffers: &mut [JxlOutputBuffer<'_>],
) -> crate::api::JxlDecoder<states::WithImageInfo> {
    // Advance to WithFrameInfo.
    let mut decoder_wfi = loop {
        match decoder_wii
            .process(&mut input)
            .expect("process(wii) failed")
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    panic!("unexpected end of input before frame info");
                }
                decoder_wii = fallback;
            }
        }
    };
    // Render the frame into `buffers`.
    loop {
        match decoder_wfi
            .process(&mut input, buffers)
            .expect("process(wfi) failed")
        {
            ProcessingResult::Complete { result } => return result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    panic!("unexpected end of input before frame complete");
                }
                decoder_wfi = fallback;
            }
        }
    }
}

/// Per-channel storage descriptor: tight buffer + bytes-per-row.
struct ChannelStorage {
    bytes_per_row: usize,
    storage: Vec<u8>,
}

/// Allocate tight (zero-padding) per-channel buffers matching the
/// decoder's currently-selected pixel format. The first entry is the
/// color plane; subsequent entries are one per extra channel.
fn allocate_tight_channels(
    decoder_wii: &crate::api::JxlDecoder<states::WithImageInfo>,
) -> Vec<ChannelStorage> {
    let (w, h) = decoder_wii.basic_info().size;
    let pf = decoder_wii.current_pixel_format();
    let color_row = w * color_bytes_per_pixel(decoder_wii);
    let mut out = vec![ChannelStorage {
        bytes_per_row: color_row,
        storage: vec![0u8; h * color_row],
    }];
    for fmt in &pf.extra_channel_format {
        let bps = fmt.as_ref().unwrap().bytes_per_sample();
        let row = w * bps;
        out.push(ChannelStorage {
            bytes_per_row: row,
            storage: vec![0u8; h * row],
        });
    }
    out
}

/// Decode `data` into the supplied set of tight channel buffers, returning
/// the consumed buffers (so callers can inspect color-plane contents). The
/// decoder is configured at default pixel format unless `force_rgba8` is
/// requested.
fn decode_into_tight_channels(
    data: &[u8],
    force_rgba8: bool,
) -> (usize, usize, Vec<ChannelStorage>) {
    let decoder_wii = decode_to_image_info(data, force_rgba8);
    let (w, h) = decoder_wii.basic_info().size;
    let mut channels = allocate_tight_channels(&decoder_wii);
    {
        let mut buffers: Vec<JxlOutputBuffer<'_>> = channels
            .iter_mut()
            .map(|c| JxlOutputBuffer::new(&mut c.storage, h, c.bytes_per_row))
            .collect();
        let _final = decode_one_frame(decoder_wii, data, &mut buffers);
    }
    (w, h, channels)
}

/// Copy non-padding regions of a padded `buf` (with `byte_stride`) into a
/// tight packed Vec for comparison against a reference packed decode.
fn tighten(buf: &[u8], num_rows: usize, bytes_per_row: usize, byte_stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(num_rows * bytes_per_row);
    for y in 0..num_rows {
        let start = y * byte_stride;
        out.extend_from_slice(&buf[start..start + bytes_per_row]);
    }
    out
}

#[test]
fn new_with_stride_padded_matches_packed() {
    let data = load("basic.jxl");
    let (w, h, packed) = decode_into_tight_channels(&data, true);
    let bpr = packed[0].bytes_per_row;
    let reference = packed[0].storage.clone();
    assert_eq!(reference.len(), h * bpr);

    // Deliberately pad each row by an extra 37 bytes; prefilled with a
    // sentinel so we can also verify the padding bytes were left untouched.
    let byte_stride = bpr + 37;
    let mut padded = vec![0xA5u8; h * byte_stride];
    {
        let decoder_wii = decode_to_image_info(&data, true);
        // No extra channels for basic.jxl when forced to rgba8(0).
        let out = JxlOutputBuffer::new_with_stride(&mut padded, h, bpr, byte_stride);
        let _final = decode_one_frame(decoder_wii, &data, &mut [out]);
    }

    let tight = tighten(&padded, h, bpr, byte_stride);
    assert_eq!(
        tight, reference,
        "padded decode with new_with_stride diverged from packed reference for {w}x{h}",
    );

    // Padding bytes must be untouched.
    for y in 0..h {
        let pad_start = y * byte_stride + bpr;
        let pad_end = pad_start + 37;
        assert!(
            padded[pad_start..pad_end].iter().all(|&b| b == 0xA5),
            "padding bytes overwritten at row {y}",
        );
    }
}

#[test]
fn new_with_stride_1x1_edge_case() {
    // The 1x1 edge case catches stride-zero/num_rows-1 bugs in the
    // capacity calculation of `new_with_stride`. basic.jxl is not 1x1, so
    // we drive the API directly: allocate a minimum-size output buffer and
    // call `new_with_stride` with `num_rows = 1`, `byte_stride = bytes_per_row`.
    let bpr = 4; // 1 RGBA u8 pixel
    let mut buf = vec![0u8; bpr];
    let out = JxlOutputBuffer::new_with_stride(&mut buf, 1, bpr, bpr);
    let (w_bytes, rows) = out.byte_size();
    assert_eq!(w_bytes, bpr);
    assert_eq!(rows, 1);

    // And with a non-trivial stride that is still valid for a single row.
    let mut buf2 = vec![0u8; bpr];
    let out2 = JxlOutputBuffer::new_with_stride(&mut buf2, 1, bpr, 100);
    assert_eq!(out2.byte_size(), (bpr, 1));
}

#[test]
fn new_with_stride_multichannel_cmyk() {
    // `cmyk_layers.jxl` has extra channels. We let the decoder pick its
    // default pixel format and decode it twice: once into a tight color
    // buffer (the reference), once into a padded color buffer built with
    // `new_with_stride`. Extra-channel buffers stay tight in both runs.
    let data = load("conformance_test_images/cmyk_layers.jxl");
    let (_w, h, packed) = decode_into_tight_channels(&data, false);
    let bpr = packed[0].bytes_per_row;
    let reference = packed[0].storage.clone();

    // Re-decode with a padded color plane.
    let byte_stride = bpr + 128;
    let mut padded_color = vec![0u8; h * byte_stride];
    {
        let decoder_wii = decode_to_image_info(&data, false);
        // Same extra-channel layout as the reference run.
        let mut ec = allocate_tight_channels(&decoder_wii);
        // Drop the auto-allocated color plane so we can splice in our padded one.
        let _color_unused = ec.remove(0);
        let mut buffers: Vec<JxlOutputBuffer<'_>> = vec![JxlOutputBuffer::new_with_stride(
            &mut padded_color,
            h,
            bpr,
            byte_stride,
        )];
        // Push extra-channel buffers in order.
        for c in ec.iter_mut() {
            buffers.push(JxlOutputBuffer::new(&mut c.storage, h, c.bytes_per_row));
        }
        let _final = decode_one_frame(decoder_wii, &data, &mut buffers);
    }

    let tight = tighten(&padded_color, h, bpr, byte_stride);
    assert_eq!(
        tight, reference,
        "cmyk_layers padded color plane diverged from tight color plane",
    );
}

/// `new_uninit_with_stride` requires `feature = "allow-unsafe"` because the
/// safe default forbids the `MaybeUninit<u8>` -> `u8` slice reinterpret. CI
/// runs the full matrix including `--all-features`, so this test is covered
/// there. Under default features the constructor does not exist and the
/// test is not compiled.
#[cfg(feature = "allow-unsafe")]
#[test]
fn new_uninit_with_stride_matches_zeroed() {
    use std::mem::MaybeUninit;

    let data = load("basic.jxl");
    let (_w, h, packed) = decode_into_tight_channels(&data, true);
    let bpr = packed[0].bytes_per_row;
    let reference = packed[0].storage.clone();

    let byte_stride = bpr + 64;
    let total = h * byte_stride;
    // `Box::new_uninit_slice` is stable and safe; it allocates a block of
    // `MaybeUninit<u8>` without initializing.
    let mut buf: Box<[MaybeUninit<u8>]> = Box::new_uninit_slice(total);

    {
        let decoder_wii = decode_to_image_info(&data, true);
        let out = JxlOutputBuffer::new_uninit_with_stride(&mut buf, h, bpr, byte_stride);
        let _final = decode_one_frame(decoder_wii, &data, &mut [out]);
    }

    // After decode, the bytes covered by the useful row layout have all
    // been initialized by the decoder. Reading back requires one unsafe
    // reinterpret which mirrors what real callers do after decode.
    #[allow(unsafe_code)]
    let init: &[u8] = {
        // SAFETY: We called `new_uninit_with_stride(buf, h, bpr, byte_stride)`
        // and the decoder contract guarantees every byte in
        // `[y*byte_stride .. y*byte_stride+bpr]` for `y in 0..h` was
        // written. `tighten` below only reads those bytes (not padding),
        // so all bytes read are initialized.
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, total) }
    };
    let tight = tighten(init, h, bpr, byte_stride);
    assert_eq!(
        tight, reference,
        "uninit padded decode diverged from packed reference",
    );
}
