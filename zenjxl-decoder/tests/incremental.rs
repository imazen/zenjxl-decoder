// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::path::{Path, PathBuf};
use zenjxl_decoder::api::{JxlDecoderOptions, JxlIncrementalDecoder};
use zencodec::decode::{
    DynIncrementalDecoder, IncrementalDecode, IncrementalDecoderShim, PushStatus,
};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/test")
}

fn resource(name: &str) -> PathBuf {
    fixture_root().join(name)
}

fn decode_incremental_chunks(
    data: &[u8],
    chunk_size: usize,
    batch_rows: u32,
) -> (u32, u32, Vec<u8>) {
    let options = JxlDecoderOptions::default();
    let preferred = [
        zenpixels::PixelDescriptor::RGBA8_SRGB,
        zenpixels::PixelDescriptor::GRAYA8_SRGB,
    ];
    let mut decoder =
        JxlIncrementalDecoder::with_preferred(options, &preferred).with_batch_rows(batch_rows);

    let mut offset = 0;
    let mut collected_pixels = Vec::new();
    let mut total_rows = 0;

    while offset < data.len() {
        let end = (offset + chunk_size).min(data.len());
        let chunk = &data[offset..end];
        let is_eof = end == data.len();

        let outcome = decoder
            .push_chunk(chunk, is_eof, None)
            .expect("push_chunk should succeed");

        offset += outcome.bytes_consumed;

        while let Some((y, slice)) = decoder.pull_batch().expect("pull_batch should succeed") {
            assert_eq!(y, total_rows);
            collected_pixels.extend_from_slice(slice.as_strided_bytes());
            total_rows += slice.rows();
        }

        if outcome.status == PushStatus::Complete {
            break;
        }
    }

    assert!(decoder.is_complete());
    assert_eq!(total_rows, decoder.height());

    (decoder.width(), decoder.height(), collected_pixels)
}

#[test]
fn test_incremental_lossy_and_lossless_chunks() {
    let test_fixtures = [
        "3x3_srgb_lossless.jxl",
        "grayscale_patches_modular.jxl",
        "candle.jxl",
    ];

    let chunk_sizes = [1, 17, 64, 4096, 65536, usize::MAX];

    for fixture_name in test_fixtures {
        let path = resource(fixture_name);
        if !path.exists() {
            eprintln!("Skipping missing fixture: {}", fixture_name);
            continue;
        }
        let data = std::fs::read(&path).expect("failed to read fixture");

        // Baseline whole-buffer decode
        let base_image = zenjxl_decoder::decode(&data).expect("base decode failed");
        eprintln!(
            "Fixture: {}, w={}, h={}, channels={}, data_len={}",
            fixture_name, base_image.width, base_image.height, base_image.channels, base_image.data.len()
        );

        for &chunk_size in &chunk_sizes {
            let chunk_sz = if chunk_size == usize::MAX {
                data.len()
            } else {
                chunk_size
            };
            eprintln!("Testing {fixture_name} with chunk_size {chunk_sz}");
            let (w, h, pixels) = decode_incremental_chunks(&data, chunk_sz, 8);

            assert_eq!(
                w, base_image.width as u32,
                "width mismatch on {fixture_name} chunk {chunk_sz}"
            );
            assert_eq!(
                h, base_image.height as u32,
                "height mismatch on {fixture_name} chunk {chunk_sz}"
            );
            assert_eq!(
                pixels.len(),
                base_image.data.len(),
                "len mismatch on {fixture_name} chunk {chunk_sz}"
            );
            assert_eq!(
                pixels, base_image.data,
                "byte mismatch on {fixture_name} chunk {chunk_sz}"
            );
        }
    }
}

#[test]
fn test_batch_rows_sweep() {
    let path = resource("candle.jxl");
    let data = std::fs::read(&path).expect("failed to read candle.jxl");
    let base_image = zenjxl_decoder::decode(&data).expect("base decode failed");

    for batch_rows in [1, 3, 7, 16, 64, 1000] {
        let (w, h, pixels) = decode_incremental_chunks(&data, 4096, batch_rows);
        assert_eq!(w, base_image.width as u32);
        assert_eq!(h, base_image.height as u32);
        assert_eq!(pixels, base_image.data);
    }
}

#[test]
fn test_unexpected_eof() {
    let path = resource("candle.jxl");
    let data = std::fs::read(&path).expect("failed to read candle.jxl");

    // Send half the file with is_eof = true
    let half = &data[..data.len() / 2];
    let mut decoder = JxlIncrementalDecoder::new(JxlDecoderOptions::default());
    let res = decoder.push_chunk(half, true, None);
    assert!(
        res.is_err(),
        "pushing incomplete data with is_eof=true should return an error"
    );
}

#[test]
fn test_dyn_incremental_decoder() {
    let path = resource("candle.jxl");
    let data = std::fs::read(&path).expect("failed to read candle.jxl");
    let base_image = zenjxl_decoder::decode(&data).expect("base decode failed");

    let decoder = JxlIncrementalDecoder::new(JxlDecoderOptions::default()).with_batch_rows(16);
    let mut dyn_decoder: Box<dyn DynIncrementalDecoder> =
        Box::new(IncrementalDecoderShim(decoder));

    let outcome = dyn_decoder
        .push_chunk(&data, true, None)
        .expect("push chunk failed");
    assert_eq!(outcome.bytes_consumed, data.len());

    let mut collected = Vec::new();
    while let Some((_, slice)) = dyn_decoder.pull_batch().expect("pull failed") {
        collected.extend_from_slice(slice.as_strided_bytes());
    }

    assert_eq!(collected, base_image.data);
    assert!(dyn_decoder.is_complete());
}
