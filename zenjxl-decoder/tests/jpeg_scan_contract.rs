// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Unmodified native producer controls for exact scan-entropy reconstruction.
#![cfg(feature = "jpeg")]
fn bytes(text: &str) -> Vec<u8> {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(compact.len() % 2, 0);
    compact
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn native_baseline_and_progressive_scan_signals_reconstruct_exactly() {
    let rows = [
        (
            "000",
            include_str!("testdata/jpeg_scan_v012_000.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_000.jpg.hex"),
        ),
        (
            "001",
            include_str!("testdata/jpeg_scan_v012_001.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_001.jpg.hex"),
        ),
        (
            "010",
            include_str!("testdata/jpeg_scan_v012_010.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_010.jpg.hex"),
        ),
        (
            "011",
            include_str!("testdata/jpeg_scan_v012_011.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_011.jpg.hex"),
        ),
        (
            "100",
            include_str!("testdata/jpeg_scan_v012_100.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_100.jpg.hex"),
        ),
        (
            "101",
            include_str!("testdata/jpeg_scan_v012_101.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_101.jpg.hex"),
        ),
        (
            "110",
            include_str!("testdata/jpeg_scan_v012_110.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_110.jpg.hex"),
        ),
        (
            "111",
            include_str!("testdata/jpeg_scan_v012_111.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_111.jpg.hex"),
        ),
    ];
    let mut failures = Vec::new();
    let mut expected = None;
    for (label, stream, original) in rows {
        let input = bytes(stream);
        let original = bytes(original);
        let image = zenjxl_decoder::decode(&input).unwrap();
        assert_eq!((image.width, image.height), (16, 8));
        if let Some(ref previous) = expected {
            assert_eq!(&image.data, previous);
        } else {
            expected = Some(image.data);
        }
        match zenjxl_decoder::reconstruct_jpeg(&input) {
            Ok(Some(output)) if output == original => {}
            other => failures.push(format!("{label}: {other:?}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn native_nonzero_restart_edge_and_long_code_tokens_reconstruct_exactly() {
    let rows = [
        (
            "nonzero-before-zrl",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_nonzero-before-zrl.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_nonzero-before-zrl.jpg.hex"),
        ),
        (
            "zrl-before-nonzero",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_zrl-before-nonzero.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_zrl-before-nonzero.jpg.hex"),
        ),
        (
            "three-zrl",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_three-zrl.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_three-zrl.jpg.hex"),
        ),
        (
            "zrl-finishes-block",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_zrl-finishes-block.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_zrl-finishes-block.jpg.hex"),
        ),
        (
            "long-codes-nonzero-before",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_long-codes-nonzero-before.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_long-codes-nonzero-before.jpg.hex"),
        ),
        (
            "long-codes-zrl-before-nonzero",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_long-codes-zrl-before-nonzero.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_long-codes-zrl-before-nonzero.jpg.hex"),
        ),
        (
            "gray-absolute22",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_gray-absolute22.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_gray-absolute22.jpg.hex"),
        ),
        (
            "interleaved-last",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_interleaved-last.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_interleaved-last.jpg.hex"),
        ),
        (
            "interleaved-first-long",
            16,
            8,
            include_str!("testdata/jpeg_scan_v012_interleaved-first-long.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_interleaved-first-long.jpg.hex"),
        ),
        (
            "edge-padding-luma",
            17,
            9,
            include_str!("testdata/jpeg_scan_v012_edge-padding-luma.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_edge-padding-luma.jpg.hex"),
        ),
        (
            "edge-visible-chroma-long",
            17,
            9,
            include_str!("testdata/jpeg_scan_v012_edge-visible-chroma-long.jxl.hex"),
            include_str!("testdata/jpeg_scan_v012_edge-visible-chroma-long.jpg.hex"),
        ),
    ];
    let mut failures = Vec::new();
    for (label, width, height, stream, original) in rows {
        let stream = bytes(stream);
        let original = bytes(original);
        let image = zenjxl_decoder::decode(&stream).unwrap();
        assert_eq!((image.width, image.height), (width, height));
        let pixel_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/testdata")
            .join(format!("jpeg_scan_v012_{label}.rgb16.txt"));
        let expected = std::fs::read_to_string(pixel_path).unwrap();
        assert_native_rgb16(&stream, width, height, label, &expected);
        match zenjxl_decoder::reconstruct_jpeg(&stream) {
            Ok(Some(output)) if output == original => {}
            other => failures.push(format!("{label}: {other:?}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn assert_native_rgb16(stream: &[u8], width: usize, height: usize, label: &str, expected: &str) {
    use zenjxl_decoder::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult,
    };
    let mut input = stream;
    let ProcessingResult::Complete { mut result } = JxlDecoder::new(JxlDecoderOptions::default())
        .process(&mut input)
        .unwrap()
    else {
        panic!("incomplete image header")
    };
    result.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U16 {
            endianness: zenjxl_decoder::api::Endianness::native(),
            bit_depth: 16,
        }),
        extra_channel_format: vec![],
    });
    let ProcessingResult::Complete { result } = result.process(&mut input).unwrap() else {
        panic!("incomplete frame header")
    };
    let stride = width * 3 * 2;
    let mut pixels = vec![0; stride * height];
    let output = JxlOutputBuffer::new(&mut pixels, height, stride);
    let ProcessingResult::Complete { .. } = result.process(&mut input, &mut [output]).unwrap()
    else {
        panic!("incomplete frame pixels")
    };
    let mut values = expected
        .split_whitespace()
        .map(|v| v.parse::<u16>().unwrap());
    assert_eq!(values.next(), Some(width as u16));
    assert_eq!(values.next(), Some(height as u16));
    let expected: Vec<_> = values.collect();
    assert_eq!(expected.len(), width * height * 3);
    let actual: Vec<_> = pixels
        .chunks_exact(2)
        .map(|v| u16::from_ne_bytes(v.try_into().unwrap()))
        .collect();
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    println!("{label}: native PNG16 maximum sample difference {max_error}");
    assert_eq!(actual, expected, "{label}: complete native PNG16 samples");
}
