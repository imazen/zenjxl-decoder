// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Native v0.12.0 a7a9c78 controls: empty segment versus zero-symbol table.
#![cfg(feature = "jpeg")]

fn bytes(text: &str) -> Vec<u8> {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(compact.len() % 2, 0);
    compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn native_empty_dht_and_zero_symbol_table_reconstruct_exactly() {
    let empty = bytes(include_str!("testdata/jpeg_dht_v012_empty.jxl.hex"));
    let zero = bytes(include_str!("testdata/jpeg_dht_v012_zero_dc3.jxl.hex"));
    // The metadata-only variation must leave the complete rendered frame alone.
    let rendered_empty = zenjxl_decoder::decode(&empty).unwrap();
    let rendered_zero = zenjxl_decoder::decode(&zero).unwrap();
    assert_eq!((rendered_empty.width, rendered_empty.height), (67, 53));
    assert_eq!(rendered_empty.data, rendered_zero.data);
    for (label, stream, original) in [
        (
            "zero-symbol DC3",
            zero,
            include_str!("testdata/jpeg_dht_v012_zero_dc3.jpg.hex"),
        ),
        (
            "zero-symbol AC3",
            bytes(include_str!("testdata/jpeg_dht_v012_zero_ac3.jxl.hex")),
            include_str!("testdata/jpeg_dht_v012_zero_ac3.jpg.hex"),
        ),
        (
            "empty DHT",
            empty,
            include_str!("testdata/jpeg_dht_v012_empty.jpg.hex"),
        ),
    ] {
        let rendered = zenjxl_decoder::decode(&stream).unwrap();
        assert_eq!(
            rendered.data, rendered_empty.data,
            "{label}: complete rendered frame"
        );
        for dither in [false, true] {
            let mut options = zenjxl_decoder::api::JxlDecoderOptions::default();
            options.dither_u8 = dither;
            let actual = zenjxl_decoder::reconstruct_jpeg_with(&stream, options)
                .unwrap_or_else(|error| panic!("{label}, dither={dither}: {error}"))
                .unwrap_or_else(|| panic!("{label}: native control retains JBRD"));
            assert!(
                actual == bytes(original),
                "{label}, dither={dither}: original JPEG bytes"
            );
        }
    }
}
