// Copyright (c) Imazen LLC. BSD-3-Clause.
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
