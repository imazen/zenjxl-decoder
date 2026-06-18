// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests for the convenience decode/read_header API.

#[cfg(test)]
mod tests {
    use crate::api::{decode, read_header};

    #[test]
    fn decode_basic() {
        let data = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        let image = decode(&data).unwrap();
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert_eq!(image.channels, 4); // RGBA
        assert!(!image.is_grayscale);
        assert_eq!(image.data.len(), image.width * image.height * 4);
    }

    #[test]
    fn decode_grayscale() {
        let data =
            std::fs::read(crate::util::test::fixture_path("gray_alpha_lossless.jxl")).unwrap();
        let image = decode(&data).unwrap();
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert_eq!(image.channels, 2); // GrayAlpha
        assert!(image.is_grayscale);
        assert_eq!(image.data.len(), image.width * image.height * 2);
    }

    #[test]
    fn decode_3x3_srgb_lossless() {
        let data = std::fs::read(crate::util::test::fixture_path("3x3_srgb_lossless.jxl")).unwrap();
        let image = decode(&data).unwrap();
        assert_eq!(image.width, 3);
        assert_eq!(image.height, 3);
        assert_eq!(image.channels, 4);
        // All pixels should be non-zero (opaque alpha at minimum)
        for y in 0..3 {
            for x in 0..3 {
                let offset = (y * 3 + x) * 4;
                let alpha = image.data[offset + 3];
                assert_eq!(alpha, 255, "pixel ({x},{y}) alpha should be 255");
            }
        }
    }

    #[test]
    fn decode_with_icc() {
        let data = std::fs::read(crate::util::test::fixture_path("with_icc.jxl")).unwrap();
        let image = decode(&data).unwrap();
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert_eq!(
            image.data.len(),
            image.width * image.height * image.channels
        );
    }

    #[test]
    fn read_header_basic() {
        let data = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        let header = read_header(&data).unwrap();
        let (w, h) = header.info.size;
        assert!(w > 0);
        assert!(h > 0);
    }

    #[test]
    fn read_header_minimal_bytes() {
        // read_header should work with just the header bytes, not the whole file
        let data =
            std::fs::read(crate::util::test::fixture_path("green_queen_vardct_e3.jxl")).unwrap();
        let full_header = read_header(&data).unwrap();

        // It should also work with just the first few hundred bytes
        let partial = &data[..256.min(data.len())];
        let partial_header = read_header(partial).unwrap();
        assert_eq!(full_header.info.size, partial_header.info.size);
    }

    #[test]
    fn decode_truncated_returns_error() {
        let data = std::fs::read(crate::util::test::fixture_path("basic.jxl")).unwrap();
        // Truncate to just 10 bytes — not enough for a full decode
        let result = decode(&data[..10]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_dice() {
        let data = std::fs::read(crate::util::test::fixture_path("dice.jxl")).unwrap();
        let image = decode(&data).unwrap();
        assert_eq!(
            image.data.len(),
            image.width * image.height * image.channels
        );
    }

    /// Build a JXL container: signature + ftyp + leading boxes + jxlc + trailing boxes.
    fn build_container(
        codestream: &[u8],
        boxes_before: &[(&[u8; 4], &[u8])],
        boxes_after: &[(&[u8; 4], &[u8])],
    ) -> Vec<u8> {
        fn push_box(out: &mut Vec<u8>, ty: &[u8; 4], payload: &[u8]) {
            out.extend_from_slice(&u32::try_from(8 + payload.len()).unwrap().to_be_bytes());
            out.extend_from_slice(ty);
            out.extend_from_slice(payload);
        }
        let mut out = Vec::new();
        // Container signature box + ftyp box (`jxl ` brand), per ISO/IEC 18181-2.
        out.extend_from_slice(&[0, 0, 0, 0xC, b'J', b'X', b'L', b' ', 0xD, 0xA, 0x87, 0xA]);
        push_box(&mut out, b"ftyp", b"jxl \0\0\0\0jxl ");
        for (ty, payload) in boxes_before {
            push_box(&mut out, ty, payload);
        }
        push_box(&mut out, b"jxlc", codestream);
        for (ty, payload) in boxes_after {
            push_box(&mut out, ty, payload);
        }
        out
    }

    fn test_gain_map_bundle(codestream: &[u8]) -> crate::container::gain_map::GainMapBundle {
        crate::container::gain_map::GainMapBundle {
            metadata: b"ISO21496-1 metadata blob".to_vec(),
            color_encoding: None,
            alt_icc_compressed: Some(vec![0xCC; 64]),
            gain_map_codestream: codestream.to_vec(),
        }
    }

    const TEST_EXIF_TIFF: &[u8] = b"II*\x00test-exif-payload";
    const TEST_XMP: &[u8] = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>";

    /// Exif box payload: 4-byte TIFF header offset prefix + TIFF data.
    fn test_exif_payload() -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(TEST_EXIF_TIFF);
        payload
    }

    /// #20: jhgm / Exif / xml boxes that FOLLOW the codestream — the layout
    /// jxl-encoder's `append_gain_map_bundle` writes — must be captured by
    /// `decode`, and the pixels must match the bare-codestream decode.
    #[test]
    fn decode_captures_trailing_boxes() {
        let codestream =
            std::fs::read(crate::util::test::fixture_path("3x3_srgb_lossless.jxl")).unwrap();
        let bundle = test_gain_map_bundle(&codestream);
        let jhgm = bundle.serialize();
        let exif = test_exif_payload();

        let file = build_container(
            &codestream,
            &[],
            &[
                (b"jhgm", jhgm.as_slice()),
                (b"Exif", exif.as_slice()),
                (b"xml ", TEST_XMP),
            ],
        );

        let image = decode(&file).unwrap();
        let gm = image.gain_map.expect("trailing jhgm box must be captured");
        assert_eq!(gm, bundle);
        assert_eq!(image.exif.as_deref(), Some(TEST_EXIF_TIFF));
        assert_eq!(image.xmp.as_deref(), Some(TEST_XMP));

        let bare = decode(&codestream).unwrap();
        assert_eq!(image.data, bare.data);
    }

    /// Boxes that precede the codestream were already captured before #20;
    /// guard that the trailing-box drain didn't disturb that path.
    #[test]
    fn decode_captures_leading_boxes() {
        let codestream =
            std::fs::read(crate::util::test::fixture_path("3x3_srgb_lossless.jxl")).unwrap();
        let bundle = test_gain_map_bundle(&codestream);
        let jhgm = bundle.serialize();
        let exif = test_exif_payload();

        let file = build_container(
            &codestream,
            &[
                (b"jhgm", jhgm.as_slice()),
                (b"Exif", exif.as_slice()),
                (b"xml ", TEST_XMP),
            ],
            &[],
        );

        let image = decode(&file).unwrap();
        assert_eq!(image.gain_map, Some(bundle));
        assert_eq!(image.exif.as_deref(), Some(TEST_EXIF_TIFF));
        assert_eq!(image.xmp.as_deref(), Some(TEST_XMP));
    }

    /// A bare codestream followed by junk bytes must still decode — the
    /// trailing-box drain only engages for containers.
    #[test]
    fn decode_bare_codestream_ignores_trailing_bytes() {
        let mut data =
            std::fs::read(crate::util::test::fixture_path("3x3_srgb_lossless.jxl")).unwrap();
        let bare = decode(&data).unwrap();
        data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let image = decode(&data).unwrap();
        assert_eq!(image.data, bare.data);
        assert!(image.gain_map.is_none());
    }
}
