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
        let data = std::fs::read("resources/test/basic.jxl").unwrap();
        let image = decode(&data).unwrap();
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert_eq!(image.channels, 4); // RGBA
        assert!(!image.is_grayscale);
        assert_eq!(image.data.len(), image.width * image.height * 4);
    }

    #[test]
    fn decode_grayscale() {
        let data = std::fs::read("resources/test/gray_alpha_lossless.jxl").unwrap();
        let image = decode(&data).unwrap();
        assert!(image.width > 0);
        assert!(image.height > 0);
        assert_eq!(image.channels, 2); // GrayAlpha
        assert!(image.is_grayscale);
        assert_eq!(image.data.len(), image.width * image.height * 2);
    }

    #[test]
    fn decode_3x3_srgb_lossless() {
        let data = std::fs::read("resources/test/3x3_srgb_lossless.jxl").unwrap();
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
        let data = std::fs::read("resources/test/with_icc.jxl").unwrap();
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
        let data = std::fs::read("resources/test/basic.jxl").unwrap();
        let header = read_header(&data).unwrap();
        let (w, h) = header.info.size;
        assert!(w > 0);
        assert!(h > 0);
    }

    #[test]
    fn read_header_minimal_bytes() {
        // read_header should work with just the header bytes, not the whole file
        let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let full_header = read_header(&data).unwrap();

        // It should also work with just the first few hundred bytes
        let partial = &data[..256.min(data.len())];
        let partial_header = read_header(partial).unwrap();
        assert_eq!(full_header.info.size, partial_header.info.size);
    }

    #[test]
    fn decode_truncated_returns_error() {
        let data = std::fs::read("resources/test/basic.jxl").unwrap();
        // Truncate to just 10 bytes — not enough for a full decode
        let result = decode(&data[..10]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_dice() {
        let data = std::fs::read("resources/test/dice.jxl").unwrap();
        let image = decode(&data).unwrap();
        assert_eq!(
            image.data.len(),
            image.width * image.height * image.channels
        );
    }
}
