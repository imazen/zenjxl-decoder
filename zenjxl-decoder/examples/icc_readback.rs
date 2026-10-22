// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Exercise one-shot and fragmented ICC header parsing on caller-supplied files.
//! Inputs are mandatory; a missing file or failed case exits unsuccessfully.
use zenjxl_decoder::api::{JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult};

fn main() {
    let files: Vec<_> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "supply ICC-bearing JXL files");
    for file in files {
        let data = std::fs::read(&file).expect("read fixture");
        let header = zenjxl_decoder::read_header(&data).expect("one-shot header");
        let JxlColorProfile::Icc(expected) = &header.embedded_profile else {
            panic!("fixture must carry an ICC profile");
        };
        let image = zenjxl_decoder::decode(&data).expect("pixel decode");
        assert_eq!(&image.embedded_profile, &header.embedded_profile);
        for chunk in [1, 7, 64, 4096] {
            let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
            let (mut offset, mut end) = (0, chunk.min(data.len()));
            loop {
                let mut input = &data[offset..end];
                let result = decoder.process(&mut input).expect("incremental header");
                offset = end - input.len();
                match result {
                    ProcessingResult::Complete { result } => {
                        assert_eq!(result.embedded_color_profile(), &header.embedded_profile);
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        assert!(end < data.len(), "truncated header at chunk {chunk}");
                        decoder = fallback;
                        end = (end + chunk).min(data.len());
                    }
                }
            }
        }
        let mut limits = zenjxl_decoder::api::JxlDecoderLimits::default();
        limits.max_icc_size = Some(128);
        assert!(zenjxl_decoder::read_header_with(&data, limits).is_err());
        println!("{file}\t{}\tok", expected.len());
    }
}
