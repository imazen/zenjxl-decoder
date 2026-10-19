// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Exercise one-shot and fragmented ICC header parsing on caller-supplied files.
//! Inputs are mandatory; a missing file or failed case exits unsuccessfully.
use zenjxl_decoder::api::{JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult};

fn main() {
    let mut files: Vec<_> = std::env::args().skip(1).collect();
    let output = if files.first().is_some_and(|arg| arg == "--write-icc") {
        files.remove(0);
        let directory = std::path::PathBuf::from(files.remove(0));
        std::fs::create_dir(&directory).expect("new ICC output directory");
        Some(directory)
    } else {
        None
    };
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
        if let Some(directory) = &output {
            let name = std::path::Path::new(&file).file_name().unwrap();
            std::fs::write(directory.join(name).with_extension("icc"), expected)
                .expect("retain ICC bytes");
            assert_eq!(image.channels, 4, "RGB oracle fixtures");
            let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
            for pixel in image.data.as_chunks::<4>().0 {
                assert_eq!(pixel[3], 255, "opaque oracle fixtures");
                ppm.extend_from_slice(&pixel[..3]);
            }
            std::fs::write(directory.join(name).with_extension("ppm"), ppm)
                .expect("retain native RGB samples");
        }
        println!("{file}\t{}\tok", expected.len());
    }
}
