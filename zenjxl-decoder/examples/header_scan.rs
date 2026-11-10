// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Profile header-only scans with explicit memory admission and optional layout.
use zenjxl_decoder::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult};

fn scan(data: &[u8], record: bool) -> Result<usize, String> {
    let mut options = JxlDecoderOptions::default()
        .with_scan_frames_only(true)
        .with_record_layout(record);
    options.limits.max_memory_bytes = Some(8 << 20);
    let mut input = data;
    let initialized = JxlDecoder::new(options);
    let ProcessingResult::Complete {
        result: mut decoder,
    } = initialized.process(&mut input).map_err(|e| e.to_string())?
    else {
        return Err("incomplete image header".into());
    };
    while decoder.has_more_frames() {
        let ProcessingResult::Complete { result: frame } =
            decoder.process(&mut input).map_err(|e| e.to_string())?
        else {
            return Err("incomplete frame header".into());
        };
        let ProcessingResult::Complete { result } =
            frame.skip_frame(&mut input).map_err(|e| e.to_string())?
        else {
            return Err("incomplete frame sections".into());
        };
        decoder = result;
    }
    Ok(decoder.scanned_frames().len())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let file = args.next().expect("supply a JXL input");
    let record = args.next().is_some_and(|s| s == "--layout");
    let data = std::fs::read(file).expect("read input");
    match scan(&data, record) {
        Ok(frames) => println!("frames={frames}; complete"),
        Err(error) => {
            eprintln!("scan failed: {error}");
            std::process::exit(1);
        }
    }
}
