// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![no_main]
use libfuzzer_sys::fuzz_target;
use zenjxl_decoder::api::{
    CodestreamLayout, Error, JxlDecoder, JxlDecoderOptions, ProcessingResult, VisibleFrameInfo,
};

#[derive(Debug)]
enum ScanFailure {
    Other,
    Memory { actual: u64, limit: u64 },
}

fn classify(error: Error) -> ScanFailure {
    match error {
        Error::LimitExceeded {
            resource: "memory_bytes",
            actual,
            limit,
        } => ScanFailure::Memory { actual, limit },
        _ => ScanFailure::Other,
    }
}

fn scan(
    data: &[u8],
    record: bool,
) -> Result<(CodestreamLayout, Vec<VisibleFrameInfo>), ScanFailure> {
    let mut options = JxlDecoderOptions::default()
        .with_scan_frames_only(true)
        .with_record_layout(record);
    options.limits.max_pixels = Some(1_000_000);
    options.limits.max_memory_bytes = Some(8 << 20);
    options.limits.max_icc_size = Some(1 << 20);
    options.parallel = false;
    let mut input = data;
    let initialized = JxlDecoder::new(options);
    let ProcessingResult::Complete {
        result: mut decoder,
    } = initialized
        .process(&mut input)
        .map_err(|e| classify(e.decompose().0))?
    else {
        return Err(ScanFailure::Other);
    };
    while decoder.has_more_frames() {
        let ProcessingResult::Complete { result: frame } = decoder
            .process(&mut input)
            .map_err(|e| classify(e.decompose().0))?
        else {
            return Err(ScanFailure::Other);
        };
        let ProcessingResult::Complete { result } = frame
            .skip_frame(&mut input)
            .map_err(|e| classify(e.decompose().0))?
        else {
            return Err(ScanFailure::Other);
        };
        decoder = result;
    }
    let layout = decoder.codestream_layout().clone();
    let frames = decoder.scanned_frames().to_vec();
    // Poll past the final frame so junk draining and trailing-box errors
    // participate in the same acceptance comparison as the codestream scan.
    let _ = decoder
        .process(&mut input)
        .map_err(|e| classify(e.decompose().0))?;
    Ok((layout, frames))
}

fuzz_target!(|data: &[u8]| {
    let plain = scan(data, false);
    let recorded = scan(data, true);
    // Recorded names and spans share the same memory budget as the scan.
    // Exhausting that extra allocation is the sole permitted acceptance change.
    match (&plain, &recorded) {
        (Ok(_), Err(ScanFailure::Memory { actual, limit })) => {
            assert_eq!(*limit, 8 << 20);
            assert!(*actual > *limit);
        }
        _ => assert_eq!(
            plain.is_ok(),
            recorded.is_ok(),
            "recording changed scan acceptance"
        ),
    }
    if let (Ok((empty, frames)), Ok((layout, recorded_frames))) = (plain, recorded) {
        assert_eq!(empty, CodestreamLayout::default());
        assert_eq!(frames, recorded_frames);
        let limit = data.len() as u64 * 8;
        let bounded = |r: &std::ops::Range<u64>| {
            assert!(r.start <= r.end && r.end <= limit);
        };
        if let Some(header) = &layout.image_header_bits {
            bounded(header);
        }
        if let Some(icc) = &layout.icc_bits {
            bounded(icc);
        }
        for (_, name) in &layout.extra_channel_names {
            bounded(&name.bits);
        }
        for extension in &layout.image_extensions {
            bounded(extension);
        }
        for frame in &layout.frames {
            bounded(&frame.header_bits);
            bounded(&frame.toc_bits);
            assert_eq!(frame.header_bits.end, frame.toc_bits.start);
            if let Some(name) = &frame.name {
                bounded(&name.bits);
                assert!(
                    name.bits.start >= frame.header_bits.start
                        && name.bits.end <= frame.header_bits.end
                );
            }
            for extension in &frame.extensions {
                bounded(extension);
            }
            for section in &frame.sections {
                assert!(section.start <= section.end && section.end <= data.len() as u64);
            }
            for pair in frame.sections.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
        }
    }
});
