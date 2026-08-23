// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjxl_decoder::api::{
    JxlDecoder, JxlDecoderLimits, JxlDecoderOptions, JxlOutputBuffer, ProcessingResult, states,
};

/// The first input byte picks the chunk size (1..=256 bytes per `process`
/// call, or the whole input), the rest is the bitstream. Decoding proceeds
/// with `parallel = true` on a small rayon pool, and with the input handed
/// over in chunks the decoder takes the incremental path of
/// `decode_groups_parallel`: groups arrive in several batches, borders at
/// batch boundaries are only partially ready, and a final re-render corrects
/// them. A 30000-byte-chunk decode of `cafe_web_q80.jxl` used to panic in
/// that path (a fragment narrower than its rectangle); the corpus below is
/// what finds the next one.
fn decode_chunked(data: &[u8], chunk: usize) {
    let mut limits = JxlDecoderLimits::restrictive();
    limits.max_pixels = Some(4_000_000);
    limits.max_memory_bytes = Some(64 * 1024 * 1024);
    let mut options = JxlDecoderOptions::default();
    options.limits = limits;
    options.parallel = true;

    // `input` is the unconsumed remainder; each call sees at most `chunk`
    // bytes of it.
    let mut input = data;
    macro_rules! step {
        ($dec:expr $(, $bufs:expr)?) => {{
            let mut w = &input[..chunk.min(input.len())];
            let before = w.len();
            let r = $dec.process(&mut w $(, $bufs)?);
            input = &input[before - w.len()..];
            r
        }};
    }

    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder = loop {
        match step!(decoder) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return;
                }
                decoder = fallback;
            }
            Err(_) => return,
        }
    };
    let (width, height) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    let mut decoder = loop {
        match step!(decoder) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return;
                }
                decoder = fallback;
            }
            Err(_) => return,
        }
    };
    // One buffer per requested output (colour + each extra channel), in the
    // decoder's default format.
    let mut storage: Vec<Vec<u8>> = Vec::new();
    let mut strides = Vec::new();
    if let Some(f) = format.color_data_format {
        let bpr = width * format.color_type.samples_per_pixel() * f.bytes_per_sample();
        storage.push(vec![0; bpr * height]);
        strides.push(bpr);
    }
    for f in format.extra_channel_format.iter().flatten() {
        let bpr = width * f.bytes_per_sample();
        storage.push(vec![0; bpr * height]);
        strides.push(bpr);
    }
    let mut buffers: Vec<JxlOutputBuffer<'_>> = storage
        .iter_mut()
        .zip(&strides)
        .map(|(s, &bpr)| JxlOutputBuffer::new(s, height, bpr))
        .collect();
    loop {
        match step!(decoder, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => return,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                if input.is_empty() {
                    return;
                }
                decoder = fallback;
            }
            Err(_) => return,
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    let chunk = if sel == 0 {
        usize::MAX
    } else {
        sel as usize * 128
    };
    // A small pool: more leaves per group count than the default pool, and
    // a fixed size so findings reproduce.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    pool.install(|| decode_chunked(rest, chunk));
});
