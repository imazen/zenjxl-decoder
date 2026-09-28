// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Actual thread migration is a native contract. wasm32-wasip1 does not
// provide std::thread::spawn; the ordinary decoder tests cover that target.
#![cfg(not(target_family = "wasm"))]

use zenjxl_decoder::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};

const INPUT: &[u8] = include_bytes!("testdata/send-state-animation.jxl");

#[test]
fn decoder_migrates_between_workers_at_every_frame_boundary() {
    let options = JxlDecoderOptions::default();
    #[cfg(feature = "cms")]
    let options = options.with_cms(Box::new(zenjxl_decoder::api::MoxCms));
    let mut decoder = std::thread::spawn(move || {
        let mut input = INPUT;
        let ProcessingResult::Complete { mut result } =
            JxlDecoder::new(options).process(&mut input).unwrap()
        else {
            panic!("incomplete image header")
        };
        result.set_pixel_format(JxlPixelFormat::rgb_f32(0));
        (result, INPUT.len() - input.len())
    })
    .join()
    .unwrap();
    for index in 0..3 {
        let (state, offset) = decoder;
        let (frame, offset) = std::thread::spawn(move || {
            let mut input = &INPUT[offset..];
            let ProcessingResult::Complete { result } = state.process(&mut input).unwrap() else {
                panic!("incomplete frame header")
            };
            (result, INPUT.len() - input.len())
        })
        .join()
        .unwrap();
        let (state, offset, bytes) = std::thread::spawn(move || {
            let mut input = &INPUT[offset..];
            let mut pixels = vec![0; 17 * 13 * 3 * 4];
            let out = JxlOutputBuffer::new(&mut pixels, 13, 17 * 3 * 4);
            let ProcessingResult::Complete { result } =
                frame.process(&mut input, &mut [out]).unwrap()
            else {
                panic!("incomplete frame pixels")
            };
            (result, INPUT.len() - input.len(), pixels)
        })
        .join()
        .unwrap();
        for (sample, value) in bytes.chunks_exact(4).enumerate() {
            let actual = f32::from_ne_bytes(value.try_into().unwrap());
            let expected =
                ((sample * 73 + sample / 17 * 113 + index.min(1) * 29) & 1023) as f32 / 1023.0;
            assert!((actual - expected).abs() < 0.5 / 1023.0);
        }
        assert_eq!(state.has_more_frames(), index < 2);
        decoder = (state, offset);
    }
}
