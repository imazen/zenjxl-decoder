// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Paired header-only scanning: current main versus layout recording disabled.
//! Inputs are shared; neither arm decodes pixels. Both use default resource caps.
use zenbench::prelude::*;

macro_rules! scanner {
    ($function:ident, $decoder:ident) => {
        fn $function(mut input: &[u8]) -> usize {
            use $decoder::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult};
            let ProcessingResult::Complete {
                result: mut decoder,
            } = JxlDecoder::new(JxlDecoderOptions::default().with_scan_frames_only(true))
                .process(&mut input)
                .unwrap()
            else {
                panic!("image header incomplete")
            };
            let mut count = 0;
            while decoder.has_more_frames() {
                let ProcessingResult::Complete { result: frame } =
                    decoder.process(&mut input).unwrap()
                else {
                    panic!("frame header incomplete")
                };
                let ProcessingResult::Complete { result } = frame.skip_frame(&mut input).unwrap()
                else {
                    panic!("frame incomplete")
                };
                decoder = result;
                count += 1;
            }
            count
        }
    };
}
scanner!(scan_main, baseline);
scanner!(scan_layout_disabled, candidate);

fn compare_scans(suite: &mut Suite) {
    for &(name, input) in cases() {
        let expected = scan_main(input);
        assert!(expected > 0);
        assert_eq!(scan_layout_disabled(input), expected);
        suite.compare(
            format!("header-scan/layout-disabled/{name}"),
            move |group| {
                group.bench("main-6d060afb", move |b| {
                    b.with_input(move || input).run(scan_main)
                });
                group.bench("layout-disabled", move |b| {
                    b.with_input(move || input).run(scan_layout_disabled)
                });
            },
        );
    }
}
// Repeated operations live only in the benchmark binary so perf/callgrind can
// sample the same scanners without instrumenting decoder code or timing loops.
fn cases() -> &'static [(&'static str, &'static [u8])] {
    &[
        (
            "basic",
            include_bytes!("../../../zenjxl-decoder/resources/test/basic.jxl"),
        ),
        (
            "layered",
            include_bytes!(
                "../../../zenjxl-decoder/resources/test/multiple_layers_noise_spline.jxl"
            ),
        ),
        (
            "cmyk",
            include_bytes!(
                "../../../zenjxl-decoder/resources/test/conformance_test_images/cmyk_layers.jxl"
            ),
        ),
    ]
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if let Some(arm) = args.iter().find_map(|a| a.strip_prefix("--profile=")) {
        let name = args
            .iter()
            .find_map(|a| a.strip_prefix("--case="))
            .unwrap_or("basic");
        let iterations: usize = args
            .iter()
            .find_map(|a| a.strip_prefix("--iterations="))
            .unwrap_or("100000")
            .parse()
            .unwrap();
        assert!(iterations > 0);
        let input = cases()
            .iter()
            .find(|(key, _)| *key == name)
            .expect("unknown fixture")
            .1;
        let scanner = match arm {
            "baseline" => scan_main,
            "candidate" => scan_layout_disabled,
            _ => panic!("unknown profile arm"),
        };
        for _ in 0..iterations {
            black_box(scanner(black_box(input)));
        }
        return;
    }
    let filter = args.iter().find_map(|a| a.strip_prefix("--group="));
    let result = zenbench::run(|suite| {
        if let Some(filter) = filter {
            suite.set_group_filter(filter.to_owned());
        }
        compare_scans(suite);
    });
    zenbench::postprocess_result(&result);
}
