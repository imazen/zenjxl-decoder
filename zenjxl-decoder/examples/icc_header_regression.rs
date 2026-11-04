//! Caller-supplied ICC/header regression, including incremental input.
use std::path::PathBuf;
use zenjxl_decoder::api::{
    JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult, states,
};

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert_eq!(
        args.len(),
        2,
        "usage: icc_header_regression <stream.jxl> <expected.icc>"
    );
    let data = std::fs::read(&args[0]).expect("read regression stream");
    let expected = std::fs::read(&args[1]).expect("read expected profile");
    for chunk in [usize::MAX, 1, 7, 257] {
        let mut options = JxlDecoderOptions::default();
        options.limits = zenjxl_decoder::api::JxlDecoderLimits::unlimited();
        let mut state = JxlDecoder::<states::Initialized>::new(options);
        let (mut consumed, mut offered) = (0, 0usize);
        let mut calls = 0;
        loop {
            calls += 1;
            assert!(calls <= data.len() * 2 + 100, "chunk={chunk}: stalled");
            offered = offered.saturating_add(chunk).min(data.len());
            let mut input = &data[consumed..offered];
            let previous = consumed;
            let result = state
                .process(&mut input)
                .expect("decode complete ICC header");
            consumed = offered - input.len();
            match result {
                ProcessingResult::Complete { result } => {
                    let JxlColorProfile::Icc(icc) = result.embedded_color_profile() else {
                        panic!("missing ICC profile");
                    };
                    assert_eq!(icc, &expected, "chunk={chunk}: profile differs");
                    println!(
                        "chunk={chunk}: ICC exact, size={:?}",
                        result.basic_info().size
                    );
                    break;
                }
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(
                        offered < data.len() || consumed > previous,
                        "chunk={chunk}: all input supplied but header is incomplete"
                    );
                    state = fallback;
                }
            }
        }
    }
}
