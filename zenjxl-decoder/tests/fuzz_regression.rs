//! Replay seed inputs from `../fuzz/regression/` (top-level cargo-fuzz
//! workspace) through every fuzz target entry point. Shared scaffolding
//! lives in `zen-fuzz-regress`.

use zenjxl_decoder::api::{JxlDecoderLimits, JxlDecoderOptions};
use zenutils_fuzz::RegressionSuite;

#[test]
fn fuzz_regression() {
    // CARGO_MANIFEST_DIR is the inner crate; the fuzz workspace lives at
    // the repo root, alongside it.
    let seed_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fuzz")
        .join("regression");

    RegressionSuite::new(seed_dir)
        .target("decode", |input| {
            let _ = zenjxl_decoder::decode(input);
        })
        .target("decode_with_limits", |input| {
            let mut limits = JxlDecoderLimits::restrictive();
            limits.max_pixels = Some(4_000_000);
            limits.max_memory_bytes = Some(64 * 1024 * 1024);
            let mut options = JxlDecoderOptions::default();
            options.limits = limits;
            options.parallel = false;
            let _ = zenjxl_decoder::decode_with(input, options);
        })
        .target("probe", |input| {
            let _ = zenjxl_decoder::read_header(input);
        })
        .run();
}
