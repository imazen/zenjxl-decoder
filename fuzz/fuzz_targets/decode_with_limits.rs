#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjxl_decoder::api::{JxlDecoderLimits, JxlDecoderOptions};

fuzz_target!(|data: &[u8]| {
    // Full decode with restrictive limits — prevents OOM from masking real bugs.
    // Exercises the same paths as decode but with resource caps.
    let mut limits = JxlDecoderLimits::restrictive();
    limits.max_pixels = Some(4_000_000);
    limits.max_memory_bytes = Some(64 * 1024 * 1024);

    let mut options = JxlDecoderOptions::default();
    options.limits = limits;
    options.parallel = false;

    let _ = zenjxl_decoder::decode_with(data, options);
});
