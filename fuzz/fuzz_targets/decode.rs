#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Full decode — exercises container parsing, header parsing, entropy coding,
    // modular/VarDCT pipelines, color management, and rendering.
    let _ = zenjxl_decoder::decode(data);
});
