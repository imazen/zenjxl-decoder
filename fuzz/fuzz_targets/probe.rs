#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Header-only parsing — exercises container demux, file header, ICC profile
    // decode, and metadata extraction without decoding pixels.
    // Very fast (microseconds), high exec/s.
    let _ = zenjxl_decoder::read_header(data);
});
