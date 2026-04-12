// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Header-only parsing — exercises container demux, file header, ICC profile
    // decode, and metadata extraction without decoding pixels.
    // Very fast (microseconds), high exec/s.
    let _ = zenjxl_decoder::read_header(data);
});
