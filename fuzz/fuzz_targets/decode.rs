// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Full decode — exercises container parsing, header parsing, entropy coding,
    // modular/VarDCT pipelines, color management, and rendering.
    let _ = zenjxl_decoder::decode(data);
});
