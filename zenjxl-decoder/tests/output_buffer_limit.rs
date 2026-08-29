// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! zenjxl-decoder#55: the convenience decoders' output buffers are cache-line
//! padded per row, so a stream that passes `max_pixels` can still demand far
//! more memory than its pixel count suggests. The farm's seed is a 1x235875981
//! image (235 MP, under the 256 MP default) whose padded RGB output is
//! 64 B/row = 15.1 GB. `max_memory_bytes` (default 1 GiB) must refuse it before
//! the allocation is attempted — on an overcommitting kernel the allocation
//! otherwise "succeeds" and the prefault OOM-kills the process.

use zenjxl_decoder::api::Error;

const SEED: &[u8] = include_bytes!("../../fuzz/regression/oom-15gb-container-issue55");

#[test]
fn issue55_padded_output_buffer_is_charged_against_max_memory_bytes() {
    let err = match zenjxl_decoder::decode(SEED) {
        Ok(image) => panic!(
            "a 15 GB output buffer must be refused, but decoded {}x{}",
            image.width, image.height
        ),
        Err(err) => err,
    };
    // 64-bit: the padded footprint is representable and exceeds the 1 GiB
    // default budget. 32-bit: 15.1 GB does not fit `usize`, so the size check
    // refuses it before the budget is even consulted. Either way the decode
    // fails before the allocation is attempted.
    #[cfg(target_pointer_width = "64")]
    assert!(
        matches!(
            err.error(),
            Error::LimitExceeded {
                resource: "memory_bytes",
                ..
            }
        ),
        "expected LimitExceeded(memory_bytes), got {err:?}"
    );
    #[cfg(not(target_pointer_width = "64"))]
    assert!(
        matches!(err.error(), Error::ImageSizeTooLarge(..)),
        "expected ImageSizeTooLarge, got {err:?}"
    );
}
