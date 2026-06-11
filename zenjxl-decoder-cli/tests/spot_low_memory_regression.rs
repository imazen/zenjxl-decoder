//! Regression: spot-colour JXL decode through the low-memory pipeline.
//!
//! `spot.jxl` (colour + a spot-colour extra channel, with alpha folded into the
//! interleaved colour output) leaves a `None` gap in the per-channel output
//! formats. The low-memory render pipeline's user-output save-stage buffer index
//! must be the PACKED position — matching `num_api_buffers` and upstream
//! jxl-rs's `save_idx` running counter — not the absolute extra-channel index.
//! The fork had regressed to `1 + i` (absolute), which overshot the packed
//! output-buffer Vec and panicked: "index out of bounds: the len is 3 but the
//! index is 3" in `low_memory_pipeline::check_buffer_sizes`.
//!
//! This decodes the committed `spot.jxl` fixture through the exact `decode_frames`
//! path the `decode` bench exercises, and fails loudly (no graceful skip) if the
//! fixture is missing.

use jxl::api::JxlDecoderOptions;
use std::path::Path;
use zenjxl_decoder_cli::dec::{OutputDataType, decode_frames};

#[test]
fn spot_color_low_memory_decode_does_not_panic() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("zenjxl-decoder")
        .join("resources")
        .join("test")
        .join("conformance_test_images")
        .join("spot.jxl");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("committed fixture {} must exist: {e}", path.display()));

    let mut input = bytes.as_slice();
    decode_frames(
        &mut input,
        JxlDecoderOptions::default(),
        None,
        None,
        &[
            OutputDataType::U8,
            OutputDataType::U16,
            OutputDataType::F16,
            OutputDataType::F32,
        ],
        true,
        false,
        None,
        false,
    )
    .expect("spot.jxl must decode through the low-memory pipeline without panicking");
}
