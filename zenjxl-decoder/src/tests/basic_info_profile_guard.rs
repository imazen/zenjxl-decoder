// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Regression tests for the `basic_info()` embedded-profile guard.
//!
//! Ports the test added in upstream jxl-rs 28ddaeb (PR #745,
//! "decoder: hide basic info until embedded profile is available"). Ensures
//! `JxlDecoderInner::basic_info()` matches the typed `WithImageInfo`
//! transition: image info is not observable before the embedded color profile
//! has been parsed.

use crate::api::{JxlDecoderInner, JxlDecoderOptions};

/// Tiny chunk size to exercise the incremental parse path and deliberately
/// stall inside the image header, before the embedded color profile box
/// finishes.
const CHUNK_SIZE: usize = 16;

fn test_resource(name: &str) -> Vec<u8> {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("resources/test").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// `cmyk_layers.jxl` carries an embedded ICC color profile. Feeding bytes in
/// small chunks, `basic_info()` must stay `None` until the embedded profile
/// has been parsed, and the transition to `Some(_)` must coincide with the
/// embedded profile becoming visible.
#[test]
fn basic_info_hidden_until_embedded_profile_cmyk() {
    let data = test_resource("conformance_test_images/cmyk_layers.jxl");
    let mut decoder = JxlDecoderInner::new(JxlDecoderOptions::default());

    let mut saw_some = false;
    for chunk in data.chunks(CHUNK_SIZE) {
        let mut input = chunk;
        let _ = decoder.process(&mut input, None);

        // Invariant: basic_info() is observable only once the embedded
        // color profile has been parsed.
        if decoder.embedded_color_profile().is_none() {
            assert!(
                decoder.basic_info().is_none(),
                "basic_info() leaked before embedded color profile parsed"
            );
        }

        if decoder.basic_info().is_some() {
            assert!(
                decoder.embedded_color_profile().is_some(),
                "basic_info() became Some without embedded color profile"
            );
            saw_some = true;
            break;
        }
    }

    assert!(
        saw_some,
        "failed to reach image-info state while parsing cmyk_layers.jxl"
    );
}

/// Negative regression test: `basic.jxl` has no embedded ICC profile — the
/// color profile is signalled by `ColorEncoding`, which still populates
/// `embedded_color_profile` as soon as the image header has been parsed.
/// `basic_info()` must become `Some(_)` once the header lands.
#[test]
fn basic_info_visible_for_non_icc_file() {
    let data = test_resource("basic.jxl");
    let mut decoder = JxlDecoderInner::new(JxlDecoderOptions::default());

    let mut saw_some = false;
    for chunk in data.chunks(CHUNK_SIZE) {
        let mut input = chunk;
        let _ = decoder.process(&mut input, None);

        if decoder.basic_info().is_some() {
            assert!(
                decoder.embedded_color_profile().is_some(),
                "basic.jxl transitioned to Some(basic_info) but embedded profile is still None"
            );
            let info = decoder.basic_info().unwrap();
            assert!(info.size.0 > 0, "basic.jxl width should be positive");
            assert!(info.size.1 > 0, "basic.jxl height should be positive");
            saw_some = true;
            break;
        }
    }

    assert!(
        saw_some,
        "failed to reach image-info state while parsing basic.jxl"
    );
}
