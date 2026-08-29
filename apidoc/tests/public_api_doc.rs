// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//! Auto-discovers the publishable library members (zenjxl-decoder,
//! zenjxl-decoder-macros, zenjxl-decoder-simd; the cli is publish = false).
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new().workspace_dir("..").run();
}
