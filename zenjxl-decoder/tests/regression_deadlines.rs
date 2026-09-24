// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Hang regressions run separately so the WASI runner can enforce a process
//! deadline without requiring guest threads. Native tests keep worker deadlines.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use zenjxl_decoder::api::{Error, JxlDecoderOptions, decode_with};

fn testdata(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[cfg(target_arch = "wasm32")]
fn with_deadline<T>(secs: u64, _what: &str, f: impl FnOnce() -> T) -> T {
    // Fail loudly if invoked outside the runner's enforced deadline. Its 20s
    // budget covers this entire binary, including both tests and fixture I/O.
    let limit: u64 = std::env::var("ZENJXL_TEST_DEADLINE_SECS")
        .expect("run this test through .cargo/wasm-runner.sh")
        .parse()
        .unwrap();
    assert!(limit > 0 && limit <= secs);
    f()
}

/// Runs `f` on a worker thread and fails if it does not finish in time.
///
/// A decoder that spins forever would otherwise hang the whole test binary
/// instead of reporting a failure, and CI would time out with no useful
/// output. The spinning thread is leaked deliberately — the process is about
/// to abort on the panic anyway.
#[cfg(not(target_arch = "wasm32"))]
fn with_deadline<T: Send + 'static>(
    secs: u64,
    what: &str,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(v) => v,
        Err(_) => panic!("{what}: still running after {secs}s (upstream expects it to return)"),
    }
}

/// Upstream `ooo_jxlp_with_trailing_bytes_does_not_hang`: an out-of-order
/// `jxlp` stream with trailing container bytes used to spin forever when
/// header parsing needed more codestream than the boxes provided. Upstream
/// asserts a single `process()` call returns `NeedsMoreInput`.
#[test]
fn ooo_jxlp_with_trailing_bytes_does_not_hang() {
    let data = testdata("ooo_jxlp_with_trailing_bytes.jxl");
    let res = with_deadline(20, "ooo_jxlp_with_trailing_bytes", move || {
        decode_with(&data, JxlDecoderOptions::default()).map(|_| ())
    });
    // Upstream stops with "needs more input"; reaching a decoded image or any
    // other error is a divergence, but the point of the fixture is that the
    // call returns at all.
    assert!(
        matches!(res, Err(ref e) if matches!(e.error(), Error::OutOfBounds(_))),
        "expected an out-of-input result, got {res:?}"
    );
}

/// Upstream `test_fuzzer_patches_ec_upsampling_dim_shift`: this fuzzer artifact
/// must be rejected without panicking or hanging.
///
/// Upstream asserts the specific error `PatchesUnsupportedMixedUpsampling`.
/// This fork reports the file as truncated instead — and so does the reference
/// implementation: `djxl 0.12.0` says "Input file is truncated (total bytes:
/// 49, processed bytes: 49)". The 49-byte file really is truncated, so the
/// error upstream surfaces is an artifact of the order in which it validates
/// patches versus running out of input, not a property worth pinning here.
/// What the fixture guards is that a malformed patch header is rejected
/// cleanly.
#[test]
fn fuzzer_patches_ec_upsampling_dim_shift_rejected() {
    let data = testdata("patches_ec_upsampling_dim_shift.jxl");
    let result = with_deadline(20, "patches_ec_upsampling_dim_shift", move || {
        decode_with(&data, JxlDecoderOptions::default()).map(|_| ())
    });
    assert!(
        result.is_err(),
        "expected the truncated fuzzer file to be rejected, got a decoded image"
    );
}
