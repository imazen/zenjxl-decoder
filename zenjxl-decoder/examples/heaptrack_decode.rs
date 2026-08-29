// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Heaptrack harness for JPEG XL decode-from-bytes allocation profiling.
//!
//! Profiles the production-critical path: `zenjxl_decoder::decode(&bytes)` —
//! decoding a JXL file (untrusted input) all the way to interleaved RGBA8 / GrayA8
//! pixels. The goal is to surface allocation *pathologies* that don't show up in a
//! wall-clock benchmark: a high allocation *count* relative to image size, per-pixel
//! or per-DCT-block/per-group mallocs, large transient peaks, or unbounded growth
//! across repeated decodes (a leak). High allocation churn hurts most under
//! contended allocators (Windows, multi-threaded servers) where a single decode of
//! an untrusted upload turns into thousands of lock round-trips.
//!
//! Usage:
//!   cargo build -p zenjxl-decoder --release --example heaptrack_decode
//!   heaptrack ./target/release/examples/heaptrack_decode                 # default fixture
//!   heaptrack ./target/release/examples/heaptrack_decode <file.jxl> [iters]
//!
//! Then inspect:
//!   heaptrack_print heaptrack.heaptrack_decode.*.zst | less
//!
//! Defaults to the committed `resources/test/bike_web_q85.jxl` (a real VarDCT
//! photo: many 256x256 groups and 8x8 DCT blocks, so the allocation count can be
//! judged relative to image size) decoded 8 times. A large fixture should be
//! decoded fewer times (pass a smaller `iters`).

use std::hint::black_box;
use std::path::{Path, PathBuf};

/// Resolve the default bundled fixture relative to the crate manifest so the
/// example runs from any working directory.
fn default_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/test/bike_web_q85.jxl")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let path: PathBuf = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => default_fixture(),
    };
    // Default 8 iterations; a leak shows up as monotonic growth across them, and a
    // healthy decoder's steady-state per-decode allocation count is iterations-stable.
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let data = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", path.display());
        std::process::exit(1);
    });

    // Probe once so the report can state the dimensions the alloc count is relative to.
    match zenjxl_decoder::read_header(&data) {
        Ok(hdr) => {
            let (w, h) = hdr.info.size;
            eprintln!("fixture: {} ({} bytes on disk)", path.display(), data.len());
            eprintln!(
                "  decoded image: {}x{} ({:.2} MP)",
                w,
                h,
                (w as f64 * h as f64) / 1.0e6
            );
        }
        Err(e) => {
            eprintln!("probe (read_header) failed for {}: {e:?}", path.display());
            std::process::exit(1);
        }
    }

    eprintln!("decoding {iters}x via zenjxl_decoder::decode(..) ...");

    let mut total_pixels: u64 = 0;
    for i in 0..iters {
        let image = zenjxl_decoder::decode(&data).unwrap_or_else(|e| {
            eprintln!("decode iteration {i} failed: {e:?}");
            std::process::exit(1);
        });
        total_pixels += image.width as u64 * image.height as u64;
        // Consume the decoded buffer so the optimizer can't elide the decode or the
        // allocation of the output Vec.
        black_box(&image.data);
        black_box(image.width);
        black_box(image.height);
        black_box(image.channels);
    }

    eprintln!("done: decoded {total_pixels} total pixels across {iters} iterations");
}
