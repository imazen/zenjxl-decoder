// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Buffer-reuse corruption gate for the VarDCT scratch-buffer reuse in issue #40
//! (reusing the per-pass `num_nzeros` maps and replacing the per-block `Vec`
//! scratch in the AFV transform with stack arrays). That change must not alter
//! the decoded image; buffer reuse that fails to fully reset state would
//! silently corrupt the decode — exactly the class of bug this guards against.
//!
//! The historical strided references use a per-byte tolerance because they
//! were captured before the fused/unfused SIMD distinction was repaired. The
//! separate `cross_tier_determinism` gate now requires identical bytes across
//! the executed tiers, including float output and dithering. Keeping these
//! existing references and expectations preserves their buffer-corruption
//! coverage while the strict gate checks current dispatch consistency.
//!
//! The reference samples every `STRIDE`th pixel in both axes, covering every
//! 256x256 group. `JXL_GEN_BITID_REF=1` regenerates them only after an intended,
//! independently verified output change.

use std::path::{Path, PathBuf};

/// Sample every 32nd pixel in each axis: 64x80 = 5120 samples for the 2048x2560
/// fixtures (20 KB), with 8x8 = 64 samples inside every 256x256 group.
const STRIDE: usize = 32;

/// Historical buffer-reuse reference tolerance, retained unchanged. Current
/// cross-tier consistency is checked byte-exactly by its dedicated gate.
const MAX_DIFF: u8 = 2;

struct Expected {
    file: &'static str,
    width: usize,
    height: usize,
    channels: usize,
    len: usize,
    /// Committed strided-reference file, relative to `resources/test`.
    reference: &'static str,
}

/// Root of the fixture tree: the local checkout when present (the normal dev/CI
/// case, no network), otherwise downloaded on demand via codec-corpus. The
/// integration-test mirror of the lib's `crate::util::test::fixture_dir`;
/// `resources/test/` is not packaged in the published crate (#8). Panics loudly
/// on failure — a test must never pass without its data.
fn fixture_root() -> PathBuf {
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/test");
    #[cfg(not(target_arch = "wasm32"))]
    if !local.is_dir() {
        return codec_corpus::Corpus::new()
            .expect("initialize codec-corpus to download test fixtures")
            .github_repo(
                "imazen/zenjxl-decoder",
                "zenjxl-decoder/resources/test",
                "main",
            )
            .expect("download zenjxl-decoder test fixtures via codec-corpus");
    }
    local
}

fn resource(name: &str) -> PathBuf {
    fixture_root().join(name)
}

/// In-checkout path for the regeneration path: writing a new reference only
/// makes sense in the dev repo, never against the download cache.
fn local_resource(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources/test")
        .join(name)
}

/// Extract every `STRIDE`-th pixel (in both axes) as a flat RGBA byte vector.
fn strided_sample(data: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let i = (y * width + x) * channels;
            out.extend_from_slice(&data[i..i + channels]);
            x += STRIDE;
        }
        y += STRIDE;
    }
    out
}

fn assert_decode_matches(e: &Expected) {
    let path = resource(e.file);
    let data = std::fs::read(&path)
        .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()));

    let image = zenjxl_decoder::decode(&data)
        .unwrap_or_else(|err| panic!("decode failed for {}: {err:?}", e.file));

    assert_eq!(image.width, e.width, "{}: width changed", e.file);
    assert_eq!(image.height, e.height, "{}: height changed", e.file);
    assert_eq!(
        image.channels, e.channels,
        "{}: channel count changed",
        e.file
    );
    assert_eq!(
        image.data.len(),
        e.len,
        "{}: output byte length changed",
        e.file
    );

    let sample = strided_sample(&image.data, image.width, image.height, image.channels);

    // Regeneration path: write the current strided sample as the new reference.
    if std::env::var("JXL_GEN_BITID_REF").is_ok() {
        std::fs::write(local_resource(e.reference), &sample).unwrap();
        eprintln!("regenerated {} ({} bytes)", e.reference, sample.len());
        return;
    }

    let reference = std::fs::read(resource(e.reference)).unwrap_or_else(|err| {
        panic!(
            "missing reference {} ({err}); regenerate with JXL_GEN_BITID_REF=1",
            e.reference
        )
    });
    assert_eq!(
        sample.len(),
        reference.len(),
        "{}: strided sample length changed",
        e.file
    );

    let mut max_diff = 0u8;
    let mut worst_at = 0usize;
    for (i, (&a, &b)) in sample.iter().zip(reference.iter()).enumerate() {
        let d = a.abs_diff(b);
        if d > max_diff {
            max_diff = d;
            worst_at = i;
        }
    }
    assert!(
        max_diff <= MAX_DIFF,
        "{}: decoded pixels diverge from the clean-main reference by {} (> {} tolerance) \
         at strided byte {} -- a buffer-reuse change has corrupted output \
         (cross-backend FP rounding is <= 1 LSB, so this is a real regression)",
        e.file,
        max_diff,
        MAX_DIFF,
        worst_at,
    );
}

/// `bike_web_q85.jxl` is the 5.24 MP VarDCT photo used by the allocation profiler
/// in `examples/heaptrack_decode.rs`; it exercises the AFV transform and
/// `num_nzeros` (multi-group) paths touched by the issue #40 fix.
#[test]
fn bike_web_q85_decode_matches_reference() {
    assert_decode_matches(&Expected {
        file: "bike_web_q85.jxl",
        width: 2048,
        height: 2560,
        channels: 4,
        len: 20_971_520,
        reference: "bike_web_q85_strided_ref.bin",
    });
}

/// A second VarDCT fixture at a different quality, so the gate covers more than
/// one coefficient distribution / block-type mix.
#[test]
fn bike_web_q75_decode_matches_reference() {
    assert_decode_matches(&Expected {
        file: "bike_web_q75.jxl",
        width: 2048,
        height: 2560,
        channels: 4,
        len: 20_971_520,
        reference: "bike_web_q75_strided_ref.bin",
    });
}
