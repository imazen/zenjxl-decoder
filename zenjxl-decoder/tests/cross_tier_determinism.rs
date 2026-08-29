// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Cross-tier determinism gate: the same bytes in must give the same bytes out
//! no matter which SIMD tier the dispatcher picks.
//!
//! The decoder dispatches per call site through `simd_function!`, which summons
//! the best token the CPU offers (avx512 -> avx -> sse42 -> scalar on x86_64,
//! neon -> scalar on aarch64, wasm128 -> scalar on wasm32). Every one of those
//! tiers is a separate hand-written implementation, and nothing used to check
//! that they agreed. Three defects lived in that gap:
//!
//! * `round_store_u8` stored 255 on avx512 where every other tier stored 0 for
//!   a negative sample — a black/white inversion selected by the CPU.
//! * `round_store_u8` disagreed between sse42/avx (0) and neon (255) on values
//!   above the i32 range.
//! * The scalar tier rounded exact halves away from zero while all five SIMD
//!   tiers rounded to even.
//!
//! None were reachable by the fuzzers, which build with
//! `default-features = false` and therefore only ever exercise the scalar tier.
//!
//! `archmage::testing::for_each_token_permutation` disables tokens
//! process-wide, so `summon()` falls through to the next tier and the whole
//! decoder — not just one kernel — runs on it. Decoding every fixture under
//! every permutation and comparing the bytes is what makes the tiers testable
//! against each other.
//!
//! # What is asserted, and why it is not simply "all tiers are byte-equal"
//!
//! Byte-equality is the right contract for the integer pipeline and is
//! asserted exactly there: [`LOSSLESS_FIXTURES`] decoded with dithering off go
//! through modular decode and an integer-valued conversion, so every tier must
//! agree to the byte. Lossless means exact by definition; any divergence there
//! is a defect, and that is where the squeeze inverse's SIMD/scalar split
//! lives.
//!
//! The float pipeline cannot be held to byte-equality, and the reason is
//! structural rather than a defect: `mul_add` is a fused multiply-add on
//! avx/avx512/neon and an unfused multiply-then-add on sse42 and wasm128,
//! which have no FMA instruction. Fused rounds once, unfused rounds twice, so
//! the tiers differ by an ULP in the last place no matter how carefully each
//! is written. Removing FMA from the tiers that have it to buy exactness would
//! be a large, deliberate performance decision, not a bug fix. So the float
//! path is gated by an envelope instead: no byte may differ by more than
//! [`MAX_TIER_DIFF`], and no more than [`MAX_DIFFERING_FRACTION`] of the image
//! may differ at all.
//!
//! That envelope is not a weakened form of the byte-equality check — it is
//! chosen to be far tighter than any of the defects above, all of which move a
//! byte from 0 to 255. A tie-rounding change is a 1-LSB move and *is* inside
//! the envelope; that one is caught exactly, on every tier, by
//! `round_store_u8_contract` in zenjxl-decoder-simd, which is where a
//! per-sample rule belongs. The two gates are complementary: this one covers
//! the whole decoder including paths no unit test reaches, that one pins the
//! per-sample rule the envelope cannot see.
//!
//! # Coverage is bounded by the host
//!
//! A tier the CPU does not implement cannot be exercised here — an aarch64 host
//! can reach neon and scalar, and nothing on x86. The test prints the
//! permutations it actually ran, so a tier that is silently absent (AVX-512 on
//! a runner without it) shows up in the log rather than passing quietly.

use std::path::{Path, PathBuf};

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenjxl_decoder::api::JxlDecoderOptions;

/// Lossless fixtures: modular decode with no float quantization on the output
/// path, so with dithering off every tier must agree to the byte.
///
/// This is the list the squeeze inverse is gated by — `squeeze_*` exercise the
/// transform whose SIMD body and scalar tail split at `h & !(lanes - 1)`.
const LOSSLESS_FIXTURES: &[&str] = &[
    "3x3_srgb_lossless.jxl",
    "3x3a_srgb_lossless.jxl",
    "gray_alpha_lossless.jxl",
    "squeeze_alpha.jxl",
    "squeeze_edge.jxl",
    "squeeze_empty_residual.jxl",
    "grayscale_patches_modular.jxl",
    "orientation5_transpose.jxl",
];

/// Fixtures whose decode goes through the float pipeline: VarDCT, noise,
/// splines, upsampling, and the non-sRGB transfer functions.
///
/// PQ and HLG matter specifically because they take the Gamma/DCI branch of
/// `XybToU8Stage`, which clamps the *absolute* value before the transfer
/// function and then restores the sign with `copysign` — the one caller that
/// reached `round_store_u8` with a signed, unbounded value.
const FLOAT_FIXTURES: &[&str] = &[
    "3x3_srgb_lossy.jxl",
    "3x3a_srgb_lossy.jxl",
    "8x8_noise.jxl",
    "grayscale_patches_var_dct.jxl",
    "hdr_pq_test.jxl",
    "hdr_hlg_test.jxl",
    "pq_gradient.jxl",
    "extra_channels.jxl",
    "upsampled_alpha.jxl",
    "splines.jxl",
    "spline_on_first_frame.jxl",
];

/// Largest per-byte difference any two tiers may show on the float pipeline.
///
/// One ULP of the u8 quantization, i.e. the most a single fused-vs-unfused
/// `mul_add` can move a sample. Every defect this test exists for moves a byte
/// between 0 and 255, so they are two orders of magnitude outside this.
const MAX_TIER_DIFF: u8 = 1;

/// Largest fraction of an image that may differ at all between tiers.
///
/// Bounds a *systematic* 1-LSB shift — a wholesale rounding-mode change would
/// stay within `MAX_TIER_DIFF` per byte but move a large share of the image,
/// and that is a defect, not FMA noise. Measured worst case on this aarch64
/// host (neon vs scalar) is 2 bytes of 16384, i.e. 0.012%.
const MAX_DIFFERING_FRACTION: f64 = 0.001;

/// Root of the fixture tree: the local checkout when present (the normal dev/CI
/// case, no network), otherwise downloaded on demand via codec-corpus.
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

/// One decoded image, reduced to what has to match across tiers.
#[derive(PartialEq, Eq)]
struct Decoded {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>,
}

/// One decoded grid entry: its label, how strictly it is compared, and either
/// the decoded image or the error text.
type GridResult = (String, Strictness, Result<Decoded, String>);

/// A whole grid decoded under one token permutation, tagged with that
/// permutation's label.
type PermutationRun = (String, Vec<GridResult>);

/// How strictly a given (fixture, option set) pair is compared.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strictness {
    /// Every byte must match. Integer pipeline, no float quantization.
    Exact,
    /// Bounded by `MAX_TIER_DIFF` / `MAX_DIFFERING_FRACTION`.
    FmaEnvelope,
}

/// The (fixture, option set, strictness) grid.
///
/// `dither_u8` defaults to true, and the dithered branch of `XybToU8Stage`
/// clamps at the call site — so the unclamped store is only reachable with
/// dithering off, which is why that configuration is swept explicitly rather
/// than relying on the default.
///
/// Dithering deliberately adds sub-LSB noise before rounding, and that noise is
/// computed in float, so a dithered decode is held to the envelope even for a
/// lossless fixture. With dithering off a lossless decode must be exact.
fn grid() -> Vec<(&'static str, &'static str, JxlDecoderOptions, Strictness)> {
    let mut out = Vec::new();
    for name in LOSSLESS_FIXTURES {
        out.push((
            *name,
            "dither_u8=false",
            JxlDecoderOptions::default().with_dither_u8(false),
            Strictness::Exact,
        ));
        out.push((
            *name,
            "default",
            JxlDecoderOptions::default(),
            Strictness::FmaEnvelope,
        ));
    }
    for name in FLOAT_FIXTURES {
        out.push((
            *name,
            "dither_u8=false",
            JxlDecoderOptions::default().with_dither_u8(false),
            Strictness::FmaEnvelope,
        ));
        out.push((
            *name,
            "default",
            JxlDecoderOptions::default(),
            Strictness::FmaEnvelope,
        ));
    }
    out
}

/// Decode the whole grid, in a fixed order.
///
/// A fixture that fails to decode records the error text instead of pixels:
/// *whether* a stream decodes, and which error it reports, must also not depend
/// on the CPU.
fn decode_grid(inputs: &[(&'static str, Vec<u8>)]) -> Vec<GridResult> {
    let mut out = Vec::new();
    for (name, opt_name, opts, strictness) in grid() {
        let data = &inputs
            .iter()
            .find(|(n, _)| *n == name)
            .expect("grid names a fixture that was not loaded")
            .1;
        let result = match zenjxl_decoder::decode_with(data, opts) {
            Ok(image) => Ok(Decoded {
                width: image.width,
                height: image.height,
                channels: image.channels,
                data: image.data,
            }),
            Err(e) => Err(format!("{e:?}")),
        };
        out.push((format!("{name} [{opt_name}]"), strictness, result));
    }
    out
}

fn describe(result: &Result<Decoded, String>) -> String {
    match result {
        Ok(d) => format!(
            "ok {}x{}x{} ({} bytes)",
            d.width,
            d.height,
            d.channels,
            d.data.len()
        ),
        Err(e) => format!("error {e}"),
    }
}

/// `(count of differing bytes, largest difference, index of the first one)`.
fn compare(a: &Decoded, b: &Decoded) -> (usize, u8, usize) {
    let mut count = 0;
    let mut max = 0u8;
    let mut first = 0usize;
    for (i, (x, y)) in a.data.iter().zip(b.data.iter()).enumerate() {
        if x != y {
            if count == 0 {
                first = i;
            }
            count += 1;
            max = max.max(x.abs_diff(*y));
        }
    }
    (count, max, first)
}

#[test]
fn decode_is_identical_on_every_dispatch_tier() {
    let root = fixture_root();
    let names: Vec<&'static str> = LOSSLESS_FIXTURES
        .iter()
        .chain(FLOAT_FIXTURES.iter())
        .copied()
        .collect();
    let inputs: Vec<(&'static str, Vec<u8>)> = names
        .iter()
        .map(|name| {
            let path = root.join(name);
            let data = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
            (*name, data)
        })
        .collect();

    let mut baseline: Option<PermutationRun> = None;
    let mut labels: Vec<String> = Vec::new();
    let mut worst_diff = 0u8;
    let mut worst_fraction = 0.0f64;

    let report = for_each_token_permutation(CompileTimePolicy::WarnStderr, |perm| {
        labels.push(perm.label.clone());
        let got = decode_grid(&inputs);

        let Some((base_label, base)) = &baseline else {
            baseline = Some((perm.label.clone(), got));
            return;
        };

        assert_eq!(
            base.len(),
            got.len(),
            "permutation '{}' produced a different number of results than '{base_label}'",
            perm.label
        );

        for ((key, strictness, want), (got_key, _, have)) in base.iter().zip(got.iter()) {
            assert_eq!(key, got_key, "result order changed between permutations");

            let (want, have) = match (want, have) {
                (Ok(w), Ok(h)) => (w, h),
                _ if want == have => continue,
                _ => panic!(
                    "{key}: decoding succeeded on one tier and not another.\n  \
                     '{base_label}': {}\n  '{}': {}",
                    describe(want),
                    perm.label,
                    describe(have),
                ),
            };

            assert_eq!(
                (want.width, want.height, want.channels, want.data.len()),
                (have.width, have.height, have.channels, have.data.len()),
                "{key}: image geometry depends on the SIMD tier ('{base_label}' vs '{}')",
                perm.label
            );

            let (count, max, first) = compare(want, have);
            if count == 0 {
                continue;
            }
            let fraction = count as f64 / want.data.len() as f64;
            worst_diff = worst_diff.max(max);
            worst_fraction = worst_fraction.max(fraction);

            let context = format!(
                "{key}: {count} of {} bytes differ (max {max}, first at index {first}, \
                 {:.4}%) between '{base_label}' and '{}'",
                want.data.len(),
                fraction * 100.0,
                perm.label,
            );

            match strictness {
                Strictness::Exact => panic!(
                    "{context}\n  A lossless decode must be byte-identical on every \
                     dispatch tier; lossless is exact by definition."
                ),
                Strictness::FmaEnvelope => {
                    assert!(
                        max <= MAX_TIER_DIFF,
                        "{context}\n  Exceeds MAX_TIER_DIFF ({MAX_TIER_DIFF}): a difference \
                         this large is not fused-vs-unfused multiply-add rounding."
                    );
                    assert!(
                        fraction <= MAX_DIFFERING_FRACTION,
                        "{context}\n  Exceeds MAX_DIFFERING_FRACTION \
                         ({MAX_DIFFERING_FRACTION}): a shift affecting this much of the \
                         image is systematic, not FMA noise."
                    );
                }
            }
        }
    });

    eprintln!(
        "cross-tier decode: {} comparison(s) over {} permutation(s); \
         worst per-byte difference {worst_diff}, worst differing fraction {:.4}%\n\
         cross-tier decode: permutations: {}",
        grid().len(),
        report.permutations_run,
        worst_fraction * 100.0,
        labels.join(" | "),
    );
    for warning in &report.warnings {
        eprintln!("cross-tier decode: {warning}");
    }

    assert!(
        report.permutations_run >= 1,
        "no dispatch permutation ran; the differential test proved nothing"
    );
}
