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
//! None were reachable by the fuzzers, which pinned `default-features = false`
//! and therefore compiled only the scalar tier into the fuzz binaries — the
//! code carrying two of the three defects was not present to be fuzzed. The
//! root fuzz package now builds with `all-simd`; this test covers the same
//! ground deterministically and on every CI target rather than opportunistically.
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
//! asserted exactly there: [`EXACT_FIXTURES`] decoded with dithering off go
//! through modular decode and an integer-valued conversion, so every tier must
//! agree to the byte. Any divergence there is a defect, and that is where the
//! squeeze inverse's SIMD/scalar split lives.
//!
//! "Lossless" alone is *not* the criterion, and assuming it was is how this
//! test first went wrong: `squeeze_empty_residual.jxl` was put in the exact
//! list on the strength of its name, is actually lossy, and failed on
//! windows-on-arm while passing on aarch64 macOS. See [`EXACT_FIXTURES`] for
//! the four conditions that actually have to hold and how each entry was
//! checked.
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
//! can reach neon and scalar, and nothing on x86. Building the avx512 tier is
//! not the same as running it: CI's `avx512` job proves it compiles, and only
//! proves it *executes* if the runner's CPU actually has AVX-512.
//!
//! Two things keep that from being mistaken for coverage. On x86_64 and aarch64
//! the run asserts that at least two permutations executed, so a baseline
//! compared against nothing fails instead of passing vacuously. And on x86_64
//! it reports whether the avx512 token could be summoned at all. Both matter
//! because `cargo test` captures output for passing tests: a printed summary
//! alone is invisible in CI unless something fails, so the guarantee has to be
//! an assertion.
//!
//! i686 and wasm32 are exempt from that floor, and the exemption is real rather
//! than convenient: archmage registers disable-able token slots for x86_64 and
//! aarch64 only, so there is nothing to permute on either. On i686 — a target
//! this crate treats as primary — the scalar tier is simply what runs, which is
//! why the per-sample rules it must obey are pinned by
//! `round_store_u8_contract` in zenjxl-decoder-simd instead.

use std::path::{Path, PathBuf};

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenjxl_decoder::api::JxlDecoderOptions;

/// Fixtures held to byte-equality: 8-bit modular lossless, whose decode carries
/// no float quantization onto the output path, so with dithering off every tier
/// must agree to the byte.
///
/// Membership is measured, not inferred from the filename — every entry was
/// checked with `jxlinspect`, which reports mode, bit depth and channels. Two
/// fixtures were originally put here on the strength of their names and do not
/// belong: `squeeze_alpha.jxl` and `squeeze_empty_residual.jxl` are both
/// **lossy** (jxlinspect: "203x354, lossy" and "64x64, lossy"), so they go
/// through VarDCT and are held to the envelope below instead. Windows-on-arm
/// CI is what caught that — `squeeze_empty_residual` differed by one byte of
/// 16384 there while being byte-equal on this aarch64 host, because a 1-ULP
/// fused-vs-unfused difference only crosses a rounding boundary on some
/// codegen.
///
/// Three more constraints are load-bearing, and "the header says lossless" does
/// not imply them, which is why several `(possibly) lossless` fixtures are in
/// the envelope list instead:
///
/// * **8-bit output.** `k / 255.0 * 255.0` round-trips exactly in f32 for every
///   `k` in `0..=255`, so the conversion is exact on every tier. At 10- or
///   16-bit (`hdr_pq_test`, `hdr_hlg_test`, `pq_gradient`) it is not.
/// * **An sRGB transfer function.** PQ and HLG are evaluated in float.
/// * **No spline or noise synthesis**, both of which are float
///   (`splines.jxl`, `spline_on_first_frame.jxl`).
///
/// This is the list the squeeze inverse is gated by. `squeeze_edge.jxl` is the
/// one that matters there: 513x513 is deliberately not a multiple of any lane
/// count, so the decode crosses from the vector body into the scalar tail at
/// `h & !(lanes - 1)` — exactly the split the two implementations disagreed
/// across.
const EXACT_FIXTURES: &[&str] = &[
    "3x3_srgb_lossless.jxl",
    "3x3a_srgb_lossless.jxl",
    "gray_alpha_lossless.jxl",
    "squeeze_edge.jxl",
    "grayscale_patches_modular.jxl",
    "orientation5_transpose.jxl",
    "extra_channels.jxl",
];

/// Fixtures whose decode involves float arithmetic somewhere, and which are
/// therefore held to the FMA envelope rather than to byte-equality: VarDCT,
/// noise, splines, upsampling, non-sRGB transfer functions, and any output
/// deeper than 8 bits.
///
/// PQ and HLG matter specifically because they take the Gamma/DCI branch of
/// `XybToU8Stage`, which clamps the *absolute* value before the transfer
/// function and then restores the sign with `copysign` — the one caller that
/// reached `round_store_u8` with a signed, unbounded value. Measured on
/// `3x3_srgb_lossy.jxl`, samples from -44.6 to +262.3 arrive at that store.
const FLOAT_FIXTURES: &[&str] = &[
    "3x3_srgb_lossy.jxl",
    "3x3a_srgb_lossy.jxl",
    "8x8_noise.jxl",
    "grayscale_patches_var_dct.jxl",
    "hdr_pq_test.jxl",
    "hdr_hlg_test.jxl",
    "pq_gradient.jxl",
    "upsampled_alpha.jxl",
    "splines.jxl",
    "spline_on_first_frame.jxl",
    "squeeze_alpha.jxl",
    "squeeze_empty_residual.jxl",
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

/// Write a line that survives libtest's output capture.
///
/// The harness captures `print!`/`eprint!` for a *passing* test and only shows
/// it under `--nocapture`, which CI does not pass — so a coverage summary
/// emitted with `eprintln!` is invisible in exactly the case that matters, a
/// green run that silently exercised fewer tiers than it looks like. The
/// capture is installed for those macros only, so writing to the process's
/// stderr handle directly bypasses it and lands in the CI log.
fn report_uncaptured(line: &str) {
    use std::io::Write;
    let mut err = std::io::stderr();
    let _ = writeln!(err, "{line}");
    let _ = err.flush();
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
    // Single-threaded, always. The dispatch tier is the only thing this test is
    // allowed to vary; `parallel` defaults to `true` whenever the `threads`
    // feature is on, and leaving it on would let rayon's scheduling — which
    // depends on the machine's core count — confound a tier difference with a
    // scheduling one. `decode_is_deterministic_when_parallel` covers the
    // parallel path separately, where run-to-run equality is the actual claim.
    let base = || JxlDecoderOptions::default().with_parallel(false);
    let mut out = Vec::new();
    for name in EXACT_FIXTURES {
        out.push((
            *name,
            "dither_u8=false",
            base().with_dither_u8(false),
            Strictness::Exact,
        ));
        out.push((*name, "default", base(), Strictness::FmaEnvelope));
    }
    for name in FLOAT_FIXTURES {
        out.push((
            *name,
            "dither_u8=false",
            base().with_dither_u8(false),
            Strictness::FmaEnvelope,
        ));
        out.push((*name, "default", base(), Strictness::FmaEnvelope));
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
    let names: Vec<&'static str> = EXACT_FIXTURES
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
                    "{context}\n  This fixture's decode carries no float quantization \
                     onto the output path (8-bit modular lossless, sRGB transfer, no \
                     splines or noise), so every dispatch tier must agree to the byte. \
                     If the fixture does not actually satisfy that, fix its \
                     classification in EXACT_FIXTURES rather than this assertion."
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

    report_uncaptured(&format!(
        "cross-tier decode: {} comparison(s) over {} permutation(s); \
         worst per-byte difference {worst_diff}, worst differing fraction {:.4}%",
        grid().len(),
        report.permutations_run,
        worst_fraction * 100.0,
    ));
    report_uncaptured(&format!(
        "cross-tier decode: permutations: {}",
        labels.join(" | ")
    ));
    for warning in &report.warnings {
        report_uncaptured(&format!("cross-tier decode: {warning}"));
    }

    // Anti-vacuity gate. `cargo test` captures stdout and stderr for passing
    // tests, so the summary above is only visible on failure or under
    // `--nocapture` — it cannot be relied on to reveal that a tier went
    // unexercised. One permutation means the baseline was compared against
    // nothing at all, which passes while proving nothing.
    //
    // Only x86_64 and aarch64 can be held to that. archmage registers
    // disable-able token slots for those two architectures and no others, so on
    // i686 and wasm32 there is nothing to permute — i686 has no archmage tokens
    // at all, and the wasm128 tier is selected at compile time rather than
    // summoned. On those targets this test degenerates to a repeat-decode
    // check, which is still worth running but proves nothing about tiers.
    let min_permutations = if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        2
    } else {
        1
    };
    assert!(
        report.permutations_run >= min_permutations,
        "only {} dispatch permutation(s) ran ({}), so nothing was compared against \
         anything. This target registers disable-able SIMD tokens (and the archmage \
         `testable_dispatch` dev-feature makes even compile-time guaranteed ones \
         disable-able), so this means token discovery is broken, not that the CPU is \
         unusual.{}",
        report.permutations_run,
        labels.join(" | "),
        if report.warnings.is_empty() {
            String::new()
        } else {
            format!(" Warnings: {}", report.warnings.join("; "))
        }
    );

    // Which tiers a run can reach is a property of the host, not of this test,
    // and an absent tier is not a failure — but it must not be mistaken for
    // coverage. Record it where it will be read: the x86 CI legs build the
    // avx512 tier, yet a runner whose CPU lacks AVX-512 exercises none of it,
    // and the `avx512` job then proves only that it compiles.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        use archmage::SimdToken;
        report_uncaptured(&format!(
            "cross-tier decode: avx512 tier {} on this host",
            if archmage::X64V4Token::summon().is_some() {
                "AVAILABLE and exercised"
            } else {
                "ABSENT - compiled but never executed here"
            }
        ));
    }
}

/// Decoding the same bytes twice must give the same bytes, including when the
/// work is spread across rayon's thread pool.
///
/// The tier comparison above deliberately runs single-threaded so that the
/// dispatch tier is the only variable. That leaves a gap: if parallel decode
/// were order-dependent, the tier test could neither see it nor be trusted,
/// since it would attribute a scheduling difference to the tier. This closes
/// that gap by holding the tier fixed and varying nothing at all — any
/// difference here is nondeterminism in the decoder, not a tier divergence.
///
/// Without the `threads` feature `parallel` is already false and this reduces
/// to a repeat-decode check, which is still worth running.
#[test]
fn decode_is_deterministic_when_parallel() {
    let root = fixture_root();
    for name in EXACT_FIXTURES.iter().chain(FLOAT_FIXTURES.iter()) {
        let path = root.join(name);
        let data = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));

        let first =
            zenjxl_decoder::decode_with(&data, JxlDecoderOptions::default().with_parallel(true));
        for round in 1..4 {
            let again = zenjxl_decoder::decode_with(
                &data,
                JxlDecoderOptions::default().with_parallel(true),
            );
            match (&first, &again) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(
                        (a.width, a.height, a.channels),
                        (b.width, b.height, b.channels),
                        "{name}: geometry changed between identical decodes (round {round})"
                    );
                    let differing = a
                        .data
                        .iter()
                        .zip(b.data.iter())
                        .filter(|(x, y)| x != y)
                        .count();
                    assert_eq!(
                        differing,
                        0,
                        "{name}: {differing} of {} bytes differ between two identical decodes \
                         (round {round}) — the decoder is not deterministic",
                        a.data.len()
                    );
                }
                (Err(a), Err(b)) => assert_eq!(
                    format!("{a:?}"),
                    format!("{b:?}"),
                    "{name}: identical decodes failed differently (round {round})"
                ),
                _ => panic!("{name}: identical decodes disagreed on success (round {round})"),
            }
        }
    }
}
