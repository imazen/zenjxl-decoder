// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Every executed dispatch tier must produce identical bytes for the same
//! fixture and options, including float rendering and dithered output.
//!
//! Scalar, SSE and WASM fallbacks use fused multiply-add semantics so the
//! operation rounds once, as it does on AVX2, AVX-512 and NEON. The old float
//! envelope admitted one-code differences; this gate now requires exact bytes.
//!
//! Coverage is bounded by the host. Token permutations exercise the tiers the
//! CPU implements; the assertions below reject vacuous native coverage and
//! explicitly report whether AVX-512 actually executed. i686 and WASM have no
//! disable-able token slots and only prove repeat-decode behavior here. Their
//! arithmetic is also checked by the SIMD primitive contract tests.

use std::path::{Path, PathBuf};

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenjxl_decoder::api::JxlDecoderOptions;

/// Integer-oriented fixtures, also exercised with dithering enabled.
const EXACT_FIXTURES: &[&str] = &[
    "3x3_srgb_lossless.jxl",
    "3x3a_srgb_lossless.jxl",
    "gray_alpha_lossless.jxl",
    "squeeze_edge.jxl",
    "grayscale_patches_modular.jxl",
    "orientation5_transpose.jxl",
    "extra_channels.jxl",
];

/// Fixtures exercising float rendering, noise, splines, upsampling and HDR.
const FLOAT_FIXTURES: &[&str] = &[
    "crop2x_small.jxl",
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

/// One decoded grid entry: its label and either pixels or the error text.
type GridResult = (String, Result<Decoded, String>);

/// A whole grid decoded under one token permutation, tagged with that
/// permutation's label.
type PermutationRun = (String, Vec<GridResult>);

/// Keep thread count fixed while varying dispatch. Parallel repeat-decode
/// determinism is exercised separately below.
fn grid() -> Vec<(&'static str, &'static str, JxlDecoderOptions)> {
    let mut out = Vec::new();
    for name in EXACT_FIXTURES.iter().chain(FLOAT_FIXTURES.iter()) {
        let base = || JxlDecoderOptions::default().with_parallel(false);
        out.push((*name, "dither_u8=false", base().with_dither_u8(false)));
        out.push((*name, "default", base()));
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
    for (name, opt_name, opts) in grid() {
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
        out.push((format!("{name} [{opt_name}]"), result));
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
        if let Some(directory) = std::env::var_os("ZENJXL_TIER_OUTPUT_DIR") {
            let directory = PathBuf::from(directory).join(format!("tier-{}", labels.len() - 1));
            std::fs::create_dir(&directory).expect("fresh tier output directory");
            std::fs::write(directory.join("label.txt"), &perm.label).unwrap();
            for (index, (key, result)) in got.iter().enumerate() {
                if let Ok(image) = result {
                    std::fs::write(directory.join(format!("{index}.rgba8")), &image.data).unwrap();
                    std::fs::write(
                        directory.join(format!("{index}.txt")),
                        format!(
                            "{key}\n{} {} {}\n",
                            image.width, image.height, image.channels
                        ),
                    )
                    .unwrap();
                }
            }
        }

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

        for ((key, want), (got_key, have)) in base.iter().zip(got.iter()) {
            assert_eq!(key, got_key, "result order changed between permutations");

            let (want, have) = match (want, have) {
                (Ok(w), Ok(h)) => (w, h),
                // Both failed the same way: agreeing on the rejection is the
                // correct outcome for a stream that should not decode.
                (Err(a), Err(b)) if a == b => continue,
                (Err(_), Err(_)) => panic!(
                    "{key}: both tiers rejected the stream but with different errors, so \
                     which failure a caller sees depends on the CPU.\n  \
                     '{base_label}': {}\n  '{}': {}",
                    describe(want),
                    perm.label,
                    describe(have),
                ),
                _ => panic!(
                    "{key}: the stream decoded on one tier and not the other, so whether \
                     the file is accepted at all depends on the CPU.\n  \
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

            report_uncaptured(&context);
            panic!("{context}\n  Every dispatch tier must produce identical output bytes.");
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
