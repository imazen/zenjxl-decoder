//! Fuzz crash regression suite (DEDUP-J template, ported from zenwebp).
//!
//! Runs every file in `../fuzz/regression/` through every decoder entry point
//! that has a fuzz target. Each seed file is a previously-found crash that has
//! been fixed; this test ensures none of them re-introduce a panic.
//!
//! Reproduces what the `decode`, `decode_with_limits`, and `probe` fuzz
//! targets (under the top-level `fuzz/` cargo-fuzz workspace) do, but as a
//! regular `cargo test` — no nightly toolchain needed. Failures here mean a
//! regression of a previously-fixed bug.
//!
//! To add a new seed: drop the (preferably minimized) crash file into
//! `<repo-root>/fuzz/regression/` with a descriptive name, no other action
//! required.

use std::fs;
use std::path::PathBuf;

use zenjxl_decoder::api::{JxlDecoderLimits, JxlDecoderOptions};

fn regression_dir() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is the inner crate dir; the fuzz workspace lives
    // alongside it at the repo root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fuzz")
        .join("regression")
}

/// Recursively collect every regular file under `dir`. Skips dotfiles and
/// silently tolerates a missing directory.
fn collect_seeds(dir: &PathBuf, out: &mut Vec<PathBuf>) {
    let read = match fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        match entry.file_type() {
            Ok(t) if t.is_file() => out.push(path),
            Ok(t) if t.is_dir() => collect_seeds(&path, out),
            _ => {}
        }
    }
}

fn run_decode(input: &[u8]) {
    // Mirrors fuzz_targets/decode.rs.
    let _ = zenjxl_decoder::decode(input);
}

fn run_decode_with_limits(input: &[u8]) {
    // Mirrors fuzz_targets/decode_with_limits.rs.
    let mut limits = JxlDecoderLimits::restrictive();
    limits.max_pixels = Some(4_000_000);
    limits.max_memory_bytes = Some(64 * 1024 * 1024);
    let mut options = JxlDecoderOptions::default();
    options.limits = limits;
    options.parallel = false;
    let _ = zenjxl_decoder::decode_with(input, options);
}

fn run_probe(input: &[u8]) {
    // Mirrors fuzz_targets/probe.rs.
    let _ = zenjxl_decoder::read_header(input);
}

#[test]
fn fuzz_regression_seeds_do_not_panic() {
    let dir = regression_dir();
    let mut seeds = Vec::new();
    collect_seeds(&dir, &mut seeds);

    if seeds.is_empty() {
        eprintln!(
            "note: no regression seeds found under {} — nothing to check",
            dir.display()
        );
        return;
    }

    for path in seeds {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>")
            .to_owned();
        let input = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));

        // Each entry point may return Err but must not panic. If any panics,
        // the test fails with the seed name in the unwind message.
        run_decode(&input);
        run_decode_with_limits(&input);
        run_probe(&input);

        eprintln!("ok: {name} ({} bytes)", input.len());
    }
}
