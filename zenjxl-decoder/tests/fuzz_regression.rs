//! Replay seed inputs from `../fuzz/regression/` (top-level cargo-fuzz
//! workspace) through every fuzz target entry point. Shared scaffolding
//! lives in `zen-fuzz-regress`.

use std::path::Path;
use zenjxl_decoder::api::{JxlDecoderLimits, JxlDecoderOptions};
use zenutils_fuzz::RegressionSuite;

/// Lower bound on the replayable seed corpus committed under `../fuzz/regression/`.
///
/// `RegressionSuite` treats a missing or empty seed directory as a clean no-op,
/// so an emptied, renamed, or never-checked-out corpus would let this test pass
/// without replaying a single seed. That risk is sharper here than in the other
/// zen codecs because the corpus lives one level up from this crate, outside
/// `CARGO_MANIFEST_DIR` — a layout change would silently strand the path.
///
/// The CI job's own `ls fuzz/regression/ | wc -l` check (b952a93) does not close
/// this: it counts `README.md` and dotfiles, which the suite skips, so a corpus
/// stripped down to its README passes the workflow and replays nothing. This
/// constant counts only what actually gets replayed.
///
/// Raise this when seeds are added; only lower it when deleting seeds on purpose.
const MIN_SEEDS: usize = 21;

/// Count the files `RegressionSuite::run` will actually replay, using its own
/// filters: recurse into subdirectories, skip dotfiles, `*.md` and `*.txt`.
fn replayable_seeds(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut found = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            found += replayable_seeds(&path);
        } else if path.is_file() {
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".md") && !lower.ends_with(".txt") {
                found += 1;
            }
        }
    }
    found
}

#[test]
fn fuzz_regression() {
    // CARGO_MANIFEST_DIR is the inner crate; the fuzz workspace lives at
    // the repo root, alongside it.
    let seed_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fuzz")
        .join("regression");

    // Fail loudly when the corpus this suite exists to replay is not there.
    let found = replayable_seeds(&seed_dir);
    assert!(
        found >= MIN_SEEDS,
        "{} holds {found} replayable seeds, expected at least {MIN_SEEDS} — \
         the committed regression corpus is missing or was renamed, which would \
         otherwise let this test pass without replaying anything",
        seed_dir.display()
    );

    RegressionSuite::new(seed_dir)
        .target("decode", |input| {
            let _ = zenjxl_decoder::decode(input);
        })
        .target("decode_with_limits", |input| {
            let mut limits = JxlDecoderLimits::restrictive();
            limits.max_pixels = Some(4_000_000);
            limits.max_memory_bytes = Some(64 * 1024 * 1024);
            let mut options = JxlDecoderOptions::default();
            options.limits = limits;
            options.parallel = false;
            let _ = zenjxl_decoder::decode_with(input, options);
        })
        .target("probe", |input| {
            let _ = zenjxl_decoder::read_header(input);
        })
        .run();
}
