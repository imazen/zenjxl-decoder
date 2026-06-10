# Public-API Ablation Report: zenjxl-decoder-macros

**Date:** 2026-06-10
**Snapshot commit:** 9b213828c07d (main)
**Items in snapshot:** 3 (default features) / 4 (all features)
**Grep template:** `grep -rn "<symbol>" /home/lilith/work/ --include="*.rs" --exclude-dir=target --exclude-dir=".jj" --exclude-dir="zenjxl-decoder" 2>/dev/null`

---

## Summary

| Total items | Flagged A | Flagged B | Flagged % |
|-------------|-----------|-----------|-----------|
| 4           | 1         | 0         | 25%       |

Note: the 25% figure exceeds the nominal 10% threshold. However, this is a 1-item flag on a 4-item crate. The three unflagged items (`UnconditionalCoder`, `noop`, the module itself) are all consumed internally. The one flagged item is genuinely a test-only leak and the small denominator makes the percentage misleading.

---

## Item table

| Item | Class | Evidence (as of 2026-06-10 scan) | Proposed action | Semver impact |
|------|-------|----------------------------------|-----------------|---------------|
| `#[derive(UnconditionalCoder)]` | KEEP | Used pervasively inside `zenjxl-decoder` src across ~20 files. The macro generates bitstream decoding impls; it is the whole point of this crate. No external consumers found outside jxl-rs forks in `third-party/`. | None | — |
| `#[noop]` | KEEP | Used internally; a no-op attribute macro for conditional compilation guards. No external consumers found. Present in default features. | None | — |
| `for_each_test_file!()` | A | Present only in `all features` (feature-gated). Consumed in `third-party/jxl-rs/jxl/src/api/decoder.rs` and `third-party/jxl-rs/jxl_cli/src/bin/jxlinspect.rs` — these are upstream read-only reference copies, not dependents. Zero hits in `/home/lilith/work/zen/` outside `zenjxl-decoder` itself. This is a test-generation helper that expands to test functions over a corpus directory. It is not part of the decoder contract. | A: add `#[doc(hidden)]` to the macro (or gate behind a `_test-macros` feature so it is excluded from the `all features` surface) | Non-breaking — doc visibility only |
| `pub mod zenjxl_decoder_macros` | KEEP | Module root required to re-export the above. | None | — |

---

## Top 3 highest-confidence ablations

1. **`for_each_test_file!()`** (Class A) — a test-generation macro that emits `#[test]` functions for each `.jxl` file in the corpus directory. It is gated behind a non-default feature and has zero production consumers in the zen org. Adding `#[doc(hidden)]` removes it from rendered docs and signals "internal use only" without a semver break. If a dedicated `_test-macros` feature already exists or can be added (underscore-prefix = excluded from the public-api snapshot), that is the cleaner route.

No further items meet the conservative flagging bar for a 4-item crate.

---

## Notes

This crate's surface is intentionally narrow. `UnconditionalCoder` and `noop` are both consumed exclusively by `zenjxl-decoder`'s internals (bitstream struct derives and conditional-compilation no-ops). There is no realistic scenario where an external crate would derive `UnconditionalCoder` on its own types — the trait it implements (`headers::encodings::UnconditionalCoder`) is not publicly exported from `zenjxl-decoder`. The publish presence of this crate exists only to satisfy `zenjxl-decoder`'s `jxl_macros = { package = "zenjxl-decoder-macros" }` dep; it is effectively an implementation-detail crate.
