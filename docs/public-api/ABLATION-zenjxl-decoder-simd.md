# Public-API Ablation Report: zenjxl-decoder-simd

**Date:** 2026-06-10
**Snapshot commit:** 9b213828c07d (main)
**Items in snapshot:** 352 (default features) / 486 (all features)
**Grep template:** `grep -rn "<symbol>" /home/lilith/work/ --include="*.rs" --exclude-dir=target --exclude-dir=".jj" --exclude-dir="zenjxl-decoder" 2>/dev/null`

---

## Summary

| Total items | Flagged A (crate-level) | Flagged B | Flagged % |
|-------------|-------------------------|-----------|-----------|
| 486         | All (crate-level)       | 0         | n/a (see below) |

The instruction says: ">10% of a crate flagged → bar too low, re-filter. Exception: `zenjxl-decoder-simd` MAY legitimately exceed 10% if evidence shows the whole crate is parent-only — then propose the crate-level 'internal, no semver promises' banner instead of itemized flags, which is the conservative move."

This report proposes the crate-level banner. See rationale below.

---

## Crate role analysis

`zenjxl-decoder-simd` is the SIMD substrate crate for `zenjxl-decoder`. Its relationship is `jxl_simd = { package = "zenjxl-decoder-simd", path = "../zenjxl-decoder-simd", version = "=0.3.9" }` — pinned to the exact same version with a path dep. This is a canonical "sibling implementation crate" pattern.

**External consumer check (as of 2026-06-10 scan):**

```
grep -rn "zenjxl.decoder.simd\|zenjxl_decoder_simd\|SimdDescriptor\|F32SimdVec\|I32SimdVec\|SimdMask\|U8SimdVec\|U16SimdVec\|U32SimdVec\|AvxDescriptor\|Sse42Descriptor\|Avx512Descriptor\|ScalarDescriptor" \
  /home/lilith/work/ --include="*.rs" --include="Cargo.toml" \
  --exclude-dir=target --exclude-dir=".jj" --exclude-dir="zenjxl-decoder" 2>/dev/null \
  | grep -v "third-party/"
```

Result: **zero hits** outside `zenjxl-decoder` itself and the `third-party/jxl-rs` read-only upstream copies. No crates in the zen org (`zenjxl`, `jxl-encoder`, `zenpipe`, `zencodecs`, or any others) import or name any type from `zenjxl-decoder-simd`.

---

## What the crate publishes

The 486 items (all-features) comprise:

| Module / surface | Items | Nature |
|-----------------|-------|--------|
| `float16::f16` + impls | ~30 | An f16 type; all methods are standard numeric conversions |
| `scalar::ScalarDescriptor` + impls | ~20 | Scalar fallback descriptor |
| `SimdDescriptor` trait + impls | ~50 | Core dispatch trait |
| `F32SimdVec`, `I32SimdVec`, `SimdMask`, `U8/16/32SimdVec` traits + impls | ~150 | SIMD vector operation traits + scalar impls |
| `AvxDescriptor`, `Sse42Descriptor` (default features) | ~40 | x86 ISA descriptors |
| `Avx512Descriptor` (avx512 feature) | ~35 | AVX-512 descriptor |
| `bench_all_instruction_sets!`, `test_all_instruction_sets!`, `simd_function!`, `shl!`, `shr!` macros | 5 | Test/bench generation and SIMD helper macros |
| NEON / WASM descriptors (non-default features) | ~100 | ARM and WASM backends |
| Concrete SIMD vector types per ISA (`F32VecAvx`, `I32VecSse42`, etc.) | ~50 | Backend impl types |

All of this is the SIMD dispatch infrastructure that `zenjxl-decoder` uses internally for its decode kernels (DCT, dequantization, color conversion, etc.). None of it is part of the decoder's user-facing contract.

The 134 extra items in `all features` vs `default features` are the avx512 and platform-specific backends — again all internal.

---

## Proposed action (Class A — non-breaking, crate-level)

**Add a prominent `# Stability` section to `zenjxl-decoder-simd/README.md` and to the crate `lib.rs` top-level doc comment:**

```
## Stability

`zenjxl-decoder-simd` is an implementation-detail crate for `zenjxl-decoder`.
It is published on crates.io only because Cargo requires all workspace members
in a published dependency graph to be themselves published.

**No semver guarantees are made.** Any type, trait, or macro in this crate may
change or be removed in any patch release. Do not depend on this crate directly.
Use `zenjxl-decoder` instead.
```

Additionally, add `#[doc(hidden)]` to the crate root in `lib.rs` so the entire crate's contents are excluded from docs.rs rendering (making the instability self-evident to any user who navigates there).

This is **Class A** (non-breaking). It does not remove any items, does not change any signatures, and does not break existing callers (of which there are none outside the repo).

No itemized B-class (breaking) removals are proposed. Demoting individual items to `pub(crate)` is impossible across crate boundaries; the crate-level banner is both correct and sufficient.

---

## Top 3 highest-confidence observations

1. **Crate-level internal-banner (Class A):** The entire `zenjxl-decoder-simd` surface is consumed exclusively by `zenjxl-decoder`. Zero external org hits. The crate is published because Cargo requires it, not because it offers an external API. A README and lib.rs doc banner removes the implicit stability promise without any semver impact.

2. **`bench_all_instruction_sets!` / `test_all_instruction_sets!` macros (would be Class A/B):** These expand to benchmarks and tests over all ISA tiers. They have no production use outside the crate's own test suite. If the crate-level banner is added, these are covered. If itemizing, add `#[doc(hidden)]` individually.

3. **`for_each_test_file!`-analogue macros `simd_function!` / `shl!` / `shr!`:** These are SIMD dispatch helpers used by the decoder's kernel authors. Zero external hits. Covered by the crate-level banner.

---

## Notes

Cross-crate visibility reduction (`pub` → `pub(crate)`) is not achievable for a published crate without splitting the codebase — the two crates are separate compilation units and `pub(crate)` only scopes to the defining crate. The correct mechanism for an internal-crate situation is the stability banner and `#[doc(hidden)]`, which flags "no promises" to any downstream that reaches in despite the warning. This is the approach used by several well-known Rust crates with internal-crate helper deps (e.g. `syn-utils`, `tokio-macros` before they stabilized).
