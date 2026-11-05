# Decoder API review set — 2026-11-05

The proposals below were reconstructed against runtime main `552da568`, with
test-tool main `43d3069e` included in all except the operational proposal.
Original PR heads remain ancestors; no force push was used. Public API changes
remain in review branches. Main contains the compatible CMS error/profile,
HF-table/metadata-drain and LCMS context-lifetime repairs.

| Review | Proposed contract | Review status / limitation |
|---|---|---|
| [60: Send](https://github.com/imazen/zenjxl-decoder/pull/60) | Require `JxlCms: Send`; move incremental decoder states between threads | Ready for API review; breaks existing non-Send providers. No Sync provider bound. |
| [68: source metadata](https://github.com/imazen/zenjxl-decoder/pull/68) | Extra-channel precision/spot/CFA details, timecodes, fallible source-layer snapshot and frame selection | Ready for API review; composed pixel behavior is unchanged. |
| [74: operational limits](https://github.com/imazen/zenjxl-decoder/pull/74) | Preserve native stop reasons and add an optional frame-header budget | Ready for API review; default budget remains None and coarse cancellation classification remains unchanged. |
| [66: layout](https://github.com/imazen/zenjxl-decoder/pull/66) | Opt-in bounded codestream spans and seek-stable records | Ready for API/cost review; disabled header scans regress on two measured fixtures. |
| [75: local F32 layers](https://github.com/imazen/zenjxl-decoder/pull/75) | Qualified local samples before composition, with explicit unsupported-mode errors | Draft, stacked on 68; broader blend/reference qualification and RGB bit exactness remain incomplete. |

Review 60, 68 and 74 independently, then 66's span/lifetime API and measured
cost. Review 75's constrained API separately; it does not establish general
raw-layer completeness. Review branches contain their detailed contracts and
reproduction recipes under `docs/api-review/` and `justfile`. Check each PR's
current CI rather than treating these local results as platform-wide coverage.

## LCMS ownership and thread migration

The CLI provider is already Send + Sync. lcms2 6.2.0's transform carries only a
context type marker; it does not own the actual ThreadContext. Main now retains
the context after the transform field so destruction runs in the required
order. ThreadContext is Send rather than Sync; private Mutex ownership retains
the transform callback's existing Send + Sync contract without unsafe code.

Both CLI feature configurations pass seven tests, including an initialized
transform moved across four threads with identical float bits and destroyed on
another thread. Candidate and baseline Memcheck identity controls both report
zero errors; no reproduced invalid memory access is claimed. The unchanged
whole-image LCMS CMYK golden is exact. The private repair is `552da568`.

This ownership repair alone does not make `Box<dyn JxlCms>` Send. Proposal 60
supplies that separate trait-object bound and verifies real LCMS-backed decoder
migration through header, frame, render and drop states. Its coverage workflow
also prepares the required corpus references using pinned libjxl source;
missing-reference cases are not converted into silent skips.

## Fresh qualification and cost findings

Metadata's standalone test crate generates inputs using the pinned native
encoder and checks them with a libjxl v0.12 C API reader. Four independent tests
pass for seven extra-channel roles, precision/exponent details, spot/CFA,
present-zero versus absent timecodes and reverse seek selection. The layer
snapshot preserves signed local facts while composed output retains canvas
geometry; its blend-list allocation is fallible.

Layout's current-source full all-feature workspace, both feature-specific
contract gates and strict all-target Clippy pass. Fresh exploratory 30-round
zenbench pairs compare the disabled candidate with pinned runtime main in one
process. Both arms scan headers and skip pixels, on r5900xt CPUs 0–3:

| Fixture | Baseline mean | Candidate mean | Paired change (95% CI) |
|---|---:|---:|---:|
| Basic | 2.78 µs | 2.99 µs | +7.72% [+7.10%, +8.32%] |
| Layered | 9.68 µs | 10.07 µs | +4.79% [+4.15%, +5.45%] |
| CMYK | 15.19 ms | 14.86 ms | No significant difference; CV 15.2% / 17.5% |

This measures the complete disabled-layout change. Whole-image decode cost
remains unmeasured. The raw pairs, command and pins live on the layout review
branch in `benchmarks/layout_disabled_2026-11-05.txt`.

Local F32 qualification against checksum-pinned libjxl source
`a7a9c787341cf703dede03c2009fa460cae5e5df` finds:

- RGB/alpha: five layers, 18,874,368 samples, 2,265,549 different float bit
  patterns; maximum absolute error 0.00036650896072387695. Every sample passes
  the unchanged 1/256 F32 bound. Alpha samples are exact; RGB is not bit-exact.
- CMYK/alpha: four layers, 1,587,120 samples, zero different float bits.
- Geometry, channel and blend/reference metadata agree exactly. A one-bit CMYK
  mantissa corruption is rejected even below the lossy float tolerance.
- Native renderer/chunk comparisons require exact float bits and preserve
  strided-row padding. Both feature configurations and strict Clippy pass.
  The full all-feature workspace passes, and all six frozen downstream RGBA
  frames remain byte-exact.

The combined raw/layout full-suite, frozen-reference and benchmark wrapper
reports `rc=0 567s | peak-RSS 7.29GiB | min-avail 36471MiB | peak-load 7.03`.
This is a validation process-tree receipt, not decoder-only memory usage.
No pre-existing assertion, accuracy threshold, ignore or corpus policy is
weakened. Existing binary fixtures on the original PRs remain unchanged; no
newly generated binary fixture is committed.

## Earlier PR dispositions

[61](https://github.com/imazen/zenjxl-decoder/pull/61) remains an engineering
adaptation: preserve current transform limits/render scheduling, account for
LUT/cache memory, then rerun parity and paired performance. No speedup over
current main has been measured. Its historical benchmark receipts do not
establish one.

[65](https://github.com/imazen/zenjxl-decoder/pull/65)'s compatible HF-table
budgeting and valid metadata draining are already on main. Its blanket trailing
error recovery remains excluded because it suppresses I/O and structural
errors. Optional malformed-metadata recovery needs a separate policy and
implementation. Do not merge the full old PR as a substitute for that work.

The nine-audit precision repairs remain on main. The large 64×64 AVX2 IDCT
kernel measured 9.4 µs versus 3.8 µs for the f32 baseline while retaining the
established accuracy gate; whole-decoder impact remains unmeasured. These API
reviews do not change that kernel or its threshold.
