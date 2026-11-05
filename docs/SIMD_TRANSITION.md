# SIMD descriptor transition and remaining work

Observed 2026-11-05 against main `b04af24c` and the review below. This is a
source/work inventory, not release qualification.

[PR 77](https://github.com/imazen/zenjxl-decoder/pull/77) adds token-free
feature-context construction and deprecates retained token constructors and
accessors. It is open and assigned for API review. These additions are not a
published release. Verify the consumer's resolved source before migrating;
no removal, merge or publication is authorized by this work.

## Calling the API

Baseline code uses `SimdDescriptor::new()` for runtime detection. Inside a
function with the complete CPU feature set, use a concrete descriptor's
`from_context()` or `target_features::descriptor_v2`, `_v3`, `_v4`, `_neon`
or `_wasm128`. Keep tokens inside the library and compose the existing vector
traits over arrays/slices. `ScalarDescriptor` remains a unit constructor.
The checked [module example](../zenjxl-decoder-simd/src/target_features.rs)
shows an array-in/array-out operation without a token in its signature.

Native E0133 means the calling function lacks a proven feature context;
a runtime check alone does not discharge that requirement. Use a matching
feature function or runtime construction, not an unsafe block. WASM permits
safe baseline calls because the engine validates module features on load.
SIMD128 still requires an engine supporting it.

V4 needs all V3 features, `aes`, `pclmulqdq`, and `avx512f,avx512bw,avx512cd,
avx512dq,avx512vl`. Full requirements live in the module. Internal descriptors
still store proof tokens privately. Legacy signatures and semantics remain;
new consumer operation boundaries should use arrays or correctly described
slices/strided rows. There is no new inlining, zero-copy or speed guarantee.

## Validation

All five SIMD descriptor tiers have constructor/vector bit-parity coverage:
V2, V3, V4, NEON and WASM128, plus the unchanged scalar constructor. Named
tiers are explicitly required in validation. Native E0133/safe-composition
doctests and a passing WASM baseline-call integration case are included.

Native default/no-default tests, strict lint/docs, ARM/WASM test typechecks,
i686 no-default library checks and complete x86/ARM snapshots passed locally.
The full decoder consumer library also passed strict all-feature lint with
its existing dependency graph. Existing assertions, tolerances and ignores
remain intact. CI must be checked on the current review head; a local pass
is not a claim that every external corpus or platform was exercised.

Use `just simd-api-stage <new-directory>` and
`just simd-api-check <stage>/Cargo.toml` for isolated primitive checks.
`just simd-api-doc-check` verifies this architecture's complete inventory in
`docs/public-api/{x86_64,aarch64}/`. The snapshot runner is excluded from the
product workspace and pins the verified, not-yet-published zenutils 0.1.2
implementation by immutable git revision.

## Branch and issue inventory

The live inventory contains 48 remote branches, including main and PR 77,
and 23 local branch heads. Thirteen remote tips are ancestors of main; 35 are
not. Several divergent tips link to merged PRs, including builder, README,
limits and lean-scan work. A divergent tip does not establish missing
functionality. Full names, SHAs, graph counts, PR states and local refs are
retained privately by `just review-inventory <private-directory>`.
No branch or issue was deleted or closed.

Every remote tip is accounted for by these mutually exclusive dispositions:

| Disposition | Branches | Action |
|---|---:|---|
| Ancestor of main, including main | 13 | Already reachable; retain refs. |
| Open PR | 4 | Review 61, 66, 75 and 77 independently. |
| Divergent tip with a merged PR | 9 | Compare residual patches; do not replay landed features. |
| Closed alternative | 4 | Preserve the decision and work; no reopening implied. |
| Superseded | 6 | Preserve history; compare residual work before adoption. |
| Preserved experiment or residual | 12 | Requalify against current contracts before proposing adoption. |

The 23 local heads are retained and included in the private evidence, along
with all local remote-tracking refs and jj bookmarks.

Four local heads have tips absent from the live remote branch list. They remain
preserved, with these dispositions:

- `abandoned/pipelined-batch`: one abandoned rayon pipeline commit; no approval
  to revive it.
- `alternate-simd/eliminate-jxl-simd`: residual cleanup/dependency work on the
  closed replacement approach; compare individual changes before adoption.
- `security-fixes-h1-h4`: six historical allocation/bitreader/revert commits;
  verify each against current bounds and allocation handling before replaying.
- `superseded/safe-parallel`: nine historical ownership/parallel-render and
  benchmark commits; preserve as superseded experiments, with fresh parity and
  performance qualification required for any extracted change.

The other local tips match live remote tips in the capture. Local aliases do not
create additional approved work.

Other open PRs remain separate:

- [66](https://github.com/imazen/zenjxl-decoder/pull/66): bounded codestream
  layout records; preserve the disabled-path cost and seek/fragmentation gates.
- [75](https://github.com/imazen/zenjxl-decoder/pull/75): local F32 layers;
  preserve representation, compositing, arithmetic and stride qualification.
- [61](https://github.com/imazen/zenjxl-decoder/pull/61): fused prefix decoding;
  adapt current transform limits, memory accounting and rendering before
  claiming a speedup against current main.

[PR 68](https://github.com/imazen/zenjxl-decoder/pull/68) and
[PR 74](https://github.com/imazen/zenjxl-decoder/pull/74) are merged. Older SIMD
replacement branches and closed PRs 3/4/5 are preserved alternatives, not
approval to replace the current descriptors. The older sans-I/O, progressive
preview, stride-constructor, parallel and entropy experiments need individual
contract review against current typestate/seek APIs. Preserve rejected and
superseded research rather than replaying it wholesale.

All 10 open issues were inspected; none was closed from source inspection:

| Issue | Current source fact and remaining gate |
|---|---|
| [73](https://github.com/imazen/zenjxl-decoder/issues/73) | PQ near-black reference-build disagreement remains an attribution task; preserve the existing numeric gate. |
| [72](https://github.com/imazen/zenjxl-decoder/issues/72) | CMS error/profile repairs are on main; reconcile the issue against retained callback and independent CMYK evidence. |
| [71](https://github.com/imazen/zenjxl-decoder/issues/71) | Main exposes `max_frame_headers`; retain hidden-frame, retry, seek/reset and admitted-pixel qualification. |
| [70](https://github.com/imazen/zenjxl-decoder/issues/70) | Main exposes `Error::Stopped` and `stop_reason()`; retain actual polling and error-chain qualification. |
| [69](https://github.com/imazen/zenjxl-decoder/issues/69) | `FrameHeader::postprocess` now always applies source dimension shifts; reconcile the complete auxiliary-role/shift qualification. |
| [67](https://github.com/imazen/zenjxl-decoder/issues/67) | Source precision/spot/CFA details and timecodes landed with PR 68; raw pixel behavior remains separate on PR 75. |
| [64](https://github.com/imazen/zenjxl-decoder/issues/64) | HF table budget guards and valid codestream draining are on main. Broad malformed-box recovery remains excluded. |
| [63](https://github.com/imazen/zenjxl-decoder/issues/63) | ICC short-read handling changed on main; recheck the issue's retained large/compressible profiles and explicit limits before closure. |
| [57](https://github.com/imazen/zenjxl-decoder/issues/57) | `coalescing=false` remains documented as unimplemented. Review PR 75's constrained layer API separately; do not silently remove the public option. |
| [18](https://github.com/imazen/zenjxl-decoder/issues/18) | The opt-in uninitialized allocation remains behind `allow-unsafe`; changing it is separate policy/API work. |

See the existing [PR triage](PR_TRIAGE_2026-11-05.md) and
[API review](API_REVIEW_2026-11-05.md) for their measured contracts and limits.
These source observations do not rerun or supersede those measurements.

## Source markers and guidance

The initial tracked-text scan found 65 TODO/FIXME matches, including historical
notes. The private inventory retains every location. Current groups are:

- Safety/correctness qualification: downsampled-extra-channel patch checks,
  spline derivative limits, render-stage error handling, and color metadata
  range/precision questions. Preserve current numerical expectations.
- Resource/performance work: incremental permutation parsing still carries a
  quadratic-input note; coefficient storage, transform scheduling, buffer
  initialization and scratch/row bookkeeping have separate follow-ons.
- Test coverage: entropy/LZ77/Huffman, noise, corpus breadth and frame skipping
  have remaining source notes. Existing missing-file early returns in older
  diagnostic tests need explicit caller-visible corpus policy; they are not
  evidence that those fixtures were tested here.
- CLI output: EXR alpha association and white-luminance metadata have separate
  markers. They do not belong in the SIMD constructor migration.

The old streaming-test note names missing `JxlDecoderSkipFrames` compatibility,
not an absence of current seek support: `scanned_frames()` and
`start_new_frame(seek_target)` already exist. Keep those contracts distinct.
