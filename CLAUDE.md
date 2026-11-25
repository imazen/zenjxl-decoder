# Decoder investigation notes

## Modular inverse RCT input grids — 2026-11-25

The inverse reversible color transform reads the same tile index from
three channels. Their dimensions, shifts and bit depths are checked before
grid assignment. Grid construction now checks that all three inputs also
use the same grid kind and returns `Error::MixingDifferentChannels` on a
mismatch. The error propagates through frame setup before tile allocation.

A scheduler test exercises global, LF and HF placement and two mismatched
assignments directly in memory. It checks the propagated output grids and
the typed refusal. Codestream conformance for mixed-grid RCT remains open.

## Direct Modular RGB8 builder test — 2026-11-25

The builder test used a parallel-rendering accessor and failed to compile
without `threads`. It now inspects the finished stages through a test-only
accessor in every feature configuration. The same assertion checks that
identity RGB selects the direct save and that BGR and flipped RGB do not.

## Resolved: selected-level Black, MA-tree and HF bounds — 2026-11-19

Libjxl v0.12's encoder rejects Level 5 with a Black extra channel. The old
decoder accepted a native 4×3 CMYK/ICC control at Level 5. Level 5 now refuses
Black while Level 10 admits the existing CMYK fixture. The LCMS whole-image
and thread migration goldens retain their exact hash with Level 10 selected.

Both MA-tree readers enforce the selected root-to-leaf depth, with boundary
checks at 64/65 and 2048/2049 edges. A 131-node entropy-coded tree exercises
the inclusive node cap and Level 5 rejection. The global cap uses coded
dimensions and widened arithmetic.

Both VarDCT paths check each shifted HF pass value and cumulative coefficient
at the selected signed width before narrowing. Direct boundary controls cover
16/32-bit endpoints, shifted values and progressive sums. The high-width
boundary controls exercise the coefficient step rather than a complete stream.
The release workspace passes with 1,123 library passes and 29 existing ignores.
All six frozen v0.12 RGBA frames retain exact bytes. The decoder tests and
CMYK CLI checks pass with all features and no default features.

## Repaired: JPEG empty DHT and lost reconstruction errors — 2026-11-18

Native v0.12 empty-DHT streams render but fail public reconstruction because
the JBRD reader rejects the legal all-zero record. A zero-symbol DC3 table is
a distinct, passing control. A truncated optional JBRD is also silently treated
as absent by checked reconstruction. Fresh committed-source before evidence
and candidate controls are retained in [the DHT contract](docs/JPEG_DHT_CONTRACT.md).
The repair passes the broad Rust 1.100 feature/scalar/threaded and strict lint
gates, selected native DC/AC reconstruction, malformed-record/error-boundary
controls, five deliberate faults and the changed production-region windows.
The encoder pin update remains separate work.


## Resolved: upstream prefix-test flush schedule — 2026-11-17

The prefix comparison's first adaptation flushed after every chunk, while
upstream's DecodeParams defaults flush only at the supplied prefix end. ARM CI
exposed a retained LF preview at squeeze_empty_residual prefix 470/chunk 217;
the same case reproduces on x86. The test helper now selects its flush schedule
explicitly. Prefix comparisons retain the upstream 5e-3 and pixel-recovery
assertions; existing truncated-input tests retain intermediate flushes. A
separate positive preview comparison uses matching schedules. Production
rendering is unchanged by this test correction.

## Resolved: external LCMS intensity target — 2026-11-17

The CLI LCMS provider accepted an intensity argument but generated structured
ICC profiles through no-argument conversion. The 1000-nit PQ control returned
white [0.8704195, 0.8705389, 0.87049097], outside the unchanged 0.01 bound around
[1, 1, 1]. The owner-approved additive profile method now exposes intensity-aware
conversion, retaining borrowed ICC bytes and existing no-argument defaults.
LCMS white is now [0.9999189, 1.000056, 1.0000011]. All nine original HDR bounds
and the exact CMYK golden pass in both CLI feature configurations. The API
addition follows the required PR path. `just review-intensity-check` reproduces
the profile, HDR, CMYK and strict Clippy gates.

The combined level-validation run also identified 29 positive fixture tests
using Level 5 for streams declaring 32-bit Modular buffers. Released libjxl's
VerifyLevelSettings identifies that flag as requiring Level 10. Feature/pixel
helpers now select Level 10 explicitly; their existing numerical, file-length
and pixel assertions remain unchanged. The independent Level 5 rejection and
boundary controls remain active.

## Resolved: cumulative frame and metadata consistency — 2026-11-17

The Level 5 zero-duration area counter is separate from the frame-header work
budget. Boundary/reset controls and the existing seek/work-budget suite pass;
seek restores the counter snapshot and rewind clears stream state. Preview
frames are excluded, and Level 10 has no cumulative cap. The boundary control
does not decode a full-size frame sequence.

The core parser validates grayscale ICC correspondence and Black/CMYK profile
correspondence, including an encoded Black-without-ICC negative control.
Container admission rejects a second gain-map box but permits rereading the
same box position after seeking. Required external JPEG Exif/XMP/ICC data now
fails through the existing reconstruction error path when absent; present and
inline payloads retain their bytes. Five focused metadata controls pass.

## Resolved: structured XYB descriptor parsing — 2026-11-17

The ten-bit XYB/perceptual ColorEncoding bundle previously returned
InvalidColorEncoding and left have_gamma false. Its transfer-function bundle
now consumes no bits and initializes have_gamma=true, gamma=3333333. The valid
wire control reaches the existing public XYB variant; Relative intent remains
an error. This establishes header parsing and representation, not exhaustive
whole-image color accuracy. Unknown color/transfer handling still needs a
truthful source API and management policy.

## Resolved: before-transform extra-channel references — 2026-11-17

A procedural stream saves a full-canvas before-transform reference and adds
its alpha to the next frame. Pinned djxl v0.12 accepts it; the former shared
guard rejects it with BlendingPreColorTransform(1). Extra-channel references
now retain private origin metadata and validate covering geometry and planes.
Color reads remain protected; full-canvas color Replace does not read a source.
Both renderers and whole-file/1/7-byte input pass the exact doubled-alpha check.
The 36 RGBA8 output bytes match the independent decoder exactly. The old EC
rejection expectation described the over-rejection and is replaced with valid
EC/header geometry controls; the unchanged color-rejection contract now has
an explicit Add operation in its test.

## Resolved: parallel partial-prefix rendering — 2026-11-17

The adapted upstream partial-prefix regression renders the 29,726-byte bicycle
prefix sequentially but leaves NaNs with parallel decoding and 2,675-byte input
chunks. Parallel progressive flushes also differ from sequential snapshots.
The batch renderer previously omitted intermediate flush work. Frames now pick
one rendering strategy before the first HF decode/flush: partial frames use
the incremental renderer and complete buffered frames can retain batch rendering.
The strategy stays fixed after groups are stored. Thirteen selected release
controls pass, including the ordinary-fixture prefix sweep and exact flush
offset/frame/sample comparisons on four fixtures. Final pixel gates remain
exact. Streaming throughput is unmeasured.

## Resolved: selected-level header bounds — 2026-11-17

The parser previously applied a fixed 2^30 image-area cap but did not enforce
selected-level image/frame dimensions, area, sample precision or reconstructed
ICC size. Header-local global limits now permit Level 10 sizes, with separate
selected-level and caller resource checks before setup. Level 5 also requires
the modular 16-bit-sufficient flag. The ICC predictor preamble checks its output
limit before allocating the profile; its encoded-stream cap remains separate.
`just review-level-check` exercises exact boundaries, extra-channel precision
and an accepted header-only 2^31-area Level 10 image without allocating pixels.
The new dimension rejection failed before the repair. These controls do not
establish exhaustive level conformance or cumulative frame accounting.

## Upstream regression import validation — 2026-11-17

The integrated release all-feature workspace passes: the decoder library
reports 1,088 passes with 29 existing ignores; integration, CLI, SIMD and
doctest gates also pass. Caller-provided codec-corpus inputs and the two
required PNG references were present, with those reference/input hashes
verified. This is regular-suite coverage, not all ignored conformance tests.
Strict all-feature/all-target workspace Clippy and scoped formatting pass.
The no-default-feature library Clippy gate and all 41 selected import/boundary
regressions pass. The native i686 overflow reproduction remains its separately
executed before/after control; final platform CI is tracked independently.

Six frozen libjxl v0.12 RGBA frames have zero differing bytes: both crop
animation frames, extra-channel shift, squeeze corner patch and both reordered
multiframe frames. The comparison verifies input/reference SHA-256 and keeps
both renderers and chunk controls exact. Reproduce with the `upstream-import-*`
and `upstream-frozen-oracle` justfile targets. Fifteen imported fixture files
retain upstream blob and SHA-256 identities; the sync ledger records test
adaptations, absent APIs and remaining validation questions. No new ignore,
silent missing-file skip or relaxed numerical/pixel tolerance was added.

## Resolved: rectangle byte offsets and clipping overflow — 2026-11-17

The upstream byte-offset regression does not panic in the old release build:
an out-of-range f32 rectangle wraps back into the image. The large-coordinate
clip test returns width 0 instead of 90, and panics with debug overflow checks.
Byte conversion now checks multiplication in every build, and rectangle ends
saturate for clipping. Debug controls pass after the repair. These are direct
rectangle API tests, not a demonstrated conformant encoded-stream exploit.

## Resolved: patch-coordinate arithmetic bounds — 2026-11-17

The imported upstream delta-bound test reproduces negation overflow for
i32::MIN in the former arithmetic, and its usize::MAX positive-bound control
fails in release mode. Patch positions now use checked signed addition;
reference/destination rectangle ends also use checked addition. Both valid
delta controls retain their exact results. The debug regression passes after
the repair. These are direct helper/coordinate controls; no complete encoded
stream reaching the artificial usize::MAX helper input is claimed.

## Resolved: Huffman allocation guard on an incomplete prefix — 2026-11-17

The full suite after byte-accurate refill reproduced
`AlphabetTooLargeHuff(520)` in chunked fixture decoding and the progressive LF
header-budget test. The input-proportional allocation guard treated a short
prefix as a fatal alphabet error. It now requests enough buffered bytes to
satisfy the same cap before allocating. The two-byte header-fragment test
requires a one-byte refill and retains a fatal hard-alphabet-limit control.
The formerly failing LF header-budget control passes. No allocation cap was
removed or raised.

## Resolved: inclusive prefix alphabet endpoint — 2026-11-19

A complete 19-byte Modular RGB stream with alphabet size 32,768 returned
`AlphabetTooLargeHuff(32768)`. ISO/IEC 18181-1:2024 C.2.1 includes that value.
The inclusive check now matches libjxl v0.12. The compact simple table decodes
all 36 F32 samples exactly with either renderer in whole-file, 1-byte and
7-byte chunks. Size 32,769 still fails. Simple tables use fixed-size storage.
The buffered-input allocation guard retains its budget for complex tables,
and the total alphabet cap still applies. See `docs/UPSTREAM_SYNC.md` and
`just review-prefix-alphabet-check`.

## Resolved: sprite-sheet F32 noise rounding — 2026-11-19

The first six sequential frames of `animation_seek_sprite_sheets.jxl`
differed from pinned libjxl v0.12 AVX2 at 207 RGB F32 samples. The first
eight convolved noise samples matched exactly. Fusing the source's final
noise update alone left every differing sample unchanged. The compiled
reference also fuses the red product into the sum and difference after
rounding the green product. Making that order explicit gives exact frame
hashes for all 12 frames with both renderers, three chunk sizes and both
feature configurations. `just review-noise-check` retains the previous noise
kernel expectations and the independent animation regression. The golden is
specific to the pinned reference build.

## Resolved: frame-header dimension validation — 2026-11-17

Upstream's frame-dimension regression first accepted a zero width in the
header check. The guard now rejects zero dimensions, either dimension above
2^30 and area above 2^40 before frame setup. The original negative cases and
positive exact global boundaries pass without allocating those images.
Selected-level dimensions and cumulative animation bounds remain distinct
validation work; this change does not claim those complete.

## Resolved: bit-reader refill deficit units — 2026-11-17

The upstream byte-deficit regression fails before the repair: skipping 17
bits from a one-byte input returns a deficit of 9 rather than 2. Both skip
error paths and optimistic read-error reporting now convert missing bits to
bytes, matching ordinary reads and the caller's refill contract. All seven
bit-reader-filtered controls pass. This direct reader test does not by itself
establish a separate full-stream failure.

## Resolved: CMS HDR profile intensity and interpolation — 2026-11-17

The imported upstream 1000-nit PQ test reproduced white 0.86840016 where its
unchanged tolerance requires 1.0 within 0.01. MoxCMS ignored intensity_target,
and profile generation used the older coarse Lab LUT. The upstream mAB/XYZ
builder and private intensity plumbing pass all four original CMS tests.
The public no-argument profile generator retains its 10000-nit default.
The upstream CLUT macro test now requires exact scalar/SSE4.2/AVX2 table bytes
at 1000 and 10000 nits, retaining its fixed black/gray/white controls. AVX512
execution is not claimed. All 27 color/profile checks and strict all-feature
workspace Clippy pass. Existing numerical expectations remain unchanged;
profile-shape expectations follow the new mAB/XYZ representation.

## Resolved: color descriptor boundaries and DCI ICC tagging — 2026-11-17

Three direct before/after regressions reject invalid grayscale white points,
reject encoded primary determinants below 1e-10, and accept the actual
s15Fixed16 endpoints. Exact integer-coordinate determinant checks preserve
the inclusive boundary and imaginary XYZ primaries. The generic matrix
inverse threshold is unchanged. Relative DCI ICC export omits CICP17 because
that code point's extra scaling contradicts the retained relative TRC.
The tag-table regression retains positive sRGB/PQ/HLG controls. These are
header-check and generated-profile controls; they do not establish a
conformant encoded-stream witness for every boundary.

## Resolved: playback after seeking past a live dictionary — 2026-11-17

Upstream's mixed reverse/forward seek test failed with
`PatchesInvalidReference(1)` on `animation_seek_noise_references.jxl`.
The fork rebuilt empty decoder state and replayed only dependencies read by
the target header. Later frames can still need older saved dictionaries.
Seek starts now restore every live reference and LF slot, preserving exact
continuation pixels. This can increase replay work; cached/minimal seek costs
are not implemented or claimed. The nine-fixture reused/fresh seek control,
interrupted seek, reset, scan-only and unaligned RGB16 checks pass, as do all
eleven existing start-new-frame tests in release mode. No pixel tolerance
or established expectation changed.

## Resolved: gain-map alternate profile validation — 2026-11-17

`GainMapBundle::parse` previously retained malformed ColorEncoding bytes and
accepted a missing required ICC. It now validates both complete fields using
the existing header and ICC readers, retaining the public encoded-byte fields.
Upstream's five profile fixtures and two semantic tests are adapted without
new public API. Real-profile round trips replace arbitrary invalid profile
bytes in old tests; 19 focused checks pass, including consistency, trailing
data and fatal truncation controls. This does not validate the gain-map image
codestream or ISO 21496-1 metadata contents.

## Resolved: VarDCT nonzero-count overflow — 2026-11-17

The imported upstream `08bb971` regression reproduces an addition-overflow
panic on native i686 before rejecting a u32::MAX coded nonzero count.
Widening the bound to u64 returns the existing `InvalidNumNonZeros` error.
The original 182-byte fixture and exact error assertion are retained in
`src/tests/upstream_2026_11.rs`; the i686 before/after control executed.

## RESOLVED: global inverse-squeeze grid transition — 2026-11-17

[Encoder issue 144](https://github.com/imazen/jxl-encoder/issues/144) retains
valid tall and transposed Gray8 streams whose deeply squeezed small bands are
coded globally. The original scheduler forced an HF grid when an intermediate
inverse output grew past group size, even when both inputs remained global.
The resulting zero-sized group view panicked at the row bound. Intermediates
now retain the maximum input grid; a coded LF/HF residual introduces its grid.

Both text fixtures recover every source grayscale and opaque-alpha code under
both dithering modes. Original-source mutation replay reproduces both panics.
Pinned v0.12 native metadata and exact djxl samples supply independent controls.
All-feature, scalar and parallel/SIMD release workspace suites and both strict
Clippy permutations pass with existing expectations/ignores unchanged. The
integrated run took 975 seconds, run-heavy peak RSS 8.56 GiB, minimum available
13,917 MiB. The exact instrumented object/profile covers 4/4 mapped regions in the changed
squeeze propagation expressions without LLVM mapping diagnostics. Other
transform arms and allocation scheduling are outside that fraction. Remote
CI retains a separate verdict.
[Mechanism, reproductions and limits](docs/GLOBAL_SQUEEZE_GRIDS.md).

## RESOLVED locally: Rust 1.100 CI iterator diagnostics — 2026-11-17

The squeeze repair's CI run 44007706069 fails strict Clippy on five existing
index-only loops in spline area estimation, extra-channel output, pipeline
pruning and JPEG table emission. The local integrated pass used Rust 1.99.1;
CI uses 1.100.0. The scoped iterator conversion preserves source order,
checked slice bounds and every arithmetic operation. Matching-compiler strict
all-feature/minimal Clippy and full all-feature workspace validation pass:
236 seconds, run-heavy peak RSS 8.14 GiB, minimum available 15,168 MiB.
Remote CI remains separate. No lint allowance, threshold or existing
expected value changes.

## Spline point spacing investigation — 2026-11-05

The former point walker precomputed a unit direction with hypot and advanced
by repeated unit steps. Pinned libjxl v0.12 instead recomputes the distance from
the last emitted point using sqrt(dx*dx+dy*dy), divides the remaining distance
once, and multiplies the delta. Those operation orders accumulate different
rounding and shift spline samples. The pinned C++ oracle and exact Rust
regression are retained under perf/raw-layer-validation and features/spline.rs.
The old walker fails that new golden; the reference-order walker passes it,
including singleton and repeated-point cases. Existing numerical expectations
remain unchanged. See perf/spline-precision/README.md for reproduction and cost
measurement. Full all-feature workspace, strict Clippy, all/no-feature numeric
and approved API checks, LCMS and six frozen exact RGBA frames pass. The paired
whole-decoder layered run is564.69ms before and560.92ms after, paired effect
-0.11%,95%CI[-0.46,+0.37]%,not significant. Basic control is also nonsignificant.
The exploratory two-fixture check does not establish general performance.
benchmarks/spline_spacing_2026-11-05.json retains pins and raw pairs. This does
not establish bit-exact raw RGB output.

## Known bug: noncoalesced output option — 2026-11-05

Main already exposes coalescing / with_coalescing(), defaults true, and stores
false without consuming it in the decoder. Output remains composed. Draft
PR75 implements qualified local F32 output and explicit unsupported-mode errors;
that runtime work is not merged. This corrects the earlier inventory claim
that the option itself was absent.

The spline precision documentation also previously said false matched libjxl's
default. Pinned v0.12 common.h defines JXL_HIGH_PRECISION=1 unless overridden;
its spline cutoff exponent is 5. Native false uses3 and true uses5. Source
and generated Ninja definitions confirm the pinned reference has no override.
Matching comparisons need the existing true option; no default is changed.

## LCMS context ownership and Send qualification — 2026-11-05

lcms2 6.2.0's Transform holds a PhantomData context marker. Passing a
ThreadContext by value to new_flags_context drops that actual context on return.
The callback now retains it, with the transform field dropped first. The
context is Send rather than Sync; a private Mutex retains ownership while
preserving the callback object's existing Send + Sync contract. No unsafe
code or public API change is required.

Compile assertions establish that Lcms2Cms and the bundled transformer are
Send + Sync. The runtime test moves an initialized real transform across four
threads, requires identical float bits at each step, and destroys it on another
thread. Both CLI configurations pass all seven tests, including the unchanged
exact whole-image CMYK golden. Candidate Memcheck reports zero errors. The
baseline identity test also reports zero errors: this work does not claim a
reproduced invalid memory access. Strict all-feature workspace Clippy passes.

Context ownership alone does not make Box<dyn JxlCms> Send. Approved PR60
now adds that separate provider trait-object bound (0458ab95), without requiring
Sync on providers. Combined main6d060afb passes the full workspace, strict
Clippy, both feature-specific API/LCMS gates and six frozen exact RGBA frames.

## Resolved: HF table accounting and trailing metadata — 2026-11-05

The parser allocated group/pass section tables before the TOC and outside its
shared memory budget, including header-only scans. Allocation now happens after
the TOC/frame setup, skips scans, checks size arithmetic, and reserves the outer
and inner table together. A guard releases the reservation on frame reset/seek.
Tests reject oversized tables before allocation, exercise competition with
other frame storage, and check release.

Unused bytes inside finite codestream boxes could hide later Exif, including
across jxlp parts. The parser now drains those bytes before continuing ordinary
metadata parsing. Tests cover 0 through 70,000 unused bytes, partitioned
codestreams, seek-relative lengths, and memory-free scanning.

PR65's blanket post-codestream recovery is excluded: it hides injected I/O
errors and accepts a duplicate ftyp that djxl v0.12 rejects. Valid metadata
draining preserves current errors; new tests retain duplicate-ftyp and invalid
jxlp rejection and injected I/O failure. Recovery of malformed optional
metadata is separate work. The seven targeted checks pass with all features
and without default features. The combined full workspace suite and six
frozen-reference RGBA frames also pass. `just inventory-behavior-check`
reproduces the focused behavior and memory checks.

## Resolved: CMS callback contracts — 2026-11-05

Current-main reproduction passes only one of eight imported callback-contract
cases. RGB/gray callback errors panic, CMYK callback errors silently take the
mathematical fallback, invalid initializer counts/channels are not consistently
rejected, and the CMYK path substitutes an sRGB destination.

Both renderer runners now use a private fallible stage entry point. Numerical
kernels retain their existing implementation. A shared validated CMS stage
propagates the callback's original error and location context, rejects malformed
initializer responses before transforming pixels, and receives the requested
destination profile. The mathematical path remains when no applicable CMS is
configured. Reflectance inputs and raw-CMYK handling retain their contracts.

All eight callback cases pass with all features; the seven applicable cases
pass without default features (the parallel case requires `threads`). They
exercise both renderers, first/later/parallel failures, initializer refusals,
channel/count validation and CMYK destination selection. All 121 renderer
checks, both CLI LCMS test configurations and the combined full workspace suite
pass. The six frozen djxl regression frames have zero differing RGBA bytes.
`just downstream-cms-check` reproduces the focused gates. No public API change,
MoxCMS interpolation change or raw-layer API is included.

## Resolved: 64-point IDCT after fused fallbacks — 2026-11-04

The deterministic `test_idct1d_64_eq_slow` exceeds its unchanged 5e-6
absolute/relative bound after scalar multiply-adds become fused. Index 32
produces -1.9676225185394287 versus -1.9676337791674359: absolute error
1.1260628007159212e-5, relative error 5.7229287921271194e-6. The full
library run passes the other 1,024 cases, with 29 existing ignores.

Retaining extra secant-coefficient bits alone does not fix it. A candidate
that retains f64 intermediates through the existing large-transform
factorization passes all eight original 1D size checks (2 through 256)
without changing assertions. Its scalar-per-lane implementation was rejected
on measured cost. The current register-based implementation reuses the
existing generated 32-point factorization through magetypes f64 vectors;
all original 1D/2D accuracy checks and all nine large-shape raw-f32 parity
checks pass on scalar/SSE4.2/AVX2. ARM64 passes 54 selected transform
checks and all 40 decoder regressions; WASI passes 31 scalar and 54 SIMD128
transform checks. The full native workspace gate passes (1,029 library cases,
29 existing ignores, integrations and doctests), with strict Clippy and
formatting. Six fixture frames retain exact frozen djxl reference bytes on
x86 and ARM64. LCMS whole-image checks pass in both CLI configurations.
The paired 64x64 AVX2 microbenchmark measures 9.4 us versus 3.8 us for the
f32 baseline; whole-decoder impact is unmeasured. No expectation is weakened.


## Resolved: custom CMYK CMS conversion — 2026-11-04

A caller-supplied CMYK CMS was bypassed unless the optional MoxCMS feature
was compiled. The dedicated stage now compiles independently of that backend
and passes the public transform contract's reflectance convention (1 = no
ink); each backend performs its own ICC normalization. The LCMS backend now
converts to percentage ink and enables black-point compensation and high
resolution precalculation, matching libjxl v0.12's transform configuration.

The existing LCMS implementation with CMYK CMS enabled differed from djxl
in 94,239 of 1,048,576 RGBA bytes, maximum difference 252. The corrected
implementation matches every byte. A strict whole-image hash regression
passes both the default CLI and no-default-feature CLI builds, without
requiring MoxCMS. The existing MoxCMS color assertions retain their expected
values and tolerance; their direct inputs now use documented reflectance.
A custom-transformer stage test checks the same contract explicitly.

## Known Bugs: remaining display parity — 2026-11-04

The same djxl v0.12 runtime produces a host-dependent green byte at
canvas (18,18) of crop frame zero: 72 on i270 and 73 on r5900xt. Rust
produces 72 on both hosts, with and without optimization. This is a
reference-portability finding, not evidence that optimization changed Rust's
pixels. Cross-host gates can use frozen v0.12 reference images with verified
input and reference SHA-256 records; comparison remains byte-exact.

The previous cross-tier gate permitted float rounding differences: optimized
x86 output differed between AVX2 and SSE/scalar in
`grayscale_patches_var_dct.jxl` (one byte) and
`squeeze_empty_residual.jxl` (two bytes), all by one code. Fused fallback
arithmetic eliminates these differences across all 40 fixture/configuration
comparisons and seven available x86 permutations. ARM64 also passes all 40
configurations over nine permutations, all 40 selected decoder regressions,
and both exact crop frames. Native scalar/SSE4.2/AVX2/NEON and WASI
scalar/wasm128 fused-operation contracts pass. The cross-tier gate requires
exact bytes throughout. The final full workspace, CMS, WASI and frozen-reference
gates pass after the large-IDCT precision repair. AVX-512 execution and
fallback performance on CPUs without FMA are unmeasured.

Optional MoxCMS CMYK display output retains whole-image differences from
LCMS-backed djxl. The CLI's supplied LCMS backend has a strict whole-image
regression and matches every byte. This report does not claim that the two
independent CMS implementations have identical interpolation results.

## Resolved: negative canvas dither coordinates — 2026-11-04

[RESOLVED] The canvas-coordinate repair represents negative padding/crop
positions with wrapping usize arithmetic. Later dither chunk additions and
channel offsets still used checked addition. The full debug-assertion suite
reproduced an overflow in `convert.rs` through the existing
`serial_parallel_parity_palette_across_thread_counts` regression.

The repair wraps chunk offsets and reduces coordinates/channel phase to
the 32-pixel table period before adding. New table, conversion and fused-stage
regressions compare wrapped negative coordinates with equivalent positive
coordinates. The full workspace all-feature suite passes with debug overflow
checks enabled: 1,024 library tests pass, 29 existing ignores; integrations,
CLI tests and doctests also pass. All 18 selected no-default-feature
regressions, strict Clippy and scoped formatting pass. No expectation or
overflow check was removed.

## Resolved: inclusive sRGB transfer thresholds (2026-11-04)

The exact linear-to-sRGB threshold 0.0031308 and inverse threshold 0.04045
were selecting the rational high branch rather than the linear branch.
Strict raw-f32-bit regressions reproduced both failures on scalar, SSE4.2
and AVX2. Both equality cases now take the linear branch, preserving sign
(including negative zero). Separate float output and the two fused u8 stages
share one nonnegative sRGB magnitude implementation.

Reference: libjxl v0.12
[`TF_SRGB`](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/cms/transfer_functions-inl.h#L197).
The six cases on the available scalar/SSE4.2/AVX2 tiers pass after the repair;
AVX-512 is unavailable on the local host. Strict workspace all-target,
all-feature Clippy passes. Full integration and post-repair fixture-oracle
validation are tracked separately.

## Resolved: inverse-XYB rounding boundary (2026-11-04)

The inverse XYB kernels now share the same implementation and accumulate
matrix terms in libjxl v0.12's order, scaling the matrix coefficients for
intensity rather than the cubic operands. This resolves the remaining blue
byte at canvas pixel (22,57): the previous 7 now matches the reference's 8.

The exact candidate source was built with Rust 1.99.1 on ARM64. All 12
selected crop, XYB, palette, LF smoothing and spline checks passed. Strict
RGBA comparison against the frozen independent djxl v0.12 reference reports
zero different bytes in each 64x64 animation frame, with no tolerance.
The existing NEON and scalar XYB consistency tests also executed. Full
integrated and platform CI validation remains a separate gate.

## 2026-11-04: cropped u8 animation references and coordinates

[RESOLVED] Fused XYB/transfer/u8 output could defer color conversion past a
reference save. The saved float planes then contained XYB or linear RGB where
a later frame expected post-transform values. Fusion is now disabled when
a frame saves a reference after color conversion. The low-memory pipeline
also uses the extended canvas height for outside-frame row lookup and passes
canvas coordinates after Extend to position-dependent stages such as dithering.

`cropped_animation_preserves_reference_background` and
`cropped_animation_u8_pipelines_and_chunks_are_bit_exact` fail before the
repair and pass for both pipelines with whole-file, 1- and 7-byte input.
The 198-byte crop fixture is retained with owner approval. Independent
libjxl byte comparison is tracked separately; these tests establish internal
reference, pipeline and chunking invariants.

## 2026-11-04: ICC header read-ahead can exceed the whole stream

The ICC parser's `OutOfBounds(c + remaining / 8)` estimate assumed at least
one coded bit per remaining predicted ICC byte. LZ77 can encode many such
bytes with fewer bits. The refill loop then waited for bytes beyond the
valid file and returned `UnexpectedCodestreamBoxEnd` without its final
parse attempt. Requesting the actual deficit `c` fixes that retry.

Reproduction: `just icc-header-repro <stream.jxl> <original.icc>`. The
caller-supplied regression fails with the old estimate, then recovers the
ICC bytes exactly with whole-file, 1-, 7- and 257-byte input chunks after
the repair. Tested on the 1x1 CMYK Probev1_ICCv4 stream retained at
`~/tmp/jxl128-read-ahead-controls-2026-11-04/4.jxl`, expected profile
`~/tmp/icc-fetch-test/Probev1_ICCv4.icc` (SHA256
`47f4e71af3c03b5801f6555a57be645040c8a83011947a35bc50b599bfc8613a`).
Before/after logs: `~/tmp/jxl128-native-header-{before,after}.log`.
This is header/ICC validation; CMYK display conversion remains a separate
CMS capability. No trailing padding workaround or limit relaxation is used.
`just feature-corpus` provisions the existing feature corpus through
codec-corpus and prints `CODEC_CORPUS_PATH`. With that caller setting,
`just icc-check` passes formatting, strict library/example Clippy and all
24 ICC-filtered library tests; log `~/tmp/jxl128-native-final-check.log`.

## Resolved 2026-10-22: short out-of-order `jxlp` splice stalled the box parser

[RESOLVED] `SmallBuffer::inject_bytes_front` (`src/api/inner/process.rs`)
shrank the buffer to the spliced payload; a payload shorter than the next box
header (12 bytes, 20 with a 64-bit size) left `refill()` unable to ever hold
that header, so `process` returned `NeedsMoreInput` forever without consuming
input (`decode`: `OutOfBounds`). The buffer now keeps its capacity. Test
`short_spliced_ooo_jxlp_followed_by_more_boxes`, 12 procedurally built files;
it fails without the fix. djxl v0.12 renders all 12 with the in-order file's
pixels. The chunked helper `api::decoder::tests::decode` now panics after 64
consecutive zero-consumption `NeedsMoreInput` calls with all input offered,
so a stall fails the test instead of hanging it.

## JPEG standalone restart markers — 2026-09-24

[PROVEN] `jpeg::writer::write_jpeg` omitted RST0–RST7 entries from JBRD
`marker_order`. These are standalone markers, including a restart after the
last MCU; restart markers inside scans are reconstructed separately in
`write_sos`. libjxl v0.12 `lib/jxl/jpeg/dec_jpeg_data_writer.cc::EncodeRestart`
writes the standalone marker's two bytes without deriving a new restart index.

`jpeg::writer::tests::standalone_restart_markers_keep_original_order` failed
before the correction: only SOI, EOI and trailing data survived. It covers all
eight marker values in a deliberately nonsequential order. The matching encoder
scanner correction is jxl-encoder `1a40b8f3`; its conformance harness at
`a6a9a116` records 21 camera originals reconstructing two bytes short before
this writer fix, and validates reconstruction with both Rust and libjxl v0.12.
No public API or scan-entropy algorithm changes are required.

Validation: release library tests with `jpeg` pass under CI's explicit
`ZENJXL_ALLOW_MISSING_CORPUS=1` configuration (834 reported passes, 28 existing
ignores). Of those reported passes, 103 corpus-backed feature cases do not run
because the Mac lacks their corpus; the unconfigured run fails those 103 cases
and passes the other 731. All eight JPEG reconstruction tests and the new
standalone-marker regression execute and pass. Logs are under
`~/tmp/jxl-backlog/decoder120-*` and `~/tmp/zenjxl-decoder/jpeg-issue120.log`.
`just jpeg-reconstruction-check <label>` reproduces the release library gate;
set the corpus policy explicitly at invocation, as CI does.
Workspace all-target, all-feature Clippy and scoped formatting also pass.

## WASI regression deadlines — 2026-09-24

[PROVEN] The two deadline regressions in `jxlrs_testdata_ports` used
`std::thread::spawn`, which traps on wasm32-wasip1. They now live in the
`regression_deadlines` integration binary with unchanged decode assertions.
Native execution keeps each 20-second worker deadline. The WASI runner applies
Wasmtime's `-W timeout=20s` to the entire dedicated binary, including both
regressions; the guest refuses to run without the runner's deadline marker.
This also bounds a decoder that never returns or polls cancellation. The
WASI suites pass with no features (767 library tests reported) and `wasm128`
(808), each with 37 existing ignores; all integration and doctest binaries
pass. Local runs use `ARBTEST_BUDGET_MS=100` and CI's explicit missing-corpus
policy; these counts do not imply external-corpus coverage. An intentional
infinite-loop module run through the same runner traps with `interrupt` at
the configured deadline.
Both WASI jobs also pass on commit `440cc002` in
[CI run 36008146603](https://github.com/imazen/zenjxl-decoder/actions/runs/36008146603).
`just wasm-ci-check <label>` runs both CI feature configurations and saves
complete logs under `~/tmp/zenjxl-decoder/` with CI's explicit missing-corpus
policy.

## i686 fixture sweep memory — 2026-09-24

[PROVEN] `run_fixture_sweep` creates up to eight workers independently of
libtest's `--test-threads=1`. The exported Linux i686 baseline (executed through QEMU on WSL) reproduces CI's
`compare_pipelines_sweep` failure at `ImageOutOfMemory(42048, 5377)`, plus two
`ImageOutOfMemory(32512, 2704)` failures. The comparison helper deliberately disables decoder request limits
and renders full f64 reference planes. The baseline's process peak RSS was
2.16 GiB under `run-heavy --mem 12G --jobs 4`; address-space allocation failed
without reaching the host memory ceiling.

The correction serializes fixtures and sweeps on 32-bit targets.
It retains the current fixture selection, including the large TOC fixture,
and all pixel-hash comparisons. The duplicate `issue865_large_toc.jxl` in
`resources/test/` is byte-identical to the dedicated `tests/testdata/jxlrs-865/`
copy; neither is removed. Before/after logs are under
`~/tmp/jxl-backlog/decoder-ci-i686-*.log`.
The rebuilt comparison passes every selected fixture on the same target, with
2.63 GiB process peak RSS and 15,562 MiB minimum host available RAM under
`run-heavy`. The failing baseline peaked lower because it could not complete;
this is not a claimed reduction in completed-decode memory.
`just i686-ci-check <label>` reproduces both CI feature configurations on a
Linux multilib host; invoke through `run-heavy` on shared hosts.

Source-export trap: `rsync -a` preserves source timestamps. When a local edit
predates completion of the remote baseline build, Cargo can reuse that baseline
binary after the copy. Refresh changed source timestamps in the exported tree
and confirm a compile occurs before treating an after-run as evidence. The
invalid first after-run is retained as `decoder-ci-i686-stale-binary.log`.


## Full CI confirmation — 2026-09-24

[CI run 36009778067](https://github.com/imazen/zenjxl-decoder/actions/runs/36009778067)
passes all 19 jobs at `420d38859ab6ee9ed9a9c213d8b301fb8afe0b38`:
both WASI configurations, i686 no/all features, Windows ARM64, Linux x64/ARM64,
macOS Intel, formatting and Clippy. This independently verifies the WASI
deadline repair (`440cc002`) and 32-bit fixture serialization (`420d3885`).
Remote `main` was verified at that commit after the run. The earlier preserved
`940d2c51` work remains in its ancestry. CI's configured corpus policy remains
in effect; a green job is not a claim that every external corpus was present.

## Fast-decode lossless exploration — 2026-10-01/02

[MEASURED] M4 Pro, 4 CLIC 2025 photos (~2.8 MP, 48 groups each), CLI
`--speedtest` u8 `--no-cms`; a smoke comparison, not zenbench. Files come
from jxl-encoder `LosslessConfig::with_fast_decode()` (`02850e60`), which
makes every channel one Gradient MA leaf with prefix codes and no LZ77.

- Entropy decoding, not prediction, bounds single-thread lossless decode.
  `main` reaches 68 MP/s on these files and 62 MP/s with ANS.
- `explore/fused-prefix-lut` (draft PR #61) has three steps:
  - `0441e81b`: a fused 12-bit table (prefix code, extra bits and signed
    unpack in one load) reaches 132 MP/s. It needs the register-resident
    `FastBits` cursor; with `&mut BitReader` in the hot loop the cursor is
    stored and reloaded per sample, and the table gives only +10%.
  - `31ad13d6`: two residuals per lookup reach 169 MP/s.
  - `e4d32db2`: two groups decoded interleaved reach 201.5 MP/s at 1 thread
    (~605 MB/s RGB8); 4 threads give 581 and 12 give 935, against 237 and
    480 on `main`. Interleaving gained nothing until the two sides were
    passed by value: behind `&mut`, both cursors spilled to memory.
- Table window on M4: 11 bits 164, 12 bits ~170, 13 bits 172, 14 bits
  161 MP/s.
- [MEASURED] Ryzen 9 7900X (Zen 4), Linux, same files, 1 thread:
  `main` 60, fused table 89, two-symbol 97, interleaved 96 MP/s
  (~288 MB/s RGB8). 4 threads: 252 to ~417; 12 threads: 383 to ~517.
  11-bit and 12-bit windows tie there. Interleaving gives nothing on
  Zen 4. Zen 5 is not measured.
- [MEASURED] On that Linux box only ~52% of the 1-thread run is decoder
  code: ~37% is kernel page faults and ~10% libc memset/memmove. The
  cause is per-group modular channel buffers. `with_buffers` allocates
  and zero-fills 144 buffers of ~248 KiB per decode (35.8 MB for a
  2.8 MP RGB image) through `alloc_zeroed_fallible`
  (try_reserve + resize, an explicit memset over fresh pages). The
  render pipeline recycles rendered group buffers into its scratch pool,
  but the modular reader never takes from it. Reusing those buffers
  across groups is the next lever for Linux throughput on every modular
  decode, not only this fast path.
- `explore/two-pass-top` (`92b9b69b`) and `explore/fused-two-pass-top`
  (`af2dfd2d`) are negative results. Two-pass gradient is slower (68 to
  52 MP/s), and fused Top (126 MP/s) is slower than fused gradient, with
  ~15% larger files.
- Conformance: preset files decode exactly in libjxl v0.12, jxl-rs 0.7.4
  and jxl-oxide 0.12.6 (gray/alpha/8/16-bit, efforts 1-9). See jxl-encoder
  `just fast-decode-check` and `just fast-decode-conformance`.
- Prefix codes cost ~3-5% size against ANS for the same settings; the
  preset is ~17% smaller than PNG on the CLIC photos.
- `sansio` (`5fc1a3e`, origin) holds unreviewed prior-session work: a new
  public `JxlIncrementalDecoder` and default zencodec/zenpixels deps. It
  is not on `main`.

Benchmark files and logs are under `~/tmp/fastll/`.

## Modular buffer recycling and parallel transforms — 2026-10-02

[MEASURED] Ryzen 9 7900X, Linux, CLI `--speedtest` u8, 3 interleaved
rounds; 11.2 MP numbers from `/usr/bin/time -v`.

- `97f08d29`: modular group buffers come from a `RecyclePool` (zero-filled,
  exact geometry) fed by the render pipeline's `take_recycled_inputs`.
  Sequential modular frames without noise or border stages output each
  two-group chunk right after decoding (`eager_modular` in
  `frame/render.rs`); parallel frames with > 16 groups per thread use
  batches of 8 groups per thread. 1 thread on 48-group frames: 60 to
  83 MP/s; sys time 0.27 to 0.04 s; 11.2 MP peak RSS 228 to 94 MiB
  (1 thread), 230 to 128 MiB (4 threads). Multi-thread speed is
  unchanged: glibc's per-thread arenas already kept those pages mapped.
- `e23e8697`: a transform layer's `do_run` calls run on rayon when their
  inputs are disjoint (`FullModularImage::run_layer`). The parallel
  path's Phase 3a-store was 1.8 of 5.5 ms per 48-group frame at 12 threads.
- Combined with `explore/fused-prefix-lut` (fast-decode files): 1 thread
  166.5, 4 threads 448.5, 12 threads 738.3 MP/s (main before both:
  60.1 / 245.0 / 376.8). 12-thread scaling is still ~4.4x the 1-thread rate;
  the remaining serial work is per-frame setup and Phase 3a output passing.
- Content of recycled buffers is always fully overwritten by decode
  (poisoning them did not change output), but the zero-fill is kept so a
  reused buffer is byte-identical to a fresh one, padding included.

## Fast lossy (VarDCT) decoding — 2026-10-02

[MEASURED] 4 CLIC 2025 photos (~2.8 MP), CLI `--speedtest`, MP/s.

- Fastest lossy encoding to decode: VarDCT with libjxl `--faster_decoding=4`
  (EPF and gaborish off, no 32x32+, simple block contexts, fixed gradient
  DC tree), about 10% larger than e7 at the same distance. Modular lossy
  (`-m 1`) is the slowest (~25 MP/s on M4). Dithering is not a cost: u16
  output (undithered) is no faster than u8.
- Decoder changes `082c50ed`, `c3d92091`, `24809159`, `04612ee4`.
  Ryzen 7900X, cjxl d1 fd4: 1 thread 122 -> 134, 12 threads 416 -> 485;
  cjxl d1 e7: 76 -> 80, 286 -> 304. M4 Pro fd4: ~101 -> ~108 / ~430 -> ~470.
  Upstream jxl-rs `5122960` on M4: within 1-2% single-threaded, slower at
  12 threads on JPEG transcodes and jxl-encoder files.
- Tried without gain (reverted): a register-cursor rewrite of the
  gradient-lookup DC loop, and hoisting the ANS/prefix match out of the AC
  loop (<1%).
- Serial floor: an image up to 2048x2048 has one LF group, decoded on one
  thread before any HF group (bitstream order: DC stream, then HF
  metadata). On M4 that is ~1.2 ms DC + ~1.3 ms HF metadata for 2.8 MP,
  about 9-10 ns per sample along a value -> context -> ANS chain; at 12
  threads it is ~40% of the frame. HF groups cannot start earlier because
  the HF metadata follows all of the DC data in the same section.
- jxl-encoder `--faster-decoding` (`32caff3d`) now follows libjxl v0.12
  (EPF by tier, kGradientFixedDC DC tree, no 64x32 from tier 2); its tier-4
  files decode within 1% of cjxl's. libjxl's tier-1 simple block context map
  and 6-histogram AC cap are not mirrored there and measured as not needed
  for decode speed.

## Lossy gap to djxl, profiled — 2026-10-02

[MEASURED] M4 Pro, 4 photos (2.8-3.1 MP), `--data-type f32` (djxl
`--disable_output` decodes to float, so u8 runs overstate the gap by the
u8 conversion), MP/s, ours / djxl v0.12:

- d1 e7: 1T 65.6 / 72.7, 12T 284 / 349. JPEG q90 transcode: 1T 124 / 136,
  12T 509 / 668. d1 fd4: 1T 110 / 105, 12T 459 / 521.
- Fixed: EPF stage 1 checked bounds on all 54 loads per vector and
  reloaded row pointers after each store (`e1463a0c`): d1 e7 1T 62.2 -> 65.9.
  Per-decode `sample` counts put EPF1 at ~1.6x djxl's before, ~1.5x after
  (sampling estimate).
- 12T is the larger gap. On the JPEG transcode at 12T the main thread spends
  ~35% of wall time in the serial LF group (DC through the weighted
  predictor, `WpOnlyLookupConfig420`, then HF metadata) and workers are idle
  ~57% of the time; the rest of the idle time is the gap between the decode
  and render parallel phases. Single-threaded sample fractions put our WP
  DC decode at roughly 1.2x djxl's time (sampling estimate, not measured
  per function).
- Register cursor (`PlainCursor`) for modular channels helps single-leaf
  trees only: fast-decode preset 66 -> 89 MP/s 1T, 487 -> 603 12T. On
  learned trees and libjxl's gradient DC tree it was neutral 1T and 3-10%
  slower at 12T in back-to-back runs, so it is off there
  (`ModularChannelDecoder::USE_CURSOR`).
- 12T numbers on this Mac move 5-10% with background load (Spotlight,
  mediaanalysisd, rust-analyzer); compare A-B-A, not A-B.


## Multi-threaded parity work — 2026-10-03

Scripts used for these measurements: `scripts/perf/` (see its README).

[MEASURED] M4 Pro (8 P + 4 E cores), f32, interleaved runs; the machine
was loaded (Spotlight indexing, load 12-24), so treat MT figures as +-5%.

- False sharing (`e404d64b`): two 4-thread processes decoded e7 lossless at
  a combined ~60 MP/s while one 8-thread process reached 50. The cause was
  the per-channel MA-tree property buffer (a small `Vec<i32>` written every
  sample) sharing a cache line with another thread's data. `IsolatedBuf`
  (`util/isolated.rs`) pads it by 128 bytes per side. The two-process vs
  one-process comparison is the quick test for this class of problem.
- Per-group timing (`JXL_GDBG`-style instrumentation, not committed) shows
  E-core threads run groups 2-3.7x slower and macOS starts some threads on
  E-cores; P-core groups also slow ~1.3x at 12 threads (all-core clocks).
  From 8 to 12 threads we gained nothing on fd4 while djxl gained 9%.
- Serial LF group (one LF group up to 2048x2048): d1e7 DC 1.87 ms + HF
  metadata 0.85 ms; JPEG q90 DC 1.03 + 0.2; fd4 DC 1.13 + 0.85. DC per
  sample matches our e7 lossless speed, itself at djxl parity. Reading
  `GradientLookup` (fd4 DC) through the register cursor was slower (LF
  2.02 -> 2.30 ms); keep `USE_CURSOR` off there.
- The batched VarDCT path is expensive: forcing 2 batches of 24 groups
  raised P3b from 2.81 to 5.12 ms (d1e7) and 0.95 to 3.03 ms (JPEG), so
  overlapping render with the next batch's decode is not worth it on top of it.
- In-task render: `8c4e8175` for `border_size == 0` pipelines (fd4, JPEG
  4:4:4), `a820cac9` for the rest (a group renders when its 3x3
  neighbourhood is decoded; inputs published through per-group
  `OnceLock`s). Those frames decode in raster order: largest-first left the
  renders to the end and lost 2.7% on JPEG 4:2:0 at 4 threads.
- Tried without gain on d3 1T: a single-pass `get_distinct_indices` (the
  profile's 2.6% there is mis-attributed inlined code; instruction count
  unchanged) and a vector-index EPF1 loop with one up-front bound (+0.2%
  instructions).
- Rejected: packing cluster + hybrid-uint config into one per-context
  table for the AC loop (0.3% fewer instructions, 1.5-3% more cycles).
- Parity at `63370cd7`, interleaved, zen/djxl (4 CLIC photos per set):
  1 thread d1e7 1.05, fd4 1.19, JPEG 4:2:0 1.10, 4:4:4 1.08, d3 1.02,
  e7 lossless 0.97, e3 1.19, e1 1.28; 12 threads d1e7 1.02, fd4 1.03,
  JPEG 4:2:0 1.03, 4:4:4 1.05, d3 1.01, e7 1.30, e3 1.36, e1 1.61. Ahead of
  upstream jxl-rs on every set at both thread counts.
- EPF gains came from codegen, not algorithm: per-vector bounds checks on
  15-21 row windows and output rows read through the row table every vector
  (`85d33a94`, `5592d393`), and SmallVec pushes building each stage call's
  row lists (`63370cd7`).
- e7 lossless 1T (0.97): per decode djxl 889M cycles / 3.02G instructions,
  zen 947M / 3.63G. Not instruction-bound: removing the tree walk's property
  bounds checks (fixed 256-entry buffer, -3% instructions), caching the WP
  weights (-1.2%) and a 2x unrolled walk (-2.4%) each left cycles equal or
  higher. The time is in the data-dependent tree branches and the leaf
  predictor switch, as in libjxl.
- d3 1T: per decode djxl 146M cycles / 710M instructions, zen 152M / 900M.
  Our EPF1 and EPF2 kernels take ~2x djxl's share each; the sRGB transfer
  (`linear_to_srgb_simd`, 9% here) is the same algorithm as libjxl's.

## Known Bugs: output-buffer initialization (resolved 2026-11-11)

`RawImageBuffer::try_allocate` previously used `Vec::set_len` on uninitialized
storage under `allow-unsafe`, contradicting the initialized-byte invariant of
`OwnedRawImage`. Image allocations now use
`bytemuck::allocation::try_zeroed_vec` with `extern_crate_alloc`, mapping allocation
and layout failures to the existing `ImageOutOfMemory` error. The historical
`new_uninit` names remain compatible but always return initialized storage.
Alignment, padded footprint accounting and parallel prefaulting are unchanged.
The `allow-unsafe` feature still controls typed-row casts; this correction does
not remove that feature. `just allocation-check` exercises initialization, OOM,
memory accounting and existing exact-pixel regressions in release mode, with
all-feature and minimal-feature Clippy checks.

The issue-18 measurements are retained in
[`benchmarks/fallible_zeroed_allocation_2026-11-11.md`](benchmarks/fallible_zeroed_allocation_2026-11-11.md).
On Ryzen 9 5900XT, 64 MiB bytemuck allocation measured 7.923 µs versus
7.865 µs for its infallible zeroed control. Whole-decode results were mixed:
the eight-thread café A–B–A witness measured 17.617 / 18.691 / 17.711 ms,
while both 4K cases improved in the exploratory sweep. Do not treat allocator
parity as a guarantee of matching the former uninitialized path's throughput.

## Known Bugs: baseline JPEG extra-zero-run placement

Independent native v0.12 scan controls reconstruct exactly in native decoding
but fail Rust reconstruction for a baseline ZRL before EOB. Pixel rendering
is unchanged. The baseline writer emits the extra AC token before DC, rather
than after the final nonzero AC coefficient. Progressive controls pass. See
`zenjxl-decoder/tests/jpeg_scan_contract.rs` and its native fixture provenance.

Expanded native controls also reproduce baseline non-interleaved grid loss:
16x8 grayscale with absolute 2x2 SOF sampling rebuilds 155 bytes instead of
149 by walking padded coefficient rows. Baseline iteration now uses the
visible component grid, retaining padded stride for coefficient access.


## Known Bugs: JPEG reconstruction channel mapping (resolved 2026-11-18)

Unmodified native v0.12 streams with RGB identifiers plus Adobe transform 1
or JFIF, and custom YCbCr identifiers, render correctly but reconstruct different
JPEG entropy bytes before repair. JBRD component_type is an identifier shortcut;
the frame do_ycbcr flag determines coefficient plane order. The verified repair
uses that flag, retaining the grayscale route. Native asymmetric fixtures in
`jpeg_scan_contract` cover both channel mappings independently of identifiers.
See `docs/JPEG_SCAN_CONTRACT.md` for verification and remaining limits.

## Resolved: near-unity HLG OOTF — 2026-11-19

The former 0.1 cutoff suppresses nontrivial OOTF corrections. Both kernels now
use v0.12's closed 0.01 deadband. Native component and complete packed/padded
pixel controls pass in default/minimal builds; restoring the old cutoff fails.
Color/stage/HDR, strict all-feature library/test lint and scoped formatting
pass. [Exact mapped regions and limits](docs/HLG_OOTF_CONTRACT.md) distinguish
direct wrapper tests from the unmounted forward production route. Platform
verdicts remain separate.

### Resolved: post-header output-profile admission — 2026-11-19

The typed setter now enforces the parser's existing non-XYB/no-CMS restriction
before changing state. The retained lossless HLG source exposed ignored
conversion before repair. Refusal, exact original-profile pixels and a separate
CMS admission observer pass; removing the guard fails. The owning contract
records 16/16 mapped setter regions. No CMS-free forward implementation or
complete external-CMS qualification is claimed.
