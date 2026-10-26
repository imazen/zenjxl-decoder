# Source interpretation for selected images

Additive accessors expose source fields needed by image collections and
transcoders. They leave existing public struct literals and `Eq` implementations
unchanged.

- `JxlDecoder<WithImageInfo>::extra_channel_details(index)` returns source
  integer/float precision, including exponent bits, and spot RGBA/CFA parameters
  when those roles apply. The index is the same as `basic_info().extra_channels`;
  an out-of-range index returns `None`. The owned result survives decoder drop.
- `JxlDecoder<WithFrameInfo>::frame_timecode()` returns the current selected
  frame's original 32-bit timecode. `Some(0)` is distinct from absence. It is
  separate from duration ticks and presentation timestamps.

The low-level `JxlDecoderInner` exposes the same accessors, returning `None`
before the relevant header is available. Extra-channel details obey the existing
embedded-profile guard; reset clears their availability. Reading details does
not request an extra pixel buffer or alter output configuration.

Source precision describes the encoded samples. It does not change when a caller
requests u8 or f32 output. Likewise, spot/CFA parameters do not authorize display
conversion or metadata passthrough. Frame names and opaque profiles can contain
information an application's retention policy must inspect or discard.

## Qualification

The standalone [native experiment](../perf/interpretation-validation/src/lib.rs)
uses `jxl-encoder` revision `0255e5a9f61b7270cb3ed3f7b85fec2283deec55`
to generate fixtures at runtime. libjxl independently inspects the encoded fields.
No generated binary fixtures are committed, and the encoder/oracle are outside
the decoder library's dependency graph.

The executed cases cover all seven named extra roles: alpha, depth, spot,
selection mask, black, CFA and thermal. Native decoding returns the exact
original u8 extra samples. Tests also check half/float alpha source precision,
non-default intrinsic size and tone-mapping fields, and absent/present-zero/
nonzero timecodes across reverse and repeated frame selection. The float-alpha
test qualifies headers, not float pixel equivalence. Unknown/reserved channel
roles and arbitrary combinations remain open.

The subsampled-channel extension generates an odd 17x13 canvas for every named
role at source shifts 1/2/3. libjxl decodes the full image and supplies independent
float samples; applying the requested no-dither round-to-even conversion gives
exact native u8 color and auxiliary samples. Direct u8 comparison is not an
equivalent configuration: libjxl 0.11.1 uses ordered dithering, while this decoder
offers blue-noise dithering. No numeric tolerance is used. These checks qualify
that sample representation, not cross-decoder bit-identical F32 arithmetic.
They exposed [issue 69](https://github.com/imazen/zenjxl-decoder/issues/69): the
source extra-channel shift must apply even when color is full resolution.
The fix also checks the effective 8x maximum in that case. Source `dim_shift`
stays distinct from the full-size emitted plane geometry.

From an export outside workspace dependency overrides, run:

```sh
just interpretation-check /path/to/fresh/persistent/artifacts
just interpretation-core-check
```

The first requires a C compiler and libjxl development library. Artifacts use
exclusive creation so earlier evidence is not overwritten. The second checks
incremental-header/reset visibility, existing scan/decode parity and workspace
all-feature/all-target Clippy. The standalone CI job uses the same tests.

Native gate receipt: `zenjxl-interpretation-native2-2026-10-26.log`, run-heavy
rc=0, 6 s, peak RSS 0.56 GiB, minimum available 13356 MiB, peak load 0.81.
Core gate: `zenjxl-interpretation-core-2026-10-26.log`, rc=0, 47 s,
peak RSS 0.99 GiB, minimum available 12858 MiB, peak load 1.40. These measure
the build/test processes, not a native decoder resource guarantee.

The integration requirement is tracked in
[issue 67](https://github.com/imazen/zenjxl-decoder/issues/67). Raw layer output
remains a separate contract in
[issue 57](https://github.com/imazen/zenjxl-decoder/issues/57).

The [source-subsampling receipt](../benchmarks/source_subsampling_2026-10-26.json)
records the 21 independent decodes, corrected header-test selection and native
regression checks. Native comparison: run-heavy rc=0, 6 s, peak RSS 0.40 GiB,
minimum available 13688 MiB, peak load 0.23 (test-process usage).

## Layer placement is separate from output extent

The native parser retains each regular frame's local size and signed origin.
The regression `coalescing_source_layers_keep_local_extents` decodes the pinned
`multiple_layers_noise_spline.jxl` with two input chunk sizes and observes all
five regular frame headers through the existing internal test hook. Those
headers match independent libjxl 0.11.1 discovery with coalescing disabled.
Two layers extend beyond the 2048x1152 presentation canvas: the 1024x576 frames
at (512,640) and (1860,950). Their stored extents remain intact.

This initial gate qualifies source headers. The public F32 path described below
now returns those local layers while retaining the composed references needed
by later frames. A contained pixel-region type cannot substitute for signed
layer placement.

The [placement receipt](../benchmarks/layer_placement_2026-10-26.json) records
`just coalescing-check`; it makes no latency or total-memory claim.

## Native discovery has its own header allowance

A visible-frame count does not bound native discovery: one presentation in the
layer fixture requires five regular frame headers, and previews, LF frames and
reference dictionaries can add hidden work. `JxlDecoderLimits::max_frame_headers`
is an opt-in bound on completed header parses. Configure it before decoding:

```rust
let limits = JxlDecoderLimits::default().with_max_frame_headers(5);
let options = JxlDecoderOptions::default().with_limits(limits);
```

The decoder checks before the next frame-header parse. Completed headers remain
charged if later validation fails. Partial input retries do not charge again;
seek and rewind preserve the count and charge reparsed headers. A full reset
starts a new operation. Exhausted input is not another frame. The error uses the
existing `LimitExceeded` variant with `resource: "frame_headers"`, `actual`
(the requested next count), and `limit`; no capability query is required.

This is a per-decoder allowance. Separate decoder instances do not share it.
It bounds neither bytes nor time/memory within one header, and it does not bound
metadata boxes outside the frame stream. Existing pixel, memory, profile and
stop controls remain necessary. Defaults preserve the existing unrestricted
header count; a caller requiring a bound must supply it.

`just header-budget-check` checks hidden regular, preview, reference-only and
LF frames, one-byte/large input delivery, EOF, seek, rewind, reset and native
pixel equality under an admitted count. The LF control confirms actual LF frame
types through the existing internal decode callback. The earlier candidate
fixture `multiple_lf_420.jxl` did not exercise LF frames in that callback;
the qualified fixture is `conformance_test_images/progressive.jxl`.

Final native gate and warning-free docs/API snapshot: all four header-budget
controls and strict all-feature/all-target workspace Clippy pass.
`run-heavy: done rc=0 40s | peak-RSS 0.39GiB | min-avail 13168MiB | peak-load 0.84`.
Log `zenjxl-header-budget-final-2026-10-26.log`, SHA-256
`1055bbbf3459692842861c46e55637b9c4162bf79c0c42c9f6960762fee6141e`.
These are test-process measurements, not a decoder resource guarantee.
Tracking: [issue 71](https://github.com/imazen/zenjxl-decoder/issues/71).

## Native local-plane probe

A test-only output tap now retains F32 planes immediately before blending and
canvas extension. It saves every local sample, including off-canvas pixels,
while the existing pipeline continues producing composed references and display
output. Allocation uses the decoder's existing memory tracker. This test-only
tap supplied the native reference for the public F32 path below; production raw
output writes into the caller's buffers without allocating these tap planes.

`just raw-layer-probe <fresh-artifacts>` qualifies five RGB/alpha layers from
`multiple_layers_noise_spline.jxl` and four CMYK/alpha layers from
`conformance_test_images/cmyk_layers.jxl`. The simple and low-memory renderers,
with complete and 16384-byte input delivery, return identical captured samples.
Their composed output is bit-for-bit identical to the uncaptured path. NaN
poisoning checks that every local sample is written. Optional export produces
plane files and geometry for a separate independent-oracle check; it does not
change the native test assertions.

The [probe receipt](../benchmarks/raw_layer_probe_2026-10-26.json) records the
strict lint gate and test-process resource observation:
`run-heavy: done rc=0 90s | peak-RSS 1.13GiB | min-avail 12479MiB | peak-load 1.64`.
Independent raw-sample comparison, public publication/selection and blend facts,
preview/state restoration, output conversion and complete workspace bounds are
not established by this gate.

### Independent layer samples and composition facts

The next [oracle gate](../perf/raw-layer-validation/compare.py) runs libjxl
0.11.1 with coalescing and spot rendering disabled and stored orientation kept.
It compares all nine captured local planesets and exact placement, color and
extra-channel blend facts, reference-save slots, duration and last-frame flags.
CMYK/alpha samples agree bit-for-bit (1,587,120 samples), as do the alpha planes
of the RGB fixture. The lossy RGB/alpha fixture contains 18,874,368 samples;
5,485,741 float bit patterns differ, with maximum absolute difference
0.00036650896072387695. All satisfy the pre-existing F32 conformance threshold
1/256 from `src/tests/parity.rs`. This is conformance, not bit-identical RGB
arithmetic between independent decoders. Native cross-path equality stays exact.

The oracle also rejects a deliberately changed single mantissa bit in a CMYK
plane, even though that error is below the lossy threshold, and verifies the
original artifact is unchanged. Each run creates fresh artifact directories.
The [receipt](../benchmarks/raw_layer_oracle_2026-10-26.json) preserves the initial
missing-NumPy failure and the dependency-free standard-library replacement.
Run `just raw-layer-oracle <native-artifacts> <fresh-oracle-artifacts>` after
the native probe; it requires a C compiler, libjxl development files and Python.

These fixtures exercise replace, add and alpha blending, but do not establish
all modes or independent extra-channel blend combinations. Random layer
selection and representation conversion remain open. No binary fixture or
sample dump is committed.

The all-feature configuration also passes the same native and independent gates,
including threaded and SIMD rendering: `just raw-layer-probe-all <artifacts>`,
then the same oracle command. Native gate:
`run-heavy: done rc=0 61s | peak-RSS 1.55GiB | min-avail 12162MiB | peak-load 1.16`.
These observations cover the test process, not a layer workspace guarantee or a
throughput comparison. The oracle receipt includes both configurations.

## Qualified public raw-layer output

Use the existing typed decoder and buffers. Disable coalescing and orientation
adjustment before feeding input; keep previews skipped and spot channels separate.
After image information arrives, select F32 color and extra-channel formats.
The next frame header reports the **local** dimensions, including samples beyond
the canvas. Allocate buffers at that size and decode normally. Both tightly
packed and padded rows are supported.

```rust
let options = JxlDecoderOptions::default()
    .with_coalescing(false)
    .with_adjust_orientation(false)
    .with_render_spot_colors(false);
// Feed input until the existing WithImageInfo state is returned.
decoder.set_pixel_format(JxlPixelFormat {
    color_type: JxlColorType::Rgb,
    color_data_format: Some(JxlDataFormat::f32()),
    extra_channel_format: vec![Some(JxlDataFormat::f32()); extra_count],
});
// Feed input until WithFrameInfo; allocate each row buffer from this local size.
let layer = frame.frame_layer_info();
let size = frame.frame_header().size;
assert_eq!(size, layer.size);
// frame.process(input, buffers) writes this local layer; repeat for subsequent layers.
```

`frame_layer_info()` supplies signed origin, complete local extent, color and
per-source-extra blend instructions, reference slot saves, source duration ticks
and the last-frame flag. Reference slots belong to this decoder's context; they
are not stable collection IDs. LF and reference-only helpers remain internal.
In coalesced mode the same accessor supplies source facts, while output size
continues to describe the composed canvas.

This is not a lossless re-encoding instruction set. LF and reference-only helpers
are not published as images. A reference saved before color transformation can
contain a different representation from these reconstructed local samples;
`saved_before_color_transform` reports that fact without claiming those internal
samples are exposed.

Native CMYK components, including a separate black plane, use reversed ink
polarity: zero means full ink, one means no ink for normalized F32. This matches
[libjxl's CMS sample convention](https://github.com/libjxl/libjxl/blob/v0.11.1/lib/include/jxl/cms_interface.h#L119).
The embedded ICC profile alone does not declare this sample convention. A
consumer must retain polarity with its samples and explicitly qualify conversion
to conventional ICC device values. No inversion is performed by raw output.

Raw CMYK retains the black component, packed or as an extra plane, and reports
its embedded CMYK profile as soon as the output format is configured. A packed
three-component buffer in that configuration holds CMY, not display RGB; its
profile and separate black plane are essential. Source-associated alpha remains
source-associated. No tone mapping, CMS conversion, orientation adjustment,
premultiplication, spot flattening or integer/half output is implied.

Unqualified requests return `UncoalescedOutputUnsupported` (global configuration)
or `UncoalescedOutputRequirement` (the unmet requirement). Format validation runs
before processing, including low-level changes after a header. Progressive
`flush_pixels` refuses before writing. Full sequential decode can continue after
that refusal. The F32 path saves local samples before the existing composition
stages, so future frames still receive the composed references they require.

Visible-presentation seek targets do not identify raw layers. Use the new
`try_start_new_frame(target) -> Result<()>`: raw mode refuses before changing
state, and sequential decoding can continue. The existing infallible
`start_new_frame` remains for compatibility and panics on refusal; new callers
should use the fallible method. A future breaking revision can consolidate them.

Run `just raw-layer-output-check <fresh-artifacts>` and
`just raw-layer-output-all <fresh-artifacts>`, then apply `raw-layer-oracle`
to each output. The public-path gate checks all nine local layers against the
independently qualified tap, both renderers, complete/chunked input, untouched
row padding, early CMYK interpretation and refusal recovery. The
[public-output receipt](../benchmarks/raw_layer_output_2026-10-26.json) records
actual gate results. `just raw-layer-regressions` retains existing coalesced,
scan/seek, profile, header-allowance and cooperative-stop checks.

Still unqualified: all blend-mode combinations, random raw-layer seeks,
progressive publication, other sample representations/display transformations,
negative-origin pixel fixtures, arbitrary preview/reference combinations and a
complete native workspace bound. These limitations do not become capability
claims merely because a source header can describe them.

Executed public-output gates: no-default-features plus strict workspace lint
passed (`run-heavy: done rc=0 73s | peak-RSS 0.86GiB | min-avail 12812MiB | peak-load 0.94`).
All-feature output passed (`rc=0 46s | peak-RSS 1.46GiB | min-avail 12151MiB | peak-load 0.86`).
Both artifact sets pass the independent sample/composition gate and single-bit
CMYK negative control. Existing coalescing, discovery/stop/profile and all eleven
visible-seek regressions passed (`rc=0 212s | peak-RSS 0.38GiB | min-avail 13213MiB | peak-load 2.35`).
Warning-free rustdoc and regenerated API snapshots also pass. These are
build/test process observations, not throughput or decoder memory guarantees.

## Fallible CMS callbacks

A supplied `JxlCms` is honored independently of the `cms` feature, which enables
our moxcms adapter. The same internal render stage handles RGB, grayscale and
CMYK. Initialization must return the requested number of transforms and the
output profile's channel count. A malformed response returns an error before
any transform runs. A callback error keeps its original `whereat` context and
ends decoding; it never requests a mathematical fallback.

The callback receives native JXL CMYK values: zero is full ink, one is no ink.
The destination passed to the callback is the caller's requested profile. The
moxcms adapter performs its own conversion to ICC ink polarity in a fallibly
allocated input scratch buffer. Source/native sample carriage does not invert
or relabel values. CMYK conversion remains after composition where reference
frames require it.

`src/tests/cms_contract.rs` exercises real RGB, grayscale and layered CMYK files
through both renderers, with padded output rows. It covers initializer refusal,
invalid counts/channels, first and later chunk refusal, exact error context,
custom CMS without the bundled backend, native polarity, requested destination,
and parallel failure. Existing lcms2 color goldens, CMYK display checks,
parallel ICC parity and stage equivalence retain their expectations.

Caller-provided output buffers can contain partial pixels after a decode error.
Those buffers are not a completed result and must not be published as one. This
qualification does not catch arbitrary callback panics, bound callback workspace
or execution time, validate every ICC profile, or qualify resuming the low-level
mutable decoder after failure. The legacy mathematical path when no applicable
CMS is configured still exists; strict consumers must require a qualified
conversion route. Receipts: [CMS contract](../benchmarks/cms_contract_2026-10-26.json).
