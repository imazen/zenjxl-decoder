# Source interpretation for selected images

Two additive accessors expose source fields needed by image collections and
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

This is source-header qualification, not a raw-layer output API. The native
consumer still receives one composited presentation, and requesting
`coalescing=false` refuses before consuming input. A raw-output implementation
must preserve the composed references needed by later frames while exposing
uncomposited samples, geometry and blend instructions. A contained pixel-region
type cannot substitute for signed layer placement.

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
