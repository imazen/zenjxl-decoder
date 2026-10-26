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
roles, nonzero source dimension shifts and arbitrary combinations remain open.

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
