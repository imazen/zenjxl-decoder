# Remaining decoder API decisions

These are proposals, not implemented interfaces. Existing public signatures
remain unchanged. New public interfaces require owner approval and a PR.

## Intensity-aware ICC conversion

The bundled MoxCMS path can use the decoder's private intensity-aware profile
builder. An external CMS provider, including the separate CLI's LCMS provider,
receives the intensity argument but can only call the public no-argument ICC
conversion, whose generated-profile default is 10000 nits.

Proposed additive entry point:

```rust,ignore
impl JxlColorProfile {
    pub fn try_as_icc_with_intensity_target(
        &self,
        intensity_target: f32,
    ) -> Result<Option<std::borrow::Cow<'_, Vec<u8>>>>;
}
```

An existing ICC returns borrowed bytes unchanged. A structured profile uses
the existing private builder with the supplied target. Errors are preserved;
an unsupported profile remains `Ok(None)`. The existing `try_as_icc()` and
`as_icc()` retain their present signatures and defaults. The CLI can then use
the new method for both profiles without duplicating profile generation.

Gate: port the upstream 1000-nit PQ-to-linear BT.2100 control to LCMS, retaining
the original nine expected values and 0.01 absolute bounds. Test both CLI
feature configurations and the existing exact CMYK whole-image golden.

Executed provider control: `just review-lcms-intensity` returns
`[0.8704195, 0.8705389, 0.87049097]` for the 1000-nit white, against the
unchanged expected `[1, 1, 1]` within 0.01. Supplying 10000 instead returns the
same nine float values. `just review-lcms-intensity --check` is the explicit
failing reproduction; it is not a silently passing regression test.

Recommendation: approve this single profile-level method. A second public
method on `JxlColorEncoding` is unnecessary to fix the provider boundary.

## Unknown source color information

`JxlColorEncoding` and `JxlTransferFunction` are exhaustive public enums.
Adding Unknown variants affects downstream exhaustive matches. A truthful
source descriptor also needs a policy for automatic CMS conversion; accepting
the syntax and silently returning ordinary managed sRGB is insufficient.

Recommended additive design:

- A non-exhaustive `JxlUnknownColorPolicy` with `Reject` and
  `PreserveUnmanaged`; default `Reject` preserves existing admission behavior.
- `JxlDecoderOptions::with_unknown_color_policy(policy)`.
- A non-exhaustive `JxlSourceColorDescription` containing a rendering fallback
  `JxlColorEncoding` and separate `color_space_unknown` and
  `transfer_function_unknown` flags.
- `JxlDecoder<WithImageInfo>::source_color_description()` returns that source
  description. ICC-backed images continue to expose their authoritative ICC
  through the existing profile API; Unknown descriptor flags concern the
  no-ICC structured case.

In the opt-in unmanaged mode, the specified RGB/sRGB rendering interpretation
can be used for pixel reconstruction, while source flags survive. Automatic
CMS conversion must not run for those samples. An explicit incompatible
output-profile request returns an error rather than relabeling pixels. API
documentation must distinguish the rendering fallback returned by the legacy
profile accessors from the retained source description.

Gate: valid-wire Unknown color and Unknown transfer controls, preserved flags,
no automatic CMS callback, explicit output-profile rejection, and unchanged
known-color controls. XYB parsing is independent: it already has a public enum
variant and does not need this API addition.

Decision: adopt this opt-in design or approve an enum/API break. Recommendation:
the additive opt-in design, with precise signatures reviewed before implementation.

## Raw cropped layers

The existing `coalescing(false)` option currently has no runtime effect.
`frame_layer_info()` already returns `JxlLayerInfo` source geometry and blend
facts; those facts do not by themselves change output-buffer dimensions or
pixel meaning.

Recommended first contract: opt-in local F32 layer output, with signed source
origin, local frame dimensions, explicit before/after-color-transform meaning,
and a defined extra-channel selection. Existing coalesced canvas output remains
the default. Unsupported combinations return an explicit error. Define buffer
geometry per frame before decoding its pixels; do not infer it from the image's
canvas-sized basic info. Reuse `JxlLayerInfo` where sufficient rather than
introducing a duplicate geometry type.

Decision: whether this feature is in scope, and whether the first supported
contract is local F32. The Add-cropping question belongs here, not in malformed
stream rejection. A display crop/region-of-interest API is another feature:
this fork currently has no equivalent to upstream's `crop_region`.

## Diagnostics for additional format checks

Timecode ordering and out-of-range spot-color solidity need accurate diagnostics.
Proposed additions to the already non-exhaustive `Error` enum:

```rust,ignore
InvalidTimecode { previous: u32, current: u32 },
InvalidSpotColorSolidity(f32),
```

Both classify as `InvalidBitstream`. These avoid reporting format consistency
failures as resource exhaustion or unrelated CMS errors. The associated checks
must have valid-wire positive and negative controls; cross-frame state must be
snapshotted for seek, reset and rewind.

## Squeeze compatibility target

Keep wide-arithmetic reachability separate from syntax compatibility. The
reviewed dated draft adds a longest-valid-prefix rule for implicit squeeze
defaults, while the pinned released/reference implementation generates the
full default list. Any change in implicit transforms changes how subsequent
entropy data is interpreted. Do not adopt a draft-only interpretation as a
routine rejection fix.

The explicit-grid case has an executed compatibility conflict too. The retained
`truncated_squeeze_flush_small_tiles.jxl` uses 128-pixel groups and explicit
squeezes producing a nonempty 119x33 channel with shifts (2,8). Its group-tile
height is zero. A candidate blanket admission check rejects it before the
existing partial-render assertion can execute. Released libjxl's group loop
omits zero-sized rectangles. The candidate check was not landed, and the
existing regression was not weakened. This is separate from the shared
shift-at-most-30-before-a-step check, which can be corrected independently.

Recommendation: preserve released interoperability by default. If draft-strict
admission is desired, make it an explicit mode with separate malformed-header
and partial-recovery contracts. Do not change implicit defaults silently.
