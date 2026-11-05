# Source metadata and fallible frame selection

The source-metadata API landed in PR68 at `3d282df2`. It changes no
pixel-output behavior. Raw local F32 output remains separate in draft PR75;
the independent operational limits landed in PR74.

- `extra_channel_details(index)` returns an owned optional snapshot with
  integer/float source precision, exponent bits, spot RGBA parameters and CFA
  channel number. It uses the existing extra-channel index; out-of-range is None.
- `frame_timecode()` distinguishes absent timecodes from a present zero value.
  It returns the source u32 without decimal reinterpretation.
- `frame_layer_info() -> Result<JxlLayerInfo>` returns signed stored-canvas
  origin, full local dimensions, per-channel blend instructions, reference-slot
  facts, duration ticks and last-frame status. It allocates its blend list
  fallibly. These facts do not change the requested output geometry.
- `try_start_new_frame(target) -> Result<()>` exposes the existing native seek
  errors. The infallible convenience remains source compatible. Targets are
  source-specific; they can be used by initialized decoders for the same input,
  and are not portable image identities.

New public snapshot types are non-exhaustive. Floating-point spot details live
in the new type, preserving the existing Eq structs and fields. A layer's
reference-slot number belongs to native decoder/reference state; the snapshot
is not an export of every internal reference representation.

Current-main conformance bounds, ICC-before-basic-info visibility, rendering,
CMS, stored-orientation fields and crop/precision repairs remain intact. No
normal encoder/oracle dependency is added. The standalone qualification crate
pins the native encoder, generates fixtures at runtime and checks them with an
independent libjxl v0.12 C reader.

Fresh all-feature/no-default-feature source/profile/geometry gates and strict
workspace Clippy pass. Four independent-oracle tests pass for seven extra roles,
integer/float/exponent precision, spot/CFA parameters, present-zero/absent
timecodes and reverse seek selection. The source-geometry getter also retains
composited output dimensions while exposing the source's off-canvas extent.
No generated binary fixture or old benchmark dump is committed.
