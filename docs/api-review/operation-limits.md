# Stop reasons and frame-header discovery

This proposal is independent of source metadata and raw-layer output.

## Stop API

- Keep explicit cancellation as `Error::Cancelled`.
- Add `Error::Stopped { reason: StopReason }` for timeout and future reasons.
- Add `Error::stop_reason() -> Option<StopReason>`.
- Keep both variants in the existing `ErrorClass::Cancelled` coarse class.

The original `whereat` context is retained. This changes the precise native
variant for timeouts; consumers inspecting variants should review that behavior.
It does not turn callback execution into an interruptible operation.

## Header budget

Add `JxlDecoderLimits::max_frame_headers: Option<u64>` and
`with_max_frame_headers(u64)`. All existing presets default to `None`.

The cap counts completed preview, reference-only, LF and regular frame headers,
including replay through seek or rewind. It checks admission before another
header starts. Partial-input retries do not consume another count. Full reset
starts a new operation; seek and rewind preserve the accumulated charge.

This bounds the number of headers, not bytes or work inside one header. Existing
memory, image, transform, ICC and stop limits remain necessary and unchanged.
Refusal uses `LimitExceeded { resource: "frame_headers", actual, limit }`.

## Fresh qualification

All-feature and no-default-feature tests cover hidden frames, previews, LF
frames, one-byte input, repeated exhausted calls, seek/rewind/reset boundaries,
unchanged admitted pixels, exact stop reasons and existing cancellation.
Strict all-target/all-feature workspace Clippy passes. No existing accuracy
threshold, expected pixel, ignore, or default resource cap is relaxed.

Source is adapted to main `552da568`, including its current conformance checks
and shared HF section-table budgeting. The public additions remain in this PR
for owner review; none is landed on main.
