# Codestream layout recording

This proposal adds `CodestreamLayout`, `FrameLayout`, `NamedSpan`,
`JxlDecoderOptions::record_layout` / `with_record_layout(bool)`, and
`JxlDecoder::codestream_layout() -> &CodestreamLayout`.

Recording defaults to false. The returned layout is filled as headers and
TOCs complete; observing a partial layout does not mean image decoding has
completed. Records include preview/visibility/reference facts, names,
extensions, compressed ICC bounds, TOCs and section byte ranges.

Offsets refer to the logical codestream: concatenated jxlc/jxlp payloads,
excluding jxlp index words. Header/name/extension ranges are bits; ICC and TOC
ranges are bits; section ranges are bytes in file order. These are not
container-file offsets. Seek retains absolute offsets and replaces repeated
frame records instead of appending duplicates. Preview is marked separately.

Retained vectors/text and temporary growth share the decoder memory budget.
Growth reserves both old and replacement allocations until the move completes.
Parse-log scopes restore enclosing logs on success and failure. Caller-created
clones are caller-owned allocations.

The refresh starts from current main `43d3069e`; it preserves its transform,
frame-rate, ICC, HF-table and structural-container fixes. The original PR head
is retained as a merge parent, and the PR now targets main rather than PR65.
The existing 10,454-byte ICC fixture is unchanged, with SHA-256
`37c66f4835f4759d70f1529b094ce19a06b66b767bbca44d1d6b642df2ac1d02`.

Targeted all-feature/no-default-feature gates cover fragmented input, preview,
seeks, unchanged pixel bits, the large ICC profile, shared-budget competition,
allocation growth and parse-log restoration. Strict all-target workspace
Clippy passes. Full-suite and fresh default-off overhead receipts are being
completed before the final review summary; old performance/fuzz numbers are
not presented as current-source measurements.

`perf/layout-overhead` pairs the recording-disabled candidate with pinned
runtime main `552da568` using zenbench in one process. Both arms perform header
scanning and skip pixels. This measures the complete disabled-layout change,
not only its thread-local checks, and does not claim whole-image decode cost.
