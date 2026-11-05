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

The proposal is adapted to main `b04af24c`, including the approved Send,
source metadata and operation contracts, strict inventory resolution and spline
point-spacing repair. The original PR head remains in ancestry. The existing
10,454-byte ICC fixture is unchanged, SHA-256
`37c66f4835f4759d70f1529b094ce19a06b66b767bbca44d1d6b642df2ac1d02`.

The disabled logger uses a constant-initialized TLS Cell to return before
initializing/borrowing its drop-bearing RefCell. Nested sessions, taking logs,
error cleanup and budget behavior retain coverage.

Paired zenbench header scans on i270 use a real uniform cpuset (CPUs0–3),
release/thinLTO, no target-cpu=native and30 exploratory rounds. The same binary
compares the refactored candidate, the previous layout implementation
`65e776ab`, and approved runtime main `6d060afb`:

| Fixture | Refactor versus prior layout | Refactored layout versus main |
|---|---|---|
| Basic | -1.42%,95%CI[-1.82,-1.01]%,significant | +0.62%,not significant |
| Layered | -1.44%,not significant | +1.00%,not significant |
| CMYK | -0.84%,not significant | -0.90%,not significant |

These observations measure complete header scans with pixels skipped. They do
not establish zero overhead or full-image throughput. The later main spline
repair changes a pixel kernel, not header scanning.

Perf stat3 runs of1,000,000 basic scans measures instructions21,632,466,925
before and21,451,488,628 after; cycles12,432,842,563 before and12,189,520,325
after. Earlier cycle sampling puts35.89% self samples in memmove. This does not
attribute the complete layout cost to TLS logging.

Both feature contract gates cover fragmented input, preview, seek, unchanged
pixel bits, the large ICC profile, shared-budget competition, allocation growth
and nested/error parse-log restoration. Strict all-target Clippy and the full
all-feature workspace pass on the profiled implementation. The combined
arithmetic/layout gate completed rc0 in364s, peakRSS7.06GiB,
minimum available18951MiB,peakload11.48. That is validation process-tree usage.
[`benchmarks/layout_disabled_fast_path_2026-11-05.txt`](../../benchmarks/layout_disabled_fast_path_2026-11-05.txt)
retains source pins, raw pairs and counters. The older, different-host/base
measurement is retained as a historical receipt in
[`benchmarks/layout_disabled_2026-11-05.txt`](../../benchmarks/layout_disabled_2026-11-05.txt).
No threshold, assertion, ignore or corpus policy is relaxed.
