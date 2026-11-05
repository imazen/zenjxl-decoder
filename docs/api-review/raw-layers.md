# Qualified local F32 layers

This draft is stacked on the source-metadata proposal. It implements the existing
`JxlDecoderOptions::coalescing` / `with_coalescing(bool)` option, default true.
On main, false is currently accepted but ignored.
False emits regular local layers before compositing; stored source geometry
and blend facts come from `frame_layer_info() -> Result<JxlLayerInfo>`.
Full local extents include samples outside the canvas. Composed reference
frames remain internal for subsequent decoding.

The supported mode requires stored orientation, skipped previews, F32 samples,
no CMS, tone mapping or premultiplication, separate unrendered spot channels,
and native CMYK components including black. Header-only scans need no F32
buffer. Unsupported formats/transforms, visible-frame seeking and progressive
raw flushing return explicit errors. Low-level pixel-format changes are
rechecked before writing buffers. Refused seeks/flushes retain sequential
progress. No flattened substitute is returned.

Missing qualification: every blend-mode/reference combination, negative-origin
pixel fixtures, arbitrary preview/reference combinations, random raw seeks,
progressive raw output and non-F32 representations. RGB agreement with libjxl
is not bit-exact. Keep this draft until its API and supported scope are accepted.

Current-source evidence uses the existing two layered fixtures and an
independent C API reader built from checksum-pinned libjxl v0.12 source
`a7a9c787341cf703dede03c2009fa460cae5e5df`. This runtime is distinct from the
separately pinned Clang static oracle used in the PQ investigation.

- RGB/alpha: five layers, 18,874,368 samples, 2,265,549 different float bit
  patterns; maximum absolute error 0.00036650896072387695. Every sample retains
  the established 1/256 F32 conformance bound. Alpha samples are exact.
- CMYK/alpha: four layers, 1,587,120 samples, zero different float bits.
- Geometry, channel layout and blend/reference metadata agree exactly.
- The independent gate rejects one changed CMYK mantissa bit even though that
  change is below the lossy float tolerance. Original captures remain unchanged.
- Native simple/low-memory renderers and input chunkings require identical float
  bits, and padded-row sentinels remain unchanged. Both feature configurations,
  strict all-target Clippy and the full all-feature workspace pass. The six
  frozen downstream RGBA frames remain byte-exact; no existing gate is relaxed.

[`benchmarks/raw_layers_2026-11-05.json`](../../benchmarks/raw_layers_2026-11-05.json)
retains the compact receipt. `just raw-layer-oracle <artifacts> <oracle-root>`
exports through the public decoder, builds the independent reader, compares
both fixtures and runs the negative control. Use a fresh artifact directory
and a pinned reference root built by `just build-feature-oracle`. The caller
holds the shared heavy lock and invokes run-heavy. No encoder/oracle enters the
production dependency graph, and no generated binary fixture is committed.
