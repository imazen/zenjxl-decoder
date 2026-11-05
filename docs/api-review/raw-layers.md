# Qualified local F32 layers

This draft is adapted to main after the approved Send, source-metadata and
operation-contract APIs, the safe inventory resolution and the spline point-spacing repair. It implements the existing
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

Current evidence uses the two existing layered fixtures and an independent C
API reader built from checksum-pinned libjxl v0.12 source
`a7a9c787341cf703dede03c2009fa460cae5e5df`. This GNU-built runtime differs from
the Clang static oracle at4128790 used in the separate PQ investigation; no
frozen oracle is replaced.

The RGB fixture has five Modular layers: layer0 has noise, layers1/4 splines,
layers2/3 neither. The four CMYK layers have neither noise nor splines.
Independent post-repair results:

| Mode | RGB/alpha samples | Different float bits | Maximum absolute error |
|---|---:|---:|---:|
| Existing default precision | 18,874,368 | 2,265,542 | 0.0003052426764043048 |
| Existing high_precision=true | 18,874,368 | 2,195,471 | 0.0000010728836059570312 |

All1,587,120 CMYK/alpha samples, RGB alpha samples, geometry and blend/reference
metadata agree exactly in both comparisons. The one-bit CMYK negative control
rejects exactly the altered mantissa bit; original captures are unchanged.
RGB retains the existing1/256 F32 conformance bound and is not called bit-exact.

Native default false uses spline cutoff exponent3; true uses5. Pinned v0.12
common.h defaults JXL_HIGH_PRECISION=1 and its build has no override, so matching
policy requires the existing true option. The decoder's default is unchanged.

The separate private repair on main preserves the reference point walker's
operation order, recomputing the remaining distance instead of advancing a
precomputed unit vector. At matched precision the pre-repair maximum was
0.000348508358001709; the corrected maximum is0.0000010728836059570312.
The C++ float-bit point-spacing golden fails before and passes after the repair.
This draft consumes that repair from main.

Further isolated arithmetic diagnostics cover fused noise/fast-math operations,
a fixed eight-lane spline reduction, double DCT constants and the reference
color floor. The combined matched-precision prototype reaches maximum error
0.000000476837158203125 but still has1,812,678 differing float patterns. Those
experimental changes remain in the test tool; they are not production fixes.
DCT constants alone made no observed difference on this fixture. Existing
numerical checks retain their assertions and existing ignores.

Both feature-specific raw contract gates, strict all-target Clippy, the full
all-feature workspace and six frozen exact downstream RGBA frames pass after
the main repair. Native simple/low-memory renderers and chunkings require
identical float bits; packed rows retain padding sentinels. These fixture gates
do not establish complete raw-layer qualification.

[`benchmarks/raw_layers_spacing_2026-11-05.json`](../../benchmarks/raw_layers_spacing_2026-11-05.json)
retains current source/oracle pins and results. The pre-repair receipt remains
unchanged in
[`benchmarks/raw_layers_2026-11-05.json`](../../benchmarks/raw_layers_2026-11-05.json).
`just raw-layer-oracle <artifacts> <oracle-root>` reproduces the public export,
independent comparison and negative control. Use a fresh artifact directory
and the pinned reference root from `just build-feature-oracle`; hold the shared
heavy lock and invoke run-heavy. No oracle enters the production dependency
graph and no generated binary fixture is committed.

`precision_experiments.py --case unchanged --case high-precision` compares the
current policy settings. To reproduce historical point-spacing experiments,
use `--source-root` with an export of34a99d25 and label it with
`--source-revision`. The tool rejects a reference-spacing experiment if the
selected source already contains the repair, and records a Rust-source hash.
