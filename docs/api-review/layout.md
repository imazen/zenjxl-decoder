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
Clippy and the full all-feature workspace pass. Old performance/fuzz numbers
are not presented as current-source measurements.

`perf/layout-overhead` pairs the recording-disabled candidate with pinned
runtime main `552da568` using zenbench in one process. Both arms perform header
scanning and skip pixels. This measures the complete disabled-layout change,
not only its thread-local checks, and does not claim whole-image decode cost.

The fresh paired header-scan measurement finds a disabled-recording cost:

| Fixture | Baseline mean | Candidate mean | Paired change (95% CI) |
|---|---:|---:|---:|
| Basic | 2.78 µs | 2.99 µs | +7.72% [+7.10%, +8.32%] |
| Layered | 9.68 µs | 10.07 µs | +4.79% [+4.15%, +5.45%] |
| CMYK | 15.19 ms | 14.86 ms | No significant difference; CV 15.2% / 17.5% |

These are exploratory 30-round zenbench pairs on r5900xt, release/thin-LTO,
without target-cpu=native. Both arms run on CPUs 0–3. The complete raw/layout
validation wrapper finished rc=0 in 567s, peak RSS 7.29 GiB, minimum available
36,471 MiB, peak load 7.03. That resource receipt covers the combined tests
and benchmark, rather than standalone decoder memory usage. The small-header
regression is a review trade-off; zero overhead is not established.
[`benchmarks/layout_disabled_2026-11-05.txt`](../../benchmarks/layout_disabled_2026-11-05.txt)
retains the raw pairs and source identities. No runtime threshold is changed.

## Further investigation after the API merges

The proposal is adapted to approved runtime main `6d060afb`; current main's
later documentation corrections are retained. Existing frame-header budgets,
source facts, Send migration and structural/I/O errors remain intact.

Hardware profiling on i270 finds substantial state movement in the header-only
workload: memmove accounts for 35.89% of candidate cycle samples. This does not
attribute the entire patch's overhead to span logging. The disabled logger
previously initialized/borrowed its drop-bearing TLS RefCell for every tracked
field. A constant-initialized TLS Cell now returns before that borrow when off;
nested sessions and taking/restoring logs retain the original behavior.

The new same-process before/candidate zenbench pairs use an enforced uniform
performance-core cpuset, CPUs0-3, release/thinLTO, no target-cpu=native and30
exploratory rounds. Against the unrefactored layout code, the basic scan improves
1.42% (95%CI[-1.82%,-1.01%], significant). Layered -1.44% and CMYK -0.84% do
not satisfy the harness significance gate and are not claimed as speedups.

Against approved main, the refactored candidate reports basic +0.62%, layered
+1.00% and CMYK -0.90%; none meets that significance gate. These observations
are specific to this host/workload and do not establish zero cost or full-image
throughput. The earlier measurements above remain historical receipts.

Both feature contract gates, strict all-target Clippy and the full all-feature
workspace pass. The combined validation wrapper completed rc0 in364s, peakRSS
7.06GiB, minimum available18951MiB, peakload11.48. This is validation process-tree
usage. The raw paired measurements and counters are retained in
[`benchmarks/layout_disabled_fast_path_2026-11-05.txt`](../../benchmarks/layout_disabled_fast_path_2026-11-05.txt).
