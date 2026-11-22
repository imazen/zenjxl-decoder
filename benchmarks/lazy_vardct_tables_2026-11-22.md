# Lazy standard VarDCT tables on i270

[Upstream jxl-rs](https://github.com/libjxl/jxl-rs/commit/7d250dbd2) defers
building standard quantization matrices and natural coefficient orders
until the corresponding transform is used. The candidate adapts that
policy to this decoder's existing per-transform static caches. Custom
tables and custom coefficient orders still parse and compute at HF-global.

Compare decoder `e58a4838725704e37d90f2acd1d11dcccd7ab98a` with the lazy
candidate using `all-simd,threads` release builds on i270. Use 24
streams from six imazen-26 SDR sources, center cropped to 64 or 256
pixels and encoded at effort 7, distances 1 and 4. Warm timing adds
progressive, alpha, floating-point Modular, and bike controls. The timed
endpoint allocates and produces u8 output with CMS disabled. Each sample
starts a new process. Cold timing measures its first in-memory decode,
and warm timing measures the immediate second decode. Three blocks of
five decodes per arm include a second baseline for A/A.

| Workers | Cold baseline / candidate median (ms) | Cold median paired latency reduction | Warm median latency change |
|---|---:|---:|---:|
| 1 | 4.866 / 0.895 | 81.9% | −0.22% |
| 4 | 5.247 / 0.997 | 81.2% | +0.21% |
| 8 | 5.380 / 1.051 | 80.8% | +1.53% |

The columns report medians across cells. Latency change first divides
the candidate cell median by the baseline cell median. All cold cells
beat the 20% latency target. All 72 cold and 84 warm PNG16 comparisons
match exactly across builds and worker counts.

Nearest-rank p95 over 15 warm samples is the worst observation. The
floating-point Modular control has a +12.7% one-worker p95 with a
−0.3% A/A and −0.2% median change. The compact SDR classes meet the
class p95 limit after pooling latency normalized by each stream's
baseline median. The table retains every cell's median and p95.

A nine-block confirmation used 45 samples per arm on the floating-point
control, the progressive control, and two affected SDR streams. The
floating-point control's one-worker p95 was −0.04% on replay, with
+1.34% A/A. All 12 replay cells met the p95 limit. The 64 px photo at
d1 still had +3.1% and +5.1% median latency at 4 and 8 workers,
respectively. The selected terminal crop had +3.1% at 8 workers.
The full 28-cell warm result captures the aggregate cost. The replay
used 0.43 GiB peak RSS over 138 seconds.

All 24 encoded streams fully render in djxl v0.12.0 (`a7a9c78`) and
jxl-rs (`f370be423933628eb7f783f1b61d4c765eca5f79`). The
coefficient-order and quantization-table goldens and 13 selected
progressive, HDR, alpha flush, and streaming library tests pass. The
first corpus-backed test invocation lacked `CODEC_CORPUS_PATH`; the
passing run supplies the existing codec corpus with no skipped tests.

Run-heavy peak RSS was 1.50 GiB for the build and focused tests, and
0.34 GiB for crop preparation, cold timing and the first warm attempt.
That attempt reached a Level 5 admission error for a 32-bit float
fixture in the baseline. The warm rerun explicitly enables Level 10
for both binaries and used 0.42 GiB peak RSS.

[Per-stream results](lazy_vardct_tables_2026-11-22.tsv). Per-decode durations
are derived from the CLI's per-sample MP/s and pixel count. The streams,
pinned commands, binary and stream hashes, exact readback hashes, and
logs are retained on i270 under
`/mnt/v/output/zenjxl-decoder/lazy-vardct-2026-11-22`.
