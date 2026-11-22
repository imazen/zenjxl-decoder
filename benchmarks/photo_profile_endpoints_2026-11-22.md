# Cold and caller-owned output performance of the photo encode profiles

The benchmark reuses the selected imazen-26 streams from
[jxl-encoder 25ca0c65](https://github.com/imazen/jxl-encoder/commit/25ca0c659e3f55a0526219c50b8817878f0e1be7).
The lossless balanced comparison uses tier 0 as its reference and
the throughput comparison uses stock `--fast-decode -g 0`. Both
use three native photos. The lossy comparison uses five native
photos near d1 and d4 with the existing matched rendered-quality
selection and tier-0 references. Both d4 profiles selected the
same five streams. The Display P3 photo keeps its identical
tier-0 stream in both distance bands. A mixed page and a small RGBA
fixture serve as identical-file controls for lossless balanced.

The cold endpoint starts a fresh process per sample and times its
first in-memory decode into freshly allocated RGB8 or RGBA8.
Allocation and prefaulting fall inside the timing boundary. The
warm reuse endpoint uses [paired-output](../scripts/perf/README.md#timing)
from `245165e18e5f020da26f20b2daca3f532a340a0d`; each timed decode
builds a fresh decoder and writes into its arm's preallocated
output buffer. Its baseline and candidate render into the same
dimensions and layout. Each arm reproduces its own allocating
decode before timing, after the first timed decode into a cleared
buffer, and after timing. Lossless pairs require cross-stream
pixel equality at both endpoints. Lossy quality and same-stream
accuracy are bound to the earlier oracle checks by stream hash.

All timings use the current decoder library from `6000956d`, CMS
off and the same pinned P-core sets on i270 at 1, 4 and 8 workers.
Every comparison includes three interleaved blocks of five
samples plus A/A. Each speedup is the median of paired block
median latency ratios. Aggregate speedups are medians across
sources. Class p95 uses pooled latencies normalized by each
stream's baseline median. A cell with 15 samples uses its largest
latency as nearest-rank p95.

Strict Clippy passed for the benchmark binary. Smoke checks cover
the existing single-list mode, the exact-pair mode, paired input
counts, differing output shapes, and pixel mismatches in exact
mode. The benchmark retains the exact input and binary hashes,
commands, samples, and cold lossless PNG16 readback hashes.

| Profile | Cold 1 / 4 / 8 workers | Warm reused 1 / 4 / 8 workers |
|---|---:|---:|
| Lossless balanced | 2.645× / 2.361× / 2.210× | 2.674× / 2.324× / 2.153× |
| Lossless throughput | 1.373× / 1.404× / 1.147× | 1.414× / 1.555× / 1.532× |
| Lossy balanced near d1 | 1.085× / 1.083× / 1.064× | 1.091× / 1.085× / 1.095× |
| Lossy throughput near d1 | 1.208× / 1.541× / 1.540× | 1.225× / 1.479× / 1.502× |
| Lossy throughput near d4 | 1.367× / 1.224× / 1.218× | 1.388× / 1.318× / 1.364× |

Lossless throughput reaches 146.756 / 394.747 / 516.357 MP/s
cold and 167.795 / 530.027 / 703.904 MP/s with warm reused output.
Its eight-worker cold median is below 550 MP/s. The warm
allocating production result was 573.60 MP/s. The codecs retain
their profile gains on a reused destination, where allocation and
prefaulting have already happened. The two lossy throughput bands
clear the relative photo targets at both measured endpoints.

The initial individual p95 ratios above 1.10 are all on unchanged
streams. At eight workers the cold d4 Display P3 fallback reaches
1.105×, the cold RGBA control 2.049×, and the reused RGBA control
1.311×. Each corresponding 75-sample replay falls within 10%
at 0.982×, 1.020× and 1.033× respectively. The reused RGBA replay
has comparable median variation in the two identical arms
(0.959× candidate and 0.948× A/A speedup). The first 4-worker
cold mixed-page control also has a 1.388× A/A p95 versus a
0.865× candidate p95. All photo class p95 ratios are within
the 10% limit in the primary data.

[Class medians and p95](photo_profile_endpoints_aggregate_2026-11-22.tsv).
[All 138 stream and endpoint cells](photo_profile_endpoints_2026-11-22.tsv).
[Cold](photo_profile_endpoints_cold_control_repeat_2026-11-22.tsv)
and [reused](photo_profile_endpoints_reused_control_repeat_2026-11-22.tsv)
eight-worker replays retain the original tests as well as the
extended controls.

The initial cold sweep took 209 s with 0.20 GiB peak RSS.
The build, remaining cold pairs and all reused pairs took
604 s with 0.91 GiB peak RSS. The cold tail replay took 12 s
with 0.22 GiB peak RSS. The reused tail replay used 0.01 GiB.
Raw evidence and reproductions are under
`/mnt/v/output/zenjxl-decoder/photo-profile-endpoints-2026-11-22`
on i270.
