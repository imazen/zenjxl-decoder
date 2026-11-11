# Fallible zeroed image allocation — issue 18

Measured 2026-11-11 UTC on AMD Ryzen 9 5900XT (16 cores / 32 threads), Linux,
Rust 1.99.1 / LLVM 23.1.1, glibc 2.43, zenbench 0.1.10. This is an exploratory measurement, not a formal
performance-equivalence qualification.

The implementation uses `bytemuck::allocation::try_zeroed_vec::<u8>` for all
`RawImageBuffer` allocations. It removes the uninitialized `Vec::set_len` path,
keeps historical constructor signatures, and preserves alignment, allocation
footprints, OOM error mapping and parallel prefaulting. `allow-unsafe` still
controls typed-row casts.

## Allocation-only comparisons

In-memory zenbench pairs, 30 rounds per group, two arms per group. Both arms
allocate and free a vector; the benchmark does not subsequently touch every
page. Allocator history and lazy zero pages therefore matter. Means below
include all rounds; zenbench's paired analysis can remove paired outliers.

| Requested bytes | Unit | `vec![0u8; n]` mean | Bytemuck fallible mean |
|---:|:---|---:|---:|
| 64 | ns | 5.897 | 5.321 |
| 4,096 | ns | 48.646 | 49.063 |
| 4,194,304 | µs | 55.179 | 53.088 |
| 67,108,864 | µs | 7.865 | 7.923 |

At 64 MiB the paired difference's bootstrap 95% interval is
−0.517 to +0.620 µs. These observations support using the same zeroed-allocator
path without the explicit reserve/resize initialization pass; they do not
prove equivalence within a predeclared margin. The 4 MiB interval is wide;
the 4 KiB paired analysis retains 17 of 30 samples. Full paired statistics,
including outlier counts, are in
[`fallible_zeroed_allocation_2026-11-11.tsv`](fallible_zeroed_allocation_2026-11-11.tsv).

In a separate 64 MiB pair, reserve/resize measured 29.854 ms and its
`vec![0; n]` control measured 39.549 µs. That control is noisier than in the
bytemuck pair, and reserve/resize has positive drift. Do not combine the two
pairs into a claimed speedup ratio, or extrapolate these allocation-only times
to decoder throughput.

## Whole-decode comparison

The complete cross-binary sweep has 14 cases at each thread count. Positive
latency change means slower. These runs are sequential exploratory comparisons,
not paired confidence intervals across processes:

| Threads | Median case latency change | Case range |
|---:|---:|---:|
| 1 | +1.29% | −7.10% to +3.29% |
| 8 | +1.46% | −4.24% to +14.89% |

The 25,137,152-pixel portrait measured 789.05 → 771.86 ms at one thread and
178.76 → 171.18 ms at eight threads. The 10,317,824-pixel city measured
296.51 → 293.76 ms and 75.19 → 72.06 ms respectively. The eight-thread bicycle
case instead measured 5.77 → 6.63 ms (+14.89%), prompting one bounded web-case
A–B–A recheck. Every case, its rounds and throughput are retained in
[`fallible_zeroed_decode_2026-11-11.tsv`](fallible_zeroed_decode_2026-11-11.tsv).

The recheck uses the same retained binaries at eight threads, with only the
`vardct_web` groups selected:

| Fixture | Baseline A1, ms | Candidate B, ms | Baseline A2, ms |
|:---|---:|---:|---:|
| café | 17.6167 | 18.6907 | 17.7106 |
| bicycles | 5.3693 | 5.8927 | 5.8024 |

Café is 5.53–6.10% slower against the surrounding baseline runs. Bicycle is
1.56–9.75% slower, with baseline drift; the original +14.89% magnitude does
not repeat. These observations show that matching infallible **zeroed**
allocation does not guarantee matching the historical **uninitialized**
output path's decoder speed. They do not establish a performance guarantee
across allocators, platforms or workloads. No further tuning or threshold
changes are part of this change. Recheck data is in
[`fallible_zeroed_web_witness_2026-11-11.tsv`](fallible_zeroed_web_witness_2026-11-11.tsv).

The original full sweeps report 31/108 noisy rounds for baseline 1T/8T and
21/180 for candidate 1T/8T. All four runs and all three witness runs report
`unreliable=false`; that field is not a substitute for process-level
confirmation. No allocation-group warmup precedes these decoder runs.

Completed full-sweep resource records (not a claim of reduced decoder memory):

| Run | Elapsed | Peak RSS | Min available RAM | Peak load |
|:---|---:|---:|---:|---:|
| Baseline 1T | 226 s | 0.50 GiB | 46,846 MiB | 1.31 |
| Baseline 8T | 310 s | 0.47 GiB | 46,961 MiB | 2.40 |
| Candidate 1T | 228 s | 0.50 GiB | 46,996 MiB | 1.43 |
| Candidate 8T | 353 s | 0.47 GiB | 46,047 MiB | 2.64 |

Every benchmark ran through `run-heavy --mem 8G --jobs 8` with the same
low-priority scheduling. Witness runs report 0.41 GiB peak RSS each.

## Source and reproduction

- Baseline implementation: `803d6ab07127ab0a5022ed52223421f0d7d71fdd`.
- Allocation implementation: `1cb9773339854ed6e510a3ad167ff5c78a4639fa`.
- Isolated decoder benchmark harness: `9429cfd66a9ebc06ed9cec229b6fc84b66344019`.
- Release profile: optimization level 3, thin LTO, one codegen unit. Default
  SIMD features plus `threads,allow-unsafe`; no `target-cpu=native` or x86
  rustflags. Both binaries use the same `Cargo.lock`.
- Decode work is `zenjxl_decoder::decode(&bytes)` in memory; fixture loading,
  probing and process startup are outside timers. `RAYON_NUM_THREADS` is 1 or 8.
  The suite has 14 benchmark cases covering 13 unique bundled fixtures; the
  grayscale modular fixture appears in two groups.

Build and retain each revision's executable before rebuilding:

```sh
TMPDIR="$HOME/tmp" CARGO_TARGET_DIR="$HOME/tmp/decoder-issue18/target" \
  run-heavy --mem 8G --jobs 4 -- \
  cargo bench --locked -p zenjxl-decoder --bench decode_bench \
    --features threads,allow-unsafe --no-run
```

Copy the executable reported by Cargo outside `target/`. On an eligible idle
host, invoke each retained binary through `run-heavy`:

```sh
run-heavy --mem 8G --jobs 8 -- \
  just allocation-decode-measure "$HOME/tmp/decoder-issue18/baseline" base-before 1
run-heavy --mem 8G --jobs 8 -- \
  just allocation-decode-measure "$HOME/tmp/decoder-issue18/candidate-bench-v3" candidate-v3 1
# Repeat each with threads=8.
just allocation-summary "$HOME/tmp/decoder-issue18/logs"
# Allocation-only groups use a separate invocation:
run-heavy --mem 8G --jobs 1 -- \
  just allocation-decode-measure "$HOME/tmp/decoder-issue18/candidate-bench-v3" allocator 1 zeroed_allocation
```

The retained allocator measurements used the earlier two-arm harness (v2),
whose source and binary identity are archived with the raw results. Its
allocation groups ran before the decoder groups. Only its allocation results
are used; its decoder results are excluded. The final v3 decoder harness runs
no allocation benchmarks before the decode groups. An initial three-arm
allocation attempt was interrupted because calibration assigned too many
repetitions to reserve/resize. Both interrupted attempts and the mixed v2
run remain archived and excluded from the decoder comparison.

## Validation

`just allocation-check` passed on Ryzen 9 7900X with a separate Cargo output
directory, `run-heavy --mem 6G --jobs 3`, four Rayon and libtest threads:

- Workspace all-target Clippy with all features and no default features.
- Release image tests with all features and no default features, including
  initialized storage/padding, layout-allocation failure, strided output and
  padded-memory-budget accounting.
- Existing release byte-exact dispatch and parallel determinism regressions.
- Existing modular-recycling regression against its libjxl-derived pixel hash,
  sequentially and with one-/four-thread pools.
- Existing oversized padded-output memory-limit regression.

No test expectations, ignores or corpus policies changed. These are targeted
local gates, not a claim that the entire library suite ran locally.

Validation wrapper: `run-heavy: done rc=0 302s | peak-RSS 4.35GiB |
min-avail 5323MiB | peak-load 2.73`. The final benchmark harness also passed
release compilation and targeted Clippy. Production source did not change
between the validated implementation and the harness correction.

## Retained evidence

Raw JSON timing samples, complete stdout/stderr logs, wrapper resource records,
source identities and the earlier allocator harness are retained on `dev` at
`/mnt/v/output/zenjxl-decoder/issue18-2026-11-11/`, mirrored under the same
project directory in `/mnt/tower/output/`. `SHA256SUMS` records file identities.
The raw data is outside git; the committed scalar summaries remain below 30 KB.
No diagnostic image maps or decoded image arrays were saved.

Retained executable SHA-256 identities:

- Baseline: `624284704c28d0e7169e48503d58fa62519fb7480978b5e854c67c322b580d48`.
- Candidate v3 decoder: `7f5110453192fe76225bf459250feb72a0add08f2d603cd2355d9d501d0d4953`.
- Candidate v2 allocator: `14f4fa2b80c16b93866d61df51368503186b17cfc13e974692b788bff33091f4`.

The executables were copied out of Cargo output before subsequent builds. Source
exports carry the pinned commit identities and per-file hashes because their
benchmark JSON has no Git checkout to identify automatically.
