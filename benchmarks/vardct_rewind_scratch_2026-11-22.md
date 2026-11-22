# One retained VarDCT scratch set across decoder rewinds

[Upstream jxl-rs](https://github.com/libjxl/jxl-rs/commit/e33465298)
moves its recycler and per-thread scratch ownership to the decoder.
Zen's baseline already reuses VarDCT scratch within each frame and
keeps one scratch set after each parallel render call. Its frame
finalization drops that set.

Candidate `dd4146d6879ef7f1b536cb7d59edfee41aeb3c03` carries that one
set through frame finalization, decoder rewind and the internal
full reset. It clears the tracked nonzero-count maps before moving
the remaining buffers to a new frame. The benchmark isolates this
one retained set. The upstream per-thread recycler has a wider scope.

On i270, the two binaries use Rust 1.99.1, locked release dependencies,
`all-simd,threads` and no CMS. The fixture set contains three native
VarDCT photos, one 1024 px VarDCT crop and three Modular controls,
including RGBA. Tests of partial-input reset and frame-header
accounting across rewind and reset both pass. Every benchmark process
checks exact RGB8 or RGBA8 rows across all arms before timing, after
the first timed pass with zeroed output buffers, and after timing.

For the three native VarDCT photos, the paired median speedup of
the reuse-and-rewind arm is 1.005×, 0.999× and 0.981× at one, four
and eight workers. A/A medians are 1.001×, 1.005× and 0.996×.
For the same photos, the allocating control is 1.005×, 0.998×
and 0.987×. All 84 endpoint and stream cells meet the 10% p95
latency limit. The one-set policy misses the 5% throughput target
and is archived under `rejected/vardct-single-scratch-rewind`.

The [paired data](vardct_rewind_scratch_2026-11-22.tsv) compare
baseline `9c745952e5290e6feb2ed553fa0010c3f5a221bc` and the
candidate at identical allocation or reuse endpoints. Three outer
blocks rotate the baseline, candidate and A/A process order. Each
process interleaves three blocks of five samples in four endpoint
arms. Thus p95 uses 45 individual samples per binary, stream,
worker count and endpoint. Timing repeatedly decodes the same
file. Cross-file reset is exercised by the existing functional test.

The builds, tests and measurements took 836 s with 1.50 GiB peak
RSS. Both executable hashes, source and stream identities, exact
commands and raw samples are under
`/mnt/v/output/zenjxl-decoder/vardct-rewind-scratch-2026-11-22`
on i270.
