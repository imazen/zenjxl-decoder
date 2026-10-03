# Decoder investigation notes

## JPEG standalone restart markers — 2026-09-24

[PROVEN] `jpeg::writer::write_jpeg` omitted RST0–RST7 entries from JBRD
`marker_order`. These are standalone markers, including a restart after the
last MCU; restart markers inside scans are reconstructed separately in
`write_sos`. libjxl v0.12 `lib/jxl/jpeg/dec_jpeg_data_writer.cc::EncodeRestart`
writes the standalone marker's two bytes without deriving a new restart index.

`jpeg::writer::tests::standalone_restart_markers_keep_original_order` failed
before the correction: only SOI, EOI and trailing data survived. It covers all
eight marker values in a deliberately nonsequential order. The matching encoder
scanner correction is jxl-encoder `1a40b8f3`; its conformance harness at
`a6a9a116` records 21 camera originals reconstructing two bytes short before
this writer fix, and validates reconstruction with both Rust and libjxl v0.12.
No public API or scan-entropy algorithm changes are required.

Validation: release library tests with `jpeg` pass under CI's explicit
`ZENJXL_ALLOW_MISSING_CORPUS=1` configuration (834 reported passes, 28 existing
ignores). Of those reported passes, 103 corpus-backed feature cases do not run
because the Mac lacks their corpus; the unconfigured run fails those 103 cases
and passes the other 731. All eight JPEG reconstruction tests and the new
standalone-marker regression execute and pass. Logs are under
`~/tmp/jxl-backlog/decoder120-*` and `~/tmp/zenjxl-decoder/jpeg-issue120.log`.
`just jpeg-reconstruction-check <label>` reproduces the release library gate;
set the corpus policy explicitly at invocation, as CI does.
Workspace all-target, all-feature Clippy and scoped formatting also pass.

## WASI regression deadlines — 2026-09-24

[PROVEN] The two deadline regressions in `jxlrs_testdata_ports` used
`std::thread::spawn`, which traps on wasm32-wasip1. They now live in the
`regression_deadlines` integration binary with unchanged decode assertions.
Native execution keeps each 20-second worker deadline. The WASI runner applies
Wasmtime's `-W timeout=20s` to the entire dedicated binary, including both
regressions; the guest refuses to run without the runner's deadline marker.
This also bounds a decoder that never returns or polls cancellation. The
WASI suites pass with no features (767 library tests reported) and `wasm128`
(808), each with 37 existing ignores; all integration and doctest binaries
pass. Local runs use `ARBTEST_BUDGET_MS=100` and CI's explicit missing-corpus
policy; these counts do not imply external-corpus coverage. An intentional
infinite-loop module run through the same runner traps with `interrupt` at
the configured deadline.
Both WASI jobs also pass on commit `440cc002` in
[CI run 36008146603](https://github.com/imazen/zenjxl-decoder/actions/runs/36008146603).
`just wasm-ci-check <label>` runs both CI feature configurations and saves
complete logs under `~/tmp/zenjxl-decoder/` with CI's explicit missing-corpus
policy.

## i686 fixture sweep memory — 2026-09-24

[PROVEN] `run_fixture_sweep` creates up to eight workers independently of
libtest's `--test-threads=1`. The exported Linux i686 baseline (executed through QEMU on WSL) reproduces CI's
`compare_pipelines_sweep` failure at `ImageOutOfMemory(42048, 5377)`, plus two
`ImageOutOfMemory(32512, 2704)` failures. The comparison helper deliberately disables decoder request limits
and renders full f64 reference planes. The baseline's process peak RSS was
2.16 GiB under `run-heavy --mem 12G --jobs 4`; address-space allocation failed
without reaching the host memory ceiling.

The correction serializes fixtures and sweeps on 32-bit targets.
It retains the current fixture selection, including the large TOC fixture,
and all pixel-hash comparisons. The duplicate `issue865_large_toc.jxl` in
`resources/test/` is byte-identical to the dedicated `tests/testdata/jxlrs-865/`
copy; neither is removed. Before/after logs are under
`~/tmp/jxl-backlog/decoder-ci-i686-*.log`.
The rebuilt comparison passes every selected fixture on the same target, with
2.63 GiB process peak RSS and 15,562 MiB minimum host available RAM under
`run-heavy`. The failing baseline peaked lower because it could not complete;
this is not a claimed reduction in completed-decode memory.
`just i686-ci-check <label>` reproduces both CI feature configurations on a
Linux multilib host; invoke through `run-heavy` on shared hosts.

Source-export trap: `rsync -a` preserves source timestamps. When a local edit
predates completion of the remote baseline build, Cargo can reuse that baseline
binary after the copy. Refresh changed source timestamps in the exported tree
and confirm a compile occurs before treating an after-run as evidence. The
invalid first after-run is retained as `decoder-ci-i686-stale-binary.log`.


## Full CI confirmation — 2026-09-24

[CI run 36009778067](https://github.com/imazen/zenjxl-decoder/actions/runs/36009778067)
passes all 19 jobs at `420d38859ab6ee9ed9a9c213d8b301fb8afe0b38`:
both WASI configurations, i686 no/all features, Windows ARM64, Linux x64/ARM64,
macOS Intel, formatting and Clippy. This independently verifies the WASI
deadline repair (`440cc002`) and 32-bit fixture serialization (`420d3885`).
Remote `main` was verified at that commit after the run. The earlier preserved
`940d2c51` work remains in its ancestry. CI's configured corpus policy remains
in effect; a green job is not a claim that every external corpus was present.

## Fast-decode lossless exploration — 2026-10-01/02

[MEASURED] M4 Pro, 4 CLIC 2025 photos (~2.8 MP, 48 groups each), CLI
`--speedtest` u8 `--no-cms`; a smoke comparison, not zenbench. Files come
from jxl-encoder `LosslessConfig::with_fast_decode()` (`02850e60`), which
makes every channel one Gradient MA leaf with prefix codes and no LZ77.

- Entropy decoding, not prediction, bounds single-thread lossless decode.
  `main` reaches 68 MP/s on these files and 62 MP/s with ANS.
- `explore/fused-prefix-lut` (draft PR #61) has three steps:
  - `0441e81b`: a fused 12-bit table (prefix code, extra bits and signed
    unpack in one load) reaches 132 MP/s. It needs the register-resident
    `FastBits` cursor; with `&mut BitReader` in the hot loop the cursor is
    stored and reloaded per sample, and the table gives only +10%.
  - `31ad13d6`: two residuals per lookup reach 169 MP/s.
  - `e4d32db2`: two groups decoded interleaved reach 201.5 MP/s at 1 thread
    (~605 MB/s RGB8); 4 threads give 581 and 12 give 935, against 237 and
    480 on `main`. Interleaving gained nothing until the two sides were
    passed by value: behind `&mut`, both cursors spilled to memory.
- Table window on M4: 11 bits 164, 12 bits ~170, 13 bits 172, 14 bits
  161 MP/s.
- [MEASURED] Ryzen 9 7900X (Zen 4), Linux, same files, 1 thread:
  `main` 60, fused table 89, two-symbol 97, interleaved 96 MP/s
  (~288 MB/s RGB8). 4 threads: 252 to ~417; 12 threads: 383 to ~517.
  11-bit and 12-bit windows tie there. Interleaving gives nothing on
  Zen 4. Zen 5 is not measured.
- [MEASURED] On that Linux box only ~52% of the 1-thread run is decoder
  code: ~37% is kernel page faults and ~10% libc memset/memmove. The
  cause is per-group modular channel buffers. `with_buffers` allocates
  and zero-fills 144 buffers of ~248 KiB per decode (35.8 MB for a
  2.8 MP RGB image) through `alloc_zeroed_fallible`
  (try_reserve + resize, an explicit memset over fresh pages). The
  render pipeline recycles rendered group buffers into its scratch pool,
  but the modular reader never takes from it. Reusing those buffers
  across groups is the next lever for Linux throughput on every modular
  decode, not only this fast path.
- `explore/two-pass-top` (`92b9b69b`) and `explore/fused-two-pass-top`
  (`af2dfd2d`) are negative results. Two-pass gradient is slower (68 to
  52 MP/s), and fused Top (126 MP/s) is slower than fused gradient, with
  ~15% larger files.
- Conformance: preset files decode exactly in libjxl v0.12, jxl-rs 0.7.4
  and jxl-oxide 0.12.6 (gray/alpha/8/16-bit, efforts 1-9). See jxl-encoder
  `just fast-decode-check` and `just fast-decode-conformance`.
- Prefix codes cost ~3-5% size against ANS for the same settings; the
  preset is ~17% smaller than PNG on the CLIC photos.
- `sansio` (`5fc1a3e`, origin) holds unreviewed prior-session work: a new
  public `JxlIncrementalDecoder` and default zencodec/zenpixels deps. It
  is not on `main`.

Benchmark files and logs are under `~/tmp/fastll/`.

## Modular buffer recycling and parallel transforms — 2026-10-02

[MEASURED] Ryzen 9 7900X, Linux, CLI `--speedtest` u8, 3 interleaved
rounds; 11.2 MP numbers from `/usr/bin/time -v`.

- `97f08d29`: modular group buffers come from a `RecyclePool` (zero-filled,
  exact geometry) fed by the render pipeline's `take_recycled_inputs`.
  Sequential modular frames without noise or border stages output each
  two-group chunk right after decoding (`eager_modular` in
  `frame/render.rs`); parallel frames with > 16 groups per thread use
  batches of 8 groups per thread. 1 thread on 48-group frames: 60 to
  83 MP/s; sys time 0.27 to 0.04 s; 11.2 MP peak RSS 228 to 94 MiB
  (1 thread), 230 to 128 MiB (4 threads). Multi-thread speed is
  unchanged: glibc's per-thread arenas already kept those pages mapped.
- `e23e8697`: a transform layer's `do_run` calls run on rayon when their
  inputs are disjoint (`FullModularImage::run_layer`). The parallel
  path's Phase 3a-store was 1.8 of 5.5 ms per 48-group frame at 12 threads.
- Combined with `explore/fused-prefix-lut` (fast-decode files): 1 thread
  166.5, 4 threads 448.5, 12 threads 738.3 MP/s (main before both:
  60.1 / 245.0 / 376.8). 12-thread scaling is still ~4.4x the 1-thread rate;
  the remaining serial work is per-frame setup and Phase 3a output passing.
- Content of recycled buffers is always fully overwritten by decode
  (poisoning them did not change output), but the zero-fill is kept so a
  reused buffer is byte-identical to a fresh one, padding included.

## Fast lossy (VarDCT) decoding — 2026-10-02

[MEASURED] 4 CLIC 2025 photos (~2.8 MP), CLI `--speedtest`, MP/s.

- Fastest lossy encoding to decode: VarDCT with libjxl `--faster_decoding=4`
  (EPF and gaborish off, no 32x32+, simple block contexts, fixed gradient
  DC tree), about 10% larger than e7 at the same distance. Modular lossy
  (`-m 1`) is the slowest (~25 MP/s on M4). Dithering is not a cost: u16
  output (undithered) is no faster than u8.
- Decoder changes `082c50ed`, `c3d92091`, `24809159`, `04612ee4`.
  Ryzen 7900X, cjxl d1 fd4: 1 thread 122 -> 134, 12 threads 416 -> 485;
  cjxl d1 e7: 76 -> 80, 286 -> 304. M4 Pro fd4: ~101 -> ~108 / ~430 -> ~470.
  Upstream jxl-rs `5122960` on M4: within 1-2% single-threaded, slower at
  12 threads on JPEG transcodes and jxl-encoder files.
- Tried without gain (reverted): a register-cursor rewrite of the
  gradient-lookup DC loop, and hoisting the ANS/prefix match out of the AC
  loop (<1%).
- Serial floor: an image up to 2048x2048 has one LF group, decoded on one
  thread before any HF group (bitstream order: DC stream, then HF
  metadata). On M4 that is ~1.2 ms DC + ~1.3 ms HF metadata for 2.8 MP,
  about 9-10 ns per sample along a value -> context -> ANS chain; at 12
  threads it is ~40% of the frame. HF groups cannot start earlier because
  the HF metadata follows all of the DC data in the same section.
- jxl-encoder `--faster-decoding` (`32caff3d`) now follows libjxl v0.12
  (EPF by tier, kGradientFixedDC DC tree, no 64x32 from tier 2); its tier-4
  files decode within 1% of cjxl's. libjxl's tier-1 simple block context map
  and 6-histogram AC cap are not mirrored there and measured as not needed
  for decode speed.

## Lossy gap to djxl, profiled — 2026-10-02

[MEASURED] M4 Pro, 4 photos (2.8-3.1 MP), `--data-type f32` (djxl
`--disable_output` decodes to float, so u8 runs overstate the gap by the
u8 conversion), MP/s, ours / djxl v0.12:

- d1 e7: 1T 65.6 / 72.7, 12T 284 / 349. JPEG q90 transcode: 1T 124 / 136,
  12T 509 / 668. d1 fd4: 1T 110 / 105, 12T 459 / 521.
- Fixed: EPF stage 1 checked bounds on all 54 loads per vector and
  reloaded row pointers after each store (`e1463a0c`): d1 e7 1T 62.2 -> 65.9.
  Per-decode `sample` counts put EPF1 at ~1.6x djxl's before, ~1.5x after
  (sampling estimate).
- 12T is the larger gap. On the JPEG transcode at 12T the main thread spends
  ~35% of wall time in the serial LF group (DC through the weighted
  predictor, `WpOnlyLookupConfig420`, then HF metadata) and workers are idle
  ~57% of the time; the rest of the idle time is the gap between the decode
  and render parallel phases. Single-threaded sample fractions put our WP
  DC decode at roughly 1.2x djxl's time (sampling estimate, not measured
  per function).
- Register cursor (`PlainCursor`) for modular channels helps single-leaf
  trees only: fast-decode preset 66 -> 89 MP/s 1T, 487 -> 603 12T. On
  learned trees and libjxl's gradient DC tree it was neutral 1T and 3-10%
  slower at 12T in back-to-back runs, so it is off there
  (`ModularChannelDecoder::USE_CURSOR`).
- 12T numbers on this Mac move 5-10% with background load (Spotlight,
  mediaanalysisd, rust-analyzer); compare A-B-A, not A-B.


## Multi-threaded parity work — 2026-10-03

[MEASURED] M4 Pro (8 P + 4 E cores), f32, interleaved runs; the machine
was loaded (Spotlight indexing, load 12-24), so treat MT figures as +-5%.

- False sharing (`e404d64b`): two 4-thread processes decoded e7 lossless at
  a combined ~60 MP/s while one 8-thread process reached 50. The cause was
  the per-channel MA-tree property buffer (a small `Vec<i32>` written every
  sample) sharing a cache line with another thread's data. `IsolatedBuf`
  (`util/isolated.rs`) pads it by 128 bytes per side. The two-process vs
  one-process comparison is the quick test for this class of problem.
- Per-group timing (`JXL_GDBG`-style instrumentation, not committed) shows
  E-core threads run groups 2-3.7x slower and macOS starts some threads on
  E-cores; P-core groups also slow ~1.3x at 12 threads (all-core clocks).
  From 8 to 12 threads we gained nothing on fd4 while djxl gained 9%.
- Serial LF group (one LF group up to 2048x2048): d1e7 DC 1.87 ms + HF
  metadata 0.85 ms; JPEG q90 DC 1.03 + 0.2; fd4 DC 1.13 + 0.85. DC per
  sample matches our e7 lossless speed, itself at djxl parity. Reading
  `GradientLookup` (fd4 DC) through the register cursor was slower (LF
  2.02 -> 2.30 ms); keep `USE_CURSOR` off there.
- The batched VarDCT path is expensive: forcing 2 batches of 24 groups
  raised P3b from 2.81 to 5.12 ms (d1e7) and 0.95 to 3.03 ms (JPEG), so
  overlapping render with the next batch's decode is not worth it on top of it.
- In-task render (`8c4e8175`) covers only pipelines with `border_size == 0`
  (fd4, JPEG 4:4:4). JPEG 4:2:0, d1e7 and d3 still use P2 -> P3b.
- Rejected: packing cluster + hybrid-uint config into one per-context
  table for the AC loop (0.3% fewer instructions, 1.5-3% more cycles).
- Parity after these changes, 12 threads, zen/djxl: d1e7 0.94, fd4 ~0.99,
  JPEG 4:2:0 0.91, JPEG 4:4:4 ~0.93-1.0, d3 0.89. 1 thread: ahead or tied
  everywhere except d3 (78 vs 81) and e7 lossless (10.7 vs 11.2).
