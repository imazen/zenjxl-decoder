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
