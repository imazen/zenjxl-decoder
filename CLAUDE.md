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
