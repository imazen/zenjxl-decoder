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
`just wasm-ci-check <label>` runs both CI feature configurations and saves
complete logs under `~/tmp/zenjxl-decoder/` with CI's explicit missing-corpus
policy.
