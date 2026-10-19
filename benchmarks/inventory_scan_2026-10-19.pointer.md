# Inventory scan and ICC checks, 2026-10-19

Artifacts were checked on dev under
`/mnt/v/output/imazen-26/decoder-inventory-2026-10-19/`.
The scanner and ICC readback examples and the `scan-heap-profile` / `icc-oracle`
just recipes reproduce these checks with explicit inputs and output paths.

The 1,517-byte malformed HF input has SHA-1
`ba9c55c5dc09ccd128de72910ee999feed23f08d` and is retained in
`/mnt/v/fuzzes/zenjxl/corpus/inventory/`. Header-only scanning rejects its
incomplete frame header. Heaptrack reports a peak heap of 122.39K bytes;
this is an absolute measurement of the candidate, without an old-version
comparison. `scan-hf-memory.zst` and `scan-hf-memory-summary.txt` retain the trace
and analysis. run-heavy: rc=0, 1s, peak RSS 0.09GiB, min available 49,275MiB.

Seven independently valid compressed-ICC cases pass native one-shot decoding
and header parsing with 1, 7, 64 and 4,096-byte input chunks. Native RGB samples
and complete ICC bytes match libjxl 0.12 for all seven; the largest profile has
2,121,892 bytes. Explicit 128-byte ICC limits reject the same files. Native and
oracle results are under `icc-native-v3/` and `icc-libjxl/`.
The oracle run took 1s; run-heavy peak RSS 0.03GiB, min available 49,279MiB.

The owner-approved `resources/test/icc_zero_2m.jxl` fixture is 10,454 bytes,
SHA-256 `37c66f4835f4759d70f1529b094ce19a06b66b767bbca44d1d6b642df2ac1d02`.
It contains a synthetic sRGB profile and a 1×1 RGB image. libjxl independently
reads RGB (200, 30, 90). Its native regression also checks fragmented headers
and the caller's ICC cap. Existing amplification and size guards remain enabled.

The GNU-target `layout_scan` fuzz run completed 2,440,840 executions in
601 seconds without a finding. It compares plain and recorded header scans,
including a poll after the final frame. Command: `just layout-fuzz 600`, under
the shared build lock and `run-heavy --jobs 8`; the wrapper includes compilation
and reports rc0, 662s, peak RSS 2.03GiB, minimum available 46,898MiB, peak load 2.83.
The complete log is `task5-layout-fuzz2.log` in the private measurement archive.
Accumulated corpus inputs remain in `/mnt/v/fuzzes/zenjxl-decoder/corpus/layout_scan/`;
`fuzz-sync.sh push zenjxl-decoder` completed after the run.
