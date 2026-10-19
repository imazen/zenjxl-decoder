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
