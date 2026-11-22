# jxl-rs main on faster-decode photo streams

On i270, Zen decodes the four selected native imazen-26 lossless photo
streams 18–23% faster than jxl-rs main at one worker. At four workers the
decoders are within 4% on each photo. At eight workers upstream wins
three of four. Upstream also wins the single VarDCT control at every
worker count, with a different RGB8 readback.

| Photos | 1 worker | 4 workers | 8 workers |
|---|---:|---:|---:|
| Lossless median throughput, Zen / upstream | 1.195× | 0.992× | 0.903× |
| VarDCT control throughput, Zen / upstream | 0.920× | 0.727× | 0.715× |

## Changes worth testing upstream

1. Fuse the Huffman lookup and short HybridUint payload into a residual or
   repeat count inside upstream's existing
   [fixed-Gradient distance-1 RLE loop](https://github.com/libjxl/jxl-rs/blob/f370be423933628eb7f783f1b61d4c765eca5f79/jxl/src/frame/modular/decode/bitstream.rs#L46).
   Share the lookup across groups with the same tree and keep the ordinary
   path for wider tokens and short input. Preserve upstream's i16 and i32
   storage paths and partial-input behavior. Then time on-demand refill
   as its own change. Zen's
   [refill ablation](cold_modular_photo_2026-11-22.md#on-demand-refill-for-the-fused-rle-window)
   measures 1.058× same-file cold throughput at one worker over the already
   fused version.
2. Measure the row-buffer handoff for identity Modular RGB8. Upstream already
   [fuses i16 conversion and interleave](https://github.com/libjxl/jxl-rs/blob/f370be423933628eb7f783f1b61d4c765eca5f79/jxl/src/render/low_memory_pipeline/save/identity.rs#L485)
   from render rows. The question is whether it can skip row staging when
   the frame only needs an identity save, or gains from an i32 equivalent.
   [Zen's direct group-plane save](cold_modular_photo_2026-11-22.md#write-identity-modular-rgb8-directly)
   separately measured 1.121× / 1.080× / 1.056× cold on these streams in
   Zen's pipeline. The upstream benefit needs its own ablation.
3. Keep upstream's group scheduling and VarDCT pipeline as controls.
   Prioritize one-worker Modular while checking the faster upstream
   eight-worker cases for regressions. Compare VarDCT only after accounting
   for the rendered-sample difference on the existing control.

## Reproduction

Upstream source
[`f370be42`](https://github.com/libjxl/jxl-rs/commit/f370be423933628eb7f783f1b61d4c765eca5f79)
was built with `cargo build --locked --release -p jxl_cli --no-default-features
--features all-simd`. Zen source is
[`05042e1b`](https://github.com/imazen/zenjxl-decoder/commit/05042e1b5b46151a507da3f116eadee805117599).
Neither build used `target-cpu=native`. Endpoints were fresh processes,
memory-resident input, allocating RGB8, and pinned one, four and eight
i270 P-cores. Zen passed `--no-cms`; upstream's CLI creates a CMS transform
for linear output or a CMYK conversion, neither of which applies to the
RGB8 endpoints. The timing run used three rotated blocks of five decodes
per arm with an upstream A/A control. A/A throughput ratios range from
0.989× to 1.013×. Timing used the first-decode times reported by the two CLIs.

Full PPM rasters match byte-for-byte in all 12 lossless pairs. The VarDCT
control has identical dimensions and worker-invariant pixels within each
decoder. Its cross-decoder raster hashes differ. All reported rates cover
non-reused output only. The upstream build took 13 s at 0.80 GiB peak RSS.
The comparison took 38 s at 0.20 GiB peak RSS.

[Per-stream throughput and p95](jxl_rs_main_photo_comparison_2026-11-22.tsv).
The source archive, binaries, stream and readback hashes, exact commands
and raw timings are under
`/mnt/v/output/zenjxl-decoder/jxl-rs-main-profile-i270-2026-11-22` on i270.
