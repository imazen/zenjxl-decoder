# Caller-owned output buffers on i270

The paired still-image harness compares a fresh aligned buffer, an A/A
repeat, the same caller-owned buffer with a new decoder, and the reused
buffer with the public decoder rewind. All arms use the same stream and
pixel format, with CMS off. The decoder core is `88ce5198` with Rust
1.99.1, locked release dependencies and `all-simd,threads`. Timings are warm,
with nine interleaved blocks of five individually measured decodes at
one, four and eight workers on pinned P cores.

| Stream | Output reuse at 1 / 4 / 8 workers | Output reuse + rewind at 1 / 4 / 8 workers |
|---|---:|---:|
| 1007 native Modular fast | 1.059× / 1.043× / 1.053× | 1.058× / 1.064× / 1.072× |
| 1007 native VarDCT d4 | 1.050× / 1.057× / 1.097× | 1.053× / 1.058× / 1.085× |
| 1019 native VarDCT d4.08 | 1.082× / 1.089× / 1.159× | 1.080× / 1.093× / 1.159× |
| 8457 native Modular fast | 1.013× / 1.029× / 0.994× | 1.013× / 1.035× / 1.040× |

The three native photos benefit from reusing output. Rewind has no
consistent additional gain on these streams. The current rewind rebuilds
the codestream parser and preserves pixel-format configuration. It does
not carry the per-frame VarDCT scratch pool across files. That leaves
internal scratch recycling as a separate experiment.

Both 64 px and 256 px Modular controls have no material throughput
change from output reuse. The small RGBA fixture has an 8-worker A/A
p95 ratio of 1.152×. Its speed and tail comparisons are sensitive to
variation at that size. The seven streams pass exact row-wise RGB8 or
RGBA8 comparisons between all four arms before timing, after the first
timed decode into zeroed reused buffers, and after timing, in each
worker setting. The run includes 3,780 measured decodes. Strict
Clippy, the build and the timing run used 0.91 GiB peak RSS over 166 seconds.

Build with `just build-paired-output` and run
`taskset -c 0-7 target/release/paired-output --workers 8 --blocks 9 --allow-level10 FILE...`.
Use CPU 2 at one worker and 2,3,0,1 at four. The runner times the decoder
from header parsing through the single-frame result. It includes the
creation and prefault of fresh aligned output where applicable and rewind
where requested. Retained prior output is released inside each fresh
allocation arm. The measured path uses the still-image decoder API.

[Per-stream medians and p95](paired_output_reuse_2026-11-22.tsv).
Raw samples, inputs, binary and stream SHA-256s and the command are
under `/mnt/v/output/zenjxl-decoder/paired-output-2026-11-22` on i270.
