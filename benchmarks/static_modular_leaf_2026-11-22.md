# Static Modular leaf setup in VarDCT LF groups

Candidate `71b67180914be24a2b16d355e560952723c9b6d5` resolves the constant channel and stream splits
before allocating the tree-pruning queue. If the result is a unit,
zero-offset Zero or Gradient leaf, it uses the existing single-leaf
predictor and entropy path.

On i270, five native VarDCT files and three Modular controls decoded
identically to `88ce5198e3649b490875eda30b6f7eda6d5e1271` at one, four
and eight workers. The exact check compares 16-bit PNG hashes. The
timed endpoint decodes into freshly allocated RGB8 and disables CMS.

Median paired LF-group latency ratios at eight workers range from
0.993 to 1.008, with A/A medians from 0.990 to 1.020. Same-stream
VarDCT throughput medians range from 0.989 to 1.001 at eight workers.
The optimization misses both the 15% phase reduction and the 5%
end-to-end throughput target. The largest apparent one-worker gain
is 1.050× on the smooth tier-1 photo. The three other timed photos
and the tier-4 stream do not repeat that gain.

[End-to-end timing](static_modular_leaf_2026-11-22.tsv) uses three
paired blocks of five decodes at each worker count. The independent
[phase capture](static_modular_leaf_phase_2026-11-22.tsv) uses
three paired blocks of eight decodes at eight workers with the
existing phase diagnostic enabled. Both include A/A. The phase output
excludes the three warmup decodes per process. The release build uses
Rust 1.99.1 and `--no-default-features --features all-simd,threads`.
The build and run took 155 s with 0.91 GiB peak RSS. Commands, stream
hashes, raw timings and readbacks are retained on i270 under
`/mnt/v/output/zenjxl-decoder/static-leaf-lf-2026-11-22`.
