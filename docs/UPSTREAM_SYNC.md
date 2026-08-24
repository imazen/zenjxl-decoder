# Upstream sync ledger — libjxl/jxl-rs → zenjxl-decoder

This is the living record of where this fork stands relative to
[libjxl/jxl-rs](https://github.com/libjxl/jxl-rs): the fork point, which
upstream PRs have been ported, what upstream has changed since, and which of
those changes are worth taking. Update it whenever a port lands.

Per-commit evidence (file:line in both trees, port effort, verification
results) for the 2026-08-22 audit lives in
[`upstream-audit-2026-08-22.md`](upstream-audit-2026-08-22.md). The tools
used are in [`scripts/upstream-audit/`](../scripts/upstream-audit/README.md).

## Status (audited 2026-08-22)

| | |
|---|---|
| Fork point (merge-base) | upstream `da89c6c` — "Make feature `all-simd` enabled by default", 2026-03-07 (jxl-rs 0.3.0 line, pre-0.4.0) |
| Last port sweep | **2026-08-23** (fork commits `784a545`..`15ccc9e`, see "Port log 2026-08-22/23" below). Before that: 2026-06-01 (`7c7ee08`..`f8b3e85`, upstream through `841842a` / #784). |
| Upstream HEAD audited | `088ec7f` — "Remove support and use of internal image padding. (#888)", 2026-08-21; release **v0.6.0** (2026-08-18) |
| Upstream commits since fork point | **161** (verified with `git merge-base` on a scratch clone) |
| Already ported (cherry-ported, re-implemented) | 24 of them, plus 5 perf items from the never-merged upstream draft PR #705 |
| Fork output vs upstream HEAD | **bit-identical** on all 116 fixtures that both decode (58 small + 22 large + 36 conformance, 16-bit output, 1 thread) — see "Verification" |

Upstream versions vs fork releases:

| fork | date | upstream baseline |
|---|---|---|
| 0.3.0 | 2026-03-06 | fork point `da89c6c` |
| 0.3.3 – 0.3.5 | 2026-03-30 → 04-01 | ports from 0.4.1: #725 #731 #735 #740 #741 #742 + draft-#705 perf items (BitReader 8-byte section padding, property used-mask, HybridUint fast path, inline audit, blending SmallVec). Note: upstream never merged #705; the section padding and HybridUint fast path have **no upstream counterpart**. |
| 0.3.6 – 0.3.8 | 2026-04-10 → 04-17 | 0.4.2: #745, subset of #743 |
| 0.3.9 – 0.3.10 | 2026-06-09 → 06-11 | 0.5.0-era fixes: #738 #749 #756 #757 #766 #774 #775 #776 #784, c60408d (spline SIMD consistency), 83db36f/#751 (PQ/HLG F16 clamp). #758 (drop uninit buffers) benchmarked, **decision still open** (`benchmarks/uninit_buffer_measurement_2026-06-01.md`). |
| main (unreleased) | — | nothing from 0.5.1 / 0.6.0 |

## Port log 2026-08-22/23

What landed from this audit, in push order (all on `main`, CI green unless
noted). Decisions taken with the user: dithering on by default, out-of-order
`jxlp` supported, uninit output buffers kept (#758 not adopted), no `unsafe`,
perf pursued until within 8 % of upstream.

| batch | fork commits | contents |
|---|---|---|
| 1 | `784a545` | CI green: `kernel_tiers.rs` aarch64 gate, #887 `chunks_exact_to_as_chunks` allow; f16 subnormal encode fix (#706 hunk) with IEEE-sweep tests; #858 `Mul` blend without extra channels; #833 modular-XYB overflow; #791 weighted-predictor delta palette (bit-patched `delta_palette.jxl` fixture, hash cross-checked with djxl 0.12 and jxl-rs 0.6); #823–#825 / cflite POCs as `fuzz/regression/` seeds (the #822 POC is 33 KB — over the 8 KB regression-seed ceiling — and the fork rejects it on a path the smaller POCs cover; the 94-byte chromium 541318910 POC joined in `d09682f`'s follow-up) |
| 2 | `1cbb414` | #856 section buffers grown from available input (was a 1 GB allocation on a truncated huge-TOC file), #845 + #873 + 43e2db6 render-edge padding keyed on the rectangle, TOC `check_for_error`, #828 empty/size-0 box acceptance, `restrictive()` ICC cap 16 MiB, `Lz77Params` |
| 3 | `29c6dd4` | parallel-CMS fix (pool-with-return transformers, fail loud on exhaustion; fe3b3c9 idea) + thread-count parity test; #752/#777 out-of-order `jxlp`; blue-noise u8 dithering (#841) on by default, `with_dither_u8(false)`; proc-macro-error2 dropped (#799); benchmark-comment workflow removed |
| 4 | `cddf926` | perf: single-symbol entropy path (#787 + #817), config-420 reader specialisation, `prefault_parallel` gate, `--no-cms` CLI flag, blending bundle (#709 #818 #821: vectorised, allocation-free, in-place) |
| 5 | `728fc6e`, `d588318` | perf: modular palette applied one channel per thread (fork-only; closes the 1.25× MT gap on `delta_palette`); parallel render renders every group once (tiled full-readiness rectangles), pool lock scope, render-context pool |
| 6 | `9b1998f`, `0038580`, `15ccc9e` | perf: f94cc26 `BufferFiller`; **fix** two pre-existing streamed-decode panics (fragment-path band split with overlapping columns; STEP 5 flush of an undecoded neighbour group), parallel/sequential path chosen per frame; fixtures #728 #734 #772, #875 LF-group colour test, e12b99b flush test, `decode_parallel` fuzz target, large-chunk incremental sweep, corpus-wide thread-count / chunked-parallel parity tests |

Perf at the end of the sweep (fork `ceacc3b` vs upstream `088ec7f`, CLI
`--speedtest`, M4 Pro 8P+4E, idle machine, `just speed-compare`; full table
in `benchmarks/upstream_speed_compare_2026-08-23.tsv`). 12 threads, 17
fixtures: fork ahead on 10 (`patches` 1.84×, `bike` 1.41×,
`green_queen_vardct_e3` 1.49×, `delta_palette` 1.26×, `cafe_web_q80` 1.12×),
at parity on 5, behind by 10 % on `cafe` (conformance) and `blendmodes` (a
15 MP/s animation where threads barely matter). 1 thread: within 8 % on 14,
behind by 9–12 % on `multiple_lf_420` (1.10×), `cafe` (1.09×) and the
4:2:2 JPEG transcode `test_600x600_422_libjxl` (1.12×). Where the 1-thread
gap sits (measured by elimination on `multiple_lf_420`, see the batch-5/6
commit messages): the VarDCT coefficient entropy loop (~1.1 ms of 30 ms per
decode; identical source, the fork's `decode_vardct_group` compiles to a
larger function with more spills), and the per-row render glue (save stage,
`run_stage_on` setup; the stage kernels themselves compile to the same inner
loops). Tried and measured as no gain on aarch64: `mimalloc` in the CLI,
an out-of-line coefficient decoder, bulk instead of per-block coefficient
zeroing, force-inlined context helpers, inlining the NEON kernel into the
`simd_function!` dispatcher.

Follow-up sweep 2026-08-23 (`15f332f`, `325cf98` + this docs commit): the
#812 scratch cap is ported; `issue865_large_toc.jxl` lives in `tests/testdata/jxlrs-865/`
with a 64-bit-gated streamed-equals-one-shot test (it stays out of
`resources/test/` so the fixture sweeps do not decode 28 MP on 32-bit CI).
Measured and NOT taken, on top of the earlier list: the #722
`#[inline(always)]` sites (applied 1:1, within measurement noise on the
M4 Pro, reverted); LPT-sorting Phase 2's groups by coded size (no change:
the phase's wall clock is bounded by the densest single group, not by
imbalance); and a full fused decode+render pass for one-shot VarDCT
(per-group `OnceLock` plane publication + atomic 3x3-neighbourhood
counters triggering each group's render on whichever thread completes it,
raster-order work queue, pre-split disjoint output fragments — implemented,
pixel-exact, and **no faster**: the wall stays bounded by the densest
group's decode, the per-group synchronisation costs about what the barrier
did, and a 9-group image paid 1.36x for the setup; reverted, negative
result recorded here so it is not re-attempted without new evidence).

Still open from the audit: #716 flat-tree enum (measured slower on
aarch64, reverted; re-measure on x86 — no x86 box in this environment),
#793 weighted-predictor layout (skipped: the fork's WP decode paths
measure *faster* than upstream's where WP dominates, e.g. cafe's
`WpOnlyLookupConfig420` 0.6 vs 0.73 ms/decode, and upstream's version
needs `get_unchecked`), #797 reader-generic trees (the remaining candidate
for the ~9 % single-thread entropy-decode gap), #888 padding removal
(deferred behind #797: the fork's specialised tree decoders rely on the
padded rows for branch-free `row_top[x + 2]` loads, so removing padding
without the #797 restructure would put edge branches in the hottest loop
for ~3 % of modular-buffer memory), the depth-first modular transform
engine, the additive API items marked TAKE (`new_with_stride`, `rgb*`
helpers — awaiting public-API approval), and the breaking-change batch
(`flush_pixels -> bool`, `progressive_mode`/`unconsume`, `rgba*`
extra-channel semantics — now queued in the CHANGELOG). CI also runs
`threads` without `allow-unsafe` (`8b58423`) and wasm32-wasip1 under
wasmtime with and without simd128 (`6051605`,
`.cargo/config.toml` + `.cargo/wasm-runner.sh`).

## What upstream did since the fork point (by theme)

- **0.5.0 (2026-07-28):** out-of-order `jxlp` containers (#752), big modular
  decode speedups (#787 single-symbol path, #716 flat-tree, #793 weighted
  predictor, #797 more fast paths, #812 transform-application rewrite),
  blending made allocation-free / in-place / SIMD (#709 #818 #821), box +
  codestream parser rewrite (#828) with a batch of fuzzer fixes (#829–#834),
  CmsStage removed from the decoder (#754 — opposite of our design),
  smooth progressive previews for modular images, wasm simd128 (#706).
- **0.5.1:** progressive-preview fix (#837).
- **0.6.0 (2026-08-18):** upstream's own multithreading (#849–#854, #862
  shuttle testing, #864), **blue-noise dithering of u8 output, always on**
  (#841), section buffers grown from available input (#856), `Mul` blend
  clamp-bit parse fix (#858), subsampled-frame rendering fix (#875 — upstream
  bug introduced by their MT LF rewrite; our layout is unaffected), modular
  border buffers + eager deallocation (2d0b720..4495876), internal image
  padding removed (#888).

## Verification performed 2026-08-22

All with the fork at `affc97f` and upstream at `088ec7f`, release builds, on
an Apple M-series (aarch64/NEON) machine. Commands are reproducible with
`scripts/upstream-audit/`.

1. **Pixel equality, 1 thread, 16-bit output** (bypasses upstream's new u8
   dithering): every one of the 58 in-tree fixtures, the 22 large
   `*_4k_* / clic_* / *_web_*` fixtures, and the 36 conformance images that
   both decoders accept are **bit-identical**. (`cropped_traffic_light` and the
   5 animation fixtures produce APNG, not compared.) `cmyk_layers` differs by
   design: the fork converts CMYK→RGB in-decoder via CMS, upstream no longer
   has CMS.
2. **Upstream fixtures we lack:** `issue728_minimal`, `strategic_solid_blue`
   (#731/#735 regression cases — ported fixes, fixtures never copied),
   `issue865_large_toc`, `issue772_blendbug`: all decode, bit-identical to
   upstream; `invalid_animated_ooo_jxlp` is rejected by both.
3. **Upstream's July fuzzer crashes** (#822 #823 #824 #825, POCs downloaded):
   all five are rejected cleanly by the fork with the same error upstream now
   returns. They were introduced by #812, which we do not have.
4. **Regressions found in the fork** (details + repro in the audit file):
   - `Mul` blend mode without extra channels → **"Source file truncated"** on
     a valid 32-byte codestream (conformance PR libjxl/conformance#48
     `mul_no_extra_channels`); upstream and djxl decode it. Fix = upstream #858.
   - **Parallel CMS decode renders most tiles without the colour transform**
     (`threads` + `cms` + an ICC-only profile): 4096×2519 ACEScg-tagged photo,
     `--num-threads 2` → 157/160 groups wrong, `--num-threads 8` → 160/160
     wrong, max error 18818/65535; `--num-threads 1` matches upstream within
     ±1 LSB. Mechanism: `CmsStage`'s transformer pool (`current_num_threads()+2`)
     is popped once per rayon leaf and never returned; leaves without a
     transformer skip the stage silently. Not reachable through `decode()`
     (no CMS by default) or zenjxl (sets no CMS); reachable in the CLI and any
     `with_cms` caller. Fork-only bug.
   - `f32 → f16` conversion halves every f16-subnormal value and flushes the
     smallest ones (`util/float16.rs` and `zenjxl-decoder-simd/src/float16.rs`,
     both copies). Affects `JxlDataFormat::F16` output below ~6.1e-5. Fix =
     upstream #706's `float16.rs` hunk.
   - ClusterFuzzLite batch run 32539550999 (2026-08-22) hit
     `render/stages/convert.rs:114` "attempt to add with overflow" — this is
     upstream #833.
   - CI on `origin/main` has been **red since 2026-08-01**: `benches/kernel_tiers.rs`
     references `NeonDescriptor` unconditionally (x86 clippy job fails) and is
     not `cargo fmt`-clean; stable 1.98 additionally raises
     `clippy::chunks_exact_to_as_chunks` at 53 sites (upstream allowed it in #887).
5. **Rough single-thread speed** (`speed_compare.sh`, CLI `--speedtest`,
   interleaved, NOT a zenbench measurement): VarDCT within 0–8 % of upstream
   (upstream faster); modular: `squeeze_edge` upstream 1.78×, `sunset_logo`
   1.63× (#787's single-symbol path), `issue648_palette0` 1.12×,
   `delta_palette` 1.10×, `green_queen_modular_e3` 1.06×, `lz77_flower` 1.03×.
   With default threads (12) upstream is 2.0× on `squeeze_edge` and 1.5× on
   `sunset_logo` because it parallelises modular transforms; VarDCT is within
   3–8 %. Raw numbers in the audit file.

## Disposition of the 161 upstream commits

Legend: **PORTED** already in the fork · **TAKE** port as-is / adapted ·
**LATER** worth porting, lower priority or needs prerequisites · **N/A** not
applicable (feature absent, design differs, or upstream-only regression fix) ·
**INFRA** CI/build/docs only.

### Correctness, hardening, memory

| upstream | what | disposition |
|---|---|---|
| aed4e9e #858 | parse `Mul` blend `clamp` bit when there are no extra channels | **TAKE now** — fork fails valid files (`headers/frame_header.rs:126-129`) |
| 814612d #791 | DeltaPalette + Weighted predictor: call WP for every pixel, save/restore full WP state across group rows | **TAKE now** — pre-fix code is byte-for-byte in `transforms/palette.rs:242-255,598-620`, `predict.rs:456-464`; silent wrong pixels; needs a fixture (libjxl's encoder never emits Weighted delta palettes, jxl-art/other encoders do) |
| 204c0d0 #833 | `(b as f32 + y as f32) * scale` in modular-XYB conversion | **TAKE now** — confirmed by our own fuzzer (`render/stages/convert.rs:114`) |
| b78c54a #706 (float16 hunk only) | correct f16 subnormal conversion with RNE | **TAKE now** — both `float16.rs` copies; bring upstream's tests |
| 7809f88 #856 | grow section buffers from available input, `try_reserve` | **TAKE now** — `codestream_parser/mod.rs:243` does an infallible `resize` to the TOC-declared size (≤ ~1.07 GB per entry, untracked by the memory budget; abort on i686); keep the fork's 8-byte `SECTION_PADDING` |
| 5c6d8d6 #845 + a608c0b #873 | clip ready rects to the image, edge padding keyed on `start_of_row/end_of_row`, guard `x1<x0` | **TAKE now** — `group_scheduler.rs:65,70,123,129,382,390,555,561`, `render_group.rs:338,353`; wrong right-edge pixels for permuted-TOC streaming + latent usize underflow |
| 6430786 #813 | ICC size cap 1 MiB → 16 MiB | **TAKE now** for `JxlDecoderLimits::restrictive()` (`options.rs:108`); real CMYK press profiles are 1.8–3.5 MiB. Default is already 256 MiB. Also flag zenjxl's 1 MiB probe cap (other repo) |
| d5c1f17 #828 (items only) | accept empty boxes / size-0-to-EOF for any box type; `br.check_for_error()` after TOC `read_step`/`read_permutation` | **TAKE now** (two trivial pieces of a large rewrite we otherwise skip) — `box_parser.rs:331-338`, `headers/toc.rs:89,102` |
| 43e2db6 | x-padding keyed on rect position, not group index | **TAKE now** (trivial, provably equivalent today, protects a future sector-ownership change) |
| 57e515e #868 | floor semantics for vertically subsampled border rows | **LATER** (mismatch present, no reachable repro found; bundle with #845) |
| 35fb0ad #829, e749ffd #780, 67cb896 #831, eb60b47 #846, e025e92 #830, 24db91e #832, 452f35e #749, 28ddaeb #745, f1514f1 #743 (non-seek part), 33864e8 #757, 2a6f9ec #774, f20f7d1 #775, c184321 #776, 841842a #784, 81dc81e #766, 0664c90 #756, 159c60b #731, 3d1d0c2 #735, 8de0b29 #740, 1e909aa #741, 226f47e #742, a47d786 #725, 371f033 #738, 83db36f #751, c60408d, a737779 #699 | previously ported or independently fixed | **PORTED** — each re-verified at file:line in the audit file |
| ebeed75 #773 | clipped blending with missing references (copy_from_slice length) | **N/A** — fork's blending is immune; fixture + block-mean test **PORTED** (`0038580`) |
| 365eb80 #875 + 6401d6e | subsampled-frame LF rect | **N/A — do not port**: fixes a bug upstream introduced in 85f23a1; our packed LF layout is self-consistent (bit-identical on `multiple_lf_420`). Pixel test **PORTED** (`0038580`) |
| 57e29ed #861 | allow adaptive LF smoothing on subsampled frames | **N/A** (libjxl still rejects such files; spec clarification pending; would also need a JPEG-reconstruction guard) |
| 3c4f224 #777, 4413527 #866, 4aa221f #834, 2ccf6af #867, 4fde1db #869, 5a689fc, c4d102d #876, b9bb475 #870, 600f977 #863 | fixes to upstream-only code (OOO-jxlp state, #828 parser, MT locks) | **N/A** (verified the fork has no analogous state); the `squeeze_empty_residual` chunk-1..16 flush test from e12b99b is worth adding |

### Features

| upstream | what | disposition |
|---|---|---|
| f694be5 #841 | blue-noise dithering of u8 output (libjxl behaviour, always on upstream) | **DECISION NEEDED** — changes every lossy u8 pixel by ≤1 (lossless 8-bit unchanged); brings `decode()` in line with djxl (fork vs djxl: 25 % of pixels ±1 today, upstream vs djxl: 7 %). Three fork stages need the term (`convert.rs`, `xyb.rs` fused kernels, `from_linear.rs`); recommend a `dither_u8` option — default is the user's call |
| e7405e0 #752 (+3c4f224 #777) | out-of-order `jxlp` boxes (ftyp minor version 1; `cjxl --output_mode 2`) | **TAKE** — fork returns `InvalidBox` on valid files (`box_parser.rs:363`); keep #752's permissive semantics, not #828's seek-oriented restriction |
| e883140 | safe `JxlOutputBuffer::new_with_stride` | **TAKE** (trivial, additive; matches our stride rule) |
| c5528f6 #767 | `JxlPixelFormat::rgb*` helpers; `rgba*` stop requesting planar extra channels | **TAKE** rgb* now; rgba* semantic change → queued breaking |
| d782c19 #755, 0977812 #880, 1cc9ab7 #820 | `flush_pixels -> bool`, drop no-op `progressive_mode`, drop dead `unconsume` / add `file_length` | **LATER** — all breaking; add to CHANGELOG "QUEUED BREAKING CHANGES" |
| 035477c #678, 2cddd90 #702 | animation scan/seek API | **LATER**, only if a consumer needs seeking; port the post-#828 design, not these commits |
| 2556ead #732 | 16-bit PPM/PGM in the CLI | **LATER** (cherry-pick) |
| 8b8dd57, 0d75b8f, d8359cf, 6e82649 #779, 6fa7f5c #785, e1fc42e #781, 6cb5810 #788, 245728e #800, c41cbfa #760, 4e43b32 #763, 1f92de5 #764, 3bddd4f #816, f39af49 #835, 15a1e01 #837, 5ad6a8c #883, ad7cbd4 #885, 85aa6ee/d05edc9 #871 | progressive-preview rendering (partial LF-global decode, smooth unsqueeze, Jinc2) | **N/A** — the fork kept the pre-March flush design; only relevant if a progressive-preview UI becomes a requirement |

### Performance / memory

| upstream | what | disposition |
|---|---|---|
| 654a985 #787 + 8e8769b #817 | single-symbol fast path (must ship together) | **PORTED** (`cddf926`, batch 4) — sunset_logo 1T now 1.07× behind, 12T at parity |
| ad5ead5 #716 | enum flat tree + `Box<[i32;256]>` property buffer | **N/A (measured)** — ported and measured slower on aarch64 (2026-08-22), reverted; re-measure on x86 before retrying |
| 4fcfb24 #793 | weighted predictor layout/unrolling | **N/A (measured)** — the fork's WP paths already measure faster than upstream's (2026-08-23), and the upstream version relies on `get_unchecked` |
| afd41c9 #797 | reader-generic flat trees, fast-lossless path | **LATER** (medium–large); `Tree::num_properties` piece is trivial |
| 7f8ee4f #812 (scratch cap only) | `group_scratch_buffers_limit = Some(0)` for modular frames | **PORTED** (`15f332f`) — 300 cached tiles / 78 MB dropped on a 2333x2333 lossless decode, speed unchanged |
| 7f8ee4f #812 (rest), 2d0b720, c066ee2, 07cb870, abf9c4f, 4495876, f76be0c | depth-first transform engine, border buffers, eager dealloc, parallel modular transforms | **LATER** as one project; this is where upstream's 2× MT modular win comes from |
| 088ec7f #888 | remove internal image padding | **LATER, behind #797** — the fork's specialised tree decoders use the padded rows for branch-free `row_top[x + 2]` loads; port only together with the #797 loop restructure |
| f1388c4 #709, 012a292 #818, 7b223de #821, a76e651 #819 | allocation-free → in-place → SIMD blending + bench | **PORTED** (`cddf926`, batch 4; `blendmodes` 1T 1.07×, 12T 1.04×). Bench not ported (criterion); a zenbench bench is still to do |
| fe3b3c9 | pool-with-return `PerThreadStorage` | **PORTED** (`29c6dd4`, batch 3: CMS transformer pool; `d588318`, batch 5: render-context pool) |
| 9d64f77 #722 | 20 `#[inline(always)]` sites (`shrc`, `mirror`, cmap, …) | **N/A (measured)** — applied 1:1 on 2026-08-23, within noise on aarch64 (codegen-units=1 + thin LTO already inline these), reverted |
| 0196204 #720 | `assert!(permutation.len() >= num_coeffs)` before the coefficient loop | **N/A (measured)** — tried inside an out-of-line coefficient decoder on 2026-08-23, no gain on aarch64 |
| 36c9c3b #801, da96606 #710, 6c44dc3 #805 | XYB param precompute (≤0.5 %), SmallVec inlining, LF-image copy removal | **LATER** (the fork uses the `smallvec` crate; `f94cc26` BufferFiller, the bigger per-row item, is **PORTED** in `9b1998f`) |
| bf17fa7 #717, de77265 #721 | property-buffer assert, small inlines | **PORTED** (fork additionally keeps a used-mask gate upstream dropped — A/B it) |
| 625af01, c17fcf3 #872, 032d101 | small-image parallelism limiter | **LATER** — measure tiny/small MT vs ST first |
| 462454f #723, 065f477 #855, 8d16561 #826 | release profile, mimalloc in CLI, reciprocal in dead `ToLinearStage` | **N/A** / skip |

### Multithreading (upstream #849–#854, #860, #862, #864, 8 follow-ups)

**N/A as code.** Upstream built a lock/atomic dataflow scheduler with a
caller-supplied `JxlParallelRunner` (and re-introduced `unsafe` in its buffer
splitter); the fork's rayon design is bulk-synchronous over owned buffers and
stays `#![forbid(unsafe_code)]`. Nothing in the upstream race-fix series has
analogous state here. Borrowed as ideas (2026-08-23): pool-with-return
storage (`29c6dd4`, `d588318`), single-owner rendering of the full-readiness
pass (`d588318`: tiled rectangles when every group renders in one pass —
not for the last batch of an incremental decode, see `0038580`), the
`decode_parallel` fuzz target and the thread-count / chunked-input parity
sweeps (`0038580`). Still to do: a CI job for `threads` without
`allow-unsafe`.

### Infra / deps / CI

| upstream | what | disposition |
|---|---|---|
| e7436b8 #799 | drop `proc-macro-error2` (RUSTSEC-2026-0173) | **TAKE** — near-cherry-pickable into `zenjxl-decoder-macros` |
| ae13a2d #887 | `clippy::chunks_exact_to_as_chunks = "allow"` | **TAKE** (CI on stable 1.98) |
| 9bd5b83 #748 | delete benchmark-comment workflow, pin actions, `persist-credentials: false` | **TAKE** the workflow deletion (token-abuse vector); pin later |
| c588001 #882 | rustfmt import grouping (146-file churn) | **skip**; strip `use`-block hunks when porting later commits |
| version bumps, nix flake, cflite timeout, README, clippy/fmt fixups, #694/#727/#790 (jxl_cms crate) | — | **INFRA / N/A** |

## Recommended order

1. Fix CI (`kernel_tiers.rs` aarch64 gate + fmt, #887 allow) — everything
   else is unverifiable while it is red. Push the unpushed `affc97f`.
2. Correctness, smallest first: #858, #833, f16 (#706 hunk), #791 (+ a
   fixture), #856, #845+#873 (+43e2db6), TOC `check_for_error` + empty-box
   acceptance from #828, `restrictive()` ICC cap.
3. Parallel-CMS bug (fork-only): pool-with-return + fail loud + a multi-group
   ICC parity test swept over thread counts; add a `decode_parallel` fuzz target.
4. Fixtures (all ≤ 33 KB, staged in `~/tmp/audit-fixtures/`, re-downloadable):
   `mul_no_extra_channels.jxl` (32 B), `issue728_minimal`, `strategic_solid_blue`,
   `issue772_blendbug`, `issue865_large_toc`, `animated_ooo_jxlp` (#752 original),
   `clusterfuzz_541318910` (94 B), the #822–#825 POCs and our own cflite
   crash (188 B) as `fuzz/regression/` seeds.
5. Features needing a decision: u8 dithering default; #752 OOO jxlp.
6. Perf batch, measured with zenbench per item: blending bundle, #787+#817,
   #716, #888, #812 scratch cap, #720 assert; then the #812/border-buffer
   modular project.
7. Breaking-change batch (next 0.x minor): `flush_pixels -> bool`, drop
   `progressive_mode` and `unconsume`, `rgba*` extra-channel semantics.

## How to redo this audit

```sh
git clone https://github.com/libjxl/jxl-rs ~/tmp/jxl-rs   # scratch clone, never in-repo
cd ~/tmp/jxl-rs && git remote add fork https://github.com/imazen/zenjxl-decoder && git fetch fork main
git merge-base fork/main origin/main                        # fork point
git log --reverse --oneline <fork-point>..origin/main       # everything since
nice -n 19 cargo build --release -j 8 -p jxl_cli            # upstream CLI
cd ~/work/zen/zenjxl-decoder && cargo build --release -p zenjxl-decoder-cli
scripts/upstream-audit/corpus_compare.sh zenjxl-decoder/resources/test/*.jxl \
    zenjxl-decoder/resources/test/conformance_test_images/*.jxl > ~/tmp/cmp.tsv
```

Then, for each unported upstream commit, read `git show <hash>` against the
fork's matching file (`jxl/src/X` ↔ `zenjxl-decoder/src/X`) and record the
disposition here. Upstream release notes:
<https://github.com/libjxl/jxl-rs/releases>.
