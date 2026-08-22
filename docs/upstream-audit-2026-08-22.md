# Upstream audit 2026-08-22 — measurements and per-commit evidence

Companion to [`UPSTREAM_SYNC.md`](UPSTREAM_SYNC.md). Fork `affc97f`
(main, 2026-08-13), upstream libjxl/jxl-rs `088ec7f` (2026-08-21, v0.6.0+7).
Machine: Apple M-series, 12 cores (8P+4E), macOS; release builds of both CLIs
(`nice -n 19 cargo build --release -j 8`), no `target-cpu=native`.
Tools: `scripts/upstream-audit/`. Scratch inputs (POCs, generated test files,
staged fixtures) are under `~/tmp/audit-fixtures/` and `~/tmp/conf/`.

The six per-subsystem reports with file:line evidence for every one of the
161 upstream commits (233 KB) are at `~/tmp/jxlrs_audit_{A..F}_*.md`; their
summary tables are reproduced at the end of this file.

## 1. Pixel equality, fork vs upstream HEAD (1 thread, 16-bit output)

`corpus_compare.sh` over `resources/test/*.jxl` (58 small + 22 large) and
`resources/test/conformance_test_images/*.jxl` (38): every file both decoders
accept is **bit-identical** at full 16-bit precision, except:

| file | result | why |
|---|---|---|
| `cmyk_layers` | 12,424 px differ, up to full range | fork converts CMYK→RGB in-decoder via CMS (`CmsCmykToRgbStage`); upstream removed CMS (#754) and emits untransformed data |
| `cropped_traffic_light`, 5 `animation_*` | not compared | APNG output; the stdlib PNG reader does not parse animation chunks |
| `patches`, `progressive` (+`_5`) | identical at 16-bit | (an earlier 8-bit-truncated pass showed ±1; artefact of the `>>8` comparison, gone with `FULL16=1`) |

Upstream-only fixtures (copied from the scratch clone):

| file | bytes | fork | upstream | vs djxl 0.12 (u8) |
|---|---|---|---|---|
| `issue728_minimal.jxl` (#739, vsqueeze) | 4314 | ok | ok | identical |
| `strategic_solid_blue.jxl` (#734, hsqueeze) | 131 | ok | ok | identical |
| `issue865_large_toc.jxl` (#866) | 6005 | ok | ok | identical |
| `issue772_blendbug.jxl` (#773) | 11554 | ok | ok | ±1 on 25 % (dither; see §4) — fork == upstream at 16-bit |
| `invalid_animated_ooo_jxlp.jxl` (#752/#828) | 4329 | `Invalid ISOBMMF container` | `InvalidBox` | — |

## 2. Upstream fuzzer crash POCs (issues #822–#825, July 2026)

All introduced upstream by #812 (transform-application rewrite) and fixed by
#829–#833. The fork, which never took #812, rejects every POC cleanly with the
same error string upstream HEAD now produces:

| POC | bytes | fork & upstream result |
|---|---|---|
| #822 (unwrap in `step.rs`) | 33371 | `Modular stream requested a global tree but there isn't one` |
| #823a / #823b (div-by-zero in `meta_apply.rs`) | 110 / 197 | `ANS stream checksum mismatch` / `Non-zero padding bits` |
| #824 (splines unwrap) | 112 | `Too large area for spline: 170076677760, limit is 4296376320` |
| #825 (OOB in `meta_apply.rs`) | 64 | `Modular stream requested a global tree but there isn't one` |

## 3. Bugs reproduced in the fork

### 3.1 `Mul` blend without extra channels (upstream #858)

`libjxl/conformance#48` adds `testcases/mul_no_extra_channels/input.jxl`
(32-byte codestream, 8×8, Modular, `Mul` blending, zero extra channels).

| decoder | result |
|---|---|
| fork `affc97f` | **`Error: Source file truncated`** |
| upstream `088ec7f` | decodes; identical to djxl 0.12 |
| djxl 0.12.0 | decodes |

Cause: `headers/frame_header.rs:126-129` only reads the `clamp` bit when
`num_extra_channels > 0`; libjxl (`frame_header.cc`) and upstream since #858
read it for `Mul` unconditionally, so every later field is shifted by one bit.

### 3.2 Parallel decode + CMS: colour transform skipped on most tiles (fork-only)

Input: `city_4k_q75` decoded by djxl to PNG, tagged with the ACEScg Linear ICC
profile via `sips -e`, re-encoded `cjxl -d 1 -e 3` → 4096×2519 XYB image whose
embedded profile is ICC-only (`jxlinfo`: "600-byte ICC profile"), so the CLI
(which always installs `Lcms2Cms`) must run `CmsStage` (ICC → sRGB).

| comparison (16-bit output) | differing pixels | max |Δ| / 65535 |
|---|---|---|
| `--num-threads 1` vs upstream | 4,627,138 / 10,317,824 | **1** |
| `--num-threads 1` vs `--num-threads 2` | 8,360,620 | **18818** |
| `--num-threads 1` vs `--num-threads 8` | 9,601,155 | **18818** |

`groupdiff.py` (256-px groups, 16×10): with 2 threads the first **3** groups
are correct and the other 157 are wrong; with 8 threads all 160 are wrong.
That matches the mechanism read from the code: `frame/render.rs:1781-1786`
creates `rayon::current_num_threads() + 2` transformers, the main render
context takes one, `try_for_each_init(|| factory.create(1).ok(), ..)`
(`frame/render.rs:1155`, `:1202`) pops one per rayon *leaf* and never returns
it, and `render/stages/cms.rs:159-161` silently passes pixels through when a
context has no transformer. rayon 1.12 splits into ≥ 2·T leaves before
stealing, so the pool is exhausted immediately.

Reach: the CLI (`zenjxl-decoder-cli/src/main.rs:140`) and any caller using
`JxlDecoderOptions::with_cms` together with `parallel` (the default under
`threads`). `decode()` / `decode_with` default to `cms: None`, and zenjxl sets
no CMS, so imageflow is not affected today. The CMYK path uses a single
mutex-guarded transformer (`cms_cmyk.rs`) and is correct.

Same image with a Display P3 / AdobeRGB / Generic RGB profile is unaffected,
because cjxl stores those as enum colour encodings and no CMS stage is built.

### 3.3 f32→f16 subnormal conversion (upstream #706, `float16.rs` hunk)

Python emulation of `util/float16.rs:from_f32` (identical copy in
`zenjxl-decoder-simd/src/float16.rs`) against IEEE round-to-nearest-even:

| input | fork | IEEE |
|---|---|---|
| 2^-15 | 0x0100 | 0x0200 |
| 2^-16 | 0x0080 | 0x0100 |
| 6.0976e-5 (largest subnormal) | 0x01ff | 0x03ff |
| 2^-24 (smallest subnormal) | 0x0000 | 0x0001 |

Every f16-subnormal result is half the correct value (shift off by one,
`>> (shift + 14)` should be `>> (shift + 13)`), no rounding, and inputs in
[2^-25, 2^-24) flush to zero. Reachable through `JxlDataFormat::F16` output
(`ConvertF32ToF16Stage`, `simple_pipeline/save.rs`); `decode()` is u8-only.

### 3.4 Integer overflow in modular-XYB conversion (upstream #833)

ClusterFuzzLite batch run 32539550999 (2026-08-22 01:10 UTC, target
`decode_with_limits`, ASan) panicked at
`render/stages/convert.rs:114:30 attempt to add with overflow`
(`(input_b[0][i] + input_y[0][i]) as f32 * scale_b`). Crash input: 188 bytes
(`~/tmp/audit-fixtures/cflite_xyb_overflow_convert114.jxl`; the release CLI
wraps and then reports `Float is NaN or Inf`). Upstream's fix converts each
operand to f32 before adding.

### 3.5 Section buffers sized from the untrusted TOC (upstream #856)

Read from code, not reproduced with a crafted file:
`api/inner/codestream_parser/mod.rs:243` `buf.data.resize(buf.len + SECTION_PADDING, 0)`
is infallible and uses the TOC-declared length (one entry can encode up to
2^30 + 4,211,712 bytes, `headers/toc.rs:78-86`); the allocation is not charged
to `max_memory_bytes`. A tiny truncated file therefore costs a ~1 GB
calloc/memset per frame on 64-bit and aborts on i686.

### 3.6 CI red on `origin/main` since 2026-08-01

Run 30684960419 (`4d7860f`): `clippy` and `format` jobs fail —
`benches/kernel_tiers.rs:61,81` use `jxl_simd::NeonDescriptor` without a
`cfg(target_arch = "aarch64")` gate (the x86 clippy job builds it under
`--all-targets --all-features`), and the file is not `cargo fmt`-clean. Stable
1.98 (2026-08-20) additionally warns `clippy::chunks_exact_to_as_chunks` at 53
sites; upstream allowed the lint in #887. Local commit `affc97f`
("drop 24 clippy allow attributes", 2026-08-13) is **not pushed**.

## 4. u8 dithering (upstream #841) — what it changes

Upstream 0.6.0 adds libjxl's 32×32 blue-noise dither (values in ±0.49219,
mean ≈ 0) to every `ConvertF32ToU8Stage` output, unconditionally. Measured on
`issue772_blendbug.jxl` (lossy, RGBA):

| pair (u8) | pixels differing by 1 |
|---|---|
| fork vs djxl 0.12 | 183,966 / 750,000 (24.5 %) |
| upstream (dithered) vs djxl | 56,204 / 750,000 (7.5 %) |
| fork vs upstream | 183,898 / 750,000 |

Lossless 8-bit content is unchanged by the dither (|d| < 0.5), confirmed by the
exact u8 matches on `issue728_minimal`, `strategic_solid_blue`,
`issue865_large_toc` against djxl.

## 5. Rough speed comparison (CLI `--speedtest`, not zenbench)

`speed_compare.sh`: 2 interleaved passes × 5 reps, `nice -n 19`, fork
`--num-threads 1`, upstream `RAYON_NUM_THREADS=1`. MP/s; higher is faster.

| file | fork p1 | upstream p1 | fork p2 | upstream p2 | upstream/fork |
|---|---|---|---|---|---|
| green_queen_modular_e3 | 14.6 | 16.0 | 15.4 | 16.0 | 1.06 |
| issue648_palette0 | 15.8 | 17.9 | 16.0 | 17.9 | 1.12 |
| grayscale_patches_modular | 36.0 | 35.9 | 35.6 | 35.9 | 1.00 |
| squeeze_edge | 45.4 | 80.7 | 45.2 | 80.6 | **1.78** |
| sunset_logo | 9.4 | 15.4 | 9.5 | 15.4 | **1.63** |
| lz77_flower | 6.93 | 7.17 | 6.94 | 7.17 | 1.03 |
| delta_palette | 18.6 | 20.5 | 18.5 | 20.5 | 1.10 |
| green_queen_vardct_e3 | 45.1 | 46.4 | 45.5 | 46.6 | 1.03 |
| efb | 27.4 | 29.7 | 27.3 | 29.8 | 1.08 |
| zoltan_tasi_unsplash | 44.0 | 44.1 | 43.7 | 43.9 | 1.00 |
| bike_web_q85 | 65.2 | 66.2 | 65.0 | 66.3 | 1.02 |
| cafe_web_q80 | 49.7 | 51.3 | 49.9 | 51.3 | 1.03 |
| city_4k_q75 | 49.3 | 51.3 | 49.6 | 51.3 | 1.04 |
| bicycles (conformance) | 19.4 | 22.8 | 19.6 | 22.8 | 1.17 |

Default thread count (12) on both:

| file | fork | upstream | upstream/fork |
|---|---|---|---|
| green_queen_modular_e3 | 48–56 | 59 | ~1.1 |
| squeeze_edge | 52–53 | 106–111 | **2.0** |
| sunset_logo | 16.1–16.3 | 24.7 | **1.5** |
| bike_web_q85 | 277–279 | 299–301 | 1.08 |
| city_4k_q75 | 272–274 | 278–282 | 1.03 |
| bicycles | 46.6–47.2 | 49.4–51.9 | 1.08 |

Reading: VarDCT is at parity; the modular gap is upstream's 2026-06/07 modular
work (#787 single-symbol path, #716, #793, #797, #812) and, multi-threaded,
its parallel modular transforms. These are aarch64/NEON numbers; re-measure on
x86 with zenbench before and after any port.

## 6. Fixtures staged for the fork (`~/tmp/audit-fixtures/`)

| file | bytes | source | intended use |
|---|---|---|---|
| `mul_no_extra_channels.jxl` | 32 | libjxl/conformance#48 | regression for #858 (decode must succeed, 8×8) |
| `issue728_minimal.jxl`, `strategic_solid_blue.jxl` | 4314, 131 | jxl-rs #739/#734 | regressions for the already-ported #731/#735 |
| `issue772_blendbug.jxl` | 11554 | jxl-rs #773 | blending with missing references |
| `issue865_large_toc.jxl` | 6005 | jxl-rs #866 | large-TOC incremental parse |
| `animated_ooo_jxlp_752_original.jxl` | 4329 | jxl-rs #752 (`e7405e0`) | positive test once OOO jxlp is supported (djxl decodes it) |
| `clusterfuzz_541318910.jxl` | 94 | jxl-rs `600f977` | flush-on-truncated-modular-frame seed |
| `jxlrs_issue82{2,3a,3b,4,5}.poc.jxl` | 64–33371 | jxl-rs issues | `fuzz/regression/` seeds (all currently rejected cleanly) |
| `cflite_xyb_overflow_convert114.jxl` | 188 | our ClusterFuzzLite run 32539550999 | regression for #833 |

All are below the 30 KB binary threshold except `jxlrs_issue822.poc.jxl`
(33 KB — keep in block storage per the fuzz-corpus rule).

---


# Per-subsystem top picks and re-verification (from the six audit reports)

Each group's full per-commit entries (status, fork file:line evidence, impact, port effort) are in `~/tmp/jxlrs_audit_<group>.md`; the dispositions are consolidated in `UPSTREAM_SYNC.md`.


## Group A — modular

### Previously-ported items re-verified
| Item | Verdict | Evidence |
|---|---|---|
| #731 vsqueeze grid boundary (upstream 159c60b) | **PRESENT** | `frame/modular/transforms/squeeze.rs:604-612` `avg_row_next = if !has_tail && h == 1 { match in_next_avg { None => in_avg.row(0), Some(mc) => mc.row(0) } }` — identical to upstream's fix. |
| #735 hsqueeze grid boundary (upstream 3d1d0c2) | **PRESENT** | `squeeze.rs:475-484` `let w = in_res.size().0; if w == 0 { let out_h = out.data.size().1; for y in 0..out_h {...} }` — loops over output height, as upstream. |
| #738 SingleGradientOnly direct path (upstream 371f033) | **PRESENT** | `decode/specialized_trees.rs:349-371` uses `clamped_gradient(...)` + `read_signed_clustered_inline` + `dec.wrapping_add(pred as i32)`; selected at `:491-503`. |
| #766 LZ77 distance cluster capture (upstream 81dc81e) | **PRESENT** | `entropy_coding/decode.rs:76` field `lz_dist_cluster: u8`, captured at `:636-637` in `Histograms::decode` before any `resize`, used at `:263` (RLE detection) and `:396-401` (distance read). No remaining `context_map.last()` in `SymbolReader`. |

### Top picks
1. **814612d — DeltaPalette + Weighted predictor.** Silent wrong pixels on spec-valid streams; the fork's `palette.rs:242-255,598-620` and `predict.rs:456-464` are exactly upstream's pre-fix code, and the upstream patch is against the fork's current `WeightedPredictorState` layout. Cherry-pick + add a multi-group-row delta-palette/Weighted fixture.
2. **aed4e9e — Mul-blend clamp bit.** One-attribute fix; any `Mul`-blended frame without extra channels is currently misparsed (`headers/frame_header.rs:126-129`).
3. **7f8ee4f sub-commit 1 — `group_scratch_buffers_limit = Some(0)` for modular frames.** The fork's low-memory pipeline keeps every rendered modular tile in `scratch_channel_buffers` forever in modular mode (`group_scheduler.rs:250-257,416-423`); upstream cut exactly this. Small, measurable with `examples/heaptrack_decode` on a modular fixture.
4. **204c0d0 — XYB conversion overflow** (`render/stages/convert.rs:114`). One-liner, removes a fuzz-panic.
5. **088ec7f — drop internal padding.** Small, ~7%/tile memory by the fork's own allocation formula, simplifies the channel decoder.
6. **Perf batch: 654a985+8e8769b (single-symbol), ad5ead5 (flat-tree enum), then afd41c9 (reader specialisation / fast-lossless), then a safe-Rust adaptation of 4fcfb24.** Upstream's quoted wins: 1.4× (jxl-art) for #787, ~10% for #716, ~7% e3 / ~30% jxl-art for #793 (with unsafe); all against code whose shape the fork still has.
7. **Later / conditional:** #812 depth-first rewrite + border-buffer series (2d0b720, c066ee2, 07cb870, 4495876) — only worth it after the fork's own modular retentions (`preserve_buffers` cloning at `frame/modular/mod.rs:717`, pipeline scratch cache) are fixed and heaptrack still shows squeeze/palette neighbour tiles dominating. Smooth previews (6e82649 & co.) only if a progressive-preview UI becomes a requirement.

Cross-group note (render pipeline, not modular): `render/low_memory_pipeline/group_scheduler.rs:395-401,570` compute `x1 - x0`/`y1 - y0` without the `x1 < x0 || y1 < y0` guard that #812 added (and that upstream's later a608c0b "Clip low-memory ready rectangles to the input size" extends). Reachability in the fork is UNVERIFIED; worth a look by whoever owns the render group.


## Group B — vardct render

### Previously-ported items re-verified
| Item | Fork evidence |
|---|---|
| #725 stage pruning with shift (a47d786) | `render/builder.rs:151-153` `if stage.shift() != (0, 0) { stage_is_used[i] = true; }` |
| #740 LF preview alpha overflow + BGR order | `frame/lf_preview.rs:65,73` `((1u16 << bit_depth) - 1) as u8` / `((1u32 << bit_depth) - 1) as u16`; `:233-244` `(r,g,b) = (2,1,0)` for `Bgr|Bgra` |
| #741 EC upsampling validated after dim_shift | `headers/frame_header.rs:669-685` (`effective_upsampling = ec_upsampling << dim_shift`, `<upsampling || >8`) |
| #742 reject mixed-upsampling patches | `headers/frame_header.rs:687-698` `PatchesUnsupportedMixedUpsampling` |
| #756 modular output channels with grid None | `frame/modular/mod.rs:713-763` (`grid_is_none`, crop to group rect); fixture `resources/test/upsampled_alpha.jxl` present |
| #784 non-existent LF frame error | `frame/decode.rs:180-184` `Err(at!(Error::NoLfFrame(lf_level)))`; `error.rs:280` |
| #776 save_lowmem 1px-wide overflow | `render/low_memory_pipeline/save/mod.rs:110` `display_row_step()`; `headers/image_metadata.rs:87` |

### Top picks
1. **#845 + a608c0b** (`group_scheduler.rs` clamp + `x1>=x0` guard, `render_group.rs` edge-padding on `start_of_row/end_of_row`): closes a
   real wrong-pixel path at the right image edge for streamed permuted-TOC files with a tiny last group column, removes an unguarded usize
   underflow, and makes the one-shot rayon/final-rerender paths correct by construction instead of by copy-back order. ~30 lines.
2. **Regression tests that are free**: 6401d6e's `multiple_lf_420` pixel assertions (guards the packed LF layout end-to-end — the exact
   thing a careless port of 365eb80/#805 would break) and e12b99b's `squeeze_empty_residual.jxl` chunk 1..=16 flush loop.
3. **57e515e** floor-semantics fill: small, output-neutral, removes a latent mismatch that upstream only caught with randomized scheduling.
4. **#806's 4-line "already rendered → return" guard** in `decode_hf_group`: makes the LF-only re-render of complete single-pass groups
   structurally impossible rather than relying on `changed_since_last_flush`.
5. Perf crumbs, measure before/after with zenbench: #720 assert (1 line), #801 XybParams (≤0.5%), da96606 inline/`get_rows_mut_into`.
6. Do NOT take: 365eb80 code change (wrong for the fork's layout), #861 (libjxl rejects; spec pending; would also need a JPEG-recon guard).


## Group C — blend color features

### Previously-ported items re-verified
| Item | Verdict | Evidence |
|---|---|---|
| c60408d spline SIMD-width consistency (fork `8474b77`) | **PRESENT** | `src/features/spline.rs:574-606`: single `for iter in 0..=num_chunks` loop with the `[0.0; 16]`-padded remainder (`into_remainder()`), and `:632` calls `draw_segment_inner` once — the old `maybe_downgrade_256bit/128bit → ScalarDescriptor` chain that produced width-dependent rounding is gone. Matches upstream's diff hunk-for-hunk. |
| 83db36f PQ/HLG F16 clamp (fork `94373e0`) | **PRESENT** | `src/frame/render.rs:1628-1632` derives `clamp_range_for_f16` (PQ → (0,1), HLG → (−0.074, 1.1)); passed via `add_conversion_stages` (`:75,97`, call site `:2086-2091`; extra channels get `None`, `:2106`); applied in `ConvertF32ToF16Stage::process_row_chunk` (`src/render/stages/convert.rs:594-598`). |
| v0.3.3 "Blending SmallVec" perf port | **PRESENT but over-claimed** | `src/render/stages/blending.rs:114,116` (`SmallVec<[&mut [f32]; 8]>`, `SmallVec<[&[f32]; 8]>`) and `src/features/patches.rs:721,764`. The accompanying commit message "eliminate per-row heap allocations in blending stage" (`e7166ed`) is not true: three per-row-chunk heap allocations remain — `slice!(..).collect::<Vec<_>>()` (`blending.rs:145` via `src/util/ndarray.rs:13-17`), the `ec_blending_info` `.collect()` (`:138-142`), and `tmp = vec![..]` inside `perform_blending` (`src/features/blending.rs:31`). Upstream #709 removes all of these. |
| #707 identical-ICC CMS skip (fork's moxcms path) | **PRESENT, predates upstream** | `src/api/color.rs:1269` (`a == b && !self.is_cmyk()`), consumed at `src/frame/render.rs:1739,1747`. Fork commit `71b661a` (2026-02-19). |

### Top picks
1. **Blending bundle: f1388c4 → 012a292 → 7b223de, measured by a zenbench port of a76e651.** Bit-identical, zero-allocation, in-place, SIMD blending mapped 1:1 onto the fork's `simd_function!`/`F32SimdVec` layer (same names). Upstream measured ≈3–4× on the alpha-composite modes; the fork's patches path (text/screenshot content) currently allocates per patch-row. Medium total effort, near-cherry-pickable.
2. **6430786 — raise `JxlDecoderLimits::restrictive().max_icc_size` from 1 MiB to 16 MiB.** One-line fix for a false rejection of real CMYK profiles (3.3 MiB GRACoL, 2.6 MiB SWOP found locally). Also tell the user about the identical 1 MiB cap in `zenjxl/src/decode.rs:630` (other repo).
3. **f694be5 — blue-noise dithering for 8-bit output.** The fork's `decode()` is u8-only and its parity reference (`djxl`) already dithers; porting closes that gap and reduces banding, but it changes default pixels and requires a test-tolerance relaxation (0.003 → 0.004) — needs a user decision on default-on vs `dither_u8` opt-in before any code lands. Three stages need the term (`convert.rs`, `xyb.rs`, `from_linear.rs`).
4. **ebeed75 — add the `issue772_blendbug.jxl` fixture** (11.5 KB, auto-picked-up by `all_jxl_fixtures()`) as a free regression gate for the missing-reference blending path, even though the fork is already immune.
5. Everything else (35fb0ad, 5410803, 041b060, 8d16561, fbf05df, 7374aa3, 6b6b005): nothing to do.


## Group D — api parsing

### Top picks
1. **7809f88** — section buffers: `resize` to full TOC-declared size is infallible (`codestream_parser/mod.rs:243`) → abort on i686, ~1 GB memset/RSS from a tiny crafted file on 64-bit; small fix that must keep `SECTION_PADDING`.
2. **e7405e0 + 3c4f224** — out-of-order `jxlp` (ftyp v1): the one real "valid file fails to decode" gap in this group (`box_parser.rs:363-365`; ftyp never parsed `:457`). Implement in `src/api/inner/box_parser.rs`; use the original #752 fixture as a positive test; add `try_reserve` and stop cloning/memsetting box buffers per step (`:18,93`).
3. **#828 item 3 (#811)** — two missing `br.check_for_error()` calls in `src/headers/toc.rs:89,102`: silent overread of a permuted TOC's permutation during incremental parsing can corrupt the parse position (wrong TOC → decode error or wrong pixels). Trivial.
4. **#828 item 1** — accept empty boxes (`box_len == header`) and size-0-to-EOF for any box type (`box_parser.rs:331-338`) to match libjxl; and decide what `decode()` should do with errors from trailing boxes after the last frame (`convenience.rs:240-249` currently fails the whole decode).
5. **e883140** — safe strided `JxlOutputBuffer::new_with_stride` (trivial, additive; aligns with the fork's own stride rule).
6. **c5528f6** — `JxlPixelFormat::rgb*` helpers (trivial, additive).
7. Breaking-release batch: **d782c19** (`flush_pixels -> Result<bool>`), **0977812** (drop no-op `progressive_mode`), **1cc9ab7** (drop dead `unconsume`, add `file_length`), `rgba*` extra-channel semantics.
8. **#828 item 9 / #771** (+ #834) — keep the finished frame until the next one exists so `flush_pixels` can render at a frame boundary (CLI `--allow-partial-files` path); medium.
9. **#678/#702 seek API** — only if an animation consumer needs it; port the post-#828 design as one project, not the March commits.


## Group E — multithreading

### Top picks
1. Fix 4.1 (CMS pool exhaustion -> untransformed tiles) using the fe3b3c9 pool-with-return pattern, fail loud on
   exhaustion, and add a multi-group non-sRGB-ICC parallel parity test with a thread-count sweep. Pixel-correctness class.
2. Add a >= 2x2-group 4:2:2 JPEG-reconstruction fixture to the parallel parity tests to confirm/refute 4.2, then fix
   via disjoint work items (3.2) or a column-overlap check in `can_band_split`.
3. Port a608c0b's rect clipping (group B) and include the parallel one-shot case in its regression test.
4. `decode_parallel` fuzz target (8b4ecb0 analogue) + chunked-input parallel parity tests (15fb1de analogue) + a
   `threads`-without-`allow-unsafe` CI job.
5. Single-owner sector emission + border-lifetime counting (b839216 idea): removes double rendering, re-enables the
   fragment path for bordered pipelines, and deletes the sequential final re-render of incremental decodes. Land 43e2db6's
   `start_of_row`/`end_of_row` alignment in the same change.
6. Move noise generation into Phase 2.
7. Measure small-image MT vs ST; adopt a 032d101-style work gate only if the numbers say so.
8. Later, bundled with the #812 modular rewrite: f76be0c + 347dc98 + 2ccf6af + 4fde1db for parallel modular transforms.


## Group F — perf infra

### Top picks
1. **Fix fork CI first** (not an upstream port): gate `benches/kernel_tiers.rs` to aarch64 + `cargo fmt` (red since 2026-08-01), and add `chunks_exact_to_as_chunks = "allow"` (#887) before pushing anything else — every other port is unverifiable while clippy/format are red.
2. **#706 f16 denormal fix** — wrong F16 output pixels (half value, unrounded) for every sample below 6.1e-5, in both `util/float16.rs` and `zenjxl-decoder-simd/src/float16.rs`; bring upstream's tests along. Also a candidate upstream report (upstream's main-crate copy is still broken) — needs user sign-off before filing against libjxl.
3. **#833 cross-reference** (other group): confirmed reproducible panic in the fork's own ClusterFuzzLite run of 2026-08-22 (`convert.rs:114`). Treat as port-now wherever it is assigned.
4. **#799 proc-macro-error2 removal** — RUSTSEC-2026-0173, near-cherry-pickable.
5. **#748 delete `benchmark_comment.yml` + PR benchmark job** — closes a token-abuse vector; then SHA-pin actions at leisure.
6. **#722 inline annotations (20 sites)** — cheap, but measure with zenbench before claiming anything; consider dropping the fork's used-mask gating (#717) in the same measured pass.
7. **#709 completion** (not assigned): the two remaining per-row heap allocations in blending (`ec_blending_info` rebuild, `perform_blending` tmp Vec).
8. wasm32-wasip1 CI job (#706) so the default-enabled `wasm128` feature is tested; #732 16-bit PNM for the CLI; README refresh (#879).
