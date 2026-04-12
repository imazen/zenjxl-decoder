# Progressive LF preview — implementation plan

Target branch: `feat/progressive-lf-preview` on `imazen/zenjxl-decoder`.
Worktree: `/home/lilith/work/zen/zenjxl-decoder-progressive-lf/`

## Current state vs target

Our main already contains commit `c80cfd2` (Progressive rendering of LF frames — that commit
is an ancestor of our main and `lf_preview.rs` / `maybe_preview_lf_frame` are present). That
commit is NOT to be ported — it is already in place.

Three commits remain to port:

1. **`8b8dd57`** — "Initialize the render pipeline as early as possible" (~348 LOC, 17 files)
   — structural: moves patches/splines/noise/lf_quant/color_correlation/epf_sigma fields out
   of `LfGlobalState` and into `Frame` as `Arc<AtomicRefCell<_>>`. Stage constructors change
   to accept `Arc<AtomicRefCell<_>>`. Adds `SigmaSource::Default`, `PatchesDictionary::new`,
   `Splines::is_initialized`. `prepare_render_pipeline` moves from sections.rs (called after
   HF global) to `non_section.rs` (called just after Frame::from_header_and_toc).

2. **`0d75b8f`** — "Do partial renders of Modular frames even before we have all LF" (~85
   LOC, 5 files) — depends on 8b8dd57.
   Adds `Frame::can_do_early_rendering()`, `FullModularImage::can_do_early_partial_render()`
   plus flag in struct. Adds call to `decode_and_render_hf_groups` with empty groups when
   `do_flush && !called_render_hf && frame.can_do_early_rendering()`. Changes
   `zero_fill_empty_channels` signature to also loop over LF groups. Adds `mark_group_to_be_read`
   inside `decode_lf_group` (was previously inside `read_stream`). Rearranges VarDCT group
   decode paths. NOTE: removes `maybe_preview_lf_frame` call from sections.rs top. That call
   is already in our sections.rs and is correct; 0d75b8f moves the functionality into the
   hf-group rendering path.

3. **`d8359cf`** — "Allow partial decodes of the LF global section in modular images" (~270
   LOC, 10 files) — depends on 0d75b8f.
   Adds `allow_partial: bool` param to `decode_lf_global`, lets re-entry resume via stored
   `total_bits_read` in `LfGlobalState`. Splits `FullModularImage::read` into `read` (global
   header parse only) + `read_section0` (actual section 0 channel decode). Adds
   `partial_decoded_buffers` param to `decode_modular_subbitstream`. Adds
   `BUFFER_STATUS_PARTIAL_RENDER` paths in section 0. Adds `lf_global_flush_len` tracking in
   `SectionState`. In the parser, when flushing with incomplete LF global buffer, speculatively
   try a partial decode (only Modular regular/LF frames). Also a minor save_idx fix in render.rs.
   Commit `4f3cf59` (our release) might already include the save_idx fix — need to verify.

## File map (upstream path → our path)

| upstream | ours | upstream LOC | ours LOC | divergence |
|----------|------|--------------|----------|------------|
| `jxl/src/api/inner/codestream_parser/mod.rs` | `zenjxl-decoder/src/api/inner/codestream_parser/mod.rs` | 695 | 499 | ours smaller — fewer helpers |
| `jxl/src/api/inner/codestream_parser/sections.rs` | `zenjxl-decoder/src/api/inner/codestream_parser/sections.rs` | 297 | 417 | ours adds parallel/CMYK paths |
| `jxl/src/api/inner/codestream_parser/non_section.rs` | `zenjxl-decoder/src/api/inner/codestream_parser/non_section.rs` | n/a | n/a | similar |
| `jxl/src/features/epf.rs` | `zenjxl-decoder/src/features/epf.rs` | n/a | n/a | |
| `jxl/src/features/patches.rs` | `zenjxl-decoder/src/features/patches.rs` | | | ours adds limits |
| `jxl/src/features/spline.rs` | `zenjxl-decoder/src/features/spline.rs` | | | ours adds limits |
| `jxl/src/frame/decode.rs` | `zenjxl-decoder/src/frame/decode.rs` | 771 | 1415 | ours +644 (limits, memory, jpeg) |
| `jxl/src/frame/mod.rs` | `zenjxl-decoder/src/frame/mod.rs` | 508 | 774 | ours +266 (jpeg reconstruct) |
| `jxl/src/frame/modular/mod.rs` | `zenjxl-decoder/src/frame/modular/mod.rs` | 1229 | 1252 | near-parity |
| `jxl/src/frame/modular/decode/bitstream.rs` | `zenjxl-decoder/src/frame/modular/decode/bitstream.rs` | | | |
| `jxl/src/frame/modular/decode/channel.rs` | `zenjxl-decoder/src/frame/modular/decode/channel.rs` | | | |
| `jxl/src/frame/quantizer.rs` | `zenjxl-decoder/src/frame/quantizer.rs` | | | need Default impl for LfQuantFactors |
| `jxl/src/frame/quant_weights.rs` | `zenjxl-decoder/src/frame/quant_weights.rs` | | | add None to decode_modular_subbitstream calls |
| `jxl/src/frame/render.rs` | `zenjxl-decoder/src/frame/render.rs` | 859 | 2145 | ours +1286 (parallel, CMYK, tone mapping) |
| `jxl/src/headers/modular.rs` | `zenjxl-decoder/src/headers/modular.rs` | | | derive Clone |
| `jxl/src/render/stages/convert.rs` | `zenjxl-decoder/src/render/stages/convert.rs` | | | ConvertModularXYBToF32 takes Arc<AtomicRefCell<LfQuantFactors>> |
| `jxl/src/render/stages/epf/epf{0,1,2}.rs` | `zenjxl-decoder/src/render/stages/epf/epf{0,1,2}.rs` | | | sigma field |
| `jxl/src/render/stages/noise.rs` | `zenjxl-decoder/src/render/stages/noise.rs` | | | AddNoiseStage fields |
| `jxl/src/render/stages/patches.rs` | `zenjxl-decoder/src/render/stages/patches.rs` | | | PatchesStage fields + constructor |
| `jxl/src/render/stages/splines.rs` | `zenjxl-decoder/src/render/stages/splines.rs` | | | SplinesStage fields + lazy init_draw_cache |

## Invariants to preserve

- `#![forbid(unsafe_code)]`
- `MemoryTracker` threading through every allocation path
- `JxlDecoderLimits` security budgets (max_patches, max_spline_points, max_tree_size,
  max_reference_frames, max_memory_bytes, etc.)
- Parallel rendering via `decode_groups_parallel` (`e59aa01`)
- Fragment-based parallel render with zero-cost column splitting
- CMYK pipeline (`render/stages/{cms_cmyk,black,tone_mapping,unpremultiply_alpha}.rs`)
- moxcms wrapper integration
- Our transforms/ module layout (inline, not separate crate)
- Cancellation checks (`check_cancelled`)
- XYB + u8 fusion fast path
- JPEG reconstruction path

## Staging strategy

Each upstream commit becomes at least one named commit on `feat/progressive-lf-preview`:

- Commit A: `port: 8b8dd57 Arc<AtomicRefCell<_>> refactor — move LF global fields into Frame`
  - Add `PatchesDictionary::new`, `Splines::is_initialized`, `SigmaSource: Default`,
    `LfQuantFactors: Default`, `Clone`
  - Change `LfGlobalState` to drop `patches/splines/noise`
  - Add fields to `Frame`: `patches/splines/noise/lf_quant/color_correlation_params/epf_sigma`
    as `Arc<AtomicRefCell<_>>`
  - Update `decode_lf_global` to store into `self.xxx.borrow_mut()` instead of option
  - Update `decode_hf_global` to compute `epf_sigma` at end when epf_iters > 0
  - Update `build_render_pipeline` signature and all 6 stages' constructors to take
    `Arc<AtomicRefCell<_>>`
  - Update `prepare_render_pipeline` to read from `self.xxx.clone()` instead of
    computing `epf_sigma` locally
  - Move `prepare_render_pipeline` call from `sections.rs` to `non_section.rs` (just after
    `Frame::from_header_and_toc`). Delete the sections.rs call sites.
  - Update test sites (`frame::test::splines`, `frame::test::noise`) to read from
    `frame.splines.borrow()` / `frame.noise.borrow()`
  - Update render stage tests (epf consistency, noise, patches, splines) to wrap in
    `Arc::new(AtomicRefCell::new(_))`

- Commit B: `port: 0d75b8f partial modular rendering without full LF`
  - Add `Frame::can_do_early_rendering()`
  - Add `FullModularImage::can_do_early_partial_render` flag + getter
  - Detect `has_squeeze_transform` in `FullModularImage::read`
  - Move `mark_group_to_be_read` call out of `read_stream` into the caller
    (`decode_lf_group` in decode.rs)
  - Change `zero_fill_empty_channels` to also fill LF groups; update caller
  - In parser, emit `frame.decode_and_render_hf_groups(..., vec![], do_flush, output_profile)`
    when `do_flush && !called_render_hf && frame.can_do_early_rendering()`
  - Change `render.decode_and_render_hf_groups` to accept `self.lf_global.is_none()` without
    error

- Commit C: `port: d8359cf partial LF global decode in modular images`
  - Add `total_bits_read: usize` to `LfGlobalState`
  - Change `decode_lf_global(&mut self, br, allow_partial: bool)`:
    - If `self.lf_global` already set, skip the re-decode and do `br.skip_bits(total_bits_read)`
    - Store total_bits_read when constructing the Some(LfGlobalState)
    - Move the modular_global read OUT of the LfGlobalState construction and call
      `lf_global.modular_global.read_section0(header, tree, br, allow_partial)` at the end
  - Change `FullModularImage::read` → only reads the GroupHeader and builds the struct, does
    NOT decode section 0
  - Add `FullModularImage::read_section0(header, tree, br, allow_partial)` which calls
    `decode_modular_subbitstream` with a partial_decoded_buffers counter
  - Add `decoded_section0_channels` + `needed_section0_channels_for_early_render` fields
  - Add `global_header: Option<GroupHeader>` field
  - Modify `can_do_early_partial_render` to `&& decoded_section0_channels >= needed...`
  - Add `partial_decoded_buffers: Option<&mut usize>` param to `decode_modular_subbitstream`
  - Add `br.check_for_error()` at end of `decode_modular_channel`
  - Derive `Clone` on `Transform` and `GroupHeader`
  - Update all call sites of `decode_modular_subbitstream` and `FullModularImage::read` to
    pass the new args (and `None` for `partial_decoded_buffers` where not progressive)
  - In parser, add `lf_global_flush_len` logic in `SectionState`; handle speculative partial
    LF global decode on `do_flush` for Modular Regular/LF frames
  - In parser mod.rs, allow the `has_more_frames=false + do_flush` case

- Commit D: tests — 6 integration tests for progressive LF preview semantics
- Commits E..N: follow-up fixes as needed

## Risk register

1. **Parallel-rendering thread safety**: `Arc<AtomicRefCell<Splines>>` with `splines.borrow_mut()`
   inside `SplinesStage::process_row_inplace` is a race condition under multi-threaded pipelining.
   Upstream even marks this with `// TODO(veluca): this is wrong!! Race condition in MT.` This
   is a known upstream bug. Our parallel render path needs either (a) per-thread copies, (b) a
   barrier to ensure initialization happens once, or (c) we note it as upstream's limitation and
   follow their pattern. MITIGATION: match upstream exactly; add regression note; if parallel
   render with splines frame hits this, document as known limitation.

2. **`epf_sigma` lifetime**: upstream computes `epf_sigma` after `decode_hf_global` and writes
   into `self.epf_sigma.borrow_mut()`. Our `prepare_render_pipeline` currently computes it right
   before building pipeline. The pipeline MUST be built before `decode_hf_global` now (that's the
   whole point of 8b8dd57 — initialize pipeline early). So the EPF stage holds an
   `Arc<AtomicRefCell<SigmaSource>>` that is filled in later. This works because the stage uses
   `.borrow()` inside the hot loop.

3. **`prepare_pipeline_and_finalize_lf`**: our `render.rs:2135` has a fused helper that runs
   `prepare_render_pipeline` and `finalize_lf` in parallel. With 8b8dd57, `prepare_render_pipeline`
   moves to before decode_hf_global, so this helper is called from a different place. Needs
   rethinking — may be simpler to drop the overlap.

4. **Modular section 0 partial decode**: requires `decode_modular_subbitstream` to bail out
   cleanly on a BitReader "not enough data" error AND zero-fill the partial buffer. Our
   MemoryTracker allocations inside channel decode must tolerate mid-decode error (dropping
   happens automatically via RAII).

5. **Parser's `lf_global_flush_len` heuristic**: `2 * ready > 3 * last_flush_len` means "only
   try again after at least 50% more data arrived". This prevents spinning on tiny buffer
   increments. Our parser already has `ready_section_data`.

6. **`maybe_preview_lf_frame` duplicate**: upstream 0d75b8f REMOVES the top-of-function
   `maybe_preview_lf_frame` call in sections.rs and instead relies on the new `can_do_early_rendering`
   fallback. But our sections.rs top-call is from c80cfd2. We need to either (a) keep the top
   call and add the fallback or (b) follow upstream exactly and only rely on the fallback. Given
   our frame/render.rs also calls `maybe_preview_lf_frame` at line 409, I'll match upstream.

## Out of scope

- Animation seek API (separate PR)
- Bumping crate version
- New CLI flags
- Fuzz target changes for progressive paths

## Incrementalism

After each commit A–C, the crate MUST build and the existing test suite MUST pass (no
regressions). If a commit cannot achieve this cleanly, it is split into smaller commits or
marked WIP with the explicit blocker in the message.
