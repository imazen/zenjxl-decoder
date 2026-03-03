# Plan: Sound Safe Parallel Decoding

Goal: `#![forbid(unsafe_code)]` on the `jxl` crate (without `allow-unsafe` feature) by
replacing all three unsafe parallel abstractions with heic-style "parallel decode into
owned buffers, sequential blit" patterns.

## Current Unsafe Inventory (threads feature)

### 1. SharedImageView (image/typed.rs)
- Raw pointer view into Image<T> for parallel rect access
- Used in: LF group decode (decode.rs ~11 call sites), adaptive LF smoothing
- Manual Send/Sync impls, no debug overlap tracking
- Weakest abstraction: no lifetime enforcement, &self -> &mut T without UnsafeCell

### 2. DisjointRowAccess (image/disjoint.rs)
- Raw pointer row access for HF coefficient images
- Used in: HF group decode (render.rs)
- Has debug-mode borrow tracking via Mutex<HashSet>
- Best of the three, but still raw pointer + manual Send/Sync

### 3. ParallelOutputAccess / SharedOutputView (render/buffer_splitter.rs, image/output_buffer.rs)
- Raw pointer sub-views of output buffers for parallel render
- Used in: parallel render pipeline (render.rs Phase 3b)
- Has debug-mode rect overlap tracking

### 4. Image::row_info_mut (image/typed.rs)
- Exposes raw pointer for DisjointRowAccess construction
- Removed when DisjointRowAccess is removed

## Replacement Strategy

### SharedImageView -> Per-group owned scratch images (LF decode)

Current: Each parallel LF group task calls `unsafe { lf_views[c].get_rect_mut(r) }` to
write directly into sub-rects of shared lf_image/quant_lf/hf_meta maps.

Replacement:
- Allocate per-group scratch Images for each output (lf channels, quant_lf, hf_meta maps)
- Size each to the group's lf_group_rect dimensions
- Decode into owned scratch images inside rayon tasks
- Collect results via `.collect::<Result<Vec<_>>>()?`
- Sequential loop copies each group's scratch into the correct rect of the shared image
- Use `ImageRectMut::copy_from()` or row-by-row memcpy for the blit

Cost: Extra allocation per group + memcpy. Mitigate with buffer pool (reuse across frames).

### DisjointRowAccess -> Pre-split owned coefficient images (HF decode)

Current: HF coefficient images are wrapped in DisjointRowAccess, parallel tasks get
row_guard() for their group's rows.

Replacement options:
a) Allocate per-group coefficient scratch, collect + blit (same as LF pattern)
b) Pre-split the coefficient image into per-group owned Vec<i32> slices at allocation
   time (the group grid is known), pass owned slices into rayon tasks
c) Actually: HF coefficients already use per-group pixel buffers from a pool (render.rs
   Phase 2). Check if DisjointRowAccess is even needed or if the pixel pool already
   covers this.

### ParallelOutputAccess -> Collect rendered strips, write sequentially (render)

Current: Phase 3b renders work items in parallel, each writing to non-overlapping
sub-views of the output buffer via ParallelOutputAccess.

Replacement:
- Each parallel render task writes into a small owned strip buffer (already partially true)
- Collect rendered strips via `.collect::<Vec<_>>()`
- Sequential loop copies strips into the output buffer
- Or: just run Phase 3b sequentially (it's typically fast relative to Phase 2 decode)

### adaptive_lf_smoothing.rs

Uses SharedImageView for parallel smoothing output. Same fix: per-chunk scratch + blit.

## Non-blockers (already safe without allow-unsafe)

- `get_distinct_indices` - has split_at_mut fallback
- `set_len` uninit optimization - has resize fallback
- `as_maybe_uninit_slice` + SIMD casts - has scalar fallback
- `new_from_ptr` - allow-unsafe gated public API for embedders

## Risks

1. **Allocation overhead**: 518757a showed sequential pre-allocation of per-group buffers
   caused 0.59x regression. Mitigation: allocate inside rayon tasks (parallel page faults)
   and use buffer pools.

2. **Memcpy overhead**: Unknown. Never benchmarked. For 4K images, each LF group is ~256x256
   floats x 3 channels = ~768KB. Total blit for a 4K image with ~16 LF groups = ~12MB memcpy.
   Should be fast relative to decode, but must measure.

3. **Phase 3b render parallelism**: If we make render sequential, we lose the parallel
   EPF/color/save speedup. Need to benchmark whether Phase 3b parallelism matters vs
   Phase 2 decode parallelism.

## Verification

- All 885 tests must pass
- Conformance: 184/184 codec-corpus, 17/23 Level 5
- Benchmark before/after on multi-tile 4K images (city, forest, landscape, portrait)
- Compare 1-thread and 8-thread performance

## Success Criteria

- `#![forbid(unsafe_code)]` on jxl crate compiles without allow-unsafe feature
- No performance regression >10% on 8-thread 4K decode
- All tests pass
