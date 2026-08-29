# Changelog

All notable changes to this project will be documented in this file.

This project is a fork of [libjxl/jxl-rs](https://github.com/libjxl/jxl-rs). The changelog covers changes made in this fork.

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next 0.x minor release.
     Add items here as you discover them. Do NOT ship these piecemeal. -->
- `JxlDecoderOptions::coalescing` and `with_coalescing()` — accepted but read
  nowhere; either implement the `false` case (return each frame's uncomposited
  rectangle with its crop origin and blend mode, as libjxl's
  `JXL_DEC_SET_COALESCING(false)` does) or remove both. Documented as
  unimplemented in the meantime (#57).

### Fixed

- **The same file decoded differently depending on the CPU, in three separate
  places.** Six hand-written SIMD tiers dispatch at runtime, and nothing
  checked that they agreed. The fuzzers could not have found any of this:
  `fuzz/` builds with `default-features = false`, so it only ever exercises the
  scalar tier.
  - `round_store_u8`/`round_store_u16` documented out-of-range input as
    "unspecified", and the tiers duly disagreed. A **negative sample stored 255
    on avx512 and 0 on every other tier** — a black/white inversion selected by
    the CPU — because `_mm512_cvtusepi32_epi8` (`vpmovusdb`) reads its i32 lanes
    as unsigned. The reachable caller is the Gamma/DCI branch of
    `XybToU8Stage`, which clamps the *absolute* value before the transfer
    function and restores the sign with `copysign`; measured on
    `3x3_srgb_lossy.jxl`, samples from **-44.6 to +262.3** reach that store, and
    reproducing the avx512 conversion on neon flips 7 of its 36 output bytes.
    A second divergence was found while writing the test: **large positive
    samples stored 0 on sse42/avx and 255 on neon**, because `cvtps_epi32` maps
    out-of-range floats (`+inf` included) to `INT_MIN` while `vcvtq_s32_f32`
    saturates to `INT_MAX`. Both are fixed by clamping in float space *before*
    the integer conversion, at the trait layer so every caller is covered
    (5afe0d5).
  - **Exact halves rounded away from zero on the scalar tier and to even on all
    five SIMD tiers.** Ties-to-even is what the hardware rounding mode gives, so
    the scalar outlier moved. This is the tier that runs on **i686**, on wasm
    without simd128, and in the fuzzers, so the crate's "matches djxl
    bit-for-bit" claim did not hold there (5afe0d5).
  - **A lossless file decoded differently per CPU.** `hsqueeze`/`vsqueeze` split
    at `h & !(lanes - 1)`, sending leading rows through the vector body and the
    remainder through a scalar tail — two independent implementations, the body
    in wrapping i32 and the tail in i64 narrowed with `as i32`. They agree until
    something overflows i32 and then diverge, because `diff / 2` does not
    commute with wrapping; for `(0, i32::MAX, i32::MIN, i32::MAX)` they returned
    `(1700091220, -1700091221)` vs `(1073741823, -1073741824)`. Since the split
    point is the lane count, the output moved with the CPU. The duplicate is
    deleted: the tail now calls the same `unsqueeze_impl` at `LEN == 1`. Streams
    that do not overflow — everything a valid encoder emits — decode to exactly
    the same bytes as before (83fb368).
- **A 256-cluster context map panicked (debug) or silently decoded nothing
  (release).** `Histograms::decode` computed `num_histograms` as
  `*context_map.iter().max().unwrap() + 1` in **u8**, while both sibling sites
  (`Histograms::num_histograms`, `verify_context_map`) widen first. Cluster
  index 255 is legal — `decode_context_map` rejects only values above
  `u8::MAX`, and `verify_context_map` requires every value below the maximum to
  be present, so a 256-cluster map is constructible — and `255u8 + 1` overflows:
  a panic under debug assertions, and a wrap to zero in release, leaving
  `uint_configs` empty and the codes decoded for no histograms at all. Now
  widened to `usize` before the add.
- **Two entropy sub-streams were never validated with `check_final_state`**: the
  patches dictionary and the permuted TOC. That check verifies the reader's
  deferred errors, the bit reader's, and the ANS final-state checksum, and every
  other sub-stream in the decoder already ran it (splines, the modular tree,
  coefficient orders, ICC, the context map), as does libjxl. Corruption in
  either is now caught at the sub-stream that carries it, with
  `AnsChecksumMismatch`, instead of incidentally further downstream. **This is
  hardening, not a silent-corruption fix**: a randomized search of 200,000
  one-to-three-byte corruptions per fixture across `has_permutation`,
  `has_permutation_with_container` and both `grayscale_patches` fixtures, plus
  an exhaustive single-bit sweep of `has_permutation.jxl`, produced 160 inputs
  that these checks reject and **zero** that were accepted without them — the
  downstream section-length and bit-padding checks caught every one, just later
  and with a less specific error.
- **The fuzz regression harness could still pass without replaying a seed.**
  b952a93 already made the `Fuzz regression` CI step a real gate (it exits 1 on
  a missing or empty `fuzz/regression/` instead of the old
  `cargo test … 2>/dev/null || echo …`, which reported green for the #54 seeds
  on the same push whose fuzz targets crashed on them). One vacuity path
  remained on the harness side: `zenutils_fuzz::RegressionSuite` documents a
  missing or empty seed directory as a clean no-op, and the workflow's
  `ls fuzz/regression/ | wc -l` counts `README.md` and dotfiles that the suite
  skips — so a corpus stripped to its README passed both checks and replayed
  nothing. The risk is sharper here than in sibling codecs because the corpus
  lives one level above the crate, outside `CARGO_MANIFEST_DIR`. The harness now
  asserts at least `MIN_SEEDS` (21) *replayable* files, counted with the suite's
  own filters. Mutation-verified: removing the corpus and injecting a panic into
  the `decode` target each fail the test with exit code 101.
- **Both `test-wasm` legs went red the moment that assertion landed**, because
  the harness located the corpus with `CARGO_MANIFEST_DIR.join("..")` — a path
  carrying a literal `..` component, which WASI's sandboxed resolution refuses
  to traverse even though `.cargo/wasm-runner.sh` preopens both the crate
  directory and the repo root. `read_dir` therefore failed under wasmtime, the
  replayable count came back 0, and the new assertion fired. It had been hiding
  behind the old no-op-on-missing-directory behaviour: the wasm legs were
  replaying nothing and reporting green. The harness now uses `.parent()`,
  which names the same directory with no `..` to resolve, so the corpus is
  reachable on wasm too. Verified under wasmtime locally on both legs
  (`--no-default-features` and `--features wasm128`): all 21 seeds replay.

### Added

- **A cross-tier differential decode gate** (`tests/cross_tier_determinism.rs`).
  `archmage::testing::for_each_token_permutation` disables SIMD tokens
  process-wide, so `summon()` falls through and the *whole decoder* — not one
  kernel — runs on the lower tier; the test decodes a fixture grid under every
  permutation the host supports and compares the results. Fixtures whose decode
  carries no float quantization onto the output path must be **byte-identical**
  when decoded with dithering off; float-pipeline
  fixtures are held to an envelope of at most 1 per byte and 0.1% of the image,
  because `mul_add` is fused on avx/avx512/neon and unfused on sse42/wasm128
  (which have no FMA instruction), so those tiers round once versus twice. That
  envelope is two orders of magnitude tighter than any of the defects above,
  all of which move a byte between 0 and 255. Per-sample rules the envelope
  cannot see are pinned exactly instead by `round_store_u8_contract` in
  zenjxl-decoder-simd, which runs on every tier via
  `test_all_instruction_sets!`. Measured on aarch64 across 25 permutations:
  worst per-byte difference 1, worst differing fraction 0.0122%. Coverage is
  bounded by the host — an aarch64 machine cannot execute the x86 tiers — so
  the test prints the permutations it actually ran, making an absent tier
  visible in the log rather than silently green (61dac52).

  Two follow-up corrections, both prompted by CI on platforms this host cannot
  run (b04c2a1). Exact-set membership is now **measured with `jxlinspect`, not
  inferred from filenames**: the first version classified
  `squeeze_empty_residual.jxl` as lossless because of its name, and
  windows-11-arm failed with "1 of 16384 bytes differ (max 1)" while the same
  fixture was byte-equal on aarch64 macOS — it is in fact `64x64, lossy`, as is
  `squeeze_alpha.jxl`. Both moved to the envelope, where a 1-ULP
  fused-vs-unfused difference that only crosses a rounding boundary on some
  codegen belongs. The criterion is now stated rather than assumed — 8-bit
  output (`k / 255.0 * 255.0` round-trips exactly in f32, which is untrue at 10
  or 16 bits), an sRGB transfer function, and no spline or noise synthesis —
  which also moves `hdr_pq_test`, `hdr_hlg_test`, `pq_gradient`, `splines` and
  `spline_on_first_frame` out despite their headers reading
  "(possibly) lossless". `squeeze_edge.jxl` stays, and is the fixture that
  matters for the squeeze inverse: 513x513 is not a multiple of any lane count,
  so the decode crosses from the vector body into the scalar tail. Second, the
  tier comparison now forces `parallel = false`, since it defaults to `true`
  with the `threads` feature and rayon scheduling varies with core count —
  otherwise a scheduling difference could be reported as a tier difference; a
  companion `decode_is_deterministic_when_parallel` covers the parallel path by
  holding the tier fixed and varying nothing.

- **The root fuzz package now builds with `all-simd`**, so fuzzing exercises the
  SIMD tiers instead of only the scalar one. It pinned `default-features =
  false`, which is the structural reason two of the three CPU-dependent decode
  defects above were unreachable by fuzzing: the avx512 negative-sample
  inversion and the squeeze inverse's vector body were not compiled into the
  fuzz binaries at all. Tier selection is at runtime, so a fuzzing host now
  reaches whatever it supports — on the x86 CI runners the avx/avx512 path. The
  nested `zenjxl-decoder/fuzz/` package already used default features and needed
  no change. All 21 committed regression seeds replay clean with SIMD enabled.

- **A `Fuzz targets compile` gate covering all six fuzz targets, on push and on
  pull requests.** Three of them were reached by no workflow at all:
  `decode_parallel` was missing from the `Fuzz` campaign matrix, and the whole
  nested `zenjxl-decoder/fuzz/` package (`decode`, `decode_header`) is invisible
  to every root-level cargo command because it declares its own `[workspace]`
  table — `.clusterfuzzlite/build.sh` globs the *root* `fuzz/fuzz_targets/`, so
  it does not cover it either. They could rot indefinitely with every workflow
  green. The gate runs `cargo check --all-targets` per workspace on stable,
  which catches the same class of rot as `cargo fuzz build` (type and
  resolution errors) without nightly or per-target sanitizer codegen. All six
  still compiled when it was added. `decode_parallel` is now also in the
  campaign matrix, and the campaign itself is skipped on pull requests
  (`if: github.event_name != 'pull_request'`) so the new PR trigger buys the
  cheap gate without an hour of sanitizer builds per PR.

- **Animation frame seeking** (issue #11; port of upstream jxl-rs #678 +
  #702, adapted to this fork's parser). `JxlDecoder::scanned_frames()`
  returns a `VisibleFrameInfo` per visible frame parsed so far (index,
  duration, file offset, `is_keyframe`, name) with a
  `VisibleFrameSeekTarget`; `JxlDecoder<WithImageInfo>::start_new_frame`
  repositions the decoder so that feeding raw file bytes from
  `target.decode_start_file_offset` yields that frame, passing over the
  intermediate visible frames internally. Seeks are bit-exact with a
  from-start decode: the decode start is resolved through blending-source,
  patch-dictionary and LF-frame dependencies; the container box state
  (`jxlc`/`jxlp`, frames straddling box boundaries included) is restored;
  and the frame counters that seed the noise RNG are restored. New
  `JxlDecoderOptions::scan_frames_only` / `with_scan_frames_only` parse only
  frame headers and TOCs to build the seek table without decoding reference
  frames. Containers with out-of-order `jxlp` boxes report
  `seek_target: None`. The inner API's `start_new_frame` returns the new
  `Error::SeekBeforeImageInfo` (`ErrorClass::OutputConfiguration`) when
  called before the image header. Covered by a corpus-wide sweep that seeks
  to every visible frame of every fixture.

### Fixed

- **Scalar `i32` `abs` panicked on `i32::MIN`** (#54). A corrupt stream can
  leave `i32::MIN` in a quantized coefficient; `adjust_quant_bias` takes its
  `abs`, and the scalar SIMD tier used `i32::abs`, which panics under overflow
  checks (the fuzz build) where every vector tier wraps. Now `wrapping_abs`,
  matching the vector tiers.
- **Output buffers are charged against `max_memory_bytes`** (#55). The
  convenience decoders (`decode`, `decode_with`, `reconstruct_jpeg*`) allocate
  cache-line-padded output rows, so a stream that passes `max_pixels` could
  still request far more than its pixel count suggests — the farm's seed is a
  1×235875981 image (under the 256 MP default) whose padded RGB output is
  64 B/row = 15.1 GB, and on an overcommitting kernel the allocation
  "succeeded" and the prefault OOM-killed the process. The padded footprint
  is now reserved from the decoder's memory tracker (shared with its internal
  buffers) before the allocation; the seed fails with
  `LimitExceeded { resource: "memory_bytes" }`.
- **`ready_image_area` underflowed on an empty strip.** The low-memory
  pipeline's readiness rectangle was built with `bool::then_some`, whose
  argument is evaluated before the condition, so a strip with `x1 <= x0`
  (a 3-px last group column inside a 7-px border) computed `x1 - x0` anyway:
  a panic under overflow checks (debug builds, the fuzz build) and a
  wrapped-then-discarded value in release. Now `then` (lazy), and the
  remaining border subtractions saturate. Found by the `permuted_toc_tiny_
  last_column_*` tests, which only ever ran in release on CI.
- **JPEG-reconstruction writer could silently emit corrupt streams** (issue
  #56, 2026-08-26 ultracode sweep, adversarially verified): `write_huffman`
  wrote ZERO bits for any symbol missing from its DHT (the zenjpeg #194/#196
  mechanism); the progressive scan writer substituted the DC table for a
  missing AC table and silently skipped pending EOB-run flushes; and
  `from_counts_values` tolerated DHT counts/values mismatches, leaving
  symbols codeless. Now: a sticky missing-symbol flag on the bit writer is
  checked at every scan flush (loud `InvalidJbrd` instead of corrupt bytes),
  progressive scans require the table class they declare, EOB-run flushes
  error on a missing table, and counts/values consistency is validated at
  table build. All 8 byte-exact reconstruction fixtures still pass.
- `take_jpeg_reconstruction` no longer swallows serialization errors with
  `.ok()` — a failed reconstruction is recorded and exposed via the new
  `jpeg_reconstruction_error()` accessor, distinguishable from "no jbrd box".
- Corpus-backed feature tests no longer skip silently when codec-corpus is
  absent: they fail loudly unless `ZENJXL_ALLOW_MISSING_CORPUS=1` is set
  explicitly (CI sets it, visibly, in ci.yml).

The next release is **0.4.0**: the entries under "Changed (BREAKING)" and
"Removed (BREAKING)" ship together as one breaking batch.

### Changed (BREAKING)
- `JxlDecoder::flush_pixels` (both states) returns `Result<bool>`: `true`
  when new pixels were written to the buffers since the previous call,
  `false` for a no-op flush, so callers can skip redundant post-processing
  (upstream jxl-rs d782c19, #755). A flush before any frame data now also
  returns `false` without running the parser -- previously it silently
  derailed header staging so the next `process()` call consumed the rest of
  the file before returning frame info. (617ee66)
- The `JxlPixelFormat::rgba8/rgba16/rgba_f16/rgba_f32` constructors declare
  their `num_extra_channels` extra channels as not-requested (`None`)
  instead of requesting every extra channel plane at the color format
  (upstream c5528f6, #767). Callers that passed `num_extra_channels > 0`
  and supplied per-channel buffers now pass only the color buffer, or fill
  the `extra_channel_format` entries explicitly. (c0c66bd)
- `JxlColorType::add_alpha` returns `Option<JxlColorType>` (`None` for the
  new `Cmyk` variant, which has no alpha-carrying counterpart) (upstream
  29a3e5e, #891). (b9c8ca5)
- `OwnedRawImage::prefault_parallel` (a `threads`-only helper) takes the
  decoder's `parallel` flag and no-ops when the decode is sequential or the
  buffer is small, instead of always pre-faulting. (87f4392)
- The public `Error` is now wrapped as [`whereat::At<Error>`](https://docs.rs/whereat)
  via the `Result` alias, so decode errors carry a `file:line` source location for
  server-side stack traces. Match on the cause with `e.error()` (borrow) or
  `e.decompose().0` (owned); `into_inner()` is deprecated. The low-level
  bitstream/entropy hot path keeps a bare `Error` (no `At<>` in inner loops);
  only the frame/render/api layer carries the wrapper. (#28)

### Removed (BREAKING)
- The no-op `JxlDecoderOptions::progressive_mode` field,
  `with_progressive_mode()` and the `JxlProgressiveMode` enum: nothing read
  the option anywhere in the decode path, so setting it silently did
  nothing (upstream 0977812, #880). The functional `reject_progressive`
  option is unchanged. (f273a2a)
- `JxlBitstreamInput::unconsume` (default method and the `BufReader` impl):
  the decoder never called it. Callers that used it to find where the file
  ended use the new `file_length()` instead (upstream 1cc9ab7, #820).
  (f8039de)

### Added
- CMYK interleaved output: `JxlColorType::Cmyk` + `JxlPixelFormat::cmyk8()`
  decode C, M, Y from the color channels and K from the image's Black extra
  channel; requesting it on a non-CMYK image fails with the new
  `Error::NotCmyk` (upstream 29a3e5e, #891). (b9c8ca5)
- `JxlDecoder::file_length()`: total length of the JPEG XL file once
  decoding finishes, so callers that over-fed the decoder can tell which
  trailing bytes were not part of the file (upstream 1cc9ab7, #820).
  (f8039de)
- `JxlPixelFormat::rgb8/rgb16/rgb_f16/rgb_f32` constructors (upstream
  c5528f6, #767). (c0c66bd)
- `JxlOutputBuffer::new_with_stride`: hand the decoder a sub-rectangle of a
  larger buffer, rows `byte_stride` apart; safe implementation (no
  `MaybeUninit`, unlike upstream e883140). (b45bec6)

### Fixed
- Premultiplied RGBA output from a grayscale image stays gray: the
  premultiply stage covered only the single source color channel, leaving
  the expanded G/B copies straight (upstream 775837f, #903). (6dc641c)
- Blending now rejects reference frames saved before color transforms
  (new `Error::BlendingPreColorTransform`) and asserts full-image-size
  references, preventing out-of-bounds row access (upstream 2b4a36a,
  #902). (da205ec)
- `flush_pixels` with color output ignored no longer hits a latent unwrap
  panic in the LF-preview path (`color_data_format` was unwrapped before
  the `None` check guarding it). (617ee66)
- `tests/testdata/jxlrs-865/issue865_large_toc.jxl` (5249x5377, 462-group
  TOC) with a streamed-equals-one-shot regression test (jxl-rs #865's
  incremental-parser stall; 64-bit only — the decode is 28 MP). (325cf98)
- Regression fixtures and tests ported from jxl-rs: `issue728_minimal.jxl`,
  `strategic_solid_blue.jxl` (#728/#734 squeeze boundaries), `issue772_blendbug.jxl`
  (#772 clipped blending), the #875 `multiple_lf_420` LF-group colour check and
  the e12b99b `squeeze_empty_residual` chunk-1..16 flush check; a
  `decode_parallel` fuzz target (threads on, chunked input, 3-thread pool).
  (0038580) A 600x600 4:2:2 JPEG-transcoded fixture (3x3 groups) with a
  byte-exact reconstruction test. (6051605)
- CI: the `threads` feature is tested without `allow-unsafe`, and the crate's
  tests run on `wasm32-wasip1` under wasmtime with and without `simd128`
  (`.cargo/config.toml` carries the runner and target features). (8b58423,
  6051605, and the wasm job commit)
- **Blue-noise dithering of 8-bit output, on by default** (port of libjxl
  `stage_write.cc` / jxl-rs #841). Every `U8` conversion stage (plain, fused
  XYB→sRGB/gamma, fused linear→sRGB) adds the 32×32 pattern, indexed by
  absolute position and channel, before rounding. `djxl`'s RGB8 output is now
  reproduced byte-for-byte on `with_preview.jxl` (was 25 % of samples ±1);
  lossless 8-bit content is unchanged; `U16`/`F16`/`F32` output is never
  dithered. New `JxlDecoderOptions::dither_u8` / `with_dither_u8(false)`
  restores plain rounding. The u8-vs-f32 consistency tests' tolerance moves
  from 0.003 to 0.004 = (0.5 + 0.49219) / 255, the derived bound.
- **Out-of-order `jxlp` boxes** (ISO/IEC 18181-2 with `ftyp` minor version 1,
  as written by `cjxl --output_mode 2`; port of jxl-rs #752/#777). A `jxlp`
  box ahead of the next expected index is buffered (bounded: 1024 boxes,
  growth in 64 KB steps with fallible allocation, no upfront allocation of
  the declared size) and spliced in when its turn comes. Version-0 files
  with out-of-order boxes, duplicate indices and size-0 out-of-order boxes
  are rejected as before. `ftyp` is now required to be the second box,
  exactly once, as in libjxl.
- `Error::kind() -> ErrorClass` and the `ErrorClass` enum (re-exported from
  `zenjxl_decoder::api`): a coarse, best-effort classification of decode errors —
  `InvalidBitstream` / `LimitExceeded` / `OutOfMemory` / `Cancelled` / `Io` /
  `OutputConfiguration` / `Unsupported` / `Internal`, each documented with its
  typical HTTP status — so a server can bucket a failure without matching all
  ~130 `Error` variants. Also documented the `Error::LimitExceeded.resource`
  string values (`pixels`, `memory_bytes`, `icc_size`, `icc_amplification`,
  `extra_channels`, `reference_frames`). Additive — `ErrorClass` is
  `#[non_exhaustive]`. (#29, 2c7a68d)
- Re-exported `enough::StopReason` from `zenjxl_decoder::api` (alongside the
  existing `Stop` / `Unstoppable`) so a hand-rolled cancellation `impl Stop` needs
  no direct `enough` / `almost-enough` dependency; the README now shows that
  dep-free `AtomicBool` pattern. (#29, 2c7a68d)
- `examples/heaptrack_decode.rs`: a reusable heaptrack/valgrind harness that
  decodes a JXL file from bytes via `zenjxl_decoder::decode(..)` in a loop, for
  profiling heap-allocation behaviour. Defaults to the committed
  `resources/test/bike_web_q85.jxl` (2048×2560, 5.24 MP VarDCT photo) decoded 8×;
  a path + iteration count can be passed. Driven by `just heaptrack-decode`.
  Profiled result: heap **size** and leaks are healthy — peak 44.7 MiB (~2.1× the
  20 MiB RGBA8 output, O(image)), and leaked allocations are pinned at 31 across
  2/8/16 iterations (one-time statics, no per-decode growth). Allocation **count**
  is elevated at ~29,300/decode, dominated by ~17,400/decode of small transient
  zeroed scratch in the per-group `frame::group::dequant_and_transform_to_pixels`
  path, including the per-pass `PassInfo::num_nzeros: [Image<u32>; 3]` already
  flagged with an in-code `// TODO(veluca): reuse this allocation.` Tracked as a
  resource follow-up.
- `JxlDecoderOptions::reject_progressive` (default `false`): when `true`, decode
  fails with the new `Error::ProgressiveRejected` as soon as a progressive frame
  header is seen — before decoding its passes — for untrusted-input policies that
  forbid progressive content. A frame counts as progressive when its header has
  `num_passes > 1` or its frame type is `LFFrame`; `ReferenceOnly` (patch/blend
  dictionary) and `SkipProgressive` frames do not trip the gate, and the check
  applies to the first non-preview frame. Both additions are additive
  (`JxlDecoderOptions` and `Error` are `#[non_exhaustive]`); the probe
  (`JxlBasicInfo`) is unchanged — progressive is enforced during decode, not
  surfaced on the probe. (966f9c5)

### Changed
- Modular decode no longer caches one rendered tile per channel per group
  for the whole frame (78 MB on a 2333x2333 lossless decode): the
  low-memory pipeline's group-sized scratch cache is capped at zero for
  modular frames, as upstream's #812 does. (15f332f)
- **Multi-threaded decode is faster on modular palette images and on
  VarDCT images with many groups.** The inverse palette transform runs one
  channel per rayon thread (`delta_palette` 43.6 → 67 MP/s at 12 threads,
  `issue648_palette0` 21 → 24 MP/s); the one-shot parallel render gives every
  group a disjoint tile of the output and writes it directly instead of
  rendering overlapping rectangles into temporaries and copying the whole
  image back on one thread (`multiple_lf_420` 540 → 670 MP/s, `portrait_4k_q90`
  347 → 378 MP/s at 12 threads); the first group on each thread no longer
  allocates its scratch buffers while holding the buffer-pool mutex; render
  contexts are pooled across rayon leaves instead of created per leaf. The
  per-row input-copy plan of the low-memory pipeline is computed once per
  rendered rectangle (jxl-rs f94cc26). Output is pixel-identical. (728fc6e,
  d588318, 9b1998f)
- Single-threaded VarDCT decode is 4–12 % faster from the single-symbol
  entropy fast path (jxl-rs #787/#817), a config-420 reader specialisation and
  allocation-free, vectorised, in-place blending (jxl-rs #709/#818/#821);
  against upstream jxl-rs 0.6 every fixture of the audit set is within 8 % at
  12 threads and all but `multiple_lf_420` (1.09×) at 1 thread. (cddf926)
- The parallel-vs-sequential decode path is chosen per frame (more than one
  group) instead of per `process` call, so a streamed decode no longer
  switches paths mid-frame. (0038580)
- VarDCT decode no longer makes per-block / per-pass scratch heap allocations on
  the `frame::group::dequant_and_transform_to_pixels` hot path. The AFV transform
  arms allocated three transient `Vec<f32>`s per AFV block (a 64-element input
  snapshot plus two `4x4`/`4x8` scratch buffers); these are now fixed-size stack
  arrays, matching the sibling IDENTITY/DCT2X2/DCT4X4 arms. The per-pass
  `PassInfo::num_nzeros: [Image<u32>; 3]` non-zero-count maps (which carried an
  in-code `// TODO(veluca): reuse this allocation.`) are now reused across groups
  via a pool on `VarDctBuffers`, resized only on a dimension change and re-zeroed
  per group. Measured on `resources/test/bike_web_q85.jxl` (5.24 MP) via
  `examples/heaptrack_decode`: total allocations dropped from ~29,300/decode to
  ~3,200/decode (234,333 → 25,805 over 8 iterations, −89%), with the
  `dequant_and_transform_to_pixels` allocation site eliminated entirely. Peak heap
  (44.7 MiB) and leaked allocations (31, all one-time statics) are unchanged.
  Pure internal allocation-lifetime change, gated on a bit-identical-output
  regression test (`tests/decode_bit_identical.rs`) — no public API or output
  change. (closes #40)
- `get_distinct_slices` (the per-row multi-slice helper on the modular /
  VarDCT-LF decode hot path) no longer heap-allocates a transient `Vec` on every
  call. Because the slice count `S` is a const generic, the scratch now lives on
  the stack (`[Option<&mut [u8]>; S]`, filled by original index so no sort-back is
  needed). It was called once per scanline per modular channel, so this removes
  ~16k short-lived allocations from a 64 MP decode (the count scaled linearly with
  image area). Pure internal change — no public API or output change. (#35)

### Fixed
- **Two panics when streaming input in pieces with `parallel = true`** (the
  default): (1) in the parallel render's fragment path, an incremental batch
  whose group rectangles overlapped in x but not in y cut a fragment narrower
  than its rectangle (`JxlOutputBuffer::rect` assertion; `cafe_web_q80.jxl`
  in 30000-byte chunks); band splitting now requires disjoint columns too.
  (2) with a flush after every chunk, the flush re-sent the extra channels of
  every group the modular dry run reported, including not-yet-decoded
  neighbours of a newly decoded group, and unwrapped their missing data
  (`tirr_photo.jxl` in 30000-byte chunks); buffers without data are skipped.
  Every fixture now also decodes pixel-identically to one-shot in 4096- and
  30000-byte chunks with flushes, and in parallel with 777/4096/30000-byte
  chunks and with 1/2/3/5/8-thread pools. (0038580)
- Decoding a malformed Squeeze transform whose output tile has both dimensions 0
  no longer panics (`called Option::unwrap() on a None value` at
  `frame/modular/transforms/squeeze.rs:663`). `with_buffers` legitimately skips a
  both-dimensions-0 output tile (a degenerate channel that still participates in
  numbering but carries no pixels), so `do_vsqueeze_step` / `do_hsqueeze_step` can
  receive an empty `buffers` slice; both now no-op in that case (via `let Some(out)
  = buffers.first_mut() else { return }`) instead of `.first_mut().unwrap()`,
  matching the existing zero-size early-returns immediately below. There is no
  output to write, so decode output for valid inputs is unchanged. Fuzz-found via
  target `decode` (#46); repro gated in `fuzz/regression/`.
- Decoding a malformed modular stream whose reference-channel pixel reaches
  `i32::MIN` no longer panics (`attempt to negate with overflow` inside
  `num-traits`). `precompute_references` (the MA-tree decision-property
  precomputation in `frame/modular/decode/common.rs`) now takes `wrapping_abs()`
  of the reference value instead of the panicking `num_traits::abs`, matching the
  sibling neighbour properties in `frame/modular/tree.rs` and the C++ reference's
  `std::abs(int32_t)` wrap. This is a decision property, not a pixel output, so
  output for valid inputs is unchanged. Fuzz-found via targets `decode` (#45) and
  `decode_with_limits` (#44); both repros gated in `fuzz/regression/`. (c239d544)
- The published crate's test suite now builds. `resources/test/` (~31 MB, 155
  fixtures) is not packaged (it exceeds the crates.io size budget), so the tests
  no longer reference it at compile time: the 17 `include_bytes!` fixtures and the
  `for_each_test_file!` proc macro (which read the directory during macro
  expansion and panicked when it was absent — `cargo test` on the published crate
  failed to compile at all) are replaced by runtime resolution. Fixtures resolve
  to the local checkout when present (the normal dev/CI case, no network),
  otherwise download on demand via the `codec-corpus` crate; resolution panics
  loudly on failure, so there is no silent skip. Helpers live in
  `crate::util::test` (`fixture_dir`/`fixture_path`/`fixture_bytes`/
  `all_jxl_fixtures`); the four `for_each_test_file!` sweeps became runtime sweeps
  that report every failing fixture together. The integration test
  (`decode_bit_identical.rs`) and decode benchmark resolve fixtures the same way.
  (#8)
- docs: the `JxlDecoderLimits::max_pixels` field doc stated the default was
  `2^30 (~1 billion)`, but the actual default is `1 << 28` (~256 megapixels).
  Corrected the doc comment to match the code and noted that `restrictive()`
  lowers it to a 120-megapixel house cap. Doc-only; no behavior change. Found in
  a production-readiness audit.
- docs(readme): the Basic decode example now shows a real decode-to-pixels flow
  (`width`/`height`/`data`/`channels` off `JxlImage`) instead of stopping short,
  and documents the output format (8-bit interleaved RGBA8/GrayAlpha8, straight
  alpha by default). Corrected the limits table (default `max_pixels` is 2^28
  ~256M not 2^30; `restrictive()` `max_pixels` is 120M not 100M; default
  `max_memory_bytes` is 4 GB / 2 GB-on-32-bit, not `None`) and the cancellation
  section (the `stop` field is `Arc<dyn enough::Stop>` defaulting to the
  `enough::Unstoppable` no-op; `almost_enough::Stopper` at `0.4` is how you
  actually cancel). Found by an insulated external-developer usability test.

### Docs
- Overhauled the repo-root `README.md` to the zen crate conventions (badge row
  gains lib.rs and drops the `branch=` pin; adds `## Quick start`; documents
  `Error::kind`/`ErrorClass` for server-side bucketing), fixed the
  `JxlDecoderOptions` examples to the `#[non_exhaustive]` builder form (the old
  `JxlDecoderOptions { .. }` struct-literal examples could not compile in a
  downstream crate), corrected the stale `JxlImageInfo::animation` reference to
  `JxlBasicInfo::animation`, and split the crates.io README into a generated,
  CI-badge-only `README.crates.md` (`readme = "../README.crates.md"`).

## [0.3.10] - 2026-06-11

### Added
- `JxlDecoderOptions::adjust_orientation` is now load-bearing in the render
  pipeline. When `true` (the default, "Correct") the stored EXIF/container
  orientation is baked into the output as before; when `false` ("Preserve") the
  bake is skipped — pixels are emitted in their stored (coded) orientation and
  dimensions. `JxlBasicInfo` gains `coded_size` (stored dims, never transposed)
  and `intrinsic_orientation` (the stored orientation tag, regardless of mode);
  `JxlBasicInfo::size` and `JxlBasicInfo::orientation` now describe the *emitted*
  pixels (display dims + `Identity` in Correct mode; coded dims + the stored
  orientation in Preserve mode). All four are additive (`JxlBasicInfo` is
  `#[non_exhaustive]`). (cf97249)
- Versioned public-API surface snapshots at `docs/public-api/<crate>.txt` (zenjxl-decoder, zenjxl-decoder-macros, zenjxl-decoder-simd), regenerated by `zenjxl-decoder/tests/public_api_doc.rs` on every `cargo test` (`ZEN_API_DOC=check` verifies in CI, `=off` skips); justfile `api-doc` / `api-doc-check` recipes.

### Fixed
- `decode()` / `decode_with()` now capture `jhgm` (gain map), `Exif`, and
  `xml ` boxes that follow the codestream — the layout jxl-encoder's
  `append_gain_map_bundle` writes — instead of returning `None` for them
  (closes #20). After the last frame, the decoder drains the remaining
  container boxes; the low-level API gets the same drain via one extra
  `process()` call. Boxes trailing a multi-frame (animation) codestream are
  still only reachable through the low-level API.
- Spot-colour / extra-channel images (e.g. the `spot.jxl` conformance image) no
  longer panic in the low-memory render pipeline. The user-output save-stage
  output-buffer index was the absolute extra-channel index (`1 + i`), but the
  caller's output-buffer list is *packed* — one slot per *requested* channel,
  with `None` extra channels (e.g. alpha folded into interleaved colour output)
  skipped. A `None` gap before a requested extra channel made the absolute index
  overshoot the packed buffer Vec, panicking ("index out of bounds") in
  `check_buffer_sizes`. The index is now a packed running counter matching
  `num_api_buffers` (and upstream jxl-rs's `save_idx`). Regression test:
  `zenjxl-decoder-cli/tests/spot_low_memory_regression.rs`. (fork-introduced
  regression; upstream jxl-rs is unaffected)

## [0.3.9] - 2026-06-09

### Added
- `JxlDecoder::vardct_quantizer()` and the `VardctQuantizer` type
  (`global_scale`, `quant_lf`, `inv_global_scale()`): recover the first regular
  VarDCT frame's quantizer from a decoded JXL, so callers can estimate the lossy
  encode quality without re-parsing the bitstream. Captured when the first
  regular frame's `LfGlobal` is decoded; `None` for Modular (lossless) images
  and LF/preview helper frames (d218116).
- `reconstruct_jpeg` / `reconstruct_jpeg_with` (`jpeg` feature): one-shot
  pure-Rust reconstruction of the original JPEG bytes from a JXL with a JBRD
  box — the in-crate equivalent of `djxl --reconstruct_jpeg` (e3011867).

### Fixed
- **JPEG (JBRD) reconstruction byte-exactness**, found via a cross-crate
  round-trip conformance gate:
  - Progressive (SOF2) JPEGs now reconstruct byte-for-byte (spectral selection,
    successive approximation, EOB-run coding, reset_points/extra_zero_runs);
    previously a baseline stream was written for progressive scans (e17a482e).
  - Grayscale reconstructed an all-zero (truncated) image — the lone luma
    component was read from JXL channel 0 instead of the Y channel (1) (e3011867).
  - SOF1 (extended-sequential) and SOF2 SOF markers were silently dropped from
    the marker stream; now emitted (e3011867).
  - Restart (DRI) markers were never emitted (driven by the progressive-only
    reset_points list); now interval-driven in the baseline writer (e3011867).
  - Progressive Huffman tables are now tracked through marker order (a
    progressive JPEG redefines table slots per scan) (e17a482e).
  - EXIF / XMP / ICC metadata round-trips byte-exact (1d1ea27e, closes #19):
    `brob` (brotli-compressed) `Exif`/`xml ` boxes are decompressed and
    re-stitched into their APPn markers, and the chunked ICC profile is
    recovered from the codestream color encoding and re-split into
    `ICC_PROFILE` APP2 markers. Reconstruction defers metadata stitching to
    `take_jpeg_reconstruction` (container boxes follow the codestream).
  - Trailing metadata boxes are now drained for **large** codestreams too. The
    `Exif`/`xml `/`brob` boxes follow the frame's codestream; for small files
    they were incidentally read during frame decode, but a codestream larger
    than the read-ahead window left them unparsed, so EXIF/XMP (incl. the EXIF
    `Orientation` / "rotflip" tag) and chunked ICC were silently dropped from
    the reconstruction. `codestream_parser` now drives the box parser through
    the trailing boxes when a JBRD reconstruction is pending (gated on
    `jpeg_recon`, so non-JPEG decodes are unaffected).

### Changed
- `zenjxl-decoder/tests/fuzz_regression.rs` now uses the shared
  `zen-fuzz-regress` test-helper crate (DEDUP-J2). Per-target payloads
  preserved verbatim; only the walk-dir + read-bytes + dispatch
  scaffolding moved out. Same `../fuzz/regression/` seed path, same
  three targets (`decode`, `decode_with_limits`, `probe`).

### Added

- `zenjxl-decoder/tests/fuzz_regression.rs` regression-harness template
  ported from zenwebp (DEDUP-J). Walks the top-level `fuzz/regression/`
  directory and runs every seed through `decode`, `decode_with` (with
  the restrictive limits the fuzz target uses), and `read_header` on
  the stable toolchain — no nightly required. Drop minimized crash
  files into `fuzz/regression/` to gate future regressions of fixed
  bugs.

## [0.3.8] - 2026-04-17

### Added

- **`basic_info()` embedded-profile guard** -- Ported from jxl-rs #745; hides `basic_info()` until the embedded ICC/color-profile box is parsed, preventing callers from observing partial image-info state. Adds integration tests using `cmyk_layers.jxl` and `basic.jxl` (fa4400f, 470a6f4).
- **Shared `apply_decoder_options()` helper** -- Routes both the primary `DecoderState` creation and the preview-recovery recreation through one helper so the two sites cannot drift (5bb4632).
- **Chunked drip-decode animation stress test** -- Feeds `animation_newtons_cradle.jxl` through `JxlDecoderInner::process` in 1 KiB chunks and asserts no error or panic at any boundary, mirroring the Chrome integration repro from jxl-rs #743 (5bb4632).
- **EC-upsampling-after-dim_shift regression tests** -- Ports the test harness from jxl-rs #741 (negative, positive, and real-file cases) so future refactors cannot silently regress the existing `check()` validation (4c7cbcc, e96e434).

### Changed

- **Decoder options preserved across preview-frame recovery** -- When a JXL file contains a preview frame, `sections::handle_frame_finalized` previously recreated `DecoderState` for the main frame while copying only four of nine fields. `high_precision`, `premultiply_output`, `parallel`, `memory_tracker`, and `embedded_color_profile` silently reverted to defaults. Ported as the independent subset of upstream jxl-rs #743 that does not depend on the animation seek API in #678 (5bb4632, 3e369a1).
- **Blanket `#![allow(dead_code, unused_imports)]` removed from `lib.rs`** -- Dead code is now surfaced and handled individually instead of suppressed crate-wide; removed 9 unused imports and added targeted `#[cfg(test)]` / `#[allow(dead_code)]` with comments where items are intentionally kept (c84147f, b83639d, PR #13).
- **Clippy passes under `-D warnings` for both `--all-features` and `--no-default-features`** -- Added `#[cfg(feature = ...)]` gates and targeted `#[allow(dead_code)]` on thread-only and jpeg-only items; removed unused imports; suppressed `field_reassign_with_default` where the `#[non_exhaustive]` struct-literal rewrite is not available to external callers (265dc07, b1b8e39, a385d59, cd6d5fa).

### Fixed

- **Memory budget bypass on preview-bearing files** -- A caller asking for a restrictive `max_memory_bytes` saw the budget enforced on the preview frame but silently dropped to unlimited for the main frame; `parallel=false` was flipped back on; CMYK ICC detection misfired (5bb4632).
- **ClusterFuzzLite build paths** -- `build.sh` referenced the upstream `jxl-rs/jxl/` subdirectory layout; corrected to the fork's root workspace and `zenjxl-decoder/resources/test/` location (1a49043).
- **Benchmark test image paths** -- Updated from the upstream `jxl/resources/test/` prefix to `zenjxl-decoder/resources/test/` (3f18f86).
- **Missing copyright headers** -- Added headers to fuzz targets, the nightly fuzz workflow, and the fuzz script, caught by the `source_checks` CI job (e91149d).

### Dependencies

- Bumped `rand` 0.10.0 -> 0.10.1 (ee04a49).

### Docs

- README now lists `wasm128` under the all-SIMD row and per-ISA row (already wired in `Cargo.toml` and the `-simd` crate) (5055dae).
- Added crate-level rustdoc to `zenjxl-decoder/src/lib.rs` describing the fork, entry points, SIMD dispatch, safety posture, and feature flags, with credit to upstream libjxl/jxl-rs and libjxl under BSD-3-Clause (5055dae).
- Backfilled `[0.3.6]` and `[0.3.7]` CHANGELOG sections from git log (5055dae).

## [0.3.7] - 2026-04-10

### Fixed

- **i686 address space exhaustion in test suite** -- Test suite ran out of 32-bit virtual address space under parallel execution (d4b1167).
- **32-bit memory limit** -- Raised the default 32-bit memory limit to 2 GB so correctness tests fit on i686 (ce5b5e0).
- **Large-image tests gated on 64-bit** -- Tests that require >2 GB address space are now excluded from 32-bit targets (ee65a6f).
- **slow_probe_regression timing** -- Raised threshold from 5 ms to 10 ms to stabilise CI against loaded runners (4ea9ba5).
- **Memory limit disabled in correctness tests** -- Correctness tests no longer trip the default cap on large conformance images (727f00c).

## [0.3.6] - 2026-04-10

### Added

- **cargo-fuzz infrastructure** -- Three fuzz targets and a JXL format dictionary for continuous fuzzing (d9cfa74).
- **Nightly fuzz workflow** -- 60-second fuzz run on every push, 5-minute run nightly (8086be0).
- **BitReader panic regression seeds** -- Captured regression seeds for the `BitReader::new_padded` panic fixed in 0.3.5 (c5460e2).
- **Minimized OOM regression seed** -- 781-byte seed reproducing the crafted-header OOM fixed in 0.3.1 (91cc64d).

### Changed

- **Default `max_memory_bytes` lowered to 4 GB** -- Prevents OOM from crafted inputs in default configuration; raise explicitly via `JxlDecoderLimits` for large images (b1693bf).
- **Clippy runs once on Ubuntu** -- Removed redundant per-platform clippy jobs from CI; other platforms still run tests (c2026cd).

### Fixed

- **MemoryGuard::forget() leak** -- `MemoryGuard::forget()` leaked 32 bytes per tracked image allocation; the guard now releases its accounting slot on drop as well as on explicit forget (14e7739).
- **OOM-test clippy lint** -- Resolved `field_reassign_with_default` on the non-exhaustive options struct in the OOM regression test (8686bea).

## [0.3.5] - 2026-04-01

### Fixed

- **Huffman alphabet ratio overflow** -- Increased `ALPHABET_BITS_RATIO` from 32 to 256 to prevent false rejections of valid streams.
- **Shift overflow in property mask** -- Prevent shift overflow in `compute_used_property_mask` for large property indices.
- **Section padding for non-section buffers** -- Add `SECTION_PADDING` to non-section buffer allocation to prevent out-of-bounds reads during BitReader refill.
- **ANS alias map validation** -- Replace `assert!` in `build_alias_map` with proper error returns for malformed streams.
- **Flat tree child_id bounds checking** -- Validate child_id references in flat trees to prevent out-of-bounds access.
- **HybridUint nbits overflow** -- Track `nbits>=32` overflow in `ErrorState` for deferred reporting instead of silent corruption.
- **Memory tracker threading** -- Thread `MemoryTracker` to local modular tree decoding for accurate accounting.

### Performance

- **HybridUint OR-accumulator** -- Use OR-accumulator for overflow detection, reducing branches in the hot path.

### Dependencies

- Updated moxcms to 0.8.1 (from crates.io, with `extended_range` + `options` features).
- Updated wasm-bindgen 0.2.117, js-sys/web-sys 0.3.94.
- Updated archmage 0.9.16, zenbench 0.1.3, libc 0.2.184.

### CI

- Added full CI matrix with i686 cross-compilation, macOS Intel, windows-11-arm.
- Reduced i686 test parallelism to 2 threads for address space constraints.

## [0.3.4] - 2026-03-30

### Fixed

- **ICC amplification DoS** -- A crafted 19-byte JXL codestream could claim a huge ICC profile, causing the entropy decode loop to iterate hundreds of millions of times. Added a per-symbol progress check that detects degenerate streams producing output without consuming input (>1024:1 amplification ratio). Works correctly in both one-shot and incremental decode modes.

### Dependencies

- Updated all 55 dependencies to latest compatible versions, including `time` 0.3.46→0.3.47 (fixes GHSA-r6v5-fh4h-64xc stack exhaustion DoS), `archmage` 0.9.5→0.9.15, `zerocopy` 0.8.27→0.8.48.

## [0.3.3] - 2026-03-30

Ports 6 upstream bugfixes from libjxl/jxl-rs (March 2026) and 5 performance optimizations from PR #705. Yanks broken 0.3.2 release.

### Fixed (upstream ports)

- **vsqueeze grid boundary** (PR #731) -- Grid-based processing used wrong row when `has_tail=false` but `in_next_avg` exists, corrupting squeeze output.
- **hsqueeze grid boundary** (PR #735) -- Single-pixel-width shortcut looped over residual height instead of output height.
- **Stage pruning with shift** (PR #725, fuzzer-found) -- Pruning render pipeline stages with non-zero shift corrupted downstream channel dimensions.
- **EC upsampling validation** (PR #741) -- `check()` tested raw `ec_upsampling` instead of the effective value after `dim_shift`, allowing invalid configurations.
- **Mixed-upsampling patches** (PR #742) -- Patches with EC upsampling differing from color upsampling were silently accepted instead of rejected.
- **LF preview alpha overflow + BGR order** (PR #740) -- `1u8 << 8` overflowed to 0 (should be 255); BGR/BGRA output formats got RGB channel order.

### Performance

- **BitReader section padding** -- Append 8 zero bytes to section buffers so `refill()` always takes the fast 8-byte path, eliminating `refill_slow()` calls.
- **Property used-mask** -- Skip unused property computation per pixel. Trees typically split on 2-4 of 16 properties; the rest are now skipped.
- **HybridUint fast path** -- When `msb_in_token == 0`, simplify entropy decoding to `(1 << nbits) | bits`.
- **Inline annotation audit** -- 14 hot-path functions upgraded to `#[inline(always)]` across BitReader, ANS, Huffman, LZ77, and modular predict.
- **Blending SmallVec** -- Replace per-row `Vec` heap allocations with stack-based `SmallVec<[_; 8]>`.

Combined effect: **+4% to +16%** across VarDCT and modular images (single-threaded).

### Yanked

- **0.3.2** -- Broken release: `BitReader::new_padded` was changed to return `Result`, causing 478 of 1277 tests to fail (`SectionTooShort` on valid files).

## [0.3.1] - 2026-03-30

### Fixed

- **OOM from crafted JXL codestream headers** -- A 26-byte JXL header could request a 4.2GB allocation. Three fixes:
  - `Size::check()` now rejects `width * height > 2^30` during header parsing, before any pixel buffer allocation.
  - `alloc_zeroed_fallible` uses `try_reserve` instead of `vec![0u8; n]`, returning an error instead of aborting on allocation failure.
  - Default `max_pixels` lowered from 2^30 to 2^28 (256 megapixels).

## [0.3.0] - 2026-03-06

Initial public release of the zenjxl-decoder fork.

### Added

- **Resource limits** — `JxlDecoderLimits` API caps pixels, memory, ICC size, tree size, and more. `LimitExceeded` errors include the resource name, actual value, and limit.
- **Memory tracking** — `max_memory_bytes` budget with atomic, lock-free tracking across threads.
- **Fallible allocation** — All significant allocations return `TryReserveError` instead of panicking.
- **Cooperative cancellation** — `enough::Stop` trait integration lets any thread cancel or timeout decoding.
- **Parallel decoding** — Rayon-based parallel group decode and render via the `threads` feature.
- **CMS-based CMYK→RGB** — ICC profile-based CMYK conversion via optional `moxcms` backend (`cms` feature).
- **JPEG reconstruction** — Lossless JPEG reconstruction from JXL containers (`jpeg` feature).
- **`allow-unsafe` feature** — Opt-in `unsafe` fast paths in the main crate; safe fallbacks used by default.
- **`#![forbid(unsafe_code)]`** by default in the main `jxl` crate (without `allow-unsafe`).

### Fixed

- sRGB transfer function applied by default for XYB-encoded images (was outputting linear).
- RCT overflow panic via wrapping arithmetic on edge-case pixel values.
- Extra channel format slot allocation for all extra channels, not just the first.
- Progressive AC validation: `last_pass` must be strictly increasing.
- Extra channel bit depth: use the channel's own `bit_depth` for modular-to-f32 conversion.
- Noise seeding: separate RNG seeds per subregion for upsampled frames.
- CMYK blending order: blend in CMYK space, then CMS-convert to RGB.
