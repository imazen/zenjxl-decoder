# Decoder investigation notes

## LCMS context ownership and Send qualification — 2026-11-05

lcms2 6.2.0's Transform holds a PhantomData context marker. Passing a
ThreadContext by value to new_flags_context drops that actual context on return.
The callback now retains it, with the transform field dropped first. The
context is Send rather than Sync; a private Mutex retains ownership while
preserving the callback object's existing Send + Sync contract. No unsafe
code or public API change is required.

Compile assertions establish that Lcms2Cms and the bundled transformer are
Send + Sync. The runtime test moves an initialized real transform across four
threads, requires identical float bits at each step, and destroys it on another
thread. Both CLI configurations pass all seven tests, including the unchanged
exact whole-image CMYK golden. Candidate Memcheck reports zero errors. The
baseline identity test also reports zero errors: this work does not claim a
reproduced invalid memory access. Strict all-feature workspace Clippy passes.

Context ownership alone does not make Box<dyn JxlCms> Send. Approved PR60
now adds that separate provider trait-object bound (0458ab95), without requiring
Sync on providers. Combined main6d060afb passes the full workspace, strict
Clippy, both feature-specific API/LCMS gates and six frozen exact RGBA frames.

## Resolved: HF table accounting and trailing metadata — 2026-11-05

The parser allocated group/pass section tables before the TOC and outside its
shared memory budget, including header-only scans. Allocation now happens after
the TOC/frame setup, skips scans, checks size arithmetic, and reserves the outer
and inner table together. A guard releases the reservation on frame reset/seek.
Tests reject oversized tables before allocation, exercise competition with
other frame storage, and check release.

Unused bytes inside finite codestream boxes could hide later Exif, including
across jxlp parts. The parser now drains those bytes before continuing ordinary
metadata parsing. Tests cover 0 through 70,000 unused bytes, partitioned
codestreams, seek-relative lengths, and memory-free scanning.

PR65's blanket post-codestream recovery is excluded: it hides injected I/O
errors and accepts a duplicate ftyp that djxl v0.12 rejects. Valid metadata
draining preserves current errors; new tests retain duplicate-ftyp and invalid
jxlp rejection and injected I/O failure. Recovery of malformed optional
metadata is separate work. The seven targeted checks pass with all features
and without default features. The combined full workspace suite and six
frozen-reference RGBA frames also pass. `just inventory-behavior-check`
reproduces the focused behavior and memory checks.

## Resolved: CMS callback contracts — 2026-11-05

Current-main reproduction passes only one of eight imported callback-contract
cases. RGB/gray callback errors panic, CMYK callback errors silently take the
mathematical fallback, invalid initializer counts/channels are not consistently
rejected, and the CMYK path substitutes an sRGB destination.

Both renderer runners now use a private fallible stage entry point. Numerical
kernels retain their existing implementation. A shared validated CMS stage
propagates the callback's original error and location context, rejects malformed
initializer responses before transforming pixels, and receives the requested
destination profile. The mathematical path remains when no applicable CMS is
configured. Reflectance inputs and raw-CMYK handling retain their contracts.

All eight callback cases pass with all features; the seven applicable cases
pass without default features (the parallel case requires `threads`). They
exercise both renderers, first/later/parallel failures, initializer refusals,
channel/count validation and CMYK destination selection. All 121 renderer
checks, both CLI LCMS test configurations and the combined full workspace suite
pass. The six frozen djxl regression frames have zero differing RGBA bytes.
`just downstream-cms-check` reproduces the focused gates. No public API change,
MoxCMS interpolation change or raw-layer API is included.

## Resolved: 64-point IDCT after fused fallbacks — 2026-11-04

The deterministic `test_idct1d_64_eq_slow` exceeds its unchanged 5e-6
absolute/relative bound after scalar multiply-adds become fused. Index 32
produces -1.9676225185394287 versus -1.9676337791674359: absolute error
1.1260628007159212e-5, relative error 5.7229287921271194e-6. The full
library run passes the other 1,024 cases, with 29 existing ignores.

Retaining extra secant-coefficient bits alone does not fix it. A candidate
that retains f64 intermediates through the existing large-transform
factorization passes all eight original 1D size checks (2 through 256)
without changing assertions. Its scalar-per-lane implementation was rejected
on measured cost. The current register-based implementation reuses the
existing generated 32-point factorization through magetypes f64 vectors;
all original 1D/2D accuracy checks and all nine large-shape raw-f32 parity
checks pass on scalar/SSE4.2/AVX2. ARM64 passes 54 selected transform
checks and all 40 decoder regressions; WASI passes 31 scalar and 54 SIMD128
transform checks. The full native workspace gate passes (1,029 library cases,
29 existing ignores, integrations and doctests), with strict Clippy and
formatting. Six fixture frames retain exact frozen djxl reference bytes on
x86 and ARM64. LCMS whole-image checks pass in both CLI configurations.
The paired 64x64 AVX2 microbenchmark measures 9.4 us versus 3.8 us for the
f32 baseline; whole-decoder impact is unmeasured. No expectation is weakened.


## Resolved: custom CMYK CMS conversion — 2026-11-04

A caller-supplied CMYK CMS was bypassed unless the optional MoxCMS feature
was compiled. The dedicated stage now compiles independently of that backend
and passes the public transform contract's reflectance convention (1 = no
ink); each backend performs its own ICC normalization. The LCMS backend now
converts to percentage ink and enables black-point compensation and high
resolution precalculation, matching libjxl v0.12's transform configuration.

The existing LCMS implementation with CMYK CMS enabled differed from djxl
in 94,239 of 1,048,576 RGBA bytes, maximum difference 252. The corrected
implementation matches every byte. A strict whole-image hash regression
passes both the default CLI and no-default-feature CLI builds, without
requiring MoxCMS. The existing MoxCMS color assertions retain their expected
values and tolerance; their direct inputs now use documented reflectance.
A custom-transformer stage test checks the same contract explicitly.

## Known Bugs: remaining display parity — 2026-11-04

The same djxl v0.12 runtime produces a host-dependent green byte at
canvas (18,18) of crop frame zero: 72 on i270 and 73 on r5900xt. Rust
produces 72 on both hosts, with and without optimization. This is a
reference-portability finding, not evidence that optimization changed Rust's
pixels. Cross-host gates can use frozen v0.12 reference images with verified
input and reference SHA-256 records; comparison remains byte-exact.

The previous cross-tier gate permitted float rounding differences: optimized
x86 output differed between AVX2 and SSE/scalar in
`grayscale_patches_var_dct.jxl` (one byte) and
`squeeze_empty_residual.jxl` (two bytes), all by one code. Fused fallback
arithmetic eliminates these differences across all 40 fixture/configuration
comparisons and seven available x86 permutations. ARM64 also passes all 40
configurations over nine permutations, all 40 selected decoder regressions,
and both exact crop frames. Native scalar/SSE4.2/AVX2/NEON and WASI
scalar/wasm128 fused-operation contracts pass. The cross-tier gate requires
exact bytes throughout. The final full workspace, CMS, WASI and frozen-reference
gates pass after the large-IDCT precision repair. AVX-512 execution and
fallback performance on CPUs without FMA are unmeasured.

Optional MoxCMS CMYK display output retains whole-image differences from
LCMS-backed djxl. The CLI's supplied LCMS backend has a strict whole-image
regression and matches every byte. This report does not claim that the two
independent CMS implementations have identical interpolation results.

## Resolved: negative canvas dither coordinates — 2026-11-04

[RESOLVED] The canvas-coordinate repair represents negative padding/crop
positions with wrapping usize arithmetic. Later dither chunk additions and
channel offsets still used checked addition. The full debug-assertion suite
reproduced an overflow in `convert.rs` through the existing
`serial_parallel_parity_palette_across_thread_counts` regression.

The repair wraps chunk offsets and reduces coordinates/channel phase to
the 32-pixel table period before adding. New table, conversion and fused-stage
regressions compare wrapped negative coordinates with equivalent positive
coordinates. The full workspace all-feature suite passes with debug overflow
checks enabled: 1,024 library tests pass, 29 existing ignores; integrations,
CLI tests and doctests also pass. All 18 selected no-default-feature
regressions, strict Clippy and scoped formatting pass. No expectation or
overflow check was removed.

## Resolved: inclusive sRGB transfer thresholds (2026-11-04)

The exact linear-to-sRGB threshold 0.0031308 and inverse threshold 0.04045
were selecting the rational high branch rather than the linear branch.
Strict raw-f32-bit regressions reproduced both failures on scalar, SSE4.2
and AVX2. Both equality cases now take the linear branch, preserving sign
(including negative zero). Separate float output and the two fused u8 stages
share one nonnegative sRGB magnitude implementation.

Reference: libjxl v0.12
[`TF_SRGB`](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/cms/transfer_functions-inl.h#L197).
The six cases on the available scalar/SSE4.2/AVX2 tiers pass after the repair;
AVX-512 is unavailable on the local host. Strict workspace all-target,
all-feature Clippy passes. Full integration and post-repair fixture-oracle
validation are tracked separately.

## Resolved: inverse-XYB rounding boundary (2026-11-04)

The inverse XYB kernels now share the same implementation and accumulate
matrix terms in libjxl v0.12's order, scaling the matrix coefficients for
intensity rather than the cubic operands. This resolves the remaining blue
byte at canvas pixel (22,57): the previous 7 now matches the reference's 8.

The exact candidate source was built with Rust 1.99.1 on ARM64. All 12
selected crop, XYB, palette, LF smoothing and spline checks passed. Strict
RGBA comparison against the frozen independent djxl v0.12 reference reports
zero different bytes in each 64x64 animation frame, with no tolerance.
The existing NEON and scalar XYB consistency tests also executed. Full
integrated and platform CI validation remains a separate gate.

## 2026-11-04: cropped u8 animation references and coordinates

[RESOLVED] Fused XYB/transfer/u8 output could defer color conversion past a
reference save. The saved float planes then contained XYB or linear RGB where
a later frame expected post-transform values. Fusion is now disabled when
a frame saves a reference after color conversion. The low-memory pipeline
also uses the extended canvas height for outside-frame row lookup and passes
canvas coordinates after Extend to position-dependent stages such as dithering.

`cropped_animation_preserves_reference_background` and
`cropped_animation_u8_pipelines_and_chunks_are_bit_exact` fail before the
repair and pass for both pipelines with whole-file, 1- and 7-byte input.
The 198-byte crop fixture is retained with owner approval. Independent
libjxl byte comparison is tracked separately; these tests establish internal
reference, pipeline and chunking invariants.

## 2026-11-04: ICC header read-ahead can exceed the whole stream

The ICC parser's `OutOfBounds(c + remaining / 8)` estimate assumed at least
one coded bit per remaining predicted ICC byte. LZ77 can encode many such
bytes with fewer bits. The refill loop then waited for bytes beyond the
valid file and returned `UnexpectedCodestreamBoxEnd` without its final
parse attempt. Requesting the actual deficit `c` fixes that retry.

Reproduction: `just icc-header-repro <stream.jxl> <original.icc>`. The
caller-supplied regression fails with the old estimate, then recovers the
ICC bytes exactly with whole-file, 1-, 7- and 257-byte input chunks after
the repair. Tested on the 1x1 CMYK Probev1_ICCv4 stream retained at
`~/tmp/jxl128-read-ahead-controls-2026-11-04/4.jxl`, expected profile
`~/tmp/icc-fetch-test/Probev1_ICCv4.icc` (SHA256
`47f4e71af3c03b5801f6555a57be645040c8a83011947a35bc50b599bfc8613a`).
Before/after logs: `~/tmp/jxl128-native-header-{before,after}.log`.
This is header/ICC validation; CMYK display conversion remains a separate
CMS capability. No trailing padding workaround or limit relaxation is used.
`just feature-corpus` provisions the existing feature corpus through
codec-corpus and prints `CODEC_CORPUS_PATH`. With that caller setting,
`just icc-check` passes formatting, strict library/example Clippy and all
24 ICC-filtered library tests; log `~/tmp/jxl128-native-final-check.log`.

## Resolved 2026-10-22: short out-of-order `jxlp` splice stalled the box parser

[RESOLVED] `SmallBuffer::inject_bytes_front` (`src/api/inner/process.rs`)
shrank the buffer to the spliced payload; a payload shorter than the next box
header (12 bytes, 20 with a 64-bit size) left `refill()` unable to ever hold
that header, so `process` returned `NeedsMoreInput` forever without consuming
input (`decode`: `OutOfBounds`). The buffer now keeps its capacity. Test
`short_spliced_ooo_jxlp_followed_by_more_boxes`, 12 procedurally built files;
it fails without the fix. djxl v0.12 renders all 12 with the in-order file's
pixels. The chunked helper `api::decoder::tests::decode` now panics after 64
consecutive zero-consumption `NeedsMoreInput` calls with all input offered,
so a stall fails the test instead of hanging it.

## JPEG standalone restart markers — 2026-09-24

[PROVEN] `jpeg::writer::write_jpeg` omitted RST0–RST7 entries from JBRD
`marker_order`. These are standalone markers, including a restart after the
last MCU; restart markers inside scans are reconstructed separately in
`write_sos`. libjxl v0.12 `lib/jxl/jpeg/dec_jpeg_data_writer.cc::EncodeRestart`
writes the standalone marker's two bytes without deriving a new restart index.

`jpeg::writer::tests::standalone_restart_markers_keep_original_order` failed
before the correction: only SOI, EOI and trailing data survived. It covers all
eight marker values in a deliberately nonsequential order. The matching encoder
scanner correction is jxl-encoder `1a40b8f3`; its conformance harness at
`a6a9a116` records 21 camera originals reconstructing two bytes short before
this writer fix, and validates reconstruction with both Rust and libjxl v0.12.
No public API or scan-entropy algorithm changes are required.

Validation: release library tests with `jpeg` pass under CI's explicit
`ZENJXL_ALLOW_MISSING_CORPUS=1` configuration (834 reported passes, 28 existing
ignores). Of those reported passes, 103 corpus-backed feature cases do not run
because the Mac lacks their corpus; the unconfigured run fails those 103 cases
and passes the other 731. All eight JPEG reconstruction tests and the new
standalone-marker regression execute and pass. Logs are under
`~/tmp/jxl-backlog/decoder120-*` and `~/tmp/zenjxl-decoder/jpeg-issue120.log`.
`just jpeg-reconstruction-check <label>` reproduces the release library gate;
set the corpus policy explicitly at invocation, as CI does.
Workspace all-target, all-feature Clippy and scoped formatting also pass.

## WASI regression deadlines — 2026-09-24

[PROVEN] The two deadline regressions in `jxlrs_testdata_ports` used
`std::thread::spawn`, which traps on wasm32-wasip1. They now live in the
`regression_deadlines` integration binary with unchanged decode assertions.
Native execution keeps each 20-second worker deadline. The WASI runner applies
Wasmtime's `-W timeout=20s` to the entire dedicated binary, including both
regressions; the guest refuses to run without the runner's deadline marker.
This also bounds a decoder that never returns or polls cancellation. The
WASI suites pass with no features (767 library tests reported) and `wasm128`
(808), each with 37 existing ignores; all integration and doctest binaries
pass. Local runs use `ARBTEST_BUDGET_MS=100` and CI's explicit missing-corpus
policy; these counts do not imply external-corpus coverage. An intentional
infinite-loop module run through the same runner traps with `interrupt` at
the configured deadline.
Both WASI jobs also pass on commit `440cc002` in
[CI run 36008146603](https://github.com/imazen/zenjxl-decoder/actions/runs/36008146603).
`just wasm-ci-check <label>` runs both CI feature configurations and saves
complete logs under `~/tmp/zenjxl-decoder/` with CI's explicit missing-corpus
policy.

## i686 fixture sweep memory — 2026-09-24

[PROVEN] `run_fixture_sweep` creates up to eight workers independently of
libtest's `--test-threads=1`. The exported Linux i686 baseline (executed through QEMU on WSL) reproduces CI's
`compare_pipelines_sweep` failure at `ImageOutOfMemory(42048, 5377)`, plus two
`ImageOutOfMemory(32512, 2704)` failures. The comparison helper deliberately disables decoder request limits
and renders full f64 reference planes. The baseline's process peak RSS was
2.16 GiB under `run-heavy --mem 12G --jobs 4`; address-space allocation failed
without reaching the host memory ceiling.

The correction serializes fixtures and sweeps on 32-bit targets.
It retains the current fixture selection, including the large TOC fixture,
and all pixel-hash comparisons. The duplicate `issue865_large_toc.jxl` in
`resources/test/` is byte-identical to the dedicated `tests/testdata/jxlrs-865/`
copy; neither is removed. Before/after logs are under
`~/tmp/jxl-backlog/decoder-ci-i686-*.log`.
The rebuilt comparison passes every selected fixture on the same target, with
2.63 GiB process peak RSS and 15,562 MiB minimum host available RAM under
`run-heavy`. The failing baseline peaked lower because it could not complete;
this is not a claimed reduction in completed-decode memory.
`just i686-ci-check <label>` reproduces both CI feature configurations on a
Linux multilib host; invoke through `run-heavy` on shared hosts.

Source-export trap: `rsync -a` preserves source timestamps. When a local edit
predates completion of the remote baseline build, Cargo can reuse that baseline
binary after the copy. Refresh changed source timestamps in the exported tree
and confirm a compile occurs before treating an after-run as evidence. The
invalid first after-run is retained as `decoder-ci-i686-stale-binary.log`.


## Full CI confirmation — 2026-09-24

[CI run 36009778067](https://github.com/imazen/zenjxl-decoder/actions/runs/36009778067)
passes all 19 jobs at `420d38859ab6ee9ed9a9c213d8b301fb8afe0b38`:
both WASI configurations, i686 no/all features, Windows ARM64, Linux x64/ARM64,
macOS Intel, formatting and Clippy. This independently verifies the WASI
deadline repair (`440cc002`) and 32-bit fixture serialization (`420d3885`).
Remote `main` was verified at that commit after the run. The earlier preserved
`940d2c51` work remains in its ancestry. CI's configured corpus policy remains
in effect; a green job is not a claim that every external corpus was present.

## Fast-decode lossless exploration — 2026-10-01/02

[MEASURED] M4 Pro, 4 CLIC 2025 photos (~2.8 MP, 48 groups each), CLI
`--speedtest` u8 `--no-cms`; a smoke comparison, not zenbench. Files come
from jxl-encoder `LosslessConfig::with_fast_decode()` (`02850e60`), which
makes every channel one Gradient MA leaf with prefix codes and no LZ77.

- Entropy decoding, not prediction, bounds single-thread lossless decode.
  `main` reaches 68 MP/s on these files and 62 MP/s with ANS.
- `explore/fused-prefix-lut` (draft PR #61) has three steps:
  - `0441e81b`: a fused 12-bit table (prefix code, extra bits and signed
    unpack in one load) reaches 132 MP/s. It needs the register-resident
    `FastBits` cursor; with `&mut BitReader` in the hot loop the cursor is
    stored and reloaded per sample, and the table gives only +10%.
  - `31ad13d6`: two residuals per lookup reach 169 MP/s.
  - `e4d32db2`: two groups decoded interleaved reach 201.5 MP/s at 1 thread
    (~605 MB/s RGB8); 4 threads give 581 and 12 give 935, against 237 and
    480 on `main`. Interleaving gained nothing until the two sides were
    passed by value: behind `&mut`, both cursors spilled to memory.
- Table window on M4: 11 bits 164, 12 bits ~170, 13 bits 172, 14 bits
  161 MP/s.
- [MEASURED] Ryzen 9 7900X (Zen 4), Linux, same files, 1 thread:
  `main` 60, fused table 89, two-symbol 97, interleaved 96 MP/s
  (~288 MB/s RGB8). 4 threads: 252 to ~417; 12 threads: 383 to ~517.
  11-bit and 12-bit windows tie there. Interleaving gives nothing on
  Zen 4. Zen 5 is not measured.
- [MEASURED] On that Linux box only ~52% of the 1-thread run is decoder
  code: ~37% is kernel page faults and ~10% libc memset/memmove. The
  cause is per-group modular channel buffers. `with_buffers` allocates
  and zero-fills 144 buffers of ~248 KiB per decode (35.8 MB for a
  2.8 MP RGB image) through `alloc_zeroed_fallible`
  (try_reserve + resize, an explicit memset over fresh pages). The
  render pipeline recycles rendered group buffers into its scratch pool,
  but the modular reader never takes from it. Reusing those buffers
  across groups is the next lever for Linux throughput on every modular
  decode, not only this fast path.
- `explore/two-pass-top` (`92b9b69b`) and `explore/fused-two-pass-top`
  (`af2dfd2d`) are negative results. Two-pass gradient is slower (68 to
  52 MP/s), and fused Top (126 MP/s) is slower than fused gradient, with
  ~15% larger files.
- Conformance: preset files decode exactly in libjxl v0.12, jxl-rs 0.7.4
  and jxl-oxide 0.12.6 (gray/alpha/8/16-bit, efforts 1-9). See jxl-encoder
  `just fast-decode-check` and `just fast-decode-conformance`.
- Prefix codes cost ~3-5% size against ANS for the same settings; the
  preset is ~17% smaller than PNG on the CLIC photos.
- `sansio` (`5fc1a3e`, origin) holds unreviewed prior-session work: a new
  public `JxlIncrementalDecoder` and default zencodec/zenpixels deps. It
  is not on `main`.

Benchmark files and logs are under `~/tmp/fastll/`.

## Modular buffer recycling and parallel transforms — 2026-10-02

[MEASURED] Ryzen 9 7900X, Linux, CLI `--speedtest` u8, 3 interleaved
rounds; 11.2 MP numbers from `/usr/bin/time -v`.

- `97f08d29`: modular group buffers come from a `RecyclePool` (zero-filled,
  exact geometry) fed by the render pipeline's `take_recycled_inputs`.
  Sequential modular frames without noise or border stages output each
  two-group chunk right after decoding (`eager_modular` in
  `frame/render.rs`); parallel frames with > 16 groups per thread use
  batches of 8 groups per thread. 1 thread on 48-group frames: 60 to
  83 MP/s; sys time 0.27 to 0.04 s; 11.2 MP peak RSS 228 to 94 MiB
  (1 thread), 230 to 128 MiB (4 threads). Multi-thread speed is
  unchanged: glibc's per-thread arenas already kept those pages mapped.
- `e23e8697`: a transform layer's `do_run` calls run on rayon when their
  inputs are disjoint (`FullModularImage::run_layer`). The parallel
  path's Phase 3a-store was 1.8 of 5.5 ms per 48-group frame at 12 threads.
- Combined with `explore/fused-prefix-lut` (fast-decode files): 1 thread
  166.5, 4 threads 448.5, 12 threads 738.3 MP/s (main before both:
  60.1 / 245.0 / 376.8). 12-thread scaling is still ~4.4x the 1-thread rate;
  the remaining serial work is per-frame setup and Phase 3a output passing.
- Content of recycled buffers is always fully overwritten by decode
  (poisoning them did not change output), but the zero-fill is kept so a
  reused buffer is byte-identical to a fresh one, padding included.

## Fast lossy (VarDCT) decoding — 2026-10-02

[MEASURED] 4 CLIC 2025 photos (~2.8 MP), CLI `--speedtest`, MP/s.

- Fastest lossy encoding to decode: VarDCT with libjxl `--faster_decoding=4`
  (EPF and gaborish off, no 32x32+, simple block contexts, fixed gradient
  DC tree), about 10% larger than e7 at the same distance. Modular lossy
  (`-m 1`) is the slowest (~25 MP/s on M4). Dithering is not a cost: u16
  output (undithered) is no faster than u8.
- Decoder changes `082c50ed`, `c3d92091`, `24809159`, `04612ee4`.
  Ryzen 7900X, cjxl d1 fd4: 1 thread 122 -> 134, 12 threads 416 -> 485;
  cjxl d1 e7: 76 -> 80, 286 -> 304. M4 Pro fd4: ~101 -> ~108 / ~430 -> ~470.
  Upstream jxl-rs `5122960` on M4: within 1-2% single-threaded, slower at
  12 threads on JPEG transcodes and jxl-encoder files.
- Tried without gain (reverted): a register-cursor rewrite of the
  gradient-lookup DC loop, and hoisting the ANS/prefix match out of the AC
  loop (<1%).
- Serial floor: an image up to 2048x2048 has one LF group, decoded on one
  thread before any HF group (bitstream order: DC stream, then HF
  metadata). On M4 that is ~1.2 ms DC + ~1.3 ms HF metadata for 2.8 MP,
  about 9-10 ns per sample along a value -> context -> ANS chain; at 12
  threads it is ~40% of the frame. HF groups cannot start earlier because
  the HF metadata follows all of the DC data in the same section.
- jxl-encoder `--faster-decoding` (`32caff3d`) now follows libjxl v0.12
  (EPF by tier, kGradientFixedDC DC tree, no 64x32 from tier 2); its tier-4
  files decode within 1% of cjxl's. libjxl's tier-1 simple block context map
  and 6-histogram AC cap are not mirrored there and measured as not needed
  for decode speed.

## Lossy gap to djxl, profiled — 2026-10-02

[MEASURED] M4 Pro, 4 photos (2.8-3.1 MP), `--data-type f32` (djxl
`--disable_output` decodes to float, so u8 runs overstate the gap by the
u8 conversion), MP/s, ours / djxl v0.12:

- d1 e7: 1T 65.6 / 72.7, 12T 284 / 349. JPEG q90 transcode: 1T 124 / 136,
  12T 509 / 668. d1 fd4: 1T 110 / 105, 12T 459 / 521.
- Fixed: EPF stage 1 checked bounds on all 54 loads per vector and
  reloaded row pointers after each store (`e1463a0c`): d1 e7 1T 62.2 -> 65.9.
  Per-decode `sample` counts put EPF1 at ~1.6x djxl's before, ~1.5x after
  (sampling estimate).
- 12T is the larger gap. On the JPEG transcode at 12T the main thread spends
  ~35% of wall time in the serial LF group (DC through the weighted
  predictor, `WpOnlyLookupConfig420`, then HF metadata) and workers are idle
  ~57% of the time; the rest of the idle time is the gap between the decode
  and render parallel phases. Single-threaded sample fractions put our WP
  DC decode at roughly 1.2x djxl's time (sampling estimate, not measured
  per function).
- Register cursor (`PlainCursor`) for modular channels helps single-leaf
  trees only: fast-decode preset 66 -> 89 MP/s 1T, 487 -> 603 12T. On
  learned trees and libjxl's gradient DC tree it was neutral 1T and 3-10%
  slower at 12T in back-to-back runs, so it is off there
  (`ModularChannelDecoder::USE_CURSOR`).
- 12T numbers on this Mac move 5-10% with background load (Spotlight,
  mediaanalysisd, rust-analyzer); compare A-B-A, not A-B.


## Multi-threaded parity work — 2026-10-03

Scripts used for these measurements: `scripts/perf/` (see its README).

[MEASURED] M4 Pro (8 P + 4 E cores), f32, interleaved runs; the machine
was loaded (Spotlight indexing, load 12-24), so treat MT figures as +-5%.

- False sharing (`e404d64b`): two 4-thread processes decoded e7 lossless at
  a combined ~60 MP/s while one 8-thread process reached 50. The cause was
  the per-channel MA-tree property buffer (a small `Vec<i32>` written every
  sample) sharing a cache line with another thread's data. `IsolatedBuf`
  (`util/isolated.rs`) pads it by 128 bytes per side. The two-process vs
  one-process comparison is the quick test for this class of problem.
- Per-group timing (`JXL_GDBG`-style instrumentation, not committed) shows
  E-core threads run groups 2-3.7x slower and macOS starts some threads on
  E-cores; P-core groups also slow ~1.3x at 12 threads (all-core clocks).
  From 8 to 12 threads we gained nothing on fd4 while djxl gained 9%.
- Serial LF group (one LF group up to 2048x2048): d1e7 DC 1.87 ms + HF
  metadata 0.85 ms; JPEG q90 DC 1.03 + 0.2; fd4 DC 1.13 + 0.85. DC per
  sample matches our e7 lossless speed, itself at djxl parity. Reading
  `GradientLookup` (fd4 DC) through the register cursor was slower (LF
  2.02 -> 2.30 ms); keep `USE_CURSOR` off there.
- The batched VarDCT path is expensive: forcing 2 batches of 24 groups
  raised P3b from 2.81 to 5.12 ms (d1e7) and 0.95 to 3.03 ms (JPEG), so
  overlapping render with the next batch's decode is not worth it on top of it.
- In-task render: `8c4e8175` for `border_size == 0` pipelines (fd4, JPEG
  4:4:4), `a820cac9` for the rest (a group renders when its 3x3
  neighbourhood is decoded; inputs published through per-group
  `OnceLock`s). Those frames decode in raster order: largest-first left the
  renders to the end and lost 2.7% on JPEG 4:2:0 at 4 threads.
- Tried without gain on d3 1T: a single-pass `get_distinct_indices` (the
  profile's 2.6% there is mis-attributed inlined code; instruction count
  unchanged) and a vector-index EPF1 loop with one up-front bound (+0.2%
  instructions).
- Rejected: packing cluster + hybrid-uint config into one per-context
  table for the AC loop (0.3% fewer instructions, 1.5-3% more cycles).
- Parity at `63370cd7`, interleaved, zen/djxl (4 CLIC photos per set):
  1 thread d1e7 1.05, fd4 1.19, JPEG 4:2:0 1.10, 4:4:4 1.08, d3 1.02,
  e7 lossless 0.97, e3 1.19, e1 1.28; 12 threads d1e7 1.02, fd4 1.03,
  JPEG 4:2:0 1.03, 4:4:4 1.05, d3 1.01, e7 1.30, e3 1.36, e1 1.61. Ahead of
  upstream jxl-rs on every set at both thread counts.
- EPF gains came from codegen, not algorithm: per-vector bounds checks on
  15-21 row windows and output rows read through the row table every vector
  (`85d33a94`, `5592d393`), and SmallVec pushes building each stage call's
  row lists (`63370cd7`).
- e7 lossless 1T (0.97): per decode djxl 889M cycles / 3.02G instructions,
  zen 947M / 3.63G. Not instruction-bound: removing the tree walk's property
  bounds checks (fixed 256-entry buffer, -3% instructions), caching the WP
  weights (-1.2%) and a 2x unrolled walk (-2.4%) each left cycles equal or
  higher. The time is in the data-dependent tree branches and the leaf
  predictor switch, as in libjxl.
- d3 1T: per decode djxl 146M cycles / 710M instructions, zen 152M / 900M.
  Our EPF1 and EPF2 kernels take ~2x djxl's share each; the sRGB transfer
  (`linear_to_srgb_simd`, 9% here) is the same algorithm as libjxl's.
