# Global inverse-squeeze intermediates

A valid modular channel can grow beyond one group while both inverse-squeeze
inputs still occupy global buffers. Keep that intermediate global. A later
coded LF or HF residual introduces its own grid when it participates in the
inverse transform. Output size alone does not identify that transition.

The original scheduler promoted such intermediates to HF. On a 2x16449 Gray8
stream with 128-pixel groups, a global input has vertical shift 8 while its
inverse output has shift 7. An HF view of the input has height `128 >> 8 = 0`;
reading it panics at the image row bound. The transposed case has the same
problem along the other axis. Part 1 G.4.2 governs coded channel placement;
intermediate inverse buffers are not additional coded HF channels.

The regression streams and independently retained source tiles are text
literals in `tests/global_squeeze_grid.rs` and its support module. They came
from [encoder issue 144](https://github.com/imazen/jxl-encoder/issues/144).
Pinned libjxl v0.12.0 default/forward metadata establishes legal coded-band
geometry, and djxl recovers every source code. The decoder regression compares
every grayscale sample and opaque alpha byte, in both orientations, with
integer dithering off/on and serial/parallel execution when threads are built.
No tolerance or existing expectation changes.

Replay the original source with `just global-squeeze-mutation <revision>
<new-output-directory>`. The script preserves the original executable and
both source versions, requires the actual row-bound panic in each orientation,
and restores the fixed source byte for byte. A compiler failure is not a
successful mutation result. The recorded parent is
`7a139dbc2ded10e181f56390d5fa6af4fe134433`.

`just global-squeeze-integrated` requires the feature corpus and pinned-reference
PNGs before checking formatting, strict all-feature/minimal Clippy and release
workspace tests in all-feature, scalar and parallel/SIMD configurations.
`just global-squeeze-coverage` retains instrumented objects/profiles for
inspection of `make_grids`. Corpus absence fails explicitly. This is selected
scheduler regression coverage, not proof of every legal squeeze schedule or
complete decoder conformance. Cross-platform CI has a separate verdict.

The isolated November 17 validation passes all three integrated configurations,
including both new full-decode regressions. The original-source replay fails
both with the row-bound assertion. A single exact instrumented executable and
its completed profile cover 4/4 mapped regions in the changed squeeze-grid
propagation expressions, with no LLVM mapping diagnostics. The retained
`identity.json` binds executable, profile and source hashes. This fraction
excludes other transform arms and allocation/dependency scheduling. Use
`just global-squeeze-regions <llvm-directory> <executable> <profile> <new-output>`
to preserve those inputs and reproduce the selected report.
