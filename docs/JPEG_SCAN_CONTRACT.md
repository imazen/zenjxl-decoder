# JPEG scan reconstruction contracts

Native v0.12.0 a7a9c78 controls distinguish baseline and progressive scans,
extra ZRL tokens, and the padding bits that terminate entropy segments. The
original decoder reconstructs the baseline ZRL input incorrectly while native
reconstruction retains every original byte. Complete pixel decoding remains
available in both cases. The source fixtures and unmodified native identities
live in `zenjxl-decoder/tests/testdata/jpeg_scan_v012.md`.

Two corrections are independent. Baseline trailing ZRLs belong after the final
nonzero AC coefficient, after DC; a run that finishes zigzag position 63 has no
following EOB. Non-interleaved scans walk the visible component grid while
coefficient addressing keeps its padded stride. The 16x8/2x2 grayscale control
reconstructs 155 bytes instead of the native original's 149 after correcting
AC ordering alone. Restoring the old grid in a fresh fault replay reproduces
that separate failure with an exact retained executable.

The expanded native inputs cover nonzero AC before/after ZRL, multiple runs,
an exact block finish, long Huffman codes, grayscale absolute sampling,
restart markers and interleaved 4:4:4/odd 4:2:0 grids. Every native reference
renders fully and rebuilds its original exactly. Default SIMD and scalar Rust
reconstruction retain those bytes. The flat controls compare complete decoded
buffers. Every expanded control also compares every RGB16 output sample with
unmodified native PNG16 output: all eleven maximum differences are zero.
The sample text is extracted with the existing audit PNG reader, with both
PNG and JXL hashes checked against the native run identity. This selected
fixture set does not establish general rendering accuracy.

Private malformed-state tests separately check padding fallback bytes,
out-of-block and arithmetic errors, and that a failed scan appends no partial
entropy output. These are internal consistency controls, not valid encoded
JBRD inputs. The serialized JBRD ranges cannot overflow the run accumulator:
there are at most 65,555 extra records with at most 275 runs each. Those limits
come from the existing reader selectors, not a population measurement.

The bound source-region report observes 17/17 regions in the visible-grid
selection and 35/35 in the extra-run handoff, combining native inputs with
private fallback/error controls. The AC helper observes 44/45: the absent
branch is zero `extra_len` after an already nonzero coefficient, which the
integer categorizer cannot return. No full JPEG/Annex A or decoder coverage
is claimed. Every object and profile is retained in the scan coverage archive.
LLVM 23's default-thread dump crashes on the retained inputs; exporting those
same objects/profiles with one thread succeeds. Earlier failed exports remain.

Five independent runtime faults are detected and restored by checksum:
missing trailing runs, runs before a nonzero AC value, the old padded grid,
an extra EOB at an exact block finish, and bypassed extent validation. The
fault chain passes in `decoder-baseline-zrl-faults.log` (68 seconds; build/test
peak RSS 1.36 GiB, minimum available 18,957 MiB). Focused default/scalar,
existing DHT, helper and strict all-feature lint checks pass in
`decoder-baseline-zrl-focused.log` (53 seconds; peak RSS 1.35 GiB, minimum
available 18,854 MiB).

The first broad run was interrupted at low disk headroom; its rc130 log
provides no complete verdict. After archiving and checksum-verifying inactive
Cargo output, the resumed workspace release/all-feature, scalar, threads/SIMD,
source and strict lint checks pass (`decoder-baseline-zrl-integrated-v2.log`,
1,039 seconds; peak RSS 8.70 GiB, minimum available 12,449 MiB). The subsequent
RGB16 assertions pass in `decoder-baseline-native-pixels-v2.log`. Production
arithmetic, accepted fixtures and established thresholds are unchanged
outside the documented baseline reconstruction correction.

The final restored source/strict lint/default/scalar check and fresh bound
region capture pass with the RGB16 assertions included:
`decoder-scan-final-pixels-v2.log`, 37 seconds, peak RSS 1.30 GiB, minimum
available 18,832 MiB. The region windows retain 17/17, 35/35 and 44/45.

## Frame color transform and component identifiers

JPEG reconstruction uses the frame's `do_ycbcr` flag for plane order: YCbCr
maps JPEG components to JXL channels `[1, 0, 2]`; an RGB frame retains order.
Grayscale still reads channel 1. JBRD `component_type` only compresses the
component identifiers (Part 2 Annex A); it does not define the frame transform.

The prior mapping selected the swap only for the standard YCbCr identifier
shortcut. Unmodified v0.12 streams with RGB identifiers and Adobe transform 1,
RGB identifiers with JFIF, custom YCbCr identifiers, or standard YCbCr identifiers with Adobe transform 0
reconstruct different
JPEG bytes before repair. Complete native PNG16 samples match exactly in all
nine controls, separating reconstruction from displayed pixels. The ordinary RGB/YCbCr controls retain exact JPEG reconstruction before repair.
Source/stream/sample provenance is in
[the native color fixtures](../zenjxl-decoder/tests/testdata/jpeg_color_v012.md).
The retained original executable is `decoder-color-channel-original-v2`;
its runtime returns 101 with those three independent reconstruction failures.

The final control inventory also includes standard YCbCr identifiers with
Adobe transform 0, and custom RGB identifiers with Adobe transform 0. The
retained original-source replay detects four independent reconstruction
mismatches; every tested native PNG16 sample still matches. All nine repaired
sources reconstruct exactly. The replay restores all five overlaid production
files byte for byte (`decoder-color-channel-original-complete`).

`decoder-color-channel-final-coverage` retains the exact fixed source, objects
and profiles. The selected channel-order block executes 9/9 mapped regions,
including grayscale and both color transforms. Existing scan regions remain
17/17 for the visible grid, 35/35 for the extra-run handoff and 44/45 for AC
emission; the remaining AC region is the previously documented impossible
zero-category alternative. These observations do not establish every JPEG
coefficient value, scan schedule or complete decoder conformance.

The final normal workspace all-feature tests/doctests, default-disabled JPEG
controls, strict workspace all-target Clippy with all/default-disabled features,
source checks and scoped formatting pass. The run requires the existing prepared
feature corpus with its two native reference PNGs, verified before testing.
`decoder-color-channel-integrated-prepared.log`: rc0, 226 seconds; run-heavy
peak RSS 7.88 GiB, minimum available 12,811 MiB. Earlier broad attempts retained
103 missing-corpus failures and then two missing-reference failures; neither is
a passing full-suite verdict. No threshold, expected byte or ignore is changed.
The removed identifier classification field was private, redundant after JBRD
parsing, and is not a public API change. Remote CI remains a separate verdict.
