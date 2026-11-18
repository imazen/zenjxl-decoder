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
