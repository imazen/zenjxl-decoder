# JPEG DHT reconstruction contract

Encoder issue 144 retains independent native v0.12.0 `a7a9c78` controls for
an empty DHT segment and unused zero-symbol DC3/AC3 tables. All three fully render
the same image and native JPEG reconstruction preserves every original byte.
The textual fixtures in `tests/testdata/jpeg_dht_v012_*` retain those actual
streams and originals; they are not hand-written JBRD syntax.

## Observed before and candidate

An unmodified committed decoder source export at `bb1abf42`, with its own
Cargo target directory, fails the public empty-DHT reconstruction. The legal
zero-symbol control reconstructs exactly. A separate one-byte truncated JBRD
inside a well-formed container also exposes a failure: ordinary pixel decoding
still succeeds, but checked reconstruction returns `Ok(None)` instead of an
error. The labelled before executable SHA256 is
`f7a81e99cba09c72fa7c88e89959dfc0d8393c6eb43f1c62cf271ccd44b9c315`;
its rc101 run takes 29 seconds, run-heavy peak RSS 0.87 GiB, minimum available
20,239 MiB. Source and executable identities are retained in
`decoder-dht-native-before-labelled`. The earlier unlabelled failure must not
be attributed to the zero-symbol control.

Direct metadata parsing independently accepts the zero-symbol record and
rejects the empty record as having no symbols. The candidate distinguishes
an empty segment from a table whose wire representation contains only a
sentinel, preserving that distinction through JPEG writing. An empty segment
writes no class/id or count bytes and does not replace an active Huffman table.
A zero-symbol table retains its class/id and sixteen JPEG count bytes.

The frame finalizer also retains reconstruction failures for the existing
checked reconstruction boundary instead of dropping them. Pixel decoding
remains non-fatal for optional reconstruction failures; the malformed-JBRD
control requires identical complete rendered bytes before inspecting the
reconstruction error. No public signature changes.

The candidate passes direct parsing and both public regressions: exact JPEG
reconstruction for both native controls and an error for malformed JBRD.
The initial candidate run passes in 91 seconds, peak RSS 1.48 GiB, minimum
available 19,637 MiB. These are resource observations, not a performance
comparison. The complete Rust 1.100 all-feature, no-default scalar and threaded/SIMD
workspace runs pass, as do strict all-feature/minimal Clippy, formatting and
source checks (rc0, 1,174 seconds; peak RSS 8.50 GiB, minimum available
12,843 MiB). Explicit scalar **with JPEG enabled** and dither on/off controls
also pass; the no-default workspace run alone does not establish JPEG coverage.
The additional AC3 native control is checked against the same complete rendered
frame and exact original JPEG bytes. No encoder pin has advanced yet.

## Caller and evidence boundaries

`frame::jpeg_reconstruct` calls `jpeg::decode_jbrd`; successful metadata and
captured coefficients reach `jpeg::write_jpeg` through the public reconstruction
helpers or the decoder's take method. The DHT writer runs only with the `jpeg`
feature. Ordinary frame rendering shares the finalizer but does not require
successful JPEG reconstruction. Parsing and writing errors must remain visible
through checked reconstruction.

The native fixtures distinguish an empty segment and unused zero-symbol DC3
and AC3 tables. The malformed-record unit control uses those actual decoded
records: an empty record joined after a real table, and an empty record marked
nonterminal, must reach the named writer error. A separate fully decoded
truncated-JBRD control calls the public unchecked take first and then requires
an error from the existing checked boundary. The low-level public method keeps
its existing Option signature; no new public error-accessor API is introduced.

The fixtures do not enumerate every table identifier or malformed JPEG grammar.
They establish the empty/nonempty distinction, both table classes, unchanged
active tables for the empty segment, and original-byte reconstruction for the
selected native streams. These observations are not inferred from test counts.
Reproduce the direct and public controls with `just jpeg-dht-candidate-check`;
`jpeg-dht-retained-evidence` retains an exact ELF and source checksums.

## Changed-region and fault evidence

`decoder-dht-regions-final` retains source checksums, both exact executables,
individual successful run commands, raw profiles and per-object LLVM exports.
The selected production windows execute 8/8 mapped reader regions, 9/9 writer
regions, 4/4 frame-error-retention regions and 4/4 take-error-transfer regions.
The existing no-JBRD control executes the finalizer's otherwise uncovered
closing region. These fractions describe those windows, not complete-function,
whole-file or whole-decoder coverage. The retained source checksums match the final formatted candidate exactly.

The first combined report was rejected. LLVM 23's detailed per-object dump
identifies missing zero-hash dependency records in bytemuck, byteorder,
num_traits, safe_unaligned_simd, smallvec and whereat. No decoder function has
that mismatch. The report helper retains every diagnostic and accepts only
that inspected zero-hash dependency set; any decoder mismatch, other hash or
other diagnostic fails. It binds each profile set to its own executable before
combining source-location observations.

`decoder-dht-faults` retains five executable/source fault pairs. Each runtime
regression fails for its intended fault: reject a legal empty record, write an
empty record as a table, drop the frame's reconstruction error, drop its
transfer at take, or permit a mixed/nonterminal empty record. The runner restores
the fixed file exactly after each experiment. The complete fault command
passes in 203 seconds, peak RSS 1.37 GiB, minimum available 19,729 MiB. Fault
execution is separate from positive coverage; no failed run contributes to
those source-region fractions.

Reproduce focused positive checks with `just jpeg-dht-focused`; run
`jpeg-dht-mutations <new-output>` in an owned checkout with its own Cargo target.
For region evidence, use a distinct target with `RUSTFLAGS="-C instrument-coverage"`
and the matching toolchain's llvm-tools component, then
`just jpeg-dht-coverage <new-output>`. The helper refuses zero-test success and
records the exact object/profile/source binding.

The final AC3-inclusive focused run passes in 117 seconds, peak RSS 1.36 GiB,
minimum available 19,731 MiB. The final formatted-source coverage capture passes
in 70 seconds, peak RSS 1.46 GiB, minimum available 19,078 MiB. These resource
observations describe validation runs, not performance comparisons.
