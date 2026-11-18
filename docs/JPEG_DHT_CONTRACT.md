# JPEG DHT reconstruction contract

Unmodified libjxl v0.12.0 a7a9c78 supplies native empty-DHT and unused
zero-symbol DC3/AC3 streams. Each fully renders the same 67x53 image and
reconstructs its original JPEG exactly in the native decoder. Text fixtures,
byte hashes and the generator are recorded in
[the fixture provenance](../zenjxl-decoder/tests/testdata/jpeg_dht_v012.md).

Before this repair, the decoder rejects the empty record as having no symbols;
the zero-symbol DC3 control already reconstructs exactly. The labelled original
executable f7a81e99cba09c72fa7c88e89959dfc0d8393c6eb43f1c62cf271ccd44b9c315
and frozen source are retained in decoder-dht-native-before-labelled.

Part 2 A.2 distinguishes an all-zero empty-segment record from a real table
containing a sentinel. The reader preserves that distinction in private state.
The writer emits only the two-byte empty segment length and leaves active
tables unchanged; DC3 and AC3 zero-symbol definitions retain their class/id
and sixteen count bytes. Mixed or nonterminal empty records return a named
InvalidJbrd error. The malformed-record controls use actual decoded native
records rather than a separately invented JBRD model.

The complete Rust 1.100 feature/scalar/threaded workspace and strict lint
gates pass for the combined repair. Focused JPEG-enabled scalar/SIMD controls
also pass with dither on/off, exact JPEG bytes and complete rendered buffers.
The selected reader/writer windows execute 8/8 and 9/9 mapped regions, and
faults rejecting or miswriting the empty record and permitting malformed
placement are detected. Complete exact-object evidence and the separate
checked-reconstruction error repair follow in the expanded report.

Other table identifiers and arbitrary malformed JPEG grammar are not
exhaustively enumerated by these native controls. No whole-decoder coverage
or complete format-audit claim follows from the selected regressions.
