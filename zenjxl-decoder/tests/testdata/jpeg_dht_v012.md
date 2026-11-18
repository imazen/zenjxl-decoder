# Native JPEG DHT reconstruction controls

These hexadecimal text files retain actual unmodified libjxl v0.12.0
`a7a9c78` outputs and their original JPEG bytes. They are generated test data,
not a hand-written model of the JBRD fields.

The source is the public encoder's
[base_b_444.jpg fixture](https://github.com/imazen/jxl-encoder/blob/18b09c7f0b912e5d2d5f8cb8dca4f318493e3db9/jxl-encoder/tests/fixtures/jbrd/base_b_444.jpg).
The [native control generator](https://github.com/imazen/jxl-encoder/blob/18b09c7f0b912e5d2d5f8cb8dca4f318493e3db9/scripts/jpeg_native_edge_controls.py)
inserts either an empty DHT marker or an unused DC3 or AC3 table definition with
sixteen zero counts before SOS. Pinned cjxl encodes each with lossless JPEG
reconstruction at effort three. Pinned djxl renders each completely and
reconstructs every original byte. All three controls render the same image.

Whitespace is ignored when decoding the hexadecimal files. SHA256 values
below describe decoded binary bytes, not the text representation.

| Control | Format | Bytes | SHA256 |
|---|---|---:|---|
| Empty DHT | JPEG | 2153 | `a13b5e7e6025bfa23d845a434798cba35b186c2502ee50362c8941a96ae1e029` |
| Empty DHT | JXL | 2188 | `daad63a7c15d1aec74764e0d84f3f3473b9a1992015c07e042393630d539ff6a` |
| Zero-symbol DC3 | JPEG | 2170 | `3b2cc849ce355f6a67dbffa345c5b5e8beaedec36c4408f70eec239a1c258953` |
| Zero-symbol DC3 | JXL | 2189 | `5d89b4b99ccfa8dba4bef1dbeaa94fb4e7808bc3383ffe62a00a25418a4b6437` |

| Zero-symbol AC3 | JPEG | 2170 | `586368b251a21893aedcfadacfa1a4ba4d0315d950a9a6392a3c097b62282e7e` |
| Zero-symbol AC3 | JXL | 2189 | `9a80037d72fe8905e6fd57a505b4cbda4f33c1689293d95de976a3e95cf1ab68` |

These inputs establish native validity and original-byte identity for the
selected cases. Decoder repair, other table IDs and complete source
coverage remain separate work.
