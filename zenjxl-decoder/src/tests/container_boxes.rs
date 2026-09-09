// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! ISOBMFF container edge cases that libjxl accepts.

fn box_(ty: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut v = ((8 + payload.len()) as u32).to_be_bytes().to_vec();
    v.extend_from_slice(ty);
    v.extend_from_slice(payload);
    v
}

/// `basic.jxl` (a bare codestream) wrapped in a container, with `extra`
/// boxes spliced in `before` the `jxlc` box and `after` it.
fn container(before: &[Vec<u8>], after: &[Vec<u8>]) -> Vec<u8> {
    let codestream = crate::util::test::fixture_bytes("basic.jxl");
    assert_eq!(
        &codestream[..2],
        &[0xff, 0x0a],
        "basic.jxl is a bare codestream"
    );
    let mut v = Vec::new();
    v.extend_from_slice(&[
        0, 0, 0, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ]);
    v.extend_from_slice(&box_(b"ftyp", b"jxl \0\0\0\0jxl "));
    for b in before {
        v.extend_from_slice(b);
    }
    v.extend_from_slice(&box_(b"jxlc", &codestream));
    for b in after {
        v.extend_from_slice(b);
    }
    v
}

fn decodes(data: &[u8]) {
    let img = crate::decode(data).unwrap_or_else(|e| panic!("decode failed: {e:?}"));
    assert_eq!((img.width, img.height), (1, 1));
    let plain = crate::decode(&crate::util::test::fixture_bytes("basic.jxl")).unwrap();
    assert_eq!(
        img.data, plain.data,
        "container wrapping changed the pixels"
    );
}

#[test]
fn plain_container_decodes() {
    decodes(&container(&[], &[]));
}

/// An 8-byte box (size == header size, no payload) is legal ISOBMFF and
/// libjxl accepts it (`decode.cc` only rejects `box_size < header_size`).
/// The fork used to return `InvalidBox`. (jxl-rs #828, item 1)
#[test]
fn empty_box_before_codestream_is_accepted() {
    decodes(&container(&[box_(b"junk", &[])], &[]));
}

#[test]
fn empty_box_after_codestream_is_accepted() {
    decodes(&container(&[], &[box_(b"junk", &[])]));
}

/// A box with size 0 extends to the end of the file -- for *any* box type,
/// not only `jxlc`/`jxlp` (libjxl `decode.cc`).
#[test]
fn zero_size_trailing_box_is_skipped_to_eof() {
    let mut trailing = 0u32.to_be_bytes().to_vec();
    trailing.extend_from_slice(b"junk");
    trailing.extend_from_slice(&[0xAA; 37]);
    decodes(&container(&[], &[trailing]));
}

/// A box claiming to be smaller than its own header is still invalid.
#[test]
fn box_smaller_than_its_header_is_rejected() {
    let mut bad = 5u32.to_be_bytes().to_vec();
    bad.extend_from_slice(b"junk");
    let data = container(&[bad], &[]);
    assert!(crate::decode(&data).is_err());
}

/// A `jxlp` box with an index but no payload (size == 12) is an empty
/// partial-codestream box; the stream continues in the next one.
#[test]
fn empty_jxlp_box_is_accepted() {
    let codestream = crate::util::test::fixture_bytes("basic.jxl");
    let mut v = Vec::new();
    v.extend_from_slice(&[
        0, 0, 0, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ]);
    v.extend_from_slice(&box_(b"ftyp", b"jxl \0\0\0\0jxl "));
    v.extend_from_slice(&box_(b"jxlp", &0u32.to_be_bytes()));
    let mut last = 0x8000_0001u32.to_be_bytes().to_vec();
    last.extend_from_slice(&codestream);
    v.extend_from_slice(&box_(b"jxlp", &last));
    decodes(&v);
}

// ---- out-of-order jxlp boxes (ftyp minor version 1; jxl-rs #752) --------

/// Split a container into its boxes as (type, whole box bytes).
fn boxes(data: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let ty: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();
        let end = if size == 0 { data.len() } else { pos + size };
        out.push((ty, data[pos..end].to_vec()));
        pos = end;
    }
    out
}

fn jxlp_index(b: &[u8]) -> u32 {
    u32::from_be_bytes(b[8..12].try_into().unwrap()) & 0x7fff_ffff
}

/// The #752 fixture: `ftyp` minor version 1, `jxlp` boxes stored in the
/// order 0, 2, 1, 3(last); 500x160, 4 frames; djxl decodes it.
fn ooo_fixture() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/jxlrs-752/animated_ooo_jxlp.jxl");
    std::fs::read(&path).unwrap()
}

/// The same file with its jxlp boxes physically sorted by index.
fn in_order_rewrite(data: &[u8]) -> Vec<u8> {
    let mut bx = boxes(data);
    let first_jxlp = bx.iter().position(|(t, _)| t == b"jxlp").unwrap();
    let mut jxlps: Vec<Vec<u8>> = bx
        .iter()
        .filter(|(t, _)| t == b"jxlp")
        .map(|(_, b)| b.clone())
        .collect();
    jxlps.sort_by_key(|b| jxlp_index(b));
    bx.retain(|(t, _)| t != b"jxlp");
    for (i, b) in jxlps.into_iter().enumerate() {
        bx.insert(first_jxlp + i, (*b"jxlp", b));
    }
    bx.into_iter().flat_map(|(_, b)| b).collect()
}

/// All frames, as the f32 test helper returns them, decoded with `chunk`-byte
/// input slices.
fn frames(data: &[u8], chunk: usize) -> Vec<Vec<crate::image::Image<f32>>> {
    crate::api::decoder::tests::decode(data, chunk, false, false, None)
        .unwrap()
        .1
}

fn assert_same_frames(
    a: &[Vec<crate::image::Image<f32>>],
    b: &[Vec<crate::image::Image<f32>>],
    what: &str,
) {
    assert_eq!(a.len(), b.len(), "{what}: frame count");
    for (fa, fb) in a.iter().zip(b) {
        assert_eq!(fa.len(), fb.len());
        for (ia, ib) in fa.iter().zip(fb) {
            assert_eq!(ia.size(), ib.size());
            for y in 0..ia.size().1 {
                assert_eq!(ia.row(y), ib.row(y), "{what}: row {y} differs");
            }
        }
    }
}

#[test]
fn ooo_fixture_is_really_out_of_order() {
    let idx: Vec<u32> = boxes(&ooo_fixture())
        .iter()
        .filter(|(t, _)| t == b"jxlp")
        .map(|(_, b)| jxlp_index(b))
        .collect();
    assert_eq!(idx, [0, 2, 1, 3]);
}

/// Out-of-order `jxlp` boxes (allowed by `ftyp` minor version 1) must decode
/// to exactly what the in-order file decodes to. The fork used to return
/// `InvalidBox` on the first out-of-order box.
#[test]
fn out_of_order_jxlp_decodes_like_in_order() {
    let ooo = ooo_fixture();
    let ordered = in_order_rewrite(&ooo);
    assert_ne!(ooo, ordered);
    let reference = frames(&ordered, usize::MAX);
    assert_eq!(reference.len(), 4, "4 visible frames expected");
    assert_same_frames(&frames(&ooo, usize::MAX), &reference, "one-shot");
    for chunk in [1usize, 13, 100, 1000] {
        assert_same_frames(&frames(&ooo, chunk), &reference, &format!("chunk {chunk}"));
    }
    // and through the public API
    let a = crate::decode(&ooo).unwrap();
    let b = crate::decode(&ordered).unwrap();
    assert_eq!((a.width, a.height), (500, 160));
    assert_eq!(a.data, b.data);
}

/// With `ftyp` minor version 0 an out-of-order `jxlp` box is still invalid
/// (libjxl: "jxlp boxes require file format version 1").
#[test]
fn out_of_order_jxlp_rejected_without_ftyp_version_1() {
    let mut data = ooo_fixture();
    // ftyp payload: 'jxl ' + minor version (u32) at box offset 8..16.
    let ftyp = 12usize;
    assert_eq!(&data[ftyp + 4..ftyp + 8], b"ftyp");
    assert_eq!(&data[ftyp + 12..ftyp + 16], &[0, 0, 0, 1]);
    data[ftyp + 15] = 0;
    assert!(crate::decode(&data).is_err());
    // the in-order rewrite is fine with version 0
    let ordered = in_order_rewrite(&data);
    assert!(crate::decode(&ordered).is_ok());
}

/// A duplicated out-of-order index is an error, not a silent overwrite.
#[test]
fn duplicate_ooo_jxlp_index_is_rejected() {
    let bx = boxes(&ooo_fixture());
    let mut out = Vec::new();
    for (t, b) in &bx {
        out.extend_from_slice(b);
        if t == b"jxlp" && jxlp_index(b) == 2 {
            out.extend_from_slice(b); // duplicate index 2
        }
    }
    assert!(crate::decode(&out).is_err());
}

// ---- `jxlp` boxes smaller than a codestream section ----------------------

/// A version-1 container (out-of-order `jxlp` allowed) whose codestream is
/// `parts[i]` concatenated in index order, with the boxes written in `order`.
/// The highest index carries the "last" flag; a part may be empty, which is a
/// legal 12-byte `jxlp` box that carries only an index.
fn jxlp_stream(parts: &[Vec<u8>], order: &[usize]) -> Vec<u8> {
    let last = parts.len() - 1;
    let mut v = Vec::new();
    v.extend_from_slice(&[
        0, 0, 0, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ]);
    v.extend_from_slice(&box_(b"ftyp", b"jxl \0\0\0\x01jxl "));
    for &i in order {
        let mut payload = (i as u32 | if i == last { 0x8000_0000 } else { 0 })
            .to_be_bytes()
            .to_vec();
        payload.extend_from_slice(&parts[i]);
        v.extend_from_slice(&box_(b"jxlp", &payload));
    }
    v
}

/// `codestream` cut into `chunk`-byte pieces.
fn split(codestream: &[u8], chunk: usize) -> Vec<Vec<u8>> {
    codestream.chunks(chunk).map(<[u8]>::to_vec).collect()
}

/// Decode `data` (one-shot and in small chunks) and require it to match what
/// the bare `codestream` decodes to.
fn decodes_like_bare_codestream(data: &[u8], codestream: &[u8]) {
    let reference = frames(codestream, usize::MAX);
    assert_same_frames(&frames(data, usize::MAX), &reference, "one-shot");
    for chunk in [1usize, 7, 64, 1000] {
        assert_same_frames(&frames(data, chunk), &reference, &format!("chunk {chunk}"));
    }
    let a = crate::decode(data).unwrap_or_else(|e| panic!("decode failed: {e:?}"));
    let b = crate::decode(codestream).unwrap();
    assert_eq!((a.width, a.height), (b.width, b.height));
    assert_eq!(a.data, b.data, "streamed container changed the pixels");
}

/// `jxlp` boxes smaller than a codestream section, in index order. The fork
/// stopped as soon as a round could not finish a section, even though the
/// bytes that finish it were in the next box; `djxl` and upstream jxl-rs
/// decode these. Unrelated to box *ordering* -- this file is in order.
#[test]
fn small_in_order_jxlp_boxes_decode() {
    let codestream = crate::util::test::fixture_bytes("3x3_srgb_lossy.jxl");
    for chunk in [4usize, 16, 64] {
        let parts = split(&codestream, chunk);
        let data = jxlp_stream(&parts, &(0..parts.len()).collect::<Vec<_>>());
        decodes_like_bare_codestream(&data, &codestream);
    }
}

// ---- the `cjxl --output_mode=2` streaming layout -------------------------
//
// `cjxl -e 7 --output_mode=2` writes its `jxlp` boxes in the order the encoder
// finishes them, not in index order: the boxes that carry the *start* of the
// codestream (image header, ICC, TOC) are patched up last and land at the very
// end of the file, after the box whose index has the "last" bit set, and the
// encoder leaves empty placeholder boxes for indices it never filled. libjxl's
// `djxl` decodes such files.

/// Every index except 0 and 1 first, then index 1 -- so the box that completes
/// the beginning of the codestream is the physically last box in the file and
/// the whole input is consumed before the image header is complete. This is
/// the shape `cjxl --output_mode=2` produces.
fn head_last_order(n: usize) -> Vec<usize> {
    let mut order = vec![0];
    order.extend(2..n);
    order.push(1);
    order
}

/// The header is only complete once the physically last box has been read and
/// the buffered boxes behind it have been spliced in. The fork used
/// `input.available_bytes() > 0` as its sole "can I get more data" test, so at
/// end of input it reported a truncated file while holding every remaining
/// byte in its own out-of-order buffer.
#[test]
fn ooo_jxlp_header_completed_after_input_eof() {
    let codestream = crate::util::test::fixture_bytes("3x3_srgb_lossy.jxl");
    let parts = split(&codestream, 4);
    assert!(parts.len() > 8, "header must span several boxes");
    let data = jxlp_stream(&parts, &head_last_order(parts.len()));
    decodes_like_bare_codestream(&data, &codestream);
}

/// An empty `jxlp` box in the middle of the buffered run advances the expected
/// index without producing codestream bytes. The fork stopped injecting there
/// and went looking for the next box in the *file*, which at end of file is an
/// unsatisfiable read reported as a truncated file. A box with no payload also
/// used to wedge the parser mid-buffering, which showed up as a hang.
#[test]
fn ooo_jxlp_empty_box_does_not_stop_the_injection_chain() {
    let codestream = crate::util::test::fixture_bytes("3x3_srgb_lossy.jxl");
    let mut parts = split(&codestream, 4);
    // Two adjacent placeholder boxes, plus one more later in the run.
    parts.insert(2, Vec::new());
    parts.insert(3, Vec::new());
    parts.insert(9, Vec::new());
    let data = jxlp_stream(&parts, &head_last_order(parts.len()));
    decodes_like_bare_codestream(&data, &codestream);
}

/// A genuinely truncated out-of-order file must still fail, promptly. The
/// retry paths that keep reading while the box parser still holds buffered
/// `jxlp` payloads only continue after a round that made progress, so a file
/// that can never be completed terminates instead of spinning. (If it did
/// spin, this test would hang rather than fail.)
#[test]
fn truncated_ooo_jxlp_file_still_fails() {
    let codestream = crate::util::test::fixture_bytes("3x3_srgb_lossy.jxl");
    let mut parts = split(&codestream, 4);
    parts.insert(2, Vec::new());
    let full = jxlp_stream(&parts, &head_last_order(parts.len()));
    assert!(crate::decode(&full).is_ok(), "the complete file decodes");
    // Cutting anywhere before the end drops the boxes that carry the start of
    // the codestream, so every prefix is undecodable.
    for num in [1usize, 2, 3, 5, 6, 8] {
        let cut = full.len() * num / 10;
        assert!(
            crate::decode(&full[..cut]).is_err(),
            "{num}0% prefix ({cut} bytes) must not decode"
        );
    }
}

/// The number of boxes held ahead of the one being decoded scales with image
/// size: a 12000x9000 `cjxl -e 7 --output_mode=2` file measured here needs
/// 1724 buffered at once. The fork capped it at 1024 and rejected such files
/// as an invalid container; libjxl's limit is `kNumBuffersLimit = 1 << 20`.
#[test]
fn ooo_jxlp_buffers_more_than_a_thousand_boxes() {
    let codestream = crate::util::test::fixture_bytes("8x8_noise.jxl");
    let mut parts = split(&codestream, 4);
    let tail = parts.split_off(2);
    // 1100 placeholder indices between the head and the rest of the stream:
    // every one of them is buffered before index 1 arrives at end of file.
    parts.resize(parts.len() + 1100, Vec::new());
    parts.extend(tail);
    let data = jxlp_stream(&parts, &head_last_order(parts.len()));
    assert!(data.len() < 32 * 1024, "synthetic file stays small");
    decodes_like_bare_codestream(&data, &codestream);
}
