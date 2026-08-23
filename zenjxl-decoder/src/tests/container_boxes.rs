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
