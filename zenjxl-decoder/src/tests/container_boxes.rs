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
