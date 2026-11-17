// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

use crate::{
    bit_reader::BitReader,
    container::ContainerParser,
    headers::{
        FileHeader, JxlHeader,
        encodings::UnconditionalCoder,
        frame_header::{Encoding, FrameHeader},
        toc::{Toc, TocNonserialized},
    },
    util::test::{fixture_bytes, pack_bits},
};

/// Reuse the entropy payload of an existing RGB+alpha Modular image. Only
/// frame headers/TOCs are authored here; the source payload stays unchanged.
fn before_ct_ec_stream(extra_add: bool) -> Vec<u8> {
    let source =
        ContainerParser::collect_codestream(&fixture_bytes("3x3a_srgb_lossless.jxl")).unwrap();
    let mut reader = BitReader::new(&source);
    let file = FileHeader::read(&mut reader).unwrap();
    assert!(!file.image_metadata.xyb_encoded);
    assert_eq!(file.image_metadata.extra_channel_info.len(), 1);
    let file_end = reader.total_bits_read().div_ceil(8);
    let header =
        FrameHeader::read_unconditional(&(), &mut reader, &file.frame_header_nonserialized())
            .unwrap();
    assert_eq!(header.encoding, Encoding::Modular);
    assert_eq!(header.num_toc_entries(), 1);
    let toc =
        Toc::read_unconditional(&(), &mut reader, &TocNonserialized { num_entries: 1 }).unwrap();
    reader.jump_to_byte_boundary().unwrap();
    let payload_start = reader.total_bits_read() / 8;
    let payload = &source[payload_start..payload_start + toc.entries[0] as usize];
    assert!(payload.len() < 1024);
    let mut output = source[..file_end].to_vec();
    for first in [true, false] {
        let mut fields = vec![
            (0, 1),
            (0, 2),
            (1, 1),
            (0, 2), // regular Modular, flags=0
            (0, 1),
            (0, 2),
            (0, 2),
            (1, 2), // RGB, upsampling 1, group shift 1
            (0, 2),
            (0, 1), // one pass, no crop
            (0, 2), // color Replace, full canvas: no source field
        ];
        if !first && extra_add {
            fields.extend([(1, 2), (1, 2)]); // extra Add from reference slot 1
        } else {
            fields.push((0, 2)); // extra Replace
        }
        fields.push((u64::from(!first), 1)); // is_last
        if first {
            fields.extend([(1, 2), (1, 1)]); // save slot 1, before CT
        }
        fields.extend([(0, 2), (1, 1), (0, 2), (0, 1)]); // name, filter, extensions, TOC permutation
        let bytes = pack_bits(&fields);
        let mut check = BitReader::new(&bytes);
        let parsed =
            FrameHeader::read_unconditional(&(), &mut check, &file.frame_header_nonserialized())
                .unwrap();
        assert_eq!(parsed.save_before_ct, first);
        assert_eq!(parsed.is_last, !first);
        assert_eq!(parsed.width, file.size.xsize());
        output.extend(bytes);
        output.extend(pack_bits(&[(0, 2), (payload.len() as u64, 10)]));
        output.extend(payload);
    }
    output
}

#[test]
fn before_ct_extra_add_uses_full_canvas_reference() {
    use crate::api::decoder::tests::decode;
    let control = before_ct_ec_stream(false);
    let stream = before_ct_ec_stream(true);
    // Optional oracle capture is selected explicitly by the caller.
    if let Ok(directory) = std::env::var("ZENJXL_BEFORE_CT_ORACLE_DIR") {
        std::fs::write(
            std::path::Path::new(&directory).join("extra-add.jxl"),
            &stream,
        )
        .unwrap();
        std::fs::write(
            std::path::Path::new(&directory).join("extra-replace.jxl"),
            &control,
        )
        .unwrap();
        std::fs::write(
            std::path::Path::new(&directory).join("native.rgba8"),
            crate::api::decode(&stream).unwrap().data,
        )
        .unwrap();
    }
    let (_, reference) = decode(&control, usize::MAX, false, false, None).unwrap();
    assert_eq!(reference.len(), 1);
    for simple in [false, true] {
        for chunk in [usize::MAX, 1, 7] {
            let (_, frames) = decode(&stream, chunk, simple, false, None).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].len(), 2); // interleaved RGBA and separate alpha
            for y in 0..frames[0][1].size().1 {
                for (&actual, &base) in frames[0][1].row(y).iter().zip(reference[0][1].row(y)) {
                    assert_eq!(actual, 2.0 * base);
                }
            }
        }
    }
}
