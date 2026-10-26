// Copyright (c) Imazen LLC.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Native probe only. The public uncoalesced request continues to refuse.

use std::{cell::RefCell, io::Write, rc::Rc};

use crate::{
    api::decoder::tests::{decode, decode_with_layer_capture},
    frame::Frame,
    headers::frame_header::FrameType,
    image::Image,
};

#[derive(Debug, PartialEq, Eq)]
struct Layer {
    origin: (i32, i32),
    size: (usize, usize),
    planes: Vec<Vec<u32>>,
}

fn bits(plane: &Image<f32>) -> Vec<u32> {
    (0..plane.size().1)
        .flat_map(|y| plane.row(y).iter().map(|sample| sample.to_bits()))
        .collect()
}

fn exact_samples(actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(mismatch.is_none(), "first differing sample: {mismatch:?}");
}

fn layer(frame: &Frame) -> Option<Layer> {
    let header = frame.header();
    if header.frame_type != FrameType::RegularFrame {
        assert!(frame.raw_layer_data.is_none());
        return None;
    }
    let planes = frame
        .raw_layer_data
        .as_ref()
        .expect("regular layer not captured");
    let size = header.size_upsampled();
    for plane in planes {
        assert_eq!(plane.size(), size, "capture clipped a layer to the canvas");
        for y in 0..size.1 {
            assert!(
                plane.row(y).iter().all(|sample| sample.is_finite()),
                "unwritten or nonfinite layer sample at row {y}"
            );
        }
    }
    Some(Layer {
        origin: (header.x0, header.y0),
        size,
        planes: planes.iter().map(bits).collect(),
    })
}

// Optional artifact export changes no assertions. The independent oracle gate
// explicitly supplies this directory, then requires and compares every file.
fn export(name: &str, layers: &[Layer]) {
    if let Some(root) = std::env::var_os("JXL_RAW_LAYER_ARTIFACTS") {
        let root = std::path::Path::new(&root).join(name.replace('/', "_"));
        std::fs::create_dir(&root).unwrap();
        let mut metadata = Vec::new();
        for (index, layer) in layers.iter().enumerate() {
            let mut files = Vec::new();
            for (channel, samples) in layer.planes.iter().enumerate() {
                let name = format!("layer-{index}-{channel}.f32le");
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join(&name))
                    .unwrap();
                let mut file = std::io::BufWriter::new(file);
                for sample in samples {
                    file.write_all(&sample.to_le_bytes()).unwrap();
                }
                file.flush().unwrap();
                files.push(name);
            }
            metadata.push(serde_json::json!({
                "origin": layer.origin, "size": layer.size, "planes": files,
            }));
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("layers.json"))
            .unwrap();
        serde_json::to_writer_pretty(file, &metadata).unwrap();
    }
}

#[test]
fn raw_layer_probe_keeps_reference_pixels_and_all_local_samples() {
    for name in [
        "multiple_layers_noise_spline.jxl",
        "conformance_test_images/cmyk_layers.jxl",
    ] {
        let bytes = crate::util::test::fixture_bytes(name);
        let (_, baseline) = decode(&bytes, usize::MAX, false, false, None).unwrap();
        let mut expected = None;
        for simple in [false, true] {
            for chunk in [usize::MAX, 16_384] {
                let captured = Rc::new(RefCell::new(Vec::new()));
                let target = captured.clone();
                let (_, output) = decode_with_layer_capture(
                    &bytes,
                    chunk,
                    simple,
                    false,
                    Some(Box::new(move |frame, _| {
                        if let Some(layer) = layer(frame) {
                            target.borrow_mut().push(layer);
                        }
                        Ok(())
                    })),
                    true,
                )
                .unwrap();
                assert_eq!(output.len(), baseline.len());
                for (actual, reference) in output.iter().zip(&baseline) {
                    assert_eq!(actual.len(), reference.len());
                    for (actual, reference) in actual.iter().zip(reference) {
                        exact_samples(&bits(actual), &bits(reference));
                    }
                }
                let captured = Rc::try_unwrap(captured).unwrap().into_inner();
                assert!(!captured.is_empty());
                if let Some(expected) = &expected {
                    let expected: &Vec<Layer> = expected;
                    assert_eq!(captured.len(), expected.len());
                    for (actual, expected) in captured.iter().zip(expected) {
                        assert_eq!(
                            (actual.origin, actual.size),
                            (expected.origin, expected.size)
                        );
                        assert_eq!(actual.planes.len(), expected.planes.len());
                        for (actual, expected) in actual.planes.iter().zip(&expected.planes) {
                            exact_samples(actual, expected);
                        }
                    }
                } else {
                    expected = Some(captured);
                }
            }
        }
        let expected = expected.unwrap();
        if name == "multiple_layers_noise_spline.jxl" {
            assert_eq!(
                expected
                    .iter()
                    .map(|l| (l.origin, l.size))
                    .collect::<Vec<_>>(),
                vec![
                    ((0, 0), (2048, 1152)),
                    ((512, 164), (1024, 576)),
                    ((512, 640), (1024, 576)),
                    ((512, 164), (1024, 576)),
                    ((1860, 950), (1024, 576)),
                ]
            );
        }
        export(name, &expected);
    }
}
