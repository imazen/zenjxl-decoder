// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Decode benchmarks using zenbench with real JXL images.
//!
//! Run: cargo bench -p zenjxl-decoder --bench decode_bench

use std::path::PathBuf;
use zenbench::{Suite, Throughput};

fn test_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test")
}

/// Load a test image and return (data, pixel_count).
fn load_image(name: &str) -> (Vec<u8>, u64) {
    let path = test_dir().join(name);
    let data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let image = zenjxl_decoder::decode(&data).expect("failed to decode for pixel count");
    let pixels = (image.width * image.height) as u64;
    (data, pixels)
}

macro_rules! bench_image {
    ($group:expr, $name:literal) => {{
        let (data, pixels) = load_image($name);
        $group.throughput(Throughput::Elements(pixels));
        $group.throughput_unit("pixels");
        let label = $name.strip_suffix(".jxl").unwrap();
        $group.bench(label, move |b| {
            b.iter(|| zenjxl_decoder::decode(zenbench::black_box(&data)).unwrap())
        });
    }};
}

fn bench_decode(suite: &mut Suite) {
    // 4K VarDCT images — main decode performance target
    suite.compare("vardct_4k", |group| {
        bench_image!(group, "portrait_4k_q75.jxl");
        bench_image!(group, "city_4k_q75.jxl");
    });

    // Web-sized VarDCT images
    suite.compare("vardct_web", |group| {
        bench_image!(group, "cafe_web_q80.jxl");
        bench_image!(group, "bicycles_web_q85.jxl");
    });

    // Modular images — where the biggest optimization gains are
    suite.compare("modular", |group| {
        bench_image!(group, "green_queen_modular_e3.jxl");
        bench_image!(group, "issue648_palette0.jxl");
    });

    // Small images — tests overhead and fast paths
    suite.compare("small", |group| {
        bench_image!(group, "green_queen_vardct_e3.jxl");
        bench_image!(group, "grayscale_patches_modular.jxl");
    });
}

zenbench::main!(bench_decode);
