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
    // Local checkout when present (the normal dev case, no network), otherwise
    // downloaded on demand via codec-corpus. `resources/test/` is not packaged
    // in the published crate (#8).
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test");
    #[cfg(not(target_arch = "wasm32"))]
    if !local.is_dir() {
        return codec_corpus::Corpus::new()
            .expect("initialize codec-corpus to download bench fixtures")
            .github_repo(
                "imazen/zenjxl-decoder",
                "zenjxl-decoder/resources/test",
                "main",
            )
            .expect("download zenjxl-decoder bench fixtures via codec-corpus");
    }
    local
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

    // Modular images — where the biggest optimization gains are.
    //
    // Widened 2026-07-28: a NEON-vs-scalar A/B (default features vs
    // --no-default-features) showed modular gains almost nothing from SIMD
    // (1.02-1.05x, vs 1.71-1.83x for VarDCT) and ONE image,
    // grayscale_patches_modular, came out 14% SLOWER with SIMD on. Two images
    // could not distinguish "that image" from "the mode", so every modular
    // fixture in resources/test is now benched.
    suite.compare("modular", |group| {
        bench_image!(group, "green_queen_modular_e3.jxl");
        bench_image!(group, "issue648_palette0.jxl");
        bench_image!(group, "grayscale_patches_modular.jxl");
        bench_image!(group, "small_grayscale_patches_modular.jxl");
        bench_image!(group, "small_grayscale_patches_modular_with_icc.jxl");
        bench_image!(group, "gray_alpha_lossless.jxl");
        bench_image!(group, "3x3_srgb_lossless.jxl");
        bench_image!(group, "3x3a_srgb_lossless.jxl");
    });

    // Small images — tests overhead and fast paths
    suite.compare("small", |group| {
        bench_image!(group, "green_queen_vardct_e3.jxl");
        bench_image!(group, "grayscale_patches_modular.jxl");
    });
}

zenbench::main!(bench_decode);
