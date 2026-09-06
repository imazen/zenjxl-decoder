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
    ($suite:expr, $group:literal, $name:literal) => {{
        let (data, pixels) = load_image($name);
        $suite.compare(concat!($group, "/", $name), |group| {
            group.throughput(Throughput::Elements(pixels));
            group.throughput_unit("pixels");
            group.bench("decode", move |b| {
                #[cfg(target_arch = "aarch64")]
                archmage::NeonToken::dangerously_disable_token_process_wide(false)
                    .expect("restore NEON for normal decode benchmarks");
                b.iter(|| zenjxl_decoder::decode(zenbench::black_box(&data)).unwrap())
            });
        });
    }};
}

fn bench_decode(suite: &mut Suite) {
    #[cfg(target_arch = "aarch64")]
    for name in [
        "cafe_web_q80.jxl",
        "portrait_4k_q75.jxl",
        "green_queen_modular_e3.jxl",
        "gray_alpha_lossless.jxl",
    ] {
        let (data, pixels) = load_image(name);
        suite.compare(format!("decode_tiers/{name}"), |g| {
            g.throughput(Throughput::Elements(pixels));
            for (label, enabled) in [("neon", true), ("scalar", false)] {
                let data = data.clone();
                g.bench(label, move |b| {
                    b.with_input(move || {
                        archmage::NeonToken::dangerously_disable_token_process_wide(!enabled)
                            .expect("benchmark requires toggleable NEON");
                    })
                    .run(|_| zenjxl_decoder::decode(zenbench::black_box(&data)).unwrap())
                });
            }
        });
    }

    // 4K VarDCT images — main decode performance target
    bench_image!(suite, "vardct_4k", "portrait_4k_q75.jxl");
    bench_image!(suite, "vardct_4k", "city_4k_q75.jxl");

    // Web-sized VarDCT images
    bench_image!(suite, "vardct_web", "cafe_web_q80.jxl");
    bench_image!(suite, "vardct_web", "bicycles_web_q85.jxl");

    // Modular images — where the biggest optimization gains are.
    //
    // Widened 2026-07-28: a NEON-vs-scalar A/B (default features vs
    // --no-default-features) showed modular gains almost nothing from SIMD
    // (1.02-1.05x, vs 1.71-1.83x for VarDCT) and ONE image,
    // grayscale_patches_modular, came out 14% SLOWER with SIMD on. Two images
    // could not distinguish "that image" from "the mode", so every modular
    // fixture in resources/test is now benched.
    bench_image!(suite, "modular", "green_queen_modular_e3.jxl");
    bench_image!(suite, "modular", "issue648_palette0.jxl");
    bench_image!(suite, "modular", "grayscale_patches_modular.jxl");
    bench_image!(suite, "modular", "small_grayscale_patches_modular.jxl");
    bench_image!(
        suite,
        "modular",
        "small_grayscale_patches_modular_with_icc.jxl"
    );
    bench_image!(suite, "modular", "gray_alpha_lossless.jxl");
    bench_image!(suite, "modular", "3x3_srgb_lossless.jxl");
    bench_image!(suite, "modular", "3x3a_srgb_lossless.jxl");

    // Small images — tests overhead and fast paths
    bench_image!(suite, "small", "green_queen_vardct_e3.jxl");
    bench_image!(suite, "small", "grayscale_patches_modular.jxl");
}

zenbench::main!(bench_decode);
