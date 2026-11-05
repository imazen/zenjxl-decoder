// Copyright (c) Imazen LLC.
// Use of this source code is governed by a BSD-style license in LICENSE.

//! Whole-image decode cost of reference spline point placement.
//! Both arms use identical default options and run without the threads feature.
use zenbench::prelude::*;

fn main() {
    let result = zenbench::run(|suite| {
        for (name, input) in [
            (
                "basic-control",
                include_bytes!("../../../zenjxl-decoder/resources/test/basic.jxl").as_slice(),
            ),
            (
                "layered-noise-spline",
                include_bytes!(
                    "../../../zenjxl-decoder/resources/test/multiple_layers_noise_spline.jxl"
                )
                .as_slice(),
            ),
        ] {
            let before = baseline::decode(input).expect("baseline decode");
            let after = candidate::decode(input).expect("candidate decode");
            assert_eq!(
                (
                    before.width,
                    before.height,
                    before.channels,
                    before.data.len()
                ),
                (after.width, after.height, after.channels, after.data.len()),
            );
            // The geometry repair changes spline pixels. The independent C++
            // golden regression checks correctness; this compares decode cost.
            suite.compare(format!("spline-spacing/{name}"), move |group| {
                group.bench("main-6d060afb", move |b| {
                    b.with_input(move || input)
                        .run(|input| baseline::decode(input).expect("baseline decode"))
                });
                group.bench("reference-spacing", move |b| {
                    b.with_input(move || input)
                        .run(|input| candidate::decode(input).expect("candidate decode"))
                });
            });
        }
    });
    zenbench::postprocess_result(&result);
}
