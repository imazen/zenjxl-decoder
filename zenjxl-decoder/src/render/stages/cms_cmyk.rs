// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! CMS-based CMYK to RGB conversion stage.
//!
//! This stage uses a caller-provided Color Management System to convert CMYK data
//! to RGB using an embedded ICC profile. This is needed for proper CMYK
//! color reproduction, as simple mathematical conversion (R=C*K) does not
//! account for ink characteristics, dot gain, and color gamut mapping that
//! ICC profiles encode.

use std::sync::Mutex;

use crate::api::JxlCmsTransformer;
use crate::render::RenderPipelineInPlaceStage;

/// CMS-based CMYK to RGB conversion stage.
///
/// This stage:
/// 1. Reads CMY from channels 0, 1, 2 and K from the specified extra channel
/// 2. Packs them into CMYK format (4 channels)
/// 3. Transforms through the ICC profile using the supplied CMS
/// 4. Outputs RGB to channels 0, 1, 2
pub struct CmsCmykToRgbStage {
    /// Black channel index (offset from 3)
    black_c: usize,
    /// The CMS transformer for CMYK → sRGB conversion (uses Mutex for interior mutability)
    transformer: Mutex<Box<dyn JxlCmsTransformer + Send + Sync>>,
}

impl std::fmt::Display for CmsCmykToRgbStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CMS CMYK to RGB stage for channel {}", self.black_c)
    }
}

impl CmsCmykToRgbStage {
    pub fn new(
        black_c_offset: usize,
        transformer: Box<dyn JxlCmsTransformer + Send + Sync>,
    ) -> Self {
        Self {
            black_c: 3 + black_c_offset,
            transformer: Mutex::new(transformer),
        }
    }
}

impl RenderPipelineInPlaceStage for CmsCmykToRgbStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3 || c == self.black_c
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut (dyn std::any::Any + Send)>,
    ) {
        let [row_c, row_m, row_y, row_k] = row else {
            panic!(
                "incorrect number of channels; expected 4, found {}",
                row.len()
            );
        };

        assert!(
            xsize <= row_c.len()
                && xsize <= row_m.len()
                && xsize <= row_y.len()
                && xsize <= row_k.len()
        );

        // Allocate buffers for CMYK input and RGB output
        // In moxcms, CMYK uses Layout::Rgba (4 channels)
        let mut cmyk_input = vec![0.0f32; xsize * 4];
        let mut rgb_output = vec![0.0f32; xsize * 3];

        // JxlCmsTransformer takes JPEG XL reflectance values: 1 = no ink.
        // Backend-specific ICC inversion and percentage conversion belong in
        // the backend, so custom implementations see the documented contract.
        for idx in 0..xsize {
            cmyk_input[idx * 4] = row_c[idx];
            cmyk_input[idx * 4 + 1] = row_m[idx];
            cmyk_input[idx * 4 + 2] = row_y[idx];
            cmyk_input[idx * 4 + 3] = row_k[idx];
        }

        // Transform CMYK → RGB through ICC profile
        let mut transformer = self.transformer.lock().unwrap();

        match transformer.do_transform(&cmyk_input, &mut rgb_output) {
            Ok(()) => {
                // Unpack RGB output
                for idx in 0..xsize {
                    row_c[idx] = rgb_output[idx * 3];
                    row_m[idx] = rgb_output[idx * 3 + 1];
                    row_y[idx] = rgb_output[idx * 3 + 2];
                }
            }
            Err(_e) => {
                // Fall back to simple CMYK math if transform fails
                for idx in 0..xsize {
                    let k = row_k[idx];
                    row_c[idx] *= k;
                    row_m[idx] *= k;
                    row_y[idx] *= k;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReflectanceCms;
    impl JxlCmsTransformer for ReflectanceCms {
        fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> crate::error::Result<()> {
            assert_eq!(input, [1.0, 0.5, 0.25, 0.75]);
            output.copy_from_slice(&[0.3, 0.4, 0.5]);
            Ok(())
        }
        fn do_transform_inplace(&mut self, _: &mut [f32]) -> crate::error::Result<()> {
            panic!("CMYK to RGB requires separate buffers")
        }
    }

    #[test]
    fn custom_cms_receives_documented_reflectance() {
        let stage = CmsCmykToRgbStage::new(0, Box::new(ReflectanceCms));
        let mut c = [1.0];
        let mut m = [0.5];
        let mut y = [0.25];
        let mut k = [0.75];
        stage.process_row_chunk((0, 0), 1, &mut [&mut c, &mut m, &mut y, &mut k], None);
        assert_eq!([c[0], m[0], y[0], k[0]], [0.3, 0.4, 0.5, 0.75]);
    }
}
