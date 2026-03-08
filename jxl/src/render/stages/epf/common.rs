// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::BLOCK_DIM;
use crate::features::epf::SigmaSource;

/// Sigma row source for EPF processing.
/// Either a slice from the variable sigma image, or a constant value.
#[derive(Clone, Copy)]
pub(super) enum SigmaRow<'a> {
    Variable(&'a [f32]),
    Constant(f32),
}

impl SigmaSource {
    /// Get the sigma row for a given y position.
    #[inline(always)]
    pub(super) fn row(&self, y: usize) -> SigmaRow<'_> {
        match self {
            SigmaSource::Variable(image) => SigmaRow::Variable(image.row(y)),
            SigmaSource::Constant(sigma) => SigmaRow::Constant(*sigma),
        }
    }
}

#[inline(always)]
pub(super) fn prepare_sad_mul_storage(x: usize, y: usize, sm: f32, bsm: f32) -> [f32; 24] {
    let mut sad_mul_storage = [bsm; 24];
    if ![0, BLOCK_DIM - 1].contains(&(y % BLOCK_DIM)) {
        for (i, s) in sad_mul_storage.iter_mut().enumerate().take(16) {
            if ![0, BLOCK_DIM - 1].contains(&((x + i) % BLOCK_DIM)) {
                *s = sm;
            }
        }
    }
    sad_mul_storage
}

/// Get the sigma value for a single pixel at absolute x position.
#[inline(always)]
pub(super) fn get_sigma_scalar(x: usize, row_sigma: SigmaRow<'_>) -> f32 {
    match row_sigma {
        SigmaRow::Constant(sigma) => sigma,
        SigmaRow::Variable(row) => row[x / BLOCK_DIM],
    }
}
