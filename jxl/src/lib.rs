// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg_attr(not(feature = "allow-unsafe"), forbid(unsafe_code))]
#![cfg_attr(feature = "allow-unsafe", deny(unsafe_code))]
// Internal modules expose items that are used across the crate but not
// externally.  Some items are kept for future use (container, color TFs,
// render stages) — suppress dead-code noise so clippy -D warnings passes.
#![allow(dead_code, unused_imports)]

pub mod api;
pub use api::{decode, decode_with, read_header, read_header_with};
pub(crate) mod bit_reader;
pub(crate) mod color;
pub(crate) mod container;
pub(crate) mod entropy_coding;
pub(crate) mod error;
pub(crate) mod features;
pub(crate) mod frame;
pub(crate) mod headers;
pub(crate) mod icc;
pub(crate) mod image;
pub(crate) mod render;
pub(crate) mod transforms;
pub(crate) mod util;

#[cfg(feature = "jpeg")]
pub(crate) mod jpeg;

#[cfg(test)]
mod tests;

// TODO: Move these to a more appropriate location.
const GROUP_DIM: usize = 256;
const BLOCK_DIM: usize = 8;
const BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;
#[allow(clippy::excessive_precision)]
const MIN_SIGMA: f32 = -3.90524291751269967465540850526868;
