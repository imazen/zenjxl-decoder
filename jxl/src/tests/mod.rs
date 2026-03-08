// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Test suite ported from libjxl C++ tests.
//!
//! These tests verify parity with the reference implementation.
//! Tests that fail indicate implementation bugs - DO NOT WEAKEN TOLERANCES.

pub mod codec_corpus;
#[cfg(feature = "threads")]
pub mod concurrency;
pub mod conformance;
pub mod coverage_boost;
pub mod decode_api;
pub mod entropy;
pub mod feature_tests;
pub mod hdr;
#[cfg(feature = "jpeg")]
pub mod jpeg_reconstruction;
pub mod parity;
pub mod streaming;
pub mod synthetic;
