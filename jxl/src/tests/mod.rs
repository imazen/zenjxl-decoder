// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Test suite ported from libjxl C++ tests.
//!
//! These tests verify parity with the reference implementation.
//! Tests that fail indicate implementation bugs - DO NOT WEAKEN TOLERANCES.

pub mod codec_corpus;
pub mod decode_api;
pub mod entropy;
pub mod parity;
pub mod streaming;
