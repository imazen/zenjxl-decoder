// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG reconstruction from JXL containers with JBRD boxes.
//!
//! When a JPEG file is losslessly transcoded to JXL, the JXL container
//! includes a `jbrd` (JPEG Bitstream Reconstruction Data) box alongside
//! the codestream. This module decodes that box and reconstructs the
//! original JPEG byte-exactly.

pub(crate) mod data;
mod jbrd;
mod writer;

pub use jbrd::decode_jbrd;
pub use writer::write_jpeg;
