// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! The decoder's clock. Its readings only feed phase timing (printed when
//! `JXL_PHASE_TIMING` is set). On `wasm32-unknown-unknown`, where
//! `std::time::Instant::now()` panics, a stand-in never reads a clock and
//! reports zero elapsed time; everywhere else this is `std::time::Instant`.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl Instant {
    pub(crate) fn now() -> Self {
        Instant
    }

    pub(crate) fn elapsed(&self) -> core::time::Duration {
        core::time::Duration::ZERO
    }
}
