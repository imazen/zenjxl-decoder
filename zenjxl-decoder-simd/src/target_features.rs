// Copyright (c) Imazen LLC. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Descriptors constructed from the calling function's CPU features.
//!
//! The inherent `from_context()` constructors and these free constructors are
//! equivalent. They perform no runtime detection. Baseline callers use
//! [`crate::SimdDescriptor::new`]; E0133 means the caller lacks a statically
//! proven native feature context. Do not add an unsafe block to bypass that diagnostic.
//!
//! Complete required feature sets:
//! - `_v2`: sse, sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b.
//! - `_v3`: v2 plus avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe.
//! - `_v4`: v3 plus pclmulqdq, aes, avx512f, avx512bw, avx512cd, avx512dq, avx512vl.
//! - `_neon`: neon. `_wasm128`: simd128.
//!
//! Descriptors retain the existing vector traits' array/slice interfaces. Their
//! private token storage and dependency version do not cross user API boundaries.
//! On wasm32, Rust also permits safe baseline calls: the engine validates module
//! features at load time. That platform does not apply the native E0133 rule;
//! SIMD128 functions still require an engine supporting SIMD128.
//! No speed or inlining guarantee is made.

#![cfg_attr(
    all(target_arch = "x86_64", feature = "avx"),
    doc = r#"
A baseline caller cannot construct an AVX descriptor without runtime detection:

```compile_fail,E0133
let _ = zenjxl_decoder_simd::target_features::descriptor_v3();
```

Safe composition inside a matching feature function needs no token parameter:

```
use zenjxl_decoder_simd::{F32SimdVec, SimdDescriptor, AvxDescriptor};
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe")]
fn add(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let d = AvxDescriptor::from_context();
    let result = <AvxDescriptor as SimdDescriptor>::F32Vec::load(d, a)
        + <AvxDescriptor as SimdDescriptor>::F32Vec::load(d, b);
    let mut out = [0.0; 8];
    result.store(&mut out);
    out
}
```
"#
)]

/// Construct an SSE4.2 descriptor from a v2 feature context.
#[cfg(all(target_arch = "x86_64", feature = "sse42"))]
#[archmage::rite(v2)]
pub fn descriptor_v2() -> crate::Sse42Descriptor {
    crate::Sse42Descriptor::from_context()
}

/// Construct an AVX2 descriptor from a v3 feature context.
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[archmage::rite(v3)]
pub fn descriptor_v3() -> crate::AvxDescriptor {
    crate::AvxDescriptor::from_context()
}

/// Construct an AVX-512 descriptor from a v4 feature context.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::rite(v4)]
pub fn descriptor_v4() -> crate::Avx512Descriptor {
    crate::Avx512Descriptor::from_context()
}

/// Construct a NEON descriptor from a neon feature context.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[archmage::rite(neon)]
pub fn descriptor_neon() -> crate::NeonDescriptor {
    crate::NeonDescriptor::from_context()
}

/// Construct a SIMD128 descriptor; WASM also permits safe baseline calls.
/// The engine validates SIMD128 support when loading the module.
///
/// ```
/// let _ = zenjxl_decoder_simd::target_features::descriptor_wasm128();
/// ```
#[cfg(all(target_arch = "wasm32", feature = "wasm128"))]
#[archmage::rite(wasm128)]
pub fn descriptor_wasm128() -> crate::Wasm128Descriptor {
    crate::Wasm128Descriptor::from_context()
}
