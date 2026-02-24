// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    arch::wasm32::*,
    mem::MaybeUninit,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
    },
};

use crate::U32SimdVec;

use super::super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U8SimdVec, U16SimdVec};

#[derive(Clone, Copy, Debug)]
pub struct Wasm128Descriptor(());

impl Wasm128Descriptor {
    #[inline]
    pub fn from_token(_token: archmage::Wasm128Token) -> Self {
        Self(())
    }
}

/// Prepared 8-entry BF16 lookup table for wasm SIMD128.
/// Contains 8 BF16 values packed into 16 bytes (v128).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Wasm128(v128);

impl SimdDescriptor for Wasm128Descriptor {
    type F32Vec = F32VecWasm128;
    type I32Vec = I32VecWasm128;
    type U32Vec = U32VecWasm128;
    type U16Vec = U16VecWasm128;
    type U8Vec = U8VecWasm128;
    type Mask = MaskWasm128;
    type Bf16Table8 = Bf16Table8Wasm128;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    fn new() -> Option<Self> {
        // SIMD128 is a compile-time feature on wasm32.
        // If this code compiles and runs, SIMD128 is available.
        Some(Self(()))
    }

    #[inline]
    fn maybe_downgrade_256bit(self) -> Self {
        self
    }

    #[inline]
    fn maybe_downgrade_128bit(self) -> Self {
        self
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        // On wasm32, simd128 is a compile-time feature — no runtime gate needed.
        f(self)
    }
}

// Wrapper macro for wasm SIMD methods that need #[target_feature(enable = "simd128")].
// Similar to fn_neon! but for wasm32.
macro_rules! fn_wasm128 {
    {} => {};
    {$(
        fn $name:ident($this:ident: $self_ty:ty $(, $arg:ident: $ty:ty)* $(,)?) $(-> $ret:ty )?
        $body: block
    )*} => {$(
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            // On wasm32, simd128 intrinsics are always available at compile time —
            // no #[target_feature] gate or unsafe block needed.
            let $this = self;
            $body
        }
    )*};
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecWasm128(v128, Wasm128Descriptor);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
unsafe impl F32SimdVec for F32VecWasm128 {
    type Descriptor = Wasm128Descriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        Self(f32x4_splat(v), d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(f32x4_splat(0.0), d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        Self(unsafe { v128_load(mem.as_ptr() as *const v128) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        unsafe { v128_store(mem.as_mut_ptr() as *mut v128, self.0) }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<f32>]) {
        debug_assert!(dest.len() >= 2 * Self::LEN);
        // a=[a0,a1,a2,a3], b=[b0,b1,b2,b3] → [a0,b0,a1,b1, a2,b2,a3,b3]
        let lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            a.0, b.0,
        );
        let hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            a.0, b.0,
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, lo);
            v128_store(ptr.add(1), hi);
        }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<f32>]) {
        debug_assert!(dest.len() >= 3 * Self::LEN);
        // a=[a0,a1,a2,a3], b=[b0,b1,b2,b3], c=[c0,c1,c2,c3]
        // → [a0,b0,c0,a1, b1,c1,a2,b2, c2,a3,b3,c3]
        let out0 = f32x4(
            f32x4_extract_lane::<0>(a.0),
            f32x4_extract_lane::<0>(b.0),
            f32x4_extract_lane::<0>(c.0),
            f32x4_extract_lane::<1>(a.0),
        );
        let out1 = f32x4(
            f32x4_extract_lane::<1>(b.0),
            f32x4_extract_lane::<1>(c.0),
            f32x4_extract_lane::<2>(a.0),
            f32x4_extract_lane::<2>(b.0),
        );
        let out2 = f32x4(
            f32x4_extract_lane::<2>(c.0),
            f32x4_extract_lane::<3>(a.0),
            f32x4_extract_lane::<3>(b.0),
            f32x4_extract_lane::<3>(c.0),
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, out0);
            v128_store(ptr.add(1), out1);
            v128_store(ptr.add(2), out2);
        }
    }

    #[inline(always)]
    fn store_interleaved_4_uninit(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        dest: &mut [MaybeUninit<f32>],
    ) {
        debug_assert!(dest.len() >= 4 * Self::LEN);
        // Two-stage interleave: first pairs, then combine
        let ab_lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            a.0, b.0,
        );
        let ab_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            a.0, b.0,
        );
        let cd_lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            c.0, d.0,
        );
        let cd_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            c.0, d.0,
        );
        // Combine: low 64 bits of each pair
        let out0 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            ab_lo, cd_lo,
        );
        let out1 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            ab_lo, cd_lo,
        );
        let out2 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            ab_hi, cd_hi,
        );
        let out3 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            ab_hi, cd_hi,
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, out0);
            v128_store(ptr.add(1), out1);
            v128_store(ptr.add(2), out2);
            v128_store(ptr.add(3), out3);
        }
    }

    #[inline(always)]
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    ) {
        debug_assert!(dest.len() >= 8 * Self::LEN);
        // Interleave abcd and efgh separately (4-way), then interleave the two halves.
        // abcd interleave (reuse the same pattern as store_interleaved_4):
        let ab_lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            a.0, b.0,
        );
        let ab_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            a.0, b.0,
        );
        let cd_lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            c.0, d.0,
        );
        let cd_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            c.0, d.0,
        );
        let abcd_0 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            ab_lo, cd_lo,
        );
        let abcd_1 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            ab_lo, cd_lo,
        );
        let abcd_2 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            ab_hi, cd_hi,
        );
        let abcd_3 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            ab_hi, cd_hi,
        );

        // efgh interleave
        let ef_lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            e.0, f.0,
        );
        let ef_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            e.0, f.0,
        );
        let gh_lo = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            g.0, h.0,
        );
        let gh_hi = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            g.0, h.0,
        );
        let efgh_0 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            ef_lo, gh_lo,
        );
        let efgh_1 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            ef_lo, gh_lo,
        );
        let efgh_2 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            ef_hi, gh_hi,
        );
        let efgh_3 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            ef_hi, gh_hi,
        );

        // Output: [abcd_0, efgh_0, abcd_1, efgh_1, abcd_2, efgh_2, abcd_3, efgh_3]
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, abcd_0);
            v128_store(ptr.add(1), efgh_0);
            v128_store(ptr.add(2), abcd_1);
            v128_store(ptr.add(3), efgh_1);
            v128_store(ptr.add(4), abcd_2);
            v128_store(ptr.add(5), efgh_2);
            v128_store(ptr.add(6), abcd_3);
            v128_store(ptr.add(7), efgh_3);
        }
    }

    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        debug_assert!(src.len() >= 2 * Self::LEN);
        // src = [a0,b0,a1,b1, a2,b2,a3,b3]
        // SAFETY: we just checked that `src` has enough space.
        let (lo, hi) = unsafe {
            let ptr = src.as_ptr() as *const v128;
            (v128_load(ptr), v128_load(ptr.add(1)))
        };
        let a = i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(
            lo, hi,
        );
        let b = i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31>(
            lo, hi,
        );
        (Self(a, d), Self(b, d))
    }

    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        debug_assert!(src.len() >= 3 * Self::LEN);
        // src = [a0,b0,c0,a1, b1,c1,a2,b2, c2,a3,b3,c3]
        // SAFETY: we just checked that `src` has enough space.
        let (v0, v1, v2) = unsafe {
            let ptr = src.as_ptr() as *const v128;
            (v128_load(ptr), v128_load(ptr.add(1)), v128_load(ptr.add(2)))
        };
        let a = f32x4(
            f32x4_extract_lane::<0>(v0),
            f32x4_extract_lane::<3>(v0),
            f32x4_extract_lane::<2>(v1),
            f32x4_extract_lane::<1>(v2),
        );
        let b = f32x4(
            f32x4_extract_lane::<1>(v0),
            f32x4_extract_lane::<0>(v1),
            f32x4_extract_lane::<3>(v1),
            f32x4_extract_lane::<2>(v2),
        );
        let c = f32x4(
            f32x4_extract_lane::<2>(v0),
            f32x4_extract_lane::<1>(v1),
            f32x4_extract_lane::<0>(v2),
            f32x4_extract_lane::<3>(v2),
        );
        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        debug_assert!(src.len() >= 4 * Self::LEN);
        // SAFETY: we just checked that `src` has enough space.
        let (v0, v1, v2, v3) = unsafe {
            let ptr = src.as_ptr() as *const v128;
            (
                v128_load(ptr),
                v128_load(ptr.add(1)),
                v128_load(ptr.add(2)),
                v128_load(ptr.add(3)),
            )
        };
        // v0=[a0,b0,c0,d0], v1=[a1,b1,c1,d1], v2=[a2,b2,c2,d2], v3=[a3,b3,c3,d3]
        // Step 1: group pairs
        let ab_lo = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            v0, v1,
        );
        let cd_lo = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            v0, v1,
        );
        let ab_hi = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            v2, v3,
        );
        let cd_hi = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            v2, v3,
        );
        // Step 2: deinterleave pairs
        let a = i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(
            ab_lo, ab_hi,
        );
        let b = i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31>(
            ab_lo, ab_hi,
        );
        let c = i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(
            cd_lo, cd_hi,
        );
        let e = i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31>(
            cd_lo, cd_hi,
        );
        (Self(a, d), Self(b, d), Self(c, d), Self(e, d))
    }

    #[inline(always)]
    fn transpose_square(_d: Wasm128Descriptor, data: &mut [[f32; 4]], stride: usize) {
        assert!(data.len() > 3 * stride);
        // SAFETY: we just verified bounds.
        let p0 = unsafe { v128_load(data[0].as_ptr() as *const v128) };
        let p1 = unsafe { v128_load(data[stride].as_ptr() as *const v128) };
        let p2 = unsafe { v128_load(data[2 * stride].as_ptr() as *const v128) };
        let p3 = unsafe { v128_load(data[3 * stride].as_ptr() as *const v128) };

        // 4x4 transpose via two rounds of 2-element interleave
        // Stage 1: trn1/trn2 equivalent — interleave 32-bit pairs
        let t0 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27>(
            p0, p1,
        );
        let t1 = i8x16_shuffle::<4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31>(
            p0, p1,
        );
        let t2 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27>(
            p2, p3,
        );
        let t3 = i8x16_shuffle::<4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31>(
            p2, p3,
        );

        // Stage 2: combine 64-bit halves
        let r0 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            t0, t2,
        );
        let r1 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
            t1, t3,
        );
        let r2 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            t0, t2,
        );
        let r3 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
            t1, t3,
        );

        // SAFETY: we verified bounds above.
        unsafe {
            v128_store(data[0].as_mut_ptr() as *mut v128, r0);
            v128_store(data[stride].as_mut_ptr() as *mut v128, r1);
            v128_store(data[2 * stride].as_mut_ptr() as *mut v128, r2);
            v128_store(data[3 * stride].as_mut_ptr() as *mut v128, r3);
        }
    }

    crate::impl_f32_array_interface!();

    fn_wasm128! {
        fn mul_add(this: F32VecWasm128, mul: F32VecWasm128, add: F32VecWasm128) -> F32VecWasm128 {
            // No hardware FMA on wasm32 — use mul+add.
            F32VecWasm128(f32x4_add(f32x4_mul(this.0, mul.0), add.0), this.1)
        }

        fn neg_mul_add(this: F32VecWasm128, mul: F32VecWasm128, add: F32VecWasm128) -> F32VecWasm128 {
            // add - this * mul
            F32VecWasm128(f32x4_sub(add.0, f32x4_mul(this.0, mul.0)), this.1)
        }

        fn abs(this: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_abs(this.0), this.1)
        }

        fn floor(this: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_floor(this.0), this.1)
        }

        fn sqrt(this: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_sqrt(this.0), this.1)
        }

        fn neg(this: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_neg(this.0), this.1)
        }

        fn copysign(this: F32VecWasm128, sign: F32VecWasm128) -> F32VecWasm128 {
            // Select sign bit from `sign`, magnitude from `this`
            let sign_mask = i32x4_splat(0x7FFF_FFFFu32 as i32);
            let magnitude = v128_and(this.0, sign_mask);
            let sign_bit = v128_andnot(sign.0, sign_mask);
            F32VecWasm128(v128_or(magnitude, sign_bit), this.1)
        }

        fn max(this: F32VecWasm128, other: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_max(this.0, other.0), this.1)
        }

        fn min(this: F32VecWasm128, other: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_min(this.0, other.0), this.1)
        }

        fn gt(this: F32VecWasm128, other: F32VecWasm128) -> MaskWasm128 {
            MaskWasm128(f32x4_gt(this.0, other.0), this.1)
        }

        fn as_i32(this: F32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(i32x4_trunc_sat_f32x4(this.0), this.1)
        }

        fn bitcast_to_i32(this: F32VecWasm128) -> I32VecWasm128 {
            // v128 is untyped — reinterpret is a no-op.
            I32VecWasm128(this.0, this.1)
        }

        fn round_store_u8(this: F32VecWasm128, dest: &mut [u8]) {
            debug_assert!(dest.len() >= F32VecWasm128::LEN);
            let rounded = f32x4_nearest(this.0);
            let i32s = i32x4_trunc_sat_f32x4(rounded);
            // Narrow i32→i16→u8 with saturation
            let zeros = i32x4_splat(0);
            let i16s = i16x8_narrow_i32x4(i32s, zeros);
            let zeros_i16 = i16x8_splat(0);
            let u8s = u8x16_narrow_i16x8(i16s, zeros_i16);
            // First 4 bytes contain our values
            let packed = i32x4_extract_lane::<0>(u8s);
            // SAFETY: we checked dest has enough space.
            unsafe {
                *(dest.as_mut_ptr() as *mut i32) = packed;
            }
        }

        fn round_store_u16(this: F32VecWasm128, dest: &mut [u16]) {
            debug_assert!(dest.len() >= F32VecWasm128::LEN);
            let rounded = f32x4_nearest(this.0);
            let i32s = i32x4_trunc_sat_f32x4(rounded);
            // Narrow i32→u16 with unsigned saturation
            let zeros = i32x4_splat(0);
            let u16s = u16x8_narrow_i32x4(i32s, zeros);
            // First 4 u16 values = 8 bytes = one i64 lane
            let packed = i64x2_extract_lane::<0>(u16s);
            // SAFETY: we checked dest has enough space.
            unsafe {
                *(dest.as_mut_ptr() as *mut i64) = packed;
            }
        }

        fn store_f16_bits(this: F32VecWasm128, dest: &mut [u16]) {
            debug_assert!(dest.len() >= F32VecWasm128::LEN);
            // Software f32→f16 conversion (no hardware f16 on wasm).
            let mut arr = [0.0f32; 4];
            // SAFETY: arr has exactly 4 elements.
            unsafe { v128_store(arr.as_mut_ptr() as *mut v128, this.0); }
            dest[0] = crate::f16::from_f32(arr[0]).to_bits();
            dest[1] = crate::f16::from_f32(arr[1]).to_bits();
            dest[2] = crate::f16::from_f32(arr[2]).to_bits();
            dest[3] = crate::f16::from_f32(arr[3]).to_bits();
        }
    }

    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        debug_assert!(mem.len() >= Self::LEN);
        // Software f16→f32 conversion (no hardware f16 on wasm).
        let v0 = crate::f16::from_bits(mem[0]).to_f32();
        let v1 = crate::f16::from_bits(mem[1]).to_f32();
        let v2 = crate::f16::from_bits(mem[2]).to_f32();
        let v3 = crate::f16::from_bits(mem[3]).to_f32();
        Self(f32x4(v0, v1, v2, v3), d)
    }

    #[inline(always)]
    fn prepare_table_bf16_8(_d: Wasm128Descriptor, table: &[f32; 8]) -> Bf16Table8Wasm128 {
        // Convert f32 table to BF16 packed into 16 bytes.
        // BF16 is the upper 16 bits of f32.
        // SAFETY: table has exactly 8 elements.
        let table_lo = unsafe { v128_load(table.as_ptr() as *const v128) };
        let table_hi = unsafe { v128_load(table.as_ptr().add(4) as *const v128) };

        // Shift right by 16 to get BF16 in lower 16 bits of each 32-bit lane.
        let bf16_lo_u32 = u32x4_shr(table_lo, 16);
        let bf16_hi_u32 = u32x4_shr(table_hi, 16);

        // Narrow u32→u16 (takes lower 16 bits of each lane with saturation).
        // Since we shifted, values are in 0..65535 range, so no saturation occurs.
        let bf16_table = u16x8_narrow_i32x4(bf16_lo_u32, bf16_hi_u32);

        Bf16Table8Wasm128(bf16_table)
    }

    #[inline(always)]
    fn table_lookup_bf16_8(
        d: Wasm128Descriptor,
        table: Bf16Table8Wasm128,
        indices: I32VecWasm128,
    ) -> Self {
        // Build shuffle mask: for each index i (0-7), select bytes [2*i, 2*i+1]
        // from the BF16 table, place them in the high 16 bits of each 32-bit lane,
        // with low 16 bits zeroed (0x80 in swizzle gives 0).
        //
        // Output byte pattern per lane (little-endian): [0x80, 0x80, 2*i, 2*i+1]
        // As u32: 0x80 | (0x80 << 8) | (2*i << 16) | ((2*i+1) << 24)
        //       = (i << 17) | (i << 25) | 0x01008080
        let indices_u32 = indices.0;
        let shl17 = i32x4_shl(indices_u32, 17);
        let shl25 = i32x4_shl(indices_u32, 25);
        let base = i32x4_splat(0x01008080u32 as i32);
        let shuffle_mask = v128_or(v128_or(shl17, shl25), base);

        // Perform the table lookup (out of range indices give 0).
        let result = i8x16_swizzle(table.0, shuffle_mask);

        // Result has BF16 in high 16 bits of each 32-bit lane = valid f32.
        F32VecWasm128(result, d)
    }
}

impl Add<F32VecWasm128> for F32VecWasm128 {
    type Output = Self;
    fn_wasm128! {
        fn add(this: F32VecWasm128, rhs: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_add(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<F32VecWasm128> for F32VecWasm128 {
    type Output = Self;
    fn_wasm128! {
        fn sub(this: F32VecWasm128, rhs: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_sub(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<F32VecWasm128> for F32VecWasm128 {
    type Output = Self;
    fn_wasm128! {
        fn mul(this: F32VecWasm128, rhs: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_mul(this.0, rhs.0), this.1)
        }
    }
}

impl Div<F32VecWasm128> for F32VecWasm128 {
    type Output = Self;
    fn_wasm128! {
        fn div(this: F32VecWasm128, rhs: F32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_div(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<F32VecWasm128> for F32VecWasm128 {
    fn_wasm128! {
        fn add_assign(this: &mut F32VecWasm128, rhs: F32VecWasm128) {
            this.0 = f32x4_add(this.0, rhs.0);
        }
    }
}

impl SubAssign<F32VecWasm128> for F32VecWasm128 {
    fn_wasm128! {
        fn sub_assign(this: &mut F32VecWasm128, rhs: F32VecWasm128) {
            this.0 = f32x4_sub(this.0, rhs.0);
        }
    }
}

impl MulAssign<F32VecWasm128> for F32VecWasm128 {
    fn_wasm128! {
        fn mul_assign(this: &mut F32VecWasm128, rhs: F32VecWasm128) {
            this.0 = f32x4_mul(this.0, rhs.0);
        }
    }
}

impl DivAssign<F32VecWasm128> for F32VecWasm128 {
    fn_wasm128! {
        fn div_assign(this: &mut F32VecWasm128, rhs: F32VecWasm128) {
            this.0 = f32x4_div(this.0, rhs.0);
        }
    }
}

// --- I32 ---

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecWasm128(v128, Wasm128Descriptor);

impl I32SimdVec for I32VecWasm128 {
    type Descriptor = Wasm128Descriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        Self(i32x4_splat(v), d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        Self(unsafe { v128_load(mem.as_ptr() as *const v128) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        unsafe { v128_store(mem.as_mut_ptr() as *mut v128, self.0) }
    }

    fn_wasm128! {
        fn abs(this: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(i32x4_abs(this.0), this.1)
        }

        fn as_f32(this: I32VecWasm128) -> F32VecWasm128 {
            F32VecWasm128(f32x4_convert_i32x4(this.0), this.1)
        }

        fn bitcast_to_f32(this: I32VecWasm128) -> F32VecWasm128 {
            // v128 is untyped — reinterpret is a no-op.
            F32VecWasm128(this.0, this.1)
        }

        fn bitcast_to_u32(this: I32VecWasm128) -> U32VecWasm128 {
            // v128 is untyped — reinterpret is a no-op.
            U32VecWasm128(this.0, this.1)
        }

        fn gt(this: I32VecWasm128, other: I32VecWasm128) -> MaskWasm128 {
            MaskWasm128(i32x4_gt(this.0, other.0), this.1)
        }

        fn lt_zero(this: I32VecWasm128) -> MaskWasm128 {
            MaskWasm128(i32x4_lt(this.0, i32x4_splat(0)), this.1)
        }

        fn eq(this: I32VecWasm128, other: I32VecWasm128) -> MaskWasm128 {
            MaskWasm128(i32x4_eq(this.0, other.0), this.1)
        }

        fn eq_zero(this: I32VecWasm128) -> MaskWasm128 {
            MaskWasm128(i32x4_eq(this.0, i32x4_splat(0)), this.1)
        }

        fn mul_wide_take_high(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            // Multiply pairs of i32 to get i64, then take the high 32 bits.
            let lo = i64x2_extmul_low_i32x4(this.0, rhs.0);
            let hi = i64x2_extmul_high_i32x4(this.0, rhs.0);
            // Shift right by 32 to get high halves
            let lo_high = i64x2_shr(lo, 32);
            let hi_high = i64x2_shr(hi, 32);
            // Pack: extract bytes 0-3 and 8-11 from each, interleave
            let result = i8x16_shuffle::<
                0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
            >(lo_high, hi_high);
            I32VecWasm128(result, this.1)
        }
    }

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i32x4_shl(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(i32x4_shr(self.0, AMOUNT_U), self.1)
    }

    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        debug_assert!(dest.len() >= Self::LEN);
        // Extract lower 16 bits of each i32 lane and store as u16.
        let lane0 = i32x4_extract_lane::<0>(self.0) as u16;
        let lane1 = i32x4_extract_lane::<1>(self.0) as u16;
        let lane2 = i32x4_extract_lane::<2>(self.0) as u16;
        let lane3 = i32x4_extract_lane::<3>(self.0) as u16;
        dest[0] = lane0;
        dest[1] = lane1;
        dest[2] = lane2;
        dest[3] = lane3;
    }
}

impl Add<I32VecWasm128> for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn add(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(i32x4_add(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<I32VecWasm128> for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn sub(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(i32x4_sub(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<I32VecWasm128> for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn mul(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(i32x4_mul(this.0, rhs.0), this.1)
        }
    }
}

impl Neg for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn neg(this: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(i32x4_neg(this.0), this.1)
        }
    }
}

impl BitAnd<I32VecWasm128> for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn bitand(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(v128_and(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<I32VecWasm128> for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn bitor(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(v128_or(this.0, rhs.0), this.1)
        }
    }
}

impl BitXor<I32VecWasm128> for I32VecWasm128 {
    type Output = I32VecWasm128;
    fn_wasm128! {
        fn bitxor(this: I32VecWasm128, rhs: I32VecWasm128) -> I32VecWasm128 {
            I32VecWasm128(v128_xor(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<I32VecWasm128> for I32VecWasm128 {
    fn_wasm128! {
        fn add_assign(this: &mut I32VecWasm128, rhs: I32VecWasm128) {
            this.0 = i32x4_add(this.0, rhs.0);
        }
    }
}

impl SubAssign<I32VecWasm128> for I32VecWasm128 {
    fn_wasm128! {
        fn sub_assign(this: &mut I32VecWasm128, rhs: I32VecWasm128) {
            this.0 = i32x4_sub(this.0, rhs.0);
        }
    }
}

impl MulAssign<I32VecWasm128> for I32VecWasm128 {
    fn_wasm128! {
        fn mul_assign(this: &mut I32VecWasm128, rhs: I32VecWasm128) {
            this.0 = i32x4_mul(this.0, rhs.0);
        }
    }
}

impl BitAndAssign<I32VecWasm128> for I32VecWasm128 {
    fn_wasm128! {
        fn bitand_assign(this: &mut I32VecWasm128, rhs: I32VecWasm128) {
            this.0 = v128_and(this.0, rhs.0);
        }
    }
}

impl BitOrAssign<I32VecWasm128> for I32VecWasm128 {
    fn_wasm128! {
        fn bitor_assign(this: &mut I32VecWasm128, rhs: I32VecWasm128) {
            this.0 = v128_or(this.0, rhs.0);
        }
    }
}

impl BitXorAssign<I32VecWasm128> for I32VecWasm128 {
    fn_wasm128! {
        fn bitxor_assign(this: &mut I32VecWasm128, rhs: I32VecWasm128) {
            this.0 = v128_xor(this.0, rhs.0);
        }
    }
}

// --- U32 ---

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecWasm128(v128, Wasm128Descriptor);

impl U32SimdVec for U32VecWasm128 {
    type Descriptor = Wasm128Descriptor;

    const LEN: usize = 4;

    fn_wasm128! {
        fn bitcast_to_i32(this: U32VecWasm128) -> I32VecWasm128 {
            // v128 is untyped — reinterpret is a no-op.
            I32VecWasm128(this.0, this.1)
        }
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        Self(u32x4_shr(self.0, AMOUNT_U), self.1)
    }
}

// --- U8 ---

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecWasm128(v128, Wasm128Descriptor);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
unsafe impl U8SimdVec for U8VecWasm128 {
    type Descriptor = Wasm128Descriptor;
    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        Self(unsafe { v128_load(mem.as_ptr() as *const v128) }, d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        Self(u8x16_splat(v), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        unsafe { v128_store(mem.as_mut_ptr() as *mut v128, self.0) }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<u8>]) {
        debug_assert!(dest.len() >= 2 * Self::LEN);
        // Interleave bytes: [a0,b0,a1,b1,...,a15,b15]
        let lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
            a.0, b.0,
        );
        let hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
            a.0, b.0,
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, lo);
            v128_store(ptr.add(1), hi);
        }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<u8>]) {
        debug_assert!(dest.len() >= 3 * Self::LEN);
        // 3-way byte interleave: use array-based approach for clarity
        let mut a_arr = [0u8; 16];
        let mut b_arr = [0u8; 16];
        let mut c_arr = [0u8; 16];
        // SAFETY: arrays are large enough.
        unsafe {
            v128_store(a_arr.as_mut_ptr() as *mut v128, a.0);
            v128_store(b_arr.as_mut_ptr() as *mut v128, b.0);
            v128_store(c_arr.as_mut_ptr() as *mut v128, c.0);
        }
        let mut out = [0u8; 48];
        for i in 0..16 {
            out[3 * i] = a_arr[i];
            out[3 * i + 1] = b_arr[i];
            out[3 * i + 2] = c_arr[i];
        }
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, v128_load(out.as_ptr() as *const v128));
            v128_store(ptr.add(1), v128_load(out.as_ptr().add(16) as *const v128));
            v128_store(ptr.add(2), v128_load(out.as_ptr().add(32) as *const v128));
        }
    }

    #[inline(always)]
    fn store_interleaved_4_uninit(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        dest: &mut [MaybeUninit<u8>],
    ) {
        debug_assert!(dest.len() >= 4 * Self::LEN);
        // 4-way byte interleave: two rounds of 2-way interleave
        let ab_lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
            a.0, b.0,
        );
        let ab_hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
            a.0, b.0,
        );
        let cd_lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
            c.0, d.0,
        );
        let cd_hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
            c.0, d.0,
        );
        // Now interleave the pairs (treating as u16 elements)
        let out0 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
            ab_lo, cd_lo,
        );
        let out1 = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
            ab_lo, cd_lo,
        );
        let out2 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
            ab_hi, cd_hi,
        );
        let out3 = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
            ab_hi, cd_hi,
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, out0);
            v128_store(ptr.add(1), out1);
            v128_store(ptr.add(2), out2);
            v128_store(ptr.add(3), out3);
        }
    }
}

// --- U16 ---

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecWasm128(v128, Wasm128Descriptor);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
unsafe impl U16SimdVec for U16VecWasm128 {
    type Descriptor = Wasm128Descriptor;
    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        Self(unsafe { v128_load(mem.as_ptr() as *const v128) }, d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        Self(u16x8_splat(v), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        debug_assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space.
        unsafe { v128_store(mem.as_mut_ptr() as *mut v128, self.0) }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<u16>]) {
        debug_assert!(dest.len() >= 2 * Self::LEN);
        // Interleave u16 elements
        let lo = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
            a.0, b.0,
        );
        let hi = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
            a.0, b.0,
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, lo);
            v128_store(ptr.add(1), hi);
        }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<u16>]) {
        debug_assert!(dest.len() >= 3 * Self::LEN);
        // 3-way u16 interleave: use array-based approach
        let mut a_arr = [0u16; 8];
        let mut b_arr = [0u16; 8];
        let mut c_arr = [0u16; 8];
        // SAFETY: arrays are large enough.
        unsafe {
            v128_store(a_arr.as_mut_ptr() as *mut v128, a.0);
            v128_store(b_arr.as_mut_ptr() as *mut v128, b.0);
            v128_store(c_arr.as_mut_ptr() as *mut v128, c.0);
        }
        let mut out = [0u16; 24];
        for i in 0..8 {
            out[3 * i] = a_arr[i];
            out[3 * i + 1] = b_arr[i];
            out[3 * i + 2] = c_arr[i];
        }
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, v128_load(out.as_ptr() as *const v128));
            v128_store(ptr.add(1), v128_load(out.as_ptr().add(8) as *const v128));
            v128_store(ptr.add(2), v128_load(out.as_ptr().add(16) as *const v128));
        }
    }

    #[inline(always)]
    fn store_interleaved_4_uninit(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        dest: &mut [MaybeUninit<u16>],
    ) {
        debug_assert!(dest.len() >= 4 * Self::LEN);
        // 4-way u16 interleave: two rounds of 2-way
        let ab_lo = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
            a.0, b.0,
        );
        let ab_hi = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
            a.0, b.0,
        );
        let cd_lo = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
            c.0, d.0,
        );
        let cd_hi = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
            c.0, d.0,
        );
        // Interleave the 32-bit pairs
        let out0 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            ab_lo, cd_lo,
        );
        let out1 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            ab_lo, cd_lo,
        );
        let out2 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
            ab_hi, cd_hi,
        );
        let out3 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
            ab_hi, cd_hi,
        );
        // SAFETY: we just checked that `dest` has enough space.
        unsafe {
            let ptr = dest.as_mut_ptr() as *mut v128;
            v128_store(ptr, out0);
            v128_store(ptr.add(1), out1);
            v128_store(ptr.add(2), out2);
            v128_store(ptr.add(3), out3);
        }
    }
}

// --- Mask ---

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskWasm128(v128, Wasm128Descriptor);

impl SimdMask for MaskWasm128 {
    type Descriptor = Wasm128Descriptor;

    fn_wasm128! {
        fn if_then_else_f32(
            this: MaskWasm128,
            if_true: F32VecWasm128,
            if_false: F32VecWasm128,
        ) -> F32VecWasm128 {
            // v128_bitselect(v1, v2, c) = (v1 & c) | (v2 & ~c)
            F32VecWasm128(v128_bitselect(if_true.0, if_false.0, this.0), this.1)
        }

        fn if_then_else_i32(
            this: MaskWasm128,
            if_true: I32VecWasm128,
            if_false: I32VecWasm128,
        ) -> I32VecWasm128 {
            I32VecWasm128(v128_bitselect(if_true.0, if_false.0, this.0), this.1)
        }

        fn maskz_i32(this: MaskWasm128, v: I32VecWasm128) -> I32VecWasm128 {
            // v & ~mask (same semantics as NEON's vbicq)
            I32VecWasm128(v128_andnot(v.0, this.0), this.1)
        }

        fn andnot(this: MaskWasm128, rhs: MaskWasm128) -> MaskWasm128 {
            // !self & rhs = rhs & ~self
            MaskWasm128(v128_andnot(rhs.0, this.0), this.1)
        }

        fn all(this: MaskWasm128) -> bool {
            i32x4_all_true(this.0)
        }
    }
}

impl BitAnd<MaskWasm128> for MaskWasm128 {
    type Output = MaskWasm128;
    fn_wasm128! {
        fn bitand(this: MaskWasm128, rhs: MaskWasm128) -> MaskWasm128 {
            MaskWasm128(v128_and(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<MaskWasm128> for MaskWasm128 {
    type Output = MaskWasm128;
    fn_wasm128! {
        fn bitor(this: MaskWasm128, rhs: MaskWasm128) -> MaskWasm128 {
            MaskWasm128(v128_or(this.0, rhs.0), this.1)
        }
    }
}
