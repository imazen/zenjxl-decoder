// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]

use crate::{impl_f32_array_interface, U32SimdVec};

use super::super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U16SimdVec, U8SimdVec};
use std::{
    mem::MaybeUninit,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
    },
};

use archmage::{NeonToken, SimdToken};
use magetypes::simd::generic::{f32x4, i32x4, u16x8, u32x4, u8x16};

type Token = NeonToken;

/// NEON descriptor wrapping an `archmage::NeonToken`.
/// Zero-sized; the token proves CPU support at construction time.
#[derive(Clone, Copy, Debug)]
pub struct NeonDescriptor(Token);

impl NeonDescriptor {
    /// Creates a NeonDescriptor without runtime checks.
    ///
    /// # Safety
    /// The caller must guarantee that the NEON target feature is available.
    #[allow(unsafe_code)]
    pub unsafe fn new_unchecked() -> Self {
        Self(Token::summon().unwrap())
    }

    #[inline(always)]
    fn token(self) -> Token {
        self.0
    }
}

/// Prepared 8-entry BF16 lookup table for NEON. Array fallback.
#[derive(Clone, Copy, Debug)]
pub struct Bf16Table8Neon([f32; 8]);

impl SimdDescriptor for NeonDescriptor {
    type F32Vec = F32VecNeon;
    type I32Vec = I32VecNeon;
    type U32Vec = U32VecNeon;
    type U8Vec = U8VecNeon;
    type U16Vec = U16VecNeon;
    type Mask = MaskNeon;
    type Bf16Table8 = Bf16Table8Neon;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }

    fn new() -> Option<Self> {
        Token::summon().map(NeonDescriptor)
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

// ============================================================================
// F32VecNeon
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecNeon(f32x4<Token>);

impl Add for F32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F32VecNeon(self.0 + rhs.0)
    }
}

impl Sub for F32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32VecNeon(self.0 - rhs.0)
    }
}

impl Mul for F32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32VecNeon(self.0 * rhs.0)
    }
}

impl Div for F32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F32VecNeon(self.0 / rhs.0)
    }
}

impl AddAssign for F32VecNeon {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

impl SubAssign for F32VecNeon {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

impl MulAssign for F32VecNeon {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

impl DivAssign for F32VecNeon {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = self.0 / rhs.0;
    }
}

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
#[allow(unsafe_code)]
unsafe impl F32SimdVec for F32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        F32VecNeon(f32x4::splat(d.token(), v))
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        F32VecNeon(f32x4::zero(d.token()))
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        F32VecNeon(f32x4::from_slice(d.token(), mem))
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        let arr = self.0.to_array();
        mem[..4].copy_from_slice(&arr);
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        // magetypes mul_add: self * a + b
        F32VecNeon(self.0.mul_add(mul.0, add.0))
    }

    #[inline(always)]
    fn neg_mul_add(self, mul: Self, add: Self) -> Self {
        // Compute add - self * mul = (-self) * mul + add
        F32VecNeon((-self.0).mul_add(mul.0, add.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        F32VecNeon(self.0.abs())
    }

    #[inline(always)]
    fn floor(self) -> Self {
        F32VecNeon(self.0.floor())
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        F32VecNeon(self.0.sqrt())
    }

    #[inline(always)]
    fn neg(self) -> Self {
        F32VecNeon(-self.0)
    }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        let a = self.0.to_array();
        let s = sign.0.to_array();
        let out: [f32; 4] = core::array::from_fn(|i| {
            f32::from_bits((a[i].to_bits() & 0x7FFF_FFFF) | (s[i].to_bits() & 0x8000_0000))
        });
        F32VecNeon(f32x4::from_array(Token::summon().unwrap(), out))
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        F32VecNeon(self.0.max(other.0))
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        F32VecNeon(self.0.min(other.0))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> MaskNeon {
        MaskNeon(self.0.simd_gt(other.0))
    }

    #[inline(always)]
    fn as_i32(self) -> I32VecNeon {
        I32VecNeon(self.0.to_i32_round())
    }

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecNeon {
        I32VecNeon(self.0.bitcast_to_i32())
    }

    #[inline(always)]
    fn prepare_table_bf16_8(_d: NeonDescriptor, table: &[f32; 8]) -> Bf16Table8Neon {
        Bf16Table8Neon(*table)
    }

    #[inline(always)]
    fn table_lookup_bf16_8(d: NeonDescriptor, table: Bf16Table8Neon, indices: I32VecNeon) -> Self {
        let idx = indices.0.to_array();
        let out: [f32; 4] = core::array::from_fn(|i| table.0[idx[i] as usize]);
        F32VecNeon(f32x4::from_array(d.token(), out))
    }

    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        assert!(dest.len() >= Self::LEN);
        let rounded = self.0.round();
        let arr = rounded.to_array();
        for i in 0..4 {
            dest[i] = arr[i].clamp(0.0, 255.0) as u8;
        }
    }

    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        let rounded = self.0.round();
        let arr = rounded.to_array();
        for i in 0..4 {
            dest[i] = arr[i].clamp(0.0, 65535.0) as u16;
        }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<f32>]) {
        assert!(dest.len() >= 2 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        for i in 0..4 {
            dest[2 * i].write(aa[i]);
            dest[2 * i + 1].write(bb[i]);
        }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<f32>]) {
        assert!(dest.len() >= 3 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        let cc = c.0.to_array();
        for i in 0..4 {
            dest[3 * i].write(aa[i]);
            dest[3 * i + 1].write(bb[i]);
            dest[3 * i + 2].write(cc[i]);
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
        assert!(dest.len() >= 4 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        let cc = c.0.to_array();
        let dd = d.0.to_array();
        for i in 0..4 {
            dest[4 * i].write(aa[i]);
            dest[4 * i + 1].write(bb[i]);
            dest[4 * i + 2].write(cc[i]);
            dest[4 * i + 3].write(dd[i]);
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
        assert!(dest.len() >= 8 * Self::LEN);
        let vecs = [
            a.0.to_array(),
            b.0.to_array(),
            c.0.to_array(),
            d.0.to_array(),
            e.0.to_array(),
            f.0.to_array(),
            g.0.to_array(),
            h.0.to_array(),
        ];
        for i in 0..4 {
            for j in 0..8 {
                dest[8 * i + j] = vecs[j][i];
            }
        }
    }

    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        assert!(src.len() >= 2 * Self::LEN);
        let a: [f32; 4] = core::array::from_fn(|i| src[2 * i]);
        let b: [f32; 4] = core::array::from_fn(|i| src[2 * i + 1]);
        (
            F32VecNeon(f32x4::from_array(d.token(), a)),
            F32VecNeon(f32x4::from_array(d.token(), b)),
        )
    }

    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        assert!(src.len() >= 3 * Self::LEN);
        let a: [f32; 4] = core::array::from_fn(|i| src[3 * i]);
        let b: [f32; 4] = core::array::from_fn(|i| src[3 * i + 1]);
        let c: [f32; 4] = core::array::from_fn(|i| src[3 * i + 2]);
        (
            F32VecNeon(f32x4::from_array(d.token(), a)),
            F32VecNeon(f32x4::from_array(d.token(), b)),
            F32VecNeon(f32x4::from_array(d.token(), c)),
        )
    }

    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        assert!(src.len() >= 4 * Self::LEN);
        let a: [f32; 4] = core::array::from_fn(|i| src[4 * i]);
        let b: [f32; 4] = core::array::from_fn(|i| src[4 * i + 1]);
        let c: [f32; 4] = core::array::from_fn(|i| src[4 * i + 2]);
        let dv: [f32; 4] = core::array::from_fn(|i| src[4 * i + 3]);
        (
            F32VecNeon(f32x4::from_array(d.token(), a)),
            F32VecNeon(f32x4::from_array(d.token(), b)),
            F32VecNeon(f32x4::from_array(d.token(), c)),
            F32VecNeon(f32x4::from_array(d.token(), dv)),
        )
    }

    impl_f32_array_interface!();

    #[inline(always)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        assert!(data.len() > stride * 3);
        let mut rows: [f32x4<Token>; 4] =
            core::array::from_fn(|i| f32x4::load(d.token(), &data[i * stride]));
        f32x4::transpose_4x4(&mut rows);
        for i in 0..4 {
            rows[i].store(&mut data[i * stride]);
        }
    }

    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        let out: [f32; 4] = core::array::from_fn(|i| crate::f16::from_bits(mem[i]).to_f32());
        F32VecNeon(f32x4::from_array(d.token(), out))
    }

    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        let arr = self.0.to_array();
        for i in 0..4 {
            dest[i] = crate::f16::from_f32(arr[i]).to_bits();
        }
    }
}

// ============================================================================
// I32VecNeon
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecNeon(i32x4<Token>);

impl Add for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        I32VecNeon(self.0 + rhs.0)
    }
}

impl Sub for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        I32VecNeon(self.0 - rhs.0)
    }
}

impl Mul for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        I32VecNeon(self.0 * rhs.0)
    }
}

impl Neg for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        I32VecNeon(-self.0)
    }
}

impl BitAnd for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        I32VecNeon(self.0 & rhs.0)
    }
}

impl BitOr for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        I32VecNeon(self.0 | rhs.0)
    }
}

impl BitXor for I32VecNeon {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        I32VecNeon(self.0 ^ rhs.0)
    }
}

impl AddAssign for I32VecNeon {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

impl SubAssign for I32VecNeon {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

impl MulAssign for I32VecNeon {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

impl BitAndAssign for I32VecNeon {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = self.0 & rhs.0;
    }
}

impl BitOrAssign for I32VecNeon {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = self.0 | rhs.0;
    }
}

impl BitXorAssign for I32VecNeon {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = self.0 ^ rhs.0;
    }
}

impl I32SimdVec for I32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        I32VecNeon(i32x4::splat(d.token(), v))
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        I32VecNeon(i32x4::from_slice(d.token(), mem))
    }

    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        assert!(mem.len() >= Self::LEN);
        let arr = self.0.to_array();
        mem[..4].copy_from_slice(&arr);
    }

    #[inline(always)]
    fn abs(self) -> Self {
        I32VecNeon(self.0.abs())
    }

    #[inline(always)]
    fn as_f32(self) -> F32VecNeon {
        F32VecNeon(self.0.to_f32())
    }

    #[inline(always)]
    fn bitcast_to_f32(self) -> F32VecNeon {
        F32VecNeon(self.0.bitcast_to_f32())
    }

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecNeon {
        let arr = self.0.to_array();
        let u_arr: [u32; 4] = core::array::from_fn(|i| arr[i] as u32);
        U32VecNeon(u32x4::from_array(Token::summon().unwrap(), u_arr))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> MaskNeon {
        let result = self.0.simd_gt(other.0);
        MaskNeon(result.bitcast_to_f32())
    }

    #[inline(always)]
    fn lt_zero(self) -> MaskNeon {
        let zero = i32x4::zero(Token::summon().unwrap());
        let result = zero.simd_gt(self.0);
        MaskNeon(result.bitcast_to_f32())
    }

    #[inline(always)]
    fn eq(self, other: Self) -> MaskNeon {
        let result = self.0.simd_eq(other.0);
        MaskNeon(result.bitcast_to_f32())
    }

    #[inline(always)]
    fn eq_zero(self) -> MaskNeon {
        let zero = i32x4::zero(Token::summon().unwrap());
        let result = self.0.simd_eq(zero);
        MaskNeon(result.bitcast_to_f32())
    }

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        I32VecNeon(self.0.shl_const::<AMOUNT_I>())
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        I32VecNeon(self.0.shr_arithmetic_const::<AMOUNT_I>())
    }

    #[inline(always)]
    fn mul_wide_take_high(self, rhs: Self) -> Self {
        let a = self.0.to_array();
        let b = rhs.0.to_array();
        let out: [i32; 4] = core::array::from_fn(|i| {
            let wide = (a[i] as i64) * (b[i] as i64);
            (wide >> 32) as i32
        });
        I32VecNeon(i32x4::from_array(Token::summon().unwrap(), out))
    }

    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        let arr = self.0.to_array();
        for i in 0..4 {
            dest[i] = arr[i] as u16;
        }
    }

    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        assert!(dest.len() >= Self::LEN);
        let arr = self.0.to_array();
        for i in 0..4 {
            dest[i] = arr[i] as u8;
        }
    }
}

// ============================================================================
// U32VecNeon
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecNeon(u32x4<Token>);

impl U32SimdVec for U32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecNeon {
        I32VecNeon(self.0.bitcast_to_i32())
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        U32VecNeon(self.0.shr_logical_const::<AMOUNT_I>())
    }
}

// ============================================================================
// U8VecNeon
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecNeon(u8x16<Token>);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
#[allow(unsafe_code)]
unsafe impl U8SimdVec for U8VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        assert!(mem.len() >= Self::LEN);
        U8VecNeon(u8x16::from_slice(d.token(), mem))
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        U8VecNeon(u8x16::splat(d.token(), v))
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        assert!(mem.len() >= Self::LEN);
        let arr = self.0.to_array();
        mem[..16].copy_from_slice(&arr);
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<u8>]) {
        assert!(dest.len() >= 2 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        for i in 0..16 {
            dest[2 * i].write(aa[i]);
            dest[2 * i + 1].write(bb[i]);
        }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<u8>]) {
        assert!(dest.len() >= 3 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        let cc = c.0.to_array();
        for i in 0..16 {
            dest[3 * i].write(aa[i]);
            dest[3 * i + 1].write(bb[i]);
            dest[3 * i + 2].write(cc[i]);
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
        assert!(dest.len() >= 4 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        let cc = c.0.to_array();
        let dd = d.0.to_array();
        for i in 0..16 {
            dest[4 * i].write(aa[i]);
            dest[4 * i + 1].write(bb[i]);
            dest[4 * i + 2].write(cc[i]);
            dest[4 * i + 3].write(dd[i]);
        }
    }
}

// ============================================================================
// U16VecNeon
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecNeon(u16x8<Token>);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
#[allow(unsafe_code)]
unsafe impl U16SimdVec for U16VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        U16VecNeon(u16x8::from_slice(d.token(), mem))
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        U16VecNeon(u16x8::splat(d.token(), v))
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        assert!(mem.len() >= Self::LEN);
        let arr = self.0.to_array();
        mem[..8].copy_from_slice(&arr);
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<u16>]) {
        assert!(dest.len() >= 2 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        for i in 0..8 {
            dest[2 * i].write(aa[i]);
            dest[2 * i + 1].write(bb[i]);
        }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<u16>]) {
        assert!(dest.len() >= 3 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        let cc = c.0.to_array();
        for i in 0..8 {
            dest[3 * i].write(aa[i]);
            dest[3 * i + 1].write(bb[i]);
            dest[3 * i + 2].write(cc[i]);
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
        assert!(dest.len() >= 4 * Self::LEN);
        let aa = a.0.to_array();
        let bb = b.0.to_array();
        let cc = c.0.to_array();
        let dd = d.0.to_array();
        for i in 0..8 {
            dest[4 * i].write(aa[i]);
            dest[4 * i + 1].write(bb[i]);
            dest[4 * i + 2].write(cc[i]);
            dest[4 * i + 3].write(dd[i]);
        }
    }
}

// ============================================================================
// MaskNeon
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskNeon(f32x4<Token>);

impl BitAnd for MaskNeon {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        MaskNeon(self.0 & rhs.0)
    }
}

impl BitOr for MaskNeon {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        MaskNeon(self.0 | rhs.0)
    }
}

impl SimdMask for MaskNeon {
    type Descriptor = NeonDescriptor;

    #[inline(always)]
    fn if_then_else_f32(self, if_true: F32VecNeon, if_false: F32VecNeon) -> F32VecNeon {
        // blend: where mask is all-1s pick if_true, else if_false
        F32VecNeon(f32x4::blend(self.0, if_true.0, if_false.0))
    }

    #[inline(always)]
    fn if_then_else_i32(self, if_true: I32VecNeon, if_false: I32VecNeon) -> I32VecNeon {
        let mask_i32 = self.0.bitcast_to_i32();
        I32VecNeon(i32x4::blend(mask_i32, if_true.0, if_false.0))
    }

    #[inline(always)]
    fn maskz_i32(self, v: I32VecNeon) -> I32VecNeon {
        // maskz: !mask & v (zero where mask is set)
        let mask_i32 = self.0.bitcast_to_i32();
        let not_mask = mask_i32.not();
        I32VecNeon(not_mask & v.0)
    }

    #[inline(always)]
    fn all(self) -> bool {
        let mask_i32 = self.0.bitcast_to_i32();
        mask_i32.all_true()
    }

    #[inline(always)]
    fn andnot(self, rhs: Self) -> Self {
        // !self & rhs
        MaskNeon(self.0.not() & rhs.0)
    }
}
