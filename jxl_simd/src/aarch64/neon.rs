// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use archmage::intrinsics::aarch64::*;

use crate::U32SimdVec;

use super::super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U8SimdVec, U16SimdVec};

// Safety invariant: this type is only ever constructed if neon is available.
#[derive(Clone, Copy, Debug)]
pub struct NeonDescriptor(());

impl NeonDescriptor {
    #[inline]
    pub fn from_token(_token: archmage::NeonToken) -> Self {
        Self(())
    }
}

/// Prepared 8-entry BF16 lookup table for NEON.
/// Contains 8 BF16 values packed into 16 bytes (uint8x16_t).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Neon(uint8x16_t);

impl SimdDescriptor for NeonDescriptor {
    type F32Vec = F32VecNeon;

    type I32Vec = I32VecNeon;

    type U32Vec = U32VecNeon;

    type U16Vec = U16VecNeon;

    type U8Vec = U8VecNeon;

    type Mask = MaskNeon;
    type Bf16Table8 = Bf16Table8Neon;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    fn new() -> Option<Self> {
        archmage::NeonToken::summon().map(Self::from_token)
    }

    #[inline]
    fn maybe_downgrade_256bit(self) -> Self {
        self
    }

    #[inline]
    fn maybe_downgrade_128bit(self) -> Self {
        self
    }

    #[allow(unsafe_code)]
    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn inner<R>(d: NeonDescriptor, f: impl FnOnce(NeonDescriptor) -> R) -> R {
            f(d)
        }
        unsafe { inner(self, f) }
    }
}

macro_rules! fn_neon {
    {} => {};
    {$(
        fn $name:ident($this:ident: $self_ty:ty $(, $arg:ident: $ty:ty)* $(,)?) $(-> $ret:ty )?
        $body: block
    )*} => {$(
        #[inline(always)]
        #[allow(unsafe_code)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "neon")]
            #[inline]
            #[allow(unsafe_code)]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            unsafe { inner(self, $($arg),*) }
        }
    )*};
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecNeon(float32x4_t, NeonDescriptor);

#[allow(unsafe_code)]
impl F32SimdVec for F32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    #[allow(unsafe_code)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        Self(unsafe { vdupq_n_f32(v) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn zero(d: Self::Descriptor) -> Self {
        Self(unsafe { vdupq_n_f32(0.0) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        Self(unsafe { vld1q_f32(mem.first_chunk::<4>().unwrap()) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        unsafe { vst1q_f32(mem.first_chunk_mut::<4>().unwrap(), self.0) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 2 * Self::LEN);
        unsafe {
            vst2q_f32(
                dest.first_chunk_mut::<8>().unwrap(),
                float32x4x2_t(a.0, b.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 3 * Self::LEN);
        unsafe {
            vst3q_f32(
                dest.first_chunk_mut::<12>().unwrap(),
                float32x4x3_t(a.0, b.0, c.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        assert!(dest.len() >= 4 * Self::LEN);
        unsafe {
            vst4q_f32(
                dest.first_chunk_mut::<16>().unwrap(),
                float32x4x4_t(a.0, b.0, c.0, d.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
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
        #[target_feature(enable = "neon")]
        #[inline]
        fn store_interleaved_8_impl(
            a: float32x4_t,
            b: float32x4_t,
            c: float32x4_t,
            d: float32x4_t,
            e: float32x4_t,
            f: float32x4_t,
            g: float32x4_t,
            h: float32x4_t,
            dest: &mut [f32],
        ) {
            assert!(dest.len() >= 8 * F32VecNeon::LEN);

            let ae_lo = vzip1q_f32(a, e);
            let ae_hi = vzip2q_f32(a, e);
            let bf_lo = vzip1q_f32(b, f);
            let bf_hi = vzip2q_f32(b, f);
            let cg_lo = vzip1q_f32(c, g);
            let cg_hi = vzip2q_f32(c, g);
            let dh_lo = vzip1q_f32(d, h);
            let dh_hi = vzip2q_f32(d, h);

            let aebf_0 = vzip1q_f32(ae_lo, bf_lo);
            let aebf_1 = vzip2q_f32(ae_lo, bf_lo);
            let aebf_2 = vzip1q_f32(ae_hi, bf_hi);
            let aebf_3 = vzip2q_f32(ae_hi, bf_hi);
            let cgdh_0 = vzip1q_f32(cg_lo, dh_lo);
            let cgdh_1 = vzip2q_f32(cg_lo, dh_lo);
            let cgdh_2 = vzip1q_f32(cg_hi, dh_hi);
            let cgdh_3 = vzip2q_f32(cg_hi, dh_hi);

            let out0 = vreinterpretq_f32_f64(vzip1q_f64(
                vreinterpretq_f64_f32(aebf_0),
                vreinterpretq_f64_f32(cgdh_0),
            ));
            let out1 = vreinterpretq_f32_f64(vzip2q_f64(
                vreinterpretq_f64_f32(aebf_0),
                vreinterpretq_f64_f32(cgdh_0),
            ));
            let out2 = vreinterpretq_f32_f64(vzip1q_f64(
                vreinterpretq_f64_f32(aebf_1),
                vreinterpretq_f64_f32(cgdh_1),
            ));
            let out3 = vreinterpretq_f32_f64(vzip2q_f64(
                vreinterpretq_f64_f32(aebf_1),
                vreinterpretq_f64_f32(cgdh_1),
            ));
            let out4 = vreinterpretq_f32_f64(vzip1q_f64(
                vreinterpretq_f64_f32(aebf_2),
                vreinterpretq_f64_f32(cgdh_2),
            ));
            let out5 = vreinterpretq_f32_f64(vzip2q_f64(
                vreinterpretq_f64_f32(aebf_2),
                vreinterpretq_f64_f32(cgdh_2),
            ));
            let out6 = vreinterpretq_f32_f64(vzip1q_f64(
                vreinterpretq_f64_f32(aebf_3),
                vreinterpretq_f64_f32(cgdh_3),
            ));
            let out7 = vreinterpretq_f32_f64(vzip2q_f64(
                vreinterpretq_f64_f32(aebf_3),
                vreinterpretq_f64_f32(cgdh_3),
            ));

            vst1q_f32(dest[0..4].first_chunk_mut::<4>().unwrap(), out0);
            vst1q_f32(dest[4..8].first_chunk_mut::<4>().unwrap(), out1);
            vst1q_f32(dest[8..12].first_chunk_mut::<4>().unwrap(), out2);
            vst1q_f32(dest[12..16].first_chunk_mut::<4>().unwrap(), out3);
            vst1q_f32(dest[16..20].first_chunk_mut::<4>().unwrap(), out4);
            vst1q_f32(dest[20..24].first_chunk_mut::<4>().unwrap(), out5);
            vst1q_f32(dest[24..28].first_chunk_mut::<4>().unwrap(), out6);
            vst1q_f32(dest[28..32].first_chunk_mut::<4>().unwrap(), out7);
        }

        unsafe { store_interleaved_8_impl(a.0, b.0, c.0, d.0, e.0, f.0, g.0, h.0, dest) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        assert!(src.len() >= 2 * Self::LEN);
        let float32x4x2_t(a, b) = unsafe { vld2q_f32(src.first_chunk::<8>().unwrap()) };
        (Self(a, d), Self(b, d))
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        assert!(src.len() >= 3 * Self::LEN);
        let float32x4x3_t(a, b, c) = unsafe { vld3q_f32(src.first_chunk::<12>().unwrap()) };
        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        assert!(src.len() >= 4 * Self::LEN);
        let float32x4x4_t(a, b, c, e) = unsafe { vld4q_f32(src.first_chunk::<16>().unwrap()) };
        (Self(a, d), Self(b, d), Self(c, d), Self(e, d))
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn transpose_square(d: NeonDescriptor, data: &mut [[f32; 4]], stride: usize) {
        #[target_feature(enable = "neon")]
        #[inline]
        fn transpose4x4f32(d: NeonDescriptor, data: &mut [[f32; 4]], stride: usize) {
            assert!(data.len() > 3 * stride);

            let p0 = F32VecNeon::load_array(d, &data[0]).0;
            let p1 = F32VecNeon::load_array(d, &data[1 * stride]).0;
            let p2 = F32VecNeon::load_array(d, &data[2 * stride]).0;
            let p3 = F32VecNeon::load_array(d, &data[3 * stride]).0;

            let tr0 = vreinterpretq_f64_f32(vtrn1q_f32(p0, p1));
            let tr1 = vreinterpretq_f64_f32(vtrn2q_f32(p0, p1));
            let tr2 = vreinterpretq_f64_f32(vtrn1q_f32(p2, p3));
            let tr3 = vreinterpretq_f64_f32(vtrn2q_f32(p2, p3));

            let p0 = vreinterpretq_f32_f64(vzip1q_f64(tr0, tr2));
            let p1 = vreinterpretq_f32_f64(vzip1q_f64(tr1, tr3));
            let p2 = vreinterpretq_f32_f64(vzip2q_f64(tr0, tr2));
            let p3 = vreinterpretq_f32_f64(vzip2q_f64(tr1, tr3));

            F32VecNeon(p0, d).store_array(&mut data[0]);
            F32VecNeon(p1, d).store_array(&mut data[1 * stride]);
            F32VecNeon(p2, d).store_array(&mut data[2 * stride]);
            F32VecNeon(p3, d).store_array(&mut data[3 * stride]);
        }

        #[target_feature(enable = "neon")]
        #[inline]
        fn transpose4x4f32_contiguous(d: NeonDescriptor, data: &mut [[f32; 4]]) {
            assert!(data.len() > 3);

            let float32x4x4_t(p0, p1, p2, p3) =
                vld4q_f32(data.as_flattened().first_chunk::<16>().unwrap());

            F32VecNeon(p0, d).store_array(&mut data[0]);
            F32VecNeon(p1, d).store_array(&mut data[1]);
            F32VecNeon(p2, d).store_array(&mut data[2]);
            F32VecNeon(p3, d).store_array(&mut data[3]);
        }

        if stride == 1 {
            unsafe { transpose4x4f32_contiguous(d, data) }
        } else {
            unsafe { transpose4x4f32(d, data, stride) }
        }
    }

    crate::impl_f32_array_interface!();

    fn_neon! {
        fn mul_add(this: F32VecNeon, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vfmaq_f32(add.0, this.0, mul.0), this.1)
        }

        fn neg_mul_add(this: F32VecNeon, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vfmsq_f32(add.0, this.0, mul.0), this.1)
        }

        fn abs(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vabsq_f32(this.0), this.1)
        }

        fn floor(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vrndmq_f32(this.0), this.1)
        }

        fn sqrt(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vsqrtq_f32(this.0), this.1)
        }

        fn neg(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vnegq_f32(this.0), this.1)
        }

        fn copysign(this: F32VecNeon, sign: F32VecNeon) -> F32VecNeon {
            F32VecNeon(
                vbslq_f32(vdupq_n_u32(0x8000_0000), sign.0, this.0),
                this.1,
            )
        }

        fn max(this: F32VecNeon, other: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vmaxq_f32(this.0, other.0), this.1)
        }

        fn min(this: F32VecNeon, other: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vminq_f32(this.0, other.0), this.1)
        }

        fn gt(this: F32VecNeon, other: F32VecNeon) -> MaskNeon {
            MaskNeon(vcgtq_f32(this.0, other.0), this.1)
        }

        fn as_i32(this: F32VecNeon) -> I32VecNeon {
            I32VecNeon(vcvtq_s32_f32(this.0), this.1)
        }

        fn bitcast_to_i32(this: F32VecNeon) -> I32VecNeon {
            I32VecNeon(vreinterpretq_s32_f32(this.0), this.1)
        }

        fn round_store_u8(this: F32VecNeon, dest: &mut [u8]) {
            assert!(dest.len() >= F32VecNeon::LEN);
            let rounded = vrndnq_f32(this.0);
            let i32s = vcvtq_s32_f32(rounded);
            let u16s = vqmovun_s32(i32s);
            let u8s = vqmovn_u16(vcombine_u16(u16s, u16s));
            // Extract 4 bytes via lane extraction (no safe wrapper for vst1_lane_u32)
            let val = vget_lane_u32::<0>(vreinterpret_u32_u8(u8s));
            dest[..4].copy_from_slice(&val.to_ne_bytes());
        }

        fn round_store_u16(this: F32VecNeon, dest: &mut [u16]) {
            assert!(dest.len() >= F32VecNeon::LEN);
            let rounded = vrndnq_f32(this.0);
            let i32s = vcvtq_s32_f32(rounded);
            let u16s = vqmovun_s32(i32s);
            vst1_u16(dest.first_chunk_mut::<4>().unwrap(), u16s);
        }

        fn store_f16_bits(this: F32VecNeon, dest: &mut [u16]) {
            assert!(dest.len() >= F32VecNeon::LEN);
            let f16_bits: uint16x4_t;
            // Inline asm required: stdarch incorrectly requires fp16 target feature for vcvt_f16_f32
            unsafe {
                std::arch::asm!(
                    "fcvtn {out:v}.4h, {inp:v}.4s",
                    inp = in(vreg) this.0,
                    out = out(vreg) f16_bits,
                    options(pure, nomem, nostack),
                );
            }
            vst1_u16(dest.first_chunk_mut::<4>().unwrap(), f16_bits);
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        let result: float32x4_t;
        // Inline asm required: stdarch incorrectly requires fp16 target feature for vcvt_f32_f16
        unsafe {
            let f16_bits = vld1_u16(mem.first_chunk::<4>().unwrap());
            std::arch::asm!(
                "fcvtl {out:v}.4s, {inp:v}.4h",
                inp = in(vreg) f16_bits,
                out = out(vreg) result,
                options(pure, nomem, nostack),
            );
        }
        F32VecNeon(result, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn prepare_table_bf16_8(_d: NeonDescriptor, table: &[f32; 8]) -> Bf16Table8Neon {
        #[target_feature(enable = "neon")]
        #[inline]
        fn prepare_impl(table: &[f32; 8]) -> uint8x16_t {
            let table_lo = vld1q_f32(table.first_chunk::<4>().unwrap());
            let table_hi = vld1q_f32(table[4..].first_chunk::<4>().unwrap());

            let table_lo_u32 = vreinterpretq_u32_f32(table_lo);
            let table_hi_u32 = vreinterpretq_u32_f32(table_hi);

            let bf16_lo_u16 = vshrn_n_u32::<16>(table_lo_u32);
            let bf16_hi_u16 = vshrn_n_u32::<16>(table_hi_u32);

            let bf16_table_u16 = vcombine_u16(bf16_lo_u16, bf16_hi_u16);
            vreinterpretq_u8_u16(bf16_table_u16)
        }
        Bf16Table8Neon(unsafe { prepare_impl(table) })
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn table_lookup_bf16_8(d: NeonDescriptor, table: Bf16Table8Neon, indices: I32VecNeon) -> Self {
        #[target_feature(enable = "neon")]
        #[inline]
        fn lookup_impl(bf16_table: uint8x16_t, indices: int32x4_t) -> float32x4_t {
            let indices_u32 = vreinterpretq_u32_s32(indices);
            let shl17 = vshlq_n_u32::<17>(indices_u32);
            let shl25 = vshlq_n_u32::<25>(indices_u32);
            let base = vdupq_n_u32(0x01008080);
            let shuffle_mask = vorrq_u32(vorrq_u32(shl17, shl25), base);

            let result = vqtbl1q_u8(bf16_table, vreinterpretq_u8_u32(shuffle_mask));

            vreinterpretq_f32_u8(result)
        }
        F32VecNeon(unsafe { lookup_impl(table.0, indices.0) }, d)
    }
}

impl Add<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn add(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vaddq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn sub(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vsubq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn mul(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vmulq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Div<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn div(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vdivq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn add_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vaddq_f32(this.0, rhs.0);
        }
    }
}

impl SubAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vsubq_f32(this.0, rhs.0);
        }
    }
}

impl MulAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vmulq_f32(this.0, rhs.0);
        }
    }
}

impl DivAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn div_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vdivq_f32(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecNeon(int32x4_t, NeonDescriptor);

#[allow(unsafe_code)]
impl I32SimdVec for I32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    #[allow(unsafe_code)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        Self(unsafe { vdupq_n_s32(v) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        Self(unsafe { vld1q_s32(mem.first_chunk::<4>().unwrap()) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store(&self, mem: &mut [i32]) {
        assert!(mem.len() >= Self::LEN);
        unsafe { vst1q_s32(mem.first_chunk_mut::<4>().unwrap(), self.0) }
    }

    fn_neon! {
        fn abs(this: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vabsq_s32(this.0), this.1)
        }

        fn as_f32(this: I32VecNeon) -> F32VecNeon {
            F32VecNeon(vcvtq_f32_s32(this.0), this.1)
        }

        fn bitcast_to_f32(this: I32VecNeon) -> F32VecNeon {
            F32VecNeon(vreinterpretq_f32_s32(this.0), this.1)
        }

        fn bitcast_to_u32(this: I32VecNeon) -> U32VecNeon {
            U32VecNeon(vreinterpretq_u32_s32(this.0), this.1)
        }

        fn gt(this: I32VecNeon, other: I32VecNeon) -> MaskNeon {
            MaskNeon(vcgtq_s32(this.0, other.0), this.1)
        }

        fn lt_zero(this: I32VecNeon) -> MaskNeon {
            MaskNeon(vcltzq_s32(this.0), this.1)
        }

        fn eq(this: I32VecNeon, other: I32VecNeon) -> MaskNeon {
            MaskNeon(vceqq_s32(this.0, other.0), this.1)
        }

        fn eq_zero(this: I32VecNeon) -> MaskNeon {
            MaskNeon(vceqzq_s32(this.0), this.1)
        }

        fn mul_wide_take_high(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            let l = vmull_s32(vget_low_s32(this.0), vget_low_s32(rhs.0));
            let l = vreinterpretq_s32_s64(l);
            let h = vmull_high_s32(this.0, rhs.0);
            let h = vreinterpretq_s32_s64(h);
            I32VecNeon(vuzp2q_s32(l, h), this.1)
        }

        fn wrapping_add(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            this + rhs
        }

        fn wrapping_sub(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            this - rhs
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        unsafe { Self(vshlq_n_s32::<AMOUNT_I>(self.0), self.1) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        unsafe { Self(vshrq_n_s32::<AMOUNT_I>(self.0), self.1) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_u16(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        unsafe {
            let narrowed = vmovn_s32(self.0);
            vst1_u16(
                dest.first_chunk_mut::<4>().unwrap(),
                vreinterpret_u16_s16(narrowed),
            );
        }
    }
}

impl Add<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn add(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vaddq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn sub(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vsubq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn mul(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vmulq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Neg for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn neg(this: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vnegq_s32(this.0), this.1)
        }
    }
}

impl BitAnd<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn bitand(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vandq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn bitor(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vorrq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl BitXor<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn bitxor(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(veorq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn add_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vaddq_s32(this.0, rhs.0)
        }
    }
}

impl SubAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vsubq_s32(this.0, rhs.0)
        }
    }
}

impl MulAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vmulq_s32(this.0, rhs.0)
        }
    }
}

impl BitAndAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn bitand_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vandq_s32(this.0, rhs.0);
        }
    }
}

impl BitOrAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn bitor_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vorrq_s32(this.0, rhs.0);
        }
    }
}

impl BitXorAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn bitxor_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = veorq_s32(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecNeon(uint32x4_t, NeonDescriptor);

#[allow(unsafe_code)]
impl U32SimdVec for U32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    fn_neon! {
        fn bitcast_to_i32(this: U32VecNeon) -> I32VecNeon {
            I32VecNeon(vreinterpretq_s32_u32(this.0), this.1)
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        unsafe { Self(vshrq_n_u32::<AMOUNT_I>(self.0), self.1) }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecNeon(uint8x16_t, NeonDescriptor);

#[allow(unsafe_code)]
impl U8SimdVec for U8VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 16;

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        assert!(mem.len() >= Self::LEN);
        Self(unsafe { vld1q_u8(mem.first_chunk::<16>().unwrap()) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        Self(unsafe { vdupq_n_u8(v) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store(&self, mem: &mut [u8]) {
        assert!(mem.len() >= Self::LEN);
        unsafe { vst1q_u8(mem.first_chunk_mut::<16>().unwrap(), self.0) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 2 * Self::LEN);
        unsafe {
            vst2q_u8(
                dest.first_chunk_mut::<32>().unwrap(),
                uint8x16x2_t(a.0, b.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 3 * Self::LEN);
        unsafe {
            vst3q_u8(
                dest.first_chunk_mut::<48>().unwrap(),
                uint8x16x3_t(a.0, b.0, c.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        assert!(dest.len() >= 4 * Self::LEN);
        unsafe {
            vst4q_u8(
                dest.first_chunk_mut::<64>().unwrap(),
                uint8x16x4_t(a.0, b.0, c.0, d.0),
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecNeon(uint16x8_t, NeonDescriptor);

#[allow(unsafe_code)]
impl U16SimdVec for U16VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 8;

    #[inline(always)]
    #[allow(unsafe_code)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        Self(unsafe { vld1q_u16(mem.first_chunk::<8>().unwrap()) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        Self(unsafe { vdupq_n_u16(v) }, d)
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store(&self, mem: &mut [u16]) {
        assert!(mem.len() >= Self::LEN);
        unsafe { vst1q_u16(mem.first_chunk_mut::<8>().unwrap(), self.0) }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 2 * Self::LEN);
        unsafe {
            vst2q_u16(
                dest.first_chunk_mut::<16>().unwrap(),
                uint16x8x2_t(a.0, b.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 3 * Self::LEN);
        unsafe {
            vst3q_u16(
                dest.first_chunk_mut::<24>().unwrap(),
                uint16x8x3_t(a.0, b.0, c.0),
            )
        }
    }

    #[inline(always)]
    #[allow(unsafe_code)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        assert!(dest.len() >= 4 * Self::LEN);
        unsafe {
            vst4q_u16(
                dest.first_chunk_mut::<32>().unwrap(),
                uint16x8x4_t(a.0, b.0, c.0, d.0),
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskNeon(uint32x4_t, NeonDescriptor);

impl SimdMask for MaskNeon {
    type Descriptor = NeonDescriptor;

    fn_neon! {
        fn if_then_else_f32(
            this: MaskNeon,
            if_true: F32VecNeon,
            if_false: F32VecNeon,
        ) -> F32VecNeon {
            F32VecNeon(vbslq_f32(this.0, if_true.0, if_false.0), this.1)
        }

        fn if_then_else_i32(
            this: MaskNeon,
            if_true: I32VecNeon,
            if_false: I32VecNeon,
        ) -> I32VecNeon {
            I32VecNeon(vbslq_s32(this.0, if_true.0, if_false.0), this.1)
        }

        fn maskz_i32(this: MaskNeon, v: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vbicq_s32(v.0, vreinterpretq_s32_u32(this.0)), this.1)
        }

        fn andnot(this: MaskNeon, rhs: MaskNeon) -> MaskNeon {
            MaskNeon(vbicq_u32(rhs.0, this.0), this.1)
        }

        fn all(this: MaskNeon) -> bool {
            vminvq_u32(this.0) == u32::MAX
        }
    }
}

impl BitAnd<MaskNeon> for MaskNeon {
    type Output = MaskNeon;
    fn_neon! {
        fn bitand(this: MaskNeon, rhs: MaskNeon) -> MaskNeon {
            MaskNeon(vandq_u32(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<MaskNeon> for MaskNeon {
    type Output = MaskNeon;
    fn_neon! {
        fn bitor(this: MaskNeon, rhs: MaskNeon) -> MaskNeon {
            MaskNeon(vorrq_u32(this.0, rhs.0), this.1)
        }
    }
}
