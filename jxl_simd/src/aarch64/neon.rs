// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

use archmage::SimdToken;
use archmage::arcane;
use archmage::intrinsics::aarch64::*;

use crate::U32SimdVec;

use super::super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U8SimdVec, U16SimdVec};

fn token() -> archmage::NeonToken {
    archmage::NeonToken::summon().unwrap()
}

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

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[arcane]
        fn impl_<R>(
            _: archmage::NeonToken,
            d: NeonDescriptor,
            f: impl FnOnce(NeonDescriptor) -> R,
        ) -> R {
            f(d)
        }
        impl_(token(), self, f)
    }
}

macro_rules! fn_neon {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[arcane]
            #[inline(always)]
            fn impl_(_t: archmage::NeonToken, $this: $self_ty, $($arg: $ty),*) $(-> $ret)? $body
            impl_(token(), self, $($arg),*)
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecNeon(float32x4_t, NeonDescriptor);

impl F32SimdVec for F32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: f32) -> float32x4_t {
            vdupq_n_f32(v)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken) -> float32x4_t {
            vdupq_n_f32(0.0)
        }
        Self(impl_(token()), d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, mem: &[f32]) -> float32x4_t {
            assert!(mem.len() >= F32VecNeon::LEN);
            vld1q_f32(mem.first_chunk::<4>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: float32x4_t, mem: &mut [f32]) {
            assert!(mem.len() >= F32VecNeon::LEN);
            vst1q_f32(mem.first_chunk_mut::<4>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, a: float32x4_t, b: float32x4_t, dest: &mut [f32]) {
            assert!(dest.len() >= 2 * F32VecNeon::LEN);
            vst2q_f32(dest.first_chunk_mut::<8>().unwrap(), float32x4x2_t(a, b))
        }
        impl_(token(), a.0, b.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            a: float32x4_t,
            b: float32x4_t,
            c: float32x4_t,
            dest: &mut [f32],
        ) {
            assert!(dest.len() >= 3 * F32VecNeon::LEN);
            vst3q_f32(
                dest.first_chunk_mut::<12>().unwrap(),
                float32x4x3_t(a, b, c),
            )
        }
        impl_(token(), a.0, b.0, c.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            a: float32x4_t,
            b: float32x4_t,
            c: float32x4_t,
            d: float32x4_t,
            dest: &mut [f32],
        ) {
            assert!(dest.len() >= 4 * F32VecNeon::LEN);
            vst4q_f32(
                dest.first_chunk_mut::<16>().unwrap(),
                float32x4x4_t(a, b, c, d),
            )
        }
        impl_(token(), a.0, b.0, c.0, d.0, dest)
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
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
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

        impl_(token(), a.0, b.0, c.0, d.0, e.0, f.0, g.0, h.0, dest)
    }

    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, src: &[f32]) -> (float32x4_t, float32x4_t) {
            assert!(src.len() >= 2 * F32VecNeon::LEN);
            let float32x4x2_t(a, b) = vld2q_f32(src.first_chunk::<8>().unwrap());
            (a, b)
        }
        let (a, b) = impl_(token(), src);
        (Self(a, d), Self(b, d))
    }

    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, src: &[f32]) -> (float32x4_t, float32x4_t, float32x4_t) {
            assert!(src.len() >= 3 * F32VecNeon::LEN);
            let float32x4x3_t(a, b, c) = vld3q_f32(src.first_chunk::<12>().unwrap());
            (a, b, c)
        }
        let (a, b, c) = impl_(token(), src);
        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            src: &[f32],
        ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
            assert!(src.len() >= 4 * F32VecNeon::LEN);
            let float32x4x4_t(a, b, c, e) = vld4q_f32(src.first_chunk::<16>().unwrap());
            (a, b, c, e)
        }
        let (a, b, c, e) = impl_(token(), src);
        (Self(a, d), Self(b, d), Self(c, d), Self(e, d))
    }

    #[inline(always)]
    fn transpose_square(d: NeonDescriptor, data: &mut [[f32; 4]], stride: usize) {
        #[arcane]
        fn transpose4x4f32(
            _: archmage::NeonToken,
            d: NeonDescriptor,
            data: &mut [[f32; 4]],
            stride: usize,
        ) {
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

        #[arcane]
        fn transpose4x4f32_contiguous(
            _: archmage::NeonToken,
            d: NeonDescriptor,
            data: &mut [[f32; 4]],
        ) {
            assert!(data.len() > 3);

            let float32x4x4_t(p0, p1, p2, p3) =
                vld4q_f32(data.as_flattened().first_chunk::<16>().unwrap());

            F32VecNeon(p0, d).store_array(&mut data[0]);
            F32VecNeon(p1, d).store_array(&mut data[1]);
            F32VecNeon(p2, d).store_array(&mut data[2]);
            F32VecNeon(p3, d).store_array(&mut data[3]);
        }

        if stride == 1 {
            transpose4x4f32_contiguous(token(), d, data)
        } else {
            transpose4x4f32(token(), d, data, stride)
        }
    }

    crate::impl_f32_array_interface!();

    fn_neon!(this: F32VecNeon, fn mul_add(mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vfmaq_f32(add.0, this.0, mul.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn neg_mul_add(mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vfmsq_f32(add.0, this.0, mul.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn abs() -> F32VecNeon {
        F32VecNeon(vabsq_f32(this.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn floor() -> F32VecNeon {
        F32VecNeon(vrndmq_f32(this.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn sqrt() -> F32VecNeon {
        F32VecNeon(vsqrtq_f32(this.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn neg() -> F32VecNeon {
        F32VecNeon(vnegq_f32(this.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn copysign(sign: F32VecNeon) -> F32VecNeon {
        F32VecNeon(
            vbslq_f32(vdupq_n_u32(0x8000_0000), sign.0, this.0),
            this.1,
        )
    });

    fn_neon!(this: F32VecNeon, fn max(other: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vmaxq_f32(this.0, other.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn min(other: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vminq_f32(this.0, other.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn gt(other: F32VecNeon) -> MaskNeon {
        MaskNeon(vcgtq_f32(this.0, other.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn as_i32() -> I32VecNeon {
        I32VecNeon(vcvtq_s32_f32(this.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn bitcast_to_i32() -> I32VecNeon {
        I32VecNeon(vreinterpretq_s32_f32(this.0), this.1)
    });

    fn_neon!(this: F32VecNeon, fn round_store_u8(dest: &mut [u8]) {
        assert!(dest.len() >= F32VecNeon::LEN);
        let rounded = vrndnq_f32(this.0);
        let i32s = vcvtq_s32_f32(rounded);
        let u16s = vqmovun_s32(i32s);
        let u8s = vqmovn_u16(vcombine_u16(u16s, u16s));
        let val = vget_lane_u32::<0>(vreinterpret_u32_u8(u8s));
        dest[..4].copy_from_slice(&val.to_ne_bytes());
    });

    fn_neon!(this: F32VecNeon, fn round_store_u16(dest: &mut [u16]) {
        assert!(dest.len() >= F32VecNeon::LEN);
        let rounded = vrndnq_f32(this.0);
        let i32s = vcvtq_s32_f32(rounded);
        let u16s = vqmovun_s32(i32s);
        vst1_u16(dest.first_chunk_mut::<4>().unwrap(), u16s);
    });

    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: float32x4_t, dest: &mut [u16]) {
            assert!(dest.len() >= F32VecNeon::LEN);
            // Scalar f16 conversion: stdarch incorrectly requires fp16 target feature for vcvt_f16_f32
            let mut arr = [0f32; 4];
            vst1q_f32(arr.as_mut_ptr(), v);
            for i in 0..4 {
                dest[i] = crate::f16::from_f32(arr[i]).to_bits();
            }
        }
        impl_(token(), self.0, dest)
    }

    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, mem: &[u16]) -> float32x4_t {
            assert!(mem.len() >= F32VecNeon::LEN);
            // Scalar f16 conversion: stdarch incorrectly requires fp16 target feature for vcvt_f32_f16
            let mut arr = [0f32; 4];
            for i in 0..4 {
                arr[i] = crate::f16::from_bits(mem[i]).to_f32();
            }
            vld1q_f32(arr.as_ptr())
        }
        F32VecNeon(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn prepare_table_bf16_8(_d: NeonDescriptor, table: &[f32; 8]) -> Bf16Table8Neon {
        #[arcane]
        fn impl_(_: archmage::NeonToken, table: &[f32; 8]) -> uint8x16_t {
            let table_lo = vld1q_f32(table.first_chunk::<4>().unwrap());
            let table_hi = vld1q_f32(table[4..].first_chunk::<4>().unwrap());

            let table_lo_u32 = vreinterpretq_u32_f32(table_lo);
            let table_hi_u32 = vreinterpretq_u32_f32(table_hi);

            let bf16_lo_u16 = vshrn_n_u32::<16>(table_lo_u32);
            let bf16_hi_u16 = vshrn_n_u32::<16>(table_hi_u32);

            let bf16_table_u16 = vcombine_u16(bf16_lo_u16, bf16_hi_u16);
            vreinterpretq_u8_u16(bf16_table_u16)
        }
        Bf16Table8Neon(impl_(token(), table))
    }

    #[inline(always)]
    fn table_lookup_bf16_8(d: NeonDescriptor, table: Bf16Table8Neon, indices: I32VecNeon) -> Self {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            bf16_table: uint8x16_t,
            indices: int32x4_t,
        ) -> float32x4_t {
            let indices_u32 = vreinterpretq_u32_s32(indices);
            let shl17 = vshlq_n_u32::<17>(indices_u32);
            let shl25 = vshlq_n_u32::<25>(indices_u32);
            let base = vdupq_n_u32(0x01008080);
            let shuffle_mask = vorrq_u32(vorrq_u32(shl17, shl25), base);

            let result = vqtbl1q_u8(bf16_table, vreinterpretq_u8_u32(shuffle_mask));

            vreinterpretq_f32_u8(result)
        }
        F32VecNeon(impl_(token(), table.0, indices.0), d)
    }
}

impl Add<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon!(this: F32VecNeon, fn add(rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vaddq_f32(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon!(this: F32VecNeon, fn sub(rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vsubq_f32(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon!(this: F32VecNeon, fn mul(rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vmulq_f32(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon!(this: F32VecNeon, fn div(rhs: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vdivq_f32(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecNeon> for F32VecNeon {
    fn_neon!(this: &mut F32VecNeon, fn add_assign(rhs: F32VecNeon) {
        this.0 = vaddq_f32(this.0, rhs.0);
    });
}

impl SubAssign<F32VecNeon> for F32VecNeon {
    fn_neon!(this: &mut F32VecNeon, fn sub_assign(rhs: F32VecNeon) {
        this.0 = vsubq_f32(this.0, rhs.0);
    });
}

impl MulAssign<F32VecNeon> for F32VecNeon {
    fn_neon!(this: &mut F32VecNeon, fn mul_assign(rhs: F32VecNeon) {
        this.0 = vmulq_f32(this.0, rhs.0);
    });
}

impl DivAssign<F32VecNeon> for F32VecNeon {
    fn_neon!(this: &mut F32VecNeon, fn div_assign(rhs: F32VecNeon) {
        this.0 = vdivq_f32(this.0, rhs.0);
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecNeon(int32x4_t, NeonDescriptor);

impl I32SimdVec for I32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: i32) -> int32x4_t {
            vdupq_n_s32(v)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, mem: &[i32]) -> int32x4_t {
            assert!(mem.len() >= I32VecNeon::LEN);
            vld1q_s32(mem.first_chunk::<4>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: int32x4_t, mem: &mut [i32]) {
            assert!(mem.len() >= I32VecNeon::LEN);
            vst1q_s32(mem.first_chunk_mut::<4>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    fn_neon!(this: I32VecNeon, fn abs() -> I32VecNeon {
        I32VecNeon(vabsq_s32(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn as_f32() -> F32VecNeon {
        F32VecNeon(vcvtq_f32_s32(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn bitcast_to_f32() -> F32VecNeon {
        F32VecNeon(vreinterpretq_f32_s32(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn bitcast_to_u32() -> U32VecNeon {
        U32VecNeon(vreinterpretq_u32_s32(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn gt(other: I32VecNeon) -> MaskNeon {
        MaskNeon(vcgtq_s32(this.0, other.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn lt_zero() -> MaskNeon {
        MaskNeon(vcltzq_s32(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn eq(other: I32VecNeon) -> MaskNeon {
        MaskNeon(vceqq_s32(this.0, other.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn eq_zero() -> MaskNeon {
        MaskNeon(vceqzq_s32(this.0), this.1)
    });

    fn_neon!(this: I32VecNeon, fn mul_wide_take_high(rhs: I32VecNeon) -> I32VecNeon {
        let l = vmull_s32(vget_low_s32(this.0), vget_low_s32(rhs.0));
        let l = vreinterpretq_s32_s64(l);
        let h = vmull_high_s32(this.0, rhs.0);
        let h = vreinterpretq_s32_s64(h);
        I32VecNeon(vuzp2q_s32(l, h), this.1)
    });

    fn_neon!(this: I32VecNeon, fn wrapping_add(rhs: I32VecNeon) -> I32VecNeon {
        this + rhs
    });

    fn_neon!(this: I32VecNeon, fn wrapping_sub(rhs: I32VecNeon) -> I32VecNeon {
        this - rhs
    });

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        #[arcane]
        fn impl_<const AMOUNT_I: i32>(_: archmage::NeonToken, v: int32x4_t) -> int32x4_t {
            vshlq_n_s32::<AMOUNT_I>(v)
        }
        Self(impl_::<AMOUNT_I>(token(), self.0), self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        #[arcane]
        fn impl_<const AMOUNT_I: i32>(_: archmage::NeonToken, v: int32x4_t) -> int32x4_t {
            vshrq_n_s32::<AMOUNT_I>(v)
        }
        Self(impl_::<AMOUNT_I>(token(), self.0), self.1)
    }

    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: int32x4_t, dest: &mut [u16]) {
            assert!(dest.len() >= I32VecNeon::LEN);
            let narrowed = vmovn_s32(v);
            vst1_u16(
                dest.first_chunk_mut::<4>().unwrap(),
                vreinterpret_u16_s16(narrowed),
            );
        }
        impl_(token(), self.0, dest)
    }
}

impl Add<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn add(rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vaddq_s32(this.0, rhs.0), this.1)
    });
}

impl Sub<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn sub(rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vsubq_s32(this.0, rhs.0), this.1)
    });
}

impl Mul<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn mul(rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vmulq_s32(this.0, rhs.0), this.1)
    });
}

impl Neg for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn neg() -> I32VecNeon {
        I32VecNeon(vnegq_s32(this.0), this.1)
    });
}

impl BitAnd<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn bitand(rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vandq_s32(this.0, rhs.0), this.1)
    });
}

impl BitOr<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn bitor(rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vorrq_s32(this.0, rhs.0), this.1)
    });
}

impl BitXor<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon!(this: I32VecNeon, fn bitxor(rhs: I32VecNeon) -> I32VecNeon {
        I32VecNeon(veorq_s32(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I32VecNeon> for I32VecNeon {
    fn_neon!(this: &mut I32VecNeon, fn add_assign(rhs: I32VecNeon) {
        this.0 = vaddq_s32(this.0, rhs.0)
    });
}

impl SubAssign<I32VecNeon> for I32VecNeon {
    fn_neon!(this: &mut I32VecNeon, fn sub_assign(rhs: I32VecNeon) {
        this.0 = vsubq_s32(this.0, rhs.0)
    });
}

impl MulAssign<I32VecNeon> for I32VecNeon {
    fn_neon!(this: &mut I32VecNeon, fn mul_assign(rhs: I32VecNeon) {
        this.0 = vmulq_s32(this.0, rhs.0)
    });
}

impl BitAndAssign<I32VecNeon> for I32VecNeon {
    fn_neon!(this: &mut I32VecNeon, fn bitand_assign(rhs: I32VecNeon) {
        this.0 = vandq_s32(this.0, rhs.0);
    });
}

impl BitOrAssign<I32VecNeon> for I32VecNeon {
    fn_neon!(this: &mut I32VecNeon, fn bitor_assign(rhs: I32VecNeon) {
        this.0 = vorrq_s32(this.0, rhs.0);
    });
}

impl BitXorAssign<I32VecNeon> for I32VecNeon {
    fn_neon!(this: &mut I32VecNeon, fn bitxor_assign(rhs: I32VecNeon) {
        this.0 = veorq_s32(this.0, rhs.0);
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecNeon(uint32x4_t, NeonDescriptor);

impl U32SimdVec for U32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    fn_neon!(this: U32VecNeon, fn bitcast_to_i32() -> I32VecNeon {
        I32VecNeon(vreinterpretq_s32_u32(this.0), this.1)
    });

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        #[arcane]
        fn impl_<const AMOUNT_I: i32>(_: archmage::NeonToken, v: uint32x4_t) -> uint32x4_t {
            vshrq_n_u32::<AMOUNT_I>(v)
        }
        Self(impl_::<AMOUNT_I>(token(), self.0), self.1)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecNeon(uint8x16_t, NeonDescriptor);

impl U8SimdVec for U8VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, mem: &[u8]) -> uint8x16_t {
            assert!(mem.len() >= U8VecNeon::LEN);
            vld1q_u8(mem.first_chunk::<16>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: u8) -> uint8x16_t {
            vdupq_n_u8(v)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: uint8x16_t, mem: &mut [u8]) {
            assert!(mem.len() >= U8VecNeon::LEN);
            vst1q_u8(mem.first_chunk_mut::<16>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, a: uint8x16_t, b: uint8x16_t, dest: &mut [u8]) {
            assert!(dest.len() >= 2 * U8VecNeon::LEN);
            vst2q_u8(dest.first_chunk_mut::<32>().unwrap(), uint8x16x2_t(a, b))
        }
        impl_(token(), a.0, b.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            a: uint8x16_t,
            b: uint8x16_t,
            c: uint8x16_t,
            dest: &mut [u8],
        ) {
            assert!(dest.len() >= 3 * U8VecNeon::LEN);
            vst3q_u8(dest.first_chunk_mut::<48>().unwrap(), uint8x16x3_t(a, b, c))
        }
        impl_(token(), a.0, b.0, c.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            a: uint8x16_t,
            b: uint8x16_t,
            c: uint8x16_t,
            d: uint8x16_t,
            dest: &mut [u8],
        ) {
            assert!(dest.len() >= 4 * U8VecNeon::LEN);
            vst4q_u8(
                dest.first_chunk_mut::<64>().unwrap(),
                uint8x16x4_t(a, b, c, d),
            )
        }
        impl_(token(), a.0, b.0, c.0, d.0, dest)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecNeon(uint16x8_t, NeonDescriptor);

impl U16SimdVec for U16VecNeon {
    type Descriptor = NeonDescriptor;
    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, mem: &[u16]) -> uint16x8_t {
            assert!(mem.len() >= U16VecNeon::LEN);
            vld1q_u16(mem.first_chunk::<8>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: u16) -> uint16x8_t {
            vdupq_n_u16(v)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, v: uint16x8_t, mem: &mut [u16]) {
            assert!(mem.len() >= U16VecNeon::LEN);
            vst1q_u16(mem.first_chunk_mut::<8>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::NeonToken, a: uint16x8_t, b: uint16x8_t, dest: &mut [u16]) {
            assert!(dest.len() >= 2 * U16VecNeon::LEN);
            vst2q_u16(dest.first_chunk_mut::<16>().unwrap(), uint16x8x2_t(a, b))
        }
        impl_(token(), a.0, b.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            a: uint16x8_t,
            b: uint16x8_t,
            c: uint16x8_t,
            dest: &mut [u16],
        ) {
            assert!(dest.len() >= 3 * U16VecNeon::LEN);
            vst3q_u16(dest.first_chunk_mut::<24>().unwrap(), uint16x8x3_t(a, b, c))
        }
        impl_(token(), a.0, b.0, c.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(
            _: archmage::NeonToken,
            a: uint16x8_t,
            b: uint16x8_t,
            c: uint16x8_t,
            d: uint16x8_t,
            dest: &mut [u16],
        ) {
            assert!(dest.len() >= 4 * U16VecNeon::LEN);
            vst4q_u16(
                dest.first_chunk_mut::<32>().unwrap(),
                uint16x8x4_t(a, b, c, d),
            )
        }
        impl_(token(), a.0, b.0, c.0, d.0, dest)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskNeon(uint32x4_t, NeonDescriptor);

impl SimdMask for MaskNeon {
    type Descriptor = NeonDescriptor;

    fn_neon!(this: MaskNeon, fn if_then_else_f32(if_true: F32VecNeon, if_false: F32VecNeon) -> F32VecNeon {
        F32VecNeon(vbslq_f32(this.0, if_true.0, if_false.0), this.1)
    });

    fn_neon!(this: MaskNeon, fn if_then_else_i32(if_true: I32VecNeon, if_false: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vbslq_s32(this.0, if_true.0, if_false.0), this.1)
    });

    fn_neon!(this: MaskNeon, fn maskz_i32(v: I32VecNeon) -> I32VecNeon {
        I32VecNeon(vbicq_s32(v.0, vreinterpretq_s32_u32(this.0)), this.1)
    });

    fn_neon!(this: MaskNeon, fn andnot(rhs: MaskNeon) -> MaskNeon {
        MaskNeon(vbicq_u32(rhs.0, this.0), this.1)
    });

    fn_neon!(this: MaskNeon, fn all() -> bool {
        vminvq_u32(this.0) == u32::MAX
    });
}

impl BitAnd<MaskNeon> for MaskNeon {
    type Output = MaskNeon;
    fn_neon!(this: MaskNeon, fn bitand(rhs: MaskNeon) -> MaskNeon {
        MaskNeon(vandq_u32(this.0, rhs.0), this.1)
    });
}

impl BitOr<MaskNeon> for MaskNeon {
    type Output = MaskNeon;
    fn_neon!(this: MaskNeon, fn bitor(rhs: MaskNeon) -> MaskNeon {
        MaskNeon(vorrq_u32(this.0, rhs.0), this.1)
    });
}
