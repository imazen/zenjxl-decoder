// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{U32SimdVec, impl_f32_array_interface};

use super::super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U8SimdVec, U16SimdVec};
use archmage::SimdToken;
use archmage::arcane;
use archmage::intrinsics::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Sub, SubAssign,
};

fn token() -> archmage::X64V2Token {
    archmage::X64V2Token::summon().unwrap()
}

#[derive(Clone, Copy, Debug)]
pub struct Sse42Descriptor(());

impl Sse42Descriptor {
    #[inline]
    pub fn from_token(_token: archmage::X64V2Token) -> Self {
        Self(())
    }
}

impl SimdDescriptor for Sse42Descriptor {
    type F32Vec = F32VecSse42;
    type I32Vec = I32VecSse42;
    type U32Vec = U32VecSse42;
    type U16Vec = U16VecSse42;
    type U8Vec = U8VecSse42;
    type Mask = MaskSse42;
    type Bf16Table8 = Bf16Table8Sse42;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    #[inline]
    fn maybe_downgrade_256bit(self) -> Self::Descriptor256 {
        self
    }

    #[inline]
    fn maybe_downgrade_128bit(self) -> Self::Descriptor128 {
        self
    }

    fn new() -> Option<Self> {
        archmage::X64V2Token::summon().map(Self::from_token)
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[arcane]
        fn impl_<R>(_: archmage::X64V2Token, d: Sse42Descriptor, f: impl FnOnce(Sse42Descriptor) -> R) -> R {
            f(d)
        }
        impl_(token(), self, f)
    }
}

macro_rules! fn_sse42 {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[arcane]
            #[inline(always)]
            fn impl_(_t: archmage::X64V2Token, $this: $self_ty, $($arg: $ty),*) $(-> $ret)? $body
            impl_(token(), self, $($arg),*)
        }
    };
}

/// Prepared 8-entry BF16 lookup table for SSE4.2.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8Sse42(__m128i);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecSse42(__m128, Sse42Descriptor);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskSse42(__m128, Sse42Descriptor);

impl F32SimdVec for F32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, mem: &[f32]) -> __m128 {
            _mm_loadu_ps(mem.first_chunk::<4>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128, mem: &mut [f32]) {
            _mm_storeu_ps(mem.first_chunk_mut::<4>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [f32]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128, b: __m128, dest: &mut [f32]) {
            assert!(dest.len() >= 2 * F32VecSse42::LEN);
            let lo = _mm_unpacklo_ps(a, b);
            let hi = _mm_unpackhi_ps(a, b);
            _mm_storeu_ps(dest[..4].first_chunk_mut::<4>().unwrap(), lo);
            _mm_storeu_ps(dest[4..8].first_chunk_mut::<4>().unwrap(), hi);
        }
        impl_(token(), a.0, b.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [f32]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128, b: __m128, c: __m128, dest: &mut [f32]) {
            assert!(dest.len() >= 3 * F32VecSse42::LEN);
            let p_ab_lo = _mm_unpacklo_ps(a, b);
            let p_ab_hi = _mm_unpackhi_ps(a, b);
            let p_ca_lo = _mm_unpacklo_ps(c, a);
            let p_ca_hi = _mm_unpackhi_ps(c, a);
            let p_bc_hi = _mm_unpackhi_ps(b, c);

            let out0 = _mm_shuffle_ps::<0xC4>(p_ab_lo, p_ca_lo);
            let out1_tmp1 = _mm_shuffle_ps::<0xAF>(p_ab_lo, p_ca_lo);
            let out1 = _mm_shuffle_ps::<0x48>(out1_tmp1, p_ab_hi);
            let out2 = _mm_shuffle_ps::<0xEC>(p_ca_hi, p_bc_hi);

            _mm_storeu_ps(dest[..4].first_chunk_mut::<4>().unwrap(), out0);
            _mm_storeu_ps(dest[4..8].first_chunk_mut::<4>().unwrap(), out1);
            _mm_storeu_ps(dest[8..12].first_chunk_mut::<4>().unwrap(), out2);
        }
        impl_(token(), a.0, b.0, c.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [f32]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128, b: __m128, c: __m128, d: __m128, dest: &mut [f32]) {
            assert!(dest.len() >= 4 * F32VecSse42::LEN);
            let ab_lo = _mm_unpacklo_ps(a, b);
            let ab_hi = _mm_unpackhi_ps(a, b);
            let cd_lo = _mm_unpacklo_ps(c, d);
            let cd_hi = _mm_unpackhi_ps(c, d);

            let out0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo)));
            let out1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo)));
            let out2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi)));
            let out3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi)));

            _mm_storeu_ps(dest[..4].first_chunk_mut::<4>().unwrap(), out0);
            _mm_storeu_ps(dest[4..8].first_chunk_mut::<4>().unwrap(), out1);
            _mm_storeu_ps(dest[8..12].first_chunk_mut::<4>().unwrap(), out2);
            _mm_storeu_ps(dest[12..16].first_chunk_mut::<4>().unwrap(), out3);
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
            _: archmage::X64V2Token,
            a: __m128,
            b: __m128,
            c: __m128,
            d: __m128,
            e: __m128,
            f: __m128,
            g: __m128,
            h: __m128,
            dest: &mut [f32],
        ) {
            assert!(dest.len() >= 8 * F32VecSse42::LEN);
            let ab_lo = _mm_unpacklo_ps(a, b);
            let ab_hi = _mm_unpackhi_ps(a, b);
            let cd_lo = _mm_unpacklo_ps(c, d);
            let cd_hi = _mm_unpackhi_ps(c, d);
            let ef_lo = _mm_unpacklo_ps(e, f);
            let ef_hi = _mm_unpackhi_ps(e, f);
            let gh_lo = _mm_unpacklo_ps(g, h);
            let gh_hi = _mm_unpackhi_ps(g, h);

            let abcd_0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo)));
            let abcd_1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_lo), _mm_castps_pd(cd_lo)));
            let abcd_2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi)));
            let abcd_3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ab_hi), _mm_castps_pd(cd_hi)));
            let efgh_0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ef_lo), _mm_castps_pd(gh_lo)));
            let efgh_1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ef_lo), _mm_castps_pd(gh_lo)));
            let efgh_2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(ef_hi), _mm_castps_pd(gh_hi)));
            let efgh_3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(ef_hi), _mm_castps_pd(gh_hi)));

            _mm_storeu_ps(dest[..4].first_chunk_mut::<4>().unwrap(), abcd_0);
            _mm_storeu_ps(dest[4..8].first_chunk_mut::<4>().unwrap(), efgh_0);
            _mm_storeu_ps(dest[8..12].first_chunk_mut::<4>().unwrap(), abcd_1);
            _mm_storeu_ps(dest[12..16].first_chunk_mut::<4>().unwrap(), efgh_1);
            _mm_storeu_ps(dest[16..20].first_chunk_mut::<4>().unwrap(), abcd_2);
            _mm_storeu_ps(dest[20..24].first_chunk_mut::<4>().unwrap(), efgh_2);
            _mm_storeu_ps(dest[24..28].first_chunk_mut::<4>().unwrap(), abcd_3);
            _mm_storeu_ps(dest[28..32].first_chunk_mut::<4>().unwrap(), efgh_3);
        }
        impl_(token(), a.0, b.0, c.0, d.0, e.0, f.0, g.0, h.0, dest)
    }

    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, src: &[f32]) -> (__m128, __m128) {
            assert!(src.len() >= 2 * F32VecSse42::LEN);
            let in0 = _mm_loadu_ps(src[..4].first_chunk::<4>().unwrap());
            let in1 = _mm_loadu_ps(src[4..8].first_chunk::<4>().unwrap());
            let a = _mm_shuffle_ps::<0x88>(in0, in1);
            let b = _mm_shuffle_ps::<0xDD>(in0, in1);
            (a, b)
        }
        let (a, b) = impl_(token(), src);
        (Self(a, d), Self(b, d))
    }

    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, src: &[f32]) -> (__m128, __m128, __m128) {
            assert!(src.len() >= 3 * F32VecSse42::LEN);
            let in0 = _mm_loadu_ps(src[..4].first_chunk::<4>().unwrap());
            let in1 = _mm_loadu_ps(src[4..8].first_chunk::<4>().unwrap());
            let in2 = _mm_loadu_ps(src[8..12].first_chunk::<4>().unwrap());

            let a_lo = _mm_shuffle_ps::<0xC0>(in0, in0);
            let a_hi = _mm_shuffle_ps::<0x98>(in1, in2);
            let a = _mm_shuffle_ps::<0x9C>(a_lo, a_hi);

            let b_lo = _mm_shuffle_ps::<0x01>(in0, in1);
            let b_hi = _mm_shuffle_ps::<0x2C>(in1, in2);
            let b = _mm_shuffle_ps::<0x98>(b_lo, b_hi);

            let c_lo = _mm_shuffle_ps::<0x12>(in0, in1);
            let c_hi = _mm_shuffle_ps::<0x30>(in2, in2);
            let c = _mm_shuffle_ps::<0x98>(c_lo, c_hi);

            (a, b, c)
        }
        let (a, b, c) = impl_(token(), src);
        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, src: &[f32]) -> (__m128, __m128, __m128, __m128) {
            assert!(src.len() >= 4 * F32VecSse42::LEN);
            let in0 = _mm_loadu_ps(src[..4].first_chunk::<4>().unwrap());
            let in1 = _mm_loadu_ps(src[4..8].first_chunk::<4>().unwrap());
            let in2 = _mm_loadu_ps(src[8..12].first_chunk::<4>().unwrap());
            let in3 = _mm_loadu_ps(src[12..16].first_chunk::<4>().unwrap());

            let t0 = _mm_unpacklo_ps(in0, in1);
            let t1 = _mm_unpackhi_ps(in0, in1);
            let t2 = _mm_unpacklo_ps(in2, in3);
            let t3 = _mm_unpackhi_ps(in2, in3);

            let a = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(t0), _mm_castps_pd(t2)));
            let b = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(t0), _mm_castps_pd(t2)));
            let c = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(t1), _mm_castps_pd(t3)));
            let dv = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(t1), _mm_castps_pd(t3)));

            (a, b, c, dv)
        }
        let (a, b, c, dv) = impl_(token(), src);
        (Self(a, d), Self(b, d), Self(c, d), Self(dv, d))
    }

    fn_sse42!(this: F32VecSse42, fn mul_add(mul: F32VecSse42, add: F32VecSse42) -> F32VecSse42 {
        this * mul + add
    });

    fn_sse42!(this: F32VecSse42, fn neg_mul_add(mul: F32VecSse42, add: F32VecSse42) -> F32VecSse42 {
        add - this * mul
    });

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: f32) -> __m128 {
            _mm_set1_ps(v)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token) -> __m128 {
            _mm_setzero_ps()
        }
        Self(impl_(token()), d)
    }

    fn_sse42!(this: F32VecSse42, fn abs() -> F32VecSse42 {
        F32VecSse42(
            _mm_castsi128_ps(_mm_andnot_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(this.0),
            )),
            this.1)
    });

    fn_sse42!(this: F32VecSse42, fn floor() -> F32VecSse42 {
        F32VecSse42(_mm_floor_ps(this.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn sqrt() -> F32VecSse42 {
        F32VecSse42(_mm_sqrt_ps(this.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn neg() -> F32VecSse42 {
        F32VecSse42(
            _mm_castsi128_ps(_mm_xor_si128(
                _mm_set1_epi32(i32::MIN),
                _mm_castps_si128(this.0),
            )),
            this.1)
    });

    fn_sse42!(this: F32VecSse42, fn copysign(sign: F32VecSse42) -> F32VecSse42 {
        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(i32::MIN));
        F32VecSse42(
            _mm_or_ps(
                _mm_andnot_ps(sign_mask, this.0),
                _mm_and_ps(sign_mask, sign.0),
            ),
            this.1,
        )
    });

    fn_sse42!(this: F32VecSse42, fn max(other: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_max_ps(this.0, other.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn min(other: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_min_ps(this.0, other.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn gt(other: F32VecSse42) -> MaskSse42 {
        MaskSse42(_mm_cmpgt_ps(this.0, other.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn as_i32() -> I32VecSse42 {
        I32VecSse42(_mm_cvtps_epi32(this.0), this.1)
    });

    fn_sse42!(this: F32VecSse42, fn bitcast_to_i32() -> I32VecSse42 {
        I32VecSse42(_mm_castps_si128(this.0), this.1)
    });

    #[inline(always)]
    fn prepare_table_bf16_8(_d: Sse42Descriptor, table: &[f32; 8]) -> Bf16Table8Sse42 {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, table: &[f32; 8]) -> __m128i {
            let table_lo = _mm_loadu_ps(table[..4].first_chunk::<4>().unwrap());
            let table_hi = _mm_loadu_ps(table[4..8].first_chunk::<4>().unwrap());
            let table_lo_i32 = _mm_castps_si128(table_lo);
            let table_hi_i32 = _mm_castps_si128(table_hi);

            let bf16_extract =
                _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
            let bf16_lo = _mm_shuffle_epi8(table_lo_i32, bf16_extract);
            let bf16_hi = _mm_shuffle_epi8(table_hi_i32, bf16_extract);
            _mm_unpacklo_epi64(bf16_lo, bf16_hi)
        }
        Bf16Table8Sse42(impl_(token(), table))
    }

    #[inline(always)]
    fn table_lookup_bf16_8(
        d: Sse42Descriptor,
        table: Bf16Table8Sse42,
        indices: I32VecSse42,
    ) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, bf16_table: __m128i, indices: __m128i) -> __m128 {
            let shl17 = _mm_slli_epi32::<17>(indices);
            let shl25 = _mm_slli_epi32::<25>(indices);
            let base = _mm_set1_epi32(0x01008080u32 as i32);
            let shuffle_mask = _mm_or_si128(_mm_or_si128(shl17, shl25), base);
            let result = _mm_shuffle_epi8(bf16_table, shuffle_mask);
            _mm_castsi128_ps(result)
        }
        F32VecSse42(impl_(token(), table.0, indices.0), d)
    }

    #[inline(always)]
    fn round_store_u8(self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128, dest: &mut [u8]) {
            assert!(dest.len() >= F32VecSse42::LEN);
            let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
            let i32s = _mm_cvtps_epi32(rounded);
            let u16s = _mm_packus_epi32(i32s, i32s);
            let u8s = _mm_packus_epi16(u16s, u16s);
            let val = _mm_cvtsi128_si32(u8s);
            dest[..4].copy_from_slice(&val.to_ne_bytes());
        }
        impl_(token(), self.0, dest)
    }

    #[inline(always)]
    fn round_store_u16(self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128, dest: &mut [u16]) {
            assert!(dest.len() >= F32VecSse42::LEN);
            let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
            let i32s = _mm_cvtps_epi32(rounded);
            let u16s = _mm_packus_epi32(i32s, i32s);
            dest[0] = _mm_extract_epi16::<0>(u16s) as u16;
            dest[1] = _mm_extract_epi16::<1>(u16s) as u16;
            dest[2] = _mm_extract_epi16::<2>(u16s) as u16;
            dest[3] = _mm_extract_epi16::<3>(u16s) as u16;
        }
        impl_(token(), self.0, dest)
    }

    impl_f32_array_interface!();

    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SSE4.2 doesn't have F16C, use scalar conversion
        let mut result = [0.0f32; 4];
        for i in 0..4 {
            result[i] = crate::f16::from_bits(mem[i]).to_f32();
        }
        Self::load(d, &result)
    }

    #[inline(always)]
    fn store_f16_bits(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        // SSE4.2 doesn't have F16C, use scalar conversion
        let mut tmp = [0.0f32; 4];
        self.store(&mut tmp);
        for i in 0..4 {
            dest[i] = crate::f16::from_f32(tmp[i]).to_bits();
        }
    }

    #[inline(always)]
    fn transpose_square(d: Self::Descriptor, data: &mut [Self::UnderlyingArray], stride: usize) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, d: Sse42Descriptor, data: &mut [[f32; 4]], stride: usize) {
            assert!(data.len() > stride * 3);

            let p0 = F32VecSse42::load_array(d, &data[0]).0;
            let p1 = F32VecSse42::load_array(d, &data[1 * stride]).0;
            let p2 = F32VecSse42::load_array(d, &data[2 * stride]).0;
            let p3 = F32VecSse42::load_array(d, &data[3 * stride]).0;

            let q0 = _mm_unpacklo_ps(p0, p2);
            let q1 = _mm_unpacklo_ps(p1, p3);
            let q2 = _mm_unpackhi_ps(p0, p2);
            let q3 = _mm_unpackhi_ps(p1, p3);

            let r0 = _mm_unpacklo_ps(q0, q1);
            let r1 = _mm_unpackhi_ps(q0, q1);
            let r2 = _mm_unpacklo_ps(q2, q3);
            let r3 = _mm_unpackhi_ps(q2, q3);

            F32VecSse42(r0, d).store_array(&mut data[0]);
            F32VecSse42(r1, d).store_array(&mut data[1 * stride]);
            F32VecSse42(r2, d).store_array(&mut data[2 * stride]);
            F32VecSse42(r3, d).store_array(&mut data[3 * stride]);
        }
        impl_(token(), d, data, stride)
    }
}

impl Add<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn add(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn sub(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn mul(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecSse42> for F32VecSse42 {
    type Output = F32VecSse42;
    fn_sse42!(this: F32VecSse42, fn div(rhs: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn add_assign(rhs: F32VecSse42) {
        this.0 = _mm_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn sub_assign(rhs: F32VecSse42) {
        this.0 = _mm_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn mul_assign(rhs: F32VecSse42) {
        this.0 = _mm_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecSse42> for F32VecSse42 {
    fn_sse42!(this: &mut F32VecSse42, fn div_assign(rhs: F32VecSse42) {
        this.0 = _mm_div_ps(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecSse42(__m128i, Sse42Descriptor);

impl I32SimdVec for I32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, mem: &[i32]) -> __m128i {
            _mm_loadu_si128(mem.first_chunk::<4>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128i, mem: &mut [i32]) {
            _mm_storeu_si128(mem.first_chunk_mut::<4>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: i32) -> __m128i {
            _mm_set1_epi32(v)
        }
        Self(impl_(token(), v), d)
    }

    fn_sse42!(this: I32VecSse42, fn as_f32() -> F32VecSse42 {
        F32VecSse42(_mm_cvtepi32_ps(this.0), this.1)
    });

    fn_sse42!(this: I32VecSse42, fn bitcast_to_f32() -> F32VecSse42 {
        F32VecSse42(_mm_castsi128_ps(this.0), this.1)
    });

    #[inline(always)]
    fn bitcast_to_u32(self) -> U32VecSse42 {
        U32VecSse42(self.0, self.1)
    }

    fn_sse42!(this: I32VecSse42, fn abs() -> I32VecSse42 {
        I32VecSse42(
            _mm_abs_epi32(
                this.0,
            ),
            this.1)
    });

    fn_sse42!(this: I32VecSse42, fn gt(rhs: I32VecSse42) -> MaskSse42 {
        MaskSse42(
            _mm_castsi128_ps(_mm_cmpgt_epi32(this.0, rhs.0)),
            this.1,
        )
    });

    fn_sse42!(this: I32VecSse42, fn lt_zero() -> MaskSse42 {
        I32VecSse42(_mm_setzero_si128(), this.1).gt(this)
    });

    fn_sse42!(this: I32VecSse42, fn eq(rhs: I32VecSse42) -> MaskSse42 {
        MaskSse42(
            _mm_castsi128_ps(_mm_cmpeq_epi32(this.0, rhs.0)),
            this.1,
        )
    });

    fn_sse42!(this: I32VecSse42, fn eq_zero() -> MaskSse42 {
        this.eq(I32VecSse42(_mm_setzero_si128(), this.1))
    });

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        #[arcane]
        fn impl_<const AMOUNT_I: i32>(_: archmage::X64V2Token, v: __m128i) -> __m128i {
            _mm_slli_epi32::<AMOUNT_I>(v)
        }
        Self(impl_::<AMOUNT_I>(token(), self.0), self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        #[arcane]
        fn impl_<const AMOUNT_I: i32>(_: archmage::X64V2Token, v: __m128i) -> __m128i {
            _mm_srai_epi32::<AMOUNT_I>(v)
        }
        Self(impl_::<AMOUNT_I>(token(), self.0), self.1)
    }

    fn_sse42!(this: I32VecSse42, fn mul_wide_take_high(rhs: I32VecSse42) -> I32VecSse42 {
        let l = _mm_mul_epi32(this.0, rhs.0);
        let h = _mm_mul_epi32(_mm_srli_epi64::<32>(this.0), _mm_srli_epi64::<32>(rhs.0));
        let p0 = _mm_unpacklo_epi32(l, h);
        let p1 = _mm_unpackhi_epi32(l, h);
        I32VecSse42(_mm_unpackhi_epi64(p0, p1), this.1)
    });

    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128i, dest: &mut [u16]) {
            assert!(dest.len() >= I32VecSse42::LEN);
            let mut tmp = [0i32; 4];
            _mm_storeu_si128(&mut tmp, v);
            for i in 0..4 {
                dest[i] = tmp[i] as u16;
            }
        }
        impl_(token(), self.0, dest)
    }
}

impl Add<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn add(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_add_epi32(this.0, rhs.0), this.1)
    });
}

impl Sub<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn sub(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_sub_epi32(this.0, rhs.0), this.1)
    });
}

impl Mul<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn mul(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_mullo_epi32(this.0, rhs.0), this.1)
    });
}

impl Neg for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn neg() -> I32VecSse42 {
        I32VecSse42(_mm_setzero_si128(), this.1) - this
    });
}

impl BitAnd<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn bitand(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_and_si128(this.0, rhs.0), this.1)
    });
}

impl BitOr<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn bitor(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_or_si128(this.0, rhs.0), this.1)
    });
}

impl BitXor<I32VecSse42> for I32VecSse42 {
    type Output = I32VecSse42;
    fn_sse42!(this: I32VecSse42, fn bitxor(rhs: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_xor_si128(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn add_assign(rhs: I32VecSse42) {
        this.0 = _mm_add_epi32(this.0, rhs.0)
    });
}

impl SubAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn sub_assign(rhs: I32VecSse42) {
        this.0 = _mm_sub_epi32(this.0, rhs.0)
    });
}

impl MulAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn mul_assign(rhs: I32VecSse42) {
        this.0 = _mm_mullo_epi32(this.0, rhs.0)
    });
}

impl BitAndAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn bitand_assign(rhs: I32VecSse42) {
        this.0 = _mm_and_si128(this.0, rhs.0)
    });
}

impl BitOrAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn bitor_assign(rhs: I32VecSse42) {
        this.0 = _mm_or_si128(this.0, rhs.0)
    });
}

impl BitXorAssign<I32VecSse42> for I32VecSse42 {
    fn_sse42!(this: &mut I32VecSse42, fn bitxor_assign(rhs: I32VecSse42) {
        this.0 = _mm_xor_si128(this.0, rhs.0)
    });
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecSse42(__m128i, Sse42Descriptor);

impl U32SimdVec for U32VecSse42 {
    type Descriptor = Sse42Descriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn bitcast_to_i32(self) -> I32VecSse42 {
        I32VecSse42(self.0, self.1)
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        #[arcane]
        fn impl_<const AMOUNT_I: i32>(_: archmage::X64V2Token, v: __m128i) -> __m128i {
            _mm_srli_epi32::<AMOUNT_I>(v)
        }
        Self(impl_::<AMOUNT_I>(token(), self.0), self.1)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecSse42(__m128i, Sse42Descriptor);

impl U8SimdVec for U8VecSse42 {
    type Descriptor = Sse42Descriptor;
    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, mem: &[u8]) -> __m128i {
            _mm_loadu_si128(mem.first_chunk::<16>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: u8) -> __m128i {
            _mm_set1_epi8(v as i8)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128i, mem: &mut [u8]) {
            _mm_storeu_si128(mem.first_chunk_mut::<16>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128i, b: __m128i, dest: &mut [u8]) {
            assert!(dest.len() >= 2 * U8VecSse42::LEN);
            let lo = _mm_unpacklo_epi8(a, b);
            let hi = _mm_unpackhi_epi8(a, b);
            _mm_storeu_si128(dest[..16].first_chunk_mut::<16>().unwrap(), lo);
            _mm_storeu_si128(dest[16..32].first_chunk_mut::<16>().unwrap(), hi);
        }
        impl_(token(), a.0, b.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128i, b: __m128i, c: __m128i, dest: &mut [u8]) {
            assert!(dest.len() >= 3 * U8VecSse42::LEN);

            let mask_a0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
            let mask_b0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
            let mask_c0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);

            let mask_a1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
            let mask_b1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
            let mask_c1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);

            let mask_a2 = _mm_setr_epi8(
                -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
            );
            let mask_b2 = _mm_setr_epi8(
                -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
            );
            let mask_c2 = _mm_setr_epi8(
                10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
            );

            let out0 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, mask_a0), _mm_shuffle_epi8(b, mask_b0)),
                _mm_shuffle_epi8(c, mask_c0),
            );
            let out1 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, mask_a1), _mm_shuffle_epi8(b, mask_b1)),
                _mm_shuffle_epi8(c, mask_c1),
            );
            let out2 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, mask_a2), _mm_shuffle_epi8(b, mask_b2)),
                _mm_shuffle_epi8(c, mask_c2),
            );

            _mm_storeu_si128(dest[..16].first_chunk_mut::<16>().unwrap(), out0);
            _mm_storeu_si128(dest[16..32].first_chunk_mut::<16>().unwrap(), out1);
            _mm_storeu_si128(dest[32..48].first_chunk_mut::<16>().unwrap(), out2);
        }
        impl_(token(), a.0, b.0, c.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u8]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128i, b: __m128i, c: __m128i, d: __m128i, dest: &mut [u8]) {
            assert!(dest.len() >= 4 * U8VecSse42::LEN);
            let ab_lo = _mm_unpacklo_epi8(a, b);
            let ab_hi = _mm_unpackhi_epi8(a, b);
            let cd_lo = _mm_unpacklo_epi8(c, d);
            let cd_hi = _mm_unpackhi_epi8(c, d);

            let out0 = _mm_unpacklo_epi16(ab_lo, cd_lo);
            let out1 = _mm_unpackhi_epi16(ab_lo, cd_lo);
            let out2 = _mm_unpacklo_epi16(ab_hi, cd_hi);
            let out3 = _mm_unpackhi_epi16(ab_hi, cd_hi);

            _mm_storeu_si128(dest[..16].first_chunk_mut::<16>().unwrap(), out0);
            _mm_storeu_si128(dest[16..32].first_chunk_mut::<16>().unwrap(), out1);
            _mm_storeu_si128(dest[32..48].first_chunk_mut::<16>().unwrap(), out2);
            _mm_storeu_si128(dest[48..64].first_chunk_mut::<16>().unwrap(), out3);
        }
        impl_(token(), a.0, b.0, c.0, d.0, dest)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecSse42(__m128i, Sse42Descriptor);

impl U16SimdVec for U16VecSse42 {
    type Descriptor = Sse42Descriptor;
    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, mem: &[u16]) -> __m128i {
            _mm_loadu_si128(mem.first_chunk::<8>().unwrap())
        }
        Self(impl_(token(), mem), d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: u16) -> __m128i {
            _mm_set1_epi16(v as i16)
        }
        Self(impl_(token(), v), d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, v: __m128i, mem: &mut [u16]) {
            _mm_storeu_si128(mem.first_chunk_mut::<8>().unwrap(), v)
        }
        impl_(token(), self.0, mem)
    }

    #[inline(always)]
    fn store_interleaved_2(a: Self, b: Self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128i, b: __m128i, dest: &mut [u16]) {
            assert!(dest.len() >= 2 * U16VecSse42::LEN);
            let lo = _mm_unpacklo_epi16(a, b);
            let hi = _mm_unpackhi_epi16(a, b);
            _mm_storeu_si128(dest[..8].first_chunk_mut::<8>().unwrap(), lo);
            _mm_storeu_si128(dest[8..16].first_chunk_mut::<8>().unwrap(), hi);
        }
        impl_(token(), a.0, b.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_3(a: Self, b: Self, c: Self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128i, b: __m128i, c: __m128i, dest: &mut [u16]) {
            assert!(dest.len() >= 3 * U16VecSse42::LEN);

            let mask_a0 = _mm_setr_epi8(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
            let mask_b0 = _mm_setr_epi8(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
            let mask_c0 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);

            let mask_a1 = _mm_setr_epi8(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
            let mask_b1 = _mm_setr_epi8(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
            let mask_c1 = _mm_setr_epi8(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);

            let mask_a2 = _mm_setr_epi8(
                -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
            );
            let mask_b2 = _mm_setr_epi8(
                10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
            );
            let mask_c2 = _mm_setr_epi8(
                -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
            );

            let out0 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, mask_a0), _mm_shuffle_epi8(b, mask_b0)),
                _mm_shuffle_epi8(c, mask_c0),
            );
            let out1 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, mask_a1), _mm_shuffle_epi8(b, mask_b1)),
                _mm_shuffle_epi8(c, mask_c1),
            );
            let out2 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, mask_a2), _mm_shuffle_epi8(b, mask_b2)),
                _mm_shuffle_epi8(c, mask_c2),
            );

            _mm_storeu_si128(dest[..8].first_chunk_mut::<8>().unwrap(), out0);
            _mm_storeu_si128(dest[8..16].first_chunk_mut::<8>().unwrap(), out1);
            _mm_storeu_si128(dest[16..24].first_chunk_mut::<8>().unwrap(), out2);
        }
        impl_(token(), a.0, b.0, c.0, dest)
    }

    #[inline(always)]
    fn store_interleaved_4(a: Self, b: Self, c: Self, d: Self, dest: &mut [u16]) {
        #[arcane]
        fn impl_(_: archmage::X64V2Token, a: __m128i, b: __m128i, c: __m128i, d: __m128i, dest: &mut [u16]) {
            assert!(dest.len() >= 4 * U16VecSse42::LEN);
            let ab_lo = _mm_unpacklo_epi16(a, b);
            let ab_hi = _mm_unpackhi_epi16(a, b);
            let cd_lo = _mm_unpacklo_epi16(c, d);
            let cd_hi = _mm_unpackhi_epi16(c, d);

            let out0 = _mm_unpacklo_epi32(ab_lo, cd_lo);
            let out1 = _mm_unpackhi_epi32(ab_lo, cd_lo);
            let out2 = _mm_unpacklo_epi32(ab_hi, cd_hi);
            let out3 = _mm_unpackhi_epi32(ab_hi, cd_hi);

            _mm_storeu_si128(dest[..8].first_chunk_mut::<8>().unwrap(), out0);
            _mm_storeu_si128(dest[8..16].first_chunk_mut::<8>().unwrap(), out1);
            _mm_storeu_si128(dest[16..24].first_chunk_mut::<8>().unwrap(), out2);
            _mm_storeu_si128(dest[24..32].first_chunk_mut::<8>().unwrap(), out3);
        }
        impl_(token(), a.0, b.0, c.0, d.0, dest)
    }
}

impl SimdMask for MaskSse42 {
    type Descriptor = Sse42Descriptor;

    fn_sse42!(this: MaskSse42, fn if_then_else_f32(if_true: F32VecSse42, if_false: F32VecSse42) -> F32VecSse42 {
        F32VecSse42(_mm_blendv_ps(if_false.0, if_true.0, this.0), this.1)
    });

    fn_sse42!(this: MaskSse42, fn if_then_else_i32(if_true: I32VecSse42, if_false: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_blendv_epi8(if_false.0, if_true.0, _mm_castps_si128(this.0)), this.1)
    });

    fn_sse42!(this: MaskSse42, fn maskz_i32(v: I32VecSse42) -> I32VecSse42 {
        I32VecSse42(_mm_andnot_si128(_mm_castps_si128(this.0), v.0), this.1)
    });

    fn_sse42!(this: MaskSse42, fn all() -> bool {
        _mm_movemask_ps(this.0) == 0b1111
    });

    fn_sse42!(this: MaskSse42, fn andnot(rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_andnot_ps(this.0, rhs.0), this.1)
    });
}

impl BitAnd<MaskSse42> for MaskSse42 {
    type Output = MaskSse42;
    fn_sse42!(this: MaskSse42, fn bitand(rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_and_ps(this.0, rhs.0), this.1)
    });
}

impl BitOr<MaskSse42> for MaskSse42 {
    type Output = MaskSse42;
    fn_sse42!(this: MaskSse42, fn bitor(rhs: MaskSse42) -> MaskSse42 {
        MaskSse42(_mm_or_ps(this.0, rhs.0), this.1)
    });
}
