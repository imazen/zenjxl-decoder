// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! IEEE 754 half-precision (binary16) floating-point type.
//!
//! This is a minimal implementation providing only the operations needed for JPEG XL decoding,
//! avoiding external dependencies like `half` which pulls in `zerocopy`.

/// IEEE 754 binary16 half-precision floating-point type.
///
/// Format: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct f16(u16);

impl f16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0);

    /// Creates an f16 from its raw bit representation.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x3FF) as u32;

        let f32_bits = if exp == 0 {
            if mant == 0 {
                // Zero (signed)
                sign << 31
            } else {
                // Denormal f16 -> normalized f32
                // Find the leading 1 bit in mantissa
                let mut m = mant;
                let mut e = 0u32;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                m &= 0x3FF; // Remove the implicit leading 1
                // f16 denormal exponent is -14 (not -15), adjust by shift count
                let new_exp = 127 - 14 - e;
                (sign << 31) | (new_exp << 23) | (m << 13)
            }
        } else if exp == 31 {
            // Infinity or NaN
            if mant == 0 {
                // Infinity
                (sign << 31) | (0xFF << 23)
            } else {
                // NaN - preserve some payload bits, ensure quiet NaN
                (sign << 31) | (0xFF << 23) | (mant << 13) | 0x0040_0000
            }
        } else {
            // Normal number
            // Rebias: f16 uses bias 15, f32 uses bias 127
            // new_exp = exp - 15 + 127 = exp + 112
            let new_exp = exp + 112;
            (sign << 31) | (new_exp << 23) | (mant << 13)
        };

        f32::from_bits(f32_bits)
    }

    /// Creates an f16 from an f32.
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        let bits = f.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        let h_bits = if exp == 0 {
            // Zero or f32 denormal -> f16 zero (too small)
            sign << 15
        } else if exp == 255 {
            // Infinity or NaN
            if mant == 0 {
                (sign << 15) | (0x1F << 10) // Infinity
            } else {
                (sign << 15) | (0x1F << 10) | 0x0200 // Quiet NaN
            }
        } else {
            let unbiased = exp - 127;

            if unbiased < -25 {
                // Too small, underflow to zero
                sign << 15
            } else if unbiased < -14 {
                // Denormal f16. The 24-bit significand (implicit bit included)
                // is scaled to units of 2^-24 (the f16 subnormal ULP):
                // value = full * 2^(unbiased - 23), so the result mantissa is
                // full >> (24 - 1 - (unbiased + 14)) = full >> (13 - 14 - unbiased).
                // Round to nearest, ties to even, like the normal path.
                let shift = (-14 - unbiased + 13) as u32;
                let full = mant | 0x0080_0000;
                let m = (full >> shift) as u16;
                let round_bit = (full >> (shift - 1)) & 1;
                let sticky = full & ((1 << (shift - 1)) - 1);
                let m = if round_bit == 1 && (sticky != 0 || (m & 1) == 1) {
                    m + 1
                } else {
                    m
                };
                (sign << 15) | m
            } else if unbiased > 15 {
                // Overflow to infinity
                (sign << 15) | (0x1F << 10)
            } else {
                // Normal f16
                let h_exp = (unbiased + 15) as u16;
                let h_mant = (mant >> 13) as u16;

                // Round to nearest, ties to even
                let round_bit = (mant >> 12) & 1;
                let sticky = mant & 0x0FFF;
                let h_mant = if round_bit == 1 && (sticky != 0 || (h_mant & 1) == 1) {
                    h_mant + 1
                } else {
                    h_mant
                };

                // Handle mantissa overflow from rounding
                if h_mant > 0x3FF {
                    if h_exp >= 30 {
                        // Overflow to infinity
                        (sign << 15) | (0x1F << 10)
                    } else {
                        (sign << 15) | ((h_exp + 1) << 10)
                    }
                } else {
                    (sign << 15) | (h_exp << 10) | h_mant
                }
            }
        };

        Self(h_bits)
    }

    /// Creates an f16 from an f64.
    #[inline]
    pub fn from_f64(f: f64) -> Self {
        // Convert via f32 - sufficient precision for f16
        Self::from_f32(f as f32)
    }

    /// Converts to f64.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Returns true if this is neither infinite nor NaN.
    #[inline]
    pub fn is_finite(self) -> bool {
        // Exponent of 31 means infinity or NaN
        ((self.0 >> 10) & 0x1F) != 31
    }

    /// Returns the bytes in little-endian order.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Returns the bytes in big-endian order.
    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 2] {
        self.0.to_be_bytes()
    }
}

impl From<f16> for f32 {
    #[inline]
    fn from(f: f16) -> f32 {
        f.to_f32()
    }
}

impl From<f16> for f64 {
    #[inline]
    fn from(f: f16) -> f64 {
        f.to_f64()
    }
}

impl core::fmt::Debug for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl core::fmt::Display for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let z = f16::ZERO;
        assert_eq!(z.to_bits(), 0);
        assert_eq!(z.to_f32(), 0.0);
        assert!(z.is_finite());
    }

    #[test]
    fn test_one() {
        // 1.0 in f16: sign=0, exp=15 (biased), mant=0 -> 0x3C00
        let one = f16::from_bits(0x3C00);
        assert!((one.to_f32() - 1.0).abs() < 1e-6);
        assert!(one.is_finite());
    }

    #[test]
    fn test_negative_one() {
        // -1.0 in f16: sign=1, exp=15, mant=0 -> 0xBC00
        let neg_one = f16::from_bits(0xBC00);
        assert!((neg_one.to_f32() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_infinity() {
        // +Inf: sign=0, exp=31, mant=0 -> 0x7C00
        let inf = f16::from_bits(0x7C00);
        assert!(inf.to_f32().is_infinite());
        assert!(!inf.is_finite());

        // -Inf: 0xFC00
        let neg_inf = f16::from_bits(0xFC00);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(!neg_inf.is_finite());
    }

    #[test]
    fn test_nan() {
        // NaN: exp=31, mant!=0 -> 0x7C01 (or any mant != 0)
        let nan = f16::from_bits(0x7C01);
        assert!(nan.to_f32().is_nan());
        assert!(!nan.is_finite());
    }

    #[test]
    fn test_denormal() {
        // Smallest positive denormal: 0x0001
        let tiny = f16::from_bits(0x0001);
        let val = tiny.to_f32();
        assert!(val > 0.0);
        assert!(val < 1e-6);
        assert!(tiny.is_finite());
    }

    #[test]
    fn test_roundtrip_normal() {
        let test_values: [f32; 8] = [0.5, 1.0, 2.0, 100.0, 0.001, -0.5, -1.0, -100.0];
        for &v in &test_values {
            let h = f16::from_f32(v);
            let back = h.to_f32();
            // f16 has limited precision, allow ~0.1% error for normal values
            let rel_err = ((v - back) / v).abs();
            assert!(
                rel_err < 0.002,
                "Roundtrip failed for {}: got {}, rel_err {}",
                v,
                back,
                rel_err
            );
        }
    }

    #[test]
    fn test_roundtrip_special() {
        // Zero
        assert_eq!(f16::from_f32(0.0).to_f32(), 0.0);

        // Infinity
        assert!(f16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(f16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());

        // NaN
        assert!(f16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn test_overflow_to_infinity() {
        // f16 max is ~65504, values above should overflow to infinity
        let big = f16::from_f32(100000.0);
        assert!(big.to_f32().is_infinite());
    }

    #[test]
    fn test_underflow_to_zero() {
        // Very small values should underflow to zero
        let tiny = f16::from_f32(1e-10);
        assert_eq!(tiny.to_f32(), 0.0);
    }

    #[test]
    fn test_bytes() {
        let h = f16::from_bits(0x1234);
        assert_eq!(h.to_le_bytes(), [0x34, 0x12]);
        assert_eq!(h.to_be_bytes(), [0x12, 0x34]);
    }
    /// Reference f32 -> f16 conversion (IEEE 754 round-to-nearest-even),
    /// written independently of the implementation under test: it goes
    /// through integer arithmetic on the f32 bit pattern with an explicit
    /// round/sticky computation, so a shift-by-one error in `from_f32`
    /// cannot be mirrored here.
    fn reference_f32_to_f16_bits(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;
        if exp == 0xFF {
            return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
        }
        // Value = 1.mant * 2^(exp-127) (or 0.mant * 2^-126 when exp == 0).
        let (full, e) = if exp == 0 {
            (mant, -126)
        } else {
            (mant | 0x0080_0000, exp - 127)
        };
        // Number of mantissa bits to drop so that the result is a 10-bit f16
        // mantissa at the right exponent. For normals that's 13; subnormals
        // need more.
        let shift: i32 = if e >= -14 { 13 } else { 13 + (-14 - e) };
        if shift >= 32 {
            return sign; // far below the smallest subnormal
        }
        let m = full >> shift;
        let round_bit = (full >> (shift - 1)) & 1;
        let sticky = full & ((1u32 << (shift - 1)) - 1);
        let m = if round_bit == 1 && (sticky != 0 || (m & 1) == 1) {
            m + 1
        } else {
            m
        };
        // Assemble: for normals m has the implicit bit at position 10 and the
        // exponent field is e+15; carries out of the mantissa are absorbed
        // by the field arithmetic below (which is exactly what IEEE wants).
        let h = if e >= -14 {
            (((e + 15) as u32) << 10) + (m - 0x400)
        } else {
            m // subnormal (or rounds up into the smallest normal)
        };
        if h >= 0x7C00 {
            return sign | 0x7C00;
        }
        sign | h as u16
    }

    /// Every f32 in [2^-26, 2^-13) -- the whole f16-subnormal range plus one
    /// binade on each side -- must convert exactly like the IEEE reference.
    /// Before the fix every subnormal came out at half its value
    /// (0x0100 for 2^-15 instead of 0x0200) and the smallest ones flushed.
    #[test]
    fn test_from_f32_matches_ieee_over_subnormal_range() {
        let lo = (2.0f32).powi(-26).to_bits();
        let hi = (2.0f32).powi(-13).to_bits();
        // ~27M values; stride keeps the test fast while still covering every
        // exponent and every rounding pattern many times over.
        let mut checked = 0u32;
        let mut bits = lo;
        while bits < hi {
            let v = f32::from_bits(bits);
            let got = f16::from_f32(v).to_bits();
            let want = reference_f32_to_f16_bits(v);
            assert_eq!(
                got, want,
                "f16::from_f32({v:e}) = {got:#06x}, IEEE RNE wants {want:#06x}"
            );
            let gotn = f16::from_f32(-v).to_bits();
            assert_eq!(gotn, want | 0x8000, "negative mirror of {v:e}");
            checked += 1;
            bits += 97; // prime stride
        }
        assert!(checked > 100_000);
    }

    #[test]
    fn test_subnormal_exact_powers_of_two() {
        // 2^-15 is a subnormal with a single mantissa bit: 0x0200.
        assert_eq!(f16::from_f32((2.0f32).powi(-15)).to_bits(), 0x0200);
        assert_eq!(f16::from_f32((2.0f32).powi(-16)).to_bits(), 0x0100);
        // Smallest subnormal 2^-24 must not flush to zero.
        assert_eq!(f16::from_f32((2.0f32).powi(-24)).to_bits(), 0x0001);
        // Largest subnormal: (1023/1024) * 2^-14.
        assert_eq!(f16::from_f32(1023.0 * (2.0f32).powi(-24)).to_bits(), 0x03ff);
        // Just below the smallest normal rounds up into it (ties-to-even).
        assert_eq!(
            f16::from_f32((2.0f32).powi(-14) - (2.0f32).powi(-30)).to_bits(),
            0x0400
        );
    }

    #[test]
    fn test_subnormal_rounding_near_smallest() {
        // 2^-25 + 1 ULP should round to the smallest subnormal, not flush to zero.
        let v = f32::from_bits(0x33000001);
        assert_eq!(f16::from_f32(v).to_bits(), 0x0001);
        assert_eq!(f16::from_f32(-v).to_bits(), 0x8001);
        // Exactly 2^-25 is a tie and rounds to even (zero).
        assert_eq!(f16::from_f32(f32::from_bits(0x33000000)).to_bits(), 0x0000);
    }

    #[test]
    fn test_subnormal_roundtrip_is_monotonic_and_close() {
        // Converting a subnormal to f16 and back must stay within one f16 ULP
        // (2^-24) of the input -- the old code was off by a factor of two.
        let ulp = (2.0f32).powi(-24);
        let mut v = (2.0f32).powi(-24);
        let mut prev_bits = 0u16;
        while v < (2.0f32).powi(-14) {
            let h = f16::from_f32(v);
            assert!(
                (h.to_f32() - v).abs() <= ulp * 0.5 + f32::EPSILON,
                "roundtrip of {v:e} gave {:e}",
                h.to_f32()
            );
            assert!(h.to_bits() >= prev_bits, "non-monotonic at {v:e}");
            prev_bits = h.to_bits();
            v *= 1.01;
        }
    }
    #[test]
    fn test_to_f32_subnormals_exact() {
        // Every f16 subnormal is mant * 2^-24 exactly (representable in f32).
        let ulp = (2.0f32).powi(-24);
        for mant in 1u16..=0x3FF {
            let want = mant as f32 * ulp;
            assert_eq!(f16::from_bits(mant).to_f32(), want, "mant={mant:#05x}");
            assert_eq!(f16::from_bits(0x8000 | mant).to_f32(), -want);
            // and from_f32 must invert it exactly
            assert_eq!(f16::from_f32(want).to_bits(), mant);
        }
        // Smallest normal for reference: 2^-14.
        assert_eq!(f16::from_bits(0x0400).to_f32(), (2.0f32).powi(-14));
    }
}
