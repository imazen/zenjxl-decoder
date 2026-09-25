// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::{Add, Shl, Shr, Sub};

pub trait ShiftRightCeil: Copy {
    fn shrc<T: Copy>(self, rhs: T) -> Self
    where
        Self: Shr<T, Output = Self> + Shl<T, Output = Self>;
}

impl<S: Copy + PartialEq + Add<Self, Output = Self> + Sub<Self, Output = Self> + From<u8>>
    ShiftRightCeil for S
{
    fn shrc<T: Copy>(self, rhs: T) -> Self
    where
        Self: Shr<T, Output = Self> + Shl<T, Output = Self>,
    {
        // `(self + (1 << rhs) - 1) >> rhs` overflows for values near MAX (and
        // wraps to a tiny result in release builds). This form cannot.
        // Ported from upstream jxl-rs ee7c7c5.
        if self == Self::from(0u8) {
            Self::from(0u8)
        } else {
            ((self - Self::from(1u8)) >> rhs) + Self::from(1u8)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::util::ShiftRightCeil;

    #[test]
    fn test_shrc() {
        assert_eq!(0u32, 0u32.shrc(1u32));
        assert_eq!(0u32, 0u32.shrc(3u32));
        // Near MAX the old `(x + (1 << r) - 1) >> r` overflowed.
        assert_eq!(1u32 << 31, u32::MAX.shrc(1u32));
        assert_eq!(1usize << (usize::BITS - 3), usize::MAX.shrc(3usize));
        assert_eq!(1u8, 1u8.shrc(1u8));
        assert_eq!(1u8, 2u8.shrc(1u8));
        assert_eq!(2u8, 9u8.shrc(3u8));
        assert_eq!(1u32, 1u32.shrc(1u32));
        assert_eq!(1u32, 2u32.shrc(1u32));
        assert_eq!(2u32, 9u32.shrc(3u32));
        assert_eq!(1u32, 1u32.shrc(1u8));
        assert_eq!(1u32, 2u32.shrc(1u8));
        assert_eq!(2u32, 9u32.shrc(3u8));
    }
}
