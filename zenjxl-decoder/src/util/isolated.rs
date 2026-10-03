// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Small buffers that a decoding thread writes for every sample.

use std::ops::{Deref, DerefMut};

/// Bytes kept unused on each side of the elements: the largest cache line
/// of the supported targets (128 on Apple M-series).
const GUARD_BYTES: usize = 128;

/// A zeroed buffer whose elements share no cache line with any other
/// allocation.
///
/// A few-dozen-byte `Vec` written per sample can land in the same cache line
/// as another thread's data (its MA tree, its own property buffer), and the
/// line then moves between cores on every write. On M4 Pro this made 8
/// threads decode lossless effort-7 groups ~20% slower each than 4 threads.
pub struct IsolatedBuf<T> {
    buf: Vec<T>,
    len: usize,
}

impl<T: Copy + Default> IsolatedBuf<T> {
    const GUARD: usize = GUARD_BYTES.div_ceil(std::mem::size_of::<T>());

    pub fn new(len: usize) -> Self {
        Self {
            buf: vec![T::default(); len + 2 * Self::GUARD],
            len,
        }
    }
}

impl<T: Copy + Default> Deref for IsolatedBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        &self.buf[Self::GUARD..Self::GUARD + self.len]
    }
}

impl<T: Copy + Default> DerefMut for IsolatedBuf<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf[Self::GUARD..Self::GUARD + self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::IsolatedBuf;

    #[test]
    fn elements_have_a_guard_on_both_sides() {
        let mut b = IsolatedBuf::<i32>::new(13);
        assert_eq!(b.len(), 13);
        assert!(b.iter().all(|&v| v == 0));
        b[12] = 7;
        let start = b.as_ptr() as usize - b.buf.as_ptr() as usize;
        let end = b.buf.len() * 4 - (start + 13 * 4);
        assert!(start >= 128 && end >= 128);
        assert_eq!(b.buf[IsolatedBuf::<i32>::GUARD + 12], 7);
    }
}
