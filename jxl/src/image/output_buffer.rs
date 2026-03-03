// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use super::{RawImageRectMut, Rect, internal::RawImageBuffer};

pub struct JxlOutputBuffer<'a> {
    storage: &'a mut [u8],
    bytes_per_row: usize,
    num_rows: usize,
    bytes_between_rows: usize,
}

impl<'a> JxlOutputBuffer<'a> {
    pub fn from_image_rect_mut(raw: RawImageRectMut<'a>) -> Self {
        Self {
            storage: raw.storage,
            bytes_per_row: raw.bytes_per_row,
            num_rows: raw.num_rows,
            bytes_between_rows: raw.bytes_between_rows,
        }
    }

    /// Creates a new JxlOutputBuffer from a mutable byte slice.
    pub fn new(buf: &'a mut [u8], num_rows: usize, bytes_per_row: usize) -> Self {
        RawImageBuffer::check_vals(num_rows, bytes_per_row, bytes_per_row);
        let expected_len = if num_rows == 0 {
            0
        } else {
            (num_rows - 1) * bytes_per_row + bytes_per_row
        };
        assert!(buf.len() >= expected_len);
        Self {
            storage: if expected_len == 0 {
                &mut []
            } else {
                &mut buf[..expected_len]
            },
            bytes_per_row,
            num_rows,
            bytes_between_rows: bytes_per_row,
        }
    }

    /// Creates a new JxlOutputBuffer from raw pointers.
    /// It is guaranteed that `buf` will never be used to write uninitialized data.
    ///
    /// # Safety
    /// - `buf` must be valid for writes for all bytes in the range
    ///   `buf[i*bytes_between_rows..i*bytes_between_rows+bytes_per_row]` for all values of `i`
    ///   from `0` to `num_rows-1`.
    /// - The bytes in these ranges must not be accessed as long as the returned `Self` is in scope.
    /// - All the bytes in those ranges (and in between) must be part of the same allocated object.
    #[cfg(feature = "allow-unsafe")]
    #[allow(unsafe_code)]
    pub unsafe fn new_from_ptr(
        buf: *mut std::mem::MaybeUninit<u8>,
        num_rows: usize,
        bytes_per_row: usize,
        bytes_between_rows: usize,
    ) -> Self {
        RawImageBuffer::check_vals(num_rows, bytes_per_row, bytes_between_rows);
        let total_len = if num_rows == 0 {
            0
        } else {
            (num_rows - 1) * bytes_between_rows + bytes_per_row
        };
        let storage = if total_len == 0 {
            &mut []
        } else {
            // SAFETY: Caller guarantees `buf` is valid for `total_len` bytes.
            // MaybeUninit<u8> and u8 have identical layout.
            unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, total_len) }
        };
        Self {
            storage,
            bytes_per_row,
            num_rows,
            bytes_between_rows,
        }
    }

    pub(crate) fn reborrow(lender: &'a mut JxlOutputBuffer<'_>) -> JxlOutputBuffer<'a> {
        JxlOutputBuffer {
            storage: &mut *lender.storage,
            bytes_per_row: lender.bytes_per_row,
            num_rows: lender.num_rows,
            bytes_between_rows: lender.bytes_between_rows,
        }
    }

    /// Returns a mutable row as a byte slice.
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [u8] {
        assert!(row < self.num_rows);
        let start = row * self.bytes_between_rows;
        &mut self.storage[start..start + self.bytes_per_row]
    }

    #[inline]
    pub fn write_bytes(&mut self, row: usize, col: usize, bytes: &[u8]) {
        let slice = self.row_mut(row);
        slice[col..col + bytes.len()].copy_from_slice(bytes);
    }

    pub fn byte_size(&self) -> (usize, usize) {
        (self.bytes_per_row, self.num_rows)
    }

    pub fn rect(&mut self, rect: Rect) -> JxlOutputBuffer<'_> {
        if rect.size.0 == 0 || rect.size.1 == 0 {
            return JxlOutputBuffer {
                storage: &mut [],
                bytes_per_row: 0,
                num_rows: 0,
                bytes_between_rows: 0,
            };
        }
        assert!(rect.origin.1 + rect.size.1 <= self.num_rows);
        assert!(rect.origin.0 + rect.size.0 <= self.bytes_per_row);
        let new_start = rect.origin.1 * self.bytes_between_rows + rect.origin.0;
        let data_span = (rect.size.1 - 1) * self.bytes_between_rows + rect.size.0;
        assert!(new_start + data_span <= self.storage.len());
        JxlOutputBuffer {
            storage: &mut self.storage[new_start..new_start + data_span],
            bytes_per_row: rect.size.0,
            num_rows: rect.size.1,
            bytes_between_rows: self.bytes_between_rows,
        }
    }
}

impl Debug for JxlOutputBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "JxlOutputBuffer {}x{}",
            self.bytes_per_row, self.num_rows
        )
    }
}
