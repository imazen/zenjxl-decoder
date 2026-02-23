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
        let expected_len = if num_rows == 0 { 0 } else { (num_rows - 1) * bytes_per_row + bytes_per_row };
        assert!(buf.len() >= expected_len);
        Self {
            storage: if expected_len == 0 { &mut [] } else { &mut buf[..expected_len] },
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

    /// Creates a shared view for parallel access to non-overlapping sub-regions.
    ///
    /// # Safety
    /// - The returned SharedOutputView must not outlive this buffer's backing memory
    /// - This JxlOutputBuffer should not be used while SharedOutputView sub-views exist
    /// - All sub-views created from the SharedOutputView must access non-overlapping regions
    #[cfg(feature = "threads")]
    #[allow(unsafe_code)]
    pub(crate) unsafe fn shared_view(&self) -> SharedOutputView {
        SharedOutputView {
            ptr: self.storage.as_ptr() as *mut u8,
            len: self.storage.len(),
            bytes_per_row: self.bytes_per_row,
            num_rows: self.num_rows,
            bytes_between_rows: self.bytes_between_rows,
        }
    }
}

impl Debug for JxlOutputBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JxlOutputBuffer {}x{}", self.bytes_per_row, self.num_rows)
    }
}

/// Thread-safe shared access to an output buffer for parallel rendering.
///
/// Stores a raw pointer to allow creating multiple mutable sub-views for
/// non-overlapping regions across threads.
#[cfg(feature = "threads")]
pub(crate) struct SharedOutputView {
    ptr: *mut u8,
    len: usize,
    bytes_per_row: usize,
    num_rows: usize,
    bytes_between_rows: usize,
}

// SAFETY: SharedOutputView's raw pointer is only used to create non-overlapping
// &mut [u8] sub-views, with the safety contract requiring exclusive access to each region.
// The original data comes from a &mut [u8] which was valid for writes.
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
unsafe impl Send for SharedOutputView {}
// SAFETY: See Send impl above — non-overlapping access contract ensures soundness.
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
unsafe impl Sync for SharedOutputView {}

#[cfg(feature = "threads")]
#[allow(unsafe_code)]
impl SharedOutputView {
    /// Creates a sub-view for a rectangle (coordinates in bytes).
    ///
    /// # Safety
    /// The rectangle must not overlap with any other concurrently-accessed sub-view.
    pub(crate) unsafe fn sub_view(&self, rect: Rect) -> JxlOutputBuffer<'_> {
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
        assert!(new_start + data_span <= self.len);
        // SAFETY: Caller guarantees non-overlapping sub-views. The pointer arithmetic
        // stays within the original allocation (verified by the assert above).
        let sub_ptr = unsafe { self.ptr.add(new_start) };
        JxlOutputBuffer {
            // SAFETY: The sub-region [new_start..new_start+data_span] is within the
            // original buffer, and the caller guarantees exclusive access to this region.
            storage: unsafe { std::slice::from_raw_parts_mut(sub_ptr, data_span) },
            bytes_per_row: rect.size.0,
            num_rows: rect.size.1,
            bytes_between_rows: self.bytes_between_rows,
        }
    }
}
