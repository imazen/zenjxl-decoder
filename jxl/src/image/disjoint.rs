// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Safe parallel row access for `Image<i32>`.
//!
//! [`DisjointRowAccess`] takes ownership of an `Image<i32>` and provides
//! concurrent access to individual rows. Each row can be borrowed by at most
//! one thread at a time; this invariant is enforced at runtime in debug builds
//! via a `Mutex<HashSet>` borrow tracker.
//!
//! In release builds the tracking is compiled out, leaving only a bounds check
//! and a raw-pointer dereference — the same cost as the old `SplitRowAccess`.

use super::Image;

/// Safe parallel row access backed by raw pointers with debug-mode borrow tracking.
///
/// Takes ownership of an `Image<i32>` and exposes individual rows through
/// [`DisjointRowGuard`] RAII guards.
///
/// # Safety Model
///
/// This type contains `unsafe` code internally but exposes a **safe** API.
/// The internal `unsafe` is sound because:
/// - The raw pointer is derived from the owned Image's backing `Vec<u8>`
/// - The Image is kept alive for the lifetime of this struct
/// - Each row occupies non-overlapping memory (`row_stride >= row_len`)
/// - Debug builds verify no two concurrent accesses use the same row
pub(crate) struct DisjointRowAccess {
    image: Image<i32>,
    ptr: *mut i32,
    row_stride: usize, // in i32 elements
    row_len: usize,    // in i32 elements
    num_rows: usize,
    #[cfg(debug_assertions)]
    borrowed: std::sync::Mutex<std::collections::HashSet<usize>>,
}

// SAFETY: The backing memory is owned (via Image) and each parallel worker
// accesses a distinct row. Debug builds verify this with runtime tracking.
#[allow(unsafe_code)]
unsafe impl Send for DisjointRowAccess {}
// SAFETY: Concurrent access to different rows is safe because rows occupy
// non-overlapping memory. Debug builds verify no two threads access the same row.
#[allow(unsafe_code)]
unsafe impl Sync for DisjointRowAccess {}

impl DisjointRowAccess {
    /// Takes ownership of an `Image<i32>` and prepares it for disjoint row access.
    #[allow(unsafe_code)]
    pub(crate) fn from_image(mut image: Image<i32>) -> Self {
        let (ptr, row_stride, row_len, num_rows) = image.row_info_mut();
        DisjointRowAccess {
            image,
            ptr,
            row_stride,
            row_len,
            num_rows,
            #[cfg(debug_assertions)]
            borrowed: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Returns a guard providing `&mut [i32]` access to the specified row.
    ///
    /// Panics if `row >= num_rows` or (in debug builds) if the row is already borrowed.
    #[allow(unsafe_code)]
    pub(crate) fn row_guard(&self, row: usize) -> DisjointRowGuard<'_> {
        assert!(
            row < self.num_rows,
            "DisjointRowAccess: row {row} out of bounds (num_rows: {})",
            self.num_rows
        );
        #[cfg(debug_assertions)]
        {
            let mut borrowed = self.borrowed.lock().unwrap();
            assert!(
                borrowed.insert(row),
                "DisjointRowAccess: row {row} is already borrowed"
            );
        }
        let offset = row * self.row_stride;
        // SAFETY: ptr is valid for the image's lifetime (we own it).
        // Each row at ptr + row*row_stride occupies non-overlapping memory.
        // Debug builds verify no double-borrow via the Mutex<HashSet>.
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), self.row_len) };
        DisjointRowGuard {
            slice,
            #[cfg(debug_assertions)]
            access: self,
            #[cfg(debug_assertions)]
            row,
        }
    }

    /// Consumes this wrapper and returns the underlying `Image<i32>`.
    pub(crate) fn into_image(self) -> Image<i32> {
        #[cfg(debug_assertions)]
        {
            let borrowed = self.borrowed.lock().unwrap();
            assert!(
                borrowed.is_empty(),
                "DisjointRowAccess dropped with active borrows: {borrowed:?}"
            );
        }
        self.image
    }
}

/// RAII guard providing `&mut [i32]` access to a single row.
///
/// In debug builds, releases the borrow tracking on drop.
pub(crate) struct DisjointRowGuard<'a> {
    slice: &'a mut [i32],
    #[cfg(debug_assertions)]
    access: &'a DisjointRowAccess,
    #[cfg(debug_assertions)]
    row: usize,
}

impl DisjointRowGuard<'_> {
    /// Returns a mutable reference to the row's data.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [i32] {
        self.slice
    }
}

impl std::ops::Deref for DisjointRowGuard<'_> {
    type Target = [i32];
    fn deref(&self) -> &[i32] {
        self.slice
    }
}

impl std::ops::DerefMut for DisjointRowGuard<'_> {
    fn deref_mut(&mut self) -> &mut [i32] {
        self.slice
    }
}

impl Drop for DisjointRowGuard<'_> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        {
            self.access.borrowed.lock().unwrap().remove(&self.row);
        }
    }
}
