// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;
use whereat::at;

use crate::{
    error::{Error, Result},
    util::{CACHE_LINE_BYTE_SIZE, tracing_wrappers::*},
};

/// Allocates zeroed memory, returning an error instead of aborting on OOM.
///
/// Always uses `try_reserve` to ensure proper error handling for all allocation
/// sizes. The `vec!` macro calls the global allocator which aborts on failure,
/// so we avoid it for any size that could plausibly fail. After `try_reserve`
/// succeeds, `resize` is guaranteed not to re-allocate.
fn alloc_zeroed_fallible(
    total_len: usize,
    bytes_per_row: usize,
    num_rows: usize,
) -> Result<Vec<u8>> {
    let mut storage = Vec::new();
    storage
        .try_reserve(total_len)
        .map_err(|_| Error::ImageOutOfMemory(bytes_per_row, num_rows))?;
    storage.resize(total_len, 0);
    Ok(storage)
}

/// Safe image buffer backed by `Vec<u8>` with cache-line alignment via offset.
///
/// For owned buffers, `storage` holds the allocation and `offset` points to the first
/// cache-line-aligned byte.
///
/// Invariants:
///  - If `num_rows > 0`, then `bytes_per_row > 0` and `bytes_per_row <= bytes_between_rows`.
///  - All accessible byte ranges `[offset + i*bytes_between_rows .. offset + i*bytes_between_rows + bytes_per_row]`
///    for i in 0..num_rows are within `storage.len()`.
///  - The computation `bytes_between_rows * (num_rows-1) + bytes_per_row` does not overflow
///    and has a result that is at most `isize::MAX`, or `num_rows` is 0.
#[derive(Debug, Clone)]
pub(super) struct RawImageBuffer {
    storage: Vec<u8>,
    offset: usize,
    bytes_per_row: usize,
    num_rows: usize,
    bytes_between_rows: usize,
}

impl RawImageBuffer {
    pub(super) fn check_vals(num_rows: usize, bytes_per_row: usize, bytes_between_rows: usize) {
        if num_rows > 0 {
            assert!(bytes_per_row > 0);
            assert!(bytes_between_rows >= bytes_per_row);
            assert!(
                bytes_between_rows
                    .checked_mul(num_rows - 1)
                    .unwrap()
                    .checked_add(bytes_per_row)
                    .unwrap()
                    <= isize::MAX as usize
            );
        }
    }

    /// Checks that the data pointer, bytes_per_row, and bytes_between_rows are all multiples of `align`.
    #[inline(always)]
    pub(super) fn is_aligned(&self, align: usize) -> bool {
        if self.num_rows == 0 {
            return true;
        }
        self.bytes_per_row.is_multiple_of(align)
            && self.bytes_between_rows.is_multiple_of(align)
            && self.data_ptr_addr().is_multiple_of(align)
    }

    /// Returns the address of the first data byte (for alignment checking).
    #[inline(always)]
    fn data_ptr_addr(&self) -> usize {
        if self.storage.is_empty() {
            0
        } else {
            (self.storage.as_ptr() as usize) + self.offset
        }
    }

    /// Returns the minimum size that the accessible data spans, or 0 if empty.
    pub(super) fn minimum_allocation_size(&self) -> usize {
        if self.num_rows == 0 {
            0
        } else {
            (self.num_rows - 1) * self.bytes_between_rows + self.bytes_per_row
        }
    }

    #[inline]
    pub(super) fn byte_size(&self) -> (usize, usize) {
        (self.bytes_per_row, self.num_rows)
    }

    /// Returns (bytes_per_row, num_rows, bytes_between_rows).
    #[inline]
    pub(super) fn dimensions(&self) -> (usize, usize, usize) {
        (self.bytes_per_row, self.num_rows, self.bytes_between_rows)
    }

    /// Returns the accessible data as an immutable slice, starting at the first row.
    #[inline]
    pub(super) fn data_slice(&self) -> &[u8] {
        let size = self.minimum_allocation_size();
        if size == 0 {
            &[]
        } else {
            &self.storage[self.offset..self.offset + size]
        }
    }

    /// Returns the accessible data as a mutable slice, starting at the first row.
    #[inline]
    pub(super) fn data_slice_mut(&mut self) -> &mut [u8] {
        let size = self.minimum_allocation_size();
        if size == 0 {
            &mut []
        } else {
            let start = self.offset;
            &mut self.storage[start..start + size]
        }
    }

    #[inline(always)]
    pub(super) fn row(&self, row: usize) -> &[u8] {
        assert!(row < self.num_rows);
        let start = self.offset + row * self.bytes_between_rows;
        &self.storage[start..start + self.bytes_per_row]
    }

    #[inline(always)]
    pub(super) fn row_mut(&mut self, row: usize) -> &mut [u8] {
        assert!(row < self.num_rows);
        let start = self.offset + row * self.bytes_between_rows;
        &mut self.storage[start..start + self.bytes_per_row]
    }

    /// Returns mutable slices for distinct rows. Panics if any rows are equal.
    /// Note: this is quadratic in the number of rows.
    #[inline(always)]
    pub(super) fn distinct_rows_mut<I: DistinctRowsIndexes>(&mut self, rows: I) -> I::Output<'_> {
        rows.get_rows_mut(self)
    }

    /// The number of bytes [`Self::try_allocate`] requests from the allocator
    /// for `byte_size`: rows are padded to a cache line and the buffer carries
    /// one extra cache line of alignment slack. A memory budget must be charged
    /// with this number, not `bytes_per_row * num_rows` — for a 1-pixel-wide
    /// image the padding is a 16-64x multiplier (zenjxl-decoder#55).
    pub(crate) fn allocation_len(byte_size: (usize, usize)) -> Result<usize> {
        let (bytes_per_row, num_rows) = byte_size;
        if bytes_per_row == 0 || num_rows == 0 {
            return Ok(0);
        }
        if bytes_per_row as u64 >= i64::MAX as u64 / 4 || num_rows as u64 >= i64::MAX as u64 / 4 {
            return Err(at!(Error::ImageSizeTooLarge(bytes_per_row, num_rows)));
        }
        let bytes_between_rows =
            bytes_per_row.div_ceil(CACHE_LINE_BYTE_SIZE) * CACHE_LINE_BYTE_SIZE;
        (num_rows - 1)
            .checked_mul(bytes_between_rows)
            .and_then(|v| v.checked_add(bytes_per_row))
            .and_then(|v| v.checked_add(CACHE_LINE_BYTE_SIZE - 1))
            .ok_or_else(|| at!(Error::ImageSizeTooLarge(bytes_per_row, num_rows)))
    }

    /// Returns zeroed memory. The returned buffer is aligned to
    /// CACHE_LINE_BYTE_SIZE bytes via offset.
    pub(super) fn try_allocate(byte_size: (usize, usize), uninit: bool) -> Result<RawImageBuffer> {
        let (bytes_per_row, num_rows) = byte_size;
        if bytes_per_row == 0 || num_rows == 0 {
            return Ok(RawImageBuffer {
                storage: Vec::new(),
                offset: 0,
                bytes_per_row: 0,
                num_rows: 0,
                bytes_between_rows: 0,
            });
        }
        if bytes_per_row as u64 >= i64::MAX as u64 / 4 || num_rows as u64 >= i64::MAX as u64 / 4 {
            return Err(at!(Error::ImageSizeTooLarge(bytes_per_row, num_rows)));
        }
        debug!("trying to allocate image");
        let bytes_between_rows =
            bytes_per_row.div_ceil(CACHE_LINE_BYTE_SIZE) * CACHE_LINE_BYTE_SIZE;
        let data_len = (num_rows - 1)
            .checked_mul(bytes_between_rows)
            .and_then(|v| v.checked_add(bytes_per_row))
            .ok_or(Error::ImageSizeTooLarge(bytes_per_row, num_rows))?;
        assert_ne!(data_len, 0);

        // Allocate with extra space for alignment padding
        let total_len = data_len
            .checked_add(CACHE_LINE_BYTE_SIZE - 1)
            .ok_or(Error::ImageSizeTooLarge(bytes_per_row, num_rows))?;
        let storage = if uninit {
            #[cfg(feature = "allow-unsafe")]
            {
                // Skip zeroing entirely: pages fault on first write.
                let mut v = Vec::new();
                v.try_reserve(total_len)
                    .map_err(|_| Error::ImageOutOfMemory(bytes_per_row, num_rows))?;
                #[allow(unsafe_code)]
                // SAFETY: try_reserve succeeded so capacity >= total_len.
                // Caller guarantees all bytes will be written before being read.
                unsafe {
                    v.set_len(total_len);
                }
                v
            }
            #[cfg(not(feature = "allow-unsafe"))]
            {
                alloc_zeroed_fallible(total_len, bytes_per_row, num_rows)?
            }
        } else {
            alloc_zeroed_fallible(total_len, bytes_per_row, num_rows)?
        };

        // Compute offset to first cache-line-aligned byte
        let base_ptr = storage.as_ptr() as usize;
        let aligned_ptr = base_ptr.div_ceil(CACHE_LINE_BYTE_SIZE) * CACHE_LINE_BYTE_SIZE;
        let offset = aligned_ptr - base_ptr;

        Ok(RawImageBuffer {
            storage,
            offset,
            bytes_per_row,
            num_rows,
            bytes_between_rows,
        })
    }

    /// Returns a copy of the current buffer contents in a new buffer.
    pub(super) fn try_clone(&self) -> Result<Self> {
        let out = RawImageBuffer::try_allocate(self.byte_size(), true)?;
        assert_eq!(self.bytes_per_row, out.bytes_per_row);
        assert_eq!(self.bytes_between_rows, out.bytes_between_rows);
        assert_eq!(self.num_rows, out.num_rows);
        let data_len = self.minimum_allocation_size();
        if data_len != 0 {
            let mut result = out;
            let src = &self.storage[self.offset..self.offset + data_len];
            let dst = &mut result.storage[result.offset..result.offset + data_len];
            dst.copy_from_slice(src);
            Ok(result)
        } else {
            Ok(out)
        }
    }

    /// Clears the buffer, releasing the backing Vec.
    pub(super) fn deallocate(&mut self) {
        self.storage = Vec::new();
        self.offset = 0;
        self.num_rows = 0;
        self.bytes_per_row = 0;
        self.bytes_between_rows = 0;
    }
}

// RawImageBuffer is Send + Sync automatically because Vec<u8> is Send + Sync.

#[allow(private_interfaces)]
pub trait DistinctRowsIndexes {
    type Output<'a>;
    type CastOutput<'a, T: 'static>;

    fn get_rows_mut<'a>(&self, image: &'a mut RawImageBuffer) -> Self::Output<'a>;

    fn cast_rows<'a, T: crate::image::ImageDataType>(
        rows: Self::Output<'a>,
    ) -> Self::CastOutput<'a, T>;
}

#[allow(private_interfaces)]
impl<const S: usize> DistinctRowsIndexes for [usize; S] {
    type Output<'a> = [&'a mut [u8]; S];
    type CastOutput<'a, T: 'static> = [&'a mut [T]; S];

    #[inline(always)]
    fn get_rows_mut<'a>(&self, image: &'a mut RawImageBuffer) -> Self::Output<'a> {
        for i in 0..S {
            assert!(self[i] < image.num_rows);
            for j in i + 1..S {
                assert_ne!(self[i], self[j]);
            }
        }

        // Compute byte ranges for each row
        let ranges: [std::ops::Range<usize>; S] = std::array::from_fn(|i| {
            let start = image.offset + self[i] * image.bytes_between_rows;
            start..start + image.bytes_per_row
        });

        // Use split_at_mut to safely create non-overlapping mutable slices.
        let storage = &mut image.storage[..];
        get_distinct_slices(storage, ranges)
    }

    #[inline(always)]
    fn cast_rows<'a, T: crate::image::ImageDataType>(
        rows: Self::Output<'a>,
    ) -> Self::CastOutput<'a, T> {
        rows.map(|row| crate::image::typed::cast_row_mut(row))
    }
}

/// Safely extract multiple non-overlapping mutable slices from a single slice.
fn get_distinct_slices<const S: usize>(
    data: &mut [u8],
    ranges: [std::ops::Range<usize>; S],
) -> [&mut [u8]; S] {
    // Create index-range pairs sorted by start position
    let mut indexed: [(usize, std::ops::Range<usize>); S] =
        std::array::from_fn(|i| (i, ranges[i].clone()));
    // Sort by range start (simple insertion sort for small S)
    for i in 1..S {
        let mut j = i;
        while j > 0 && indexed[j].1.start < indexed[j - 1].1.start {
            indexed.swap(j, j - 1);
            j -= 1;
        }
    }

    // Verify non-overlapping
    for i in 1..S {
        assert!(
            indexed[i].1.start >= indexed[i - 1].1.end,
            "overlapping row ranges"
        );
    }

    // Peel the slices off the front in ascending-start order via split_at_mut,
    // placing each directly at its original index. `S` is a const generic, so
    // this scratch lives on the stack — no per-call heap allocation, and no
    // sort-back is needed because we index by `orig_idx` as we go.
    let mut out: [Option<&mut [u8]>; S] = std::array::from_fn(|_| None);
    let mut remaining = data;
    let mut consumed = 0usize;
    for item in indexed.iter() {
        let (orig_idx, ref range) = *item;
        let skip = range.start - consumed;
        let len = range.len();
        let (_, rest) = remaining.split_at_mut(skip);
        let (chunk, rest2) = rest.split_at_mut(len);
        out[orig_idx] = Some(chunk);
        remaining = rest2;
        consumed = range.end;
    }

    // Every original index 0..S was assigned exactly once, so no slot is None.
    out.map(|s| s.unwrap())
}

#[cfg(test)]
mod tests {
    use super::get_distinct_slices;

    #[test]
    fn distinct_slices_preserve_input_order_when_ranges_descend() {
        // The hot modular caller passes [y+2, y+1, y] — i.e. ranges in
        // *descending* start order. The returned slices must come back in the
        // original input order, not the internal ascending-start sort order.
        let mut data: Vec<u8> = (0..12).collect();
        let slices = get_distinct_slices(&mut data, [6..9, 3..6, 0..3]);
        assert_eq!(&*slices[0], [6, 7, 8].as_slice());
        assert_eq!(&*slices[1], [3, 4, 5].as_slice());
        assert_eq!(&*slices[2], [0, 1, 2].as_slice());
    }

    #[test]
    fn distinct_slices_are_independently_writable() {
        // Writing through each returned slice must land in the matching range,
        // proving the slices are non-overlapping and correctly mapped.
        let mut data: Vec<u8> = vec![0; 9];
        {
            let slices = get_distinct_slices(&mut data, [0..3, 3..6, 6..9]);
            slices[0].copy_from_slice(&[1, 1, 1]);
            slices[1].copy_from_slice(&[2, 2, 2]);
            slices[2].copy_from_slice(&[3, 3, 3]);
        }
        assert_eq!(data, [1, 1, 1, 2, 2, 2, 3, 3, 3]);
    }

    #[test]
    fn distinct_slices_handle_gaps_and_arbitrary_order() {
        // Ranges separated by padding (as when bytes_between_rows >
        // bytes_per_row), requested in a non-monotonic order.
        let mut data: Vec<u8> = (0..16).collect();
        let slices = get_distinct_slices(&mut data, [6..9, 12..15, 0..3]);
        assert_eq!(&*slices[0], [6, 7, 8].as_slice());
        assert_eq!(&*slices[1], [12, 13, 14].as_slice());
        assert_eq!(&*slices[2], [0, 1, 2].as_slice());
    }

    #[test]
    // The single-element array of ranges is intentional: this exercises the
    // `S == 1` case of the const-generic `[Range; S]` parameter.
    #[allow(clippy::single_range_in_vec_init)]
    fn distinct_slices_single_range() {
        let mut data: Vec<u8> = (0..6).collect();
        let slices = get_distinct_slices(&mut data, [3..6]);
        assert_eq!(&*slices[0], [3, 4, 5].as_slice());
    }
}

#[cfg(test)]
mod allocation_len_tests {
    use super::RawImageBuffer;

    /// `allocation_len` must equal what `try_allocate` really asks for, or the
    /// memory budget drifts from the allocation (#55).
    #[test]
    fn allocation_len_matches_try_allocate() {
        for byte_size in [
            (1, 1),
            (3, 235),
            (64, 1),
            (65, 2),
            (4, 7),
            (200, 3),
            (1, 4096),
        ] {
            let buf = RawImageBuffer::try_allocate(byte_size, false).unwrap();
            assert_eq!(
                buf.storage.len(),
                RawImageBuffer::allocation_len(byte_size).unwrap(),
                "{byte_size:?}"
            );
        }
        assert_eq!(RawImageBuffer::allocation_len((0, 9)).unwrap(), 0);
        assert_eq!(RawImageBuffer::allocation_len((9, 0)).unwrap(), 0);
        // The #55 shape: a 3-byte RGB row padded to a cache line, 235875981 rows
        // (15.1 GB). On a 32-bit target that footprint does not fit `usize`, and
        // the size check must say so rather than wrap.
        let expected: u64 = 235_875_980 * 64 + 64 + 63;
        match RawImageBuffer::allocation_len((64, 235_875_981)) {
            Ok(len) => assert_eq!(len as u64, expected),
            Err(e) => assert!(
                usize::try_from(expected).is_err(),
                "footprint fits usize but allocation_len failed: {e:?}"
            ),
        }
        assert!(RawImageBuffer::allocation_len((usize::MAX / 2, 2)).is_err());
    }
}
