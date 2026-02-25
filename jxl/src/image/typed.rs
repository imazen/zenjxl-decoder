// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{fmt::Debug, marker::PhantomData};

use crate::{
    error::Result,
    image::internal::DistinctRowsIndexes,
    util::{CACHE_LINE_BYTE_SIZE, MemoryGuard, MemoryTracker, tracing_wrappers::*},
};

use super::{ImageDataType, OwnedRawImage, RawImageRect, RawImageRectMut, Rect};

#[repr(transparent)]
pub struct Image<T: ImageDataType> {
    // Safety invariant: self.raw.data.is_aligned(T::DATA_TYPE_ID.size()) is true.
    raw: OwnedRawImage,
    _ph: PhantomData<T>,
}

impl<T: ImageDataType> Image<T> {
    #[instrument(ret, err)]
    pub fn new_with_padding(
        size: (usize, usize),
        offset: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Image<T>> {
        let s = T::DATA_TYPE_ID.size();
        let img = OwnedRawImage::new_zeroed_with_padding(
            (size.0 * s, size.1),
            (offset.0 * s, offset.1),
            (padding.0 * s, padding.1),
        )?;
        Ok(Self::from_raw(img))
    }

    #[instrument(ret, err)]
    pub fn new(size: (usize, usize)) -> Result<Image<T>> {
        Self::new_with_padding(size, (0, 0), (0, 0))
    }

    /// Allocates an uninitialized image buffer.
    ///
    /// With the `allow-unsafe` feature, the memory is left truly uninitialized
    /// (saving page-fault and zeroing costs). Without it, the buffer is zeroed.
    ///
    /// # Safety contract
    /// The caller MUST write every pixel before reading it.
    pub fn new_uninit(size: (usize, usize)) -> Result<Image<T>> {
        let s = T::DATA_TYPE_ID.size();
        let img = OwnedRawImage::new_uninit((size.0 * s, size.1))?;
        Ok(Self::from_raw(img))
    }

    pub fn new_with_value(size: (usize, usize), value: T) -> Result<Image<T>> {
        // TODO(veluca): skip zero-initializing the allocation if this becomes
        // performance-sensitive.
        let mut ret = Self::new(size)?;
        ret.fill(value);
        Ok(ret)
    }

    /// Computes the allocation size in bytes for an image of the given dimensions.
    /// This accounts for cache-line alignment of rows.
    pub fn allocation_size(size: (usize, usize)) -> u64 {
        let (width, height) = size;
        if width == 0 || height == 0 {
            return 0;
        }
        let bytes_per_row = width.saturating_mul(T::DATA_TYPE_ID.size());
        let bytes_between_rows =
            bytes_per_row.div_ceil(CACHE_LINE_BYTE_SIZE) * CACHE_LINE_BYTE_SIZE;
        let total = (height - 1)
            .saturating_mul(bytes_between_rows)
            .saturating_add(bytes_per_row);
        total as u64
    }

    /// Creates a new image after checking the memory tracker budget.
    /// The image tracks its allocation and releases the budget on drop.
    #[instrument(ret, err)]
    pub fn new_tracked(size: (usize, usize), tracker: &MemoryTracker) -> Result<Image<T>> {
        let alloc_size = Self::allocation_size(size);
        tracker.try_allocate(alloc_size)?;
        // Guard ensures budget is rolled back if the allocation below fails.
        let guard = MemoryGuard::new(tracker.clone(), alloc_size);
        let mut img = Self::new(size)?;
        img.raw.set_tracker(tracker.clone(), alloc_size);
        guard.forget(); // Ownership transferred to OwnedRawImage's Drop.
        Ok(img)
    }

    /// Creates a new image with padding after checking the memory tracker budget.
    /// The image tracks its allocation and releases the budget on drop.
    #[instrument(ret, err)]
    pub fn new_with_padding_tracked(
        size: (usize, usize),
        offset: (usize, usize),
        padding: (usize, usize),
        tracker: &MemoryTracker,
    ) -> Result<Image<T>> {
        let total_width = size.0.saturating_add(offset.0).saturating_add(padding.0);
        let total_height = size.1.saturating_add(offset.1).saturating_add(padding.1);
        let alloc_size = Self::allocation_size((total_width, total_height));
        tracker.try_allocate(alloc_size)?;
        let guard = MemoryGuard::new(tracker.clone(), alloc_size);
        let mut img = Self::new_with_padding(size, offset, padding)?;
        img.raw.set_tracker(tracker.clone(), alloc_size);
        guard.forget();
        Ok(img)
    }

    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (
            self.raw.byte_size().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_size().1,
        )
    }

    pub fn offset(&self) -> (usize, usize) {
        (
            self.raw.byte_offset().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_offset().1,
        )
    }

    pub fn padding(&self) -> (usize, usize) {
        (
            self.raw.byte_padding().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_padding().1,
        )
    }

    pub fn fill(&mut self, v: T) {
        if self.size().0 == 0 {
            return;
        }
        for y in 0..self.size().1 {
            self.row_mut(y).fill(v);
        }
    }

    #[inline]
    pub fn get_rect_including_padding_mut(&mut self, rect: Rect) -> ImageRectMut<'_, T> {
        ImageRectMut::from_raw(
            self.raw
                .get_rect_including_padding_mut(rect.to_byte_rect(T::DATA_TYPE_ID)),
        )
    }

    #[inline]
    pub fn get_rect_including_padding(&mut self, rect: Rect) -> ImageRect<'_, T> {
        ImageRect::from_raw(
            self.raw
                .get_rect_including_padding(rect.to_byte_rect(T::DATA_TYPE_ID)),
        )
    }

    #[inline]
    pub fn get_rect_mut(&mut self, rect: Rect) -> ImageRectMut<'_, T> {
        ImageRectMut::from_raw(self.raw.get_rect_mut(rect.to_byte_rect(T::DATA_TYPE_ID)))
    }

    #[inline]
    pub fn get_rect(&self, rect: Rect) -> ImageRect<'_, T> {
        ImageRect::from_raw(self.raw.get_rect(rect.to_byte_rect(T::DATA_TYPE_ID)))
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self::from_raw(self.raw.try_clone()?))
    }

    pub fn into_raw(self) -> OwnedRawImage {
        self.raw
    }

    pub fn from_raw(raw: OwnedRawImage) -> Self {
        const { assert!(CACHE_LINE_BYTE_SIZE.is_multiple_of(T::DATA_TYPE_ID.size())) };
        assert!(raw.data.is_aligned(T::DATA_TYPE_ID.size()));
        Image {
            raw,
            _ph: PhantomData,
        }
    }

    #[inline(always)]
    pub fn row(&self, row: usize) -> &[T] {
        bytemuck::cast_slice(self.raw.row(row))
    }

    #[inline(always)]
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        bytemuck::cast_slice_mut(self.raw.row_mut(row))
    }

    /// Note: this is quadratic in the number of rows. Indexing *ignores any padding rows*, i.e.
    /// the row at index 0 will be the first row of the *padding*, unlike with all the other row
    /// accessors.
    #[inline(always)]
    pub fn distinct_full_rows_mut<I: DistinctRowsIndexes>(&mut self, rows: I) -> I::CastOutput<'_, T> {
        let rows = self.raw.data.distinct_rows_mut(rows);
        I::cast_rows(rows)
    }

    /// Returns the raw pointer, stride (in elements), row length (in elements),
    /// and number of rows for this image's data.
    ///
    /// This is intended for building split-access wrappers that provide
    /// `&mut [T]` for individual rows without borrowing the entire image.
    /// The caller must ensure aliasing rules are upheld.
    #[cfg(feature = "threads")]
    #[allow(unsafe_code)]
    pub fn row_info_mut(&mut self) -> (*mut T, usize, usize, usize) {
        let (bytes_per_row, num_rows, bytes_between_rows) = self.raw.data.dimensions();
        let row_len = bytes_per_row / std::mem::size_of::<T>();
        let row_stride = bytes_between_rows / std::mem::size_of::<T>();
        let ptr = self.raw.data.data_slice_mut().as_mut_ptr() as *mut T;
        (ptr, row_stride, row_len, num_rows)
    }
}

#[derive(Clone, Copy)]
pub struct ImageRect<'a, T: ImageDataType> {
    // Invariant: self.raw.is_aligned(T::DATA_TYPE_ID.size()) is true.
    raw: RawImageRect<'a>,
    _ph: PhantomData<T>,
}

impl<'a, T: ImageDataType> ImageRect<'a, T> {
    #[inline(always)]
    pub fn rect(&self, rect: Rect) -> ImageRect<'a, T> {
        Self::from_raw(self.raw.rect(rect.to_byte_rect(T::DATA_TYPE_ID)))
    }

    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (
            self.raw.byte_size().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_size().1,
        )
    }

    #[inline(always)]
    pub fn row(&self, row: usize) -> &'a [T] {
        // RawImageRect::row() returns &'a [u8] (the lifetime of the underlying storage),
        // and bytemuck::cast_slice preserves that lifetime.
        bytemuck::cast_slice(self.raw.row(row))
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.size().1).flat_map(|x| self.row(x).iter().cloned())
    }

    pub fn into_raw(self) -> RawImageRect<'a> {
        self.raw
    }

    #[inline]
    pub fn from_raw(raw: RawImageRect<'a>) -> Self {
        const { assert!(CACHE_LINE_BYTE_SIZE.is_multiple_of(T::DATA_TYPE_ID.size())) };
        assert!(raw.is_aligned(T::DATA_TYPE_ID.size()));
        ImageRect {
            raw,
            _ph: PhantomData,
        }
    }
}

pub struct ImageRectMut<'a, T: ImageDataType> {
    // Invariant: self.raw.is_aligned(T::DATA_TYPE_ID.size()) is true.
    raw: RawImageRectMut<'a>,
    _ph: PhantomData<T>,
}

impl<'a, T: ImageDataType> ImageRectMut<'a, T> {
    #[inline]
    pub fn rect(&'a mut self, rect: Rect) -> ImageRectMut<'a, T> {
        Self::from_raw(self.raw.rect_mut(rect.to_byte_rect(T::DATA_TYPE_ID)))
    }

    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (
            self.raw.byte_size().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_size().1,
        )
    }

    #[inline(always)]
    pub fn row(&mut self, row: usize) -> &mut [T] {
        bytemuck::cast_slice_mut(self.raw.row(row))
    }

    pub fn as_rect(&'a self) -> ImageRect<'a, T> {
        ImageRect::from_raw(self.raw.as_rect())
    }

    pub fn into_raw(self) -> RawImageRectMut<'a> {
        self.raw
    }

    #[inline]
    pub fn from_raw(raw: RawImageRectMut<'a>) -> Self {
        const { assert!(CACHE_LINE_BYTE_SIZE.is_multiple_of(T::DATA_TYPE_ID.size())) };
        assert!(raw.is_aligned(T::DATA_TYPE_ID.size()));
        ImageRectMut {
            raw,
            _ph: PhantomData,
        }
    }
}

impl<T: ImageDataType> Debug for Image<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {}x{}",
            T::DATA_TYPE_ID,
            self.size().0,
            self.size().1
        )
    }
}

impl<T: ImageDataType> Debug for ImageRect<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} rect {}x{}",
            T::DATA_TYPE_ID,
            self.size().0,
            self.size().1
        )
    }
}

impl<T: ImageDataType> Debug for ImageRectMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} mutrect {}x{}",
            T::DATA_TYPE_ID,
            self.size().0,
            self.size().1
        )
    }
}

/// Raw-pointer view into an `Image<T>` that can be shared across threads.
///
/// Created via [`Image::shared_view`], which exclusively borrows the image.
/// Provides `unsafe fn get_rect_mut()` to create non-overlapping `ImageRectMut`
/// sub-views for parallel access. The safety contract matches [`SharedOutputView`]
/// and [`ParallelOutputAccess`]: the caller must ensure concurrent rects do not overlap.
///
/// [`SharedOutputView`]: crate::image::output_buffer::SharedOutputView
/// [`ParallelOutputAccess`]: crate::render::buffer_splitter::ParallelOutputAccess
#[cfg(feature = "threads")]
pub(crate) struct SharedImageView<T: ImageDataType> {
    ptr: *mut u8,
    len: usize,
    /// Bytes per row of actual content (bytes_per_row after offset).
    bytes_per_row: usize,
    num_rows: usize,
    bytes_between_rows: usize,
    /// Byte offset from ptr to the first content byte.
    offset: (usize, usize),
    /// Content size in bytes (without padding).
    byte_size: (usize, usize),
    _marker: PhantomData<T>,
}

// SAFETY: SharedImageView's raw pointer is derived from an exclusively-borrowed
// Image. The safety contract requires non-overlapping rect access, same as
// SharedOutputView and DisjointRowAccess.
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
unsafe impl<T: ImageDataType> Send for SharedImageView<T> {}
#[cfg(feature = "threads")]
#[allow(unsafe_code)]
unsafe impl<T: ImageDataType> Sync for SharedImageView<T> {}

#[cfg(feature = "threads")]
#[allow(unsafe_code)]
impl<T: ImageDataType> SharedImageView<T> {
    /// Creates a mutable rect sub-view for parallel access.
    ///
    /// # Safety
    /// The rect must not overlap with any other concurrently-accessed sub-view
    /// from the same `SharedImageView`. Coordinates are in elements (not bytes).
    /// Returns a mutable slice for a single row.
    ///
    /// # Safety
    /// The row must not overlap with any other concurrently-accessed row
    /// from the same `SharedImageView`.
    pub(crate) unsafe fn row_mut(&self, y: usize) -> &mut [T] {
        let row_start = (y + self.offset.1) * self.bytes_between_rows + self.offset.0;
        let row_bytes = self.byte_size.0;
        assert!(row_start + row_bytes <= self.len, "row out of bounds");
        let ptr = unsafe { self.ptr.add(row_start) };
        let count = row_bytes / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) }
    }

    pub(crate) unsafe fn get_rect_mut(&self, rect: Rect) -> ImageRectMut<'_, T> {
        let byte_rect = rect.to_byte_rect(T::DATA_TYPE_ID);
        // Shift by offset (same as OwnedRawImage::shift_rect)
        let shifted = Rect {
            origin: (
                byte_rect.origin.0 + self.offset.0,
                byte_rect.origin.1 + self.offset.1,
            ),
            size: byte_rect.size,
        };
        if shifted.size.0 == 0 || shifted.size.1 == 0 {
            return ImageRectMut::from_raw(super::RawImageRectMut {
                storage: &mut [],
                bytes_per_row: 0,
                num_rows: 0,
                bytes_between_rows: 0,
            });
        }
        assert!(
            shifted.origin.1 + shifted.size.1 <= self.num_rows,
            "rect y out of bounds"
        );
        assert!(
            shifted.origin.0 + shifted.size.0 <= self.bytes_per_row + self.offset.0,
            "rect x out of bounds"
        );
        let new_start = shifted.origin.1 * self.bytes_between_rows + shifted.origin.0;
        let data_span = (shifted.size.1 - 1) * self.bytes_between_rows + shifted.size.0;
        assert!(new_start + data_span <= self.len);
        // SAFETY: Caller guarantees non-overlapping sub-views. The pointer arithmetic
        // stays within the original allocation (verified by the assert above).
        let sub_ptr = unsafe { self.ptr.add(new_start) };
        ImageRectMut::from_raw(super::RawImageRectMut {
            // SAFETY: The sub-region is within the original buffer, and the caller
            // guarantees exclusive access to this region.
            storage: unsafe { std::slice::from_raw_parts_mut(sub_ptr, data_span) },
            bytes_per_row: shifted.size.0,
            num_rows: shifted.size.1,
            bytes_between_rows: self.bytes_between_rows,
        })
    }
}

#[cfg(feature = "threads")]
#[allow(unsafe_code)]
impl<T: ImageDataType> Image<T> {
    /// Creates a shared view for parallel access to non-overlapping rects.
    ///
    /// The returned view holds raw pointers into this image's backing store.
    /// The caller must ensure the image outlives all uses of the view.
    pub(crate) fn shared_view(&mut self) -> SharedImageView<T> {
        let (bpr, nr, bbr) = self.raw.data.dimensions();
        let offset = (self.raw.offset.0 * T::DATA_TYPE_ID.size(), self.raw.offset.1);
        let byte_size = self.raw.byte_size();
        let storage = self.raw.data.data_slice_mut();
        SharedImageView {
            ptr: storage.as_mut_ptr(),
            len: storage.len(),
            bytes_per_row: bpr,
            num_rows: nr,
            bytes_between_rows: bbr,
            offset,
            byte_size,
            _marker: PhantomData,
        }
    }
}
