// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::util::SmallVec;

/// Multi-row channel accessor for immutable access.
///
/// Provides 2D indexing where `channels[ch]` returns `&[&[T]]` (all rows for a channel),
/// and `channels[ch][row]` returns `&[T]` (pixels for a specific row).
///
/// This eliminates nested Vec collections while maintaining the same indexing syntax.
pub struct Channels<'a, T> {
    // The number of input rows should be maximized by the EPF0 stage, which has 21.
    // A plain array and a length: a `SmallVec` checked its capacity and
    // inline/heap state on every push, several times per row per stage.
    buf: [&'a [T]; MAX_INPUT_ROWS],
    num_rows: usize,
    num_channels: usize,
    pub(crate) rows_per_channel: usize,
}

/// Rows a [`Channels`] holds at most.
const MAX_INPUT_ROWS: usize = 32;
/// Rows a [`ChannelsMut`] holds at most.
const MAX_OUTPUT_ROWS: usize = 8;

impl<'a, T> Channels<'a, T> {
    /// Create a new Channels accessor.
    ///
    /// # Arguments
    /// * `row_data` - Flat vector of all rows for all channels (length = num_channels * rows_per_channel)
    /// * `num_channels` - Number of channels
    /// * `rows_per_channel` - Number of rows per channel (typically 2*BORDER+1)
    pub fn new(
        row_data: SmallVec<[&'a [T]; 32]>,
        num_channels: usize,
        rows_per_channel: usize,
    ) -> Self {
        debug_assert_eq!(
            row_data.len(),
            num_channels * rows_per_channel,
            "row_data length must equal num_channels * rows_per_channel"
        );
        let mut this = Self::empty(num_channels, rows_per_channel);
        for row in row_data {
            this.push(row);
        }
        this
    }

    /// An accessor with no rows yet, to [`Self::push`] into.
    #[inline(always)]
    pub fn empty(num_channels: usize, rows_per_channel: usize) -> Self {
        Self {
            buf: [&[]; MAX_INPUT_ROWS],
            num_rows: 0,
            num_channels,
            rows_per_channel,
        }
    }

    /// Appends a row; rows are channel-major.
    #[inline(always)]
    pub fn push(&mut self, row: &'a [T]) {
        self.buf[self.num_rows] = row;
        self.num_rows += 1;
    }

    /// All rows, channel-major.
    #[inline(always)]
    pub fn rows(&self) -> &[&'a [T]] {
        &self.buf[..self.num_rows]
    }

    /// Returns the number of channels.
    pub fn len(&self) -> usize {
        self.num_channels
    }

    /// Returns true if there are no channels.
    #[allow(dead_code)] // Companion to len()
    pub fn is_empty(&self) -> bool {
        self.num_channels == 0
    }

    /// Returns an iterator over channel slices.
    #[allow(dead_code)] // Part of Channels accessor API
    pub fn iter(&self) -> impl Iterator<Item = &[&'a [T]]> {
        (0..self.num_channels).map(move |ch| &self[ch])
    }
}

/// Implement indexing: channels[ch] returns &[&[T]]
impl<'a, T> std::ops::Index<usize> for Channels<'a, T> {
    type Output = [&'a [T]];

    fn index(&self, ch: usize) -> &[&'a [T]] {
        let start = ch * self.rows_per_channel;
        &self.rows()[start..start + self.rows_per_channel]
    }
}

/// Multi-row channel accessor for mutable access.
///
/// Provides 2D indexing where `channels[ch]` returns `&[&mut [T]]` or `&mut [&mut [T]]`,
/// and `channels[ch][row]` returns `&mut [T]` (pixels for a specific row).
pub struct ChannelsMut<'a, T> {
    // The number of output rows should be maximized by the Upsample8 stage, which has 8.
    buf: [&'a mut [T]; MAX_OUTPUT_ROWS],
    num_rows: usize,
    num_channels: usize,
    pub(crate) rows_per_channel: usize,
}

impl<'a, T> ChannelsMut<'a, T> {
    /// Create a new ChannelsMut accessor.
    ///
    /// # Arguments
    /// * `row_data` - Flat vector of all mutable rows for all channels
    /// * `num_channels` - Number of channels
    /// * `rows_per_channel` - Number of rows per channel (typically 1 << SHIFT)
    pub fn new(
        row_data: SmallVec<[&'a mut [T]; 8]>,
        num_channels: usize,
        rows_per_channel: usize,
    ) -> Self {
        debug_assert_eq!(
            row_data.len(),
            num_channels * rows_per_channel,
            "row_data length must equal num_channels * rows_per_channel"
        );
        let mut this = Self::empty(num_channels, rows_per_channel);
        for row in row_data {
            this.push(row);
        }
        this
    }

    /// See [`Channels::empty`].
    #[inline(always)]
    pub fn empty(num_channels: usize, rows_per_channel: usize) -> Self {
        Self {
            buf: std::array::from_fn(|_| Default::default()),
            num_rows: 0,
            num_channels,
            rows_per_channel,
        }
    }

    /// Appends a row; rows are channel-major.
    #[inline(always)]
    pub fn push(&mut self, row: &'a mut [T]) {
        self.buf[self.num_rows] = row;
        self.num_rows += 1;
    }

    /// All rows, channel-major.
    #[inline(always)]
    pub fn rows_mut(&mut self) -> &mut [&'a mut [T]] {
        &mut self.buf[..self.num_rows]
    }

    /// Returns the number of channels.
    pub fn len(&self) -> usize {
        self.num_channels
    }

    /// Returns true if there are no channels.
    #[allow(dead_code)] // Companion to len()
    pub fn is_empty(&self) -> bool {
        self.num_channels == 0
    }

    /// Splits the first 3 channels into separate mutable slices.
    /// Returns a tuple containing mutable references to each channel's rows.
    #[allow(clippy::type_complexity)]
    pub fn split_first_3_mut(
        &mut self,
    ) -> (&mut [&'a mut [T]], &mut [&'a mut [T]], &mut [&'a mut [T]]) {
        assert!(
            3 <= self.num_channels,
            "requested 3 channels but only have {}",
            self.num_channels
        );
        let rpc = self.rows_per_channel;
        let (first, rest) = self.rows_mut().split_at_mut(rpc);
        let (second, rest) = rest.split_at_mut(rpc);
        let (third, _) = rest.split_at_mut(rpc);
        (first, second, third)
    }

    /// Returns a mutable iterator over all channels.
    /// Each item is a mutable slice of rows for that channel.
    #[allow(dead_code)] // Part of ChannelsMut accessor API
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [&'a mut [T]]> {
        let rpc = self.rows_per_channel;
        self.rows_mut().chunks_mut(rpc)
    }
}

/// Implement immutable indexing: channels[ch] returns &[&mut [T]]
impl<'a, T> std::ops::Index<usize> for ChannelsMut<'a, T> {
    type Output = [&'a mut [T]];

    fn index(&self, ch: usize) -> &[&'a mut [T]] {
        let start = ch * self.rows_per_channel;
        &self.buf[..self.num_rows][start..start + self.rows_per_channel]
    }
}

/// Implement mutable indexing: &mut channels[ch] returns &mut [&mut [T]]
impl<'a, T> std::ops::IndexMut<usize> for ChannelsMut<'a, T> {
    fn index_mut(&mut self, ch: usize) -> &mut [&'a mut [T]] {
        let start = ch * self.rows_per_channel;
        let end = start + self.rows_per_channel;
        &mut self.rows_mut()[start..end]
    }
}
