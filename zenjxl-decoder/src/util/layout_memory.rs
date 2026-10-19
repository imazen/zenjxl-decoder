// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Reservations for retained layout vectors, including temporary growth.

use super::{MemoryGuard, MemoryTracker};
use crate::error::{Error, Result};
use whereat::at;

pub(crate) struct LayoutMemory {
    tracker: MemoryTracker,
    bytes: u64,
}

impl LayoutMemory {
    pub(crate) fn new(tracker: MemoryTracker) -> Self {
        Self { tracker, bytes: 0 }
    }

    pub(crate) fn reserve<T>(&mut self, vector: &mut Vec<T>, additional: usize) -> Result<()> {
        let needed = vector.len().checked_add(additional).ok_or_else(overflow)?;
        if needed <= vector.capacity() {
            return Ok(());
        }
        let capacity = needed.max(vector.capacity().saturating_mul(2)).max(4);
        let bytes = allocation_bytes::<T>(capacity)?;
        // The old allocation remains live during growth. Charge both until
        // the new vector has been allocated and the elements moved into it.
        self.tracker.try_allocate(bytes)?;
        let guard = MemoryGuard::new(self.tracker.clone(), bytes);
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(capacity)
            .map_err(|e| at!(Error::from(e)))?;
        let actual = allocation_bytes::<T>(replacement.capacity())?;
        let extra = actual.saturating_sub(bytes);
        self.tracker.try_allocate(extra)?;
        let extra_guard = MemoryGuard::new(self.tracker.clone(), extra);
        let old_bytes = allocation_bytes::<T>(vector.capacity())?;
        let total = self
            .bytes
            .checked_add(actual)
            .and_then(|v| v.checked_sub(old_bytes))
            .ok_or_else(overflow)?;
        replacement.append(vector);
        *vector = replacement;
        self.tracker.release(old_bytes);
        self.bytes = total;
        guard.disarm();
        extra_guard.disarm();
        Ok(())
    }

    pub(crate) fn push<T>(&mut self, vector: &mut Vec<T>, value: T) -> Result<()> {
        self.reserve(vector, 1)?;
        vector.push(value);
        Ok(())
    }

    pub(crate) fn string(&mut self, text: &str) -> Result<String> {
        let mut bytes = Vec::new();
        self.reserve(&mut bytes, text.len())?;
        bytes.extend_from_slice(text.as_bytes());
        Ok(String::from_utf8(bytes).expect("copied UTF-8"))
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        self.tracker.release(bytes as u64);
        self.bytes -= bytes as u64;
    }
}

fn allocation_bytes<T>(capacity: usize) -> Result<u64> {
    capacity
        .checked_mul(size_of::<T>())
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(overflow)
}

fn overflow() -> whereat::At<Error> {
    at!(Error::LimitExceeded {
        resource: "layout_bytes",
        actual: u64::MAX,
        limit: isize::MAX as u64
    })
}

impl Drop for LayoutMemory {
    fn drop(&mut self) {
        self.tracker.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_allocations_share_and_release_the_budget() {
        let tracker = MemoryTracker::with_limit(4096);
        let mut vector = Vec::<u64>::new();
        let mut memory = LayoutMemory::new(tracker.clone());
        for value in 0..20 {
            memory.push(&mut vector, value).unwrap();
        }
        let text = memory.string("retained frame name").unwrap();
        let expected = vector.capacity() * size_of::<u64>() + text.capacity();
        assert_eq!(tracker.allocated(), expected as u64);
        assert!(tracker.try_allocate(4096 - expected as u64 + 1).is_err());
        drop(vector);
        drop(text);
        drop(memory);
        assert_eq!(tracker.allocated(), 0);
    }

    #[test]
    fn layout_growth_reserves_old_and_new_allocations() {
        let tracker = MemoryTracker::with_limit(10);
        let mut memory = LayoutMemory::new(tracker.clone());
        let mut vector = Vec::<u8>::new();
        for value in 0..4 {
            memory.push(&mut vector, value).unwrap();
        }
        assert_eq!(tracker.allocated(), 4);
        // Eight replacement bytes plus four still-live old bytes exceed ten.
        assert!(memory.push(&mut vector, 4).is_err());
        assert_eq!(vector, [0, 1, 2, 3]);
        assert_eq!(tracker.allocated(), 4);
        drop(vector);
        drop(memory);
        assert_eq!(tracker.allocated(), 0);
    }
}
