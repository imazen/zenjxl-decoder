// Copyright (c) Imazen LLC.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Bit-span log for header fields that can carry text or skipped bits.
//!
//! The header derive macro records the span of every field named `name` or
//! `extensions` while a log is active (see [`begin`]). It is how the codestream
//! layout API finds frame names, extra-channel names and extension bits
//! without a second parser. Off by default: `record` is one thread-local
//! check.

use std::cell::RefCell;

/// `(field name, first bit, end bit)`, relative to the reader the header was
/// read from.
pub(crate) type Entry = (&'static str, u64, u64);

pub(crate) struct Log {
    entries: Vec<Entry>,
    memory: crate::util::LayoutMemory,
}

impl Log {
    fn new(tracker: crate::util::MemoryTracker) -> Self {
        Self {
            entries: Vec::new(),
            memory: crate::util::LayoutMemory::new(tracker),
        }
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Entry> {
        self.entries.iter()
    }
}

thread_local! {
    static LOG: RefCell<Option<Log>> = const { RefCell::new(None) };
}

/// Restores an enclosing parse's log on success or error.
pub(crate) struct Session(Option<Log>);

impl Drop for Session {
    fn drop(&mut self) {
        LOG.with(|log| *log.borrow_mut() = self.0.take());
    }
}

pub(crate) fn begin(tracker: crate::util::MemoryTracker) -> Session {
    Session(LOG.with(|log| log.borrow_mut().replace(Log::new(tracker))))
}

/// Retains the log's memory reservation while its entries are being consumed.
pub(crate) fn take() -> Log {
    LOG.with(|l| l.borrow_mut().take())
        .unwrap_or_else(|| Log::new(crate::util::MemoryTracker::unlimited()))
}

/// Called by the header derive macro after reading a tracked field.
pub(crate) fn record(
    field: &'static str,
    start: usize,
    end: usize,
) -> crate::error::Result<(), crate::error::Error> {
    LOG.with(|l| {
        if let Some(log) = l.borrow_mut().as_mut() {
            log.memory
                .push(&mut log.entries, (field, start as u64, end as u64))
                .map_err(|error| error.decompose().0)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_or_nested_parse_restores_the_previous_log() {
        let tracker = crate::util::MemoryTracker::with_limit(4096);
        {
            let _outer = begin(tracker.clone());
            record("name", 1, 9).unwrap();
            {
                let _nested = begin(tracker.clone());
                record("extensions", 2, 4).unwrap();
            }
            assert_eq!(take().entries, vec![("name", 1, 9)]);
        }
        record("name", 99, 100).unwrap();
        assert!(take().entries.is_empty());
        assert_eq!(tracker.allocated(), 0);
    }

    #[test]
    fn failed_log_reservation_is_released() {
        let tracker = crate::util::MemoryTracker::with_limit(1);
        {
            let _session = begin(tracker.clone());
            assert!(record("name", 0, 8).is_err());
        }
        assert_eq!(tracker.allocated(), 0);
    }
}
