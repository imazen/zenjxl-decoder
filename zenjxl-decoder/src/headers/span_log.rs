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

thread_local! {
    static LOG: RefCell<Option<Vec<Entry>>> = const { RefCell::new(None) };
}

/// Starts (or restarts) logging, discarding earlier entries.
pub(crate) struct Session(Option<Vec<Entry>>);

impl Drop for Session {
    fn drop(&mut self) {
        LOG.with(|log| *log.borrow_mut() = self.0.take());
    }
}

pub(crate) fn begin() -> Session {
    Session(LOG.with(|log| log.borrow_mut().replace(Vec::new())))
}

/// Stops logging and returns what was recorded since [`begin`].
pub(crate) fn take() -> Vec<Entry> {
    LOG.with(|l| l.borrow_mut().take().unwrap_or_default())
}

/// Called by the header derive macro after reading a tracked field.
pub(crate) fn record(field: &'static str, start: usize, end: usize) {
    LOG.with(|l| {
        if let Some(v) = l.borrow_mut().as_mut() {
            v.push((field, start as u64, end as u64));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_or_nested_parse_restores_the_previous_log() {
        {
            let _outer = begin();
            record("name", 1, 9);
            {
                let _nested = begin();
                record("extensions", 2, 4);
            }
            assert_eq!(take(), vec![("name", 1, 9)]);
        }
        record("name", 99, 100);
        assert!(take().is_empty());
    }
}
