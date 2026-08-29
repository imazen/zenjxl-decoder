// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Section buffers must be sized by the input that is actually available,
//! not by the (untrusted) TOC-declared section length.
//!
//! A 34-byte codestream whose single TOC entry claims a 2^30 + 4211712 byte
//! section used to make the decoder `resize()` a ~1 GB zeroed buffer before
//! reading a single section byte (1.08 GB peak RSS for a 34-byte file; on
//! 32-bit targets the infallible `resize` aborts the process). Upstream
//! jxl-rs fixed the same pattern in #856.
//!
//! The check uses a counting global allocator, so this lives in its own
//! integration-test binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates every operation to `System` unchanged; the counters are
// plain atomics and never touch the returned memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        account(layout.size());
        // SAFETY: same layout forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        account(layout.size());
        // SAFETY: same layout forwarded to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: ptr/layout come from a matching alloc above.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        account(new_size);
        // SAFETY: ptr/layout come from a matching alloc above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn account(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
    LARGEST.fetch_max(size, Ordering::Relaxed);
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn truncated_stream_with_huge_toc_entry_does_not_allocate_the_declared_size() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/huge-toc/huge_toc_truncated_34b.jxl");
    let data = std::fs::read(&path).unwrap();
    assert_eq!(data.len(), 34);

    let peak_before = PEAK.load(Ordering::Relaxed);
    LARGEST.store(0, Ordering::Relaxed);
    let result = zenjxl_decoder::decode(&data);
    let largest = LARGEST.load(Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed);

    assert!(result.is_err(), "a truncated stream must not decode");
    // Anything in the MB range means a section buffer was sized from the TOC.
    const LIMIT: usize = 4 << 20;
    assert!(
        largest < LIMIT,
        "largest single allocation was {largest} bytes for a 34-byte input \
         (TOC-declared section size leaked into the buffer size)"
    );
    assert!(
        peak - peak_before.min(peak) < LIMIT,
        "peak live allocation grew by {} bytes for a 34-byte input",
        peak - peak_before.min(peak)
    );
}
