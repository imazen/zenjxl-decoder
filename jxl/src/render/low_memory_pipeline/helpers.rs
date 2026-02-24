// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::low_memory_pipeline::render_group::ChannelVec;

/// Returns a vector of &mut vals[idx[i].0][idx[i].1], in order of idx[i].2.
/// Panics if any of the indices are out of bounds or
/// (idx[i].0, idx[i].1) == (idx[j].0, idx[j].1) for i != j or indices are not
/// sorted lexicographically.
#[inline]
#[allow(unsafe_code)]
pub(super) fn get_distinct_indices<'a, T>(
    vals: &'a mut [impl AsMut<[T]>],
    idx: &[(usize, usize, usize)],
) -> ChannelVec<&'a mut T> {
    let mut answer_buffer: ChannelVec<Option<&'a mut T>> = ChannelVec::new();
    for _ in 0..idx.len() {
        answer_buffer.push(None);
    }

    let mut prev_outer = usize::MAX;
    let mut prev_inner = usize::MAX;
    for &(outer, inner, pos) in idx {
        // Verify sorted and distinct (same check as before, just explicit)
        debug_assert!(
            prev_outer == usize::MAX || outer > prev_outer || (outer == prev_outer && inner > prev_inner),
            "indices must be sorted and distinct"
        );
        prev_outer = outer;
        prev_inner = inner;

        let buf = vals[outer].as_mut();
        assert!(inner < buf.len(), "inner index out of bounds");
        // SAFETY: indices are guaranteed distinct (no two (outer, inner) pairs are equal),
        // so each &mut T points to a different element. We use raw pointers to avoid
        // the borrow checker's inability to prove non-aliasing across loop iterations.
        let ptr = unsafe { buf.as_mut_ptr().add(inner) };
        answer_buffer[pos] = Some(unsafe { &mut *ptr });
    }

    answer_buffer
        .into_iter()
        .map(|x| x.expect("Not all elements were found"))
        .collect()
}
