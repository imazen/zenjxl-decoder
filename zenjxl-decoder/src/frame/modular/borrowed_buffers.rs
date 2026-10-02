// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::DerefMut;

use crate::{
    error::Result,
    frame::modular::{IMAGE_OFFSET, IMAGE_PADDING},
    image::Image,
    util::AtomicRefMut,
};

use super::{ModularBufferInfo, ModularChannel};

/// Buffers `with_buffers` took from a recycle pool instead of allocating.
#[cfg(test)]
pub(crate) static RECYCLED_REUSES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub fn with_buffers<T>(
    buffers: &[ModularBufferInfo],
    indices: &[usize],
    grid: usize,
    f: impl FnOnce(Vec<&mut ModularChannel>) -> Result<T>,
) -> Result<T> {
    let mut bufs = vec![];
    for i in indices {
        // Allocate buffers if they are not present.
        let buf = &buffers[*i];
        let b = &buf.buffer_grid[grid];
        let mut data = b.data.borrow_mut();
        if data.is_none() {
            // A recycled buffer of the same geometry, zeroed, is identical to
            // a fresh one and skips the allocation and its page faults.
            let recycled = buf.pool.as_ref().and_then(|pool| {
                pool.take(|raw| Image::<i32>::fits_padded(raw, b.size, IMAGE_OFFSET, IMAGE_PADDING))
            });
            let image = match recycled {
                Some(mut raw) => {
                    #[cfg(test)]
                    RECYCLED_REUSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    raw.zero_fill();
                    Image::from_raw(raw)
                }
                None => Image::new_with_padding(b.size, IMAGE_OFFSET, IMAGE_PADDING)?,
            };
            *data = Some(ModularChannel {
                data: image,
                auxiliary_data: None,
                shift: buf.info.shift,
                bit_depth: buf.info.bit_depth,
            });
        }

        // Skip zero-sized *tiles*.
        //
        // Note that some bitstreams can contain channels with one dimension being 0 (e.g. palette
        // meta-channel with 0 colors has size (0, 3)). Those must still participate in channel
        // numbering (but carry no entropy-coded pixels), so we only skip when both dimensions are 0.
        if b.size.0 == 0 && b.size.1 == 0 {
            continue;
        }

        bufs.push(AtomicRefMut::map(data, |x| x.as_mut().unwrap()));
    }
    f(bufs.iter_mut().map(|x| x.deref_mut()).collect())
}

#[cfg(test)]
mod tests {
    use super::RECYCLED_REUSES;
    use std::sync::atomic::Ordering;

    /// A sequential decode of a 1806-group modular frame renders each
    /// two-group chunk as it goes and must draw its group buffers from the
    /// recycle pool rather than allocating one set per group. Output
    /// equality is covered by `tests/modular_buffer_recycling.rs`.
    #[test]
    fn sequential_decode_reuses_group_buffers() {
        let data = include_bytes!("../../../tests/testdata/jxlrs-865/issue865_large_toc.jxl");
        let options = crate::api::JxlDecoderOptions {
            parallel: false,
            ..Default::default()
        };
        let before = RECYCLED_REUSES.load(Ordering::Relaxed);
        crate::api::decode_with(data, options).expect("decode");
        let reused = RECYCLED_REUSES.load(Ordering::Relaxed) - before;
        assert!(reused >= 1000, "only {reused} group buffers were reused");
    }
}
