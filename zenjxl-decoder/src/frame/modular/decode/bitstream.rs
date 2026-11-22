// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::channel::{decode_modular_channel, try_decode_fused_prefix};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::{
        GlobalTree, ModularChannel, Tree,
        transforms::apply::{TransformLimits, meta_apply_local_transforms},
    },
    headers::{JxlHeader, modular::GroupHeader},
    util::MemoryTracker,
};
use whereat::at;

// This function will decode a header and apply local transforms if a header is not given.
// The intended use of passing a header is for the DcGlobal section.
pub fn decode_modular_subbitstream(
    buffers: Vec<&mut ModularChannel>,
    stream_id: usize,
    header: Option<GroupHeader>,
    global_tree: &Option<GlobalTree>,
    br: &mut BitReader,
    memory_tracker: &MemoryTracker,
    limits: TransformLimits,
) -> Result<()> {
    // Skip decoding if all grids are zero-sized.
    let is_empty = buffers
        .iter()
        .all(|buffer| matches!(buffer.data.size(), (0, _) | (_, 0)));
    if is_empty {
        return Ok(());
    }
    let mut transform_steps = vec![];
    let mut buffer_storage = vec![];

    let buffers = buffers.into_iter().collect::<Vec<_>>();
    let (header, mut buffers) = match header {
        Some(h) => (h, buffers),
        None => {
            let h = GroupHeader::read(br)?;
            if !h.transforms.is_empty() {
                // Note: reassigning to `buffers` here convinces the borrow checker that the borrow of
                // `buffer_storage` ought to outlive `buffers[..]`'s lifetime, which obviously breaks
                // applying transforms later.
                let new_bufs;
                (new_bufs, transform_steps) =
                    meta_apply_local_transforms(buffers, &mut buffer_storage, &h, limits)?;
                (h, new_bufs)
            } else {
                (h, buffers)
            }
        }
    };

    if header.use_global_tree && global_tree.is_none() {
        return Err(at!(Error::NoGlobalTree));
    }
    let local_tree = if !header.use_global_tree {
        let num_local_samples = buffers
            .iter()
            .map(|buf| {
                let (width, height) = buf.channel_info().size;
                width * height
            })
            .sum::<usize>();
        let size_limit = (1024 + num_local_samples).min(1 << 20);
        Some(Tree::read(
            br,
            size_limit,
            limits.max_tree_depth,
            memory_tracker,
        )?)
    } else {
        None
    };
    let (tree, prefix_tree) = if header.use_global_tree {
        let shared = global_tree.as_ref().unwrap();
        (&shared.tree, shared.has_prefix_cache().then_some(shared))
    } else {
        (local_tree.as_ref().unwrap(), None)
    };

    let image_width = buffers
        .iter()
        .map(|info| info.channel_info().size.0)
        .max()
        .unwrap_or(0);
    let mut reader = SymbolReader::new(&tree.histograms, br, Some(image_width))?;

    for i in 0..buffers.len() {
        // Keep channel numbering stable, but skip actually decoding empty channels.
        // This matches libjxl, which continues the loop without renumbering.
        let (w, h) = buffers[i].data.size();
        if w == 0 || h == 0 {
            continue;
        }
        if let Some(shared) = prefix_tree
            && try_decode_fused_prefix(buffers[i], shared, i, stream_id, &mut reader, br)
        {
            br.check_for_error().map_err(|e| at!(e))?;
            continue;
        }
        decode_modular_channel(&mut buffers, i, stream_id, &header, tree, &mut reader, br)?;
    }

    reader.check_final_state(&tree.histograms, br)?;

    drop(buffers);

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage)?;
    }

    Ok(())
}
