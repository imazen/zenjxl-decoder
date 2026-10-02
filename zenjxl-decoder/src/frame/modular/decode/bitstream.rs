// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::channel::{decode_modular_channel, fused_gradient_ctx};
use super::fused_gradient::{self, FusedSide};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::{ModularChannel, Tree, transforms::apply::meta_apply_local_transforms},
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
    global_tree: &Option<Tree>,
    br: &mut BitReader,
    memory_tracker: &MemoryTracker,
    max_channels: usize,
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
                    meta_apply_local_transforms(buffers, &mut buffer_storage, &h, max_channels)?;
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
        Some(Tree::read(br, size_limit, memory_tracker)?)
    } else {
        None
    };
    let tree = if header.use_global_tree {
        global_tree.as_ref().unwrap()
    } else {
        local_tree.as_ref().unwrap()
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
        decode_modular_channel(&mut buffers, i, stream_id, &header, tree, &mut reader, br)?;
    }

    reader.check_final_state(&tree.histograms, br)?;

    drop(buffers);

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage)?;
    }

    Ok(())
}

/// Decodes the modular sub-bitstreams of two groups (no header given), with
/// the same result as two `decode_modular_subbitstream` calls. Channels that
/// take the fused single-gradient path in both groups are decoded together,
/// alternating between the two independent bitstreams so their serial
/// dependency chains overlap; every other channel is decoded group A first,
/// then group B, exactly as before.
#[allow(clippy::too_many_arguments)]
pub fn decode_modular_subbitstream_pair(
    buffers_a: Vec<&mut ModularChannel>,
    stream_id_a: usize,
    br_a: &mut BitReader,
    buffers_b: Vec<&mut ModularChannel>,
    stream_id_b: usize,
    br_b: &mut BitReader,
    global_tree: &Option<Tree>,
    memory_tracker: &MemoryTracker,
    max_channels: usize,
) -> Result<()> {
    let empty = |b: &[&mut ModularChannel]| {
        b.iter()
            .all(|buffer| matches!(buffer.data.size(), (0, _) | (_, 0)))
    };
    if empty(&buffers_a) || empty(&buffers_b) {
        decode_modular_subbitstream(
            buffers_a,
            stream_id_a,
            None,
            global_tree,
            br_a,
            memory_tracker,
            max_channels,
        )?;
        return decode_modular_subbitstream(
            buffers_b,
            stream_id_b,
            None,
            global_tree,
            br_b,
            memory_tracker,
            max_channels,
        );
    }

    // Header, local transforms and tree for one group, as in
    // `decode_modular_subbitstream`.
    macro_rules! prepare {
        ($buffers:ident, $br:ident, $steps:ident, $storage:ident, $local_tree:ident) => {{
            let h = GroupHeader::read($br)?;
            let bufs = if !h.transforms.is_empty() {
                let new_bufs;
                (new_bufs, $steps) =
                    meta_apply_local_transforms($buffers, &mut $storage, &h, max_channels)?;
                new_bufs
            } else {
                $buffers
            };
            if h.use_global_tree && global_tree.is_none() {
                return Err(at!(Error::NoGlobalTree));
            }
            if !h.use_global_tree {
                let num_local_samples = bufs
                    .iter()
                    .map(|buf| {
                        let (width, height) = buf.channel_info().size;
                        width * height
                    })
                    .sum::<usize>();
                let size_limit = (1024 + num_local_samples).min(1 << 20);
                $local_tree = Some(Tree::read($br, size_limit, memory_tracker)?);
            }
            (h, bufs)
        }};
    }

    let mut steps_a = vec![];
    let mut storage_a = vec![];
    let mut local_tree_a = None;
    let (header_a, mut bufs_a) = prepare!(buffers_a, br_a, steps_a, storage_a, local_tree_a);
    let tree_a = if header_a.use_global_tree {
        global_tree.as_ref().unwrap()
    } else {
        local_tree_a.as_ref().unwrap()
    };
    let width = |b: &[&mut ModularChannel]| {
        b.iter()
            .map(|info| info.channel_info().size.0)
            .max()
            .unwrap_or(0)
    };
    let mut reader_a = SymbolReader::new(&tree_a.histograms, br_a, Some(width(&bufs_a)))?;

    let mut steps_b = vec![];
    let mut storage_b = vec![];
    let mut local_tree_b = None;
    let (header_b, mut bufs_b) = prepare!(buffers_b, br_b, steps_b, storage_b, local_tree_b);
    let tree_b = if header_b.use_global_tree {
        global_tree.as_ref().unwrap()
    } else {
        local_tree_b.as_ref().unwrap()
    };
    let mut reader_b = SymbolReader::new(&tree_b.histograms, br_b, Some(width(&bufs_b)))?;

    let nonempty = |b: &[&mut ModularChannel], i: usize| {
        b.get(i).is_some_and(|c| {
            let (w, h) = c.data.size();
            w != 0 && h != 0
        })
    };
    for i in 0..bufs_a.len().max(bufs_b.len()) {
        let (has_a, has_b) = (nonempty(&bufs_a, i), nonempty(&bufs_b, i));
        if has_a && has_b {
            let ctx_a = fused_gradient_ctx(&bufs_a, i, stream_id_a, &header_a, tree_a, &reader_a)?;
            let ctx_b = fused_gradient_ctx(&bufs_b, i, stream_id_b, &header_b, tree_b, &reader_b)?;
            if let (Some(ctx_a), Some(ctx_b)) = (ctx_a, ctx_b) {
                let lut_a = tree_a.histograms.fused_prefix_lut(ctx_a).unwrap();
                let lut_b = tree_b.histograms.fused_prefix_lut(ctx_b).unwrap();
                let sa = FusedSide::new(lut_a, &mut reader_a, br_a, &tree_a.histograms, ctx_a);
                let sb = FusedSide::new(lut_b, &mut reader_b, br_b, &tree_b.histograms, ctx_b);
                fused_gradient::decode_two(&mut *bufs_a[i], sa, &mut *bufs_b[i], sb);
                br_a.check_for_error().map_err(|e| at!(e))?;
                br_b.check_for_error().map_err(|e| at!(e))?;
                continue;
            }
        }
        if has_a {
            decode_modular_channel(
                &mut bufs_a,
                i,
                stream_id_a,
                &header_a,
                tree_a,
                &mut reader_a,
                br_a,
            )?;
        }
        if has_b {
            decode_modular_channel(
                &mut bufs_b,
                i,
                stream_id_b,
                &header_b,
                tree_b,
                &mut reader_b,
                br_b,
            )?;
        }
    }

    reader_a.check_final_state(&tree_a.histograms, br_a)?;
    reader_b.check_final_state(&tree_b.histograms, br_b)?;

    drop(bufs_a);
    drop(bufs_b);

    for step in steps_a.iter().rev() {
        step.local_apply(&mut storage_a)?;
    }
    for step in steps_b.iter().rev() {
        step.local_apply(&mut storage_b)?;
    }

    Ok(())
}
