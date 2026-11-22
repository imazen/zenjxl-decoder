// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::channel::{ChannelStream, decode_modular_channel, try_decode_channel_pair};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::{
        ModularChannel, Tree,
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
    global_tree: &Option<Tree>,
    br: &mut BitReader,
    memory_tracker: &MemoryTracker,
    limits: TransformLimits,
) -> Result<()> {
    with_modular_subbitstream(
        buffers,
        header,
        global_tree,
        br,
        memory_tracker,
        limits,
        |buffers, header, tree, reader, br| {
            for i in 0..buffers.len() {
                if channel_nonempty(buffers, i) {
                    decode_modular_channel(buffers, i, stream_id, header, tree, reader, br)?;
                }
            }
            Ok(())
        },
    )
}

fn channel_nonempty(buffers: &[&mut ModularChannel], index: usize) -> bool {
    buffers.get(index).is_some_and(|buffer| {
        let (width, height) = buffer.data.size();
        width != 0 && height != 0
    })
}

fn subbitstream_empty(buffers: &[&mut ModularChannel]) -> bool {
    buffers
        .iter()
        .all(|buffer| matches!(buffer.data.size(), (0, _) | (_, 0)))
}

fn with_modular_subbitstream<'data>(
    buffers: Vec<&mut ModularChannel>,
    header: Option<GroupHeader>,
    global_tree: &Option<Tree>,
    br: &mut BitReader<'data>,
    memory_tracker: &MemoryTracker,
    limits: TransformLimits,
    decode: impl FnOnce(
        &mut [&mut ModularChannel],
        &GroupHeader,
        &Tree,
        &mut SymbolReader,
        &mut BitReader<'data>,
    ) -> Result<()>,
) -> Result<()> {
    // Skip decoding if all grids are zero-sized.
    if subbitstream_empty(&buffers) {
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

    decode(&mut buffers, &header, tree, &mut reader, br)?;

    reader.check_final_state(&tree.histograms, br)?;

    drop(buffers);

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage)?;
    }

    Ok(())
}

pub(crate) struct GroupInput<'a, 'data> {
    pub buffers: Vec<&'a mut ModularChannel>,
    pub id: usize,
    pub bits: &'a mut BitReader<'data>,
}

pub(crate) fn decode_modular_subbitstream_pair(
    a: GroupInput<'_, '_>,
    b: GroupInput<'_, '_>,
    global_tree: &Option<Tree>,
    memory_tracker: &MemoryTracker,
    limits: TransformLimits,
) -> Result<()> {
    if subbitstream_empty(&a.buffers) || subbitstream_empty(&b.buffers) {
        decode_modular_subbitstream(
            a.buffers,
            a.id,
            None,
            global_tree,
            a.bits,
            memory_tracker,
            limits,
        )?;
        return decode_modular_subbitstream(
            b.buffers,
            b.id,
            None,
            global_tree,
            b.bits,
            memory_tracker,
            limits,
        );
    }
    with_modular_subbitstream(
        a.buffers,
        None,
        global_tree,
        a.bits,
        memory_tracker,
        limits,
        |bufs_a, header_a, tree_a, reader_a, bits_a| {
            with_modular_subbitstream(
                b.buffers,
                None,
                global_tree,
                b.bits,
                memory_tracker,
                limits,
                |bufs_b, header_b, tree_b, reader_b, bits_b| {
                    for index in 0..bufs_a.len().max(bufs_b.len()) {
                        let has_a = channel_nonempty(bufs_a, index);
                        let has_b = channel_nonempty(bufs_b, index);
                        if has_a
                            && has_b
                            && try_decode_channel_pair(
                                index,
                                bufs_a[index],
                                ChannelStream {
                                    id: a.id,
                                    header: header_a,
                                    tree: tree_a,
                                    reader: reader_a,
                                    bits: bits_a,
                                },
                                bufs_b[index],
                                ChannelStream {
                                    id: b.id,
                                    header: header_b,
                                    tree: tree_b,
                                    reader: reader_b,
                                    bits: bits_b,
                                },
                            )?
                        {
                            continue;
                        }
                        if has_a {
                            decode_modular_channel(
                                bufs_a, index, a.id, header_a, tree_a, reader_a, bits_a,
                            )?;
                        }
                        if has_b {
                            decode_modular_channel(
                                bufs_b, index, b.id, header_b, tree_b, reader_b, bits_b,
                            )?;
                        }
                    }
                    Ok(())
                },
            )
        },
    )
}
