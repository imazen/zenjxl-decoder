// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;
use whereat::at;

use super::{Predictor, predict::WeightedPredictorState};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::predict::PredictionData,
    image::Image,
    util::{MemoryTracker, NewWithCapacity, tracing_wrappers::*},
};

#[derive(Debug, Clone, Copy)]
pub enum TreeNode {
    Split {
        property: u8,
        val: i32,
        left: u32,
        right: u32,
    },
    Leaf {
        predictor: Predictor,
        offset: i32,
        multiplier: u32,
        id: u32,
    },
}

/// Flattened tree node for optimized traversal (matches C++ FlatDecisionNode).
/// Stores parent + info about both children to evaluate 3 nodes per iteration.
// TODO(hjanuschka): investigate performance of using a Rust enum here, and whether
// separating internal nodes and leaves into two arrays could save a branch.
#[derive(Debug, Clone, Copy)]
pub(super) struct FlatTreeNode {
    property0: i32,                    // Property to test, -1 if leaf
    splitval0_or_predictor: i32,       // Split value, or predictor if leaf
    splitvals_or_multiplier: [i32; 2], // Child splitvals, or multiplier if leaf
    child_id: u32,                     // Index to first grandchild, or context if leaf
    properties_or_offset: [i16; 2],    // Child properties, or offset if leaf
    /// Pre-validated predictor enum for leaf nodes (avoids per-pixel try_from + unwrap).
    /// Only valid when property0 < 0 (leaf node).
    predictor: Predictor,
}

impl FlatTreeNode {
    #[inline]
    fn leaf(predictor: Predictor, offset: i32, multiplier: u32, context: u32) -> Self {
        Self {
            property0: -1,
            splitval0_or_predictor: predictor as i32,
            splitvals_or_multiplier: [multiplier as i32, 0],
            child_id: context,
            properties_or_offset: [offset as i16, 0],
            predictor,
        }
    }

    #[inline(always)]
    fn prediction(&self, data: PredictionData, weighted: i64) -> PredictionResult {
        PredictionResult {
            guess: self.predictor.predict_one(data, weighted) + self.properties_or_offset[0] as i64,
            multiplier: self.splitvals_or_multiplier[0] as u32,
            context: self.child_id,
        }
    }
}

pub(super) enum PrunedTree {
    Flat(Vec<FlatTreeNode>),
    Mask(BitsetTree),
}

impl PrunedTree {
    pub(super) fn new(nodes: &[TreeNode], width: usize) -> Result<Self> {
        if width >= 64
            && let Some(tree) = BitsetTree::build(nodes)?
        {
            return Ok(Self::Mask(tree));
        }
        Ok(Self::Flat(Tree::build_flat_tree(nodes)?))
    }
}

const MIN_MASK_PROPERTY: i32 = -1024;
const MAX_MASK_PROPERTY: i32 = 1023;
const MASK_LOOKUP_SIZE: usize = (MAX_MASK_PROPERTY - MIN_MASK_PROPERTY + 1) as usize;
const MAX_MASK_PROPERTIES: usize = 8;

struct PropertyMask {
    property: usize,
    lookup: [u8; MASK_LOOKUP_SIZE],
    intervals: Vec<u64>,
}

pub(super) struct BitsetTree {
    masks: Vec<PropertyMask>,
    leaves: Vec<FlatTreeNode>,
}

impl BitsetTree {
    // One mask bit per leaf. Values outside the lookup range select the same
    // interval as the clamped value because all splits are inside that range.
    fn build(nodes: &[TreeNode]) -> Result<Option<Self>> {
        if nodes.len() < 5 || nodes.len() > 127 {
            return Ok(None);
        }
        let mut properties = [0u8; MAX_MASK_PROPERTIES];
        let mut num_properties = 0;
        for node in nodes {
            match *node {
                TreeNode::Leaf { offset, .. } => {
                    if i16::try_from(offset).is_err() {
                        return Ok(None);
                    }
                }
                TreeNode::Split { property, val, .. } => {
                    if !(MIN_MASK_PROPERTY..MAX_MASK_PROPERTY).contains(&val) {
                        return Ok(None);
                    }
                    if !properties[..num_properties].contains(&property) {
                        if num_properties == MAX_MASK_PROPERTIES {
                            return Ok(None);
                        }
                        properties[num_properties] = property;
                        num_properties += 1;
                    }
                }
            }
        }
        if num_properties < 2 {
            return Ok(None);
        }

        let reserve = |err: std::collections::TryReserveError| at!(Error::from(err));
        let mut masks = Vec::new_with_capacity(num_properties).map_err(reserve)?;
        let mut splits: [Vec<i32>; MAX_MASK_PROPERTIES] = std::array::from_fn(|_| Vec::new());
        let mut ranges = [(0, 0); MAX_MASK_PROPERTIES];
        for (slot, &property) in properties[..num_properties].iter().enumerate() {
            let cuts = &mut splits[slot];
            cuts.try_reserve_exact(nodes.len() / 2).map_err(reserve)?;
            cuts.extend(nodes.iter().filter_map(|node| match *node {
                TreeNode::Split {
                    property: p, val, ..
                } if p == property => Some(val),
                _ => None,
            }));
            cuts.sort_unstable();
            cuts.dedup();
            let mut lookup = [0; MASK_LOOKUP_SIZE];
            let mut interval = 0;
            for (index, entry) in lookup.iter_mut().enumerate() {
                if cuts
                    .get(interval)
                    .is_some_and(|&cut| cut < MIN_MASK_PROPERTY + index as i32)
                {
                    interval += 1;
                }
                *entry = interval as u8;
            }
            let mut intervals = Vec::new_with_capacity(cuts.len() + 1).map_err(reserve)?;
            intervals.resize(cuts.len() + 1, 0);
            ranges[slot].1 = intervals.len();
            masks.push(PropertyMask {
                property: property as usize,
                lookup,
                intervals,
            });
        }

        let mut leaves = Vec::new_with_capacity(nodes.len().div_ceil(2)).map_err(reserve)?;
        let mut stack = Vec::new_with_capacity(nodes.len()).map_err(reserve)?;
        stack.push((0, ranges));
        while let Some((index, mut ranges)) = stack.pop() {
            match nodes[index as usize] {
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    let slot = properties[..num_properties]
                        .iter()
                        .position(|&p| p == property)
                        .unwrap();
                    let boundary = splits[slot].binary_search(&val).unwrap() + 1;
                    let (start, end) = ranges[slot];
                    if !(start < boundary && boundary < end) {
                        return Ok(None);
                    }
                    ranges[slot] = (boundary, end);
                    stack.push((left, ranges));
                    ranges[slot] = (start, boundary);
                    stack.push((right, ranges));
                }
                TreeNode::Leaf {
                    predictor,
                    multiplier,
                    offset,
                    id,
                } => {
                    let bit = 1u64 << leaves.len();
                    for (mask, &(start, end)) in masks.iter_mut().zip(&ranges) {
                        for interval in &mut mask.intervals[start..end] {
                            *interval |= bit;
                        }
                    }
                    leaves.push(FlatTreeNode::leaf(predictor, offset, multiplier, id));
                }
            }
        }
        Ok(Some(Self { masks, leaves }))
    }

    #[inline(always)]
    fn leaf(&self, properties: &[i32]) -> &FlatTreeNode {
        let mut candidates = u64::MAX;
        for mask in &self.masks {
            let value = properties[mask.property].clamp(MIN_MASK_PROPERTY, MAX_MASK_PROPERTY);
            let index = (value - MIN_MASK_PROPERTY) as usize;
            candidates &= mask.intervals[mask.lookup[index] as usize];
        }
        debug_assert_eq!(candidates.count_ones(), 1);
        &self.leaves[candidates.trailing_zeros() as usize]
    }
}

pub struct Tree {
    pub nodes: Vec<TreeNode>,
    pub histograms: Histograms,
}

fn validate_tree(tree: &[TreeNode], num_properties: usize, height_limit: usize) -> Result<()> {
    if tree.is_empty() {
        return Ok(());
    }

    // This mirrors libjxl's ValidateTree(), but avoids allocating
    // `num_properties * tree.len()` entries.
    //
    // We do an explicit DFS and keep the property ranges only for the current root->node path.
    // When descending into a child we update exactly one property's range (the one we split on)
    // and store the previous range in the child frame; when returning from that child we restore
    // it. This makes memory O(num_properties + height) instead of O(num_properties * tree_size).

    #[derive(Clone, Copy, Debug)]
    enum Stage {
        Enter,
        AfterLeft,
        AfterRight,
    }

    struct Frame {
        node: usize,
        depth: usize,
        stage: Stage,
        restore: Option<(usize, (i32, i32))>,
    }

    let mut property_ranges: Vec<(i32, i32)> = vec![(i32::MIN, i32::MAX); num_properties];
    let mut stack = vec![Frame {
        node: 0,
        depth: 0,
        stage: Stage::Enter,
        restore: None,
    }];

    while let Some(mut frame) = stack.pop() {
        if frame.depth > height_limit {
            return Err(at!(Error::TreeTooTall(frame.depth, height_limit)));
        }

        match (frame.stage, tree[frame.node]) {
            (Stage::Enter, TreeNode::Leaf { .. }) => {
                if let Some((p, old)) = frame.restore {
                    property_ranges[p] = old;
                }
            }
            (
                Stage::Enter,
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right: _,
                },
            ) => {
                let p = property as usize;
                let (l, u) = property_ranges[p];
                if l > val || u <= val {
                    return Err(at!(Error::TreeSplitOnEmptyRange(property, val, l, u)));
                }

                frame.stage = Stage::AfterLeft;
                let depth = frame.depth;
                stack.push(frame);

                // Descend into left child: range becomes (val+1, u).
                let old = property_ranges[p];
                property_ranges[p] = (val + 1, u);
                stack.push(Frame {
                    node: left as usize,
                    depth: depth + 1,
                    stage: Stage::Enter,
                    restore: Some((p, old)),
                });
            }
            (
                Stage::AfterLeft,
                TreeNode::Split {
                    property,
                    val,
                    left: _,
                    right,
                },
            ) => {
                let p = property as usize;
                let (l, u) = property_ranges[p];
                if l > val || u <= val {
                    return Err(at!(Error::TreeSplitOnEmptyRange(property, val, l, u)));
                }

                frame.stage = Stage::AfterRight;
                let depth = frame.depth;
                stack.push(frame);

                // Descend into right child: range becomes (l, val).
                let old = property_ranges[p];
                property_ranges[p] = (l, val);
                stack.push(Frame {
                    node: right as usize,
                    depth: depth + 1,
                    stage: Stage::Enter,
                    restore: Some((p, old)),
                });
            }
            (Stage::AfterRight, TreeNode::Split { .. }) => {
                if let Some((p, old)) = frame.restore {
                    property_ranges[p] = old;
                }
            }
            _ => unreachable!("invalid tree validation state"),
        }
    }

    Ok(())
}

impl Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tree[{:?}]", self.nodes)
    }
}

#[derive(Debug)]
pub struct PredictionResult {
    pub guess: i64,
    pub multiplier: u32,
    pub context: u32,
}

pub const NUM_NONREF_PROPERTIES: usize = 16;
pub const PROPERTIES_PER_PREVCHAN: usize = 4;

const SPLIT_VAL_CONTEXT: usize = 0;
const PROPERTY_CONTEXT: usize = 1;
const PREDICTOR_CONTEXT: usize = 2;
const OFFSET_CONTEXT: usize = 3;
const MULTIPLIER_LOG_CONTEXT: usize = 4;
const MULTIPLIER_BITS_CONTEXT: usize = 5;
const NUM_TREE_CONTEXTS: usize = 6;

// Note: `property_buffer` is passed as input because this implementation relies on having the
// previous values available for computing the local gradient property.
// Also, the first two properties (the static properties) should be already set by the caller.
// All other properties should be 0 on the first call in a row.

/// Computes properties for tree traversal. Shared between flat and non-flat prediction.
/// Returns the weighted predictor prediction value.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compute_properties(
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    used_mask: u32,
) -> i64 {
    let PredictionData {
        left,
        top,
        toptop,
        topleft,
        topright,
        leftleft,
        toprightright: _,
    } = prediction_data;

    // Assert length so the compiler can eliminate bounds checks for constant indices 0..16.
    assert!(property_buffer.len() >= NUM_NONREF_PROPERTIES);

    // Position (properties 2-3)
    if used_mask & 0b1100 != 0 {
        property_buffer[2] = y as i32;
        property_buffer[3] = x as i32;
    }

    // Neighbours (properties 4-7)
    if used_mask & 0b1111_0000 != 0 {
        property_buffer[4] = top.wrapping_abs();
        property_buffer[5] = left.wrapping_abs();
        property_buffer[6] = top;
        property_buffer[7] = left;
    }

    // Local gradient (properties 8-9, always computed together)
    if used_mask & 0b11_0000_0000 != 0 {
        property_buffer[8] = left.wrapping_sub(property_buffer[9]);
        property_buffer[9] = left.wrapping_add(top).wrapping_sub(topleft);
    }

    // FFV1 context properties (10-14)
    if used_mask & 0b0111_1100_0000_0000 != 0 {
        property_buffer[10] = left.wrapping_sub(topleft);
        property_buffer[11] = topleft.wrapping_sub(top);
        property_buffer[12] = top.wrapping_sub(topright);
        property_buffer[13] = top.wrapping_sub(toptop);
        property_buffer[14] = left.wrapping_sub(leftleft);
    }

    // Weighted predictor property (15).
    // We always need wp_pred for the return value when wp_state is present,
    // but only write property_buffer[15] when the tree uses it.
    let (wp_pred, wp_prop) = wp_state
        .map(|wp_state| wp_state.predict_and_property((x, y), xsize, &prediction_data))
        .unwrap_or((0, 0));
    if used_mask & (1 << 15) != 0 {
        property_buffer[15] = wp_prop;
    }

    // Reference properties (16+).
    if used_mask >> 16 != 0 {
        let num_refs = references.size().0;
        if num_refs != 0 {
            let ref_properties = &mut property_buffer[NUM_NONREF_PROPERTIES..];
            ref_properties[..num_refs].copy_from_slice(&references.row(x)[..num_refs]);
        }
    }

    wp_pred
}

/// Prediction using standard tree traversal.
/// Used for small channels where building a flat tree isn't worth it.
#[inline]
#[instrument(level = "trace", ret)]
#[allow(clippy::too_many_arguments)]
pub(super) fn predict(
    tree: &[TreeNode],
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    used_mask: u32,
) -> PredictionResult {
    let wp_pred = compute_properties(
        prediction_data,
        xsize,
        wp_state,
        x,
        y,
        references,
        property_buffer,
        used_mask,
    );

    trace!(?property_buffer, "new properties");

    let mut tree_node = 0;
    while let TreeNode::Split {
        property,
        val,
        left,
        right,
    } = tree[tree_node]
    {
        if property_buffer[property as usize] > val {
            trace!(
                "left at node {tree_node} [{} > {val}]",
                property_buffer[property as usize]
            );
            tree_node = left as usize;
        } else {
            trace!(
                "right at node {tree_node} [{} <= {val}]",
                property_buffer[property as usize]
            );
            tree_node = right as usize;
        }
    }

    trace!(leaf = ?tree[tree_node]);

    let TreeNode::Leaf {
        predictor,
        offset,
        multiplier,
        id,
    } = tree[tree_node]
    else {
        unreachable!();
    };

    let pred = predictor.predict_one(prediction_data, wp_pred);

    PredictionResult {
        guess: pred + offset as i64,
        multiplier,
        context: id,
    }
}

/// Optimized prediction using flat tree (matches C++ context_predict.h:351-371).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_pruned(
    tree: &PrunedTree,
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    used_mask: u32,
) -> PredictionResult {
    let wp_pred = compute_properties(
        prediction_data,
        xsize,
        wp_state,
        x,
        y,
        references,
        property_buffer,
        used_mask,
    );

    let flat_tree = match tree {
        PrunedTree::Mask(tree) => {
            return tree
                .leaf(property_buffer)
                .prediction(prediction_data, wp_pred);
        }
        PrunedTree::Flat(flat) => flat,
    };

    // Flat tree traversal — all child_id + offset values are validated
    // in build_flat_tree, so indexing here cannot go out of bounds.
    let mut pos = 0;
    loop {
        let node = &flat_tree[pos];

        if node.property0 < 0 {
            // Leaf node — predictor was pre-validated at tree build time
            return node.prediction(prediction_data, wp_pred);
        }

        // Split node: C++ logic from context_predict.h:361-365
        let p0 = property_buffer[node.property0 as usize] <= node.splitval0_or_predictor;
        let off0 = if property_buffer[node.properties_or_offset[0] as usize]
            <= node.splitvals_or_multiplier[0]
        {
            1
        } else {
            0
        };
        let off1 = if property_buffer[node.properties_or_offset[1] as usize]
            <= node.splitvals_or_multiplier[1]
        {
            3
        } else {
            2
        };

        pos = (node.child_id + if p0 { off1 } else { off0 }) as usize;
    }
}

impl Tree {
    #[instrument(level = "debug", skip(br, memory_tracker), err)]
    pub fn read(
        br: &mut BitReader,
        size_limit: usize,
        height_limit: usize,
        memory_tracker: &MemoryTracker,
    ) -> Result<Tree> {
        assert!(size_limit <= u32::MAX as usize);
        trace!(pos = br.total_bits_read());
        let tree_histograms = Histograms::decode(NUM_TREE_CONTEXTS, br, true)?;
        let mut tree_reader = SymbolReader::new(&tree_histograms, br, None)?;
        // TODO(veluca): consider early-exiting for trees known to be infinite.
        let mut tree: Vec<TreeNode> = vec![];
        let mut to_decode = 1;
        let mut leaf_id = 0;
        let mut max_property = 0;
        while to_decode > 0 {
            if tree.len() >= size_limit {
                return Err(at!(Error::TreeTooLarge(tree.len() + 1, size_limit)));
            }
            if tree.len() >= tree.capacity() {
                tree.try_reserve(tree.len() * 2 + 1)
                    .map_err(|e| at!(Error::from(e)))?;
            }
            to_decode -= 1;
            let property = tree_reader.read_unsigned(&tree_histograms, br, PROPERTY_CONTEXT);
            trace!(property);
            if let Some(property) = property.checked_sub(1) {
                // inner node.
                if property > 255 {
                    return Err(at!(Error::InvalidProperty(property)));
                }
                max_property = max_property.max(property);
                let splitval = tree_reader.read_signed(&tree_histograms, br, SPLIT_VAL_CONTEXT);
                let left_child = (tree.len() + to_decode + 1) as u32;
                let node = TreeNode::Split {
                    property: property as u8,
                    val: splitval,
                    left: left_child,
                    right: left_child + 1,
                };
                trace!("split node {:?}", node);
                to_decode += 2;
                tree.push(node);
            } else {
                let predictor = Predictor::try_from(tree_reader.read_unsigned(
                    &tree_histograms,
                    br,
                    PREDICTOR_CONTEXT,
                ))?;
                let offset = tree_reader.read_signed(&tree_histograms, br, OFFSET_CONTEXT);
                let mul_log =
                    tree_reader.read_unsigned(&tree_histograms, br, MULTIPLIER_LOG_CONTEXT);
                if mul_log >= 31 {
                    return Err(at!(Error::TreeMultiplierTooLarge(mul_log, 31)));
                }
                let mul_bits =
                    tree_reader.read_unsigned(&tree_histograms, br, MULTIPLIER_BITS_CONTEXT);
                let multiplier = (mul_bits as u64 + 1) << mul_log;
                if multiplier > (u32::MAX as u64) {
                    return Err(at!(Error::TreeMultiplierBitsTooLarge(mul_bits, mul_log)));
                }
                let node = TreeNode::Leaf {
                    predictor,
                    offset,
                    id: leaf_id,
                    multiplier: multiplier as u32,
                };
                leaf_id += 1;
                trace!("leaf node {:?}", node);
                tree.push(node);
            }
        }
        tree_reader.check_final_state(&tree_histograms, br)?;

        let num_properties = max_property as usize + 1;
        validate_tree(&tree, num_properties, height_limit)?;

        // Check memory budget for flat tree allocation
        memory_tracker.check_alloc((tree.len() * std::mem::size_of::<FlatTreeNode>()) as u64)?;

        let histograms = Histograms::decode(tree.len().div_ceil(2), br, true)?;

        Ok(Tree {
            nodes: tree,
            histograms,
        })
    }

    /// Compute a bitmask of which properties the tree actually splits on.
    /// Only those properties need to be computed per pixel in `compute_properties`.
    pub(super) fn compute_used_property_mask(nodes: &[TreeNode]) -> u32 {
        let mut mask = 0u32;
        for node in nodes {
            if let TreeNode::Split { property, .. } = node {
                // Properties >= 32 can't be tracked in a u32 mask; set all high
                // bits so they are always computed in compute_properties.
                mask |= 1u32.checked_shl(*property as u32).unwrap_or(u32::MAX);
            }
        }
        // Properties 8 and 9 are coupled: property 8 reads the previous value
        // of property 9, and property 9 is written alongside 8. If either is
        // used by the tree, both must be computed.
        if mask & ((1 << 8) | (1 << 9)) != 0 {
            mask |= (1 << 8) | (1 << 9);
        }
        mask
    }

    /// Build flat tree using BFS traversal (matches C++ encoding.cc:81-144).
    /// Each flat node stores parent + both children info to reduce branches.
    pub(super) fn build_flat_tree(nodes: &[TreeNode]) -> Result<Vec<FlatTreeNode>> {
        use std::collections::VecDeque;

        if nodes.is_empty() {
            return Ok(vec![]);
        }

        let mut flat_nodes =
            Vec::new_with_capacity(nodes.len()).map_err(|e| at!(Error::from(e)))?;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(0); // Start with root

        while let Some(cur_idx) = queue.pop_front() {
            match &nodes[cur_idx] {
                TreeNode::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    id,
                } => {
                    flat_nodes.push(FlatTreeNode::leaf(*predictor, *offset, *multiplier, *id));
                }
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    // childID points to first of 4 grandchildren in output
                    let child_id = (flat_nodes.len() + queue.len() + 1) as u32;

                    let mut flat = FlatTreeNode {
                        property0: *property as i32,
                        splitval0_or_predictor: *val,
                        splitvals_or_multiplier: [0, 0],
                        child_id,
                        properties_or_offset: [0, 0],
                        predictor: Predictor::Zero, // unused for split nodes
                    };

                    // Process left (i=0) and right (i=1) children
                    for (i, &child_idx) in [*left as usize, *right as usize].iter().enumerate() {
                        match &nodes[child_idx] {
                            TreeNode::Leaf { .. } => {
                                // Child is leaf: set property=0 and enqueue leaf twice
                                flat.properties_or_offset[i] = 0;
                                flat.splitvals_or_multiplier[i] = 0;
                                queue.push_back(child_idx);
                                queue.push_back(child_idx);
                            }
                            TreeNode::Split {
                                property: cp,
                                val: cv,
                                left: cl,
                                right: cr,
                            } => {
                                // Child is split: store property/splitval and enqueue grandchildren
                                flat.properties_or_offset[i] = *cp as i16;
                                flat.splitvals_or_multiplier[i] = *cv;
                                queue.push_back(*cl as usize);
                                queue.push_back(*cr as usize);
                            }
                        }
                    }

                    flat_nodes.push(flat);
                }
            }
        }

        // Validate that all child_id references are within bounds.
        // Split nodes can index up to child_id + 3 (the maximum offset from off1).
        let len = flat_nodes.len();
        for node in &flat_nodes {
            if node.property0 >= 0 {
                // Split node: child_id + 3 must be a valid index
                let max_child = node.child_id as usize + 3;
                if max_child >= len {
                    return Err(at!(Error::TreeTooLarge(max_child + 1, len)));
                }
            }
        }

        Ok(flat_nodes)
    }

    pub fn max_property_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|x| match x {
                TreeNode::Leaf { .. } => 0,
                TreeNode::Split { property, .. } => *property,
            })
            .max()
            .unwrap_or_default() as usize
            + 1
    }

    #[allow(dead_code)] // Part of Tree analysis API
    pub fn num_prev_channels(&self) -> usize {
        self.max_property_count()
            .saturating_sub(NUM_NONREF_PROPERTIES)
            .div_ceil(PROPERTIES_PER_PREVCHAN)
    }
}

#[cfg(test)]
mod level_tests {
    use super::*;
    use crate::api::JxlCodestreamLevel;

    fn encoded_comb(depth: usize) -> Vec<u8> {
        // Three prefix tables: implicit zero for leaf fields, property 0/1,
        // and a fixed split-value token followed by twelve bits.
        let mut bits = vec![
            (0, 1),
            (1, 1),
            (2, 2), // no LZ77, simple context map
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (1, 1),
            (1, 4),
            (0, 1),
            (0, 1), // prefix, HybridUint 100
            (1, 4),
            (0, 1),
            (0, 1),
            (1, 4),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 4),
            (1, 1),
            (3, 4),
            (5, 3), // alphabets 1, 2, 14
            (1, 2),
            (1, 2),
            (0, 1),
            (1, 1), // two property symbols
            (1, 2),
            (0, 2),
            (13, 4), // one split-value token
        ];
        for i in 0..depth {
            bits.extend([(1, 1), (2 * i as u64, 12)]);
            if i != 0 {
                bits.push((0, 1)); // right sibling at this depth
            }
        }
        bits.extend([(0, 1), (0, 1)]); // children of the deepest split
        bits.extend([(0, 1), (1, 1), (0, 2), (1, 1), (0, 4), (0, 1)]);
        crate::util::test::pack_bits(&bits)
    }

    #[test]
    fn ma_tree_encoded_node_and_depth_limits() {
        let bytes = encoded_comb(65);
        let tracker = MemoryTracker::unlimited();
        let read = |nodes, depth| Tree::read(&mut BitReader::new(&bytes), nodes, depth, &tracker);
        assert_eq!(read(131, 2048).unwrap().nodes.len(), 131);
        assert!(matches!(
            read(130, 2048).err().unwrap().error(),
            Error::TreeTooLarge(131, 130)
        ));
        assert!(matches!(
            read(131, 64).err().unwrap().error(),
            Error::TreeTooTall(65, 64)
        ));
    }

    fn comb(depth: usize) -> Vec<TreeNode> {
        let leaf = TreeNode::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
            id: 0,
        };
        let mut nodes = vec![leaf; 2 * depth + 1];
        for i in 0..depth {
            nodes[2 * i] = TreeNode::Split {
                property: 0,
                val: i as i32,
                left: (2 * i + 2) as u32,
                right: (2 * i + 1) as u32,
            };
        }
        nodes
    }

    #[test]
    fn ma_tree_depth_selected_level_boundaries() {
        for level in [JxlCodestreamLevel::Level5, JxlCodestreamLevel::Level10] {
            let bound = level.max_ma_tree_depth();
            validate_tree(&comb(bound), 1, bound).unwrap();
            let error = validate_tree(&comb(bound + 1), 1, bound).unwrap_err();
            assert!(matches!(error.error(), Error::TreeTooTall(actual, limit)
                if *actual == bound + 1 && *limit == bound));
        }
        validate_tree(
            &comb(65),
            1,
            JxlCodestreamLevel::Level10.max_ma_tree_depth(),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod bitset_tests {
    use super::*;

    #[test]
    fn property_masks_match_branch_boundaries_and_repeated_splits() {
        let leaf = |id| TreeNode::Leaf {
            predictor: Predictor::Gradient,
            offset: id as i32 - 3,
            multiplier: id + 1,
            id,
        };
        let nodes = [
            TreeNode::Split {
                property: 9,
                val: 17,
                left: 1,
                right: 2,
            },
            TreeNode::Split {
                property: 2,
                val: -11,
                left: 3,
                right: 4,
            },
            TreeNode::Split {
                property: 9,
                val: -40,
                left: 5,
                right: 6,
            },
            leaf(0),
            leaf(1),
            leaf(2),
            leaf(3),
        ];
        let tree = BitsetTree::build(&nodes).unwrap().unwrap();
        let boundaries = [
            i32::MIN,
            -1024,
            -41,
            -40,
            -39,
            -12,
            -11,
            -10,
            16,
            17,
            18,
            1023,
            i32::MAX,
        ];
        let mut properties = [0; 10];
        for y in boundaries {
            for g in boundaries {
                properties[2] = y;
                properties[9] = g;
                let mut index = 0;
                while let TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } = nodes[index]
                {
                    index = if properties[property as usize] > val {
                        left
                    } else {
                        right
                    } as usize;
                }
                let TreeNode::Leaf {
                    id,
                    multiplier,
                    offset,
                    ..
                } = nodes[index]
                else {
                    unreachable!()
                };
                let result = tree.leaf(&properties);
                assert_eq!(result.child_id, id);
                assert_eq!(result.splitvals_or_multiplier[0] as u32, multiplier);
                assert_eq!(result.properties_or_offset[0] as i32, offset);
            }
        }
    }
}
