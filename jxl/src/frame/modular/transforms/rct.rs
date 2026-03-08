// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    frame::modular::{
        ModularChannel,
        transforms::{RctOp, RctPermutation},
    },
    image::Image,
    util::tracing_wrappers::*,
};

#[inline(always)]
fn rct_row<const OP: u32>(r: &mut [i32], g: &mut [i32], b: &mut [i32]) {
    const { assert!(OP <= 6) };
    let len = r.len();
    for i in 0..len {
        let (v0, v1, v2) = (r[i], g[i], b[i]);
        // All arithmetic must be wrapping to match SIMD semantics.
        let (w0, w1, w2) = match OP {
            0 => (v0, v1, v2),
            1 => (v0, v1, v2.wrapping_add(v0)),
            2 => (v0, v1.wrapping_add(v0), v2),
            3 => (v0, v1.wrapping_add(v0), v2.wrapping_add(v0)),
            4 => {
                let avg = v0.wrapping_add(v2) >> 1;
                (v0, v1.wrapping_add(avg), v2)
            }
            5 => {
                let v2_new = v0.wrapping_add(v2);
                let avg = v0.wrapping_add(v2_new) >> 1;
                (v0, v1.wrapping_add(avg), v2_new)
            }
            6 => {
                let (y, co, cg) = (v0, v1, v2);
                let y = y.wrapping_sub(cg >> 1);
                let green = cg.wrapping_add(y);
                let y = y.wrapping_sub(co >> 1);
                let red = y.wrapping_add(co);
                (red, green, y)
            }
            _ => unreachable!(),
        };
        r[i] = w0;
        g[i] = w1;
        b[i] = w2;
    }
}

#[inline(always)]
fn rct_loop_impl<const OP: u32>(r: &mut Image<i32>, g: &mut Image<i32>, b: &mut Image<i32>) {
    let h = r.size().1;
    for y in 0..h {
        rct_row::<OP>(r.row_mut(y), g.row_mut(y), b.row_mut(y));
    }
}

fn rct_loop(r: &mut Image<i32>, g: &mut Image<i32>, b: &mut Image<i32>, op: RctOp) {
    match op {
        RctOp::Noop => {}
        RctOp::AddFirstToThird => rct_loop_impl::<1>(r, g, b),
        RctOp::AddFirstToSecond => rct_loop_impl::<2>(r, g, b),
        RctOp::AddFirstToSecondAndThird => rct_loop_impl::<3>(r, g, b),
        RctOp::AddAvgToSecond => rct_loop_impl::<4>(r, g, b),
        RctOp::AddFirstToThirdAndAvgToSecond => rct_loop_impl::<5>(r, g, b),
        RctOp::YCoCg => rct_loop_impl::<6>(r, g, b),
    }
}

// Applies a RCT in-place to the given buffers.
#[instrument(level = "debug", skip(buffers), ret)]
pub fn do_rct_step(buffers: &mut [&mut ModularChannel], op: RctOp, perm: RctPermutation) {
    let [r, g, b] = buffers else {
        unreachable!("incorrect buffer count for RCT");
    };

    if op != RctOp::Noop {
        rct_loop(&mut r.data, &mut g.data, &mut b.data, op);
    }

    // Note: Gbr and Brg use the *inverse* permutation compared to libjxl, because we *first* write
    // to the buffers and then permute them, while in libjxl the buffers to be written to are
    // permuted first.
    // The same is true for Rbg/Grb/Bgr, but since those are involutions it doesn't change
    // anything.
    match perm {
        RctPermutation::Rgb => {}
        RctPermutation::Gbr => {
            // out[1, 2, 0] = in[0, 1, 2]
            std::mem::swap(&mut g.data, &mut b.data); // [1, 0, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Brg => {
            // out[2, 0, 1] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut b.data); // [1, 0, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Rbg => {
            // out[0, 2, 1] = in[0, 1, 2]
            std::mem::swap(&mut b.data, &mut g.data);
        }
        RctPermutation::Grb => {
            // out[1, 0, 2] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Bgr => {
            // out[2, 1, 0] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut b.data);
        }
    }
}
