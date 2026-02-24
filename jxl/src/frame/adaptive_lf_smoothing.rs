// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::image::Image;
use num_traits::abs;

#[allow(clippy::excessive_precision)]
const W_SIDE: f32 = 0.20345139757231578;
#[allow(clippy::excessive_precision)]
const W_CORNER: f32 = 0.0334829185968739;
const W_CENTER: f32 = 1.0 - 4.0 * (W_SIDE + W_CORNER);

fn compute_pixel_channel(
    dc_factor: f32,
    gap: f32,
    x: usize,
    row_top: &[f32],
    row: &[f32],
    row_bottom: &[f32],
) -> (f32, f32, f32) {
    let tl = row_top[x - 1];
    let tc = row_top[x];
    let tr = row_top[x + 1];
    let ml = row[x - 1];
    let mc = row[x];
    let mr = row[x + 1];
    let bl = row_bottom[x - 1];
    let bc = row_bottom[x];
    let br = row_bottom[x + 1];
    let corner = tl + tr + bl + br;
    let side = ml + mr + tc + bc;
    let sm = corner * W_CORNER + side * W_SIDE + mc * W_CENTER;
    (mc, sm, gap.max(abs((mc - sm) / dc_factor)))
}

pub fn adaptive_lf_smoothing(
    lf_factors: [f32; 3],
    lf_image: &mut [Image<f32>; 3],
    #[cfg(feature = "threads")] parallel: bool,
) -> Result<()> {
    let xsize = lf_image[0].size().0;
    let ysize = lf_image[0].size().1;
    if ysize <= 2 || xsize <= 2 {
        return Ok(());
    }
    let mut smoothed: [Image<f32>; 3] = [
        Image::<f32>::new((xsize, ysize))?,
        Image::<f32>::new((xsize, ysize))?,
        Image::<f32>::new((xsize, ysize))?,
    ];
    // Copy border rows (row 0 and ysize-1) and border columns (x=0, x=xsize-1).
    for c in 0..3 {
        for y in [0, ysize - 1] {
            smoothed[c].row_mut(y).copy_from_slice(lf_image[c].row(y));
        }
    }

    #[cfg(feature = "threads")]
    if parallel && ysize > 4 {
        use crate::image::SharedImageView;
        use rayon::prelude::*;

        // Create shared views for parallel mutable access to disjoint rows.
        let smoothed_views: [SharedImageView<f32>; 3] = [
            smoothed[0].shared_view(),
            smoothed[1].shared_view(),
            smoothed[2].shared_view(),
        ];

        // Each row y in 1..ysize-1 reads from lf_image rows y-1, y, y+1
        // and writes to smoothed row y. Rows are independent.
        (1..ysize - 1).into_par_iter().try_for_each(|y| -> Result<()> {
            let rows_in: [(&[f32], &[f32], &[f32]); 3] = core::array::from_fn(|c| {
                (lf_image[c].row(y - 1), lf_image[c].row(y), lf_image[c].row(y + 1))
            });

            // SAFETY: each iteration writes to a distinct row y of smoothed.
            let rows_out: [&mut [f32]; 3] = core::array::from_fn(|c| {
                #[allow(unsafe_code)]
                unsafe {
                    smoothed_views[c].row_mut(y)
                }
            });

            // Copy border columns.
            for c in 0..3 {
                rows_out[c][0] = rows_in[c].1[0];
                rows_out[c][xsize - 1] = rows_in[c].1[xsize - 1];
            }

            for x in 1..xsize - 1 {
                let gap = 0.5;
                let (mc_x, sm_x, gap) =
                    compute_pixel_channel(lf_factors[0], gap, x, rows_in[0].0, rows_in[0].1, rows_in[0].2);
                let (mc_y, sm_y, gap) =
                    compute_pixel_channel(lf_factors[1], gap, x, rows_in[1].0, rows_in[1].1, rows_in[1].2);
                let (mc_b, sm_b, gap) =
                    compute_pixel_channel(lf_factors[2], gap, x, rows_in[2].0, rows_in[2].1, rows_in[2].2);
                let factor = (3.0 - 4.0 * gap).max(0.0);
                rows_out[0][x] = (sm_x - mc_x) * factor + mc_x;
                rows_out[1][x] = (sm_y - mc_y) * factor + mc_y;
                rows_out[2][x] = (sm_b - mc_b) * factor + mc_b;
            }
            Ok(())
        })?;

        // Suppress unused warnings — views are consumed.
        drop(smoothed_views);
    } else {
        smooth_serial(lf_factors, lf_image, &mut smoothed, xsize, ysize);
    }

    #[cfg(not(feature = "threads"))]
    smooth_serial(lf_factors, lf_image, &mut smoothed, xsize, ysize);

    *lf_image = smoothed;
    Ok(())
}

fn smooth_serial(
    lf_factors: [f32; 3],
    lf_image: &[Image<f32>; 3],
    smoothed: &mut [Image<f32>; 3],
    xsize: usize,
    ysize: usize,
) {
    for y in 1..ysize - 1 {
        for x in [0, xsize - 1] {
            for c in 0..3 {
                smoothed[c].row_mut(y)[x] = lf_image[c].row(y)[x];
            }
        }
        for x in 1..xsize - 1 {
            let gap = 0.5;
            let (mc_x, sm_x, gap) = compute_pixel_channel(
                lf_factors[0],
                gap,
                x,
                lf_image[0].row(y - 1),
                lf_image[0].row(y),
                lf_image[0].row(y + 1),
            );
            let (mc_y, sm_y, gap) = compute_pixel_channel(
                lf_factors[1],
                gap,
                x,
                lf_image[1].row(y - 1),
                lf_image[1].row(y),
                lf_image[1].row(y + 1),
            );
            let (mc_b, sm_b, gap) = compute_pixel_channel(
                lf_factors[2],
                gap,
                x,
                lf_image[2].row(y - 1),
                lf_image[2].row(y),
                lf_image[2].row(y + 1),
            );
            let factor = (3.0 - 4.0 * gap).max(0.0);
            smoothed[0].row_mut(y)[x] = (sm_x - mc_x) * factor + mc_x;
            smoothed[1].row_mut(y)[x] = (sm_y - mc_y) * factor + mc_y;
            smoothed[2].row_mut(y)[x] = (sm_b - mc_b) * factor + mc_b;
        }
    }
}
