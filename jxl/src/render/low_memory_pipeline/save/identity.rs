// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    render::low_memory_pipeline::row_buffers::RowBuffer,
};

/// Generic interleave: N planar channels into packed output.
/// Returns the number of pixels processed.
fn interleave_channels<T: Copy>(inputs: &[&[T]], output: &mut [T]) -> usize {
    let n = inputs.len();
    if n < 2 || n > 4 {
        return 0;
    }
    let pixel_count = inputs[0].len();
    for input in inputs.iter() {
        assert!(input.len() >= pixel_count);
    }
    assert!(output.len() >= pixel_count * n);

    for i in 0..pixel_count {
        for (c, input) in inputs.iter().enumerate() {
            output[i * n + c] = input[i];
        }
    }
    pixel_count
}

pub(super) fn store(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
) -> usize {
    let byte_start = xrange.start * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let byte_end = xrange.end * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let is_native_endian = match data_format {
        JxlDataFormat::U8 { .. } => true,
        JxlDataFormat::F16 { endianness, .. }
        | JxlDataFormat::U16 { endianness, .. }
        | JxlDataFormat::F32 { endianness, .. } => endianness == Endianness::native(),
    };
    let output_buf = output_buf.row_mut(output_y);
    let output_buf = &mut output_buf[0..(byte_end - byte_start) * input_buf.len()];
    match (
        input_buf.len(),
        data_format.bytes_per_sample(),
        is_native_endian,
    ) {
        (1, _, true) => {
            // We can just do a memcpy.
            let input_buf = &input_buf[0].get_row::<u8>(input_y)[byte_start..byte_end];
            assert_eq!(input_buf.len(), output_buf.len());
            output_buf.copy_from_slice(input_buf);
            input_buf.len() / data_format.bytes_per_sample()
        }
        (channels, 1, true) if (2..=4).contains(&channels) => {
            let start_u8 = byte_start;
            let end_u8 = byte_end;
            let mut slices = [&[] as &[u8]; 4];
            for (i, buf) in input_buf.iter().enumerate() {
                slices[i] = &buf.get_row::<u8>(input_y)[start_u8..end_u8];
            }
            interleave_channels(&slices[..channels], output_buf)
        }
        (channels, 2, true) if (2..=4).contains(&channels) => {
            if let Ok(output_u16) = bytemuck::try_cast_slice_mut::<u8, u16>(output_buf) {
                let start_u16 = byte_start / 2;
                let end_u16 = byte_end / 2;
                let mut slices = [&[] as &[u16]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<u16>(input_y)[start_u16..end_u16];
                }
                interleave_channels(&slices[..channels], output_u16)
            } else {
                0
            }
        }
        (channels, 4, true) if (2..=4).contains(&channels) => {
            if let Ok(output_f32) = bytemuck::try_cast_slice_mut::<u8, f32>(output_buf) {
                let start_f32 = byte_start / 4;
                let end_f32 = byte_end / 4;
                let mut slices = [&[] as &[f32]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<f32>(input_y)[start_f32..end_f32];
                }
                interleave_channels(&slices[..channels], output_f32)
            } else {
                0
            }
        }
        _ => 0,
    }
}
