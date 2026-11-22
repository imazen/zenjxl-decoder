// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use jxl_simd::{
    F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U8SimdVec, U16SimdVec, simd_function,
};

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    render::low_memory_pipeline::row_buffers::RowBuffer,
};

macro_rules! define_run_interleaved {
    ($fn_name:ident, $ty:ty, $vec_trait:ident, $store_fn:ident, $cnt:expr, $($arg:ident),+) => {
        #[inline(always)]
        fn $fn_name<D: SimdDescriptor>(
            d: D,
            $($arg: &[$ty]),+,
            out: &mut [$ty],
        ) -> usize {
            let len = D::$vec_trait::LEN;
            let mut n = 0;
            let limit = [$($arg.len()),+][0];

            {
                let out_chunks = out[..limit * $cnt].chunks_exact_mut(len * $cnt);
                $(let mut $arg = $arg.chunks_exact(len);)+
                for out_chunk in out_chunks {
                    $(let $arg = D::$vec_trait::load(d, $arg.next().unwrap());)+
                    D::$vec_trait::$store_fn($($arg),+, out_chunk);
                    n += len;
                }
            }

            let d256 = d.maybe_downgrade_256bit();
            let len256 = <D::Descriptor256 as SimdDescriptor>::$vec_trait::LEN;
            if len256 < len {
                let out_chunks = out[n * $cnt..limit * $cnt].chunks_exact_mut(len256 * $cnt);
                $(let mut $arg = $arg[n..limit].chunks_exact(len256);)+
                for out_chunk in out_chunks {
                    $(let $arg = <D::Descriptor256 as SimdDescriptor>::$vec_trait::load(d256, $arg.next().unwrap());)+
                    <D::Descriptor256 as SimdDescriptor>::$vec_trait::$store_fn($($arg),+, out_chunk);
                    n += len256;
                }
            }

            let d128 = d.maybe_downgrade_128bit();
            let len128 = <D::Descriptor128 as SimdDescriptor>::$vec_trait::LEN;
            if len128 < len {
                let out_chunks = out[n * $cnt..limit * $cnt].chunks_exact_mut(len128 * $cnt);
                $(let mut $arg = $arg[n..limit].chunks_exact(len128);)+
                for out_chunk in out_chunks {
                    $(let $arg = <D::Descriptor128 as SimdDescriptor>::$vec_trait::load(d128, $arg.next().unwrap());)+
                    <D::Descriptor128 as SimdDescriptor>::$vec_trait::$store_fn($($arg),+, out_chunk);
                    n += len128;
                }
            }

            n
        }
    };
}

define_run_interleaved!(
    run_interleaved_2_f32,
    f32,
    F32Vec,
    store_interleaved_2,
    2,
    a,
    b
);
define_run_interleaved!(
    run_interleaved_3_f32,
    f32,
    F32Vec,
    store_interleaved_3,
    3,
    a,
    b,
    c
);
define_run_interleaved!(
    run_interleaved_4_f32,
    f32,
    F32Vec,
    store_interleaved_4,
    4,
    a,
    b,
    c,
    e
);

simd_function!(
    store_interleaved_f32,
    d: D,
    fn store_interleaved_impl_f32(
        inputs: &[&[f32]],
        output: &mut [f32]
    ) -> usize {
        match inputs.len() {
            2 => run_interleaved_2_f32(d, inputs[0], inputs[1], output),
            3 => run_interleaved_3_f32(d, inputs[0], inputs[1], inputs[2], output),
            4 => run_interleaved_4_f32(d, inputs[0], inputs[1], inputs[2], inputs[3], output),
            _ => 0,
        }
    }
);

define_run_interleaved!(
    run_interleaved_2_u8,
    u8,
    U8Vec,
    store_interleaved_2,
    2,
    a,
    b
);
define_run_interleaved!(
    run_interleaved_3_u8,
    u8,
    U8Vec,
    store_interleaved_3,
    3,
    a,
    b,
    c
);
define_run_interleaved!(
    run_interleaved_4_u8,
    u8,
    U8Vec,
    store_interleaved_4,
    4,
    a,
    b,
    c,
    e
);

simd_function!(
    store_interleaved_u8,
    d: D,
    fn store_interleaved_impl_u8(
        inputs: &[&[u8]],
        output: &mut [u8]
    ) -> usize {
        match inputs.len() {
            2 => run_interleaved_2_u8(d, inputs[0], inputs[1], output),
            3 => run_interleaved_3_u8(d, inputs[0], inputs[1], inputs[2], output),
            4 => run_interleaved_4_u8(d, inputs[0], inputs[1], inputs[2], inputs[3], output),
            _ => 0,
        }
    }
);

simd_function!(
    store_modular_rgb8,
    d: D,
    pub(in crate::render::low_memory_pipeline) fn store_modular_rgb8_impl(
        inputs: [&[i32]; 3],
        output: &mut [u8],
    ) {
        let width = inputs[0].len();
        let block = D::U8Vec::LEN;
        let lanes = D::I32Vec::LEN;
        const { assert!(D::U8Vec::LEN <= 64 && D::U8Vec::LEN.is_multiple_of(D::I32Vec::LEN)) };
        let zero = D::I32Vec::splat(d, 0);
        let max = D::I32Vec::splat(d, 255);
        let mut output_blocks = output[..width * 3].chunks_exact_mut(block * 3);
        let mut x = 0;
        for dst in &mut output_blocks {
            let [r, g, b] = inputs.map(|src| {
                let mut bytes = [0u8; 64];
                for (input, narrow) in src[x..x + block]
                    .chunks_exact(lanes)
                    .zip(bytes[..block].chunks_exact_mut(lanes))
                {
                    let value = D::I32Vec::load(d, input);
                    let low = value.lt_zero().if_then_else_i32(zero, value);
                    value.gt(max).if_then_else_i32(max, low).store_u8(narrow);
                }
                D::U8Vec::load(d, &bytes[..block])
            });
            D::U8Vec::store_interleaved_3(r, g, b, dst);
            x += block;
        }
        for (pixel, x) in output_blocks.into_remainder().chunks_exact_mut(3).zip(x..width) {
            for (dst, src) in pixel.iter_mut().zip(inputs) {
                *dst = src[x].clamp(0, 255) as u8;
            }
        }
    }
);

#[cfg(test)]
#[test]
fn modular_rgb8_save_clamps_odd_rows_without_touching_padding() {
    let mut r = (0..259).map(|x| x - 1).collect::<Vec<_>>();
    r[0] = i32::MIN;
    r[258] = i32::MAX;
    let g = (0..259).map(|x| 258 - x).collect::<Vec<_>>();
    let b = (0..259).map(|x| (x * 17) % 259).collect::<Vec<_>>();
    for width in [1, 15, 31, 65, 127, 259] {
        let mut output = vec![19u8; width * 3 + 7];
        store_modular_rgb8([&r[..width], &g[..width], &b[..width]], &mut output);
        for x in 0..width {
            assert_eq!(
                &output[3 * x..3 * x + 3],
                &[r[x], g[x], b[x]].map(|value| value.clamp(0, 255) as u8)
            );
        }
        assert_eq!(&output[width * 3..], &[19; 7]);
    }
}

define_run_interleaved!(
    run_interleaved_2_u16,
    u16,
    U16Vec,
    store_interleaved_2,
    2,
    a,
    b
);
define_run_interleaved!(
    run_interleaved_3_u16,
    u16,
    U16Vec,
    store_interleaved_3,
    3,
    a,
    b,
    c
);
define_run_interleaved!(
    run_interleaved_4_u16,
    u16,
    U16Vec,
    store_interleaved_4,
    4,
    a,
    b,
    c,
    e
);

simd_function!(
    store_interleaved_u16,
    d: D,
    fn store_interleaved_impl_u16(
        inputs: &[&[u16]],
        output: &mut [u16]
    ) -> usize {
        match inputs.len() {
            2 => run_interleaved_2_u16(d, inputs[0], inputs[1], output),
            3 => run_interleaved_3_u16(d, inputs[0], inputs[1], inputs[2], output),
            4 => run_interleaved_4_u16(d, inputs[0], inputs[1], inputs[2], inputs[3], output),
            _ => 0,
        }
    }
);

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
            store_interleaved_u8(&slices[..channels], output_buf)
        }
        (channels, 2, true) if (2..=4).contains(&channels) => {
            if let Ok(output_u16) = bytemuck::try_cast_slice_mut::<u8, u16>(output_buf) {
                let start_u16 = byte_start / 2;
                let end_u16 = byte_end / 2;
                let mut slices = [&[] as &[u16]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<u16>(input_y)[start_u16..end_u16];
                }
                store_interleaved_u16(&slices[..channels], output_u16)
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
                store_interleaved_f32(&slices[..channels], output_f32)
            } else {
                0
            }
        }
        _ => 0,
    }
}
