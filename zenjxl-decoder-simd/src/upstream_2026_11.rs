// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.

// Tests retained verbatim from jxl-rs f370be423933628eb7f783f1b61d4c765eca5f79,
// jxl_simd/src/lib.rs. Only module placement/imports differ.
use crate::{F32SimdVec, I32SimdVec, SimdDescriptor, test_all_instruction_sets};

fn test_store_u8<D: SimdDescriptor>(d: D) {
    let data = [
        0xba_i32,
        0x12345678_i32,
        0xdeadbabeu32 as i32,
        0x76543210_i32,
        0x11111111_i32,
        0x00000000_i32,
        0xffffffffu32 as i32,
        0x12345678_i32,
        0x87654321u32 as i32,
        0xabcdef01u32 as i32,
        0x10203040_i32,
        0x50607080_i32,
        0x01020304_i32,
        0x05060708_i32,
        0x090a0b0c_i32,
        0x0d0e0f00_i32,
    ];
    let mut output = [0u8; 16];
    for i in (0..16).step_by(D::I32Vec::LEN) {
        let vec = D::I32Vec::load(d, &data[i..]);
        vec.store_u8(&mut output[i..]);
    }

    for i in 0..16 {
        let expected = (data[i] & 0xff) as u8;
        assert_eq!(
            output[i], expected,
            "store_u8 failed at index {}: expected 0x{:02x}, got 0x{:02x}",
            i, expected, output[i]
        );
    }
}
test_all_instruction_sets!(test_store_u8);

fn test_f16_load_roundtrip<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    let values: &[f32] = &[0.0, 1.0, -1.0, 0.5, 100.0, -42.5, 65504.0, 0.001];
    for chunk in values.chunks(len) {
        let mut input = vec![0.0f32; len];
        input[..chunk.len()].copy_from_slice(chunk);
        let v = D::F32Vec::load(d, &input);
        let mut f16_buf = vec![0u16; len];
        v.store_f16_bits(&mut f16_buf);
        let back = D::F32Vec::load_f16_bits(d, &f16_buf);
        let mut output = vec![0.0f32; len];
        back.store(&mut output);
        for i in 0..chunk.len() {
            let expected = input[i];
            if expected == 0.0 {
                assert_eq!(
                    output[i].to_bits() & 0x7FFF_FFFF,
                    0,
                    "zero not preserved at {i}"
                );
            } else {
                let rel_err = ((expected - output[i]) / expected).abs();
                assert!(
                    rel_err < 0.001,
                    "roundtrip failed at {i}: {expected} -> {}",
                    output[i]
                );
            }
        }
    }
}
test_all_instruction_sets!(test_f16_load_roundtrip);

fn test_f16_load_special_values<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    let cases: &[(u16, f32)] = &[
        (0x0001, 5.960_464_5e-8),    // smallest positive subnormal
        (0x03FF, 6.097_555e-5),      // largest positive subnormal
        (0x8001, -5.960_464_5e-8),   // smallest negative subnormal
        (0x7C00, f32::INFINITY),     // +inf
        (0xFC00, f32::NEG_INFINITY), // -inf
        (0x7C01, f32::NAN),          // NaN
        (0x0000, 0.0),               // +zero
    ];
    for chunk in cases.chunks(len) {
        let mut bits = vec![0u16; len];
        for (i, &(b, _)) in chunk.iter().enumerate() {
            bits[i] = b;
        }
        let v = D::F32Vec::load_f16_bits(d, &bits);
        let mut output = vec![0.0f32; len];
        v.store(&mut output);
        for (i, &(_, expected)) in chunk.iter().enumerate() {
            if expected.is_nan() {
                assert!(output[i].is_nan(), "expected NaN at {i}, got {}", output[i]);
            } else if expected.is_infinite() {
                assert_eq!(output[i], expected, "inf mismatch at {i}");
            } else if expected == 0.0 {
                assert_eq!(output[i], 0.0, "zero mismatch at {i}");
            } else {
                assert!(
                    (output[i] - expected).abs() < expected.abs() * 1e-3,
                    "subnormal mismatch at {i}: expected {expected}, got {}",
                    output[i],
                );
            }
        }
    }
}
test_all_instruction_sets!(test_f16_load_special_values);

fn test_f16_store_roundtrip<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    let known: &[(f32, u16)] = &[(1.0, 0x3C00), (-2.0, 0xC000), (0.5, 0x3800), (0.0, 0x0000)];
    let mut input = vec![0.0f32; len];
    let mut expected_bits = vec![0u16; len];
    for i in 0..len.min(known.len()) {
        input[i] = known[i].0;
        expected_bits[i] = known[i].1;
    }
    let v = D::F32Vec::load(d, &input);
    let mut output = vec![0u16; len];
    v.store_f16_bits(&mut output);
    for i in 0..len.min(known.len()) {
        assert_eq!(
            output[i], expected_bits[i],
            "store_f16_bits wrong at {i}: input {}, expected 0x{:04X}, got 0x{:04X}",
            input[i], expected_bits[i], output[i],
        );
    }
}
test_all_instruction_sets!(test_f16_store_roundtrip);

fn test_f16_store_special_values<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    // (f32 input, expected f16 bits)
    let cases: &[(f32, u16)] = &[
        (5.960_464_5e-8, 0x0001),    // smallest positive subnormal
        (6.097_555e-5, 0x03FF),      // largest positive subnormal
        (-5.960_464_5e-8, 0x8001),   // smallest negative subnormal
        (f32::INFINITY, 0x7C00),     // +inf
        (f32::NEG_INFINITY, 0xFC00), // -inf
        (100000.0, 0x7C00),          // overflow to inf
        (65504.0, 0x7BFF),           // max normal f16
        (f32::NAN, 0x7E00),          // NaN preserved
    ];
    for chunk in cases.chunks(len) {
        let mut input = vec![0.0f32; len];
        for (i, &(v, _)) in chunk.iter().enumerate() {
            input[i] = v;
        }
        let v = D::F32Vec::load(d, &input);
        let mut output = vec![0u16; len];
        v.store_f16_bits(&mut output);
        for (i, &(inp, expected)) in chunk.iter().enumerate() {
            assert_eq!(
                output[i], expected,
                "store_f16_bits special at {i}: input {inp}, expected 0x{expected:04X}, got 0x{:04X}",
                output[i],
            );
        }
    }
}
test_all_instruction_sets!(test_f16_store_special_values);

fn test_f16_underflow_to_zero<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    let mut input = vec![0.0f32; len];
    input[0] = 1e-8;
    if len > 1 {
        input[1] = -1e-8;
    }
    if len > 2 {
        input[2] = 1e-10;
    }
    let v = D::F32Vec::load(d, &input);
    let mut f16_buf = vec![0u16; len];
    v.store_f16_bits(&mut f16_buf);
    // Magnitude bits should be zero (sign may be preserved)
    for (i, &bits) in f16_buf.iter().enumerate().take(3.min(len)) {
        assert_eq!(
            bits & 0x7FFF,
            0,
            "underflow not flushed to zero at {i}: got 0x{bits:04X}",
        );
    }
}
test_all_instruction_sets!(test_f16_underflow_to_zero);

fn compare_f16_bits(s: u16, m: u16) {
    let (s_exp, m_exp) = ((s >> 10) & 0x1F, (m >> 10) & 0x1F);
    if s_exp == 0x1F || m_exp == 0x1F {
        assert_eq!(s_exp, m_exp, "inf/nan: scalar 0x{s:04X}, simd 0x{m:04X}");
        return;
    }
    let (s_mag, m_mag) = (s & 0x7FFF, m & 0x7FFF);
    let diff_mag = (s_mag as i32 - m_mag as i32).unsigned_abs();
    assert!(diff_mag <= 1, "magnitude: scalar 0x{s:04X}, simd 0x{m:04X}");
    if s_mag != 0 || m_mag != 0 {
        let (s_sign, m_sign) = (s & 0x8000, m & 0x8000);
        assert_eq!(s_sign, m_sign, "sign: scalar 0x{s:04X}, simd 0x{m:04X}");
    }
}

fn test_f16_store_scalar_equivalent<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    arbtest::arbtest(|u| {
        let mut input = vec![0.0f32; len];
        for v in input.iter_mut() {
            *v = f32::from_bits(u.arbitrary::<u32>()?);
        }
        let mut simd_bits = vec![0u16; len];
        D::F32Vec::load(d, &input).store_f16_bits(&mut simd_bits);
        for i in 0..len {
            compare_f16_bits(crate::f16::from_f32(input[i]).to_bits(), simd_bits[i]);
        }
        Ok(())
    });
}
test_all_instruction_sets!(test_f16_store_scalar_equivalent);

fn test_f16_load_scalar_equivalent<D: SimdDescriptor>(d: D) {
    let len = D::F32Vec::LEN;
    arbtest::arbtest(|u| {
        let mut input = vec![0u16; len];
        for v in input.iter_mut() {
            *v = u.arbitrary::<u16>()?;
        }
        let mut simd_f32 = vec![0.0f32; len];
        D::F32Vec::load_f16_bits(d, &input).store(&mut simd_f32);
        for i in 0..len {
            let (s, m) = (crate::f16::from_bits(input[i]).to_f32(), simd_f32[i]);
            if s.is_nan() {
                assert!(m.is_nan(), "expected NaN for 0x{:04X}, got {m}", input[i]);
                continue;
            }
            let (s_b, m_b) = (s.to_bits(), m.to_bits());
            assert_eq!(s_b, m_b, "load 0x{:04X}: scalar {s}, simd {m}", input[i]);
        }
        Ok(())
    });
}
test_all_instruction_sets!(test_f16_load_scalar_equivalent);
