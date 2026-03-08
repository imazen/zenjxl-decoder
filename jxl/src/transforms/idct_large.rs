// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains a generic implementation of large (>32x32) 2d IDCTs.
// They are not implemented in the same way as smaller 2d IDCTs to reduce code size.

#![allow(clippy::excessive_precision)]

use std::f32::consts::SQRT_2;

use super::apply_idct_32;

const WC_WEIGHTS_64: [f32; 32] = [
    0.500150636020651,
    0.5013584524464084,
    0.5037887256810443,
    0.5074711720725553,
    0.5124514794082247,
    0.5187927131053328,
    0.52657731515427,
    0.535909816907992,
    0.5469204379855088,
    0.5597698129470802,
    0.57465518403266,
    0.5918185358574165,
    0.6115573478825099,
    0.6342389366884031,
    0.6603198078137061,
    0.6903721282002123,
    0.7251205223771985,
    0.7654941649730891,
    0.8127020908144905,
    0.8683447152233481,
    0.9345835970364075,
    1.0144082649970547,
    1.1120716205797176,
    1.233832737976571,
    1.3892939586328277,
    1.5939722833856311,
    1.8746759800084078,
    2.282050068005162,
    2.924628428158216,
    4.084611078129248,
    6.796750711673633,
    20.373878167231453,
];

const WC_WEIGHTS_128: [f32; 64] = [
    0.5000376519155477,
    0.5003390374428216,
    0.5009427176380873,
    0.5018505174842379,
    0.5030651913013697,
    0.5045904432216454,
    0.5064309549285542,
    0.5085924210498143,
    0.5110815927066812,
    0.5139063298475396,
    0.5170756631334912,
    0.5205998663018917,
    0.524490540114724,
    0.5287607092074876,
    0.5334249333971333,
    0.538499435291984,
    0.5440022463817783,
    0.549953374183236,
    0.5563749934898856,
    0.5632916653417023,
    0.5707305880121454,
    0.5787218851348208,
    0.5872989370937893,
    0.5964987630244563,
    0.606362462272146,
    0.6169357260050706,
    0.6282694319707711,
    0.6404203382416639,
    0.6534518953751283,
    0.6674352009263413,
    0.6824501259764195,
    0.6985866506472291,
    0.7159464549705746,
    0.7346448236478627,
    0.7548129391165311,
    0.776600658233963,
    0.8001798956216941,
    0.8257487738627852,
    0.8535367510066064,
    0.8838110045596234,
    0.9168844461846523,
    0.9531258743921193,
    0.9929729612675466,
    1.036949040910389,
    1.0856850642580145,
    1.1399486751015042,
    1.2006832557294167,
    1.2690611716991191,
    1.346557628206286,
    1.4350550884414341,
    1.5369941008524954,
    1.6555965242641195,
    1.7952052190778898,
    1.961817848571166,
    2.163957818751979,
    2.4141600002500763,
    2.7316450287739396,
    3.147462191781909,
    3.7152427383269746,
    4.5362909369693565,
    5.827688377844654,
    8.153848602466814,
    13.58429025728446,
    40.744688103351834,
];

const WC_WEIGHTS_256: [f32; 128] = [
    0.5000094125358878,
    0.500084723455784,
    0.5002354020255269,
    0.5004615618093246,
    0.5007633734146156,
    0.5011410648064231,
    0.5015949217281668,
    0.502125288230386,
    0.5027325673091954,
    0.5034172216566842,
    0.5041797745258774,
    0.5050208107132756,
    0.5059409776624396,
    0.5069409866925212,
    0.5080216143561264,
    0.509183703931388,
    0.5104281670536573,
    0.5117559854927805,
    0.5131682130825206,
    0.5146659778093218,
    0.516250484068288,
    0.5179230150949777,
    0.5196849355823947,
    0.5215376944933958,
    0.5234828280796439,
    0.52552196311921,
    0.5276568203859896,
    0.5298892183652453,
    0.5322210772308335,
    0.5346544231010253,
    0.537191392591309,
    0.5398342376841637,
    0.5425853309375497,
    0.545447171055775,
    0.5484223888484947,
    0.551513753605893,
    0.554724179920619,
    0.5580567349898085,
    0.5615146464335654,
    0.5651013106696203,
    0.5688203018875696,
    0.5726753816701664,
    0.5766705093136241,
    0.5808098529038624,
    0.5850978012111273,
    0.58953897647151,
    0.5941382481306648,
    0.5989007476325463,
    0.6038318843443582,
    0.6089373627182432,
    0.614223200800649,
    0.6196957502119484,
    0.6253617177319102,
    0.6312281886412079,
    0.6373026519855411,
    0.6435930279473415,
    0.6501076975307724,
    0.6568555347890955,
    0.6638459418498757,
    0.6710888870233562,
    0.6785949463131795,
    0.6863753486870501,
    0.6944420255086364,
    0.7028076645818034,
    0.7114857693151208,
    0.7204907235796304,
    0.7298378629074134,
    0.7395435527641373,
    0.749625274727372,
    0.7601017215162176,
    0.7709929019493761,
    0.7823202570613161,
    0.7941067887834509,
    0.8063772028037925,
    0.8191580674598145,
    0.83247799080191,
    0.8463678182968619,
    0.860860854031955,
    0.8759931087426972,
    0.8918035785352535,
    0.9083345588266809,
    0.9256319988042384,
    0.9437459026371479,
    0.962730784794803,
    0.9826461881778968,
    1.0035572754078206,
    1.0255355056139732,
    1.048659411496106,
    1.0730154944316674,
    1.0986992590905857,
    1.1258164135986009,
    1.1544842669978943,
    1.184833362908442,
    1.217009397314603,
    1.2511754798461228,
    1.287514812536712,
    1.326233878832723,
    1.3675662599582539,
    1.411777227500661,
    1.459169302866857,
    1.5100890297227016,
    1.5649352798258847,
    1.6241695131835794,
    1.6883285509131505,
    1.7580406092704062,
    1.8340456094306077,
    1.9172211551275689,
    2.0086161135167564,
    2.1094945286246385,
    2.22139377701127,
    2.346202662531156,
    2.486267909203593,
    2.644541877144861,
    2.824791402350551,
    3.0318994541759925,
    3.2723115884254845,
    3.5547153325075804,
    3.891107790700307,
    4.298537526449054,
    4.802076008665048,
    5.440166215091329,
    6.274908408039339,
    7.413566756422303,
    9.058751453879703,
    11.644627325175037,
    16.300023088031555,
    27.163977662448232,
    81.48784219222516,
];

/// Recursive 1D IDCT on a contiguous buffer.
/// `data` must have exactly n elements where n is 32, 64, 128, or 256.
/// `scratch` must have at least n elements.
fn idct_recursive(data: &mut [f32], scratch: &mut [f32]) {
    let n = data.len();
    debug_assert!(scratch.len() >= n);

    if n == 32 {
        apply_idct_32(data, 0, 1);
        return;
    }

    let weights = match n {
        64 => &WC_WEIGHTS_64[..],
        128 => &WC_WEIGHTS_128[..],
        256 => &WC_WEIGHTS_256[..],
        _ => unreachable!("invalid large-dct size: {n}"),
    };

    let (first_half, second_half) = scratch[..n].split_at_mut(n / 2);
    for i in 0..n / 2 {
        first_half[i] = data[i * 2];
        second_half[i] = data[2 * i + 1];
    }

    idct_recursive(first_half, &mut data[..n / 2]);

    for i in (1..n / 2).rev() {
        second_half[i] += second_half[i - 1];
    }
    second_half[0] *= SQRT_2;

    idct_recursive(second_half, &mut data[..n / 2]);

    for i in 0..n / 2 {
        data[i] = second_half[i].mul_add(weights[i], first_half[i]);
        data[n - i - 1] = (-second_half[i]).mul_add(weights[i], first_half[i]);
    }
}

/// Generic 2D IDCT implementation for large sizes (64, 128, 256).
/// For tall matrices (rows > cols), input is in C×R (transposed) layout per JPEG XL convention.
fn idct2d_large_impl(data: &mut [f32], rows: usize, cols: usize) {
    let n = rows.max(cols);
    let mut col_buf = vec![0.0f32; n];
    let mut scratch = vec![0.0f32; n];

    if rows > cols {
        // Tall: input is stored as cols×rows (transposed).
        // Step 1: cols-point column IDCTs on cols×rows layout (stride = rows)
        for c in 0..rows {
            for r in 0..cols {
                col_buf[r] = data[r * rows + c];
            }
            idct_recursive(&mut col_buf[..cols], &mut scratch[..cols]);
            for r in 0..cols {
                data[r * rows + c] = col_buf[r];
            }
        }
        // Step 2: Transpose cols×rows → rows×cols
        let mut tmp = vec![0.0f32; rows * cols];
        for i in 0..cols {
            for j in 0..rows {
                tmp[j * cols + i] = data[i * rows + j];
            }
        }
        data[..rows * cols].copy_from_slice(&tmp);
        // Step 3: rows-point column IDCTs on rows×cols layout (stride = cols)
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = data[r * cols + c];
            }
            idct_recursive(&mut col_buf[..rows], &mut scratch[..rows]);
            for r in 0..rows {
                data[r * cols + c] = col_buf[r];
            }
        }
    } else {
        // Square or wide: input is in rows×cols layout.
        // Step 1: rows-point column IDCTs (stride = cols)
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = data[r * cols + c];
            }
            idct_recursive(&mut col_buf[..rows], &mut scratch[..rows]);
            for r in 0..rows {
                data[r * cols + c] = col_buf[r];
            }
        }
        // Step 2: Transpose rows×cols → cols×rows
        let mut tmp = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                tmp[j * rows + i] = data[i * cols + j];
            }
        }
        data[..rows * cols].copy_from_slice(&tmp);
        // Step 3: cols-point column IDCTs on cols×rows layout (stride = rows)
        for c in 0..rows {
            for r in 0..cols {
                col_buf[r] = data[r * rows + c];
            }
            idct_recursive(&mut col_buf[..cols], &mut scratch[..cols]);
            for r in 0..cols {
                data[r * rows + c] = col_buf[r];
            }
        }
        // Step 4: For wide, transpose back to rows×cols
        if rows != cols {
            for i in 0..cols {
                for j in 0..rows {
                    tmp[j * cols + i] = data[i * rows + j];
                }
            }
            data[..rows * cols].copy_from_slice(&tmp);
        }
    }
}

macro_rules! make_idct2d {
    ($name:ident, $h:literal, $w:literal) => {
        pub fn $name(data: &mut [f32]) {
            assert_eq!(data.len(), $h * $w);
            idct2d_large_impl(data, $h, $w);
        }
    };
}

make_idct2d!(idct2d_32_64, 32, 64);
make_idct2d!(idct2d_64_32, 64, 32);
make_idct2d!(idct2d_64_64, 64, 64);
make_idct2d!(idct2d_64_128, 64, 128);
make_idct2d!(idct2d_128_64, 128, 64);
make_idct2d!(idct2d_128_128, 128, 128);
make_idct2d!(idct2d_128_256, 128, 256);
make_idct2d!(idct2d_256_128, 256, 128);
make_idct2d!(idct2d_256_256, 256, 256);

#[cfg(test)]
pub fn apply_idct_64(data: &mut [f32], col: usize, stride: usize) {
    let mut buf = [0.0f32; 64];
    let mut scratch = [0.0f32; 64];
    for i in 0..64 {
        buf[i] = data[col + i * stride];
    }
    idct_recursive(&mut buf, &mut scratch);
    for i in 0..64 {
        data[col + i * stride] = buf[i];
    }
}

#[cfg(test)]
pub fn apply_idct_128(data: &mut [f32], col: usize, stride: usize) {
    let mut buf = [0.0f32; 128];
    let mut scratch = [0.0f32; 128];
    for i in 0..128 {
        buf[i] = data[col + i * stride];
    }
    idct_recursive(&mut buf, &mut scratch);
    for i in 0..128 {
        data[col + i * stride] = buf[i];
    }
}

#[cfg(test)]
pub fn apply_idct_256(data: &mut [f32], col: usize, stride: usize) {
    let mut buf = [0.0f32; 256];
    let mut scratch = [0.0f32; 256];
    for i in 0..256 {
        buf[i] = data[col + i * stride];
    }
    idct_recursive(&mut buf, &mut scratch);
    for i in 0..256 {
        data[col + i * stride] = buf[i];
    }
}
