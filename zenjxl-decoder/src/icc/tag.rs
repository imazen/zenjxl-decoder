// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::{Cursor, Write};
use whereat::at;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{Error, Result};
use crate::util::NewWithCapacity;
use crate::util::tracing_wrappers::warn;

use super::{ICC_HEADER_SIZE, IccStream, read_varint_from_reader};

const COMMON_TAGS: [&[u8; 4]; 19] = [
    b"rTRC", b"rXYZ", b"cprt", b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"rTRC",
    b"gTRC", b"bTRC", b"kTRC", b"chad", b"desc", b"chrm", b"dmnd", b"dmdd", b"lumi",
];

const COMMON_DATA: [&[u8; 4]; 8] = [
    b"XYZ ", b"desc", b"text", b"mluc", b"para", b"curv", b"sf32", b"gbd ",
];

pub(super) fn read_tag_list(
    data_stream: &mut IccStream,
    commands_stream: &mut Cursor<Vec<u8>>,
    decoded_profile: &mut Cursor<&mut [u8]>,
    num_tags: u32,
    output_size: u64,
) -> Result<()> {
    let mut prev_tagstart = num_tags
        .checked_mul(12)
        .and_then(|v| v.checked_add(ICC_HEADER_SIZE as u32))
        .ok_or(Error::InvalidIccStream)?;
    let mut prev_tagsize = 0u32;

    loop {
        let Ok(command) = commands_stream.read_u8() else {
            // End of commands stream
            return Ok(());
        };

        let tagcode = command & 63;
        let tag = match tagcode {
            0 => break,
            1 => {
                let mut tag = [0u8; 4];
                data_stream.read_exact(&mut tag)?;
                tag
            }
            2..=20 => *COMMON_TAGS[(tagcode - 2) as usize],
            _ => return Err(at!(Error::InvalidIccStream)),
        };

        let tagstart = if command & 64 == 0 {
            prev_tagstart
                .checked_add(prev_tagsize)
                .ok_or(Error::InvalidIccStream)?
        } else {
            // `as u32` truncated a 64-bit varint, so a huge offset wrapped to
            // a small one and passed the bounds check below. Upstream jxl-rs
            // 0f41860.
            u32::try_from(read_varint_from_reader(commands_stream)?)
                .map_err(|_| Error::InvalidIccStream)?
        };
        let tagsize = match &tag {
            _ if command & 128 != 0 => u32::try_from(read_varint_from_reader(commands_stream)?)
                .map_err(|_| Error::InvalidIccStream)?,
            b"rXYZ" | b"gXYZ" | b"bXYZ" | b"kXYZ" | b"wtpt" | b"bkpt" | b"lumi" => 20,
            _ => prev_tagsize,
        };
        if (tagstart as u64 + tagsize as u64) > output_size {
            warn!(output_size, tagstart, tagsize, "tag size overflow");
            return Err(at!(Error::InvalidIccStream));
        }

        prev_tagstart = tagstart;
        prev_tagsize = tagsize;

        let tagstart_plus_size = tagstart
            .checked_add(tagsize)
            .ok_or(Error::InvalidIccStream)?;
        let tagstart_plus_size_x2 = tagstart
            .checked_add(tagsize.checked_mul(2).ok_or(Error::InvalidIccStream)?)
            .ok_or(Error::InvalidIccStream)?;

        let write_result = (|| -> std::io::Result<()> {
            decoded_profile.write_all(&tag)?;
            decoded_profile.write_u32::<BigEndian>(tagstart)?;
            decoded_profile.write_u32::<BigEndian>(tagsize)?;
            if tagcode == 2 {
                decoded_profile.write_all(b"gTRC")?;
                decoded_profile.write_u32::<BigEndian>(tagstart)?;
                decoded_profile.write_u32::<BigEndian>(tagsize)?;
                decoded_profile.write_all(b"bTRC")?;
                decoded_profile.write_u32::<BigEndian>(tagstart)?;
                decoded_profile.write_u32::<BigEndian>(tagsize)?;
            } else if tagcode == 3 {
                decoded_profile.write_all(b"gXYZ")?;
                decoded_profile.write_u32::<BigEndian>(tagstart_plus_size)?;
                decoded_profile.write_u32::<BigEndian>(tagsize)?;
                decoded_profile.write_all(b"bXYZ")?;
                decoded_profile.write_u32::<BigEndian>(tagstart_plus_size_x2)?;
                decoded_profile.write_u32::<BigEndian>(tagsize)?;
            }
            Ok(())
        })();
        write_result.map_err(|_| Error::InvalidIccStream)?;
    }

    Ok(())
}

fn shuffle_w2(bytes: &[u8]) -> Result<Vec<u8>> {
    let len = bytes.len();
    let mut out = Vec::new_with_capacity(len).map_err(|e| at!(Error::from(e)))?;

    let height = len / 2;
    let odd = len % 2;
    for idx in 0..height {
        out.push(bytes[idx]);
        out.push(bytes[idx + height + odd]);
    }
    if odd != 0 {
        out.push(bytes[height]);
    }
    Ok(out)
}

fn shuffle_w4(bytes: &[u8]) -> Result<Vec<u8>> {
    let len = bytes.len();
    let mut out = Vec::new_with_capacity(len).map_err(|e| at!(Error::from(e)))?;

    let step = len / 4;
    let wide_count = len % 4;
    for idx in 0..step {
        let mut base = idx;
        for _ in 0..wide_count {
            out.push(bytes[base]);
            base += step + 1;
        }
        for _ in wide_count..4 {
            out.push(bytes[base]);
            base += step;
        }
    }
    for idx in 1..=wide_count {
        out.push(bytes[(step + 1) * idx - 1]);
    }
    Ok(out)
}

pub(super) fn read_single_command(
    data_stream: &mut IccStream,
    commands_stream: &mut Cursor<Vec<u8>>,
    decoded_profile: &mut Cursor<&mut [u8]>,
    command: u8,
) -> Result<()> {
    use std::num::Wrapping;

    match command {
        1 => {
            let num = read_varint_from_reader(commands_stream)? as usize;
            data_stream.copy_bytes(decoded_profile, num)?;
        }
        2 | 3 => {
            let num = read_varint_from_reader(commands_stream)? as usize;
            let bytes = data_stream.read_to_vec_exact(num)?;
            let bytes = if command == 2 {
                shuffle_w2(&bytes)?
            } else {
                shuffle_w4(&bytes)?
            };
            decoded_profile
                .write_all(&bytes)
                .map_err(|_| Error::InvalidIccStream)?;
        }
        4 => {
            let flags = commands_stream
                .read_u8()
                .map_err(|_| Error::InvalidIccStream)?;
            let width = ((flags & 3) + 1) as usize;
            let order = (flags >> 2) & 3;
            if width == 3 || order == 3 {
                return Err(at!(Error::InvalidIccStream));
            }

            let stride = if (flags & 16) == 0 {
                width
            } else {
                let stride = read_varint_from_reader(commands_stream)? as usize;
                if stride < width {
                    return Err(at!(Error::InvalidIccStream));
                }
                stride
            };
            if stride.saturating_mul(4) >= decoded_profile.position() as usize {
                return Err(at!(Error::InvalidIccStream));
            }

            let num = read_varint_from_reader(commands_stream)? as usize;
            let bytes = data_stream.read_to_vec_exact(num)?;
            let bytes = match width {
                1 => bytes,
                2 => shuffle_w2(&bytes)?,
                4 => shuffle_w4(&bytes)?,
                _ => unreachable!(),
            };

            for i in (0..num).step_by(width) {
                let base_position = decoded_profile.position() as usize;
                let buffer_slice = decoded_profile.get_ref();

                let mut prev = [Wrapping(0u32); 3];
                for (j, p) in prev[..=order as usize].iter_mut().enumerate() {
                    let offset = base_position - (stride * (j + 1));
                    let mut p_bytes = [0u8; 4];
                    p_bytes[(4 - width)..].copy_from_slice(&buffer_slice[offset..offset + width]);
                    p.0 = u32::from_be_bytes(p_bytes);
                }

                let p = match order {
                    0 => prev[0],
                    1 => Wrapping(2) * prev[0] - prev[1],
                    2 => Wrapping(3) * (prev[0] - prev[1]) + prev[2],
                    _ => unreachable!(),
                };

                decoded_profile.set_position(base_position as u64);
                for j in 0..width.min(num - i) {
                    let val = Wrapping(bytes[i + j] as u32) + (p >> (8 * (width - 1 - j)));
                    decoded_profile
                        .write_u8(val.0 as u8)
                        .map_err(|_| Error::InvalidIccStream)?;
                }
            }
        }
        10 => {
            let mut bytes = [0u8; 20];
            bytes[..4].copy_from_slice(b"XYZ ");
            data_stream.read_exact(&mut bytes[8..])?;
            decoded_profile
                .write_all(&bytes)
                .map_err(|_| Error::InvalidIccStream)?;
        }
        16..=23 => {
            decoded_profile
                .write_all(COMMON_DATA[command as usize - 16])
                .and_then(|_| decoded_profile.write_all(&[0u8; 4]))
                .map_err(|_| Error::InvalidIccStream)?;
        }
        _ => {
            return Err(at!(Error::InvalidIccStream));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (v & 127) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 128);
        }
    }

    fn run(commands: Vec<u8>, num_tags: u32) -> Result<()> {
        let mut data = IccStream::new(Vec::new());
        let mut commands = Cursor::new(commands);
        let mut profile = vec![0u8; 1024];
        let mut profile = Cursor::new(&mut profile[..]);
        read_tag_list(&mut data, &mut commands, &mut profile, num_tags, 1024)
    }

    #[test]
    fn tag_start_above_u32_is_rejected() {
        // rXYZ (tag code 3) with an explicit start of 2^32 + 128. Truncating
        // the varint to u32 turned it into 128, which fits a 1024-byte profile.
        let mut commands = vec![3 | 64];
        commands.extend(varint((1 << 32) + 128));
        assert!(run(commands.clone(), 1).is_err());
        commands[0] = 3 | 64 | 128;
        commands.truncate(1);
        commands.extend(varint(128));
        commands.extend(varint((1 << 32) + 20));
        assert!(run(commands, 1).is_err());
        let mut ok = vec![3 | 64];
        ok.extend(varint(128));
        assert!(run(ok, 1).is_ok());
    }

    #[test]
    fn tag_count_overflow_is_rejected() {
        assert!(run(vec![0], u32::MAX / 4).is_err());
    }
}
