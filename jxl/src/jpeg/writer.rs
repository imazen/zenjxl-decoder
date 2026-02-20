// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG bitstream writer for reconstruction from JXL.
//!
//! Writes a valid JPEG file from reconstructed coefficient data and
//! JBRD metadata (Huffman tables, quant tables, scan headers, markers).

use crate::error::{Error, Result};

use super::data::*;

/// Write a complete JPEG file from reconstructed data.
///
/// Returns the byte-exact original JPEG.
pub fn write_jpeg(jpeg: &JpegData) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    {
        let mut writer = JpegWriter::new(&mut out);

        // SOI
        writer.write_marker(0xD8)?;

        // Process markers in original order
        let mut app_idx = 0usize;
        let mut com_idx = 0usize;
        let mut scan_idx = 0usize;
        let mut dqt_idx = 0usize;
        let mut dht_idx = 0usize;
        let mut intermarker_idx = 0usize;

        for &marker in &jpeg.marker_order {
            match marker {
                0xD9 => {
                    // EOI
                    writer.write_marker(0xD9)?;
                }
                0xDA => {
                    // SOS - write scan header + entropy-coded data
                    if scan_idx >= jpeg.scan_info.len() {
                        return Err(Error::InvalidJbrd("too many SOS markers".into()));
                    }
                    writer.write_sos(jpeg, scan_idx)?;
                    scan_idx += 1;
                }
                0xE0..=0xEF => {
                    // APP markers
                    if app_idx >= jpeg.app_data.len() {
                        return Err(Error::InvalidJbrd("too many APP markers".into()));
                    }
                    writer.write_app_marker(marker, &jpeg.app_data[app_idx])?;
                    app_idx += 1;
                }
                0xFE => {
                    // COM
                    if com_idx >= jpeg.com_data.len() {
                        return Err(Error::InvalidJbrd("too many COM markers".into()));
                    }
                    writer.write_com_marker(&jpeg.com_data[com_idx])?;
                    com_idx += 1;
                }
                0xDB => {
                    // DQT
                    writer.write_dqt(jpeg, &mut dqt_idx)?;
                }
                0xC4 => {
                    // DHT
                    writer.write_dht(jpeg, &mut dht_idx)?;
                }
                0xC0 => {
                    // SOF0 (baseline)
                    writer.write_sof0(jpeg)?;
                }
                0xDD => {
                    // DRI
                    writer.write_dri(jpeg.restart_interval)?;
                }
                0xFF => {
                    // Inter-marker data
                    if intermarker_idx >= jpeg.inter_marker_data.len() {
                        return Err(Error::InvalidJbrd(
                            "too many inter-marker data".into(),
                        ));
                    }
                    writer.write_intermarker_data(
                        &jpeg.inter_marker_data[intermarker_idx],
                    );
                    intermarker_idx += 1;
                }
                _ => {
                    // Other markers (SOF1, etc.) — shouldn't appear for baseline
                }
            }
        }
    }

    // Tail data (bytes after EOI)
    out.extend_from_slice(&jpeg.tail_data);

    Ok(out)
}

/// JPEG bitstream writer.
struct JpegWriter<'a> {
    out: &'a mut Vec<u8>,
}

impl<'a> JpegWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out }
    }

    fn write_intermarker_data(&mut self, data: &[u8]) {
        self.out.extend_from_slice(data);
    }

    fn write_marker(&mut self, marker: u8) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(marker);
        Ok(())
    }

    fn write_app_marker(&mut self, marker: u8, data: &[u8]) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(marker);
        // APP data includes marker_byte + length + payload in encoder format
        // But in the JBRD decoder, data is the raw payload as stored
        // The length field covers the payload + 2 bytes for the length itself
        let len = (data.len() + 2) as u16;
        self.out.extend_from_slice(&len.to_be_bytes());
        self.out.extend_from_slice(data);
        Ok(())
    }

    fn write_com_marker(&mut self, data: &[u8]) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(0xFE);
        let len = (data.len() + 2) as u16;
        self.out.extend_from_slice(&len.to_be_bytes());
        self.out.extend_from_slice(data);
        Ok(())
    }

    fn write_dri(&mut self, restart_interval: u32) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(0xDD);
        self.out.extend_from_slice(&4u16.to_be_bytes()); // length = 4
        self.out
            .extend_from_slice(&(restart_interval as u16).to_be_bytes());
        Ok(())
    }

    fn write_dqt(&mut self, jpeg: &JpegData, idx: &mut usize) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(0xDB);

        // Collect tables until is_last
        let start = *idx;
        let mut total_payload = 0usize;
        loop {
            if *idx >= jpeg.quant.len() {
                return Err(Error::InvalidJbrd("too many DQT tables".into()));
            }
            let qt = &jpeg.quant[*idx];
            let precision_bytes = if qt.precision == 0 { 1 } else { 2 };
            total_payload += 1 + 64 * precision_bytes; // 1 byte for Pq|Tq
            let is_last = qt.is_last;
            *idx += 1;
            if is_last {
                break;
            }
        }

        let length = (total_payload + 2) as u16;
        self.out.extend_from_slice(&length.to_be_bytes());

        for i in start..*idx {
            let qt = &jpeg.quant[i];
            let pq_tq = ((qt.precision as u8) << 4) | (qt.index as u8);
            self.out.push(pq_tq);
            if qt.precision == 0 {
                // 8-bit values
                for &v in &qt.values {
                    self.out.push(v as u8);
                }
            } else {
                // 16-bit values
                for &v in &qt.values {
                    self.out.extend_from_slice(&(v as u16).to_be_bytes());
                }
            }
        }

        Ok(())
    }

    fn write_dht(&mut self, jpeg: &JpegData, idx: &mut usize) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(0xC4);

        // Collect tables until is_last
        let start = *idx;
        let mut total_payload = 0usize;
        loop {
            if *idx >= jpeg.huffman_code.len() {
                return Err(Error::InvalidJbrd("too many DHT tables".into()));
            }
            let hc = &jpeg.huffman_code[*idx];
            let num_values: u32 = hc.counts.iter().sum();
            total_payload += 1 + 16 + num_values as usize;
            let is_last = hc.is_last;
            *idx += 1;
            if is_last {
                break;
            }
        }

        let length = (total_payload + 2) as u16;
        self.out.extend_from_slice(&length.to_be_bytes());

        for i in start..*idx {
            let hc = &jpeg.huffman_code[i];
            let tc_th = if hc.is_ac { 0x10 } else { 0x00 } | (hc.id as u8);
            self.out.push(tc_th);
            for &count in &hc.counts {
                self.out.push(count as u8);
            }
            for &val in &hc.values {
                self.out.push(val);
            }
        }

        Ok(())
    }

    fn write_sof0(&mut self, jpeg: &JpegData) -> Result<()> {
        self.out.push(0xFF);
        self.out.push(0xC0);

        let nc = jpeg.components.len();
        let length = (8 + 3 * nc) as u16;
        self.out.extend_from_slice(&length.to_be_bytes());

        self.out.push(8); // sample precision = 8 bits
        self.out
            .extend_from_slice(&(jpeg.height as u16).to_be_bytes());
        self.out
            .extend_from_slice(&(jpeg.width as u16).to_be_bytes());
        self.out.push(nc as u8);

        for comp in &jpeg.components {
            self.out.push(comp.id as u8);
            let hv = ((comp.h_samp_factor as u8) << 4) | (comp.v_samp_factor as u8);
            self.out.push(hv);
            self.out.push(comp.quant_idx as u8);
        }

        Ok(())
    }

    fn write_sos(&mut self, jpeg: &JpegData, scan_idx: usize) -> Result<()> {
        let scan = &jpeg.scan_info[scan_idx];

        // SOS header
        self.out.push(0xFF);
        self.out.push(0xDA);

        let length = (6 + 2 * scan.num_components) as u16;
        self.out.extend_from_slice(&length.to_be_bytes());
        self.out.push(scan.num_components as u8);

        for i in 0..scan.num_components as usize {
            let comp_idx = scan.component_indices[i] as usize;
            self.out.push(jpeg.components[comp_idx].id as u8);
            let td_ta = ((scan.dc_tbl_idx[i] as u8) << 4) | (scan.ac_tbl_idx[i] as u8);
            self.out.push(td_ta);
        }

        self.out.push(scan.ss as u8);
        self.out.push(scan.se as u8);
        let ah_al = ((scan.ah as u8) << 4) | (scan.al as u8);
        self.out.push(ah_al);

        // Huffman encode coefficients
        self.write_scan_data(jpeg, scan_idx)?;

        Ok(())
    }

    fn write_scan_data(&mut self, jpeg: &JpegData, scan_idx: usize) -> Result<()> {
        let scan = &jpeg.scan_info[scan_idx];

        // Build Huffman encode tables for each table used in this scan
        let mut dc_tables: [Option<HuffmanEncodeTable>; 4] = [None, None, None, None];
        let mut ac_tables: [Option<HuffmanEncodeTable>; 4] = [None, None, None, None];
        for hc in &jpeg.huffman_code {
            let table = HuffmanEncodeTable::from_counts_values(&hc.counts, &hc.values);
            if hc.is_ac {
                ac_tables[hc.id as usize] = Some(table);
            } else {
                dc_tables[hc.id as usize] = Some(table);
            }
        }

        let mut bw = BitWriter::new();
        let mut padding_bit_idx = 0usize;
        let mut reset_point_idx = 0usize;
        let mut extra_zero_idx = 0usize;

        // Track DC predictions (one per component)
        let mut dc_pred = vec![0i32; jpeg.components.len()];

        // For baseline sequential JPEG (ss=0, se=63, ah=0, al=0):
        // interleaved components, MCU-based ordering
        let is_interleaved = scan.num_components > 1;

        // Calculate MCU dimensions
        let (mcu_rows, mcu_cols) = if is_interleaved {
            let max_h: u32 = jpeg.components.iter().map(|c| c.h_samp_factor).max().unwrap_or(1);
            let max_v: u32 = jpeg.components.iter().map(|c| c.v_samp_factor).max().unwrap_or(1);
            let mcu_cols = (jpeg.width + max_h * 8 - 1) / (max_h * 8);
            let mcu_rows = (jpeg.height + max_v * 8 - 1) / (max_v * 8);
            (mcu_rows, mcu_cols)
        } else {
            let comp_idx = scan.component_indices[0] as usize;
            let comp = &jpeg.components[comp_idx];
            (comp.height_in_blocks, comp.width_in_blocks)
        };

        let mut block_count: u32 = 0;

        for mcu_row in 0..mcu_rows {
            for mcu_col in 0..mcu_cols {
                // Check for reset point (RST marker)
                if reset_point_idx < scan.reset_points.len()
                    && block_count == scan.reset_points[reset_point_idx]
                {
                    // Flush bits, emit RST marker
                    bw.pad_to_byte(&jpeg.padding_bits, &mut padding_bit_idx);
                    self.out.extend_from_slice(&bw.finish());
                    bw = BitWriter::new();

                    let rst_marker = 0xD0 + ((reset_point_idx % 8) as u8);
                    self.out.push(0xFF);
                    self.out.push(rst_marker);

                    // Reset DC prediction
                    for dc in &mut dc_pred {
                        *dc = 0;
                    }
                    reset_point_idx += 1;
                }

                for sci in 0..scan.num_components as usize {
                    let comp_idx = scan.component_indices[sci] as usize;
                    let comp = &jpeg.components[comp_idx];
                    let dc_table = dc_tables[scan.dc_tbl_idx[sci] as usize]
                        .as_ref()
                        .ok_or_else(|| Error::InvalidJbrd("missing DC table".into()))?;
                    let ac_table = ac_tables[scan.ac_tbl_idx[sci] as usize]
                        .as_ref()
                        .ok_or_else(|| Error::InvalidJbrd("missing AC table".into()))?;

                    // How many blocks per MCU for this component
                    let (h_blocks, v_blocks) = if is_interleaved {
                        (comp.h_samp_factor, comp.v_samp_factor)
                    } else {
                        (1, 1)
                    };

                    for v in 0..v_blocks {
                        for h in 0..h_blocks {
                            let (by, bx) = if is_interleaved {
                                (
                                    mcu_row * comp.v_samp_factor + v,
                                    mcu_col * comp.h_samp_factor + h,
                                )
                            } else {
                                (mcu_row, mcu_col)
                            };

                            if by >= comp.height_in_blocks || bx >= comp.width_in_blocks {
                                // Padding block — encode as zero
                                encode_dc(&mut bw, 0, &mut dc_pred[comp_idx], dc_table);
                                encode_ac_eob(&mut bw, ac_table);
                            } else {
                                let block_offset =
                                    (by * comp.width_in_blocks + bx) as usize * 64;
                                let coeffs =
                                    &comp.coeffs[block_offset..block_offset + 64];

                                // Check for extra zero runs before this block
                                while extra_zero_idx < scan.extra_zero_runs.len()
                                    && scan.extra_zero_runs[extra_zero_idx].0
                                        == block_count
                                {
                                    let num_runs =
                                        scan.extra_zero_runs[extra_zero_idx].1;
                                    for _ in 0..num_runs {
                                        // Emit ZRL (15 zero run, zero amplitude)
                                        bw.write_huffman(ac_table, 0xF0);
                                    }
                                    extra_zero_idx += 1;
                                }

                                encode_dc(
                                    &mut bw,
                                    coeffs[0] as i32,
                                    &mut dc_pred[comp_idx],
                                    dc_table,
                                );
                                encode_ac(&mut bw, &coeffs[1..], ac_table);
                            }

                            block_count += 1;
                        }
                    }
                }
            }
        }

        // Flush remaining bits
        bw.pad_to_byte(&jpeg.padding_bits, &mut padding_bit_idx);
        self.out.extend_from_slice(&bw.finish());

        Ok(())
    }
}

/// Encode a DC coefficient using DPCM + Huffman.
fn encode_dc(bw: &mut BitWriter, dc: i32, dc_pred: &mut i32, table: &HuffmanEncodeTable) {
    let diff = dc - *dc_pred;
    *dc_pred = dc;

    let (category, extra_bits, extra_len) = categorize(diff);
    bw.write_huffman(table, category as u8);
    if extra_len > 0 {
        bw.write_bits(extra_bits as u32, extra_len);
    }
}

/// Encode AC coefficients (positions 1-63) using run-length + Huffman.
fn encode_ac(bw: &mut BitWriter, coeffs: &[i16], table: &HuffmanEncodeTable) {
    let mut zero_run = 0u32;
    let mut last_nonzero = 62; // index in coeffs (which is 0-based for positions 1-63)
    // Find last nonzero
    while last_nonzero > 0 && coeffs[last_nonzero] == 0 {
        last_nonzero -= 1;
    }

    if coeffs[0] == 0 && last_nonzero == 0 {
        // All zeros — emit EOB
        bw.write_huffman(table, 0x00);
        return;
    }

    for i in 0..=last_nonzero {
        if coeffs[i] == 0 {
            zero_run += 1;
            continue;
        }
        // Emit ZRL for runs > 15
        while zero_run > 15 {
            bw.write_huffman(table, 0xF0); // ZRL
            zero_run -= 16;
        }
        let (category, extra_bits, extra_len) = categorize(coeffs[i] as i32);
        let symbol = ((zero_run as u8) << 4) | (category as u8);
        bw.write_huffman(table, symbol);
        if extra_len > 0 {
            bw.write_bits(extra_bits as u32, extra_len);
        }
        zero_run = 0;
    }

    // EOB if not at position 63
    if last_nonzero < 62 {
        bw.write_huffman(table, 0x00);
    }
}

/// Encode AC EOB (for padding blocks).
fn encode_ac_eob(bw: &mut BitWriter, table: &HuffmanEncodeTable) {
    bw.write_huffman(table, 0x00);
}

/// Categorize a coefficient value for Huffman encoding.
/// Returns (category, extra_bits, extra_bit_length).
fn categorize(value: i32) -> (u32, u32, u32) {
    if value == 0 {
        return (0, 0, 0);
    }
    let abs_val = value.unsigned_abs();
    let category = 32 - abs_val.leading_zeros(); // = ceil(log2(abs+1))
    // For positive values: extra_bits = value
    // For negative values: extra_bits = value + (1 << category) - 1
    let extra_bits = if value > 0 {
        value as u32
    } else {
        (value + (1 << category) - 1) as u32
    };
    (category, extra_bits, category)
}

/// Huffman encode table: symbol → (code, length).
struct HuffmanEncodeTable {
    codes: [u32; 256],
    lengths: [u8; 256],
}

impl HuffmanEncodeTable {
    fn from_counts_values(counts: &[u32; 16], values: &[u8]) -> Self {
        let mut codes = [0u32; 256];
        let mut lengths = [0u8; 256];

        // Generate Huffman codes from counts (JPEG standard algorithm)
        let mut code: u32 = 0;
        let mut val_idx = 0;
        for (bits_minus_1, &count) in counts.iter().enumerate() {
            let bits = bits_minus_1 as u8 + 1;
            for _ in 0..count {
                if val_idx < values.len() {
                    let symbol = values[val_idx] as usize;
                    codes[symbol] = code;
                    lengths[symbol] = bits;
                    val_idx += 1;
                }
                code += 1;
            }
            code <<= 1;
        }

        Self { codes, lengths }
    }
}

/// Bitstream writer for JPEG entropy-coded data.
struct BitWriter {
    buffer: Vec<u8>,
    bit_buffer: u32,
    bits_in_buffer: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    fn write_huffman(&mut self, table: &HuffmanEncodeTable, symbol: u8) {
        let code = table.codes[symbol as usize];
        let length = table.lengths[symbol as usize];
        if length > 0 {
            self.write_bits(code, length as u32);
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: u32) {
        // JPEG uses MSB-first bit packing
        // We accumulate bits MSB-first in bit_buffer
        self.bit_buffer = (self.bit_buffer << num_bits) | (value & ((1 << num_bits) - 1));
        self.bits_in_buffer += num_bits;

        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = ((self.bit_buffer >> self.bits_in_buffer) & 0xFF) as u8;
            self.buffer.push(byte);
            if byte == 0xFF {
                self.buffer.push(0x00); // byte stuffing
            }
        }
    }

    fn pad_to_byte(&mut self, padding_bits: &[u8], padding_idx: &mut usize) {
        while self.bits_in_buffer % 8 != 0 {
            let bit = if *padding_idx < padding_bits.len() {
                let b = padding_bits[*padding_idx];
                *padding_idx += 1;
                b
            } else {
                1 // Default pad with 1s (standard JPEG)
            };
            self.write_bits(bit as u32, 1);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Any remaining bits should have been padded already
        debug_assert!(self.bits_in_buffer == 0);
        std::mem::take(&mut self.buffer)
    }
}
