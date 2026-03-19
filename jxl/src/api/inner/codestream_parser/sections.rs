// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{JxlDecoderOptions, JxlOutputBuffer},
    bit_reader::BitReader,
    error::Result,
    frame::Section,
};

use super::CodestreamParser;

pub(super) struct SectionState {
    lf_global_done: bool,
    remaining_lf: usize,
    hf_global_done: bool,
    completed_passes: Vec<u8>,
}

impl SectionState {
    pub(super) fn new(num_lf_groups: usize, num_groups: usize) -> Self {
        Self {
            lf_global_done: false,
            remaining_lf: num_lf_groups,
            hf_global_done: false,
            completed_passes: vec![0; num_groups],
        }
    }

    /// Returns the number of passes that are fully completed across all groups.
    /// A pass is fully completed when all groups have decoded that pass.
    pub(super) fn num_completed_passes(&self) -> usize {
        self.completed_passes.iter().copied().min().unwrap_or(0) as usize
    }
}

impl CodestreamParser {
    pub(super) fn process_sections(
        &mut self,
        decode_options: &JxlDecoderOptions,
        output_buffers: &mut Option<&mut [JxlOutputBuffer<'_>]>,
        do_flush: bool,
    ) -> Result<Option<usize>> {
        let frame = self.frame.as_mut().unwrap();

        let output_profile = self
            .output_color_profile
            .as_ref()
            .expect("output_color_profile should be set before pipeline preparation");

        if do_flush && let Some(buf) = output_buffers {
            frame.maybe_preview_lf_frame(
                self.pixel_format.as_ref().unwrap(),
                buf,
                None,
                output_profile,
            )?;
        }

        let frame_header = frame.header();

        // Dequeue ready sections.
        while self
            .sections
            .front()
            .is_some_and(|s| s.len <= self.ready_section_data)
        {
            let s = self.sections.pop_front().unwrap();
            self.ready_section_data -= s.len;

            match s.section {
                Section::LfGlobal => {
                    self.lf_global_section = Some(s);
                }
                Section::HfGlobal => {
                    self.hf_global_section = Some(s);
                }
                Section::Lf { .. } => {
                    self.lf_sections.push(s);
                }
                Section::Hf { group, pass } => {
                    self.hf_sections[group][pass] = Some(s);
                    self.candidate_hf_sections.insert(group);
                }
            }
        }

        let mut processed_section = false;
        let pixel_format = self.pixel_format.as_ref().unwrap();
        'process: {
            if frame_header.num_groups() == 1 && frame_header.passes.num_passes == 1 {
                // Single-group special case.
                let Some(sec) = self.lf_global_section.take() else {
                    break 'process;
                };
                assert!(self.sections.is_empty());
                let mut br = BitReader::new(&sec.data);
                frame.decode_lf_global(&mut br)?;
                frame.decode_lf_group(0, &mut br)?;
                frame.decode_hf_global(&mut br)?;
                frame.prepare_render_pipeline(
                    self.pixel_format.as_ref().unwrap(),
                    decode_options.cms.as_deref(),
                    self.embedded_color_profile
                        .as_ref()
                        .expect("embedded_color_profile should be set before pipeline preparation"),
                    output_profile,
                )?;
                frame.finalize_lf()?;
                frame.decode_and_render_hf_groups(
                    output_buffers,
                    pixel_format,
                    vec![(0, vec![(0, br)])],
                    do_flush,
                    output_profile,
                )?;
                processed_section = true;
            } else {
                let section_timing = std::env::var("JXL_PHASE_TIMING").is_ok();
                let t0 = std::time::Instant::now();

                if let Some(lf_global) = self.lf_global_section.take() {
                    frame.decode_lf_global(&mut BitReader::new(&lf_global.data))?;
                    self.section_state.lf_global_done = true;
                    processed_section = true;
                }

                if !self.section_state.lf_global_done {
                    break 'process;
                }

                let lf_global_dur = t0.elapsed();

                #[cfg(feature = "threads")]
                let use_parallel_lf = frame.decoder_state.parallel && self.lf_sections.len() > 1;
                #[cfg(not(feature = "threads"))]
                let use_parallel_lf = false;

                // When VarDCT parallel and HF global section is ready, overlap LF group
                // decode with HF global parsing. decode_hf_global only depends on lf_global
                // (the LF Global section), not on LF group data — so both can run concurrently.
                #[cfg(feature = "threads")]
                let use_overlap = use_parallel_lf
                    && frame.header().encoding != crate::headers::frame_header::Encoding::Modular
                    && self.hf_global_section.is_some();

                #[cfg(not(feature = "threads"))]
                let use_overlap = false;

                if use_overlap {
                    #[cfg(feature = "threads")]
                    {
                        let lf_sections: Vec<_> = self.lf_sections.drain(..).collect();
                        let count = lf_sections.len();
                        let sections: Vec<(usize, Vec<u8>)> = lf_sections
                            .into_iter()
                            .map(|s| {
                                let Section::Lf { group } = s.section else {
                                    unreachable!()
                                };
                                (group, s.data)
                            })
                            .collect();
                        let hf_data = self.hf_global_section.take().unwrap().data;
                        frame.decode_lf_and_hf_global_parallel(sections, hf_data)?;
                        self.section_state.remaining_lf -= count;
                        self.section_state.hf_global_done = true;
                        processed_section = true;
                    }
                } else if use_parallel_lf {
                    #[cfg(feature = "threads")]
                    {
                        let lf_sections: Vec<_> = self.lf_sections.drain(..).collect();
                        let count = lf_sections.len();
                        let sections: Vec<(usize, Vec<u8>)> = lf_sections
                            .into_iter()
                            .map(|s| {
                                let Section::Lf { group } = s.section else {
                                    unreachable!()
                                };
                                (group, s.data)
                            })
                            .collect();
                        if frame.header().encoding
                            == crate::headers::frame_header::Encoding::Modular
                        {
                            frame.decode_lf_groups_modular_parallel(sections)?;
                        } else {
                            frame.decode_lf_groups_vardct_parallel(sections)?;
                        }
                        self.section_state.remaining_lf -= count;
                        processed_section = true;
                    }
                } else {
                    for lf_section in self.lf_sections.drain(..) {
                        let Section::Lf { group } = lf_section.section else {
                            unreachable!()
                        };
                        frame.decode_lf_group(group, &mut BitReader::new(&lf_section.data))?;
                        processed_section = true;
                        self.section_state.remaining_lf -= 1;
                    }
                }

                if self.section_state.remaining_lf != 0 {
                    break 'process;
                }

                let lf_groups_dur = t0.elapsed().saturating_sub(lf_global_dur);

                let mut decode_hf_dur = std::time::Duration::ZERO;
                let mut pipeline_dur = std::time::Duration::ZERO;
                let mut finalize_lf_dur = std::time::Duration::ZERO;
                // If HF global was already decoded in the overlap path, skip decoding.
                // Otherwise decode it now (sequential path or no overlap).
                let hf_newly_done = if self.section_state.hf_global_done {
                    // Already decoded via overlap path
                    true
                } else if let Some(hf_global) = self.hf_global_section.take() {
                    let t = std::time::Instant::now();
                    frame.decode_hf_global(&mut BitReader::new(&hf_global.data))?;
                    decode_hf_dur = t.elapsed();
                    self.section_state.hf_global_done = true;
                    processed_section = true;
                    true
                } else {
                    false
                };
                if hf_newly_done && frame.render_pipeline_not_ready() {
                    let t = std::time::Instant::now();

                    #[cfg(feature = "threads")]
                    {
                        frame.prepare_render_pipeline(
                            self.pixel_format.as_ref().unwrap(),
                            decode_options.cms.as_deref(),
                            self.embedded_color_profile.as_ref().expect(
                                "embedded_color_profile should be set before pipeline preparation",
                            ),
                            self.output_color_profile.as_ref().expect(
                                "output_color_profile should be set before pipeline preparation",
                            ),
                        )?;
                        pipeline_dur = t.elapsed();

                        let t = std::time::Instant::now();
                        frame.finalize_lf()?;
                        finalize_lf_dur = t.elapsed();
                    }

                    #[cfg(not(feature = "threads"))]
                    {
                        frame.prepare_render_pipeline(
                            self.pixel_format.as_ref().unwrap(),
                            decode_options.cms.as_deref(),
                            self.embedded_color_profile.as_ref().expect(
                                "embedded_color_profile should be set before pipeline preparation",
                            ),
                            self.output_color_profile.as_ref().expect(
                                "output_color_profile should be set before pipeline preparation",
                            ),
                        )?;
                        pipeline_dur = t.elapsed();

                        let t = std::time::Instant::now();
                        frame.finalize_lf()?;
                        finalize_lf_dur = t.elapsed();
                    }
                }
                if !self.section_state.hf_global_done {
                    break 'process;
                }

                let prep_start = std::time::Instant::now();
                let mut group_readers = vec![];
                let mut processed_groups = vec![];

                let mut check_group = |g: usize| {
                    let mut sections = vec![];
                    for (pass, grp) in self.hf_sections[g]
                        .iter()
                        .enumerate()
                        .skip(self.section_state.completed_passes[g] as usize)
                    {
                        let Some(s) = &grp else {
                            break;
                        };
                        self.section_state.completed_passes[g] += 1;
                        sections.push((pass, BitReader::new(&s.data)));
                    }
                    if !sections.is_empty() {
                        group_readers.push((g, sections));
                        processed_groups.push(g);
                    }
                };

                if self.candidate_hf_sections.len() * 4 < self.hf_sections.len() {
                    for g in self.candidate_hf_sections.drain() {
                        check_group(g)
                    }
                    // Processing sections in order is more efficient because it lets us flush
                    // the pipeline faster.
                    group_readers.sort_by_key(|x| x.0);
                } else {
                    for g in 0..self.hf_sections.len() {
                        if self.candidate_hf_sections.contains(&g) {
                            check_group(g);
                        }
                    }
                    self.candidate_hf_sections.clear();
                }
                let prep_dur = prep_start.elapsed();

                if section_timing {
                    eprintln!(
                        "[JXL_SECTION_TIMING] lf_global: {:.2}ms | lf_groups: {:.2}ms | \
                         decode_hf: {:.2}ms | pipeline: {:.2}ms | finalize_lf: {:.2}ms | \
                         hf_prep: {:.2}ms | total: {:.2}ms",
                        lf_global_dur.as_secs_f64() * 1000.0,
                        lf_groups_dur.as_secs_f64() * 1000.0,
                        decode_hf_dur.as_secs_f64() * 1000.0,
                        pipeline_dur.as_secs_f64() * 1000.0,
                        finalize_lf_dur.as_secs_f64() * 1000.0,
                        prep_dur.as_secs_f64() * 1000.0,
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }

                frame.decode_and_render_hf_groups(
                    output_buffers,
                    pixel_format,
                    group_readers,
                    do_flush,
                    output_profile,
                )?;

                for g in processed_groups.into_iter() {
                    for i in 0..self.section_state.completed_passes[g] {
                        self.hf_sections[g][i as usize] = None;
                    }
                    processed_section = true;
                }
            }
        }

        if !processed_section {
            let data_for_next_section =
                self.sections.front().unwrap().len - self.ready_section_data;
            return Ok(Some(data_for_next_section));
        }

        // Frame is not yet complete.
        if !self.sections.is_empty() {
            return Ok(None);
        }

        #[cfg(test)]
        {
            self.frame_callback.as_mut().map_or(Ok(()), |cb| {
                cb(self.frame.as_ref().unwrap(), self.decoded_frames)
            })?;
            self.decoded_frames += 1;
        }

        // Check if this might be a preview frame (skipped frame with preview enabled)
        let has_preview = self
            .basic_info
            .as_ref()
            .is_some_and(|info| info.preview_size.is_some());
        let might_be_preview = self.process_without_output && has_preview;

        // Reconstruct JPEG if we have JBRD data (before frame is consumed by finalize)
        #[cfg(feature = "jpeg")]
        if let Some(jbrd_data) = &self.jbrd_data
            && let Some(frame) = &self.frame
            && let Ok(bytes) = frame.jpeg_reconstruct(jbrd_data)
        {
            // Reconstruction failure is non-fatal; normal decode continues
            self.jpeg_bytes = Some(bytes);
        }

        let decoder_state = self.frame.take().unwrap().finalize()?;
        if let Some(state) = decoder_state {
            self.decoder_state = Some(state);
        } else if might_be_preview {
            // Preview frame has is_last=true but the main frame follows.
            // Recreate decoder state from saved file header for the main frame.
            if let Some(fh) = self.saved_file_header.take() {
                let mut new_state = crate::frame::DecoderState::new(fh);
                new_state.render_spotcolors = decode_options.render_spot_colors;
                new_state.desired_intensity_target = decode_options.desired_intensity_target;
                new_state.limits = decode_options.limits.clone();
                new_state.stop = std::sync::Arc::clone(&decode_options.stop);
                self.decoder_state = Some(new_state);
            }
        } else {
            self.has_more_frames = false;
        }
        Ok(None)
    }
}
