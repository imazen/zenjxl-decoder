// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use smallvec::SmallVec;

use whereat::at;

use crate::{
    error::{Error, Result},
    features::{
        blending::perform_blending,
        patches::{PatchBlendMode, PatchBlending},
    },
    frame::ReferenceFrame,
    headers::{FileHeader, extra_channels::ExtraChannelInfo, frame_header::*},
    render::RenderPipelineInPlaceStage,
};

pub struct BlendingStage {
    pub frame_origin: (isize, isize),
    pub image_size: (isize, isize),
    pub blending_info: BlendingInfo,
    pub ec_blending_info: Vec<BlendingInfo>,
    /// `ec_blending_info` converted once (it used to be rebuilt per row chunk).
    ec_patch_blending: Vec<PatchBlending>,
    pub extra_channels: Vec<ExtraChannelInfo>,
    pub reference_frames: Arc<[Option<ReferenceFrame>; 4]>,
    pub zeros: Vec<f32>,
}

/// Per-render-context scratch for the stage: the blending kernels snapshot
/// the previous extra-channel values here instead of allocating per row.
struct BlendingScratch {
    tmp: Vec<f32>,
}

impl From<&BlendingInfo> for PatchBlending {
    fn from(info: &BlendingInfo) -> Self {
        let mode = match info.mode {
            BlendingMode::Replace => PatchBlendMode::None,
            BlendingMode::Add => PatchBlendMode::Add,
            BlendingMode::Mul => PatchBlendMode::Mul,
            BlendingMode::Blend => PatchBlendMode::BlendBelow,
            BlendingMode::AlphaWeightedAdd => PatchBlendMode::AlphaWeightedAddBelow,
        };
        PatchBlending {
            mode,
            alpha_channel: info.alpha_channel as usize,
            clamp: info.clamp,
        }
    }
}

impl BlendingStage {
    pub fn new(
        frame_header: &FrameHeader,
        file_header: &FileHeader,
        reference_frames: Arc<[Option<ReferenceFrame>; 4]>,
    ) -> Result<BlendingStage> {
        let image_size = (
            file_header.size.xsize() as usize,
            file_header.size.ysize() as usize,
        );
        for (index, info) in std::iter::once(&frame_header.blending_info)
            .chain(frame_header.ec_blending_info.iter())
            .enumerate()
        {
            if index == 0
                && info.mode == BlendingMode::Replace
                && frame_header.x0 <= 0
                && frame_header.y0 <= 0
                && frame_header.x0 as i64 + frame_header.width as i64 >= image_size.0 as i64
                && frame_header.y0 as i64 + frame_header.height as i64 >= image_size.1 as i64
            {
                // Full-canvas Replace does not read color from a reference.
                continue;
            }
            let source = info.source as usize;
            let Some(frame) = reference_frames[source].as_ref() else {
                continue;
            };
            if index == 0 && frame.saved_before_color_transform {
                return Err(at!(Error::BlendingPreColorTransform(source)));
            }
            // EC samples do not undergo the color transform. A before-CT
            // reference may supply them, provided it covers the canvas at
            // origin zero. Check every plane we can read instead of asserting
            // only the color-plane size on untrusted input.
            let channels = if index == 0 {
                0..3
            } else {
                index + 2..index + 3
            };
            if frame.origin != (0, 0)
                || channels.into_iter().any(|channel| {
                    frame.frame.get(channel).is_none_or(|image| {
                        image.size().0 < image_size.0 || image.size().1 < image_size.1
                    })
                })
            {
                return Err(at!(Error::NonPatchReferenceWithCrop));
            }
        }

        let ec_patch_blending = frame_header
            .ec_blending_info
            .iter()
            .map(PatchBlending::from)
            .collect();
        Ok(BlendingStage {
            frame_origin: (frame_header.x0 as isize, frame_header.y0 as isize),
            image_size: (image_size.0 as isize, image_size.1 as isize),
            blending_info: frame_header.blending_info.clone(),
            ec_blending_info: frame_header.ec_blending_info.clone(),
            ec_patch_blending,
            extra_channels: file_header.image_metadata.extra_channel_info.clone(),
            reference_frames,
            zeros: vec![0f32; image_size.0],
        })
    }
}

impl std::fmt::Display for BlendingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "blending")
    }
}

impl RenderPipelineInPlaceStage for BlendingStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3 + self.extra_channels.len()
    }

    fn init_local_state(
        &self,
        _thread_index: usize,
    ) -> Result<Option<Box<dyn std::any::Any + Send>>> {
        Ok(Some(Box::new(BlendingScratch { tmp: Vec::new() })))
    }

    fn process_row_chunk(
        &self,
        position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        state: Option<&mut (dyn std::any::Any + Send)>,
    ) {
        let num_ec = self.extra_channels.len();
        let fg_y0 = self.frame_origin.1 + position.1 as isize;
        let mut fg_x0 = self.frame_origin.0 + position.0 as isize;
        let mut fg_x1 = fg_x0 + xsize as isize;
        let mut bg_x0: isize = 0;
        let mut bg_x1: isize = xsize as isize;

        if fg_x1 <= 0 || fg_x0 >= self.image_size.0 || fg_y0 < 0 || fg_y0 >= self.image_size.1 {
            return;
        }

        if fg_x0 < 0 {
            bg_x0 -= fg_x0;
            fg_x0 = 0;
        }
        if fg_x1 > self.image_size.0 {
            bg_x1 = bg_x0 + self.image_size.0 - fg_x0;
            fg_x1 = self.image_size.0;
        }

        let fg_x0: usize = fg_x0 as usize;
        let fg_x1: usize = fg_x1 as usize;
        let bg_x0: usize = bg_x0 as usize;
        let bg_x1: usize = bg_x1 as usize;
        let fg_y0: usize = fg_y0 as usize;

        let mut out: SmallVec<[&mut [f32]; 8]> = row.iter_mut().map(|s| &mut s[..xsize]).collect();

        let mut fg: SmallVec<[&[f32]; 8]> = smallvec::smallvec![self.zeros.as_slice(); 3 + num_ec];

        for (c, fg_ptr) in fg.iter_mut().enumerate().take(3) {
            if self.blending_info.mode != BlendingMode::Replace
                && self.reference_frames[self.blending_info.source as usize].is_some()
            {
                *fg_ptr = &(self.reference_frames[self.blending_info.source as usize]
                    .as_ref()
                    .unwrap()
                    .frame[c]
                    .row(fg_y0)[fg_x0..fg_x1]);
            }
        }
        for i in 0..num_ec {
            if self.reference_frames[self.ec_blending_info[i].source as usize].is_some() {
                fg[3 + i] = &(self.reference_frames[self.ec_blending_info[i].source as usize]
                    .as_ref()
                    .unwrap()
                    .frame[3 + i]
                    .row(fg_y0)[fg_x0..fg_x1]);
            }
        }

        let blending_info = PatchBlending::from(&self.blending_info);

        // The scratch lives in the per-context state (allocated once per
        // render context); a missing state only happens in unit tests that
        // drive the stage directly, so fall back to a local buffer there.
        let mut local_tmp = Vec::new();
        let tmp = match state {
            Some(state) => &mut state.downcast_mut::<BlendingScratch>().unwrap().tmp,
            None => &mut local_tmp,
        };
        let mut bg: SmallVec<[&mut [f32]; 8]> =
            out.iter_mut().map(|s| &mut s[bg_x0..bg_x1]).collect();
        perform_blending(
            &mut bg,
            &fg,
            &blending_info,
            &self.ec_patch_blending,
            &self.extra_channels,
            tmp,
        );
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::util::test::read_headers_and_toc;

    #[test]
    fn blending_consistency() -> Result<()> {
        let (file_header, frame_header, _) =
            read_headers_and_toc(&crate::util::test::fixture_bytes("basic.jxl")).unwrap();
        let image_size = (
            file_header.size.xsize() as usize,
            file_header.size.ysize() as usize,
        );
        let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(0);
        let mut reference_frame =
            || ReferenceFrame::random(&mut rng, image_size.0, image_size.1, 4, false);
        let reference_frames = Arc::new([
            Some(reference_frame()?),
            Some(reference_frame()?),
            Some(reference_frame()?),
            Some(reference_frame()?),
        ]);
        crate::render::test::test_stage_consistency(
            || BlendingStage::new(&frame_header, &file_header, reference_frames.clone()).unwrap(),
            (500, 500),
            4,
        )
    }

    #[test]
    fn reject_blending_with_pre_color_transform_reference() -> Result<()> {
        let (file_header, mut frame_header, _) =
            read_headers_and_toc(&crate::util::test::fixture_bytes("basic.jxl")).unwrap();
        frame_header.blending_info.source = 3;
        frame_header.blending_info.mode = BlendingMode::Add;
        let image_size = (
            file_header.size.xsize() as usize,
            file_header.size.ysize() as usize,
        );
        let reference_frames = Arc::new([
            None,
            None,
            None,
            Some(ReferenceFrame::blank(image_size.0, image_size.1, 4, true)?),
        ]);

        let result = BlendingStage::new(&frame_header, &file_header, reference_frames);
        assert!(
            matches!(&result, Err(e) if matches!(e.error(), Error::BlendingPreColorTransform(3))),
            "expected BlendingPreColorTransform(3)"
        );
        Ok(())
    }

    #[test]
    fn extra_channel_before_ct_reference_geometry_is_checked() -> Result<()> {
        use crate::headers::{bit_depth::BitDepth, extra_channels::ExtraChannel};
        let (mut file_header, mut frame_header, _) =
            read_headers_and_toc(&crate::util::test::fixture_bytes("basic.jxl")).unwrap();
        file_header
            .image_metadata
            .extra_channel_info
            .push(ExtraChannelInfo::new(
                false,
                ExtraChannel::Alpha,
                BitDepth::integer_samples(8),
                0,
                String::new(),
                false,
                None,
                None,
            ));
        let mut ec_blending_info = frame_header.blending_info.clone();
        ec_blending_info.mode = BlendingMode::Add;
        ec_blending_info.source = 2;
        frame_header.ec_blending_info = vec![ec_blending_info];
        let image_size = (
            file_header.size.xsize() as usize,
            file_header.size.ysize() as usize,
        );
        for (size, origin, channels, allowed) in [
            (image_size, (0, 0), 4, true),
            ((image_size.0 + 1, image_size.1 + 1), (0, 0), 4, true),
            ((image_size.0 - 1, image_size.1), (0, 0), 4, false),
            (image_size, (1, 0), 4, false),
            (image_size, (0, -1), 4, false),
            (image_size, (0, 0), 3, false),
        ] {
            let mut reference = ReferenceFrame::blank(size.0, size.1, channels, true)?;
            reference.origin = origin;
            let result = BlendingStage::new(
                &frame_header,
                &file_header,
                Arc::new([None, None, Some(reference), None]),
            );
            if allowed {
                assert!(result.is_ok());
            } else {
                assert!(
                    matches!(&result, Err(e) if matches!(e.error(), Error::NonPatchReferenceWithCrop))
                );
            }
        }
        Ok(())
    }
}
