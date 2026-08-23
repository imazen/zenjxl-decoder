// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlCms;

use std::sync::Arc;

/// Security limits for the JXL decoder to prevent resource exhaustion attacks.
///
/// These limits protect against "JXL bombs" - maliciously crafted files designed
/// to exhaust memory or CPU. All limits are optional; `None` means use the default.
///
/// # Example
/// ```
/// use zenjxl_decoder::api::JxlDecoderLimits;
///
/// // Use restrictive preset for untrusted input
/// let limits = JxlDecoderLimits::restrictive();
///
/// // Or use defaults for normal operation
/// let defaults = JxlDecoderLimits::default();
///
/// // Or unlimited for trusted input (use with caution)
/// let unlimited = JxlDecoderLimits::unlimited();
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct JxlDecoderLimits {
    /// Maximum total pixels (width × height). Default: 1 << 28 (~256 megapixels).
    /// This is checked early during header parsing.
    /// [`JxlDecoderLimits::restrictive`] lowers this to a 120-megapixel house cap.
    pub max_pixels: Option<usize>,

    /// Maximum number of extra channels (alpha, depth, etc.). Default: 256.
    /// Each extra channel requires memory proportional to image size.
    pub max_extra_channels: Option<usize>,

    /// Maximum ICC profile size in bytes. Default: 2^28 (256 MB).
    pub max_icc_size: Option<usize>,

    /// Maximum modular tree size (number of nodes). Default: 2^22.
    /// Limits memory and CPU for tree-based entropy coding.
    pub max_tree_size: Option<usize>,

    /// Maximum number of patches. Default: derived from image size.
    /// Set to limit patch-based attacks.
    pub max_patches: Option<usize>,

    /// Maximum number of spline control points. Default: 2^20.
    pub max_spline_points: Option<u32>,

    /// Maximum number of reference frames stored. Default: 4.
    /// Each reference frame consumes memory equal to the image size.
    pub max_reference_frames: Option<usize>,

    /// Maximum total memory budget in bytes. Default: None (unlimited).
    /// When set, the decoder tracks allocations and fails if budget exceeded.
    /// This provides defense-in-depth against memory exhaustion attacks.
    pub max_memory_bytes: Option<u64>,
}

impl Default for JxlDecoderLimits {
    fn default() -> Self {
        // On 32-bit targets, a 4 GB memory budget exceeds the usable address
        // space (~3 GB). Cap at 2 GB so the guard triggers before the global
        // allocator aborts on OOM.
        let max_memory = if cfg!(target_pointer_width = "32") {
            2u64 << 30 // 2 GB
        } else {
            4u64 << 30 // 4 GB
        };
        Self {
            max_pixels: Some(1 << 28),        // ~256 megapixels
            max_extra_channels: Some(256),    // 256 extra channels
            max_icc_size: Some(1 << 28),      // 256 MB
            max_tree_size: Some(1 << 22),     // 4M nodes
            max_patches: None,                // Use image-size-based default
            max_spline_points: Some(1 << 20), // 1M points
            max_reference_frames: Some(4),    // 4 reference frames
            max_memory_bytes: Some(max_memory),
        }
    }
}

impl JxlDecoderLimits {
    /// Returns limits with no restrictions (all None).
    /// Use with caution - only for trusted input.
    pub fn unlimited() -> Self {
        Self {
            max_pixels: None,
            max_extra_channels: None,
            max_icc_size: None,
            max_tree_size: None,
            max_patches: None,
            max_spline_points: None,
            max_reference_frames: None,
            max_memory_bytes: None,
        }
    }

    /// Returns restrictive limits suitable for untrusted web content.
    pub fn restrictive() -> Self {
        Self {
            max_pixels: Some(120_000_000), // 120 megapixels (admits common ~108 MP camera photos)
            max_extra_channels: Some(16),  // 16 extra channels
            // 16 MB: real press CMYK profiles run 1.8-3.5 MB (GRACoL, SWOP, ISO
            // Coated), which the old 1 MB cap rejected. Same value upstream jxl-rs
            // settled on (#813).
            max_icc_size: Some(16 << 20),
            max_tree_size: Some(1 << 20),     // 1M nodes
            max_patches: Some(1 << 16),       // 64K patches
            max_spline_points: Some(1 << 16), // 64K points
            max_reference_frames: Some(2),    // 2 reference frames
            max_memory_bytes: Some(1 << 30),  // 1 GB total memory
        }
    }

    /// Set the maximum total pixel count (width × height).
    #[must_use]
    pub fn with_max_pixels(mut self, max: usize) -> Self {
        self.max_pixels = Some(max);
        self
    }

    /// Set the maximum number of extra channels (alpha, depth, etc.).
    #[must_use]
    pub fn with_max_extra_channels(mut self, max: usize) -> Self {
        self.max_extra_channels = Some(max);
        self
    }

    /// Set the maximum ICC profile size in bytes.
    #[must_use]
    pub fn with_max_icc_size(mut self, max: usize) -> Self {
        self.max_icc_size = Some(max);
        self
    }

    /// Set the maximum modular tree size (number of nodes).
    #[must_use]
    pub fn with_max_tree_size(mut self, max: usize) -> Self {
        self.max_tree_size = Some(max);
        self
    }

    /// Set the maximum number of patches.
    #[must_use]
    pub fn with_max_patches(mut self, max: usize) -> Self {
        self.max_patches = Some(max);
        self
    }

    /// Set the maximum number of spline control points.
    #[must_use]
    pub fn with_max_spline_points(mut self, max: u32) -> Self {
        self.max_spline_points = Some(max);
        self
    }

    /// Set the maximum number of reference frames.
    #[must_use]
    pub fn with_max_reference_frames(mut self, max: usize) -> Self {
        self.max_reference_frames = Some(max);
        self
    }

    /// Set the maximum total memory budget in bytes.
    #[must_use]
    pub fn with_max_memory_bytes(mut self, max: u64) -> Self {
        self.max_memory_bytes = Some(max);
        self
    }
}

pub enum JxlProgressiveMode {
    /// Renders all pixels in every call to Process.
    Eager,
    /// Renders pixels once passes are completed.
    Pass,
    /// Renders pixels only once the final frame is ready.
    FullFrame,
}

/// Decoder configuration.
///
/// This struct is `#[non_exhaustive]`, so downstream crates **cannot** build it
/// with a struct literal. Start from [`default`](Self::default) and chain the
/// `with_*` setters (or assign the public fields on a `mut` binding):
///
/// ```
/// use zenjxl_decoder::api::{JxlDecoderOptions, JxlDecoderLimits};
///
/// let options = JxlDecoderOptions::default()
///     .with_limits(JxlDecoderLimits::restrictive().with_max_pixels(120_000_000))
///     .with_reject_progressive(true);
///
/// assert!(options.reject_progressive);
/// assert_eq!(options.limits.max_pixels, Some(120_000_000));
/// ```
#[non_exhaustive]
pub struct JxlDecoderOptions {
    pub adjust_orientation: bool,
    /// Reject progressive content during frame decode.
    ///
    /// When `true`, decode fails as soon as a progressive frame header
    /// (multi-pass or LF frame) is seen — before decoding its passes — for
    /// untrusted-input policies that forbid progressive content. A frame counts
    /// as progressive when its header has `num_passes > 1` or its frame type is
    /// `LFFrame`; patch/blend dictionary frames (`ReferenceOnly`) and
    /// `SkipProgressive` frames do **not** trip the gate. The check is applied
    /// to the first non-preview frame.
    ///
    /// Default: `false` (progressive content is decoded normally).
    pub reject_progressive: bool,
    /// Apply blue-noise dithering when quantising to 8-bit output.
    ///
    /// libjxl dithers every `u8` output sample with a fixed 32x32 blue-noise
    /// pattern (`stage_write.cc`), which breaks the banding that plain
    /// rounding leaves in smooth gradients; jxl-rs does the same since 0.6.
    /// The pattern is indexed by absolute pixel position, so output is
    /// deterministic, identical for streamed and one-shot decodes, and
    /// matches `djxl` bit-for-bit wherever the float pipelines agree. Exact
    /// 8-bit values (lossless 8-bit content) are never changed; a lossy
    /// sample moves by at most one code relative to plain rounding. Only
    /// `U8` output is affected; `U16`/`F16`/`F32` are never dithered.
    ///
    /// Default: `true`. Set to `false` for plain round-to-nearest.
    pub dither_u8: bool,
    pub render_spot_colors: bool,
    pub coalescing: bool,
    pub desired_intensity_target: Option<f32>,
    pub skip_preview: bool,
    pub progressive_mode: JxlProgressiveMode,
    pub cms: Option<Box<dyn JxlCms>>,
    /// Use high precision mode for decoding.
    /// When false (default), uses lower precision settings that match libjxl's default.
    /// When true, uses higher precision at the cost of performance.
    ///
    /// This affects multiple decoder decisions including spline rendering precision
    /// and potentially intermediate buffer storage (e.g., using f32 vs f16).
    pub high_precision: bool,
    /// If true, multiply RGB by alpha before writing to output buffer.
    /// This produces premultiplied alpha output, which is useful for compositing.
    /// Default: false (output straight alpha)
    pub premultiply_output: bool,
    /// Security limits to prevent resource exhaustion attacks.
    /// Use `JxlDecoderLimits::restrictive()` for untrusted input.
    pub limits: JxlDecoderLimits,
    /// Cooperative cancellation / timeout handle.
    /// Default: `Arc::new(enough::Unstoppable)` (no cancellation).
    pub stop: Arc<dyn enough::Stop>,
    /// Enable parallel decoding and rendering using rayon.
    ///
    /// When `true` (the default when the `threads` feature is enabled),
    /// group decoding and rendering are parallelized across rayon's global
    /// thread pool. Control thread count via `RAYON_NUM_THREADS` or
    /// `rayon::ThreadPoolBuilder::build_global()`.
    ///
    /// When `false`, all decoding is single-threaded.
    pub parallel: bool,
}

impl Default for JxlDecoderOptions {
    fn default() -> Self {
        Self {
            adjust_orientation: true,
            reject_progressive: false,
            dither_u8: true,
            render_spot_colors: true,
            coalescing: true,
            skip_preview: true,
            desired_intensity_target: None,
            progressive_mode: JxlProgressiveMode::Pass,
            cms: None,
            high_precision: false,
            premultiply_output: false,
            limits: JxlDecoderLimits::default(),
            stop: Arc::new(enough::Unstoppable),
            parallel: cfg!(feature = "threads"),
        }
    }
}

impl JxlDecoderOptions {
    /// Apply EXIF orientation to the decoded image.
    #[must_use]
    pub fn with_adjust_orientation(mut self, v: bool) -> Self {
        self.adjust_orientation = v;
        self
    }

    /// Reject progressive content during frame decode (untrusted-input policy).
    #[must_use]
    pub fn with_reject_progressive(mut self, v: bool) -> Self {
        self.reject_progressive = v;
        self
    }

    /// Enable or disable blue-noise dithering of 8-bit output (see
    /// [`JxlDecoderOptions::dither_u8`]; default on).
    #[must_use]
    pub fn with_dither_u8(mut self, v: bool) -> Self {
        self.dither_u8 = v;
        self
    }

    /// Render spot colors.
    #[must_use]
    pub fn with_render_spot_colors(mut self, v: bool) -> Self {
        self.render_spot_colors = v;
        self
    }

    /// Coalesce animation frames onto the canvas.
    #[must_use]
    pub fn with_coalescing(mut self, v: bool) -> Self {
        self.coalescing = v;
        self
    }

    /// Set the desired display intensity target (nits).
    #[must_use]
    pub fn with_desired_intensity_target(mut self, nits: f32) -> Self {
        self.desired_intensity_target = Some(nits);
        self
    }

    /// Skip decoding the preview frame.
    #[must_use]
    pub fn with_skip_preview(mut self, v: bool) -> Self {
        self.skip_preview = v;
        self
    }

    /// Set the progressive rendering mode.
    #[must_use]
    pub fn with_progressive_mode(mut self, mode: JxlProgressiveMode) -> Self {
        self.progressive_mode = mode;
        self
    }

    /// Set the color-management system used for ICC conversions.
    #[must_use]
    pub fn with_cms(mut self, cms: Box<dyn JxlCms>) -> Self {
        self.cms = Some(cms);
        self
    }

    /// Use higher-precision decoding (slower).
    #[must_use]
    pub fn with_high_precision(mut self, v: bool) -> Self {
        self.high_precision = v;
        self
    }

    /// Emit premultiplied alpha in the output.
    #[must_use]
    pub fn with_premultiply_output(mut self, v: bool) -> Self {
        self.premultiply_output = v;
        self
    }

    /// Set the resource limits (see [`JxlDecoderLimits::restrictive`] for a
    /// safe untrusted-input preset).
    #[must_use]
    pub fn with_limits(mut self, limits: JxlDecoderLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the cooperative-cancellation token (an [`enough::Stop`]).
    #[must_use]
    pub fn with_stop(mut self, stop: Arc<dyn enough::Stop>) -> Self {
        self.stop = stop;
        self
    }

    /// Enable multi-threaded decoding (requires the `threads` feature).
    #[must_use]
    pub fn with_parallel(mut self, v: bool) -> Self {
        self.parallel = v;
        self
    }
}
