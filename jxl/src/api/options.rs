// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlCms;

/// Security limits for the JXL decoder to prevent resource exhaustion attacks.
///
/// These limits protect against "JXL bombs" - maliciously crafted files designed
/// to exhaust memory or CPU. All limits are optional; `None` means use the default.
///
/// # Example
/// ```
/// use jxl::api::JxlDecoderLimits;
///
/// // Restrictive limits for untrusted input
/// let limits = JxlDecoderLimits {
///     max_pixels: Some(100_000_000),      // 100 megapixels
///     max_extra_channels: Some(16),        // 16 extra channels
///     max_icc_size: Some(1 << 20),         // 1 MB ICC profile
///     ..JxlDecoderLimits::default()
/// };
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct JxlDecoderLimits {
    /// Maximum total pixels (width × height). Default: 2^30 (~1 billion).
    /// This is checked early during header parsing.
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
}

impl Default for JxlDecoderLimits {
    fn default() -> Self {
        Self {
            max_pixels: Some(1 << 30),           // ~1 billion pixels
            max_extra_channels: Some(256),       // 256 extra channels
            max_icc_size: Some(1 << 28),         // 256 MB
            max_tree_size: Some(1 << 22),        // 4M nodes
            max_patches: None,                   // Use image-size-based default
            max_spline_points: Some(1 << 20),    // 1M points
            max_reference_frames: Some(4),       // 4 reference frames
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
        }
    }

    /// Returns restrictive limits suitable for untrusted web content.
    pub fn restrictive() -> Self {
        Self {
            max_pixels: Some(100_000_000),       // 100 megapixels
            max_extra_channels: Some(16),        // 16 extra channels
            max_icc_size: Some(1 << 20),         // 1 MB
            max_tree_size: Some(1 << 20),        // 1M nodes
            max_patches: Some(1 << 16),          // 64K patches
            max_spline_points: Some(1 << 16),    // 64K points
            max_reference_frames: Some(2),       // 2 reference frames
        }
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

#[non_exhaustive]
pub struct JxlDecoderOptions {
    pub adjust_orientation: bool,
    pub render_spot_colors: bool,
    pub coalescing: bool,
    pub desired_intensity_target: Option<f32>,
    pub skip_preview: bool,
    pub progressive_mode: JxlProgressiveMode,
    pub xyb_output_linear: bool,
    pub enable_output: bool,
    pub cms: Option<Box<dyn JxlCms>>,
    /// Fail decoding images with more than this number of pixels, or with frames with
    /// more than this number of pixels. The limit counts the product of pixels and
    /// channels, so for example an image with 1 extra channel of size 1024x1024 has 4
    /// million pixels.
    ///
    /// **Deprecated**: Use `limits.max_pixels` instead.
    pub pixel_limit: Option<usize>,
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
}

impl Default for JxlDecoderOptions {
    fn default() -> Self {
        Self {
            adjust_orientation: true,
            render_spot_colors: true,
            coalescing: true,
            skip_preview: true,
            desired_intensity_target: None,
            progressive_mode: JxlProgressiveMode::Pass,
            xyb_output_linear: false,
            enable_output: true,
            cms: None,
            pixel_limit: None,
            high_precision: false,
            premultiply_output: false,
            limits: JxlDecoderLimits::default(),
        }
    }
}
