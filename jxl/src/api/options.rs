// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlCms;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Re-export the cooperative cancellation trait and types from `enough`.
// Library users can implement `Stop` for custom cancellation logic (timeouts, etc.).
pub use enough::{Stop, StopReason, Unstoppable};

/// A cancellation token that can be used to abort decoding.
///
/// This is a concrete implementation of [`Stop`] using an atomic boolean.
/// For custom cancellation logic (timeouts, integration with async runtimes, etc.),
/// implement [`Stop`] directly or use types from the [`almost-enough`](https://docs.rs/almost-enough) crate.
///
/// # Example
/// ```
/// use jxl::api::{CancellationToken, Stop};
/// use std::thread;
///
/// let token = CancellationToken::new();
/// let token_clone = token.clone();
///
/// // Cancel from another thread after timeout
/// thread::spawn(move || {
///     thread::sleep(std::time::Duration::from_secs(5));
///     token_clone.cancel();
/// });
///
/// // Check cancellation status
/// assert!(!token.should_stop());
/// ```
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a new cancellation token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signals cancellation. All decoders using this token will abort.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Checks if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Resets the token, allowing reuse after cancellation.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

impl Stop for CancellationToken {
    fn check(&self) -> Result<(), StopReason> {
        if self.is_cancelled() {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Security limits for the JXL decoder to prevent resource exhaustion attacks.
///
/// These limits protect against "JXL bombs" - maliciously crafted files designed
/// to exhaust memory or CPU. All limits are optional; `None` means use the default.
///
/// # Example
/// ```
/// use jxl::api::JxlDecoderLimits;
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

    /// Maximum total memory budget in bytes. Default: None (unlimited).
    /// When set, the decoder tracks allocations and fails if budget exceeded.
    /// This provides defense-in-depth against memory exhaustion attacks.
    pub max_memory_bytes: Option<u64>,
}

impl Default for JxlDecoderLimits {
    fn default() -> Self {
        Self {
            max_pixels: Some(1 << 30),        // ~1 billion pixels
            max_extra_channels: Some(256),    // 256 extra channels
            max_icc_size: Some(1 << 28),      // 256 MB
            max_tree_size: Some(1 << 22),     // 4M nodes
            max_patches: None,                // Use image-size-based default
            max_spline_points: Some(1 << 20), // 1M points
            max_reference_frames: Some(4),    // 4 reference frames
            max_memory_bytes: None,           // No overall memory limit by default
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
            max_pixels: Some(100_000_000),    // 100 megapixels
            max_extra_channels: Some(16),     // 16 extra channels
            max_icc_size: Some(1 << 20),      // 1 MB
            max_tree_size: Some(1 << 20),     // 1M nodes
            max_patches: Some(1 << 16),       // 64K patches
            max_spline_points: Some(1 << 16), // 64K points
            max_reference_frames: Some(2),    // 2 reference frames
            max_memory_bytes: Some(1 << 30),  // 1 GB total memory
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

/// Decoder options with cooperative cancellation support.
///
/// The type parameter `S` is the [`Stop`] implementation for cooperative cancellation.
/// It defaults to [`Unstoppable`], which has zero runtime overhead.
///
/// # Examples
///
/// Default (no cancellation, zero overhead):
/// ```
/// use jxl::api::JxlDecoderOptions;
///
/// let options = JxlDecoderOptions::default();
/// ```
///
/// With cancellation using [`CancellationToken`]:
/// ```
/// use jxl::api::{CancellationToken, JxlDecoderOptions};
///
/// let token = CancellationToken::new();
/// let options = JxlDecoderOptions::with_stop(token.clone());
/// // Later, from another thread: token.cancel();
/// ```
///
/// With a custom [`Stop`] implementation (e.g., from `almost-enough` crate):
/// ```ignore
/// use almost_enough::Stopper;
/// use jxl::api::JxlDecoderOptions;
///
/// let stopper = Stopper::new();
/// let options = JxlDecoderOptions::with_stop(stopper.clone());
/// // stopper.cancel(); // Cancels from anywhere
/// ```
#[non_exhaustive]
pub struct JxlDecoderOptions<S: Stop = Unstoppable> {
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
    /// Cooperative cancellation. Defaults to [`Unstoppable`] (zero overhead).
    ///
    /// The decoder checks this at various points during decoding and will
    /// return [`Error::Cancelled`](crate::error::Error::Cancelled) if stopped.
    pub stop: S,
}

impl<S: Stop> JxlDecoderOptions<S> {
    /// Create options with a custom stop implementation.
    pub fn with_stop(stop: S) -> Self {
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
            stop,
        }
    }
}

impl<S: Stop + Clone + 'static> JxlDecoderOptions<S> {
    /// Get a type-erased Arc reference to the stop implementation.
    ///
    /// This is used internally for storing the stop in decoder state
    /// without making the entire decoder generic.
    pub(crate) fn stop_arc(&self) -> Arc<dyn Stop> {
        Arc::new(self.stop.clone())
    }
}

impl Default for JxlDecoderOptions<Unstoppable> {
    fn default() -> Self {
        Self::with_stop(Unstoppable)
    }
}
