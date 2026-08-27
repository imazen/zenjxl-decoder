// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::TryReserveError;

use thiserror::Error;
use whereat::At;

use crate::{
    api::{JxlColorType, JxlDataFormat},
    entropy_coding::huffman::HUFFMAN_MAX_BITS,
    features::spline::Point,
    image::DataTypeTag,
};

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("Invalid raw quantization table")]
    InvalidRawQuantTable,
    #[error("Invalid distance band {0}: {1}")]
    InvalidDistanceBand(usize, f32),
    #[error("Invalid AFV bands")]
    InvalidAFVBands,
    #[error("Invalid quantization table weight: {0}")]
    InvalidQuantizationTableWeight(f32),
    #[error("Read out of bounds; size hint: {0}")]
    OutOfBounds(usize),
    #[error("Section is too short")]
    SectionTooShort,
    #[error("Non-zero padding bits")]
    NonZeroPadding,
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid exponent_bits_per_sample: {0}")]
    InvalidExponent(u32),
    #[error("Invalid mantissa_bits: {0}")]
    InvalidMantissa(i32),
    #[error("Invalid bits_per_sample: {0}")]
    InvalidBitsPerSample(u32),
    #[error("Invalid enum value {0} for {1}")]
    InvalidEnum(u32, String),
    #[error("Value of dim_shift {0} is too large")]
    DimShiftTooLarge(u32),
    #[error("Float is NaN or Inf")]
    FloatNaNOrInf,
    #[error("Invalid gamma value: {0}")]
    InvalidGamma(f32),
    #[error("Invalid color encoding: no ICC and unknown TF / ColorSpace")]
    InvalidColorEncoding,
    #[error("Invalid color space: should be one of RGB, Gray or XYB")]
    InvalidColorSpace,
    #[error("Only perceptual rendering intent implemented for XYB ICC profile.")]
    InvalidRenderingIntent,
    #[error("Invalid intensity_target: {0}")]
    InvalidIntensityTarget(f32),
    #[error("Invalid min_nits: {0}")]
    InvalidMinNits(f32),
    #[error("Invalid linear_below {1}, relative_to_max_display is {0}")]
    InvalidLinearBelow(bool, f32),
    #[error("Overflow when computing a bitstream size")]
    SizeOverflow,
    #[error("Invalid ISOBMMF container")]
    InvalidBox,
    #[error("ICC is too large")]
    IccTooLarge,
    #[error("Invalid ICC stream: unexpected end of stream")]
    IccEndOfStream,
    #[error("Invalid ICC stream")]
    InvalidIccStream,
    #[error("Invalid HybridUintConfig: {0} {1} {2:?}")]
    InvalidUintConfig(u32, u32, Option<u32>),
    #[error("LZ77 enabled when explicitly disallowed")]
    Lz77Disallowed,
    #[error("LZ77 repeat symbol encountered without decoding any symbols")]
    UnexpectedLz77Repeat,
    #[error("Huffman alphabet too large: {0}, max is {max}", max = 1 << HUFFMAN_MAX_BITS)]
    AlphabetTooLargeHuff(usize),
    #[error("Invalid Huffman code")]
    InvalidHuffman,
    #[error("Invalid ANS histogram")]
    InvalidAnsHistogram,
    #[error("ANS stream checksum mismatch")]
    AnsChecksumMismatch,
    #[error("Integer too large: nbits {0} > 29")]
    IntegerTooLarge(u32),
    #[error("Invalid context map: context id {0} > 255")]
    InvalidContextMap(u32),
    #[error("Invalid context map: number of histogram {0}, number of distinct histograms {1}")]
    InvalidContextMapHole(u32, u32),
    #[error(
        "Invalid permutation: skipped elements {skip} and encoded elements {end} don't fit in permutation of size {size}"
    )]
    InvalidPermutationSize { size: u32, skip: u32, end: u32 },
    #[error(
        "Invalid permutation: Lehmer code {lehmer} out of bounds in permutation of size {size} at index {idx}"
    )]
    InvalidPermutationLehmerCode { size: u32, idx: u32, lehmer: u32 },
    #[error("Invalid quant encoding mode")]
    InvalidQuantEncodingMode,
    #[error("Invalid quant encoding with mode {mode} and required size {required_size}")]
    InvalidQuantEncoding { mode: u8, required_size: usize },
    // FrameHeader format errors
    #[error("Invalid extra channel upsampling: upsampling: {0} dim_shift: {1} ec_upsampling: {2}")]
    InvalidEcUpsampling(u32, u32, u32),
    #[error("Patches with mixed upsampling not supported: color {0}, ec {1}")]
    PatchesUnsupportedMixedUpsampling(u32, u32),
    #[error("Invalid lf level in UseLFFrame frame: {0}")]
    InvalidLfLevel(u32),
    #[error("Num_ds: {0} should be smaller than num_passes: {1}")]
    NumPassesTooLarge(u32, u32),
    #[error("Passes::downsample is non-decreasing")]
    PassesDownsampleNonDecreasing,
    #[error("Passes::last_pass is non-increasing")]
    PassesLastPassNonIncreasing,
    #[error("Passes::last_pass has too large elements")]
    PassesLastPassTooLarge,
    #[error("Non-patch reference frame with a crop")]
    NonPatchReferenceWithCrop,
    #[error("Non-444 chroma subsampling is not allowed when adaptive DC smoothing is enabled")]
    Non444ChromaSubsampling,
    #[error("Non-444 chroma subsampling is not allowed for bigger than 8x8 transforms")]
    InvalidBlockSizeForChromaSubsampling,
    #[error("Out of memory: {0}")]
    OutOfMemory(#[from] TryReserveError),
    #[error("Out of memory when allocating image of byte size {0}x{1}")]
    ImageOutOfMemory(usize, usize),
    #[error("Image size too large: {0}x{1}")]
    ImageSizeTooLarge(usize, usize),
    #[error("Image dimension too large: {0}")]
    ImageDimensionTooLarge(u64),
    #[error("Invalid image size: {0}x{1}")]
    InvalidImageSize(usize, usize),
    // Generic arithmetic overflow. Prefer using other errors if possible.
    #[error("Arithmetic overflow")]
    ArithmeticOverflow,
    #[error(
        "Pipeline channel type mismatch: stage {0} channel {1}, expected {2:?} but found {3:?}"
    )]
    PipelineChannelTypeMismatch(String, usize, DataTypeTag, DataTypeTag),
    #[error("Invalid stage {0} after extend stage")]
    PipelineInvalidStageAfterExtend(String),
    #[error("Trying to copy rects of different size, src: {0}x{1} dst {2}x{3}")]
    CopyOfDifferentSize(usize, usize, usize, usize),
    #[error("LF quantization factor is too small: {0}")]
    LfQuantFactorTooSmall(f32),
    #[error("HF quantization factor is too small: {0}")]
    HfQuantFactorTooSmall(f32),
    #[error("Invalid modular mode predictor: {0}")]
    InvalidPredictor(u32),
    #[error("Invalid modular mode property: {0}")]
    InvalidProperty(u32),
    #[error("Invalid alpha channel for blending: {0}, limit is {1}")]
    InvalidBlendingAlphaChannel(usize, usize),
    #[error("Blending cannot use reference frame {0} saved before color transforms")]
    BlendingPreColorTransform(usize),
    #[error("Invalid alpha channel for blending: {0}, limit is {1}")]
    PatchesInvalidAlphaChannel(usize, usize),
    #[error("Invalid patch blend mode: {0}, limit is {1}")]
    PatchesInvalidBlendMode(u8, u8),
    #[error("Invalid Patch: negative {0}-coordinate: {1} base {0},  {2} delta {0}")]
    PatchesInvalidDelta(String, usize, i32),
    #[error(
        "Invalid position specified in reference frame in {0}-coordinate: {0}0 + {0}size = {1} + {2} > {3} = reference_frame {0}size"
    )]
    PatchesInvalidPosition(String, usize, usize, usize),
    #[error("Patches invalid reference frame at index {0}")]
    PatchesInvalidReference(usize),
    #[error("Invalid Patch {0}: at {1} + {2} > {3}")]
    PatchesOutOfBounds(String, usize, usize, usize),
    #[error("Patches cannot use frames saved post color transforms")]
    PatchesPostColorTransform(),
    #[error("Too many {0}: {1}, limit is {2}")]
    PatchesTooMany(String, usize, usize),
    #[error("Reference too large: {0}, limit is {1}")]
    PatchesRefTooLarge(usize, usize),
    #[error("Point list is empty")]
    PointListEmpty,
    #[error("Too large area for spline: {0}, limit is {1}")]
    SplinesAreaTooLarge(u64, u64),
    #[error("Too large manhattan_distance reached: {0}, limit is {1}")]
    SplinesDistanceTooLarge(u64, u64),
    #[error("Too many splines: {0}, limit is {1}")]
    SplinesTooMany(u32, u32),
    #[error("Spline has adjacent coinciding control points: point[{0}]: {1:?}, point[{2}]: {3:?}")]
    SplineAdjacentCoincidingControlPoints(usize, Point, usize, Point),
    #[error("Too many control points for splines: {0}, limit is {1}")]
    SplinesTooManyControlPoints(u32, u32),
    #[error(
        "Spline point outside valid bounds: coordinates: {0:?}, out of bounds: {1}, bounds: {2:?}"
    )]
    SplinesPointOutOfRange(Point, i32, std::ops::Range<i32>),
    #[error("Spline coordinates out of bounds: {0}, limit is {1}")]
    SplinesCoordinatesLimit(isize, isize),
    #[error("Spline delta-delta is out of bounds: {0}, limit is {1}")]
    SplinesDeltaLimit(i64, i64),
    #[error("Modular tree too large: {0}, limit is {1}")]
    TreeTooLarge(usize, usize),
    #[error("Modular tree too tall: {0}, limit is {1}")]
    TreeTooTall(usize, usize),
    #[error("Modular tree multiplier too large: {0}, limit is {1}")]
    TreeMultiplierTooLarge(u32, u32),
    #[error("Modular tree multiplier too large: {0}, multiplier log is {1}")]
    TreeMultiplierBitsTooLarge(u32, u32),
    #[error(
        "Modular tree splits on property {0} at value {1}, which is outside the possible range of [{2}, {3}]"
    )]
    TreeSplitOnEmptyRange(u8, i32, i32, i32),
    #[error("Modular stream requested a global tree but there isn't one")]
    NoGlobalTree,
    #[error("Invalid transform id")]
    InvalidTransformId,
    #[error("Invalid RCT type {0}")]
    InvalidRCT(u32),
    #[error("Invalid channel range: {0}..{1}, {2} total channels")]
    InvalidChannelRange(usize, usize, usize),
    #[error("Invalid transform: mixing different channels (different shape or different shift)")]
    MixingDifferentChannels,
    #[error("Invalid transform: squeezing meta-channels needs an in-place transform")]
    MetaSqueezeRequiresInPlace,
    #[error("Invalid transform: too many squeezes (shift > 30)")]
    TooManySqueezes,
    #[error("Invalid BlockConextMap: too big: num_lf_context: {0}, num_qf_thresholds: {1}")]
    BlockContextMapSizeTooBig(usize, usize),
    #[error("Invalid BlockConextMap: too many distinct contexts.")]
    TooManyBlockContexts,
    #[error("Base color correlation out of range.")]
    BaseColorCorrelationOutOfRange,
    #[error("Invalid EPF sharpness param {0}")]
    InvalidEpfValue(i32),
    #[error("Invalid VarDCT transform type {0}")]
    InvalidVarDCTTransform(usize),
    #[error("Invalid VarDCT transform map")]
    InvalidVarDCTTransformMap,
    #[error("VarDCT transform overflows HF group")]
    HFBlockOutOfBounds,
    #[error("Invalid AC: nonzeros {0} is too large for {1} 8x8 blocks")]
    InvalidNumNonZeros(usize, usize),
    #[error("Invalid AC: histogram index {0} is out of bounds (num_histograms = {1})")]
    InvalidHistogramIndex(usize, usize),
    #[error("Invalid AC: {0} nonzeros after decoding block")]
    EndOfBlockResidualNonZeros(usize),
    #[error("Unknown transfer function for ICC profile")]
    TransferFunctionUnknown,
    #[error("Attempting to write out of Bounds when writing ICC")]
    IccWriteOutOfBounds,
    #[error("Invalid tag string when writing ICC: {0}")]
    IccInvalidTagString(String),
    #[error("Invalid text for ICC MLuc string, not ascii: {0}")]
    IccMlucTextNotAscii(String),
    #[error("ICC value is out of range / NaN: {0}")]
    IccValueOutOfRangeS15Fixed16(f32),
    #[error("Y value is too small: {0}")]
    IccInvalidWhitePointY(f32),
    #[error("{2}: wx: {0}, wy: {1}")]
    IccInvalidWhitePoint(f32, f32, String),
    #[error("Determinant is zero or too small, matrix is close to singular: |det| = {0}.")]
    MatrixInversionFailed(f64),
    #[error("Unsupported transfer function when writing ICC")]
    IccUnsupportedTransferFunction,
    #[error("Table size too large when writing ICC: {0}")]
    IccTableSizeExceeded(usize),
    #[error("Invalid CMS configuration: requested ICC but no CMS is configured")]
    ICCOutputNoCMS,
    #[error("Non-XYB image requires CMS to convert to different output color profile")]
    NonXybOutputNoCMS,
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Wrong buffer count: {0} buffers given, {1} buffers expected")]
    WrongBufferCount(usize, usize),
    #[error("Image is not grayscale, but grayscale output was requested")]
    NotGrayscale,
    #[error("Image is not CMYK, but CMYK output was requested")]
    NotCmyk,
    #[error("Invalid output buffer byte size {0}x{1} for {2}x{3} image with type {4:?} {5:?}")]
    InvalidOutputBufferSize(usize, usize, usize, usize, JxlColorType, JxlDataFormat),
    #[error("Attempting to save channels with different downsample amounts: {0:?} and {1:?}")]
    SaveDifferentDownsample((u8, u8), (u8, u8)),
    #[error("Image has {0} extra channels, more than the maximum of 256")]
    TooManyExtraChannels(usize),
    #[error("No LF frame for level {0}")]
    NoLfFrame(u32),
    #[error(
        "CMS transform increases channel count from {in_channels} to {out_channels}, which is not supported"
    )]
    CmsChannelCountIncrease {
        in_channels: usize,
        out_channels: usize,
    },
    #[error(
        "Cannot output extra channel {channel_index} ({channel_type:?}): it was consumed by CMS color conversion"
    )]
    CmsConsumedChannelRequested {
        channel_index: usize,
        channel_type: String,
    },
    #[error("CMS error: {0}")]
    CmsError(String),
    /// A configurable security/resource limit was exceeded. `resource` is one of
    /// the stable identifiers `"pixels"`, `"memory_bytes"`, `"icc_size"`,
    /// `"icc_amplification"`, `"extra_channels"`, or `"reference_frames"` —
    /// the budgets in `JxlDecoderLimits`. For coarse bucketing (e.g. to an HTTP
    /// status) use [`Error::kind`], which maps this to [`ErrorClass::LimitExceeded`].
    #[error("Security limit exceeded: {resource} is {actual}, limit is {limit}")]
    LimitExceeded {
        resource: &'static str,
        actual: u64,
        limit: u64,
    },
    #[error("Decoding cancelled")]
    Cancelled,
    #[cfg(feature = "jpeg")]
    #[error("Invalid JBRD box: {0}")]
    InvalidJbrd(String),
    #[error("Invalid gain map bundle (jhgm): {0}")]
    InvalidGainMap(String),
    #[error(
        "Progressive content rejected: a progressive frame header (multi-pass or LF frame) was seen but `JxlDecoderOptions::reject_progressive` forbids it"
    )]
    ProgressiveRejected,
    #[error(
        "start_new_frame called before the image header was parsed: seeking needs the image info first"
    )]
    SeekBeforeImageInfo,
}

/// A coarse, best-effort classification of an [`Error`], for operational
/// decisions: mapping a decode failure to an HTTP status, choosing a log level,
/// or deciding whether a retry could help. The exact [`Error`] variant remains
/// available for diagnostics; [`Error::kind`] exists so callers don't have to
/// match all ~130 variants.
///
/// `#[non_exhaustive]`: new classes may be added, and existing errors may be
/// reclassified, without a breaking change — always include a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorClass {
    /// The input bitstream is malformed, corrupt, truncated, or violates the
    /// JPEG XL specification. Retrying the same bytes will not help (HTTP ~400).
    InvalidBitstream,
    /// A configurable decode limit was exceeded — one of `JxlDecoderLimits`'
    /// budgets (`pixels`, `memory_bytes`, `icc_size`, `icc_amplification`,
    /// `extra_channels`, `reference_frames`). The same input may succeed with a
    /// higher limit (HTTP ~413).
    LimitExceeded,
    /// A memory allocation failed (HTTP ~500/507).
    OutOfMemory,
    /// The decode was cancelled or timed out via the cooperative `stop` handle
    /// (HTTP ~499/408).
    Cancelled,
    /// An I/O error occurred reading the input (HTTP ~500).
    Io,
    /// The caller's output request or colour-management configuration is invalid
    /// for this image (wrong output buffer size/count, grayscale mismatch, or an
    /// ICC/CMS configuration that cannot satisfy the request). A usage error, not
    /// an input problem (HTTP ~400/500).
    OutputConfiguration,
    /// The input is valid but uses a feature this decoder or the current
    /// configuration does not support — e.g. a progressive frame when
    /// `reject_progressive` is set (HTTP ~415).
    Unsupported,
    /// An internal invariant was violated (arithmetic overflow, a pipeline
    /// inconsistency, ICC synthesis failure) — most likely a decoder bug rather
    /// than a problem with the input (HTTP ~500).
    Internal,
}

impl Error {
    /// Classify this error into a coarse [`ErrorClass`] for operational
    /// decisions (HTTP status, log level, retry). Best-effort: see the variant
    /// itself for exact diagnostics.
    pub fn kind(&self) -> ErrorClass {
        match self {
            // A configurable decode limit (see `JxlDecoderLimits`) was hit.
            Error::LimitExceeded { .. }
            | Error::ImageSizeTooLarge(..)
            | Error::ImageDimensionTooLarge(..)
            | Error::IccTooLarge => ErrorClass::LimitExceeded,

            // An allocation failed.
            Error::OutOfMemory(..) | Error::ImageOutOfMemory(..) => ErrorClass::OutOfMemory,

            // Cooperative cancellation / timeout.
            Error::Cancelled => ErrorClass::Cancelled,

            // I/O reading the input.
            Error::IOError(..) => ErrorClass::Io,

            // The caller's output request or colour-management configuration is
            // invalid for this image — a usage error, not an input problem.
            Error::WrongBufferCount(..)
            | Error::NotGrayscale
            | Error::NotCmyk
            | Error::InvalidOutputBufferSize(..)
            | Error::ICCOutputNoCMS
            | Error::NonXybOutputNoCMS
            | Error::CmsConsumedChannelRequested { .. }
            | Error::CmsChannelCountIncrease { .. }
            | Error::SeekBeforeImageInfo => ErrorClass::OutputConfiguration,

            // Valid input using a feature this decoder / configuration refuses.
            Error::ProgressiveRejected | Error::InvalidRenderingIntent => ErrorClass::Unsupported,

            // Internal invariant violations — most likely a decoder bug. Includes
            // failures while *synthesising* an output ICC profile.
            Error::ArithmeticOverflow
            | Error::SizeOverflow
            | Error::MatrixInversionFailed(..)
            | Error::CopyOfDifferentSize(..)
            | Error::PipelineChannelTypeMismatch(..)
            | Error::PipelineInvalidStageAfterExtend(..)
            | Error::IccWriteOutOfBounds
            | Error::IccInvalidTagString(..)
            | Error::IccMlucTextNotAscii(..)
            | Error::IccValueOutOfRangeS15Fixed16(..)
            | Error::IccInvalidWhitePointY(..)
            | Error::IccInvalidWhitePoint(..)
            | Error::IccUnsupportedTransferFunction
            | Error::IccTableSizeExceeded(..) => ErrorClass::Internal,

            // Everything else: the bitstream is malformed, corrupt, truncated, or
            // violates the JPEG XL spec. New variants default here; revisit this
            // when adding a variant that is a limit, config, or internal error.
            _ => ErrorClass::InvalidBitstream,
        }
    }
}

impl From<enough::StopReason> for Error {
    fn from(reason: enough::StopReason) -> Self {
        match reason {
            enough::StopReason::Cancelled | enough::StopReason::TimedOut | _ => Error::Cancelled,
        }
    }
}

/// Result alias whose default error type carries a `whereat` source-location
/// trace ([`At<Error>`]). Origins are created with `at!(Error::X)`; propagation
/// via `?` preserves the trace through `whereat`'s blanket `From<E> for At<E>`.
pub type Result<T, E = At<Error>> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_kind_classification() {
        assert_eq!(
            Error::LimitExceeded {
                resource: "pixels",
                actual: 9,
                limit: 8,
            }
            .kind(),
            ErrorClass::LimitExceeded,
        );
        assert_eq!(Error::Cancelled.kind(), ErrorClass::Cancelled);
        assert_eq!(
            Error::ImageOutOfMemory(4, 4).kind(),
            ErrorClass::OutOfMemory
        );
        assert_eq!(Error::NotGrayscale.kind(), ErrorClass::OutputConfiguration);
        assert_eq!(Error::NotCmyk.kind(), ErrorClass::OutputConfiguration);
        assert_eq!(
            Error::SeekBeforeImageInfo.kind(),
            ErrorClass::OutputConfiguration
        );
        assert_eq!(Error::ProgressiveRejected.kind(), ErrorClass::Unsupported,);
        assert_eq!(Error::ArithmeticOverflow.kind(), ErrorClass::Internal);
        // Malformed-bitstream variants fall into the default class.
        assert_eq!(Error::InvalidSignature.kind(), ErrorClass::InvalidBitstream);
        assert_eq!(Error::InvalidHuffman.kind(), ErrorClass::InvalidBitstream);
    }
}
