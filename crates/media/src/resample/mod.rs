//! Host resampling that reproduces the reference implementations' mathematics.
//!
//! The three resamplers are what the model conditioning used before the main
//! process was pure Rust: Pillow's LANCZOS resize, torchaudio's bandlimited
//! sinc resample, and torchvision's antialiased bicubic resize
//! (`specs/worker.md` §9, "主机端重采样"). Each reproduces its reference's
//! arithmetic, not merely a similar filter, so the results agree with the
//! reference on the §10 check set.

mod bicubic;
mod lanczos;
mod sinc;

use crate::Result;

/// Resizes an RGB24 image with Pillow's LANCZOS filter.
///
/// `src` is `[height, width, 3]` uint8, row-major; the result is
/// `[out_height, out_width, 3]`. Reproduces `PIL.Image.resize(size,
/// Image.Resampling.LANCZOS)` for an RGB image: the same precomputed
/// coefficients, the same fixed-point quantization and the same two-pass
/// clipping.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the dimensions are zero or `src` does not hold
/// `height * width * 3` bytes.
pub fn lanczos_resize(
    src: &[u8],
    width: u32,
    height: u32,
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>> {
    lanczos::lanczos_resize(src, width, height, out_width, out_height)
}

/// Resamples planar float audio with torchaudio's sinc interpolation.
///
/// `input` is `channels` planes of `length` samples each, contiguous, float32;
/// the result has the same layout at `new_freq`. Reproduces
/// `torchaudio.transforms.Resample(orig_freq, new_freq)` with its default
/// parameters (`lowpass_filter_width = 6`, `rolloff = 0.99`, Hann-windowed
/// sinc), including the zero padding and the output length
/// `ceil(new_freq * length / orig_freq)`.
///
/// # Errors
///
/// [`crate::Error::Plan`] when a rate is zero or `input` is not `channels *
/// length` samples.
pub fn sinc_resample(
    input: &[f32],
    channels: usize,
    length: usize,
    orig_freq: u32,
    new_freq: u32,
) -> Result<Vec<f32>> {
    sinc::sinc_resample(input, channels, length, orig_freq, new_freq)
}

/// Resizes a planar 8-bit image with torchvision's antialiased bicubic filter.
///
/// `src` is `channels` planes of `height * width` bytes, contiguous; the result
/// is `channels` planes of `out_height * out_width` bytes. Reproduces
/// `torchvision.transforms.v2.functional.resize(..., InterpolationMode.BICUBIC,
/// antialias=True)` on a uint8 tensor, which runs torch's antialiased bicubic
/// kernel.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the dimensions are zero or `src` does not hold
/// `channels * height * width` bytes.
pub fn bicubic_antialias_u8(
    src: &[u8],
    channels: usize,
    height: u32,
    width: u32,
    out_height: u32,
    out_width: u32,
) -> Result<Vec<u8>> {
    bicubic::bicubic_antialias_u8(src, channels, height, width, out_height, out_width)
}

/// Resizes a planar float32 image with torch's antialiased bicubic filter.
///
/// The layout is the one of [`bicubic_antialias_u8`]; reproduces
/// `torch.nn.functional.interpolate(..., mode = "bicubic", align_corners =
/// false, antialias = true)` on a float32 tensor.
///
/// # Errors
///
/// [`crate::Error::Plan`] when the dimensions are zero or `src` does not hold
/// `channels * height * width` samples.
pub fn bicubic_antialias_f32(
    src: &[f32],
    channels: usize,
    height: u32,
    width: u32,
    out_height: u32,
    out_width: u32,
) -> Result<Vec<f32>> {
    bicubic::bicubic_antialias_f32(src, channels, height, width, out_height, out_width)
}

#[cfg(test)]
mod tests;
