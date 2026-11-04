//! torch's antialiased bicubic resize of 8-bit images.
//!
//! On this host torchvision's `resize(..., BICUBIC, antialias=True)` on a
//! uint8 CHW tensor runs torch's native 8-bit kernel. It computes one
//! separable filter per axis over a support scaled by the downscaling factor
//! (`_compute_indices_min_size_weights_aa`), converts the normalized weights to
//! int16 fixed point with a per-axis precision derived from the largest weight
//! (`_compute_index_ranges_int16_weights`), and applies the filter
//! horizontally and then vertically with integer accumulation, a half-unit
//! bias and a shift (`basic_loop_separable_1d_horizontal<uint8_t>` and
//! `..._vertical<uint8_t>`). This module keeps that structure.
//!
//! The reference uses Keys' cubic convolution with `a = -0.5` for the
//! antialiased path.

use crate::{Error, Result};

/// The filter's radius in output pixels, before downscaling widens it.
const SUPPORT: f64 = 2.0;

/// Keys' cubic convolution parameter of the antialiased path.
const A: f64 = -0.5;

/// The largest fixed-point precision the reference considers.
const MAX_PRECISION: u32 = 22;

/// One axis' filter: the source window of each output position and its int16
/// fixed-point weights.
struct Filter {
    starts: Vec<usize>,
    counts: Vec<usize>,
    weights: Vec<i16>,
    /// Weights reserved per output position.
    width: usize,
    /// Fractional bits of the weights.
    precision: u32,
}

/// Keys' cubic convolution, the reference's `aa_filter` with `a = -0.5`.
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// The reference's `_compute_indices_min_size_weights_aa` and the int16 weight
/// conversion of `_compute_index_ranges_int16_weights`.
fn filter(input: usize, output: usize) -> Filter {
    let scale = input as f64 / output as f64;
    // A downscaling filter widens with the scale, so it averages every source
    // pixel the output pixel covers.
    let support = if scale >= 1.0 { SUPPORT * scale } else { SUPPORT };
    let width = support.ceil() as usize * 2 + 1;
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };

    let mut starts = Vec::with_capacity(output);
    let mut counts = Vec::with_capacity(output);
    let mut raw = Vec::with_capacity(output);
    let mut largest = 0.0f64;
    for out in 0..output {
        let center = scale * (out as f64 + 0.5);
        // The reference casts these bounds to int64, which truncates toward
        // zero before the clamps.
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let count = end
            .saturating_sub(start)
            .min(width);
        let mut weights = vec![0.0f64; count];
        let mut total = 0.0;
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = cubic((index + start) as f64 - center + 0.5) * inverse_scale;
            total += *weight;
        }
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
                largest = largest.max(*weight);
            }
        }
        starts.push(start);
        counts.push(count);
        raw.push(weights);
    }

    // The precision is the first at which the largest weight, doubled and
    // rounded, fills the int16 range's high bit.
    let mut precision = MAX_PRECISION;
    for candidate in 0..MAX_PRECISION {
        let scaled = 0.5 + largest * 2f64.powi(candidate as i32 + 1);
        if scaled as i64 >= 1 << 15 {
            precision = candidate;
            break;
        }
    }
    let mut quantized = vec![0i16; output * width];
    for (out, weights) in raw.iter().enumerate() {
        for (index, weight) in weights.iter().enumerate() {
            let value = weight * 2f64.powi(precision as i32);
            quantized[out * width + index] = if value < 0.0 {
                (value - 0.5) as i16
            } else {
                (value + 0.5) as i16
            };
        }
    }
    Filter {
        starts,
        counts,
        weights: quantized,
        width,
        precision,
    }
}

/// Applies the filter along one axis of a planar 8-bit image.
fn pass(
    axis: &Filter,
    source: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    channels: usize,
    horizontal: bool,
) -> Vec<u8> {
    let bias = 1i32 << (axis.precision - 1);
    let shift = axis.precision;
    if horizontal {
        let mut out = vec![0u8; output_width * input_height * channels];
        for row in 0..input_height {
            let source_row = &source[row * input_width * channels..][..input_width * channels];
            let out_row = &mut out[row * output_width * channels..][..output_width * channels];
            for column in 0..output_width {
                let start = axis.starts[column];
                let weights = &axis.weights[column * axis.width..][..axis.counts[column]];
                for channel in 0..channels {
                    let mut sum = bias;
                    for (index, weight) in weights.iter().enumerate() {
                        sum += i32::from(source_row[(start + index) * channels + channel])
                            * i32::from(*weight);
                    }
                    out_row[column * channels + channel] = (sum >> shift).clamp(0, 255) as u8;
                }
            }
        }
        out
    } else {
        let mut out = vec![0u8; output_height * output_width * channels];
        for row in 0..output_height {
            let start = axis.starts[row];
            let weights = &axis.weights[row * axis.width..][..axis.counts[row]];
            let out_row = &mut out[row * output_width * channels..][..output_width * channels];
            for column in 0..output_width {
                for channel in 0..channels {
                    let mut sum = bias;
                    for (index, weight) in weights.iter().enumerate() {
                        let sample =
                            source[(start + index) * input_width * channels + column * channels + channel];
                        sum += i32::from(sample) * i32::from(*weight);
                    }
                    out_row[column * channels + channel] = (sum >> shift).clamp(0, 255) as u8;
                }
            }
        }
        out
    }
}

/// Resizes a planar 8-bit image with torchvision's antialiased bicubic
/// filter.
///
/// `source` is `channels` planes of `height` rows of `width` samples; the
/// output has the same channel count at `out_width` x `out_height`. The
/// horizontal pass runs first and its 8-bit result feeds the vertical pass, as
/// in the reference; an unchanged axis skips its pass, and a resize to the
/// same size is a copy.
///
/// # Errors
///
/// Returns [`Error::Plan`] when a dimension or channel count is zero, or the
/// input is not `channels * width * height` bytes.
pub fn bicubic_antialias(
    source: &[u8],
    channels: usize,
    height: u32,
    width: u32,
    out_height: u32,
    out_width: u32,
) -> Result<Vec<u8>> {
    let (height, width, out_height, out_width) =
        (height as usize, width as usize, out_height as usize, out_width as usize);
    if channels == 0 || width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return Err(Error::Plan(
            "an image raster and its channel count must be positive".to_owned(),
        ));
    }
    if source.len() != channels * width * height {
        return Err(Error::Plan(format!(
            "a {channels}-channel image of {width}x{height} needs {} bytes, {} were given",
            channels * width * height,
            source.len()
        )));
    }
    if (width, height) == (out_width, out_height) {
        return Ok(source.to_vec());
    }
    let horizontal = if out_width != width {
        pass(
            &filter(width, out_width),
            source,
            width,
            height,
            out_width,
            height,
            channels,
            true,
        )
    } else {
        source.to_vec()
    };
    if out_height != height {
        Ok(pass(
            &filter(height, out_height),
            &horizontal,
            out_width,
            height,
            out_width,
            out_height,
            channels,
            false,
        ))
    } else {
        Ok(horizontal)
    }
}
