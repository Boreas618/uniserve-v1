//! Pillow's 8-bit LANCZOS resize.
//!
//! Pillow builds one separable filter per axis (`precompute_coeffs`), stores
//! its weights as 22-bit fixed-point integers (`normalize_coeffs_8bpc`) and
//! applies it horizontally and then vertically, clipping each pass to 8 bits
//! (`_ImagingResampleHorizontal_8bpc`, `_ImagingResampleVertical_8bpc`). Both
//! passes clip through `clip8(v) = clamp(v >> 22)`, so the intermediate image
//! is 8 bits again and the two passes are not one float convolution. This
//! module keeps that structure: the same coefficient grid, the same
//! quantization of the weights, the same accumulation bias and the same clip.

use crate::{Error, Result};

/// Fixed-point bits of a weight. Pillow's `PRECISION_BITS = 32 - 8 - 2`:
/// eight bits of result, two of overflow and sign, leaving 22.
const PRECISION_BITS: i32 = 22;

/// Half of a white pixel in the fixed-point domain: the rounding bias every
/// accumulation starts from (Pillow's `1 << (PRECISION_BITS - 1)`).
const ROUNDING_BIAS: i64 = 1 << (PRECISION_BITS - 1);

/// The radius at which Pillow's LANCZOS filter reaches zero.
const LANCZOS_SUPPORT: f64 = 3.0;

/// Pillow's LANCZOS kernel: `sinc(x) * sinc(x / 3)` on `[-3, 3)`, zero
/// outside.
fn lanczos_filter(x: f64) -> f64 {
    if !(-LANCZOS_SUPPORT..LANCZOS_SUPPORT).contains(&x) {
        return 0.0;
    }
    sinc(x) * sinc(x / LANCZOS_SUPPORT)
}

/// Pillow's `sinc_filter`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let scaled = x * std::f64::consts::PI;
    scaled.sin() / scaled
}

/// One axis of Pillow's separable filter: for each output position, the first
/// input position it reads and its fixed-point weights.
struct Filter {
    /// First input position of each output position.
    starts: Vec<usize>,
    /// Number of input positions each output position reads.
    counts: Vec<usize>,
    /// `outputs * width` weights, fixed point, row-major by output position.
    weights: Vec<i32>,
    /// The number of weights reserved per output position.
    width: usize,
}

/// Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc` for the whole
/// image (`in0 = 0`, `in1 = input`).
fn filter(input: usize, output: usize) -> Filter {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = LANCZOS_SUPPORT * filter_scale;
    let width = support.ceil() as usize * 2 + 1;

    let mut starts = Vec::with_capacity(output);
    let mut counts = Vec::with_capacity(output);
    let mut weights = vec![0i32; output * width];
    let inverse_scale = 1.0 / filter_scale;
    let mut raw = Vec::with_capacity(width);
    for out in 0..output {
        let center = (out as f64 + 0.5) * scale;
        // Pillow rounds these bounds with a C cast to int, which truncates
        // toward zero before the clamp.
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let count = end.saturating_sub(start);

        // Pillow stores the raw coefficients, divides them by their sum and
        // only then quantizes, so each weight is rounded exactly once.
        raw.clear();
        let mut total = 0.0;
        for index in 0..count {
            // The filter's support widens with the downscaling factor: the
            // reference scales the filter's argument, not its value.
            let weight = lanczos_filter(((index + start) as f64 - center + 0.5) * inverse_scale);
            raw.push(weight);
            total += weight;
        }
        if total != 0.0 {
            for weight in &mut raw {
                *weight /= total;
            }
        }
        for (index, weight) in raw.iter().enumerate() {
            weights[out * width + index] = quantize(*weight);
        }
        starts.push(start);
        counts.push(count);
    }
    Filter {
        starts,
        counts,
        weights,
        width,
    }
}

/// Pillow's fixed-point quantization: a C cast of `±0.5 + w << 22`, which
/// rounds halves away from zero.
fn quantize(weight: f64) -> i32 {
    let scaled = weight * f64::from(1 << PRECISION_BITS);
    if scaled < 0.0 {
        (scaled - 0.5) as i32
    } else {
        (scaled + 0.5) as i32
    }
}

/// Applies one axis' filter to 8-bit samples.
///
/// `horizontal` resamples each row of an `input_width`-wide image into
/// `output_width`; otherwise each column of an `input_height`-tall image into
/// `output_height`. `channels` samples are interleaved per position.
#[expect(
    clippy::too_many_arguments,
    reason = "one pass needs both extents, the channel count and the axis"
)]
fn pass(
    filter: &Filter,
    source: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    channels: usize,
    horizontal: bool,
) -> Vec<u8> {
    if horizontal {
        let mut out = vec![0u8; output_width * input_height * channels];
        for row in 0..input_height {
            let source_row = &source[row * input_width * channels..][..input_width * channels];
            let out_row = &mut out[row * output_width * channels..][..output_width * channels];
            for column in 0..output_width {
                let start = filter.starts[column];
                let count = filter.counts[column];
                let weights = &filter.weights[column * filter.width..][..count];
                for channel in 0..channels {
                    let mut sum = ROUNDING_BIAS;
                    for (index, weight) in weights.iter().enumerate() {
                        let sample = source_row[(start + index) * channels + channel] as i64;
                        sum += sample * i64::from(*weight);
                    }
                    out_row[column * channels + channel] = clip8(sum);
                }
            }
        }
        out
    } else {
        let mut out = vec![0u8; output_height * output_width * channels];
        for row in 0..output_height {
            let start = filter.starts[row];
            let count = filter.counts[row];
            let weights = &filter.weights[row * filter.width..][..count];
            let out_row = &mut out[row * output_width * channels..][..output_width * channels];
            for column in 0..output_width {
                for channel in 0..channels {
                    let mut sum = ROUNDING_BIAS;
                    for (index, weight) in weights.iter().enumerate() {
                        let sample = source
                            [(start + index) * input_width * channels + column * channels + channel]
                            as i64;
                        sum += sample * i64::from(*weight);
                    }
                    out_row[column * channels + channel] = clip8(sum);
                }
            }
        }
        out
    }
}

/// Pillow's `clip8`: the fixed-point sum's integer part clamped to a byte.
///
/// The C code looks the arithmetic shift up in a table over `[-640, 639]`,
/// which is exactly the clamp.
fn clip8(sum: i64) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resizes an RGB24 image exactly as Pillow's `Image.resize(size, LANCZOS)`
/// does.
///
/// The input is `height` rows of `width` RGB triples; the output is
/// `out_height` rows of `out_width` triples. A resize to the same size is a
/// copy, and an unchanged axis skips its pass, as in Pillow.
///
/// # Errors
///
/// Returns [`Error::Plan`] when a dimension is zero or the input is not
/// `width * height * 3` bytes.
pub(super) fn lanczos_resize(
    source: &[u8],
    width: u32,
    height: u32,
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>> {
    let (width, height, out_width, out_height) = (
        width as usize,
        height as usize,
        out_width as usize,
        out_height as usize,
    );
    if width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return Err(Error::Plan("an image raster must be positive".to_owned()));
    }
    if source.len() != width * height * 3 {
        return Err(Error::Plan(format!(
            "an RGB24 image of {width}x{height} needs {} bytes, {} were given",
            width * height * 3,
            source.len()
        )));
    }
    if (width, height) == (out_width, out_height) {
        return Ok(source.to_vec());
    }

    // Pillow resamples horizontally first and clips the intermediate image to
    // 8 bits before the vertical pass.
    let horizontal = if out_width != width {
        pass(
            &filter(width, out_width),
            source,
            width,
            height,
            out_width,
            height,
            3,
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
            3,
            false,
        ))
    } else {
        Ok(horizontal)
    }
}
