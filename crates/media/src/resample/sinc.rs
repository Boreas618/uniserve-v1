//! torchaudio's bandlimited sinc resample.
//!
//! The reference builds one filter bank per resampling ratio
//! (`_get_sinc_resample_kernel`) and applies it as a strided convolution
//! (`_apply_sinc_resample_kernel`): the signal is zero-padded by `width` on the
//! left and `width + orig` on the right, and a `[new, 1, 2*width + orig]`
//! filter bank is applied with stride `orig`. Filter `i` produces the output
//! samples `i, i + new, i + 2*new, ...`, so the bank's output flattened
//! position-major and then filter-major is the resampled signal. The kernel
//! grid is computed in the waveform's dtype, float32, which this module keeps.

use crate::{Error, Result};

/// torchaudio's default low-pass filter width in zero crossings.
const LOWPASS_FILTER_WIDTH: f64 = 6.0;

/// torchaudio's default roll-off, as a fraction of the Nyquist frequency.
const ROLLOFF: f64 = 0.99;

/// Resamples planar float32 audio from `orig_freq` to `new_freq` exactly as
/// `torchaudio.transforms.Resample(orig_freq, new_freq)` does.
///
/// `input` is `channels` planes of `length` samples each, one after another;
/// the output has the same layout and `ceil(new * length / orig)` samples per
/// plane.
///
/// # Numerics
///
/// The filter bank and the accumulation are float32, as in the reference. The
/// reference computes the convolution with torch's vectorized float32 kernel,
/// whose summation order differs from this module's scalar order; the values
/// agree to a few float32 rounding steps of the filter's ~`2*width + orig`
/// taps. Tests compare within `1e-5` absolute, which is ~100 ulp of a
/// unit-amplitude sample and far below any audible or model-visible
/// difference.
///
/// # Errors
///
/// Returns [`Error::Plan`] when a rate is zero, a channel count is zero, or
/// the input is not `channels * length` samples.
pub fn sinc_resample(
    input: &[f32],
    channels: usize,
    length: usize,
    orig_freq: u32,
    new_freq: u32,
) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(Error::Plan(
            "a resampled waveform needs at least one channel".to_owned(),
        ));
    }
    if orig_freq == 0 || new_freq == 0 {
        return Err(Error::Plan(
            "audio resampling rates must be positive".to_owned(),
        ));
    }
    if input.len() != channels * length {
        return Err(Error::Plan(format!(
            "a {channels}-channel waveform of {length} samples needs {}, {} were given",
            channels * length,
            input.len()
        )));
    }
    if orig_freq == new_freq {
        return Ok(input.to_vec());
    }

    let (orig, new) = reduce(orig_freq, new_freq);
    // The reference computes the ratio and the filter width in Python's
    // float64, and the grid and the weights in the waveform's float32.
    let base_freq = orig.min(new) as f64 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig as f64 / base_freq).ceil() as usize;
    let taps = 2 * width + orig;

    let kernel = kernel(width, taps, orig, new, base_freq);
    let target = (new as f64 * length as f64 / orig as f64).ceil() as usize;

    let mut output = vec![0.0f32; channels * target];
    for channel in 0..channels {
        let plane = &input[channel * length..][..length];
        let out = &mut output[channel * target..][..target];
        for (position, sample) in out.iter_mut().enumerate() {
            // Sample `position` is filter `position % new` at convolution
            // position `position / new`.
            let filter = position % new;
            let start = (position / new) * orig;
            let weights = &kernel[filter * taps..][..taps];
            let mut sum = 0.0f32;
            for (index, weight) in weights.iter().enumerate() {
                let at = start as isize + index as isize - width as isize;
                if at >= 0 && (at as usize) < length {
                    sum += plane[at as usize] * weight;
                }
            }
            *sample = sum;
        }
    }
    Ok(output)
}

/// The reduced frequencies torchaudio convolves with: both divided by their
/// greatest common divisor.
fn reduce(orig: u32, new: u32) -> (usize, usize) {
    let mut a = orig;
    let mut b = new;
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    let divisor = a.max(1);
    ((orig / divisor) as usize, (new / divisor) as usize)
}

/// torchaudio's `_get_sinc_resample_kernel` for the Hann-windowed method:
/// `new` filters of `taps` float32 weights.
fn kernel(width: usize, taps: usize, orig: usize, new: usize, base_freq: f64) -> Vec<f32> {
    let pi = std::f32::consts::PI;
    let base_freq32 = base_freq as f32;
    let scale = (base_freq / orig as f64) as f32;
    let mut kernel = vec![0.0f32; new * taps];
    for filter in 0..new {
        for tap in 0..taps {
            // `arange` values are exact integers in float32; the division is
            // float32, as in the reference.
            let index = (tap as f32 - width as f32) / orig as f32;
            let mut t = -(filter as f32) / new as f32 + index;
            t *= base_freq32;
            t = t.clamp(-LOWPASS_FILTER_WIDTH as f32, LOWPASS_FILTER_WIDTH as f32);
            // The Hann window evaluated at the sample's offset, not on a
            // regular grid.
            let angle = t * pi / LOWPASS_FILTER_WIDTH as f32 / 2.0;
            let window = angle.cos() * angle.cos();
            t *= pi;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[filter * taps + tap] = sinc * (window * scale);
        }
    }
    kernel
}
