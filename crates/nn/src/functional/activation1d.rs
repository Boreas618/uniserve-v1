//! BigVGAN's alias-free activation: a SnakeBeta evaluated at `ratio` times the signal's rate,
//! between a Kaiser-windowed sinc upsample and the matching low-pass, as one operator.
//!
//! ```text
//! 1. replicate pad the signal by `kernel / ratio − 1` on each side
//! 2. a depthwise transposed convolution, stride `ratio`, the filter shared by every channel,
//!    times `ratio`: the ×ratio upsample
//! 3. crop `3 · (kernel − ratio) / 2` samples from each end, which leaves `length · ratio`
//! 4. SnakeBeta:  y = x + sin²(exp(alpha) · x) / (exp(beta) + 1e-9), per channel
//! 5. replicate pad by `kernel / 2 − 1` and `kernel / 2`, then a depthwise convolution at
//!    stride `ratio`: the ×ratio low-pass
//! ```
//!
//! The chain preserves the length. Each stage as its own kernel would hold two intermediates
//! of `ratio` times the signal, a wide vocoder stage's largest allocation by far; the operator
//! keeps the upsampled signal inside the kernel that produced it.

use ops::activation1d::{Activation1d, Args, BETA_FLOOR, Descriptor};
use tensor::{DType, Device, Tensor};

use super::{PDL, on_cpu};
use crate::error::{Error, Result};

/// Channels one kernel of the activation handles.
const BLOCK: u32 = 256;

/// The alias-free SnakeBeta of `x`, `[channels, length]` or `[batch, channels, length]`: a
/// tensor of `x`'s shape and element type.
///
/// `alpha` and `beta` are SnakeBeta's `[channels]` parameters in log space, as the checkpoint
/// stores them, in FP32; `filter_up` and `filter_down` are the `[1, 1, kernel]` Kaiser-sinc
/// filters of the upsample and the low-pass, in `x`'s element type; `ratio` is the up- and
/// down-sampling ratio.
///
/// # Errors
///
/// When `x` is not a signal of the parameters' channels, the filters are not one even kernel
/// no shorter than the ratio, the signal is empty, or the operator refuses its arguments.
pub fn activation1d(
    x: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    filter_up: &Tensor,
    filter_down: &Tensor,
    ratio: u32,
) -> Result<Tensor> {
    const OWNER: &str = "activation1d";

    let (batch, channels, length) = match *x.dims() {
        [channels, length] => (1, channels, length),
        [batch, channels, length] => (batch, channels, length),
        _ => {
            return Err(Error::shape(
                OWNER,
                format!(
                    "{:?} is neither [channels, samples] nor [batch, channels, samples]",
                    x.dims()
                ),
            ));
        }
    };
    let &[1, 1, kernel] = filter_up.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("a filter is [1, 1, kernel], not {:?}", filter_up.dims()),
        ));
    };
    let step = ratio as usize;
    if length == 0
        || channels == 0
        || ratio == 0
        || kernel < 2
        || !kernel.is_multiple_of(2)
        || kernel < step
        || filter_down.dims() != filter_up.dims()
        || alpha.dims() != [channels]
        || beta.dims() != [channels]
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "a signal of {:?} with {} and {} parameters and filters of {:?} and {:?} at \
                 ratio {ratio}",
                x.dims(),
                alpha.elem_count(),
                beta.elem_count(),
                filter_up.dims(),
                filter_down.dims()
            ),
        ));
    }
    let descriptor = Descriptor {
        dtype: x.dtype(),
        channels: u32::try_from(channels)
            .map_err(|_| Error::shape(OWNER, format!("{channels} channels")))?,
        kernel: u32::try_from(kernel)
            .map_err(|_| Error::shape(OWNER, format!("a filter of {kernel} taps")))?,
        ratio,
        block: BLOCK,
        pdl: PDL,
    };
    let expected = descriptor.output(length as u64)? as usize;

    if on_cpu(x) {
        // Every channel reads the same filter, so the depthwise convolutions are one
        // single-channel convolution over the channels as rows.
        let signal = x
            .to_dtype(DType::F32)?
            .reshape((batch * channels, 1, length))?;
        let (up, down) = (
            filter_up.to_dtype(DType::F32)?,
            filter_down.to_dtype(DType::F32)?,
        );

        let pad = descriptor.upsample_pad() as usize;
        let upsampled = signal
            .pad_with_same(2, pad, pad)?
            .conv_transpose1d(&up, 0, 0, step, 1, 1)?
            .broadcast_mul(&Tensor::new(ratio as f32, &Device::Cpu)?)?
            .narrow(2, descriptor.crop() as usize, length * step)?
            .reshape((batch, channels, length * step))?;

        let per_channel = |parameter: &Tensor| -> Result<Tensor> {
            Ok(parameter
                .to_dtype(DType::F32)?
                .exp()?
                .reshape((1, channels, 1))?)
        };
        let periodic = upsampled
            .broadcast_mul(&per_channel(alpha)?)?
            .sin()?
            .sqr()?
            .broadcast_div(&(per_channel(beta)? + f64::from(BETA_FLOOR))?)?;
        let activated = (upsampled + periodic)?;

        let (left, right) = descriptor.lowpass_pads();
        let filtered = activated
            .reshape((batch * channels, 1, length * step))?
            .pad_with_same(2, left as usize, right as usize)?
            .conv1d(&down, 0, step, 1, 1)?;
        return Ok(filtered.reshape(x.shape())?.to_dtype(x.dtype())?);
    }

    // A batch is the same channel set read again: one launch per batch row over one descriptor.
    let signal = x.contiguous()?.reshape((batch, channels, length))?;
    let out = ops::empty((batch, channels, expected), x.dtype())?;
    let length = u32::try_from(length)
        .map_err(|_| Error::shape(OWNER, format!("{length} samples exceed a launch value")))?;

    for row in 0..batch {
        ops::launch::<Activation1d>(
            &descriptor,
            Args {
                values: ops::operand(&signal.get(row)?)?,
                alpha: ops::operand(alpha)?,
                beta: ops::operand(beta)?,
                filter_up: ops::operand(filter_up)?,
                filter_down: ops::operand(filter_down)?,
                out: ops::operand(&out.get(row)?)?,
                length,
            },
        )?;
    }
    Ok(out.reshape(x.shape())?)
}
