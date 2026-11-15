//! The normalization a causal video convolution's pre-activation needs: the group statistics of
//! each frame, and the normalized, activated and padded frame written in one pass.
//!
//! ```text
//! mean, rstd = statistics of each frame's channel groups     frame_moments (FP32)
//! out        = pad(silu(norm(x) · weight + bias))             frame_norm_pad
//! ```
//!
//! A causal video VAE normalizes every frame by itself, so a group's statistics reduce its
//! channels over one frame's pixels and not over the clip: the variance is the mean of the
//! squared deviations, and the reciprocal deviation is `rsqrt(variance + eps)`. The convolution
//! that follows reads its input activated and causally padded
//! ([`causal_pad`](super::causal_pad)), so the second launch writes all three steps: two reads of
//! the frame where the composition makes five.

use ops::descriptor::Float;
use ops::frame::{
    FrameMoments, FrameMomentsArgs, FrameMomentsDescriptor, FrameNormPad, FrameNormPadArgs,
    FrameNormPadDescriptor, frame_reduction_class,
};
use tensor::{DType, Device, Tensor};

use super::conv::{Padding, causal_pad, packed};
use super::{Fused, compute};
use crate::error::{Error, Result};

/// The per-frame group normalization of `x`, its SiLU and its causal pad, as one operator:
/// `pad(silu(norm(x) · weight + bias))`.
///
/// `x` is `[n, c, d, h, w]` in FP32. Each frame's channels are normalized in `groups` groups
/// over that frame's pixels with variance floor `eps`; `affine` is the learned `(weight, bias)`
/// pair, one element per channel each, or `None` for a normalization without one. The result is
/// padded as [`causal_pad`] pads it (`spatial` rows and columns around each frame filled as
/// `padding` states, `front` zero frames before the first):
/// `[n, c, d + front, h + top + bottom, w + left + right]` in FP32, packed channel-first.
///
/// # Errors
///
/// When `x` is not an FP32 frame, the groups do not divide its channels, a parameter is not one
/// FP32 element per channel, or an operator refuses its arguments.
pub fn frame_norm_pad(
    x: &Tensor,
    affine: Option<(&Tensor, &Tensor)>,
    groups: u32,
    eps: f32,
    spatial: [usize; 4],
    front: usize,
    padding: Padding,
) -> Result<Tensor> {
    const OWNER: &str = FrameNormPadCall::OWNER;

    let &[_, channels, _, _, _] = x.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("a frame is rank 5, not {:?}", x.dims()),
        ));
    };
    let group_count = groups as usize;
    if groups == 0 || !channels.is_multiple_of(group_count) || x.dtype() != DType::F32 {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} frames of {channels} channels do not normalize in {groups} FP32 groups",
                ops::abi::name(x.dtype())
            ),
        ));
    }
    if let Some((weight, bias)) = affine {
        for (name, parameter) in [("weight", weight), ("bias", bias)] {
            if parameter.dims() != [channels] || parameter.dtype() != DType::F32 {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "the {name} is {} {:?}, and {channels} channels take one FP32 element \
                         each",
                        ops::abi::name(parameter.dtype()),
                        parameter.dims()
                    ),
                ));
            }
        }
    }
    compute(FrameNormPadCall {
        x,
        affine,
        groups,
        eps,
        spatial,
        front,
        padding,
    })
}

/// A checked call of [`frame_norm_pad`].
struct FrameNormPadCall<'t> {
    x: &'t Tensor,
    affine: Option<(&'t Tensor, &'t Tensor)>,
    groups: u32,
    eps: f32,
    spatial: [usize; 4],
    front: usize,
    padding: Padding,
}

impl FrameNormPadCall<'_> {
    const OWNER: &'static str = "frame_norm_pad";
}

impl Fused for FrameNormPadCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            x,
            affine,
            groups,
            eps,
            spatial,
            front,
            padding,
        } = self;
        let (batches, channels, frames, height, width) = x.dims5()?;
        let group_count = groups as usize;
        let group_channels = channels / group_count;
        let pixels = height * width;

        // Each frame's groups as rows: `[n, d, groups, channels per group · pixels]`.
        let grouped = x
            .reshape((batches, group_count, group_channels, frames, pixels))?
            .permute((0, 3, 1, 2, 4))?
            .reshape((batches, frames, group_count, group_channels * pixels))?;
        let mean = grouped.mean_keepdim(3)?;
        let variance = grouped.broadcast_sub(&mean)?.sqr()?.mean_keepdim(3)?;
        let rstd = (variance + f64::from(eps))?.sqrt()?.recip()?;

        // The statistics of each channel's group, over `[n, c, d, 1, 1]`.
        let per_channel = |statistic: &Tensor| -> Result<Tensor> {
            Ok(statistic
                .reshape((batches, frames, group_count, 1))?
                .permute((0, 2, 3, 1))?
                .broadcast_as((batches, group_count, group_channels, frames))?
                .reshape((batches, channels, frames, 1, 1))?)
        };
        let (mean, rstd) = (per_channel(&mean)?, per_channel(&rstd)?);
        let over_channels = |parameter: &Tensor| parameter.reshape((1, channels, 1, 1, 1));

        // The kernel's three branches, each in the order it evaluates: the fused parameters
        // `a = rstd · weight`, `b = −a · mean + bias` and `a · x + b`; the centred form without
        // parameters; and the centred form with them where a frame is one pixel.
        let normalized = match affine {
            Some((weight, bias)) if pixels > 1 => {
                let scale = rstd.broadcast_mul(&over_channels(weight)?)?;
                let shift = scale
                    .neg()?
                    .broadcast_mul(&mean)?
                    .broadcast_add(&over_channels(bias)?)?;

                x.broadcast_mul(&scale)?.broadcast_add(&shift)?
            }
            Some((weight, bias)) => x
                .broadcast_sub(&mean)?
                .broadcast_mul(&rstd)?
                .broadcast_mul(&over_channels(weight)?)?
                .broadcast_add(&over_channels(bias)?)?,
            None => x.broadcast_sub(&mean)?.broadcast_mul(&rstd)?,
        };
        causal_pad(&normalized.silu()?, spatial, front, padding)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            x,
            affine,
            groups,
            eps,
            spatial,
            front,
            padding,
        } = self;
        let (batches, channels, frames, height, width) = x.dims5()?;
        let group_count = groups as usize;
        let group_channels = channels / group_count;
        let pixels = height * width;

        let [top, bottom, left, right] = spatial;
        let values = packed(x)?;
        let statistics = (batches * frames, group_count);
        let mean = ops::empty(statistics, DType::F32)?;
        let rstd = ops::empty(statistics, DType::F32)?;
        let out = ops::empty(
            (
                batches,
                channels,
                frames + front,
                height + top + bottom,
                width + left + right,
            ),
            DType::F32,
        )?;

        // Both launches take the same reduction: a group's channels by a frame's pixels,
        // padded to the power of two the kernel's tolerance is stated for.
        let reduction = frame_reduction_class(group_channels as u64, pixels as u64);

        ops::launch::<FrameMoments>(
            &FrameMomentsDescriptor {
                dtype: x.dtype(),
                reduction,
                epsilon: Float::from(eps),
            },
            FrameMomentsArgs {
                values: ops::operand(&values)?,
                mean: ops::operand(&mean)?,
                rstd: ops::operand(&rstd)?,
                groups: u64::from(groups),
            },
        )?;
        ops::launch::<FrameNormPad>(
            &FrameNormPadDescriptor {
                dtype: x.dtype(),
                reduction,
                channels_last: false,
                centered: affine.is_none(),
                single_pixel: pixels == 1,
            },
            FrameNormPadArgs {
                values: ops::operand(&values)?,
                mean: ops::operand(&mean)?,
                rstd: ops::operand(&rstd)?,
                weight: affine.map(|(weight, _)| ops::operand(weight)).transpose()?,
                bias: affine.map(|(_, bias)| ops::operand(bias)).transpose()?,
                out: ops::operand(&out)?,
                groups: u64::from(groups),
                top: top as u64,
                left: left as u64,
                front: front as u64,
                reflect: padding == Padding::Reflect,
            },
        )?;
        Ok(out)
    }
}
