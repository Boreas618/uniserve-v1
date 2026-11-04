//! The video VAE's frame layer: the causal pad, the per-frame group statistics, and the fused
//! normalize-pad-activate that feeds a convolution.
//!
//! A video convolution sees a padded, normalized and activated frame. The reference's eager path
//! pads, reduces each frame's channel groups, and normalizes as three passes over the frame; the
//! operators here fuse them (specs/worker.md 9, "replicate pad"; `ops::frame`), so the layer reads
//! each frame once. Statistics stay FP32 whatever the activations' element type is; the padding is
//! replication or reflection, the video VAE's two modes.

use ops::TensorView;
use ops::descriptor::Float;
use ops::frame::{self, FrameNormPad as NormPadOp};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::weights::Weight;

/// The causal padding of a video convolution's input.
pub struct FramePad {
    reflect: bool,
}

impl FramePad {
    /// The pad that mirrors the frame's edge rows and columns (`reflect`) or replicates them.
    pub const fn new(reflect: bool) -> Self {
        Self { reflect }
    }

    /// Pad `values` `[batch, channels, frames, height, width]` into `out`, whose extra frames are
    /// zeros and whose extra rows and columns are mirrored or replicated.
    ///
    /// `top`, `left` and `front` are the padding before the source on each axis; the output's own
    /// extents give the padding after it.
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        out: TensorView,
        top: u64,
        left: u64,
        front: u64,
    ) -> Result<()> {
        let (block_pixels, block_channels) = frame::FramePadDescriptor::tiles(&values, &out);
        let descriptor = frame::FramePadDescriptor {
            dtype: values.dtype,
            reflect: self.reflect,
            block_pixels,
            block_channels,
        };
        let args = frame::FramePadArgs {
            values,
            out,
            top,
            left,
            front,
        };

        f.operate::<frame::FramePad>(&descriptor, args)
    }
}

/// The per-frame group statistics of a video convolution's input.
pub struct FrameMoments {
    groups: u64,
    eps: f32,
}

impl FrameMoments {
    /// The statistics of `groups` channel groups, with `eps` added to each variance.
    pub const fn new(groups: u64, eps: f32) -> Self {
        Self { groups, eps }
    }

    /// Reduce each frame's group means and reciprocal standard deviations.
    ///
    /// `values` is `[batch, channels, frames, height, width]` FP32 with each frame's pixels
    /// contiguous; `mean` and `rstd` are `[batch · frames, groups]` FP32.
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        mean: TensorView,
        rstd: TensorView,
    ) -> Result<()> {
        let descriptor = frame::FrameMomentsDescriptor {
            dtype: values.dtype,
            epsilon: Float::from(self.eps),
        };
        let args = frame::FrameMomentsArgs {
            values,
            mean,
            rstd,
            groups: self.groups,
        };

        f.operate::<frame::FrameMoments>(&descriptor, args)
    }
}

/// The padded, normalized and activated input of a video convolution.
pub struct FrameNormPad {
    groups: u64,
    weight: Option<Weight>,
    bias: Option<Weight>,
    reflect: bool,
    channels_last: bool,
}

impl FrameNormPad {
    /// The fused normalize-pad-activate of `groups` channel groups, with the learned affine
    /// parameters when the checkpoint has them.
    pub fn new(
        groups: u64,
        weight: Option<Weight>,
        bias: Option<Weight>,
        reflect: bool,
    ) -> Result<Self> {
        if weight.is_some() != bias.is_some() {
            return Err(Error::shape(
                "FrameNormPad",
                "the affine parameters come in a weight and a bias, or not at all".to_owned(),
            ));
        }
        Ok(Self {
            groups,
            weight,
            bias,
            reflect,
            channels_last: false,
        })
    }

    /// Write the padded frame channels-last (`[batch, frames, height, width, channels]`).
    pub fn with_channels_last(mut self) -> Self {
        self.channels_last = true;
        self
    }

    /// Normalize `values` by the frame statistics, pad and activate them into `out`.
    ///
    /// `values` is `[batch, channels, frames, height, width]` FP32 with each frame's pixels
    /// contiguous; `mean` and `rstd` are the `frame_moments` of the same frames; `out` is the padded
    /// frame in the activation's element type, channel-first or channels-last.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        mean: TensorView,
        rstd: TensorView,
        out: TensorView,
        top: u64,
        left: u64,
        front: u64,
    ) -> Result<()> {
        if out.shape.rank() != 5 {
            return Err(Error::shape(
                "FrameNormPad",
                format!("a padded frame is rank 5, not {:?}", out.shape.dims()),
            ));
        }
        // Channel-first: [batch, channels, frames, height, width]; channels-last:
        // [batch, frames, height, width, channels]. The layer states which, so a frame whose width
        // happens to equal its channel count is not guessed at.
        let channels = values.shape.dim(1);
        let (frames_axis, height_axis, width_axis) = if self.channels_last {
            (1, 2, 3)
        } else {
            (2, 3, 4)
        };
        let channel_axis = if self.channels_last { 4 } else { 1 };
        if out.shape.dim(channel_axis) != channels {
            return Err(Error::shape(
                "FrameNormPad",
                format!(
                    "an output of {} channels for a source of {channels}",
                    out.shape.dim(channel_axis)
                ),
            ));
        }
        if out.shape.dim(frames_axis) < values.shape.dim(2)
            || out.shape.dim(height_axis) < values.shape.dim(3)
            || out.shape.dim(width_axis) < values.shape.dim(4)
        {
            return Err(Error::shape(
                "FrameNormPad",
                "the output's frames, rows and columns cannot be smaller than the source's"
                    .to_owned(),
            ));
        }
        let single_pixel = out.shape.dim(height_axis) == 1 && out.shape.dim(width_axis) == 1;
        let descriptor = frame::FrameNormPadDescriptor {
            dtype: values.dtype,
            channels_last: self.channels_last,
            centered: self.weight.is_none(),
            single_pixel,
        };
        let args = frame::FrameNormPadArgs {
            values,
            mean,
            rstd,
            weight: self.weight.as_ref().map(Weight::view),
            bias: self.bias.as_ref().map(Weight::view),
            out,
            groups: self.groups,
            top,
            left,
            front,
            reflect: self.reflect,
        };

        f.operate::<NormPadOp>(&descriptor, args)
    }
}
