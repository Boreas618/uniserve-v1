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

    /// The descriptor one pad launch takes, for a walk that asks a model which programs it launches
    /// with no device.
    ///
    /// The block tiles are the operator's own split of the values' and the output's extents, so they
    /// are stated from the views rather than from an element type alone.
    pub fn pad_descriptor(&self, values: TensorView, out: TensorView) -> frame::FramePadDescriptor {
        let (block_pixels, block_channels) = frame::FramePadDescriptor::tiles(&values, &out);
        Self::descriptor_of(values.dtype, self.reflect, block_pixels, block_channels)
    }

    /// The descriptor one pad launch takes from the numbers alone: a walk has the extents, not the
    /// views, and the block split turns on two of their facts (`FramePadDescriptor::tiles_of`).
    pub fn descriptor_of(
        dtype: ops::DType,
        reflect: bool,
        block_pixels: u32,
        block_channels: u32,
    ) -> frame::FramePadDescriptor {
        frame::FramePadDescriptor {
            dtype,
            reflect,
            block_pixels,
            block_channels,
        }
    }

    /// Pad `values` `[batch, channels, frames, height, width]` into `out`, whose extra frames are
    /// zeros and whose extra rows and columns are mirrored or replicated.
    ///
    /// `top`, `left` and `front` are the padding before the source on each axis; the output's own
    /// extents give the padding after it. Both operands may be stored channel-first or
    /// channels-last: the logical axes keep their meaning and the strides state the order
    /// (`ops::conv`'s rule), so a channels-last span hands the same call a permuted view.
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        out: TensorView,
        top: u64,
        left: u64,
        front: u64,
    ) -> Result<()> {
        let descriptor = self.pad_descriptor(values, out);
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

    /// The descriptor one moments launch takes, for a walk without a device.
    ///
    /// The reduction class is derived from the frame's own shape, so a launch can never name a class
    /// narrower than the reduction it asks for: `channels_per_group · pixels`, padded to a power of
    /// two because that is what the compiled program's bound is.
    pub fn moments_descriptor(&self, values: TensorView) -> frame::FrameMomentsDescriptor {
        Self::descriptor_of(
            values.dtype,
            values.shape.dim(1),
            values.shape.dim(3) * values.shape.dim(4),
            self.groups,
            self.eps,
        )
    }

    /// The descriptor one moments launch takes from the frame's own numbers: the channels, the pixels
    /// per frame, and the groups the reduction is classed by.
    pub fn descriptor_of(
        dtype: ops::DType,
        channels: u64,
        pixels: u64,
        groups: u64,
        eps: f32,
    ) -> frame::FrameMomentsDescriptor {
        frame::FrameMomentsDescriptor {
            dtype,
            reduction: frame::frame_reduction_class(channels / groups.max(1), pixels),
            epsilon: Float::from(eps),
        }
    }

    /// Reduce each frame's group means and reciprocal standard deviations.
    ///
    /// `values` is `[batch, channels, frames, height, width]` FP32 in either layout — the operator
    /// derives a frame's pixel step from the strides; `mean` and `rstd` are the `[batch · frames,
    /// groups]` FP32.
    pub fn forward(
        &self,
        f: &Forward,
        values: TensorView,
        mean: TensorView,
        rstd: TensorView,
    ) -> Result<()> {
        let descriptor = self.moments_descriptor(values);
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
        })
    }

    /// The descriptor one normalize-pad launch takes from numbers: the frame's channels and pixels,
    /// the groups its statistics are taken over, the storage order it writes, whether the affine
    /// parameters are present, and the padded frame's pixel count (one pixel is a class of its own).
    ///
    /// A walk has the extents and not the views, so it states the order; a launch reads it off
    /// `out`'s strides, which is where the order of a frame actually lives.
    pub fn descriptor_of(
        dtype: ops::DType,
        channels: u64,
        pixels: u64,
        groups: u64,
        channels_last: bool,
        affine: bool,
        padded_pixels: u64,
    ) -> frame::FrameNormPadDescriptor {
        frame::FrameNormPadDescriptor {
            dtype,
            // The same reduction `frame_moments` takes of the same frame.
            reduction: frame::frame_reduction_class(channels / groups.max(1), pixels),
            channels_last,
            centered: !affine,
            single_pixel: padded_pixels == 1,
        }
    }

    /// Normalize `values` by the frame statistics, pad and activate them into `out`.
    ///
    /// `values` is `[batch, channels, frames, height, width]` FP32 with each frame's pixels
    /// contiguous; `mean` and `rstd` are the `frame_moments` of the same frames; `out` is the padded
    /// frame in the activation's element type, in either storage order — its strides state which,
    /// and they select the compiled variant.
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
        // The axes are logical in either storage order — `[batch, channels, frames, height, width]`
        // — and the order lives in the strides, so one view of a channels-last buffer serves this
        // layer exactly as a channels-first one does.
        let channels = values.shape.dim(1);
        if out.shape.dim(1) != channels {
            return Err(Error::shape(
                "FrameNormPad",
                format!(
                    "an output of {} channels for a source of {channels}",
                    out.shape.dim(1)
                ),
            ));
        }
        if out.shape.dim(2) < values.shape.dim(2)
            || out.shape.dim(3) < values.shape.dim(3)
            || out.shape.dim(4) < values.shape.dim(4)
        {
            return Err(Error::shape(
                "FrameNormPad",
                "the output's frames, rows and columns cannot be smaller than the source's"
                    .to_owned(),
            ));
        }
        let padded_pixels = out.shape.dim(3) * out.shape.dim(4);
        let descriptor = Self::descriptor_of(
            values.dtype,
            values.shape.dim(1),
            values.shape.dim(3) * values.shape.dim(4),
            self.groups,
            // The variant follows the output's storage, which the view states; the operator checks
            // the two agree before it launches.
            frame::is_channels_last(&out),
            self.weight.is_some(),
            padded_pixels,
        );
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
