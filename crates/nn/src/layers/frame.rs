//! The causal padding of a video convolution's input (`ops::frame`).
//!
//! A causal video convolution reads its input padded on both sides of height and width and
//! preceded by zero frames, so no output frame depends on a later input frame. The padding is
//! replication or reflection; a normalized input is padded by the
//! [`GroupNorm`](crate::layers::GroupNorm) that writes it.

use ops::TensorView;
use ops::frame;

use crate::error::Result;
use crate::forward::Forward;

/// The causal padding of a video convolution's input.
pub struct FramePad {
    reflect: bool,
}

impl FramePad {
    /// The pad that mirrors the frame's edge rows and columns (`reflect`) or replicates them.
    pub const fn new(reflect: bool) -> Self {
        Self { reflect }
    }

    /// The descriptor one pad launch takes.
    ///
    /// The block tiles are the operator's own split of the values' and the output's extents, so they
    /// are stated from the views rather than from an element type alone.
    fn descriptor(&self, values: TensorView, out: TensorView) -> frame::FramePadDescriptor {
        let (block_pixels, block_channels) = frame::FramePadDescriptor::tiles(&values, &out);
        frame::FramePadDescriptor {
            dtype: values.dtype,
            reflect: self.reflect,
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
        let descriptor = self.descriptor(values, out);
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
