//! `position_interpolate`: resample a learned position table to another grid, on the device.
//!
//! A vision tower is trained at one image resolution and runs at several. Its position table is a
//! learned `[in_side · in_side, channels]` tensor, and the reference resamples it bilinearly to the
//! grid the image actually has
//! (`uniserve/models/…`: `F.interpolate(pos_embed.reshape(1, side, side, -1), scale, mode="bilinear",
//! align_corners=False)`), per image, inside the forward pass. That cannot be host math: the table
//! lives on the device (it is a weight), and a host round trip per image would synchronize the
//! stream in the middle of a captured graph.
//!
//! The host still does what it is good at — four tap indices and four `f32` weights per output point
//! — and the program gathers and weights:
//!
//! ```text
//! out[j] = Σ_{t < 4} weight[j, t] · table[indices[j, t]]
//! ```
//!
//! **Which resampling rule this is, the program does not know.** Corner alignment or half-pixel
//! alignment, edge clamping or edge extension — those are the caller's convention, and each tower has
//! its own: H3's vision tower resamples with corner alignment and a clamped last sample
//! (`uniserve/nn/vision/position.py:167-200`, `torch.linspace(0, extent - 1, size)` with
//! `upper = (lower + 1).clamp(max = extent - 1)`), which is *not* the `align_corners = false` rule
//! any one framework defaults to. So the caller computes the four tap indices and four f32 weights
//! for every output point, once per grid, and the program applies them: no alignment rule, no
//! clamping, no division by a side — a shared operator must not bake one caller's convention in.
//!
//! The program multiplies in FP32 and rounds once into the output's dtype. The weights of one output
//! point are the caller's normalization too: a clamped point is not a uniform four-way average.
//!
//! The tolerance is a four-tap weighted sum in FP32: four products and their accumulation, so the
//! bound is [`Tolerance::dot`] over four terms with the output's own rounding.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Taps one output point reads: four for a bilinear resample of a two-dimensional grid.
pub const TAPS: u64 = 4;

/// One compilation of `position_interpolate`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the table and of the output.
    pub dtype: DType,
    /// Elements per row, the table's feature width.
    pub channels: u32,
    /// Points per side of the learned table's grid: square, because the checkpoint's table is.
    pub in_side: u32,

    /// Elements one thread block handles.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("in_side", self.in_side)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one resample.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[in_side · in_side, channels]` the learned table.
    pub table: TensorView,
    /// `[points, 4]` i32 tap indices into the table's points.
    ///
    /// Every index must be below `in_side²`: the program does not range-check a table it cannot read
    /// on the host, so building taps inside the table is the caller's obligation and a table that
    /// violates it reads a neighbouring point rather than failing.
    pub indices: TensorView,
    /// `[points, 4]` FP32 tap weights, normalized by the caller.
    pub weights: TensorView,
    /// `[points, channels]` the resampled table.
    pub out: TensorView,
}

/// Bilinear resample of a learned position table.
pub struct PositionInterpolate;

impl Op for PositionInterpolate {
    const NAME: &'static str = "position_interpolate";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let channels = u64::from(descriptor.channels);
        // The output's point count is the args' own extent: the grid geometry is the caller's, and
        // the program never divides by a side.
        let points = args.out.rows();

        args.table.require_shape(
            Self::NAME,
            Shape::new(&[
                u64::from(descriptor.in_side) * u64::from(descriptor.in_side),
                channels,
            ]),
        )?;
        args.table
            .require_row(Self::NAME, descriptor.dtype, channels)?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[points, channels]))?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, channels)?;
        args.indices
            .require_shape(Self::NAME, Shape::new(&[points, TAPS]))?;
        args.weights
            .require_shape(Self::NAME, Shape::new(&[points, TAPS]))?;
        if args.indices.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 tap indices, found {}", args.indices.dtype),
            ));
        }
        if args.weights.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected f32 tap weights, found {}", args.weights.dtype),
            ));
        }
        // The program walks the destination as rows of `channels` elements: a row that is not that
        // wide, or a last axis that is not unit-stride, would make the tap arithmetic address the
        // wrong elements.
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the output is written row-major; found strides {}",
                    args.out.strides
                ),
            ));
        }
        if args.table.strides != Strides::row_major(args.table.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the table is read row-major; found strides {}",
                    args.table.strides
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.table)
            .view(args.indices)
            .view(args.weights)
            .view(args.out)
            .u32(u32::try_from(points).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{points} output points do not fit a kernel parameter"),
                )
            })?)
            .u32(descriptor.channels)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One block per (output point, element block): the four taps of a point are read and summed
        // by one block, so the sum's order is fixed and two blocks never write the same element.
        kernel.launch(
            LaunchGrid::new([points, channels.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Four weighted products accumulated in FP32, then one rounding into the output's dtype.
        Tolerance::dot(TAPS, DType::F32, descriptor.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::BF16,
            channels: 1280,
            in_side: 48,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_key_states_the_learned_table_and_not_the_target_grid() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_bf16_channels_1280_in_side_48_block_1024_pdl_false"
        );
        // The resample of another grid is another compilation: its index and weight tables differ.
        // The target grid is the args' extent, so two grids of one table are one program; a
        // different learned table is another (its taps address a different range).
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                in_side: 24,
                ..descriptor()
            }
            .to_string()
        );
    }

    #[test]
    fn the_bound_is_the_four_tap_dot() {
        assert_eq!(
            PositionInterpolate::tolerance(&descriptor()),
            Tolerance::dot(4, DType::F32, DType::BF16)
        );
    }
}
