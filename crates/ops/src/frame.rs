//! The causal padding of a video convolution, and the normalization its pre-activation needs.
//!
//! A causal 3D convolution over `[batch, channels, frames, height, width]` reads its input padded on
//! both sides of height and width and preceded by `front` zero frames, so no output frame depends on
//! a later input frame. The pad is the eager `F.pad` composition's own: the rows and columns beside
//! the source are mirrored or replicated, and the frames before it are zero.
//!
//! ```text
//! frame_pad        out = pad(values)                         arbitrary input strides
//! frame_moments    mean, rstd = group statistics of each frame
//! frame_norm_pad   out = pad(silu(group_norm(values)))       moments read, one further pass
//! ```
//!
//! `frame_pad` is the "replicate pad" of specs/worker.md 9 and is the whole of the padding a video
//! VAE's convolutions need. `frame_moments` and `frame_norm_pad` are `frame.cu`'s two entries: the
//! first reduces each frame's channel groups with PyTorch's Welford tree, the second writes the
//! normalized, activated and padded input in one pass, which is the fused operator the design calls
//! for. Their launch shape is the package's (the entries launch their own blocks), so the call
//! structures below carry the operands and nothing about the grid.

use std::fmt;
use std::mem::{offset_of, size_of};

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The output columns one `frame_pad` program covers when the source and the output rows are
/// contiguous.
pub const ROW_PIXELS: u32 = 512;

/// The elements one `frame_pad` program handles when its rows are not contiguous.
pub const PAD_PIXELS: u32 = 32;

/// The channels one `frame_pad` program handles when its rows are not contiguous.
pub const PAD_CHANNELS: u32 = 64;

/// The elements one contiguous `frame_pad` program handles.
pub const ROW_ELEMENTS: u32 = 2048;

/// One compilation of `frame_pad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FramePadDescriptor {
    /// Element type of the values and the output.
    pub dtype: DType,
    /// Mirror the rows and columns beside the source instead of replicating them.
    pub reflect: bool,
    /// Output columns one program covers.
    pub block_pixels: u32,
    /// Channels one program covers.
    pub block_channels: u32,
}

impl fmt::Display for FramePadDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("reflect", self.reflect)
            .field("block_pixels", self.block_pixels)
            .field("block_channels", self.block_channels)
            .finish(formatter)
    }
}

impl FramePadDescriptor {
    /// The tile a launch of `values` into `out` uses.
    ///
    /// A source whose columns are contiguous with a contiguous output walks a whole output row at a
    /// time, up to [`ROW_PIXELS`]; anything else walks fixed tiles of columns and channels. The
    /// descriptor carries the result, so the caller and the compiled program agree on the variant.
    pub fn tiles(values: &TensorView, out: &TensorView) -> (u32, u32) {
        let contiguous_out = out.strides == Strides::row_major(out.shape);
        if values.strides.last() == 1 && contiguous_out {
            let columns = out.shape.dim(4);
            let pixels = ROW_PIXELS.min(columns.next_power_of_two().max(1) as u32);
            (pixels, ROW_ELEMENTS / pixels)
        } else {
            (PAD_PIXELS, PAD_CHANNELS)
        }
    }
}

/// The arguments of `frame_pad`.
pub struct FramePadArgs {
    /// `[batch, channels, frames, height, width]`, any strides, its columns unit-strided.
    pub values: TensorView,
    /// `[batch, channels, frames + front, height + top + bottom, width + left + right]`, contiguous
    /// or channels-last.
    pub out: TensorView,
    /// Output rows above the source.
    pub top: u64,
    /// Output columns left of the source.
    pub left: u64,
    /// Zero frames written before the source.
    pub front: u64,
}

/// The causal padding of a video convolution's input.
pub struct FramePad;

impl Op for FramePad {
    const NAME: &'static str = "frame_pad";

    type Descriptor = FramePadDescriptor;
    type Args<'a> = FramePadArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        args.values.require_rank(Self::NAME, 5)?;
        args.out.require_rank(Self::NAME, 5)?;
        if args.values.dtype != descriptor.dtype || args.out.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} values and output, found {} and {}",
                    descriptor.dtype, args.values.dtype, args.out.dtype
                ),
            ));
        }
        if args.values.strides.last() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the source columns are {} elements apart, the kernel reads them contiguously",
                    args.values.strides.last()
                ),
            ));
        }
        let channels_last = args.out.strides
            == Strides::new(&[
                args.out.shape.dim(1)
                    * args.out.shape.dim(2)
                    * args.out.shape.dim(3)
                    * args.out.shape.dim(4),
                args.out.shape.dim(4),
                args.out.shape.dim(1) * args.out.shape.dim(3) * args.out.shape.dim(4),
                args.out.shape.dim(1) * args.out.shape.dim(4),
                1,
            ]);
        if args.out.strides != Strides::row_major(args.out.shape) && !channels_last {
            return Err(Error::argument(
                Self::NAME,
                "the output is neither contiguous nor channels-last".to_owned(),
            ));
        }
        let (batch, channels) = (args.values.shape.dim(0), args.values.shape.dim(1));
        if args.out.shape.dim(0) != batch || args.out.shape.dim(1) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the output holds {} batches and {} channels, the source {batch} and {channels}",
                    args.out.shape.dim(0),
                    args.out.shape.dim(1)
                ),
            ));
        }
        let (height, width) = (args.values.shape.dim(3), args.values.shape.dim(4));
        let (out_width, out_height) = (args.out.shape.dim(4), args.out.shape.dim(3));
        if args.top >= out_height
            || args.left >= out_width
            || args.top + height > out_height
            || args.left + width > out_width
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a {height}x{width} source at ({}, {}) does not fit a {out_height}x{out_width} \
                     output",
                    args.top, args.left
                ),
            ));
        }
        if args.front != args.out.shape.dim(2) - args.values.shape.dim(2) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the output gains {} frames, the launch pads {}",
                    args.out.shape.dim(2) - args.values.shape.dim(2),
                    args.front
                ),
            ));
        }
        if descriptor.block_pixels == 0 || descriptor.block_channels == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a tile of zero columns or zero channels covers nothing".to_owned(),
            ));
        }
        // Every scalar the kernel takes is a 32-bit index, as the eager launch casts them.
        for (name, value) in [
            ("channels", channels),
            ("height", height),
            ("width", width),
            ("output frames", args.out.shape.dim(2)),
            ("output height", out_height),
            ("output width", out_width),
        ] {
            if value > i32::MAX as u64 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} {value} does not fit the kernel's int32"),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments.view(args.values).view(args.out);
        for stride in args.values.strides.as_slice() {
            arguments.i32(*stride as i32);
        }
        for stride in args.out.strides.as_slice() {
            arguments.i32(*stride as i32);
        }
        arguments
            .i32(height as i32)
            .i32(width as i32)
            .i32(args.out.shape.dim(2) as i32)
            .i32(out_height as i32)
            .i32(out_width as i32)
            .i32(channels as i32)
            .i32(args.top as i32)
            .i32(args.left as i32)
            .i32(args.front as i32);

        let columns = u64::from(descriptor.block_pixels);
        let rows = batch * args.out.shape.dim(2) * out_height;
        let grid = LaunchGrid::new([
            rows,
            out_width.div_ceil(columns),
            channels.div_ceil(u64::from(descriptor.block_channels)),
        ])?;
        kernel.launch(grid, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        FramePadArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Padding copies values or replicates them from their own row; two implementations agree bit
        // for bit, and the zero frames are zeros.
        Tolerance::exact()
    }
}

/// The largest message an entry writes into a call structure.
const MESSAGE: usize = 512;

/// The call structure of `frame_moments`, mirrored by the package's header.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FrameMomentsCall {
    /// The values' address.
    pub values: u64,
    /// Elements between batches.
    pub values_batch: i64,
    /// Elements between channels.
    pub values_channel: i64,
    /// Elements between frames.
    pub values_frame: i64,
    /// Frames per batch.
    pub frames: i64,
    /// Pixels per frame and channel, `height · width`.
    pub pixels: i64,
    /// Channels in one group.
    pub channels_per_group: i64,
    /// Groups per frame.
    pub groups: i64,
    /// The epsilon added to the variance before its reciprocal square root.
    pub epsilon: f32,
    /// The mean buffer's address, `[batch · frames, groups]` FP32.
    pub mean: u64,
    /// The reciprocal standard deviation buffer's address, the same shape.
    pub rstd: u64,
    /// The entry's NUL-terminated error message.
    pub error: [u8; MESSAGE],
}

impl NativeCall for FrameMomentsCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

const _: () = assert!(offset_of!(FrameMomentsCall, values) == 0);
const _: () = assert!(offset_of!(FrameMomentsCall, epsilon) == 64);
const _: () = assert!(offset_of!(FrameMomentsCall, mean) == 72);
const _: () = assert!(offset_of!(FrameMomentsCall, error) == 88);
const _: () = assert!(size_of::<FrameMomentsCall>() == 88 + MESSAGE);

/// One compilation of `frame_moments`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FrameMomentsDescriptor {
    /// Element type of the values and the statistics.
    pub dtype: DType,
    /// The epsilon added to the variance before its reciprocal square root.
    pub epsilon: crate::descriptor::Float,
}

impl fmt::Display for FrameMomentsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("epsilon", self.epsilon)
            .finish(formatter)
    }
}

/// The arguments of `frame_moments`.
pub struct FrameMomentsArgs {
    /// `[batch, channels, frames, height, width]` FP32, each frame's pixels contiguous.
    pub values: TensorView,
    /// `[batch · frames, groups]` FP32: receives each frame's group means.
    pub mean: TensorView,
    /// `[batch · frames, groups]` FP32: receives each frame's reciprocal standard deviations.
    pub rstd: TensorView,
    /// Channels per normalization group.
    pub groups: u64,
}

/// The per-frame group statistics of a video convolution's input.
pub struct FrameMoments;

impl Op for FrameMoments {
    const NAME: &'static str = "frame_moments";

    type Descriptor = FrameMomentsDescriptor;
    type Args<'a> = FrameMomentsArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        args.values.require_rank(Self::NAME, 5)?;
        args.values.require_row(Self::NAME, descriptor.dtype, 1)?;
        if args.values.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                "the reference kernel normalizes FP32 values".to_owned(),
            ));
        }
        if args.values.strides.get(4) != 1 || args.values.strides.get(3) != args.values.shape.dim(4)
        {
            return Err(Error::argument(
                Self::NAME,
                "a frame's pixels of one channel are not contiguous".to_owned(),
            ));
        }
        let (batch, channels, frames) = (
            args.values.shape.dim(0),
            args.values.shape.dim(1),
            args.values.shape.dim(2),
        );
        if args.groups == 0 || channels % args.groups != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{channels} channels do not divide into {} groups",
                    args.groups
                ),
            ));
        }
        let shape = Shape::new(&[batch * frames, args.groups]);
        for (name, view) in [("mean", &args.mean), ("rstd", &args.rstd)] {
            view.require_shape(Self::NAME, shape)?;
            view.require_row(Self::NAME, DType::F32, args.groups)?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} buffer is not FP32"),
                ));
            }
        }

        let pixels = args.values.shape.dim(3) * args.values.shape.dim(4);
        let mut call = FrameMomentsCall {
            values: args.values.ptr.address(),
            values_batch: args.values.strides.get(0) as i64,
            values_channel: args.values.strides.get(1) as i64,
            values_frame: args.values.strides.get(2) as i64,
            frames: frames as i64,
            pixels: pixels as i64,
            channels_per_group: (channels / args.groups) as i64,
            groups: args.groups as i64,
            epsilon: f32::from_bits(descriptor.epsilon.0 as u32),
            mean: args.mean.ptr.address(),
            rstd: args.rstd.ptr.address(),
            error: [0; MESSAGE],
        };
        // SAFETY: the entry is this package's `frame_moments`, whose header mirrors
        // `FrameMomentsCall` field for field.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        FrameMomentsArgs {
            mean: twins.shadow_of(args.mean),
            rstd: twins.shadow_of(args.rstd),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The moments reduce `channels_per_group · pixels` values in FP32 and are consumed in FP32:
        // the bound is the reduction's own rounding, taken at the widest frame a launch can carry.
        let _ = descriptor;
        Tolerance::rounding(1 << 16, DType::F32, DType::F32)
    }
}

/// One compilation of `frame_norm_pad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FrameNormPadDescriptor {
    /// Element type of the values, the parameters and the output.
    pub dtype: DType,
    /// The output's storage: channel-first (`false`) or channels-last (`true`).
    pub channels_last: bool,
    /// Normalize with the affine parameters (`false`), or with none (`true`).
    pub centered: bool,
    /// Every frame holds a single pixel, the reduction's shortest path.
    pub single_pixel: bool,
}

impl fmt::Display for FrameNormPadDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels_last", self.channels_last)
            .field("centered", self.centered)
            .field("single_pixel", self.single_pixel)
            .finish(formatter)
    }
}

/// The call structure of `frame_norm_pad`, mirrored by the package's header.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FrameNormPadCall {
    /// The values' address.
    pub values: u64,
    /// Elements between batches.
    pub values_batch: i64,
    /// Elements between channels.
    pub values_channel: i64,
    /// Elements between frames.
    pub values_frame: i64,
    /// The per-frame group means' address.
    pub mean: u64,
    /// The per-frame reciprocal standard deviations' address.
    pub rstd: u64,
    /// The weight's address, or zero when the kernel normalizes without one.
    pub weight: u64,
    /// The bias's address, or zero when the kernel normalizes without one.
    pub bias: u64,
    /// The padded output's address.
    pub out: u64,
    /// Elements between output batches.
    pub out_batch: i64,
    /// Elements between output channels.
    pub out_channel: i64,
    /// Elements between output frames.
    pub out_frame: i64,
    /// Elements between output rows.
    pub out_height: i64,
    /// Elements between output columns.
    pub out_width: i64,
    /// Channels of the source.
    pub channels: i64,
    /// Frames of the source.
    pub frames: i64,
    /// Rows of the source.
    pub height: i64,
    /// Columns of the source.
    pub width: i64,
    /// Frames of the output.
    pub out_frames: i64,
    /// Rows of the output.
    pub out_rows: i64,
    /// Columns of the output.
    pub out_columns: i64,
    /// Channels in one group.
    pub channels_per_group: i64,
    /// Groups per frame.
    pub groups: i64,
    /// Output rows above the source.
    pub top: i64,
    /// Output columns left of the source.
    pub left: i64,
    /// Zero frames written before the source.
    pub front: i64,
    /// Mirror the padded rows and columns instead of replicating them.
    pub reflect: i32,
    /// The normalization branch the kernel takes.
    pub normalization: i32,
    /// The entry's NUL-terminated error message.
    pub error: [u8; MESSAGE],
}

impl NativeCall for FrameNormPadCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

const _: () = assert!(offset_of!(FrameNormPadCall, values) == 0);
const _: () = assert!(offset_of!(FrameNormPadCall, weight) == 48);
const _: () = assert!(offset_of!(FrameNormPadCall, out) == 64);
const _: () = assert!(offset_of!(FrameNormPadCall, channels) == 112);
const _: () = assert!(offset_of!(FrameNormPadCall, reflect) == 208);
const _: () = assert!(offset_of!(FrameNormPadCall, error) == 216);
const _: () = assert!(size_of::<FrameNormPadCall>() == 216 + MESSAGE);

/// The arguments of `frame_norm_pad`.
pub struct FrameNormPadArgs {
    /// `[batch, channels, frames, height, width]` FP32, each frame's pixels contiguous.
    pub values: TensorView,
    /// `[batch · frames, groups]` FP32: the frame's group means.
    pub mean: TensorView,
    /// `[batch · frames, groups]` FP32: the frame's reciprocal standard deviations.
    pub rstd: TensorView,
    /// `[channels]` FP32: the learned weight, present exactly when the descriptor normalizes with
    /// the affine parameters.
    pub weight: Option<TensorView>,
    /// `[channels]` FP32: the learned bias, on the same condition.
    pub bias: Option<TensorView>,
    /// The padded output, contiguous or channels-last.
    pub out: TensorView,
    /// Channels per normalization group.
    pub groups: u64,
    /// Output rows above the source.
    pub top: u64,
    /// Output columns left of the source.
    pub left: u64,
    /// Zero frames written before the source.
    pub front: u64,
    /// Mirror the padded rows and columns instead of replicating them.
    pub reflect: bool,
}

/// The padded, normalized and activated input of a video convolution.
pub struct FrameNormPad;

impl Op for FrameNormPad {
    const NAME: &'static str = "frame_norm_pad";

    type Descriptor = FrameNormPadDescriptor;
    type Args<'a> = FrameNormPadArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        args.values.require_rank(Self::NAME, 5)?;
        args.out.require_rank(Self::NAME, 5)?;
        if args.values.dtype != DType::F32 || args.out.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                "the reference kernel normalizes and stores FP32 values".to_owned(),
            ));
        }
        if args.values.strides.get(4) != 1 || args.values.strides.get(3) != args.values.shape.dim(4)
        {
            return Err(Error::argument(
                Self::NAME,
                "a frame's pixels of one channel are not contiguous".to_owned(),
            ));
        }
        let (batch, channels, frames) = (
            args.values.shape.dim(0),
            args.values.shape.dim(1),
            args.values.shape.dim(2),
        );
        let (height, width) = (args.values.shape.dim(3), args.values.shape.dim(4));
        if args.out.shape.dim(0) != batch || args.out.shape.dim(1) != channels {
            return Err(Error::argument(
                Self::NAME,
                "the output's batch and channels differ from the source's".to_owned(),
            ));
        }
        let channels_last = descriptor.channels_last;
        let expected = if channels_last {
            Strides::new(&[
                channels * args.out.shape.dim(2) * args.out.shape.dim(3) * args.out.shape.dim(4),
                args.out.shape.dim(4),
                channels * args.out.shape.dim(3) * args.out.shape.dim(4),
                channels * args.out.shape.dim(4),
                1,
            ])
        } else {
            Strides::row_major(args.out.shape)
        };
        if args.out.strides != expected {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor compiles for {} output storage, the view is {:?}",
                    if channels_last {
                        "channels-last"
                    } else {
                        "contiguous"
                    },
                    args.out.strides.as_slice()
                ),
            ));
        }
        if args.out.shape.dim(3) != height + args.top
            || args.out.shape.dim(4) != width + args.left
            || args.out.shape.dim(2) != frames + args.front
        {
            return Err(Error::argument(
                Self::NAME,
                "the output's trailing extents do not follow from the padding".to_owned(),
            ));
        }
        if args.groups == 0 || channels % args.groups != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{channels} channels do not divide into {} groups",
                    args.groups
                ),
            ));
        }
        if descriptor.single_pixel != (height * width == 1) {
            return Err(Error::argument(
                Self::NAME,
                "the descriptor's single-pixel branch does not match the frames' extent".to_owned(),
            ));
        }
        let shape = Shape::new(&[batch * frames, args.groups]);
        for (name, view) in [("mean", &args.mean), ("rstd", &args.rstd)] {
            view.require_shape(Self::NAME, shape)?;
            view.require_row(Self::NAME, DType::F32, args.groups)?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} buffer is not FP32"),
                ));
            }
        }
        let affine = !descriptor.centered;
        for (name, parameter) in [("weight", &args.weight), ("bias", &args.bias)] {
            match (parameter, affine) {
                (Some(view), true) => {
                    view.require_shape(Self::NAME, Shape::new(&[channels]))?;
                    view.require_row(Self::NAME, DType::F32, channels)?;
                    if view.dtype != DType::F32 {
                        return Err(Error::argument(
                            Self::NAME,
                            format!("the {name} is not FP32"),
                        ));
                    }
                }
                (None, false) => {}
                (None, true) => {
                    return Err(Error::argument(
                        Self::NAME,
                        format!("the descriptor normalizes with a {name} but none was given"),
                    ));
                }
                (Some(_), false) => {
                    return Err(Error::argument(
                        Self::NAME,
                        format!("a {name} was given but the descriptor applies none"),
                    ));
                }
            }
        }
        // The reference entry addresses rows and channels with 32-bit indices.
        let rows = batch * args.out.shape.dim(2) * args.out.shape.dim(3);
        if rows >= 1 << 31 || channels >= 1 << 31 {
            return Err(Error::argument(
                Self::NAME,
                format!("{rows} output rows and {channels} channels do not fit 32-bit indices"),
            ));
        }

        // The kernel's branch: one pixel per frame, the centered reduction, or the fused affine one.
        let normalization = if descriptor.single_pixel {
            2
        } else if descriptor.centered {
            1
        } else {
            0
        };
        let mut call = FrameNormPadCall {
            values: args.values.ptr.address(),
            values_batch: args.values.strides.get(0) as i64,
            values_channel: args.values.strides.get(1) as i64,
            values_frame: args.values.strides.get(2) as i64,
            mean: args.mean.ptr.address(),
            rstd: args.rstd.ptr.address(),
            weight: args
                .weight
                .map(|view| view.ptr.address())
                .unwrap_or_default(),
            bias: args
                .bias
                .map(|view| view.ptr.address())
                .unwrap_or_default(),
            out: args.out.ptr.address(),
            out_batch: args.out.strides.get(0) as i64,
            out_channel: args.out.strides.get(1) as i64,
            out_frame: args.out.strides.get(2) as i64,
            out_height: args.out.strides.get(3) as i64,
            out_width: args.out.strides.get(4) as i64,
            channels: channels as i64,
            frames: frames as i64,
            height: height as i64,
            width: width as i64,
            out_frames: args.out.shape.dim(2) as i64,
            out_rows: args.out.shape.dim(3) as i64,
            out_columns: args.out.shape.dim(4) as i64,
            channels_per_group: (channels / args.groups) as i64,
            groups: args.groups as i64,
            top: args.top as i64,
            left: args.left as i64,
            front: args.front as i64,
            reflect: i32::from(args.reflect),
            normalization,
            error: [0; MESSAGE],
        };
        // SAFETY: the entry is this package's `frame_norm_pad`, whose header mirrors
        // `FrameNormPadCall` field for field.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        FrameNormPadArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The moments come from the same reduction as `frame_moments`; the normalization, the SiLU
        // and the padded store round in FP32.
        let _ = descriptor;
        Tolerance::rounding(1 << 16, DType::F32, DType::F32)
    }
}
