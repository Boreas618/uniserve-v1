//! Convolutions through cuDNN with torch's arguments, and the causal padding of a video
//! convolution's input.
//!
//! [`Conv1d`], [`ConvTranspose1d`] and [`Conv3d`] take the arguments their torch modules take —
//! channels, kernel, stride, zero padding, dilation, groups, bias — as one [`ConvSpec`], and load
//! the module's `weight` (or its weight-normalized `weight_v`/`weight_g` pair, folded once) and
//! `bias` from a [`Scope`]. The spec's `tf32` flag is the model's numerical declaration for an FP32
//! convolution, and `channels_last` the storage of a 3-d convolution's operands.
//!
//! A causal video convolution reads an input padded beyond what a zero padding states: the spatial
//! border mirrored or replicated ([`Padding`]) and zero frames in front, so no output frame reads a
//! later input frame. [`CausalPad`] is that padding, launched on its own (`frame_pad`) or fused
//! into the [`GroupNorm`](crate::layers::GroupNorm) that writes a normalized input.

use ops::conv::{
    Args, Conv1d as Conv1dOp, Conv3d as Conv3dOp, ConvTranspose1d as ConvTranspose1dOp, Descriptor,
    Geometry,
};
use ops::frame;
use ops::{Shape, Strides, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Bias;
use crate::weights::{Scope, Weight};

/// How a causal pad fills the rows and columns beyond a frame's edge: torch's `padding_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Padding {
    /// Mirror the rows and columns inside the edge (`reflect`).
    Reflect,
    /// Repeat the edge row and column (`replicate`).
    Replicate,
}

/// The causal padding of a video convolution's input: `top` rows, `left` columns and `front` zero
/// frames before the source, the border filled as `padding` states.
///
/// The output view's own extents give the padding after the source. The frames in front are zeros,
/// so no output frame of the convolution that reads the padded input depends on a later input
/// frame.
#[derive(Clone, Copy, Debug)]
pub struct CausalPad {
    /// Rows before the source.
    pub top: u32,
    /// Columns before the source.
    pub left: u32,
    /// Zero frames before the source.
    pub front: u32,
    /// How the rows and columns beyond the edge are filled.
    pub padding: Padding,
}

impl CausalPad {
    /// Whether the border mirrors rather than replicates: the operators' flag.
    pub(crate) fn reflect(&self) -> bool {
        self.padding == Padding::Reflect
    }

    /// Pad `values` `[batch, channels, frames, height, width]` into `out`, whose extra frames are
    /// zeros and whose extra rows and columns follow [`Self::padding`].
    ///
    /// Both operands may be stored channel-first or channels-last: the logical axes keep their
    /// meaning and the strides state the order (`ops::conv`'s rule).
    ///
    /// # Errors
    ///
    /// The operator's refusal: operands of another rank or channel count, or an output smaller than
    /// the source and its leading padding.
    pub fn forward(&self, f: &Forward, values: TensorView, out: TensorView) -> Result<()> {
        // The block tiles are the operator's own split of the values' and the output's extents.
        let (block_pixels, block_channels) = frame::FramePadDescriptor::tiles(&values, &out);
        let descriptor = frame::FramePadDescriptor {
            dtype: values.dtype,
            reflect: self.reflect(),
            block_pixels,
            block_channels,
        };
        f.operate::<frame::FramePad>(
            &descriptor,
            frame::FramePadArgs {
                values,
                out,
                top: u64::from(self.top),
                left: u64::from(self.left),
                front: u64::from(self.front),
            },
        )
    }
}

/// A convolution's torch arguments over `N` spatial axes.
///
/// [`ConvSpec::new`] states torch's defaults — stride one, no padding, no dilation, one group, a
/// bias — and a caller overrides the fields its module states with struct update syntax.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConvSpec<const N: usize> {
    /// Input channels.
    pub in_channels: u32,
    /// Output channels.
    pub out_channels: u32,
    /// The filter's extent along each spatial axis.
    pub kernel: [u32; N],
    /// The step between output positions along each spatial axis.
    pub stride: [u32; N],
    /// The zero padding before (and after) the input along each spatial axis.
    pub padding: [u32; N],
    /// The spacing between filter taps along each spatial axis.
    pub dilation: [u32; N],
    /// How many channel groups the filter is split into.
    pub groups: u32,
    /// Whether the module states a `bias`.
    pub bias: Bias,
    /// Whether the checkpoint stores the weight as torch's weight-norm pair, folded at load.
    pub weight_norm: bool,
    /// Whether cuDNN may use TF32 for an FP32 convolution.
    pub tf32: bool,
    /// Whether the operands are stored channels-last (3-d only).
    pub channels_last: bool,
}

impl<const N: usize> ConvSpec<N> {
    /// `in_channels` to `out_channels` with a `kernel` filter and torch's other defaults.
    pub const fn new(in_channels: u32, out_channels: u32, kernel: [u32; N]) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel,
            stride: [1; N],
            padding: [0; N],
            dilation: [1; N],
            groups: 1,
            bias: Bias::Present,
            weight_norm: false,
            tf32: false,
            channels_last: false,
        }
    }

    /// The operator's three-axis statement of the spec: a 1-d convolution's length is the first
    /// spatial axis, and its other two axes are a filter of one with no padding.
    fn geometry(&self) -> Geometry {
        let lift = |values: &[u32; N], fill: u32| {
            let mut axes = [fill; 3];
            axes[..N].copy_from_slice(values);
            axes
        };
        Geometry {
            kernel: lift(&self.kernel, 1),
            stride: lift(&self.stride, 1),
            padding: lift(&self.padding, 0),
            dilation: lift(&self.dilation, 1),
            groups: self.groups,
        }
    }

    /// Refuse a spec no convolution has.
    fn check(&self, owner: &'static str) -> Result<()> {
        let positive = |values: &[u32; N]| values.iter().all(|value| *value > 0);
        if self.groups == 0
            || self.in_channels == 0
            || self.out_channels == 0
            || !self.in_channels.is_multiple_of(self.groups)
            || !self.out_channels.is_multiple_of(self.groups)
            || !positive(&self.kernel)
            || !positive(&self.stride)
            || !positive(&self.dilation)
        {
            return Err(Error::shape(
                owner,
                format!("{self:?} is not a convolution"),
            ));
        }
        if self.channels_last && N != 3 {
            return Err(Error::shape(
                owner,
                "only a 3-d convolution stores its operands channels-last".to_owned(),
            ));
        }
        Ok(())
    }
}

/// The convolution state the three layers share: the weights, the spec and the operator's rank.
struct Convolution<const N: usize> {
    weight: Weight,
    bias: Option<Weight>,
    spec: ConvSpec<N>,
}

impl<const N: usize> Convolution<N> {
    /// The module at `scope` of `filter` extents (torch's weight shape for the layer).
    fn load(
        owner: &'static str,
        scope: &mut Scope<'_, '_>,
        spec: ConvSpec<N>,
        filter: Vec<u64>,
    ) -> Result<Self> {
        spec.check(owner)?;
        let weight = match spec.weight_norm {
            true => scope.weight_norm("weight")?,
            false => scope.tensor("weight", &filter)?,
        };
        let bias = match spec.bias {
            Bias::Present => Some(scope.tensor("bias", &[u64::from(spec.out_channels)])?),
            Bias::Absent => None,
        };
        Self::new(owner, weight, bias, spec, filter)
    }

    /// The convolution over already bound weights of `filter` extents.
    fn new(
        owner: &'static str,
        weight: Weight,
        bias: Option<Weight>,
        spec: ConvSpec<N>,
        filter: Vec<u64>,
    ) -> Result<Self> {
        spec.check(owner)?;
        if weight.shape().dims() != filter.as_slice() {
            return Err(Error::shape(
                owner,
                format!(
                    "a filter of {:?} for a spec whose filter is {filter:?}",
                    weight.shape().dims()
                ),
            ));
        }
        if (spec.bias == Bias::Present) != bias.is_some()
            || bias
                .as_ref()
                .is_some_and(|bias| bias.shape().dims() != [u64::from(spec.out_channels)])
        {
            return Err(Error::shape(
                owner,
                format!("the bias does not match a spec of {:?}", spec.bias),
            ));
        }
        Ok(Self { weight, bias, spec })
    }

    /// One launch of `Op` over `input` into `out`.
    fn launch(
        &self,
        f: &Forward,
        owner: &'static str,
        input: TensorView,
        out: TensorView,
        rank: u32,
        transposed: bool,
    ) -> Result<()> {
        if self
            .bias
            .as_ref()
            .is_some_and(|bias| bias.dtype() != out.dtype)
        {
            return Err(Error::shape(
                owner,
                "the bias and the output have different element types".to_owned(),
            ));
        }
        let descriptor = Descriptor {
            input: input.dtype,
            weight: self.weight.dtype(),
            output: out.dtype,
            geometry: self.spec.geometry(),
            bias: self.bias.is_some(),
            tf32: self.spec.tf32,
            channels_last: self.spec.channels_last,
            rank,
            transposed,
        };
        let args = Args {
            input,
            weight: self.weight.view(),
            bias: self.bias.as_ref().map(Weight::view),
            out,
        };
        match (rank, transposed) {
            (3, false) => f.operate::<Conv3dOp>(&descriptor, args),
            (1, false) => f.operate::<Conv1dOp>(&descriptor, args),
            (1, true) => f.operate::<ConvTranspose1dOp>(&descriptor, args),
            (rank, transposed) => Err(Error::Unsupported(format!(
                "{owner}: rank {rank} transposed={transposed} is not a convolution this layer builds"
            ))),
        }
    }
}

/// The filter of a plain convolution: `[out, in / groups, kernel...]`.
fn filter<const N: usize>(spec: &ConvSpec<N>, transposed: bool) -> Vec<u64> {
    let (first, second) = match transposed {
        // torch stores a transposed filter `[in, out / groups, kernel...]`.
        true => (spec.in_channels, spec.out_channels / spec.groups.max(1)),
        false => (spec.out_channels, spec.in_channels / spec.groups.max(1)),
    };
    let mut extents = vec![u64::from(first), u64::from(second)];
    extents.extend(spec.kernel.iter().map(|tap| u64::from(*tap)));
    extents
}

/// A three-dimensional convolution: torch's `Conv3d`.
pub struct Conv3d {
    inner: Convolution<3>,
}

/// The strides of a channels-last (`NDHWC`) buffer read as its logical `[n, c, d, h, w]` shape.
///
/// The channel axis runs fastest, then the spatial axes from the innermost out, then the batch —
/// the same order `conv3d`'s kernel derives from the descriptor's flag.
fn channels_last_strides(shape: Shape) -> Strides {
    let (channels, depth) = (shape.dim(1), shape.dim(2));
    let (rows, columns) = (shape.dim(3), shape.dim(4));
    let plane = columns * channels;
    let body = rows * plane;
    Strides::new(&[depth * body, 1, body, plane, channels])
}

/// Refuse a channels-last convolution whose operand is not stored channels-last.
///
/// The kernel derives every operand's strides from the descriptor's `channels_last` flag alone, so
/// the flag **is** the layout: a convolution declared channels-last over a channels-first buffer
/// reads the same bytes in another order and produces wrong numbers with no error anywhere.
///
/// **The filter is not part of the order.** A checkpoint stores a convolution's weight
/// `[out, in, kd, kh, kw]` packed, the KCRS form cuDNN's filter descriptor wants, and PyTorch's
/// `channels_last_3d` is a memory format of *activations*.
fn channels_last_storage(
    owner: &'static str,
    operand: &'static str,
    view: TensorView,
) -> Result<()> {
    if view.shape.rank() != 5 {
        return Err(Error::shape(
            owner,
            format!(
                "a channels-last convolution takes 5-d operands, and its {operand} is {:?}",
                view.shape.dims()
            ),
        ));
    }
    let expected = channels_last_strides(view.shape);
    if view.strides != expected {
        return Err(Error::shape(
            owner,
            format!(
                "the {operand} of a channels-last convolution must be stored NDHWC: its strides are \
                 {:?} over {:?}, and the packed channels-last layout is {:?}",
                view.strides.as_slice(),
                view.shape.dims(),
                expected.as_slice()
            ),
        ));
    }
    Ok(())
}

impl Conv3d {
    /// The module at `scope`: `weight` `[out, in / groups, kd, kh, kw]` and, when the spec states
    /// one, `bias` `[out]`.
    ///
    /// # Errors
    ///
    /// A spec no convolution has, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(scope: &mut Scope<'_, '_>, spec: ConvSpec<3>) -> Result<Self> {
        let inner = Convolution::load("Conv3d", scope, spec, filter(&spec, false))?;
        Ok(Self { inner })
    }

    /// The convolution over weights a model computes rather than reads.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the weights given.
    pub fn new(weight: Weight, bias: Option<Weight>, spec: ConvSpec<3>) -> Result<Self> {
        let inner = Convolution::new("Conv3d", weight, bias, spec, filter(&spec, false))?;
        Ok(Self { inner })
    }

    /// `out = conv3d(input)`: `input` `[n, c, d, h, w]` into `out` `[n, k, d', h', w']`.
    ///
    /// A convolution built channels-last takes both operands **stored** channels-last, which a
    /// caller states by viewing an `NDHWC` buffer as its logical shape.
    ///
    /// # Errors
    ///
    /// An operand not stored as the spec states, or the operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        if self.inner.spec.channels_last {
            channels_last_storage("Conv3d", "input", input)?;
            channels_last_storage("Conv3d", "out", out)?;
        }
        self.inner.launch(f, "Conv3d", input, out, 3, false)
    }
}

/// A one-dimensional convolution: torch's `Conv1d`.
pub struct Conv1d {
    inner: Convolution<1>,
}

impl Conv1d {
    /// The module at `scope`: `weight` `[out, in / groups, k]` (or its weight-norm pair) and, when
    /// the spec states one, `bias` `[out]`.
    ///
    /// # Errors
    ///
    /// A spec no convolution has, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(scope: &mut Scope<'_, '_>, spec: ConvSpec<1>) -> Result<Self> {
        let inner = Convolution::load("Conv1d", scope, spec, filter(&spec, false))?;
        Ok(Self { inner })
    }

    /// The convolution over weights a model computes rather than reads.
    ///
    /// # Errors
    ///
    /// As [`Self::load`], for the weights given.
    pub fn new(weight: Weight, bias: Option<Weight>, spec: ConvSpec<1>) -> Result<Self> {
        let inner = Convolution::new("Conv1d", weight, bias, spec, filter(&spec, false))?;
        Ok(Self { inner })
    }

    /// `out = conv1d(input)` on `[n, c, l]` sequences.
    ///
    /// # Errors
    ///
    /// The operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        self.inner.launch(f, "Conv1d", input, out, 1, false)
    }
}

/// A one-dimensional transposed convolution: torch's `ConvTranspose1d`.
pub struct ConvTranspose1d {
    inner: Convolution<1>,
}

impl ConvTranspose1d {
    /// The module at `scope`: `weight` `[in, out / groups, k]` (or its weight-norm pair) and, when
    /// the spec states one, `bias` `[out]`.
    ///
    /// # Errors
    ///
    /// A spec no convolution has, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(scope: &mut Scope<'_, '_>, spec: ConvSpec<1>) -> Result<Self> {
        let inner = Convolution::load("ConvTranspose1d", scope, spec, filter(&spec, true))?;
        Ok(Self { inner })
    }

    /// `out = conv_transpose1d(input)` on `[n, c, l]` sequences.
    ///
    /// # Errors
    ///
    /// The operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        self.inner.launch(f, "ConvTranspose1d", input, out, 1, true)
    }
}

/// The arguments of a [`ResnetBlock3d`].
#[derive(Clone, Copy, Debug)]
pub struct ResnetSpec {
    /// Channels the block reads.
    pub in_channels: u32,
    /// Channels the block writes.
    pub out_channels: u32,
    /// Group-normalization groups.
    pub groups: u32,
    /// Group-normalization variance floor.
    pub eps: f32,
    /// The causal pad each normalization writes its activated frame into; the spatial padding after
    /// the source is the padding before it, so a convolution of `2 · top + 1` taps keeps the extent.
    pub pad: CausalPad,
    /// The convolutions' taps along each axis.
    pub kernel: u32,
    /// Whether cuDNN may use TF32 for the block's FP32 convolutions.
    pub tf32: bool,
}

/// A causal video residual block (diffusers' `ResnetBlock3D` with causal convolutions): two stages
/// of group normalization, SiLU, causal pad and convolution, and the input — or its one-by-one
/// projection where the width changes — added to the second stage's output.
///
/// ```text
/// h   = conv1(pad(silu(norm1(x))))
/// out = conv2(pad(silu(norm2(h)))) + shortcut(x)
/// ```
///
/// Each normalization and its pad are one [`GroupNorm`](crate::layers::GroupNorm) pair of launches,
/// each convolution carries its bias, and the sum rounds once.
pub struct ResnetBlock3d {
    spec: ResnetSpec,
    norm1: crate::layers::GroupNorm,
    conv1: Conv3d,
    norm2: crate::layers::GroupNorm,
    conv2: Conv3d,
    /// `conv_shortcut`, the projection of the input when the width changes.
    shortcut: Option<Conv3d>,
}

impl ResnetBlock3d {
    /// The block at `scope`: `norm1`, `conv1`, `norm2`, `conv2` and, when the width changes, the
    /// one-by-one `conv_shortcut`, each with its weight and bias.
    ///
    /// # Errors
    ///
    /// A spec the layers refuse, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(scope: &mut Scope<'_, '_>, spec: ResnetSpec) -> Result<Self> {
        let norm = |scope: &mut Scope<'_, '_>, channels: u32| -> Result<crate::layers::GroupNorm> {
            let width = [u64::from(channels)];
            let weight = scope.tensor("weight", &width)?;
            let bias = scope.tensor("bias", &width)?;
            crate::layers::GroupNorm::new(spec.groups, spec.eps, Some(weight), Some(bias), spec.pad)
        };
        let conv = |from: u32, to: u32, kernel: u32| ConvSpec {
            tf32: spec.tf32,
            ..ConvSpec::new(from, to, [kernel; 3])
        };
        let norm1 = norm(&mut scope.scope("norm1"), spec.in_channels)?;
        let conv1 = Conv3d::load(
            &mut scope.scope("conv1"),
            conv(spec.in_channels, spec.out_channels, spec.kernel),
        )?;
        let norm2 = norm(&mut scope.scope("norm2"), spec.out_channels)?;
        let conv2 = Conv3d::load(
            &mut scope.scope("conv2"),
            conv(spec.out_channels, spec.out_channels, spec.kernel),
        )?;
        let shortcut = match spec.in_channels == spec.out_channels {
            true => None,
            false => Some(Conv3d::load(
                &mut scope.scope("conv_shortcut"),
                conv(spec.in_channels, spec.out_channels, 1),
            )?),
        };
        Ok(Self {
            spec,
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    /// The block's output for `input` `[1, in_channels, frames, height, width]` into `out`
    /// `[1, out_channels, frames, height, width]`, both FP32 and packed; `input` is never written.
    ///
    /// The padded frames, the normalizations' statistics, the first stage's output and the
    /// projection are reserved for the call and released at its end.
    ///
    /// # Errors
    ///
    /// Operands of another shape, or an operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let dims = input.shape.dims();
        let spec = &self.spec;
        if dims.len() != 5
            || dims[0] != 1
            || dims[1] != u64::from(spec.in_channels)
            || out.shape.dims() != [1, u64::from(spec.out_channels), dims[2], dims[3], dims[4]]
        {
            return Err(Error::shape(
                "ResnetBlock3d",
                format!(
                    "{:?} into {:?} for a block of {} to {} channels",
                    dims,
                    out.shape.dims(),
                    spec.in_channels,
                    spec.out_channels
                ),
            ));
        }
        let mark = f.arena().mark();
        let result = self.stages(f, input, out);
        f.arena().release(mark);
        result
    }

    fn stages(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let spec = &self.spec;
        let dims = input.shape.dims();
        let (frames, height, width) = (dims[2], dims[3], dims[4]);
        let pad = |channels: u32| {
            f.arena().tensor(
                input.dtype,
                Shape::new(&[
                    1,
                    u64::from(channels),
                    frames + u64::from(spec.pad.front),
                    height + 2 * u64::from(spec.pad.top),
                    width + 2 * u64::from(spec.pad.left),
                ]),
            )
        };
        let statistics = Shape::new(&[frames, u64::from(spec.groups)]);
        let (mean, rstd) = (
            f.arena().tensor(ops::DType::F32, statistics)?,
            f.arena().tensor(ops::DType::F32, statistics)?,
        );
        let first = pad(spec.in_channels)?;
        let working = f.arena().tensor(out.dtype, out.shape)?;
        self.norm1.forward(f, input, mean, rstd, first)?;
        self.conv1.forward(f, first, working)?;
        let second = pad(spec.out_channels)?;
        self.norm2.forward(f, working, mean, rstd, second)?;
        self.conv2.forward(f, second, out)?;
        let residual = match &self.shortcut {
            Some(shortcut) => {
                let projected = f.arena().tensor(out.dtype, out.shape)?;
                shortcut.forward(f, input, projected)?;
                projected
            }
            None => input,
        };
        // `(conv2 + bias) + residual`: the reference adds the convolution's bias in a pass of its
        // own and then the residual, and IEEE addition commutes, so this is the same rounded sum.
        let rows =
            |view: TensorView| view.reshape(Shape::new(&[view.shape.numel() / width, width]));
        crate::layers::Residual::default().forward(f, rows(out), rows(residual), None, rows(out))
    }
}

#[cfg(test)]
mod tests {
    use super::{channels_last_storage, channels_last_strides};
    use ops::tensor::{DType, DevicePtr, Shape, Strides, TensorView};

    /// The encoder's first convolution's operand: 3 channels, 19 frames, 258 x 258 pixels.
    fn operand(strides: Strides) -> TensorView {
        TensorView::with_strides(
            DevicePtr::new(0x1_0000),
            DType::F32,
            Shape::new(&[1, 3, 19, 258, 258]),
            strides,
        )
    }

    #[test]
    fn a_channels_last_operand_is_packed_ndhwc() {
        // The kernel's own order: channel fastest, then width, height, depth, batch.
        let shape = Shape::new(&[1, 3, 19, 258, 258]);
        assert_eq!(
            channels_last_strides(shape).as_slice(),
            &[3 * 19 * 258 * 258, 1, 258 * 258 * 3, 258 * 3, 3]
        );
    }

    #[test]
    fn a_channels_first_buffer_is_refused_by_a_channels_last_convolution() {
        let shape = Shape::new(&[1, 3, 19, 258, 258]);
        let error = channels_last_storage("Conv3d", "input", operand(Strides::row_major(shape)))
            .expect_err("a channels-first buffer read channels-last is wrong numbers, not a plan");
        assert!(format!("{error}").contains("must be stored NDHWC"));
        channels_last_storage("Conv3d", "input", operand(channels_last_strides(shape)))
            .expect("the packed channels-last layout is what the descriptor promises");
    }
}
