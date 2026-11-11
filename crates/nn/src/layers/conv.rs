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
};
use ops::frame::{self, channels_last_strides};
use ops::{Shape, TensorView};

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
/// frames before the source, `bottom` rows and `right` columns after it, the border filled as
/// `padding` states.
///
/// The frames in front are zeros and none follow the source, so no output frame of the convolution
/// that reads the padded input depends on a later input frame.
#[derive(Clone, Copy, Debug)]
pub struct CausalPad {
    /// Rows before the source.
    pub top: u32,
    /// Rows after the source.
    pub bottom: u32,
    /// Columns before the source.
    pub left: u32,
    /// Columns after the source.
    pub right: u32,
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

    /// The padded extent of `values` `[batch, channels, frames, height, width]`, reserved in
    /// `values`' element type and storage order.
    pub(crate) fn reserve(&self, f: &Forward, values: TensorView) -> Result<TensorView> {
        if values.shape.rank() != 5 {
            return Err(Error::shape(
                "CausalPad",
                format!("a frame is rank 5, not {:?}", values.shape.dims()),
            ));
        }
        let dims = values.shape.dims();
        let shape = Shape::new(&[
            dims[0],
            dims[1],
            dims[2] + u64::from(self.front),
            dims[3] + u64::from(self.top) + u64::from(self.bottom),
            dims[4] + u64::from(self.left) + u64::from(self.right),
        ]);
        frame(f, values.dtype, shape, frame::is_channels_last(&values))
    }

    /// `values` `[batch, channels, frames, height, width]` padded: zero frames in front, the extra
    /// rows and columns following [`Self::padding`], in `values`' element type and storage order
    /// (channel-first or channels-last; the strides state which, `ops::conv`'s rule).
    ///
    /// # Errors
    ///
    /// Operands of another rank, an arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, values: TensorView) -> Result<TensorView> {
        let out = self.reserve(f, values)?;
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
        )?;
        Ok(out)
    }
}

/// A `[n, c, d, h, w]` tensor of `dtype` from the forward's arena, stored channels-last when
/// `channels_last` holds and packed channel-first otherwise.
pub(crate) fn frame(
    f: &Forward,
    dtype: ops::DType,
    shape: Shape,
    channels_last: bool,
) -> Result<TensorView> {
    let packed = f.empty(dtype, shape)?;
    Ok(match channels_last {
        true => TensorView::with_strides(packed.ptr, dtype, shape, channels_last_strides(shape)),
        false => packed,
    })
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

    /// `values` over the operator's three spatial axes: a 1-d convolution's length is the first
    /// axis, and the axes past `N` take `fill` (a filter, stride and dilation of one, no padding).
    fn axes(values: &[u32; N], fill: u32) -> [u32; 3] {
        let mut axes = [fill; 3];
        axes[..N].copy_from_slice(values);
        axes
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

    /// The output extent of `input` `[n, c, spatial…]` along the `N` spatial axes, torch's rule:
    /// `(l + 2p − d(k − 1) − 1) / s + 1` for a convolution and `(l − 1)s − 2p + d(k − 1) + 1` for a
    /// transposed one.
    fn output_shape(
        &self,
        owner: &'static str,
        input: TensorView,
        transposed: bool,
    ) -> Result<Shape> {
        let dims = input.shape.dims();
        if dims.len() != N + 2 || dims[1] != u64::from(self.spec.in_channels) {
            return Err(Error::shape(
                owner,
                format!(
                    "an input of {:?} for a convolution of {} input channels over {N} axes",
                    dims, self.spec.in_channels
                ),
            ));
        }
        let mut out = vec![dims[0], u64::from(self.spec.out_channels)];
        for axis in 0..N {
            let length = dims[axis + 2];
            let kernel = u64::from(self.spec.kernel[axis]);
            let stride = u64::from(self.spec.stride[axis]);
            let padding = u64::from(self.spec.padding[axis]);
            let span = u64::from(self.spec.dilation[axis]) * (kernel - 1);
            let extent = match transposed {
                true => ((length.max(1) - 1) * stride + span + 1).checked_sub(2 * padding),
                false => (length + 2 * padding)
                    .checked_sub(span + 1)
                    .map(|reach| reach / stride + 1),
            };
            let extent = extent.filter(|extent| *extent > 0).ok_or_else(|| {
                Error::shape(
                    owner,
                    format!("an input of {length} along axis {axis} leaves no output"),
                )
            })?;
            out.push(extent);
        }
        Ok(Shape::new(&out))
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
            kernel: ConvSpec::axes(&self.spec.kernel, 1),
            stride: ConvSpec::axes(&self.spec.stride, 1),
            padding: ConvSpec::axes(&self.spec.padding, 0),
            dilation: ConvSpec::axes(&self.spec.dilation, 1),
            groups: self.spec.groups,
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

    /// `conv3d(input)`: `input` `[n, c, d, h, w]` to `[n, k, d', h', w']` in `input`'s element
    /// type.
    ///
    /// A convolution built channels-last takes its input **stored** channels-last, which a caller
    /// states by viewing an `NDHWC` buffer as its logical shape, and answers its output stored the
    /// same way.
    ///
    /// # Errors
    ///
    /// An input not stored as the spec states, an arena that cannot hold the output, or the
    /// operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView) -> Result<TensorView> {
        if self.inner.spec.channels_last {
            channels_last_storage("Conv3d", "input", input)?;
        }
        let shape = self.inner.output_shape("Conv3d", input, false)?;
        let out = frame(f, input.dtype, shape, self.inner.spec.channels_last)?;
        self.inner.launch(f, "Conv3d", input, out, 3, false)?;
        Ok(out)
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

    /// `conv1d(input)` on `[n, c, l]` sequences, in `input`'s element type.
    ///
    /// # Errors
    ///
    /// An arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView) -> Result<TensorView> {
        let out = self.reserve(f, input)?;
        self.launch(f, input, out)?;
        Ok(out)
    }

    /// The output [`Self::forward`] writes for an input of `input`'s extents and element type,
    /// reserved without launching: a stack that keeps it across a scope reserves it first.
    pub(crate) fn reserve(&self, f: &Forward, input: TensorView) -> Result<TensorView> {
        f.empty(
            input.dtype,
            self.inner.output_shape("Conv1d", input, false)?,
        )
    }

    /// The convolution of `input` into the reserved `out`.
    pub(crate) fn launch(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
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

    /// `conv_transpose1d(input)` on `[n, c, l]` sequences, in `input`'s element type.
    ///
    /// # Errors
    ///
    /// An arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView) -> Result<TensorView> {
        let shape = self.inner.output_shape("ConvTranspose1d", input, true)?;
        let out = f.empty(input.dtype, shape)?;
        self.inner
            .launch(f, "ConvTranspose1d", input, out, 1, true)?;
        Ok(out)
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
    /// The causal pad each normalization writes its activated frame into; a convolution of
    /// `top + bottom + 1` taps keeps the extent.
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

    /// The block's output for `input` `[1, in_channels, frames, height, width]`:
    /// `[1, out_channels, frames, height, width]`, FP32 and packed; `input` is never written.
    ///
    /// The padded frames, the normalizations' statistics, the first stage's output and the
    /// projection are reserved inside the call and handed back at its end; only the output stays.
    ///
    /// # Errors
    ///
    /// Operands of another shape, or an operator's refusal.
    pub fn forward(&self, f: &Forward, input: TensorView) -> Result<TensorView> {
        let dims = input.shape.dims();
        let spec = &self.spec;
        if dims.len() != 5 || dims[0] != 1 || dims[1] != u64::from(spec.in_channels) {
            return Err(Error::shape(
                "ResnetBlock3d",
                format!(
                    "{dims:?} for a block of {} to {} channels",
                    spec.in_channels, spec.out_channels
                ),
            ));
        }
        let out = f.empty(
            input.dtype,
            Shape::new(&[1, u64::from(spec.out_channels), dims[2], dims[3], dims[4]]),
        )?;
        // The output is reserved before the scope, so the second convolution and the residual
        // add write it while every intermediate is handed back when the scope ends.
        f.scoped(|f| {
            let working = self.conv1.forward(f, self.norm1.forward(f, input)?)?;
            let padded = self.norm2.forward(f, working)?;
            self.conv2
                .inner
                .launch(f, "Conv3d", padded, out, 3, false)?;
            let residual = match &self.shortcut {
                Some(shortcut) => shortcut.forward(f, input)?,
                None => input,
            };
            // `(conv2 + bias) + residual`: the reference adds the convolution's bias in a pass of
            // its own and then the residual, and IEEE addition commutes, so this is the same
            // rounded sum.
            let width = dims[4];
            let rows =
                |view: TensorView| view.reshape(Shape::new(&[view.shape.numel() / width, width]));
            crate::layers::Residual::default().forward(f, rows(out), rows(residual), None)
        })?;
        Ok(out)
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
            .expect_err(
                "a channels-first buffer read channels-last is wrong numbers, so it is refused",
            );
        assert!(format!("{error}").contains("must be stored NDHWC"));
        channels_last_storage("Conv3d", "input", operand(channels_last_strides(shape)))
            .expect("the packed channels-last layout is what the descriptor promises");
    }
}
