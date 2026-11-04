//! The gated activation of a transformer's MLP: the activation of one projection half times the
//! other.
//!
//! Two packings occur, each an operator of its own (specs/worker.md 9, "SwiGLU";
//! `ops::activation`): [`GatedActivation`] over two arbitrary views, and [`SwiGlu`] over one packed
//! `[value, gate]` projection with its packed bias. The H3 block's MLP uses the packed form; the
//! expert GEMMs use the separate forms.
//!
//! **Every gated activation in this module may write in place.** `out` may be the same view as one
//! of its inputs — the reference's own `_swiglu` does exactly that, and a captured graph wants it,
//! because the arena is fixed at capture and the input buffer is where the result belongs. It is
//! safe because the activation is elementwise over unit-strided rows: each element is read before it
//! is written. What is *not* safe is `out` as a shifted view of an input, where a row's write lands
//! on a position a later read has not reached yet; the operators' shape and stride checks refuse
//! mismatched extents but cannot see a shift inside one row, so the layer's documentation is the
//! contract the caller holds.

use ops::TensorView;
use ops::DType;
use ops::activation::{self, Activation};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::weights::Weight;

/// The elements one activation program walks.
///
/// The programs vary the tile with the row width, so the descriptor states the element count one
/// statement handles; a power of two keeps every vector load aligned.
const BLOCK: u32 = 1024;

/// `out = act(gate) · value` over two views.
pub struct GatedActivation {
    activation: Activation,
    stepwise: bool,
}

impl GatedActivation {
    /// The activation `activation` applied to the gate half.
    pub const fn new(activation: Activation) -> Self {
        Self {
            activation,
            stepwise: false,
        }
    }

    /// Evaluate the activation step by step, matching the eager expression's rounding rather than
    /// the contraction-free default.
    pub const fn with_stepwise(mut self) -> Self {
        self.stepwise = true;
        self
    }

    /// The activation this layer applies.
    pub const fn activation(&self) -> Activation {
        self.activation
    }

    /// The descriptor one gated activation of `width` channels launches, stated without a device.
    ///
    /// The two row strides are launch values a walk reads from the scratch layout, and `elements` is
    /// the output's element count — the descriptor's `wide` form is a second program, so it is part of
    /// the key and not a heuristic.
    pub fn descriptor_of(
        dtype: DType,
        width: u64,
        activation: Activation,
        gate_stride: u64,
        value_stride: u64,
        elements: u64,
        stepwise: bool,
    ) -> Result<activation::ActAndMulDescriptor> {
        Ok(activation::ActAndMulDescriptor {
            dtype,
            width: u32::try_from(width)
                .map_err(|_| Error::shape("GatedActivation", "a row wider than a u32".to_owned()))?,
            block: BLOCK,
            activation,
            gate_stride,
            value_stride,
            wide: elements >= u64::from(u32::MAX),
            pdl: PDL,
            stepwise,
        })
    }

    /// `out = act(gate) · value`.
    ///
    /// `gate` and `value` have `out`'s extents; their rows may be strided (the halves of one packed
    /// projection included), and `out` is contiguous. `out` may be `gate` or `value`: each element is
    /// read before it is written, but a *shifted* view of either is not allowed (module docs).
    pub fn forward(
        &self,
        f: &Forward,
        gate: TensorView,
        value: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let width = out.shape.last();
        let descriptor = Self::descriptor_of(
            out.dtype,
            width,
            self.activation,
            gate.strides.get(0),
            value.strides.get(0),
            out.shape.numel(),
            self.stepwise,
        )?;
        let args = activation::ActAndMulArgs { gate, value, out };

        f.operate::<activation::ActAndMul>(&descriptor, args)
    }
}

/// `out = (value + bias) · silu(gate + bias)` over separate views.
pub struct SwiGlu {
    width: u32,
    value_bias: Option<Weight>,
    gate_bias: Option<Weight>,
    absmax: bool,
}

impl SwiGlu {
    /// The gated activation of a projection of `width` output channels.
    ///
    /// Qwen3's dense MLPs have no bias; a checkpoint whose MLP has one adds it with
    /// [`SwiGlu::with_biases`].
    pub fn new(width: u32) -> Result<Self> {
        if width == 0 {
            return Err(Error::shape(
                "SwiGlu",
                "a width of zero channels has no activation".to_owned(),
            ));
        }
        Ok(Self {
            width,
            value_bias: None,
            gate_bias: None,
            absmax: false,
        })
    }

    /// The same activation with the per-half biases the checkpoint carries.
    pub fn with_biases(
        mut self,
        value_bias: Option<Weight>,
        gate_bias: Option<Weight>,
    ) -> Result<Self> {
        for (name, bias) in [("value", &value_bias), ("gate", &gate_bias)] {
            if let Some(bias) = bias {
                if bias.shape().rank() != 1 {
                    return Err(Error::shape(
                        "SwiGlu",
                        format!(
                            "the {name} bias is one vector, not {:?}",
                            bias.shape().dims()
                        ),
                    ));
                }
                if bias.shape().numel() != u64::from(self.width) {
                    return Err(Error::shape(
                        "SwiGlu",
                        format!(
                            "a {name} bias of {} elements for {} channels",
                            bias.shape().numel(),
                            self.width
                        ),
                    ));
                }
            }
        }
        self.value_bias = value_bias;
        self.gate_bias = gate_bias;
        Ok(self)
    }

    /// Record each program's largest output magnitude into `partials` on every call.
    pub fn with_absmax(mut self) -> Self {
        self.absmax = true;
        self
    }

    /// Channels of one output row.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// `out[r, c] = silu((gate + gate_bias)[r, c]) · (value + value_bias)[r, c]`; the three views are
    /// `[rows, width]`.
    ///
    /// The gate comes first, as in [`GatedActivation`], so the two gated activations cannot be read
    /// one for the other: `gate` is the branch the activation is applied to.
    ///
    /// `out` may be the *same* view as `gate` — the dense MLP's only call site passes it so, writing
    /// the result into the arena buffer the gate arrived in, and the reference's `_swiglu` is
    /// in-place for the same reason. Each element is read before it is written; a shifted view of
    /// `gate` (or of `value`) is not allowed (module docs).
    pub fn forward(
        &self,
        f: &Forward,
        gate: TensorView,
        value: TensorView,
        out: TensorView,
    ) -> Result<()> {
        self.run(f, gate, value, out, None)
    }

    /// The same activation, recording each program's largest magnitude into `partials`.
    pub fn forward_absmax(
        &self,
        f: &Forward,
        gate: TensorView,
        value: TensorView,
        out: TensorView,
        partials: TensorView,
    ) -> Result<()> {
        self.run(f, gate, value, out, Some(partials))
    }

    fn run(
        &self,
        f: &Forward,
        gate: TensorView,
        value: TensorView,
        out: TensorView,
        partials: Option<TensorView>,
    ) -> Result<()> {
        if self.absmax != partials.is_some() {
            return Err(Error::shape(
                "SwiGlu",
                "the magnitude buffer is present exactly when the layer records it".to_owned(),
            ));
        }
        let width = u64::from(self.width);
        for (name, view) in [("value", value), ("gate", gate), ("out", out)] {
            if view.shape.last() != width {
                return Err(Error::shape(
                    "SwiGlu",
                    format!(
                        "the {name} row is {} elements wide, {width} expected",
                        view.shape.last()
                    ),
                ));
            }
        }
        let descriptor = activation::SwigluDescriptor {
            dtype: out.dtype,
            width: self.width,
            block: self.width.next_power_of_two(),
            value_stride: value.strides.get(0),
            gate_stride: gate.strides.get(0),
            value_bias: self.value_bias.is_some(),
            gate_bias: self.gate_bias.is_some(),
            absmax: self.absmax,
        };
        let args = activation::SwigluArgs {
            value,
            gate,
            value_bias: self.value_bias.as_ref().map(Weight::view),
            gate_bias: self.gate_bias.as_ref().map(Weight::view),
            out,
            partials,
        };

        f.operate::<activation::Swiglu>(&descriptor, args)
    }
}

/// `out = silu(gate) · value` over one packed `[value, gate]` projection: the gate half is the one
/// the activation applies to, as in [`GatedActivation`] and [`SwiGlu`].
pub struct PackedSwiGlu {
    bias: Option<Weight>,
    absmax: bool,
}

impl PackedSwiGlu {
    /// The descriptor one packed gated activation of `width` channels launches, stated without a
    /// device or a projection.
    ///
    /// `input_stride` is the row stride of the projection's output, which a walk reads from the
    /// scratch tensor the layer carves; the bias's presence and the absolute-max mode are the
    /// layer's.
    pub fn descriptor_of(
        dtype: DType,
        width: u64,
        input_stride: u64,
        bias: bool,
        absmax: bool,
    ) -> Result<activation::ValueFirstSwigluDescriptor> {
        Ok(activation::ValueFirstSwigluDescriptor {
            dtype,
            width: u32::try_from(width)
                .map_err(|_| Error::shape("PackedSwiGlu", "a row wider than a u32".to_owned()))?,
            block: BLOCK,
            input_stride,
            bias,
            absmax,
        })
    }

    /// The gated activation of a projection whose rows hold the value half then the gate half, with
    /// the packed `[value_bias, gate_bias]` bias when the checkpoint has one.
    pub fn new(bias: Option<Weight>) -> Result<Self> {
        if let Some(bias) = &bias
            && bias.shape().rank() != 1
        {
            return Err(Error::shape(
                "PackedSwiGlu",
                format!(
                    "the packed bias is one vector, not {:?}",
                    bias.shape().dims()
                ),
            ));
        }
        Ok(Self {
            bias,
            absmax: false,
        })
    }

    /// Record each program's largest output magnitude into `partials` on every call.
    pub fn with_absmax(mut self) -> Self {
        self.absmax = true;
        self
    }

    /// The activation of a packed `[..., 2 · width]` projection into `[..., width]`.
    ///
    /// `out` may overlap the leading half of `projection` (the value half), element for element; it
    /// must not be a shifted view of either half (module docs).
    pub fn forward(
        &self,
        f: &Forward,
        projection: TensorView,
        out: TensorView,
        partials: Option<TensorView>,
    ) -> Result<()> {
        if self.absmax != partials.is_some() {
            return Err(Error::shape(
                "PackedSwiGlu",
                "the magnitude buffer is present exactly when the layer records it".to_owned(),
            ));
        }
        let width = out.shape.last();
        if projection.shape.last() != 2 * width {
            return Err(Error::shape(
                "PackedSwiGlu",
                format!(
                    "a packed row of {} elements for {width} output channels",
                    projection.shape.last()
                ),
            ));
        }
        let descriptor = Self::descriptor_of(
            out.dtype,
            width,
            projection.strides.get(0),
            self.bias.is_some(),
            self.absmax,
        )?;
        let args = activation::ValueFirstSwigluArgs {
            value_gate: projection,
            bias: self.bias.as_ref().map(Weight::view),
            out,
            partials,
        };
        f.operate::<activation::ValueFirstSwiglu>(&descriptor, args)
    }
}

/// GELU over a row-major `[rows, columns]` activation.
///
/// H3's audio VAE and its diffusion transformer take PyTorch's tanh form
/// (`gelu_pytorch_tanh`); a checkpoint whose config says plain `gelu` takes the exact one. They are
/// separate compilations, so the layer states which it is (specs/worker.md 9, "GELU";
/// `ops::gelu`).
pub struct Gelu {
    approximation: ops::gelu::Approximation,
}

impl Gelu {
    /// PyTorch's `gelu_pytorch_tanh`: `x·sigmoid(√(8/π)(x + 0.044715x³))`.
    pub const fn tanh() -> Self {
        Self {
            approximation: ops::gelu::Approximation::Tanh,
        }
    }

    /// PyTorch's `gelu`: `0.5x(1 + erf(x/√2))`.
    pub const fn exact() -> Self {
        Self {
            approximation: ops::gelu::Approximation::Exact,
        }
    }

    /// Which formula this layer evaluates.
    pub const fn approximation(&self) -> ops::gelu::Approximation {
        self.approximation
    }

    /// The descriptor one GELU over rows of `columns` channels launches, stated without a device.
    ///
    /// The approximation is part of the key: the checkpoint's blocks and its merger use two different
    /// ones, which are two programs.
    pub fn descriptor_of(
        dtype: DType,
        columns: u32,
        approximate: ops::gelu::Approximation,
    ) -> ops::gelu::Descriptor {
        ops::gelu::Descriptor {
            dtype,
            columns,
            block: BLOCK,
            approximate,
            pdl: PDL,
        }
    }

    /// `out = gelu(x)` over the rows.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        if x.shape.rank() != 2 || x.shape != out.shape {
            return Err(Error::shape(
                "Gelu",
                format!("{:?} into {:?}", x.shape.dims(), out.shape.dims()),
            ));
        }
        let columns = u32::try_from(x.shape.last())
            .map_err(|_| Error::shape("Gelu", "more columns than a descriptor holds".to_owned()))?;
        let descriptor = Self::descriptor_of(x.dtype, columns, self.approximation);
        f.operate::<ops::gelu::Gelu>(&descriptor, ops::gelu::Args { x, out })
    }
}

/// The audio VAE's periodic activation, `x + sin²(α·x)/(α + 1e-9)`, one frequency per channel.
///
/// The encoder's blocks apply it and the decoder's alias-free activation composes it; `α` is the
/// checkpoint's raw per-channel parameter, not a log-space pair (specs/worker.md 9, "Snake";
/// `ops::snake`).
pub struct Snake {
    channels: u64,
}

impl Snake {
    /// The activation of `channels` audio channels: the length of every `α`.
    pub fn new(channels: u64) -> Result<Self> {
        if channels == 0 {
            return Err(Error::shape(
                "Snake",
                "an audio activation has channels".to_owned(),
            ));
        }
        Ok(Self { channels })
    }

    /// The descriptor one Snake launch takes: the element type and the channel count, which is the
    /// length of every frequency. The samples per channel travel with the call.
    pub fn descriptor_of(dtype: ops::DType, channels: u64) -> ops::snake::Descriptor {
        ops::snake::Descriptor {
            dtype,
            channels: u32::try_from(channels).unwrap_or(u32::MAX),
            block: BLOCK,
            pdl: PDL,
        }
    }

    /// `out = snake(x, alpha)` over `[rows, channels, width]`.
    ///
    /// `alpha` is `[channels]` in the activation's element type; the signal is row-major in and out.
    pub fn forward(
        &self,
        f: &Forward,
        x: TensorView,
        alpha: TensorView,
        out: TensorView,
    ) -> Result<()> {
        if x.shape.rank() != 3 || x.shape.dim(1) != self.channels {
            return Err(Error::shape(
                "Snake",
                format!(
                    "{:?} is not [rows, {}, width]",
                    x.shape.dims(),
                    self.channels
                ),
            ));
        }
        if alpha.shape.numel() != self.channels {
            return Err(Error::shape(
                "Snake",
                format!(
                    "{} frequencies for {} channels",
                    alpha.shape.numel(),
                    self.channels
                ),
            ));
        }
        let descriptor = Self::descriptor_of(x.dtype, self.channels);
        f.operate::<ops::snake::Snake>(&descriptor, ops::snake::Args { x, alpha, out })
    }
}

/// The sigmoid linear unit on one buffer, `out = x / (1 + exp(−x))`.
///
/// A plain activation, not a gate: the timestep embedding is `Linear → SiLU → Linear` over one
/// buffer, and folding it into a doubled gate buffer would compute a product nobody asks for
/// (specs/worker.md 9, 门控残差; `ops::activation::Silu`).
pub struct Silu;

impl Silu {
    /// `out = silu(x)` over the rows.
    pub fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        if x.shape.rank() != 2 || x.shape != out.shape {
            return Err(Error::shape(
                "Silu",
                format!("{:?} into {:?}", x.shape.dims(), out.shape.dims()),
            ));
        }
        let columns = u32::try_from(x.shape.last())
            .map_err(|_| Error::shape("Silu", "more columns than a descriptor holds".to_owned()))?;
        let descriptor = ops::activation::SiluDescriptor {
            dtype: x.dtype,
            columns,
            block: BLOCK,
            pdl: PDL,
        };
        f.operate::<ops::activation::Silu>(&descriptor, ops::activation::SiluArgs { x, out })
    }
}
