//! Activations: the gated activation of a transformer's MLP, and the plain GELU, SiLU and Snake.
//!
//! A gated MLP projects one hidden state into two halves and multiplies the activation of one half
//! by the other. [`GatedActivation`] states that product once; its [`Rounding`] names the FP32
//! evaluation order and rounding points a model's baseline uses, and its input comes as two views or
//! as one packed `[value, gate]` projection ([`Halves`]):
//!
//! ```text
//! Rounding::Fused         out = round(act(gate) · value)                    act_and_mul
//! Rounding::Stepwise      out = round(round(act(gate)) · value)             act_and_mul
//! Rounding::ValueProduct  out = round((value · gate) / (1 + exp(−gate)))    swiglu, value_first_swiglu
//! ```
//!
//! The three orders round differently, so a model keeps the one its reference computes: a reference
//! that evaluates the activation and the product as separate rounded steps keeps `Stepwise`; the
//! SwiGLU kernels multiply value and gate before dividing by the sigmoid's denominator. A packed
//! projection under `ValueProduct` is read in one pass with its packed bias; under `Fused` or
//! `Stepwise` its halves are read as two strided views of the packed rows.
//!
//! **Every gated activation may write in place.** `out` may be the same view as one of its inputs —
//! the reference's own `_swiglu` does exactly that, and a captured graph wants it, because the arena
//! is fixed at capture and the input buffer is where the result belongs. It is safe because the
//! activation is elementwise over unit-strided rows: each element is read before it is written.
//! What is *not* safe is `out` as a shifted view of an input, where a row's write lands on a position
//! a later read has not reached yet; the operators' shape and stride checks refuse mismatched extents
//! but cannot see a shift inside one row, so this documentation is the obligation the caller holds.

use ops::Operand;
use ops::activation::{self, Activation};
use uniserve_tensor::DType;

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::weights::Weight;

/// The elements one activation program walks.
///
/// The programs vary the tile with the row width, so the descriptor states the element count one
/// statement handles; a power of two keeps every vector load aligned.
const BLOCK: u32 = 1024;

/// The elements one program of the packed gated activation walks: wide enough that each lane moves a
/// whole vector of a half-precision row, which is what this memory-bound step runs at.
const PACKED_BLOCK: u32 = 4096;

/// The FP32 evaluation order and rounding points of a gated product; see the module table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rounding {
    /// `act(gate) · value` in FP32, rounded once.
    Fused,
    /// `act(gate)` rounded to the gate's element type, then the product rounded.
    Stepwise,
    /// `(value · gate) / (1 + exp(−gate))` in FP32, rounded once: the SwiGLU kernels' order, SiLU
    /// only.
    ValueProduct,
}

/// The biases a gated product adds to its halves before the activation.
#[derive(Clone, Copy, Debug)]
pub enum GateBias {
    /// No bias.
    None,
    /// One `[width]` bias per half of a split input.
    Split {
        /// Added to the value half.
        value: Option<Weight>,
        /// Added to the gate half.
        gate: Option<Weight>,
    },
    /// One `[2 · width]` bias over a packed `[value, gate]` projection.
    Packed(Weight),
}

/// The two halves a gated product reads.
#[derive(Clone, Copy, Debug)]
pub enum Halves {
    /// Two `[rows, width]` views, each with unit stride along its rows.
    Split {
        /// The half the activation applies to.
        gate: Operand,
        /// The half the activation multiplies.
        value: Operand,
    },
    /// One `[rows, 2 · width]` projection holding the value half and then the gate half of each
    /// row, as a checkpoint that fuses `[up, gate]` value-first stores it.
    ValueFirst(Operand),
}

impl Halves {
    /// The gate and value halves of a value-first packed projection, as strided views of its rows:
    /// `value` at element zero and `gate` one half along, both striding over the packed row.
    fn split(projection: Operand) -> (Operand, Operand) {
        let width = projection.last() / 2;
        let shape = ops::Shape::new(&[projection.rows(), width]);
        let strides = projection.strides;
        let value = Operand::with_strides(
            projection.element(&[0, 0]),
            projection.dtype,
            shape,
            strides,
        );
        let gate = Operand::with_strides(
            projection.element(&[0, width]),
            projection.dtype,
            shape,
            strides,
        );
        (gate, value)
    }
}

/// The gated activation `out = act(gate) · value`, at a stated [`Rounding`].
pub struct GatedActivation {
    activation: Activation,
    rounding: Rounding,
    bias: GateBias,
}

impl GatedActivation {
    /// The product gated by `activation`, evaluated at `rounding`, with `bias` added to the halves.
    ///
    /// # Errors
    ///
    /// `ValueProduct` with an activation other than SiLU, a bias under `Fused` or `Stepwise` (whose
    /// operator adds none), or a bias that is not one vector.
    pub fn new(activation: Activation, rounding: Rounding, bias: GateBias) -> Result<Self> {
        const OWNER: &str = "GatedActivation";
        if rounding == Rounding::ValueProduct && activation != Activation::Silu {
            return Err(Error::shape(
                OWNER,
                format!("the value-product order is SiLU's, not {activation}'s"),
            ));
        }
        let vectors: &[Option<Weight>] = match bias {
            GateBias::None => &[],
            GateBias::Split { value, gate } => &[value, gate],
            GateBias::Packed(bias) => &[Some(bias)],
        };
        if rounding != Rounding::ValueProduct && vectors.iter().any(Option::is_some) {
            return Err(Error::shape(
                OWNER,
                "an activation-first product adds no bias; the projection carries it".to_owned(),
            ));
        }
        for vector in vectors.iter().flatten() {
            if vector.shape().rank() != 1 {
                return Err(Error::shape(
                    OWNER,
                    format!("a bias is one vector, not {:?}", vector.shape().dims()),
                ));
            }
        }
        Ok(Self {
            activation,
            rounding,
            bias,
        })
    }

    /// `act(gate) · value` over `input`'s halves: a fresh `[rows, width]` tensor of the input's
    /// element type.
    ///
    /// # Errors
    ///
    /// Halves of different extents, a bias of another width or packing than the input, an arena
    /// that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, input: Halves) -> Result<Operand> {
        let (dtype, rows, width) = match input {
            Halves::Split { gate, .. } => (gate.dtype, gate.rows(), gate.last()),
            Halves::ValueFirst(projection) => {
                (projection.dtype, projection.rows(), projection.last() / 2)
            }
        };
        let out = f.empty(dtype, ops::Shape::new(&[rows, width]))?;
        self.launch(f, input, out)?;
        Ok(out)
    }

    /// The product of `input`'s halves into `out`, `[rows, width]`.
    fn launch(&self, f: &Forward, input: Halves, out: Operand) -> Result<()> {
        const OWNER: &str = "GatedActivation";
        let width = out.shape.last();
        match (self.rounding, input) {
            (Rounding::ValueProduct, Halves::ValueFirst(projection)) => {
                let bias = match self.bias {
                    GateBias::None => None,
                    GateBias::Packed(bias) => Some(bias),
                    GateBias::Split { .. } => {
                        return Err(Error::shape(
                            OWNER,
                            "a packed projection takes a packed bias".to_owned(),
                        ));
                    }
                };
                self.value_first(f, projection, bias, out)
            }
            (Rounding::ValueProduct, Halves::Split { gate, value }) => {
                let (value_bias, gate_bias) = match self.bias {
                    GateBias::None => (None, None),
                    GateBias::Split { value, gate } => (value, gate),
                    GateBias::Packed(_) => {
                        return Err(Error::shape(
                            OWNER,
                            "two halves take a bias per half".to_owned(),
                        ));
                    }
                };
                self.swiglu(f, gate, value, value_bias, gate_bias, out)
            }
            (rounding, input) => {
                let (gate, value) = match input {
                    Halves::Split { gate, value } => (gate, value),
                    Halves::ValueFirst(projection) => {
                        if projection.shape.last() != 2 * width {
                            return Err(Error::shape(
                                OWNER,
                                format!(
                                    "a packed row of {} elements for {width} output channels",
                                    projection.shape.last()
                                ),
                            ));
                        }
                        Halves::split(projection)
                    }
                };
                self.act_and_mul(f, gate, value, rounding == Rounding::Stepwise, out)
            }
        }
    }

    /// `act_and_mul`: the activation of the gate in FP32 (rounded first when `stepwise`), times the
    /// value.
    fn act_and_mul(
        &self,
        f: &Forward,
        gate: Operand,
        value: Operand,
        stepwise: bool,
        out: Operand,
    ) -> Result<()> {
        let width = out.shape.last();
        let (gate_stride, value_stride) = (gate.strides.get(0), value.strides.get(0));
        // The 64-bit form follows the furthest element either input row reaches: the halves of one
        // packed projection stride over twice the output's row.
        let elements = out.shape.numel();
        let rows = elements.div_ceil(width.max(1));
        let span = elements.max(rows.saturating_mul(gate_stride.max(value_stride)));
        let descriptor = activation::ActAndMulDescriptor {
            dtype: out.dtype,
            width: u32::try_from(width).map_err(|_| {
                Error::shape("GatedActivation", "a row wider than a u32".to_owned())
            })?,
            block: PACKED_BLOCK,
            activation: self.activation,
            gate_stride,
            value_stride,
            wide: span >= 1 << 31,
            pdl: PDL,
            stepwise,
        };
        let args = activation::ActAndMulArgs { gate, value, out };
        f.operate::<activation::ActAndMul>(&descriptor, args)
    }

    /// `swiglu`: the value-product order over two views, with a bias per half.
    fn swiglu(
        &self,
        f: &Forward,
        gate: Operand,
        value: Operand,
        value_bias: Option<Weight>,
        gate_bias: Option<Weight>,
        out: Operand,
    ) -> Result<()> {
        let width = out.shape.last();
        for (name, view) in [("value", value), ("gate", gate)] {
            if view.shape.last() != width {
                return Err(Error::shape(
                    "GatedActivation",
                    format!(
                        "the {name} row is {} elements wide, {width} expected",
                        view.shape.last()
                    ),
                ));
            }
        }
        for bias in [value_bias, gate_bias].iter().flatten() {
            if bias.shape().numel() != width {
                return Err(Error::shape(
                    "GatedActivation",
                    format!(
                        "a bias of {} elements for {width} channels",
                        bias.shape().numel()
                    ),
                ));
            }
        }
        let width = u32::try_from(width)
            .ok()
            .filter(|width| *width > 0)
            .ok_or_else(|| Error::shape("GatedActivation", format!("a row of {width} channels")))?;
        let descriptor = activation::SwigluDescriptor {
            dtype: out.dtype,
            width,
            block: width.next_power_of_two(),
            value_stride: value.strides.get(0),
            gate_stride: gate.strides.get(0),
            value_bias: value_bias.is_some(),
            gate_bias: gate_bias.is_some(),
            absmax: false,
        };
        let args = activation::SwigluArgs {
            value,
            gate,
            value_bias: value_bias.as_ref().map(Weight::view),
            gate_bias: gate_bias.as_ref().map(Weight::view),
            out,
            partials: None,
        };
        f.operate::<activation::Swiglu>(&descriptor, args)
    }

    /// `value_first_swiglu`: the value-product order over one packed projection and its packed bias.
    fn value_first(
        &self,
        f: &Forward,
        projection: Operand,
        bias: Option<Weight>,
        out: Operand,
    ) -> Result<()> {
        let width = out.shape.last();
        if projection.shape.last() != 2 * width {
            return Err(Error::shape(
                "GatedActivation",
                format!(
                    "a packed row of {} elements for {width} output channels",
                    projection.shape.last()
                ),
            ));
        }
        let descriptor = activation::ValueFirstSwigluDescriptor {
            dtype: out.dtype,
            width: u32::try_from(width).map_err(|_| {
                Error::shape("GatedActivation", "a row wider than a u32".to_owned())
            })?,
            block: PACKED_BLOCK,
            input_stride: projection.strides.get(0),
            bias: bias.is_some(),
            absmax: false,
        };
        let args = activation::ValueFirstSwigluArgs {
            value_gate: projection,
            bias: bias.as_ref().map(Weight::view),
            out,
            partials: None,
        };
        f.operate::<activation::ValueFirstSwiglu>(&descriptor, args)
    }
}

/// GELU over a row-major `[rows, columns]` activation.
///
/// A checkpoint whose reference takes PyTorch's tanh form takes `gelu_pytorch_tanh`;
/// a checkpoint whose config says plain `gelu` takes the exact one. They are
/// separate compilations, so the layer states which it is (`ops::gelu`).
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

    /// The descriptor one GELU over rows of `columns` channels launches.
    ///
    /// The approximation is part of the key: the checkpoint's blocks and its merger use two different
    /// ones, which are two programs.
    fn descriptor(
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

    /// `gelu(x)` over the rows of `x`, `[rows, columns]`, in `x`'s element type.
    ///
    /// # Errors
    ///
    /// A view of another rank, an arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: Operand) -> Result<Operand> {
        if x.shape.rank() != 2 {
            return Err(Error::shape(
                "Gelu",
                format!("{:?} is not [rows, columns]", x.shape.dims()),
            ));
        }
        let columns = u32::try_from(x.shape.last())
            .map_err(|_| Error::shape("Gelu", "more columns than a descriptor holds".to_owned()))?;
        let out = f.empty(x.dtype, x.shape)?;
        let descriptor = Self::descriptor(x.dtype, columns, self.approximation);
        f.operate::<ops::gelu::Gelu>(&descriptor, ops::gelu::Args { x, out })?;
        Ok(out)
    }
}

/// A periodic activation, `x + sin²(α·x)/(α + 1e-9)`, one frequency per channel.
///
/// A block may apply it directly or compose it inside an alias-free activation; `α` is the
/// checkpoint's raw per-channel parameter, not a log-space pair (`ops::snake`).
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
    fn descriptor(dtype: uniserve_tensor::DType, channels: u64) -> ops::snake::Descriptor {
        ops::snake::Descriptor {
            dtype,
            channels: u32::try_from(channels).unwrap_or(u32::MAX),
            block: BLOCK,
            pdl: PDL,
        }
    }

    /// `snake(x, alpha)` over `[rows, channels, width]`, in `x`'s element type.
    ///
    /// `alpha` is `[channels]` in the activation's element type; the signal is row-major in and out.
    ///
    /// # Errors
    ///
    /// Views of another extent, an arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: Operand, alpha: Operand) -> Result<Operand> {
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
        let out = f.empty(x.dtype, x.shape)?;
        let descriptor = Self::descriptor(x.dtype, self.channels);
        f.operate::<ops::snake::Snake>(&descriptor, ops::snake::Args { x, alpha, out })?;
        Ok(out)
    }
}

/// The sigmoid linear unit on one buffer, `out = x / (1 + exp(−x))`.
///
/// A plain activation, not a gate: the timestep embedding is `Linear → SiLU → Linear` over one
/// buffer, and folding it into a doubled gate buffer would compute a product nobody asks for
/// (`ops::activation::Silu`).
pub struct Silu;

impl Silu {
    /// The descriptor one SiLU launch takes: the element type and the row's column count, which is the
    /// width the timestep embedding activates.
    fn descriptor(dtype: uniserve_tensor::DType, columns: u32) -> ops::activation::SiluDescriptor {
        ops::activation::SiluDescriptor {
            dtype,
            columns,
            block: BLOCK,
            pdl: PDL,
        }
    }

    /// `silu(x)` over the rows of `x`, `[rows, columns]`, in `x`'s element type.
    ///
    /// # Errors
    ///
    /// A view of another rank, an arena that cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: Operand) -> Result<Operand> {
        if x.shape.rank() != 2 {
            return Err(Error::shape(
                "Silu",
                format!("{:?} is not [rows, columns]", x.shape.dims()),
            ));
        }
        let columns = u32::try_from(x.shape.last())
            .map_err(|_| Error::shape("Silu", "more columns than a descriptor holds".to_owned()))?;
        let out = f.empty(x.dtype, x.shape)?;
        let descriptor = Self::descriptor(x.dtype, columns);
        f.operate::<ops::activation::Silu>(&descriptor, ops::activation::SiluArgs { x, out })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::Shape;

    /// The 64-bit form follows the packed input's span, which runs twice as far as the output's: with
    /// 14336 channels the last row of 74899 is past `2**31` elements of input while the output is at
    /// half of that. A 32-bit launch there wraps a row's offset and reads another row's memory.
    ///
    /// The halves of one packed `[rows, 2 · width]` projection keep the packed row stride, which is
    /// what the stepwise product reads them through.
    #[test]
    fn a_packed_projection_goes_wide_on_its_input_span() {
        let width = 14_336u64;
        let wide = |rows: u64| {
            let arena = crate::Arena::counting();
            let f = Forward::recording(&arena);
            let packed = arena
                .tensor(DType::BF16, Shape::new(&[rows, 2 * width]))
                .expect("a counting arena");
            let activation =
                GatedActivation::new(Activation::Silu, Rounding::Stepwise, GateBias::None)
                    .expect("a stepwise SiLU product");
            let (walked, launches) =
                crate::forward::recorded(|| activation.forward(&f, Halves::ValueFirst(packed)));
            walked.expect("the recorded forward");
            let (operator, key) = &launches[0];
            assert_eq!(operator, "act_and_mul");
            key.contains("_wide_true_")
        };
        assert!(!wide(74_898), "74898 rows stay inside 32 bits");
        assert!(wide(74_899), "74899 rows do not");
    }

    /// The value-product order is SiLU's, and an activation-first product has no bias operand.
    #[test]
    fn a_gated_product_refuses_an_order_its_activation_or_bias_does_not_have() {
        let arena = crate::Arena::counting();
        let bias = arena
            .tensor(DType::BF16, Shape::new(&[8]))
            .expect("a counting arena");
        let bias = Weight {
            backing: crate::Backing::new(bias.ptr, 16),
            view: bias,
        };
        let split = GateBias::Split {
            value: Some(bias),
            gate: None,
        };
        assert!(
            GatedActivation::new(Activation::GeluTanh, Rounding::ValueProduct, GateBias::None)
                .is_err()
        );
        assert!(
            GatedActivation::new(Activation::GeluTanh, Rounding::Fused, GateBias::None).is_ok()
        );
        assert!(GatedActivation::new(Activation::Silu, Rounding::Stepwise, split).is_err());
        assert!(GatedActivation::new(Activation::Silu, Rounding::ValueProduct, split).is_ok());
    }
}
