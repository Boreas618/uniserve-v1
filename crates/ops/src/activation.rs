//! Gated activations: the fused activation-times-value pair a transformer's MLP applies.
//!
//! A gated MLP projects one hidden state into two halves and multiplies the activation of one half by
//! the other. Two packings occur, and each is an operator because it has its own stride rule:
//!
//! ```text
//! act_and_mul        out = act(gate) · value            gate and value are two views
//! value_first_swiglu out = value · silu(gate)          one packed [value, gate] view
//! ```
//!
//! Rows are the last axis. In `act_and_mul` the two inputs are arbitrary views whose rows each have
//! one stride (the two halves of a packed projection, or two separately written ones); in
//! `value_first_swiglu` a single contiguous view holds both halves, so a row of the packed input is
//! twice as wide as an output row. `value_first_swiglu` also adds the packed bias and, on request,
//! records each program's largest magnitude for the caller's reduction.
//!
//! The activation runs in FP32 and the product rounds to the output type, which is what the eager
//! expression does (specs/worker.md 9, "SwiGLU").

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The activation a gated pair applies to its gate half.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Activation {
    /// `x · sigmoid(x)`.
    Silu,
    /// The exact Gaussian error function's GELU.
    Gelu,
    /// The tanh approximation of GELU.
    GeluTanh,
}

impl Activation {
    /// The name the kernel's activation argument takes and the descriptor key prints.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::GeluTanh => "gelu_tanh",
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One compilation of `act_and_mul`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ActAndMulDescriptor {
    /// Element type of the gates, the values and the output.
    pub dtype: DType,
    /// Elements per row of the output.
    pub width: u32,
    /// The elements one program walks.
    pub block: u32,
    /// The activation applied to the gate half.
    pub activation: Activation,
    /// Elements between the gate rows.
    pub gate_stride: u64,
    /// Elements between the value rows.
    pub value_stride: u64,
    /// Index in 64 bits: the views span at least `2**31` elements. The kernel's element count and
    /// both row strides widen with it, so a launch whose spans stay below `2**31` keeps 32-bit
    /// addressing.
    pub wide: bool,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
    /// Evaluate the activation step by step instead of contraction-free, for a caller that needs
    /// the eager expression's rounding rather than the kernel's default evaluation.
    pub stepwise: bool,
}

impl fmt::Display for ActAndMulDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("width", self.width)
            .field("block", self.block)
            .field("activation", self.activation)
            .field("gate_stride", self.gate_stride)
            .field("value_stride", self.value_stride)
            .field("wide", self.wide)
            .field("pdl", self.pdl)
            .field("stepwise", self.stepwise)
            .finish(formatter)
    }
}

/// The arguments of `act_and_mul`.
pub struct ActAndMulArgs {
    /// The gate half, its rows `gate_stride` elements apart and its last axis unit-strided.
    pub gate: TensorView,
    /// The value half, its rows `value_stride` elements apart and its last axis unit-strided.
    pub value: TensorView,
    /// `[..., width]`, contiguous: receives `act(gate) · value`.
    pub out: TensorView,
}

/// The activation of the gate half times the value half.
pub struct ActAndMul;

impl Op for ActAndMul {
    const NAME: &'static str = "act_and_mul";

    type Descriptor = ActAndMulDescriptor;
    type Args<'a> = ActAndMulArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        if width == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a row of zero elements has no activation".to_owned(),
            ));
        }
        if descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a block of zero elements walks nothing".to_owned(),
            ));
        }
        args.out.require_row(Self::NAME, descriptor.dtype, width)?;
        let rows = args.out.rows();
        for (name, view) in [("gate", &args.gate), ("value", &args.value)] {
            if view.shape.last() != width {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} rows hold {} elements, the output holds {width}",
                        view.shape.last()
                    ),
                ));
            }
            if view.strides.last() != 1 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} rows are not unit-strided in their last axis: {}",
                        view.strides.last()
                    ),
                ));
            }
            if view.rows() != rows {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} half holds {} rows, the output {rows}",
                        view.rows()
                    ),
                ));
            }
            if view.dtype != descriptor.dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected {} {name} elements, found {}",
                        descriptor.dtype, view.dtype
                    ),
                ));
            }
        }
        // A one-row view has no leading axis to stride over, so only a rank above one is checked.
        for (name, view, stride) in [
            ("gate", &args.gate, descriptor.gate_stride),
            ("value", &args.value, descriptor.value_stride),
        ] {
            if view.shape.rank() > 1 && view.strides.get(0) != stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} rows are {} elements apart, the descriptor says {stride}",
                        view.strides.get(0)
                    ),
                ));
            }
        }

        let elements = rows * width;
        let widest = elements.max(rows * descriptor.gate_stride.max(descriptor.value_stride));
        if descriptor.wide && widest < 1 << 31 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor indexes {widest} elements in 64 bits, which this launch does \
                     not need; build a 32-bit variant for it"
                ),
            ));
        }
        if !descriptor.wide {
            let limit = i32::MAX as u64;
            for (name, value) in [
                ("element count", elements),
                ("gate row stride", descriptor.gate_stride),
                ("value row stride", descriptor.value_stride),
            ] {
                if value > limit {
                    return Err(Error::argument(
                        Self::NAME,
                        format!("the {name} {value} needs the 64-bit variant of the kernel"),
                    ));
                }
            }
        }

        let mut arguments = Arguments::new();
        arguments.view(args.gate).view(args.value).view(args.out);
        if descriptor.wide {
            arguments
                .u64(elements)
                .u64(descriptor.gate_stride)
                .u64(descriptor.value_stride);
        } else {
            arguments
                .i32(elements as i32)
                .i32(descriptor.gate_stride as i32)
                .i32(descriptor.value_stride as i32);
        }
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ActAndMulArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The activation evaluates in FP32 against a rounded output: the sigmoid or error function,
        // its product with the gate and the product with the value, three roundings in the output
        // type at most, and no reduction.
        let roundings = match descriptor.activation {
            Activation::Silu => 3.0,
            Activation::Gelu | Activation::GeluTanh => 4.0,
        };
        Tolerance::new(0.0, roundings * descriptor.dtype.unit_roundoff() as f32)
    }
}

/// One compilation of `value_first_swiglu`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueFirstSwigluDescriptor {
    /// Element type of the packed input, the bias and the output.
    pub dtype: DType,
    /// Elements per row of the output; the packed input's rows hold twice as many.
    pub width: u32,
    /// The elements one program walks: the tile, or the magnitude tile when partials are written.
    pub block: u32,
    /// The packed input's rows are `2 · width` elements apart and start with the value half.
    pub input_stride: u64,
    /// Whether the packed `[value_bias, gate_bias]` bias is added.
    pub bias: bool,
    /// Whether each program records its largest magnitude into `partials`.
    pub absmax: bool,
}

impl fmt::Display for ValueFirstSwigluDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("width", self.width)
            .field("block", self.block)
            .field("input_stride", self.input_stride)
            .field("bias", self.bias)
            .field("absmax", self.absmax)
            .finish(formatter)
    }
}

/// The arguments of `value_first_swiglu`.
pub struct ValueFirstSwigluArgs {
    /// `[..., 2 · width]`, contiguous, value half first: the packed projection.
    pub value_gate: TensorView,
    /// `[2 · width]`, contiguous: the packed bias, present exactly when the descriptor adds one.
    pub bias: Option<TensorView>,
    /// `[..., width]`, contiguous: receives `value · silu(gate)`.
    pub out: TensorView,
    /// `[programs]`, FP32: each program's largest output magnitude, present exactly when the
    /// descriptor asks for it.
    pub partials: Option<TensorView>,
}

/// `value · silu(gate)` over a packed `[value, gate]` projection.
pub struct ValueFirstSwiglu;

impl Op for ValueFirstSwiglu {
    const NAME: &'static str = "value_first_swiglu";

    type Descriptor = ValueFirstSwigluDescriptor;
    type Args<'a> = ValueFirstSwigluArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        if width == 0 || descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a row or a block of zero elements computes nothing".to_owned(),
            ));
        }
        args.out.require_row(Self::NAME, descriptor.dtype, width)?;
        let rows = args.out.rows();

        args.value_gate
            .require_row(Self::NAME, descriptor.dtype, 2 * width)?;
        if args.value_gate.rows() != rows {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the packed projection holds {} rows, the output {rows}",
                    args.value_gate.rows()
                ),
            ));
        }
        if args.value_gate.shape.rank() > 1
            && args.value_gate.strides.get(0) != descriptor.input_stride
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the packed rows are {} elements apart, the descriptor says {}",
                    args.value_gate.strides.get(0),
                    descriptor.input_stride
                ),
            ));
        }
        match (&args.bias, descriptor.bias) {
            (Some(bias), true) => {
                bias.require_shape(Self::NAME, Shape::new(&[2 * width]))?;
                bias.require_row(Self::NAME, descriptor.dtype, 2 * width)?;
            }
            (None, false) => {}
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor adds a bias but none was given".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a bias was given but the descriptor adds none".to_owned(),
                ));
            }
        }

        let elements = rows * width;
        let programs = elements.div_ceil(u64::from(descriptor.block));
        match (&args.partials, descriptor.absmax) {
            (Some(partials), true) => {
                partials.require_shape(Self::NAME, Shape::new(&[programs]))?;
                partials.require_row(Self::NAME, DType::F32, programs)?;
            }
            (None, false) => {}
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor records magnitudes but no buffer was given".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a magnitude buffer was given but the descriptor records none".to_owned(),
                ));
            }
        }
        if descriptor.input_stride > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the packed row stride {} does not fit the kernel's int32",
                    descriptor.input_stride
                ),
            ));
        }

        // A flag is off, so its slot is never dereferenced; the kernel still takes the argument.
        let bias = args.bias.unwrap_or(args.out);
        let partials = args.partials.unwrap_or(args.out);
        let mut arguments = Arguments::new();
        arguments
            .view(args.value_gate)
            .view(bias)
            .view(args.out)
            .view(partials)
            .i32(elements as i32)
            .i32(descriptor.input_stride as i32);
        kernel.launch(LaunchGrid::linear(programs)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ValueFirstSwigluArgs {
            out: twins.shadow_of(args.out),
            partials: args.partials.map(|partials| twins.shadow_of(partials)),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The sigmoid, its product with the gate, the bias add and the product with the value: four
        // roundings in the output type, and no reduction.
        Tolerance::new(0.0, 4.0 * descriptor.dtype.unit_roundoff() as f32)
    }
}
