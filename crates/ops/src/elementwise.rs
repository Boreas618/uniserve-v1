//! The elementwise primitives: `elementwise_unary`, `elementwise_binary` and `elementwise_where`.
//!
//! ```text
//! elementwise_unary    out[i] = f(x[i])
//! elementwise_binary   out[i] = lhs[i] ∘ rhs[i]         (arithmetic, or a comparison writing u8)
//! elementwise_where    out[i] = condition[i] ≠ 0 ? on_true[i] : on_false[i]
//! ```
//!
//! `i` ranges over one shape that every operand of a call has. An input is read through its
//! strides, so a permuted view needs no copy and a broadcast operand is one whose stride is zero
//! along the axes it repeats; the output is contiguous. The shape and the strides are launch
//! words ([`crate::strided`]): a descriptor names the function, the element type, the axes the
//! call traverses and the block, and a program serves every shape with those.
//!
//! # Arithmetic
//!
//! A half-precision element is widened to FP32, the function is evaluated in FP32, and the result
//! rounds to nearest even once into the element type; FP32 and FP64 evaluate in their own type.
//! An integer evaluates in its own type and wraps, and an integer quotient truncates toward zero;
//! a zero integer divisor leaves its element unspecified. `maximum` and `minimum` return the
//! left operand unless the right one is strictly beyond it, and a comparison with a NaN is false
//! (`ne` alone is true), as IEEE 754 orders them.
//!
//! The unary functions follow the tensor library's host formulas:
//!
//! ```text
//! recip     1 / x                       sqr       x · x
//! relu      max(x, 0)                   sign      (x > 0) − (x < 0)
//! sigmoid   1 / (1 + exp(−x))           silu      x / (1 + exp(−x))
//! gelu      x / (1 + exp(−√(8/π) · x · (1 + 0.044715 · x²)))
//! gelu_erf  (erf(x / √2) + 1) · 0.5 · x
//! round     the nearest integer, a half away from zero
//! affine    x · a + b                   powf      xᵃ
//! elu       x > 0 ? x : (exp(x) − 1) · a
//! ```
//!
//! `gelu` is the tanh approximation `0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))` written as one
//! logistic, which is the same function with one exponential and no cancellation at either end. `a`
//! and `b` are the scalars of the call, converted to the type the function evaluates in. On an
//! integer element only the functions closed over the integers exist ([`UnaryFunction::integral`]).

use std::fmt;

use uniserve_tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::strided;
use crate::tensor::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The bound on `|x|`, and on elu's `|a|`, that the absolute part of `gelu`'s, `gelu_erf`'s and
/// `elu`'s tolerance is stated for: the activations a normalized layer feeds an activation
/// function.
const VALUE_ENVELOPE: f32 = 8.0;

/// The function one compilation of `elementwise_unary` evaluates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryFunction {
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Neg,
    Recip,
    Sqr,
    Sqrt,
    Abs,
    /// The tanh approximation of the Gaussian error linear unit.
    Gelu,
    /// The Gaussian error linear unit through the error function.
    GeluErf,
    Erf,
    Relu,
    Silu,
    Sigmoid,
    Ceil,
    Floor,
    /// The nearest integer, a half away from zero.
    Round,
    Sign,
    /// `x · a + b`, with the call's two scalars.
    Affine,
    /// `xᵃ`, with the call's first scalar.
    Powf,
    /// `x` where it is positive and `(exp(x) − 1) · a` elsewhere, with the call's first scalar.
    Elu,
}

impl UnaryFunction {
    /// Every function, in the order the package declares them.
    pub const ALL: [Self; 23] = [
        Self::Exp,
        Self::Log,
        Self::Sin,
        Self::Cos,
        Self::Tanh,
        Self::Neg,
        Self::Recip,
        Self::Sqr,
        Self::Sqrt,
        Self::Abs,
        Self::Gelu,
        Self::GeluErf,
        Self::Erf,
        Self::Relu,
        Self::Silu,
        Self::Sigmoid,
        Self::Ceil,
        Self::Floor,
        Self::Round,
        Self::Sign,
        Self::Affine,
        Self::Powf,
        Self::Elu,
    ];

    /// Whether the function is closed over the integers, and so exists on an integer element.
    ///
    /// `ceil`, `floor` and `round` are the identity there; the rest wrap as integer arithmetic
    /// does.
    pub const fn integral(self) -> bool {
        matches!(
            self,
            Self::Neg
                | Self::Sqr
                | Self::Abs
                | Self::Relu
                | Self::Ceil
                | Self::Floor
                | Self::Round
                | Self::Sign
                | Self::Affine
        )
    }

    /// The rounded operations the function's formula performs before the result is stored: none
    /// for a function whose result is one of the element type's own values.
    const fn roundings(self) -> u64 {
        match self {
            Self::Neg | Self::Abs | Self::Relu | Self::Ceil | Self::Floor | Self::Round => 0,
            Self::Sign => 0,
            Self::Exp | Self::Log | Self::Sin | Self::Cos | Self::Tanh | Self::Erf => 1,
            Self::Recip | Self::Sqr | Self::Sqrt | Self::Powf => 1,
            // The product and the sum.
            Self::Affine => 2,
            // The exponential, the sum with one and the quotient.
            Self::Sigmoid | Self::Silu => 3,
            // The exponential, the difference from one and the product with the scalar.
            Self::Elu => 3,
            // The scaled argument, the error function, the sum with one and the product.
            Self::GeluErf => 4,
            // The square, its two products and the sum that form the argument, then the scale,
            // the exponential, the sum with one and the quotient.
            Self::Gelu => 8,
        }
    }
}

impl fmt::Display for UnaryFunction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Neg => "neg",
            Self::Recip => "recip",
            Self::Sqr => "sqr",
            Self::Sqrt => "sqrt",
            Self::Abs => "abs",
            Self::Gelu => "gelu",
            Self::GeluErf => "gelu_erf",
            Self::Erf => "erf",
            Self::Relu => "relu",
            Self::Silu => "silu",
            Self::Sigmoid => "sigmoid",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::Round => "round",
            Self::Sign => "sign",
            Self::Affine => "affine",
            Self::Powf => "powf",
            Self::Elu => "elu",
        })
    }
}

/// One compilation of `elementwise_unary`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UnaryDescriptor {
    /// The function the module evaluates.
    pub function: UnaryFunction,
    /// Element type of the input and the output.
    pub dtype: DType,
    /// The axes the call traverses ([`crate::strided`]).
    pub rank: u8,
    /// Elements one thread block evaluates.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl UnaryDescriptor {
    /// The descriptor of the call `args` makes of `function`.
    pub fn of(function: UnaryFunction, args: &UnaryArgs, block: u32, pdl: bool) -> Self {
        Self {
            function,
            dtype: args.x.dtype,
            rank: strided::rank([args.x, args.out]),
            block,
            pdl,
        }
    }
}

impl fmt::Display for UnaryDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("function", self.function)
            .field("dtype", abi::name(self.dtype))
            .field("rank", self.rank)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `elementwise_unary`.
#[derive(Clone, Copy)]
pub struct UnaryArgs {
    /// The values, at any strides.
    pub x: Operand,
    /// The result: the shape of `x`, contiguous, in its element type.
    pub out: Operand,
    /// The function's scalars `a` and `b`: affine's factor and addend, powf's exponent or elu's
    /// factor in the first. A function that takes none reads neither.
    pub scalars: [f64; 2],
}

/// `out[i] = f(x[i])` over a strided input.
pub struct ElementwiseUnary;

/// Pack one of a unary function's scalars as the type its function evaluates in.
///
/// An integer element takes the scalar converted as a host cast converts it, saturating at the
/// element type's bounds, which is the value the tensor library's host arithmetic uses.
fn pack_scalar(arguments: &mut Arguments, dtype: DType, scalar: f64) {
    match dtype {
        DType::F64 => arguments.f64(scalar),
        DType::U8 => arguments.i64(i64::from(scalar as u8)),
        DType::U32 => arguments.i64(i64::from(scalar as u32)),
        DType::I16 => arguments.i64(i64::from(scalar as i16)),
        DType::I32 => arguments.i64(i64::from(scalar as i32)),
        DType::I64 => arguments.i64(scalar as i64),
        _ => arguments.f32(scalar as f32),
    };
}

impl Op for ElementwiseUnary {
    const NAME: &'static str = proto::program::operator::ELEMENTWISE_UNARY;

    type Descriptor = UnaryDescriptor;
    type Args<'a> = UnaryArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, &strided::ELEMENTS)?;
        if !descriptor.dtype.is_float() && !descriptor.function.integral() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} is not closed over the integers and the descriptor's dtype is {}",
                    descriptor.function, abi::name(descriptor.dtype)
                ),
            ));
        }
        strided::require_block(Self::NAME, descriptor.block)?;

        strided::require_dtype(Self::NAME, "x", &args.x, descriptor.dtype)?;
        strided::require_dtype(Self::NAME, "out", &args.out, descriptor.dtype)?;
        strided::require_shape(Self::NAME, "out", &args.out, args.x.shape)?;
        strided::require_elements(Self::NAME, "x", &args.x)?;
        strided::require_contiguous(Self::NAME, "out", &args.out)?;

        let traversed = strided::rank([args.x, args.out]);
        strided::require_rank(Self::NAME, "rank", descriptor.rank, traversed)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let [x, out] = strided::merge_axes([args.x, args.out]);
        let elements = out.numel();

        let mut arguments = Arguments::new();
        arguments.view(x).view(out).u64(elements);
        strided::pack_extents(&mut arguments, out.shape);
        strided::pack_strides(&mut arguments, x.strides);
        for scalar in args.scalars {
            pack_scalar(&mut arguments, descriptor.dtype, scalar);
        }
        arguments.flag(descriptor.pdl);

        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        UnaryArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let dtype = descriptor.dtype;
        let roundings = descriptor.function.roundings();
        if !dtype.is_float() || roundings == 0 {
            // Integer arithmetic wraps the same way everywhere, and a sign change, a selection or
            // a rounding to an integer returns one of the element type's own values.
            return Tolerance::exact();
        }

        let compute = tolerance::accumulate(dtype);
        match descriptor.function {
            // A product and a sum of either sign: the sum may cancel below its two terms.
            UnaryFunction::Affine => Tolerance::dot(roundings, compute, dtype),
            // Where the result is far below its argument, its error is not relative to it. A
            // transcendental near minus one, or near one, meets a one of the other sign, and the
            // difference keeps the transcendental's last place as an absolute error; an
            // exponential of a rounded argument carries that rounding scaled by the argument. The
            // product with the value scales either by at most the envelope, once per
            // implementation.
            UnaryFunction::Gelu | UnaryFunction::GeluErf | UnaryFunction::Elu => {
                Tolerance::rounding(roundings, compute, dtype)
                    .with_atol(2.0 * VALUE_ENVELOPE * tolerance::unit_roundoff(compute) as f32)
            }
            // Products, quotients, sums of one sign and library functions accurate to their last
            // place: each rounds relative to its own result.
            _ => Tolerance::rounding(roundings, compute, dtype),
        }
    }
}

/// The function one compilation of `elementwise_binary` evaluates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryFunction {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryFunction {
    /// Every function, in the order the package declares them.
    pub const ALL: [Self; 12] = [
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Div,
        Self::Maximum,
        Self::Minimum,
        Self::Eq,
        Self::Ne,
        Self::Lt,
        Self::Le,
        Self::Gt,
        Self::Ge,
    ];

    /// Whether the function is a comparison, whose result is `u8`: one where it holds, zero
    /// where it does not.
    pub const fn compares(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge
        )
    }

    /// The element type of the result over operands of `dtype`.
    pub const fn output(self, dtype: DType) -> DType {
        if self.compares() { DType::U8 } else { dtype }
    }
}

impl fmt::Display for BinaryFunction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        })
    }
}

/// One compilation of `elementwise_binary`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BinaryDescriptor {
    /// The function the module evaluates.
    pub function: BinaryFunction,
    /// Element type of both inputs, and of the output unless the function compares.
    pub dtype: DType,
    /// The axes the call traverses ([`crate::strided`]).
    pub rank: u8,
    /// Elements one thread block evaluates.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl BinaryDescriptor {
    /// The descriptor of the call `args` makes of `function`.
    pub fn of(function: BinaryFunction, args: &BinaryArgs, block: u32, pdl: bool) -> Self {
        Self {
            function,
            dtype: args.lhs.dtype,
            rank: strided::rank([args.lhs, args.rhs, args.out]),
            block,
            pdl,
        }
    }
}

impl fmt::Display for BinaryDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("function", self.function)
            .field("dtype", abi::name(self.dtype))
            .field("rank", self.rank)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `elementwise_binary`.
#[derive(Clone, Copy)]
pub struct BinaryArgs {
    /// The left operand, at any strides; a zero stride repeats an axis.
    pub lhs: Operand,
    /// The right operand, in the shape of `lhs`, at any strides.
    pub rhs: Operand,
    /// The result: the shape of `lhs`, contiguous; `u8` for a comparison.
    pub out: Operand,
}

/// `out[i] = lhs[i] ∘ rhs[i]` over two strided inputs.
pub struct ElementwiseBinary;

impl Op for ElementwiseBinary {
    const NAME: &'static str = proto::program::operator::ELEMENTWISE_BINARY;

    type Descriptor = BinaryDescriptor;
    type Args<'a> = BinaryArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, &strided::ELEMENTS)?;
        strided::require_block(Self::NAME, descriptor.block)?;

        let output = descriptor.function.output(descriptor.dtype);
        strided::require_dtype(Self::NAME, "lhs", &args.lhs, descriptor.dtype)?;
        strided::require_dtype(Self::NAME, "rhs", &args.rhs, descriptor.dtype)?;
        strided::require_dtype(Self::NAME, "out", &args.out, output)?;
        strided::require_shape(Self::NAME, "rhs", &args.rhs, args.lhs.shape)?;
        strided::require_shape(Self::NAME, "out", &args.out, args.lhs.shape)?;
        strided::require_elements(Self::NAME, "lhs", &args.lhs)?;
        strided::require_elements(Self::NAME, "rhs", &args.rhs)?;
        strided::require_contiguous(Self::NAME, "out", &args.out)?;

        let traversed = strided::rank([args.lhs, args.rhs, args.out]);
        strided::require_rank(Self::NAME, "rank", descriptor.rank, traversed)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let [lhs, rhs, out] = strided::merge_axes([args.lhs, args.rhs, args.out]);
        let elements = out.numel();

        let mut arguments = Arguments::new();
        arguments.view(lhs).view(rhs).view(out).u64(elements);
        strided::pack_extents(&mut arguments, out.shape);
        strided::pack_strides(&mut arguments, lhs.strides);
        strided::pack_strides(&mut arguments, rhs.strides);
        arguments.flag(descriptor.pdl);

        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        BinaryArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let dtype = descriptor.dtype;
        match descriptor.function {
            BinaryFunction::Add
            | BinaryFunction::Sub
            | BinaryFunction::Mul
            | BinaryFunction::Div
                if dtype.is_float() =>
            {
                // One operation in the computing type, rounded once into the element type.
                Tolerance::rounding(1, tolerance::accumulate(dtype), dtype)
            }
            // Integer arithmetic wraps the same way everywhere; a maximum or a minimum is one of
            // its operands, and a comparison that differs is a wrong decision, not a rounding.
            _ => Tolerance::exact(),
        }
    }
}

/// One compilation of `elementwise_where`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WhereDescriptor {
    /// Element type of both values and of the output.
    pub dtype: DType,
    /// Element type of the condition: `u8`, `u32` or `i64`.
    pub condition: DType,
    /// The axes the call traverses ([`crate::strided`]).
    pub rank: u8,
    /// Elements one thread block selects.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl WhereDescriptor {
    /// The descriptor of the call `args` makes.
    pub fn of(args: &WhereArgs, block: u32, pdl: bool) -> Self {
        Self {
            dtype: args.on_true.dtype,
            condition: args.condition.dtype,
            rank: strided::rank([args.condition, args.on_true, args.on_false, args.out]),
            block,
            pdl,
        }
    }
}

impl fmt::Display for WhereDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("condition", abi::name(self.condition))
            .field("rank", self.rank)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `elementwise_where`.
#[derive(Clone, Copy)]
pub struct WhereArgs {
    /// The condition, at any strides: an element is taken from `on_true` where it is not zero.
    pub condition: Operand,
    /// The values where the condition holds, in the shape of `condition`, at any strides.
    pub on_true: Operand,
    /// The values where it does not, in the shape of `condition`, at any strides.
    pub on_false: Operand,
    /// The result: the shape of `condition`, contiguous, in the values' element type.
    pub out: Operand,
}

/// `out[i] = condition[i] ≠ 0 ? on_true[i] : on_false[i]` over three strided inputs.
pub struct ElementwiseWhere;

impl Op for ElementwiseWhere {
    const NAME: &'static str = proto::program::operator::ELEMENTWISE_WHERE;

    type Descriptor = WhereDescriptor;
    type Args<'a> = WhereArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, &strided::ELEMENTS)?;
        strided::require_element(
            Self::NAME,
            "condition",
            descriptor.condition,
            &strided::INDICES,
        )?;
        strided::require_block(Self::NAME, descriptor.block)?;

        let shape = args.condition.shape;
        strided::require_dtype(
            Self::NAME,
            "condition",
            &args.condition,
            descriptor.condition,
        )?;
        for (name, operand) in [
            ("on_true", &args.on_true),
            ("on_false", &args.on_false),
            ("out", &args.out),
        ] {
            strided::require_dtype(Self::NAME, name, operand, descriptor.dtype)?;
            strided::require_shape(Self::NAME, name, operand, shape)?;
        }
        strided::require_elements(Self::NAME, "condition", &args.condition)?;
        strided::require_elements(Self::NAME, "on_true", &args.on_true)?;
        strided::require_elements(Self::NAME, "on_false", &args.on_false)?;
        strided::require_contiguous(Self::NAME, "out", &args.out)?;

        let traversed = strided::rank([args.condition, args.on_true, args.on_false, args.out]);
        strided::require_rank(Self::NAME, "rank", descriptor.rank, traversed)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let [condition, on_true, on_false, out] =
            strided::merge_axes([args.condition, args.on_true, args.on_false, args.out]);
        let elements = out.numel();

        let mut arguments = Arguments::new();
        arguments
            .view(condition)
            .view(on_true)
            .view(on_false)
            .view(out)
            .u64(elements);
        strided::pack_extents(&mut arguments, out.shape);
        strided::pack_strides(&mut arguments, condition.strides);
        strided::pack_strides(&mut arguments, on_true.strides);
        strided::pack_strides(&mut arguments, on_false.strides);
        arguments.flag(descriptor.pdl);

        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        WhereArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A selection moves one of its two values: any difference is a wrong element.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Shape, Strides};

    const BLOCK: u32 = 1024;

    fn operand(dtype: DType, shape: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(
            DevicePtr::new(0x1000),
            dtype,
            Shape::new(shape),
            Strides::new(strides),
        )
    }

    fn contiguous(dtype: DType, shape: &[u64]) -> Operand {
        Operand::new(DevicePtr::new(0x2000), dtype, Shape::new(shape))
    }

    fn unary(x: Operand) -> UnaryArgs {
        UnaryArgs {
            x,
            out: contiguous(x.dtype, x.shape.dims()),
            scalars: [0.0; 2],
        }
    }

    fn refusal<T: std::fmt::Debug>(result: Result<T>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn a_unary_key_names_the_function_the_type_and_the_axes_traversed() {
        let args = unary(contiguous(DType::BF16, &[2, 3, 5]));
        let descriptor = UnaryDescriptor::of(UnaryFunction::GeluErf, &args, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "function_gelu_erf_dtype_bf16_rank_1_block_1024_pdl_false"
        );
        ElementwiseUnary::check(&descriptor, &args).unwrap();
    }

    #[test]
    fn every_unary_function_has_a_name_of_its_own() {
        let names: std::collections::BTreeSet<String> =
            UnaryFunction::ALL.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), UnaryFunction::ALL.len());
    }

    #[test]
    fn contiguous_operands_of_any_shape_traverse_one_axis_and_a_permuted_one_traverses_its_own() {
        // Every shape of a contiguous call is one run of elements.
        for shape in [&[7][..], &[2, 3], &[2, 3, 5], &[2, 1, 3, 1, 5]] {
            let args = unary(contiguous(DType::F32, shape));
            let descriptor = UnaryDescriptor::of(UnaryFunction::Exp, &args, BLOCK, false);
            assert_eq!(descriptor.rank, 1, "{shape:?}");
        }

        // `[2, 3, 5]` read as `[5, 2, 3]`: the last two axes are still one run, the first is not.
        let permuted = unary(operand(DType::F32, &[5, 2, 3], &[1, 15, 5]));
        let descriptor = UnaryDescriptor::of(UnaryFunction::Exp, &permuted, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        ElementwiseUnary::check(&descriptor, &permuted).unwrap();

        // The fully reversed order shares no run at all.
        let reversed = unary(operand(DType::F32, &[5, 3, 2], &[1, 5, 15]));
        let descriptor = UnaryDescriptor::of(UnaryFunction::Exp, &reversed, BLOCK, false);
        assert_eq!(descriptor.rank, 3);

        // An operand of no axis is one element.
        let scalar = unary(contiguous(DType::F32, &[]));
        let descriptor = UnaryDescriptor::of(UnaryFunction::Exp, &scalar, BLOCK, false);
        assert_eq!(descriptor.rank, 1);
        ElementwiseUnary::check(&descriptor, &scalar).unwrap();
    }

    #[test]
    fn a_unary_call_has_one_descriptor_and_any_other_rank_is_refused() {
        let args = unary(contiguous(DType::F32, &[4, 8]));
        let stated = UnaryDescriptor {
            rank: 2,
            ..UnaryDescriptor::of(UnaryFunction::Exp, &args, BLOCK, false)
        };
        let message = refusal(ElementwiseUnary::check(&stated, &args));
        assert!(message.contains("traverse 1 axes"), "{message}");
    }

    #[test]
    fn a_unary_call_states_its_types_its_shape_and_a_contiguous_output() {
        let x = contiguous(DType::F32, &[4, 8]);
        let descriptor = UnaryDescriptor::of(UnaryFunction::Exp, &unary(x), BLOCK, false);

        let other_type = UnaryArgs {
            out: contiguous(DType::BF16, &[4, 8]),
            ..unary(x)
        };
        let message = refusal(ElementwiseUnary::check(&descriptor, &other_type));
        assert!(message.contains("expected f32 out"), "{message}");

        let other_shape = UnaryArgs {
            out: contiguous(DType::F32, &[8, 4]),
            ..unary(x)
        };
        let message = refusal(ElementwiseUnary::check(&descriptor, &other_shape));
        assert!(message.contains("expected out of shape"), "{message}");

        let strided_out = UnaryArgs {
            out: operand(DType::F32, &[4, 8], &[16, 1]),
            ..unary(x)
        };
        let message = refusal(ElementwiseUnary::check(&descriptor, &strided_out));
        assert!(message.contains("out must be contiguous"), "{message}");

        let empty = unary(contiguous(DType::F32, &[4, 0]));
        let descriptor = UnaryDescriptor::of(UnaryFunction::Exp, &empty, BLOCK, false);
        let message = refusal(ElementwiseUnary::check(&descriptor, &empty));
        assert!(message.contains("nothing to launch"), "{message}");

        let odd_block = UnaryDescriptor {
            block: 1000,
            ..UnaryDescriptor::of(UnaryFunction::Exp, &unary(x), BLOCK, false)
        };
        let message = refusal(ElementwiseUnary::check(&odd_block, &unary(x)));
        assert!(message.contains("not a power of two"), "{message}");
    }

    #[test]
    fn an_integer_element_takes_only_the_functions_closed_over_the_integers() {
        let args = unary(contiguous(DType::I64, &[16]));
        for function in UnaryFunction::ALL {
            let descriptor = UnaryDescriptor::of(function, &args, BLOCK, false);
            let checked = ElementwiseUnary::check(&descriptor, &args);
            assert_eq!(checked.is_ok(), function.integral(), "{function}");
        }

        // A narrow float is no element of a primitive.
        let packed = unary(contiguous(DType::F8E4M3, &[16]));
        let descriptor = UnaryDescriptor::of(UnaryFunction::Neg, &packed, BLOCK, false);
        let message = refusal(ElementwiseUnary::check(&descriptor, &packed));
        assert!(message.contains("dtype is f8e4m3"), "{message}");
    }

    #[test]
    fn a_unary_tolerance_follows_the_formula_and_the_element_type() {
        let tolerance = |function, dtype| {
            let args = unary(contiguous(dtype, &[16]));
            ElementwiseUnary::tolerance(&UnaryDescriptor::of(function, &args, BLOCK, false))
        };

        // A sign change or a selection returns one of the type's own values.
        for function in [
            UnaryFunction::Neg,
            UnaryFunction::Relu,
            UnaryFunction::Round,
        ] {
            assert_eq!(tolerance(function, DType::BF16), Tolerance::exact());
        }
        // Integer arithmetic has no rounding.
        assert_eq!(
            tolerance(UnaryFunction::Affine, DType::I32),
            Tolerance::exact()
        );

        // One library function in FP32, rounded once into the element type.
        assert_eq!(
            tolerance(UnaryFunction::Exp, DType::BF16),
            Tolerance::rounding(1, DType::F32, DType::BF16)
        );
        // FP64 evaluates in itself.
        assert_eq!(
            tolerance(UnaryFunction::Exp, DType::F64),
            Tolerance::rounding(1, DType::F64, DType::F64)
        );
        // A product and a sum that may cancel.
        assert_eq!(
            tolerance(UnaryFunction::Affine, DType::F16),
            Tolerance::dot(2, DType::F32, DType::F16)
        );
        // A cancelling difference keeps an absolute part over the envelope.
        let elu = tolerance(UnaryFunction::Elu, DType::F32);
        assert_eq!(
            elu.rtol,
            Tolerance::rounding(3, DType::F32, DType::F32).rtol
        );
        assert_eq!(elu.atol, 16.0 * tolerance::unit_roundoff(DType::F32) as f32);
    }

    fn binary(lhs: Operand, rhs: Operand, function: BinaryFunction) -> BinaryArgs {
        BinaryArgs {
            lhs,
            rhs,
            out: contiguous(function.output(lhs.dtype), lhs.shape.dims()),
        }
    }

    #[test]
    fn a_binary_key_names_the_function_the_type_and_the_axes_traversed() {
        let lhs = contiguous(DType::BF16, &[2, 3, 5]);
        let args = binary(lhs, lhs, BinaryFunction::Maximum);
        let descriptor = BinaryDescriptor::of(BinaryFunction::Maximum, &args, BLOCK, true);
        assert_eq!(
            descriptor.to_string(),
            "function_maximum_dtype_bf16_rank_1_block_1024_pdl_true"
        );
        ElementwiseBinary::check(&descriptor, &args).unwrap();
    }

    #[test]
    fn a_broadcast_operand_is_a_zero_stride_and_traverses_the_axes_it_repeats() {
        let lhs = contiguous(DType::F32, &[2, 3, 5]);

        // One value for every element: no stride at all, and still one run.
        let scalar = operand(DType::F32, &[2, 3, 5], &[0, 0, 0]);
        let args = binary(lhs, scalar, BinaryFunction::Mul);
        let descriptor = BinaryDescriptor::of(BinaryFunction::Mul, &args, BLOCK, false);
        assert_eq!(descriptor.rank, 1);
        ElementwiseBinary::check(&descriptor, &args).unwrap();

        // A bias along the last axis repeats over the two leading ones, which merge.
        let bias = operand(DType::F32, &[2, 3, 5], &[0, 0, 1]);
        let args = binary(lhs, bias, BinaryFunction::Add);
        let descriptor = BinaryDescriptor::of(BinaryFunction::Add, &args, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        ElementwiseBinary::check(&descriptor, &args).unwrap();

        // A value per middle index repeats on both sides of it.
        let middle = operand(DType::F32, &[2, 3, 5], &[0, 1, 0]);
        let args = binary(lhs, middle, BinaryFunction::Add);
        let descriptor = BinaryDescriptor::of(BinaryFunction::Add, &args, BLOCK, false);
        assert_eq!(descriptor.rank, 3);
    }

    #[test]
    fn a_comparison_writes_u8_and_arithmetic_writes_the_operands_type() {
        let lhs = contiguous(DType::F32, &[4, 8]);
        let args = binary(lhs, lhs, BinaryFunction::Lt);
        assert_eq!(args.out.dtype, DType::U8);
        let descriptor = BinaryDescriptor::of(BinaryFunction::Lt, &args, BLOCK, false);
        ElementwiseBinary::check(&descriptor, &args).unwrap();

        // The same operands under an arithmetic function need an output of their own type.
        let sum = BinaryDescriptor {
            function: BinaryFunction::Add,
            ..descriptor
        };
        let message = refusal(ElementwiseBinary::check(&sum, &args));
        assert!(message.contains("expected f32 out"), "{message}");

        let mixed = BinaryArgs {
            rhs: contiguous(DType::BF16, &[4, 8]),
            ..args
        };
        let message = refusal(ElementwiseBinary::check(&descriptor, &mixed));
        assert!(message.contains("expected f32 rhs"), "{message}");

        let unequal = BinaryArgs {
            rhs: contiguous(DType::F32, &[4, 1]),
            ..args
        };
        let message = refusal(ElementwiseBinary::check(&descriptor, &unequal));
        assert!(message.contains("expected rhs of shape"), "{message}");
    }

    #[test]
    fn a_binary_tolerance_is_one_rounding_for_float_arithmetic_and_exact_otherwise() {
        let tolerance = |function, dtype| {
            let lhs = contiguous(dtype, &[16]);
            let args = binary(lhs, lhs, function);
            ElementwiseBinary::tolerance(&BinaryDescriptor::of(function, &args, BLOCK, false))
        };

        assert_eq!(
            tolerance(BinaryFunction::Div, DType::BF16),
            Tolerance::rounding(1, DType::F32, DType::BF16)
        );
        assert_eq!(
            tolerance(BinaryFunction::Div, DType::I32),
            Tolerance::exact()
        );
        assert_eq!(
            tolerance(BinaryFunction::Minimum, DType::BF16),
            Tolerance::exact()
        );
        assert_eq!(
            tolerance(BinaryFunction::Ge, DType::F32),
            Tolerance::exact()
        );
    }

    fn selection(condition: Operand, values: Operand) -> WhereArgs {
        WhereArgs {
            condition,
            on_true: values,
            on_false: values,
            out: contiguous(values.dtype, values.shape.dims()),
        }
    }

    #[test]
    fn a_where_key_names_both_types_and_the_axes_traversed() {
        let args = selection(
            contiguous(DType::U8, &[4, 8]),
            contiguous(DType::BF16, &[4, 8]),
        );
        let descriptor = WhereDescriptor::of(&args, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "dtype_bf16_condition_u8_rank_1_block_1024_pdl_false"
        );
        ElementwiseWhere::check(&descriptor, &args).unwrap();
        assert_eq!(ElementwiseWhere::tolerance(&descriptor), Tolerance::exact());
    }

    #[test]
    fn a_condition_is_u8_u32_or_i64_and_shares_the_values_shape() {
        let values = contiguous(DType::F32, &[4, 8]);

        for condition in [DType::U8, DType::U32, DType::I64] {
            let args = selection(contiguous(condition, &[4, 8]), values);
            ElementwiseWhere::check(&WhereDescriptor::of(&args, BLOCK, false), &args).unwrap();
        }

        let args = selection(contiguous(DType::I32, &[4, 8]), values);
        let message = refusal(ElementwiseWhere::check(
            &WhereDescriptor::of(&args, BLOCK, false),
            &args,
        ));
        assert!(message.contains("condition is i32"), "{message}");

        let args = selection(contiguous(DType::U8, &[4]), values);
        let message = refusal(ElementwiseWhere::check(
            &WhereDescriptor::of(&args, BLOCK, false),
            &args,
        ));
        assert!(message.contains("expected on_true of shape"), "{message}");

        // A row-wise condition repeated along the last axis traverses two axes.
        let args = selection(operand(DType::U8, &[4, 8], &[1, 0]), values);
        let descriptor = WhereDescriptor::of(&args, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        ElementwiseWhere::check(&descriptor, &args).unwrap();
    }
}
