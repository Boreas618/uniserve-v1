//! Tensor Operation Enums and Traits
//!
#![allow(clippy::redundant_closure_call)]
use float8::F8E4M3 as f8e4m3;
use half::{bf16, f16};
use num_traits::float::Float;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl ReduceOp {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ArgMax => "argmax",
            Self::ArgMin => "argmin",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }
}

/// The element-wise operations of two operands of one shape and one element type. The result
/// has that element type.
///
/// On an integer element type the arithmetic is the element type's own: `Add`, `Sub` and `Mul`
/// wrap, and `Div` truncates toward zero. The operations from `Remainder` on exist on the
/// integer element types alone ([`BinaryOp::integral`]) and are named as PyTorch names them:
///
/// - `Remainder` is `torch.remainder`: the remainder of the floored quotient, which is zero or
///   has the sign of the divisor;
/// - `BitwiseAnd`, `BitwiseOr` and `BitwiseXor` combine the bits of the two elements;
/// - `BitwiseLeftShift` and `BitwiseRightShift` shift the left element by the right one. A
///   right shift of a signed element repeats its sign bit. A count that is negative or not
///   below the element's width in bits leaves its element unspecified.
///
/// A zero divisor of `Div` or `Remainder` on an integer element type is an error on the CPU
/// backend and leaves its element unspecified on a device backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
    Remainder,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseLeftShift,
    BitwiseRightShift,
}

impl BinaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => Add::NAME,
            Self::Mul => Mul::NAME,
            Self::Sub => Sub::NAME,
            Self::Div => Div::NAME,
            Self::Maximum => Maximum::NAME,
            Self::Minimum => Minimum::NAME,
            Self::Remainder => "remainder",
            Self::BitwiseAnd => "bitwise_and",
            Self::BitwiseOr => "bitwise_or",
            Self::BitwiseXor => "bitwise_xor",
            Self::BitwiseLeftShift => "bitwise_left_shift",
            Self::BitwiseRightShift => "bitwise_right_shift",
        }
    }

    /// Whether the operation exists on the integer element types alone.
    pub fn integral(&self) -> bool {
        match self {
            Self::Add | Self::Mul | Self::Sub | Self::Div | Self::Maximum | Self::Minimum => false,
            Self::Remainder
            | Self::BitwiseAnd
            | Self::BitwiseOr
            | Self::BitwiseXor
            | Self::BitwiseLeftShift
            | Self::BitwiseRightShift => true,
        }
    }
}

// Unary ops with no argument
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Neg,
    Recip,
    Sqr,
    Sqrt,
    Gelu,
    GeluErf,
    Erf,
    Relu,
    Silu,
    Sigmoid,
    Tanh,
    Floor,
    Ceil,
    Round,
    Sign,
}

impl UnaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Exp => Exp::NAME,
            Self::Log => Log::NAME,
            Self::Sin => Sin::NAME,
            Self::Cos => Cos::NAME,
            Self::Abs => Abs::NAME,
            Self::Neg => Neg::NAME,
            Self::Recip => Recip::NAME,
            Self::Sqr => Sqr::NAME,
            Self::Sqrt => Sqrt::NAME,
            Self::Gelu => Gelu::NAME,
            Self::GeluErf => GeluErf::NAME,
            Self::Erf => Erf::NAME,
            Self::Relu => Relu::NAME,
            Self::Silu => Silu::NAME,
            Self::Sigmoid => Sigmoid::NAME,
            Self::Tanh => Tanh::NAME,
            Self::Floor => Floor::NAME,
            Self::Ceil => Ceil::NAME,
            Self::Round => Round::NAME,
            Self::Sign => Sign::NAME,
        }
    }
}

pub trait UnaryOpT {
    const NAME: &'static str;
    fn bf16(v1: bf16) -> bf16;
    fn f16(v1: f16) -> f16;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u8(v1: u8) -> u8;
    fn u32(v1: u32) -> u32;
    fn i16(v1: i16) -> i16;
    fn i32(v1: i32) -> i32;
    fn i64(v1: i64) -> i64;
    fn f8e4m3(v1: f8e4m3) -> f8e4m3;

    #[inline(always)]
    fn bf16_vec(xs: &[bf16], ys: &mut [bf16]) {
        xs.iter().zip(ys).for_each(|(&x, y)| *y = Self::bf16(x))
    }
    #[inline(always)]
    fn f16_vec(xs: &[f16], ys: &mut [f16]) {
        xs.iter().zip(ys).for_each(|(&x, y)| *y = Self::f16(x))
    }
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        xs.iter().zip(ys).for_each(|(&x, y)| *y = Self::f32(x))
    }
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        xs.iter().zip(ys).for_each(|(&x, y)| *y = Self::f64(x))
    }
}

pub trait BinaryOpT {
    const NAME: &'static str;
    fn bf16(v1: bf16, v2: bf16) -> bf16;
    fn f16(v1: f16, v2: f16) -> f16;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
    fn u32(v1: u32, v2: u32) -> u32;
    fn i16(v1: i16, v2: i16) -> i16;
    fn i32(v1: i32, v2: i32) -> i32;
    fn i64(v1: i64, v2: i64) -> i64;
    fn f8e4m3(v1: f8e4m3, v2: f8e4m3) -> f8e4m3;

    #[inline(always)]
    fn bf16_vec(xs1: &[bf16], xs2: &[bf16], ys: &mut [bf16]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::bf16(a, b))
    }
    #[inline(always)]
    fn f16_vec(xs1: &[f16], xs2: &[f16], ys: &mut [f16]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::f16(a, b))
    }
    #[inline(always)]
    fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::f32(a, b))
    }
    #[inline(always)]
    fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::f64(a, b))
    }
    #[inline(always)]
    fn u8_vec(xs1: &[u8], xs2: &[u8], ys: &mut [u8]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::u8(a, b))
    }
    #[inline(always)]
    fn u32_vec(xs1: &[u32], xs2: &[u32], ys: &mut [u32]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::u32(a, b))
    }
    #[inline(always)]
    fn i16_vec(xs1: &[i16], xs2: &[i16], ys: &mut [i16]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::i16(a, b))
    }
    #[inline(always)]
    fn i32_vec(xs1: &[i32], xs2: &[i32], ys: &mut [i32]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::i32(a, b))
    }
    #[inline(always)]
    fn i64_vec(xs1: &[i64], xs2: &[i64], ys: &mut [i64]) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .for_each(|((&a, &b), y)| *y = Self::i64(a, b))
    }

    // Scalar-broadcast variants: ys[i] = f(xs[i], scalar).
    // Used by binary_map_vec for the (1,0) inner-stride branch where one tensor
    // is contiguous and the other broadcasts a single value.
    #[inline(always)]
    fn bf16_scalar_vec(scalar: bf16, xs: &[bf16], ys: &mut [bf16]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::bf16(x, scalar))
    }
    #[inline(always)]
    fn f16_scalar_vec(scalar: f16, xs: &[f16], ys: &mut [f16]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::f16(x, scalar))
    }
    #[inline(always)]
    fn f32_scalar_vec(scalar: f32, xs: &[f32], ys: &mut [f32]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::f32(x, scalar))
    }
    #[inline(always)]
    fn f64_scalar_vec(scalar: f64, xs: &[f64], ys: &mut [f64]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::f64(x, scalar))
    }
    #[inline(always)]
    fn u8_scalar_vec(scalar: u8, xs: &[u8], ys: &mut [u8]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::u8(x, scalar))
    }
    #[inline(always)]
    fn u32_scalar_vec(scalar: u32, xs: &[u32], ys: &mut [u32]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::u32(x, scalar))
    }
    #[inline(always)]
    fn i16_scalar_vec(scalar: i16, xs: &[i16], ys: &mut [i16]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::i16(x, scalar))
    }
    #[inline(always)]
    fn i32_scalar_vec(scalar: i32, xs: &[i32], ys: &mut [i32]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::i32(x, scalar))
    }
    #[inline(always)]
    fn i64_scalar_vec(scalar: i64, xs: &[i64], ys: &mut [i64]) {
        xs.iter()
            .zip(ys)
            .for_each(|(&x, y)| *y = Self::i64(x, scalar))
    }
}

pub struct Add;
pub struct Div;
pub struct Mul;
pub struct Sub;
pub struct Maximum;
pub struct Minimum;
pub struct Exp;
pub struct Log;
pub struct Sin;
pub struct Cos;
pub struct Abs;
pub struct Neg;
pub struct Recip;
pub struct Sqr;
pub struct Sqrt;
pub struct Gelu;
pub struct GeluErf;
pub struct Erf;
pub struct Relu;
pub struct Silu;
pub struct Sigmoid;
pub struct Tanh;
pub struct Floor;
pub struct Ceil;
pub struct Round;
pub struct Sign;

// `$name` is an ident; stringify! derives the NAME string.
// `$e` is the operation on a float element and `$int`, over the operands `$a` and `$b`, the one
// on an integer element, whose arithmetic wraps.
// The optional `$vec_op` and `$scalar_op` name methods of `crate::cpu::kernels::VecOps` that
// override the float slice loops with SIMD implementations.
macro_rules! bin_op {
    (
        $op:ident,
        $name:ident,
        $e:expr,
        |$a:ident, $b:ident| $int:expr
        $(, $vec_op:ident, $scalar_op:ident)?
    ) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = stringify!($name);
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bf16 { $e(v1, v2) }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> f16 { $e(v1, v2) }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> f32 { $e(v1, v2) }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> f64 { $e(v1, v2) }
            #[inline(always)]
            fn u8($a: u8, $b: u8) -> u8 { $int }
            #[inline(always)]
            fn u32($a: u32, $b: u32) -> u32 { $int }
            #[inline(always)]
            fn i16($a: i16, $b: i16) -> i16 { $int }
            #[inline(always)]
            fn i32($a: i32, $b: i32) -> i32 { $int }
            #[inline(always)]
            fn i64($a: i64, $b: i64) -> i64 { $int }
            #[inline(always)]
            fn f8e4m3(v1: f8e4m3, v2: f8e4m3) -> f8e4m3 { $e(v1, v2) }
            $(
                #[inline(always)]
                fn f32_vec(lhs: &[f32], rhs: &[f32], res: &mut [f32]) {
                    <f32 as crate::cpu::kernels::VecOps>::$vec_op(lhs, rhs, res)
                }
                #[inline(always)]
                fn f64_vec(lhs: &[f64], rhs: &[f64], res: &mut [f64]) {
                    <f64 as crate::cpu::kernels::VecOps>::$vec_op(lhs, rhs, res)
                }
                #[inline(always)]
                fn bf16_vec(lhs: &[bf16], rhs: &[bf16], res: &mut [bf16]) {
                    <bf16 as crate::cpu::kernels::VecOps>::$vec_op(lhs, rhs, res)
                }
                #[inline(always)]
                fn f16_vec(lhs: &[f16], rhs: &[f16], res: &mut [f16]) {
                    <f16 as crate::cpu::kernels::VecOps>::$vec_op(lhs, rhs, res)
                }
                #[inline(always)]
                fn bf16_scalar_vec(scalar: bf16, xs: &[bf16], ys: &mut [bf16]) {
                    <bf16 as crate::cpu::kernels::VecOps>::$scalar_op(scalar, xs, ys)
                }
                #[inline(always)]
                fn f16_scalar_vec(scalar: f16, xs: &[f16], ys: &mut [f16]) {
                    <f16 as crate::cpu::kernels::VecOps>::$scalar_op(scalar, xs, ys)
                }
                #[inline(always)]
                fn f32_scalar_vec(scalar: f32, xs: &[f32], ys: &mut [f32]) {
                    <f32 as crate::cpu::kernels::VecOps>::$scalar_op(scalar, xs, ys)
                }
                #[inline(always)]
                fn f64_scalar_vec(scalar: f64, xs: &[f64], ys: &mut [f64]) {
                    <f64 as crate::cpu::kernels::VecOps>::$scalar_op(scalar, xs, ys)
                }
            )?
        }
    };
}

bin_op!(
    Add,
    add,
    |v1, v2| v1 + v2,
    |v1, v2| v1.wrapping_add(v2),
    vec_add,
    scalar_add
);
bin_op!(Sub, sub, |v1, v2| v1 - v2, |v1, v2| v1.wrapping_sub(v2));
bin_op!(
    Mul,
    mul,
    |v1, v2| v1 * v2,
    |v1, v2| v1.wrapping_mul(v2),
    vec_mul,
    scalar_mul
);
// The integer quotient truncates toward zero; the CPU backend refuses a zero divisor before it
// divides.
bin_op!(Div, div, |v1, v2| v1 / v2, |v1, v2| v1.wrapping_div(v2));
bin_op!(
    Minimum,
    minimum,
    |v1, v2| if v1 > v2 { v2 } else { v1 },
    |v1, v2| if v1 > v2 { v2 } else { v1 }
);
bin_op!(
    Maximum,
    maximum,
    |v1, v2| if v1 < v2 { v2 } else { v1 },
    |v1, v2| if v1 < v2 { v2 } else { v1 }
);

/// `torch.remainder` on one pair of integer elements: the remainder of the floored quotient,
/// which is zero or has the sign of `divisor`.
///
/// `divisor` is not zero.
#[inline(always)]
pub(crate) fn remainder<T: num_traits::PrimInt + num_traits::CheckedRem>(
    dividend: T,
    divisor: T,
) -> T {
    // The one remainder that overflows is the smallest value by minus one, which is zero.
    let truncated = dividend.checked_rem(&divisor).unwrap_or_else(T::zero);

    // The truncated quotient leaves a remainder of the dividend's sign; the floored one is a
    // divisor further where the two signs differ. The sum has operands of opposite signs.
    if !truncated.is_zero() && (truncated < T::zero()) != (divisor < T::zero()) {
        truncated + divisor
    } else {
        truncated
    }
}

/// The number of bits a shift moves an element by, from the element that states it.
///
/// A count outside the element's width leaves the result unspecified: the wrapping shifts that
/// take this value reduce it modulo the width.
#[inline(always)]
pub(crate) fn shift_count<T: num_traits::PrimInt>(count: T) -> u32 {
    count.to_u32().unwrap_or(0)
}

#[allow(clippy::redundant_closure_call)]
macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline(always)]
            fn i16(_: i16) -> i16 {
                todo!("no unary function for i16")
            }
            #[inline(always)]
            fn i32(_: i32) -> i32 {
                todo!("no unary function for i32")
            }
            #[inline(always)]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }
            #[inline(always)]
            fn f8e4m3($a: f8e4m3) -> f8e4m3 {
                $e
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp());
unary_op!(Log, "log", v, v.ln());
unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Tanh, "tanh", v, v.tanh());
unary_op!(Neg, "neg", v, -v);
unary_op!(Recip, "recip", v, v.recip());
unary_op!(Sqr, "sqr", v, v * v);
unary_op!(Sqrt, "sqrt", v, v.sqrt());

// Hardcode the value for sqrt(2/pi)
// https://github.com/huggingface/candle/issues/1982
#[allow(clippy::excessive_precision)]
const SQRT_TWO_OVER_PI_F32: f32 = 0.79788456080286535587989211986876373;
#[allow(clippy::excessive_precision)]
const SQRT_TWO_OVER_PI_F64: f64 = 0.79788456080286535587989211986876373;

/// Tanh based approximation of the `gelu` operation
/// GeluErf is the more precise one.
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f32_const(0.5)
            * v
            * (bf16::ONE
                + bf16::tanh(
                    bf16::from_f32_const(SQRT_TWO_OVER_PI_F32)
                        * v
                        * (bf16::ONE + bf16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f32_const(0.5)
            * v
            * (f16::ONE
                + f16::tanh(
                    f16::from_f32_const(SQRT_TWO_OVER_PI_F32)
                        * v
                        * (f16::ONE + f16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        0.5 * v * (1.0 + f32::tanh(SQRT_TWO_OVER_PI_F32 * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        0.5 * v * (1.0 + f64::tanh(SQRT_TWO_OVER_PI_F64 * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f32(0.5)
            * v
            * (f8e4m3::ONE
                + f8e4m3::tanh(
                    f8e4m3::from_f32(SQRT_TWO_OVER_PI_F32)
                        * v
                        * (f8e4m3::ONE + f8e4m3::from_f32(0.044715) * v * v),
                ))
    }
}

/// `erf` operation
/// <https://en.wikipedia.org/wiki/Error_function>
impl UnaryOpT for Erf {
    const NAME: &'static str = "erf";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        crate::cpu::erf::erf_f32(v)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        crate::cpu::erf::erf_f64(v)
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f64(Self::f64(v.to_f64()))
    }
}

/// Silu operation
impl UnaryOpT for Silu {
    const NAME: &'static str = "silu";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v / (bf16::ONE + (-v).exp())
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v / (f16::ONE + (-v).exp())
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v / (f8e4m3::ONE + (-v).exp())
    }
}

/// The logistic function `1 / (1 + exp(-x))`.
///
/// A 16-bit or an 8-bit float is widened to F32, where the function is evaluated, and rounded
/// once into its element type.
impl UnaryOpT for Sigmoid {
    const NAME: &'static str = "sigmoid";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f32(Self::f32(v.to_f32()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f32(Self::f32(v.to_f32()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f32(Self::f32(v.to_f32()))
    }
}

impl UnaryOpT for Abs {
    const NAME: &'static str = "abs";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.abs()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.abs()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.abs()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.abs()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v.abs()
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v.abs()
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v.abs()
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.abs()
    }
}

impl UnaryOpT for Ceil {
    const NAME: &'static str = "ceil";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.ceil()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.ceil()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.ceil()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.ceil()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.ceil()
    }
}

impl UnaryOpT for Floor {
    const NAME: &'static str = "floor";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.floor()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.floor()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.floor()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.floor()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.floor()
    }
}

impl UnaryOpT for Round {
    const NAME: &'static str = "round";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.round()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.round()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.round()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.round()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.round()
    }
}

impl UnaryOpT for GeluErf {
    const NAME: &'static str = "gelu_erf";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        (crate::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        (crate::cpu::erf::erf_f64(v * std::f64::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        0
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        f8e4m3::from_f32(Self::f32(v.to_f32()))
    }
}

impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.max(bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.max(f16::ZERO)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.max(0f32)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.max(0f64)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        v.max(0)
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        v.max(0)
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v.max(0)
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        v.max(f8e4m3::ZERO)
    }
}

impl UnaryOpT for Sign {
    const NAME: &'static str = "sign";
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from((v > bf16::ZERO) as i8) - bf16::from((v < bf16::ZERO) as i8)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from((v > f16::ZERO) as i8) - f16::from((v < f16::ZERO) as i8)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        f32::from(v > 0.) - f32::from(v < 0.)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        f64::from(v > 0.) - f64::from(v < 0.)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        u8::min(1, v)
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        u32::min(1, v)
    }
    #[inline(always)]
    fn i16(v: i16) -> i16 {
        (v > 0) as i16 - (v < 0) as i16
    }
    #[inline(always)]
    fn i32(v: i32) -> i32 {
        (v > 0) as i32 - (v < 0) as i32
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        (v > 0) as i64 - (v < 0) as i64
    }
    #[inline(always)]
    fn f8e4m3(v: f8e4m3) -> f8e4m3 {
        if v > f8e4m3::ZERO {
            f8e4m3::ONE
        } else if v < f8e4m3::ZERO {
            -f8e4m3::ONE
        } else {
            f8e4m3::ZERO
        }
    }
}
