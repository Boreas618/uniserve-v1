//! `cast`: every element of a strided input converted to another element type.
//!
//! ```text
//! out[i] = convert(x[i])                     x of `input` elements, out of `output` elements
//! ```
//!
//! The input is read through its strides and the output is contiguous; the shape and the strides
//! are launch words ([`crate::strided`]), so a descriptor names the two element types, the axes
//! the call traverses and the block.
//!
//! The conversion is the host's `as`:
//!
//! - into an integer from an integer, the low bits are kept, a signed source extending its sign;
//! - into an integer from a float, the value truncates toward zero. A NaN, or a value outside the
//!   target's range, leaves its element unspecified;
//! - into a float, the value rounds to nearest even.
//!
//! A conversion to or from a half-precision type goes through FP32, the one such conversion a
//! program lowers: `i64 → bf16` rounds into FP32 and then into BF16, which is what the tensor
//! library's host conversion does, and `f64 → f16` rounds twice where a direct conversion would
//! round once. The tolerance of an inexact conversion covers both roundings.
//!
//! A conversion between one type and itself moves elements and is `copy_strided`'s.

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

/// One compilation of `cast`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the input.
    pub input: DType,
    /// Element type of the output.
    pub output: DType,
    /// The axes the call traverses ([`crate::strided`]).
    pub rank: u8,
    /// Elements one thread block converts.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The descriptor of the call `args` makes.
    pub fn of(args: &Args, block: u32, pdl: bool) -> Self {
        Self {
            input: args.x.dtype,
            output: args.out.dtype,
            rank: strided::rank([args.x, args.out]),
            block,
            pdl,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("input", abi::name(self.input))
            .field("output", abi::name(self.output))
            .field("rank", self.rank)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `cast`.
#[derive(Clone, Copy)]
pub struct Args {
    /// The values, at any strides.
    pub x: Operand,
    /// The result: the shape of `x`, contiguous, in the output's element type.
    pub out: Operand,
}

/// Element type conversion over a strided input.
pub struct Cast;

/// The bits a value of `dtype` is exact to: an integer's value bits, a float's significand.
fn precision(dtype: DType) -> u32 {
    match dtype {
        DType::U8 | DType::BF16 => 8,
        DType::F16 => 11,
        DType::I16 => 15,
        DType::F32 => 24,
        DType::I32 => 31,
        DType::U32 => 32,
        DType::F64 => 53,
        DType::I64 => 63,
        // No other type is an element of a conversion; `check` refuses it.
        _ => 0,
    }
}

/// Whether `output` holds every value of `input`, so that the conversion rounds nothing.
///
/// An integer target keeps the low bits or truncates, which is exact arithmetic. A float target
/// is exact for an integer that fits its significand, and for a narrower float when the target is
/// FP32 or FP64: a half-precision target has a narrower exponent range than BF16 and a narrower
/// significand than F16, so neither holds the other.
fn holds(input: DType, output: DType) -> bool {
    if !output.is_float() {
        return true;
    }
    let wide = matches!(output, DType::F32 | DType::F64);
    precision(input) <= precision(output) && (wide || !input.is_float())
}

impl Op for Cast {
    const NAME: &'static str = proto::module::operator::CAST;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "input", descriptor.input, &strided::ELEMENTS)?;
        strided::require_element(Self::NAME, "output", descriptor.output, &strided::ELEMENTS)?;
        if descriptor.input == descriptor.output {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a conversion from {} to itself moves elements, which copy_strided does",
                    abi::name(descriptor.input)
                ),
            ));
        }
        strided::require_block(Self::NAME, descriptor.block)?;

        strided::require_dtype(Self::NAME, "x", &args.x, descriptor.input)?;
        strided::require_dtype(Self::NAME, "out", &args.out, descriptor.output)?;
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
        arguments.flag(descriptor.pdl);

        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
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
        if holds(descriptor.input, descriptor.output) {
            // Low bits, a truncation, or a value the target holds: nothing rounds.
            return Tolerance::exact();
        }
        // One rounding into the output, after at most one into the type it is reached through.
        Tolerance::rounding(
            1,
            tolerance::accumulate(descriptor.output),
            descriptor.output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Shape, Strides};

    const BLOCK: u32 = 1024;

    fn contiguous(dtype: DType, shape: &[u64]) -> Operand {
        Operand::new(DevicePtr::new(0x2000), dtype, Shape::new(shape))
    }

    fn args(input: DType, output: DType, shape: &[u64]) -> Args {
        Args {
            x: contiguous(input, shape),
            out: contiguous(output, shape),
        }
    }

    fn refusal(result: Result<()>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn the_key_names_both_types_and_the_axes_traversed() {
        let args = args(DType::F32, DType::BF16, &[2, 3, 5]);
        let descriptor = Descriptor::of(&args, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "input_f32_output_bf16_rank_1_block_1024_pdl_false"
        );
        Cast::check(&descriptor, &args).unwrap();
    }

    #[test]
    fn every_pair_of_distinct_element_types_converts() {
        for input in strided::ELEMENTS {
            for output in strided::ELEMENTS {
                let args = args(input, output, &[4, 8]);
                let checked = Cast::check(&Descriptor::of(&args, BLOCK, false), &args);
                assert_eq!(checked.is_ok(), input != output, "{input:?} to {output:?}");
            }
        }

        // A packed narrow float is no element of a conversion.
        let args = args(DType::F8E4M3, DType::F32, &[4, 8]);
        let message = refusal(Cast::check(&Descriptor::of(&args, BLOCK, false), &args));
        assert!(message.contains("input is f8e4m3"), "{message}");
    }

    #[test]
    fn a_permuted_input_traverses_its_own_axes_and_the_output_stays_contiguous() {
        let permuted = Args {
            x: Operand::with_strides(
                DevicePtr::new(0x1000),
                DType::I64,
                Shape::new(&[5, 3, 2]),
                Strides::new(&[1, 5, 15]),
            ),
            out: contiguous(DType::F32, &[5, 3, 2]),
        };
        let descriptor = Descriptor::of(&permuted, BLOCK, false);
        assert_eq!(descriptor.rank, 3);
        Cast::check(&descriptor, &permuted).unwrap();

        let strided_out = Args {
            out: Operand::with_strides(
                DevicePtr::new(0x2000),
                DType::F32,
                Shape::new(&[5, 3, 2]),
                Strides::new(&[12, 4, 2]),
            ),
            ..permuted
        };
        let descriptor = Descriptor::of(&strided_out, BLOCK, false);
        let message = refusal(Cast::check(&descriptor, &strided_out));
        assert!(message.contains("out must be contiguous"), "{message}");

        let other_shape = Args {
            out: contiguous(DType::F32, &[5, 6]),
            ..permuted
        };
        let descriptor = Descriptor::of(&other_shape, BLOCK, false);
        let message = refusal(Cast::check(&descriptor, &other_shape));
        assert!(message.contains("expected out of shape"), "{message}");
    }

    #[test]
    fn a_conversion_is_exact_unless_it_rounds_into_a_float() {
        let tolerance = |input, output| {
            Cast::tolerance(&Descriptor::of(&args(input, output, &[8]), BLOCK, false))
        };

        // Between integers the low bits are kept, and a float truncates into an integer.
        assert_eq!(tolerance(DType::I64, DType::U8), Tolerance::exact());
        assert_eq!(tolerance(DType::U32, DType::I16), Tolerance::exact());
        assert_eq!(tolerance(DType::F32, DType::I32), Tolerance::exact());
        // A wider float holds a narrower one, and a float holds an integer that fits it.
        assert_eq!(tolerance(DType::BF16, DType::F32), Tolerance::exact());
        assert_eq!(tolerance(DType::F32, DType::F64), Tolerance::exact());
        assert_eq!(tolerance(DType::U8, DType::BF16), Tolerance::exact());
        assert_eq!(tolerance(DType::I32, DType::F64), Tolerance::exact());

        // A narrowing rounds once into the output, after one rounding on the way.
        assert_eq!(
            tolerance(DType::F32, DType::BF16),
            Tolerance::rounding(1, DType::F32, DType::BF16)
        );
        assert_eq!(
            tolerance(DType::I64, DType::F64),
            Tolerance::rounding(1, DType::F64, DType::F64)
        );
        // Neither half-precision type holds the other.
        assert_ne!(tolerance(DType::BF16, DType::F16), Tolerance::exact());
        assert_ne!(tolerance(DType::F16, DType::BF16), Tolerance::exact());
        // An integer wider than a significand rounds.
        assert_ne!(tolerance(DType::I32, DType::F32), Tolerance::exact());
        assert_ne!(tolerance(DType::I16, DType::F16), Tolerance::exact());
    }
}
