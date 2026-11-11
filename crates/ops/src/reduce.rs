//! `reduce`: a strided input folded along some of its axes.
//!
//! ```text
//! sum      out[k] = Σ x[k, r]                over every index r of the reduced axes
//! min      out[k] = min x[k, r]              max likewise
//! argmin   out[k] = the first r at which x[k, r] is least, as u32; argmax likewise
//! ```
//!
//! `k` ranges over the axes the call keeps and `r` over the ones it reduces, both in row-major
//! order. The input is read through its strides; the output is contiguous and has the input's
//! shape with every reduced extent one. `sum`, `min` and `max` reduce one or more axes; `argmin`
//! and `argmax` reduce exactly one, and return its index.
//!
//! # What a compilation holds
//!
//! The kept axes and the reduced axes are two index spaces, each merged on its own
//! ([`crate::strided`]): a descriptor states how many axes are left in each. A reduction along the
//! last axis of a contiguous tensor keeps one axis and reduces one, whatever the tensor's rank.
//! Extents and strides are launch words.
//!
//! A program reads `tile` elements of the reduction for each of `block / tile` outputs at a time,
//! so that a long reduction of few outputs and a short one of many both fill a block. The tile is
//! the reduction's class — the least power of two that holds its length — capped at the block.
//! The descriptor carries the class, because the tolerance of a sum follows from it, and the key
//! names the tile, which is all of the class a compilation reads.
//!
//! # Arithmetic
//!
//! A floating sum accumulates in FP32 (FP64 for an FP64 element) and rounds once into the element
//! type; an integer sum wraps. A minimum or a maximum is one of the input's own values. The first
//! of equal values wins an `argmin` or an `argmax`. A NaN's place in an order is unspecified: a
//! comparison with it is false whichever way a program makes it.
//!
//! A sum over an axis of no element is zero, and its output is launched like any other. A
//! minimum, a maximum or an index over no element does not exist and is refused.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::strided;
use crate::tensor::{DType, MAX_RANK, Operand, Shape};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The function one compilation of `reduce` folds with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Function {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl Function {
    /// Every function, in the order the package declares them.
    pub const ALL: [Self; 5] = [Self::Sum, Self::Min, Self::Max, Self::ArgMin, Self::ArgMax];

    /// Whether the function returns an index along its one axis rather than a value.
    pub const fn indexes(self) -> bool {
        matches!(self, Self::ArgMin | Self::ArgMax)
    }

    /// The element type of the result over an input of `dtype`.
    pub const fn output(self, dtype: DType) -> DType {
        if self.indexes() { DType::U32 } else { dtype }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::ArgMin => "argmin",
            Self::ArgMax => "argmax",
        })
    }
}

/// The class of a reduction over `length` elements: the least power of two that holds it.
///
/// A class is what a compilation's tile and a sum's tolerance are stated for, so that the lengths
/// a model reduces over — a sequence that grows by a token — share programs and bounds.
pub fn reduction_class(length: u64) -> u32 {
    let class = length.max(1).next_power_of_two();
    class.min(1 << (u32::BITS - 1)) as u32
}

/// One compilation of `reduce`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// The function the module folds with.
    pub function: Function,
    /// Element type of the input, and of the output unless the function returns an index.
    pub dtype: DType,
    /// The kept axes the call traverses ([`crate::strided`]).
    pub kept: u8,
    /// The reduced axes the call traverses.
    pub reduced: u8,
    /// The reduction's class ([`reduction_class`]): a power of two at or above the elements one
    /// output folds, which the launch checks and a sum's tolerance is stated for.
    pub reduction: u32,
    /// Elements one thread block reads at a time: `tile` of the reduction for `block / tile`
    /// outputs.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The descriptor of the call `args` makes of `function`.
    pub fn of(function: Function, args: &Args<'_>, block: u32, pdl: bool) -> Self {
        let (kept, reduced) = split(&args.x, args.axes);
        Self {
            function,
            dtype: args.x.dtype,
            kept: kept.shape.rank() as u8,
            reduced: reduced.shape.rank() as u8,
            reduction: reduction_class(reduced.numel()),
            block,
            pdl,
        }
    }

    /// The elements of the reduction one program reads for each output at a time: the class,
    /// capped at the block.
    pub fn tile(&self) -> u32 {
        self.reduction.min(self.block)
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The class beyond the block is absent on purpose: it travels in no word and changes no
        // program, so naming it would file one module under several keys.
        Key::new()
            .field("function", self.function)
            .field("dtype", self.dtype)
            .field("kept", self.kept)
            .field("reduced", self.reduced)
            .field("tile", self.tile())
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `reduce`.
#[derive(Clone, Copy)]
pub struct Args<'a> {
    /// The values, at any strides.
    pub x: Operand,
    /// The result: the shape of `x` with every reduced extent one, contiguous; `u32` for an
    /// index.
    pub out: Operand,
    /// The axes of `x` to reduce, each named once.
    pub axes: &'a [usize],
}

/// A strided input folded along some of its axes.
pub struct Reduce;

/// `x` along the axes a reduction keeps and along the ones it reduces, each merged into the axes
/// a program traverses.
///
/// The first is the view of `x` at index zero of every reduced axis, whose elements the outputs
/// follow in row-major order; the second is the view at index zero of every kept axis, whose
/// elements one output folds.
fn split(x: &Operand, axes: &[usize]) -> (Operand, Operand) {
    let [kept] = strided::merge_axes([strided::select_axes(x, |axis| !axes.contains(&axis))]);
    let [reduced] = strided::merge_axes([strided::select_axes(x, |axis| axes.contains(&axis))]);
    (kept, reduced)
}

impl Op for Reduce {
    const NAME: &'static str = proto::program::operator::REDUCE;

    type Descriptor = Descriptor;
    type Args<'a> = Args<'a>;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, &strided::ELEMENTS)?;
        strided::require_block(Self::NAME, descriptor.block)?;

        let rank = args.x.shape.rank();
        if args.axes.is_empty() {
            return Err(Error::argument(
                Self::NAME,
                "a reduction along no axis folds nothing".to_owned(),
            ));
        }
        for (position, &axis) in args.axes.iter().enumerate() {
            if axis >= rank || args.axes[..position].contains(&axis) {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the axes {:?} do not each name one axis of a rank-{rank} input",
                        args.axes
                    ),
                ));
            }
        }
        if descriptor.function.indexes() && args.axes.len() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} returns an index along one axis, and the call names {:?}",
                    descriptor.function, args.axes
                ),
            ));
        }

        let mut folded = [0_u64; MAX_RANK];
        for (axis, extent) in folded[..rank].iter_mut().enumerate() {
            *extent = if args.axes.contains(&axis) {
                1
            } else {
                args.x.shape.dim(axis)
            };
        }
        let output = descriptor.function.output(descriptor.dtype);
        strided::require_dtype(Self::NAME, "x", &args.x, descriptor.dtype)?;
        strided::require_dtype(Self::NAME, "out", &args.out, output)?;
        strided::require_shape(Self::NAME, "out", &args.out, Shape::new(&folded[..rank]))?;
        strided::require_elements(Self::NAME, "out", &args.out)?;
        strided::require_contiguous(Self::NAME, "out", &args.out)?;
        strided::require_words(Self::NAME, "x", &args.x)?;

        let (kept, reduced) = split(&args.x, args.axes);
        let length = reduced.numel();
        if length == 0 && descriptor.function != Function::Sum {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} over the axes {:?} of {:?} has no element to return",
                    descriptor.function, args.axes, args.x.shape
                ),
            ));
        }
        if descriptor.reduction != reduction_class(length) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "one output folds {length} elements, whose class is {} and not the \
                     descriptor's {}",
                    reduction_class(length),
                    descriptor.reduction
                ),
            ));
        }
        if u64::from(descriptor.reduction) < length {
            return Err(Error::argument(
                Self::NAME,
                format!("a reduction of {length} elements is longer than any class"),
            ));
        }

        strided::require_rank(Self::NAME, "kept", descriptor.kept, kept.shape.rank() as u8)?;
        strided::require_rank(
            Self::NAME,
            "reduced",
            descriptor.reduced,
            reduced.shape.rank() as u8,
        )
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let (kept, reduced) = split(&args.x, args.axes);
        let outputs = kept.numel();

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u64(outputs)
            .u64(reduced.numel());
        strided::pack_extents(&mut arguments, kept.shape);
        strided::pack_strides(&mut arguments, kept.strides);
        strided::pack_extents(&mut arguments, reduced.shape);
        strided::pack_strides(&mut arguments, reduced.strides);
        arguments.flag(descriptor.pdl);

        // One program folds `block / tile` outputs.
        let rows = u64::from(descriptor.block / descriptor.tile());
        kernel.launch(LaunchGrid::covering(outputs, rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let dtype = descriptor.dtype;
        if descriptor.function == Function::Sum && dtype.is_float() {
            // Terms of either sign accumulated in the computing type, in whatever order a program
            // takes them, and rounded once into the element type: the bound of the class.
            return Tolerance::dot(u64::from(descriptor.reduction), dtype.accumulate(), dtype);
        }
        // An integer sum wraps the same way in any order; a minimum or a maximum is one of the
        // input's values, and an index that differs names another element.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Strides};

    const BLOCK: u32 = 1024;

    fn contiguous(dtype: DType, shape: &[u64]) -> Operand {
        Operand::new(DevicePtr::new(0x1000), dtype, Shape::new(shape))
    }

    /// A reduction of `x` along `axes`, with the output its function and shape call for.
    fn reduction<'a>(function: Function, x: Operand, axes: &'a [usize]) -> Args<'a> {
        let folded: Vec<u64> = (0..x.shape.rank())
            .map(|axis| {
                if axes.contains(&axis) {
                    1
                } else {
                    x.shape.dim(axis)
                }
            })
            .collect();
        Args {
            x,
            out: Operand::new(
                DevicePtr::new(0x2000),
                function.output(x.dtype),
                Shape::new(&folded),
            ),
            axes,
        }
    }

    fn refusal(descriptor: &Descriptor, args: &Args<'_>) -> String {
        Reduce::check(descriptor, args).unwrap_err().to_string()
    }

    #[test]
    fn the_key_names_the_function_the_type_the_axes_traversed_and_the_tile() {
        let args = reduction(Function::Sum, contiguous(DType::BF16, &[4, 16, 100]), &[2]);
        let descriptor = Descriptor::of(Function::Sum, &args, BLOCK, false);
        assert_eq!(descriptor.reduction, 128);
        assert_eq!(
            descriptor.to_string(),
            "function_sum_dtype_bf16_kept_1_reduced_1_tile_128_block_1024_pdl_false"
        );
        Reduce::check(&descriptor, &args).unwrap();
    }

    #[test]
    fn lengths_of_one_class_share_a_key_and_a_class_past_the_block_shares_its_tile() {
        let key = |length: u64| {
            let x = contiguous(DType::F32, &[8, length]);
            let args = reduction(Function::Max, x, &[1]);
            Descriptor::of(Function::Max, &args, BLOCK, false).to_string()
        };

        // 65 to 128 elements are one class.
        assert_eq!(key(65), key(128));
        assert_ne!(key(64), key(65));
        // Past the block every class reads a block at a time: a vocabulary and a longer one.
        assert_eq!(key(151_936), key(2048));
        assert!(key(151_936).contains("tile_1024"), "{}", key(151_936));
    }

    #[test]
    fn the_kept_and_the_reduced_axes_merge_each_on_their_own() {
        let x = contiguous(DType::F32, &[2, 3, 4, 5]);
        let ranks = |axes: &[usize]| {
            let args = reduction(Function::Sum, x, axes);
            let descriptor = Descriptor::of(Function::Sum, &args, BLOCK, false);
            Reduce::check(&descriptor, &args).unwrap();
            (descriptor.kept, descriptor.reduced)
        };

        // The trailing axes are one run, and so are the leading ones they leave.
        assert_eq!(ranks(&[3]), (1, 1));
        assert_eq!(ranks(&[2, 3]), (1, 1));
        // The leading axes reduce as one run of a larger stride.
        assert_eq!(ranks(&[0, 1]), (1, 1));
        // A middle axis leaves two kept runs that do not touch.
        assert_eq!(ranks(&[1]), (2, 1));
        // The outer and the inner axis reduce as two runs.
        assert_eq!(ranks(&[0, 3]), (1, 2));
        // Every axis reduced leaves one output.
        assert_eq!(ranks(&[0, 1, 2, 3]), (1, 1));
        // The order the axes are named in does not matter.
        assert_eq!(ranks(&[3, 0]), (1, 2));
    }

    #[test]
    fn an_index_is_u32_along_exactly_one_axis() {
        let x = contiguous(DType::F32, &[4, 151_936]);
        let args = reduction(Function::ArgMax, x, &[1]);
        assert_eq!(args.out.dtype, DType::U32);
        let descriptor = Descriptor::of(Function::ArgMax, &args, BLOCK, false);
        Reduce::check(&descriptor, &args).unwrap();
        assert_eq!(Reduce::tolerance(&descriptor), Tolerance::exact());

        // The same output under a value function is the wrong type.
        let maximum = Descriptor {
            function: Function::Max,
            ..descriptor
        };
        let message = refusal(&maximum, &args);
        assert!(message.contains("expected f32 out"), "{message}");

        let both = reduction(Function::ArgMax, x, &[0, 1]);
        let descriptor = Descriptor::of(Function::ArgMax, &both, BLOCK, false);
        let message = refusal(&descriptor, &both);
        assert!(message.contains("an index along one axis"), "{message}");
    }

    #[test]
    fn the_axes_each_name_one_axis_of_the_input() {
        let x = contiguous(DType::F32, &[4, 8]);

        for axes in [&[][..], &[2], &[1, 1]] {
            let args = reduction(Function::Sum, x, axes);
            let descriptor = Descriptor::of(Function::Sum, &args, BLOCK, false);
            assert!(Reduce::check(&descriptor, &args).is_err(), "{axes:?}");
        }
    }

    #[test]
    fn the_output_has_the_inputs_shape_with_the_reduced_extents_one() {
        let x = contiguous(DType::F32, &[4, 8]);
        let args = reduction(Function::Sum, x, &[1]);
        let descriptor = Descriptor::of(Function::Sum, &args, BLOCK, false);

        let squeezed = Args {
            out: contiguous(DType::F32, &[4]),
            ..args
        };
        let message = refusal(&descriptor, &squeezed);
        assert!(message.contains("expected out of shape"), "{message}");

        let strided_out = Args {
            out: Operand::with_strides(
                DevicePtr::new(0x2000),
                DType::F32,
                Shape::new(&[4, 1]),
                Strides::new(&[2, 1]),
            ),
            ..args
        };
        let message = refusal(&descriptor, &strided_out);
        assert!(message.contains("out must be contiguous"), "{message}");

        // A descriptor of another class is not this call's.
        let other_class = Descriptor {
            reduction: 16,
            ..descriptor
        };
        let message = refusal(&other_class, &args);
        assert!(message.contains("whose class is 8"), "{message}");
    }

    #[test]
    fn a_sum_over_no_element_is_launched_and_an_extremum_over_none_is_refused() {
        let x = contiguous(DType::F32, &[4, 0]);

        let sum = reduction(Function::Sum, x, &[1]);
        let descriptor = Descriptor::of(Function::Sum, &sum, BLOCK, false);
        assert_eq!(descriptor.reduction, 1);
        Reduce::check(&descriptor, &sum).unwrap();

        let minimum = reduction(Function::Min, x, &[1]);
        let descriptor = Descriptor::of(Function::Min, &minimum, BLOCK, false);
        let message = refusal(&descriptor, &minimum);
        assert!(message.contains("no element to return"), "{message}");

        // No output at all is nothing to launch, whatever the function.
        let none = reduction(Function::Sum, x, &[0]);
        let descriptor = Descriptor::of(Function::Sum, &none, BLOCK, false);
        let message = refusal(&descriptor, &none);
        assert!(message.contains("nothing to launch"), "{message}");
    }

    #[test]
    fn a_floating_sum_is_bounded_by_its_class_and_everything_else_is_exact() {
        let tolerance = |function, dtype, length| {
            let args = reduction(function, contiguous(dtype, &[4, length]), &[1]);
            Reduce::tolerance(&Descriptor::of(function, &args, BLOCK, false))
        };

        assert_eq!(
            tolerance(Function::Sum, DType::BF16, 100),
            Tolerance::dot(128, DType::F32, DType::BF16)
        );
        assert_eq!(
            tolerance(Function::Sum, DType::F64, 5000),
            Tolerance::dot(8192, DType::F64, DType::F64)
        );
        assert_eq!(
            tolerance(Function::Sum, DType::I32, 100),
            Tolerance::exact()
        );
        assert_eq!(
            tolerance(Function::Min, DType::BF16, 100),
            Tolerance::exact()
        );
        assert_eq!(
            tolerance(Function::ArgMin, DType::F32, 100),
            Tolerance::exact()
        );
    }
}
