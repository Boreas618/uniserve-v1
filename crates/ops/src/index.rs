//! `index`: elements read or written at positions an index tensor names along one axis.
//!
//! ```text
//! gather        destination[.., j, ..]              = source[.., indices[.., j, ..], ..]
//! scatter_set   destination[.., indices[.., j, ..], ..]  = source[.., j, ..]
//! scatter_add   destination[.., indices[.., j, ..], ..] += source[.., j, ..]
//! ```
//!
//! The dots are the same positions on both sides: only the position along `axis` goes through the
//! indices. The source and the indices are read through their strides; the destination is
//! contiguous, the output of a `gather` and the tensor a scatter updates in place. Indices are
//! `u8`, `u32` or `i64`.
//!
//! A `gather` and a `scatter_set` convert nothing, so their elements are any a primitive moves
//! ([`crate::strided::MOVED`]): an FP8 (`f8e4m3`) table is looked up as the bytes its elements
//! are. A `scatter_add` computes, on the element types the primitives compute with.
//!
//! # The five backend primitives
//!
//! The indices have the shape of the side that is traversed — the destination of a `gather`, the
//! source of a scatter — and an index vector that applies to every position off the axis is that
//! shape with a zero stride off the axis, as every broadcast operand is stated
//! ([`crate::strided`]). So three functions serve five primitives:
//!
//! | Primitive | Function | Indices |
//! | --- | --- | --- |
//! | `gather` | `gather` | one per destination element |
//! | `index_select` | `gather` | one per position along the axis, repeated off it |
//! | `scatter_set` | `scatter_set` | one per source element |
//! | `scatter_add_set` | `scatter_add` | one per source element |
//! | `index_add` | `scatter_add` | one per position along the axis, repeated off it |
//!
//! # What an index means
//!
//! An index equal to its type's largest value names no element: a `gather` stores zero for it and
//! a scatter skips it, which is the tensor library's host rule. Any other index is a position
//! along the axis and **is not checked**: a device value cannot be read on the host, so an index at
//! or past the extent reads or writes outside the operand. The caller holds its indices to the
//! extent.
//!
//! # Order
//!
//! A scatter traverses the axis in order within each column — the positions off the axis — and no
//! two columns share a destination element. Repeated indices are therefore well defined: the last
//! source along the axis is the one a `scatter_set` leaves, and a `scatter_add` adds its sources in
//! the order they lie along the axis, each addition evaluated in FP32 (FP64 for an FP64 element,
//! the element's own wrapping arithmetic for an integer) and rounded to nearest even into the
//! element type. That order and that arithmetic are the operator's, so two implementations agree
//! bit for bit. The cost of the order is that a column is sequential: a scatter's work is spread
//! over its columns, not along its axis.

use std::fmt;

use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::strided;
use crate::tensor::{MAX_RANK, Operand, Strides};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The function one compilation of `index` performs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Function {
    /// The destination reads the source at the indexed positions: `gather` and `index_select`.
    Gather,
    /// The source is stored at the indexed positions of the destination.
    ScatterSet,
    /// The source is added at the indexed positions of the destination: `scatter_add` and
    /// `index_add`.
    ScatterAdd,
}

impl Function {
    /// Every function, in the order the package declares them.
    pub const ALL: [Self; 3] = [Self::Gather, Self::ScatterSet, Self::ScatterAdd];
}

impl fmt::Display for Function {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Gather => "gather",
            Self::ScatterSet => "scatter_set",
            Self::ScatterAdd => "scatter_add",
        })
    }
}

/// One compilation of `index`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// The function the module performs.
    pub function: Function,
    /// Element type of the source and the destination.
    pub dtype: DType,
    /// Element type of the indices: `u8`, `u32` or `i64`.
    pub indices: DType,
    /// The axes the call traverses ([`crate::strided`]): every axis of a `gather`'s destination,
    /// and the axes off the indexed one for a scatter, whose programs traverse columns.
    pub rank: u8,
    /// Destination elements one thread block gathers, or columns one scatters.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The descriptor of the call `args` makes of `function`.
    pub fn of(function: Function, args: &Args, block: u32, pdl: bool) -> Self {
        Self {
            function,
            dtype: args.source.dtype,
            indices: args.indices.dtype,
            rank: rank(function, args),
            block,
            pdl,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("function", self.function)
            .field("dtype", abi::name(self.dtype))
            .field("indices", abi::name(self.indices))
            .field("rank", self.rank)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `index`.
#[derive(Clone, Copy)]
pub struct Args {
    /// The elements read, at any strides. A `gather` reads it at the indexed positions and it has
    /// the destination's shape off the axis; a scatter reads all of it.
    pub source: Operand,
    /// The positions along `axis`, at any strides: the shape of the destination for a `gather`
    /// and of the source for a scatter, with a zero stride wherever one index serves a whole axis.
    pub indices: Operand,
    /// The elements written, contiguous: all of them by a `gather`, the indexed ones by a
    /// scatter, which has the source's shape off the axis.
    pub destination: Operand,
    /// The axis the indices are positions along.
    pub axis: usize,
}

/// Elements read or written at indexed positions along one axis.
pub struct Index;

/// The operands of a `gather` as one index space, merged: the destination, the source row that
/// index zero selects repeated along the axis, and the indices.
///
/// An element's source offset is the repeated row's offset plus its index times the source's step
/// along the axis, so the three traverse the destination's shape together and the axis needs no
/// special place among the others.
fn gathered(args: &Args) -> [Operand; 3] {
    let rank = args.source.shape.rank();
    let mut steps = [0_u64; MAX_RANK];
    steps[..rank].copy_from_slice(args.source.strides.as_slice());
    if args.axis < rank {
        steps[args.axis] = 0;
    }

    let repeated = Operand::with_strides(
        args.source.ptr,
        args.source.dtype,
        args.destination.shape,
        Strides::new(&steps[..rank]),
    );
    strided::merge_axes([args.destination, repeated, args.indices])
}

/// The columns of a scatter, merged: the source, the indices and the destination along every
/// axis but the indexed one, at index zero of it.
fn columns(args: &Args) -> [Operand; 3] {
    let off_axis = |axis: usize| axis != args.axis;
    strided::merge_axes([
        strided::select_axes(&args.source, off_axis),
        strided::select_axes(&args.indices, off_axis),
        strided::select_axes(&args.destination, off_axis),
    ])
}

/// The axes a program of `function` traverses for `args`.
fn rank(function: Function, args: &Args) -> u8 {
    let [first, ..] = match function {
        Function::Gather => gathered(args),
        Function::ScatterSet | Function::ScatterAdd => columns(args),
    };
    first.shape.rank() as u8
}

/// Check that `narrow` and `wide` have one rank and one extent on every axis but `axis`.
fn require_alike_off_axis(
    narrow: (&str, &Operand),
    wide: (&str, &Operand),
    axis: usize,
) -> Result<()> {
    let (narrow_name, narrow) = narrow;
    let (wide_name, wide) = wide;
    let alike = narrow.shape.rank() == wide.shape.rank()
        && narrow.strides.rank() == narrow.shape.rank()
        && (0..narrow.shape.rank())
            .all(|other| other == axis || narrow.shape.dim(other) == wide.shape.dim(other));
    if alike {
        return Ok(());
    }
    Err(Error::argument(
        Index::NAME,
        format!(
            "{narrow_name} of shape {:?} and {wide_name} of shape {:?} differ off axis {axis}",
            narrow.shape, wide.shape
        ),
    ))
}

impl Op for Index {
    const NAME: &'static str = proto::module::operator::INDEX;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        // An addition computes; the other two functions move elements.
        let elements: &[DType] = match descriptor.function {
            Function::Gather | Function::ScatterSet => &strided::MOVED,
            Function::ScatterAdd => &strided::ELEMENTS,
        };
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, elements)?;
        strided::require_element(Self::NAME, "indices", descriptor.indices, &strided::INDICES)?;
        strided::require_block(Self::NAME, descriptor.block)?;

        strided::require_dtype(Self::NAME, "source", &args.source, descriptor.dtype)?;
        strided::require_dtype(
            Self::NAME,
            "destination",
            &args.destination,
            descriptor.dtype,
        )?;
        strided::require_dtype(Self::NAME, "indices", &args.indices, descriptor.indices)?;

        if args.axis >= args.destination.shape.rank() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "axis {} is none of the destination's {} axes",
                    args.axis,
                    args.destination.shape.rank()
                ),
            ));
        }
        require_alike_off_axis(
            ("source", &args.source),
            ("destination", &args.destination),
            args.axis,
        )?;
        // The indices have the shape of the side a program traverses.
        let side = match descriptor.function {
            Function::Gather => &args.destination,
            Function::ScatterSet | Function::ScatterAdd => &args.source,
        };
        strided::require_shape(Self::NAME, "indices", &args.indices, side.shape)?;

        strided::require_elements(Self::NAME, "source", &args.source)?;
        strided::require_elements(Self::NAME, "destination", &args.destination)?;
        strided::require_words(Self::NAME, "indices", &args.indices)?;
        strided::require_contiguous(Self::NAME, "destination", &args.destination)?;

        strided::require_rank(
            Self::NAME,
            "rank",
            descriptor.rank,
            rank(descriptor.function, args),
        )
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new();
        arguments
            .view(args.source)
            .view(args.indices)
            .view(args.destination);

        let work = match descriptor.function {
            Function::Gather => {
                let [destination, repeated, indices] = gathered(&args);
                let elements = destination.numel();

                arguments.u64(elements);
                strided::pack_extents(&mut arguments, destination.shape);
                strided::pack_strides(&mut arguments, repeated.strides);
                strided::pack_strides(&mut arguments, indices.strides);
                arguments.u64(args.source.strides.get(args.axis));
                elements
            }
            Function::ScatterSet | Function::ScatterAdd => {
                let [source, indices, destination] = columns(&args);
                let count = source.numel();

                arguments.u64(count).u64(args.source.shape.dim(args.axis));
                strided::pack_extents(&mut arguments, source.shape);
                strided::pack_strides(&mut arguments, source.strides);
                strided::pack_strides(&mut arguments, indices.strides);
                strided::pack_strides(&mut arguments, destination.strides);
                arguments
                    .u64(args.source.strides.get(args.axis))
                    .u64(args.indices.strides.get(args.axis))
                    .u64(args.destination.strides.get(args.axis));
                count
            }
        };
        arguments.flag(descriptor.pdl);

        // One program gathers a block of destination elements, or scatters a block of columns.
        kernel.launch(
            LaunchGrid::covering(work, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // A scatter updates the destination in place, so its shadow buffer starts as the
        // destination's bytes. A gather writes every element, and its shadow buffer may start as
        // anything: the same bytes serve, which keeps the redirection free of the function the
        // arguments do not name.
        Args {
            destination: twins.in_place(args.destination, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A gather and a scatter_set move elements. A scatter_add's order and arithmetic are the
        // operator's own, so its sums are one sequence of roundings, not a bound on several.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Shape};

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

    fn refusal(function: Function, args: &Args) -> String {
        Index::check(&Descriptor::of(function, args, BLOCK, false), args)
            .unwrap_err()
            .to_string()
    }

    /// `table[ids]` along the first axis: an embedding lookup, as `index_select` states it.
    fn lookup() -> Args {
        Args {
            source: contiguous(DType::BF16, &[1000, 64]),
            // One index per row of the result, repeated along the row.
            indices: operand(DType::U32, &[7, 64], &[1, 0]),
            destination: contiguous(DType::BF16, &[7, 64]),
            axis: 0,
        }
    }

    #[test]
    fn the_key_names_the_function_both_types_and_the_axes_traversed() {
        let args = lookup();
        let descriptor = Descriptor::of(Function::Gather, &args, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "function_gather_dtype_bf16_indices_u32_rank_2_block_1024_pdl_false"
        );
        Index::check(&descriptor, &args).unwrap();
        assert_eq!(Index::tolerance(&descriptor), Tolerance::exact());
    }

    #[test]
    fn a_gather_traverses_the_destination_and_an_index_per_element_is_the_same_function() {
        // `gather` along the last axis: one index per destination element.
        let args = Args {
            source: contiguous(DType::F32, &[4, 151_936]),
            indices: contiguous(DType::I64, &[4, 20]),
            destination: contiguous(DType::F32, &[4, 20]),
            axis: 1,
        };
        let descriptor = Descriptor::of(Function::Gather, &args, BLOCK, false);
        assert_eq!(descriptor.function, Function::Gather);
        assert_eq!(descriptor.rank, 2);
        Index::check(&descriptor, &args).unwrap();

        // Along a middle axis of contiguous operands, the axes on either side stay apart.
        let args = Args {
            source: contiguous(DType::F32, &[2, 9, 5]),
            indices: operand(DType::U8, &[2, 3, 5], &[0, 1, 0]),
            destination: contiguous(DType::F32, &[2, 3, 5]),
            axis: 1,
        };
        assert_eq!(
            Descriptor::of(Function::Gather, &args, BLOCK, false).rank,
            3
        );
    }

    #[test]
    fn a_scatter_traverses_its_columns_and_an_index_per_position_is_the_same_function() {
        // `index_add` along the first axis of `[rows, columns]`: the columns are one run.
        let rows = Args {
            source: contiguous(DType::BF16, &[7, 64]),
            indices: operand(DType::U32, &[7, 64], &[1, 0]),
            destination: contiguous(DType::BF16, &[1000, 64]),
            axis: 0,
        };
        let descriptor = Descriptor::of(Function::ScatterAdd, &rows, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "function_scatter_add_dtype_bf16_indices_u32_rank_1_block_1024_pdl_false"
        );
        Index::check(&descriptor, &rows).unwrap();

        // `scatter_set` along the last axis with an index per element: the rows are the columns.
        let elements = Args {
            source: contiguous(DType::F32, &[4, 20]),
            indices: contiguous(DType::I64, &[4, 20]),
            destination: contiguous(DType::F32, &[4, 151_936]),
            axis: 1,
        };
        let descriptor = Descriptor::of(Function::ScatterSet, &elements, BLOCK, false);
        assert_eq!(descriptor.rank, 1);
        Index::check(&descriptor, &elements).unwrap();

        // Along a middle axis, the axes on either side are two runs of columns.
        let middle = Args {
            source: contiguous(DType::F32, &[2, 3, 5]),
            indices: contiguous(DType::U8, &[2, 3, 5]),
            destination: contiguous(DType::F32, &[2, 9, 5]),
            axis: 1,
        };
        assert_eq!(
            Descriptor::of(Function::ScatterAdd, &middle, BLOCK, false).rank,
            2
        );

        // A vector scattered into a vector has one column.
        let vector = Args {
            source: contiguous(DType::I64, &[12]),
            indices: contiguous(DType::U32, &[12]),
            destination: contiguous(DType::I64, &[5]),
            axis: 0,
        };
        let descriptor = Descriptor::of(Function::ScatterAdd, &vector, BLOCK, false);
        assert_eq!(descriptor.rank, 1);
        Index::check(&descriptor, &vector).unwrap();
    }

    #[test]
    fn an_fp8_table_is_gathered_and_set_as_its_bytes_and_never_added() {
        // The rows of an FP8 table, as an embedding lookup selects them.
        let lookup = Args {
            source: contiguous(DType::F8E4M3, &[1000, 64]),
            indices: operand(DType::U32, &[7, 64], &[1, 0]),
            destination: contiguous(DType::F8E4M3, &[7, 64]),
            axis: 0,
        };
        let descriptor = Descriptor::of(Function::Gather, &lookup, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "function_gather_dtype_f8e4m3_indices_u32_rank_2_block_1024_pdl_false"
        );
        Index::check(&descriptor, &lookup).unwrap();

        // The same rows written back, and added, into the table.
        let scatter = Args {
            source: contiguous(DType::F8E4M3, &[7, 64]),
            indices: operand(DType::U32, &[7, 64], &[1, 0]),
            destination: contiguous(DType::F8E4M3, &[1000, 64]),
            axis: 0,
        };
        let descriptor = Descriptor::of(Function::ScatterSet, &scatter, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "function_scatter_set_dtype_f8e4m3_indices_u32_rank_1_block_1024_pdl_false"
        );
        Index::check(&descriptor, &scatter).unwrap();

        let message = refusal(Function::ScatterAdd, &scatter);
        assert!(message.contains("dtype is f8e4m3"), "{message}");

        // Two NVFP4 elements share a byte: no index names one.
        let packed = Args {
            source: contiguous(DType::F4, &[1000, 64]),
            destination: contiguous(DType::F4, &[7, 64]),
            ..lookup
        };
        let message = refusal(Function::Gather, &packed);
        assert!(message.contains("dtype is f4e2m1"), "{message}");
    }

    #[test]
    fn the_indices_are_u8_u32_or_i64_in_the_shape_of_the_side_traversed() {
        let args = lookup();

        for indices in [DType::U8, DType::U32, DType::I64] {
            let typed = Args {
                indices: operand(indices, &[7, 64], &[1, 0]),
                ..args
            };
            Index::check(
                &Descriptor::of(Function::Gather, &typed, BLOCK, false),
                &typed,
            )
            .unwrap();
        }

        let signed = Args {
            indices: operand(DType::I32, &[7, 64], &[1, 0]),
            ..args
        };
        let message = refusal(Function::Gather, &signed);
        assert!(message.contains("indices is i32"), "{message}");

        // A gather's indices have the destination's shape, not the source's.
        let of_source = Args {
            indices: operand(DType::U32, &[1000, 64], &[1, 0]),
            ..args
        };
        let message = refusal(Function::Gather, &of_source);
        assert!(message.contains("expected indices of shape"), "{message}");

        // A scatter's have the source's.
        let scatter = Args {
            source: contiguous(DType::BF16, &[7, 64]),
            indices: operand(DType::U32, &[1000, 64], &[1, 0]),
            destination: contiguous(DType::BF16, &[1000, 64]),
            axis: 0,
        };
        let message = refusal(Function::ScatterSet, &scatter);
        assert!(message.contains("expected indices of shape"), "{message}");
    }

    #[test]
    fn the_two_sides_agree_off_the_axis_and_the_destination_is_contiguous() {
        let args = lookup();

        let wider = Args {
            source: contiguous(DType::BF16, &[1000, 65]),
            ..args
        };
        let message = refusal(Function::Gather, &wider);
        assert!(message.contains("differ off axis 0"), "{message}");

        let past = Args { axis: 2, ..args };
        let message = refusal(Function::Gather, &past);
        assert!(message.contains("axis 2 is none"), "{message}");

        let strided_destination = Args {
            destination: operand(DType::BF16, &[7, 64], &[128, 1]),
            ..args
        };
        let message = refusal(Function::Gather, &strided_destination);
        assert!(
            message.contains("destination must be contiguous"),
            "{message}"
        );

        let other_type = Args {
            destination: contiguous(DType::F32, &[7, 64]),
            ..args
        };
        let message = refusal(Function::Gather, &other_type);
        assert!(message.contains("expected bf16 destination"), "{message}");

        let empty = Args {
            indices: operand(DType::U32, &[0, 64], &[1, 0]),
            destination: contiguous(DType::BF16, &[0, 64]),
            ..args
        };
        let message = refusal(Function::Gather, &empty);
        assert!(message.contains("nothing to launch"), "{message}");

        // A strided source is read as it lies.
        let permuted = Args {
            source: operand(DType::BF16, &[1000, 64], &[1, 1000]),
            ..args
        };
        Index::check(
            &Descriptor::of(Function::Gather, &permuted, BLOCK, false),
            &permuted,
        )
        .unwrap();
    }
}
