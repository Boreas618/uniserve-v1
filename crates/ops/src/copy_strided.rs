//! `copy_strided`: the elements of a strided source moved into a row-major window.
//!
//! ```text
//! destination[i] = source[i]                 both of one shape and one element type
//! ```
//!
//! The source is read through its strides, so it may be any view: permuted, stepped or broadcast.
//! The destination is a row-major window — a contiguous tensor, or a block of a larger contiguous
//! one, read with the larger one's strides — and its address is where the first element lands,
//! which is how a caller states an element offset. That one form is every copy a tensor library
//! makes:
//!
//! - `copy_strided_src` and `contiguous`: a strided source into a contiguous destination, at an
//!   offset;
//! - `copy2d`: `[rows, columns]` whose row strides differ on the two sides;
//! - `cat`, `pad` and `slice_set`: a source into the block of the result it occupies.
//!
//! The shape and both sides' strides are launch words ([`crate::strided`]), so a descriptor names
//! the element type, the axes the call traverses and the block. No element is converted: the
//! destination receives the source's bits. That holds for every element a primitive moves
//! ([`crate::strided::MOVED`]): an FP8 (`f8e4m3`) tensor is copied, cloned, made contiguous and
//! concatenated as the bytes its elements are.
//!
//! The two operands do not overlap. A caller that moves elements within one tensor reads a
//! source its destination has not reached.

use std::fmt;

use uniserve_tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::strided;
use crate::tensor::Operand;
use crate::tolerance::Tolerance;
use crate::{Op, Result, Twins};

/// One compilation of `copy_strided`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of both sides; a copy does not change it.
    pub dtype: DType,
    /// The axes the call traverses ([`crate::strided`]).
    pub rank: u8,
    /// Elements one thread block moves.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The descriptor of the call `args` makes.
    pub fn of(args: &Args, block: u32, pdl: bool) -> Self {
        Self {
            dtype: args.source.dtype,
            rank: strided::rank([args.source, args.destination]),
            block,
            pdl,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("rank", self.rank)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `copy_strided`.
#[derive(Clone, Copy)]
pub struct Args {
    /// The elements to move, at any strides.
    pub source: Operand,
    /// Where they land: the shape of `source`, a row-major window, starting at its address.
    pub destination: Operand,
}

/// A strided source moved into a row-major window.
pub struct CopyStrided;

impl Op for CopyStrided {
    const NAME: &'static str = proto::module::operator::COPY_STRIDED;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, &strided::MOVED)?;
        strided::require_block(Self::NAME, descriptor.block)?;

        strided::require_dtype(Self::NAME, "source", &args.source, descriptor.dtype)?;
        strided::require_dtype(
            Self::NAME,
            "destination",
            &args.destination,
            descriptor.dtype,
        )?;
        strided::require_shape(
            Self::NAME,
            "destination",
            &args.destination,
            args.source.shape,
        )?;
        strided::require_elements(Self::NAME, "source", &args.source)?;
        strided::require_elements(Self::NAME, "destination", &args.destination)?;
        strided::require_window(Self::NAME, "destination", &args.destination)?;

        let traversed = strided::rank([args.source, args.destination]);
        strided::require_rank(Self::NAME, "rank", descriptor.rank, traversed)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let [source, destination] = strided::merge_axes([args.source, args.destination]);
        let elements = destination.numel();

        let mut arguments = Arguments::new();
        arguments.view(source).view(destination).u64(elements);
        strided::pack_extents(&mut arguments, destination.shape);
        strided::pack_strides(&mut arguments, source.strides);
        strided::pack_strides(&mut arguments, destination.strides);
        arguments.flag(descriptor.pdl);

        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The destination is what is redirected: a redirected source would be compared with
        // itself.
        Args {
            destination: strided::shadow_window(args.destination, twins, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // No arithmetic: the same bits arrive on the other side.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Shape, Strides};

    const BLOCK: u32 = 1024;

    fn operand(address: u64, shape: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(
            DevicePtr::new(address),
            DType::BF16,
            Shape::new(shape),
            Strides::new(strides),
        )
    }

    fn contiguous(address: u64, shape: &[u64]) -> Operand {
        Operand::new(DevicePtr::new(address), DType::BF16, Shape::new(shape))
    }

    fn refusal(args: &Args) -> String {
        CopyStrided::check(&Descriptor::of(args, BLOCK, false), args)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn the_key_names_the_type_and_the_axes_traversed() {
        let args = Args {
            source: contiguous(0x1000, &[4, 8]),
            destination: contiguous(0x2000, &[4, 8]),
        };
        let descriptor = Descriptor::of(&args, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "dtype_bf16_rank_1_block_1024_pdl_false"
        );
        CopyStrided::check(&descriptor, &args).unwrap();
        assert_eq!(CopyStrided::tolerance(&descriptor), Tolerance::exact());
    }

    #[test]
    fn a_permuted_source_is_made_contiguous_over_the_axes_it_does_not_share() {
        // `[2, 3, 5]` read as `[5, 2, 3]` into a contiguous destination: `contiguous()`.
        let args = Args {
            source: operand(0x1000, &[5, 2, 3], &[1, 15, 5]),
            destination: contiguous(0x2000, &[5, 2, 3]),
        };
        let descriptor = Descriptor::of(&args, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        CopyStrided::check(&descriptor, &args).unwrap();

        // A broadcast source fills a contiguous destination with its repeated row.
        let args = Args {
            source: operand(0x1000, &[4, 8], &[0, 1]),
            destination: contiguous(0x2000, &[4, 8]),
        };
        let descriptor = Descriptor::of(&args, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        CopyStrided::check(&descriptor, &args).unwrap();
    }

    #[test]
    fn rows_whose_strides_differ_on_the_two_sides_are_the_two_dimensional_copy() {
        // Three rows of eight, sixteen apart in the source, into rows twenty apart: a block of a
        // wider result, as a concatenation along the last axis writes it.
        let args = Args {
            source: operand(0x1000, &[3, 8], &[16, 1]),
            destination: operand(0x2000, &[3, 8], &[20, 1]),
        };
        let descriptor = Descriptor::of(&args, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        CopyStrided::check(&descriptor, &args).unwrap();

        // Both sides contiguous is one run, whatever the two-dimensional call said.
        let args = Args {
            source: operand(0x1000, &[3, 8], &[8, 1]),
            destination: operand(0x2000, &[3, 8], &[8, 1]),
        };
        assert_eq!(Descriptor::of(&args, BLOCK, false).rank, 1);
    }

    #[test]
    fn a_destination_is_a_row_major_window_of_the_sources_shape_and_type() {
        let source = contiguous(0x1000, &[4, 8]);

        // A block inside a larger tensor keeps the larger tensor's strides.
        let window = Args {
            source,
            destination: operand(0x2000, &[4, 8], &[32, 1]),
        };
        CopyStrided::check(&Descriptor::of(&window, BLOCK, false), &window).unwrap();

        // A permuted destination is no window.
        let permuted = Args {
            source,
            destination: operand(0x2000, &[4, 8], &[1, 4]),
        };
        let message = refusal(&permuted);
        assert!(message.contains("no row-major window"), "{message}");

        // Nor is one whose rows overlap, or one written once per repetition.
        let overlapping = Args {
            source,
            destination: operand(0x2000, &[4, 8], &[4, 1]),
        };
        assert!(refusal(&overlapping).contains("no row-major window"));
        let broadcast = Args {
            source,
            destination: operand(0x2000, &[4, 8], &[0, 1]),
        };
        assert!(refusal(&broadcast).contains("no row-major window"));

        let other_shape = Args {
            source,
            destination: contiguous(0x2000, &[8, 4]),
        };
        let message = refusal(&other_shape);
        assert!(
            message.contains("expected destination of shape"),
            "{message}"
        );

        let other_type = Args {
            source,
            destination: Operand::new(DevicePtr::new(0x2000), DType::F16, Shape::new(&[4, 8])),
        };
        let message = refusal(&other_type);
        assert!(message.contains("expected bf16 destination"), "{message}");

        let empty = Args {
            source: contiguous(0x1000, &[0, 8]),
            destination: contiguous(0x2000, &[0, 8]),
        };
        assert!(refusal(&empty).contains("nothing to launch"));
    }

    #[test]
    fn an_fp8_tensor_moves_as_its_bytes_and_a_packed_one_does_not() {
        let of = |dtype: DType, strides: &[u64]| Args {
            source: Operand::with_strides(
                DevicePtr::new(0x1000),
                dtype,
                Shape::new(&[5, 2, 3]),
                Strides::new(strides),
            ),
            destination: Operand::new(DevicePtr::new(0x2000), dtype, Shape::new(&[5, 2, 3])),
        };

        // A permuted FP8 view made contiguous.
        let narrow = of(DType::F8E4M3, &[1, 15, 5]);
        let descriptor = Descriptor::of(&narrow, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "dtype_f8e4m3_rank_2_block_1024_pdl_false"
        );
        CopyStrided::check(&descriptor, &narrow).unwrap();
        assert_eq!(CopyStrided::tolerance(&descriptor), Tolerance::exact());

        // Two NVFP4 elements share a byte: no stride addresses one.
        let packed = of(DType::F4, &[6, 3, 1]);
        let message = refusal(&packed);
        assert!(message.contains("dtype is f4e2m1"), "{message}");
    }
}
