//! `fill`: one element stored at every position of a row-major window.
//!
//! ```text
//! destination[i] = value
//! ```
//!
//! The destination is a row-major window — a contiguous tensor, or a block of a larger contiguous
//! one, read with the larger one's strides — so a caller sets a whole tensor (`zeros`, a constant)
//! or a slice of one (`const_set`) with the same operator. Its shape and strides are launch words
//! ([`crate::strided`]), and a descriptor names the element type, the axes the call traverses and
//! the block.
//!
//! The value travels as its bits: the caller states the element as it is stored, and a program
//! stores those bits without converting anything. A scalar of the host therefore becomes an
//! element once, on the host, by the host's own conversion, and every implementation of the
//! operator writes the same bytes.
//!
//! The destination holds any element a primitive moves ([`crate::strided::MOVED`]): an FP8
//! (`f8e4m3`) element is its one byte, so the bits of a zero are the zero byte.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::strided;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `fill`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the destination.
    pub dtype: DType,
    /// The axes the call traverses ([`crate::strided`]).
    pub rank: u8,
    /// Elements one thread block stores.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The descriptor of the call `args` makes.
    pub fn of(args: &Args, block: u32, pdl: bool) -> Self {
        Self {
            dtype: args.destination.dtype,
            rank: strided::rank([&args.destination]),
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

/// The arguments of one `fill`.
#[derive(Clone)]
pub struct Args {
    /// The elements to set: a row-major window.
    pub destination: Operand,
    /// The element to store, as its bits in the low bytes: `f32::to_bits` for an FP32 element, the
    /// sixteen bits of a half-precision one, an integer's two's complement at the element's width.
    pub bits: u64,
}

/// One element stored over a row-major window.
pub struct Fill;

impl Op for Fill {
    const NAME: &'static str = proto::module::operator::FILL;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        strided::require_element(Self::NAME, "dtype", descriptor.dtype, &strided::MOVED)?;
        strided::require_block(Self::NAME, descriptor.block)?;

        strided::require_dtype(
            Self::NAME,
            "destination",
            &args.destination,
            descriptor.dtype,
        )?;
        strided::require_elements(Self::NAME, "destination", &args.destination)?;
        strided::require_window(Self::NAME, "destination", &args.destination)?;

        // Every element type of a fill is a whole number of bytes.
        let width = u8::BITS * descriptor.dtype.size_in_bytes() as u32;
        if width < u64::BITS && args.bits >> width != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the bits {:#x} do not fit a {} element of {width} bits",
                    args.bits,
                    abi::name(descriptor.dtype)
                ),
            ));
        }

        let traversed = strided::rank([&args.destination]);
        strided::require_rank(Self::NAME, "rank", descriptor.rank, traversed)
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let [destination] = strided::merge_axes([&args.destination]);
        let elements = destination.numel();

        let mut arguments = Arguments::new();
        arguments.view(&destination).u64(elements);
        strided::pack_extents(&mut arguments, &destination);
        strided::pack_strides(&mut arguments, &destination);
        arguments.u64(args.bits).flag(descriptor.pdl);

        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        Args {
            destination: strided::shadow_window(args.destination, twins, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The bits the caller states are the bits stored.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::DevicePtr;

    const BLOCK: u32 = 1024;

    fn window(dtype: DType, shape: &[u64], strides: &[u64], bits: u64) -> Args {
        Args {
            destination: Operand::with_strides(
                DevicePtr::from_address(0x2000),
                dtype,
                shape,
                strides,
            ),
            bits,
        }
    }

    fn refusal(args: &Args) -> String {
        Fill::check(&Descriptor::of(args, BLOCK, false), args)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn the_key_names_the_type_and_the_axes_traversed() {
        let zeros = window(DType::BF16, &[4, 8], &[8, 1], 0);
        let descriptor = Descriptor::of(&zeros, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "dtype_bf16_rank_1_block_1024_pdl_false"
        );
        Fill::check(&descriptor, &zeros).unwrap();
        assert_eq!(Fill::tolerance(&descriptor), Tolerance::exact());
    }

    #[test]
    fn a_block_of_a_larger_tensor_traverses_the_axes_its_strides_keep_apart() {
        // Rows of eight inside rows of thirty-two.
        let block = window(DType::F32, &[4, 8], &[32, 1], 1.0_f32.to_bits().into());
        let descriptor = Descriptor::of(&block, BLOCK, false);
        assert_eq!(descriptor.rank, 2);
        Fill::check(&descriptor, &block).unwrap();

        // Every other element of a run is one axis with a step of two.
        let stepped = window(DType::F32, &[16], &[2], 0);
        let descriptor = Descriptor::of(&stepped, BLOCK, false);
        assert_eq!(descriptor.rank, 1);
        Fill::check(&descriptor, &stepped).unwrap();
    }

    #[test]
    fn a_destination_that_is_no_window_or_has_no_element_is_refused() {
        let permuted = window(DType::F32, &[4, 8], &[1, 4], 0);
        assert!(refusal(&permuted).contains("no row-major window"));

        let broadcast = window(DType::F32, &[4, 8], &[0, 1], 0);
        assert!(refusal(&broadcast).contains("no row-major window"));

        let empty = window(DType::F32, &[4, 0], &[0, 1], 0);
        assert!(refusal(&empty).contains("nothing to launch"));

        let packed = window(DType::F4, &[8], &[1], 0);
        assert!(refusal(&packed).contains("dtype is f4e2m1"));
    }

    #[test]
    fn the_bits_are_an_element_of_the_destinations_width() {
        // A half-precision element is sixteen bits: one more does not fit.
        let wide = window(DType::BF16, &[8], &[1], 0x1_0000);
        let message = refusal(&wide);
        assert!(message.contains("do not fit a bf16 element"), "{message}");

        let whole = window(DType::BF16, &[8], &[1], 0xffff);
        Fill::check(&Descriptor::of(&whole, BLOCK, false), &whole).unwrap();

        // A 64-bit element takes every pattern, a negative integer's included.
        let negative = window(DType::I64, &[8], &[1], (-1_i64) as u64);
        Fill::check(&Descriptor::of(&negative, BLOCK, false), &negative).unwrap();

        let byte = window(DType::U8, &[8], &[1], 0x100);
        assert!(refusal(&byte).contains("do not fit a u8 element"));
    }

    #[test]
    fn an_fp8_element_is_stored_as_its_byte() {
        // One in E4M3: a zero sign, the exponent's bias and no mantissa.
        let ones = window(DType::F8E4M3, &[4, 8], &[8, 1], 0x38);
        let descriptor = Descriptor::of(&ones, BLOCK, false);
        assert_eq!(
            descriptor.to_string(),
            "dtype_f8e4m3_rank_1_block_1024_pdl_false"
        );
        Fill::check(&descriptor, &ones).unwrap();
        assert_eq!(Fill::tolerance(&descriptor), Tolerance::exact());

        // A block of a larger FP8 tensor is a window like any other.
        let block = window(DType::F8E4M3, &[4, 8], &[32, 1], 0);
        Fill::check(&Descriptor::of(&block, BLOCK, false), &block).unwrap();

        let wide = window(DType::F8E4M3, &[8], &[1], 0x100);
        assert!(refusal(&wide).contains("do not fit a f8e4m3 element"));
    }
}
