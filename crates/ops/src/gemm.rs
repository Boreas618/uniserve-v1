//! Dense matrix products.
//!
//! `gemm` is one operator with one shape contract and as many descriptors as there are dtype,
//! weight-layout and epilogue combinations:
//!
//! ```text
//! input  [m, k]        row-major, k elements on the last axis
//! weight [k, n]        (WeightOrder::KxN) or [n, k] (WeightOrder::NxK), as the checkpoint stores it
//! bias   [n]           when the descriptor declares one
//! out    [m, n]        row-major
//! ```
//!
//! Each operand is a matrix of rows: its last axis is contiguous, its rows are one fixed stride
//! apart, and that stride is no shorter than a row. The library reads the row stride as a
//! leading dimension, so a window of a wider tensor is read in place, while a transposed, a
//! stepped and a broadcast view are refused; their caller makes them contiguous first.
//!
//! The weight order is a descriptor value because it is the checkpoint's layout, not this call's.
//! The contraction length `k` is a descriptor value too: a module's tile and split choices are made
//! for one `k`, so a launch whose input disagrees with the descriptor is a launch of the wrong
//! module and fails here.
//!
//! The program is a native library: cuBLASLt's algorithm choice, its workspace and its epilogue are
//! the package's tuning data and its host code, so the grid is the library's
//! and this launch computes none. The library receives [`Call`], whose layout the package's header
//! mirrors.

use std::ffi::c_void;
use std::fmt;

use platform::{DevicePtr, Stream};
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// How the weight matrix is stored.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WeightOrder {
    /// `[k, n]`: the contraction axis first.
    KxN,
    /// `[n, k]`: the output axis first, as Hugging Face checkpoints store a projection.
    NxK,
}

impl fmt::Display for WeightOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::KxN => "kxn",
            Self::NxK => "nxk",
        })
    }
}

/// What the product does after the contraction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Epilogue {
    /// `out = input · weight`.
    None,
    /// `out = input · weight + bias`.
    Bias,
}

impl fmt::Display for Epilogue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::Bias => "bias",
        })
    }
}

/// One compilation of `gemm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the input activations.
    pub input: DType,
    /// Element type of the weight matrix.
    pub weight: DType,
    /// Element type of the product before the epilogue.
    pub accumulate: DType,
    /// Element type of `out`.
    pub output: DType,
    /// How the weight is stored.
    pub order: WeightOrder,
    /// What follows the contraction.
    pub epilogue: Epilogue,
    /// The contraction length the module was compiled for.
    pub k: u32,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The contraction length is not in the key: cublasLt takes `m`, `n` and `k` with the call and
        // the module is the same code whatever they are. A key that named `k` compiled one module per
        // weight shape for no gain.
        Key::new()
            .field("input", abi::name(self.input))
            .field("weight", abi::name(self.weight))
            .field("accumulate", abi::name(self.accumulate))
            .field("output", abi::name(self.output))
            .field("order", self.order)
            .field("epilogue", self.epilogue)
            .finish(formatter)
    }
}

impl Descriptor {
    /// The descriptor of the product `args` states, with the weight stored in `order`: the
    /// operands' element types, an FP32 accumulation rounded once into the output's type, a bias
    /// epilogue exactly when the call has a bias, and the contraction over the input's last axis.
    ///
    /// This is the one place a dense product's descriptor is made: the tensor library's `matmul`
    /// on this backend and `nn`'s projections state their calls and take the descriptor from here,
    /// so one product has one key whichever path launched it.
    ///
    /// # Errors
    ///
    /// [`Error::Argument`] when the input's last axis does not fit the descriptor's 32 bits.
    pub fn of(args: &Args, order: WeightOrder) -> Result<Self> {
        let k = u32::try_from(args.input.last()).map_err(|_| {
            Error::argument(
                Gemm::NAME,
                format!(
                    "a contraction over {} elements exceeds the operator's length",
                    args.input.last()
                ),
            )
        })?;

        Ok(Self {
            input: args.input.dtype,
            weight: args.weight.dtype,
            accumulate: DType::F32,
            output: args.out.dtype,
            order,
            epilogue: if args.bias.is_some() {
                Epilogue::Bias
            } else {
                Epilogue::None
            },
            k,
        })
    }
}

/// The call structure the package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/gemm/<backend>/include/uniserve_gemm.h`; the layout is
/// pinned by [`CallLayout`], the assertions below and the test beside this module.
#[repr(C)]
pub struct Call {
    pub out: *mut c_void,
    pub input: *const c_void,
    pub weight: *const c_void,
    /// Null when the epilogue adds no bias.
    pub bias: *const c_void,
    /// Rows of the input and of the product.
    pub m: i64,
    /// Columns of the product.
    pub n: i64,
    /// The contraction length.
    pub k: i64,
    /// Elements between consecutive input rows.
    pub input_row_stride: i64,
    /// Elements between consecutive weight rows of the stored order.
    pub weight_row_stride: i64,
    /// Elements between consecutive output rows.
    pub out_row_stride: i64,
    /// A scalar applied to the product before the epilogue; 1.0 for a plain product.
    pub alpha: f32,
    /// The entry's message on a nonzero return.
    pub error: [u8; 512],
}

/// The size and field offsets the package's header must mirror, in bytes.
pub const GEMM_CALL_LAYOUT: CallLayout = CallLayout {
    size: 600,
    out: 0,
    input: 8,
    weight: 16,
    bias: 24,
    m: 32,
    n: 40,
    k: 48,
    input_row_stride: 56,
    weight_row_stride: 64,
    out_row_stride: 72,
    alpha: 80,
    error: 84,
};

/// The layout of [`Call`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub out: usize,
    pub input: usize,
    pub weight: usize,
    pub bias: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub input_row_stride: usize,
    pub weight_row_stride: usize,
    pub out_row_stride: usize,
    pub alpha: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<Call>() == GEMM_CALL_LAYOUT.size,
    "the gemm call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, error) == GEMM_CALL_LAYOUT.error,
    "the gemm call structure's `error` moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, out_row_stride) == GEMM_CALL_LAYOUT.out_row_stride,
    "the gemm call structure's strides moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, weight_row_stride) == GEMM_CALL_LAYOUT.weight_row_stride,
    "the gemm call structure's weight stride moved without its header"
);

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// The arguments of one product.
#[derive(Clone)]
pub struct Args {
    /// Activations, `[m, k]`.
    pub input: Operand,
    /// The weight, in the descriptor's order.
    pub weight: Operand,
    /// The bias, `[n]`, when the epilogue adds one.
    pub bias: Option<Operand>,
    /// The product, `[m, n]`.
    pub out: Operand,
    /// A scalar applied to the product before the epilogue.
    pub alpha: f32,
}

/// The dense product operator.
pub struct Gemm;

impl Gemm {
    /// Refuse a descriptor whose two operands disagree in type.
    ///
    /// A contraction is one type on both sides: `cublasLtMatmul` takes a single compute type and a
    /// pair of different operand types is not a cell this operator can compile. The check lives in
    /// the launch so a descriptor that slipped into a fixture — an f32 activation against a bf16
    /// weight, say — is refused here, naming both types, rather than inside the library on the
    /// first call, where the refusal says nothing about which descriptor was wrong.
    fn operands(descriptor: &Descriptor) -> Result<()> {
        if descriptor.input == descriptor.weight {
            return Ok(());
        }
        Err(Error::argument(
            Self::NAME,
            format!(
                "the operands are {} input against {} weight: a contraction takes one type on both sides",
                abi::name(descriptor.input),
                abi::name(descriptor.weight)
            ),
        ))
    }
}

impl Op for Gemm {
    const NAME: &'static str = proto::module::operator::GEMM;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let m = args.input.rows();
        let k = args.input.last();
        if k != u64::from(descriptor.k) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the module contracts over {}, the input's last axis is {k}",
                    descriptor.k
                ),
            ));
        }
        Self::operands(descriptor)?;
        let (n, weight_row_stride) = match descriptor.order {
            WeightOrder::KxN => (args.weight.last(), row_stride(&args.weight)),
            WeightOrder::NxK => (args.weight.rows(), row_stride(&args.weight)),
        };
        args.input.require_row(Self::NAME, descriptor.input, k)?;
        if args.out.dtype != descriptor.output {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} output, found {}",
                    abi::name(descriptor.output),
                    abi::name(args.out.dtype)
                ),
            ));
        }
        args.out.require_shape(Self::NAME, &[m, n])?;
        if args.out.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected a unit stride on the output's last axis, found {}",
                    args.out.last_stride()
                ),
            ));
        }
        let weight_dims = match descriptor.order {
            WeightOrder::KxN => [k, n],
            WeightOrder::NxK => [n, k],
        };
        args.weight.require_shape(Self::NAME, &weight_dims)?;
        if args.weight.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected a unit stride on the weight's last axis, found {}: a transposed \
                     weight is the other weight order over the stored matrix",
                    args.weight.last_stride()
                ),
            ));
        }
        require_rows("input", &args.input)?;
        require_rows("weight", &args.weight)?;
        require_rows("out", &args.out)?;

        match (descriptor.epilogue, &args.bias) {
            (Epilogue::Bias, Some(bias)) => {
                bias.require_shape(Self::NAME, &[n])?;
            }
            (Epilogue::Bias, None) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the descriptor's epilogue adds a bias and the call passed none",
                ));
            }
            (Epilogue::None, Some(_)) => {
                return Err(Error::argument(
                    Self::NAME,
                    "the call passed a bias the descriptor's epilogue does not add",
                ));
            }
            (Epilogue::None, None) => {}
        }

        // The package's call structure carries every extent and stride as an `i64`.
        for value in [
            m,
            n,
            k,
            row_stride(&args.input),
            weight_row_stride,
            row_stride(&args.out),
        ] {
            to_i64(value)?;
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let m = args.input.rows();
        let k = args.input.last();
        let (n, weight_row_stride) = match descriptor.order {
            WeightOrder::KxN => (args.weight.last(), row_stride(&args.weight)),
            WeightOrder::NxK => (args.weight.rows(), row_stride(&args.weight)),
        };
        // `check` pairs the bias with the epilogue that adds it.
        let bias = match &args.bias {
            Some(bias) => address(bias.ptr) as usize as *const c_void,
            None => std::ptr::null(),
        };

        // `check` holds every extent and stride within an `i64`.
        let mut call = Call {
            out: address(args.out.ptr) as usize as *mut c_void,
            input: address(args.input.ptr) as usize as *const c_void,
            weight: address(args.weight.ptr) as usize as *const c_void,
            bias,
            m: m as i64,
            n: n as i64,
            k: k as i64,
            input_row_stride: row_stride(&args.input) as i64,
            weight_row_stride: weight_row_stride as i64,
            out_row_stride: row_stride(&args.out) as i64,
            alpha: args.alpha,
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // `k` products accumulated in the accumulation type and rounded once into the output: the
        // standard bound with the reduction length, and `dot`'s square-root term because the
        // products have mixed signs and a library is free to order the sum.
        Tolerance::dot(
            u64::from(descriptor.k),
            descriptor.accumulate,
            descriptor.output,
        )
    }
}

/// Check that the rows of the operand `name` are what the entry reads: runs of its last axis,
/// one fixed stride apart, the stride no shorter than a run.
///
/// The entry receives a row count, a row length and one row stride, which its library takes as
/// a leading dimension. Rows closer than their length overlap, and a row repeated by a zero
/// stride is one address stated many times: neither is a matrix the library accepts. The axes
/// before the row axis only multiply the row count, so each of them steps over the rows inside
/// it.
fn require_rows(name: &str, operand: &Operand) -> Result<()> {
    let (stride, length) = (row_stride(operand), operand.last());
    if stride < length {
        return Err(Error::argument(
            Gemm::NAME,
            format!(
                "the {name}'s rows are {stride} elements apart and {length} long: a row stride \
                 is a leading dimension, which is no shorter than a row, in the view {:?} with \
                 strides {:?}",
                operand.dims(),
                operand.strides()
            ),
        ));
    }

    let rank = operand.rank();
    for axis in 0..rank.saturating_sub(2) {
        let inside = operand
            .dim(axis + 1)
            .saturating_mul(operand.stride(axis + 1));

        if operand.dim(axis) > 1 && operand.stride(axis) != inside {
            return Err(Error::argument(
                Gemm::NAME,
                format!(
                    "axis {axis} of the {name} advances by {} elements where the rows inside it \
                     span {inside}: its rows are not one stride apart, in the view {:?} with \
                     strides {:?}",
                    operand.stride(axis),
                    operand.dims(),
                    operand.strides()
                ),
            ));
        }
    }
    Ok(())
}

/// The elements between consecutive rows of a view.
fn row_stride(view: &Operand) -> u64 {
    if view.rank() >= 2 {
        view.stride(view.rank() - 2)
    } else {
        0
    }
}

fn address(pointer: DevicePtr) -> u64 {
    pointer.address()
}

fn to_i64(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(Gemm::NAME, "an extent above i64::MAX"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(input: DType, weight: DType) -> Descriptor {
        Descriptor {
            input,
            weight,
            accumulate: DType::F32,
            output: DType::BF16,
            order: WeightOrder::NxK,
            epilogue: Epilogue::None,
            k: 2048,
        }
    }

    #[test]
    fn a_mixed_operand_pair_is_refused_by_name() {
        let refused = Gemm::operands(&descriptor(DType::F32, DType::BF16))
            .expect_err("an f32 activation against a bf16 weight has no program")
            .to_string();
        assert!(
            refused.contains("f32 input against bf16 weight"),
            "the refusal must name both operand types: {refused}"
        );
        assert!(
            Gemm::operands(&descriptor(DType::BF16, DType::BF16)).is_ok(),
            "a pair of one type is the cell the operator declares"
        );
    }

    fn matrix(shape: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(DevicePtr::from_address(0x1000), DType::BF16, shape, strides)
    }

    /// Four rows of eight against a weight stored `[n, k]` with five outputs.
    fn product() -> (Descriptor, Args) {
        let descriptor = Descriptor {
            k: 8,
            ..descriptor(DType::BF16, DType::BF16)
        };
        let args = Args {
            input: matrix(&[4, 8], &[8, 1]),
            weight: matrix(&[5, 8], &[8, 1]),
            bias: None,
            out: matrix(&[4, 5], &[5, 1]),
            alpha: 1.0,
        };
        (descriptor, args)
    }

    fn refusal(descriptor: &Descriptor, args: &Args) -> String {
        Gemm::check(descriptor, args).unwrap_err().to_string()
    }

    #[test]
    fn rows_a_stride_apart_are_read_in_place() {
        let (descriptor, args) = product();
        Gemm::check(&descriptor, &args).unwrap();

        // Each operand as a window of a wider tensor: its rows keep the wider tensor's stride.
        let windows = Args {
            input: matrix(&[4, 8], &[12, 1]),
            weight: matrix(&[5, 8], &[16, 1]),
            out: matrix(&[4, 5], &[9, 1]),
            ..args.clone()
        };
        Gemm::check(&descriptor, &windows).unwrap();

        // A batch folded into the rows: two members of two rows each, contiguous.
        let folded = Args {
            input: matrix(&[2, 2, 8], &[16, 8, 1]),
            ..args.clone()
        };
        Gemm::check(&descriptor, &folded).unwrap();

        // The weight in the other order, `[k, n]`.
        let stored = Descriptor {
            order: WeightOrder::KxN,
            ..descriptor
        };
        let args = Args {
            weight: matrix(&[8, 5], &[5, 1]),
            ..args.clone()
        };
        Gemm::check(&stored, &args).unwrap();
    }

    #[test]
    fn a_weight_whose_last_axis_is_not_contiguous_is_refused() {
        let (descriptor, args) = product();

        // `[k, n]` storage read as `[n, k]`: its transpose.
        let transposed = Args {
            weight: matrix(&[5, 8], &[1, 5]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &transposed);
        assert!(
            message.contains("expected a unit stride on the weight's last axis, found 5"),
            "{message}"
        );

        // Every other column of a wider weight.
        let stepped = Args {
            weight: matrix(&[5, 8], &[16, 2]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &stepped);
        assert!(message.contains("found 2"), "{message}");
    }

    #[test]
    fn a_row_stride_shorter_than_a_row_is_refused_for_every_operand() {
        let (descriptor, args) = product();

        // One row repeated over the batch: a broadcast.
        let broadcast = Args {
            input: matrix(&[4, 8], &[0, 1]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &broadcast);
        assert!(
            message.contains("the input's rows are 0 elements apart and 8 long"),
            "{message}"
        );

        // Rows that overlap.
        let overlapping = Args {
            weight: matrix(&[5, 8], &[4, 1]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &overlapping);
        assert!(
            message.contains("the weight's rows are 4 elements apart and 8 long"),
            "{message}"
        );

        let out = Args {
            out: matrix(&[4, 5], &[0, 1]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &out);
        assert!(
            message.contains("the out's rows are 0 elements apart and 5 long"),
            "{message}"
        );

        // A vector has no row axis to state a stride.
        let vector = Args {
            input: matrix(&[8], &[1]),
            out: matrix(&[1, 5], &[5, 1]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &vector);
        assert!(
            message.contains("the input's rows are 0 elements apart"),
            "{message}"
        );
    }

    #[test]
    fn a_batch_whose_members_are_not_a_row_stride_apart_is_refused() {
        let (descriptor, args) = product();

        // Two members of two rows, the members three rows apart.
        let spaced = Args {
            input: matrix(&[2, 2, 8], &[24, 8, 1]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &spaced);
        assert!(
            message.contains("axis 0 of the input advances by 24 elements"),
            "{message}"
        );

        // One member repeated.
        let repeated = Args {
            input: matrix(&[2, 2, 8], &[0, 8, 1]),
            ..args.clone()
        };
        let message = refusal(&descriptor, &repeated);
        assert!(
            message.contains("axis 0 of the input advances by 0"),
            "{message}"
        );
    }

    /// A product's descriptor follows from its operands, so a `matmul` of the tensor library and
    /// a projection of `nn` over the same operands state the key the stored module carries.
    #[test]
    fn a_products_descriptor_follows_from_its_operands() {
        let (stated, mut args) = product();
        let plain = Descriptor::of(&args, WeightOrder::NxK).unwrap();
        assert_eq!(plain, stated);
        assert_eq!(
            plain.to_string(),
            "input_bf16_weight_bf16_accumulate_f32_output_bf16_order_nxk_epilogue_none"
        );

        // A bias is the bias epilogue; the order is the caller's statement of the weight.
        args.bias = Some(matrix(&[5], &[1]));
        let biased = Descriptor::of(&args, WeightOrder::KxN).unwrap();
        assert_eq!(biased.epilogue, Epilogue::Bias);
        assert_eq!(biased.k, 8);
        assert_eq!(
            biased.to_string(),
            "input_bf16_weight_bf16_accumulate_f32_output_bf16_order_kxn_epilogue_bias"
        );
    }
}
