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
//! The weight order is a descriptor value because it is the checkpoint's layout, not this call's.
//! The contraction length `k` is a descriptor value too: a module's tile and split choices are made
//! for one `k`, so a launch whose input disagrees with the descriptor is a launch of the wrong
//! module and fails here.
//!
//! The program is a native library: cuBLASLt's algorithm choice, its workspace and its epilogue are
//! the package's tuning data (specs/worker.md 5.4) and its host code, so the grid is the library's
//! and this launch computes none. The library receives [`Call`], whose layout the package's header
//! mirrors.

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::{self, DevicePtr, Stream};
use crate::tensor::{DType, Shape, Strides, TensorView};
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
        Key::new()
            .field("input", self.input)
            .field("weight", self.weight)
            .field("accumulate", self.accumulate)
            .field("output", self.output)
            .field("order", self.order)
            .field("epilogue", self.epilogue)
            .field("k", self.k)
            .finish(formatter)
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
#[derive(Clone, Copy)]
pub struct Args {
    /// Activations, `[m, k]`.
    pub input: TensorView,
    /// The weight, in the descriptor's order.
    pub weight: TensorView,
    /// The bias, `[n]`, when the epilogue adds one.
    pub bias: Option<TensorView>,
    /// The product, `[m, n]`.
    pub out: TensorView,
    /// A scalar applied to the product before the epilogue.
    pub alpha: f32,
}

/// The dense product operator.
pub struct Gemm;

impl Op for Gemm {
    const NAME: &'static str = "gemm";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
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
        let (n, weight_row_stride) = match descriptor.order {
            WeightOrder::KxN => (args.weight.last(), row_stride(&args.weight.strides)),
            WeightOrder::NxK => (args.weight.rows(), row_stride(&args.weight.strides)),
        };
        args.input.require_row(Self::NAME, descriptor.input, k)?;
        if args.out.dtype != descriptor.output {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} output, found {}",
                    descriptor.output, args.out.dtype
                ),
            ));
        }
        args.out.require_shape(Self::NAME, Shape::new(&[m, n]))?;
        if args.out.strides.last() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected a unit stride on the output's last axis, found {}",
                    args.out.strides.last()
                ),
            ));
        }
        args.weight.require_shape(
            Self::NAME,
            match descriptor.order {
                WeightOrder::KxN => Shape::new(&[k, n]),
                WeightOrder::NxK => Shape::new(&[n, k]),
            },
        )?;

        let bias = match (descriptor.epilogue, args.bias) {
            (Epilogue::Bias, Some(bias)) => {
                bias.require_shape(Self::NAME, Shape::new(&[n]))?;
                address(bias.ptr) as usize as *const c_void
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
            (Epilogue::None, None) => std::ptr::null(),
        };

        let mut call = Call {
            out: address(args.out.ptr) as usize as *mut c_void,
            input: address(args.input.ptr) as usize as *const c_void,
            weight: address(args.weight.ptr) as usize as *const c_void,
            bias,
            m: to_i64(m)?,
            n: to_i64(n)?,
            k: to_i64(k)?,
            input_row_stride: to_i64(row_stride(&args.input.strides))?,
            weight_row_stride: to_i64(weight_row_stride)?,
            out_row_stride: to_i64(row_stride(&args.out.strides))?,
            alpha: args.alpha,
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::dot(
            u64::from(descriptor.k),
            descriptor.accumulate,
            descriptor.output,
        )
    }
}

/// The elements between consecutive rows of a view.
fn row_stride(strides: &Strides) -> u64 {
    if strides.rank() >= 2 {
        strides.get(strides.rank() - 2)
    } else {
        0
    }
}

fn address(pointer: DevicePtr) -> u64 {
    platform::address(pointer)
}

fn to_i64(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(Gemm::NAME, "an extent above i64::MAX"))
}
