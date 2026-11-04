//! NVFP4: the representation a calibrated checkpoint stores a matrix in, and the products over it.
//!
//! A ModelOpt NVFP4 checkpoint stores a matrix as packed E2M1 values, two to a byte, with one E4M3
//! scale per block of sixteen elements and one FP32 tensor scale for the matrix
//! (specs/worker.md 9, "两个模型的 NVFP4 检查点与 FP8 KV 页"). Activations are encoded into the same
//! form at run time and the products read it:
//!
//! ```text
//! nvfp4_quantize   encode a BF16/FP16 row block into packed E2M1 values and E4M3 block scales
//! nvfp4_gemm       a dense product over packed weights, a packed activation and one FP32 alpha
//! ```
//!
//! Both are kernel programs whose arithmetic is the vendor kernel's (FlashInfer's quantizer and
//! CUTLASS's or cuDNN's FP4 product, which the design asks for as programs of their own); what this
//! module fixes is the descriptor of a compilation, the argument list, the scale layouts and the
//! tolerance between two implementations of the same variant.
//!
//! **Scale layouts.** A matrix's E4M3 block scales are laid out in one of two ways, and the layout
//! is part of the compilation, not a runtime choice: `linear` fills `[rows, columns / 16]` row
//! major, which is what CUTLASS's FP4 product reads, while `Swizzled128X4` pads the rows to a
//! multiple of 128 and the scale columns to a multiple of 4 and interleaves them, which is what
//! cuDNN reads. The two products are separate variants because their results differ in the last bit
//! of a few elements — a difference of the product, not of the encoding.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one E4M3 block scale covers: NVFP4's sixteen-element block.
pub const BLOCK_ELEMENTS: u32 = 16;

/// The padded row and scale-column granularity of the swizzled scale layout.
pub const SWIZZLE_ROWS: u64 = 128;
/// The scale columns the swizzled layout groups.
pub const SWIZZLE_COLUMNS: u64 = 4;

/// How a matrix's E4M3 block scales are stored.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleLayout {
    /// `[rows, columns / 16]` row major: the layout CUTLASS's FP4 product reads.
    Linear,
    /// Rows padded to 128 and scale columns to 4, interleaved: the layout cuDNN reads.
    Swizzled128X4,
}

impl fmt::Display for ScaleLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Linear => "linear",
            Self::Swizzled128X4 => "swizzled_128x4",
        })
    }
}

impl ScaleLayout {
    /// The scale columns a row of `columns` elements has.
    pub fn columns(self, columns: u64) -> u64 {
        columns.div_ceil(u64::from(BLOCK_ELEMENTS))
    }

    /// The shape a scale field of a `[rows, columns]` matrix takes in this layout.
    ///
    /// The linear layout is the matrix's own scale matrix; the swizzled one is that matrix padded
    /// to [`SWIZZLE_ROWS`] rows and [`SWIZZLE_COLUMNS`] scale columns.
    pub fn shape(self, rows: u64, columns: u64) -> Shape {
        let scales = self.columns(columns);
        match self {
            Self::Linear => Shape::new(&[rows, scales]),
            Self::Swizzled128X4 => Shape::new(&[
                rows.div_ceil(SWIZZLE_ROWS) * SWIZZLE_ROWS,
                scales.div_ceil(SWIZZLE_COLUMNS) * SWIZZLE_COLUMNS,
            ]),
        }
    }
}

/// One compilation of `nvfp4_quantize`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuantizeDescriptor {
    /// Element type of the activations, BF16 or FP16.
    pub input: DType,
    /// Elements per row; a multiple of [`BLOCK_ELEMENTS`].
    pub columns: u32,
    /// How the block scales are laid out.
    pub layout: ScaleLayout,
    /// Elements one program encodes; a multiple of [`BLOCK_ELEMENTS`].
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for QuantizeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("input", self.input)
            .field("columns", self.columns)
            .field("layout", self.layout)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `nvfp4_quantize`.
#[derive(Clone, Copy)]
pub struct QuantizeArgs {
    /// The activations, `[rows, columns]`, unit stride on the last axis.
    pub input: TensorView,
    /// The packed E2M1 values, `[rows, columns / 2]` bytes.
    pub values: TensorView,
    /// The E4M3 block scales, in the descriptor's layout.
    pub scales: TensorView,
    /// The encoder's global scale: the reciprocal of the checkpoint's FP32 tensor scale.
    pub global_scale: f32,
}

/// Encoding activations into the checkpoint's E2M1 representation.
pub struct Nvfp4Quantize;

impl Op for Nvfp4Quantize {
    const NAME: &'static str = "nvfp4_quantize";

    type Descriptor = QuantizeDescriptor;
    type Args<'a> = QuantizeArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.input != DType::BF16 && descriptor.input != DType::F16 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} activations; NVFP4 encodes BF16 or FP16",
                    descriptor.input
                ),
            ));
        }
        if !descriptor.columns.is_multiple_of(BLOCK_ELEMENTS) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} columns are not whole {BLOCK_ELEMENTS} element blocks",
                    descriptor.columns
                ),
            ));
        }
        if !descriptor.block.is_multiple_of(BLOCK_ELEMENTS) || descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} elements per program are not whole blocks",
                    descriptor.block
                ),
            ));
        }
        let columns = u64::from(descriptor.columns);
        args.input
            .require_row(Self::NAME, descriptor.input, columns)?;
        if args.input.strides.last() != 1 {
            return Err(Error::argument(
                Self::NAME,
                "the activations need a unit stride on the last axis",
            ));
        }
        let rows = args.input.rows();
        args.values
            .require_shape(Self::NAME, Shape::new(&[rows, columns / 2]))?;
        args.values
            .require_row(Self::NAME, DType::U8, columns / 2)?;
        args.scales
            .require_row(Self::NAME, DType::U8, args.scales.shape.last())?;
        args.scales
            .require_shape(Self::NAME, descriptor.layout.shape(rows, columns))?;

        let elements = rows * columns;
        let mut arguments = Arguments::new();
        arguments
            .view(args.input)
            .view(args.values)
            .view(args.scales)
            .u64(rows)
            .f32(args.global_scale);
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        QuantizeArgs {
            values: twins.shadow_of(args.values),
            scales: twins.shadow_of(args.scales),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // An encoding is a table lookup per element and per block: two implementations of the same
        // variant produce the same bytes. Two encoder backends that round differently are two
        // programs, which is why the layout is part of the descriptor.
        Tolerance::exact()
    }
}

/// What follows the contraction of `nvfp4_gemm`.
pub use crate::gemm::Epilogue;

/// One compilation of `nvfp4_gemm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GemmDescriptor {
    /// Element type of the packed activations and weights: `U8`, two E2M1 values per byte.
    pub packed: DType,
    /// Element type of the block scales: `U8`, one E4M3 byte per block.
    pub scale: DType,
    /// Element type of the product before the epilogue.
    pub accumulate: DType,
    /// Element type of `out`.
    pub output: DType,
    /// The activation scales' layout, which also names the product: `Linear` is CUTLASS's and
    /// `Swizzled128X4` is cuDNN's.
    pub layout: ScaleLayout,
    /// What follows the contraction.
    pub epilogue: Epilogue,
    /// Rows of the activations and of the product.
    pub m: u32,
    /// Columns of the product.
    pub n: u32,
    /// The contraction length in elements: the packed operands each hold `k / 2` bytes a row.
    pub k: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for GemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("packed", self.packed)
            .field("scale", self.scale)
            .field("accumulate", self.accumulate)
            .field("output", self.output)
            .field("layout", self.layout)
            .field("epilogue", self.epilogue)
            .field("m", self.m)
            .field("n", self.n)
            .field("k", self.k)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `nvfp4_gemm`.
#[derive(Clone, Copy)]
pub struct GemmArgs {
    /// The packed activations, `[m, k / 2]` bytes.
    pub input: TensorView,
    /// The packed weight, `[n, k / 2]` bytes, N×K.
    pub weight: TensorView,
    /// The activation scales, in the descriptor's layout, over `[m, k]`.
    pub input_scales: TensorView,
    /// The weight scales, swizzled, over `[n, k]`.
    pub weight_scales: TensorView,
    /// `[1]` FP32: the product of the activation's and the weight's tensor scales.
    ///
    /// It is a device operand rather than a host scalar because the epilogue reads it as each
    /// thread block starts, before waiting for the kernel that computed it.
    pub alpha: TensorView,
    /// The bias, `[n]`, when the epilogue adds one.
    pub bias: Option<TensorView>,
    /// The product, `[m, n]`.
    pub out: TensorView,
}

/// A dense product over NVFP4 operands.
pub struct Nvfp4Gemm;

impl Op for Nvfp4Gemm {
    const NAME: &'static str = "nvfp4_gemm";

    type Descriptor = GemmDescriptor;
    type Args<'a> = GemmArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.packed != DType::U8 || descriptor.scale != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "NVFP4 operands are U8 bytes, not {} and {}",
                    descriptor.packed, descriptor.scale
                ),
            ));
        }
        let m = args.input.rows();
        let n = args.weight.rows();
        let k = args.input.last();
        if k * 2 != u64::from(descriptor.k) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the module contracts over {} elements, the packed row holds {k} bytes",
                    descriptor.k
                ),
            ));
        }
        args.input
            .require_shape(Self::NAME, Shape::new(&[m, u64::from(descriptor.k / 2)]))?;
        args.input.require_row(Self::NAME, DType::U8, k)?;
        args.weight
            .require_shape(Self::NAME, Shape::new(&[n, u64::from(descriptor.k / 2)]))?;
        args.weight.require_row(Self::NAME, DType::U8, k)?;
        args.input_scales
            .require_row(Self::NAME, DType::U8, args.input_scales.shape.last())?;
        args.input_scales.require_shape(
            Self::NAME,
            descriptor.layout.shape(m, u64::from(descriptor.k)),
        )?;
        args.weight_scales
            .require_row(Self::NAME, DType::U8, args.weight_scales.shape.last())?;
        args.weight_scales.require_shape(
            Self::NAME,
            ScaleLayout::Swizzled128X4.shape(n, u64::from(descriptor.k)),
        )?;
        args.alpha.require_shape(Self::NAME, Shape::new(&[1]))?;
        args.alpha.require_row(Self::NAME, DType::F32, 1)?;
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
        let bias = match (descriptor.epilogue, args.bias) {
            (Epilogue::Bias, Some(bias)) => {
                bias.require_shape(Self::NAME, Shape::new(&[n]))?;
                bias
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
            (Epilogue::None, None) => args.alpha,
        };

        let mut arguments = Arguments::new();
        arguments
            .view(args.input)
            .view(args.weight)
            .view(args.input_scales)
            .view(args.weight_scales)
            .view(args.alpha)
            .view(bias)
            .view(args.out)
            .u64(m)
            .u64(n)
            .u64(u64::from(descriptor.k));
        kernel.launch(
            LaunchGrid::new([u64::from(descriptor.m), u64::from(descriptor.n), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The packed operands are exact and the contraction accumulates in FP32; the one rounding
        // is the epilogue's write. Two implementations of one variant differ only in the order the
        // FP32 sum is formed, which the standard bound covers.
        Tolerance::rounding(2, descriptor.accumulate, descriptor.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scale_matrix_is_the_matrix_it_covers() {
        // One E4M3 scale per sixteen elements, evenly: 4096 elements are 256 blocks a row.
        assert_eq!(ScaleLayout::Linear.columns(4096), 256);
        assert_eq!(
            ScaleLayout::Linear.columns(1),
            1,
            "a partial block has a scale"
        );
        assert_eq!(
            ScaleLayout::Linear.shape(3, 4096),
            Shape::new(&[3, 256]),
            "the linear layout is the matrix's own"
        );
        // The swizzled layout pads rows to 128 and scale columns to 4.
        assert_eq!(
            ScaleLayout::Swizzled128X4.shape(3, 4096),
            Shape::new(&[128, 256])
        );
        assert_eq!(
            ScaleLayout::Swizzled128X4.shape(129, 64),
            Shape::new(&[256, 4])
        );
    }

    #[test]
    fn the_quantizer_and_the_product_are_separate_compilations() {
        let quantize = QuantizeDescriptor {
            input: DType::BF16,
            columns: 4096,
            layout: ScaleLayout::Linear,
            block: 1024,
            pdl: false,
        };
        assert_eq!(
            crate::descriptor::key(&quantize),
            "input_bf16_columns_4096_layout_linear_block_1024_pdl_false"
        );

        let gemm = GemmDescriptor {
            packed: DType::U8,
            scale: DType::U8,
            accumulate: DType::F32,
            output: DType::BF16,
            layout: ScaleLayout::Swizzled128X4,
            epilogue: Epilogue::None,
            m: 4,
            n: 4096,
            k: 4096,
            pdl: true,
        };
        let key = crate::descriptor::key(&gemm);
        assert!(key.contains("layout_swizzled_128x4"), "{key}");
        assert_ne!(
            key,
            crate::descriptor::key(&GemmDescriptor {
                layout: ScaleLayout::Linear,
                ..gemm
            }),
            "the scale layout names the product, so it is part of the compilation"
        );
    }
}
