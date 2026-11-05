//! NVFP4: the representation a calibrated checkpoint stores a matrix in, and the products over it.
//!
//! A ModelOpt NVFP4 checkpoint stores a matrix as packed E2M1 values, two to a byte, with one E4M3
//! scale per block of sixteen elements and one FP32 tensor scale for the matrix
//! (specs/worker.md 9, "两个模型的 NVFP4 检查点与 FP8 KV 页"). Activations are encoded into the same
//! form at run time and the products read it:
//!
//! ```text
//! nvfp4_quantize   encode a BF16/FP16 row into packed E2M1 values, E4M3 block scales and one
//!                  per-token factor
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

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
use crate::platform::{self, Stream};
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
///
/// The two are distinct statements, and confusing them costs a weight load: **`Linear` is the
/// checkpoint's own layout** — what a calibrated file publishes — while `Swizzled128X4` is what a
/// product's kernel reads when its program was compiled for it. The loader reads the first and
/// transforms it into whichever the product asks for (`nn::weights::swizzled_scales`), so the same
/// checkpoint serves either. Which one a product accepts is the **package's** decision and is not
/// implied by this enum: `nvfp4_gemm/cute_dsl` serves `Swizzled128X4` only and refuses the other by
/// name (`src/build.py:48`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleLayout {
    /// `[rows, columns / 16]` row major, unpadded: the checkpoint's layout.
    Linear,
    /// Rows padded to 128 and scale columns to 4, interleaved: what a product compiled for it reads.
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
    /// `[1]` FP32: the encoder's global scale, the reciprocal of the checkpoint's activation tensor
    /// scale.
    ///
    /// A device operand rather than a launch scalar: the encoder is calibrated data of the tensor
    /// it encodes, published beside the weights, and the product that consumes the encoding reads
    /// it as each thread block starts — so a host value here would serialize the pair.
    pub global_scale: TensorView,
    /// `[rows]` FP32: the row's **dequantization factor**, the number the product multiplies a
    /// block's E4M3 scale by.
    ///
    /// Encoding is two-level micro-scaling, not one: a row is first brought into the range of the
    /// block format by a factor the row itself determines, and each sixteen-element block then
    /// carries its own E4M3 scale relative to that factor. A product reads both, so the factor is
    /// an output of the encode beside `values` and `scales`.
    ///
    /// The value is the reference encoder's: `token_scale[r] = amax(row r) * global_scale`, and
    /// the block's scale is `block_amax / (6 * amax(row r))`, so a dequantized element is
    /// `code * block_scale * token_scale`. This is a *data* output and not a descriptor field: the
    /// same module serves every row count, and no compilation depends on how many rows there are.
    pub token_scale: TensorView,
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

        args.global_scale.require_row(Self::NAME, DType::F32, 1)?;

        args.token_scale
            .require_shape(Self::NAME, Shape::new(&[rows]))?;
        args.token_scale.require_row(Self::NAME, DType::F32, rows)?;

        // One program per row: the row's amax is a reduction over the whole row and every block of
        // the row is scaled against it, so a program that owned a fragment of a row could not
        // state the factor its blocks are relative to. The block size is the chunk a program walks
        // the row in, not the grid's extent.
        let elements = rows;
        let mut arguments = Arguments::new();
        arguments
            .view(args.input)
            .view(args.values)
            .view(args.scales)
            .u64(rows)
            .view(args.global_scale)
            .view(args.token_scale);
        kernel.launch(LaunchGrid::covering(elements, 1)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        QuantizeArgs {
            values: twins.shadow_of(args.values),
            scales: twins.shadow_of(args.scales),
            token_scale: twins.shadow_of(args.token_scale),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The encode is a declared step with a bound of its own, and it is not the product's: an
        // element's error is the E2M1 rounding of its code relative to its block's scale (half a
        // step of a four-bit format, so one sixteenth of the block's scale) plus the two divisions
        // that state the per-token factor and the block scale (one rounding each, as the reference
        // encoder's approximate reciprocal does). The product over an encoded operand keeps the
        // 7.3 bound over its own contraction and absorbs none of this.
        //
        // Between two implementations of one variant the bytes are nonetheless equal: an encoding
        // is a table lookup per element and per block, and two encoders that round differently are
        // two programs, which is why the layout is part of the descriptor. So the operator states
        // the arithmetic above in its documentation and compares exactly.
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
    /// Element type of the block scales, as the checkpoint declares them: a real ModelOpt export
    /// stores them `F8_E4M3` (one per sixteen elements), and a stack's dtype is the checkpoint's,
    /// agreed across experts. Eight bits per block either way — the two names differ in what the
    /// byte *means*, not in how a program reads it, which is why one byte of it is decoded as an
    /// E4M3 scale whichever name the shard uses.
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

/// The call structure `nvfp4_gemm/cute_dsl`'s entry takes, as its header declares it.
///
/// The header asserts the two offsets this structure must match — `error` at byte 80 and a size
/// of 592 — because the entry reads seven device addresses, then `m`, `n` and `k`, and writes its
/// message into the tail. `bias` is never null: an epilogue that adds none passes `alpha`, which
/// the entry does not read in that case.
#[repr(C)]
pub struct Call {
    pub input: *const c_void,
    pub weight: *const c_void,
    pub input_scales: *const c_void,
    pub weight_scales: *const c_void,
    pub alpha: *const c_void,
    pub bias: *const c_void,
    pub out: *mut c_void,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub error: [u8; 512],
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

const _: () = assert!(
    std::mem::offset_of!(Call, error) == 80,
    "the call's message moved without the package's header"
);
const _: () = assert!(
    std::mem::size_of::<Call>() == 592,
    "the call's size moved without the package's header"
);

/// An encoded activation's block scales: one byte per scale, declared by whoever produced it.
///
/// `nvfp4_quantize` writes the E4M3 *integer* byte (`scale_bytes` in the Triton program), so the
/// encoder's output plane is bytes; a checkpoint may publish the same table as an `f8e4m3` tensor.
/// The product reads either bit-for-bit, so the element type is the producer's declaration and not
/// a property of the product. What a kernel does depend on is one byte per scale, which is what
/// this states — pinning one declaration instead refused the other producer and cost a run.
pub fn require_byte_scale(op: &'static str, view: TensorView, columns: u64) -> Result<()> {
    if !matches!(view.dtype, DType::U8 | DType::F8E4M3) {
        return Err(Error::argument(
            op,
            format!(
                "block scales are one byte each, declared U8 or f8e4m3, found {}",
                view.dtype
            ),
        ));
    }
    view.require_row(op, view.dtype, columns)
}

/// The operand pair an NVFP4 product requires, checked where a descriptor can be built.
///
/// The truth here is the descriptor's alone — the packed weight is E2M1 pairs carried as bytes and
/// its block scale is E4M3, two element types by construction. A check that compared one type
/// against both refused every real product, and because it sat in `launch` only a run could say so.
/// Called from `launch` as a backstop and from the layer that builds the descriptor, which is where
/// a test can reach it.
pub fn check_gemm(descriptor: &GemmDescriptor) -> Result<()> {
    if descriptor.packed != DType::U8 || descriptor.scale != DType::F8E4M3 {
        return Err(Error::argument(
            Nvfp4Gemm::NAME,
            format!(
                "an NVFP4 product takes U8 packed bytes and an f8e4m3 block scale, not {} and {}",
                descriptor.packed, descriptor.scale
            ),
        ));
    }
    Ok(())
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
        // The descriptor's own truth, refused here as a backstop for a caller that built one by
        // hand; the layer that builds it calls the same check (`fp4::check_gemm`).
        check_gemm(descriptor)?;
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
        // The activation's block scales are the encoder's output, so their element type is the
        // encoder's declaration (`require_byte_scale`); the weight's are the checkpoint's, which the
        // descriptor's key names.
        require_byte_scale(
            Self::NAME,
            args.input_scales,
            args.input_scales.shape.last(),
        )?;
        args.input_scales.require_shape(
            Self::NAME,
            descriptor.layout.shape(m, u64::from(descriptor.k)),
        )?;
        // An E4M3 block scale, not a byte of packed values: the scale table is its own element
        // type and the packed weight is the only U8 operand of this product.
        args.weight_scales.require_row(
            Self::NAME,
            DType::F8E4M3,
            args.weight_scales.shape.last(),
        )?;
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

        let mut call = Call {
            input: platform::address(args.input.ptr) as usize as *const c_void,
            weight: platform::address(args.weight.ptr) as usize as *const c_void,
            input_scales: platform::address(args.input_scales.ptr) as usize as *const c_void,
            weight_scales: platform::address(args.weight_scales.ptr) as usize as *const c_void,
            alpha: platform::address(args.alpha.ptr) as usize as *const c_void,
            bias: platform::address(bias.ptr) as usize as *const c_void,
            out: platform::address(args.out.ptr) as usize as *mut c_void,
            m: m as i64,
            n: n as i64,
            k: i64::from(descriptor.k),
            error: [0; 512],
        };
        // SAFETY: `Call` is the structure this package's entry declares, with the layout its
        // header asserts, and the entry the package resolved is `uniserve_nvfp4_gemm`.
        unsafe { kernel.native(&mut call, stream) }
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
        // FP32 sum is formed, which the standard bound covers — over a reduction of `k` terms, not
        // of two: `k` is the contraction the program actually runs, and a bound with two terms in
        // it was tighter than any correct kernel by the width of the reduction (programs measured
        // one element outside it on a correct program).
        Tolerance::rounding(
            u64::from(descriptor.k),
            descriptor.accumulate,
            descriptor.output,
        )
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

    /// The two operands of an NVFP4 product are two element types, and the pair is the descriptor's.
    ///
    /// The defect this pins: the check demanded `U8` for both, so every real product was refused
    /// before its launch and one run was spent finding it.
    #[test]
    fn a_product_takes_byte_weights_and_an_e4m3_scale() {
        let descriptor = |packed, scale| GemmDescriptor {
            packed,
            scale,
            accumulate: DType::F32,
            output: DType::BF16,
            layout: ScaleLayout::Swizzled128X4,
            epilogue: Epilogue::None,
            m: 128,
            n: 4096,
            k: 2048,
            pdl: false,
        };
        assert!(
            check_gemm(&descriptor(DType::U8, DType::F8E4M3)).is_ok(),
            "the operands a checkpoint states"
        );
        for (packed, scale, what) in [
            (DType::U8, DType::U8, "an E4M3 scale declared as bytes"),
            (
                DType::F8E4M3,
                DType::F8E4M3,
                "packed pairs declared as floats",
            ),
        ] {
            let error = check_gemm(&descriptor(packed, scale))
                .err()
                .unwrap_or_else(|| panic!("{what} was accepted"));
            let text = error.to_string();
            assert!(
                text.contains("U8 packed bytes") && text.contains("f8e4m3 block scale"),
                "{text}"
            );
        }
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
