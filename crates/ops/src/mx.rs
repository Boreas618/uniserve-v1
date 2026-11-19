//! Microscaling (MX) formats and the dense product over them.
//!
//! An MX matrix stores its elements in blocks of [`BLOCK`] consecutive elements along the
//! contraction. Each block has one UE8M0 scale, a power of two stored as its biased exponent
//! byte, and its elements are E4M3 codes (MXFP8) or E2M1 codes packed two a byte (MXFP4); an
//! element's value is `code · scale`. A block-scaled FP8 checkpoint's weight, one UE8M0 scale per
//! 32 × 32 block, is an MXFP8 matrix once each block's scale is repeated over its 32 rows, which is
//! exact.
//!
//! ```text
//! mxfp8_quantize   encode 16-bit rows into E4M3 codes and one UE8M0 scale per 32 elements
//! mxfp8_gemm       out[m, n] = Σ_k (a[m, k] · sa[m, k / 32]) · (w[n, k] · sw[n, k / 32])
//!                  accumulated in FP32 and rounded once into BF16
//! ```
//!
//! ## Scales
//!
//! A scale byte `b` is the value `2^(b − 127)`; `b = 255` is not a number. Tensors hold the bytes
//! as `U8` ([`ue8m0`] decodes one), or four of them packed into one `U32` word for a product that
//! loads words ([`ScaleLayout::PackedMnMajor`]); no tensor element type is a scale.
//!
//! ## The encoding
//!
//! An encoder chooses each block's scale from the block's largest magnitude by one of two rules
//! ([`ScaleRule`]) and rounds every quotient to its E4M3 code:
//!
//! ```text
//! covering   amax = max(max |x| over the block, 10⁻⁴)
//!            e    = ⌈log₂ fl(amax · fl(1/448))⌉    the product's exponent, plus one unless it
//!                                                 is a power of two
//! ocp        e    = ⌊log₂ max |x|⌋ − 8              the block's largest exponent less E4M3's
//! scale = 2^e                                      stored as e + 127
//! code  = e4m3(clamp(x / scale, ±448))             round to nearest even
//! ```
//!
//! The covering rule is the reference's dynamic activation encoding (DeepSeek's `act_quant(x,
//! 32, scale_fmt = "ue8m0")`), which `mxfp8_quantize` computes: no quotient exceeds E4M3's range.
//! The OCP Microscaling specification's rule takes a scale half as large for a block whose
//! largest magnitude has a significand above 1.75, and that block's quotients above 448 saturate;
//! trtllm-gen's gated MXFP4 product encodes its gated rows by it (`gather_gemm_mxfp4`). Every
//! step of either is exact or one rounding to nearest, so two implementations write the same
//! bytes ([`block_exponent`] is a rule's scale).
//!
//! ## Scale layouts
//!
//! A product reads its operands' scales in the arrangement its kernels load ([`ScaleLayout`]):
//! the matrix's own `[rows, columns / 32]` table, the 128 × 4 atom block-scaled tensor-core
//! operands are swizzled into, or DeepGEMM's MN-major packed words. The layout is part of the
//! compilation that writes or reads it, never a launch choice.
//!
//! ## The product
//!
//! `mxfp8_gemm`'s package is DeepGEMM's SM100 block-scaled product (`mxfp8_gemm/deepgemm`), one
//! persistent kernel that serves every row count. Its tile is chosen for a row class
//! ([`ROW_CLASSES`]): the descriptor states the smallest class that holds the call's rows, as
//! [`crate::fp4::packed_rows`] does for the NVFP4 product, and a module computes any row count.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::{self, Key};
use crate::fp8::E4M3_MAX;
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
use crate::launch::PDL;
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one UE8M0 scale covers, along the contraction.
pub const BLOCK: u32 = 32;

/// The bias of a UE8M0 exponent byte: the byte `b` is the scale `2^(b − 127)`.
pub const EXPONENT_BIAS: i32 = 127;

/// The byte that is not a number.
pub const NOT_A_NUMBER: u8 = 255;

/// The smallest block magnitude the encoding takes: a block of zeros encodes under the scale of
/// this magnitude, as the reference's encoder clamps it.
pub const AMAX_FLOOR: f32 = 1e-4;

/// Scales one packed word of [`ScaleLayout::PackedMnMajor`] holds: four consecutive blocks of a
/// row, 128 elements of the contraction.
pub const WORD_SCALES: u64 = 4;

/// Elements of the contraction one packed scale word covers.
pub const WORD_ELEMENTS: u64 = WORD_SCALES * BLOCK as u64;

/// The rows a packed scale column is padded to: one TMA load reads whole 16-byte rows of words.
pub const WORD_ROWS: u64 = 4;

/// The value of the UE8M0 scale byte `byte`: `2^(byte − 127)`, not a number for 255.
pub fn ue8m0(byte: u8) -> f32 {
    match byte {
        NOT_A_NUMBER => f32::NAN,
        // 2⁻¹²⁷ lies below FP32's normal range: its one set bit is the subnormal mantissa's top.
        0 => f32::from_bits(1 << 22),
        biased => f32::from_bits(u32::from(biased) << 23),
    }
}

/// The exponent of E4M3's largest normal value, `448 = 1.75 · 2⁸`.
const E4M3_EXPONENT: i32 = 8;

/// How an encoder chooses a block's scale from the block's largest magnitude (module
/// documentation, the encoding).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleRule {
    /// The least power of two at or above `fl(max(amax, 10⁻⁴) · fl(1/448))`: no quotient exceeds
    /// ±448. DeepSeek's `act_quant(x, 32, scale_fmt = "ue8m0")`, which `mxfp8_quantize` computes.
    Covering,
    /// The OCP Microscaling specification's `2^(⌊log₂ amax⌋ − 8)`: the block's largest exponent
    /// less E4M3's, the quotients above 448 saturating to ±448. trtllm-gen's gated product writes
    /// its gated rows by it.
    Ocp,
}

/// The UE8M0 scale byte `rule` gives a block whose largest magnitude is `amax`, computed on the
/// bits of FP32 values as the encoders do: for [`ScaleRule::Covering`]
/// `⌈log₂ fl(max(amax, 10⁻⁴) · fl(1/448))⌉ + 127`, for [`ScaleRule::Ocp`] the exponent field of
/// `amax` less 8, the smallest byte for a block of zeros, whose scale no value observes.
///
/// A magnitude that is not finite has no scale and gives [`NOT_A_NUMBER`].
pub fn block_exponent(amax: f32, rule: ScaleRule) -> u8 {
    if !amax.is_finite() {
        return NOT_A_NUMBER;
    }

    match rule {
        ScaleRule::Covering => {
            let ratio = amax.max(AMAX_FLOOR) * (1.0 / E4M3_MAX);
            let bits = ratio.to_bits();
            let exponent = ((bits >> 23) & 0xff) as i32 - EXPONENT_BIAS;
            let rounded_up = i32::from(bits & 0x7f_ffff != 0);

            // The floor keeps the ratio a normal number, so its exponent field is its exponent
            // and the biased result lies well inside a byte.
            (exponent + rounded_up + EXPONENT_BIAS) as u8
        }
        ScaleRule::Ocp => {
            // The biased exponent field of `amax` is `⌊log₂ amax⌋ + 127` for a normal magnitude;
            // the scale's biased byte is that less E4M3's exponent. A magnitude below 2⁻¹¹⁸ takes
            // the smallest scale, under which its quotients are still below one.
            let field = ((amax.abs().to_bits() >> 23) & 0xff) as i32;
            (field - E4M3_EXPONENT).max(0) as u8
        }
    }
}

/// How an MX matrix's scales are arranged for the kernel that writes or reads them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScaleLayout {
    /// `[rows, columns / 32]` `U8`, row major: a checkpoint's own table, and what a routed
    /// product that gathers its tokens reads.
    Linear,
    /// `U8` rows padded to 128 and scale columns to 4, each `[128, 4]` tile interleaved as
    /// block-scaled tensor-core operands are read (TensorRT-LLM's `R128c4`, the layout
    /// [`crate::fp4::ScaleLayout::Swizzled128X4`] states for NVFP4's sixteen-element blocks).
    Swizzled128X4,
    /// DeepGEMM's operand scales: four consecutive scales of a row packed into one `U32` word
    /// (block `4g + i` in byte `i`), the words of one group `g` of a column stored for every
    /// row before the next group's (MN-major), the rows padded to [`WORD_ROWS`]:
    /// `[columns / 128, rows padded to 4]` `U32`.
    PackedMnMajor,
}

impl fmt::Display for ScaleLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Linear => "linear",
            Self::Swizzled128X4 => "swizzled_128x4",
            Self::PackedMnMajor => "packed_mn_major",
        })
    }
}

impl ScaleLayout {
    /// The element type the layout's tensor holds: scale bytes, or packed words.
    pub const fn dtype(self) -> DType {
        match self {
            Self::Linear | Self::Swizzled128X4 => DType::U8,
            Self::PackedMnMajor => DType::U32,
        }
    }

    /// The extents of the scale tensor of a `[rows, columns]` matrix in this layout.
    ///
    /// The caller holds `columns` to whole blocks, and to whole words for the packed layout.
    pub fn shape(self, rows: u64, columns: u64) -> [u64; 2] {
        let blocks = columns.div_ceil(u64::from(BLOCK));
        match self {
            Self::Linear => [rows, blocks],
            Self::Swizzled128X4 => [rows.div_ceil(128) * 128, blocks.div_ceil(4) * 4],
            Self::PackedMnMajor => [
                columns.div_ceil(WORD_ELEMENTS),
                rows.div_ceil(WORD_ROWS) * WORD_ROWS,
            ],
        }
    }
}

/// The scale table of a `[rows, columns]` matrix whose element `(r, c)` has the scale byte
/// `linear[r · blocks + c / 32]`, rearranged into `layout`.
///
/// `linear` is the matrix's own `[rows, columns / 32]` table. The padding of a padded layout is
/// zero bytes; a product reads padding only for rows or blocks the matrix does not have, whose
/// products it never stores. The packed layout's bytes are its words' little-endian bytes, so the
/// result is the published table of either element type.
///
/// # Panics
///
/// When `linear` is not `rows · columns / 32` bytes.
pub fn arranged(linear: &[u8], rows: u64, columns: u64, layout: ScaleLayout) -> Vec<u8> {
    let blocks = columns.div_ceil(u64::from(BLOCK));
    assert_eq!(
        linear.len() as u64,
        rows * blocks,
        "a [{rows}, {blocks}] scale table"
    );

    match layout {
        ScaleLayout::Linear => linear.to_vec(),
        ScaleLayout::Swizzled128X4 => {
            let [padded_rows, padded_blocks] = layout.shape(rows, columns);
            let mut swizzled = vec![0_u8; (padded_rows * padded_blocks) as usize];
            for row in 0..rows {
                for block in 0..blocks {
                    let index = (block % 4)
                        + (block / 4) * 512
                        + (row % 32) * 16
                        + ((row % 128) / 32) * 4
                        + (row / 128) * (128 * padded_blocks);
                    swizzled[index as usize] = linear[(row * blocks + block) as usize];
                }
            }
            swizzled
        }
        ScaleLayout::PackedMnMajor => {
            let [groups, padded_rows] = layout.shape(rows, columns);
            let mut packed = vec![0_u8; (groups * padded_rows * WORD_SCALES) as usize];
            for row in 0..rows {
                for block in 0..blocks {
                    // Word `(group, row)` holds the row's blocks `4 · group ..`, byte by byte.
                    let word = (block / WORD_SCALES) * padded_rows + row;
                    let index = word * WORD_SCALES + block % WORD_SCALES;
                    packed[index as usize] = linear[(row * blocks + block) as usize];
                }
            }
            packed
        }
    }
}

/// The linear `[rows, columns / 32]` scale table a table `published` in `layout` holds: the
/// inverse of [`arranged`], which a reader of a published table (a CPU formula) decodes by.
///
/// # Panics
///
/// When `published` is shorter than `layout`'s table of a `[rows, columns]` matrix.
pub fn linear(published: &[u8], rows: u64, columns: u64, layout: ScaleLayout) -> Vec<u8> {
    let blocks = columns.div_ceil(u64::from(BLOCK));
    let [first, second] = layout.shape(rows, columns);
    let bytes = first
        * second
        * if layout == ScaleLayout::PackedMnMajor {
            4
        } else {
            1
        };
    assert!(
        published.len() as u64 >= bytes,
        "a {layout} table of {bytes} bytes"
    );

    let mut table = Vec::with_capacity((rows * blocks) as usize);
    for row in 0..rows {
        for block in 0..blocks {
            let index = match layout {
                ScaleLayout::Linear => row * blocks + block,
                ScaleLayout::Swizzled128X4 => {
                    (block % 4)
                        + (block / 4) * 512
                        + (row % 32) * 16
                        + ((row % 128) / 32) * 4
                        + (row / 128) * (128 * second)
                }
                ScaleLayout::PackedMnMajor => {
                    ((block / WORD_SCALES) * second + row) * WORD_SCALES + block % WORD_SCALES
                }
            };
            table.push(published[index as usize]);
        }
    }
    table
}

/// The row classes `mxfp8_gemm`'s tile is chosen for: powers of two from a decode step's rows to
/// a long prefill's.
pub const ROW_CLASSES: [u32; 11] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];

/// The row class of a product over `rows` activation rows: the smallest class that holds them, or
/// the largest for more. The class chooses the tile, never the rows computed.
pub fn row_class(rows: u64) -> u32 {
    ROW_CLASSES
        .iter()
        .copied()
        .find(|class| u64::from(*class) >= rows)
        .unwrap_or(ROW_CLASSES[ROW_CLASSES.len() - 1])
}

/// One compilation of `mxfp8_quantize`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuantizeDescriptor {
    /// Element type of the activations: BF16 or FP16.
    pub input: DType,
    /// Elements per row; whole blocks, and whole packed words for [`ScaleLayout::PackedMnMajor`].
    pub columns: u32,
    /// How the scales are written: [`ScaleLayout::Linear`] or [`ScaleLayout::PackedMnMajor`].
    pub layout: ScaleLayout,
    /// Elements of a row one program encodes: a power of two dividing the row, whole words.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl QuantizeDescriptor {
    /// The compilation that encodes rows of `columns` `input` elements with scales in `layout`.
    ///
    /// A program encodes a chunk of the operators' block ([`descriptor::BLOCK`]) of a row, halved
    /// until it divides the row.
    pub fn new(input: DType, columns: u32, layout: ScaleLayout) -> Self {
        let mut block = descriptor::BLOCK;
        while block > WORD_ELEMENTS as u32 && !columns.is_multiple_of(block) {
            block /= 2;
        }

        Self {
            input,
            columns,
            layout,
            block,
            pdl: PDL,
        }
    }
}

impl fmt::Display for QuantizeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("input", abi::name(self.input))
            .field("columns", self.columns)
            .field("layout", self.layout)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `mxfp8_quantize`.
#[derive(Clone)]
pub struct QuantizeArgs {
    /// `[rows, columns]`, unit stride on the last axis, rows any stride apart: the activations.
    pub input: Operand,
    /// `[rows, columns]` E4M3, contiguous: the codes.
    pub codes: Operand,
    /// The scales, the descriptor's layout's extents and element type.
    pub scales: Operand,
}

/// Encoding activations into MXFP8; see the module documentation.
pub struct Mxfp8Quantize;

impl Op for Mxfp8Quantize {
    const NAME: &'static str = proto::module::operator::MXFP8_QUANTIZE;

    type Descriptor = QuantizeDescriptor;
    type Args<'a> = QuantizeArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if !matches!(descriptor.input, DType::BF16 | DType::F16) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} activations; MXFP8 encodes BF16 or FP16",
                    abi::name(descriptor.input)
                ),
            ));
        }
        if descriptor.layout == ScaleLayout::Swizzled128X4 {
            return Err(Error::argument(
                Self::NAME,
                "the encoder writes linear or packed scales; the swizzled atom is a product's \
                 epilogue's",
            ));
        }
        let columns = u64::from(descriptor.columns);
        let block = u64::from(descriptor.block);
        if columns == 0
            || !columns.is_multiple_of(WORD_ELEMENTS)
            || !block.is_power_of_two()
            || !block.is_multiple_of(WORD_ELEMENTS)
            || !columns.is_multiple_of(block)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "rows of {columns} elements in programs of {block}: a row is whole packed \
                     words of {WORD_ELEMENTS} and a program a power of two of them that divides it"
                ),
            ));
        }

        args.input.require_rank(Self::NAME, 2)?;
        let rows = args.input.dim(0);
        args.input
            .require_row(Self::NAME, descriptor.input, columns)?;
        args.codes.require_shape(Self::NAME, &[rows, columns])?;
        args.codes.require_row(Self::NAME, DType::F8E4M3, columns)?;
        if !args.codes.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the codes are not contiguous".to_owned(),
            ));
        }
        let shape = descriptor.layout.shape(rows, columns);
        args.scales.require_shape(Self::NAME, &shape)?;
        args.scales
            .require_row(Self::NAME, descriptor.layout.dtype(), shape[1])?;
        if !args.scales.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the scales are not contiguous".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.input.dim(0);
        let columns = u64::from(descriptor.columns);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("input_ptr", &args.input)
            .view("codes_ptr", &args.codes)
            .view("scales_ptr", &args.scales)
            .u64("rows", rows)
            .u64("input_row_stride", args.input.stride(0));

        // One row a grid row, its programs across the grid's columns.
        kernel.launch(
            LaunchGrid::new([rows, columns / u64::from(descriptor.block), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        QuantizeArgs {
            codes: twins.shadow_of(args.codes),
            scales: twins.shadow_of(args.scales),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Every step of the encoding is exact or one rounding to nearest (module documentation):
        // two implementations write the same codes and the same scales.
        Tolerance::exact()
    }
}

/// One compilation of `mxfp8_gemm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GemmDescriptor {
    /// Element type of the product: BF16.
    pub output: DType,
    /// Output channels: the weight's rows.
    pub n: u32,
    /// The contraction's elements: whole packed scale words.
    pub k: u32,
    /// The row class the tile is chosen for ([`row_class`]); the module computes any row count.
    pub rows: u32,
}

impl GemmDescriptor {
    /// The layout of both operands' scales the product reads.
    pub const SCALE_LAYOUT: ScaleLayout = ScaleLayout::PackedMnMajor;

    /// The compilation of a product over `rows` encoded activation rows and an `[n, k]` MXFP8
    /// weight into BF16.
    pub fn new(rows: u64, n: u32, k: u32) -> Self {
        Self {
            output: DType::BF16,
            n,
            k,
            rows: row_class(rows),
        }
    }
}

impl fmt::Display for GemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("output", abi::name(self.output))
            .field("n", self.n)
            .field("k", self.k)
            .field("rows", self.rows)
            .finish(formatter)
    }
}

/// The arguments of one `mxfp8_gemm`.
#[derive(Clone)]
pub struct GemmArgs {
    /// `[m, k]` E4M3, contiguous: the encoded activations.
    pub input: Operand,
    /// The activations' scales in [`GemmDescriptor::SCALE_LAYOUT`] over `[m, k]`.
    pub input_scales: Operand,
    /// `[n, k]` E4M3, contiguous: the weight, one row per output channel.
    pub weight: Operand,
    /// The weight's scales in [`GemmDescriptor::SCALE_LAYOUT`] over `[n, k]`.
    pub weight_scales: Operand,
    /// `[m, n]`, contiguous: the product.
    pub out: Operand,
}

/// The call structure `mxfp8_gemm`'s entry takes, declared to the package as
/// `uniserve_mxfp8_gemm_call` (`uniserve_mxfp8_gemm.h`, generated with its layout asserted:
/// [`abi`]).
///
/// Five device addresses, the problem's extents, the scale tables' column strides in words, and
/// the multiprocessors of the device the launch runs on, which the persistent grid covers.
#[repr(C)]
#[derive(abi::Layout)]
pub struct Call {
    /// `[m, k]` E4M3 codes, contiguous.
    pub input: *const c_void,
    /// `[k / 128, padded m]` U32: the activations' packed scale words.
    pub input_scales: *const c_void,
    /// `[n, k]` E4M3 codes, contiguous.
    pub weight: *const c_void,
    /// `[k / 128, padded n]` U32: the weight's packed scale words.
    pub weight_scales: *const c_void,
    /// `[m, n]` BF16: the product.
    pub out: *mut c_void,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    /// Words between consecutive scale columns of the activations: `m` padded to four.
    pub input_scale_stride: i64,
    /// Words between consecutive scale columns of the weight: `n` padded to four.
    pub weight_scale_stride: i64,
    /// The multiprocessors of the device the launch runs on: the persistent grid's extent.
    pub multiprocessors: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The dense MXFP8 product; see the module documentation.
pub struct Mxfp8Gemm;

impl Op for Mxfp8Gemm {
    const NAME: &'static str = proto::module::operator::MXFP8_GEMM;

    type Descriptor = GemmDescriptor;
    type Args<'a> = GemmArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.output != DType::BF16 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a {} product; the package writes BF16",
                    abi::name(descriptor.output)
                ),
            ));
        }
        let (n, k) = (u64::from(descriptor.n), u64::from(descriptor.k));
        // The kernel loads 128 elements of the contraction a step, one packed scale word, and
        // stores output rows of whole 16-byte vectors.
        if k == 0 || !k.is_multiple_of(WORD_ELEMENTS) || n == 0 || !n.is_multiple_of(16) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a [{n}, {k}] weight: the contraction is whole steps of {WORD_ELEMENTS} \
                     elements and the output channels whole vectors of 16"
                ),
            ));
        }

        args.input.require_rank(Self::NAME, 2)?;
        let m = args.input.dim(0);
        for (name, operand, rows) in [("input", &args.input, m), ("weight", &args.weight, n)] {
            operand.require_shape(Self::NAME, &[rows, k])?;
            operand.require_row(Self::NAME, DType::F8E4M3, k)?;
            if !operand.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} is not contiguous"),
                ));
            }
        }
        for (operand, rows) in [(&args.input_scales, m), (&args.weight_scales, n)] {
            let shape = GemmDescriptor::SCALE_LAYOUT.shape(rows, k);
            operand.require_shape(Self::NAME, &shape)?;
            operand.require_row(Self::NAME, DType::U32, shape[1])?;
        }
        args.out.require_shape(Self::NAME, &[m, n])?;
        args.out.require_row(Self::NAME, descriptor.output, n)?;
        if !args.out.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the output is not contiguous".to_owned(),
            ));
        }
        // The kernel's tensor maps count rows and the contraction in 32 bits.
        if m.max(n).max(k) > u64::from(u32::MAX) {
            return Err(Error::argument(
                Self::NAME,
                format!("a [{m}, {n}, {k}] product past the kernel's 32-bit extents"),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let address = |operand: &Operand| operand.ptr.address() as usize as *const c_void;
        let m = args.input.dim(0);

        let mut call = Call {
            input: address(&args.input),
            input_scales: address(&args.input_scales),
            weight: address(&args.weight),
            weight_scales: address(&args.weight_scales),
            out: args.out.ptr.address() as usize as *mut c_void,
            m: m as i64,
            n: i64::from(descriptor.n),
            k: i64::from(descriptor.k),
            input_scale_stride: args.input_scales.stride(0) as i64,
            weight_scale_stride: args.weight_scales.stride(0) as i64,
            multiprocessors: i64::from(stream.device().multiprocessors()?),
            error: [0; MESSAGE],
        };
        // SAFETY: `Call` is the structure this package's entry declares, generated from this
        // definition with its layout asserted, and the entry the package resolved is
        // `uniserve_mxfp8_gemm`.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        GemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The products of two codes under power-of-two scales are exact in FP32, so two
        // implementations differ by the association of the `k` products and the one rounding
        // into the output; a matrix product's terms take either sign.
        Tolerance::dot(u64::from(descriptor.k), DType::F32, descriptor.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::DevicePtr;

    /// A scale byte is its power of two; the subnormal byte 0 and the not-a-number byte 255 are
    /// the format's own.
    #[test]
    fn a_scale_byte_is_a_power_of_two() {
        assert_eq!(ue8m0(127), 1.0);
        assert_eq!(ue8m0(128), 2.0);
        assert_eq!(ue8m0(120), 2.0_f32.powi(-7));
        assert_eq!(ue8m0(254), 2.0_f32.powi(127));
        assert_eq!(ue8m0(0), 2.0_f32.powi(-127));
        assert!(ue8m0(NOT_A_NUMBER).is_nan());
    }

    /// The covering scale is the least power of two at or above `amax / 448`, with a zero block
    /// under the floor's, and every code of a block fits E4M3 under it.
    #[test]
    fn a_blocks_covering_scale_is_the_least_power_of_two_its_codes_fit_under() {
        let covering = |amax| block_exponent(amax, ScaleRule::Covering);
        // 448 is exactly E4M3's largest code at scale one; one ulp more needs scale two.
        assert_eq!(covering(448.0), 127);
        assert_eq!(covering(f32::from_bits(448.0_f32.to_bits() + 1)), 128);
        assert_eq!(covering(224.0), 126);
        assert_eq!(covering(1.0), 127 - 8);
        // A block of zeros takes the floor's scale: 10⁻⁴ / 448 lies between 2⁻²³ and 2⁻²².
        assert_eq!(covering(0.0), 127 - 22);
        assert_eq!(covering(f32::INFINITY), NOT_A_NUMBER);
        assert_eq!(covering(f32::NAN), NOT_A_NUMBER);

        for amax in [3.0e-3_f32, 0.7, 13.0, 1000.5, 3.3e4] {
            let scale = ue8m0(covering(amax));
            assert!(amax / scale <= E4M3_MAX, "{amax} under {scale}");
            assert!(
                amax / (scale / 2.0) > E4M3_MAX,
                "{amax} under {scale}: not the least"
            );
        }
    }

    /// The OCP scale is the block's largest power of two less E4M3's, so a magnitude whose
    /// significand exceeds 1.75 takes the scale below the covering one and saturates.
    #[test]
    fn a_blocks_ocp_scale_is_its_largest_exponent_less_e4m3s() {
        let ocp = |amax| block_exponent(amax, ScaleRule::Ocp);
        assert_eq!(ocp(448.0), 127);
        assert_eq!(ocp(511.9), 127);
        assert_eq!(ocp(512.0), 128);
        assert_eq!(ocp(1.0), 127 - 8);
        // 15.4 = 1.925 · 2³: the OCP scale 2⁻⁵ against the covering 2⁻⁴.
        assert_eq!(ocp(15.4), 127 - 5);
        assert_eq!(block_exponent(15.4, ScaleRule::Covering), 127 - 4);
        assert_eq!(ocp(-3.0), ocp(3.0));
        assert_eq!(ocp(0.0), 0);
        assert_eq!(ocp(f32::NAN), NOT_A_NUMBER);
    }

    /// Each layout places scale `(r, b)` where its readers look for it, and pads with zeros.
    #[test]
    fn the_layouts_place_each_scale_where_their_readers_load_it() {
        let (rows, columns) = (5_u64, 256_u64);
        let blocks = columns / u64::from(BLOCK);
        let linear: Vec<u8> = (0..rows * blocks).map(|index| index as u8 + 1).collect();

        assert_eq!(
            arranged(&linear, rows, columns, ScaleLayout::Linear),
            linear
        );

        let packed = arranged(&linear, rows, columns, ScaleLayout::PackedMnMajor);
        assert_eq!(ScaleLayout::PackedMnMajor.shape(rows, columns), [2, 8]);
        assert_eq!(packed.len(), 2 * 8 * 4);
        // Row 3's blocks 4..8 are the second group's word of row 3, byte by byte.
        let word = (8 + 3) * 4;
        assert_eq!(&packed[word..word + 4], &linear[3 * 8 + 4..3 * 8 + 8]);
        // Rows 5..8 of every group are padding.
        assert!(packed[5 * 4..8 * 4].iter().all(|byte| *byte == 0));

        let swizzled = arranged(&linear, rows, columns, ScaleLayout::Swizzled128X4);
        assert_eq!(ScaleLayout::Swizzled128X4.shape(rows, columns), [128, 8]);
        assert_eq!(swizzled.len(), 128 * 8);
        // Row 2, block 5: second column group, lane 2 of the first row group, column 1.
        assert_eq!(swizzled[512 + 2 * 16 + 1], linear[2 * 8 + 5]);
        assert_eq!(
            swizzled.iter().filter(|byte| **byte != 0).count(),
            linear.len()
        );

        // A published table reads back as the linear one it was arranged from.
        for layout in [
            ScaleLayout::Linear,
            ScaleLayout::Swizzled128X4,
            ScaleLayout::PackedMnMajor,
        ] {
            let published = arranged(&linear, rows, columns, layout);
            assert_eq!(
                super::linear(&published, rows, columns, layout),
                linear,
                "{layout}"
            );
        }
    }

    #[test]
    fn a_row_class_holds_the_rows_and_the_largest_holds_more() {
        for (rows, class) in [(1, 16), (16, 16), (17, 32), (100, 128), (4097, 8192)] {
            assert_eq!(row_class(rows), class, "{rows} rows");
        }
        assert_eq!(row_class(100_000), 16384);
    }

    fn operand(dtype: DType, dims: &[u64]) -> Operand {
        Operand::new(DevicePtr::from_address(0x1000), dtype, dims)
    }

    /// The product reads whole packed scale words of both operands, at the extents their rows
    /// take, and a contraction of a partial word is refused.
    #[test]
    fn a_product_reads_packed_scales_of_whole_words() {
        let descriptor = GemmDescriptor::new(5, 256, 512);
        assert_eq!(
            descriptor::key(&descriptor),
            "output_bf16_n_256_k_512_rows_16"
        );
        let args = |k: u64, input_scales: &[u64]| GemmArgs {
            input: operand(DType::F8E4M3, &[5, k]),
            input_scales: operand(DType::U32, input_scales),
            weight: operand(DType::F8E4M3, &[256, k]),
            weight_scales: operand(DType::U32, &[k / 128, 256]),
            out: operand(DType::BF16, &[5, 256]),
        };

        assert!(Mxfp8Gemm::check(&descriptor, &args(512, &[4, 8])).is_ok());
        // Five rows' words are padded to eight.
        assert!(Mxfp8Gemm::check(&descriptor, &args(512, &[4, 5])).is_err());
        let partial = GemmDescriptor {
            k: 544,
            ..descriptor
        };
        assert!(Mxfp8Gemm::check(&partial, &args(544, &[5, 8])).is_err());
    }

    /// The encoder writes linear or packed scales of whole words, in programs that divide the row.
    #[test]
    fn the_encoder_takes_rows_of_whole_words() {
        let descriptor = QuantizeDescriptor::new(DType::BF16, 1280, ScaleLayout::PackedMnMajor);
        assert_eq!(descriptor.block, 256);
        let args = |scales: &[u64], dtype: DType| QuantizeArgs {
            input: operand(DType::BF16, &[3, 1280]),
            codes: operand(DType::F8E4M3, &[3, 1280]),
            scales: operand(dtype, scales),
        };
        assert!(Mxfp8Quantize::check(&descriptor, &args(&[10, 4], DType::U32)).is_ok());
        assert!(Mxfp8Quantize::check(&descriptor, &args(&[3, 40], DType::U8)).is_err());

        let linear = QuantizeDescriptor {
            layout: ScaleLayout::Linear,
            ..descriptor
        };
        assert!(Mxfp8Quantize::check(&linear, &args(&[3, 40], DType::U8)).is_ok());
        let swizzled = QuantizeDescriptor {
            layout: ScaleLayout::Swizzled128X4,
            ..descriptor
        };
        assert!(Mxfp8Quantize::check(&swizzled, &args(&[128, 40], DType::U8)).is_err());
    }
}
