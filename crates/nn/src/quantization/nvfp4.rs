//! Calibrated NVFP4 weights: the names an export spells their fields with, the arrangements the
//! products read them in, and the loading of one matrix and of a stack of experts.
//!
//! A calibrated checkpoint already holds the encoded fields of each projection (packed E2M1
//! values, K16 block scales, a tensor scale and the calibration's activation scale), so nothing
//! is quantized at load: the fields are published through retained storage under keys naming
//! their layouts, permuted once into the arrangement the deployment's products read.

use std::borrow::Cow;
use std::fmt;

use ops::fp4::ScaleLayout;
use uniserve_tensor::{DType, Shape, Tensor};

use crate::error::{Error, Result};
use crate::weights::{
    WeightKey, WeightLayout, WeightLoader, WeightSource, dim, extents, last, numel, stored_bytes,
};

/// The suffix one export spells each field of a calibrated NVFP4 projection with.
///
/// A calibrated checkpoint states a projection's fields as four tensor names that differ only in
/// their suffix, and the suffix set is a property of the export, not of the format: two
/// calibration runs of the same model written by different toolchains name the same three
/// numbers differently. The loader is told which vocabulary its checkpoint speaks, because
/// reading it as the other one would either fail or accept a half-converted checkpoint whose
/// plain `weight` is a BF16 matrix beside a `weight_scale` nothing normalizes.
///
/// The two constructors are the export shapes this tree meets. **`weight_scale` is the same name
/// in both and means the same thing**, the K16 block scales, so the two disagree about the
/// packed values and the per-tensor scale only. The per-tensor scale is `weight_scale_2` in the
/// vLLM spelling and `weight_tensor_scale` in the diffusers one, and an export that states no
/// activation scale says so by having no tensor under [`Nvfp4Names::input_scale`], which the
/// loader reports as this vocabulary and not as a missing name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Nvfp4Names {
    /// What this vocabulary is called, in refusals: the export that writes it.
    pub export: &'static str,
    /// The packed E2M1 values, two codes a byte.
    pub values: &'static str,
    /// The K16 block scales, one byte each in `F8_E4M3` or `U8`.
    pub block_scale: &'static str,
    /// The per-tensor scale the block scales are normalized against.
    pub tensor_scale: &'static str,
    /// The calibration's activation scale, when the export states one.
    pub input_scale: &'static str,
}

impl Nvfp4Names {
    /// The vLLM/ModelOpt spelling: `weight`, `weight_scale`, `weight_scale_2`, `input_scale`.
    ///
    /// This is what a ModelOpt export for vLLM states.
    pub const fn vllm() -> Self {
        Self {
            export: "vLLM ModelOpt",
            values: "weight",
            block_scale: "weight_scale",
            tensor_scale: "weight_scale_2",
            input_scale: "input_scale",
        }
    }

    /// The diffusers/ModelOpt spelling: `weight_packed`, `weight_scale`, `weight_tensor_scale`,
    /// `input_scale`.
    ///
    /// This is what a multi-stage diffusers NVFP4 export states, in every stage it quantizes
    /// (a `transformer` and a `vae`, say), and a stage may add `scale1`/`scale2` beside them for
    /// the layers ModelOpt calibrated in a second scheme.
    pub const fn diffusers() -> Self {
        Self {
            export: "diffusers ModelOpt",
            values: "weight_packed",
            block_scale: "weight_scale",
            tensor_scale: "weight_tensor_scale",
            input_scale: "input_scale",
        }
    }
}

impl fmt::Display for Nvfp4Names {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} (`{}`, `{}`, `{}`)",
            self.export, self.values, self.block_scale, self.tensor_scale
        )
    }
}

/// The arrangement of a calibrated matrix's packed E2M1 values.
///
/// The values are two codes a byte along the contraction whichever arrangement they are in; what
/// varies is the order of the **rows**, because the trtllm-gen products read their A operand
/// with `useShuffledMatrix`: rows permuted inside blocks of sixteen or thirty-two so that the
/// tile a thread block loads is contiguous. It is a permutation of the shard's own bytes, applied
/// once at load, never in a forward and never by a kernel, and a matrix whose row count is not a
/// whole number of blocks has no such arrangement, which the loader refuses by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueLayout {
    /// The shard's own row order.
    Plain,
    /// `shuffleMatrixA`'s row order, for a product whose epilogue tile is `epilogue_tile_m` rows.
    Shuffled {
        /// Rows one MMA tile of the product covers; the shuffle block is thirty-two rows when
        /// this is a whole number of 128s and sixteen otherwise, as the reference's
        /// `get_shuffle_block_size` states.
        epilogue_tile_m: u32,
    },
    /// A gated product's row order: the matrix's two row halves interleave, so row `i` of the
    /// first half sits at `2i` and row `i` of the second half at `2i + 1`
    /// (`reorderRowsForGatedActGemm`), and the result shuffles as [`Self::Shuffled`] does. The
    /// product's epilogue then reads the linear and the gate row of one output channel side by
    /// side.
    InterleavedShuffled {
        /// Rows one MMA tile of the product covers, as [`Self::Shuffled`] states it.
        epilogue_tile_m: u32,
    },
    /// A fused gated product's row order: thirty-two-row block `b` holds rows `[16b, 16b + 16)`
    /// of the matrix's second half, then the same rows of its first half.
    GateFirst,
}

impl ValueLayout {
    /// The layout name a weight published in this arrangement is filed under.
    pub fn layout(self) -> WeightLayout {
        match self {
            Self::Plain => WeightLayout::NVFP4_VALUES,
            Self::Shuffled { .. } => WeightLayout::NVFP4_VALUES_SHUFFLED,
            Self::InterleavedShuffled { .. } => WeightLayout::NVFP4_VALUES_INTERLEAVED_SHUFFLED,
            Self::GateFirst => WeightLayout::NVFP4_VALUES_GATE_FIRST,
        }
    }

    /// The rows one shuffle block holds, or `None` when the arrangement is plain.
    pub fn block(self) -> Option<u64> {
        match self {
            Self::Plain => None,
            Self::GateFirst => Some(32),
            Self::Shuffled { epilogue_tile_m } | Self::InterleavedShuffled { epilogue_tile_m } => {
                Some(if epilogue_tile_m % 128 == 0 { 32 } else { 16 })
            }
        }
    }

    /// `bytes` with every expert's rows moved into this arrangement.
    ///
    /// `bytes` holds `experts` packed matrices of `rows` rows by `row_bytes` bytes, one after
    /// another; a plain arrangement is the bytes unchanged.
    fn arrange(self, bytes: &[u8], experts: u64, rows: u64, row_bytes: u64) -> Vec<u8> {
        let Some(block) = self.block() else {
            return bytes.to_vec();
        };
        let mut arranged = bytes.to_vec();
        let stride = row_bytes as usize;

        if matches!(self, Self::GateFirst) {
            for expert in 0..experts {
                let base = expert * rows * row_bytes;
                for position in 0..rows {
                    let (group, within) = (position / 32, position % 32);
                    let half = if within < 16 { rows / 2 } else { 0 };
                    let source = (base + (group * 16 + within % 16 + half) * row_bytes) as usize;
                    let target = (base + position * row_bytes) as usize;

                    arranged[target..target + stride]
                        .copy_from_slice(&bytes[source..source + stride]);
                }
            }
            return arranged;
        }

        let lanes = block / 8;
        for expert in 0..experts {
            let base = expert * rows * row_bytes;
            for row in 0..rows {
                let interleaved = match self {
                    Self::InterleavedShuffled { .. } if row < rows / 2 => 2 * row,
                    Self::InterleavedShuffled { .. } => 2 * (row - rows / 2) + 1,
                    _ => row,
                };

                // The reference's maps in closed form: within its block, a source row `r` moves
                // to `(r % lanes) * 8 + r / lanes`, the interleave of `lanes` eight-row groups
                // that the artifact's `srcToDstBlk16RowMap` and `srcToDstBlk32RowMap` list. The
                // block is taken first: a row's offset inside its own block decides where it
                // goes, and its block decides where that block starts.
                let within = interleaved % block;
                let destination =
                    (interleaved / block) * block + (within % lanes) * 8 + within / lanes;
                let source = (base + row * row_bytes) as usize;
                let target = (base + destination * row_bytes) as usize;

                arranged[target..target + stride].copy_from_slice(&bytes[source..source + stride]);
            }
        }
        arranged
    }
}

/// One expert's linear block scales in the 128x4 layout the products read.
///
/// The launcher's own `computeSFIndex` (TensorRT-LLM's `fp8Op.h`) written out: rows padded to
/// 128, scale columns padded to 4, and each `[128, 4]` tile laid out so that a thread block's
/// scales are contiguous: column groups of four stride 512, thirty-two-row groups stride 16,
/// groups of 128 rows stride `128 · padded_columns`. The padding is zero, which is a scale of
/// zero on a block that does not exist.
fn swizzled_scales(linear: &[u8], rows: u64, blocks: u64) -> Vec<u8> {
    let padded_rows = rows.div_ceil(128) * 128;
    let padded_columns = blocks.div_ceil(4) * 4;
    let mut swizzled = vec![0_u8; (padded_rows * padded_columns) as usize];

    for row in 0..rows {
        for column in 0..blocks {
            let index = (column % 4)
                + (column / 4) * 512
                + (row % 32) * 16
                + ((row % 128) / 32) * 4
                + (row / 128) * (128 * padded_columns);
            swizzled[index as usize] = linear[(row * blocks + column) as usize];
        }
    }
    swizzled
}

/// A calibrated NVFP4 matrix, as the `nvfp4_gemm` operator reads it.
///
/// The two tensors are the encoded fields of the checkpoint, retained like any other weight;
/// the global scale is a plain value because it enters a kernel as a launch scalar. The loader
/// keeps the invariant between them: the block scales are those of the packed matrix's rows and
/// columns, in [`Self::layout`].
#[derive(Clone, Debug)]
pub struct Nvfp4Weight {
    /// `[rows, columns / 2]` U8: two E2M1 codes a byte, packed along the contraction.
    pub packed: Tensor,
    /// The block scales in [`Self::layout`], `ScaleLayout::shape(rows, columns)` elements of the
    /// one-byte type the checkpoint stores them in: one E4M3 scale per sixteen elements.
    pub scales: Tensor,
    /// The checkpoint's FP32 tensor scale for this projection: the value, not its reciprocal.
    ///
    /// It multiplies: the operand's reconstruction is `codes · block scale · token scale`, which
    /// is the operand in real units, so the weight's tensor scale has to be applied exactly once
    /// and as a factor.
    pub global_scale: f32,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
    /// The reciprocal of the checkpoint's activation tensor scale, `[1]` F32, when the caller
    /// asked for it.
    ///
    /// A calibrated export states a static per-module `input_scale` for its activations, and an
    /// encoder needs it as a **device tensor**: the product that consumes the encoding reads it
    /// as each thread block starts, so a host scalar would serialize the pair. It is calibration
    /// data of one tensor, the same for every request, so it lives with the weights. `None`
    /// says the consumer is weight-only and asked for none, which is a different fact from a
    /// calibrated checkpoint that omitted it: that one is refused by tensor name.
    pub input_scale: Option<Tensor>,
}

/// A calibrated NVFP4 **stack of experts**, as `gather_gemm_nvfp4` and `route_gemm_nvfp4` read
/// it.
///
/// A model's expert block multiplies whole stacks: its weights are `[experts, rows, columns/2]`
/// U8, its block scales hold one entry per expert, and the tensor scale varies per expert **and
/// per projection**, so the scalar [`Nvfp4Weight::global_scale`] cannot serve it. The loader
/// publishes all of them from the checkpoint's per-expert, per-projection tensors: the encoded
/// values and scales concatenated in the operator's reading order, and the tensor scales as a
/// `[projections, experts]` F32 weight, one packed table per projection, which is the shape the
/// batched-GEMM entry reads and the shape the checkpoint itself holds a tensor per.
///
/// The stack fused from an expert's `up_proj` and `gate_proj` is why the second axis exists:
/// each of those has its own `weight_scale_2` in the checkpoint, so one scalar per expert would
/// dequantize the gate half with the up half's scale and be wrong by the ratio of the two.
#[derive(Clone, Debug)]
pub struct Nvfp4Stack {
    /// `[experts, rows, columns/2]` U8: two E2M1 codes a byte, packed along the contraction.
    pub values: Tensor,
    /// `[experts, ...]` in the checkpoint's scale type (`F8_E4M3`, or `U8` where the export
    /// labels the same bytes as raw): the block scales in [`Self::layout`], one per sixteen
    /// elements. Every expert of a stack agrees on the type, and it is published unchanged: a
    /// module puts this tensor's element type in the product's descriptor and nothing converts.
    pub scales: Tensor,
    /// `[projections, experts]` F32, row-major, in the stack's own projection order: each
    /// expert's **per-projection** checkpoint tensor scale, which is the factor the product
    /// multiplies by.
    ///
    /// One packed table per projection and not one row per expert, because that is what the
    /// batched-GEMM entry takes: it reads each projection's table as `experts` consecutive
    /// floats.
    pub global_scale: Tensor,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
    /// `[1]` F32: the activation encoder's global scale for this stack's activations, the
    /// reciprocal of the widest calibrated activation scale among its experts. The encoder
    /// writes that activation scale as each row's factor, and the product multiplies it in, so
    /// the table [`Self::global_scale`] holds the weights' tensor scales alone.
    pub encode_scale: Tensor,
    /// `[projections, experts]` F32, row-major, in the same order: the reciprocal of each
    /// expert's **per-projection** activation tensor scale, when the caller asked for it.
    ///
    /// A ModelOpt MoE export calibrates each expert's projections separately, so one activation
    /// scale per stack would encode every expert but one at the wrong magnitude. `None` says the
    /// consumer is weight-only; a calibrated checkpoint asked for this and missing it is refused
    /// by tensor name.
    pub input_scale: Option<Tensor>,
    /// The activation scale [`Self::encode_scale`] is the reciprocal of.
    pub activation_scale: f32,
}

/// Whether `dtype` is one of the two element types a block-scale field may have.
///
/// The block scales are eight bits per sixteen elements, and a checkpoint stores each as an E4M3
/// float or, where an export labels the same bytes as raw, as `U8`. Both are one byte, both are
/// published unchanged, and nothing converts: a reader that multiplied a scale by a constant
/// would be wrong, and a loader that relabeled it would hide which one the checkpoint meant.
fn block_scale_dtype(tensor: &str, dtype: DType) -> Result<DType> {
    match dtype {
        DType::U8 | DType::F8E4M3 => Ok(dtype),
        other => Err(Error::weight(
            tensor,
            format!(
                "block scales are one byte each (F8_E4M3, or U8 where the export says so), not \
                 {}",
                ops::abi::name(other)
            ),
        )),
    }
}

/// `scales`, collected expert-major as `per` scales per expert, as the little-endian F32 bytes
/// of the projection-major `[per, experts]` table: one packed table per projection, the axis
/// order the batched-GEMM entry reads and the shape a checkpoint holds a tensor for.
fn projection_major(scales: &[f32], per: usize, experts: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(per * experts * 4);

    for projection in 0..per {
        for expert in 0..experts {
            bytes.extend_from_slice(&scales[expert * per + projection].to_le_bytes());
        }
    }
    bytes
}

/// The layout name block scales in `layout` are filed under.
fn scales_layout(layout: ScaleLayout) -> WeightLayout {
    match layout {
        ScaleLayout::Linear => WeightLayout::NVFP4_SCALES_LINEAR,
        ScaleLayout::Swizzled128X4 => WeightLayout::NVFP4_SCALES_SWIZZLED,
    }
}

/// The key of a table this loader computes from the checkpoint tensors `fields` of `files`: a
/// row-major region of `shape` elements of `dtype` named by `tensors`, in `layout`.
fn computed_key(
    files: Vec<crate::checkpoint::FileIdentity>,
    dtype: DType,
    layout: WeightLayout,
    tensors: Vec<String>,
    shape: &Shape,
) -> WeightKey {
    WeightKey {
        files,
        dtype,
        layout,
        tensors,
        start: 0,
        shape: shape.clone(),
        stride: None,
        provenance: None,
    }
}

/// The tensor `source` reads, when it reads one: a region or the strided rows of it.
fn single(source: &WeightSource) -> Option<&str> {
    match source {
        WeightSource::Region { tensor, .. } | WeightSource::Strided { tensor, .. } => Some(tensor),
        WeightSource::Parts { .. } | WeightSource::Blocks { .. } => None,
    }
}

/// The extents `source` gathers.
fn gathered(source: &WeightSource) -> &Shape {
    match source {
        WeightSource::Region { shape, .. }
        | WeightSource::Strided { shape, .. }
        | WeightSource::Parts { shape, .. }
        | WeightSource::Blocks { shape, .. } => shape,
    }
}

impl WeightLoader<'_> {
    /// Loads a calibrated NVFP4 matrix: its packed E2M1 values, its block scales and its tensor
    /// scale.
    ///
    /// `values` names the packed matrix, `[rows, columns / 2]` U8 (two E2M1 codes a byte), or
    /// this rank's block of it, and `scales` its block scales in the arrangement the checkpoint
    /// stores them in, in whatever one-byte type the checkpoint stores them; both name one
    /// checkpoint tensor each and both are asked to hold a value. `input_scale`, when given, is
    /// the tensor holding the calibration's activation scale, whose reciprocal is published as
    /// the encoder's operand. `names` is the export's spelling, which a refusal of a missing
    /// field names.
    ///
    /// `layouts` is `(file, product)`: the arrangement the checkpoint stores its block scales in
    /// and the one the deployment's kernels read. They are stated apart so that one cannot be
    /// passed for the other; when they differ the table is permuted here, once, and
    /// [`Nvfp4Weight::layout`] states the product's. `global_scale` is the checkpoint's FP32
    /// tensor scale of the weight, which the caller read ([`Self::tensor_scale`]).
    ///
    /// # Errors
    ///
    /// When a field is missing, of another element type or extent than the matrix takes, every
    /// packed byte or every block scale is zero, the scales are stored swizzled and asked
    /// linear, or `global_scale` is not a positive number.
    pub fn nvfp4(
        &self,
        values: &WeightSource,
        scales: &WeightSource,
        input_scale: Option<&str>,
        names: Nvfp4Names,
        layouts: (ScaleLayout, ScaleLayout),
        global_scale: f32,
    ) -> Result<Nvfp4Weight> {
        let (file, product) = layouts;
        let (Some(values_name), Some(scales_name)) = (single(values), single(scales)) else {
            return Err(Error::weight(
                &crate::weights::source_names(values).join("+"),
                "a calibrated NVFP4 matrix is one checkpoint tensor and its block scales another"
                    .to_owned(),
            ));
        };

        // The sources name the fields, so their absence is where a wrong vocabulary shows up: a
        // checkpoint opened as the other export's spelling has a `weight_packed` where this
        // asked for a `weight`, and the refusal says which spelling was expected and why.
        self.check_nvfp4_field(values_name, names, "packed E2M1 values", names.values)?;
        self.check_nvfp4_field(scales_name, names, "K16 block scales", names.block_scale)?;

        if !(global_scale.is_finite() && global_scale > 0.0) {
            return Err(Error::weight(
                values_name,
                format!("a global scale of {global_scale} is not a positive number"),
            ));
        }

        let values_shape = gathered(values);
        if last(values_shape) == 0 {
            return Err(Error::weight(
                values_name,
                "a packed NVFP4 matrix has a contraction axis".to_owned(),
            ));
        }
        let rows = numel(values_shape) / last(values_shape);
        let columns = last(values_shape) * 2;

        let expected = file.shape(rows, columns);
        let scales_shape = gathered(scales);
        let stated: Vec<u64> = scales_shape.dims().iter().map(|dim| *dim as u64).collect();
        if stated != expected.dims() {
            return Err(Error::weight(
                scales_name,
                format!(
                    "the {file} block scales of a [{rows}, {columns}] matrix are {:?}, not {:?}",
                    expected.dims(),
                    scales_shape.dims()
                ),
            ));
        }
        if file == ScaleLayout::Swizzled128X4 && product == ScaleLayout::Linear {
            return Err(Error::weight(
                scales_name,
                "a swizzled file rewritten into the linear table is a permutation this loader \
                 does not have: a calibrated export stores its block scales linear"
                    .to_owned(),
            ));
        }

        let packed = self.load(
            values_name,
            values,
            WeightLayout::NVFP4_VALUES,
            &extents(&[rows, columns / 2]),
            None,
            true,
        )?;
        if packed.dtype() != DType::U8 {
            return Err(Error::weight(
                values_name,
                format!(
                    "packed E2M1 values are U8 bytes, not {}",
                    ops::abi::name(packed.dtype())
                ),
            ));
        }

        // A calibrated export never packs an all-zero matrix: every E2M1 code would be zero and
        // the product would give zeros for every activation. One host read of a few megabytes
        // per projection, at load only, says so by name.
        self.check_values(|checkpoint| {
            if checkpoint.read(values_name)?.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    values_name,
                    "every packed E2M1 byte is zero: the product would give zeros for every \
                     activation, so this tensor's calibration did not run"
                        .to_owned(),
                ));
            }

            // The block scales are the second way a product gives zeros for every activation:
            // an E4M3 scale of zero multiplies its whole block to zero whatever the codes are.
            // One zero scale is legitimate (a block whose values are all zero); every scale
            // zero is a calibration that did not run, and it is invisible in the values.
            if checkpoint.read(scales_name)?.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    scales_name,
                    "every block scale is zero: the product would give zeros for every \
                     activation whatever the codes hold, so this tensor's calibration did not run"
                        .to_owned(),
                ));
            }
            Ok(())
        })?;

        // The file's arrangement and the product's are two statements, and the permutation
        // between them happens here once: a kernel must never read a table under a claim it
        // does not meet.
        let product_layout = scales_layout(product);
        let scales_tensor = if file == product {
            let shape = extents(expected.dims());
            let loaded = self.load(scales_name, scales, product_layout, &shape, None, true)?;

            block_scale_dtype(scales_name, loaded.dtype())?;
            loaded
        } else {
            self.swizzled(scales_name, scales, (rows, columns), (file, product))?
        };

        let input_scale = match input_scale {
            None => None,
            Some(tensor) => {
                self.check_nvfp4_field(
                    tensor,
                    names,
                    "calibration's activation scale",
                    names.input_scale,
                )?;

                // The encoder's factor: the reciprocal of the calibrated activation scale. It is
                // not bounded: a calibrated `input_scale` near `1e-3` is a normal calibration
                // (its factor is then about 724), and the encoder's block scale is computed
                // from the row it actually sees.
                let reciprocal = 1.0 / self.tensor_scale(tensor)?;
                let shape = Shape::from(1);
                let bytes = reciprocal.to_le_bytes();

                // The reciprocal is computed from one file: the suffix names the published
                // piece, not a tensor the checkpoint holds, so it is not an identity.
                let key = computed_key(
                    self.identities(&[tensor.to_owned()])?,
                    DType::F32,
                    WeightLayout::ROW_MAJOR,
                    vec![format!("{tensor}__nvfp4_input_scale")],
                    &shape,
                );
                Some(self.publish_computed(&key, &shape, |_| Ok(Cow::Borrowed(&bytes[..])))?)
            }
        };

        Ok(Nvfp4Weight {
            packed,
            scales: scales_tensor,
            global_scale,
            layout: product,
            input_scale,
        })
    }

    /// The block scales `scales` names of a `[rows, columns]` matrix, stored in `file`'s
    /// arrangement, published in `product`'s 128x4 arrangement.
    fn swizzled(
        &self,
        name: &str,
        scales: &WeightSource,
        (rows, columns): (u64, u64),
        (file, product): (ScaleLayout, ScaleLayout),
    ) -> Result<Tensor> {
        let location = self.checkpoint().locate(name)?;
        let dtype = block_scale_dtype(name, location.dtype)?;
        let blocks = file.columns(columns);

        // The source may name a column block of a row-parallel matrix, so the rows are gathered
        // at its stride before the permutation: permuting the file's first `rows · blocks`
        // elements instead would publish another rank's scales under this rank's key.
        let (start, stride) = match scales {
            WeightSource::Region { start, .. } => (*start, blocks),
            WeightSource::Strided {
                start, row_stride, ..
            } => (*start, *row_stride),
            WeightSource::Parts { .. } | WeightSource::Blocks { .. } => {
                return Err(Error::weight(
                    name,
                    "block scales gathered from fused parts have no single row stride".to_owned(),
                ));
            }
        };

        // Each row's byte range, checked against the header's length of the tensor: the scales
        // are one byte each, so elements and bytes are the same offsets.
        let length = usize::try_from(stored_bytes(location)).unwrap_or(usize::MAX);
        let mut ranges = Vec::with_capacity(rows as usize);
        for row in 0..rows {
            let first = start + row * stride;
            let end = usize::try_from(first + blocks).unwrap_or(usize::MAX);
            let first = usize::try_from(first).unwrap_or(usize::MAX);

            if first > end || end > length {
                return Err(Error::weight(
                    name,
                    format!(
                        "elements {first}..{end} of a row are past the tensor's {blocks}-block rows"
                    ),
                ));
            }
            ranges.push(first..end);
        }

        let shape = extents(product.shape(rows, columns).dims());
        let key = computed_key(
            self.identities(&[name.to_owned()])?,
            dtype,
            scales_layout(product),
            vec![format!("{name}__nvfp4_scales")],
            &shape,
        );
        self.publish_computed(&key, &shape, |loader| {
            let bytes = loader.checkpoint().read(name)?;
            let mut linear = Vec::with_capacity((rows * blocks) as usize);

            for range in ranges {
                linear.extend_from_slice(&bytes[range]);
            }
            Ok(Cow::Owned(swizzled_scales(&linear, rows, blocks)))
        })
    }

    /// Checks that the checkpoint holds one of a projection's NVFP4 fields, naming the export's
    /// spelling when it does not.
    ///
    /// The spelling is the whole point: the same number is `weight` in one export and
    /// `weight_packed` in another, so "not found" against a name the loader expected to be right
    /// hides which export the checkpoint came from. A checkpoint opened with one vocabulary and
    /// written in the other is a loading-boundary error, not a corrupt checkpoint.
    fn check_nvfp4_field(
        &self,
        tensor: &str,
        names: Nvfp4Names,
        field: &str,
        suffix: &str,
    ) -> Result<()> {
        if self.checkpoint().tensor(tensor).is_some() {
            return Ok(());
        }
        Err(Error::weight(
            tensor,
            format!(
                "this checkpoint was opened as the {names} spelling of a calibrated NVFP4 \
                 projection, whose {field} is `{suffix}`, and it holds no tensor of this name; \
                 the other export's spelling names the same field differently"
            ),
        ))
    }

    /// Loads the encoded fields of a stack of experts: the experts `slots` names, in that order,
    /// of a router's `vocabulary` of experts, each contributing one projection per module of
    /// `projections` in the order the operator reads them.
    ///
    /// Expert `e`'s projection `p` is the module `<prefix>.<e>.<p>`: the model names the
    /// modules, as its checkpoint does, and the loader reads their fields in the export's
    /// spelling `names`. A stack is a concatenation: slot-major, and within a slot the
    /// projections in order (the gated product reads up first, gate second). Each field is
    /// published through retained storage under one key, so a second load of the same
    /// checkpoint costs nothing. Slot `s` holds expert `slots[s]`, so a redistribution's order
    /// is a load of the same checkpoint.
    ///
    /// `layout` is `(values, scales)`: the row arrangement the packed values are published in
    /// and the arrangement of the block scales. `input_scale` asks for each expert's
    /// per-projection activation scale ([`Nvfp4Stack::input_scale`]). The activation encoder's
    /// scale is the whole vocabulary's: a token is encoded once on the rank that holds it, for
    /// experts on any rank, so every rank encodes with the same number.
    ///
    /// The second value returned is each held expert's tensor scales, expert-major, one per
    /// projection; a meta loader gives [`Self::tensor_scale`]'s placeholder for each.
    ///
    /// `name` names the stack in keys and errors and must not collide with a tensor's: a
    /// computed weight needs its own name.
    ///
    /// # Errors
    ///
    /// When a field is missing or of another element type, the experts disagree in extents or
    /// scale type, a block-scale table is not the linear one a checkpoint stores, or the row
    /// count is not a whole number of the value arrangement's shuffle blocks.
    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is one independent fact of the stack, stated by the module that \
                  owns it"
    )]
    pub fn nvfp4_stack(
        &self,
        name: &str,
        prefix: &str,
        projections: &[&str],
        slots: &[u32],
        vocabulary: u32,
        names: Nvfp4Names,
        layout: (ValueLayout, ScaleLayout),
        input_scale: bool,
    ) -> Result<(Nvfp4Stack, Vec<f32>)> {
        let (values_layout, layout) = layout;
        let experts = u32::try_from(slots.len())
            .map_err(|_| Error::weight(name, "more slots than a stack holds".to_owned()))?;
        if experts == 0 || projections.is_empty() {
            return Err(Error::weight(
                name,
                "a stack has experts and at least one projection".to_owned(),
            ));
        }
        let count = u64::from(experts);
        let per = projections.len() as u64;

        // Every field of every expert, by the export's own suffix set. The stack's extents and
        // types come from the headers; the payload is read only when a field is published.
        let mut fields = Vec::new();
        let mut values_names = Vec::with_capacity(slots.len() * projections.len());
        let mut scales_names = Vec::with_capacity(slots.len() * projections.len());
        let mut rows = None;
        let mut columns = None;
        let mut tensor_scales = Vec::with_capacity(experts as usize);
        let mut activation_reciprocals = Vec::with_capacity(experts as usize * projections.len());
        let mut entry = 0;
        let mut scale_dtype = None;

        for expert in slots {
            for projection in projections {
                let stem = format!("{prefix}.{expert}.{projection}");
                let values_name = format!("{stem}.{}", names.values);
                let scales_name = format!("{stem}.{}", names.block_scale);
                let tensor_name = format!("{stem}.{}", names.tensor_scale);
                let activation_name = format!("{stem}.{}", names.input_scale);

                fields.push(values_name.clone());
                fields.push(scales_name.clone());
                fields.push(tensor_name.clone());
                if input_scale {
                    fields.push(activation_name.clone());
                }

                self.check_nvfp4_field(&values_name, names, "packed E2M1 values", names.values)?;
                let values = self.checkpoint().locate(&values_name)?;
                if values.dtype != DType::U8 {
                    return Err(Error::weight(
                        &values_name,
                        format!(
                            "packed E2M1 values are U8 bytes, not {}",
                            ops::abi::name(values.dtype)
                        ),
                    ));
                }
                if values.shape.rank() != 2 {
                    return Err(Error::weight(
                        &values_name,
                        format!("a packed matrix is two axes, not {:?}", values.shape.dims()),
                    ));
                }
                let (rows_here, columns_here) = (dim(&values.shape, 0), dim(&values.shape, 1) * 2);

                // The stack's extents are the first expert's; every other projection must
                // agree, because one set of tensors serves them all.
                for (seen, here, what) in [
                    (&mut rows, rows_here, "rows"),
                    (&mut columns, columns_here, "columns"),
                ] {
                    match *seen {
                        None => *seen = Some(here),
                        Some(agreed) if agreed != here => {
                            return Err(Error::weight(
                                &values_name,
                                format!("{here} {what} against the stack's {agreed}"),
                            ));
                        }
                        Some(_) => {}
                    }
                }

                let scales = self.checkpoint().locate(&scales_name)?;
                let here = block_scale_dtype(&scales_name, scales.dtype)?;
                match scale_dtype {
                    None => scale_dtype = Some(here),
                    Some(agreed) if agreed != here => {
                        return Err(Error::weight(
                            &scales_name,
                            format!(
                                "block scales are {} here against the stack's {}: one stack is \
                                 one scale type",
                                ops::abi::name(here),
                                ops::abi::name(agreed)
                            ),
                        ));
                    }
                    Some(_) => {}
                }

                // The shard's table is the checkpoint's own arrangement, one block scale per
                // sixteen elements, row-major, and the published one is the deployment's. A
                // converted export that already holds the padded, interleaved table is refused
                // here instead of published after a second pass over it.
                let expected = ScaleLayout::Linear.shape(rows_here, columns_here);
                let stated: Vec<u64> = scales.shape.dims().iter().map(|dim| *dim as u64).collect();
                if stated != expected.dims() {
                    return Err(Error::weight(
                        &scales_name,
                        format!(
                            "a checkpoint's block scales of a [{rows_here}, {columns_here}] \
                             matrix are linear {:?}, not {:?}: the published {layout} table is \
                             this loader's to build, not the export's to state",
                            expected.dims(),
                            scales.shape.dims()
                        ),
                    ));
                }

                tensor_scales.push(self.tensor_scale(&tensor_name)?);
                if input_scale {
                    activation_reciprocals.push(1.0 / self.tensor_scale(&activation_name)?);
                }

                values_names.push(values_name);
                scales_names.push(scales_name);
                entry = expected.numel();
            }
        }

        let rows = rows.unwrap_or(0) * per;
        let columns = columns.unwrap_or(0);
        let shard_scales = ScaleLayout::Linear.shape(rows, columns);
        if shard_scales.numel() != entry * per {
            return Err(Error::weight(
                name,
                format!(
                    "a linear entry for {rows} rows of {columns} elements is {} block scales, \
                     and {per} concatenated entries hold {entry} each",
                    shard_scales.numel() / per.max(1)
                ),
            ));
        }

        // The values' row arrangement: a plain shard needs nothing, a shuffled one is permuted
        // here once, at load. A product whose A operand is read with `useShuffledMatrix` cannot
        // read a shard's own row order, so a row count that is not a whole number of shuffle
        // blocks has no arrangement at all and is refused by name.
        if let Some(block) = values_layout.block()
            && !rows.is_multiple_of(block)
        {
            return Err(Error::weight(
                name,
                format!(
                    "a {values_layout:?} arrangement permutes {block}-row blocks, and this \
                     stack's matrices have {rows} rows"
                ),
            ));
        }

        let values_shape = extents(&[count, rows, columns / 2]);
        let published = layout.shape(rows, columns);
        let scales_shape = extents(&[count, published.dim(0), published.dim(1)]);

        // `tensor_scales` was collected expert-major, in the checkpoint's own reading order;
        // the published table is projection-major, one packed row per projection.
        let projections_shape = extents(&[per, count]);
        let (per_expert, held) = (projections.len(), slots.len());
        if tensor_scales.len() != per_expert * held {
            return Err(Error::weight(
                name,
                format!(
                    "{per} projections of {count} experts are {} scales, and the checkpoint gave \
                     {}",
                    per_expert * held,
                    tensor_scales.len()
                ),
            ));
        }
        let scale_bytes = projection_major(&tensor_scales, per_expert, held);
        let files = self.identities(&fields)?;

        // The tensors a field of the stack was computed from: its own name, then every field.
        let computed_from = |piece: &str| -> Vec<String> {
            std::iter::once(format!("{name}__{piece}"))
                .chain(fields.iter().cloned())
                .collect()
        };

        let values = self.publish_computed(
            &computed_key(
                files.clone(),
                DType::U8,
                values_layout.layout(),
                computed_from("nvfp4_values"),
                &values_shape,
            ),
            &values_shape,
            |loader| {
                let mut bytes = Vec::with_capacity(numel(&values_shape) as usize);
                for values_name in &values_names {
                    bytes.extend_from_slice(&loader.checkpoint().read(values_name)?);
                }

                Ok(Cow::Owned(values_layout.arrange(
                    &bytes,
                    count,
                    rows,
                    (columns / 2).max(1),
                )))
            },
        )?;

        // The stack's scale type is the checkpoint's own: an E4M3 field stays E4M3, and the
        // module puts it in the descriptor unchanged.
        let scale_dtype = scale_dtype.unwrap_or(DType::U8);
        let scales = self.publish_computed(
            &computed_key(
                files.clone(),
                scale_dtype,
                scales_layout(layout),
                computed_from("nvfp4_scales"),
                &scales_shape,
            ),
            &scales_shape,
            |loader| {
                let blocks = ScaleLayout::Linear.columns(columns);
                let mut scales_bytes = Vec::with_capacity((count * rows * blocks) as usize);
                for scales_name in &scales_names {
                    scales_bytes.extend_from_slice(&loader.checkpoint().read(scales_name)?);
                }

                // One expert's linear table becomes the published arrangement: rows permuted
                // the way the values' rows were when the values are shuffled (the scales are
                // read beside the row), then padded and interleaved when the deployment's
                // kernels read the swizzled 128x4 table.
                let mut arranged = Vec::with_capacity(numel(&scales_shape) as usize);
                for expert in 0..usize::try_from(count).unwrap_or(0) {
                    let entry = &scales_bytes[expert * (rows * blocks) as usize
                        ..(expert + 1) * (rows * blocks) as usize];
                    let entry = values_layout.arrange(entry, 1, rows, blocks);

                    match layout {
                        ScaleLayout::Linear => arranged.extend_from_slice(&entry),
                        ScaleLayout::Swizzled128X4 => {
                            arranged.extend_from_slice(&swizzled_scales(&entry, rows, blocks));
                        }
                    }
                }
                Ok(Cow::Owned(arranged))
            },
        )?;

        let global_scale = self.publish_computed(
            &computed_key(
                files,
                DType::F32,
                WeightLayout::ROW_MAJOR,
                computed_from("nvfp4_tensor_scales"),
                &projections_shape,
            ),
            &projections_shape,
            |_| Ok(Cow::Borrowed(&scale_bytes[..])),
        )?;

        let activation_scales = if input_scale {
            let bytes = projection_major(&activation_reciprocals, per_expert, held);
            let suffix = format!(".{}", names.input_scale);
            let files = self.identities(
                &fields
                    .iter()
                    .filter(|field| field.ends_with(&suffix))
                    .cloned()
                    .collect::<Vec<_>>(),
            )?;

            Some(self.publish_computed(
                &computed_key(
                    files,
                    DType::F32,
                    WeightLayout::ROW_MAJOR,
                    vec![format!("{name}__nvfp4_input_scales")],
                    &projections_shape,
                ),
                &projections_shape,
                |_| Ok(Cow::Borrowed(&bytes[..])),
            )?)
        } else {
            None
        };

        // One encoder serves the whole stack, so it takes the widest of its experts' calibrated
        // activation scales, whose reciprocal is the smallest: no expert's block scale
        // saturates.
        let mut encode = activation_reciprocals
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        if input_scale {
            for expert in (0..vocabulary).filter(|expert| !slots.contains(expert)) {
                for projection in projections {
                    let tensor = format!("{prefix}.{expert}.{projection}.{}", names.input_scale);
                    encode = encode.min(1.0 / self.tensor_scale(&tensor)?);
                }
            }
        }
        let encode = if encode.is_finite() { encode } else { 1.0 };
        let encode_scale = self.constant(
            &format!("{name}__nvfp4_encode_scale"),
            DType::F32,
            &Shape::from(1),
            &encode.to_ne_bytes(),
            // The producer's text is part of the stored key.
            "nn",
        )?;

        Ok((
            Nvfp4Stack {
                values,
                scales,
                global_scale,
                layout,
                encode_scale,
                input_scale: activation_scales,
                activation_scale: encode.recip(),
            },
            tensor_scales,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{ValueLayout, swizzled_scales};

    /// The block-scale arrangement is the launcher's `computeSFIndex`: a permutation of the
    /// checkpoint's row-major table into the 128x4 layout, padded with zeros.
    ///
    /// Three properties: the values survive exactly once each; the result is not the padded
    /// row-major table, so an arrangement that never ran fails here; and one index follows the
    /// reference's own arithmetic
    /// (`(column % 4) + (column / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 +
    /// (row / 128) * (128 * padded_columns)`), so a bijection that moved the right values to the
    /// wrong places is not mistaken for the layout the products read.
    ///
    /// The padding is a zero, so the data starts at one: a zero scale is a block the file says
    /// is empty, indistinguishable in the arranged table from a hole.
    #[test]
    fn the_block_scales_are_permuted_into_the_laid_out_table() {
        for (rows, blocks) in [(56_u64, 4_u64), (8, 8)] {
            let linear: Vec<u8> = (1..=rows * blocks).map(|value| value as u8).collect();
            let swizzled = swizzled_scales(&linear, rows, blocks);
            let padded_rows = rows.div_ceil(128) * 128;
            let padded_columns = blocks.div_ceil(4) * 4;
            assert_eq!(
                swizzled.len() as u64,
                padded_rows * padded_columns,
                "the table is padded to 128 rows and 4 columns"
            );

            let mut present: Vec<u8> = swizzled
                .iter()
                .copied()
                .filter(|value| *value != 0)
                .collect();
            let mut expected = linear.clone();
            present.sort_unstable();
            expected.sort_unstable();
            assert_eq!(present, expected, "every scale survives exactly once");

            let mut row_major = vec![0_u8; (padded_rows * padded_columns) as usize];
            for row in 0..rows {
                for column in 0..blocks {
                    row_major[(row * padded_columns + column) as usize] =
                        linear[(row * blocks + column) as usize];
                }
            }
            assert_ne!(
                swizzled, row_major,
                "an arrangement that never ran is the row-major table"
            );
        }

        // One index from the formula, at a row inside the first thirty-two and a column past
        // the first group of four: row 5, column 6 of eight lands at 2 + 512 + 80, and the data
        // starts at one, so the value there is `5 * 8 + 6 + 1`.
        let (rows, blocks) = (128_u64, 8_u64);
        let linear: Vec<u8> = (1..=rows * blocks).map(|value| value as u8).collect();
        let swizzled = swizzled_scales(&linear, rows, blocks);
        assert_eq!(swizzled[594], (5 * blocks + 6 + 1) as u8);
    }

    /// A gated stack's rows: logical row `i` of each half reaches the position the reference
    /// states, `2i` or `2i + 1` before the shuffle, then `(r % 4) · 8 + r / 4` inside its
    /// 32-row block.
    #[test]
    fn a_gated_stack_interleaves_its_halves_before_the_shuffle() {
        let rows = 256_u64;
        let bytes: Vec<u8> = (0..rows).map(|row| row as u8).collect();
        let layout = ValueLayout::InterleavedShuffled {
            epilogue_tile_m: 128,
        };

        let arranged = layout.arrange(&bytes, 1, rows, 1);
        for half in 0..2_u64 {
            for index in 0..rows / 2 {
                let position = 2 * index + half;
                let within = position % 32;
                let stored = (position / 32) * 32 + (within % 4) * 8 + within / 4;
                assert_eq!(
                    u64::from(arranged[stored as usize]),
                    half * (rows / 2) + index,
                    "row {index} of half {half} belongs at {stored}"
                );
            }
        }
    }
}
