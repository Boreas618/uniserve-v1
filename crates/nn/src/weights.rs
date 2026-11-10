//! Weights: the layout the reading operator declares, the key retained storage files them under,
//! and the loader that converts a checkpoint's tensors once.
//!
//! ## The layout
//!
//! [`WeightLayout`] is the reading operator's arrangement of the bytes, named by the operator and
//! carried with the weight. The checkpoint's own tensors are laid out per model framework (an
//! expert's gate and up projections are separate matrices, gate first); an operator that reads one
//! fused `[experts, 2·intermediate, hidden]` weight with the up projection first declares
//! [`WeightLayout::UP_GATE`] and the loader produces that arrangement from the checkpoint's parts,
//! once, before the bytes are published.
//!
//! ## The key
//!
//! Retained weights are keyed by exactly four inputs: the identity of every checkpoint file whose
//! bytes produced the weight, the element type the weight has after conversion, the layout name,
//! and the weight's region in its logical tensor. No program name or revision enters the key: a
//! program-layer update keeps every retained weight. A change of dtype, quantization, layout or
//! checkpoint therefore changes the key, and a change of an operator's program does not.
//!
//! ## Publishing
//!
//! Conversion happens here and nowhere else, and its output goes straight to retained storage
//! through [`RetainedWeights`], which the worker implements over the Storage Server and its
//! device mappings. `nn` owns no device memory: a [`Weight`] is a [`Backing`] plus the view the
//! operator reads.
//!
//! ## Meta loading
//!
//! [`WeightLoader::meta`] runs the same loading code over the same shard headers and stops where
//! bytes would be read: it answers every weight's element type and extents at a synthetic address
//! and publishes nothing, so a recording walk of a model's load needs no payload and no device.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use ops::fp4::ScaleLayout;
use ops::values::{bf16_to_f32, f16_to_f32, f32_to_f16};
use ops::{DType, Shape, TensorView};
use sha2::{Digest, Sha256};

use crate::checkpoint::{Checkpoint, FileIdentity, TensorLocation};
use crate::error::{Error, Result};
use crate::forward::ALIGNMENT;
use crate::quantization::{Declaration, Scheme};
use crate::tensor::{Backing, DevicePtr};

/// The suffix one export spells each field of a calibrated NVFP4 projection with.
///
/// A calibrated checkpoint states a projection's fields as four tensor names that differ only in
/// their suffix, and the suffix set is a property of the export, not of the format: two calibration
/// runs of the same model written by different toolchains name the same three numbers differently.
/// The loader is told which vocabulary its checkpoint speaks, because reading it as the other one
/// would either fail or — worse — accept a half-converted checkpoint whose plain `weight` is a
/// BF16 matrix beside a `weight_scale` nothing normalizes.
///
/// The two constructors are the export shapes this tree meets. **`weight_scale` is the same name in
/// both and means the same thing** — the K16 block scales — so it is not a mistake that the two
/// disagree about the packed values and the per-tensor scale only; renaming it in either
/// constructor would be. The per-tensor scale is `weight_scale_2` in the vLLM spelling and
/// `weight_tensor_scale` in the diffusers one, and an export that carries no activation scale says
/// so by having no tensor under [`Nvfp4Names::input_scale`], which the loader reports as this
/// vocabulary rather than as a missing name.
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
    /// This is what a multi-component diffusers NVFP4 export states, in every component it
    /// quantizes (e.g. a `transformer` and a `vae`), and a component may add `scale1`/`scale2`
    /// beside them for the layers ModelOpt calibrated in a second scheme.
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

/// The arrangement of a weight's bytes that its reading operator declares.
///
/// The names enter the retained key, so each one is a stable statement about layout, not about the
/// operator's program or tuning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightLayout(&'static str);

impl WeightLayout {
    /// The checkpoint's own contiguous row-major arrangement, unchanged.
    pub const ROW_MAJOR: Self = Self("row_major");

    /// A routed expert's first projection: `[experts, 2·intermediate, hidden]`, up before gate.
    pub const UP_GATE: Self = Self("up_gate");

    /// A routed expert's second projection: `[experts, hidden, intermediate]`.
    pub const DOWN: Self = Self("down");

    /// A calibrated NVFP4 matrix's packed E2M1 values: `[rows, columns / 2]` U8, one byte per pair.
    pub const NVFP4_VALUES: Self = Self("nvfp4_values");

    /// The same values after `shuffleMatrixA`: rows permuted inside blocks of sixteen or
    /// thirty-two, which is the arrangement the trtllm-gen products read (`useShuffledMatrix`).
    pub const NVFP4_VALUES_SHUFFLED: Self = Self("nvfp4_values_shuffled");
    /// [`ValueLayout::InterleavedShuffled`]'s row order.
    /// Packed values whose two row halves interleave in sixteen-row blocks, the second half's first:
    /// the row order a gated product reads when each gate block precedes its up block.
    pub const NVFP4_VALUES_GATE_FIRST: Self = Self("nvfp4_values_gate_first");
    pub const NVFP4_VALUES_INTERLEAVED_SHUFFLED: Self = Self("nvfp4_values_interleaved_shuffled");

    /// A calibrated NVFP4 matrix's E4M3 block scales in the linear layout: `[rows, columns / 16]`.
    pub const NVFP4_SCALES_LINEAR: Self = Self("nvfp4_scales_linear");

    /// The same scales in the swizzled 128x4 layout the cuDNN product reads.
    pub const NVFP4_SCALES_SWIZZLED: Self = Self("nvfp4_scales_swizzled_128x4");

    /// The layout named `name`.
    pub const fn named(name: &'static str) -> Self {
        Self(name)
    }
}

impl fmt::Display for WeightLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// An explicitly requested quantization method.
///
/// The names are SGLang's `quantization` values: an explicit value converts
/// the checkpoint's weights once, at load; with none, the checkpoint's own representation is kept.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Quantization {
    /// FP8 E4M3 with per-tensor scales.
    Fp8,
    /// FP8 E4M3 with block scales.
    Mxfp8,
    /// NVFP4 E2M1 with block scales.
    NvFp4,
}

impl Quantization {
    /// The element type this policy's encoded weights are stored in.
    pub const fn encoding(self) -> DType {
        match self {
            Self::Fp8 | Self::Mxfp8 => DType::F8E4M3,
            Self::NvFp4 => DType::U8,
        }
    }
}

impl fmt::Display for Quantization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Fp8 => "fp8",
            Self::Mxfp8 => "mxfp8",
            Self::NvFp4 => "nvfp4",
        })
    }
}

/// Where a weight's bytes come from: one region of one checkpoint tensor, or whole tensors fused in
/// order along their first axis.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WeightSource {
    /// `shape` elements starting at element `start` of `tensor`, row-major.
    Region {
        /// The checkpoint tensor's name.
        tensor: String,
        /// The first element of the region, in the tensor's row-major order.
        start: u64,
        /// The region's extents.
        shape: Shape,
    },
    /// `shape` elements of `tensor` whose last axis begins at element `start` and repeats every
    /// `stride` elements.
    ///
    /// A tensor-parallel rank reads one column block of a row-parallel projection: every row of the
    /// checkpoint's tensor contributes `shape.last()` elements, and consecutive rows start `stride`
    /// elements apart. The loader gathers those rows and converts them like any other weight, so
    /// the retained storage of a shard is contiguous like the rest.
    Strided {
        /// The checkpoint tensor's name.
        tensor: String,
        /// The first element of the first row, in the tensor's row-major order.
        start: u64,
        /// The gathered region's extents.
        shape: Shape,
        /// Elements between the first elements of consecutive rows.
        row_stride: u64,
    },
    /// The row-major concatenation of `parts` along their first axis, viewed as `shape`.
    ///
    /// Every part must have the same extents after its first axis. This is how a routed expert's
    /// gate and up projections become one fused weight, and how per-expert matrices become one
    /// stacked weight: the parts are named in the order the reading operator expects.
    Parts {
        /// The checkpoint tensors, in fused order.
        parts: Vec<String>,
        /// The fused tensor's extents.
        shape: Shape,
        /// The block each part contributes, or the whole part when absent.
        ///
        /// A tensor-parallel rank holds its block of every routed expert: the same rows of each
        /// gate and up projection, and the same columns of each down projection. Every part gives
        /// the same block, so the fused weight is the rank's shard of every part in order.
        block: Option<PartBlock>,
    },
    /// The row-major concatenation of one block of each part, every part naming its own block,
    /// viewed as `shape`.
    ///
    /// This is how a stacked checkpoint tensor (every expert's projections in one
    /// `[experts, rows, columns]` tensor) becomes the stack its reading operator expects in
    /// another order: each block is one expert's half, in the order the operator reads them.
    Blocks {
        /// The checkpoint tensors and the block each contributes, in fused order.
        parts: Vec<(String, PartBlock)>,
        /// The fused tensor's extents.
        shape: Shape,
    },
}

/// One block of a part: `rows` rows of `columns` elements, the first at element `start` of the
/// part in its row-major order and each next one `row_stride` elements after it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PartBlock {
    /// The block's first element in the part.
    pub start: u64,
    /// Rows in the block.
    pub rows: u64,
    /// Elements in one row of the block.
    pub columns: u64,
    /// Elements between the first elements of consecutive rows in the part.
    pub row_stride: u64,
}

/// One weight the model asks the loader for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightRequest {
    /// The logical tensor's name; part of the retained key and of every diagnostic.
    pub tensor: String,
    /// Where the bytes come from.
    pub source: WeightSource,
    /// The arrangement the reading operator declares.
    pub layout: WeightLayout,
    /// The logical extents after conversion.
    pub shape: Shape,
    /// The element type this tensor is read as whatever the deployment states, when it has one.
    ///
    /// A representation the model pins per tensor - the reference keeps a timestep embedding in
    /// FP32 and a modulation projection in BF16 at every preset (`weights.py:186-224`) - cannot be
    /// expressed by the loader's policy, which is one value for every weight it reads. A pin wins
    /// over the policy, and it is part of the retained key: two loaders of one checkpoint that read
    /// one tensor at two types retain two different weights.
    pub pinned: Option<DType>,
    /// Whether this tensor must hold a value, asked by [`WeightRequest::nonzero`].
    ///
    /// A matrix that multiplies the hidden state is never all zero in a trained or a calibrated
    /// checkpoint: the product would answer zeros for every activation, and whatever the loader
    /// published from it reads as "this projection is broken" long after the loading boundary could
    /// have said so. A norm's scale or a bias may legitimately be zero, so the check is asked for by
    /// the request rather than applied to every weight, and it reads the region the caller asked for
    /// - the bytes this load already holds, not a second pass over the tensor.
    pub nonzero: bool,
}

impl WeightRequest {
    /// A weight read unchanged from `shape` elements starting at element `start` of `tensor`.
    pub fn region(tensor: impl Into<String>, start: u64, shape: Shape) -> Self {
        let tensor = tensor.into();
        Self {
            source: WeightSource::Region {
                tensor: tensor.clone(),
                start,
                shape,
            },
            tensor,
            layout: WeightLayout::ROW_MAJOR,
            shape,
            pinned: None,
            nonzero: false,
        }
    }

    /// A weight gathered from `tensor`'s rows, each row starting `stride` elements after the last.
    pub fn strided(tensor: impl Into<String>, start: u64, shape: Shape, row_stride: u64) -> Self {
        let tensor = tensor.into();
        Self {
            source: WeightSource::Strided {
                tensor: tensor.clone(),
                start,
                shape,
                row_stride,
            },
            tensor,
            layout: WeightLayout::ROW_MAJOR,
            shape,
            pinned: None,
            nonzero: false,
        }
    }

    /// A weight fused from `parts`, viewed as `shape`, in `layout`.
    pub fn parts(
        tensor: impl Into<String>,
        parts: Vec<String>,
        shape: Shape,
        layout: WeightLayout,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            source: WeightSource::Parts {
                parts,
                shape,
                block: None,
            },
            layout,
            shape,
            pinned: None,
            nonzero: false,
        }
    }

    /// [`Self::parts`] of the same `block` of every part: a tensor-parallel rank's shard of each
    /// part, fused in order and viewed as `shape`.
    pub fn part_blocks(
        tensor: impl Into<String>,
        parts: Vec<String>,
        shape: Shape,
        layout: WeightLayout,
        block: PartBlock,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            source: WeightSource::Parts {
                parts,
                shape,
                block: Some(block),
            },
            layout,
            shape,
            pinned: None,
            nonzero: false,
        }
    }

    /// Ask that this tensor hold a value: an all-zero read is refused by name.
    ///
    /// The check is on the bytes this load reads - one rank's region of the tensor - so it also
    /// catches a rank whose own block is zero while the rest of the tensor is not, and it costs no
    /// second read of the shard.
    pub fn nonzero(mut self) -> Self {
        self.nonzero = true;
        self
    }

    /// A weight fused from one block of each of `parts`, viewed as `shape`, in `layout`.
    pub fn blocks(
        tensor: impl Into<String>,
        parts: Vec<(String, PartBlock)>,
        shape: Shape,
        layout: WeightLayout,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            source: WeightSource::Blocks { parts, shape },
            layout,
            shape,
            pinned: None,
            nonzero: false,
        }
    }

    /// The same request read as `dtype`, whatever the loader's policy states.
    pub fn pinned(mut self, dtype: DType) -> Self {
        self.pinned = Some(dtype);
        self
    }

    /// The same request in another layout.
    pub fn with_layout(mut self, layout: WeightLayout) -> Self {
        self.layout = layout;
        self
    }
}

/// The region of a logical tensor a retained weight occupies.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeightRegion {
    /// The checkpoint tensors' names, in fused order (one name for a plain region).
    pub tensors: Vec<String>,
    /// The first element of the region in the logical tensor.
    pub start: u64,
    /// The logical tensor's extents.
    pub shape: Shape,
    /// The source rows' stride in elements, when the region is a strided sub-matrix. Two gatherings
    /// of one tensor differ in it, so they cannot collide in retained storage.
    pub stride: Option<u64>,
}

impl fmt::Display for WeightRegion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.tensors.join("+"))?;
        write!(
            formatter,
            "[{}..{}]",
            self.start,
            self.start + self.shape.numel()
        )?;
        write!(formatter, "{:?}", self.shape.dims())?;
        if let Some(stride) = self.stride {
            write!(formatter, "|stride_{stride}")?;
        }
        Ok(())
    }
}

/// What produced a weight that no checkpoint file names.
///
/// A checkpoint weight is named by the files it was read from and its region in them. A computed
/// one — a rotary frequency table, fused output heads, a modulation table — has no file, so the only
/// thing that can name it is the computation: the class that ran it, and the numbers it produced.
/// A label is not an identity: two models that name their own fused heads
/// `denoiser.video.student_heads.weight` are two producers, and a key holding only the name lets the
/// second one map the first one's bytes, silently, under an identical extent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Provenance {
    /// The class that computed the weight, as the crate that placed it names itself.
    pub producer: &'static str,
    /// A digest of the bytes it produced: the same numbers are one weight under any name, and
    /// different numbers are two.
    pub content: [u8; 32],
}

impl Provenance {
    /// The provenance of a weight a class computed from `bytes`, the numbers it produced.
    pub fn computed(producer: &'static str, bytes: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update(bytes);
        Self {
            producer,
            content: digest.finalize().into(),
        }
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "computed:{}:", self.producer)?;
        for byte in self.content {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// What retained storage files a weight under.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeightKey {
    /// The identity of every checkpoint file the weight was read from, in part order.
    pub files: Vec<FileIdentity>,
    /// The element type of the retained bytes.
    pub dtype: DType,
    /// The layout of the retained bytes.
    pub layout: WeightLayout,
    /// The region of the logical tensor.
    pub region: WeightRegion,
    /// What produced the bytes, for a weight no checkpoint file names.
    ///
    /// `None` for a weight read from a checkpoint, whose files and region are its provenance; a
    /// computed weight states the class that computed it and a digest of the numbers it produced,
    /// so two producers that compute different tables under one name cannot share retained bytes.
    pub provenance: Option<Provenance>,
}

impl fmt::Display for WeightKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let files = self
            .files
            .iter()
            .map(FileIdentity::as_str)
            .collect::<Vec<_>>()
            .join("+");
        write!(
            formatter,
            "{files}|{}|{}|{}",
            self.dtype, self.layout, self.region
        )?;
        if let Some(provenance) = &self.provenance {
            write!(formatter, "|{provenance}")?;
        }
        Ok(())
    }
}

/// Retained storage, as the loader publishes converted weights to it.
///
/// The worker implements this over the Storage Server and its read-only device mappings: a
/// weight already retained under the key is handed back without reading the checkpoint again, and a
/// converted weight is published once.
pub trait RetainedWeights {
    /// The weight retained under `key`, when the store already holds it.
    fn retained(&mut self, key: &WeightKey) -> Result<Option<Backing>>;

    /// Publish `bytes` as the storage of `key`, with `shape` elements of `dtype`.
    ///
    /// The store keeps the bytes and answers the read-only region a layer's view is carved from.
    /// Publishing the same key twice is the caller's error; the store may reject it or answer the
    /// existing region.
    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> Result<Backing>;

    /// A region the model writes after the load, holding `bytes` until it does.
    ///
    /// A published region is read-only on the device, so a table a model folds from its weights
    /// cannot live in one: the first launch that writes it faults. A store whose regions a kernel
    /// may write answers one of those, which is what this default does.
    fn writable(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> Result<Backing> {
        self.publish(key, dtype, shape, bytes)
    }
}

/// The arrangement of a calibrated matrix's packed E2M1 values.
///
/// The values are two codes a byte along the contraction whichever arrangement they are in; what
/// varies is the order of the **rows**, because the trtllm-gen products read their A operand with
/// `useShuffledMatrix`: rows permuted inside blocks of sixteen or thirty-two so that the tile a
/// thread block loads is contiguous. It is a permutation of the shard's own bytes, applied once at
/// load — never a step, and never a kernel — and a matrix whose row count is not a whole number of
/// blocks has no such arrangement, which the loader refuses by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueLayout {
    /// The shard's own row order.
    Plain,
    /// `shuffleMatrixA`'s row order, for a product whose epilogue tile is `epilogue_tile_m` rows.
    Shuffled {
        /// Rows one MMA tile of the product covers; the shuffle block is thirty-two rows when this
        /// is a whole number of 128s and sixteen otherwise, as the reference's
        /// `get_shuffle_block_size` states.
        epilogue_tile_m: u32,
    },
    /// A gated product's row order: the matrix's two row halves interleave, so row `i` of the first
    /// half sits at `2i` and row `i` of the second half at `2i + 1` (`reorderRowsForGatedActGemm`),
    /// and the result shuffles as [`Self::Shuffled`] does. The product's epilogue then reads the
    /// linear and the gate row of one output channel side by side.
    InterleavedShuffled {
        /// Rows one MMA tile of the product covers, as [`Self::Shuffled`] states it.
        epilogue_tile_m: u32,
    },
    /// A fused gated product's row order: thirty-two-row block `b` holds rows `[16b, 16b + 16)` of
    /// the matrix's second half, then the same rows of its first half.
    GateFirst,
}

impl ValueLayout {
    /// The layout name a weight published in this arrangement carries.
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
                // The reference's maps in closed form: within its block, a source row `r` moves to
                // `(r % lanes) * 8 + r / lanes`, the interleave of `lanes` eight-row groups that the
                // artifact's `srcToDstBlk16RowMap` and `srcToDstBlk32RowMap` list. The block is
                // taken first: a row's offset inside its own block decides where it goes, and its
                // block decides where that block starts.
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
/// The launcher's own `computeSFIndex` (TensorRT-LLM's `fp8Op.h`) written out: rows padded to 128,
/// scale columns padded to 4, and each `[128, 4]` tile laid out so that a thread block's scales are
/// contiguous — column groups of four stride 512, thirty-two-row groups stride 16, groups of 128 rows
/// stride `128 · padded_columns`. The padding is zero, which is a scale of zero on a block that does
/// not exist.
fn swizzled_scales(linear: &[u8], rows: u64, blocks: u64) -> Vec<u8> {
    let padded_rows = rows.div_ceil(128) * 128;
    let padded_columns = blocks.div_ceil(4) * 4;
    let mut swizzled = vec![0_u8; (padded_rows * padded_columns) as usize];
    for row in 0..padded_rows {
        for column in 0..padded_columns {
            if row >= rows || column >= blocks {
                continue;
            }
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
/// The two weights are the encoded fields of the checkpoint, retained like any other weight; the
/// global scale is a plain value because it enters a kernel as a launch scalar.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedWeight {
    /// `[rows, columns / 2]` U8: two E2M1 codes a byte, packed along the contraction.
    pub packed: Weight,
    /// `ScaleLayout::shape(rows, columns)` U8: one E4M3 block scale per sixteen elements.
    pub scales: Weight,
    /// The checkpoint's FP32 tensor scale for this projection — the value, not its reciprocal.
    ///
    /// It enters a kernel as a launch scalar, which is why it is a plain value, and it multiplies:
    /// the operand's reconstruction is `codes · block scale · token scale`, which is the operand in
    /// real units, so the weight's tensor scale has to be applied exactly once and as a factor.
    pub global_scale: f32,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
    /// The reciprocal of the checkpoint's activation tensor scale, `[1]` F32, when the caller asked
    /// for it.
    ///
    /// A calibrated export states a static per-module `input_scale` for its activations, and an
    /// encoder needs it as a **device operand**: the product that consumes the encoding reads it as
    /// each thread block starts, so a host scalar would serialize the pair. It is calibration data
    /// of one tensor, invariant across requests, so it lives with the weights rather than in a
    /// descriptor or a request. `None` says the consumer is weight-only and asked for none, which is
    /// a different fact from a calibrated checkpoint that omitted it — that one is refused by
    /// tensor name.
    pub input_scale: Option<Weight>,
}

/// One packed table per projection: the axis order the batched-GEMM entry reads and the shape a
/// checkpoint carries a tensor for.
struct Projections {
    /// Projections per expert: two for a fused up/gate stack, one for a down stack.
    per: usize,
    /// Experts in the stack.
    experts: usize,
}

impl Projections {
    /// The order over `per` projections of `experts` experts, both as published extents.
    fn new(per: u64, experts: u64) -> Self {
        Self {
            per: usize::try_from(per).unwrap_or(0),
            experts: usize::try_from(experts).unwrap_or(0),
        }
    }

    /// How many scales the table holds.
    fn len(&self) -> usize {
        self.per * self.experts
    }

    /// `scales`, collected expert-major, as the table's little-endian f32 bytes.
    fn bytes(&self, scales: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.len() * 4);
        for projection in 0..self.per {
            for expert in 0..self.experts {
                bytes.extend_from_slice(&scales[expert * self.per + projection].to_le_bytes());
            }
        }
        bytes
    }
}

/// A calibrated NVFP4 **stack of experts**, as `gather_gemm_nvfp4` and `route_gemm_nvfp4` read it.
///
/// A model's expert block multiplies whole stacks: its weights are `[experts, rows, columns/2]` U8,
/// its block scales carry one entry per expert, and the tensor scale varies per expert **and per
/// projection** — so the scalar [`QuantizedWeight::global_scale`] cannot serve it, and a model may
/// not build the table itself (constructing a weight is the loader's, not a model's). The loader
/// publishes all three from the checkpoint's per-expert, per-projection tensors: the encoded values
/// and scales concatenated in the operator's reading order, and the tensor scales as a
/// `[projections, experts]` f32 weight — one packed table per projection, which is the shape the
/// batched-GEMM entry reads and the shape the checkpoint itself carries a tensor per.
///
/// The stack fused from an expert's `up_proj` and `gate_proj` is why the second axis exists: each of
/// those has its own `weight_scale_2` in the checkpoint, so one scalar per expert would dequantize
/// the gate half with the up half's scale and be wrong by the ratio of the two, silently.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedStack {
    /// `[experts, rows, columns/2]` U8: two E2M1 codes a byte, packed along the contraction.
    pub values: Weight,
    /// `[experts, ...]` in the checkpoint's scale type — `F8_E4M3`, or `U8` where the export labels
    /// the same bytes as raw — the block scales in [`Self::layout`], one per sixteen elements.
    /// Every expert of a stack agrees on it, and it is published unchanged: a model puts
    /// [`Self::scales`]`.dtype()` in the product's descriptor and nothing converts.
    pub scales: Weight,
    /// `[projections, experts]` f32, row-major, in the stack's own projection order: each expert's
    /// **per-projection** checkpoint tensor scale, which is the factor the product multiplies by.
    ///
    /// One packed table per projection rather than one row per expert, because that is what the
    /// batched-GEMM entry takes: it reads each projection's table as `experts` consecutive floats. A
    /// `[experts, projections]` table handed over as two rows would walk column 0 for even experts and
    /// column 1 for odd ones, which is a wrong dequantization with nothing to show for it.
    pub global_scale: Weight,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
    /// `[1]` f32: the activation encoder's global scale for this stack's activations: the
    /// reciprocal of the widest calibrated activation scale among its experts. The encoder writes
    /// that activation scale as each row's factor, and the product multiplies it in, so the table
    /// [`Self::global_scale`] holds the weights' tensor scales alone.
    pub encode_scale: Weight,
    /// `[projections, experts]` f32, row-major, in the same order: the reciprocal of each expert's
    /// **per-projection** activation tensor scale, when the caller asked for it.
    ///
    /// A ModelOpt MoE export calibrates each expert's projections separately, so one activation
    /// scale per stack would encode every expert but one at the wrong magnitude. `None` says the
    /// consumer is weight-only; a calibrated checkpoint asked for this and missing it is refused by
    /// tensor name.
    pub input_scale: Option<Weight>,
    /// The activation scale [`Self::encode_scale`] is the reciprocal of.
    pub activation_scale: f32,
}

/// Whether `dtype` is one of the two element types a block-scale field may have.
///
/// The block scales are eight bits per sixteen elements, and a checkpoint stores each as an E4M3
/// float — or, where an export labels the same bytes as raw, as `U8`. Both are one byte, both are
/// published unchanged, and nothing converts: a reader that multiplied a scale by a constant would
/// be wrong, and a loader that relabeled it would hide which one the checkpoint meant.
fn block_scale_dtype(tensor: &str, dtype: DType) -> Result<DType> {
    match dtype {
        DType::U8 | DType::F8E4M3 => Ok(dtype),
        other => Err(Error::weight(
            tensor,
            format!(
                "block scales are one byte each — F8_E4M3, or U8 where the export says so — not                  {other}"
            ),
        )),
    }
}

/// A weight a layer reads: its view and the region that keeps the bytes alive.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    /// The read-only region the view addresses.
    pub backing: Backing,
    /// The view the operator reads.
    pub view: TensorView,
}

impl Weight {
    /// A weight over memory the caller keeps alive for every launch that reads it: a table gathered
    /// into the forward's arena, or a request table the runtime holds for the length of a capture
    /// and its replays. Nothing is retained.
    ///
    /// # Errors
    ///
    /// A view whose element type has no whole-byte size.
    pub fn borrowed(view: TensorView) -> Result<Self> {
        let element = view.dtype.bytes().ok_or_else(|| {
            Error::weight(
                "",
                format!("a borrowed {} view has no whole-byte element", view.dtype),
            )
        })?;
        Ok(Self {
            backing: Backing::new(view.ptr, view.shape.numel() * element),
            view,
        })
    }

    /// The operator-facing view.
    pub fn view(&self) -> TensorView {
        self.view
    }

    /// The weight's element type.
    pub fn dtype(&self) -> DType {
        self.view.dtype
    }

    /// The weight's extents.
    pub fn shape(&self) -> Shape {
        self.view.shape
    }

    /// Row `row` of a two-axis weight, as a weight of its own in the same region.
    ///
    /// # Errors
    ///
    /// A weight of another rank, or a row past its first axis.
    pub fn row(&self, row: u64) -> Result<Self> {
        let shape = self.view.shape;
        if shape.rank() != 2 || row >= shape.dim(0) {
            return Err(Error::weight(
                "",
                format!("row {row} of a weight of {:?}", shape.dims()),
            ));
        }
        let element = self.view.dtype.bytes().unwrap_or(0);
        let bytes = shape.dim(1) * element;
        Ok(Self {
            backing: self.backing,
            view: TensorView::new(
                self.view.ptr.offset(row * bytes),
                self.view.dtype,
                Shape::new(&[shape.dim(1)]),
            ),
        })
    }
}

/// The first address a meta loader hands out.
///
/// Far above the counting arena's base (`1 << 40`, `forward.rs`), so a weight view and an
/// activation view of one recorded walk never share an address, and far from zero, so no weight
/// view reads as a null pointer to an argument check.
const META_BASE: u64 = 1 << 48;

/// The next address a meta loader hands out.
///
/// Every meta loader of the process draws from this one cursor, so two loaders of one walk (one per
/// pipeline component, say) never answer one address for two different weights.
static META_CURSOR: AtomicU64 = AtomicU64::new(META_BASE);

/// The tensor scale a meta loader answers for every calibrated scale it is asked for.
///
/// A meta loader reads no payload, so it has no scale to answer; one is a positive, finite number,
/// so every check a caller makes on a scale passes, and a value derived from it is a valid one. A
/// scale only ever reaches a kernel as data (a launch scalar or a published table), so a recorded
/// walk over this placeholder states the same programs as a device load over the real scales.
const META_TENSOR_SCALE: f32 = 1.0;

/// The synthetic regions a meta loader answered, by the key a device load would publish them under.
///
/// A key answered once is answered again with the same region, as a retained store answers a key it
/// already holds, so two layers that share a weight on the device share its view here too.
#[derive(Default)]
struct MetaRegions {
    regions: HashMap<WeightKey, Backing>,
}

impl MetaRegions {
    /// The region answered for `key`, when one was.
    fn retained(&self, key: &WeightKey) -> Option<Backing> {
        self.regions.get(key).copied()
    }

    /// A fresh region of `bytes` bytes at an [`ALIGNMENT`]-aligned synthetic address.
    ///
    /// `retain` files it under `key` for a later [`Self::retained`]; a writable table is this
    /// load's own and is never answered twice, as on the device.
    fn reserve(&mut self, key: &WeightKey, bytes: u64, retain: bool) -> Result<Backing> {
        // Every region takes at least one granule, so even an empty weight has an address no other
        // weight shares.
        let span = bytes
            .max(1)
            .checked_next_multiple_of(ALIGNMENT)
            .ok_or_else(|| Error::weight(&key.region.to_string(), "the weight's bytes overflow"))?;
        let address = META_CURSOR.fetch_add(span, Ordering::Relaxed);
        if address.checked_add(span).is_none() {
            return Err(Error::weight(
                &key.region.to_string(),
                "the meta loader's synthetic address range is exhausted",
            ));
        }
        let backing = Backing::new(DevicePtr::new(address), bytes);
        if retain {
            self.regions.insert(key.clone(), backing);
        }
        Ok(backing)
    }
}

/// Where a loader's weights go.
enum Retention<'a> {
    /// Retained storage, which converted bytes are published to: a device load.
    Store(&'a mut dyn RetainedWeights),
    /// Synthetic regions, owned by a loader [`WeightLoader::meta`] built.
    Meta(MetaRegions),
    /// The synthetic regions of the meta loader this one was reborrowed from.
    MetaBorrowed(&'a mut MetaRegions),
}

/// One contiguous byte range of a checkpoint tensor that a weight's bytes are gathered from.
///
/// A load plans its spans from the shard headers alone, so every bound a device load checks is
/// checked by a meta load too, without a payload byte read; a device load then copies the spans in
/// order.
struct Span<'c> {
    location: &'c TensorLocation,
    /// The first byte, within the tensor's own bytes.
    from: usize,
    /// One past the last byte, within the tensor's own bytes.
    to: usize,
}

/// How many bytes the shard's header states `location` holds.
///
/// `Checkpoint::open` has checked the range against the tensor's extents and element width and
/// against the shard's length, so this is the length of the slice [`Checkpoint::bytes`] answers.
fn stored_bytes(location: &TensorLocation) -> u64 {
    location.bytes.1 - location.bytes.0
}

/// The boundary between a checkpoint and retained storage: converts a weight once and publishes it.
///
/// A loader is one of two kinds, and both run the same decisions over the same shard headers:
///
/// - a **device** loader ([`WeightLoader::new`]) reads each tensor's payload, converts it and
///   publishes it to a [`RetainedWeights`] store;
/// - a **meta** loader ([`WeightLoader::meta`]) reads no payload and publishes nothing: every weight
///   it answers has the element type and extents the device load answers, at a synthetic address.
///
/// Every refusal a device load decides from a tensor's name, element type or extents is a meta
/// load's refusal too. Only a refusal that depends on a tensor's values (an all-zero matrix asked
/// to hold a value, a zero or non-finite tensor scale) is a device load's alone.
pub struct WeightLoader<'a> {
    /// The checkpoint the weights come from.
    checkpoint: &'a Checkpoint,
    /// Where the weights go: retained storage, or synthetic regions.
    retention: Retention<'a>,
    dtype: Option<DType>,
    quantization: Option<Quantization>,
    /// What the checkpoint declares about its packed weights, resolved against its shards; dense
    /// until a model [`declares`](WeightLoader::declare) one.
    declaration: Declaration,
}

impl<'a> WeightLoader<'a> {
    /// A loader over `checkpoint` publishing to `store`.
    ///
    /// `dtype` and `quantization` are the launch's explicit precision choices; with neither, a
    /// weight keeps the checkpoint's representation.
    pub fn new(
        checkpoint: &'a Checkpoint,
        store: &'a mut dyn RetainedWeights,
        dtype: Option<DType>,
        quantization: Option<Quantization>,
    ) -> Self {
        Self {
            checkpoint,
            retention: Retention::Store(store),
            dtype,
            quantization,
            declaration: Declaration::dense(),
        }
    }

    /// A loader over `checkpoint` that reads no tensor payload and publishes nothing.
    ///
    /// Each weight's element type and extents come from the shard headers [`Checkpoint::open`]
    /// already parsed, and `dtype` and `quantization` are the same precision choices
    /// [`WeightLoader::new`] takes. Every method answers what the device load of the same request
    /// answers — the same element type, the same extents, the same representation (dense or NVFP4
    /// fields, the block-scale type the checkpoint stores, the published layout) and the same
    /// refusals — because it runs the device load's own code up to the point where bytes would be
    /// read. The views it answers sit at distinct, [`ALIGNMENT`]-aligned, non-null synthetic
    /// addresses that no memory backs; a weight's key answers the same view twice, as a retained
    /// store answers a key it holds.
    ///
    /// What a meta load cannot answer is a tensor's **value**: [`Self::tensor_scale`] answers a
    /// placeholder of one, a value-dependent refusal does not run, and the bytes of a computed
    /// constant derived from a placeholder are not the device's. The caller is a model's load driven
    /// by a recording forward, which launches nothing and reads no weight: the shapes and element
    /// types decide the programs a model launches, and the values never enter a program key.
    pub fn meta(
        checkpoint: &'a Checkpoint,
        dtype: Option<DType>,
        quantization: Option<Quantization>,
    ) -> Self {
        Self {
            checkpoint,
            retention: Retention::Meta(MetaRegions::default()),
            dtype,
            quantization,
            declaration: Declaration::dense(),
        }
    }

    /// The checkpoint this loader reads.
    pub fn checkpoint(&self) -> &'a Checkpoint {
        self.checkpoint
    }

    /// Read every later weight by `declaration`, resolved against this loader's checkpoint.
    ///
    /// The resolution reads the shard headers once: a weight is packed when its shard stores a
    /// block scale beside it, whatever the declaration's exclusion list says. A meta loader's
    /// checkpoint holds the same headers, so it resolves the same way.
    pub fn declare(&mut self, declaration: &Declaration) {
        self.declaration = declaration.resolve(self.checkpoint);
    }

    /// How the checkpoint stores the weight named `weight`, by the resolved declaration.
    pub fn scheme(&self, weight: &str) -> Scheme {
        self.declaration.scheme(weight)
    }

    /// The scope of the module at `prefix`: its tensors are named `prefix.<leaf>`, and its
    /// children's `prefix.<child>.<leaf>`, as `named_parameters()` composes them. An empty prefix
    /// is the checkpoint's root.
    pub fn scope(&mut self, prefix: impl Into<String>) -> Scope<'_, 'a> {
        Scope {
            loader: self,
            path: prefix.into(),
            pinned: None,
        }
    }

    /// Whether this loader answers synthetic regions instead of reading and publishing.
    fn is_meta(&self) -> bool {
        !matches!(self.retention, Retention::Store(_))
    }

    /// The weight already held under `key`, viewed as `shape` elements of its type.
    fn retained(&mut self, key: &WeightKey, shape: Shape) -> Result<Option<Weight>> {
        let backing = match &mut self.retention {
            Retention::Store(store) => store.retained(key)?,
            Retention::Meta(regions) => regions.retained(key),
            Retention::MetaBorrowed(regions) => regions.retained(key),
        };
        backing
            .map(|backing| {
                Ok(Weight {
                    view: backing.view(key.dtype, shape)?,
                    backing,
                })
            })
            .transpose()
    }

    /// Publish the weight of `key`, `shape` elements of `key.dtype`, and answer its view.
    ///
    /// `bytes` produces the converted bytes from the checkpoint's payload; only a device loader
    /// calls it, so everything a meta load must decide happens before this call. A `writable` table
    /// goes to the store's writable regions and is never answered for a later key.
    fn publish_with<'c>(
        &mut self,
        key: &WeightKey,
        shape: Shape,
        writable: bool,
        bytes: impl FnOnce(&Self) -> Result<Cow<'c, [u8]>>,
    ) -> Result<Weight> {
        let backing = if self.is_meta() {
            let element = key.dtype.bytes().ok_or_else(|| {
                Error::weight(
                    &key.region.to_string(),
                    format!("{} has no whole-byte element size", key.dtype),
                )
            })?;
            let length = shape.numel().checked_mul(element).ok_or_else(|| {
                Error::weight(&key.region.to_string(), "the weight's bytes overflow")
            })?;
            match &mut self.retention {
                Retention::Meta(regions) => regions.reserve(key, length, !writable)?,
                Retention::MetaBorrowed(regions) => regions.reserve(key, length, !writable)?,
                Retention::Store(_) => unreachable!("a store-backed loader is not a meta loader"),
            }
        } else {
            let produced = bytes(self)?;
            match &mut self.retention {
                Retention::Store(store) if writable => {
                    store.writable(key, key.dtype, shape, &produced)?
                }
                Retention::Store(store) => store.publish(key, key.dtype, shape, &produced)?,
                Retention::Meta(_) | Retention::MetaBorrowed(_) => {
                    unreachable!("a meta loader publishes nothing")
                }
            }
        };
        Ok(Weight {
            view: backing.view(key.dtype, shape)?,
            backing,
        })
    }

    /// Load `request`, whose tensors the checkpoint stores as weight-only NVFP4
    /// ([`Scheme::Nvfp4Weights`]), as the BF16 matrix their codes and scales state.
    ///
    /// The request names the logical matrix, `[rows, columns]` of every tensor it reads, exactly as
    /// a dense read names it: a tensor-parallel block, a strided column block or a fused stack of
    /// experts. Each tensor is decoded once for the read, `code · block_scale · tensor_scale` in
    /// FP32 rounded once to BF16 (round to nearest even), and the request's region is gathered from
    /// the decoded matrices. The product that reads the weight is then the 16-bit product the
    /// declaration states; the stored weight is BF16, twice the 16-bit-activation kernels' four.
    ///
    /// # Errors
    ///
    /// A tensor whose fields are missing or of another extent, or the request's own refusals.
    pub(crate) fn load_decoded(&mut self, request: &WeightRequest) -> Result<Weight> {
        let names = source_names(&request.source);
        let mut logical = Vec::with_capacity(names.len());
        for name in &names {
            let packed = self.checkpoint.locate(name)?;
            if packed.dtype != DType::U8 || packed.shape.rank() != 2 {
                return Err(Error::weight(
                    name,
                    format!(
                        "a weight-only NVFP4 weight is [rows, columns / 2] U8, not {} {:?}",
                        packed.dtype,
                        packed.shape.dims()
                    ),
                ));
            }
            let (rows, columns) = (packed.shape.dim(0), packed.shape.dim(1) * 2);
            logical.push(TensorLocation {
                file: packed.file,
                tensor: name.clone(),
                dtype: DType::BF16,
                shape: Shape::new(&[rows, columns]),
                bytes: (0, rows * columns * 2),
            });
        }
        let files = self.identities(&names)?;
        let (start, shape, stride) = source_region(&request.source);
        let key = WeightKey {
            files,
            dtype: DType::BF16,
            layout: request.layout,
            region: WeightRegion {
                tensors: names.clone(),
                start,
                shape,
                stride,
            },
            provenance: None,
        };
        if let Some(weight) = self.retained(&key, request.shape)? {
            return Ok(weight);
        }
        let spans = spans_in(&request.source, &key, &|name| {
            logical.iter().find(|location| location.tensor == name)
        })?;
        let read: u64 = spans.iter().map(|span| (span.to - span.from) as u64).sum();
        if read != request.shape.numel() * 2 {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "{read} bytes were read, {} expected for {:?} of bf16",
                    request.shape.numel() * 2,
                    request.shape.dims()
                ),
            ));
        }
        check_layout(&request.tensor, &request.source, request.layout)?;
        self.publish_with(&key, request.shape, false, |loader| {
            let mut decoded: Vec<(String, Vec<u8>)> = Vec::new();
            let mut bytes = Vec::with_capacity(read as usize);
            for span in &spans {
                let name = &span.location.tensor;
                if !decoded.iter().any(|(decoded, _)| decoded == name) {
                    let matrix = loader.decode_nvfp4(name, span.location.shape)?;
                    decoded.push((name.clone(), matrix));
                }
                let (_, matrix) = decoded
                    .iter()
                    .find(|(decoded, _)| decoded == name)
                    .ok_or_else(|| Error::weight(name, "a decoded matrix went missing"))?;
                bytes.extend_from_slice(&matrix[span.from..span.to]);
            }
            Ok(Cow::Owned(bytes))
        })
    }

    /// The BF16 bytes of the `[rows, columns]` matrix the NVFP4 fields of `name` state.
    fn decode_nvfp4(&self, name: &str, shape: Shape) -> Result<Vec<u8>> {
        let stem = name.strip_suffix("weight").ok_or_else(|| {
            Error::weight(name, "an NVFP4 weight is a module's `weight`".to_owned())
        })?;
        let (rows, columns) = (shape.dim(0), shape.dim(1));
        let packed = self.checkpoint.read(name)?;
        let scales = self.checkpoint.read(&format!("{stem}weight_scale"))?;
        if scales.dtype != DType::F8E4M3 || scales.shape.numel() * 16 != rows * columns {
            return Err(Error::weight(
                name,
                format!(
                    "block scales of {} {:?} for a {rows} × {columns} matrix; one F8_E4M3 per \
                     sixteen elements expected",
                    scales.dtype,
                    scales.shape.dims()
                ),
            ));
        }
        let tensor_scale = self.tensor_scale(&WeightRequest::region(
            format!("{stem}weight_scale_2"),
            0,
            Shape::new(&[1]),
        ))?;
        Ok(crate::quantization::decode_nvfp4(
            &packed.bytes,
            &scales.bytes,
            tensor_scale,
            columns as usize,
        ))
    }
    /// Load `request`, converting and publishing it the first time and reusing it afterwards.
    pub fn load(&mut self, request: &WeightRequest) -> Result<Weight> {
        if self.quantization.is_some() {
            return Err(Error::Unsupported(format!(
                "quantization `{}` is converted at load only for the FP8 and NVFP4 step; the BF16 \
                 slice keeps each checkpoint's own representation",
                self.quantization
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            )));
        }

        let names = source_names(&request.source);
        let (start, source_shape, stride) = source_region(&request.source);
        if let WeightSource::Parts { shape, .. } | WeightSource::Blocks { shape, .. } =
            &request.source
            && !request.shape.matches(shape)
        {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "the request asks for {:?} but its parts are fused as {shape:?}",
                    request.shape.dims()
                ),
            ));
        }

        let files = self.identities(&names)?;
        let source_dtype = self.checkpoint_dtype(&names)?;
        let dtype = self.target_dtype(&names, request.pinned)?;
        let region = WeightRegion {
            tensors: names.clone(),
            start,
            shape: source_shape,
            stride,
        };
        let key = WeightKey {
            files,
            dtype,
            layout: request.layout,
            region,

            provenance: None,
        };

        if let Some(weight) = self.retained(&key, request.shape)? {
            return Ok(weight);
        }

        // The bytes the request names, planned from the headers: every bound below is checked on
        // a meta load as on a device load, and only the copy waits for the publish.
        let spans = self.spans(&request.source, &key)?;
        if source_dtype != dtype {
            conversion(source_dtype, dtype, &request.tensor)?;
        }
        let source_element = source_dtype.bytes().ok_or_else(|| {
            Error::weight(
                &request.tensor,
                format!("{source_dtype} has no whole-byte size"),
            )
        })?;
        let element = dtype.bytes().ok_or_else(|| {
            Error::weight(
                &request.tensor,
                format!("{dtype} has no whole-byte element size"),
            )
        })?;
        let expected = request
            .shape
            .numel()
            .checked_mul(element)
            .ok_or_else(|| Error::weight(&request.tensor, "the weight's bytes overflow"))?;
        let read = spans
            .iter()
            .map(|span| (span.to - span.from) as u64)
            .sum::<u64>();
        // A conversion keeps the element count and changes its width.
        let converted = read / source_element * element;
        if converted != expected {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "{converted} bytes were read, {expected} expected for {:?} of {dtype}",
                    request.shape.dims()
                ),
            ));
        }
        check_layout(&request.tensor, &request.source, request.layout)?;

        let tensor = &request.tensor;
        let nonzero = request.nonzero;
        self.publish_with(&key, request.shape, false, |loader| {
            let bytes = loader.gather(&spans)?;
            // The region's own bytes, before any conversion: one read, which this load was doing
            // anyway, so the sentence costs no second pass over the shard.
            if nonzero && bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    tensor,
                    "every stored byte of the region is zero: the product would answer zeros for \
                     every activation, so this tensor is missing its export"
                        .to_owned(),
                ));
            }
            // The key carries the retained element type; the bytes still hold the checkpoint's, so
            // the conversion is decided by the source, not by the key.
            if source_dtype != dtype {
                Ok(Cow::Owned(convert(source_dtype, dtype, &bytes, tensor)?))
            } else {
                Ok(Cow::Owned(bytes))
            }
        })
    }

    /// Publish the weight-normalized form of `v` and `g` as one weight.
    ///
    /// A checkpoint stores a weight-normalized convolution as two tensors: `v` with the weight's
    /// extents and `g` with one magnitude per output channel (PyTorch keeps it as `[channels, 1,
    /// 1]`). The weight the convolution reads is `v · g / ‖v‖`, where the norm reduces every axis
    /// but the first per channel; spec 9 computes it once at loading, so a loaded model never
    /// carries the pair. The computation is FP32 and the result keeps the checkpoint's element type
    /// unless the loader was given another. Its extents are `v`'s, as the header states them.
    ///
    /// `name` is the logical tensor the key records; the key also names both sources, so it cannot
    /// be confused with a plain region of either.
    pub fn weight_norm(&mut self, name: &str, v: &str, g: &str) -> Result<Weight> {
        self.fold(name, v, g, None)
    }

    /// [`Self::weight_norm`], written as `pinned` when a scope pins its module's type.
    fn fold(&mut self, name: &str, v: &str, g: &str, pinned: Option<DType>) -> Result<Weight> {
        let value = self.checkpoint.locate(v)?;
        let gain = self.checkpoint.locate(g)?;
        if value.shape.rank() == 0 {
            return Err(Error::weight(
                name,
                "a weight-normalized tensor has an axis".to_owned(),
            ));
        }
        let channels = value.shape.dim(0);
        if gain.shape.numel() != channels {
            return Err(Error::weight(
                name,
                format!(
                    "a gain of {} elements for {channels} output channels",
                    gain.shape.numel()
                ),
            ));
        }
        let dtype = pinned.or(self.dtype).unwrap_or(value.dtype);
        if !dtype.is_float() || !value.dtype.is_float() {
            return Err(Error::weight(
                name,
                format!(
                    "weight normalization of {} is not floating point",
                    value.dtype
                ),
            ));
        }
        // The fold reads both tensors as FP32 and writes the result back: each of the three is one
        // conversion this loader must have.
        conversion(value.dtype, DType::F32, v)?;
        conversion(gain.dtype, DType::F32, g)?;
        conversion(DType::F32, dtype, name)?;

        let names = vec![v.to_owned(), g.to_owned()];
        let key = WeightKey {
            files: self.identities(&names)?,
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            region: WeightRegion {
                // The logical name first: the double underscore marks a computed weight, so the key
                // is a region neither source names on its own.
                tensors: vec![format!("{name}__weight_norm"), v.to_owned(), g.to_owned()],
                start: 0,
                shape: value.shape,
                stride: None,
            },

            provenance: None,
        };
        if let Some(weight) = self.retained(&key, value.shape)? {
            return Ok(weight);
        }

        self.publish_with(&key, value.shape, false, |loader| {
            let value = loader.checkpoint.read(v)?;
            let gain = loader.checkpoint.read(g)?;
            let values = to_f32(value.dtype, &value.bytes, v)?;
            let gains = to_f32(gain.dtype, &gain.bytes, g)?;
            let per_channel = usize::try_from(value.shape.numel() / channels.max(1))
                .map_err(|_| Error::weight(name, "more elements than a usize".to_owned()))?;
            let mut normalized = Vec::with_capacity(values.len());
            for (channel, gain) in gains.iter().enumerate() {
                let rows = &values[channel * per_channel..][..per_channel];
                let sum = rows.iter().map(|value| value * value).sum::<f32>();
                let scale = gain / sum.sqrt();
                normalized.extend(rows.iter().map(|value| value * scale));
            }
            Ok(Cow::Owned(from_f32(dtype, &normalized, name)?))
        })
    }

    /// Borrow this loader for another checkpoint: the same store, the same element-type and
    /// quantization policy, a different shard set.
    ///
    /// A pipeline root holds one component per subdirectory (`transformer/`, `vae/`,
    /// `text_encoder/`, …), each with its own index and its own file identities, so a model crate
    /// opens the root once and reads every component through one store: a weight converted once for
    /// one component is reused for another and never converted twice. A meta loader's reborrow is a
    /// meta loader over the same synthetic regions.
    ///
    /// The policy is the loader's, not the component's, so a component that its policy does not
    /// describe is refused here: an FP8-policy loader over a BF16 component, or a float policy over
    /// a component whose weights are an FP8 or NVFP4 encoding, would read weights as something they
    /// are not, which is a wrong answer rather than a convenience. A model that needs two policies
    /// builds two loaders over the same store.
    pub fn reborrow<'b>(&'b mut self, checkpoint: &'b Checkpoint) -> Result<WeightLoader<'b>> {
        self.check_policy(checkpoint)?;
        let retention = match &mut self.retention {
            Retention::Store(store) => Retention::Store(&mut **store),
            Retention::Meta(regions) => Retention::MetaBorrowed(regions),
            Retention::MetaBorrowed(regions) => Retention::MetaBorrowed(regions),
        };
        Ok(WeightLoader {
            checkpoint,
            retention,
            dtype: self.dtype,
            quantization: self.quantization,
            declaration: self.declaration.clone(),
        })
    }

    /// Whether this loader's policy describes `checkpoint`'s weights.
    fn check_policy(&self, checkpoint: &Checkpoint) -> Result<()> {
        let held = checkpoint.dtypes();
        let listed = held
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        if let Some(quantization) = self.quantization {
            let encoding = quantization.encoding();
            if !held.contains(&encoding) {
                return Err(Error::policy(format!(
                    "a {quantization} policy encodes weights as {encoding}, and this checkpoint \
                     holds {listed}"
                )));
            }
        }
        if let Some(dtype) = self.dtype {
            for encoded in [DType::F8E4M3, DType::F4E2M1] {
                if held.contains(&encoded) {
                    return Err(Error::policy(format!(
                        "this checkpoint holds {encoded} encoded weights, which a {dtype} element \
                         type does not describe; the quantization is the loader's policy, not its \
                         element type"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Run `check` over the checkpoint's payload, on a loader that reads one.
    ///
    /// A refusal that depends on a tensor's values rather than its element type or extents can only
    /// be made with the payload in hand, so it is a device load's alone; a meta load skips it.
    fn check_values(&self, check: impl FnOnce(&Checkpoint) -> Result<()>) -> Result<()> {
        if self.is_meta() {
            return Ok(());
        }
        check(self.checkpoint)
    }

    /// Load a calibrated NVFP4 matrix: its packed E2M1 values, its block scales and its tensor
    /// scale.
    ///
    /// A calibrated checkpoint already holds the encoded fields (`values`, `block_scale`,
    /// `tensor_scale`), so nothing is quantized here: the three are published through retained
    /// storage under keys naming the scale layout, and a deployment that asks to quantize a BF16
    /// checkpoint at load does it with the `nvfp4_quantize` operator on the device.
    ///
    /// `file` is the arrangement the checkpoint stores its block scales in and `product` the one the
    /// deployment's kernels read; when the two differ the table is permuted here, and
    /// [`QuantizedWeight::layout`] states the product's. `values` is `[rows, columns / 2]` U8 (two
    /// E2M1 codes a byte), `scales` holds the block scales in `file`'s arrangement, in whatever
    /// one-byte type the checkpoint stores them (`F8_E4M3`, or `U8` where the export says so;
    /// [`QuantizedWeight::scales`]`.dtype()` is that type, unchanged), and `global_scale` is the
    /// reciprocal of the checkpoint's FP32 tensor
    /// scale, the scalar `nvfp4_quantize` takes. The product op's `alpha` is the two tensor
    /// scales' product, which the runner copies into one device scalar once at load.
    // The two arrangements are separate arguments on purpose: one is the file's statement, the other
    // the deployment's, and naming both is what keeps a caller from passing one for the other.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4(
        &mut self,
        values: &WeightRequest,
        scales: &WeightRequest,
        input_scale: Option<&WeightRequest>,
        names: Nvfp4Names,
        file: ScaleLayout,
        product: ScaleLayout,
        global_scale: f32,
    ) -> Result<QuantizedWeight> {
        // The requests name the fields, so their absence is where a wrong vocabulary shows up: a
        // checkpoint opened as the other export's spelling has a `weight_packed` where this asked
        // for a `weight`, and the refusal says which spelling was expected and why.
        for (request, field, suffix) in [
            (values, "packed E2M1 values", names.values),
            (scales, "K16 block scales", names.block_scale),
        ] {
            self.check_nvfp4_field(&request.tensor, names, field, suffix)?;
        }
        if !(global_scale.is_finite() && global_scale > 0.0) {
            return Err(Error::weight(
                &values.tensor,
                format!("a global scale of {global_scale} is not a positive number"),
            ));
        }
        if values.shape.last() == 0 {
            return Err(Error::weight(
                &values.tensor,
                "a packed NVFP4 matrix has a contraction axis".to_owned(),
            ));
        }
        let rows = values.shape.numel() / values.shape.last();
        let columns = values.shape.last() * 2;
        let expected = file.shape(rows, columns);
        if scales.shape != expected {
            return Err(Error::weight(
                &scales.tensor,
                format!(
                    "the {file} block scales of a [{rows}, {columns}] matrix are {:?}, not {:?}",
                    expected.dims(),
                    scales.shape.dims()
                ),
            ));
        }
        if file == ScaleLayout::Swizzled128X4 && product == ScaleLayout::Linear {
            return Err(Error::weight(
                &scales.tensor,
                "a swizzled file rewritten into the linear table is a permutation this loader does \
                 not have: a calibrated export stores its block scales linear"
                    .to_owned(),
            ));
        }
        let product_layout = match product {
            ScaleLayout::Linear => WeightLayout::NVFP4_SCALES_LINEAR,
            ScaleLayout::Swizzled128X4 => WeightLayout::NVFP4_SCALES_SWIZZLED,
        };
        let mut values_request = values.clone().with_layout(WeightLayout::NVFP4_VALUES);
        values_request.shape = Shape::new(&[rows, columns / 2]);
        let packed = self.load(&values_request)?;
        if packed.dtype() != DType::U8 {
            return Err(Error::weight(
                &values_request.tensor,
                format!("packed E2M1 values are U8 bytes, not {}", packed.dtype()),
            ));
        }
        // A calibrated export never packs an all-zero matrix: every E2M1 code would be zero, the
        // product would answer zeros for every activation, and a cache filled from it would read as
        // "this projection is broken" long after the loading boundary could have said so. One host
        // read of a few megabytes per projection, at load only, buys that sentence.
        self.check_values(|checkpoint| {
            let bytes = checkpoint.read(&values_request.tensor)?;
            if bytes.bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    &values_request.tensor,
                    "every packed E2M1 byte is zero: the product would answer zeros for every \
                     activation, so this tensor's calibration did not run"
                        .to_owned(),
                ));
            }
            // The block scales are the second way a product answers zeros for every activation:
            // an E4M3 scale of zero multiplies its whole block to zero whatever the codes are. One
            // zero scale is legitimate (a block whose values are all zero); every scale zero is a
            // calibration that did not run, and it is invisible in the values themselves.
            let file_scales = checkpoint.read(&scales.tensor)?;
            if file_scales.bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    &scales.tensor,
                    "every block scale is zero: the product would answer zeros for every \
                     activation whatever the codes hold, so this tensor's calibration did not run"
                        .to_owned(),
                ));
            }
            Ok(())
        })?;
        // The file's arrangement and the product's are two statements, and the permutation between
        // them happens here once: a kernel must never read a table under a claim it does not meet.
        let scales = if file == product {
            let mut request = scales.clone().with_layout(product_layout);
            request.shape = expected;
            let loaded = self.load(&request)?;
            block_scale_dtype(&request.tensor, loaded.dtype())?;
            loaded
        } else {
            let location = self.checkpoint.locate(&scales.tensor)?;
            let dtype = block_scale_dtype(&scales.tensor, location.dtype)?;
            let blocks = file.columns(columns);
            // The request may name a column block of a row-parallel matrix, so the rows are gathered
            // at its stride before the permutation: permuting the file's first `rows · blocks`
            // elements instead would publish another rank's scales under this rank's key.
            let (start, stride) = match &scales.source {
                WeightSource::Region { start, .. } => (*start, blocks),
                WeightSource::Strided {
                    start, row_stride, ..
                } => (*start, *row_stride),
                WeightSource::Parts { .. } | WeightSource::Blocks { .. } => {
                    return Err(Error::weight(
                        &scales.tensor,
                        "block scales gathered from fused parts have no single row stride"
                            .to_owned(),
                    ));
                }
            };
            // Each row's byte range, checked against the header's length of the tensor: the scales
            // are one byte each, so elements and bytes are the same offsets.
            let length = usize::try_from(stored_bytes(location)).unwrap_or(usize::MAX);
            let mut ranges = Vec::with_capacity(rows as usize);
            for row in 0..rows {
                let first = start + row * stride;
                let end = first + blocks;
                let end = usize::try_from(end).unwrap_or(usize::MAX);
                let first = usize::try_from(first).unwrap_or(usize::MAX);
                if first > end || end > length {
                    return Err(Error::weight(
                        &scales.tensor,
                        format!(
                            "elements {first}..{end} of a row are past the tensor's {blocks}-block rows"
                        ),
                    ));
                }
                ranges.push(first..end);
            }
            let shape = product.shape(rows, columns);
            let files = self.identities(std::slice::from_ref(&scales.tensor))?;
            let tensor = &scales.tensor;
            self.publish_computed(
                WeightKey {
                    files,
                    dtype,
                    layout: product_layout,
                    region: WeightRegion {
                        tensors: std::iter::once(format!("{}__nvfp4_scales", scales.tensor))
                            .collect(),
                        start: 0,
                        shape,
                        stride: None,
                    },

                    provenance: None,
                },
                shape,
                |loader| {
                    let bytes = loader.checkpoint.read(tensor)?;
                    let mut linear = Vec::with_capacity((rows * blocks) as usize);
                    for range in ranges {
                        linear.extend_from_slice(&bytes.bytes[range]);
                    }
                    Ok(Cow::Owned(swizzled_scales(&linear, rows, blocks)))
                },
            )?
        };
        let input_scale = match input_scale {
            None => None,
            Some(request) => {
                self.check_nvfp4_field(
                    &request.tensor,
                    names,
                    "calibration's activation scale",
                    names.input_scale,
                )?;
                // This value *is* the encoder's factor, so it is the one place the ceiling applies.
                let reciprocal = self.encoder_factor(request)?;
                let shape = Shape::new(&[1]);
                let bytes = reciprocal.to_le_bytes();
                // The activation's reciprocal is computed from one file: the suffix names the
                // published piece, not a tensor the checkpoint holds, so it is not an identity.
                let files = self.identities(std::slice::from_ref(&request.tensor))?;
                Some(
                    self.publish_computed(
                        WeightKey {
                            files,
                            dtype: DType::F32,
                            layout: WeightLayout::ROW_MAJOR,
                            region: WeightRegion {
                                tensors: std::iter::once(format!(
                                    "{}__nvfp4_input_scale",
                                    request.tensor
                                ))
                                .collect(),
                                start: 0,
                                shape,
                                stride: None,
                            },

                            provenance: None,
                        },
                        shape,
                        |_| Ok(Cow::Borrowed(&bytes[..])),
                    )?,
                )
            }
        };
        Ok(QuantizedWeight {
            packed,
            scales,
            global_scale,
            layout: product,
            input_scale,
        })
    }

    /// Check that a checkpoint holds one of a projection's NVFP4 fields, naming the export's
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
        if self.checkpoint.tensor(tensor).is_some() {
            return Ok(());
        }
        Err(Error::weight(
            tensor,
            format!(
                "this checkpoint was opened as the {names} spelling of a calibrated NVFP4 \
                 projection, whose {field} is `{suffix}`, and it holds no tensor of this name; the \
                 other export's spelling names the same field differently"
            ),
        ))
    }

    /// Read one tensor's single floating number and answer its reciprocal.
    ///
    /// Both tensor scales a calibrated NVFP4 export states — the weight's `weight_scale_2` and the
    /// activation's `input_scale` — are one positive float each, and every kernel that consumes one
    /// takes the reciprocal. A zero, a negative or a non-finite scale is refused here rather than
    /// encoded at the wrong magnitude.
    ///
    /// The reciprocal is answered as the checkpoint states it. Whether it can encode is
    /// [`Self::encoder_factor`]'s question, and only an operand that *is* the encoder's factor is
    /// asked it: an expert stack publishes its experts' activation scales as the checkpoint's own
    /// data and no operand multiplies a value by them, so bounding those would refuse a checkpoint
    /// over a number nothing reads.
    fn activation_reciprocal(&self, request: &WeightRequest) -> Result<f32> {
        Ok(1.0 / self.tensor_scale(request)?)
    }

    /// The activation encoder's factor for one calibrated projection.
    ///
    /// Not bounded: a calibrated `input_scale` near `1e-3` is a normal calibration (its factor is
    /// then about 724), so a ceiling here would refuse a healthy checkpoint. The encoder's block
    /// scale is computed from the row it actually sees, which is why the bound does not belong to
    /// the operand's scale.
    fn encoder_factor(&self, request: &WeightRequest) -> Result<f32> {
        Ok(1.0 / self.tensor_scale(request)?)
    }

    /// One tensor's single positive scale, as the checkpoint states it.
    ///
    /// Both tensor scales a calibrated NVFP4 export writes — the weight's `weight_scale_2` and the
    /// activation's `input_scale` — are one positive float each, and a loader that synthesized one
    /// would encode at an invented magnitude. A projection reads its weight's scale to build the
    /// product's `alpha` and its activation's to build the encoder's operand, so the reading is
    /// public and the reciprocals are its callers'.
    ///
    /// The tensor must be one floating number by its header on either kind of loader. A meta loader
    /// reads no value and answers a placeholder of one; only a device loader refuses a scale that is
    /// zero, negative or not finite.
    pub fn tensor_scale(&self, request: &WeightRequest) -> Result<f32> {
        let location = self.checkpoint.locate(&request.tensor)?;
        if location.shape.numel() != 1 || !location.dtype.is_float() {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "a tensor scale is one floating number, not {} {:?}",
                    location.dtype,
                    location.shape.dims()
                ),
            ));
        }
        conversion(location.dtype, DType::F32, &request.tensor)?;
        if self.is_meta() {
            return Ok(META_TENSOR_SCALE);
        }
        let tensor = self.checkpoint.read(&request.tensor)?;
        let scale = to_f32(tensor.dtype, &tensor.bytes, &request.tensor)?[0];
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::weight(
                &request.tensor,
                format!("a tensor scale of {scale} is not a positive number"),
            ));
        }
        Ok(scale)
    }

    /// Load the encoded fields of a stack of experts: the experts `slots` names, in that order, of
    /// a vocabulary of `vocabulary` experts, each contributing one projection per module of
    /// `projections` in the order the operator reads them.
    ///
    /// Expert `e`'s projection `p` is the module `<prefix>.<e>.<p>`: the model names the modules, as
    /// its checkpoint does, and the loader reads their fields in the export's spelling `names`.
    /// A stack is a concatenation: slot-major, and within a slot the projections in order (the
    /// gated product reads up first, gate second). The three fields answer the three the operators
    /// take — packed values, block scales, and one tensor scale per **expert and projection** — and
    /// each is published through retained storage under one key, so a second load of the same
    /// checkpoint costs nothing. Slot `s` holds expert `slots[s]`, so a redistribution's order is a
    /// load of the same checkpoint.
    ///
    /// The tensor scales are two per expert for a fused `up`/`gate` stack: the checkpoint gives each
    /// projection its own `weight_scale_2`, and dequantizing the gate half with the up half's scale
    /// would be wrong by the ratio of the two with nothing to show for it. The activation encoder's
    /// scale is the whole vocabulary's: a token is encoded once on the rank that holds it, for
    /// experts on any rank, so every rank encodes with the same number. The second value answered
    /// is each held expert's tensor scales, expert-major, one per projection; a meta loader answers
    /// [`Self::tensor_scale`]'s placeholder for each.
    ///
    /// `name` names the stack in keys and errors and must not collide with a tensor's: a computed
    /// weight needs its own name. A block-scale layout whose per-expert entry is padded, or a row
    /// count that would break the shuffle's blocks, is refused here rather than read as a table no
    /// program expects.
    #[allow(
        clippy::too_many_arguments,
        reason = "the stack is named by its parts: its name, the experts' module, the projections, \
                  the slots, the vocabulary, the export's spelling and the two layouts"
    )]
    pub fn nvfp4_stack(
        &mut self,
        name: &str,
        prefix: &str,
        projections: &[&str],
        slots: &[u32],
        vocabulary: u32,
        names: Nvfp4Names,
        values_layout: ValueLayout,
        layout: ScaleLayout,
        input_scale: bool,
    ) -> Result<(QuantizedStack, Vec<f32>)> {
        let calibrated = input_scale;
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

        // Every field of every expert, by the export's own suffix set. The stack's geometry and
        // types come from the headers; the payload is read only when a field is published.
        let spelling = names;
        let mut fields = Vec::new();
        let mut values_names = Vec::with_capacity(slots.len() * projections.len());
        let mut scales_names = Vec::with_capacity(slots.len() * projections.len());
        let mut rows = None;
        let mut columns = None;
        let mut tensor_scales = Vec::with_capacity(experts as usize);
        let mut activation_reciprocals = Vec::with_capacity(experts as usize * projections.len());
        let mut scale_shape = None;
        let mut scale_dtype = None;
        for expert in slots {
            for projection in projections {
                let stem = format!("{prefix}.{expert}.{projection}");
                let (values_name, scales_name) = (
                    format!("{stem}.{}", spelling.values),
                    format!("{stem}.{}", spelling.block_scale),
                );
                let tensor_name = format!("{stem}.{}", spelling.tensor_scale);
                fields.push(values_name.clone());
                fields.push(scales_name.clone());
                fields.push(tensor_name.clone());
                let activation_name = format!("{stem}.{}", spelling.input_scale);
                if input_scale {
                    fields.push(activation_name.clone());
                }

                self.check_nvfp4_field(
                    &values_name,
                    spelling,
                    "packed E2M1 values",
                    spelling.values,
                )?;
                let values = self.checkpoint.locate(&values_name)?;
                if values.dtype != DType::U8 {
                    return Err(Error::weight(
                        &values_name,
                        format!("packed E2M1 values are U8 bytes, not {}", values.dtype),
                    ));
                }
                let (rows_here, columns_here) = match values.shape.dims() {
                    [rows, packed] => (*rows, packed * 2),
                    other => {
                        return Err(Error::weight(
                            &values_name,
                            format!("a packed matrix is two axes, not {other:?}"),
                        ));
                    }
                };
                // The stack's geometry is the first expert's; every other projection must agree,
                // because one set of views serves them all.
                let agreed = |seen: &mut Option<u64>, here: u64, what: &str| -> Result<()> {
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
                    Ok(())
                };
                agreed(&mut rows, rows_here, "rows")?;
                agreed(&mut columns, columns_here, "columns")?;

                let scales = self.checkpoint.locate(&scales_name)?;
                let here = block_scale_dtype(&scales_name, scales.dtype)?;
                match scale_dtype {
                    None => scale_dtype = Some(here),
                    Some(agreed) if agreed != here => {
                        return Err(Error::weight(
                            &scales_name,
                            format!(
                                "block scales are {here} here against the stack's {agreed}: one \
                                 stack is one scale type"
                            ),
                        ));
                    }
                    Some(_) => {}
                }
                // The shard's table is the checkpoint's own arrangement — one block scale per
                // sixteen elements, row-major — and the published one is the deployment's. A
                // converted export that already holds the padded, interleaved table is refused here
                // rather than published after a second pass over it.
                let expected = ScaleLayout::Linear.shape(rows_here, columns_here);
                if scales.shape != expected {
                    return Err(Error::weight(
                        &scales_name,
                        format!(
                            "a checkpoint's block scales of a [{rows_here}, {columns_here}] matrix \
                             are linear {:?}, not {:?}: the published {layout} table is this \
                             loader's to build, not the export's to state",
                            expected.dims(),
                            scales.shape.dims()
                        ),
                    ));
                }

                let tensor = WeightRequest::region(&tensor_name, 0, Shape::new(&[1]));
                tensor_scales.push(self.tensor_scale(&tensor)?);
                if input_scale {
                    let request = WeightRequest::region(&activation_name, 0, Shape::new(&[1]));
                    activation_reciprocals.push(self.activation_reciprocal(&request)?);
                }

                values_names.push(values_name);
                scales_names.push(scales_name);
                scale_shape = Some(expected);
            }
        }

        let rows = rows.unwrap_or(0) * per;
        let columns = columns.unwrap_or(0);
        let shard_scales = ScaleLayout::Linear.shape(rows, columns);
        let entry = scale_shape.map(|shape: Shape| shape.numel()).unwrap_or(0);
        if shard_scales.numel() != entry * per {
            return Err(Error::weight(
                name,
                format!(
                    "a linear entry for {rows} rows of {columns} elements is {} block scales, and \
                     {per} concatenated entries hold {entry} each",
                    shard_scales.numel() / per.max(1)
                ),
            ));
        }

        // The values' row arrangement: a plain shard needs nothing, a shuffled one is permuted here
        // once, at load. A product whose A operand is read with `useShuffledMatrix` cannot read a
        // shard's own row order, so a row count that is not a whole number of shuffle blocks has no
        // arrangement at all and is refused by name rather than published in an order nothing reads.
        if let Some(block) = values_layout.block()
            && !rows.is_multiple_of(block)
        {
            return Err(Error::weight(
                name,
                format!(
                    "a {values_layout:?} arrangement permutes {block}-row blocks, and this stack's                      matrices have {rows} rows"
                ),
            ));
        }

        let values_shape = Shape::new(&[count, rows, columns / 2]);
        let scales_shape = Shape::new(&[
            count,
            layout.shape(rows, columns).dim(0),
            layout.shape(rows, columns).dim(1),
        ]);
        // `tensor_scales` was collected expert-major, in the checkpoint's own reading order; the
        // published table is projection-major, one packed row per projection.
        let projections_shape = Shape::new(&[per, count]);
        let strides = Projections::new(per, count);
        if tensor_scales.len() != strides.len() {
            return Err(Error::weight(
                name,
                format!(
                    "{per} projections of {count} experts are {} scales, and the checkpoint gave {}",
                    strides.len(),
                    tensor_scales.len()
                ),
            ));
        }
        let scale_bytes: Vec<u8> = strides.bytes(&tensor_scales);
        let files = self.identities(&fields)?;

        let values = self.publish_computed(
            WeightKey {
                files: files.clone(),
                dtype: DType::U8,
                layout: values_layout.layout(),
                region: WeightRegion {
                    tensors: std::iter::once(format!("{name}__nvfp4_values"))
                        .chain(fields.iter().cloned())
                        .collect(),
                    start: 0,
                    shape: values_shape,
                    stride: None,
                },

                provenance: None,
            },
            values_shape,
            |loader| {
                let mut bytes = Vec::with_capacity(values_shape.numel() as usize);
                for values_name in &values_names {
                    bytes.extend_from_slice(&loader.checkpoint.read(values_name)?.bytes);
                }
                Ok(Cow::Owned(values_layout.arrange(
                    &bytes,
                    count,
                    rows,
                    (columns / 2).max(1),
                )))
            },
        )?;
        // The stack's scale type is the checkpoint's own: an E4M3 field stays E4M3, and the model
        // puts it in the descriptor unchanged.
        let scale_dtype = scale_dtype.unwrap_or(DType::U8);
        let scales = self.publish_computed(
            WeightKey {
                files: files.clone(),
                dtype: scale_dtype,
                layout: match layout {
                    ScaleLayout::Linear => WeightLayout::NVFP4_SCALES_LINEAR,
                    ScaleLayout::Swizzled128X4 => WeightLayout::NVFP4_SCALES_SWIZZLED,
                },
                region: WeightRegion {
                    tensors: std::iter::once(format!("{name}__nvfp4_scales"))
                        .chain(fields.iter().cloned())
                        .collect(),
                    start: 0,
                    shape: scales_shape,
                    stride: None,
                },

                provenance: None,
            },
            scales_shape,
            |loader| {
                let blocks = ScaleLayout::Linear.columns(columns);
                let mut scales_bytes = Vec::with_capacity((count * rows * blocks) as usize);
                for scales_name in &scales_names {
                    scales_bytes.extend_from_slice(&loader.checkpoint.read(scales_name)?.bytes);
                }
                // One expert's linear table becomes the published arrangement: rows permuted the
                // way the values' rows were when the values are shuffled (the scales are read
                // beside the row), then padded and interleaved when the deployment's kernels read
                // the swizzled 128x4 table.
                let mut arranged = Vec::with_capacity(scales_shape.numel() as usize);
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
            WeightKey {
                files,
                dtype: DType::F32,
                layout: WeightLayout::ROW_MAJOR,
                region: WeightRegion {
                    tensors: std::iter::once(format!("{name}__nvfp4_tensor_scales"))
                        .chain(fields.iter().cloned())
                        .collect(),
                    start: 0,
                    shape: projections_shape,
                    stride: None,
                },

                provenance: None,
            },
            projections_shape,
            |_| Ok(Cow::Borrowed(&scale_bytes[..])),
        )?;
        let input_scale = if input_scale {
            let bytes = strides.bytes(&activation_reciprocals);
            let shape = projections_shape;
            let files = self.identities(
                &fields
                    .iter()
                    .filter(|field| field.ends_with(&format!(".{}", spelling.input_scale)))
                    .cloned()
                    .collect::<Vec<_>>(),
            )?;
            Some(self.publish_computed(
                WeightKey {
                    files,
                    dtype: DType::F32,
                    layout: WeightLayout::ROW_MAJOR,
                    region: WeightRegion {
                        tensors: std::iter::once(format!("{name}__nvfp4_input_scales")).collect(),
                        start: 0,
                        shape,
                        stride: None,
                    },

                    provenance: None,
                },
                shape,
                |_| Ok(Cow::Borrowed(&bytes[..])),
            )?)
        } else {
            None
        };
        // One encoder serves the whole stack, so it takes the widest of its experts' calibrated
        // activation scales, whose reciprocal is the smallest: no expert's block scale saturates.
        let mut encode = activation_reciprocals
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        if calibrated {
            for expert in (0..vocabulary).filter(|expert| !slots.contains(expert)) {
                for projection in projections {
                    let stem = format!("{prefix}.{expert}.{projection}");
                    let tensor = format!("{stem}.{}", spelling.input_scale);
                    let request = WeightRequest::region(&tensor, 0, Shape::new(&[1]));
                    encode = encode.min(self.activation_reciprocal(&request)?);
                }
            }
        }
        let encode = if encode.is_finite() { encode } else { 1.0 };
        let encode_scale = self.constant(
            &format!("{name}__nvfp4_encode_scale"),
            DType::F32,
            Shape::new(&[1]),
            &encode.to_ne_bytes(),
            "nn",
        )?;
        Ok((
            QuantizedStack {
                values,
                scales,
                global_scale,
                layout,
                encode_scale,
                input_scale,
                activation_scale: encode.recip(),
            },
            tensor_scales,
        ))
    }

    /// Publish bytes this loader computes, under a key it built, and answer the weight.
    ///
    /// A key already held is answered without calling `bytes`, and a meta loader never calls it.
    fn publish_computed<'c>(
        &mut self,
        key: WeightKey,
        shape: Shape,
        bytes: impl FnOnce(&Self) -> Result<Cow<'c, [u8]>>,
    ) -> Result<Weight> {
        if let Some(weight) = self.retained(&key, shape)? {
            return Ok(weight);
        }
        self.publish_with(&key, shape, false, bytes)
    }

    /// Publish host-computed bytes as a weight, without a checkpoint tensor behind them.
    ///
    /// The rotary frequency table is the first such weight: `inv_freq` is computed once at loading
    /// from `rope_theta` and `rope_scaling`, and the `rotary_factors` operator
    /// reads it as a device weight, so it takes the same retained-storage path as a checkpoint
    /// weight. There is no file to name it, so the key states the **provenance**: the class that
    /// computed it, `producer`, and a digest of the bytes it produced. Two computations of the same
    /// numbers share one retained region; two that compute different numbers under one name never
    /// do. A name alone is not an identity — it is a label the caller chose, and two models that
    /// choose the same label for their own constants are exactly the collision this prevents.
    pub fn constant(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
        producer: &'static str,
    ) -> Result<Weight> {
        self.computed(name, dtype, shape, bytes, producer, false)
    }

    /// A table the model writes once after the load, holding `bytes` until then.
    ///
    /// It is a [`Self::constant`] that stays writable on the device, and it is this process's own:
    /// no later load reads it back, because the launch that fills it runs in every load.
    pub fn writable_constant(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
        producer: &'static str,
    ) -> Result<Weight> {
        self.computed(name, dtype, shape, bytes, producer, true)
    }

    fn computed(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
        producer: &'static str,
        writable: bool,
    ) -> Result<Weight> {
        let element = dtype.bytes().ok_or_else(|| {
            Error::weight(name, format!("{dtype} has no whole-byte element size"))
        })?;
        let expected = shape
            .numel()
            .checked_mul(element)
            .ok_or_else(|| Error::weight(name, "the constant's bytes overflow"))?;
        if bytes.len() as u64 != expected {
            return Err(Error::weight(
                name,
                format!(
                    "{} bytes for {:?} of {dtype}, {expected} expected",
                    bytes.len(),
                    shape.dims()
                ),
            ));
        }
        let key = WeightKey {
            files: Vec::new(),
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            region: WeightRegion {
                tensors: vec![name.to_owned()],
                start: 0,
                shape,
                stride: None,
            },
            provenance: Some(Provenance::computed(producer, bytes)),
        };
        if writable {
            self.publish_with(&key, shape, true, |_| Ok(Cow::Borrowed(bytes)))
        } else {
            self.publish_computed(key, shape, |_| Ok(Cow::Borrowed(bytes)))
        }
    }

    /// The identity of each of `names`' files, in order.
    fn identities(&self, names: &[String]) -> Result<Vec<FileIdentity>> {
        names
            .iter()
            .map(|name| {
                self.checkpoint
                    .identity(name)
                    .cloned()
                    .ok_or_else(|| Error::MissingTensor(name.clone()))
            })
            .collect()
    }

    /// The element type every named tensor has in the checkpoint.
    fn checkpoint_dtype(&self, names: &[String]) -> Result<DType> {
        let mut dtype = None;
        for name in names {
            let location = self
                .checkpoint
                .tensor(name)
                .ok_or_else(|| Error::MissingTensor(name.clone()))?;
            match dtype {
                None => dtype = Some(location.dtype),
                Some(seen) if seen != location.dtype => {
                    return Err(Error::weight(
                        name,
                        format!(
                            "the fused parts mix {seen} and {source} elements",
                            source = location.dtype
                        ),
                    ));
                }
                Some(_) => {}
            }
        }
        dtype.ok_or_else(|| Error::weight("", "a weight has no source tensor"))
    }

    /// The element type the weight is retained as.
    fn target_dtype(&self, names: &[String], pinned: Option<DType>) -> Result<DType> {
        let checkpoint = self.checkpoint_dtype(names)?;
        // A pin is the model's statement about this tensor; the policy is the deployment's about
        // every tensor it reads. The pin wins, because a representation the reference fixes at
        // every preset is not a property of the deployment.
        let dtype = pinned.or(self.dtype).unwrap_or(checkpoint);
        // A representation that is not converted (BF16 kept as BF16, a packed NVFP4 field of
        // bytes kept as bytes) is read as it is; only a stated change goes through `convert`.
        if dtype == checkpoint {
            return Ok(dtype);
        }
        if !dtype.is_float() || !checkpoint.is_float() {
            return Err(Error::weight(
                names.first().map(String::as_str).unwrap_or(""),
                format!("converting {checkpoint} to {dtype} is not a floating-point conversion"),
            ));
        }
        Ok(dtype)
    }

    /// The byte ranges a source names, in the source's logical order, planned from the headers.
    ///
    /// Every bound is checked against the length the header states for each tensor, which is the
    /// length of the payload slice a device load copies from.
    fn spans(&self, source: &WeightSource, key: &WeightKey) -> Result<Vec<Span<'a>>> {
        let checkpoint = self.checkpoint;
        spans_in(source, key, &|name| checkpoint.tensor(name))
    }

    /// The payload bytes of `spans`, concatenated in order.
    ///
    /// Consecutive spans of one tensor are read in one [`Checkpoint::payload`] call, so a strided
    /// shard of an adapter-fused tensor, one span a row, reads the adapter's factors once.
    fn gather(&self, spans: &[Span<'_>]) -> Result<Vec<u8>> {
        let total = spans.iter().map(|span| span.to - span.from).sum();
        let mut gathered = Vec::with_capacity(total);
        let mut at = 0;
        while at < spans.len() {
            let location = spans[at].location;
            let run = spans[at..]
                .iter()
                .take_while(|span| std::ptr::eq(span.location, location))
                .count();
            let ranges = spans[at..at + run]
                .iter()
                .map(|span| (span.from, span.to))
                .collect::<Vec<_>>();
            gathered.extend(self.checkpoint.payload(location, &ranges)?);
            at += run;
        }
        Ok(gathered)
    }
}

/// One module's view of a loader: the module's path, and the tensors named under it.
///
/// A layer loads itself through the scope its parent hands it: [`Scope::tensor`] reads
/// `<path>.<leaf>`, and [`Scope::scope`] gives a child the path `<path>.<child>`, so a checkpoint
/// name is composed exactly as `named_parameters()` composes it and no model writes a name out.
/// A scope may pin its tensors to one element type ([`Scope::pinned`]), which its children inherit.
///
/// A scope reads dense weights. A weight the checkpoint stores packed has a layer that can read it
/// ([`crate::layers::Linear`], which loads its packed fields itself); any other layer that asks
/// for one is refused by name, because reading packed E2M1 as a matrix makes a model that runs and
/// computes nothing.
pub struct Scope<'s, 'a> {
    loader: &'s mut WeightLoader<'a>,
    path: String,
    pinned: Option<DType>,
}

impl<'s, 'a> Scope<'s, 'a> {
    /// The module path this scope names its tensors under.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The full checkpoint name of this module's `leaf`.
    pub fn name(&self, leaf: &str) -> String {
        if self.path.is_empty() {
            leaf.to_owned()
        } else {
            format!("{}.{leaf}", self.path)
        }
    }

    /// The scope of the child module `child`, which inherits this scope's pin.
    pub fn scope(&mut self, child: impl fmt::Display) -> Scope<'_, 'a> {
        let path = self.name(&child.to_string());
        Scope {
            loader: &mut *self.loader,
            path,
            pinned: self.pinned,
        }
    }

    /// This scope with every tensor read as `dtype`, whatever the loader's policy states.
    ///
    /// A representation a model pins per module — the reference keeps a timestep embedding in
    /// FP32 and a VAE decoder's projections in FP16 at every preset — is the module's statement, and
    /// the pin is part of each weight's retained key.
    pub fn pinned(&mut self, dtype: DType) -> Scope<'_, 'a> {
        Scope {
            loader: &mut *self.loader,
            path: self.path.clone(),
            pinned: Some(dtype),
        }
    }

    /// Whether the checkpoint holds this module's `leaf`.
    pub fn contains(&self, leaf: &str) -> bool {
        self.loader.checkpoint.tensor(&self.name(leaf)).is_some()
    }

    /// The whole of this module's `leaf`, `shape` elements read from its first.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the tensor, holds another extent, stores it packed, or cannot
    /// convert it to the pinned type.
    pub fn tensor(&mut self, leaf: &str, shape: &[u64]) -> Result<Weight> {
        self.rows(leaf, 0, shape)
    }

    /// The `shape.dim(0)` rows of this module's `leaf` that start at row `first`: a contiguous
    /// window of a tensor whose rows are `shape[1..]`.
    ///
    /// # Errors
    ///
    /// As [`Self::tensor`], and when the window reaches past the tensor.
    pub fn rows(&mut self, leaf: &str, first: u64, shape: &[u64]) -> Result<Weight> {
        let row: u64 = shape.iter().skip(1).product();
        let request = WeightRequest::region(self.name(leaf), first * row, Shape::new(shape));
        self.load(request)
    }

    /// Fold this module's weight-normalized `leaf`, stored as `<leaf>_v` and `<leaf>_g`, into one
    /// weight ([`WeightLoader::weight_norm`]), written at the scope's pin.
    ///
    /// # Errors
    ///
    /// As [`WeightLoader::weight_norm`].
    pub fn weight_norm(&mut self, leaf: &str) -> Result<Weight> {
        let name = self.name(leaf);
        self.refuse_packed(std::slice::from_ref(&name))?;
        let (value, gain) = (format!("{name}_v"), format!("{name}_g"));
        self.loader.fold(&name, &value, &gain, self.pinned)
    }

    /// A table this module computes rather than reads: `values` encoded as `dtype`
    /// ([`DType::encode`], which rounds to nearest even), published as `<path>.<leaf>`.
    ///
    /// The module path names the constant, and the retained key's provenance digests the bytes
    /// ([`WeightLoader::constant`]), so two modules' constants never share a name and one module's
    /// two computations of different numbers never share storage.
    ///
    /// # Errors
    ///
    /// When `values` is not `shape`'s element count or `dtype` has no host encoding.
    pub fn constant(
        &mut self,
        leaf: &str,
        dtype: DType,
        shape: &[u64],
        values: &[f32],
    ) -> Result<Weight> {
        let bytes = dtype.encode(values).map_err(Error::from)?;
        self.publish(leaf, dtype, shape, &bytes)
    }

    /// An FP32 table this module computes, as [`Self::constant`] publishes it.
    ///
    /// # Errors
    ///
    /// As [`Self::constant`].
    pub fn constant_f32(&mut self, leaf: &str, shape: &[u64], values: &[f32]) -> Result<Weight> {
        self.constant(leaf, DType::F32, shape, values)
    }

    /// An I32 table this module computes, as [`Self::constant`] publishes it.
    ///
    /// # Errors
    ///
    /// When `values` is not `shape`'s element count.
    pub fn constant_i32(&mut self, leaf: &str, shape: &[u64], values: &[i32]) -> Result<Weight> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        self.publish(leaf, DType::I32, shape, &bytes)
    }

    /// The I32 table `[start, start + count)`: a gather's or a scatter's index row.
    ///
    /// # Errors
    ///
    /// When the range leaves I32.
    pub fn indices(&mut self, leaf: &str, start: i32, count: u64) -> Result<Weight> {
        let end = i64::from(start) + i64::try_from(count).unwrap_or(i64::MAX);
        let end = i32::try_from(end).map_err(|_| {
            Error::weight(&self.name(leaf), format!("{count} indices from {start}"))
        })?;
        let values: Vec<i32> = (start..end).collect();
        self.constant_i32(leaf, &[count], &values)
    }

    /// A table of zeros of `dtype`.
    ///
    /// # Errors
    ///
    /// When `dtype` has no whole-byte element size.
    pub fn zeros(&mut self, leaf: &str, dtype: DType, shape: &[u64]) -> Result<Weight> {
        let bytes = self.zero_bytes(leaf, dtype, shape)?;
        self.publish(leaf, dtype, shape, &bytes)
    }

    /// A table of zeros of `dtype` this module's launches write once after the load
    /// ([`WeightLoader::writable_constant`]): it stays writable on the device and is this process's
    /// own.
    ///
    /// # Errors
    ///
    /// As [`Self::zeros`].
    pub fn writable_zeros(&mut self, leaf: &str, dtype: DType, shape: &[u64]) -> Result<Weight> {
        let bytes = self.zero_bytes(leaf, dtype, shape)?;
        self.loader.writable_constant(
            &self.name(leaf),
            dtype,
            Shape::new(shape),
            &bytes,
            "nn::weights::Scope",
        )
    }

    /// The bytes of `shape` zeros of `dtype`: every supported type's zero is the all-zero pattern.
    fn zero_bytes(&self, leaf: &str, dtype: DType, shape: &[u64]) -> Result<Vec<u8>> {
        let element = dtype.bytes().ok_or_else(|| {
            Error::weight(
                &self.name(leaf),
                format!("{dtype} has no whole-byte element"),
            )
        })?;
        Ok(vec![0; (Shape::new(shape).numel() * element) as usize])
    }

    /// Publish `bytes` as this module's constant `leaf`.
    fn publish(&mut self, leaf: &str, dtype: DType, shape: &[u64], bytes: &[u8]) -> Result<Weight> {
        self.loader.constant(
            &self.name(leaf),
            dtype,
            Shape::new(shape),
            bytes,
            "nn::weights::Scope",
        )
    }

    /// Load `request` at this scope's pin, refusing a packed source.
    pub(crate) fn load(&mut self, request: WeightRequest) -> Result<Weight> {
        let names = source_names(&request.source);
        // A weight-only NVFP4 matrix is read as the dense matrix it states, by every layer.
        if !names.is_empty()
            && names
                .iter()
                .all(|name| self.loader.scheme(name) == Scheme::Nvfp4Weights)
        {
            return self.loader.load_decoded(&request);
        }
        self.refuse_packed(&names)?;
        let request = match self.pinned {
            Some(dtype) => request.pinned(dtype),
            None => request,
        };
        self.loader.load(&request)
    }

    /// The loader, for a layer that reads its own packed fields.
    pub(crate) fn loader(&mut self) -> &mut WeightLoader<'a> {
        self.loader
    }

    /// Refuse a dense read of a weight the checkpoint stores packed.
    fn refuse_packed(&self, names: &[String]) -> Result<()> {
        match names
            .iter()
            .find(|name| self.loader.scheme(name) != Scheme::Dense)
        {
            None => Ok(()),
            Some(name) => Err(Error::weight(
                name,
                "the checkpoint stores this weight packed, and the layer reading it has no \
                 encoded form: reading it as dense would read packed E2M1 as a matrix. Load a \
                 checkpoint whose calibration excludes it",
            )),
        }
    }
}

/// The checkpoint tensors `source` reads, each once, in the order it first names them.
fn source_names(source: &WeightSource) -> Vec<String> {
    match source {
        WeightSource::Region { tensor, .. } | WeightSource::Strided { tensor, .. } => {
            vec![tensor.clone()]
        }
        WeightSource::Parts { parts, .. } => parts.clone(),
        WeightSource::Blocks { parts, .. } => {
            let mut names: Vec<String> = Vec::new();
            for (name, _) in parts {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
            names
        }
    }
}

/// The first element, the extents and the row stride that key `source`'s region.
///
/// A block's start and stride key the region: two ranks' blocks of one expert stack have one fused
/// shape and must not share retained storage, and two ranks' blocks of one stacked tensor begin at
/// different rows.
fn source_region(source: &WeightSource) -> (u64, Shape, Option<u64>) {
    match source {
        WeightSource::Region { start, shape, .. } => (*start, *shape, None),
        WeightSource::Strided {
            start,
            shape,
            row_stride,
            ..
        } => (*start, *shape, Some(*row_stride)),
        WeightSource::Parts { shape, block, .. } => (
            block.map_or(0, |block| block.start),
            *shape,
            block.map(|block| block.row_stride),
        ),
        WeightSource::Blocks { parts, shape } => {
            let first = parts.first().map(|(_, block)| *block);
            (
                first.map_or(0, |block| block.start),
                *shape,
                first.map(|block| block.row_stride),
            )
        }
    }
}

/// The byte ranges `source` names, in its logical order, over the tensors `locate` answers: the
/// checkpoint's own, or the decoded matrices a weight-only NVFP4 read gathers from.
///
/// Every bound is checked against the length each location states.
fn spans_in<'l>(
    source: &WeightSource,
    key: &WeightKey,
    locate: &dyn Fn(&str) -> Option<&'l TensorLocation>,
) -> Result<Vec<Span<'l>>> {
    match source {
        WeightSource::Region {
            tensor,
            start,
            shape,
        } => {
            let location = locate(tensor).ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
            let length = stored_bytes(location);
            let element = location.dtype.bytes().ok_or_else(|| {
                Error::weight(tensor, format!("{} has no whole-byte size", location.dtype))
            })?;
            let begin = usize::try_from(start * element).map_err(|_| {
                Error::weight(tensor, "the region's first element is past usize::MAX")
            })?;
            let end = usize::try_from((start + shape.numel()) * element).map_err(|_| {
                Error::weight(tensor, "the region's last element is past usize::MAX")
            })?;
            if end as u64 > length {
                return Err(Error::weight(
                    tensor,
                    format!(
                        "elements {start}..{} of {:?} reach past the tensor's {length}",
                        start + shape.numel(),
                        location.shape.dims(),
                    ),
                ));
            }
            Ok(vec![Span {
                location,
                from: begin,
                to: end,
            }])
        }
        WeightSource::Strided {
            tensor,
            start,
            shape,
            row_stride,
        } => {
            let location = locate(tensor).ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
            let element = location.dtype.bytes().ok_or_else(|| {
                Error::weight(tensor, format!("{} has no whole-byte size", location.dtype))
            })?;
            if shape.rank() == 0 {
                return Err(Error::weight(
                    tensor,
                    "a gathered weight has an axis".to_owned(),
                ));
            }
            let row = shape.last();
            if *row_stride < row {
                return Err(Error::weight(
                    tensor,
                    format!("a stride of {row_stride} overlaps rows of {row} elements"),
                ));
            }
            let rows = shape.numel() / row.max(1);
            let length = stored_bytes(location);
            let mut spans = Vec::with_capacity(usize::try_from(rows).unwrap_or(0));
            for index in 0..rows {
                let begin = start + index * row_stride;
                let from = usize::try_from(begin * element)
                    .map_err(|_| Error::weight(tensor, "a gathered row starts past usize::MAX"))?;
                let to = usize::try_from((begin + row) * element)
                    .map_err(|_| Error::weight(tensor, "a gathered row ends past usize::MAX"))?;
                if to as u64 > length {
                    return Err(Error::weight(
                        tensor,
                        format!(
                            "row {index} (elements {begin}..{}) reaches past the tensor's {length} bytes",
                            begin + row,
                        ),
                    ));
                }
                spans.push(Span { location, from, to });
            }
            Ok(spans)
        }
        WeightSource::Blocks { parts, .. } => {
            let mut spans = Vec::new();
            for (name, block) in parts {
                spans.extend(block_spans(locate, name, block)?);
            }
            if spans.iter().all(|span| span.from == span.to) {
                return Err(Error::weight(
                    &key.region.tensors.join("+"),
                    "a fused weight has no part",
                ));
            }
            Ok(spans)
        }
        WeightSource::Parts {
            parts,
            block: Some(block),
            ..
        } => {
            let mut spans = Vec::new();
            for name in parts {
                let location = locate(name).ok_or_else(|| Error::MissingTensor(name.clone()))?;
                let element = location.dtype.bytes().ok_or_else(|| {
                    Error::weight(name, format!("{} has no whole-byte size", location.dtype))
                })?;
                if block.row_stride < block.columns {
                    return Err(Error::weight(
                        name,
                        format!(
                            "a stride of {} overlaps rows of {} elements",
                            block.row_stride, block.columns
                        ),
                    ));
                }
                let length = stored_bytes(location);
                for row in 0..block.rows {
                    let first = (block.start + row * block.row_stride) * element;
                    let last = first + block.columns * element;
                    let (Ok(from), Ok(to)) = (usize::try_from(first), usize::try_from(last)) else {
                        return Err(Error::weight(name, "a block row is past usize::MAX"));
                    };
                    if to as u64 > length {
                        return Err(Error::weight(
                            name,
                            format!(
                                "row {row} of the block (bytes {from}..{to}) reaches past the \
                                 part's {length} bytes"
                            ),
                        ));
                    }
                    spans.push(Span { location, from, to });
                }
            }
            if spans.iter().all(|span| span.from == span.to) {
                return Err(Error::weight(
                    &key.region.tensors.join("+"),
                    "a fused weight has no part",
                ));
            }
            Ok(spans)
        }
        WeightSource::Parts { parts, .. } => {
            let mut spans = Vec::with_capacity(parts.len());
            let mut part_shape: Option<Shape> = None;
            for name in parts {
                let location = locate(name).ok_or_else(|| Error::MissingTensor(name.clone()))?;
                if location.shape.rank() == 0 {
                    return Err(Error::weight(name, "a scalar cannot be fused"));
                }
                match part_shape {
                    None => part_shape = Some(location.shape),
                    Some(first) => {
                        let same_tail = first.dims().iter().skip(1).eq(location
                            .shape
                            .dims()
                            .iter()
                            .skip(1));
                        if !same_tail {
                            return Err(Error::weight(
                                name,
                                format!(
                                    "fusing {:?} with {:?} joins different tails",
                                    first.dims(),
                                    location.shape.dims()
                                ),
                            ));
                        }
                    }
                }
                let to = usize::try_from(stored_bytes(location))
                    .map_err(|_| Error::weight(name, "a part is past usize::MAX"))?;
                spans.push(Span {
                    location,
                    from: 0,
                    to,
                });
            }
            if spans.iter().all(|span| span.from == span.to) {
                return Err(Error::weight(
                    &key.region.tensors.join("+"),
                    "a fused weight has no part",
                ));
            }
            Ok(spans)
        }
    }
}

/// The byte ranges of `block`'s rows in the checkpoint tensor `name`, checked against its length.
fn block_spans<'a>(
    locate: &dyn Fn(&str) -> Option<&'a TensorLocation>,
    name: &str,
    block: &PartBlock,
) -> Result<Vec<Span<'a>>> {
    let location = locate(name).ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    let element = location
        .dtype
        .bytes()
        .ok_or_else(|| Error::weight(name, format!("{} has no whole-byte size", location.dtype)))?;
    if block.row_stride < block.columns {
        return Err(Error::weight(
            name,
            format!(
                "a stride of {} overlaps rows of {} elements",
                block.row_stride, block.columns
            ),
        ));
    }
    let length = stored_bytes(location);
    let mut spans = Vec::with_capacity(usize::try_from(block.rows).unwrap_or(0));
    for row in 0..block.rows {
        let first = (block.start + row * block.row_stride) * element;
        let last = first + block.columns * element;
        let (Ok(from), Ok(to)) = (usize::try_from(first), usize::try_from(last)) else {
            return Err(Error::weight(name, "a block row is past usize::MAX"));
        };
        if to as u64 > length {
            return Err(Error::weight(
                name,
                format!(
                    "row {row} of the block (bytes {from}..{to}) reaches past the part's {length} \
                     bytes"
                ),
            ));
        }
        spans.push(Span { location, from, to });
    }
    Ok(spans)
}

/// Verify the loader can produce `layout` from `source`.
fn check_layout(tensor: &str, source: &WeightSource, layout: WeightLayout) -> Result<()> {
    match layout {
        WeightLayout::ROW_MAJOR => Ok(()),
        WeightLayout::UP_GATE | WeightLayout::DOWN => match source {
            WeightSource::Parts { .. } | WeightSource::Blocks { .. } => Ok(()),
            WeightSource::Region { .. } | WeightSource::Strided { .. } => Err(Error::weight(
                tensor,
                format!(
                    "layout {layout} fuses several projections; a single region cannot produce it"
                ),
            )),
        },
        // The NVFP4 fields are byte buffers: a packed matrix, a fused stack of packed experts or
        // the strided rows of either are all the same bytes to the product.
        WeightLayout::NVFP4_VALUES
        | WeightLayout::NVFP4_SCALES_LINEAR
        | WeightLayout::NVFP4_SCALES_SWIZZLED => Ok(()),
        other => Err(Error::weight(
            tensor,
            format!("layout `{other}` is not one this loader produces"),
        )),
    }
}

/// Convert the floating-point elements of `bytes` from `from` to `to`.
///
/// The checkpoint is stored in one of the three floating-point formats every model here uses, and
/// a conversion goes through `f32`: the design keeps a checkpoint's own representation unless the
/// launch states another, and an explicit statement is one conversion.
///
/// It is public because a boundary between two computations can need exactly one conversion and no
/// loader: a model that hands a decoder a tensor its own layers produced states the pair it needs,
/// and `tensor` is what the refusal names when the pair is not one of the six. A caller that
/// converts on a path a launch did not state is widening the rule above, so the conversion belongs
/// where the boundary itself is stated.
pub fn convert(from: DType, to: DType, bytes: &[u8], tensor: &str) -> Result<Vec<u8>> {
    conversion(from, to, tensor)?;
    if from == to {
        return Ok(bytes.to_vec());
    }
    let source_bytes = usize::try_from(from.bytes().unwrap_or(4)).unwrap_or(4);
    let target_bytes = usize::try_from(to.bytes().unwrap_or(4)).unwrap_or(4);
    if !bytes.len().is_multiple_of(source_bytes) {
        return Err(Error::weight(
            tensor,
            format!("{} bytes are not whole {from} elements", bytes.len()),
        ));
    }
    let count = bytes.len() / source_bytes;
    let mut converted = Vec::with_capacity(count * target_bytes);
    for index in 0..count {
        let chunk = &bytes[index * source_bytes..][..source_bytes];
        let value = match from {
            DType::BF16 => bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            DType::F16 => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            DType::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            // A PCM sample is an integer that F32 holds exactly, so the read is lossless.
            DType::I16 => f32::from(i16::from_le_bytes([chunk[0], chunk[1]])),
            other => {
                return Err(Error::weight(
                    tensor,
                    format!("converting {other} is not a floating-point conversion"),
                ));
            }
        };
        match to {
            // A weight narrowed to BF16 at load keeps its top sixteen bits: the loader truncates where
            // `DType::encode` and `torch` round to nearest. Served weights were narrowed this way, so
            // the rule stands until a measured run replaces it.
            DType::BF16 => {
                converted.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes())
            }
            DType::F16 => converted.extend_from_slice(&f32_to_f16(value).to_le_bytes()),
            DType::F32 => converted.extend_from_slice(&value.to_le_bytes()),
            DType::I16 => converted.extend_from_slice(&f32_to_i16(value).to_le_bytes()),
            other => {
                return Err(Error::weight(
                    tensor,
                    format!("converting to {other} is not a floating-point conversion"),
                ));
            }
        }
    }
    Ok(converted)
}

/// Whether [`convert`] converts `from` to `to`: one of its pairs, or the same type.
///
/// A load decides this from the element types alone, before any byte is read, so a meta load
/// refuses the pairs a device load refuses.
fn conversion(from: DType, to: DType, tensor: &str) -> Result<()> {
    match (from, to) {
        (DType::BF16, DType::F32)
        | (DType::BF16, DType::F16)
        | (DType::F16, DType::F32)
        | (DType::F16, DType::BF16)
        | (DType::F32, DType::BF16)
        | (DType::F32, DType::F16)
        | (DType::F32, DType::I16)
        | (DType::I16, DType::F32) => Ok(()),
        (same, target) if same == target => Ok(()),
        (source, target) => Err(Error::weight(
            tensor,
            format!("converting {source} to {target} is not implemented"),
        )),
    }
}

/// The `i16` a sample becomes: nearest, ties to even, saturating at both ends.
///
/// A PCM track is a clip, not a wrap: a sample beyond the range stores at the end of it rather
/// than folding to the opposite sign, and a NaN stores as zero (the saturating cast's answer, and a
/// NaN in a checkpoint is already corrupt).
fn f32_to_i16(value: f32) -> i16 {
    let rounded = value.round_ties_even();
    if rounded >= f32::from(i16::MAX) {
        i16::MAX
    } else if rounded <= f32::from(i16::MIN) {
        i16::MIN
    } else {
        rounded as i16
    }
}

/// The `f32` values of `bytes`, which hold `dtype` elements.
fn to_f32(dtype: DType, bytes: &[u8], tensor: &str) -> Result<Vec<f32>> {
    let converted = convert(dtype, DType::F32, bytes, tensor)?;
    Ok(converted
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// `values` as `dtype` elements.
fn from_f32(dtype: DType, values: &[f32], tensor: &str) -> Result<Vec<u8>> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    convert(DType::F32, dtype, &bytes, tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write one shard of `tensors` - each `(name, dtype, shape, bytes)` - into `directory`.
    ///
    /// The header is composed from the format's own grammar here, rather than through the parser a
    /// test is meant to exercise.
    fn shard(directory: &std::path::Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
        let mut body = String::from("{");
        let mut payload = Vec::new();
        for (index, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if index > 0 {
                body.push(',');
            }
            let start = payload.len();
            payload.extend_from_slice(bytes);
            let extents = shape
                .iter()
                .map(|extent| extent.to_string())
                .collect::<Vec<_>>()
                .join(",");
            body.push_str(&format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{extents}],"data_offsets":[{start},{}]}}"#,
                payload.len()
            ));
        }
        body.push('}');
        // The header length is a multiple of eight so the payload stays aligned, as every exporter
        // writes it; JSON permits the spaces that padding needs.
        let padded = body.len().div_ceil(8) * 8;
        body.push_str(&" ".repeat(padded - body.len()));

        let mut file = Vec::new();
        file.extend_from_slice(&(body.len() as u64).to_le_bytes());
        file.extend_from_slice(body.as_bytes());
        file.extend_from_slice(&payload);
        std::fs::write(directory.join("model.safetensors"), &file).expect("a shard");
    }

    /// A tensor whose every stored byte is zero is a missing export, not a value — when the request
    /// asks for a value.
    ///
    /// The check is on the bytes the load already holds, so it costs no second read, and it is the
    /// region the caller asked for rather than the whole tensor: a rank whose own block is zero is
    /// refused even when the rest of the matrix holds values. A norm or a bias may be zero, which is
    /// why the request asks rather than the loader deciding.
    #[test]
    fn a_region_of_zeros_is_refused_when_the_request_asks_for_a_value() {
        let root = tempfile::tempdir().expect("a temp directory");
        let zeros = [0u8; 8];
        // Two BF16 elements each: one tensor holds nothing but zeros, the other holds a value.
        let live = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        shard(
            root.path(),
            &[
                ("layers.0.attention.v_proj.weight", "BF16", &[2, 2], &zeros),
                ("layers.0.attention.k_proj.weight", "BF16", &[2, 2], &live),
            ],
        );

        let checkpoint = Checkpoint::open(root.path()).expect("the shard opens");
        let mut store = HostBuffers::default();
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);

        let zero =
            WeightRequest::region("layers.0.attention.v_proj.weight", 0, Shape::new(&[2, 2]))
                .nonzero();
        let problem = loader
            .load(&zero)
            .expect_err("an all-zero projection is refused");
        let text = problem.to_string();
        assert!(
            text.contains("v_proj.weight"),
            "the refusal names the tensor: {text}"
        );
        assert!(
            text.contains("every stored byte of the region is zero"),
            "and what is wrong with it: {text}"
        );

        // Without the request's word the same bytes load: a zero norm's scale and a zero bias are
        // values, and the loader does not decide which tensors must hold one.
        let zero_but_allowed =
            WeightRequest::region("layers.0.attention.v_proj.weight", 0, Shape::new(&[2, 2]));
        loader
            .load(&zero_but_allowed)
            .expect("a zero tensor loads when nothing asks it to hold a value");

        let live_request =
            WeightRequest::region("layers.0.attention.k_proj.weight", 0, Shape::new(&[2, 2]))
                .nonzero();
        loader
            .load(&live_request)
            .expect("a tensor that holds a value passes");

        // A meta loader reads no payload, so the zeros are not there for it to refuse: the check
        // is a value's, and only a device load holds the value.
        drop(loader);
        WeightLoader::meta(&checkpoint, None, None)
            .load(&zero)
            .expect("a meta loader reads no payload");
    }

    /// A store that keeps every published region in host memory, at the address of its buffer.
    #[derive(Default)]
    struct HostBuffers {
        published: Vec<(WeightKey, Vec<u8>)>,
    }

    impl RetainedWeights for HostBuffers {
        fn retained(&mut self, key: &WeightKey) -> Result<Option<Backing>> {
            Ok(self
                .published
                .iter()
                .find(|(published, _)| published == key)
                .map(|(_, bytes)| {
                    Backing::new(DevicePtr::new(bytes.as_ptr() as u64), bytes.len() as u64)
                }))
        }

        fn publish(
            &mut self,
            key: &WeightKey,
            _dtype: DType,
            _shape: Shape,
            bytes: &[u8],
        ) -> Result<Backing> {
            self.published.push((key.clone(), bytes.to_vec()));
            let held = &self.published[self.published.len() - 1].1;
            Ok(Backing::new(
                DevicePtr::new(held.as_ptr() as u64),
                held.len() as u64,
            ))
        }
    }

    /// The block-scale arrangement is the launcher's `computeSFIndex`: a permutation of the
    /// checkpoint's row-major table into the 128x4 layout, padded with zeros.
    ///
    /// Three properties, and the third is the one that catches a skipped arrangement: the values
    /// survive exactly once each; the result is not the padded row-major table, so an arrangement that
    /// never ran fails here rather than passing as "the same numbers"; and one index follows the
    /// reference's own arithmetic
    /// (`(column % 4) + (column / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 +
    /// (row / 128) * (128 * padded_columns)`), so a bijection that moved the right values to the wrong
    /// places is not mistaken for the layout the products read.
    ///
    /// The shapes keep `rows * blocks` inside a byte, because the table is bytes: `(56, 4)` pads the
    /// rows only and `(8, 8)` pads both axes.
    ///
    /// **The padding is a zero, so the data starts at one.** "Every scale survives exactly once" is
    /// counted over non-zero entries here, and a table carrying a genuine zero would be one short —
    /// which is the caller's obligation, not the assertion's: a zero scale is a block the file says is
    /// empty, indistinguishable in the arranged table from a hole, so a test of this bijection uses
    /// values a checkpoint would never write as padding.
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

        // One index from the formula, at a row inside the first thirty-two and a column past the
        // first group of four: the values are `row * blocks + column`, so this one is 5 * 8 + 6.
        let (rows, blocks) = (128_u64, 8_u64);
        let linear: Vec<u8> = (1..=rows * blocks).map(|value| value as u8).collect();
        let swizzled = swizzled_scales(&linear, rows, blocks);
        let index = 594_usize;
        // The data starts at one so that a genuine zero cannot be mistaken for the table's padding;
        // the pinned value is therefore `row * blocks + column + 1`, not the position itself.
        assert_eq!(swizzled[index], (5 * blocks + 6 + 1) as u8);
    }
    #[test]
    fn a_key_names_the_checkpoint_the_type_the_layout_and_the_region() {
        let key = WeightKey {
            files: vec![
                FileIdentity::of(std::path::Path::new("/dev/null")).unwrap_or_else(|_| {
                    FileIdentity::of(std::path::Path::new("/proc/self/exe")).unwrap()
                }),
            ],
            dtype: DType::BF16,
            layout: WeightLayout::UP_GATE,
            region: WeightRegion {
                tensors: vec!["up".to_owned(), "gate".to_owned()],
                start: 0,
                shape: Shape::new(&[128, 1536, 2048]),
                stride: None,
            },

            provenance: None,
        };
        let text = key.to_string();
        assert!(text.contains("bf16"));
        assert!(text.contains("up_gate"));
        assert!(text.contains("up+gate"));
    }

    /// A PCM sample converts to its nearest integer, ties to even, and clamps at both ends: a clip
    /// does not wrap, and the two ends are the values an audio codec actually reaches.
    #[test]
    fn a_sample_converts_to_the_nearest_sixteen_bit_integer() {
        let bytes: Vec<u8> = [
            0.0f32, 0.5, 1.5, 2.5, -0.5, -1.5, 32767.4, 32768.0, -32768.4, -40000.0, 3.0,
        ]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
        let converted = convert(DType::F32, DType::I16, &bytes, "pcm").unwrap();
        let samples: Vec<i16> = converted
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        assert_eq!(
            samples,
            [0, 0, 2, 2, 0, -2, 32767, 32767, -32768, -32768, 3],
            "nearest with ties to even, saturated at both ends"
        );
        // And back: every one of those integers is a float exactly.
        let back = convert(DType::I16, DType::F32, &converted, "pcm").unwrap();
        let floats: Vec<f32> = back
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        assert_eq!(
            floats,
            [
                0.0, 0.0, 2.0, 2.0, 0.0, -2.0, 32767.0, 32767.0, -32768.0, -32768.0, 3.0
            ]
        );
    }

    /// Every other pair that names I16 is refused, and the refusal names both types.
    #[test]
    fn a_sample_converts_only_to_and_from_f32() {
        for (from, to) in [
            (DType::BF16, DType::I16),
            (DType::I16, DType::BF16),
            (DType::F16, DType::I16),
            (DType::I16, DType::F16),
            (DType::U8, DType::I16),
            (DType::I16, DType::U8),
        ] {
            let bytes = vec![0u8; 2];
            let error = convert(from, to, &bytes, "pcm").unwrap_err().to_string();
            assert!(
                error.contains(&from.to_string()) && error.contains(&to.to_string()),
                "{from} to {to} was refused without naming both: {error}"
            );
        }
    }
}

#[cfg(test)]
mod value_layout_tests {
    use super::ValueLayout;

    /// A gated stack's rows: logical row `i` of each half reaches the position the reference states,
    /// `2i` or `2i + 1` before the shuffle, then `(r % 4) · 8 + r / 4` inside its 32-row block.
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
