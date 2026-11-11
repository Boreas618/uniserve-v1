//! A linear projection `out = x · weightᵀ + bias`, in whichever representation the checkpoint
//! stores its weight, and the parallel forms built from it.
//!
//! ## Loading
//!
//! [`Linear::load`] reads one module's `weight` (and `bias`) through its builder and chooses the
//! representation from the checkpoint's quantization declaration: a dense weight is one `gemm`;
//! a calibrated NVFP4 weight is an activation encoder and a packed product whose `alpha` is the
//! weight's tensor scale times the activation's; a per-tensor FP8 weight is one `fp8_gemm`,
//! which encodes the activation by its static scale as it loads it; a weight-only NVFP4 weight
//! is the dense matrix its fields state, decoded at load. The caller names the module, its
//! widths and its split, and never the representation. An adapter attached to the checkpoint is
//! fused into the bytes the weight is read from ([`crate::adapter`]), so a projection it edits
//! is loaded fused.
//!
//! ## Parallelism
//!
//! [`Parallel::Column`] holds this rank's block of the output channels: its rows of the
//! checkpoint's `[out, in]` matrix and its slice of the bias. [`Parallel::Row`] holds this rank's
//! block of the contraction: its columns of every row, gathered at load into one contiguous
//! weight. A row-parallel product is a partial sum, so its forward reduces the output over the
//! forward's group ([`crate::collective::all_reduce`]) and refuses a forward that has none. The
//! bias of a row-parallel projection is added on rank 0 alone, so the reduction counts it once.
//!
//! [`QkvLinear`] holds the query, key and value projections of an attention (separate
//! checkpoint tensors or windows of one fused tensor, the key/value heads replicated when the
//! split is wider than they are); [`Linear::load_merged`] reads several checkpoint matrices as
//! one product.

use ops::fp4::{BLOCK_ELEMENTS, ScaleLayout};
use uniserve_tensor::{DType, Module, Shape, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{self, WeightOrder};
use crate::quantization::{Nvfp4Names, Nvfp4Weight, Scheme};
use crate::var_builder::VarBuilder;
use crate::weights::{WeightLayout, WeightSource, extents};

/// The arrangement a calibrated ModelOpt export stores its block scales in: linear, one E4M3
/// scale per sixteen elements, row-major. The published table is built from it at load.
const FILE_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Linear;

/// The arrangement the packed product reads the weight's block scales in.
const PRODUCT_SCALE_LAYOUT: ScaleLayout = functional::PRODUCT_SCALE_LAYOUT;

/// How a projection is split across a tensor-parallel group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Parallel {
    /// One rank holds the whole projection.
    None,
    /// The output channels are split: this rank computes its block of them.
    Column(Split),
    /// The contraction is split: this rank computes a partial sum, which its forward reduces.
    Row(Split),
}

/// What a checkpoint module states of a [`Linear`]: its widths, both the whole module's, and
/// this rank's share of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LinearConfig {
    /// The width of the rows the projection reads.
    pub inputs: u32,
    /// The width of the rows it writes.
    pub outputs: u32,
    /// Whether the module states a `bias` beside its `weight`.
    pub bias: bool,
    /// How the projection is split across the rank's group.
    pub parallel: Parallel,
}

/// The representation a projection's weight is held in.
enum Weight {
    /// The checkpoint's own matrix, in `order`.
    Dense { weight: Tensor, order: WeightOrder },
    /// A calibrated NVFP4 matrix and the tensors its encoder and product read.
    Nvfp4 {
        /// `[n, k/2]` U8: two E2M1 codes a byte along the contraction.
        values: Tensor,
        /// The block scales in [`PRODUCT_SCALE_LAYOUT`], in the checkpoint's one-byte scale
        /// type.
        scales: Tensor,
        /// `[1]` F32: the encoder's global scale, the reciprocal of the activation's tensor
        /// scale.
        input_scale: Tensor,
        /// `[1]` F32: the product's alpha, the weight's tensor scale times the activation's.
        alpha: Tensor,
    },
    /// A per-tensor FP8 matrix, `[n, k]` E4M3, and its two static scales: the activation is
    /// divided by `input_scale` before it is encoded, and the accumulator is multiplied by
    /// `alpha`.
    Fp8 {
        weight: Tensor,
        input_scale: f32,
        alpha: f32,
    },
}

/// A linear projection in the representation its checkpoint stores.
pub struct Linear {
    weight: Weight,
    bias: Option<Tensor>,
    k: u32,
    n: u32,
    /// Ranks this projection's partial sums are reduced over: above one for a row-parallel
    /// split.
    reduce: u32,
}

impl Linear {
    /// A dense projection over an already loaded `weight` stored in `order`.
    ///
    /// The constructor for weights no checkpoint module states as one: a table a model computes
    /// at load, or the head's shard of a vocabulary table. `weight` is `[k, n]` in
    /// [`WeightOrder::KxN`] or `[n, k]` in [`WeightOrder::NxK`]; the bias, when present, has `n`
    /// elements.
    ///
    /// # Errors
    ///
    /// When the weight is not a matrix or the bias another width.
    pub fn new(weight: Tensor, bias: Option<Tensor>, order: WeightOrder) -> Result<Self> {
        let &[first, second] = weight.dims() else {
            return Err(Error::shape(
                "Linear",
                format!("a weight is a matrix, not {:?}", weight.dims()),
            ));
        };
        let (k, n) = match order {
            WeightOrder::KxN => (first, second),
            WeightOrder::NxK => (second, first),
        };

        Self::build(Weight::Dense { weight, order }, bias, k as u64, n as u64, 1)
    }

    /// The projection the module at `vb` states, split as `config.parallel` says.
    ///
    /// The checkpoint stores the module as `weight` `[outputs, inputs]` and, with a bias, `bias`
    /// `[outputs]`; a calibrated NVFP4 export stores `weight` packed beside `weight_scale`,
    /// `weight_scale_2` and `input_scale`, and the declaration the model's loader was given
    /// decides which this module is.
    ///
    /// # Errors
    ///
    /// When the split does not divide the split axis, a tensor is missing or of another extent,
    /// a dense matrix is all zero, or a packed module states a bias.
    pub fn load(config: &LinearConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (inputs, outputs) = (u64::from(config.inputs), u64::from(config.outputs));
        let parallel = config.parallel;
        let (local_in, local_out) = match parallel {
            Parallel::None => (inputs, outputs),
            Parallel::Column(split) => (inputs, share(&vb, outputs, split, "output channels")?),
            Parallel::Row(split) => (share(&vb, inputs, split, "contraction")?, outputs),
        };
        let reduce = match parallel {
            Parallel::Row(split) => split.size,
            _ => 1,
        };
        let name = vb.name("weight");

        match vb.scheme("weight") {
            // A weight-only NVFP4 matrix is read as the dense matrix it states
            // (`VarBuilder::load`).
            Scheme::Dense | Scheme::Nvfp4Weights => {
                let weight = vb.load(
                    &name,
                    &weight_source(&name, inputs, local_out, local_in, parallel),
                    WeightLayout::ROW_MAJOR,
                    &extents(&[local_out, local_in]),
                    true,
                )?;
                let bias = load_bias(&vb, config.bias, local_out, parallel)?;

                Self::build(
                    Weight::Dense {
                        weight,
                        order: WeightOrder::NxK,
                    },
                    bias,
                    local_in,
                    local_out,
                    reduce,
                )
            }

            Scheme::Fp8 => {
                // The E4M3 matrix is read as it is stored, whatever the builder pins.
                let weight = vb.loader().load(
                    &name,
                    &weight_source(&name, inputs, local_out, local_in, parallel),
                    WeightLayout::ROW_MAJOR,
                    &extents(&[local_out, local_in]),
                    Some(DType::F8E4M3),
                    true,
                )?;
                if weight.dtype() != DType::F8E4M3 {
                    return Err(Error::weight(
                        &name,
                        format!(
                            "an FP8 projection's weight is F8_E4M3, not {}",
                            ops::abi::name(weight.dtype())
                        ),
                    ));
                }

                // One positive float each, the same for every rank's block.
                let weight_scale = vb.tensor_scale("weight_scale")?;
                let input_scale = vb.tensor_scale("input_scale")?;
                let bias = load_bias(&vb, config.bias, local_out, parallel)?;

                Self::build(
                    Weight::Fp8 {
                        weight,
                        input_scale,
                        alpha: input_scale * weight_scale,
                    },
                    bias,
                    local_in,
                    local_out,
                    reduce,
                )
            }

            Scheme::Nvfp4 => {
                if config.bias {
                    return Err(Error::weight(
                        &name,
                        "a packed projection states a bias, and no calibrated NVFP4 export \
                         writes one",
                    ));
                }

                let mut linear = Self::load_nvfp4(&vb, inputs, local_in, local_out, parallel)?;
                linear.reduce = reduce;
                Ok(linear)
            }
        }
    }

    /// The modules `modules` under `vb`, each `[outputs[i], inputs]`, read as one projection:
    /// their weights stacked along the output channels in the order named, their biases
    /// likewise. One launch computes what the modules compute one after another, and the
    /// output's columns are each module's in order.
    ///
    /// # Errors
    ///
    /// When `modules` and `outputs` disagree in count, or a module cannot be read.
    pub fn load_merged(
        vb: VarBuilder<'_, '_>,
        modules: &[&str],
        inputs: u32,
        outputs: &[u32],
        bias: bool,
    ) -> Result<Self> {
        if modules.len() != outputs.len() || modules.is_empty() {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "{} modules with {} output widths",
                    modules.len(),
                    outputs.len()
                ),
            ));
        }
        let total: u64 = outputs.iter().map(|width| u64::from(*width)).sum();
        let parts = |leaf: &str| -> Vec<String> {
            modules
                .iter()
                .map(|module| vb.pp(module).name(leaf))
                .collect()
        };
        let merged = vb.name(&modules.join("+"));

        let shape = extents(&[total, u64::from(inputs)]);
        let weight = vb.load(
            &format!("{merged}.weight"),
            &WeightSource::Parts {
                parts: parts("weight"),
                shape: shape.clone(),
                block: None,
            },
            WeightLayout::ROW_MAJOR,
            &shape,
            true,
        )?;

        let bias = if bias {
            let shape = extents(&[total]);
            Some(vb.load(
                &format!("{merged}.bias"),
                &WeightSource::Parts {
                    parts: parts("bias"),
                    shape: shape.clone(),
                    block: None,
                },
                WeightLayout::ROW_MAJOR,
                &shape,
                false,
            )?)
        } else {
            None
        };
        Self::new(weight, bias, WeightOrder::NxK)
    }

    /// The packed fields of the module at `vb`, at this rank's block.
    ///
    /// The checkpoint stores two E2M1 codes a byte along the contraction, so a row is `in / 2`
    /// bytes and its block scales are `in / 16` of them; a rank's block is the same rows or
    /// columns of both. The weight's `weight_scale_2` and the activation's `input_scale` are one
    /// positive float each: the encoder reads the activation's reciprocal, and the product's
    /// alpha is the two scales' product, published once as a `[1]` F32 constant beside the
    /// weights.
    fn load_nvfp4(
        vb: &VarBuilder<'_, '_>,
        inputs: u64,
        local_in: u64,
        local_out: u64,
        parallel: Parallel,
    ) -> Result<Self> {
        let spelling = Nvfp4Names::vllm();
        let values = weight_source(
            &vb.name(spelling.values),
            inputs / 2,
            local_out,
            local_in / 2,
            parallel,
        );
        let scales = weight_source(
            &vb.name(spelling.block_scale),
            inputs / 16,
            local_out,
            local_in / 16,
            parallel,
        );
        let input_scale = vb.name(spelling.input_scale);

        let global_scale = vb.tensor_scale(spelling.tensor_scale)?;

        // A dense product dequantizes `code * block_scale` on both sides, so its alpha is the
        // two tensor scales' product: the weight's and the activation's.
        let product_alpha = global_scale * vb.tensor_scale(spelling.input_scale)?;
        let weight = vb.loader().nvfp4(
            &values,
            &scales,
            Some(&input_scale),
            spelling,
            (FILE_SCALE_LAYOUT, PRODUCT_SCALE_LAYOUT),
            global_scale,
        )?;
        let alpha = vb.loader().constant(
            &vb.name("alpha"),
            DType::F32,
            &Shape::from(1),
            &product_alpha.to_le_bytes(),
            // The producer's text is part of the stored key.
            "nn::Linear",
        )?;

        Self::nvfp4(&weight, alpha)
    }

    /// A projection over a calibrated NVFP4 matrix and its product's `alpha`.
    ///
    /// `weight.packed` is `[n, k/2]` bytes and `weight.scales` its block scales in the product's
    /// layout; the contraction must be whole sixteen-element blocks. A weight-only export has no
    /// activation scale and **cannot serve this projection**, which is refused here by name
    /// instead of being encoded at an invented magnitude.
    ///
    /// # Errors
    ///
    /// When a field is not the element type, the extents or the layout the product reads.
    pub fn nvfp4(weight: &Nvfp4Weight, alpha: Tensor) -> Result<Self> {
        let refuse = |problem: String| Error::shape("Linear", problem);

        let &[n, packed] = weight.packed.dims() else {
            return Err(refuse(format!(
                "a packed weight is a U8 matrix, not {:?}",
                weight.packed.dims()
            )));
        };
        if weight.packed.dtype() != DType::U8 {
            return Err(refuse(format!(
                "a packed weight is a U8 matrix, not {} {:?}",
                ops::abi::name(weight.packed.dtype()),
                weight.packed.dims()
            )));
        }

        let (n, k) = (n as u64, packed as u64 * 2);
        if k == 0 || !k.is_multiple_of(u64::from(BLOCK_ELEMENTS)) {
            return Err(refuse(format!(
                "a contraction of {k} elements is not whole sixteen-element blocks"
            )));
        }

        // One layout governs a product: it states both how the activation's block scales are
        // arranged and, by naming the product, how the weight's are read.
        if weight.layout != PRODUCT_SCALE_LAYOUT {
            return Err(refuse(format!(
                "the weight's block scales are {} and the product reads {PRODUCT_SCALE_LAYOUT}",
                weight.layout
            )));
        }
        let expected = extents(weight.layout.shape(n, k).dims());
        if weight.scales.dims() != expected.dims() {
            return Err(refuse(format!(
                "the block scales are {:?}; {n} rows of {k} elements take {:?}",
                weight.scales.dims(),
                expected.dims()
            )));
        }

        let Some(input_scale) = weight.input_scale.clone() else {
            return Err(refuse(
                "this projection has no calibrated activation input scale: a weight-only export \
                 cannot be encoded by a kernel that reads one"
                    .to_owned(),
            ));
        };
        for (what, scalar) in [("input scale", &input_scale), ("alpha", &alpha)] {
            if scalar.dtype() != DType::F32 || scalar.elem_count() != 1 {
                return Err(refuse(format!(
                    "the {what} is {} {:?}, and the product reads one FP32 number",
                    ops::abi::name(scalar.dtype()),
                    scalar.dims()
                )));
            }
        }

        Self::build(
            Weight::Nvfp4 {
                values: weight.packed.clone(),
                scales: weight.scales.clone(),
                input_scale,
                alpha,
            },
            None,
            k,
            n,
            1,
        )
    }

    /// A projection of `weight`, its widths checked.
    fn build(weight: Weight, bias: Option<Tensor>, k: u64, n: u64, reduce: u32) -> Result<Self> {
        if let Some(bias) = &bias
            && bias.elem_count() as u64 != n
        {
            return Err(Error::shape(
                "Linear",
                format!(
                    "a bias of {} elements cannot be added to {n} output channels",
                    bias.elem_count()
                ),
            ));
        }

        Ok(Self {
            weight,
            bias,
            k: u32::try_from(k)
                .map_err(|_| Error::shape("Linear", format!("a contraction of {k} elements")))?,
            n: u32::try_from(n)
                .map_err(|_| Error::shape("Linear", format!("{n} output channels")))?,
            reduce,
        })
    }

    /// The contraction length: the width of the rows this rank reads.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// The number of output channels this rank computes.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// The weight: the dense matrix, or the packed values of a packed projection.
    pub fn weight(&self) -> &Tensor {
        match &self.weight {
            Weight::Dense { weight, .. } | Weight::Fp8 { weight, .. } => weight,
            Weight::Nvfp4 { values, .. } => values,
        }
    }

    /// Whether the weight is a calibrated NVFP4 matrix, whose product is compiled per row class
    /// ([`ops::fp4::packed_rows`]).
    pub const fn is_packed(&self) -> bool {
        matches!(self.weight, Weight::Nvfp4 { .. })
    }

    /// `x · weightᵀ + bias`, `[..., n]` in `x`'s element type, reduced over the forward's group
    /// when the projection is row-parallel.
    ///
    /// # Errors
    ///
    /// When `x` does not fit the projection, an operator refuses, or a row-parallel projection
    /// runs in a forward that has no group.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bias = self.bias.as_ref();
        let out = match &self.weight {
            Weight::Dense { weight, order } => functional::linear(x, weight, bias, *order)?,
            Weight::Nvfp4 {
                values,
                scales,
                input_scale,
                alpha,
            } => functional::linear_nvfp4(x, values, scales, input_scale, alpha)?,
            Weight::Fp8 {
                weight,
                input_scale,
                alpha,
            } => functional::linear_fp8(x, weight, bias, *input_scale, *alpha)?,
        };

        if self.reduce > 1 {
            // A partial sum over this rank's block of the contraction: the group sums the
            // ranks'.
            if collective::split().is_none() {
                return Err(Error::shape(
                    "Linear",
                    format!(
                        "a row-parallel projection over {} ranks runs in a forward that has no \
                         group to reduce over",
                        self.reduce
                    ),
                ));
            }
            collective::all_reduce(&out)?;
        }
        Ok(out)
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// `extent` split over `split`'s ranks: this rank's share, refused when the split does not
/// divide.
fn share(vb: &VarBuilder<'_, '_>, extent: u64, split: Split, what: &str) -> Result<u64> {
    let size = u64::from(split.size);

    if size == 0 || split.rank >= split.size || !extent.is_multiple_of(size) {
        return Err(Error::weight(
            &vb.name("weight"),
            format!(
                "rank {} of a split of {} does not hold a share of {extent} {what}",
                split.rank, split.size
            ),
        ));
    }
    Ok(extent / size)
}

/// Where this rank's block of the `[outputs, columns]` matrix `tensor` is read from; the block
/// is `rows` rows of `width` elements.
///
/// A column-parallel block is contiguous rows; a row-parallel one is every row's own columns, a
/// gather of rows `columns` elements apart.
fn weight_source(
    tensor: &str,
    columns: u64,
    rows: u64,
    width: u64,
    parallel: Parallel,
) -> WeightSource {
    let shape = extents(&[rows, width]);
    let tensor = tensor.to_owned();

    match parallel {
        Parallel::None => WeightSource::Region {
            tensor,
            start: 0,
            shape,
        },
        Parallel::Column(split) => WeightSource::Region {
            tensor,
            start: u64::from(split.rank) * rows * columns,
            shape,
        },
        Parallel::Row(split) => WeightSource::Strided {
            tensor,
            start: u64::from(split.rank) * width,
            shape,
            row_stride: columns,
        },
    }
}

/// The bias of the module at `vb`, `local_out` elements at this rank's block, when the module
/// states one and this rank adds it.
fn load_bias(
    vb: &VarBuilder<'_, '_>,
    stated: bool,
    local_out: u64,
    parallel: Parallel,
) -> Result<Option<Tensor>> {
    if !stated {
        return Ok(None);
    }

    match parallel {
        // A partial sum's bias is added once, on the first rank, so the reduction counts it
        // once.
        Parallel::Row(split) if split.rank != 0 => Ok(None),
        Parallel::Row(_) | Parallel::None => Ok(Some(vb.get(local_out as usize, "bias")?)),
        Parallel::Column(split) => Ok(Some(vb.get_rows(
            local_out as usize,
            "bias",
            (u64::from(split.rank) * local_out) as usize,
        )?)),
    }
}

/// The heads an attention's query, key and value projections hold, where its checkpoint stores
/// them, and this rank's share.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QkvLinearConfig {
    /// The hidden width the projections reduce over.
    pub hidden: u32,
    /// Query heads of the whole module.
    pub heads: u32,
    /// Key/value heads of the whole module.
    pub kv_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Whether each query head's rows are followed by as many rows of its output gate
    /// (Qwen3-Next's and Qwen3.5's `attn_output_gate`): the query projection is then twice as
    /// wide, a head's query and gate side by side in its output row.
    pub gated: bool,
    /// Where the checkpoint stores the three projections.
    pub names: QkvNames,
    /// Where the checkpoint stores the three biases.
    pub bias: QkvBias,
    /// The tensor-parallel split of the heads.
    pub split: Split,
}

/// Where a checkpoint stores an attention's three projections.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QkvNames {
    /// Three modules, query, key and value, each with its own `weight` (and `bias`).
    Separate([&'static str; 3]),
    /// One module whose `weight` stacks the three `[query; key; value]` along its rows.
    Fused(&'static str),
}

/// Where a checkpoint stores an attention's three biases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QkvBias {
    /// No projection adds a bias.
    Absent,
    /// Each projection's module states its own `bias` (for [`QkvNames::Fused`], the fused
    /// module's `bias`, stacked as the weight is).
    Present,
    /// The attention's module states one bias tensor per projection under these names, in
    /// query, key, value order.
    Windows([&'static str; 3]),
}

/// An attention's query, key and value projections, each its own product.
///
/// The three are loaded from separate checkpoint modules or as row windows of one fused module
/// and split over a tensor-parallel group by heads. When the group is wider than the key/value
/// heads, each key/value head is replicated on `size / kv_heads` consecutive ranks, the rule the
/// reference projections use, so every rank holds the head its query heads read.
pub struct QkvLinear {
    query: Linear,
    key: Linear,
    value: Linear,
}

impl QkvLinear {
    /// The three projections of the attention at `vb`, at `config.split`'s rank.
    ///
    /// # Errors
    ///
    /// When the split divides neither the query heads nor (or into) the key/value heads, or a
    /// projection cannot be loaded.
    pub fn load(config: &QkvLinearConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let refuse = |problem: String| Error::weight(vb.path(), problem);
        let split = config.split;

        if split.size == 0 || split.rank >= split.size || !config.heads.is_multiple_of(split.size) {
            return Err(refuse(format!(
                "a split of {} does not divide {} query heads",
                split.size, config.heads
            )));
        }

        // The key/value heads' own split: the group's, or, when the group is wider than the
        // heads, one head per `size / kv_heads` ranks.
        let kv_split = if split.size <= config.kv_heads {
            if !config.kv_heads.is_multiple_of(split.size) {
                return Err(refuse(format!(
                    "a split of {} does not divide {} key/value heads",
                    split.size, config.kv_heads
                )));
            }
            split
        } else {
            if !split.size.is_multiple_of(config.kv_heads) {
                return Err(refuse(format!(
                    "a split of {} does not replicate {} key/value heads evenly",
                    split.size, config.kv_heads
                )));
            }
            Split {
                size: config.kv_heads,
                rank: split.rank / (split.size / config.kv_heads),
            }
        };

        let query_width = config.heads * config.head_dim * if config.gated { 2 } else { 1 };
        let kv_width = config.kv_heads * config.head_dim;

        match config.names {
            QkvNames::Separate(modules) => {
                let bias = match config.bias {
                    QkvBias::Absent => false,
                    QkvBias::Present => true,
                    QkvBias::Windows(_) => {
                        return Err(refuse(
                            "separate projections state their own biases".to_owned(),
                        ));
                    }
                };
                let load = |module: &str, outputs: u32, split: Split| {
                    let parallel = if split.size == 1 {
                        Parallel::None
                    } else {
                        Parallel::Column(split)
                    };
                    Linear::load(
                        &LinearConfig {
                            inputs: config.hidden,
                            outputs,
                            bias,
                            parallel,
                        },
                        vb.pp(module),
                    )
                };

                Ok(Self {
                    query: load(modules[0], query_width, split)?,
                    key: load(modules[1], kv_width, kv_split)?,
                    value: load(modules[2], kv_width, kv_split)?,
                })
            }

            QkvNames::Fused(_) if config.gated => Err(refuse(
                "a fused projection holds no output gate beside its queries".to_owned(),
            )),

            QkvNames::Fused(module) => {
                let hidden = config.hidden as usize;
                let fused = vb.pp(module);

                // The query rows, then the key rows, then the value rows; each split's block is
                // its own rows of its window.
                let window = |index: usize, base: u32, width: u32, split: Split| {
                    let rows = (width / split.size) as usize;
                    let first = base as usize + split.rank as usize * rows;
                    let weight = fused.get_rows((rows, hidden), "weight", first)?;

                    let bias = match config.bias {
                        QkvBias::Absent => None,
                        QkvBias::Present => Some(fused.get_rows(rows, "bias", first)?),
                        QkvBias::Windows(names) => {
                            Some(vb.get_rows(rows, names[index], split.rank as usize * rows)?)
                        }
                    };
                    Linear::new(weight, bias, WeightOrder::NxK)
                };

                Ok(Self {
                    query: window(0, 0, query_width, split)?,
                    key: window(1, query_width, kv_width, kv_split)?,
                    value: window(2, query_width + kv_width, kv_width, kv_split)?,
                })
            }
        }
    }

    /// The query projection.
    pub fn query(&self) -> &Linear {
        &self.query
    }

    /// The key projection.
    pub fn key(&self) -> &Linear {
        &self.key
    }

    /// The value projection.
    pub fn value(&self) -> &Linear {
        &self.value
    }
}
