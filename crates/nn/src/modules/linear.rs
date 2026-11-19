//! A linear projection `out = x · weightᵀ + bias`, in whichever representation the checkpoint
//! stores its weight, and the parallel forms built from it.
//!
//! ## Loading
//!
//! [`Linear::load`] reads one module's `weight` (and `bias`) through its builder and chooses the
//! representation from the checkpoint's quantization declaration: a dense weight is one `gemm`;
//! a calibrated NVFP4 weight is an activation encoder and a packed product whose `alpha` is the
//! weight's tensor scale times the activation's; a per-tensor FP8 weight is an activation
//! encoder at its static scale and an `fp8_gemm` over the codes; a block-scaled FP8 weight is an
//! MXFP8 matrix, its activation encoded per 32 elements on every call and multiplied by
//! `mxfp8_gemm` ([`functional::linear_mxfp8`]); a weight-only NVFP4 weight is the dense matrix
//! its fields state, decoded at load. A dense weight the declaration
//! converts ([`Scheme::DynamicFp8`], [`Scheme::DynamicNvfp4`]) is encoded at load into the
//! same FP8 or NVFP4 product, and its activation's factors are measured on every call
//! ([`functional::dynamic_scales`]) instead of published. The caller names the module, its
//! widths and its split, and never the representation. An adapter attached to the checkpoint is
//! fused into the bytes the weight is read from ([`crate::adapter`]), so a projection it edits
//! is loaded fused.
//!
//! A calibrated NVFP4 module's fields are read in the spelling the declaration states
//! ([`Nvfp4Names`](crate::quantization::Nvfp4Names)), and its activation scale is the one the
//! declaration states for the module, when the export keeps its calibration in a manifest, and
//! the tensor beside the weight otherwise ([`crate::quantization`]). Its bias, when the module
//! states one, joins the rounded product ([`functional::linear_nvfp4`]).
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

use ops::dynamic_scale::{FP8_RANGE, NVFP4_RANGE};
use ops::fp4::{BLOCK_ELEMENTS, GemmDescriptor, PRODUCT_TILE, ScaleLayout};
use ops::mx::GemmDescriptor as MxGemmDescriptor;
use tensor::{DType, Module, Shape, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{self, WeightOrder};
use crate::quantization::{Mxfp8Part, Mxfp8Weight, Nvfp4Weight, Scheme};
use crate::var_builder::VarBuilder;
use crate::weights::{WeightLayout, WeightSource, extents};

/// The arrangement a calibrated ModelOpt export stores its block scales in: linear, one E4M3
/// scale per sixteen elements, row-major. The published table is built from it at load.
const FILE_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Linear;

/// The class the constants a packed projection computes at load are filed under: the producer
/// of their retained keys. Its text is part of every stored key of such a constant.
const PRODUCER: &str = "nn::Linear";

/// The arrangement the packed product reads the weight's block scales in, as its descriptor
/// declares it.
const PRODUCT_SCALE_LAYOUT: ScaleLayout = GemmDescriptor::SCALE_LAYOUT;

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
    /// An NVFP4 matrix and how its activation is scaled.
    Nvfp4 {
        /// `[n, k/2]` U8: two E2M1 codes a byte along the contraction.
        values: Tensor,
        /// The block scales in [`PRODUCT_SCALE_LAYOUT`], in a one-byte scale type.
        scales: Tensor,
        scaling: Scaling,
    },
    /// A per-tensor FP8 matrix, `[n, k]` E4M3, and how its activation is scaled.
    Fp8 { weight: Tensor, scaling: Scaling },
    /// An MXFP8 matrix, `[n, k]` E4M3 with its scales in the product's layout; its activation
    /// is encoded per call.
    Mxfp8(Mxfp8Weight),
}

/// Where an encoded projection's two factors come from: the encoder's multiplier of the
/// activation (the reciprocal of its scale) and the product's dequantization factor (the
/// activation's scale times the weight's).
enum Scaling {
    /// A calibrated export's static activation scale: both factors `[1]` F32 constants
    /// published at load.
    Calibrated { encode: Tensor, alpha: Tensor },
    /// A weight converted at load: both factors measured from each call's activation against
    /// the weight's largest magnitude, a `[1]` F32 weight, in the format's range.
    Measured { weight_amax: Tensor, range: f32 },
}

impl Scaling {
    /// The `(encode, alpha)` the product over `x` reads.
    fn factors(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Calibrated { encode, alpha } => Ok((encode.clone(), alpha.clone())),
            Self::Measured { weight_amax, range } => {
                functional::dynamic_scales(x, weight_amax, *range)
            }
        }
    }
}

/// A linear projection in the representation its checkpoint stores.
pub struct Linear {
    weight: Weight,
    bias: Option<Tensor>,
    /// The element type of the rows the projection reads and writes, when its weight states it:
    /// a dense weight's own, or a converted weight's as its dense load would hold it.
    activation: Option<DType>,
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
    /// `[outputs]`; a calibrated NVFP4 export stores the weight's packed values beside its
    /// block scales and its tensor scale, under the names of its spelling, and the declaration
    /// the model's loader was given decides which this module is.
    ///
    /// # Errors
    ///
    /// When the split does not divide the split axis, a tensor is missing or of another extent,
    /// a dense matrix is all zero, or a packed module has no activation scale or states it
    /// twice.
    pub fn load(config: &LinearConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (inputs, outputs) = (u64::from(config.inputs), u64::from(config.outputs));
        let parallel = config.parallel;
        let (local_in, local_out) = match parallel {
            Parallel::None => (inputs, outputs),
            Parallel::Column(split) => (inputs, share(&vb, outputs, split, "output channels")?),
            Parallel::Row(split) => (share(&vb, inputs, split, "contraction")?, outputs),
        };
        let reduce = match parallel {
            Parallel::Row(split) => split.size(),
            _ => 1,
        };
        let name = vb.name("weight");

        match vb.scheme("weight") {
            Scheme::Dense => {
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

                // One positive float each, the same for every rank's block, published as the two
                // factors the product reads from device memory.
                let weight_scale = vb.tensor_scale("weight_scale")?;
                let input_scale = vb.tensor_scale("input_scale")?;
                let factor = |name: &str, value: f32| {
                    vb.loader().constant(
                        &vb.name(name),
                        DType::F32,
                        &Shape::from(1),
                        &value.to_le_bytes(),
                        PRODUCER,
                    )
                };
                let scaling = Scaling::Calibrated {
                    encode: factor("encode", 1.0 / input_scale)?,
                    alpha: factor("alpha", input_scale * weight_scale)?,
                };
                let bias = load_bias(&vb, config.bias, local_out, parallel)?;

                let mut linear = Self::build(
                    Weight::Fp8 { weight, scaling },
                    bias,
                    local_in,
                    local_out,
                    reduce,
                )?;
                linear.activation = vb.dtype();
                Ok(linear)
            }

            scheme @ (Scheme::DynamicFp8 | Scheme::DynamicNvfp4) => {
                let source = weight_source(&name, inputs, local_out, local_in, parallel);
                let mut linear = Self::load_converted(&vb, scheme, &source, local_in, local_out)?;

                linear.bias = load_bias(&vb, config.bias, local_out, parallel)?;
                linear.reduce = reduce;
                Ok(linear)
            }

            Scheme::Mxfp8 => {
                // This rank's rows (a column share) or columns (a row share) of the matrix.
                let (rows, columns) = match parallel {
                    Parallel::None => (0..outputs, 0..inputs),
                    Parallel::Column(split) => {
                        let first = u64::from(split.rank()) * local_out;
                        (first..first + local_out, 0..inputs)
                    }
                    Parallel::Row(split) => {
                        let first = u64::from(split.rank()) * local_in;
                        (0..outputs, first..first + local_in)
                    }
                };
                let weight = vb.loader().mxfp8(
                    &name,
                    &[Mxfp8Part {
                        weight: name.clone(),
                        rows,
                        columns,
                    }],
                    MxGemmDescriptor::SCALE_LAYOUT,
                )?;
                let bias = load_bias(&vb, config.bias, local_out, parallel)?;

                Self::build(Weight::Mxfp8(weight), bias, local_in, local_out, reduce)
            }

            Scheme::Mxfp4 => Err(Error::weight(
                &name,
                "is an MXFP4 matrix, which the routed experts' products read; a projection \
                 multiplies dense, FP8, MXFP8 and NVFP4 weights",
            )),

            Scheme::Nvfp4 => {
                let mut linear = Self::load_nvfp4(&vb, inputs, local_in, local_out, parallel)?;

                // The bias is `local_out` elements by its read: the packed weight's rows.
                linear.bias = load_bias(&vb, config.bias, local_out, parallel)?;
                linear.reduce = reduce;
                linear.activation = vb.dtype();
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

        // Block-scaled weights merge as one MXFP8 matrix: codes and scales stacked alike.
        if vb.pp(modules[0]).scheme("weight") == Scheme::Mxfp8 {
            if bias {
                return Err(Error::weight(
                    &merged,
                    "a merged MXFP8 projection adds no bias: no block-scaled export states one"
                        .to_owned(),
                ));
            }
            let stacked: Vec<Mxfp8Part> = parts("weight")
                .into_iter()
                .zip(outputs)
                .map(|(weight, rows)| Mxfp8Part {
                    weight,
                    rows: 0..u64::from(*rows),
                    columns: 0..u64::from(inputs),
                })
                .collect();
            let weight = vb.loader().mxfp8(
                &format!("{merged}.weight"),
                &stacked,
                MxGemmDescriptor::SCALE_LAYOUT,
            )?;
            return Self::build(Weight::Mxfp8(weight), None, u64::from(inputs), total, 1);
        }

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

    /// The packed fields of the module at `vb`, at this rank's block, read in the spelling the
    /// loader's declaration states.
    ///
    /// The checkpoint stores two E2M1 codes a byte along the contraction, so a row is `in / 2`
    /// bytes and its block scales are `in / 16` of them; a rank's block is the same rows or
    /// columns of both. The weight's tensor scale and the activation's are one positive float
    /// each: the encoder reads the activation's reciprocal, and the product's alpha is the two
    /// scales' product, published once as a `[1]` F32 constant beside the weights.
    ///
    /// The activation's scale is the declaration's when the export keeps its calibration in a
    /// manifest: no checkpoint tensor holds it, so its reciprocal is published as a constant of
    /// this projection, keyed by the bytes it holds. Otherwise it is the `input_scale` tensor
    /// beside the weight, whose reciprocal the loader publishes under the tensor's file.
    fn load_nvfp4(
        vb: &VarBuilder<'_, '_>,
        inputs: u64,
        local_in: u64,
        local_out: u64,
        parallel: Parallel,
    ) -> Result<Self> {
        let representation = vb.loader().representation();
        let spelling = representation.names();
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

        let stated = representation.activation_scale(vb.path())?;
        let activation_scale = match stated {
            // Two statements of one calibration are two numbers a loader would have to choose
            // between.
            Some(_) if vb.contains(spelling.input_scale) => {
                return Err(Error::weight(
                    &input_scale,
                    "the export's manifest states this projection's activation scale, and the \
                     checkpoint stores one beside the weight as well",
                ));
            }
            Some(scale) => scale,
            None => vb.tensor_scale(spelling.input_scale)?,
        };

        let mut weight = vb.loader().nvfp4(
            &values,
            &scales,
            stated.is_none().then_some(input_scale.as_str()),
            spelling,
            (FILE_SCALE_LAYOUT, PRODUCT_SCALE_LAYOUT),
            global_scale,
        )?;
        if stated.is_some() {
            // The encoder's factor, under the name the loader publishes a tensor's under.
            weight.input_scale = Some(vb.loader().constant(
                &format!("{input_scale}__nvfp4_input_scale"),
                DType::F32,
                &Shape::from(1),
                &(1.0 / activation_scale).to_le_bytes(),
                PRODUCER,
            )?);
        }

        // A dense product dequantizes `code * block_scale` on both sides, so its alpha is the
        // two tensor scales' product: the weight's and the activation's.
        let product_alpha = global_scale * activation_scale;
        let alpha = vb.loader().constant(
            &vb.name("alpha"),
            DType::F32,
            &Shape::from(1),
            &product_alpha.to_le_bytes(),
            PRODUCER,
        )?;

        Self::nvfp4(&weight, alpha)
    }

    /// This rank's share of the dense matrix `source` names, `[local_out, local_in]`, converted
    /// at load into `scheme`'s product ([`crate::quantization::dynamic`]), its activation's
    /// factors measured on every call.
    ///
    /// The rows it reads and writes have the type the dense load would hold the weight in: the
    /// builder's pin, or the checkpoint's own.
    fn load_converted(
        vb: &VarBuilder<'_, '_>,
        scheme: Scheme,
        source: &WeightSource,
        local_in: u64,
        local_out: u64,
    ) -> Result<Self> {
        let name = vb.name("weight");
        let shape = extents(&[local_out, local_in]);
        let activation = match vb.dtype() {
            Some(pinned) => pinned,
            None => vb.loader().checkpoint().locate(&name)?.dtype,
        };

        let weight = if scheme == Scheme::DynamicNvfp4 {
            // The packed product compiles whole tiles of output channels and of the contraction.
            let tile = u64::from(PRODUCT_TILE);
            if !local_out.is_multiple_of(tile) || !local_in.is_multiple_of(tile) {
                return Err(Error::weight(
                    &name,
                    format!(
                        "an NVFP4 share of {local_out} output channels over {local_in} elements is \
                         not whole tiles of {tile}, which the packed product computes in"
                    ),
                ));
            }

            let (values, scales, weight_amax) = vb.loader().dynamic_nvfp4(source, &shape)?;
            Weight::Nvfp4 {
                values,
                scales,
                scaling: Scaling::Measured {
                    weight_amax,
                    range: NVFP4_RANGE,
                },
            }
        } else {
            let (weight, weight_amax) = vb.loader().dynamic_fp8(source, &shape)?;
            Weight::Fp8 {
                weight,
                scaling: Scaling::Measured {
                    weight_amax,
                    range: FP8_RANGE,
                },
            }
        };

        let mut linear = Self::build(weight, None, local_in, local_out, 1)?;
        linear.activation = Some(activation);
        Ok(linear)
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
        let expected = extents(&weight.layout.shape(n, k));
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
                scaling: Scaling::Calibrated {
                    encode: input_scale,
                    alpha,
                },
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

        let activation = match &weight {
            Weight::Dense { weight, .. } => Some(weight.dtype()),
            // The MXFP8 product reads and writes BF16 rows whatever the builder pins.
            Weight::Mxfp8(_) => Some(DType::BF16),
            Weight::Nvfp4 { .. } | Weight::Fp8 { .. } => None,
        };

        Ok(Self {
            weight,
            bias,
            activation,
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
            Weight::Mxfp8(weight) => &weight.codes,
        }
    }

    /// Whether the weight is an NVFP4 matrix, whose product is compiled per row class
    /// ([`ops::fp4::packed_rows`]).
    pub const fn is_packed(&self) -> bool {
        matches!(self.weight, Weight::Nvfp4 { .. })
    }

    /// The element type of the rows the projection reads and writes, when its weight states it:
    /// a dense weight's own type, or an encoded weight's builder pin or, for a weight converted
    /// at load, the checkpoint's type its dense load would hold. `None` for a calibrated
    /// encoded weight under a builder that pins no type, whose rows take the caller's.
    pub const fn dtype(&self) -> Option<DType> {
        self.activation
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
                scaling,
            } => {
                let (encode, alpha) = scaling.factors(x)?;
                functional::linear_nvfp4(x, values, scales, bias, &encode, &alpha)?
            }
            Weight::Fp8 { weight, scaling } => {
                let (encode, alpha) = scaling.factors(x)?;
                functional::linear_fp8(x, weight, bias, &encode, &alpha)?
            }
            Weight::Mxfp8(weight) => functional::linear_mxfp8(x, weight, bias)?,
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
    fn forward(&self, x: &Tensor) -> tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// `extent` split over `split`'s ranks: this rank's share, refused when the split does not
/// divide.
fn share(vb: &VarBuilder<'_, '_>, extent: u64, split: Split, what: &str) -> Result<u64> {
    let size = u64::from(split.size());

    if !extent.is_multiple_of(size) {
        return Err(Error::weight(
            &vb.name("weight"),
            format!(
                "rank {} of a split of {} does not hold a share of {extent} {what}",
                split.rank(),
                split.size()
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
            start: u64::from(split.rank()) * rows * columns,
            shape,
        },
        Parallel::Row(split) => WeightSource::Strided {
            tensor,
            start: u64::from(split.rank()) * width,
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
        Parallel::Row(split) if split.rank() != 0 => Ok(None),
        Parallel::Row(_) | Parallel::None => Ok(Some(vb.get(local_out as usize, "bias")?)),
        Parallel::Column(split) => Ok(Some(vb.get_rows(
            local_out as usize,
            "bias",
            (u64::from(split.rank()) * local_out) as usize,
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

        if !config.heads.is_multiple_of(split.size()) {
            return Err(refuse(format!(
                "a split of {} does not divide {} query heads",
                split.size(),
                config.heads
            )));
        }

        // The key/value heads' own split: the group's, or, when the group is wider than the
        // heads, one head per `size / kv_heads` ranks.
        let kv_split = if split.size() <= config.kv_heads {
            if !config.kv_heads.is_multiple_of(split.size()) {
                return Err(refuse(format!(
                    "a split of {} does not divide {} key/value heads",
                    split.size(),
                    config.kv_heads
                )));
            }
            split
        } else {
            if !split.size().is_multiple_of(config.kv_heads) {
                return Err(refuse(format!(
                    "a split of {} does not replicate {} key/value heads evenly",
                    split.size(),
                    config.kv_heads
                )));
            }
            Split::new(
                config.kv_heads,
                split.rank() / (split.size() / config.kv_heads),
            )?
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
                    let parallel = if split.size() == 1 {
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
                    let rows = (width / split.size()) as usize;
                    let first = base as usize + split.rank() as usize * rows;
                    let weight = fused.get_rows((rows, hidden), "weight", first)?;

                    let bias = match config.bias {
                        QkvBias::Absent => None,
                        QkvBias::Present => Some(fused.get_rows(rows, "bias", first)?),
                        QkvBias::Windows(names) => {
                            Some(vb.get_rows(rows, names[index], split.rank() as usize * rows)?)
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
