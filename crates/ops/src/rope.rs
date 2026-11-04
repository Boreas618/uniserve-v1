//! Rotary position factors and the query/key normalization fused with their rotation.
//!
//! Two operators, because the factors are shared by every layer and the rotation is not:
//!
//! - `rotary_factors` turns a position per token into the cosine and sine factors of that position,
//!   once per forward. The frequency table itself is a device tensor: it is computed once at the
//!   loading boundary from the checkpoint's rope parameters (`inv_freq`, specs/worker.md 9) and
//!   never recomputed per step.
//! - `qk_norm_rope` normalizes each head's row by its RMS, applies the rotation in FP32 and rounds
//!   once into the output dtype, for queries and keys in one launch (specs/worker.md 9, "Q/K norm
//!   与 RoPE 融合").
//!
//! The rotation this operator expresses is the one the checkpoint's model uses: a rotation of
//! `rotated` dimensions of each head, in either the half-split pairing (`x[i]` against
//! `x[i + d/2]`, GPT-NeoX and Qwen) or the interleaved pairing (`x[2i]` against `x[2i+1]`).
//! A model that rotates several axes with different factors (multimodal position embeddings) is a
//! further descriptor value and another kernel program, added with the model that needs it.

use std::fmt;

use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// How the rotation pairs the elements of a head.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Rotation {
    /// `x[i]` rotates against `x[i + d/2]`.
    Half,
    /// `x[2i]` rotates against `x[2i + 1]`.
    Interleaved,
}

impl fmt::Display for Rotation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Half => "half",
            Self::Interleaved => "interleaved",
        })
    }
}

/// One compilation of `rotary_factors`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FactorsDescriptor {
    /// Element type of the cosine and sine outputs (FP32, the phase domain the model computes in).
    pub dtype: DType,
    /// Element type of the positions.
    pub positions: DType,
    /// Frequencies per position: the rotated width of one head divided by two.
    pub columns: u32,
    /// Elements one program handles.
    pub block: u32,
    /// The factor every angle is multiplied by, in bit form so the key is canonical.
    pub scale: Float,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for FactorsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("positions", self.positions)
            .field("columns", self.columns)
            .field("block", self.block)
            .field("scale", self.scale)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `rotary_factors`.
pub struct FactorsArgs {
    /// `[…, tokens]` positions, in any leading shape; every element is one token's position.
    pub positions: TensorView,
    /// `[columns]`, the angle per rotated dimension, in radians per position unit.
    pub frequencies: TensorView,
    /// `[tokens, columns]`, the cosine factors.
    pub cosine: TensorView,
    /// `[tokens, columns]`, the sine factors.
    pub sine: TensorView,
}

/// Position to rotary factor.
pub struct RotaryFactors;

impl Op for RotaryFactors {
    const NAME: &'static str = "rotary_factors";

    type Descriptor = FactorsDescriptor;
    type Args<'a> = FactorsArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if args.positions.dtype != descriptor.positions {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} positions, found {}",
                    descriptor.positions, args.positions.dtype
                ),
            ));
        }
        let columns = u64::from(descriptor.columns);
        args.frequencies
            .require_shape(Self::NAME, Shape::new(&[columns]))?;
        let tokens = args.positions.numel();
        args.cosine
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.sine
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        if args.cosine.rows() != tokens || args.sine.rows() != tokens {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{tokens} positions but {} cosine and {} sine rows",
                    args.cosine.rows(),
                    args.sine.rows()
                ),
            ));
        }

        let total = tokens * columns;
        let mut arguments = Arguments::new();
        arguments
            .view(args.positions)
            .view(args.frequencies)
            .view(args.cosine)
            .view(args.sine)
            .u64(total)
            .u32(descriptor.columns)
            .f32(f32::from_bits(descriptor.scale.0 as u32));
        kernel.launch(
            LaunchGrid::covering(total, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        FactorsArgs {
            cosine: twins.shadow_of(args.cosine),
            sine: twins.shadow_of(args.sine),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Two roundings in the phase (position times frequency) and one in each factor, plus the
        // absolute envelope a factor takes when its cosine or sine passes through zero.
        Tolerance::rounding(2, DType::F32, descriptor.dtype).with_atol(2.0 * f32::EPSILON)
    }
}

/// One compilation of `qk_norm_rope`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QkNormRopeDescriptor {
    /// Element type of the queries and keys.
    pub dtype: DType,
    /// Element type of the normalized, rotated outputs.
    pub output: DType,
    /// Elements per head.
    pub head_dim: u32,
    /// Query heads per token.
    pub q_heads: u32,
    /// Key/value heads per token.
    pub k_heads: u32,
    /// How the rotation pairs the elements of a head.
    pub rotation: Rotation,
    /// The RMS floor of the per-head normalization, in bit form so the key is canonical.
    pub epsilon: Float,
    /// Tokens one program handles.
    pub rows: u32,
    /// The head tile one program loads at a time.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for QkNormRopeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("output", self.output)
            .field("head_dim", self.head_dim)
            .field("q_heads", self.q_heads)
            .field("k_heads", self.k_heads)
            .field("rotation", self.rotation)
            .field("epsilon", self.epsilon)
            .field("rows", self.rows)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `qk_norm_rope`.
pub struct QkNormRopeArgs {
    /// `[tokens, q_heads, head_dim]` queries, before normalization.
    pub query: TensorView,
    /// `[tokens, k_heads, head_dim]` keys, before normalization.
    pub key: TensorView,
    /// `[head_dim]`, the query normalization's scale.
    pub q_weight: TensorView,
    /// `[head_dim]`, the key normalization's scale.
    pub k_weight: TensorView,
    /// `[tokens, head_dim/2]`, the cosine factors of each token's position.
    pub cosine: TensorView,
    /// `[tokens, head_dim/2]`, the sine factors of each token's position.
    pub sine: TensorView,
    /// `[tokens, q_heads, head_dim]` normalized, rotated queries.
    pub q_out: TensorView,
    /// `[tokens, k_heads, head_dim]` normalized, rotated keys.
    pub k_out: TensorView,
}

/// Per-head RMS normalization fused with the rotary rotation.
pub struct QkNormRope;

impl Op for QkNormRope {
    const NAME: &'static str = "qk_norm_rope";

    type Descriptor = QkNormRopeDescriptor;
    type Args<'a> = QkNormRopeArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let head_dim = u64::from(descriptor.head_dim);
        let pairs = head_dim / 2;
        let tokens = args.query.rows();
        if args.key.rows() != tokens {
            return Err(Error::argument(
                Self::NAME,
                format!("{} query tokens but {} key tokens", tokens, args.key.rows()),
            ));
        }
        args.query.require_shape(
            Self::NAME,
            Shape::new(&[tokens, u64::from(descriptor.q_heads), head_dim]),
        )?;
        args.key.require_shape(
            Self::NAME,
            Shape::new(&[tokens, u64::from(descriptor.k_heads), head_dim]),
        )?;
        args.q_out.require_shape(
            Self::NAME,
            Shape::new(&[tokens, u64::from(descriptor.q_heads), head_dim]),
        )?;
        args.k_out.require_shape(
            Self::NAME,
            Shape::new(&[tokens, u64::from(descriptor.k_heads), head_dim]),
        )?;
        if args.q_out.dtype != descriptor.output || args.k_out.dtype != descriptor.output {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} outputs, found {} and {}",
                    descriptor.output, args.q_out.dtype, args.k_out.dtype
                ),
            ));
        }
        args.q_weight
            .require_shape(Self::NAME, Shape::new(&[head_dim]))?;
        args.k_weight
            .require_shape(Self::NAME, Shape::new(&[head_dim]))?;
        args.cosine.require_row(Self::NAME, DType::F32, pairs)?;
        args.sine.require_row(Self::NAME, DType::F32, pairs)?;
        if args.cosine.rows() != tokens || args.sine.rows() != tokens {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{tokens} token rows but {} cosine and {} sine rows",
                    args.cosine.rows(),
                    args.sine.rows()
                ),
            ));
        }

        let programs = tokens.div_ceil(u64::from(descriptor.rows));
        let mut arguments = Arguments::new();
        arguments
            .view(args.query)
            .view(args.key)
            .view(args.q_weight)
            .view(args.k_weight)
            .view(args.cosine)
            .view(args.sine)
            .view(args.q_out)
            .view(args.k_out)
            .u64(tokens)
            .f32(f32::from_bits(descriptor.epsilon.0 as u32))
            .flag(descriptor.pdl);
        kernel.launch(LaunchGrid::linear(programs)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        QkNormRopeArgs {
            q_out: twins.shadow_of(args.q_out),
            k_out: twins.shadow_of(args.k_out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The per-head RMS reduces over the head in FP32, then one rotation product and one
        // rounding to the output dtype.
        Tolerance::rounding(
            u64::from(descriptor.head_dim),
            DType::F32,
            descriptor.output,
        )
    }
}
