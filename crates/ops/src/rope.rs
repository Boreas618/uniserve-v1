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
    /// How many position axes the factors read: 1 for one-dimensional positions.
    ///
    /// A multimodal transformer's positions are one coordinate per axis (time, height, width), and
    /// the head's columns are cut into one section per axis, so column `j` rotates by the
    /// coordinate of the axis its section belongs to (M-RoPE).
    pub axes: u32,
    /// Rotated elements each axis owns, in axis order; only the first `axes` entries are read.
    pub sections: [u32; MAX_AXES],
}

/// The most position axes a sectioned rotation reads.
///
/// Qwen2.5-VL's M-RoPE cuts the head into three (time, height, width); a fourth is room for a
/// model that adds a modality axis, and the kernels' per-column search is a scan over four entries.
pub const MAX_AXES: usize = 4;

/// `widths` as a key value: `64` or `16-24-24`.
struct Sections<'a>(&'a [u32]);

impl fmt::Display for Sections<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, width) in self.0.iter().enumerate() {
            if index > 0 {
                formatter.write_str("-")?;
            }
            write!(formatter, "{width}")?;
        }
        Ok(())
    }
}

impl fmt::Display for FactorsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut key = Key::new()
            .field("dtype", self.dtype)
            .field("positions", self.positions)
            .field("columns", self.columns)
            .field("block", self.block)
            .field("scale", self.scale)
            .field("pdl", self.pdl);
        // A one-dimensional descriptor's key is the one it always had: `axes` and `sections`
        // appear only when there is more than one axis, because one axis compiles the same
        // program as no axis at all.
        if self.axes > 1 {
            let axes = self.axes as usize;
            key = key
                .field("axes", self.axes)
                .field("sections", Sections(&self.sections[..axes.min(MAX_AXES)]));
        }
        key.finish(formatter)
    }
}

/// The arguments of `rotary_factors`.
#[derive(Clone, Copy)]
pub struct FactorsArgs {
    /// `[…, tokens]` positions, in any leading shape; every element is one token's position.
    ///
    /// A sectioned rotation's positions are `[axes, tokens]` (or `[axes, …, tokens]`): one
    /// coordinate per axis per token, contiguous, so axis `a`'s coordinates start at `a * tokens`.
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
        let axes = u64::from(descriptor.axes.max(1));
        if args.positions.numel() % axes != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} position coordinates do not divide into {axes} axes",
                    args.positions.numel()
                ),
            ));
        }
        if descriptor.axes > 1 {
            let sections: u64 = descriptor
                .sections
                .iter()
                .take(descriptor.axes as usize)
                .map(|width| u64::from(*width))
                .sum();
            if sections != columns * 2 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "sections cover {sections} elements of a {} element head",
                        columns * 2
                    ),
                ));
            }
            if descriptor.axes as usize > MAX_AXES {
                return Err(Error::argument(
                    Self::NAME,
                    format!("{} axes, at most {MAX_AXES}", descriptor.axes),
                ));
            }
        }
        let tokens = args.positions.numel() / axes;
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
        // A sectioned rotation's program takes one more pair: how many axes the positions carry
        // and the rotated width each one owns. A one-dimensional program's arguments end at
        // `scale`, so its existing packages stay valid.
        if descriptor.axes > 1 {
            arguments.u32(descriptor.axes);
            for width in descriptor.sections.iter().take(MAX_AXES) {
                arguments.u32(*width);
            }
        }
        kernel.launch(
            LaunchGrid::covering(total, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
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
    /// Channels of each head the rotation turns; `head_dim` for a full-width rotation.
    ///
    /// A multimodal transformer rotates the leading channels of a head and leaves the rest as they
    /// are — H3 rotates 96 of 128 — so the tables carry `rotated / 2` pair columns rather than
    /// `head_dim / 2`, and the program copies the trailing channels through. The key names this
    /// only when it is narrower than the head, so every full-width key is unchanged.
    pub rotated: u32,
    /// Tokens one program handles.
    pub rows: u32,
    /// The head tile one program loads at a time.
    pub block: u32,
    /// Whether the per-head normalization has a learned scale.
    ///
    /// `true` is the trained transformer's form: each head's RMS is scaled by a `[head_dim]` weight
    /// per query and key head, which the program reads. `false` is a per-head normalization with no
    /// learned weight — the video VAE's decoder normalizes this way — so the program has no weight
    /// ports at all and the arguments carry none.
    pub affine: bool,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for QkNormRopeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", self.dtype)
            .field("output", self.output)
            .field("head_dim", self.head_dim)
            .field("q_heads", self.q_heads)
            .field("k_heads", self.k_heads)
            .field("rotation", self.rotation)
            .field("epsilon", self.epsilon)
            .field("rows", self.rows)
            .field("block", self.block)
            .field("pdl", self.pdl);
        // A narrowed rotation is its own variant; a full-width one keeps the key it had.
        let key = if self.rotated == self.head_dim {
            key
        } else {
            key.field("rotated", self.rotated)
        };
        // Named only for the unweighted form, so every key a trained transformer compiles stays
        // exactly the key it was: the video VAE's weightless per-head norm is a new variant, not a
        // renamed old one.
        let key = if self.affine {
            key
        } else {
            key.field("affine", false)
        };
        key.finish(formatter)
    }
}

/// The arguments of `qk_norm_rope`.
#[derive(Clone, Copy)]
pub struct QkNormRopeArgs {
    /// `[tokens, q_heads, head_dim]` queries, before normalization.
    pub query: TensorView,
    /// `[tokens, k_heads, head_dim]` keys, before normalization.
    pub key: TensorView,
    /// `[head_dim]`, the query normalization's scale; `None` when the descriptor is unweighted.
    pub q_weight: Option<TensorView>,
    /// `[head_dim]`, the key normalization's scale; `None` when the descriptor is unweighted.
    pub k_weight: Option<TensorView>,
    /// `[tokens, rotated/2]`, the cosine factors of each token's position.
    pub cosine: TensorView,
    /// `[tokens, rotated/2]`, the sine factors of each token's position.
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
        if descriptor.rotated == 0
            || u64::from(descriptor.rotated) > head_dim
            || descriptor.rotated % 2 != 0
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a rotation of {} channels does not fit a head of {head_dim} with whole pairs",
                    descriptor.rotated
                ),
            ));
        }
        let pairs = u64::from(descriptor.rotated) / 2;
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
        // The weights exist exactly when the descriptor says the normalization has them: a program
        // compiled unweighted has no weight ports, and one compiled weighted cannot normalize
        // without them.
        let weights = if descriptor.affine {
            let q_weight = args.q_weight.ok_or_else(|| {
                Error::argument(
                    Self::NAME,
                    "a weighted normalization was launched without its query weight".to_owned(),
                )
            })?;
            let k_weight = args.k_weight.ok_or_else(|| {
                Error::argument(
                    Self::NAME,
                    "a weighted normalization was launched without its key weight".to_owned(),
                )
            })?;
            q_weight.require_shape(Self::NAME, Shape::new(&[head_dim]))?;
            k_weight.require_shape(Self::NAME, Shape::new(&[head_dim]))?;
            Some((q_weight, k_weight))
        } else {
            if args.q_weight.is_some() || args.k_weight.is_some() {
                return Err(Error::argument(
                    Self::NAME,
                    "an unweighted normalization was launched with a weight; the descriptor and \
                     the arguments disagree"
                        .to_owned(),
                ));
            }
            None
        };
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
        arguments.view(args.query).view(args.key);
        if let Some((q_weight, k_weight)) = weights {
            arguments.view(q_weight).view(k_weight);
        }
        arguments
            .view(args.cosine)
            .view(args.sine)
            .view(args.q_out)
            .view(args.k_out)
            .u64(tokens)
            .f32(f32::from_bits(descriptor.epsilon.0 as u32))
            .flag(descriptor.pdl);
        // A full rotation's arguments end at `pdl`, so its existing packages stay valid; a partial
        // one names how many channels rotate.
        if descriptor.rotated != descriptor.head_dim {
            arguments.u32(descriptor.rotated);
        }
        kernel.launch(LaunchGrid::linear(programs)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        QkNormRopeArgs {
            q_out: twins.shadow_of(args.q_out),
            k_out: twins.shadow_of(args.k_out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The per-head RMS reduces over the head in FP32, then one rotation product and one
        // rounding to the output dtype; the channels past `rotated` are copied and cost nothing.
        Tolerance::rounding(
            u64::from(descriptor.head_dim),
            DType::F32,
            descriptor.output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(affine: bool) -> QkNormRopeDescriptor {
        QkNormRopeDescriptor {
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: 128,
            q_heads: 32,
            k_heads: 4,
            rotation: Rotation::Half,
            rotated: 128,
            epsilon: Float::from(1e-6_f32),
            rows: 1,
            block: 128,
            affine,
            pdl: false,
        }
    }

    #[test]
    fn a_weighted_normalization_keeps_the_key_it_already_had() {
        // The Qwen3 slice's key, exactly: the learned-scale variant does not name the flag, so no
        // package has to repackage the transformer for the video VAE's weightless norm.
        assert_eq!(
            descriptor(true).to_string(),
            "dtype_bf16_output_bf16_head_dim_128_q_heads_32_k_heads_4_rotation_half_epsilon_0x358637bd_rows_1_block_128_pdl_false"
        );
    }

    #[test]
    fn a_full_rotation_keeps_the_key_and_a_partial_one_names_its_width() {
        assert_eq!(
            descriptor(true).to_string(),
            "dtype_bf16_output_bf16_head_dim_128_q_heads_32_k_heads_4_rotation_half_epsilon_0x358637bd_rows_1_block_128_pdl_false"
        );
        // H3's 96-of-128 rotary: the same descriptor with a narrower rotation is another module.
        let partial = QkNormRopeDescriptor {
            rotated: 96,
            ..descriptor(true)
        };
        assert_eq!(
            partial.to_string(),
            "dtype_bf16_output_bf16_head_dim_128_q_heads_32_k_heads_4_rotation_half_epsilon_0x358637bd_rows_1_block_128_pdl_false_rotated_96"
        );
    }

    #[test]
    fn a_rotation_that_leaves_a_head_or_has_no_pair_is_refused() {
        for rotated in [0, 127, 130] {
            let descriptor = QkNormRopeDescriptor {
                rotated,
                ..descriptor(true)
            };
            // The check runs inside `launch`, which needs a kernel; the rule itself is pure, so it
            // is asserted here on the values it must reject.
            assert!(
                descriptor.rotated == 0
                    || descriptor.rotated > descriptor.head_dim
                    || !descriptor.rotated.is_multiple_of(2),
                "{rotated} must be rejected"
            );
        }
    }

    #[test]
    fn an_unweighted_normalization_is_its_own_variant() {
        let unweighted = descriptor(false).to_string();
        assert_eq!(unweighted, format!("{}_affine_false", descriptor(true)));
        assert_ne!(unweighted, descriptor(true).to_string());
    }
}
