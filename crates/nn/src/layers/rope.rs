//! Rotary position factors and their application, fused with per-head Q/K normalization.
//!
//! Two operators, one layer, because the factors depend only on positions and are shared by every
//! attention layer while the rotation itself is per layer: [`Rotary::factors`] turns the batch's
//! positions into FP32 cosine and sine tables, and [`Rotary::apply`] normalizes each query and key
//! head and rotates it (specs/worker.md 9, "RoPE 因子" and "Q/K norm 与 RoPE 融合"). The fused
//! operator rotates in FP32 and rounds once, which is the recipe the checkpoint's model was trained
//! with; the frequencies are computed by the loading boundary, not here (specs/worker.md 9).

use ops::TensorView;
use ops::rope::{
    FactorsArgs, FactorsDescriptor, MAX_AXES, QkNormRope, QkNormRopeArgs, QkNormRopeDescriptor,
    RotaryFactors, Rotation,
};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::{PDL, QK_NORM_TOKENS, ROTARY_BLOCK};
use crate::weights::Weight;

/// The position axes of a sectioned rotation: M-RoPE's cut of the head into one section per axis.
///
/// A multimodal transformer's positions are one coordinate per axis (time, height, width) and its
/// head's rotated width is divided among them in this order, so each column rotates by the
/// coordinate of the axis its section belongs to (specs/worker.md 9, "RoPE 因子").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sections {
    widths: [u32; MAX_AXES],
    axes: u32,
    interleaved: bool,
}

impl Sections {
    /// The sections `widths`, in rotated elements per axis, in axis order.
    ///
    /// Every width must be a positive even number of elements, the widths must sum to the head's
    /// rotated width, and there may be at most [`MAX_AXES`] of them.
    pub fn new(widths: &[u32]) -> Result<Self> {
        if widths.is_empty() || widths.len() > MAX_AXES {
            return Err(Error::shape(
                "Rotary::sections",
                format!("{} sections, one to {MAX_AXES} axes", widths.len()),
            ));
        }
        let mut stored = [0_u32; MAX_AXES];
        for (slot, width) in stored.iter_mut().zip(widths) {
            if *width == 0 || !width.is_multiple_of(2) {
                return Err(Error::shape(
                    "Rotary::sections",
                    format!("a section of {width} elements has no rotation pair"),
                ));
            }
            *slot = *width;
        }
        Ok(Self {
            widths: stored,
            axes: widths.len() as u32,
            interleaved: false,
        })
    }

    /// Cycle the axes over the whole spectrum instead of cutting it in order.
    ///
    /// Qwen3-VL's interleaved M-RoPE gives frequency `j` the axis `j % axes` while the axis owns
    /// `j`, so every axis keeps frequencies from the whole range; the contiguous cut, the default,
    /// gives axis `a` the next `sections[a]` frequencies. The two are different programs
    /// (`ops::rope::FactorsDescriptor::interleaved`).
    pub const fn interleaved(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// Whether the axes cycle over the spectrum.
    pub const fn is_interleaved(&self) -> bool {
        self.interleaved
    }

    /// How many position axes the rotation reads.
    pub const fn axes(&self) -> u32 {
        self.axes
    }

    /// The rotated elements each axis owns, in axis order.
    pub fn widths(&self) -> &[u32] {
        &self.widths[..self.axes as usize]
    }
}

/// Rotary embedding of `head_dim` with Q/K normalization.
pub struct Rotary {
    frequencies: Weight,
    head_dim: u32,
    rotated: u32,
    q_heads: u32,
    k_heads: u32,
    epsilon: f32,
    rotation: Rotation,
    scale: f32,
    sections: Option<Sections>,
}

impl Rotary {
    /// The embedding of `head_dim` with `q_heads` query and `k_heads` key/value heads.
    ///
    /// `frequencies` is `[head_dim / 2]` FP32 radians per position unit, computed once at loading;
    /// `epsilon` is the per-head normalization's floor; `scale` multiplies every angle (1.0 for the
    /// public RoPE domain).
    #[allow(
        clippy::too_many_arguments,
        reason = "every argument is a geometry or weight the design states separately; a \
                  configuration struct would restate them"
    )]
    pub fn new(
        frequencies: Weight,
        head_dim: u32,
        q_heads: u32,
        k_heads: u32,
        epsilon: f32,
        rotation: Rotation,
        scale: f32,
    ) -> Result<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(Error::shape(
                "Rotary",
                format!("a head of {head_dim} elements has no rotation pair"),
            ));
        }
        let pairs = frequencies.shape().numel();
        if pairs == 0 || pairs * 2 > u64::from(head_dim) {
            return Err(Error::shape(
                "Rotary",
                format!("{pairs} frequency pairs for a {head_dim} element head"),
            ));
        }
        Ok(Self {
            frequencies,
            head_dim,
            rotated: head_dim,
            q_heads,
            k_heads,
            epsilon,
            rotation,
            scale,
            sections: None,
        })
    }

    /// Rotate the leading `rotated` channels of each head and pass the rest through.
    ///
    /// A multimodal transformer rotates part of a head — H3 turns 96 of its 128 channels and copies
    /// the trailing 32 as they are — so the factors are `rotated / 2` pairs wide and the sections
    /// cut `rotated`, not the head. `rotated` is even, at least two, at most the head's width, and
    /// the frequency table carries exactly `rotated / 2` pairs.
    pub fn partial(mut self, rotated: u32) -> Result<Self> {
        if rotated < 2 || !rotated.is_multiple_of(2) || rotated > self.head_dim {
            return Err(Error::shape(
                "Rotary::partial",
                format!(
                    "a rotation of {rotated} channels does not fit a {} element head with whole pairs",
                    self.head_dim
                ),
            ));
        }
        let pairs = self.frequencies.shape().numel();
        if pairs != u64::from(rotated / 2) {
            return Err(Error::shape(
                "Rotary::partial",
                format!(
                    "{pairs} frequency pairs for {rotated} rotated elements of a {} element head",
                    self.head_dim
                ),
            ));
        }
        self.rotated = rotated;
        Ok(self)
    }

    /// The same rotation, reading one position coordinate per axis (M-RoPE).
    ///
    /// The sections must cut the head's rotated width exactly: their widths sum to `rotated`.
    pub fn sectioned(mut self, sections: Sections) -> Result<Self> {
        let covered: u32 = sections.widths().iter().sum();
        if covered != self.rotated {
            return Err(Error::shape(
                "Rotary::sectioned",
                format!(
                    "sections cover {covered} elements of a {} channel rotation",
                    self.rotated
                ),
            ));
        }
        self.sections = Some(sections);
        Ok(self)
    }

    /// The width of one head.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// The rotated width of one head: `head_dim` unless the rotation was narrowed.
    pub const fn rotated(&self) -> u32 {
        self.rotated
    }

    /// The rotation pairing.
    pub const fn rotation(&self) -> Rotation {
        self.rotation
    }

    /// Turn `positions` into `cosine` and `sine` tables, both `[tokens, rotated / 2]` FP32.
    ///
    /// A one-dimensional rotation's `positions` is `[…, tokens]`; a sectioned rotation's is
    /// `[axes, …, tokens]`, one coordinate per axis per token, and the tables cover the tokens.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        let columns = u64::from(self.rotated / 2);
        cosine
            .require_row("Rotary::factors", ops::DType::F32, columns)
            .map_err(Error::from)?;
        sine.require_row("Rotary::factors", ops::DType::F32, columns)
            .map_err(Error::from)?;
        let axes = self
            .sections
            .map_or(1, |sections| u64::from(sections.axes()));
        if let Some(sections) = self.sections
            && (positions.shape.rank() < 2 || positions.shape.dim(0) != u64::from(sections.axes()))
        {
            return Err(Error::shape(
                "Rotary::factors",
                format!(
                    "sectioned positions need one leading coordinate per axis, {} in all",
                    sections.axes()
                ),
            ));
        }
        if !positions.numel().is_multiple_of(axes) {
            return Err(Error::shape(
                "Rotary::factors",
                format!(
                    "{} position coordinates do not divide into {axes} axes",
                    positions.numel()
                ),
            ));
        }
        let tokens = positions.numel() / axes;
        if cosine.rows() != tokens || sine.rows() != tokens {
            return Err(Error::shape(
                "Rotary::factors",
                format!(
                    "{tokens} positions but {} and {} factor rows",
                    cosine.rows(),
                    sine.rows()
                ),
            ));
        }
        let descriptor = FactorsDescriptor {
            dtype: ops::DType::F32,
            positions: positions.dtype,
            columns: self.rotated / 2,
            block: ROTARY_BLOCK,
            scale: self.scale.into(),
            pdl: PDL,
            axes: self.sections.map_or(1, |sections| sections.axes()),
            interleaved: self
                .sections
                .is_some_and(|sections| sections.is_interleaved()),
            sections: self
                .sections
                .map_or([0; MAX_AXES], |sections| sections.widths),
        };
        let args = FactorsArgs {
            positions,
            frequencies: self.frequencies.view(),
            cosine,
            sine,
        };

        f.operate::<RotaryFactors>(&descriptor, args)
    }

    /// Normalize and rotate `query` and `key` into `q_out` and `k_out`.
    ///
    /// `query` and `q_out` are `[tokens, q_heads, head_dim]`, `key` and `k_out`
    /// `[tokens, k_heads, head_dim]`; the two weights are `[head_dim]`, given together or both
    /// absent.
    ///
    /// A trained transformer's per-head normalization scales the head by a learned weight; the video
    /// VAE's decoder normalizes each head without one, and that call passes neither weight, which
    /// compiles a program with no weight ports (specs/worker.md 9, the per-head norm row;
    /// `ops::rope::QkNormRopeDescriptor::affine`).
    #[allow(
        clippy::too_many_arguments,
        reason = "a rotation call carries the two projections, their norms, the two factor tables \
                  and the two outputs; the design states them separately"
    )]
    pub fn apply(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        q_weight: Option<TensorView>,
        k_weight: Option<TensorView>,
        cosine: TensorView,
        sine: TensorView,
        q_out: TensorView,
        k_out: TensorView,
    ) -> Result<()> {
        let head_dim = u64::from(self.head_dim);
        let q_heads = u64::from(self.q_heads);
        let k_heads = u64::from(self.k_heads);
        for (name, view, heads) in [
            ("query", query, q_heads),
            ("q_out", q_out, q_heads),
            ("key", key, k_heads),
            ("k_out", k_out, k_heads),
        ] {
            if view.shape.rank() != 3 || view.shape.dim(1) != heads || view.shape.dim(2) != head_dim
            {
                return Err(Error::shape(
                    "Rotary::apply",
                    format!(
                        "{name} is {:?}, [{rows}, {heads}, {head_dim}] expected",
                        view.shape.dims(),
                        rows = view.rows() / heads.max(1)
                    ),
                ));
            }
            view.require_row("Rotary::apply", view.dtype, head_dim)
                .map_err(Error::from)?;
        }
        let tokens = query.shape.dim(0);
        if key.shape.dim(0) != tokens
            || q_out.shape.dim(0) != tokens
            || k_out.shape.dim(0) != tokens
        {
            return Err(Error::shape(
                "Rotary::apply",
                "the query and key tensors have different token counts".to_owned(),
            ));
        }
        let columns = u64::from(self.rotated / 2);
        if cosine.numel() != tokens * columns || sine.numel() != tokens * columns {
            return Err(Error::shape(
                "Rotary::apply",
                format!(
                    "{} and {} factor elements for {tokens} tokens of {columns} factors",
                    cosine.numel(),
                    sine.numel()
                ),
            ));
        }
        let affine = match (q_weight, k_weight) {
            (Some(q_weight), Some(k_weight))
                if q_weight.numel() == head_dim && k_weight.numel() == head_dim =>
            {
                Some((q_weight, k_weight))
            }
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(Error::shape(
                    "Rotary::apply",
                    format!("per-head weights of {head_dim} elements"),
                ));
            }
            _ => {
                return Err(Error::shape(
                    "Rotary::apply",
                    "the per-head weights are given together, or both absent".to_owned(),
                ));
            }
        };
        let descriptor = QkNormRopeDescriptor {
            dtype: query.dtype,
            output: q_out.dtype,
            head_dim: self.head_dim,
            q_heads: self.q_heads,
            k_heads: self.k_heads,
            rotation: self.rotation,
            epsilon: self.epsilon.into(),
            // Full-width unless the configuration narrows it.
            rotated: self.rotated,
            rows: QK_NORM_TOKENS,
            block: head_dim.next_power_of_two().min(1024) as u32,
            affine: affine.is_some(),
            pdl: PDL,
        };
        let (q_weight, k_weight) = match affine {
            Some((q_weight, k_weight)) => (Some(q_weight), Some(k_weight)),
            None => (None, None),
        };
        let args = QkNormRopeArgs {
            query,
            key,
            q_weight,
            k_weight,
            cosine,
            sine,
            q_out,
            k_out,
        };

        f.operate::<QkNormRope>(&descriptor, args)
    }
}

#[cfg(test)]
mod tests {
    use super::Sections;

    /// A sectioned rotation cuts the head exactly: the sections are the axes' shares of it.
    #[test]
    fn sections_cut_the_head_into_one_share_per_axis() {
        let sections = Sections::new(&[16, 24, 24]).expect("three shares of a 128 element head");
        assert_eq!(sections.axes(), 3);
        assert_eq!(sections.widths(), &[16, 24, 24]);

        assert!(Sections::new(&[]).is_err(), "no axis rotates nothing");
        assert!(
            Sections::new(&[16, 24, 23]).is_err(),
            "an odd share has no pair"
        );
        assert!(
            Sections::new(&[0, 64]).is_err(),
            "an empty share rotates nothing"
        );
        assert!(
            Sections::new(&[16, 16, 16, 16, 16]).is_err(),
            "more axes than the kernel reads"
        );
    }
}
