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
    FactorsArgs, FactorsDescriptor, QkNormRope, QkNormRopeArgs, QkNormRopeDescriptor,
    RotaryFactors, Rotation,
};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::{PDL, QK_NORM_TOKENS, ROTARY_BLOCK};
use crate::weights::Weight;

/// Rotary embedding of `head_dim` with Q/K normalization.
pub struct Rotary {
    frequencies: Weight,
    head_dim: u32,
    q_heads: u32,
    k_heads: u32,
    epsilon: f32,
    rotation: Rotation,
    scale: f32,
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
        if frequencies.shape().numel() != u64::from(head_dim / 2) {
            return Err(Error::shape(
                "Rotary",
                format!(
                    "{} frequencies for {head_dim} rotated elements",
                    frequencies.shape().numel()
                ),
            ));
        }
        Ok(Self {
            frequencies,
            head_dim,
            q_heads,
            k_heads,
            epsilon,
            rotation,
            scale,
        })
    }

    /// The rotated width of one head.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// The rotation pairing.
    pub const fn rotation(&self) -> Rotation {
        self.rotation
    }

    /// Turn `positions` into `cosine` and `sine` tables, both `[tokens, head_dim / 2]` FP32.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        let columns = u64::from(self.head_dim / 2);
        cosine
            .require_row("Rotary::factors", ops::DType::F32, columns)
            .map_err(Error::from)?;
        sine.require_row("Rotary::factors", ops::DType::F32, columns)
            .map_err(Error::from)?;
        let tokens = positions.numel();
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
            columns: self.head_dim / 2,
            block: ROTARY_BLOCK,
            scale: self.scale.into(),
            pdl: PDL,
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
    /// `[tokens, k_heads, head_dim]`; the two weights are `[head_dim]`.
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
        q_weight: TensorView,
        k_weight: TensorView,
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
        let columns = head_dim / 2;
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
        let descriptor = QkNormRopeDescriptor {
            dtype: query.dtype,
            output: q_out.dtype,
            head_dim: self.head_dim,
            q_heads: self.q_heads,
            k_heads: self.k_heads,
            rotation: self.rotation,
            epsilon: self.epsilon.into(),
            rows: QK_NORM_TOKENS,
            block: head_dim.next_power_of_two().min(1024) as u32,
            pdl: PDL,
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
