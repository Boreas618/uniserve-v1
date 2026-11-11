//! The pre-norm transformer block: each sub-layer reads its input normalized, and its output joins
//! the residual stream, scaled per channel when the block has a layer scale.
//!
//! ```text
//! normalized = norm1(stream)                     (the previous block's update joined first)
//! update     = attention(normalized)             the caller's
//! stream    += update · scale1                   ┐ joined as the block's residual form states
//! normalized = norm2(stream)                     ┘
//! update     = mlp(normalized)                   returned, joined by the next norm
//! ```
//!
//! A block's last update is not joined by the block itself: it is returned as a [`Pending`] update,
//! and the next block's first norm - or the stack's final norm - joins it. That is the residual
//! convention of the references' decoder layers, and it lets the join and the norm be one launch.
//! Whether they are is the block's [`ResidualForm`], because the two forms round differently: a
//! fused join normalizes the unrounded FP32 sum, a separate join normalizes the stream it stored.

use ops::Operand;
use uniserve_tensor::DType;

use crate::error::Result;
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::{Epilogue, Gain, Residual, ResidualStream, RmsNorm, Scale};
use crate::weights::{Scope, Weight};

use super::{GatedMlp, GatedMlpSpec};

/// How a sub-layer's update joins the residual stream before the next norm reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualForm {
    /// One launch adds the update into the stream and stores it, then the norm reads the stored
    /// stream.
    Separate,
    /// The norm's launch adds the update into the stream, stores the sum and normalizes the
    /// unrounded sum.
    Fused,
}

/// A sub-layer's output not yet joined into the stream, with the layer scale it joins under.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    /// `[rows, width]`: the update.
    pub update: Operand,
    /// `[width]`: the update's per-channel scale, when the block has one.
    pub scale: Option<Weight>,
}

impl Pending {
    /// Add the update into `stream` in place, under its scale: the separate join, for a stack whose
    /// final norm does not fold the last update.
    ///
    /// # Errors
    ///
    /// Views of another shape, or the operator's refusal.
    pub fn join(&self, f: &Forward, stream: Operand) -> Result<()> {
        let scale = self.scale.map_or(Scale::None, Scale::Channel);
        Residual::new(scale, None)?.forward(f, stream, self.update, None)?;
        Ok(())
    }
}

/// The module names and arithmetic of a [`PreNormBlock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreNormSpec<'n> {
    /// The normalized width.
    pub hidden: u32,
    /// The norms' variance floor.
    pub eps: f32,
    /// The element type the norms write: the projections' activation type, which may be narrower
    /// than the stream's.
    pub activation: DType,
    /// The attention input's norm module.
    pub norm1: &'n str,
    /// The MLP input's norm module.
    pub norm2: &'n str,
    /// The per-channel layer scales of the attention's and the MLP's updates, as tensors of the
    /// block's scope, when the block has them.
    pub layer_scale: Option<[&'n str; 2]>,
    /// The element type the norm gains and layer scales are read at, when the block pins it apart
    /// from its projections (a stream wider than the rows its projections read).
    pub norm_dtype: Option<DType>,
    /// How each update joins the stream.
    pub residual: ResidualForm,
    /// The block's MLP.
    pub mlp: GatedMlpSpec<'n>,
}

/// One pre-norm transformer block: two RMS norms, the caller's attention, a gated MLP and the
/// optional layer scales of their updates.
pub struct PreNormBlock {
    norm1: Weight,
    norm2: Weight,
    eps: f32,
    activation: DType,
    layer_scale: Option<[Weight; 2]>,
    residual: ResidualForm,
    mlp: GatedMlp,
}

impl PreNormBlock {
    /// The block the module at `scope` states, its MLP at `split`'s rank.
    ///
    /// # Errors
    ///
    /// A gain, a layer scale or an MLP projection the loader refuses.
    pub fn load(scope: &mut Scope<'_, '_>, spec: PreNormSpec<'_>, split: Split) -> Result<Self> {
        let width = [u64::from(spec.hidden)];
        // A gain or a scale is read at the pinned type when the block states one.
        let vector = |scope: &mut Scope<'_, '_>, leaf: &str| -> Result<Weight> {
            match spec.norm_dtype {
                Some(dtype) => scope.pinned(dtype).tensor(leaf, &width),
                None => scope.tensor(leaf, &width),
            }
        };
        let norm1 = vector(&mut scope.scope(spec.norm1), "weight")?;
        let norm2 = vector(&mut scope.scope(spec.norm2), "weight")?;
        let layer_scale = match spec.layer_scale {
            Some([first, second]) => Some([vector(scope, first)?, vector(scope, second)?]),
            None => None,
        };
        Ok(Self {
            norm1,
            norm2,
            eps: spec.eps,
            activation: spec.activation,
            layer_scale,
            residual: spec.residual,
            mlp: GatedMlp::load(scope, spec.mlp, split)?,
        })
    }

    /// The block over `stream` `[rows, width]`, updated in place, with `attend(normalized)` the
    /// attention answering its update.
    ///
    /// `pending` is the previous block's update, joined by this block's first norm; `None` for the
    /// first block, whose norm reads the stream alone. `carried` is the stack's `[rows, width]`
    /// buffer of the activation type that holds a block's last update until the next block joins
    /// it: it may be `pending`'s own buffer, which the first norm has consumed before the MLP
    /// writes it again. The returned update is the MLP's output in `carried`, not yet joined. The
    /// block's other intermediates are handed back before return.
    ///
    /// # Errors
    ///
    /// Views a norm, the residual or the MLP refuses, or the attention's own error.
    pub fn forward(
        &self,
        f: &Forward,
        stream: Operand,
        pending: Option<Pending>,
        carried: Operand,
        attend: impl FnOnce(Operand) -> Result<Operand>,
    ) -> Result<Pending> {
        f.scoped(|f| {
            let normalized = self.normalize(f, self.norm1, stream, pending)?;
            let attended = Pending {
                update: attend(normalized)?,
                scale: self.layer_scale.map(|[first, _]| first),
            };
            let normalized = self.normalize(f, self.norm2, stream, Some(attended))?;
            self.mlp.forward_into(f, normalized, carried)
        })?;
        Ok(Pending {
            update: carried,
            scale: self.layer_scale.map(|[_, second]| second),
        })
    }

    /// `norm(stream)` after joining `pending` into the stream in the block's form, in the block's
    /// activation type.
    fn normalize(
        &self,
        f: &Forward,
        weight: Weight,
        stream: Operand,
        pending: Option<Pending>,
    ) -> Result<Operand> {
        let plain = || {
            RmsNorm::new(
                weight,
                self.eps,
                Epilogue::None,
                Gain::Plain,
                self.activation,
            )
        };
        match (pending, self.residual) {
            (None, _) => plain()?.forward(f, stream, None),
            (Some(pending), ResidualForm::Separate) => {
                pending.join(f, stream)?;
                plain()?.forward(f, stream, None)
            }
            (Some(pending), ResidualForm::Fused) => {
                let epilogue = match pending.scale {
                    Some(scale) => Epilogue::ScaledResidual { scale, bias: None },
                    None => Epilogue::None,
                };
                RmsNorm::new(weight, self.eps, epilogue, Gain::Plain, self.activation)?.forward(
                    f,
                    pending.update,
                    Some(ResidualStream::in_place(stream)),
                )
            }
        }
    }
}
