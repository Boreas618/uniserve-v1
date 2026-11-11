//! The adaptive-layer-norm (AdaLN) block of a diffusion transformer, and its final modulated norm.
//!
//! A diffusion block is conditioned on the step's noise levels through per-row affine vectors: the
//! attention's input and the MLP's input are each normalized and then shifted and scaled, and each
//! sub-layer's output joins the residual stream through a gate. One step's vectors are gathered
//! rows of a [`ModulationTable`], and every activation row selects its own parameter row (`rows`),
//! so one launch conditions rows of several timesteps at once:
//!
//! ```text
//! normalized = rms_norm(hidden) · (1 + scale_a[row]) + shift_a[row]            attention_input
//! update     = attention(normalized)                                           the caller's
//! residual   = hidden + gate_a[row] · update                                   ┐ one launch,
//! normalized = rms_norm(residual) · (1 + scale_m[row]) + shift_m[row]          ┘ the residual retained
//! hidden     = residual + gate_m[row] · mlp(normalized)                        feed_forward
//! ```
//!
//! The vectors of one parameter row are stored in the order the reference projections emit them:
//! shift, scale and gate of the attention, then shift, scale and gate of the MLP. [`AdaLnOut`] is
//! the stack's last norm, modulated by a shift and a scale and ungated.
//!
//! The attention is the caller's: a block's attention may be dense or sparse, local or exchanged
//! over a sequence-parallel group, and an exchanged attention returns its rows interval by
//! interval, so the block exposes its two modulated stages beside the whole forward. Each stage's
//! intermediates are reserved from the forward's arena and handed back before it returns.

use ops::{DType, DevicePtr, Shape, Strides, TensorView};

use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::{Epilogue, Gain, Modulation, Residual, ResidualStream, RmsNorm, Scale};
use crate::weights::{Scope, Weight};

use super::{GatedMlp, GatedMlpSpec};

/// The affine vectors one parameter row of a diffusion block holds, in storage order.
const BLOCK_VECTORS: u64 = 6;

/// The vectors one parameter row of the final norm holds: shift, then scale.
const OUTPUT_VECTORS: u64 = 2;

/// The step's gathered modulation rows: `[parameters, vectors · width]`, one parameter row per
/// timestep and token kind the step conditions on.
///
/// A block's table holds six vectors per row (shift, scale, gate of the attention, then of the
/// MLP); the final norm's holds two (shift, scale). The vectors are row-strided views of the
/// table, so nothing is copied: a launch reads vector `k` of parameter row `p` at
/// `table[p, k · width ..]`.
#[derive(Clone, Copy, Debug)]
pub struct ModulationTable {
    table: TensorView,
    width: u64,
    vectors: u64,
}

/// One step's modulation of a call's rows: the gathered table and the `[rows]` parameter row each
/// activation row reads.
#[derive(Clone, Copy, Debug)]
pub struct RowModulation<'a> {
    /// The step's gathered modulation rows.
    pub table: &'a ModulationTable,
    /// `[rows]`: the parameter row of each activation row.
    pub rows: TensorView,
}

/// The shift, scale and gate of one modulated stage.
#[derive(Clone, Copy, Debug)]
pub struct StageVectors {
    /// Added after the scale.
    pub shift: Weight,
    /// The normalized row is multiplied by `1 + scale`.
    pub scale: Weight,
    /// Weights the stage's update before it joins the residual stream.
    pub gate: Weight,
}

impl ModulationTable {
    /// A block's table: `[parameters, 6 · width]`.
    ///
    /// # Errors
    ///
    /// A table of another rank or width.
    pub fn block(table: TensorView, width: u64) -> Result<Self> {
        Self::new(table, width, BLOCK_VECTORS)
    }

    /// The final norm's table: `[parameters, 2 · width]`.
    ///
    /// # Errors
    ///
    /// A table of another rank or width.
    pub fn output(table: TensorView, width: u64) -> Result<Self> {
        Self::new(table, width, OUTPUT_VECTORS)
    }

    fn new(table: TensorView, width: u64, vectors: u64) -> Result<Self> {
        if table.shape.rank() != 2 || table.shape.last() != vectors * width {
            return Err(Error::shape(
                "ModulationTable",
                format!(
                    "a table of {:?} is not [parameters, {vectors} · {width}]",
                    table.shape.dims()
                ),
            ));
        }
        Ok(Self {
            table,
            width,
            vectors,
        })
    }

    /// Vector `index` of every parameter row, as a `[parameters, width]` row-strided weight.
    fn vector(&self, index: u64) -> Result<Weight> {
        let element = self.table.dtype.bytes().unwrap_or(1);
        let view = TensorView::with_strides(
            DevicePtr::new(self.table.ptr.address() + index * self.width * element),
            self.table.dtype,
            Shape::new(&[self.table.shape.dim(0), self.width]),
            Strides::new(&[self.table.strides.get(0), 1]),
        );
        Weight::borrowed(view)
    }

    /// The stage whose three vectors start at `first`, refusing a final norm's table.
    fn stage(&self, first: u64) -> Result<StageVectors> {
        if self.vectors != BLOCK_VECTORS {
            return Err(Error::shape(
                "ModulationTable",
                "a final norm's table holds no gated stage".to_owned(),
            ));
        }
        Ok(StageVectors {
            shift: self.vector(first)?,
            scale: self.vector(first + 1)?,
            gate: self.vector(first + 2)?,
        })
    }

    /// The attention's shift, scale and gate.
    ///
    /// # Errors
    ///
    /// A final norm's table.
    pub fn attention(&self) -> Result<StageVectors> {
        self.stage(0)
    }

    /// The MLP's shift, scale and gate.
    ///
    /// # Errors
    ///
    /// A final norm's table.
    pub fn mlp(&self) -> Result<StageVectors> {
        self.stage(3)
    }

    /// The final norm's shift and scale.
    ///
    /// # Errors
    ///
    /// A block's table.
    pub fn shift_scale(&self) -> Result<(Weight, Weight)> {
        if self.vectors != OUTPUT_VECTORS {
            return Err(Error::shape(
                "ModulationTable",
                "a block's table is read by stage, not as a final shift and scale".to_owned(),
            ));
        }
        Ok((self.vector(0)?, self.vector(1)?))
    }
}

/// The module names and geometry of an [`AdaLnBlock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaLnSpec<'n> {
    /// The normalized width.
    pub hidden: u32,
    /// The norms' variance floor.
    pub eps: f32,
    /// The attention input's norm module.
    pub norm1: &'n str,
    /// The MLP input's norm module.
    pub norm2: &'n str,
    /// The block's MLP.
    pub mlp: GatedMlpSpec<'n>,
}

/// One AdaLN diffusion block: two modulated RMS norms, the caller's attention and a gated MLP.
pub struct AdaLnBlock {
    norm1: Weight,
    norm2: Weight,
    eps: f32,
    mlp: GatedMlp,
}

impl AdaLnBlock {
    /// The block the module at `scope` states, its MLP at `split`'s rank.
    ///
    /// # Errors
    ///
    /// A norm gain or an MLP projection the loader refuses.
    pub fn load(scope: &mut Scope<'_, '_>, spec: AdaLnSpec<'_>, split: Split) -> Result<Self> {
        let width = u64::from(spec.hidden);
        Ok(Self {
            norm1: scope.scope(spec.norm1).tensor("weight", &[width])?,
            norm2: scope.scope(spec.norm2).tensor("weight", &[width])?,
            eps: spec.eps,
            mlp: GatedMlp::load(scope, spec.mlp, split)?,
        })
    }

    /// The element type of the block's norm gains, which is the stack's activation type.
    pub fn dtype(&self) -> DType {
        self.norm1.dtype()
    }

    /// The attention's input: `hidden` normalized by `norm1` and modulated by the attention's
    /// shift and scale at each row's parameter row; a fresh `[rows, width]` tensor of `hidden`'s
    /// element type.
    ///
    /// # Errors
    ///
    /// A table of another width, an arena that cannot hold the output, or the norm's refusal.
    pub fn attention_input(
        &self,
        f: &Forward,
        modulation: RowModulation<'_>,
        hidden: TensorView,
    ) -> Result<TensorView> {
        self.attention_norm(modulation, hidden.dtype)?
            .forward(f, hidden, None)
    }

    /// [`Self::attention_input`] into `out`, the caller's `[rows, width]` location: this member's
    /// block of a whole sequence a sequence-parallel attention gathers, so the gather of these rows
    /// starts as soon as they are normalized.
    ///
    /// # Errors
    ///
    /// As [`Self::attention_input`], or an `out` of another shape or element type.
    pub fn attention_input_into(
        &self,
        f: &Forward,
        modulation: RowModulation<'_>,
        hidden: TensorView,
        out: TensorView,
    ) -> Result<()> {
        self.attention_norm(modulation, hidden.dtype)?
            .forward_into(f, hidden, None, out)
    }

    /// The modulated pre-attention normalization of `modulation`'s rows, writing `dtype`.
    fn attention_norm(&self, modulation: RowModulation<'_>, dtype: DType) -> Result<RmsNorm> {
        let stage = modulation.table.attention()?;
        RmsNorm::new(
            self.norm1,
            self.eps,
            Epilogue::Modulated(Modulation {
                shift: stage.shift,
                scale: stage.scale,
                gate: None,
                rows: modulation.rows,
                retained: None,
            }),
            Gain::Plain,
            dtype,
        )
    }

    /// The block after its attention, over any rows of the block: the attention's gated `update`
    /// joins `hidden`, the MLP reads the sum normalized and modulated, and its gated output joins
    /// the sum. The block's output lands in `hidden`, in place.
    ///
    /// `update` is the attention's `[rows, width]` output for these rows and is overwritten with
    /// the residual sum. The retained sum and the MLP's input are reserved inside the call and
    /// handed back before return.
    ///
    /// # Errors
    ///
    /// A table of another width, views the norm or the MLP refuse, or an arena that cannot hold
    /// the intermediates.
    pub fn feed_forward(
        &self,
        f: &Forward,
        modulation: RowModulation<'_>,
        hidden: TensorView,
        update: TensorView,
    ) -> Result<()> {
        let attention = modulation.table.attention()?;
        let mlp = modulation.table.mlp()?;
        f.scoped(|f| {
            // One launch folds the attention's gated update into the stream, normalizes and
            // modulates the sum for the MLP, and retains the sum in `residual`: with an update
            // operand the operator stores the gated sum over it, and with a retained view it
            // copies the same value out.
            let residual = f.empty(hidden.dtype, hidden.shape)?;
            let norm = RmsNorm::new(
                self.norm2,
                self.eps,
                Epilogue::Modulated(Modulation {
                    shift: mlp.shift,
                    scale: mlp.scale,
                    gate: Some(attention.gate),
                    rows: modulation.rows,
                    retained: Some(residual),
                }),
                Gain::Plain,
                hidden.dtype,
            )?;
            let stream = ResidualStream {
                input: hidden,
                output: update,
            };
            let normalized = norm.forward(f, update, Some(stream))?;

            // `hidden` is spent once the sum is retained, so the MLP writes there and the gated
            // residual updates it in place.
            self.mlp.forward_into(f, normalized, hidden)?;
            Residual::new(Scale::Row(mlp.gate), None)?.forward(
                f,
                residual,
                hidden,
                Some(modulation.rows),
            )?;
            Ok(())
        })
    }

    /// The whole block over `hidden`, in place, with `attend(normalized)` the attention answering
    /// its update; see [`Self::feed_forward`]. The block's intermediates are handed back before
    /// return.
    ///
    /// # Errors
    ///
    /// As the two stages, or the attention's own error.
    pub fn forward(
        &self,
        f: &Forward,
        modulation: RowModulation<'_>,
        hidden: TensorView,
        attend: impl FnOnce(TensorView) -> Result<TensorView>,
    ) -> Result<()> {
        f.scoped(|f| {
            let normalized = self.attention_input(f, modulation, hidden)?;
            let update = attend(normalized)?;
            self.feed_forward(f, modulation, hidden, update)
        })
    }
}

/// The final modulated norm of a diffusion stack: `rms_norm(x) · (1 + scale[row]) + shift[row]`.
pub struct AdaLnOut {
    weight: Weight,
    eps: f32,
}

impl AdaLnOut {
    /// The norm module at `scope`: its `weight` of `width` channels and variance floor `eps`.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another width.
    pub fn load(scope: &mut Scope<'_, '_>, width: u32, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: scope.tensor("weight", &[u64::from(width)])?,
            eps,
        })
    }

    /// `hidden` `[rows, width]` normalized and modulated by the final table's shift and scale at
    /// each row's parameter row: a fresh tensor of `hidden`'s shape and element type.
    ///
    /// # Errors
    ///
    /// A block's table, a table of another width, an arena that cannot hold the output, or the
    /// norm's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        modulation: RowModulation<'_>,
        hidden: TensorView,
    ) -> Result<TensorView> {
        let (shift, scale) = modulation.table.shift_scale()?;
        RmsNorm::new(
            self.weight,
            self.eps,
            Epilogue::Modulated(Modulation {
                shift,
                scale,
                gate: None,
                rows: modulation.rows,
                retained: None,
            }),
            Gain::Plain,
            hidden.dtype,
        )?
        .forward(f, hidden, None)
    }
}
