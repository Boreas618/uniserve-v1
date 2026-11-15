//! The adaptive-layer-norm (AdaLN) block of a diffusion transformer, and its final modulated
//! norm.
//!
//! A diffusion block is conditioned on the step's noise levels through per-row affine vectors:
//! the attention's input and the MLP's input are each normalized and then shifted and scaled,
//! and each sub-layer's output joins the residual stream through a gate. One step's vectors are
//! the rows of a table, `[parameters, vectors · width]`, and every activation row selects its
//! own parameter row (`rows`), so one launch conditions rows of several timesteps at once:
//!
//! ```text
//! normalized = rms_norm(hidden) · (1 + scale_a[row]) + shift_a[row]            attention_input
//! update     = attention(normalized)                                           the caller's
//! residual   = hidden + gate_a[row] · update                                   ┐ one launch
//! normalized = rms_norm(residual) · (1 + scale_m[row]) + shift_m[row]          ┘
//! hidden     = residual + gate_m[row] · mlp(normalized)                        feed_forward
//! ```
//!
//! The vectors of one parameter row are stored in the order the reference projections emit
//! them: shift, scale and gate of the attention, then shift, scale and gate of the MLP, so
//! vector `k` of every row is the column window `table.narrow(1, k · width, width)`, a strided
//! view. [`AdaLnOut`] is the stack's last norm, whose table holds a shift and a scale.
//!
//! The attention is the caller's: a block's attention may be dense or sparse, local or exchanged
//! over a sequence-parallel group, so the block exposes its two modulated stages beside the
//! whole forward.

use tensor::{DType, Tensor};

use crate::error::{Error, Result};
use crate::functional;
use crate::modules::mlp::{Mlp, MlpConfig};
use crate::var_builder::VarBuilder;

/// Vector `index` of every parameter row of `table`, `[parameters, vectors · width]`: a
/// `[parameters, width]` view whose rows stride over the whole table row.
fn vector(
    owner: &'static str,
    table: &Tensor,
    vectors: usize,
    width: usize,
    index: usize,
) -> Result<Tensor> {
    if table.rank() != 2 || table.dims()[1] != vectors * width {
        return Err(Error::shape(
            owner,
            format!(
                "a table of {:?} is not [parameters, {vectors} · {width}]",
                table.dims()
            ),
        ));
    }
    Ok(table.narrow(1, index * width, width)?)
}

/// The module names and widths of an [`AdaLnBlock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaLnBlockConfig {
    /// The normalized width.
    pub hidden: u32,
    /// The norms' variance floor.
    pub eps: f32,
    /// The attention input's norm module.
    pub norm1: &'static str,
    /// The MLP input's norm module.
    pub norm2: &'static str,
    /// The block's MLP, its module names relative to the block.
    pub mlp: MlpConfig,
}

/// One AdaLN diffusion block: two modulated RMS norms, the caller's attention and a gated MLP.
pub struct AdaLnBlock {
    norm1: Tensor,
    norm2: Tensor,
    eps: f32,
    width: usize,
    mlp: Mlp,
}

impl AdaLnBlock {
    /// The affine vectors one parameter row of the block's table holds, in storage order:
    /// shift, scale and gate of the attention, then of the MLP. A row is `VECTORS · width`
    /// columns wide.
    pub const VECTORS: u32 = 6;

    /// The block the module at `vb` states, its MLP at the rank its config's split states.
    ///
    /// # Errors
    ///
    /// A norm gain or an MLP projection the loader refuses.
    pub fn load(config: &AdaLnBlockConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let width = config.hidden as usize;

        Ok(Self {
            norm1: vb.pp(config.norm1).get(width, "weight")?,
            norm2: vb.pp(config.norm2).get(width, "weight")?,
            eps: config.eps,
            width,
            mlp: Mlp::load(&config.mlp, vb)?,
        })
    }

    /// The element type of the block's norm gains, which is the stack's activation type.
    pub fn dtype(&self) -> DType {
        self.norm1.dtype()
    }

    /// The attention's input: `hidden`, `[rows, width]`, normalized by `norm1` and modulated by
    /// the attention's shift and scale at each row's parameter row, in `hidden`'s element type.
    ///
    /// `table` is the step's `[parameters, VECTORS · width]` modulation ([`Self::VECTORS`]) and
    /// `rows` the `[rows]` `U32` parameter row of each activation row.
    ///
    /// # Errors
    ///
    /// A table of another width, or the norm's refusal.
    pub fn attention_input(
        &self,
        table: &Tensor,
        rows: &Tensor,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let stage = |index| {
            vector(
                "AdaLnBlock",
                table,
                Self::VECTORS as usize,
                self.width,
                index,
            )
        };

        functional::modulated_rms_norm(
            hidden,
            &self.norm1,
            &stage(0)?,
            &stage(1)?,
            rows,
            self.eps,
            hidden.dtype(),
        )
    }

    /// The block after its attention, over any rows of the block: the attention's gated
    /// `update` joins `hidden`, the MLP reads the sum normalized and modulated, and its gated
    /// output joins the sum, which is the block's output.
    ///
    /// `update` is the attention's `[rows, width]` output for these rows and is spent: the
    /// modulated norm stores the residual sum over it.
    ///
    /// # Errors
    ///
    /// A table of another width, or rows the norm or the MLP refuse.
    pub fn feed_forward(
        &self,
        table: &Tensor,
        rows: &Tensor,
        hidden: &Tensor,
        update: Tensor,
    ) -> Result<Tensor> {
        let stage = |index| {
            vector(
                "AdaLnBlock",
                table,
                Self::VECTORS as usize,
                self.width,
                index,
            )
        };

        // One launch folds the attention's gated update into the stream and normalizes and
        // modulates the sum for the MLP.
        let (normalized, residual) = functional::modulated_rms_norm_(
            update,
            hidden,
            &stage(2)?,
            &self.norm2,
            &stage(3)?,
            &stage(4)?,
            rows,
            self.eps,
            hidden.dtype(),
        )?;

        // The MLP's output joins the stream through its row's gate: the gated update rounds,
        // then the sum rounds.
        let projected = self.mlp.forward(&normalized)?;
        // The gate's column window is packed before its rows are gathered: a parameter table
        // holds a few rows, and the CPU gather reads packed rows.
        let gate = stage(5)?.contiguous()?.index_select(rows, 0)?;

        Ok((residual + (gate * projected)?)?)
    }

    /// The whole block over `hidden`, with `attend(normalized)` the attention returning its
    /// update; see [`Self::feed_forward`].
    ///
    /// # Errors
    ///
    /// As the two stages, or the attention's own error.
    pub fn forward(
        &self,
        table: &Tensor,
        rows: &Tensor,
        hidden: &Tensor,
        attend: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let normalized = self.attention_input(table, rows, hidden)?;
        let update = attend(&normalized)?;

        self.feed_forward(table, rows, hidden, update)
    }
}

/// What a checkpoint states of an [`AdaLnOut`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaLnOutConfig {
    /// The normalized width.
    pub hidden: u32,
    /// The norm's variance floor.
    pub eps: f32,
}

/// The final modulated norm of a diffusion stack: `rms_norm(x) · (1 + scale[row]) + shift[row]`.
pub struct AdaLnOut {
    weight: Tensor,
    eps: f32,
    width: usize,
}

impl AdaLnOut {
    /// The vectors one parameter row of the norm's table holds: shift, then scale. A row is
    /// `VECTORS · width` columns wide.
    pub const VECTORS: u32 = 2;

    /// The norm module at `vb`: its `weight` of `config.hidden` channels.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks the weight or holds another width.
    pub fn load(config: &AdaLnOutConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let width = config.hidden as usize;

        Ok(Self {
            weight: vb.get(width, "weight")?,
            eps: config.eps,
            width,
        })
    }

    /// `hidden`, `[rows, width]`, normalized and modulated by the shift and scale of `table`,
    /// `[parameters, VECTORS · width]` ([`Self::VECTORS`]), at each row's `U32` parameter row: a
    /// tensor of `hidden`'s shape and element type.
    ///
    /// # Errors
    ///
    /// A table of another width, or the norm's refusal.
    pub fn forward(&self, table: &Tensor, rows: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let stage = |index| vector("AdaLnOut", table, Self::VECTORS as usize, self.width, index);

        functional::modulated_rms_norm(
            hidden,
            &self.weight,
            &stage(0)?,
            &stage(1)?,
            rows,
            self.eps,
            hidden.dtype(),
        )
    }
}
