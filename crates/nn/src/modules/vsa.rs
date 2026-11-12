//! Video sparse attention (VSA): pooled tile scores select the key tiles each query tile
//! attends, and a compressed branch stands for the tiles it does not.
//!
//! The sequence is cut into `tile_rows`-row tiles. A layer prepares the projected rows, selects
//! each query tile's key tiles from their pooled scores, attends the selected tiles, and
//! compresses each (head, query tile) over the pooled values of every live tile
//! ([`VsaAttention::attend`], the operators of [`crate::functional`] in order). The compressed
//! branch is folded into the attended rows by a gate the model projects: on one rank in place
//! ([`VsaAttention::add_gated_`]), and under sequence parallelism on the return path, interval
//! by interval ([`VsaAttention::compose`]). A compressed row is broadcast over every row of its
//! query tile, so the fold is not a residual add of one row per row.
//!
//! A sparse layout packs `[text | conditions | audio | generated video | padding]`: the tiles
//! before the generated video's are read by every query row, and each video query tile reads
//! the tiles its selection names. Both sets share one softmax per query row, so the fine
//! attention is one call over their union, which the selection's kept-tiles-first order already
//! is. The model states the layout as three per-tile tables it computes with tensor expressions
//! and moves to the device: each tile's valid rows, its segment and each segment's keep count.

use std::ops::Range;

use uniserve_tensor::{DType, Tensor};

use crate::error::{Error, Result};
use crate::functional;
use crate::var_builder::VarBuilder;

/// The widths, bounds and arithmetic of a [`VsaAttention`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VsaAttentionConfig {
    /// Rows of one tile: the checkpoint's block size, whole sixteen-row kernel blocks.
    pub tile_rows: u32,
    /// Heads per row: this rank's, under a sequence split.
    pub heads: u32,
    /// Channels per head.
    pub head_dim: u32,
    /// The modules of the per-head query and key gains, when the preparation normalizes Q and
    /// K.
    pub qk_norm: Option<(&'static str, &'static str)>,
    /// The variance floor of that normalization.
    pub eps: f32,
    /// Leading channels of each head the preparation rotates; zero rotates none. A rotation
    /// implies the normalization, as the reference applies them together.
    pub rotated: u32,
    /// The deployment's bound on the key tiles one selection ranks and one compression reduces.
    pub max_tiles: u32,
    /// The largest key extent the fine attention serves, in rows: whole tiles.
    pub max_keys: u32,
    /// The element type the attended rows are written in: the stack's.
    pub output: DType,
}

/// One layer's video sparse attention.
pub struct VsaAttention {
    /// The `[head_dim]` query and key gains, when the layer normalizes.
    gains: Option<(Tensor, Tensor)>,
    config: VsaAttentionConfig,
}

impl VsaAttention {
    /// The attention the module at `vb` states: the `weight` of its query and key gain modules,
    /// when the config names them.
    ///
    /// # Errors
    ///
    /// A tile that is not whole kernel blocks, a zero head, width, tile bound or key bound, a
    /// key bound that is not whole tiles, a rotation that is odd, wider than a head or without
    /// the normalization, or a gain the checkpoint lacks.
    pub fn load(config: &VsaAttentionConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        const OWNER: &str = "VsaAttention";

        if config.tile_rows == 0 || config.heads == 0 || config.head_dim == 0 {
            return Err(Error::shape(
                OWNER,
                "a tile, a head count and a feature count are each at least one".to_owned(),
            ));
        }
        if !config.tile_rows.is_multiple_of(ops::vsa::BLOCK_ROWS) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a tile of {} rows is not a whole number of {}-row kernel blocks",
                    config.tile_rows,
                    ops::vsa::BLOCK_ROWS
                ),
            ));
        }
        if config.rotated > config.head_dim || !config.rotated.is_multiple_of(2) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a rotation of {} features of {}",
                    config.rotated, config.head_dim
                ),
            ));
        }
        if config.rotated > 0 && config.qk_norm.is_none() {
            return Err(Error::shape(
                OWNER,
                "a rotation is applied with the normalization, as the reference does".to_owned(),
            ));
        }
        if config.max_tiles == 0 {
            return Err(Error::shape(
                OWNER,
                "a tile bound of zero leaves the compression nothing to reduce".to_owned(),
            ));
        }
        if config.max_keys == 0 || !config.max_keys.is_multiple_of(config.tile_rows) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a key bound of {} rows is not whole {}-row tiles",
                    config.max_keys, config.tile_rows
                ),
            ));
        }

        let width = config.head_dim as usize;
        let gains = match config.qk_norm {
            Some((query, key)) => Some((
                vb.pp(query).get(width, "weight")?,
                vb.pp(key).get(width, "weight")?,
            )),
            None => None,
        };

        Ok(Self {
            gains,
            config: *config,
        })
    }

    /// Rows of one tile.
    pub const fn tile_rows(&self) -> u32 {
        self.config.tile_rows
    }

    /// Heads per row.
    pub const fn heads(&self) -> u32 {
        self.config.heads
    }

    /// Features per head.
    pub const fn width(&self) -> u32 {
        self.config.head_dim
    }

    /// `rows`, `[rows, heads · head_dim]`, as the `[rows, heads, head_dim]` the operators read.
    fn headed(&self, rows: &Tensor) -> Result<Tensor> {
        let (heads, width) = (self.config.heads as usize, self.config.head_dim as usize);

        match rows.dims() {
            &[count, columns] if columns == heads * width => {
                Ok(rows.reshape((count, heads, width))?)
            }
            dims => Err(Error::shape(
                "VsaAttention",
                format!("{dims:?} are not [rows, {heads} · {width}]"),
            )),
        }
    }

    /// Prepares, scores, selects, attends and compresses the projections `query`, `key` and
    /// `value`, each `[tiles · tile_rows, heads · head_dim]`. Returns `(fine, compressed)`: the
    /// fine attention's rows, `[rows, heads · head_dim]`, and one compressed row per (head,
    /// query tile), `[heads, tiles, head_dim]`, both in the layer's output type.
    ///
    /// The projections are spent: the preparation normalizes, rotates and zeroes them in
    /// place. `factors` is the stack's `[rows, rotated / 2]` cosine and sine tables, present
    /// exactly when the layer rotates. The selection reads the layout's tables, each one entry
    /// per tile: `sizes` (`U32`, the tile's valid rows), `segments` (`I32`, its segment,
    /// negative outside every segment) and `keep` (`U32`, read by segment: the tiles of it a
    /// query keeps). Every query tile is the sequence tile of its index.
    ///
    /// # Errors
    ///
    /// Rows that are not whole tiles, a layout past the deployment's tile bound, factors that
    /// do not match the rotation, or tensors an operator refuses.
    pub fn attend(
        &self,
        (query, key, value): (Tensor, Tensor, Tensor),
        factors: Option<(&Tensor, &Tensor)>,
        (sizes, segments, keep): (&Tensor, &Tensor, &Tensor),
    ) -> Result<(Tensor, Tensor)> {
        const OWNER: &str = "VsaAttention";
        let config = &self.config;

        let rows = query.dims().first().copied().unwrap_or(0);
        let tile = config.tile_rows as usize;
        if !rows.is_multiple_of(tile) {
            return Err(Error::shape(
                OWNER,
                format!("{rows} rows are not whole {tile}-row tiles"),
            ));
        }
        // A layout past the deployment's bound has no compression compiled for it.
        if rows / tile > config.max_tiles as usize {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a request ranks {} key tiles, past the deployment's bound of {}",
                    rows / tile,
                    config.max_tiles
                ),
            ));
        }
        if factors.is_some() != (config.rotated > 0) {
            return Err(Error::shape(
                OWNER,
                "the factor tables are present exactly when the layer rotates".to_owned(),
            ));
        }

        let projections = [
            self.headed(&query)?,
            self.headed(&key)?,
            self.headed(&value)?,
        ];
        let ([query, key, value], pooled) = functional::vsa_prepare_tiles_(
            projections,
            self.gains.as_ref().map(|(query, key)| (query, key)),
            factors,
            sizes,
            config.tile_rows,
            config.rotated,
            config.eps,
        )?;

        // The preparation stored zeros past each tile's valid size in all three projections,
        // so the fine attention reads the same zeroed rows without a second pass.
        let scores = functional::vsa_scores(&pooled.get(0)?, &pooled.get(1)?)?;
        let selection = functional::vsa_select_tiles(&scores, sizes, segments, keep)?;
        let attended = functional::sparse_attention(
            (&query, &key, &value),
            (&selection.0, &selection.1),
            sizes,
            config.tile_rows,
            config.max_keys,
            config.output,
        )?;

        // The padding rows of a partial last tile are the packed sequence's: the same sizes
        // that bound the key tiles bound the query tiles.
        let fine = functional::vsa_zero_padding_(attended, sizes, config.tile_rows)?;

        // One compression per (head, query tile): a softmax over every live key tile.
        let compressed = functional::vsa_compress_pooled(
            &scores,
            &pooled.get(2)?,
            sizes,
            config.max_tiles,
            config.output,
        )?;

        Ok((fine.flatten_from(1)?, compressed))
    }

    /// Folds the compressed branch's gated rows into each tile's rows of `fine`, in place, and
    /// returns it: `fine += compressed[:, tile] · gate`.
    ///
    /// `fine` and `gate` are `[rows, heads · head_dim]` and `compressed` the
    /// `[heads, tiles, head_dim]` branch of [`Self::attend`].
    ///
    /// # Errors
    ///
    /// Rows of another width, or the operator's refusal.
    pub fn add_gated_(&self, fine: Tensor, compressed: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let folded = functional::vsa_add_gated_tiles_(
            self.headed(&fine)?,
            compressed,
            &self.headed(gate)?,
            self.config.tile_rows,
        )?;

        Ok(folded.flatten_from(1)?)
    }

    /// One row interval of every member's gated rows, member by member: the send buffer of a
    /// sequence-parallel return ([`functional::vsa_compose_gated_rows`]).
    ///
    /// `fine` and `gate` hold every member's rows of this member's heads,
    /// `[members · owner rows, heads · head_dim]`; `rows` is the interval of each owner's rows
    /// the result carries, and `slots` the block each member's rows land in, in member order.
    /// The result is `[members, rows.len(), heads, head_dim]`.
    ///
    /// # Errors
    ///
    /// No member or more than the operator covers, rows of another width, or the operator's
    /// refusal.
    pub fn compose(
        &self,
        fine: &Tensor,
        compressed: &Tensor,
        gate: &Tensor,
        rows: Range<usize>,
        slots: &[u32],
    ) -> Result<Tensor> {
        functional::vsa_compose_gated_rows(
            &self.headed(fine)?,
            compressed,
            &self.headed(gate)?,
            self.config.tile_rows,
            rows,
            slots,
        )
    }
}
