//! Sparse latent attention: the compressor of a layer's tokens into latents, the indexer that
//! selects the compressed latents each query reads, and the attention over a window and that
//! selection with a per-head sink.
//!
//! One shared latent of `head_dim` channels is both the key and the value of every entry. A
//! layer keeps three cache fields of it, each a paged plane its forward borrows
//! ([`functional::Plane`]): the **window** (one entry per token, the last `window` read by each
//! query), the **compressed latents** (one entry per `ratio` tokens, [`Compressor`]) and the
//! **index keys** (one per compressed entry, [`Indexer`]). Owner layers write the compressed
//! latents and the index keys; the layers after an owner read its planes. A layer's forward is,
//! in order:
//!
//! ```text
//! compressed = compressor.forward(x)            (owners: the pre-rotation latents of the groups)
//! selection  = indexer.forward(x, qr, compressed)    (index sources: writes keys, scores, selects)
//! compressor.store_(compressed)                 (owners: rotated, encoded compressed entries)
//! o          = attention.forward(q, kv, selection)   (writes the window, attends, rotates back)
//! ```
//!
//! The index keys are made from the latent before its rotation, which is why the indexer runs
//! between the compression and the store. Every rotation turns the trailing `rotated` channels
//! in adjacent pairs: a compressed entry and its index key at the position of the group's first
//! token, a query and the window entry at its own position. The modules hold their own
//! parameters (gains, sinks, the compressor's and indexer's dense projections) and take the
//! projections of block-scaled weights as already bound [`Linear`] layers from their model.

use tensor::{DType, Device, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{
    self, Compression, Format, IndexScoring, Plane, RecordRow, Ring, Selected, Window,
};
use crate::modules::{Linear, LinearConfig, Parallel};
use crate::var_builder::VarBuilder;

/// The batch columns a latent attention layer reads, as its forward borrows them.
#[derive(Clone, Copy)]
pub struct LatentBatch<'t> {
    /// `[tokens]` I32: each token's position.
    pub positions: &'t Tensor,
    /// `[tokens]` I32: the window entry each token writes, its position, or `-1` for a padding
    /// token, which writes no cache field.
    pub written: &'t Tensor,
    /// `[tokens]` U32: each token's row of the cache fields' tables.
    pub rows: &'t Tensor,
    /// `[rows]` U32: the first position whose window entry each row holds, the position a
    /// request resumed from a prefix hit recomputes its window from; zero for one that
    /// computed every token.
    pub window_starts: &'t Tensor,
    /// `[tokens, rotated / 2]` F32 each: the rotary factors of the tokens' positions.
    pub cos: &'t Tensor,
    /// See `cos`.
    pub sin: &'t Tensor,
    /// `[rotated / 2]` F32: the inverse frequencies the factors were computed from, which a
    /// cache write turns its entries by at their own positions.
    pub inv_freq: &'t Tensor,
}

/// `value` as a one-element tensor of `like`'s element type and device, for a broadcast.
fn scalar(value: i64, like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(&[value], &Device::Cpu)?
        .to_dtype(like.dtype())?
        .to_device(like.device())?)
}

// ---- Compressor ---------------------------------------------------------------------------

/// What a checkpoint states of a [`Compressor`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CompressorConfig {
    /// The width of the hidden rows it reads.
    pub hidden: u32,
    /// The width of a latent.
    pub width: u32,
    /// Tokens one latent stands for: one, or at least two for a gated pooling.
    pub ratio: u32,
    /// The trailing channels of a latent its store rotates.
    pub rotated: u32,
    /// The normalization's variance floor.
    pub eps: f32,
}

/// The latents of the groups a batch completes, as a [`Compressor`] answers them.
pub struct Compressed {
    /// `[tokens, width]` BF16: row `t` the normalized latent of the group token `t` completes,
    /// before its rotation; the row of a token that completes none is unspecified.
    pub latent: Tensor,
    /// `[tokens]` I32: the compressed entry token `t` completes, or `-1`.
    pub entries: Tensor,
}

/// The compression of a layer's tokens into one normalized latent per `ratio` tokens.
///
/// At ratio one a token's latent is `rms_norm(wkv · x)`. Above it, the module projects each
/// token into a value and a gate in FP32 (`wkv` and `wgate`, read as one `[2 · width, hidden]`
/// FP32 projection, the reference's promotion of both), and the token that completes a group
/// pools the group by a per-channel softmax of the gates ([`functional::latent_compress`]); the
/// projections of a request's last tokens wait in its ring for the step that completes their
/// group.
pub struct Compressor {
    config: CompressorConfig,
    projection: Linear,
    norm: Tensor,
}

impl Compressor {
    /// The compressor at `vb`: `wkv` (and `wgate` above ratio one) and `norm.weight`.
    ///
    /// # Errors
    ///
    /// A ratio of zero, or a weight the checkpoint lacks or holds in another extent.
    pub fn load(config: &CompressorConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.ratio == 0 || config.rotated > config.width {
            return Err(Error::shape(
                "Compressor",
                format!(
                    "a ratio of {} and {} rotated channels of {}",
                    config.ratio, config.rotated, config.width
                ),
            ));
        }
        let projection = if config.ratio == 1 {
            Linear::load(
                &LinearConfig {
                    inputs: config.hidden,
                    outputs: config.width,
                    bias: false,
                    parallel: Parallel::None,
                },
                vb.pp("wkv"),
            )?
        } else {
            Linear::load_merged(
                vb.to_dtype(DType::F32),
                &["wkv", "wgate"],
                config.hidden,
                &[config.width, config.width],
                false,
            )?
        };

        Ok(Self {
            config: *config,
            projection,
            norm: vb.pp("norm").get(config.width as usize, "weight")?,
        })
    }

    /// The configuration the module was loaded with.
    pub const fn config(&self) -> &CompressorConfig {
        &self.config
    }

    /// The latents of the groups the batch completes.
    ///
    /// `x` is `[tokens, hidden]` BF16. Above ratio one `ring` is the request's ring, which the
    /// forward reads and then updates with the projections of each state row's last tokens; at
    /// ratio one every token completes its own group, entry its position, and no ring is read.
    ///
    /// # Errors
    ///
    /// A ring missing above ratio one, operands of other shapes, or an operator's refusal.
    pub fn forward(
        &self,
        x: &Tensor,
        batch: &LatentBatch<'_>,
        ring: Option<Ring<'_>>,
    ) -> Result<Compressed> {
        let config = &self.config;
        if config.ratio == 1 {
            let latent = functional::rms_norm(
                &self.projection.forward(x)?,
                &self.norm,
                config.eps,
                false,
                DType::BF16,
            )?;
            return Ok(Compressed {
                latent,
                entries: batch.written.clone(),
            });
        }
        let Some(ring) = ring else {
            return Err(Error::shape(
                "Compressor",
                format!("a compression of {} tokens reads a ring", config.ratio),
            ));
        };

        let projected = self.projection.forward(&x.to_dtype(DType::F32)?)?;
        let Compression {
            latent,
            entries,
            ring_rows,
        } = functional::latent_compress(
            &projected,
            (&self.norm, config.eps),
            ring,
            batch.positions,
            config.ratio,
        )?;

        // The ring takes the state rows' last projections once the compression has read it.
        let (slots, capacity, doubled) = (
            ring.rows.dims()[0],
            ring.rows.dims()[1],
            ring.rows.dims()[2],
        );
        functional::place_rows_(
            &ring.rows.reshape((slots * capacity, doubled))?,
            &ring_rows,
            &projected,
        )?;
        Ok(Compressed { latent, entries })
    }

    /// Writes the batch's completed latents into the compressed plane: each rotated at its
    /// group's first position and encoded in the plane's format, at its entry.
    ///
    /// # Errors
    ///
    /// A plane of another width, or the operator's refusal.
    pub fn store_(
        &self,
        compression: &Compressed,
        plane: Plane<'_>,
        batch: &LatentBatch<'_>,
    ) -> Result<()> {
        functional::record_write_(
            &compression.latent,
            RecordRow {
                norm: None,
                inv_freq: (self.config.rotated > 0).then_some(batch.inv_freq),
                ratio: self.config.ratio,
            },
            plane,
            &compression.entries,
            batch.rows,
        )
    }
}

// ---- Indexer ------------------------------------------------------------------------------

/// The candidate blocks an indexer takes part in: a narrowing of the scoring of later indexers
/// to the blocks one earlier indexer ranks best.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Candidates {
    /// The indexer scores every visible entry and narrows no other.
    None,
    /// The indexer also ranks blocks of `block` entries by their best score, its newest block
    /// first, and keeps `blocks` of them for the indexers after it.
    Source {
        /// Blocks kept.
        blocks: u32,
        /// Entries of one block.
        block: u32,
    },
    /// The indexer scores only the entries of a source's candidate blocks of `block` entries.
    Restricted {
        /// Entries of one block.
        block: u32,
    },
}

/// What a checkpoint states of an [`Indexer`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IndexerConfig {
    /// The width of the hidden rows its head weights are projected from.
    pub hidden: u32,
    /// Index heads of the whole layer.
    pub heads: u32,
    /// Elements of an index head and of an index key.
    pub width: u32,
    /// The width of a latent, which an index key is projected from.
    pub latent: u32,
    /// The trailing channels of a query and a key that rotate.
    pub rotated: u32,
    /// The compressed entries each query keeps at most.
    pub top: u32,
    /// Tokens one compressed entry stands for.
    pub ratio: u32,
    /// The key normalization's variance floor.
    pub eps: f32,
    /// Whether the indexer writes the index keys of its layer's compressed entries.
    pub owns_keys: bool,
    /// The candidate blocks it takes part in.
    pub candidates: Candidates,
    /// How the layer's index heads are split across its tensor-parallel group.
    pub split: Split,
}

/// What an [`Indexer`] answers for a batch.
pub struct Selection {
    /// `[tokens, top]` I32: the compressed entries each query attends, `-1` for none.
    pub entries: Tensor,
    /// `[tokens, blocks]` I32: the candidate blocks of a source indexer, for the indexers after
    /// it; `None` for any other.
    pub candidates: Option<Tensor>,
}

/// The indexer of a sparse latent attention layer: which compressed entries each query reads.
///
/// ```text
/// q    = rotate(wq_b · qr)                     [tokens, heads, width], rounded to MXFP4 as read
/// k    = rotate(k_norm(wk · latent))           written into the index-key plane by an owner
/// w    = weights_proj · x · width^-1/2 · heads^-1/2
/// s    = Σ_h relu(q_h · k) · w_h               over the (p + 1) / ratio visible entries
/// sel  = top min(top, columns) of s, -1 for an invisible entry
/// ```
///
/// With a tensor-parallel split each rank scores its heads and the group sums the scores
/// before the selection, as the reference reduces them.
pub struct Indexer {
    config: IndexerConfig,
    query: Linear,
    weights: Linear,
    key: Option<(Linear, Tensor)>,
}

impl Indexer {
    /// The indexer at `vb`: `weights_proj` and, for an owner, `wk` and `k_norm.weight`; `query`
    /// is the bound index query projection `wq_b` (`[heads · width / split, q_lora]`, this rank's
    /// heads), whose block-scaled weight its model loads.
    ///
    /// # Errors
    ///
    /// A split that does not divide the heads, or a weight the checkpoint lacks or holds in
    /// another extent.
    pub fn load(config: &IndexerConfig, vb: VarBuilder<'_, '_>, query: Linear) -> Result<Self> {
        if config.heads == 0
            || !config.heads.is_multiple_of(config.split.size())
            || config.rotated > config.width
            || config.ratio == 0
        {
            return Err(Error::shape(
                "Indexer",
                format!(
                    "{} index heads of {} over {} ranks, {} rotated, ratio {}",
                    config.heads,
                    config.width,
                    config.split.size(),
                    config.rotated,
                    config.ratio
                ),
            ));
        }
        let weights = Linear::load(
            &LinearConfig {
                inputs: config.hidden,
                outputs: config.heads,
                bias: false,
                parallel: if config.split.size() > 1 {
                    Parallel::Column(config.split)
                } else {
                    Parallel::None
                },
            },
            vb.pp("weights_proj"),
        )?;
        let key = if config.owns_keys {
            Some((
                Linear::load(
                    &LinearConfig {
                        inputs: config.latent,
                        outputs: config.width,
                        bias: false,
                        parallel: Parallel::None,
                    },
                    vb.pp("wk"),
                )?,
                vb.pp("k_norm").get(config.width as usize, "weight")?,
            ))
        } else {
            None
        };

        Ok(Self {
            config: *config,
            query,
            weights,
            key,
        })
    }

    /// The configuration the module was loaded with.
    pub const fn config(&self) -> &IndexerConfig {
        &self.config
    }

    /// The selection of each query, writing the index keys of the batch's completed entries
    /// first when the indexer owns them.
    ///
    /// `x` is the layer's `[tokens, hidden]` input, `qr` the normalized `[tokens, q_lora]` query
    /// latent, `compression` the layer's compression when it owns its keys, `keys` the index-key
    /// plane (packed records keep their scales per page), and `candidates` a source's blocks when
    /// the indexer is restricted to them.
    ///
    /// # Errors
    ///
    /// An owner without its compression, a restricted indexer without candidates, operands of
    /// other shapes, or an operator's refusal.
    pub fn forward(
        &self,
        (x, qr): (&Tensor, &Tensor),
        compression: Option<&Compressed>,
        keys: Plane<'_>,
        batch: &LatentBatch<'_>,
        candidates: Option<&Tensor>,
    ) -> Result<Selection> {
        let config = &self.config;
        let tokens = x.dims()[0];
        let heads = (config.heads / config.split.size()) as usize;
        let width = config.width as usize;

        // An owner writes the keys of the groups the batch completes before it scores them.
        if let Some((wk, k_norm)) = &self.key {
            let Some(compression) = compression else {
                return Err(Error::shape(
                    "Indexer",
                    "an indexer that owns its keys reads its layer's compression".to_owned(),
                ));
            };
            functional::record_write_(
                &wk.forward(&compression.latent)?,
                RecordRow {
                    norm: Some((k_norm, config.eps)),
                    inv_freq: (config.rotated > 0).then_some(batch.inv_freq),
                    ratio: config.ratio,
                },
                keys,
                &compression.entries,
                batch.rows,
            )?;
        }

        // The index queries, rotated at their own positions, and the heads' weights in the
        // reference's rounding: the projection and its scaling both in BF16.
        let query = self
            .query
            .forward(qr)?
            .reshape((tokens, heads, width))?
            .contiguous()?;
        if config.rotated > 0 {
            functional::rotary_trailing_(&query, batch.cos, batch.sin, false)?;
        }
        let scale = (f64::from(config.width) * f64::from(config.heads)).powf(-0.5);
        let weights = self
            .weights
            .forward(x)?
            .affine(scale, 0.0)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // `(p + 1) / ratio` entries are complete, and visible, at position `p`.
        let visible = batch
            .positions
            .broadcast_add(&scalar(1, batch.positions)?)?
            .broadcast_div(&scalar(i64::from(config.ratio), batch.positions)?)?
            .to_dtype(DType::U32)?;

        let restriction = match (config.candidates, candidates) {
            (Candidates::Restricted { block }, Some(candidates)) => Some((candidates, block)),
            (Candidates::Restricted { .. }, None) => {
                return Err(Error::shape(
                    "Indexer",
                    "a restricted indexer scores inside its source's candidates".to_owned(),
                ));
            }
            _ => None,
        };

        // A query's scores, its selection and its candidate blocks read its own row alone, so
        // the rows are scored and selected a block at a time, each block's scores within
        // `SCORE_BYTES`: a prefill over a table of a million entries would otherwise hold its
        // whole `[tokens, entries]` scores.
        let columns = match restriction {
            Some((candidates, block)) => candidates.dims()[1] * block as usize,
            None => keys.table.dims()[1] * keys.records.dims()[1],
        };
        let kept = (config.top as usize).min(columns);
        // The scores are the reference's own BF16 values (`IndexScoring::scores`), two bytes
        // each, so the selection ranks them, near ties included, as the reference does. The
        // heads of a split are summed across its ranks, so each rank's share of a score stays
        // FP32 until the sum.
        let score = if config.split.size() > 1 {
            DType::F32
        } else {
            SCORE
        };
        let score_bytes = score.size_in_bytes().unwrap_or(4);
        let rows_per_block = (SCORE_BYTES / (columns.max(1) * score_bytes)).max(1);
        let mut selected = Vec::new();
        let mut ranked = Vec::new();
        for first in (0..tokens).step_by(rows_per_block) {
            let rows = rows_per_block.min(tokens - first);
            let (query, weights, visible) = (
                query.narrow(0, first, rows)?,
                weights.narrow(0, first, rows)?,
                visible.narrow(0, first, rows)?,
            );
            let restricted = restriction
                .map(|(candidates, block)| {
                    Ok::<_, Error>((candidates.narrow(0, first, rows)?, block))
                })
                .transpose()?;
            let restricted = restricted
                .as_ref()
                .map(|(candidates, block)| (candidates, *block));

            let scores = functional::index_scores(
                &query,
                keys,
                &batch.rows.narrow(0, first, rows)?,
                &visible,
                IndexScoring {
                    weights: Some(&weights),
                    query: Format::Mxfp4,
                    candidates: restricted,
                    scores: score,
                },
            )?;
            if config.split.size() > 1 {
                collective::all_reduce(&scores)?;
            }

            selected.push(functional::index_select(
                &scores, &visible, restricted, 1, kept,
            )?);
            if let Candidates::Source { blocks, block } = config.candidates {
                let blocks_of_row = columns.div_ceil(block as usize);
                ranked.push(functional::index_select(
                    &scores,
                    &visible,
                    None,
                    block,
                    (blocks as usize).min(blocks_of_row),
                )?);
            }
        }

        let mut entries = joined(selected)?;
        if kept < config.top as usize {
            // A table narrower than the selection leaves the columns past it empty.
            let empty = Tensor::full(-1_i32, (tokens, config.top as usize - kept), &Device::Cpu)?
                .to_device(x.device())?;
            entries = Tensor::cat(&[&entries, &empty], 1)?.contiguous()?;
        }
        let candidates = match config.candidates {
            Candidates::Source { .. } => Some(joined(ranked)?),
            _ => None,
        };
        Ok(Selection {
            entries,
            candidates,
        })
    }
}

/// The scores one block of an [`Indexer`]'s query rows holds at most, in bytes: 512 query rows
/// over a million entries, the whole of a decode batch's rows.
const SCORE_BYTES: usize = 1 << 30;

/// The type an [`Indexer`] scores in: DeepSeek's indexer's BF16 arithmetic, whose order its BF16
/// `topk` ranks.
const SCORE: DType = DType::BF16;

/// The rows of `blocks` in order, one block taken as it is.
fn joined(mut blocks: Vec<Tensor>) -> Result<Tensor> {
    if blocks.len() == 1 {
        return Ok(blocks.remove(0));
    }
    let parts: Vec<&Tensor> = blocks.iter().collect();
    Ok(Tensor::cat(&parts, 0)?.contiguous()?)
}

// ---- LatentAttention ----------------------------------------------------------------------

/// What a checkpoint states of a [`LatentAttention`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LatentAttentionConfig {
    /// Query heads of the whole layer.
    pub heads: u32,
    /// Elements of a head and of a latent.
    pub head_dim: u32,
    /// The trailing channels of a head and of a latent that rotate.
    pub rotated: u32,
    /// Window entries a query attends: its own position and the ones before it.
    pub window: u32,
    /// The window latent's normalization variance floor.
    pub eps: f32,
    /// The factor the scores are multiplied by.
    pub scale: f32,
    /// How the layer's query heads are split across its tensor-parallel group.
    pub split: Split,
}

/// Attention of each query over its window and its selected compressed entries, one latent as
/// key and value, with a per-head sink ([`functional::latent_attention`]).
///
/// The forward writes the batch's window entries (`kv_norm` of the layer's latent projection,
/// rotated at each token's position and encoded in the window's format), rotates the queries'
/// trailing channels, attends, and rotates the attended values' trailing channels back, so
/// every cached entry keeps one rotated form.
pub struct LatentAttention {
    config: LatentAttentionConfig,
    sink: Tensor,
    kv_norm: Tensor,
}

impl LatentAttention {
    /// The attention at `vb`: `attn_sink` (this rank's heads) and `kv_norm.weight`.
    ///
    /// # Errors
    ///
    /// A split that does not divide the heads, or a weight the checkpoint lacks or holds in
    /// another extent.
    pub fn load(config: &LatentAttentionConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let ranks = config.split.size();
        if config.heads == 0
            || !config.heads.is_multiple_of(ranks)
            || config.rotated > config.head_dim
            || config.window == 0
        {
            return Err(Error::shape(
                "LatentAttention",
                format!(
                    "{} heads of {} over {ranks} ranks, {} rotated, a window of {}",
                    config.heads, config.head_dim, config.rotated, config.window
                ),
            ));
        }
        let local = (config.heads / ranks) as usize;
        let sink = vb.to_dtype(DType::F32).get_rows(
            local,
            "attn_sink",
            local * config.split.rank() as usize,
        )?;

        Ok(Self {
            config: *config,
            sink,
            kv_norm: vb.pp("kv_norm").get(config.head_dim as usize, "weight")?,
        })
    }

    /// The configuration the module was loaded with.
    pub const fn config(&self) -> &LatentAttentionConfig {
        &self.config
    }

    /// The attended values of the batch's queries, `[tokens, heads, head_dim]` BF16 with their
    /// trailing channels rotated back.
    ///
    /// `query` is the `[tokens, heads, head_dim]` BF16 queries before their rotation, which the
    /// forward rotates in place; `kv` the `[tokens, head_dim]` latent projection before its
    /// normalization, written into `window` at each token's position; `compressed` the
    /// compressed plane and the queries' selection, for a layer that reads compressed entries;
    /// `workspace` the `[bytes]` U8 scratch the attention's programs are lent, which execution
    /// reserves with the cache group the window comes from.
    ///
    /// # Errors
    ///
    /// Operands of other shapes, or an operator's refusal.
    pub fn forward(
        &self,
        (query, kv): (&Tensor, &Tensor),
        window: Plane<'_>,
        compressed: Option<Selected<'_>>,
        batch: &LatentBatch<'_>,
        workspace: &Tensor,
    ) -> Result<Tensor> {
        let config = &self.config;
        let rotated = config.rotated > 0;

        // The window entries of the batch's tokens, before any query reads them.
        self.write_(kv, window, batch)?;

        if rotated {
            functional::rotary_trailing_(query, batch.cos, batch.sin, false)?;
        }
        let attended = functional::latent_attention(
            query,
            Window {
                plane: window,
                starts: batch.window_starts,
                size: config.window,
            },
            compressed,
            (batch.rows, batch.positions),
            &self.sink,
            config.scale,
            workspace,
        )?;
        if rotated {
            functional::rotary_trailing_(&attended, batch.cos, batch.sin, true)?;
        }
        Ok(attended)
    }

    /// Writes the window entries of `kv`, `[tokens, head_dim]` latent projections before their
    /// normalization, into `window`: each token's `kv_norm` of its row, rotated at its position
    /// and encoded in the window's format, at the entry `batch.written` names (none for `-1`).
    /// What [`Self::forward`] writes before its queries read, for a layer whose window also
    /// holds entries of rows it attends none of (a drafter's context).
    ///
    /// # Errors
    ///
    /// Operands of other shapes, or the operator's refusal.
    pub fn write_(&self, kv: &Tensor, window: Plane<'_>, batch: &LatentBatch<'_>) -> Result<()> {
        let config = &self.config;

        functional::record_write_(
            kv,
            RecordRow {
                norm: Some((&self.kv_norm, config.eps)),
                inv_freq: (config.rotated > 0).then_some(batch.inv_freq),
                ratio: 1,
            },
            window,
            batch.written,
            batch.rows,
        )
    }
}
