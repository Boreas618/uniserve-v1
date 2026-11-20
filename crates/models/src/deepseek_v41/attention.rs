//! A decoder layer's sparse latent attention (`inference/model.py:614-791`): the query and
//! latent projections, the compressor of a source layer, the indexer of an index source, the
//! attention over the window and the selected compressed entries, and the grouped output
//! projection.
//!
//! ```text
//! qr, kv  = wq_a ‖ wkv (x)                       one MXFP8 product, [T, q_lora + head_dim]
//! qr      = q_norm(qr)
//! q       = wq_b(qr)                             [T, heads, head_dim], rotated by the attention
//! c       = compressor(x)                        a source: the groups the batch completes
//! sel     = indexer(x, qr, c)                    an index source; any other reads the last one's
//! store(c)                                       a source: rotated, encoded compressed entries
//! o       = attention(q, kv, window ∪ sel)       the window written first, o rotated back
//! out     = wo_b(‖_g wo_a[g](o[:, g]))           eight groups of 4096 → 1024 each, then 8192 → d
//! ```
//!
//! Every block-scaled projection is an MXFP8 [`Linear`]: its activation is encoded per 32 by the
//! covering rule the reference's `act_quant` states and multiplied with FP32 accumulation, so the
//! grouped output projection (the reference dequantizes `wo_a` to BF16 and fake-quantizes `o` before
//! its einsum) multiplies the same values.

use nn::family::{INDEX_KEY, Record};
use nn::functional::{self, Plane, Ring, Selected};
use nn::modules::{
    Candidates, Compressed, Compressor, CompressorConfig, Indexer, IndexerConfig, LatentAttention,
    LatentAttentionConfig, LatentBatch, Linear, LinearConfig, Parallel,
};
use nn::{DType, KvCache, Result, Split, Tensor, VarBuilder};

use crate::deepseek_v41::ID;
use crate::deepseek_v41::cache::{
    COMPRESSED, COMPRESSED_RECORD, INDEX_RECORD, WINDOW, WINDOW_RECORD,
};
use crate::deepseek_v41::config::{Config, LayerSources};

/// The profiler class of a layer's attention.
const ATTENTION: &str = "models.deepseek_v41.attention.Attention";

/// What the layers of one forward pass on: the selection of the last index source and the
/// candidate blocks of the candidate source, read by the layers after them
/// (`model.py:1168-1182`).
#[derive(Default)]
pub struct Carried {
    /// `[tokens, top]` I32: the compressed entries each query of the last index source selected.
    pub selection: Option<Tensor>,
    /// `[tokens, blocks]` I32: the candidate source's blocks.
    pub candidates: Option<Tensor>,
}

/// The columns and planes one layer's attention reads in a forward.
pub struct Inputs<'t> {
    /// The batch's columns at the layer's rotation.
    pub batch: LatentBatch<'t>,
    /// The full-history group: the compressed latents and index keys.
    pub full: &'t KvCache,
    /// The sliding window's group: every layer's window latent.
    pub window: &'t KvCache,
    /// `[tokens]` I32: each token's compressed entry may be written; `-1` where its full-history
    /// slot is none (a padding token, a token a replay recomputes for its window alone).
    pub full_written: &'t Tensor,
    /// The compressor's ring of a layer that pools groups of tokens.
    pub ring: Option<Ring<'t>>,
}

/// Layer `layer`'s plane of `field` in `cache`, as the record operators read it: its records,
/// held in `records`, over the group's page table, in the format and layout of `record`, the
/// field's declaration ([`crate::deepseek_v41::cache`]).
pub(crate) fn plane<'t>(
    cache: &'t KvCache,
    (field, record): (&str, Record),
    layer: u32,
    records: &'t mut Option<Tensor>,
) -> Result<Plane<'t>> {
    let (format, layout) = record.operand()?;
    let held = records.insert(cache.plane(field, layer)?.records()?);

    Ok(Plane {
        records: held,
        table: &cache.block_tables,
        format,
        layout,
    })
}

/// A decoder layer's attention (module documentation).
pub struct Attention {
    layer: u32,
    sources: LayerSources,
    heads: u32,
    head_dim: u32,
    q_lora_rank: u32,
    eps: f32,
    /// `[wq_a; wkv]`: the query latent and the window latent of each token, one product.
    wq_a_wkv: Linear,
    /// `[q_lora]`: the query latent's norm gain.
    q_norm: Tensor,
    wq_b: Linear,
    attention: LatentAttention,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    /// `wo_a`'s groups: group `g` projects heads `g · heads / groups ..` to `o_lora`.
    wo_a: Vec<Linear>,
    wo_b: Linear,
}

impl Attention {
    /// Layer `layer`'s attention at `vb` (`layers.<layer>.attn`), whole on this rank.
    ///
    /// # Errors
    ///
    /// A weight the checkpoint lacks or holds in another extent or representation.
    pub fn load(config: &Config, layer: u32, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let attention = &config.attention;
        let sources = config.layers[layer as usize];
        let (hidden, heads, head_dim) = (config.hidden_size, attention.heads, attention.head_dim);
        let eps = config.rms_norm_eps;
        let linear = |inputs: u32, outputs: u32, parallel: Parallel, vb: VarBuilder<'_, '_>| {
            Linear::load(
                &LinearConfig {
                    inputs,
                    outputs,
                    bias: false,
                    parallel,
                },
                vb,
            )
        };

        let wq_a_wkv = Linear::load_merged(
            vb.clone(),
            &["wq_a", "wkv"],
            hidden,
            &[attention.q_lora_rank, head_dim],
            false,
        )?;
        let wq_b = linear(
            attention.q_lora_rank,
            heads * head_dim,
            Parallel::None,
            vb.pp("wq_b"),
        )?;
        let latent = LatentAttention::load(
            &LatentAttentionConfig {
                heads,
                head_dim,
                rotated: attention.rope_head_dim,
                window: attention.window,
                eps,
                scale: (head_dim as f32).powf(-0.5),
                split: Split::WHOLE,
            },
            vb.clone(),
        )?;

        let compressor = if sources.compresses(layer) {
            Some(Compressor::load(
                &CompressorConfig {
                    hidden,
                    width: head_dim,
                    ratio: sources.ratio,
                    rotated: attention.rope_head_dim,
                    eps,
                },
                vb.pp("compressor"),
            )?)
        } else {
            None
        };
        let indexer = if sources.indexes {
            let index = &config.indexer;
            let query = linear(
                attention.q_lora_rank,
                index.heads * index.head_dim,
                Parallel::None,
                vb.pp("indexer").pp("wq_b"),
            )?;
            Some(Indexer::load(
                &IndexerConfig {
                    hidden,
                    heads: index.heads,
                    width: index.head_dim,
                    latent: head_dim,
                    rotated: attention.rope_head_dim,
                    top: index.top,
                    ratio: sources.ratio,
                    eps,
                    owns_keys: sources.compresses(layer),
                    candidates: sources.candidates,
                    split: Split::WHOLE,
                },
                vb.pp("indexer"),
                query,
            )?)
        } else {
            None
        };

        // `wo_a` is `[groups · o_lora, heads · head_dim / groups]`: group `g`'s rows are its
        // block of the output channels.
        let groups = attention.o_groups;
        let wo_a = (0..groups)
            .map(|group| {
                linear(
                    heads * head_dim / groups,
                    groups * attention.o_lora_rank,
                    Parallel::Column(Split::new(groups, group)?),
                    vb.pp("wo_a"),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let wo_b = linear(
            groups * attention.o_lora_rank,
            hidden,
            Parallel::None,
            vb.pp("wo_b"),
        )?;

        Ok(Self {
            layer,
            sources,
            heads,
            head_dim,
            q_lora_rank: attention.q_lora_rank,
            eps,
            wq_a_wkv,
            q_norm: vb
                .pp("q_norm")
                .get(attention.q_lora_rank as usize, "weight")?,
            wq_b,
            attention: latent,
            compressor,
            indexer,
            wo_a,
            wo_b,
        })
    }

    /// The attention of `x`, `[tokens, hidden]` BF16, the layer's normalized input:
    /// `[tokens, hidden]` BF16. A source writes its compressed latents and index keys, an index
    /// source leaves its selection (and the candidate source its blocks) in `carried`, and every
    /// layer writes its window.
    ///
    /// # Errors
    ///
    /// A layer that reads a selection no index source before it left, or an operator's refusal.
    pub fn forward(
        &self,
        x: &Tensor,
        inputs: &Inputs<'_>,
        carried: &mut Carried,
    ) -> Result<Tensor> {
        let forward = || -> Result<Tensor> {
            let tokens = x.dim(0)?;
            let (heads, head_dim) = (self.heads as usize, self.head_dim as usize);
            let batch = &inputs.batch;

            // The query latent and the window latent of every token, from one product.
            let projected = self.wq_a_wkv.forward(x)?;
            let rank = self.q_lora_rank as usize;
            let qr = functional::rms_norm(
                &projected.narrow(1, 0, rank)?.contiguous()?,
                &self.q_norm,
                self.eps,
                false,
                DType::BF16,
            )?;
            let kv = projected.narrow(1, rank, head_dim)?.contiguous()?;
            let query = self
                .wq_b
                .forward(&qr)?
                .reshape((tokens, heads, head_dim))?
                .contiguous()?;

            // A source's groups the batch completes, written only where the token holds its
            // full-history entries.
            let compressed = match &self.compressor {
                Some(compressor) => {
                    let pooled = compressor.forward(x, batch, inputs.ring)?;
                    let entries = inputs
                        .full_written
                        .ge(0.0)?
                        .where_cond(&pooled.entries, inputs.full_written)?;
                    Some(Compressed {
                        latent: pooled.latent,
                        entries,
                    })
                }
                None => None,
            };

            // The index source scores and selects; every other layer reads its selection.
            let mut held = (None, None, None);
            if let Some(indexer) = &self.indexer {
                let keys = plane(
                    inputs.full,
                    (INDEX_KEY, INDEX_RECORD),
                    self.layer,
                    &mut held.0,
                )?;
                let restricted =
                    matches!(indexer.config().candidates, Candidates::Restricted { .. });
                let selection = indexer.forward(
                    (x, &qr),
                    compressed.as_ref(),
                    keys,
                    batch,
                    if restricted {
                        carried.candidates.as_ref()
                    } else {
                        None
                    },
                )?;
                if selection.candidates.is_some() {
                    carried.candidates = selection.candidates;
                }
                carried.selection = Some(selection.entries);
            }

            // The source's compressed entries, rotated at their groups' first positions and
            // encoded, before any query reads them.
            let compressed_plane = match self.sources.compressed {
                Some(_) => Some(plane(
                    inputs.full,
                    (COMPRESSED, COMPRESSED_RECORD),
                    self.layer,
                    &mut held.1,
                )?),
                None => None,
            };
            if let (Some(compressor), Some(compression), Some(plane)) =
                (&self.compressor, &compressed, compressed_plane)
            {
                compressor.store_(compression, plane, batch)?;
            }

            let selected = match compressed_plane {
                Some(plane) => {
                    let selection = carried.selection.as_ref().ok_or_else(|| {
                        ID.config(format!(
                            "layer {} reads the selection of an index source before it, and \
                             none ran",
                            self.layer
                        ))
                    })?;
                    Some(Selected { plane, selection })
                }
                None => None,
            };
            let window = plane(
                inputs.window,
                (WINDOW, WINDOW_RECORD),
                self.layer,
                &mut held.2,
            )?;
            let attended = self
                .attention
                .forward((&query, &kv), window, selected, batch)?;

            // The grouped output projection: each group's heads to its rank, then together.
            let groups = self.wo_a.len();
            let grouped = attended.reshape((tokens, groups, heads * head_dim / groups))?;
            let projected = self
                .wo_a
                .iter()
                .enumerate()
                .map(|(group, projection)| {
                    projection.forward(&grouped.narrow(1, group, 1)?.squeeze(1)?.contiguous()?)
                })
                .collect::<Result<Vec<_>>>()?;
            self.wo_b.forward(&Tensor::cat(&projected, 1)?)
        };

        nn::profiler::range(ATTENTION, &format!("layers.{}.attn", self.layer), forward)
    }
}
