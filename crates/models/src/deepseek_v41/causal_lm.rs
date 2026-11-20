//! The causal language model of `DeepseekV41ForCausalLM` (`inference/model.py:1184-1279`).
//!
//! ```text
//! embedding:  e = embed(ids)                                  [tokens, d] BF16
//! residual:   h = [e, e, e, e]; pre = (1, 0, 0, 0)            four streams (MhcResidual)
//! layers:     h ← layer_l(h)                                  crate::deepseek_v41::decoder
//! read-out:   x = norm(Σ_j pre_j · h_j)                       the last connection's pre-mix
//! logits:     head(x) in FP32                                 [rows, vocabulary]
//! ```
//!
//! [`CausalLm::forward`] returns the read-out rows, `[tokens, d]`, and
//! [`CausalLm::compute_logits`] projects them through the head in FP32, as the reference's FP32
//! head does.
//!
//! ## Rotation
//!
//! Window-only layers rotate the trailing channels of their queries and latents by the power
//! series over `rope_theta`; every layer that reads compressed entries by YaRN over
//! `compress_rope_theta` (`model.py:680-700`). A forward computes both factor tables once.
//!
//! ## Placement
//!
//! A rank holds every layer's attention, shared expert and dense weights whole. Its routed
//! experts are the deployment's expert split ([`TextPlacement::experts`]): over an expert union
//! of data-parallel ranks a rank holds its window of every layer's 384 experts, and the union's
//! exchange carries each token's routes. The n-gram tables, 203 GB together, are split over the
//! same union by hash heads (`nn::modules::TableShard::Heads`): each rank holds the rows of a
//! quarter of every layer's heads of a union of four, and each lookup gathers every rank's row ids
//! and returns each rank its own tokens' rows along the union's axis ([`EXPERT_AXIS`]). A rank
//! with no batch of its own joins the step in both ([`CausalLm::join_expert_step`]).

use nn::collective::{self, Split};
use nn::family::{
    CacheField, EXPERT_AXIS, ExpertOrder, ExpertShape, FusedExperts, StateShape, TextModel,
    TextPlacement, TokenHistory,
};
use nn::functional::Ring;
use nn::modules::{
    Embedding, EmbeddingConfig, LatentBatch, Linear, LinearConfig, LmHead, LmHeadConfig,
    MhcResidual, NgramMemory, NgramTable, Parallel, Rotary, TableShard, Tokens, ngram_context,
};
use nn::{
    CausalLm, Checkpoint, DType, Device, Drafter, ExpertExchange, KvCache, RecurrentState, Result,
    Shape, Tensor, TokenBatch, VarBuilder, WeightLayout, WeightSource,
};

use crate::deepseek_v41::attention::{Carried, Inputs};
use crate::deepseek_v41::cache;
use crate::deepseek_v41::config::Config;
use crate::deepseek_v41::decoder::DecoderLayer;
use crate::deepseek_v41::engram::Hashing;
use crate::deepseek_v41::{CLASS, ID};

/// The profiler class of the whole decoder stack, at path `model`.
pub(crate) const TRANSFORMER: &str = "models.deepseek_v41.transformer.Transformer";

/// `value` as an I32 tensor of one element on `like`'s device.
fn int(value: i32, like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(&[value], &Device::Cpu)?.to_device(like.device())?)
}

/// For each token, its position where `slots` (`[tokens]` U32 cache slots) writes one, and `-1`
/// where it names [`nn::NO_SLOT`]: the entries a token writes in a group.
fn written(slots: &Tensor, positions: &Tensor) -> Result<Tensor> {
    let none = int(-1, positions)?.broadcast_as(positions.shape())?;
    Ok(slots
        .view(DType::I32)?
        .ge(0.0)?
        .where_cond(positions, &none)?)
}

/// A DeepSeek-V4.1 causal language model (module documentation).
pub struct DeepseekV41ForCausalLm {
    config: Config,
    fields: Vec<CacheField>,
    embed: Embedding,
    /// The rotation of the window-only layers.
    rope: Rotary,
    /// The rotation of the layers that read compressed entries.
    compressed_rope: Rotary,
    layers: Vec<DecoderLayer>,
    /// `[d]`: the read-out's norm gain.
    norm: Tensor,
    head: LmHead,
    /// Whether the n-gram tables are split over an expert union of ranks that serve their own
    /// tokens, whose lookups collect along its axis.
    served: bool,
}

impl DeepseekV41ForCausalLm {
    /// The model `config` describes, the configuration of `checkpoint`, its modules read under
    /// `vb` (the checkpoint's root), at the ranks `placement` states, with the routed experts in
    /// the slots `order` states.
    ///
    /// # Errors
    ///
    /// A tensor, pipeline or sequence split, which this model's layers do not serve; a drafter,
    /// which it does not load; a tokenizer the n-gram hashing cannot read; or a module the loader
    /// refuses.
    pub fn load(
        config: &Config,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Self> {
        if placement.tensor.size() > 1
            || placement.pipeline.size() > 1
            || placement.sequence.split.size() > 1
            || placement.drafter
        {
            return Err(ID.config(format!(
                "a tensor split over {}, a pipeline over {}, a sequence split over {} or a \
                 drafter: a rank holds the attention whole and its share of the routed experts \
                 and n-gram tables of an expert union, and the drafter is not loaded",
                placement.tensor.size(),
                placement.pipeline.size(),
                placement.sequence.split.size()
            )));
        }

        // The n-gram tables follow the experts' split: whole on a rank of no union, by hash heads
        // over a union, each rank serving its own tokens.
        let union = placement.experts;
        let shard = if union.size() > 1 {
            TableShard::Heads(union)
        } else {
            TableShard::Whole
        };
        let hashing = match &config.engram {
            Some(_) => Some(Hashing::open(
                config,
                &checkpoint.root().join("tokenizer.json"),
            )?),
            None => None,
        };

        let layers = (0..config.num_hidden_layers)
            .map(|layer| {
                let vb = vb.pp("layers").pp(layer);
                let engram = match &hashing {
                    Some(hashing) if config.has_engram(layer) => {
                        Some(engram(config, hashing, layer, vb.pp("engram"), shard)?)
                    }
                    _ => None,
                };
                DecoderLayer::load(config, layer, vb, engram, placement, order)
            })
            .collect::<Result<Vec<_>>>()?;

        let hidden = config.hidden_size;
        Ok(Self {
            config: config.clone(),
            fields: cache::fields(config)?,
            embed: Embedding::load(
                &EmbeddingConfig {
                    vocabulary: config.vocab_size,
                    hidden,
                    split: Split::WHOLE,
                },
                vb.pp("embed"),
            )?,
            rope: Rotary::load(&config.attention.rope, vb.pp("rotary"))?,
            compressed_rope: Rotary::load(
                &config.attention.compressed_rope,
                vb.pp("compressed_rotary"),
            )?,
            layers,
            norm: vb.pp("norm").get(hidden as usize, "weight")?,
            // The reference's head is FP32 (`ParallelHead`), the checkpoint's BF16 rows widened.
            head: LmHead::load(
                &LmHeadConfig {
                    vocabulary: config.vocab_size,
                    hidden,
                    split: Split::WHOLE,
                },
                vb.pp("head").to_dtype(DType::F32),
            )?,
            served: hashing.is_some() && union.size() > 1,
        })
    }

    /// The checkpoint configuration this model was built for.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Whose tokens the n-gram tables' other ranks look up in a forward lent `exchange`: each
    /// rank's own over a union, padded to the most rows one rank exchanges in any step.
    ///
    /// The members of one step may run graphs of different capacity classes (a member's
    /// prefill bucket and another's join are captured for different classes), so a class's rows
    /// are not one size for every member; the transport's rows per rank are, and the lookups'
    /// collectives move blocks of that size.
    fn tokens(&self, exchange: Option<&ExpertExchange>) -> Result<Tokens> {
        if !self.served {
            return Ok(Tokens::Shared);
        }
        let exchange = exchange.ok_or_else(|| {
            ID.config(
                "the n-gram tables are split over the expert union, and the forward was lent no \
                 exchange to shape its lookups by"
                    .to_owned(),
            )
        })?;
        Ok(Tokens::Own {
            block: exchange.rank_rows as usize,
        })
    }
}

/// Layer `layer`'s n-gram memory at `vb` (`layers.<layer>.engram`), its table held as `shard`
/// states: the rows of the rank's hash heads, a contiguous range of the table since every head's
/// rows follow the one before it.
fn engram(
    config: &Config,
    hashing: &Hashing,
    layer: u32,
    vb: VarBuilder<'_, '_>,
    shard: TableShard,
) -> Result<NgramMemory> {
    let ngram = config
        .ngram()
        .ok_or_else(|| ID.config("the model has no n-gram memory".to_owned()))?;
    let (Some(primes), Some(offsets)) = (hashing.primes(layer), hashing.offsets(layer)) else {
        return Err(ID.config(format!("layer {layer} holds no n-gram memory")));
    };
    // The heads' ranges in column order: n-gram length first, then head.
    let ranges: Vec<(u64, u64)> = offsets
        .iter()
        .flatten()
        .zip(primes.iter().flatten())
        .map(|(&first, &rows)| (first, rows))
        .collect();
    let columns = ranges.len();
    let (first, rows) = match shard {
        TableShard::Whole => (0, ranges.iter().map(|(_, rows)| rows).sum::<u64>()),
        TableShard::Heads(split) => {
            let share = columns / split.size() as usize;
            let held = &ranges[split.rank() as usize * share..][..share];
            let contiguous = held
                .windows(2)
                .all(|pair| pair[0].0 + pair[0].1 == pair[1].0);
            if !columns.is_multiple_of(split.size() as usize) || !contiguous {
                return Err(ID.config(format!(
                    "layer {layer}'s {columns} hash heads are no contiguous share of {} ranks",
                    split.size()
                )));
            }
            (held[0].0, held.iter().map(|(_, rows)| rows).sum::<u64>())
        }
        TableShard::Rows(_) => {
            return Err(ID.config("the n-gram tables are split by hash heads".to_owned()));
        }
    };

    // The table's E4M3 rows and their UE8M0 exponent bytes, read as the checkpoint stores them.
    let width = u64::from(ngram.row);
    let read = |name: &str, row: u64, dtype: DType| -> Result<Tensor> {
        let tensor = vb.pp("embed").name(name);
        let shape = Shape::from((rows as usize, row as usize));
        vb.loader().load(
            &tensor,
            &WeightSource::Region {
                tensor: tensor.clone(),
                start: first * row,
                shape: shape.clone(),
            },
            WeightLayout::ROW_MAJOR,
            &shape,
            Some(dtype),
            false,
        )
    };
    let table = NgramTable::new(
        read("weight", width, DType::F8E4M3)?,
        read("scale", width / 32, DType::U8)?,
        first,
        shard,
    )?;

    let (streams, hidden) = (ngram.streams as usize, ngram.hidden as usize);
    let projection = Linear::load(
        &LinearConfig {
            inputs: ngram.columns() * ngram.row,
            outputs: (ngram.streams + 1) * ngram.hidden,
            bias: false,
            parallel: Parallel::None,
        },
        vb.pp("wkv"),
    )?;
    NgramMemory::new(
        &ngram,
        hashing.layer(layer, &vb)?,
        table,
        projection,
        vb.get((streams, hidden), "q_weight")?,
        vb.get((streams, hidden), "k_weight")?,
    )
}

impl CausalLm for DeepseekV41ForCausalLm {
    fn embed_input_ids(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed.forward(ids)
    }

    /// The layers over the batch's embeddings `hidden`, `[tokens, d]`, which every stream opens
    /// as: the read-out rows, `[tokens, d]` BF16. `cache` is the full-history group and then the
    /// window's; `state` holds the compressor rings.
    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &[KvCache],
        state: Option<&RecurrentState>,
        exchange: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let (tokens, width) = (batch.tokens(), self.config.hidden_size as usize);
        if hidden.dims() != [tokens, width] {
            return Err(ID.config(format!(
                "hidden is {:?}, [{tokens}, {width}] expected",
                hidden.dims()
            )));
        }
        let [full, window] = cache else {
            return Err(ID.config(format!(
                "the full-history group and the window's, and the forward was lent {} groups",
                cache.len()
            )));
        };
        let state = state.ok_or_else(|| {
            ID.config("the compressors keep a ring, and the forward was lent no state".to_owned())
        })?;
        batch.check()?;
        full.check(batch)?;
        window.check(batch)?;

        let forward = || -> Result<Tensor> {
            let starts = window
                .starts
                .as_ref()
                .ok_or_else(|| ID.config("the window's group states no rows' starts".to_owned()))?;
            let (rows, _) = batch.token_rows()?;
            let window_written = written(&window.slot_mapping, &batch.positions)?;
            let full_written = written(&full.slot_mapping, &batch.positions)?;

            // The factor tables of both rotations, over the batch's positions.
            let (cos, sin) = self.rope.forward(&batch.positions)?;
            let (compressed_cos, compressed_sin) =
                self.compressed_rope.forward(&batch.positions)?;
            let latent = |cos, sin, inv_freq| LatentBatch {
                positions: &batch.positions,
                written: &window_written,
                rows: &rows,
                window_starts: starts,
                cos,
                sin,
                inv_freq,
            };
            let window_only = latent(&cos, &sin, self.rope.inverse_frequencies());
            let compressing = latent(
                &compressed_cos,
                &compressed_sin,
                self.compressed_rope.inverse_frequencies(),
            );

            let context = if self.config.engram.is_some() {
                Some(ngram_context(batch)?)
            } else {
                None
            };
            let lookups = self.tokens(exchange)?;

            let mut residual = MhcResidual::expand(&self.config.mhc(), hidden)?;
            let mut carried = Carried::default();
            let mut pooling = 0;
            for (index, layer) in self.layers.iter().enumerate() {
                let sources = &self.config.layers[index];
                let ring = if sources.compresses(index as u32) && sources.ratio > 1 {
                    let rows = state
                        .live
                        .get(pooling)
                        .and_then(|fields| fields.first())
                        .ok_or_else(|| {
                            ID.config(format!("the state lends no ring for layer {index}"))
                        })?;
                    pooling += 1;
                    Some(Ring {
                        rows,
                        slots: &state.slots,
                        query_offsets: &state.query_offsets,
                        seq_lens: &state.seq_lens,
                    })
                } else {
                    None
                };
                let inputs = Inputs {
                    batch: if sources.ratio == 0 {
                        window_only
                    } else {
                        compressing
                    },
                    full,
                    window,
                    full_written: &full_written,
                    ring,
                };
                layer.forward(
                    &mut residual,
                    (context.as_ref(), lookups),
                    &inputs,
                    &mut carried,
                    batch.live_tokens.as_ref(),
                    exchange,
                )?;
            }
            residual.collapse(&self.norm, self.config.rms_norm_eps)
        };

        nn::profiler::range(TRANSFORMER, "model", forward)
    }

    /// The FP32 logits of read-out rows `hidden`, `[rows, d]`.
    fn compute_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.head.forward(&hidden.to_dtype(DType::F32)?)
    }

    /// Every layer's part of a step this rank holds no tokens of, in the order a forward takes
    /// them: the n-gram lookup of a layer with a memory, then the routed experts.
    fn join_expert_step(&self, exchange: &ExpertExchange) -> Result<()> {
        for layer in &self.layers {
            if let Some(memory) = layer.engram().filter(|_| self.served) {
                collective::along(EXPERT_AXIS, || {
                    memory.join(exchange.rank_rows as usize, DType::BF16)
                })?;
            }
            layer.experts().join_expert_step(exchange)?;
        }
        Ok(())
    }
}

impl TextModel for DeepseekV41ForCausalLm {
    fn hidden(&self) -> u32 {
        self.config.hidden_size
    }

    fn vocabulary(&self) -> u32 {
        self.config.vocab_size
    }

    fn dtype(&self) -> DType {
        self.embed.weight().dtype()
    }

    fn cache(&self) -> Vec<CacheField> {
        self.fields.clone()
    }

    fn state_layers(&self) -> Vec<StateShape> {
        cache::remainder(&self.config)
    }

    fn experts(&self) -> Option<ExpertShape> {
        let experts = &self.config.experts;
        Some(ExpertShape {
            experts: experts.experts,
            intermediate: experts.intermediate,
            top: experts.top,
            // Every layer's experts load alike: the first states what a fused launch runs.
            fused: self
                .layers
                .first()
                .map_or(FusedExperts::NONE, |layer| layer.experts().fused_experts()),
        })
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        None
    }

    /// The n-gram memories hash each token with the `N − 1` before it, a position before the
    /// request reading the checkpoint's padding id.
    fn token_history(&self) -> Option<TokenHistory> {
        self.config.engram.as_ref().map(|engram| TokenHistory {
            width: engram.max_ngram_size - 1,
            fill: engram.pad_token_id,
        })
    }

    /// The n-gram lookups of a rank of an expert union collect along it.
    fn expert_collectives(&self) -> bool {
        self.served
    }
}

impl std::fmt::Debug for DeepseekV41ForCausalLm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct(CLASS)
            .field("layers", &self.layers.len())
            .field("served", &self.served)
            .finish_non_exhaustive()
    }
}
