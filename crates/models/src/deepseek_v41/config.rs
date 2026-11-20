//! The configuration of DeepSeek-V4.1, from the checkpoint's `config.json` (`text_config`) and
//! its quantization declaration, normalized once at the loading boundary into typed values the
//! modules read.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`].
//! The values compose along the numerical submodules:
//!
//! - the residual: `hc_mult` streams of `hidden_size`, each combination matrix taking
//!   `hc_sinkhorn_iters` Sinkhorn iterations under `hc_eps`; the mixes' statistic and every
//!   sub-layer input's normalization take the model's `rms_norm_eps`
//!   (`inference/model.py:950-976`);
//! - the attention ([`AttentionConfig`]): `num_attention_heads` query heads of `head_dim` over
//!   one shared latent of `head_dim`, the trailing `qk_rope_head_dim` channels rotated, the
//!   query's low-rank `q_lora_rank`, the grouped output projection (`o_groups` groups to
//!   `o_lora_rank` each) and the `sliding_window`; window-only layers rotate by `rope_theta`,
//!   compressing layers by YaRN over `compress_rope_theta` (`model.py:680-700`);
//! - each layer's sources ([`LayerSources`]): its compression ratio (`compress_ratios`), the owner
//!   of the compressed latents and index keys it reads (the nearest of `kv_source_layer_ids` at or
//!   below it), whether it runs an indexer of its own (`index_source_layer_ids`) or reads the
//!   selection of the indexer before it, and its part in the candidate blocks
//!   (`candidate_source_layer_id`, `model.py:503-611`);
//! - the indexer ([`IndexConfig`]): `index_n_heads` heads of `index_head_dim`, the best
//!   `index_topk` entries, `candidate_topk_blocks` blocks of `candidate_block_size`;
//! - the routed experts ([`ExpertConfig`]): `n_routed_experts`, `num_experts_per_tok` of them per
//!   token, `moe_intermediate_size` wide, `sqrtsoftplus` scores renormalized and scaled by
//!   `routed_scaling_factor`, SwiGLU clamped at `swiglu_limit`, beside one shared expert;
//! - the n-gram memories of the layers `engram_layer_ids` ([`EngramConfig`]), which hash the 2- to
//!   `engram_max_ngram_size`-grams of the normalized vocabulary into `engram_n_heads` heads each
//!   (`inference/engram.py`);
//! - the DSpark drafter ([`DsparkConfig`]): `num_nextn_predict_layers` window-only layers after
//!   the decoder's (`compress_ratios` states them zero), each a hyper-connected layer of the
//!   decoder's attention shape over `dspark_n_routed_experts` experts, `dspark_num_experts_per_tok`
//!   a token; blocks of `dspark_block_size` tokens filled with `dspark_noise_token_id`; the
//!   residual entering the layers `dspark_target_layer_ids` as its input; a Markov term of rank
//!   `dspark_markov_rank` (`model.py:1023-1160`).
//!
//! The vision tower (`vision_config`) is read by no module here.

use nn::config::Fields;
use nn::modules::{Candidates, MhcConfig, NgramConfig, Precision, RotaryConfig, Spectrum, Yarn};
use nn::quantization::Declaration;
use nn::{Checkpoint, Result};

use crate::deepseek_v41::ID;

/// What the n-gram memories of the model state.
#[derive(Clone, PartialEq, Debug)]
pub struct EngramConfig {
    /// The layers that add an n-gram memory into their streams before their attention.
    pub layer_ids: Vec<u32>,
    /// The table rows of each of those layers, in their order.
    pub num_embeddings: Vec<u64>,
    /// The longest n-gram, `N`.
    pub max_ngram_size: u32,
    /// The number each head's prime modulus is searched above.
    pub vocab_size: u64,
    /// Heads of each n-gram length.
    pub heads: u32,
    /// Elements of one table row.
    pub head_dim: u32,
    /// The vocabulary id a context holds before a request's first token.
    pub pad_token_id: u32,
    /// The size of the normalized vocabulary every multiplier is drawn for.
    pub compressed_vocab_size: u32,
}

/// The sparse latent attention every layer shares the shape of.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AttentionConfig {
    /// Query heads.
    pub heads: u32,
    /// Elements of a head and of the latent that is every entry's key and value.
    pub head_dim: u32,
    /// The trailing channels of a head and of a latent that rotate.
    pub rope_head_dim: u32,
    /// The width of the normalized query latent the heads are projected from.
    pub q_lora_rank: u32,
    /// The groups of heads the output projection takes apart.
    pub o_groups: u32,
    /// The width each group's heads are projected to.
    pub o_lora_rank: u32,
    /// The window entries a query attends, its own included.
    pub window: u32,
    /// The rotation of a window-only layer: the power series over `rope_theta`.
    pub rope: RotaryConfig,
    /// The rotation of a layer that reads compressed entries: YaRN over `compress_rope_theta`.
    pub compressed_rope: RotaryConfig,
}

/// The indexer every index source runs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IndexConfig {
    /// Index heads.
    pub heads: u32,
    /// Elements of an index head and of an index key.
    pub head_dim: u32,
    /// The compressed entries each query keeps at most.
    pub top: u32,
    /// The candidate blocks the candidate source keeps.
    pub candidate_blocks: u32,
    /// Entries of one candidate block.
    pub candidate_block: u32,
}

/// The routed experts of every layer, and the shared expert beside them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExpertConfig {
    /// Routed experts of a layer.
    pub experts: u32,
    /// Experts each token is routed to.
    pub top: u32,
    /// One expert's gated width, and the shared expert's.
    pub intermediate: u32,
    /// The factor every route's weight is multiplied by after the renormalization.
    pub routed_scale: f32,
    /// The clamp of the gated activation, `silu(min(gate, L)) · clamp(up, ±L)`.
    pub limit: f32,
}

/// The DSpark drafter (`model.py:1023-1160`): draft layers after the decoder's that draft a
/// block of tokens in one forward from the decoder's residual.
#[derive(Clone, PartialEq, Debug)]
pub struct DsparkConfig {
    /// Draft layers: layer `num_hidden_layers + i` is the checkpoint's `mtp.<i>`.
    pub layers: u32,
    /// Tokens of a block: the anchor and its fills, the drafts one forward proposes.
    pub block: u32,
    /// The token id every position of a block after its anchor embeds.
    pub noise_token_id: u32,
    /// The decoder layers whose input residual, averaged over its streams, the drafter reads,
    /// in the order its projection concatenates them.
    pub target_layer_ids: Vec<u32>,
    /// The width of the Markov term's token embedding.
    pub markov_rank: u32,
    /// The draft layers' routed experts, beside each layer's shared expert of the decoder's
    /// width.
    pub experts: ExpertConfig,
}

/// Which cache entries and selection one layer's attention reads (`model.py:640-660`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayerSources {
    /// Tokens one compressed entry stands for; zero for a layer that attends its window alone.
    pub ratio: u32,
    /// The layer whose compressed latents and index keys this layer reads: itself when it
    /// compresses its own; `None` for a window-only layer.
    pub compressed: Option<u32>,
    /// Whether the layer runs an indexer of its own; a layer that does not reads the selection
    /// of the last index source before it.
    pub indexes: bool,
    /// The layer's part in the candidate blocks, when it indexes.
    pub candidates: Candidates,
}

impl LayerSources {
    /// Whether layer `layer`, whose sources these are, compresses its tokens into the
    /// compressed latents and index keys other layers read.
    pub fn compresses(&self, layer: u32) -> bool {
        self.compressed == Some(layer)
    }
}

/// The configuration of the model (module documentation).
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The hidden width, `d`.
    pub hidden_size: u32,
    /// The vocabulary of the embedding and the head.
    pub vocab_size: u32,
    /// The decoder layers.
    pub num_hidden_layers: u32,
    /// The longest sequence the positions cover.
    pub max_position_embeddings: u32,
    /// The epsilon of every RMS normalization.
    pub rms_norm_eps: f32,
    /// The residual's streams, `hc_mult`.
    pub hc_mult: u32,
    /// Sinkhorn iterations of a combination matrix.
    pub hc_sinkhorn_iters: u32,
    /// The epsilon of a pre-mix and of every Sinkhorn step.
    pub hc_eps: f32,
    /// The attention's shape.
    pub attention: AttentionConfig,
    /// The indexer's shape.
    pub indexer: IndexConfig,
    /// The routed and shared experts.
    pub experts: ExpertConfig,
    /// Each decoder layer's sources, in layer order.
    pub layers: Vec<LayerSources>,
    /// The n-gram memories, when the model has any.
    pub engram: Option<EngramConfig>,
    /// The DSpark drafter, when the checkpoint carries one.
    pub dspark: Option<DsparkConfig>,
    /// What the checkpoint's export declares of its weights' representation.
    pub quantization: Declaration,
}

impl Config {
    /// Reads what `checkpoint`'s `config.json` and its quantization declaration state.
    ///
    /// # Errors
    ///
    /// No `config.json`, a declaration the loader does not read, or what [`Self::parse`]
    /// refuses.
    pub fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let value = checkpoint.sidecar("config.json")?.ok_or_else(|| {
            ID.config(format!(
                "{} holds no config.json",
                checkpoint.root().display()
            ))
        })?;
        Self::parse(
            &Fields::of("config.json", &value)?,
            Declaration::read(checkpoint)?,
        )
    }

    /// Reads `text_config` of a parsed `config.json`, its weights declared as `quantization`
    /// states.
    ///
    /// # Errors
    ///
    /// A field missing or of another type, n-gram memories whose layers and tables disagree,
    /// compression ratios that do not cover every layer, or sources whose ratios disagree.
    pub fn parse(root: &Fields<'_>, quantization: Declaration) -> Result<Self> {
        let text = root.object("text_config")?;
        let engram = match text.optional_indices("engram_layer_ids")? {
            Some(layer_ids) if !layer_ids.is_empty() => {
                let num_embeddings = text
                    .numbers("engram_num_embeddings")?
                    .into_iter()
                    .map(|rows| rows as u64)
                    .collect::<Vec<_>>();
                if num_embeddings.len() != layer_ids.len() {
                    return Err(ID.config(format!(
                        "{} n-gram layers state {} tables",
                        layer_ids.len(),
                        num_embeddings.len()
                    )));
                }
                Some(EngramConfig {
                    layer_ids,
                    num_embeddings,
                    max_ngram_size: text.count("engram_max_ngram_size")?,
                    vocab_size: text.number("engram_vocab_size")? as u64,
                    heads: text.count("engram_n_heads")?,
                    head_dim: text.count("engram_head_dim")?,
                    pad_token_id: text.index("engram_pad_token_id")?,
                    compressed_vocab_size: text.count("engram_compressed_vocab_size")?,
                })
            }
            _ => None,
        };

        let rope_head_dim = text.count("qk_rope_head_dim")?;
        let scaling = text.object("rope_scaling")?;
        if scaling.text("rope_type")? != "yarn" {
            return Err(ID.config(format!(
                "rope_scaling states {:?}; the compressing layers rotate by YaRN",
                scaling.text("rope_type")?
            )));
        }
        // The reference's table (`precompute_freqs_cis`): f32 arithmetic and no attention
        // factor or magnitude scale.
        let yarn = Yarn {
            factor: scaling.positive("factor")? as f32,
            original_max_position_embeddings: scaling.count("original_max_position_embeddings")?,
            attention_factor: None,
            beta_fast: scaling.positive("beta_fast")? as f32,
            beta_slow: scaling.positive("beta_slow")? as f32,
            mscale: None,
            mscale_all_dim: None,
        };
        let attention = AttentionConfig {
            heads: text.count("num_attention_heads")?,
            head_dim: text.count("head_dim")?,
            rope_head_dim,
            q_lora_rank: text.count("q_lora_rank")?,
            o_groups: text.count("o_groups")?,
            o_lora_rank: text.count("o_lora_rank")?,
            window: text.count("sliding_window")?,
            rope: RotaryConfig {
                theta: text.positive("rope_theta")? as f32,
                dim: rope_head_dim,
                spectrum: Spectrum::Default,
                precision: Precision::F32,
            },
            compressed_rope: RotaryConfig {
                theta: text.positive("compress_rope_theta")? as f32,
                dim: rope_head_dim,
                spectrum: Spectrum::Yarn(yarn),
                precision: Precision::F32,
            },
        };
        let indexer = IndexConfig {
            heads: text.count("index_n_heads")?,
            head_dim: text.count("index_head_dim")?,
            top: text.count("index_topk")?,
            candidate_blocks: text.count("candidate_topk_blocks")?,
            candidate_block: text.count("candidate_block_size")?,
        };

        if text.text("scoring_func")? != "sqrtsoftplus"
            || !text.flag("norm_topk_prob")?
            || text.count("n_shared_experts")? != 1
        {
            return Err(ID.config(
                "the routed experts score by sqrtsoftplus, renormalize their weights and sit \
                 beside one shared expert"
                    .to_owned(),
            ));
        }
        let experts = ExpertConfig {
            experts: text.count("n_routed_experts")?,
            top: text.count("num_experts_per_tok")?,
            intermediate: text.count("moe_intermediate_size")?,
            routed_scale: text.positive("routed_scaling_factor")? as f32,
            limit: text.positive("swiglu_limit")? as f32,
        };

        let num_hidden_layers = text.count("num_hidden_layers")?;
        let ratios = text.indices("compress_ratios")?;
        let dspark = dspark(&text, num_hidden_layers, &ratios, &experts)?;
        let layers = sources(
            num_hidden_layers,
            &ratios,
            &text.indices("kv_source_layer_ids")?,
            &text.indices("index_source_layer_ids")?,
            text.optional_index("candidate_source_layer_id")?,
            &indexer,
        )?;

        Ok(Self {
            hidden_size: text.count("hidden_size")?,
            vocab_size: text.count("vocab_size")?,
            num_hidden_layers,
            max_position_embeddings: text.count("max_position_embeddings")?,
            rms_norm_eps: text.positive("rms_norm_eps")? as f32,
            hc_mult: text.count("hc_mult")?,
            hc_sinkhorn_iters: text.count("hc_sinkhorn_iters")?,
            hc_eps: text.positive("hc_eps")? as f32,
            attention,
            indexer,
            experts,
            layers,
            engram,
            dspark,
            quantization,
        })
    }

    /// The residual's configuration: every connection's mixes and its input's normalization.
    pub fn mhc(&self) -> MhcConfig {
        MhcConfig {
            streams: self.hc_mult,
            hidden: self.hidden_size,
            iterations: self.hc_sinkhorn_iters,
            eps: self.hc_eps,
            rms_eps: self.rms_norm_eps,
            norm_eps: self.rms_norm_eps,
        }
    }

    /// An n-gram memory's configuration, when the model has n-gram memories.
    pub fn ngram(&self) -> Option<NgramConfig> {
        self.engram.as_ref().map(|engram| NgramConfig {
            order: engram.max_ngram_size,
            heads: engram.heads,
            row: engram.head_dim,
            streams: self.hc_mult,
            hidden: self.hidden_size,
            eps: self.rms_norm_eps,
        })
    }

    /// The layers that compress their tokens, each with its ratio, ascending: the owners of the
    /// compressed latents and the index keys.
    pub fn compressors(&self) -> Vec<(u32, u32)> {
        (0..self.num_hidden_layers)
            .filter(|&layer| self.layers[layer as usize].compresses(layer))
            .map(|layer| (layer, self.layers[layer as usize].ratio))
            .collect()
    }

    /// The draft layers' indices, after the decoder's: the layers whose window the DSpark
    /// drafter keeps, empty for a checkpoint that carries none.
    pub fn draft_layers(&self) -> std::ops::Range<u32> {
        let layers = self.dspark.as_ref().map_or(0, |dspark| dspark.layers);
        self.num_hidden_layers..self.num_hidden_layers + layers
    }

    /// Whether layer `layer` adds an n-gram memory into its streams.
    pub fn has_engram(&self, layer: u32) -> bool {
        self.engram
            .as_ref()
            .is_some_and(|engram| engram.layer_ids.contains(&layer))
    }
}

/// The DSpark drafter `text` states after the decoder's `layers` layers, whose routed experts
/// beside the drafter's are `experts`: `None` for a checkpoint of no draft layer or no block.
///
/// # Errors
///
/// A field missing or of another type, draft layers that `ratios` does not state window-only
/// (the reference's draft attention reads its window alone), target layers outside the decoder,
/// or a block of fewer than two tokens.
fn dspark(
    text: &Fields<'_>,
    layers: u32,
    ratios: &[u32],
    experts: &ExpertConfig,
) -> Result<Option<DsparkConfig>> {
    let draft_layers = text
        .optional_count("num_nextn_predict_layers")?
        .unwrap_or(0);
    let block = text.optional_count("dspark_block_size")?.unwrap_or(0);
    if draft_layers == 0 || block == 0 {
        return Ok(None);
    }

    let window_only =
        (layers..layers + draft_layers).all(|layer| ratios.get(layer as usize).copied() == Some(0));
    let target_layer_ids = text.indices("dspark_target_layer_ids")?;
    if !window_only
        || block < 2
        || target_layer_ids.is_empty()
        || target_layer_ids.iter().any(|&layer| layer >= layers)
    {
        return Err(ID.config(format!(
            "DSpark layers {layers}..{} of compression ratios {:?}, blocks of {block} and target \
             layers {target_layer_ids:?}: the draft layers attend their window alone, a block \
             holds an anchor and a fill, and the targets are decoder layers",
            layers + draft_layers,
            ratios.get(layers as usize..)
        )));
    }

    Ok(Some(DsparkConfig {
        layers: draft_layers,
        block,
        noise_token_id: text.index("dspark_noise_token_id")?,
        target_layer_ids,
        markov_rank: text.count("dspark_markov_rank")?,
        experts: ExpertConfig {
            experts: text.count("dspark_n_routed_experts")?,
            top: text.count("dspark_num_experts_per_tok")?,
            ..*experts
        },
    }))
}

/// Each of `layers` layers' sources (`model.py:640-660`): layer `l` compresses by `ratios[l]`,
/// reads the compressed latents of the nearest of `kv` at or below it (of its own ratio), indexes
/// when `index` names it, and takes part in the candidate blocks of `candidate`, the source, as
/// the indexers after it do.
fn sources(
    layers: u32,
    ratios: &[u32],
    kv: &[u32],
    index: &[u32],
    candidate: Option<u32>,
    indexer: &IndexConfig,
) -> Result<Vec<LayerSources>> {
    if ratios.len() < layers as usize {
        return Err(ID.config(format!(
            "{} compression ratios for {layers} layers",
            ratios.len()
        )));
    }

    (0..layers)
        .map(|layer| {
            let ratio = ratios[layer as usize];
            if ratio == 0 {
                return Ok(LayerSources {
                    ratio,
                    compressed: None,
                    indexes: false,
                    candidates: Candidates::None,
                });
            }
            let source = kv
                .iter()
                .copied()
                .filter(|&source| source <= layer)
                .max()
                .filter(|&source| ratios[source as usize] == ratio)
                .ok_or_else(|| {
                    ID.config(format!(
                        "layer {layer} compresses by {ratio} and no compressed source of that \
                         ratio precedes it among {kv:?}"
                    ))
                })?;
            let indexes = index.contains(&layer);
            if !index
                .iter()
                .any(|&earlier| earlier <= layer && earlier >= source)
            {
                return Err(ID.config(format!(
                    "layer {layer} reads a selection, and no indexer at or after its source \
                     {source} precedes it"
                )));
            }
            let candidates = match candidate {
                Some(source) if indexes && layer == source => Candidates::Source {
                    blocks: indexer.candidate_blocks,
                    block: indexer.candidate_block,
                },
                Some(source) if indexes && layer > source => Candidates::Restricted {
                    block: indexer.candidate_block,
                },
                _ => Candidates::None,
            };
            Ok(LayerSources {
                ratio,
                compressed: Some(source),
                indexes,
                candidates,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The release's sources: two window-only layers, three owners of ratio two and their
    /// readers, then layer 20's latents every later layer reads, its candidate blocks and the
    /// four indexers restricted to them.
    #[test]
    fn each_layer_reads_the_nearest_source_and_indexer_before_it() {
        let ratios: Vec<u32> = [vec![0, 0], vec![2; 18], vec![1; 20]].concat();
        let indexer = IndexConfig {
            heads: 32,
            head_dim: 128,
            top: 512,
            candidate_blocks: 2048,
            candidate_block: 8,
        };
        let layers = sources(
            40,
            &ratios,
            &[2, 8, 14, 20],
            &[2, 8, 14, 20, 24, 28, 32, 36],
            Some(20),
            &indexer,
        )
        .unwrap();

        assert_eq!(layers[1].compressed, None);
        assert_eq!(layers[7].compressed, Some(2));
        assert!(!layers[7].indexes);
        assert!(layers[8].compresses(8));
        assert_eq!(
            layers[20].candidates,
            Candidates::Source {
                blocks: 2048,
                block: 8
            }
        );
        assert_eq!(layers[23].compressed, Some(20));
        assert_eq!(layers[24].candidates, Candidates::Restricted { block: 8 });
        assert!(!layers[25].indexes);
        assert_eq!(layers[39].compressed, Some(20));

        // A layer of another ratio than the source before it, and ratios short of the layers.
        assert!(sources(3, &[0, 2, 1], &[1], &[1], None, &indexer).is_err());
        assert!(sources(3, &[0, 2], &[1], &[1], None, &indexer).is_err());
    }
}
