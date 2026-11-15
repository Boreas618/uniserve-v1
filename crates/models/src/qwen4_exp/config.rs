//! The family's checkpoint configuration, read once at the loading boundary.
//!
//! A checkpoint's `config.json` is read here and nowhere else, through [`nn::config::Fields`]:
//! absent keys take the defaults vLLM's configuration defines, and every field is validated
//! before a configuration exists. The modules consume typed fields and never parse a
//! dictionary again.
//!
//! One [`Config`] describes a checkpoint, composed along the model's numerical submodules:
//!
//! | Submodule | Configuration |
//! | --- | --- |
//! | decoder stack, attention and its rotary embedding | [`TextConfig`] |
//! | gated delta networks | [`TextConfig::gated_delta_net`] ([`GatedDeltaNetConfig`]) |
//! | hyper-connections | [`HyperConnectionConfig`] |
//! | the attention layers' indexer | [`IndexerConfig`] |
//! | n-gram hash embedding and its PLE layers | [`NgramEmbeddingConfig`] |
//! | routed experts | [`crate::qwen3::MoeConfig`] |
//! | vision tower | [`crate::qwen3::VisionConfig`] |
//!
//! [`Config::open`] also reads what the checkpoint's other sidecars state of the same modules:
//! the quantization its export declares and the processor rules its images are sized by.
//!
//! What a deployment chooses (the tokens a step drafts, the element type of the key/value
//! cache, the tensor-parallel split) is no part of a configuration: a load is given it.

use nn::config::Fields;
use nn::functional::{Activation, StateAxes};
use nn::modules::{GatedDeltaNetConfig, Precision, RotaryConfig};
use nn::quantization::Declaration;
use nn::{PixelNorm, Split};
use processing::qwen_vl::ProcessorConfig;

use crate::qwen3::{MoeConfig, VisionConfig};
use crate::qwen4_exp::{Error, Result};

/// The model class this family builds, as request processing claims it.
pub const CLASS: &str = "Qwen4ExpForConditionalGeneration";

/// The arithmetic the family's reference evaluates its rotary spectrum in.
const PRECISION: Precision = Precision::F32;

/// A tensor every checkpoint's vision tower holds: the patch embedding.
const TOWER_WEIGHT: &str = "model.visual.patch_embed.proj.weight";

/// The decoder stack: the fields of `text_config` its layers read.
#[derive(Clone, PartialEq, Debug)]
pub struct TextConfig {
    /// Vocabulary size, before any tensor-parallel sharding.
    pub vocab_size: u32,
    /// Width of one residual stream, `H`.
    pub hidden_size: u32,
    /// Depth of the decoder stack.
    pub num_hidden_layers: u32,
    /// Query heads of one attention layer.
    pub num_attention_heads: u32,
    /// Key/value heads of one attention layer.
    pub num_key_value_heads: u32,
    /// Width of one attention head.
    pub head_dim: u32,
    /// The gate of the routed and the shared experts, and of the gated delta network's
    /// convolution.
    pub hidden_act: Activation,
    /// Variance floor of every norm of the text model.
    pub rms_norm_eps: f32,
    /// The rotary embedding over a head's leading channels: interleaved M-RoPE over the
    /// `partial_rotary_factor` of a head.
    pub rope: RotaryConfig,
    /// The context length the checkpoint was trained for: the bound on one request's keys.
    pub max_position_embeddings: u32,
    /// The token that ends a document, which opens an n-gram segment
    /// (`text_config.eos_token_id`; not the chat stop token).
    pub eos_token_id: u32,
    /// Whether each layer mixes its tokens with a gated delta network (`linear_attention`) in
    /// place of indexed attention (`full_attention`), in stack order.
    pub linear_layers: Vec<bool>,
    /// The gated delta network of the linear layers, before any split and for a deployment
    /// that drafts nothing; a load states its rank and its draft tokens.
    pub gated_delta_net: GatedDeltaNetConfig,
}

/// The hyper-connections: the residual is `streams` streams of the hidden width, read and
/// written through a low-rank gate (`hc_count`, `hc_lowrank`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HyperConnectionConfig {
    /// Residual streams, `C`; more than one.
    pub streams: u32,
    /// Rank of the gate that mixes the streams into a block's input.
    pub rank: u32,
}

/// The indexer of an attention layer: a few index heads score one compressed key per
/// `compress_ratio` tokens, and a query attends the tokens of the best groups.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndexerConfig {
    /// Index query heads (`indexer_n_heads`).
    pub heads: u32,
    /// Elements of one index head and of one index key (`indexer_head_dim`).
    pub head_dim: u32,
    /// Tokens a query row selects through its groups (`indexer_budget`).
    pub budget: u32,
    /// Tokens one compressed key stands for (`indexer_compress_ratio`).
    pub compress_ratio: u32,
}

impl IndexerConfig {
    /// Groups a query row selects: the budget over the tokens of one group.
    pub const fn block_topk(&self) -> u32 {
        self.budget / self.compress_ratio
    }

    /// Positions one query row may attend: the tokens of its selected groups and of its own
    /// unfinished group, which holds at most `compress_ratio − 1`.
    pub const fn selection(&self) -> u32 {
        self.budget + self.compress_ratio - 1
    }

    /// Raw index keys a request's ring keeps for a deployment that drafts `draft_tokens`
    /// tokens a step: the members of the group open at the last committed token and the
    /// draft rows, rounded up to whole groups
    /// (`refs/vllm/vllm/models/qwen4_exp/common/qsa_indexer.py:848-866`).
    pub const fn ring_capacity(&self, draft_tokens: u32) -> u32 {
        (self.compress_ratio + draft_tokens).next_multiple_of(self.compress_ratio)
    }
}

/// The n-gram hash embedding and the layers that add it to the streams.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NgramEmbeddingConfig {
    /// The decoder layers that hold a PLE layer, zero-based and increasing. The checkpoint
    /// states them one-based (`ple_layer_ids`); a layer's dense id is its rank here.
    pub layers: Vec<u32>,
    /// Width of the concatenated head rows (`ple_embed_dim`).
    pub embed_dim: u32,
    /// Taps of the PLE layer's convolution (`ple_conv_kernel_size`); its dilation is
    /// `ngram_size`.
    pub conv_kernel_size: u32,
    /// The longest n-gram hashed (`ngram_size`): each of the lengths `2..=ngram_size` has
    /// `heads_per_ngram` heads.
    pub ngram_size: u32,
    /// Heads of one n-gram length.
    pub heads_per_ngram: u32,
    /// The size below the first head's: head sizes are the consecutive primes above it.
    pub vocab_size_base: u64,
    /// The multiple the table's row count is rounded up to.
    pub divisible_by: u64,
    /// Shards the checkpoint stores the table in.
    pub parts: u32,
    /// The seed of the hash multipliers.
    pub seed: u64,
}

impl NgramEmbeddingConfig {
    /// Hash heads of one token: `heads_per_ngram` for each n-gram length.
    pub const fn heads(&self) -> u32 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Elements of one head's row.
    pub const fn head_dim(&self) -> u32 {
        self.embed_dim / self.heads()
    }

    /// Each head's partition of the table, `(sizes, offsets)`: the sizes are the consecutive
    /// primes above the base and the offsets their prefix sums
    /// (`refs/vllm/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:342-360`).
    pub fn head_partitions(&self) -> (Vec<u64>, Vec<u64>) {
        let (mut sizes, mut offsets) = (Vec::new(), Vec::new());
        let (mut size, mut offset) = (self.vocab_size_base, 0);

        for _ in 0..self.heads() {
            size += 1;
            while !is_prime(size) {
                size += 1;
            }
            sizes.push(size);
            offsets.push(offset);
            offset += size;
        }
        (sizes, offsets)
    }

    /// Rows of the table: the heads' partitions, rounded up to the stated multiple.
    pub fn table_rows(&self) -> u64 {
        let (sizes, _) = self.head_partitions();

        sizes
            .iter()
            .sum::<u64>()
            .next_multiple_of(self.divisible_by.max(1))
    }

    /// The odd multipliers of the PLE layer of dense id `dense_id`, one per n-gram position,
    /// newest token first: `2 · (splitmix64(seed + 10007 · dense_id + γ · (i + 1)) mod half) + 1`
    /// with `half = ((2⁶³ − 1) / vocab_size) / 2`, so the product of a token id and a
    /// multiplier stays below `2⁶³`
    /// (`refs/vllm/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:266-340`).
    pub fn multipliers(&self, dense_id: u64, vocab_size: u32) -> Vec<u64> {
        const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
        let half = (i64::MAX as u64) / u64::from(vocab_size.max(1)) / 2;

        (1..=u64::from(self.ngram_size))
            .map(|position| {
                let state = self
                    .seed
                    .wrapping_add(dense_id.wrapping_mul(10_007))
                    .wrapping_add(GAMMA.wrapping_mul(position));

                2 * (splitmix64(state) % half.max(1)) + 1
            })
            .collect()
    }
}

/// One step of the SplitMix64 generator's output function.
const fn splitmix64(state: u64) -> u64 {
    let mut value = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Whether `value` is prime, by trial division: the head sizes are near twenty million.
fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    let mut divisor = 2;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 1;
    }
    true
}

/// One checkpoint's configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The decoder stack.
    pub text: TextConfig,
    /// The hyper-connections around every block.
    pub hyper_connection: HyperConnectionConfig,
    /// The indexer of every attention layer.
    pub indexer: IndexerConfig,
    /// The n-gram hash embedding and its PLE layers.
    pub ngram: NgramEmbeddingConfig,
    /// The routed experts and the shared expert of every layer.
    pub moe: MoeConfig,
    /// The vision tower and the placeholder tokens its rows replace.
    pub vision: VisionConfig,
    /// The multi-token-prediction layers the checkpoint carries (`mtp.*`). They embed with the
    /// model's own table and project with its head.
    pub mtp_num_hidden_layers: u32,
    /// What the checkpoint's export declares about its packed weights; dense for a
    /// configuration read from `config.json` alone ([`Config::parse`]).
    pub quantization: Declaration,
    /// The processor rules the tower's images are sized by and their pixel normalization,
    /// when the checkpoint holds both the tower's weights and its processor files; `None` for
    /// a checkpoint that serves text alone.
    pub images: Option<(ProcessorConfig, PixelNorm)>,
}

impl Config {
    /// Reads and validates what `checkpoint` states: its `config.json`, the quantization its
    /// export declares, and, when it holds a vision tower, its processor files.
    ///
    /// # Errors
    ///
    /// No `config.json`, what [`Self::parse`] refuses, a declaration the loader does not
    /// serve, or processor files that are not a Qwen-VL processor's.
    pub fn open(checkpoint: &nn::Checkpoint) -> Result<Self> {
        let value = checkpoint.sidecar("config.json")?.ok_or_else(|| {
            Error::Config(format!(
                "{} holds no config.json",
                checkpoint.root().display()
            ))
        })?;
        let mut config = Self::parse(&Fields::of("config.json", &value)?)?;
        config.quantization = Declaration::read(checkpoint)?;

        // The tower serves when the checkpoint holds both its weights and its processor
        // files; otherwise the checkpoint serves text alone.
        if checkpoint.tensor(TOWER_WEIGHT).is_some() {
            config.images = crate::qwen3::config::processor(checkpoint)?;
        }
        Ok(config)
    }

    /// Validates a parsed `config.json` of [`CLASS`].
    ///
    /// # Errors
    ///
    /// Another class, a field of the wrong type or outside its domain, or an architecture
    /// feature that is not served: a recurrent state other than FP32, an output gate other
    /// than the sigmoid, a drafter with its own embedding, an indexer of several key heads, a
    /// tower with DeepStack features.
    pub fn parse(fields: &Fields<'_>) -> Result<Self> {
        let architectures = fields
            .optional_texts("architectures")?
            .ok_or_else(|| Error::Config("checkpoint declares no architectures".to_owned()))?;
        if architectures.as_slice() != [CLASS] {
            return Err(Error::Config(format!(
                "{architectures:?} is not the class this crate builds ({CLASS})"
            )));
        }
        if fields
            .optional_flag("tie_word_embeddings")?
            .unwrap_or(false)
        {
            return Err(Error::Config(
                "the vocabulary head is its own tensor; a tied head is not served".to_owned(),
            ));
        }

        let values = fields.object("text_config")?;
        let text = TextConfig::read(&values)?;

        let hyper_connection = HyperConnectionConfig {
            streams: values.count("hc_count")?,
            rank: values.count("hc_lowrank")?,
        };
        if hyper_connection.streams < 2 {
            return Err(Error::Config(
                "a hyper-connected residual has more than one stream (hc_count)".to_owned(),
            ));
        }

        let moe = MoeConfig {
            num_experts: values.count("num_experts")?,
            num_experts_per_tok: values.count("num_experts_per_tok")?,
            moe_intermediate_size: values.count("moe_intermediate_size")?,
            // The router's weights are the softmax over the selected logits whether or not the
            // export states the flag.
            norm_topk_prob: values.optional_flag("norm_topk_prob")?.unwrap_or(true),
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            shared_expert_intermediate_size: Some(values.count("shared_expert_intermediate_size")?),
        };

        let vision = VisionConfig::read(fields)?;
        if !vision.deepstack_visual_indexes.is_empty() {
            return Err(Error::Config(
                "a causal language model's prefill adds no DeepStack features, but the tower \
                 states some"
                    .to_owned(),
            ));
        }

        let config = Self {
            indexer: IndexerConfig::read(&values, &text)?,
            ngram: NgramEmbeddingConfig::read(&values, &text)?,
            text,
            hyper_connection,
            moe,
            vision,
            mtp_num_hidden_layers: mtp_layers(&values)?,
            quantization: Declaration::dense(),
            images: None,
        };
        config.check()?;
        Ok(config)
    }

    /// The checks across submodules.
    fn check(&self) -> Result<()> {
        let (text, moe) = (&self.text, &self.moe);

        if moe.num_experts == 0 || moe.num_experts_per_tok > moe.num_experts {
            return Err(Error::Config(format!(
                "{} experts per token of {} experts",
                moe.num_experts_per_tok, moe.num_experts
            )));
        }
        if self.vision.out_hidden_size != text.hidden_size {
            return Err(Error::Config(format!(
                "vision tokens are {} wide and the language model {}",
                self.vision.out_hidden_size, text.hidden_size
            )));
        }
        for token in [
            self.vision.image_token_id,
            self.vision.video_token_id,
            text.eos_token_id,
        ] {
            if token >= text.vocab_size {
                return Err(Error::Config(format!(
                    "token {token} is outside the {}-entry vocabulary",
                    text.vocab_size
                )));
            }
        }
        Ok(())
    }

    /// The layers whose mixer is indexed attention: the layers the key/value cache holds.
    pub fn attention_layers(&self) -> u32 {
        self.text
            .linear_layers
            .iter()
            .filter(|linear| !**linear)
            .count() as u32
    }

    /// The layers whose mixer is a gated delta network.
    pub fn linear_layers(&self) -> u32 {
        self.text.num_hidden_layers - self.attention_layers()
    }

    /// The index of `layer` among the layers of its own mixer: its key/value cache layer when
    /// it attends, its recurrent state layer when it is linear.
    pub fn ordinal(&self, layer: usize) -> usize {
        let linear = self.text.linear_layers[layer];

        self.text.linear_layers[..layer]
            .iter()
            .filter(|other| **other == linear)
            .count()
    }

    /// The dense id of the PLE layer `layer` holds: its rank among the layers that hold one.
    pub fn ple(&self, layer: u32) -> Option<usize> {
        self.ngram.layers.iter().position(|held| *held == layer)
    }

    /// Checks a deployment's tensor and expert splits against what they shard: the query heads
    /// and the key heads of the gated delta networks, the experts' intermediate width, the
    /// vocabulary-sized tables' shards, and the experts.
    ///
    /// The key/value heads need not divide: a group wider than them holds each head on
    /// several consecutive ranks.
    ///
    /// # Errors
    ///
    /// A split that does not divide what it shards, or a rank outside its split.
    pub fn check_splits(&self, tensor: Split, experts: Split) -> Result<()> {
        for (name, split) in [("tensor", tensor), ("expert", experts)] {
            if split.size == 0 || split.rank >= split.size {
                return Err(Error::Config(format!(
                    "{name}-parallel rank {} of a group of {}",
                    split.rank, split.size
                )));
            }
        }

        let network = &self.text.gated_delta_net;
        for (name, extent) in [
            ("attention heads", self.text.num_attention_heads),
            ("linear key heads", network.key_heads),
            (
                "expert intermediate columns",
                self.moe.moe_intermediate_size,
            ),
            ("n-gram table shards", self.ngram.parts),
        ] {
            if !extent.is_multiple_of(tensor.size) {
                return Err(Error::Config(format!(
                    "tensor parallel degree {} does not divide {extent} {name}",
                    tensor.size
                )));
            }
        }
        if !self.moe.num_experts.is_multiple_of(experts.size) {
            return Err(Error::Config(format!(
                "expert parallel degree {} does not divide {} experts",
                experts.size, self.moe.num_experts
            )));
        }
        Ok(())
    }
}

impl TextConfig {
    /// The fields of `text_config` the decoder stack reads.
    fn read(values: &Fields<'_>) -> Result<Self> {
        let hidden_size = values.count("hidden_size")?;
        let num_hidden_layers = values.count("num_hidden_layers")?;
        let num_attention_heads = values.count("num_attention_heads")?;
        let num_key_value_heads = values.count("num_key_value_heads")?;
        let head_dim = values.count("head_dim")?;
        if !num_attention_heads.is_multiple_of(num_key_value_heads) || !head_dim.is_multiple_of(2) {
            return Err(Error::Config(format!(
                "{num_attention_heads} attention heads of {head_dim} over {num_key_value_heads} \
                 key/value heads"
            )));
        }
        if values.optional_flag("attention_bias")?.unwrap_or(false) {
            return Err(Error::Config(
                "attention projections with a bias are not served".to_owned(),
            ));
        }

        // The state is FP32 in vLLM, which takes the checkpoint's `mamba_ssm_dtype`
        // (`refs/vllm/vllm/model_executor/models/config.py:1284-1309`).
        if let Some(dtype) = values.optional_text("mamba_ssm_dtype")?
            && dtype != "float32"
        {
            return Err(Error::Config(format!(
                "a recurrent state of {dtype} is not served; the state is float32"
            )));
        }
        // The gated delta network of this family gates its output norm with a sigmoid
        // (`refs/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:817-831`).
        let gate = values.optional_text("output_gate_type")?.unwrap_or("silu");
        let sigmoid_gate = match gate {
            "sigmoid" => true,
            "silu" => false,
            other => {
                return Err(Error::Config(format!(
                    "an output gate of {other} is neither sigmoid nor silu"
                )));
            }
        };

        let rms_norm_eps = values
            .optional_positive("rms_norm_eps")?
            .map_or(1e-6, |value| value as f32);
        let network = GatedDeltaNetConfig {
            hidden: hidden_size,
            key_heads: values.count("linear_num_key_heads")?,
            value_heads: values.count("linear_num_value_heads")?,
            key_dim: values.count("linear_key_head_dim")?,
            value_dim: values.count("linear_value_head_dim")?,
            conv_width: values.count("linear_conv_kernel_dim")?,
            eps: rms_norm_eps,
            sigmoid_gate,
            // vLLM stores a state as rows of values over the keys
            // (`refs/vllm/vllm/model_executor/layers/mamba/mamba_utils.py:296-300`).
            axes: StateAxes::ValueKey,
            draft_tokens: 0,
            split: Split::WHOLE,
        };
        if !network.value_heads.is_multiple_of(network.key_heads.max(1)) || network.conv_width < 2 {
            return Err(Error::Config(format!(
                "{} linear value heads over {} key heads with a convolution of {} taps",
                network.value_heads, network.key_heads, network.conv_width
            )));
        }

        let max_position_embeddings = values
            .optional_count("max_position_embeddings")?
            .unwrap_or(262_144);

        Ok(Self {
            vocab_size: values.count("vocab_size")?,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act: values
                .optional_activation("hidden_act")?
                .unwrap_or(Activation::Silu),
            rms_norm_eps,
            // The recipe states its own partial rotation.
            rope: RotaryConfig::from_hf(
                values,
                head_dim,
                max_position_embeddings,
                None,
                PRECISION,
            )?,
            max_position_embeddings,
            eos_token_id: values.index("eos_token_id")?,
            linear_layers: values.linear_attention(num_hidden_layers)?,
            gated_delta_net: network,
        })
    }
}

impl IndexerConfig {
    /// The indexer fields of `text_config`.
    fn read(values: &Fields<'_>, text: &TextConfig) -> Result<Self> {
        let config = Self {
            heads: values.count("indexer_n_heads")?,
            head_dim: values.count("indexer_head_dim")?,
            budget: values.count("indexer_budget")?,
            compress_ratio: values.count("indexer_compress_ratio")?,
        };

        if values.optional_count("indexer_kv_heads")?.unwrap_or(1) != 1 {
            return Err(Error::Config(
                "an indexer has one key head (indexer_kv_heads)".to_owned(),
            ));
        }
        // The index heads rotate by the layer's rotary embedding, so its rotated channels fit
        // an index head; a group is at least two tokens and the budget whole groups
        // (`refs/vllm/vllm/models/qwen4_exp/common/qsa_indexer.py:783-786`).
        if config.compress_ratio < 2
            || !config.budget.is_multiple_of(config.compress_ratio)
            || config.head_dim < text.rope.dim
        {
            return Err(Error::Config(format!(
                "an indexer of {}-element heads selecting {} tokens in groups of {}, under a \
                 rotary embedding of {} channels",
                config.head_dim, config.budget, config.compress_ratio, text.rope.dim
            )));
        }
        Ok(config)
    }
}

impl NgramEmbeddingConfig {
    /// The n-gram and PLE fields of `text_config`.
    fn read(values: &Fields<'_>, text: &TextConfig) -> Result<Self> {
        // The checkpoint counts layers from one: `ple_layer_ids: [2]` is the layer whose
        // tensors are `layers.1.ple.*` (`refs/vllm/vllm/models/qwen4_exp/nvidia/model.py:218-224`).
        let mut layers = Vec::new();
        for id in values
            .optional_indices("ple_layer_ids")?
            .unwrap_or_default()
        {
            if id == 0 || id > text.num_hidden_layers {
                return Err(Error::Config(format!(
                    "ple_layer_ids names layer {id} of a stack of {} (counted from one)",
                    text.num_hidden_layers
                )));
            }
            layers.push(id - 1);
        }
        layers.sort_unstable();
        layers.dedup();

        let config = Self {
            layers,
            embed_dim: values.count("ple_embed_dim")?,
            conv_kernel_size: values.count("ple_conv_kernel_size")?,
            ngram_size: values.count("ngram_size")?,
            heads_per_ngram: values.count("heads_per_ngram")?,
            vocab_size_base: u64::from(values.count("ngram_vocab_size_base")?),
            divisible_by: u64::from(
                values
                    .optional_count("make_ngram_vocab_size_divisible_by")?
                    .unwrap_or(1),
            ),
            // vLLM assumes 512 shards when the field is absent
            // (`refs/vllm/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:393`).
            parts: values.optional_count("split_ngram_parts")?.unwrap_or(512),
            seed: u64::from(values.optional_index("seed")?.unwrap_or(1234)),
        };

        // The hash reads the token and the two before it, which is what a batch states of a
        // row's history ([`nn::TokenBatch::ngram_context`]).
        if config.ngram_size != 3 {
            return Err(Error::Config(format!(
                "an n-gram size of {} is not served: the hash reads a token and the two before it",
                config.ngram_size
            )));
        }
        if config.conv_kernel_size < 2
            || config.heads() == 0
            || !config.embed_dim.is_multiple_of(config.heads())
            || !config.table_rows().is_multiple_of(u64::from(config.parts))
        {
            return Err(Error::Config(format!(
                "an n-gram embedding of {} elements over {} heads in {} shards of a table of {} \
                 rows, convolved by {} taps",
                config.embed_dim,
                config.heads(),
                config.parts,
                config.table_rows(),
                config.conv_kernel_size
            )));
        }
        if let Some(dtype) = values.optional_text("ple_embedding_dtype")?
            && dtype != "float8_e4m3fn"
        {
            return Err(Error::Config(format!(
                "an n-gram table of {dtype} is not served; the table is float8_e4m3fn"
            )));
        }
        Ok(config)
    }
}

/// The multi-token-prediction layers `text_config` states, which embed with the model's table.
fn mtp_layers(values: &Fields<'_>) -> Result<u32> {
    if values
        .optional_flag("mtp_use_dedicated_embeddings")?
        .unwrap_or(false)
    {
        return Err(Error::Config(
            "the drafter embeds with a table of its own (mtp_use_dedicated_embeddings), which \
             is not served"
                .to_owned(),
        ));
    }

    let layers = values.optional_index("mtp_num_hidden_layers")?.unwrap_or(0);
    if let Some(mtp) = values.optional_object("mtp")?
        && let Some(kinds) = mtp.optional_texts("layer_types")?
        && kinds.iter().any(|kind| *kind != "full_attention")
    {
        return Err(Error::Config(format!(
            "a drafter of layer types {kinds:?} is not served: its layers attend"
        )));
    }
    Ok(layers)
}

/// A released configuration, as this module's unit tests read it.
#[cfg(test)]
pub(crate) mod fixtures {
    /// The Qwen3.8-Flash-Next configuration, eight layers deep (six linear, two attention).
    pub(crate) fn flash_next() -> serde_json::Value {
        // The stack's own fields, then the fields of its other submodules.
        let mut text = serde_json::json!({
                "attention_bias": false,
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ],
                "hc_count": 4,
                "hc_lowrank": 320,
                "head_dim": 256,
                "heads_per_ngram": 8,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "indexer_head_dim": 128,
                "indexer_kv_heads": 1,
                "indexer_n_heads": 4,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "num_attention_heads": 24,
                "num_hidden_layers": 8,
                "num_key_value_heads": 2,
                "output_gate_type": "sigmoid",
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "vocab_size": 248320
        });
        let others = serde_json::json!({
                "make_ngram_vocab_size_divisible_by": 128,
                "moe_intermediate_size": 640,
                "mtp": {"layer_types": ["full_attention"], "num_hidden_layers": 1},
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "ple_layer_ids": [2],
                "shared_expert_intermediate_size": 640,
                "split_ngram_parts": 128
        });
        if let (Some(text), Some(others)) = (text.as_object_mut(), others.as_object()) {
            text.extend(others.clone());
        }

        serde_json::json!({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "image_token_id": 248056,
            "video_token_id": 248057,
            "tie_word_embeddings": false,
            "text_config": text,
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use nn::modules::Spectrum;

    use super::*;

    fn parse(value: &serde_json::Value) -> Result<Config> {
        Config::parse(&Fields::of("config.json", value)?)
    }

    #[test]
    fn a_checkpoint_reads_its_stack_its_streams_and_its_indexer() {
        let config = parse(&fixtures::flash_next()).expect("parsed");
        let text = &config.text;

        assert_eq!((text.hidden_size, text.vocab_size), (2560, 248_320));
        assert_eq!(
            text.linear_layers,
            [true, true, true, false, true, true, true, false]
        );
        assert_eq!((config.linear_layers(), config.attention_layers()), (6, 2));
        assert_eq!(
            (config.ordinal(3), config.ordinal(4), config.ordinal(7)),
            (0, 3, 1)
        );

        // A quarter of a 256-element head rotates, by interleaved M-RoPE over three axes.
        assert_eq!((text.rope.dim, text.rope.theta), (64, 1e7));
        assert!(matches!(
            text.rope.spectrum,
            Spectrum::Sectioned { axes: 3, .. }
        ));

        let network = text.gated_delta_net;
        assert_eq!((network.key_heads, network.value_heads), (16, 48));
        assert!(network.sigmoid_gate);
        assert_eq!(network.axes, StateAxes::ValueKey);

        assert_eq!(
            config.hyper_connection,
            HyperConnectionConfig {
                streams: 4,
                rank: 320
            }
        );
        let indexer = config.indexer;
        assert_eq!((indexer.block_topk(), indexer.selection()), (512, 2051));
        // Four keys of the open group and three drafts round up to two groups; no draft is one.
        assert_eq!((indexer.ring_capacity(3), indexer.ring_capacity(0)), (8, 4));

        assert_eq!(
            (config.moe.num_experts, config.moe.num_experts_per_tok),
            (512, 10)
        );
        assert!(config.moe.norm_topk_prob);
        assert_eq!(config.mtp_num_hidden_layers, 1);
    }

    /// The hash constants are the ones the released checkpoints store
    /// (`layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets`).
    #[test]
    fn the_hash_constants_follow_from_the_seed_and_the_base() {
        let config = parse(&fixtures::flash_next()).expect("parsed");
        let ngram = &config.ngram;

        // `ple_layer_ids` counts from one: the layer is the stack's second.
        assert_eq!(ngram.layers, [1]);
        assert_eq!((config.ple(1), config.ple(2)), (Some(0), None));
        assert_eq!((ngram.heads(), ngram.head_dim()), (16, 160));

        assert_eq!(
            ngram.multipliers(0, config.text.vocab_size),
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );

        let (sizes, offsets) = ngram.head_partitions();
        assert_eq!(
            sizes,
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171
            ]
        );
        assert_eq!((offsets[0], offsets[1]), (0, 20_000_003));
        assert_eq!(sizes.iter().sum::<u64>(), 320_001_446);
        assert_eq!(ngram.table_rows(), 320_001_536);
        assert_eq!(ngram.table_rows() / u64::from(ngram.parts), 2_500_012);
    }

    #[test]
    fn what_the_references_do_not_compute_is_refused() {
        let refused = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut value = fixtures::flash_next();
            edit(&mut value);
            parse(&value).is_err()
        };

        assert!(refused(&|value| {
            value["architectures"] = serde_json::json!(["Qwen3MoeForCausalLM"]);
        }));
        assert!(refused(&|value| {
            value["text_config"]["mamba_ssm_dtype"] = serde_json::json!("bfloat16");
        }));
        assert!(refused(&|value| {
            value["text_config"]["hc_count"] = serde_json::json!(1);
        }));
        assert!(refused(&|value| {
            value["text_config"]["indexer_kv_heads"] = serde_json::json!(2);
        }));
        assert!(refused(&|value| {
            value["text_config"]["indexer_budget"] = serde_json::json!(2046);
        }));
        assert!(refused(&|value| {
            value["text_config"]["ple_layer_ids"] = serde_json::json!([0]);
        }));
        assert!(refused(&|value| {
            value["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(true);
        }));
        assert!(refused(&|value| {
            value["text_config"]["layer_types"] = serde_json::json!(["full_attention"]);
        }));
    }

    #[test]
    fn a_split_divides_the_heads_the_experts_width_and_the_table() {
        let config = parse(&fixtures::flash_next()).expect("parsed");
        let split = |size| Split { size, rank: 0 };

        // Four ranks hold six query heads, four key heads, 160 expert columns and 32 shards
        // each; the two key/value heads are each held by two ranks.
        assert!(config.check_splits(split(4), split(1)).is_ok());
        assert!(config.check_splits(split(1), split(8)).is_ok());

        assert!(config.check_splits(split(5), split(1)).is_err());
        assert!(config.check_splits(split(1), split(3)).is_err());
        assert!(
            config
                .check_splits(Split { size: 4, rank: 4 }, split(1))
                .is_err()
        );
    }
}
