//! The DSpark drafter of DeepSeek-V4.1 (`inference/model.py:1023-1160`, the checkpoint's
//! `mtp.*`): three hyper-connected layers after the decoder's that draft a block of tokens in
//! one forward, from the decoder's residual and a Markov term chained over the block.
//!
//! ```text
//! context:  main_x = main_norm(main_proj(‖_l mean_j h_l[j]))     l ∈ dspark_target_layer_ids,
//!                                                                [T, 3d] → [T, d] (MXFP8)
//!           window_i[p] = kv_norm_i(wkv_i(main_x[p]))           every draft layer i, rotated at
//!                                                               p and encoded as the decoder's
//! block:    x = embed([anchor, noise, …, noise])                 positions P .. P + K − 1
//!           h = [x, x, x, x]; layers 40, 41, 42 over h           each: mHC attention, mHC experts
//!           attention: each query of the block reads the window entries P − W .. P + K − 1,
//!                      the block's own written first: non-causal within the block
//!           rows = norm(Σ_j pre_j · h_j)                         the last connection's pre-mix
//! drafts:   d_0 = argmax(head(rows_0) + markov(anchor))          markov(t) = head_m · embed_m(t)
//!           d_j = argmax(head(rows_j) + markov(d_{j−1}))         rank 256, FP32
//! ```
//!
//! The context is the reference's `main_kv` of each step's tokens (`DSparkAttention.forward`):
//! a draft layer's window holds the latents it projects from the decoder's rows, at the
//! positions of those rows, and its block attends them beside its own (`get_dspark_topk_idxs`:
//! the last `W` context entries and every block position). The block's entries are written
//! into the window at the block's positions before any query reads them, and a later step's
//! context overwrites them; a query reads the entries from `W + K − 1` before the block's last
//! position to it, which is the window anchored at the block's end
//! ([`nn::functional::latent_attention`] with a window of `W + K` at that position).
//!
//! The draft layers attend their window alone (`compress_ratios` states them zero) and rotate
//! by the window-only layers' power series over `rope_theta`. Their routed experts are held
//! whole on every rank of an expert union: a forward routes the `K` block tokens of a rank's own
//! rows, which a rank computes faster alone than through two exchanges a layer, and no member
//! takes part in another's draft.
//!
//! The confidence head (`mtp.<last>.confidence_head`) scores the reference's adaptive
//! verification, which no draft here reads: it is not loaded.

use nn::collective::Split;
use nn::family::TextPlacement;
use nn::functional::{self, Plane};
use nn::modules::{
    Embedding, EmbeddingConfig, LatentAttention, LatentAttentionConfig, LatentBatch, Linear,
    LinearConfig, LmHead, LmHeadConfig, MhcConfig, MhcConnection, MhcResidual, MoE, Parallel,
    Rotary, SharedExpert,
};
use nn::{
    BlockDrafter, DType, Device, DraftBlock, DraftContext, ExpertExchange, KvCache, Result, Tensor,
    VarBuilder,
};

use crate::deepseek_v41::ID;
use crate::deepseek_v41::attention::plane;
use crate::deepseek_v41::cache::{WINDOW, WINDOW_RECORD};
use crate::deepseek_v41::causal_lm::written;
use crate::deepseek_v41::config::{Config, DsparkConfig};
use crate::deepseek_v41::decoder::{routed, shared};

/// The profiler class of the drafter's forward.
const DRAFTER: &str = "models.deepseek_v41.dspark.Dspark";

/// The module path the drafter's experts are named by: `mtp.<stage>.ffn.experts`.
pub fn expert_module(stage: u32) -> String {
    format!("mtp.{stage}.ffn.experts")
}

/// The window plane of layer `layer` in `cache`, the window group, held in `records`.
fn window_plane<'t>(
    cache: &'t KvCache,
    layer: u32,
    records: &'t mut Option<Tensor>,
) -> Result<Plane<'t>> {
    plane(cache, (WINDOW, WINDOW_RECORD), layer, records)
}

/// The window group among the model's cache groups: the full-history group, then the window's.
fn window_group<'t>(cache: &'t [KvCache], what: &str) -> Result<&'t KvCache> {
    match cache {
        [_, window] => Ok(window),
        _ => Err(ID.config(format!(
            "the drafter's {what} reads the full-history group and the window's, and was lent {} \
             groups",
            cache.len()
        ))),
    }
}

/// A draft layer's attention (`DSparkAttention`): the decoder's projections and latent
/// attention, over a window that also holds the layer's context.
struct DraftAttention {
    heads: u32,
    head_dim: u32,
    q_lora_rank: u32,
    eps: f32,
    /// `[wq_a; wkv]`: the query latent and the window latent of each row, one product.
    wq_a_wkv: Linear,
    /// `[q_lora]`: the query latent's norm gain.
    q_norm: Tensor,
    wq_b: Linear,
    /// The attention over a window of `W + K` entries, anchored at each block's last position.
    attention: LatentAttention,
    /// `wo_a`'s groups: group `g` projects heads `g · heads / groups ..` to `o_lora`.
    wo_a: Vec<Linear>,
    wo_b: Linear,
}

impl DraftAttention {
    /// The attention at `vb` (`mtp.<stage>.attn`), whole on this rank, its window `window`
    /// entries wide.
    fn load(config: &Config, window: u32, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let attention = &config.attention;
        let (hidden, heads, head_dim) = (config.hidden_size, attention.heads, attention.head_dim);
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

        // `wo_a` is `[groups · o_lora, heads · head_dim / groups]`: group `g`'s rows are its
        // block of the output channels, as the decoder's (`crate::deepseek_v41::attention`).
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

        Ok(Self {
            heads,
            head_dim,
            q_lora_rank: attention.q_lora_rank,
            eps: config.rms_norm_eps,
            wq_a_wkv: Linear::load_merged(
                vb.clone(),
                &["wq_a", "wkv"],
                hidden,
                &[attention.q_lora_rank, head_dim],
                false,
            )?,
            q_norm: vb
                .pp("q_norm")
                .get(attention.q_lora_rank as usize, "weight")?,
            wq_b: linear(
                attention.q_lora_rank,
                heads * head_dim,
                Parallel::None,
                vb.pp("wq_b"),
            )?,
            attention: LatentAttention::load(
                &LatentAttentionConfig {
                    heads,
                    head_dim,
                    rotated: attention.rope_head_dim,
                    window,
                    eps: config.rms_norm_eps,
                    scale: (head_dim as f32).powf(-0.5),
                    split: Split::WHOLE,
                },
                vb.clone(),
            )?,
            wo_a,
            wo_b: linear(
                groups * attention.o_lora_rank,
                hidden,
                Parallel::None,
                vb.pp("wo_b"),
            )?,
        })
    }

    /// The window latent projection of `x`, `[tokens, head_dim]`, before its normalization, and
    /// the query latent, `[tokens, q_lora]`, from one product.
    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let projected = self.wq_a_wkv.forward(x)?;
        let rank = self.q_lora_rank as usize;

        let qr = projected.narrow(1, 0, rank)?.contiguous()?;
        let kv = projected
            .narrow(1, rank, self.head_dim as usize)?
            .contiguous()?;
        Ok((qr, kv))
    }

    /// Writes the context entries of `main_x`, `[tokens, hidden]` BF16, into `window`: each
    /// token's `kv_norm(wkv(main_x))` at the entry `batch.written` names.
    fn write_context(
        &self,
        main_x: &Tensor,
        window: Plane<'_>,
        batch: &LatentBatch<'_>,
    ) -> Result<()> {
        let (_, kv) = self.project(main_x)?;
        self.attention.write_(&kv, window, batch)
    }

    /// The attention of the block rows `x`, `[tokens, hidden]` BF16, the layer's normalized
    /// input: `[tokens, hidden]` BF16. The block's own entries are written into `window` first;
    /// `batch.positions` is each token's block's last position, which anchors its window.
    fn forward(&self, x: &Tensor, window: Plane<'_>, batch: &LatentBatch<'_>) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let (heads, head_dim) = (self.heads as usize, self.head_dim as usize);

        let (qr, kv) = self.project(x)?;
        let qr = functional::rms_norm(&qr, &self.q_norm, self.eps, false, DType::BF16)?;
        let query = self
            .wq_b
            .forward(&qr)?
            .reshape((tokens, heads, head_dim))?
            .contiguous()?;
        let attended = self.attention.forward((&query, &kv), window, None, batch)?;

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
    }
}

/// One draft layer (`DSparkBlock`): the decoder layer's form, its attention a draft attention
/// and its routed experts the drafter's.
struct DraftLayer {
    /// The layer's index after the decoder's: its window's owner in the cache.
    layer: u32,
    attention_connection: MhcConnection,
    attention: DraftAttention,
    ffn_connection: MhcConnection,
    experts: MoE,
    shared: SharedExpert,
}

impl DraftLayer {
    /// The draft layer `stage` at `vb` (`mtp.<stage>`), the model's layer `layer`, whose
    /// attention reads a window of `window` entries.
    fn load(
        config: &Config,
        dspark: &DsparkConfig,
        (stage, layer, window): (u32, u32, u32),
        placement: &TextPlacement,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let mhc: MhcConfig = config.mhc();
        // Whole on every rank: no exchange carries a draft token (module documentation).
        let experts = routed(
            config,
            &dspark.experts,
            expert_module(stage),
            None,
            (placement.tensor, placement.tensor_experts, Split::WHOLE),
        );

        Ok(Self {
            layer,
            attention_connection: MhcConnection::load(&mhc, &vb, "hc_attn", vb.pp("attn_norm"))?,
            attention: DraftAttention::load(config, window, vb.pp("attn"))?,
            ffn_connection: MhcConnection::load(&mhc, &vb, "hc_ffn", vb.pp("ffn_norm"))?,
            experts: MoE::load(&experts, vb.pp("ffn"))?,
            shared: SharedExpert::load(&shared(config, &dspark.experts), vb.pp("ffn"))?,
        })
    }

    /// The layer over the block's `residual`: its attention over `window`, then its experts, as
    /// sub-layers of the residual.
    fn forward(
        &self,
        residual: &mut MhcResidual,
        window: Plane<'_>,
        batch: &LatentBatch<'_>,
        live_tokens: Option<&Tensor>,
    ) -> Result<()> {
        residual.sublayer(&self.attention_connection, |x| {
            self.attention.forward(x, window, batch)
        })?;
        residual.sublayer(&self.ffn_connection, |x| {
            let routed = self.experts.forward(x, live_tokens, None)?;
            self.shared.forward(x, routed)
        })
    }
}

/// The DSpark drafter (module documentation).
pub struct Dspark {
    config: DsparkConfig,
    hidden: u32,
    eps: f32,
    mhc: MhcConfig,
    /// The projection of the decoder's rows read at the target layers, `[3d] → [d]`.
    main_proj: Linear,
    /// `[d]`: the projection's norm gain.
    main_norm: Tensor,
    /// The window-only layers' rotation, which the draft layers take.
    rope: Rotary,
    layers: Vec<DraftLayer>,
    /// `[d]`: the norm gain of the block's read-out (`mtp.<last>.norm`).
    norm: Tensor,
    /// The Markov term's token embedding, `[vocabulary, rank]` BF16.
    markov_embed: Embedding,
    /// The Markov term's head, `[vocabulary, rank]` widened to FP32, as the reference's
    /// `ParallelHead` holds it.
    markov_head: LmHead,
}

impl Dspark {
    /// The drafter `config` states, read under `vb` (the checkpoint's root) at the ranks
    /// `placement` states.
    ///
    /// # Errors
    ///
    /// A checkpoint that carries no drafter, a target layer that holds an n-gram memory (the
    /// residual the drafter reads is the one after it, which a decoder layer's forward does not
    /// expose), or a weight the checkpoint lacks or holds otherwise.
    pub fn load(
        config: &Config,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
    ) -> Result<Self> {
        let dspark = config
            .dspark
            .clone()
            .ok_or_else(|| ID.config("the checkpoint carries no DSpark drafter".to_owned()))?;
        if let Some(layer) = dspark
            .target_layer_ids
            .iter()
            .find(|&&layer| config.has_engram(layer))
        {
            return Err(ID.config(format!(
                "the drafter reads the residual entering layer {layer}, which adds an n-gram \
                 memory first"
            )));
        }

        let hidden = config.hidden_size;
        let window = config.attention.window + dspark.block;
        let first = config.num_hidden_layers;
        let last = dspark.layers - 1;
        let mtp = vb.pp("mtp");

        let layers = (0..dspark.layers)
            .map(|stage| {
                DraftLayer::load(
                    config,
                    &dspark,
                    (stage, first + stage, window),
                    placement,
                    mtp.pp(stage),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let targets = dspark.target_layer_ids.len() as u32;
        let rank = dspark.markov_rank;
        let head = mtp.pp(last);

        Ok(Self {
            hidden,
            eps: config.rms_norm_eps,
            mhc: config.mhc(),
            main_proj: Linear::load(
                &LinearConfig {
                    inputs: targets * hidden,
                    outputs: hidden,
                    bias: false,
                    parallel: Parallel::None,
                },
                mtp.pp(0).pp("main_proj"),
            )?,
            main_norm: mtp.pp(0).pp("main_norm").get(hidden as usize, "weight")?,
            rope: Rotary::load(&config.attention.rope, mtp.pp("rotary"))?,
            layers,
            norm: head.pp("norm").get(hidden as usize, "weight")?,
            markov_embed: Embedding::load(
                &EmbeddingConfig {
                    vocabulary: config.vocab_size,
                    hidden: rank,
                    split: Split::WHOLE,
                },
                head.pp("markov_head").pp("embed"),
            )?,
            markov_head: LmHead::load(
                &LmHeadConfig {
                    vocabulary: config.vocab_size,
                    hidden: rank,
                    split: Split::WHOLE,
                },
                head.pp("markov_head").pp("head").to_dtype(DType::F32),
            )?,
            config: dspark,
        })
    }

    /// The configuration the drafter was loaded with.
    pub const fn config(&self) -> &DsparkConfig {
        &self.config
    }

    /// The Markov term of `tokens` (`[rows]` U32), the bias the drafter adds to the head's
    /// logits of the block position after each token (`DSparkMarkovHead`): `head_m ·
    /// embed_m(token)`, `[rows, vocabulary]` FP32.
    ///
    /// # Errors
    ///
    /// Tokens of another extent or type, or what a module refuses.
    pub fn markov(&self, tokens: &Tensor) -> Result<Tensor> {
        self.markov_head
            .forward(&self.markov_embed.forward(tokens)?.to_dtype(DType::F32)?)
    }

    /// The columns a forward reads at the window group `window`: `positions`, the positions
    /// that anchor the tokens' windows; `written` and `rows`, the entries the tokens write and
    /// their rows of the tables; and the tokens' rotary factors.
    fn latent<'t>(
        &'t self,
        positions: &'t Tensor,
        (written, rows): (&'t Tensor, &'t Tensor),
        window: &'t KvCache,
        (cos, sin): (&'t Tensor, &'t Tensor),
    ) -> Result<LatentBatch<'t>> {
        let starts = window
            .starts
            .as_ref()
            .ok_or_else(|| ID.config("the window's group states no rows' starts".to_owned()))?;

        Ok(LatentBatch {
            positions,
            written,
            rows,
            window_starts: starts,
            cos,
            sin,
            inv_freq: self.rope.inverse_frequencies(),
        })
    }

    /// Writes every draft layer's context entries (module documentation, context).
    fn write_context(&self, context: &DraftContext<'_>) -> Result<()> {
        let width = self.hidden as usize;
        let targets = self.config.target_layer_ids.len();
        let window = window_group(context.cache, "context")?;
        let batch = context.batch;
        if context.hidden.dims() != [batch.tokens(), (1 + targets) * width] {
            return Err(ID.config(format!(
                "the context's rows are {:?}, [{}, {}] expected: the read-out, then the residual \
                 at each target layer",
                context.hidden.dims(),
                batch.tokens(),
                (1 + targets) * width
            )));
        }

        // `main_x`: the target layers' rows, projected and normalized (`forward_embed`).
        let read = context
            .hidden
            .narrow(1, width, targets * width)?
            .contiguous()?;
        let main_x = functional::rms_norm(
            &self.main_proj.forward(&read)?,
            &self.main_norm,
            self.eps,
            false,
            DType::BF16,
        )?;

        let (rows, _) = batch.token_rows()?;
        let entries = written(&window.slot_mapping, &batch.positions)?;
        let (cos, sin) = self.rope.forward(&batch.positions)?;
        let latent = self.latent(&batch.positions, (&entries, &rows), window, (&cos, &sin))?;

        for layer in &self.layers {
            let mut held = None;
            let plane = window_plane(window, layer.layer, &mut held)?;
            layer.attention.write_context(&main_x, plane, &latent)?;
        }
        Ok(())
    }
}

impl BlockDrafter for Dspark {
    fn block(&self) -> u32 {
        self.config.block
    }

    fn fill(&self) -> u32 {
        self.config.noise_token_id
    }

    fn forward(
        &self,
        context: &DraftContext<'_>,
        block: &DraftBlock<'_>,
        exchange: Option<&ExpertExchange>,
    ) -> Result<Tensor> {
        // The draft layers' experts are whole on every rank: no forward of theirs is an expert
        // step of the union (module documentation).
        let _ = exchange;

        let forward = || -> Result<Tensor> {
            self.write_context(context)?;

            let batch = block.batch;
            let tokens = batch.tokens();
            let size = self.config.block as usize;
            if !tokens.is_multiple_of(size)
                || block.embedded.dims() != [tokens, self.hidden as usize]
            {
                return Err(ID.config(format!(
                    "blocks of {tokens} tokens and embeddings of {:?}: whole blocks of {size} \
                     rows of {}",
                    block.embedded.dims(),
                    self.hidden
                )));
            }
            let window = window_group(block.cache, "block")?;

            // A block's token `j` lies `size − 1 − j` before the block's last position, which
            // anchors its window: every query of a block reads the same entries.
            let ahead: Vec<i32> = (0..tokens)
                .map(|token| (size - 1 - token % size) as i32)
                .collect();
            let anchors = batch.positions.add(
                &Tensor::from_vec(ahead, tokens, &Device::Cpu)?
                    .to_device(batch.positions.device())?,
            )?;

            let (rows, _) = batch.token_rows()?;
            let entries = written(&window.slot_mapping, &batch.positions)?;
            let (cos, sin) = self.rope.forward(&batch.positions)?;
            let latent = self.latent(&anchors, (&entries, &rows), window, (&cos, &sin))?;

            let mut residual = MhcResidual::expand(&self.mhc, block.embedded)?;
            for layer in &self.layers {
                let mut held = None;
                let plane = window_plane(window, layer.layer, &mut held)?;
                layer.forward(&mut residual, plane, &latent, batch.live_tokens.as_ref())?;
            }
            residual.collapse(&self.norm, self.eps)
        };

        nn::profiler::range(DRAFTER, "mtp", forward)
    }

    /// The Markov chain over each block (`forward_head`): draft `j` is the argmax of the head's
    /// logits at token `j` plus the Markov term of the token before it, the anchor for the first.
    fn propose(&self, logits: &Tensor, anchors: &Tensor) -> Result<Tensor> {
        let size = self.config.block as usize;
        let rows = anchors.dim(0)?;
        let (count, vocabulary) = logits.dims2()?;
        if count != rows * size || logits.dtype() != DType::F32 {
            return Err(ID.config(format!(
                "{:?} logits of [{count}, {vocabulary}] for {rows} blocks of {size}: FP32 rows \
                 of every block token expected",
                logits.dtype()
            )));
        }
        let logits = logits.reshape((rows, size, vocabulary))?;

        let mut previous = anchors.clone();
        let mut drafts = Vec::with_capacity(size);
        for offset in 0..size {
            let term = self.markov(&previous)?;
            let shaped = logits.narrow(1, offset, 1)?.squeeze(1)?.add(&term)?;
            let draft = functional::argmax(&shaped)?;

            drafts.push(draft.unsqueeze(1)?);
            previous = draft;
        }
        Ok(Tensor::cat(&drafts, 1)?)
    }

    /// The draft layers' experts are whole on every rank, so a member with no tokens of its own
    /// runs none of them.
    fn join_expert_step(&self, exchange: &ExpertExchange) -> Result<()> {
        let _ = exchange;
        Ok(())
    }
}

impl std::fmt::Debug for Dspark {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Dspark")
            .field("layers", &self.layers.len())
            .field("block", &self.config.block)
            .finish_non_exhaustive()
    }
}
