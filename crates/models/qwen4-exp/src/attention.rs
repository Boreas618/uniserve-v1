//! Gated attention over the tokens an indexer selects (vLLM's Qwen sparse attention,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/qsa.py:281-631`, with its indexer,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/qsa_indexer.py:90-448` and
//! `refs/vllm/vllm/models/qwen4_exp/common/qsa_indexer.py`).
//!
//! ```text
//! projections   [q | gate] per head, k, v;   q, k = rope(gemma(·))        over a head's channels
//! cache         k, v of every row are written to the layer's pages
//! indexer       [q_idx (heads) | k_raw] = index_qk_proj(x)
//!               q_idx = rope(gemma(q_idx; q_layernorm))
//!               key[b] = rope(gemma(mean(k_raw of the group's tokens); k_layernorm))
//!                        at the rotary position of the group's first token, when a row
//!                        completes group b of `ratio` tokens
//!               s[n]  = Σ_h relu(q_idx[h] · key[n])          for the groups complete before
//!                                                            the row's token is counted in
//!               sel   = the tokens of the best `budget / ratio` groups, then the tokens of
//!                       the row's own unfinished group
//! attention     o = Σ_{n ∈ sel} softmax(q · k_n · head_dim^-½) · v_n
//! output        o_proj(o ⊙ sigmoid(gate))
//! ```
//!
//! ## What a request keeps
//!
//! Beside the keys and values, a layer keeps the compressed **index keys**, one per group, on
//! the cache's pages ([`nn::KvPlanes::index_keys`]), and a **ring** of the request's most
//! recent raw index keys with their rotary positions, from which a group whose first tokens
//! an earlier forward saw is pooled. The ring holds `round_up(ratio + drafts, ratio)` rows at
//! `position mod capacity` (`common/qsa_indexer.py:817-866`): a step's draft rows never
//! overwrite a member of the group still open at the last committed token. Nothing is rolled
//! back after a verification: the rows of rejected positions are overwritten when those
//! positions are computed again, and a compressed key a rejected draft completed is not
//! visible before its group completes again.
//!
//! ## Selection layout
//!
//! A row's selection is `budget + ratio − 1` positions: the tokens of its groups in rank
//! order, then its unfinished group's, with `−1` where a row has fewer. vLLM compacts the
//! valid entries and appends their count (`nvidia/ops/qsa_indexer.py:236-277`); the entries
//! are the same.

use nn::family::StateField;
use nn::functional::Activation;
use nn::modules::{
    Linear, LinearConfig, Parallel, QkvBias, QkvLinear, QkvLinearConfig, QkvNames, Rotary,
};
use nn::{
    DType, Device, KvCache, RecurrentState, Split, Tensor, TokenBatch, VarBuilder, functional,
};

use crate::causal_lm::{LEADING, POSITION, ROW, SLOT, TRAILING};
use crate::config::{Config, IndexerConfig};
use crate::ngram::int;
use crate::{Error, Result};

/// An attention layer with its indexer, its modules named as the checkpoint names them.
pub struct IndexedAttention {
    /// `q_proj` (each head's query rows, then its gate's), `k_proj` and `v_proj`.
    qkv: QkvLinear,
    o_proj: Linear,
    /// `[head_dim]` each: the zero-centred gains of the query and key norms.
    q_norm: Tensor,
    k_norm: Tensor,
    /// `indexer.index_qk_proj`: the index queries of every index head, then the raw index key.
    index_qk_proj: Linear,
    /// `[index head_dim]` each: the zero-centred gains of the index query and key norms.
    q_layernorm: Tensor,
    k_layernorm: Tensor,
    indexer: IndexerConfig,
    /// Query heads and key/value heads this rank holds.
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    /// The scale of the attention scores.
    scale: f32,
}

impl IndexedAttention {
    /// The attention at `vb` (`<layer>.self_attn`), at `split`'s rank of the tensor-parallel
    /// group: a share of the query heads with the key/value head they read (a group wider
    /// than the key/value heads holds each on several ranks) and of `o_proj`'s columns; the
    /// indexer is replicated.
    ///
    /// # Errors
    ///
    /// A split that does not divide the query heads, or a module the loader refuses.
    pub fn load(config: &Config, vb: VarBuilder<'_, '_>, split: Split) -> Result<Self> {
        let (text, indexer) = (&config.text, config.indexer);
        let whole = split.size == 1;

        let qkv = QkvLinear::load(
            &QkvLinearConfig {
                hidden: text.hidden_size,
                heads: text.num_attention_heads,
                kv_heads: text.num_key_value_heads,
                head_dim: text.head_dim,
                // Every query head projects a gate beside it: the configuration has no flag
                // for it and `q_proj` holds both.
                gated: true,
                names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
                bias: QkvBias::Absent,
                split,
            },
            vb.clone(),
        )?;
        let o_proj = Linear::load(
            &LinearConfig {
                inputs: text.num_attention_heads * text.head_dim,
                outputs: text.hidden_size,
                bias: false,
                parallel: if whole {
                    Parallel::None
                } else {
                    Parallel::Row(split)
                },
            },
            vb.pp("o_proj"),
        )?;

        let index = vb.pp("indexer");
        let index_qk_proj = Linear::load(
            &LinearConfig {
                inputs: text.hidden_size,
                outputs: (indexer.heads + 1) * indexer.head_dim,
                bias: false,
                parallel: Parallel::None,
            },
            index.pp("index_qk_proj"),
        )?;
        let (head_dim, index_dim) = (text.head_dim as usize, indexer.head_dim as usize);

        Ok(Self {
            qkv,
            o_proj,
            q_norm: vb.pp("q_norm").get(head_dim, "weight")?,
            k_norm: vb.pp("k_norm").get(head_dim, "weight")?,
            index_qk_proj,
            q_layernorm: index.pp("q_layernorm").get(index_dim, "weight")?,
            k_layernorm: index.pp("k_layernorm").get(index_dim, "weight")?,
            indexer,
            heads: (text.num_attention_heads / split.size) as usize,
            kv_heads: (text.num_key_value_heads / split.size).max(1) as usize,
            head_dim,
            eps: text.rms_norm_eps,
            scale: text.rope.score_scale(text.head_dim)?,
        })
    }

    /// The state a layer keeps per request slot, in the order its buffers are lent: the
    /// ring of raw index keys in the activations' type `dtype`, then the rotary coordinates
    /// of the ring's rows.
    pub fn state_fields(config: &Config, dtype: DType, draft_tokens: u32) -> Vec<StateField> {
        let capacity = config.indexer.ring_capacity(draft_tokens) as usize;

        vec![
            StateField {
                name: "index_ring",
                dtype,
                dims: vec![capacity, config.indexer.head_dim as usize],
            },
            StateField {
                name: "index_ring_positions",
                dtype: DType::I32,
                dims: vec![capacity, qwen3::config::POSITION_AXES as usize],
            },
        ]
    }

    /// `hidden`, `[tokens, hidden]`, attended: `[tokens, hidden]` in its element type, summed
    /// over the tensor-parallel group.
    ///
    /// `coordinates` is `[axes, tokens]` I32, each token's rotary coordinates, which `rotary`
    /// turns into the factors `rope` (`(cos, sin)` of this forward's tokens) and into those of
    /// a group's first token. `columns` are the per-token columns of the forward
    /// ([`crate::causal_lm::token_columns`]). The layer writes its keys, values and index keys
    /// into layer `layer` of `cache` and reads and advances its ring, recurrent layer `ring`
    /// of `state`, in place.
    ///
    /// # Errors
    ///
    /// A cache layer without index keys or of another group size, a state whose layer is not
    /// a ring, or what a module or a function refuses.
    #[expect(
        clippy::too_many_arguments,
        reason = "the tensors an attention layer reads and the memory execution lends it"
    )]
    pub fn forward(
        &self,
        hidden: &Tensor,
        rope: (&Tensor, &Tensor),
        rotary: &Rotary,
        coordinates: &Tensor,
        columns: &[Tensor],
        batch: &TokenBatch,
        cache: &KvCache,
        layer: usize,
        state: &RecurrentState,
        ring: usize,
    ) -> Result<Tensor> {
        let (tokens, _) = hidden.dims2()?;
        let (heads, kv_heads, head_dim) = (self.heads, self.kv_heads, self.head_dim);

        // 1. The projections, as heads; a head's query and gate sit side by side in its row.
        let projected = self.qkv.query().forward(hidden)?;
        let pairs = projected.reshape((tokens * heads, 2 * head_dim))?;
        let query = pairs
            .narrow(1, 0, head_dim)?
            .contiguous()?
            .reshape((tokens, heads, head_dim))?;
        let gate = pairs.narrow(1, head_dim, head_dim)?;
        let per_head = (tokens, kv_heads, head_dim);
        let key = self.qkv.key().forward(hidden)?.reshape(per_head)?;
        let value = self.qkv.value().forward(hidden)?.reshape(per_head)?;

        // 2. Each head normalized and turned, and the keys and values of every row stored.
        let (query, key) = functional::qk_norm_rope(
            &query,
            &key,
            Some((&self.q_norm, &self.k_norm)),
            rope.0,
            rope.1,
            self.eps,
            true,
        )?;
        functional::kv_write_(cache, layer, batch, &key, &value)?;

        // 3. The tokens each row attends, and the attention over them.
        let selection = self.select(
            hidden,
            rope,
            rotary,
            coordinates,
            columns,
            batch,
            cache,
            layer,
            state,
            ring,
        )?;
        let attended = functional::indexed_attention(
            &query,
            cache,
            layer,
            &columns[ROW],
            &selection,
            self.scale,
        )?;

        // 4. Each attended head scaled by the sigmoid of its own gate, and projected out.
        let gated = functional::act_and_mul(
            &gate,
            &attended.reshape((tokens * heads, head_dim))?,
            Activation::Sigmoid,
            false,
        )?;
        Ok(self
            .o_proj
            .forward(&gated.reshape((tokens, heads * head_dim))?)?)
    }

    /// The indexer: stores the compressed key of every group this forward's rows complete,
    /// advances the ring, and selects the positions each row attends: `[tokens, budget +
    /// ratio − 1]` I32, `−1` where a row names no token (module documentation).
    #[expect(
        clippy::too_many_arguments,
        reason = "the tensors the indexer reads and the memory execution lends it"
    )]
    fn select(
        &self,
        hidden: &Tensor,
        rope: (&Tensor, &Tensor),
        rotary: &Rotary,
        coordinates: &Tensor,
        columns: &[Tensor],
        batch: &TokenBatch,
        cache: &KvCache,
        layer: usize,
        state: &RecurrentState,
        ring: usize,
    ) -> Result<Tensor> {
        let (tokens, _) = hidden.dims2()?;
        let dtype = hidden.dtype();
        let (index_heads, width) = (self.indexer.heads as usize, self.indexer.head_dim as usize);
        let ratio = self.indexer.compress_ratio as usize;
        let device = hidden.device().clone();

        // The layer's index keys and ring, as rows.
        let plane = cache
            .layer(layer)?
            .index_keys
            .as_ref()
            .ok_or_else(|| Error::Config(format!("cache layer {layer} holds no index keys")))?;
        let (&[pages, groups, key_width], &[_, max_pages]) =
            (plane.dims(), cache.block_tables.dims())
        else {
            return Err(Error::Config(format!(
                "index keys of {:?} under a page table of {:?}",
                plane.dims(),
                cache.block_tables.dims()
            )));
        };
        if key_width != width || groups * ratio != cache.page_size as usize {
            return Err(Error::Config(format!(
                "index keys of {:?} for pages of {} tokens in groups of {ratio} keys of {width}",
                plane.dims(),
                cache.page_size
            )));
        }
        let fields = state.live.get(ring).map_or(&[][..], Vec::as_slice);
        let [ring_keys, ring_positions] = fields else {
            return Err(Error::Config(format!(
                "recurrent layer {ring} holds {} fields, a ring's two expected",
                fields.len()
            )));
        };
        let (&[slots, capacity, _], axes) = (ring_keys.dims(), coordinates.dims()[0]) else {
            return Err(Error::Config(format!(
                "a ring of {:?} is not [slots, capacity, width]",
                ring_keys.dims()
            )));
        };
        let ring_keys = ring_keys.reshape((slots * capacity, width))?;
        let ring_positions = ring_positions.reshape((slots * capacity, axes))?;

        // Integer columns of the forward, I32: the token's index in the batch, its logical
        // position, its request slot (negative for none) and its place in its state row.
        let (position, slot) = (&columns[POSITION], &columns[SLOT]);
        let token = Tensor::arange(0_i32, tokens as i32, &Device::Cpu)?.to_device(&device)?;
        let constant = |value: usize| int(value as i64, position);
        let safe_slot = slot.maximum(0.0)?;

        // The index queries of every head and the row's raw index key.
        let projected = self.index_qk_proj.forward(hidden)?;
        let raw_key = projected
            .narrow(1, index_heads * width, width)?
            .contiguous()?;
        let index_query = functional::rms_norm(
            &projected
                .narrow(1, 0, index_heads * width)?
                .contiguous()?
                .reshape((tokens * index_heads, width))?,
            &self.q_layernorm,
            self.eps,
            true,
            dtype,
        )?;
        let index_query = functional::rotary(
            &index_query.reshape((tokens, index_heads, width))?,
            rope.0,
            rope.1,
        )?;

        // The row of the token `back` tokens before each token, among this forward's raw
        // keys followed by the ring's rows: this forward's when the request's state row
        // reaches that far back, the ring's otherwise.
        let ring_capacity = constant(capacity)?;
        let member = |back: usize| -> Result<Tensor> {
            let back = constant(back)?;
            let earlier = position.broadcast_sub(&back)?.maximum(0.0)?;
            let in_ring = safe_slot
                .broadcast_mul(&ring_capacity)?
                .add(&earlier.broadcast_remainder(&ring_capacity)?)?
                .broadcast_add(&constant(tokens)?)?;
            let in_forward = token.broadcast_sub(&back)?.maximum(0.0)?;

            Ok(columns[LEADING]
                .broadcast_ge(&back)?
                .where_cond(&in_forward, &in_ring)?
                .to_dtype(DType::U32)?)
        };

        // The mean of each group's raw keys, in FP32, for every row; only a row that
        // completes a group stores it.
        let recent = Tensor::cat(&[&raw_key, &ring_keys], 0)?;
        let mut pooled = recent.index_select(&member(0)?, 0)?.to_dtype(DType::F32)?;
        for back in 1..ratio {
            pooled = pooled.add(
                &recent
                    .index_select(&member(back)?, 0)?
                    .to_dtype(DType::F32)?,
            )?;
        }
        let pooled = pooled.affine(1.0 / ratio as f64, 0.0)?.to_dtype(dtype)?;

        // The compressed key turns by the rotary position of its group's first token.
        let own = coordinates.t()?.contiguous()?;
        let first = Tensor::cat(&[&own, &ring_positions], 0)?
            .index_select(&member(ratio - 1)?, 0)?
            .t()?
            .contiguous()?;
        let (cos, sin) = rotary.forward(&first)?;
        let compressed = functional::rotary(
            &functional::rms_norm(&pooled, &self.k_layernorm, self.eps, true, dtype)?
                .reshape((tokens, 1, width))?,
            &cos,
            &sin,
        )?
        .reshape((tokens, width))?;

        // A row whose position ends a group stores the group's key on the row's page; a
        // padding token stores nothing.
        let ratio_i = constant(ratio)?;
        let next = position.broadcast_add(&constant(1)?)?;
        let visible = next.broadcast_div(&ratio_i)?;
        let partial = next.broadcast_remainder(&ratio_i)?;
        let group = position.broadcast_div(&ratio_i)?;
        let groups_i = constant(groups)?;

        let table_rows = cache.block_tables.dims()[0];
        let entry = columns[ROW]
            .to_dtype(DType::I32)?
            .broadcast_mul(&constant(max_pages)?)?
            .add(
                &group
                    .broadcast_div(&groups_i)?
                    .minimum((max_pages - 1) as f64)?,
            )?
            .minimum((table_rows * max_pages - 1) as f64)?
            .to_dtype(DType::U32)?;
        let page = cache
            .block_tables
            .flatten_all()?
            .index_select(&entry, 0)?
            .to_dtype(DType::I32)?;
        let stored_at = page
            .broadcast_mul(&groups_i)?
            .add(&group.broadcast_remainder(&groups_i)?)?;
        let written = batch.slot_mapping.view(DType::I32)?.ge(0.0)?;
        let none = int(-1, position)?.broadcast_as(position.shape())?;
        let stored_at = partial
            .eq(0.0)?
            .mul(&written)?
            .where_cond(&stored_at, &none)?;
        functional::place_rows_(
            &plane.reshape((pages * groups, width))?,
            &stored_at,
            &compressed,
        )?;

        // The ring takes the request's last `capacity` rows of this forward, after the
        // compression has read it.
        let ring_row = safe_slot
            .broadcast_mul(&ring_capacity)?
            .add(&position.broadcast_remainder(&ring_capacity)?)?;
        let kept = slot
            .ge(0.0)?
            .mul(&columns[TRAILING].broadcast_lt(&ring_capacity)?)?;
        let ring_row = kept.where_cond(&ring_row, &none)?;
        functional::place_rows_(&ring_keys, &ring_row, &raw_key)?;
        functional::place_rows_(&ring_positions, &ring_row, &own)?;

        // The scores of the groups complete before each row, and the best of them.
        let scores = functional::index_scores(
            &index_query,
            cache,
            layer,
            &columns[ROW],
            &visible.to_dtype(DType::U32)?,
        )?;
        let block_topk = self.indexer.block_topk() as usize;
        let kept_groups = block_topk.min(scores.dims()[1]);
        // The ids are below `i32::MAX`, so their U32 bits read as the I32 the columns below are.
        let (_, best) = functional::top_k(&scores, kept_groups as u32)?;
        let best = best.view(DType::I32)?;

        // The tokens of the selected groups, in rank order; a row that sees fewer groups
        // than it may keep names none past them.
        let host = |values: Vec<i32>, shape: (usize, usize, usize)| -> Result<Tensor> {
            Ok(Tensor::from_vec(values, shape, &Device::Cpu)?.to_device(&device)?)
        };
        let rank = host((0..kept_groups as i32).collect(), (1, kept_groups, 1))?;
        let member_of = host((0..ratio as i32).collect(), (1, 1, ratio))?;
        let selected = visible
            .minimum(kept_groups as f64)?
            .reshape((tokens, 1, 1))?;
        let named = rank
            .broadcast_lt(&selected)?
            .broadcast_as((tokens, kept_groups, ratio))?;
        let group_tokens = best
            .unsqueeze(2)?
            .broadcast_mul(&ratio_i.reshape((1, 1, 1))?)?
            .broadcast_add(&member_of)?;
        let nothing = none.reshape((tokens, 1, 1))?;
        let group_tokens = named
            .where_cond(
                &group_tokens,
                &nothing.broadcast_as((tokens, kept_groups, ratio))?,
            )?
            .reshape((tokens, kept_groups * ratio))?;

        // The row's own unfinished group: the positions from its first token to the row's.
        let tail =
            host((0..ratio as i32 - 1).collect(), (1, ratio - 1, 1))?.reshape((1, ratio - 1))?;
        let tail_tokens = visible
            .broadcast_mul(&ratio_i)?
            .unsqueeze(1)?
            .broadcast_add(&tail)?;
        let tail_tokens = tail.broadcast_lt(&partial.unsqueeze(1)?)?.where_cond(
            &tail_tokens,
            &none.unsqueeze(1)?.broadcast_as((tokens, ratio - 1))?,
        )?;

        // A cache of fewer groups than a row may keep leaves the columns past them empty.
        let unfilled = (block_topk - kept_groups) * ratio;
        let mut parts = vec![group_tokens];
        if unfilled > 0 {
            parts.push(none.unsqueeze(1)?.broadcast_as((tokens, unfilled))?);
        }
        parts.push(tail_tokens);

        // The rows side by side, as rows of the selection's width whatever their count: the
        // attention addresses a row from the tensor's first element.
        let width = self.indexer.selection() as usize;
        Ok(Tensor::cat(&parts, 1)?
            .contiguous()?
            .reshape((tokens, width))?)
    }
}

impl std::fmt::Debug for IndexedAttention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IndexedAttention")
            .field("heads", &self.heads)
            .field("kv_heads", &self.kv_heads)
            .field("head_dim", &self.head_dim)
            .field("indexer", &self.indexer)
            .finish_non_exhaustive()
    }
}
