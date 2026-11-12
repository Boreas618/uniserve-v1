//! The n-gram hash embedding and the layer that gates it into the residual streams.
//!
//! [`NgramEmbedding`] (vLLM's `Qwen4ExpNgramEmbedding`,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/ngram_embedding.py`, ids
//! `refs/vllm/vllm/models/qwen4_exp/common/ops/ple.py:24-168`) hashes each token with the one
//! and the two before it into rows of one large table:
//!
//! ```text
//! p1' = p1;   p2' = eos when p1 = eos, else p2          an n-gram stops at an end of text
//! mix2 = (t0 · m0) xor (p1' · m1)                       64-bit, wrapping
//! mix3 = mix2 xor (p2' · m2)
//! id_h = (mix2 mod P_h) + O_h     for the bigram heads h
//! id_h = (mix3 mod P_h) + O_h     for the trigram heads h          mod is non-negative
//! e    = [table[id_0] | table[id_1] | …] · weight_scale
//! ```
//!
//! The table is E4M3 with one scale: a row is read as its E4M3 values, exactly, in the
//! activations' type, and multiplied by the scale rounded to that type, the product rounded
//! once (`refs/vllm/vllm/models/qwen4_exp/common/ngram_embedding.py:390-402`).
//!
//! [`PleLayer`] (vLLM's `Qwen4ExpPLELayer`,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/ple_layer.py:74-484`, kernels
//! `refs/vllm/vllm/models/qwen4_exp/common/ops/ple.py:185-537`) projects the embedding to a key
//! per stream and one value, gates the value by how each stream's key agrees with the stream,
//! and adds the gated value and a dilated causal convolution of it:
//!
//! ```text
//! key   = e · key_projᵀ      [tokens, C, H]          value = e · value_projᵀ   [tokens, H]
//! kn_c  = gemma(key_c; norm_key_c)                   qn_c  = gemma(S_c; norm_query_c)
//! d_c   = (Σ_j kn_c[j] · qn_c[j]) / √H
//! g_c   = sigmoid(sign(d_c) · √max(|d_c|, 1e-6))
//! gv_c  = g_c · value                                gvn_c = gemma(gv_c; norm_conv_c)
//! conv  = Σ_k w[·, k] · gvn[t − (K − 1 − k) · dilation]         over the request's tokens
//! out   = gv + silu(conv)
//! ```
//!
//! with `gemma(x; w) = x · rsqrt(mean(x²) + eps) · (1 + w)`. Every named value is rounded once
//! into the activations' type, and so are the products `kn · qn` before they are summed, the
//! sum, `d`, the square root and `silu(conv)`, as the kernel materializes them
//! (`ple.py:212-239`).

use nn::family::StateField;
use nn::modules::Linear;
use nn::{
    DType, Device, RecurrentState, Shape, Split, Tensor, TokenBatch, VarBuilder, WeightLayout,
    WeightSource, functional,
};

use crate::config::Config;
use crate::{Error, Result};

/// The value of an E4M3 byte (`F8_E4M3`: a sign, four exponent bits biased by seven, three
/// mantissa bits, no infinities, `S.1111.111` not a number).
fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let (exponent, mantissa) = (i32::from((byte >> 3) & 0x0f), f32::from(byte & 0x07));

    match (exponent, byte & 0x07) {
        (15, 7) => f32::NAN,
        // A subnormal has no implicit one and the smallest exponent.
        (0, _) => sign * (mantissa / 8.0) * 2f32.powi(-6),
        _ => sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7),
    }
}

/// The checkpoint tensor `name` of `vb`'s module, in the element type `dtype` its shard
/// stores it in: an integer table or an E4M3 table is read as its own elements, whatever the
/// loader's policy converts floating weights to.
fn stored(vb: &VarBuilder<'_, '_>, name: &str, shape: Shape, dtype: DType) -> Result<Tensor> {
    let tensor = vb.name(name);
    let source = WeightSource::Region {
        tensor: tensor.clone(),
        start: 0,
        shape: shape.clone(),
    };

    Ok(vb.loader().load(
        &tensor,
        &source,
        WeightLayout::ROW_MAJOR,
        &shape,
        Some(dtype),
        false,
    )?)
}

/// The n-gram hash embedding of one PLE layer.
pub struct NgramEmbedding {
    /// `[ngram_size]` I64: the odd multipliers, the row's own token's first.
    layer_multipliers: Tensor,
    /// `[heads]` I64: each head's partition size, a prime.
    ngram_heads_vocab_sizes: Tensor,
    /// `[heads]` I64: the first table row of each head's partition.
    ngram_heads_offsets: Tensor,
    /// `[rows, head_dim]` E4M3: this rank's rows of the table, whole shards of the checkpoint.
    ngram_embedding: Tensor,
    /// `[1]` in the activations' type: the table's one scale.
    weight_scale: Tensor,
    /// `[256]` in the activations' type: the value of each E4M3 byte.
    values: Tensor,
    /// `[1]` I64: the end-of-text token, which stops an n-gram.
    eos: Tensor,
    /// The first table row this rank holds, and how many it holds.
    held: (u64, u64),
    /// Heads of one n-gram length.
    heads_per_ngram: usize,
    /// Elements of one head's row.
    head_dim: usize,
    /// Whether the table is split over a tensor-parallel group, whose ranks sum their rows.
    sharded: bool,
}

impl NgramEmbedding {
    /// The embedding at `vb` (`<layer>.ple.ple_embedding`), at `split`'s rank of the
    /// tensor-parallel group: the three hash tables, the table's shards
    /// `ngram_embedding.shard_<i>.weight` read as one tensor, and its `weight_scale`.
    ///
    /// A rank holds `parts / size` consecutive shards, so every table row is on one rank and
    /// the ranks sum the rows they read
    /// (`refs/vllm/vllm/models/qwen4_exp/common/ngram_embedding.py:764-772`).
    ///
    /// # Errors
    ///
    /// A split that does not divide the shards, or a tensor the loader refuses.
    pub fn load(config: &Config, vb: VarBuilder<'_, '_>, split: Split) -> Result<Self> {
        let ngram = &config.ngram;
        let (heads, head_dim) = (ngram.heads() as usize, ngram.head_dim() as usize);
        if split.size == 0 || split.rank >= split.size || !ngram.parts.is_multiple_of(split.size) {
            return Err(Error::Config(format!(
                "rank {} of a tensor group of {} does not hold whole shards of {}",
                split.rank, split.size, ngram.parts
            )));
        }

        // The shards this rank holds, fused in order into its rows of the table.
        let shard_rows = ngram.table_rows() / u64::from(ngram.parts);
        let held_parts = ngram.parts / split.size;
        let first_part = split.rank * held_parts;
        let rows = u64::from(held_parts) * shard_rows;

        let table = vb.pp("ngram_embedding");
        let shape = Shape::from((rows as usize, head_dim));
        let source = WeightSource::Parts {
            parts: (first_part..first_part + held_parts)
                .map(|part| table.pp(format!("shard_{part}")).name("weight"))
                .collect(),
            shape: shape.clone(),
            block: None,
        };
        let ngram_embedding = vb.loader().load(
            &table.name("weight"),
            &source,
            WeightLayout::ROW_MAJOR,
            &shape,
            Some(DType::F8E4M3),
            false,
        )?;

        let weight_scale = table.get(1, "weight_scale")?;
        let values: Vec<f32> = (0..=u8::MAX).map(e4m3).collect();
        let eos = Tensor::new(&[i64::from(config.text.eos_token_id)], &Device::Cpu)?;

        Ok(Self {
            layer_multipliers: stored(
                &vb,
                "layer_multipliers",
                Shape::from(ngram.ngram_size as usize),
                DType::I64,
            )?,
            ngram_heads_vocab_sizes: stored(
                &vb,
                "ngram_heads_vocab_sizes",
                Shape::from(heads),
                DType::I64,
            )?,
            ngram_heads_offsets: stored(
                &vb,
                "ngram_heads_offsets",
                Shape::from(heads),
                DType::I64,
            )?,
            ngram_embedding,
            values: table.constant("e4m3_values", weight_scale.dtype(), 256, &values)?,
            weight_scale,
            eos: vb.constant_from("eos", &eos)?,
            held: (u64::from(first_part) * shard_rows, rows),
            heads_per_ngram: ngram.heads_per_ngram as usize,
            head_dim,
            sharded: split.size > 1,
        })
    }

    /// The table rows each token reads: `[tokens, heads]` I64, for `context`, `[tokens, 3]`
    /// U32 token ids `(p2, p1, t0)`: the token two before the row's, the one before it, and
    /// the row's own, in the request's token order (module documentation).
    ///
    /// # Errors
    ///
    /// A context of another shape.
    pub fn ids(&self, context: &Tensor) -> Result<Tensor> {
        let &[tokens, 3] = context.dims() else {
            return Err(Error::Config(format!(
                "an n-gram context of {:?} is not [tokens, 3]",
                context.dims()
            )));
        };

        let wide = context.to_dtype(DType::I64)?;
        let token = |column: usize| wide.narrow(1, column, 1);
        let multiplier = |position: usize| self.layer_multipliers.narrow(0, position, 1);
        let (older, previous, current) = (token(0)?, token(1)?, token(2)?);

        // An n-gram stops at an end of text: every token before one counts as one.
        let ended = previous.broadcast_eq(&self.eos)?;
        let older = ended.where_cond(&self.eos.broadcast_as(older.shape())?, &older)?;

        // Wrapping 64-bit products: a multiplier is small enough that none overflows.
        let bigram = current
            .broadcast_mul(&multiplier(0)?)?
            .bitwise_xor(&previous.broadcast_mul(&multiplier(1)?)?)?;
        let trigram = bigram.bitwise_xor(&older.broadcast_mul(&multiplier(2)?)?)?;

        let per = (tokens, self.heads_per_ngram);
        let hashes = Tensor::cat(&[bigram.broadcast_as(per)?, trigram.broadcast_as(per)?], 1)?;

        // The remainder of a table partition is non-negative, as `torch.remainder`'s is
        // (`ple.py:99-101`).
        let sizes = self.ngram_heads_vocab_sizes.unsqueeze(0)?;
        let remainders = hashes.broadcast_remainder(&sizes)?;
        let remainders = remainders
            .lt(0.0)?
            .where_cond(&remainders.broadcast_add(&sizes)?, &remainders)?;

        Ok(remainders.broadcast_add(&self.ngram_heads_offsets.unsqueeze(0)?)?)
    }

    /// The embedding of `context` ([`Self::ids`]): `[tokens, heads · head_dim]` in the
    /// activations' type, the head rows side by side, summed over the tensor-parallel group.
    ///
    /// # Errors
    ///
    /// A context of another shape, a split table in a forward that has no group, or what a
    /// tensor operation refuses.
    pub fn forward(&self, context: &Tensor) -> Result<Tensor> {
        let ids = self.ids(context)?;
        let (tokens, heads) = ids.dims2()?;
        let (first, rows) = self.held;
        let dtype = self.weight_scale.dtype();

        // A row another rank holds reads the nearest row this rank holds, and is zeroed.
        let local = ids
            .flatten_all()?
            .broadcast_sub(&int(first as i64, &ids)?)?;
        let owned = local.ge(0.0)?.mul(&local.lt(rows as f64)?)?;
        let index = local
            .maximum(0.0)?
            .minimum((rows - 1) as f64)?
            .to_dtype(DType::U32)?;

        // The E4M3 rows move as bytes; each byte's value is exact in the activations' type.
        let bytes = self
            .ngram_embedding
            .index_select(&index, 0)?
            .view(DType::U8)?
            .to_dtype(DType::U32)?;
        let values = self
            .values
            .index_select(&bytes.flatten_all()?, 0)?
            .reshape((tokens * heads, self.head_dim))?;

        let scaled = values
            .to_dtype(DType::F32)?
            .broadcast_mul(&self.weight_scale.to_dtype(DType::F32)?)?
            .to_dtype(dtype)?;
        if !self.sharded {
            return Ok(scaled.reshape((tokens, heads * self.head_dim))?);
        }

        let held = owned
            .unsqueeze(1)?
            .broadcast_as(scaled.shape())?
            .where_cond(&scaled, &scaled.zeros_like()?)?
            .reshape((tokens, heads * self.head_dim))?
            .contiguous()?;
        if nn::collective::split().is_none() {
            return Err(Error::Config(
                "an n-gram table split over a tensor group needs the group its ranks sum over"
                    .to_owned(),
            ));
        }
        nn::collective::all_reduce(&held)?;
        Ok(held)
    }
}

/// One element of `value` in `like`'s element type on `like`'s device: an integer constant of
/// a tensor expression, exact where a scalar operand would pass through floating point.
pub(crate) fn int(value: i64, like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(&[value], &Device::Cpu)?
        .to_dtype(like.dtype())?
        .to_device(like.device())?)
}

/// The layer that adds the n-gram embedding into the residual streams.
pub struct PleLayer {
    ple_embedding: NgramEmbedding,
    /// `[key_proj; value_proj]`: `[C · H + H, embed_dim]`, one product for both.
    key_proj: Linear,
    /// `[C · H]` each: the zero-centred gains of the three per-stream norms.
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// `[C · H, taps]`: each channel's taps of the dilated convolution, oldest input first.
    conv1d: Tensor,
    /// `[1]` FP32: `√H`, which the agreement of a key and a stream is divided by.
    root_hidden: Tensor,
    hidden: usize,
    streams: usize,
    eps: f32,
    /// Inputs between consecutive taps: the n-gram size.
    dilation: u32,
}

impl PleLayer {
    /// The layer at `vb` (`<layer>.ple`), at `split`'s rank: its `ple_embedding`, `key_proj`
    /// and `value_proj`, the three norms and the depthwise `conv1d.weight` of
    /// `[C · H, 1, taps]`. Everything but the table is replicated.
    ///
    /// # Errors
    ///
    /// A module the loader refuses.
    pub fn load(config: &Config, vb: VarBuilder<'_, '_>, split: Split) -> Result<Self> {
        let (hidden, streams) = (config.text.hidden_size, config.hyper_connection.streams);
        let width = (hidden * streams) as usize;
        let taps = config.ngram.conv_kernel_size as usize;
        let gain = |module: &str| vb.pp(module).get(width, "weight");

        Ok(Self {
            ple_embedding: NgramEmbedding::load(config, vb.pp("ple_embedding"), split)?,
            key_proj: Linear::load_merged(
                vb.clone(),
                &["key_proj", "value_proj"],
                config.ngram.embed_dim,
                &[hidden * streams, hidden],
                false,
            )?,
            norm_key: gain("norm_key")?,
            norm_query: gain("norm_query")?,
            norm_conv: gain("norm_conv")?,
            conv1d: vb
                .pp("conv1d")
                .get((width, 1, taps), "weight")?
                .reshape((width, taps))?,
            root_hidden: vb.constant(
                "root_hidden",
                DType::F32,
                1,
                &[f64::from(hidden).sqrt() as f32],
            )?,
            hidden: hidden as usize,
            streams: streams as usize,
            eps: config.text.rms_norm_eps,
            dilation: config.ngram.ngram_size,
        })
    }

    /// The state the layer keeps per request slot: the convolution's window of normalized
    /// gated values, its history and one row per token a step drafts, in the activations'
    /// type `dtype`.
    pub fn state_fields(config: &Config, dtype: DType, draft_tokens: u32) -> Vec<StateField> {
        let ngram = &config.ngram;
        let history = (ngram.conv_kernel_size - 1) * ngram.ngram_size;

        vec![StateField {
            name: "ple_conv",
            dtype,
            dims: vec![
                (history + draft_tokens) as usize,
                (config.text.hidden_size * config.hyper_connection.streams) as usize,
            ],
            stepwise: false,
        }]
    }

    /// What the layer adds to `streams`, `[tokens, C · H]` with nothing pending on them:
    /// `[tokens, C · H]` in their element type. `context` is each token's n-gram context
    /// ([`NgramEmbedding::ids`]); the convolution continues from field zero of recurrent layer
    /// `layer` of `state` and advances it in place.
    ///
    /// # Errors
    ///
    /// Streams of another width, or what a module or a function refuses.
    pub fn forward(
        &self,
        streams: &Tensor,
        context: &Tensor,
        batch: &TokenBatch,
        state: &RecurrentState,
        layer: usize,
    ) -> Result<Tensor> {
        let (tokens, width) = streams.dims2()?;
        let (count, hidden) = (self.streams, self.hidden);
        if width != count * hidden {
            return Err(Error::Config(format!(
                "streams of {width} elements against {count} streams of {hidden}"
            )));
        }
        let dtype = streams.dtype();
        // A value in FP32 after its one rounding into the activations' type.
        let rounded =
            |x: &Tensor| -> Result<Tensor> { Ok(x.to_dtype(dtype)?.to_dtype(DType::F32)?) };
        let per_stream = (tokens * count, hidden);
        let norm =
            |rows: &Tensor, gain: &Tensor| functional::rms_norm(rows, gain, self.eps, true, dtype);

        let embedded = self.ple_embedding.forward(context)?;
        let projected = self.key_proj.forward(&embedded)?;
        let key = projected.narrow(1, 0, width)?.contiguous()?;
        let value = projected.narrow(1, width, hidden)?;

        // How each stream's key agrees with the stream.
        let key = norm(&key.reshape(per_stream)?, &self.norm_key)?;
        let query = norm(&streams.reshape(per_stream)?, &self.norm_query)?;
        let products = rounded(
            &key.to_dtype(DType::F32)?
                .mul(&query.to_dtype(DType::F32)?)?,
        )?;
        let agreement = rounded(&products.sum_keepdim(1)?)?;
        let scaled = rounded(&agreement.broadcast_div(&self.root_hidden)?)?;
        let magnitude = rounded(&scaled.abs()?.maximum(1e-6)?.sqrt()?)?;
        let gate = rounded(&scaled.sign()?.mul(&magnitude)?.sigmoid()?)?;

        // The gated value of every stream, and its normalization the convolution reads.
        let gated = gate
            .reshape((tokens, count, 1))?
            .broadcast_mul(&value.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .to_dtype(dtype)?;
        let normalized = norm(&gated.reshape(per_stream)?, &self.norm_conv)?;

        let convolved = functional::causal_conv1d(
            &normalized.reshape((tokens, width))?,
            &self.conv1d,
            self.dilation,
            false,
            batch,
            state,
            layer,
            0,
        )?;
        let activated = rounded(&convolved.to_dtype(DType::F32)?.silu()?)?;

        Ok(gated
            .reshape((tokens, width))?
            .to_dtype(DType::F32)?
            .add(&activated)?
            .to_dtype(dtype)?)
    }
}

impl std::fmt::Debug for PleLayer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PleLayer")
            .field("streams", &self.streams)
            .field("hidden", &self.hidden)
            .field("dilation", &self.dilation)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::e4m3;

    #[test]
    fn an_e4m3_byte_is_its_sign_exponent_and_mantissa() {
        assert_eq!(e4m3(0x00), 0.0);
        // The smallest subnormal and the largest finite value.
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0xfe), -448.0);
        // One: exponent seven, no mantissa.
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xb9), -1.125);
        assert!(e4m3(0x7f).is_nan() && e4m3(0xff).is_nan());
    }
}
