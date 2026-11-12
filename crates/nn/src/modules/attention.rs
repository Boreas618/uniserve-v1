//! The attention of a transformer block: project, normalize and rotate each head, attend,
//! project out.
//!
//! The module holds the query, key and value projections ([`QkvLinear`], column parallel over
//! the split's heads), the per-head q/k normalization in the form the model's reference
//! computes, and the row-parallel output projection. The rotary factors are the forward's:
//! every layer of a stack turns by the same tables, which the stack's
//! [`Rotary`](crate::modules::Rotary) computes once.
//!
//! ## The forwards
//!
//! ```text
//! forward          dense attention over the rows as one sequence
//! forward_varlen   dense attention over the sequences two offset tables name
//! forward_paged    the rows' keys and values written into the layer's cache pages, then
//!                  causal attention over the pages the batch's rows address
//! ```
//!
//! ## The q/k normalization
//!
//! ```text
//! QkNorm::None      no normalization; the rotation alone, when the forward rotates
//! QkNorm::Fused     a per-head RMS normalization fused into the rotation (qk_norm_rope)
//! QkNorm::Separate  a per-head RmsNorm of each projection, then the rotation alone if any
//! ```
//!
//! The fused and the separate forms round differently (the fused kernel normalizes and rotates
//! in FP32 and rounds once), so they are two forms and a model keeps its reference's.
//!
//! ## The output gate
//!
//! A gated attention (Qwen3-Next's and Qwen3.5's `attn_output_gate`,
//! [`QkvLinearConfig::gated`]) projects each query head together with a gate of the head's
//! width; the attended heads are multiplied by the sigmoid of their gates before the output
//! projection:
//!
//! ```text
//! [q_h | g_h] = q_proj(x)_h;   o_h = attention(q, k, v)_h · sigmoid(g_h);   out = o_proj(o)
//! ```

use uniserve_tensor::Tensor;

use crate::capability::{KvCache, TokenBatch};
use crate::error::{Error, Result};
use crate::functional::{self, Activation};
use crate::modules::linear::{Linear, LinearConfig, Parallel, QkvLinear, QkvLinearConfig};
use crate::modules::norm::{RmsNorm, RmsNormConfig};
use crate::var_builder::VarBuilder;

/// The per-head normalization of an attention's queries and keys, and where the checkpoint
/// stores it; see the module table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QkNorm {
    /// No normalization.
    None,
    /// A per-head RMS normalization fused into the forward's rotation.
    Fused {
        /// The modules of the queries' and the keys' learned gains (`<module>.weight`,
        /// `[head_dim]`), or `None` for a normalization without a gain.
        gains: Option<[&'static str; 2]>,
        /// The variance floor.
        eps: f32,
        /// Whether the checkpoint stores the gains zero-centred, so a head is scaled by one
        /// plus its gain.
        centred: bool,
    },
    /// A per-head [`RmsNorm`] of the queries and of the keys, computed before any rotation in
    /// the projections' element type.
    Separate {
        /// The queries' and the keys' normalization modules.
        modules: [&'static str; 2],
        /// The variance floor.
        eps: f32,
    },
}

/// The names, widths and arithmetic of an [`Attention`], and this rank's share of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionConfig {
    /// The query, key and value projections: the heads of the whole module, where the
    /// checkpoint stores them, and the tensor-parallel split.
    pub qkv: QkvLinearConfig,
    /// The output projection's module name.
    pub output: &'static str,
    /// Whether the output projection adds its module's bias.
    pub output_bias: bool,
    /// The per-head normalization.
    pub qk_norm: QkNorm,
    /// The softmax scale applied to `q · k`.
    pub scale: f32,
    /// Whether each query sees only its own past. A paged forward is causal by construction
    /// and requires it.
    pub causal: bool,
    /// An inclusive history window, which only the causal mask reads.
    pub window: Option<u32>,
    /// The longest key reduction the deployment serves. A dense forward's key bound stays
    /// under it; a paged forward states it in its descriptor, so a deployment that serves
    /// longer sequences loads another module.
    pub max_kv_len: u32,
}

/// The per-head normalization an attention holds.
enum Norm {
    None,
    Fused {
        /// The queries' and the keys' gains, when the normalization has them.
        gains: Option<(Tensor, Tensor)>,
        eps: f32,
        centred: bool,
    },
    Separate {
        query: RmsNorm,
        key: RmsNorm,
    },
}

/// The attention of a transformer block, over its own projections.
pub struct Attention {
    qkv: QkvLinear,
    norm: Norm,
    output: Linear,
    /// Query heads this rank holds.
    heads: usize,
    /// Key/value heads this rank holds.
    kv_heads: usize,
    head_dim: usize,
    /// Whether each query head is projected with an output gate beside it.
    gated: bool,
    scale: f32,
    causal: bool,
    window: Option<u32>,
    max_kv_len: u32,
}

impl Attention {
    /// The attention the module at `vb` states, at `config.qkv.split`'s rank of its
    /// tensor-parallel group.
    ///
    /// The query and key heads are this rank's share (the key heads replicated when the group
    /// is wider than they are); a group of one holds the output projection whole.
    ///
    /// # Errors
    ///
    /// A projection or a gain the loader refuses, a split that does not divide the heads, a
    /// head or a key bound of zero, a window without the causal mask, or a scale that is zero
    /// or not finite.
    pub fn load(config: &AttentionConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        const OWNER: &str = "Attention";

        let head_dim = config.qkv.head_dim;
        if head_dim == 0 || config.max_kv_len == 0 {
            return Err(Error::shape(
                OWNER,
                "a head and a key bound must be nonzero".to_owned(),
            ));
        }
        if config.window.is_some() && !config.causal {
            return Err(Error::shape(
                OWNER,
                "a history window reads the causal mask only".to_owned(),
            ));
        }
        if !config.scale.is_finite() || config.scale == 0.0 {
            return Err(Error::shape(
                OWNER,
                format!("a softmax scale of {}", config.scale),
            ));
        }

        let qkv = QkvLinear::load(&config.qkv, vb.clone())?;
        let gated = config.qkv.gated;
        let heads = qkv.query().n() / head_dim / if gated { 2 } else { 1 };
        let kv_heads = qkv.key().n() / head_dim;
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(Error::shape(
                OWNER,
                format!("{heads} query heads do not group {kv_heads} key heads"),
            ));
        }

        // Row parallel: the output projection sums each rank's partial output over the split.
        let split = config.qkv.split;
        let output = Linear::load(
            &LinearConfig {
                inputs: config.qkv.heads * head_dim,
                outputs: config.qkv.hidden,
                bias: config.output_bias,
                parallel: if split.size == 1 {
                    Parallel::None
                } else {
                    Parallel::Row(split)
                },
            },
            vb.pp(config.output),
        )?;

        let norm = match config.qk_norm {
            QkNorm::None => Norm::None,
            QkNorm::Fused {
                gains,
                eps,
                centred,
            } => {
                let gain = |module: &str| vb.pp(module).get(head_dim as usize, "weight");
                let gains = match gains {
                    Some([query, key]) => Some((gain(query)?, gain(key)?)),
                    None => None,
                };

                Norm::Fused {
                    gains,
                    eps,
                    centred,
                }
            }
            QkNorm::Separate {
                modules: [query, key],
                eps,
            } => {
                let per_head = RmsNormConfig {
                    width: head_dim,
                    eps,
                    centred: false,
                    output: None,
                };

                Norm::Separate {
                    query: RmsNorm::load(&per_head, vb.pp(query))?,
                    key: RmsNorm::load(&per_head, vb.pp(key))?,
                }
            }
        };

        Ok(Self {
            qkv,
            norm,
            output,
            heads: heads as usize,
            kv_heads: kv_heads as usize,
            head_dim: head_dim as usize,
            gated,
            scale: config.scale,
            causal: config.causal,
            window: config.window,
            max_kv_len: config.max_kv_len,
        })
    }

    /// The query projection, whose representation a caller may ask about (a packed product
    /// writes whole row classes).
    pub fn query(&self) -> &Linear {
        self.qkv.query()
    }

    /// `hidden`, `[rows, hidden]`, attended as **one sequence** and projected out: `[rows,
    /// hidden]` in `hidden`'s element type.
    ///
    /// `rope` is the forward's factor tables `(cos, sin)`, FP32 `[rows, pairs]` each, and
    /// `None` for an attention that does not rotate; a fused q/k normalization needs them.
    ///
    /// # Errors
    ///
    /// A fused normalization without a rotation, more rows than the module's key bound, or
    /// tensors a projection or a function refuses.
    pub fn forward(&self, hidden: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        self.attend(hidden, rope, |query, key, value| {
            self.bounded(key.dims()[0])?;
            functional::dense_attention(query, key, value, self.scale, self.causal, self.window)
        })
    }

    /// [`Self::forward`] over the variable-length sequences `cu_seqlens_q` and `cu_seqlens_k`
    /// name, with the longest query sequence `max_seqlen_q` when the caller knows it and the
    /// key bound `max_kv_len` of this launch ([`functional::dense_attention_varlen`]).
    ///
    /// # Errors
    ///
    /// As [`Self::forward`], and for a key bound above the module's or tables the function
    /// refuses.
    pub fn forward_varlen(
        &self,
        hidden: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        cu_seqlens_q: &Tensor,
        cu_seqlens_k: &Tensor,
        max_seqlen_q: Option<u32>,
        max_kv_len: u32,
    ) -> Result<Tensor> {
        self.attend(hidden, rope, |query, key, value| {
            self.bounded(max_kv_len as usize)?;
            functional::dense_attention_varlen(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_kv_len,
                self.scale,
                self.causal,
                self.window,
            )
        })
    }

    /// `hidden`, `[tokens, hidden]`, the rows of `batch`, attended over the key/value cache
    /// and projected out: the tokens' keys and values are written into layer `layer` of
    /// `cache`, in place, and each token attends the keys its row holds there up to its own.
    ///
    /// # Errors
    ///
    /// An attention loaded without the causal mask, a cache page longer than the module's key
    /// bound, a fused normalization without a rotation, or a batch, a cache or tensors a
    /// function refuses.
    pub fn forward_paged(
        &self,
        hidden: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        batch: &TokenBatch,
        cache: &KvCache,
        layer: usize,
    ) -> Result<Tensor> {
        const OWNER: &str = "Attention";

        if !self.causal {
            return Err(Error::shape(
                OWNER,
                "attention over the cache is causal, and this attention was loaded without the \
                 mask"
                    .to_owned(),
            ));
        }

        // Refused before the write, so a cache laid out in longer pages is never addressed.
        if self.max_kv_len < cache.page_size {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a {}-key bound is shorter than a {}-token page",
                    self.max_kv_len, cache.page_size
                ),
            ));
        }

        self.attend(hidden, rope, |query, key, value| {
            functional::kv_write_(cache, layer, batch, key, value)?;
            functional::paged_attention(
                query,
                batch,
                cache,
                layer,
                self.scale,
                self.window,
                self.max_kv_len,
            )
        })
    }

    /// Refuses a launch whose key bound `keys` is beyond the one the module was loaded for.
    fn bounded(&self, keys: usize) -> Result<()> {
        if keys > self.max_kv_len as usize {
            return Err(Error::shape(
                "Attention",
                format!(
                    "a {keys}-key bound beyond the {} this attention was loaded for",
                    self.max_kv_len
                ),
            ));
        }
        Ok(())
    }

    /// The forward around the attention itself: `attention(query, key, value)` returns the
    /// attended heads, `[rows, heads, head_dim]`, of the normalized and turned projections.
    fn attend(
        &self,
        hidden: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        attention: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        const OWNER: &str = "Attention";

        let &[rows, _] = hidden.dims() else {
            return Err(Error::shape(
                OWNER,
                format!("{:?} is not [rows, hidden]", hidden.dims()),
            ));
        };
        let (heads, kv_heads, head_dim) = (self.heads, self.kv_heads, self.head_dim);

        // 1. The projections, as heads.
        let projected = self.qkv.query().forward(hidden)?;
        let key = self
            .qkv
            .key()
            .forward(hidden)?
            .reshape((rows, kv_heads, head_dim))?;
        let value = self
            .qkv
            .value()
            .forward(hidden)?
            .reshape((rows, kv_heads, head_dim))?;

        // A gated head's query and gate sit side by side in its row: the queries are packed
        // for the normalization and rotation, and the gates are read where the projection
        // wrote them.
        let (query, gate) = if self.gated {
            let pairs = projected.reshape((rows * heads, 2 * head_dim))?;
            let query = pairs
                .narrow(1, 0, head_dim)?
                .contiguous()?
                .reshape((rows, heads, head_dim))?;

            (query, Some(pairs.narrow(1, head_dim, head_dim)?))
        } else {
            (projected.reshape((rows, heads, head_dim))?, None)
        };

        // 2. Each head normalized and turned, in the module's form.
        let (query, key) = match (&self.norm, rope) {
            (Norm::None, None) => (query, key),

            (Norm::None, Some((cos, sin))) => (
                functional::rotary(&query, cos, sin)?,
                functional::rotary(&key, cos, sin)?,
            ),

            (Norm::Fused { .. }, None) => {
                return Err(Error::shape(
                    OWNER,
                    "a q/k norm fused into the rotation needs the forward's rotation".to_owned(),
                ));
            }

            (
                Norm::Fused {
                    gains,
                    eps,
                    centred,
                },
                Some((cos, sin)),
            ) => functional::qk_norm_rope(
                &query,
                &key,
                gains.as_ref().map(|(query, key)| (query, key)),
                cos,
                sin,
                *eps,
                *centred,
            )?,

            // The norm is over one head's channels, so it reads each projection as one row per
            // head.
            (
                Norm::Separate {
                    query: query_norm,
                    key: key_norm,
                },
                rope,
            ) => {
                let (query, key) = (query_norm.forward(&query)?, key_norm.forward(&key)?);

                match rope {
                    None => (query, key),
                    Some((cos, sin)) => (
                        functional::rotary(&query, cos, sin)?,
                        functional::rotary(&key, cos, sin)?,
                    ),
                }
            }
        };

        // 3. The attention, and each attended head scaled by the sigmoid of its own gate.
        let attended = attention(&query, &key, &value)?;
        let attended = match gate {
            Some(gate) => functional::act_and_mul(
                &gate,
                &attended.reshape((rows * heads, head_dim))?,
                Activation::Sigmoid,
                false,
            )?,
            None => attended,
        };

        self.output
            .forward(&attended.reshape((rows, heads * head_dim))?)
    }
}
