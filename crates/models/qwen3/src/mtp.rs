//! The Qwen3.5 drafter: the checkpoint's multi-token-prediction layers (`mtp.*`), Qwen3-Next's
//! `Qwen3NextMultiTokenPredictor` (vLLM's `qwen3_next_mtp.py` and `qwen3_5_mtp.py`).
//!
//! An entry at position `i` reads the model's final hidden state `h` there and the embedding `e`
//! of the token at `i + 1`:
//!
//! ```text
//! x      = fc([pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)])     [2·hidden] → [hidden]
//! x      = layers(x)       full-attention decoder layers over the draft cache, every one routed
//! h'     = norm(x)                                      the drafter's hidden state at `i`
//! ```
//!
//! The layers are the model's own decoder layer, rotated by the model's text spectrum at the
//! entry's position. The embedding and the head are the model's ([`nn::family::Drafter`]); the
//! drafter's experts keep the checkpoint's order, because an expert redistribution names only the
//! model's layers.

use nn::blocks::{AttentionKind, Keys, RopeFactors};
use nn::capability::{KvView, TokenBatch};
use nn::family::{CacheShape, Drafter, ExpertOrder, Split};
use nn::layers::{Bias, Linear, PagedAttention, Parallel, ResidualStream, RmsNorm, Rotary};
use nn::weights::{Scope, WeightLoader};
use nn::{DType, Forward, Operand, Shape};

use crate::causal_lm::{expert_slots, text_spectrum};
use crate::config::{Config, Mixer};
use crate::decoder::{self, DecoderStack, Mlp};
use crate::{Error, Result, copy_block};

/// The module path the drafter's modules sit under.
const PREFIX: &str = "mtp";

/// The fence class of the drafter's whole forward, at path `mtp`.
const DRAFTER: &str = "models.qwen3.transformer.Drafter";

/// The multi-token-prediction layers of a Qwen3.5 checkpoint.
pub struct Mtp {
    hidden: u32,
    cache: CacheShape,
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc: Linear,
    rotary: Rotary,
    stack: DecoderStack,
    norm: RmsNorm,
}

impl Mtp {
    /// The drafter of `config` at the ranks `splits` state, its norms writing `dtype`.
    ///
    /// # Errors
    ///
    /// A checkpoint that carries no drafter, or a module the loader refuses.
    pub(crate) fn load(
        config: &Config,
        splits: (Split, Split),
        dtype: DType,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let layers = config.mtp_layers;
        if layers == 0 {
            return Err(Error::Config(
                "the deployment speculates and the checkpoint carries no multi-token-prediction \
                 layer"
                    .to_owned(),
            ));
        }
        let text = &config.text;
        let (tensor, _) = splits;
        // The drafter's stack: its own layers, every one full attention and routed.
        let mut drafter = config.clone();
        drafter.text.num_hidden_layers = layers;
        drafter.mixers = vec![Mixer::Attention; layers as usize];

        let mut scope = loader.scope(PREFIX);
        let norm = |scope: &mut Scope<'_, '_>, module: &str| {
            RmsNorm::load(
                &mut scope.scope(module),
                text.hidden_size,
                text.rms_norm_eps,
                text.rms_norm_gain,
                dtype,
            )
        };
        let pre_fc_norm_embedding = norm(&mut scope, "pre_fc_norm_embedding")?;
        let pre_fc_norm_hidden = norm(&mut scope, "pre_fc_norm_hidden")?;
        let fc = Linear::load(
            &mut scope.scope("fc"),
            2 * text.hidden_size,
            text.hidden_size,
            Bias::Absent,
            Parallel::None,
        )?;
        let norm = norm(&mut scope, "norm")?;
        let rotary = decoder::rotary(
            &mut scope.scope("rotary_emb"),
            text,
            text_spectrum(text.rope),
            tensor,
        )?;
        let routed: Vec<u32> = (0..layers).collect();
        let slots = expert_slots(&drafter, &routed, &ExpertOrder::new(), loader.scope(PREFIX))?;
        let stack = DecoderStack::load(
            &mut loader.scope(PREFIX),
            &drafter,
            0..layers,
            AttentionKind::Paged {
                max_kv_len: text.max_position_embeddings,
                window: None,
            },
            tensor,
            dtype,
            &mut |layer, scope| match &slots {
                Some(slots) if drafter.routes(layer) => {
                    crate::causal_lm::routed(scope, &drafter, slots, layer, splits)
                }
                _ => Mlp::dense(scope, text, tensor),
            },
        )?;
        Ok(Self {
            hidden: text.hidden_size,
            cache: CacheShape {
                layers,
                kv_heads: text.num_key_value_heads / tensor.size,
                head_dim: text.head_dim,
            },
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            rotary,
            stack,
            norm,
        })
    }

    /// `fc`'s input: the two normalized inputs side by side, `[tokens, 2·hidden]`, the embedding's
    /// in the first half and the hidden state's in the second.
    fn join(&self, f: &Forward, embedded: Operand, hidden: Operand) -> Result<Operand> {
        let tokens = hidden.shape.dim(0);
        let width = u64::from(self.hidden);
        let joined = f.empty(hidden.dtype, Shape::new(&[tokens, 2, 1, width]))?;
        for (half, (input, norm)) in [
            (embedded, &self.pre_fc_norm_embedding),
            (hidden, &self.pre_fc_norm_hidden),
        ]
        .into_iter()
        .enumerate()
        {
            let normalized = norm.forward(f, input, None)?;
            copy_block(
                f,
                normalized.reshape(Shape::new(&[tokens, 1, 1, width])),
                joined.narrow(1, half as u64, 1)?,
            )?;
        }
        Ok(joined.reshape(Shape::new(&[tokens, 2 * width])))
    }
}

impl Drafter for Mtp {
    fn cache(&self) -> CacheShape {
        self.cache
    }

    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        embedded: Operand,
        hidden: Operand,
    ) -> nn::Result<Operand> {
        let tokens = batch.tokens();
        let width = u64::from(self.hidden);
        for (name, view) in [("embedded", embedded), ("hidden", hidden)] {
            if view.shape.dims() != [tokens, width] {
                return Err(Error::Config(format!(
                    "the drafter's {name} is {:?}, [{tokens}, {width}] expected",
                    view.shape.dims()
                ))
                .into());
            }
        }
        // The drafter's tensors are a fixed handful per forward ([tokens, 2·hidden] and a few
        // [tokens, hidden]); its layers hand their intermediates back in their own scopes.
        Ok(f.fenced(DRAFTER, format_args!("{PREFIX}"), || {
            let stream = self.fc.forward(f, self.join(f, embedded, hidden)?)?;
            let rope = RopeFactors {
                rotary: &self.rotary,
                factors: self.rotary.factors(f, batch.positions)?,
            };
            // The decode program's arrival counters are cleared once per forward, as the model's
            // own forward clears them.
            PagedAttention::zero_counters(f, cache.workspace)?;
            let (last, residual) = self.stack.forward(
                f,
                stream,
                rope,
                |layer| {
                    Ok(Keys::Paged {
                        batch,
                        cache: cache.layer(0, layer)?,
                    })
                },
                |_, _| Ok(()),
            )?;
            Ok::<_, Error>(
                self.norm
                    .forward(f, last, Some(ResidualStream::in_place(residual)))?,
            )
        })?)
    }
}

impl std::fmt::Debug for Mtp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Mtp")
            .field("layers", &self.stack.layers().len())
            .field("hidden", &self.hidden)
            .finish_non_exhaustive()
    }
}
