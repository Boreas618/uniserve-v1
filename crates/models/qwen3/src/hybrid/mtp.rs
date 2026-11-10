//! The Qwen3.5-MoE drafter: the checkpoint's multi-token-prediction layers (`mtp.*`), Qwen3-Next's
//! `Qwen3NextMultiTokenPredictor` (vLLM's `qwen3_next_mtp.py` and `qwen3_5_mtp.py`, SGLang's
//! `qwen3_next_mtp.py`).
//!
//! An entry at position `i` reads the model's final hidden state `h` there and the embedding `e`
//! of the token at `i + 1`:
//!
//! ```text
//! x      = fc([pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)])     [2·hidden] → [hidden]
//! x      = layers(x)       full-attention decoder layers over the draft cache, every one routed
//! h'     = norm(x)                                      the drafter's hidden state at `i`
//! logits = lm_head(h')                                  the distribution of the token at `i + 2`
//! ```
//!
//! The norms are the checkpoint's zero-centred RMS norms, the layers the model's own decoder layer
//! with its gated attention and shared-expert MoE, rotated by the model's text spectrum at the
//! entry's position. The embedding and the head are the model's ([`nn::family::Drafter`]).

use nn::blocks::Keys;
use nn::capability::{KvView, TokenBatch};
use nn::family::{CacheShape, Drafter};
use nn::layers::{Access, Copy, Linear, PagedAttention, ResidualStream, RmsNorm};
use nn::{Forward, Shape, TensorView};

use crate::decoder::{Components, DecoderStack, Hidden};
use crate::{Error, Result};

/// The components a deployment's fence probes name the drafter's modules by.
pub(crate) const DRAFTER_COMPONENTS: Components = Components {
    layer: "models.qwen3.hybrid.DrafterLayer",
    attention: Some("models.qwen3.hybrid.DrafterMixer"),
    mlp: Some("models.qwen3.hybrid.DrafterMoE"),
};

/// The drafter's own fence component, around its whole forward.
const DRAFTER: &str = "models.qwen3.hybrid.Drafter";

/// The multi-token-prediction layers of a Qwen3.5-MoE checkpoint.
pub struct Mtp {
    hidden: u32,
    /// The draft cache this rank holds: one key/value layer per decoder layer.
    cache: CacheShape,
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc: Linear,
    stack: DecoderStack,
    norm: RmsNorm,
}

impl Mtp {
    /// The drafter of `layers` decoder layers over a `hidden`-wide stream, whose attention stores
    /// `kv_heads` heads of `head_dim` per layer on this rank, from its loaded modules.
    #[allow(
        clippy::too_many_arguments,
        reason = "the drafter's geometry and its five modules, as the loading boundary read them"
    )]
    pub(crate) fn new(
        hidden: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        pre_fc_norm_embedding: RmsNorm,
        pre_fc_norm_hidden: RmsNorm,
        fc: Linear,
        stack: DecoderStack,
        norm: RmsNorm,
    ) -> Self {
        Self {
            hidden,
            cache: CacheShape {
                layers,
                kv_heads,
                head_dim,
            },
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            stack,
            norm,
        }
    }

    /// The drafter's decoder layers.
    pub fn stack(&self) -> &DecoderStack {
        &self.stack
    }

    /// Writes the two normalized inputs side by side into `joined [tokens, 2·hidden]`: the
    /// embedding's in the first half, the hidden state's in the second, as `fc` contracts them.
    fn join(
        &self,
        f: &Forward,
        embedded: TensorView,
        hidden: TensorView,
        joined: TensorView,
    ) -> Result<()> {
        let tokens = hidden.shape.dim(0);
        let width = u64::from(self.hidden);
        let rows = Shape::new(&[tokens, width]);
        let arena = f.arena();
        let copy = Copy::new(hidden.dtype, Access::Vector)?;
        for (half, (input, norm)) in [
            (embedded, &self.pre_fc_norm_embedding),
            (hidden, &self.pre_fc_norm_hidden),
        ]
        .into_iter()
        .enumerate()
        {
            let normalized = arena.tensor(hidden.dtype, rows)?;
            norm.forward(f, input, None, normalized)?;
            // The half is a `[tokens, hidden]` block of rows `2·hidden` apart, which the copy's
            // four axes state as `[tokens, 1, 1, hidden]`.
            let block = Shape::new(&[tokens, 1, 1, width]);
            let source = TensorView::new(normalized.ptr, hidden.dtype, block);
            let destination = TensorView::new(
                joined.offset_elements(half as u64 * width),
                hidden.dtype,
                block,
            );
            let stride = i64::try_from(2 * width)
                .map_err(|_| Error::Config(format!("a {width}-wide hidden row")))?;
            copy.forward(f, source, destination, &[stride, stride, stride, 1])?;
        }
        Ok(())
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
        embedded: TensorView,
        hidden: TensorView,
    ) -> nn::Result<()> {
        let tokens = batch.tokens();
        let width = u64::from(self.hidden);
        for (name, view) in [("embedded", embedded), ("hidden", hidden)] {
            if view.shape.rank() != 2 || view.shape.dim(0) != tokens || view.shape.dim(1) != width {
                return Err(nn::Error::from(Error::Config(format!(
                    "the drafter's {name} is {:?}, [{tokens}, {width}] expected",
                    view.shape.dims()
                ))));
            }
        }
        let dtype = hidden.dtype;
        f.fenced(DRAFTER, format_args!("mtp"), || {
            let arena = f.arena();
            let mark = arena.mark();
            let joined = arena.tensor(dtype, Shape::new(&[tokens, 2 * width]))?;
            self.join(f, embedded, hidden, joined)
                .map_err(nn::Error::from)?;
            let input = arena.tensor(dtype, Shape::new(&[tokens, width]))?;
            self.fc.forward(f, joined, input)?;

            let (cosine, sine) = self
                .stack
                .factors(f, batch.positions, tokens)
                .map_err(nn::Error::from)?;
            let buffers = self
                .stack
                .buffers(arena, tokens, self.hidden, dtype, None)
                .map_err(nn::Error::from)?;
            // The decode program's arrival counters at the workspace's head are cleared once per
            // forward, as the model's own forward clears them.
            PagedAttention::zero_counters(f, cache.workspace)?;
            let last = self
                .stack
                .forward(
                    f,
                    Hidden {
                        hidden: input,
                        residual: None,
                    },
                    cosine,
                    sine,
                    |layer| Keys::Paged {
                        batch,
                        cache,
                        group: 0,
                        layer,
                    },
                    &buffers,
                    |_, state| Ok(state),
                )
                .map_err(nn::Error::from)?;
            let residual = last.residual.ok_or_else(|| {
                nn::Error::from(Error::Config(
                    "a drafter needs at least one layer".to_owned(),
                ))
            })?;
            // The output norm adds the last update into the stream and writes the drafter's hidden
            // state over its input, which no later launch reads.
            self.norm.forward(
                f,
                last.hidden,
                Some(ResidualStream::in_place(residual)),
                hidden,
            )?;
            arena.release(mark);
            Ok(())
        })
    }
}

impl std::fmt::Debug for Mtp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Mtp")
            .field("layers", &self.stack.layers().len())
            .field("hidden", &self.hidden)
            .finish()
    }
}
