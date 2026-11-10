//! The Qwen3 decoder layer and the stack of them, which every Qwen3 model composes.
//!
//! The layer is Hugging Face's `Qwen3DecoderLayer` (and `Qwen3MoeDecoderLayer`, which differs only
//! in its feed-forward block), under vLLM's residual contract:
//!
//! ```text
//! residual None:     residual = hidden;            hidden = input_layernorm(hidden)
//! residual Some(r):  hidden, residual = add_rms_norm(hidden, r, input_layernorm)
//! attention:         hidden = self_attn(hidden)     (q/k RMSNorm fused into the rotation)
//!                    hidden, residual = add_rms_norm(hidden, residual, post_attention_layernorm)
//! feed-forward:      hidden = mlp(hidden)           (gated MLP or routed experts)
//! ```
//!
//! A layer returns its update unsummed with the residual stream, and the addition is fused into
//! the next normalization: the stack never computes `hidden + residual` and then normalizes the
//! rounded sum. The residual stream is summed in FP32 inside each `add_rms_norm`.
//!
//! ## What a consumer chooses
//!
//! The consumers' baselines differ in arithmetic the layer carries as named parameters
//! ([`LayerForm`]), never as a second layer:
//!
//! - the attention: paged over the prefix cache for a causal LM, dense over packed sequences for
//!   an encoder ([`AttentionKind`]);
//! - the dense gated product's rounding ([`Rounding`]);
//! - the rotary spectrum and its precision, which the stack's one [`Rotary`] holds;
//! - the fence components the deployment's probes name the layer and its sublayers by.
//!
//! ## Buffers
//!
//! A stack's activations are one set for the whole forward ([`StackBuffers`]): the normalized
//! values, the layer's update and the residual stream are reused as the layer loop advances. Each
//! layer's own intermediates are reserved under an arena mark released when the layer's launches
//! are enqueued, so a captured bucket replays over the same addresses.

use std::ops::Range;

use nn::blocks::{
    AttentionKind, GateUp, GatedMlp, GatedMlpSpec, Keys, QkNormNames, QkvAttention,
    QkvAttentionSpec, RopeFactors,
};
use nn::family::Split;
use nn::layers::{
    Bias, Moe, QkNorm, QkvBias, QkvNames, QkvSpec, ResidualStream, RmsNorm, Rotary, RotarySpec,
    Rounding,
};
use nn::weights::Scope;
use nn::{Arena, DType, Forward, Shape, TensorView};

use crate::Result;
use crate::config::TextConfig;

/// The fence components a deployment's probes name a stack's modules by
/// (`profiling_config.fence_components`, `docs/profiling.md`).
///
/// The layer is always fenced; a consumer that states no component for a sublayer runs it
/// unfenced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Components {
    /// One decoder layer, at path `layers.<i>`.
    pub layer: &'static str,
    /// A layer's attention, at path `layers.<i>.self_attn`.
    pub attention: Option<&'static str>,
    /// A layer's feed-forward block, at path `layers.<i>.mlp`.
    pub mlp: Option<&'static str>,
}

/// The arithmetic and attention a consumer's baseline computes its decoder layers with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerForm {
    /// Paged causal attention over the prefix cache, or dense attention over packed sequences.
    pub attention: AttentionKind,
    /// Where a dense layer's gated product rounds: `Rounding::ValueProduct` is the swiglu
    /// kernel's order, `Rounding::Fused` the reference's `act_fn(gate) * up` in one expression.
    pub rounding: Rounding,
    /// The modules the deployment's fence probes name.
    pub components: Components,
}

/// A layer's feed-forward block.
///
/// The expert block is by far the larger variant and the common one (every layer of
/// Qwen3-30B-A3B routes); boxing it would put an indirection on every forward to save a few
/// hundred bytes in a vector of layers.
#[allow(
    clippy::large_enum_variant,
    reason = "the routed block is the common case and is held once per layer"
)]
pub enum Mlp {
    /// `down(act(gate(x)) · up(x))`, column parallel in and row parallel out.
    Dense(GatedMlp),
    /// Routed experts: a replicated router and the grouped products over the local expert set.
    Experts(Moe),
}

/// One decoder layer: two pre-norms and the two sublayers between them, named as the checkpoint
/// names its modules.
pub struct DecoderLayer {
    pub input_layernorm: RmsNorm,
    /// Q/K/V column parallel over the heads, per-head gains fused into the stack's rotation, the
    /// form's attention, and the row-parallel output projection.
    pub self_attn: QkvAttention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
}

/// A layer's input and output: the update, and the residual stream it has not been added into
/// yet (`None` before the first layer, where the update is the stream).
#[derive(Clone, Copy, Debug)]
pub struct Hidden {
    /// The update the next normalization adds into the stream, or the stream itself.
    pub hidden: TensorView,
    /// The residual stream, when `hidden` is an update not yet added into it.
    pub residual: Option<TensorView>,
}

/// One forward's activations, shared by every layer of the stack.
#[derive(Clone, Copy, Debug)]
pub struct StackBuffers {
    /// `[tokens, hidden]`: one normalization.
    pub normalized: TensorView,
    /// `[tokens, hidden]`: a sublayer's update, in a region a packed product's row class fits.
    pub update: TensorView,
    /// `[tokens, hidden]`: where every layer leaves the residual stream.
    pub residual: TensorView,
}

impl DecoderLayer {
    /// The layer the module at `scope` states, at `split`'s rank of its tensor-parallel group.
    ///
    /// `scale` is the softmax scale the stack's spectrum states for a head; `experts` answers the
    /// routed block read from the layer's `mlp` scope when the layer routes, and `None` for a dense
    /// layer. The modules are read in the checkpoint's order: the norms, the attention, the MLP.
    ///
    /// # Errors
    ///
    /// A module the loader refuses or a split that does not divide what it shards.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        text: &TextConfig,
        form: &LayerForm,
        scale: f32,
        split: Split,
        experts: impl FnOnce(&mut Scope<'_, '_>) -> Result<Option<Moe>>,
    ) -> Result<Self> {
        let eps = text.rms_norm_eps;
        let gain = text.rms_norm_gain;
        let input_layernorm = RmsNorm::load(
            &mut scope.scope("input_layernorm"),
            text.hidden_size,
            eps,
            gain,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            &mut scope.scope("post_attention_layernorm"),
            text.hidden_size,
            eps,
            gain,
        )?;
        let bias = if text.attention_bias {
            (QkvBias::Present, Bias::Present)
        } else {
            (QkvBias::Absent, Bias::Absent)
        };
        let self_attn = QkvAttention::load(
            &mut scope.scope("self_attn"),
            QkvAttentionSpec {
                qkv: QkvSpec {
                    hidden: text.hidden_size,
                    heads: text.num_attention_heads,
                    kv_heads: text.num_key_value_heads,
                    head_dim: text.head_dim,
                },
                names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
                bias: bias.0,
                output: "o_proj",
                output_bias: bias.1,
                qk_norm: QkNormNames::Fused {
                    query: "q_norm",
                    key: "k_norm",
                },
                kind: form.attention,
                scale,
            },
            split,
        )?;
        let mut mlp_scope = scope.scope("mlp");
        let mlp = match experts(&mut mlp_scope)? {
            Some(moe) => Mlp::Experts(moe),
            None => Mlp::Dense(GatedMlp::load(
                &mut mlp_scope,
                GatedMlpSpec {
                    hidden: text.hidden_size,
                    intermediate: text.intermediate_size,
                    activation: text.hidden_act,
                    rounding: form.rounding,
                    gate_up: GateUp::Split {
                        gate: "gate_proj",
                        up: "up_proj",
                    },
                    down: "down_proj",
                    bias: Bias::Absent,
                },
                split,
            )?),
        };
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Run the layer over `input`: normalize (adding the update into the stream when there is
    /// one), attend, add and normalize, then the feed-forward block.
    ///
    /// The update lands in `buffers.update` and the stream in `buffers.residual`, which the
    /// answer names. When `input` carries no residual its `hidden` is the stream, read in place;
    /// the post-attention norm then writes the stream into `buffers.residual`, which may be
    /// `hidden` itself.
    ///
    /// # Errors
    ///
    /// A view a layer refuses, or keys of another kind than the layer's attention.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer runs the reference's tuple through the forward's buffers, with the \
                  stack's factors, the layer's keys and the fence path"
    )]
    pub fn forward(
        &self,
        f: &Forward,
        input: Hidden,
        rope: RopeFactors<'_>,
        keys: Keys<'_>,
        buffers: &StackBuffers,
        components: &Components,
        index: usize,
    ) -> Result<Hidden> {
        let normalized = buffers.normalized;
        let update = buffers.update;

        // Without a residual the input is the stream itself: it normalizes alone and is read
        // as the stream by the post-attention add.
        let stream = match input.residual {
            None => {
                self.input_layernorm
                    .forward(f, input.hidden, None, normalized)?;
                input.hidden
            }
            Some(residual) => {
                self.input_layernorm.forward(
                    f,
                    input.hidden,
                    Some(ResidualStream::in_place(residual)),
                    normalized,
                )?;
                residual
            }
        };

        fence(f, components.attention, index, "self_attn", || {
            Ok(self
                .self_attn
                .forward(f, normalized, Some(rope), keys, update)?)
        })?;

        self.post_attention_layernorm.forward(
            f,
            update,
            Some(ResidualStream {
                input: stream,
                output: buffers.residual,
            }),
            normalized,
        )?;

        fence(f, components.mlp, index, "mlp", || {
            // Both blocks sum their tensor-parallel partial outputs themselves: the routed block
            // after its experts, the dense one in its row-parallel down projection.
            match &self.mlp {
                Mlp::Experts(moe) => moe.forward(f, normalized, update)?,
                Mlp::Dense(dense) => dense.forward(f, normalized, update)?,
            }
            Ok(())
        })?;
        Ok(Hidden {
            hidden: update,
            residual: Some(buffers.residual),
        })
    }
}

/// Run `body` under the fence of `component` at path `layers.<index>.<module>`, or unfenced when
/// the consumer states no component for the module.
fn fence(
    f: &Forward,
    component: Option<&'static str>,
    index: usize,
    module: &str,
    body: impl FnOnce() -> Result<()>,
) -> Result<()> {
    match component {
        Some(component) => f.fenced(component, format_args!("layers.{index}.{module}"), body),
        None => body(),
    }
}

/// A stack of decoder layers and the one rotary embedding every layer turns by.
pub struct DecoderStack {
    /// The stack's rotation: the forward's factor tables, and every layer's q/k norm and turn.
    rotary: Rotary,
    layers: Vec<DecoderLayer>,
    components: Components,
}

impl DecoderStack {
    /// The decoder layers `layers` of the checkpoint module at `scope` (`<scope>.layers.<i>`), with
    /// the stack's rotary (`<scope>.rotary_emb`), at `split`'s rank of the tensor-parallel group.
    ///
    /// A consumer that truncates the checkpoint's stack loads a prefix of its layers. `experts`
    /// answers the routed block of a layer that routes, read from the layer's scope, and `None`
    /// for a dense layer.
    ///
    /// # Errors
    ///
    /// A module the loader refuses, a split that does not divide the heads, or an empty range.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        text: &TextConfig,
        form: LayerForm,
        split: Split,
        layers: Range<u32>,
        mut experts: impl FnMut(u32, &mut Scope<'_, '_>) -> Result<Option<Moe>>,
    ) -> Result<Self> {
        if layers.is_empty() || layers.end > text.num_hidden_layers {
            return Err(crate::Error::Config(format!(
                "decoder layers {layers:?} of a {}-layer stack",
                text.num_hidden_layers
            )));
        }
        let (heads, kv_heads) = text.local_heads(split)?;
        // One rotary for the stack: every layer turns its queries and keys by the same factors,
        // and the per-head norm before the rotation is fused into it at the model's epsilon.
        let rotary = Rotary::load(
            &mut scope.scope("rotary_emb"),
            text.rope,
            RotarySpec {
                head_dim: text.head_dim,
                heads,
                kv_heads,
                qk_norm: QkNorm::Fused {
                    epsilon: text.rms_norm_eps,
                    gain: text.rms_norm_gain,
                },
            },
        )?;
        let scale = text.rope.score_scale(text.head_dim)?;
        let layers = layers
            .map(|layer| {
                DecoderLayer::load(
                    &mut scope.scope(format!("layers.{layer}")),
                    text,
                    &form,
                    scale,
                    split,
                    |mlp| experts(layer, mlp),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rotary,
            layers,
            components: form.components,
        })
    }

    /// The layers, in stack order.
    pub fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }

    /// Reserve the `[tokens, head_dim / 2]` FP32 cosine and sine of `positions` from `f`'s arena
    /// and compute them: one forward's worth of trigonometry, which every layer shares.
    ///
    /// `positions` is `[tokens]` for a plain spectrum and `[axes, tokens]` for M-RoPE.
    ///
    /// # Errors
    ///
    /// An arena that cannot hold the tables, or positions the rotary refuses.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        tokens: u64,
    ) -> Result<(TensorView, TensorView)> {
        let shape = Shape::new(&[tokens, u64::from(self.rotary.columns())]);
        let cosine = f.arena().tensor(DType::F32, shape)?;
        let sine = f.arena().tensor(DType::F32, shape)?;
        self.rotary.factors(f, positions, cosine, sine)?;
        Ok((cosine, sine))
    }

    /// Reserve one forward's activations over `tokens` rows of the `hidden`-wide stream.
    ///
    /// `residual` is the region the stream lives in, reserved here when `None`. The update is
    /// reserved at the row class a packed projection writes whole tiles of, when the stack's
    /// projections are packed: every other op reads the request's prefix of it.
    ///
    /// # Errors
    ///
    /// A row count above the largest packed class, or an arena that cannot hold the buffers.
    pub fn buffers(
        &self,
        arena: &Arena,
        tokens: u64,
        hidden: u32,
        dtype: DType,
        residual: Option<TensorView>,
    ) -> Result<StackBuffers> {
        let hidden = u64::from(hidden);
        let rows = Shape::new(&[tokens, hidden]);
        let packed = self
            .layers
            .first()
            .is_some_and(|layer| layer.self_attn.query().is_packed());
        let held = if packed {
            let counted = u32::try_from(tokens)
                .map_err(|_| crate::Error::Config(format!("{tokens} activation rows")))?;
            u64::from(ops::fp4::packed_rows(counted).map_err(nn::Error::from)?)
        } else {
            tokens
        };
        let normalized = arena.tensor(dtype, rows)?;
        // The region is the row class an encoded projection writes whole tiles into, while the
        // view every other op reads is the request's prefix of it.
        let region = arena.tensor(dtype, Shape::new(&[held, hidden]))?;
        let update = TensorView::new(region.ptr, dtype, rows);
        let residual = match residual {
            Some(residual) => residual,
            None => arena.tensor(dtype, rows)?,
        };
        Ok(StackBuffers {
            normalized,
            update,
            residual,
        })
    }

    /// Run every layer over `input`, answering the last layer's update with the residual stream
    /// it has not been added into; the caller's output norm or add performs that addition.
    ///
    /// `cosine` and `sine` are [`Self::factors`]' tables, `keys` the keys of each layer by its
    /// index, and `after` runs after each layer, answering the state the next layer reads: a
    /// consumer that adds features into the stream between layers (Qwen3-VL's DeepStack) does it
    /// there, and every other consumer answers its input.
    ///
    /// # Errors
    ///
    /// What a layer or `after` refuses.
    #[allow(
        clippy::too_many_arguments,
        reason = "the stack runs over the caller's buffers and factors, with its per-layer keys and \
                  hook"
    )]
    pub fn forward<'k>(
        &self,
        f: &Forward,
        input: Hidden,
        cosine: TensorView,
        sine: TensorView,
        keys: impl Fn(usize) -> Keys<'k>,
        buffers: &StackBuffers,
        mut after: impl FnMut(usize, Hidden) -> Result<Hidden>,
    ) -> Result<Hidden> {
        let rope = RopeFactors {
            rotary: &self.rotary,
            cosine,
            sine,
        };
        let mut state = input;
        for (index, layer) in self.layers.iter().enumerate() {
            // One layer's scratch is dead once its launches are enqueued: the next layer is handed
            // the same addresses, which is what a replayed graph expects. The stack's buffers were
            // reserved before this mark.
            let mark = f.arena().mark();
            state = f.fenced(
                self.components.layer,
                format_args!("layers.{index}"),
                || {
                    layer.forward(
                        f,
                        state,
                        rope,
                        keys(index),
                        buffers,
                        &self.components,
                        index,
                    )
                },
            )?;
            f.arena().release(mark);
            state = after(index, state)?;
        }
        Ok(state)
    }
}
