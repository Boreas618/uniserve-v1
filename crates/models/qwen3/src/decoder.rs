//! The Qwen3 decoder layer and the stack of them, which every Qwen3 model composes.
//!
//! The layer is Hugging Face's `Qwen3DecoderLayer`, with the feed-forward block of
//! `Qwen3MoeDecoderLayer` and the token mixer and shared expert of `Qwen3_5MoeDecoderLayer`:
//!
//! under vLLM's residual contract (`Qwen3DecoderLayer.forward(positions, hidden_states, residual)`):
//!
//! ```text
//! residual None:     residual = hidden;            x = input_layernorm(hidden)
//! residual Some(r):  x, residual = fused_add_rms_norm(hidden, r, input_layernorm)
//! mixer:             hidden = self_attn(x)          (q/k RMSNorm fused into the rotation)
//!                 or hidden = linear_attn(x)        (a gated delta network over the slot's state)
//!                    x, residual = fused_add_rms_norm(hidden, residual, post_attention_layernorm)
//! feed-forward:      hidden = mlp(x)                (gated MLP, or routed experts and a shared one)
//! ```
//!
//! A layer answers its MLP output unsummed with the residual stream; the next layer's input norm,
//! or the model's final norm, adds it in the same launch that normalizes the sum. The residual
//! stream accumulates in place in the stack's input tensor.
//!
//! The MLP output crosses from one layer to the next, so it lands in one `[tokens, hidden]` buffer
//! the stack reserves before its first layer ([`Mlp::forward_into`]); everything else a layer
//! computes is reserved inside the layer's scope ([`nn::Forward::scoped`]) and handed back once its
//! launches are enqueued, so every layer of a stack reuses the same addresses.
//!
//! The stack does not own a rotation: a model turns its queries and keys by the factors of its
//! positions ([`rotary`]), and may hold more than one spectrum (a text batch and an image batch of
//! a Qwen3.5 model turn by different spectra of the same rotary width).

use std::ops::Range;

use nn::blocks::{
    AttentionKind, GateUp, GatedDeltaNet, GatedMlp, GatedMlpSpec, Keys, QkNormNames, QkvAttention,
    QkvAttentionSpec, RopeFactors, SharedExpert,
};
use nn::family::Split;
use nn::layers::{
    Bias, Moe, QkNorm, QkvBias, QkvNames, QkvSpec, ResidualStream, RmsNorm, Rotary, RotarySpec,
    Rounding, Spectrum,
};
use nn::weights::Scope;
use nn::{DType, Forward, Shape, TensorView};

use crate::config::{Config, Mixer, TextConfig};
use crate::{Error, Result};

/// The fence class of one decoder layer, at path `<stack>.layers.<i>`, which the deployment's
/// fence patterns select (`docs/profiling.md`).
pub const LAYER: &str = "models.qwen3.transformer.TransformerLayer";
/// The fence class of a layer's token mixer, at path `<layer>.self_attn` or `<layer>.linear_attn`.
pub const ATTENTION: &str = "models.qwen3.transformer.Attention";
/// The fence class of a layer's feed-forward block, at path `<layer>.mlp`.
pub const MLP: &str = "models.qwen3.transformer.MoE";

/// A layer's feed-forward block.
#[allow(
    clippy::large_enum_variant,
    reason = "one block per layer, held for the model's lifetime; the routed block is the common one"
)]
pub enum Mlp {
    /// `down(act(gate(x)) · up(x))`, column parallel in and row parallel out.
    Dense(GatedMlp),
    /// Routed experts, and the gated shared expert every token also passes through when the
    /// checkpoint has one.
    Experts {
        routed: Moe,
        shared: Option<SharedExpert>,
    },
}

impl Mlp {
    /// The dense gated MLP of `text` at `scope` (`gate_proj`, `up_proj`, `down_proj`).
    ///
    /// # Errors
    ///
    /// A module the loader refuses or a split that does not divide the intermediate width.
    pub fn dense(scope: &mut Scope<'_, '_>, text: &TextConfig, split: Split) -> Result<Self> {
        Ok(Self::Dense(GatedMlp::load(
            scope,
            gated_mlp(text.hidden_size, text.intermediate_size, text.hidden_act),
            split,
        )?))
    }

    /// Whether the block's output is a packed product that writes whole row classes, so the
    /// buffer [`Self::forward_into`] writes must hold the class of its rows.
    fn is_packed(&self) -> bool {
        matches!(self, Self::Dense(mlp) if mlp.is_packed())
    }

    /// The block's output over `x`, `[rows, hidden]`, written into `update`.
    fn forward_into(&self, f: &Forward, x: TensorView, update: TensorView) -> Result<()> {
        match self {
            Self::Dense(mlp) => mlp.forward_into(f, x, update)?,
            Self::Experts { routed, shared } => {
                routed.forward_into(f, x, update)?;
                if let Some(shared) = shared {
                    shared.forward(f, x, update)?;
                }
            }
        }
        Ok(())
    }
}

/// The spec of a Qwen3 gated MLP of `hidden → intermediate → hidden` gated by `activation`, named
/// as the checkpoint names it.
///
/// The product `act(gate) · up` is one FP32 expression rounded once, as the references' fused
/// `SiluAndMul` evaluates it.
pub(crate) fn gated_mlp(
    hidden: u32,
    intermediate: u32,
    activation: ops::activation::Activation,
) -> GatedMlpSpec<'static> {
    GatedMlpSpec {
        hidden,
        intermediate,
        activation,
        rounding: Rounding::Fused,
        gate_up: GateUp::Split {
            gate: "gate_proj",
            up: "up_proj",
        },
        down: "down_proj",
        bias: Bias::Absent,
    }
}

/// A layer's token mixer.
#[allow(
    clippy::large_enum_variant,
    reason = "one mixer per layer, held for the model's lifetime and read on every forward"
)]
pub enum TokenMixer {
    /// Q/K/V column parallel over the heads, per-head gains fused into the rotation, and the
    /// row-parallel output projection (`self_attn`).
    Attention(QkvAttention),
    /// A gated delta network over the request slots' recurrent state (`linear_attn`).
    Linear(GatedDeltaNet),
}

/// One decoder layer, its modules named as the checkpoint names them.
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    mixer: TokenMixer,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    /// The layer's module path, which names it to the fence probes.
    path: String,
}

impl DecoderLayer {
    /// Layer `layer` of `config` at `scope` (`<stack>.layers.<layer>`), at `split`'s rank of the
    /// tensor-parallel group, its norms writing `dtype`, attending with `attention` when its mixer
    /// is attention, and its feed-forward block read by `mlp` from the layer's `mlp` scope.
    ///
    /// # Errors
    ///
    /// A module the loader refuses or a split that does not divide what it shards.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        config: &Config,
        layer: u32,
        attention: AttentionKind,
        split: Split,
        dtype: DType,
        mlp: &mut dyn FnMut(u32, &mut Scope<'_, '_>) -> Result<Mlp>,
    ) -> Result<Self> {
        let text = &config.text;
        let norm = |scope: &mut Scope<'_, '_>, module: &str| {
            RmsNorm::load(
                &mut scope.scope(module),
                text.hidden_size,
                text.rms_norm_eps,
                text.rms_norm_gain,
                dtype,
            )
        };
        let input_layernorm = norm(scope, "input_layernorm")?;
        let post_attention_layernorm = norm(scope, "post_attention_layernorm")?;
        let mixer = match config.mixers[layer as usize] {
            Mixer::Linear(spec) => TokenMixer::Linear(GatedDeltaNet::load(
                &mut scope.scope("linear_attn"),
                spec,
                split,
            )?),
            Mixer::Attention => TokenMixer::Attention(QkvAttention::load(
                &mut scope.scope("self_attn"),
                QkvAttentionSpec {
                    qkv: QkvSpec {
                        hidden: text.hidden_size,
                        heads: text.num_attention_heads,
                        kv_heads: text.num_key_value_heads,
                        head_dim: text.head_dim,
                        gated: text.attn_output_gate,
                    },
                    names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
                    bias: if text.attention_bias {
                        QkvBias::Present
                    } else {
                        QkvBias::Absent
                    },
                    output: "o_proj",
                    output_bias: if text.attention_bias {
                        Bias::Present
                    } else {
                        Bias::Absent
                    },
                    qk_norm: QkNormNames::Fused {
                        query: "q_norm",
                        key: "k_norm",
                    },
                    kind: attention,
                    scale: text.rope.score_scale(text.head_dim)?,
                },
                split,
            )?),
        };
        let mlp = mlp(layer, &mut scope.scope("mlp"))?;
        Ok(Self {
            input_layernorm,
            mixer,
            post_attention_layernorm,
            mlp,
            path: scope.path().to_owned(),
        })
    }

    /// The layer's feed-forward block.
    pub const fn mlp(&self) -> &Mlp {
        &self.mlp
    }

    /// Run the layer over `hidden`, `[tokens, hidden]`, with the residual stream it has not been
    /// added into (`None` for the first layer, whose `hidden` becomes the stream), and answer
    /// `(hidden, residual)`: the layer's MLP output, written into `update`, and the stream, updated
    /// in place.
    ///
    /// `rope` turns the attention's queries and keys; `keys` is what the mixer attends to or
    /// advances. `update` may be `hidden` itself: the input norm has consumed it before the MLP
    /// writes it. Every other intermediate is handed back before return.
    ///
    /// # Errors
    ///
    /// Keys of another kind than the layer's mixer, or a view a layer refuses.
    pub fn forward(
        &self,
        f: &Forward,
        (hidden, residual): (TensorView, Option<TensorView>),
        update: TensorView,
        rope: RopeFactors<'_>,
        keys: Keys<'_>,
    ) -> Result<(TensorView, TensorView)> {
        f.scoped(|f| {
            let (normalized, residual) = match residual {
                None => (self.input_layernorm.forward(f, hidden, None)?, hidden),
                Some(residual) => (
                    self.input_layernorm.forward(
                        f,
                        hidden,
                        Some(ResidualStream::in_place(residual)),
                    )?,
                    residual,
                ),
            };
            let attended = match &self.mixer {
                TokenMixer::Attention(attention) => {
                    f.fenced(ATTENTION, format_args!("{}.self_attn", self.path), || {
                        Ok::<_, Error>(attention.forward(f, normalized, Some(rope), keys)?)
                    })?
                }
                TokenMixer::Linear(delta) => {
                    let Keys::Recurrent {
                        batch,
                        cache,
                        layer,
                    } = keys
                    else {
                        return Err(Error::Config(format!(
                            "{} is a gated delta network and was handed no recurrent state",
                            self.path
                        )));
                    };
                    f.fenced(ATTENTION, format_args!("{}.linear_attn", self.path), || {
                        Ok::<_, Error>(delta.forward(
                            f,
                            normalized,
                            batch,
                            cache.state()?,
                            layer,
                        )?)
                    })?
                }
            };
            let normalized = self.post_attention_layernorm.forward(
                f,
                attended,
                Some(ResidualStream::in_place(residual)),
            )?;
            f.fenced(MLP, format_args!("{}.mlp", self.path), || {
                self.mlp.forward_into(f, normalized, update)
            })?;
            Ok((update, residual))
        })
    }
}

/// The rotary embedding of a stack of `text`'s heads at `split`'s rank, turning by `spectrum`, its
/// per-head q/k RMSNorm fused at the stack's epsilon and gain, publishing its spectrum under
/// `scope`.
///
/// # Errors
///
/// A split that does not divide the heads, or a spectrum the rotation refuses.
pub fn rotary(
    scope: &mut Scope<'_, '_>,
    text: &TextConfig,
    spectrum: Spectrum,
    split: Split,
) -> Result<Rotary> {
    let (heads, kv_heads) = text.local_heads(split)?;
    Ok(Rotary::load(
        scope,
        spectrum,
        RotarySpec {
            head_dim: text.head_dim,
            heads,
            kv_heads,
            qk_norm: QkNorm::Fused {
                epsilon: text.rms_norm_eps,
                gain: text.rms_norm_gain,
            },
        },
    )?)
}

/// A stack of decoder layers.
pub struct DecoderStack {
    layers: Vec<DecoderLayer>,
}

impl DecoderStack {
    /// Layers `layers` of `config` under `scope` (`<scope>.layers.<i>`), as
    /// [`DecoderLayer::load`] reads each. A consumer that truncates the checkpoint's stack loads a
    /// prefix of its layers.
    ///
    /// # Errors
    ///
    /// An empty range or one past the checkpoint's layers, or what a layer's load refuses.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        config: &Config,
        layers: Range<u32>,
        attention: AttentionKind,
        split: Split,
        dtype: DType,
        mlp: &mut dyn FnMut(u32, &mut Scope<'_, '_>) -> Result<Mlp>,
    ) -> Result<Self> {
        if layers.is_empty() || layers.end as usize > config.mixers.len() {
            return Err(Error::Config(format!(
                "decoder layers {layers:?} of a {}-layer stack",
                config.mixers.len()
            )));
        }
        let layers = layers
            .map(|layer| {
                DecoderLayer::load(
                    &mut scope.scope(format!("layers.{layer}")),
                    config,
                    layer,
                    attention,
                    split,
                    dtype,
                    mlp,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    /// The layers, in stack order.
    pub fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }

    /// Run every layer over `hidden`, `[tokens, hidden]`, and answer the last layer's
    /// `(hidden, residual)`: its MLP output and the residual stream it has not been added into,
    /// which the caller's final norm adds. The stream accumulates in place in `hidden`.
    ///
    /// The MLP output each layer hands the next is one buffer reserved here before the first layer,
    /// at the packed row class when a layer's product writes whole classes. `keys` answers what
    /// layer `index` attends to or advances; `after` runs after each layer over its MLP output,
    /// where Qwen3-VL's DeepStack features join it. Each layer is fenced at its module path.
    ///
    /// # Errors
    ///
    /// What a layer, `keys` or `after` refuses, or an arena that cannot hold the carried output.
    pub fn forward<'k>(
        &self,
        f: &Forward,
        hidden: TensorView,
        rope: RopeFactors<'_>,
        keys: impl Fn(usize) -> Result<Keys<'k>>,
        mut after: impl FnMut(usize, TensorView) -> Result<()>,
    ) -> Result<(TensorView, TensorView)> {
        let update = self.carried(f, hidden)?;
        let (mut hidden, mut residual) = (hidden, None);
        for (index, layer) in self.layers.iter().enumerate() {
            let keys = keys(index)?;
            let (output, stream) = f.fenced(LAYER, format_args!("{}", layer.path), || {
                layer.forward(f, (hidden, residual), update, rope, keys)
            })?;
            after(index, output)?;
            (hidden, residual) = (output, Some(stream));
        }
        let residual =
            residual.ok_or_else(|| Error::Config("a decoder stack of no layers".to_owned()))?;
        Ok((hidden, residual))
    }

    /// The buffer the layers' MLP outputs are carried in: `[tokens, hidden]` of `hidden`'s type,
    /// at the start of a region of the packed row class when a layer's product is packed.
    fn carried(&self, f: &Forward, hidden: TensorView) -> Result<TensorView> {
        let tokens = hidden.shape.dim(0);
        let held = if self.layers.iter().any(|layer| layer.mlp.is_packed()) {
            let counted = u32::try_from(tokens)
                .map_err(|_| Error::Config(format!("{tokens} activation rows")))?;
            u64::from(ops::fp4::packed_rows(counted)?)
        } else {
            tokens
        };
        let region = f.empty(hidden.dtype, Shape::new(&[held, hidden.shape.dim(1)]))?;
        Ok(region.narrow(0, 0, tokens)?)
    }
}
