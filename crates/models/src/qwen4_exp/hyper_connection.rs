//! The gated hyper-connection: how a block reads and writes a residual of several streams
//! (vLLM's `Qwen4ExpHyperConnection`,
//! `vllm:vllm/models/qwen4_exp/nvidia/hyperconnection.py:56-231`, kernels
//! `vllm:vllm/models/qwen4_exp/nvidia/ops/hc.py`).
//!
//! The residual is `S`, `[tokens, C · H]`: `C` streams of the hidden width `H`. A block reads
//! one row of `H` mixed from the normalized streams and writes its output back into every
//! stream under a per-stream gain:
//!
//! ```text
//! norm      Sn_c = S_c · rsqrt(mean_H(S_c²) + eps) · (1 + w_c)             one gain row a stream
//! mix       d = Sn · downᵀ            [tokens, rank]
//!           r = Sn · injectᵀ          [tokens, C]        the block's combine logits
//!           g = silu(d / C) · upᵀ     [tokens, C · H]
//!           h = mean_c(sigmoid(g_c) ⊙ Sn_c)              the block's input
//! combine   S'_c = S_c + y ⊙ 2 · sigmoid(r_c / C)        y the block's output
//! ```
//!
//! Each line is computed in FP32 and rounded once into the activations' type where it stores
//! a tensor: `S'`, `Sn`, `d` and `r` (one product over `[down; inject]`), `silu(d / C)`, `g`
//! and `h`.
//!
//! ## The deferred combine
//!
//! As vLLM evaluates it (`hyperconnection.py:197-222`, `model.py:327-353`), a block's combine
//! is deferred into the next read: [`HyperConnection::forward`] takes the streams a block read
//! with that block's output and logits, folds them, and normalizes and mixes the folded
//! streams. The norm reads the rounded `S'`, so the arithmetic is the three lines above in
//! order.

use nn::modules::{Linear, LinearConfig, Parallel};
use nn::{DType, Result, Tensor, VarBuilder, functional};

use crate::qwen4_exp::ID;
use crate::qwen4_exp::config::Config;

/// A gated hyper-connection, its modules named as the checkpoint names them.
pub struct HyperConnection {
    /// `[C · H]`: the zero-centred gain of the per-stream norm, one row of `H` a stream.
    hc_norm: Tensor,
    /// `input_mix_weight_down`, with `block_inject_weight` stacked under it when the module
    /// writes: `[rank (+ C), C · H]`, one product for the gate's input and the combine logits.
    input_mix_weight_down: Linear,
    /// `input_mix_weight_up`: `[C · H, rank]`.
    input_mix_weight_up: Linear,
    /// The width of one stream, `H`.
    hidden: usize,
    /// The streams, `C`.
    streams: usize,
    /// The rank of the gate.
    rank: usize,
    eps: f32,
    /// Whether the module states combine logits; a module that only reads the streams out
    /// (the terminal mixers) states none.
    injects: bool,
}

impl HyperConnection {
    /// The hyper-connection at `vb`: `hc_norm`, `input_mix_weight_down`,
    /// `input_mix_weight_up` and, when `injects`, `block_inject_weight`. Every tensor is
    /// replicated on the ranks of a tensor-parallel group.
    ///
    /// # Errors
    ///
    /// A module the loader refuses.
    pub fn load(config: &Config, vb: VarBuilder<'_, '_>, injects: bool) -> Result<Self> {
        let (hidden, streams) = (config.text.hidden_size, config.hyper_connection.streams);
        let (width, rank) = (hidden * streams, config.hyper_connection.rank);

        let input_mix_weight_down = if injects {
            Linear::load_merged(
                vb.clone(),
                &["input_mix_weight_down", "block_inject_weight"],
                width,
                &[rank, streams],
                false,
            )?
        } else {
            projection(&vb, "input_mix_weight_down", width, rank)?
        };

        Ok(Self {
            hc_norm: vb.pp("hc_norm").get(width as usize, "weight")?,
            input_mix_weight_down,
            input_mix_weight_up: projection(&vb, "input_mix_weight_up", rank, width)?,
            hidden: hidden as usize,
            streams: streams as usize,
            rank: rank as usize,
            eps: config.text.rms_norm_eps,
            injects,
        })
    }

    /// `streams`, `[tokens, C · H]`, with a block's `output`, `[tokens, H]`, added into every
    /// stream under the gain `2 · sigmoid(logits_c / C)` of its combine `logits`,
    /// `[tokens, C]`, or under a gain of one without logits: `[tokens, C · H]` in the streams'
    /// element type (`hc.py:219-262`).
    ///
    /// # Errors
    ///
    /// Tensors of other extents.
    pub fn combine(
        streams: &Tensor,
        output: &Tensor,
        logits: Option<&Tensor>,
        count: usize,
    ) -> Result<Tensor> {
        let (tokens, width) = streams.dims2()?;
        let hidden = width / count.max(1);
        if count == 0 || hidden * count != width || output.dims() != [tokens, hidden] {
            return Err(ID.config(format!(
                "streams of {:?} and a block output of {:?} are not {count} streams of one \
                 width",
                streams.dims(),
                output.dims()
            )));
        }

        let wide = streams
            .to_dtype(DType::F32)?
            .reshape((tokens, count, hidden))?;
        let update = output.to_dtype(DType::F32)?.unsqueeze(1)?;
        let update = match logits {
            Some(logits) => {
                // The gain of stream `c` is twice the sigmoid of its logit over the stream
                // count.
                let gain = logits
                    .to_dtype(DType::F32)?
                    .affine(1.0 / count as f64, 0.0)?
                    .sigmoid()?
                    .affine(2.0, 0.0)?;

                update.broadcast_mul(&gain.unsqueeze(2)?)?
            }
            None => update,
        };

        Ok(wide
            .broadcast_add(&update)?
            .to_dtype(streams.dtype())?
            .reshape((tokens, width))?)
    }

    /// The streams normalized, each with its own gain: `[tokens, C · H]` (`hc.py:12-51`).
    ///
    /// # Errors
    ///
    /// Streams of another width, or the operator's refusal.
    pub fn norm(&self, streams: &Tensor) -> Result<Tensor> {
        let (tokens, width) = streams.dims2()?;
        if width != self.streams * self.hidden {
            return Err(ID.config(format!(
                "streams of {width} elements against {} streams of {}",
                self.streams, self.hidden
            )));
        }

        // One row a stream, so stream `c` of every token reads gain row `c`.
        let rows = streams.reshape((tokens * self.streams, self.hidden))?;
        let normalized =
            functional::rms_norm(&rows, &self.hc_norm, self.eps, true, streams.dtype())?;
        Ok(normalized.reshape((tokens, width))?)
    }

    /// The block input mixed from the `normalized` streams, `[tokens, H]`, and the block's
    /// combine logits, `[tokens, C]`, when the module states them (`hyperconnection.py:145-180`).
    ///
    /// # Errors
    ///
    /// Rows a projection refuses.
    pub fn mix(&self, normalized: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (tokens, _) = normalized.dims2()?;
        let dtype = normalized.dtype();
        let (streams, hidden) = (self.streams, self.hidden);

        // One product gives the gate's low-rank input and, under it, the combine logits.
        let both = self.input_mix_weight_down.forward(normalized)?;
        let (low, logits) = if self.injects {
            (
                both.narrow(1, 0, self.rank)?,
                Some(both.narrow(1, self.rank, streams)?.contiguous()?),
            )
        } else {
            (both, None)
        };

        let low = low
            .to_dtype(DType::F32)?
            .affine(1.0 / streams as f64, 0.0)?
            .silu()?
            .to_dtype(dtype)?;
        let gate = self.input_mix_weight_up.forward(&low)?;

        // The gated mean of the streams, accumulated in FP32.
        let opened = gate
            .to_dtype(DType::F32)?
            .sigmoid()?
            .reshape((tokens, streams, hidden))?;
        let mixed = opened
            .mul(
                &normalized
                    .to_dtype(DType::F32)?
                    .reshape((tokens, streams, hidden))?,
            )?
            .mean(1)?
            .to_dtype(dtype)?;
        Ok((mixed, logits))
    }

    /// One read of the streams with the block before it folded in: `(streams, input, logits)`.
    ///
    /// `pending` is the output and the combine logits of the block that read `streams` last,
    /// or `None` for streams nothing is pending on. Returns the folded streams (which the
    /// next read takes), the input of this module's block, `[tokens, H]`, and the block's
    /// combine logits, which the next read takes with the block's output.
    ///
    /// # Errors
    ///
    /// Tensors of other extents, or what a projection refuses.
    pub fn forward(
        &self,
        streams: &Tensor,
        pending: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let streams = match pending {
            Some((output, logits)) => Self::combine(streams, output, Some(logits), self.streams)?,
            None => streams.clone(),
        };
        let (input, logits) = self.mix(&self.norm(&streams)?)?;

        Ok((streams, input, logits))
    }
}

/// The projection `module` under `vb`, `[outputs, inputs]` without a bias, whole on the rank.
fn projection(vb: &VarBuilder<'_, '_>, module: &str, inputs: u32, outputs: u32) -> Result<Linear> {
    Linear::load(
        &LinearConfig {
            inputs,
            outputs,
            bias: false,
            parallel: Parallel::None,
        },
        vb.pp(module),
    )
}

impl std::fmt::Debug for HyperConnection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HyperConnection")
            .field("streams", &self.streams)
            .field("hidden", &self.hidden)
            .field("rank", &self.rank)
            .field("injects", &self.injects)
            .finish_non_exhaustive()
    }
}
