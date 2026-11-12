//! The gated delta network: a linear-attention token mixer whose state a request slot keeps.
//!
//! The module is Qwen3-Next's and Qwen3.5's `GatedDeltaNet` (Hugging Face's
//! `Qwen3_5MoeGatedDeltaNet`, vLLM's and SGLang's `Qwen3_5GatedDeltaNet`):
//!
//! ```text
//! qkv = in_proj_qkv(x)                       [q | k | v], key heads, key heads, value heads
//! qkv = causal_conv1d(qkv)                   depthwise, SiLU, continuing from the slot's inputs
//! o   = gated_delta_rule(q, k, v, a = in_proj_a(x), b = in_proj_b(x))
//! o   = rms_norm(o) · weight · silu(in_proj_z(x))     per value head
//! out = out_proj(o)
//! ```
//!
//! Both recurrent stages read and advance the state the batch's request slots keep: field 0 of
//! a layer's buffers is the convolution's last inputs, field 1 the delta rule's matrices
//! ([`GatedDeltaNetConfig::state_fields`]).

use uniserve_tensor::{DType, Tensor};

use crate::capability::{RecurrentState, TokenBatch};
use crate::collective::Split;
use crate::error::{Error, Result};
use crate::family::StateField;
use crate::functional;
use crate::modules::linear::{Linear, LinearConfig, Parallel};
use crate::modules::norm::{GatedRmsNorm, GatedRmsNormConfig};
use crate::var_builder::VarBuilder;

/// The field of a layer's state that holds the convolution's last inputs.
const CONV_FIELD: usize = 0;

/// The field of a layer's state that holds the delta rule's matrices.
const RULE_FIELD: usize = 1;

/// The widths of a [`GatedDeltaNet`], before any split, and this rank's share of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaNetConfig {
    /// The hidden width the input projections read and the output projection writes.
    pub hidden: u32,
    /// Query and key heads.
    pub key_heads: u32,
    /// Value heads, each with its own state; a multiple of the key heads.
    pub value_heads: u32,
    /// Elements of a query or key head.
    pub key_dim: u32,
    /// Elements of a value head.
    pub value_dim: u32,
    /// Taps of the short convolution.
    pub conv_width: u32,
    /// The output norm's variance floor.
    pub eps: f32,
    /// The tensor-parallel split; a gated delta network is served on one rank.
    pub split: Split,
}

impl GatedDeltaNetConfig {
    /// Channels the short convolution mixes: the queries, the keys and the values.
    pub const fn conv_channels(&self) -> u32 {
        2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
    }

    /// The state one layer keeps per request slot, in the order its buffers are lent to it:
    /// the convolution's last inputs in the activations' type `dtype`, then the delta rule's
    /// FP32 matrices.
    pub fn state_fields(&self, dtype: DType) -> Vec<StateField> {
        vec![
            StateField {
                name: "conv",
                dtype,
                dims: vec![
                    self.conv_width.saturating_sub(1) as usize,
                    self.conv_channels() as usize,
                ],
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: vec![
                    self.value_heads as usize,
                    self.key_dim as usize,
                    self.value_dim as usize,
                ],
            },
        ]
    }
}

/// A gated delta network.
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    /// `[channels, width]`: each channel's taps of the short convolution, oldest input first.
    conv1d: Tensor,
    /// `[value_heads]`: the log of each head's decay rate.
    a_log: Tensor,
    /// `[value_heads]`: the bias of each head's time step.
    dt_bias: Tensor,
    norm: GatedRmsNorm,
    out_proj: Linear,
    config: GatedDeltaNetConfig,
}

impl GatedDeltaNet {
    /// The network the module at `vb` states: its four input projections, the depthwise
    /// `conv1d.weight` of `[channels, 1, width]`, the per-head `A_log` and `dt_bias`, the
    /// output `norm` and `out_proj`.
    ///
    /// # Errors
    ///
    /// A module the loader refuses, value heads the key heads do not group, head widths the
    /// convolved row is not whole heads of, or a split of more than one rank: the
    /// convolution's channels interleave three head windows, which no loaded layout shards.
    pub fn load(config: &GatedDeltaNetConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.key_heads == 0 || !config.value_heads.is_multiple_of(config.key_heads) {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "{} value heads do not group over {} key heads",
                    config.value_heads, config.key_heads
                ),
            ));
        }

        // The delta rule reads the queries, keys and values as three runs of heads of one
        // convolved row, each a view of the row's own elements ([`heads_of_rows`]).
        let (key_width, value_width) = (
            config.key_heads * config.key_dim,
            config.value_heads * config.value_dim,
        );
        if config.key_dim == 0
            || config.value_dim == 0
            || !value_width.is_multiple_of(config.key_dim)
            || !(2 * key_width).is_multiple_of(config.value_dim)
        {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "heads of {} and {} elements do not both divide a convolved row of {} query \
                     and key channels and {value_width} value channels",
                    config.key_dim,
                    config.value_dim,
                    2 * key_width
                ),
            ));
        }
        if config.split.size != 1 {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "a gated delta network is served on one tensor rank, not {}",
                    config.split.size
                ),
            ));
        }

        let channels = config.conv_channels();
        let projection = |module: &str, inputs: u32, outputs: u32| {
            Linear::load(
                &LinearConfig {
                    inputs,
                    outputs,
                    bias: false,
                    parallel: Parallel::None,
                },
                vb.pp(module),
            )
        };

        let in_proj_qkv = projection("in_proj_qkv", config.hidden, channels)?;
        let in_proj_z = projection("in_proj_z", config.hidden, value_width)?;
        let in_proj_b = projection("in_proj_b", config.hidden, config.value_heads)?;
        let in_proj_a = projection("in_proj_a", config.hidden, config.value_heads)?;

        // PyTorch's depthwise `Conv1d` stores one input channel per group; the taps are its
        // rows.
        let taps = (channels as usize, config.conv_width as usize);
        let conv1d = vb
            .pp("conv1d")
            .get((taps.0, 1, taps.1), "weight")?
            .reshape(taps)?;

        let heads = config.value_heads as usize;
        let a_log = vb.get(heads, "A_log")?;
        let dt_bias = vb.get(heads, "dt_bias")?;

        let norm = GatedRmsNorm::load(
            &GatedRmsNormConfig {
                width: config.value_dim,
                eps: config.eps,
            },
            vb.pp("norm"),
        )?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            a_log,
            dt_bias,
            norm,
            out_proj: projection("out_proj", value_width, config.hidden)?,
            config: *config,
        })
    }

    /// The output projection, whose representation a caller may ask about (a packed product
    /// writes whole row classes).
    pub fn output(&self) -> &Linear {
        &self.out_proj
    }

    /// `hidden`, `[tokens, hidden]`, mixed through the network, reading and advancing
    /// recurrent layer `layer` of `state` in place: `[tokens, hidden]` in `hidden`'s element
    /// type.
    ///
    /// `batch` is the batch the tokens are of: its phase decides how the delta rule advances
    /// its rows ([`functional::gated_delta_rule`]).
    ///
    /// # Errors
    ///
    /// A state whose layer does not hold the network's two fields, or tensors a projection or
    /// a function refuses.
    pub fn forward(
        &self,
        hidden: &Tensor,
        batch: &TokenBatch,
        state: &RecurrentState,
        layer: usize,
    ) -> Result<Tensor> {
        const OWNER: &str = "GatedDeltaNet";

        let &[tokens, _] = hidden.dims() else {
            return Err(Error::shape(
                OWNER,
                format!("{:?} is not [tokens, hidden]", hidden.dims()),
            ));
        };

        // The layer keeps its two fields in each of the state's buffers.
        let trail = state.trail.iter();
        for (name, buffers) in [("live", &state.live), ("checkpoint", &state.checkpoint)]
            .into_iter()
            .chain(trail.map(|trail| ("trail", trail)))
        {
            let fields = buffers.get(layer).map_or(0, Vec::len);
            if fields != 2 {
                return Err(Error::shape(
                    OWNER,
                    format!("recurrent layer {layer} holds {fields} {name} fields, two expected"),
                ));
            }
        }

        let config = self.config;
        let (key_heads, key_dim) = (config.key_heads as usize, config.key_dim as usize);
        let (value_heads, value_dim) = (config.value_heads as usize, config.value_dim as usize);

        let qkv = self.in_proj_qkv.forward(hidden)?;
        let z = self.in_proj_z.forward(hidden)?;
        let b = self.in_proj_b.forward(hidden)?;
        let a = self.in_proj_a.forward(hidden)?;

        let mixed = functional::causal_conv1d(&qkv, &self.conv1d, state, layer, CONV_FIELD)?;

        // The convolved row is [q | k | v]: three runs of heads of one row.
        let key_width = key_heads * key_dim;
        let query = heads_of_rows(&mixed, 0, key_heads, key_dim)?;
        let key = heads_of_rows(&mixed, key_width, key_heads, key_dim)?;
        let value = heads_of_rows(&mixed, 2 * key_width, value_heads, value_dim)?;

        let core = functional::gated_delta_rule(
            &query,
            &key,
            &value,
            &a,
            &b,
            &self.a_log,
            &self.dt_bias,
            batch,
            state,
            layer,
            RULE_FIELD,
        )?;

        // The output norm runs per value head, gated by the same head's slice of `z`.
        let per_head = (tokens * value_heads, value_dim);
        let normed = self
            .norm
            .forward(&core.reshape(per_head)?, &z.reshape(per_head)?)?;

        self.out_proj
            .forward(&normed.reshape((tokens, value_heads * value_dim))?)
    }
}

/// The `heads` heads of `dim` elements that start at column `first` of each row of `rows`,
/// `[tokens, width]`, as a view of the rows: `[tokens, heads, dim]`, its rows one row of
/// `rows` apart.
///
/// The delta rule's kernel reads its three inputs at one row stride, so the heads are views
/// and not copies. The row is viewed as whole `dim`-wide heads and the run taken of them,
/// which is why a network's head widths divide its convolved row.
fn heads_of_rows(rows: &Tensor, first: usize, heads: usize, dim: usize) -> Result<Tensor> {
    let (tokens, width) = rows.dims2()?;

    Ok(rows
        .reshape((tokens, width / dim, dim))?
        .narrow(1, first / dim, heads)?)
}
