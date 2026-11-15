//! The gated delta network: a linear-attention token mixer whose state a request slot keeps.
//!
//! The module is Qwen3-Next's and Qwen3.5's `GatedDeltaNet` (Hugging Face's
//! `Qwen3_5MoeGatedDeltaNet`, vLLM's and SGLang's `Qwen3_5GatedDeltaNet`):
//!
//! ```text
//! qkv = in_proj_qkv(x)                       [q | k | v], key heads, key heads, value heads
//! qkv = causal_conv1d(qkv)                   depthwise, SiLU, continuing from the slot's inputs
//! o   = gated_delta_rule(q, k, v, a = in_proj_a(x), b = in_proj_b(x))
//! o   = rms_norm(o) · weight · act(in_proj_z(x))      per value head, act SiLU or sigmoid
//! out = out_proj(o)
//! ```
//!
//! Both recurrent stages read and advance the state the batch's request slots keep: field 0 of
//! a layer's buffers is the convolution's last inputs, field 1 the delta rule's matrices
//! ([`GatedDeltaNetConfig::state_fields`]).
//!
//! ## Tensor parallelism
//!
//! A rank of a tensor-parallel group holds a contiguous run of the key heads and the value
//! heads they serve (vLLM's shards, `refs/vllm/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py:27-65`):
//! its rows of `in_proj_qkv` (three runs, one of each of `q`, `k` and `v`), of `in_proj_z`,
//! `in_proj_b` and `in_proj_a`, its channels of the convolution, its entries of `A_log` and
//! `dt_bias`, and its columns of `out_proj`, whose partial products the group sums. The state
//! is the rank's heads' own.

use tensor::{DType, Tensor};

use crate::capability::{RecurrentState, TokenBatch};
use crate::collective::Split;
use crate::error::{Error, Result};
use crate::family::StateField;
use crate::functional::{self, StateAxes, WeightOrder};
use crate::modules::linear::{Linear, LinearConfig, Parallel};
use crate::modules::norm::{GatedRmsNorm, GatedRmsNormConfig};
use crate::var_builder::VarBuilder;
use crate::weights::{PartBlock, WeightLayout, WeightSource, extents};

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
    /// Whether the output norm's gate passes through a sigmoid in place of the SiLU
    /// (`output_gate_type: "sigmoid"`).
    pub sigmoid_gate: bool,
    /// The order of a stored state's matrix axes.
    pub axes: StateAxes,
    /// The tokens a step of the deployment drafts: the rows a slot's convolution window holds
    /// beyond its history ([`crate::capability::VerifySlots`]); zero for a deployment whose
    /// requests own one slot.
    pub draft_tokens: u32,
    /// The tensor-parallel split of the heads.
    pub split: Split,
}

impl GatedDeltaNetConfig {
    /// Channels the short convolution mixes on this rank: its queries, keys and values.
    pub const fn conv_channels(&self) -> u32 {
        (2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim) / self.split.size()
    }

    /// The state one layer keeps per request slot on this rank, in the order its buffers are
    /// lent to it: the convolution's window in the activations' type `dtype`, then the delta
    /// rule's FP32 matrices in the order [`Self::axes`] states.
    ///
    /// A deployment that drafts keeps the matrices after each token of a decode-phase row in a
    /// slot of their own (`stepwise`), and the window's draft rows inside its one slot.
    pub fn state_fields(&self, dtype: DType) -> Vec<StateField> {
        let heads = (self.value_heads / self.split.size()) as usize;
        let (key, value) = (self.key_dim as usize, self.value_dim as usize);

        vec![
            StateField {
                name: "conv",
                dtype,
                dims: vec![
                    (self.conv_width.saturating_sub(1) + self.draft_tokens) as usize,
                    self.conv_channels() as usize,
                ],
                stepwise: false,
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: match self.axes {
                    StateAxes::KeyValue => vec![heads, key, value],
                    StateAxes::ValueKey => vec![heads, value, key],
                },
                stepwise: self.draft_tokens > 0,
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
    /// The network the module at `vb` states, at this rank's share of its heads: its four
    /// input projections, the depthwise `conv1d.weight` of `[channels, 1, width]`, the
    /// per-head `A_log` and `dt_bias`, the output `norm` and `out_proj`.
    ///
    /// # Errors
    ///
    /// A module the loader refuses, value heads the key heads do not group, head widths the
    /// convolved row is not whole heads of, or a split that does not divide the key heads.
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

        let split = config.split;
        if !config.key_heads.is_multiple_of(split.size()) {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "rank {} of a tensor group of {} does not hold a share of {} key heads",
                    split.rank(),
                    split.size(),
                    config.key_heads
                ),
            ));
        }

        // The delta rule reads the queries, keys and values as three runs of heads of one
        // convolved row, each a view of the row's own elements ([`heads_of_rows`]).
        let (key_width, value_width) = (
            config.key_heads * config.key_dim,
            config.value_heads * config.value_dim,
        );
        let (local_key, local_value) = (key_width / split.size(), value_width / split.size());
        if config.key_dim == 0
            || config.value_dim == 0
            || !local_value.is_multiple_of(config.key_dim)
            || !(2 * local_key).is_multiple_of(config.value_dim)
        {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "heads of {} and {} elements do not both divide a convolved row of {} query \
                     and key channels and {local_value} value channels",
                    config.key_dim,
                    config.value_dim,
                    2 * local_key
                ),
            ));
        }

        let parallel = |parallel: fn(Split) -> Parallel| {
            if split.size() == 1 {
                Parallel::None
            } else {
                parallel(split)
            }
        };
        let projection = |module: &str, inputs: u32, outputs: u32, parallel: Parallel| {
            Linear::load(
                &LinearConfig {
                    inputs,
                    outputs,
                    bias: false,
                    parallel,
                },
                vb.pp(module),
            )
        };

        // This rank's runs of the `[q | k | v]` rows: `rows` elements a row, in a tensor whose
        // rows are `row` elements apart.
        let rank = u64::from(split.rank());
        let (local_key, local_value) = (u64::from(local_key), u64::from(local_value));
        let runs = |row: u64| -> Vec<PartBlock> {
            [
                (rank * local_key, local_key),
                (u64::from(key_width) + rank * local_key, local_key),
                (2 * u64::from(key_width) + rank * local_value, local_value),
            ]
            .into_iter()
            .map(|(first, rows)| PartBlock {
                start: first * row,
                rows,
                columns: row,
                row_stride: row,
            })
            .collect()
        };
        let channels = u64::from(config.conv_channels());
        let gathered = |module: &str, row: u64| -> Result<Tensor> {
            let name = vb.pp(module).name("weight");
            let parts = runs(row)
                .into_iter()
                .map(|block| (name.clone(), block))
                .collect();
            let shape = extents(&[channels, row]);

            vb.load(
                &name,
                &WeightSource::Blocks {
                    parts,
                    shape: shape.clone(),
                },
                WeightLayout::ROW_MAJOR,
                &shape,
                false,
            )
        };

        let hidden = u64::from(config.hidden);
        let in_proj_qkv = if split.size() == 1 {
            projection(
                "in_proj_qkv",
                config.hidden,
                2 * key_width + value_width,
                Parallel::None,
            )?
        } else {
            Linear::new(gathered("in_proj_qkv", hidden)?, None, WeightOrder::NxK)?
        };
        let column = parallel(Parallel::Column);
        let in_proj_z = projection("in_proj_z", config.hidden, value_width, column)?;
        let in_proj_b = projection("in_proj_b", config.hidden, config.value_heads, column)?;
        let in_proj_a = projection("in_proj_a", config.hidden, config.value_heads, column)?;

        // PyTorch's depthwise `Conv1d` stores one input channel per group; the taps are its
        // rows.
        let width = u64::from(config.conv_width);
        let conv1d = if split.size() == 1 {
            vb.pp("conv1d")
                .get((channels as usize, 1, width as usize), "weight")?
                .reshape((channels as usize, width as usize))?
        } else {
            gathered("conv1d", width)?
        };

        let heads = (config.value_heads / split.size()) as usize;
        let first = split.rank() as usize * heads;
        let a_log = vb.get_rows(heads, "A_log", first)?;
        let dt_bias = vb.get_rows(heads, "dt_bias", first)?;

        let norm = GatedRmsNorm::load(
            &GatedRmsNormConfig {
                width: config.value_dim,
                eps: config.eps,
                sigmoid: config.sigmoid_gate,
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
            out_proj: projection(
                "out_proj",
                value_width,
                config.hidden,
                parallel(Parallel::Row),
            )?,
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
    /// type, summed over the tensor-parallel group.
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
        let ranks = config.split.size();
        let (key_heads, key_dim) = ((config.key_heads / ranks) as usize, config.key_dim as usize);
        let (value_heads, value_dim) = (
            (config.value_heads / ranks) as usize,
            config.value_dim as usize,
        );

        let qkv = self.in_proj_qkv.forward(hidden)?;
        let z = self.in_proj_z.forward(hidden)?;
        let b = self.in_proj_b.forward(hidden)?;
        let a = self.in_proj_a.forward(hidden)?;

        let mixed = functional::causal_conv1d(
            &qkv,
            &self.conv1d,
            1,
            true,
            batch,
            state,
            layer,
            CONV_FIELD,
        )?;

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
            config.axes,
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
