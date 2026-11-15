//! The diffusion transformer: the packed rows in, each modality's velocity rows out.
//!
//! One [`Transformer`] is one network of the checkpoint: a projection of each modality's latent
//! rows into the hidden width, the stack of blocks, the final modulated norm and the two output
//! heads. A block is an [`AdaLnBlock`] around the network's attention ([`Attention`]): dense
//! over the packed sequence for the base release, video sparse attention for a FastH3 student.
//!
//! ```text
//! video rows  = patchify(latent)  · proj_in        FP32, narrowed once into the stack's type
//! audio rows  = latent rows       · audio_proj_in  FP32, narrowed once
//! hidden      = block_49(… block_0(hidden))        each conditioned by the step's modulation
//! h           = norm_out(hidden), widened to FP32
//! velocity    = h · proj_out,  h · audio_proj_out  FP32
//! ```
//!
//! ## The two crossings
//!
//! A trajectory is FP32 and the stack computes in the checkpoint's element type. The four end
//! projections are FP32 parameters at every preset, as the reference builds them
//! (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:556-575`), so a sample crosses to
//! the stack's type once after its input projection and back once before the output heads, and
//! both products are FP32 on both sides.
//!
//! ## Parallelism
//!
//! A rank holds its place in the deployment's partition ([`DenoiserSplit`]):
//!
//! - **Sequence.** A rank holds a contiguous interval of the packed rows. [`UlyssesAttention`]
//!   trades a Ulysses group's intervals for its members' heads of the group's rows around the
//!   attention and back. Under an all-gather degree each group holds one interval of the
//!   sequence: its queries attend the whole sequence's keys and values, which the groups gather
//!   ([`VsaAttention::attend_gathered`]). A sparse block of a Ulysses split alone returns its
//!   attended rows interval by interval, so an interval's output projection and feed-forward run
//!   while the later intervals travel, and under the gather projection the next block's
//!   normalized rows of a finished interval start travelling beside the rest of the block
//!   ([`Block::forward_exchanged`]).
//! - **Tensor.** Every block's heads and feed-forward width are split; the output projection and
//!   the feed-forward's down projection reduce over the tensor group.
//! - **Layers.** A stage holds consecutive blocks; its output is broadcast to every stage, the
//!   next stage's input and, from the last stage, every stage's rows, which each reads the heads
//!   over.

use nn::family::{DenoiserSplit, GATHER_AXIS, LAYER_AXIS, Projection, TENSOR_AXIS, ULYSSES_AXIS};
use nn::functional::{self, Activation, Rounding, WeightOrder};
use nn::modules::{
    AdaLnBlock, AdaLnBlockConfig, AdaLnOut, AdaLnOutConfig, GateUp, Linear, LinearConfig,
    MlpConfig, Parallel, Precision, Rotary, RotaryConfig, Spectrum, TimestepTable,
    UlyssesAttention, UlyssesAttentionConfig, VsaAttention, VsaAttentionConfig,
};
use nn::{DType, Device, Split, Tensor, VarBuilder};
use nn::{Result, collective};

use crate::minimax_h3::ID;
use crate::minimax_h3::config::{DenoiserConfig, TransformerConfig};

/// The profiler classes of the stack and of one of its blocks.
pub(crate) const TRANSFORMER: &str = "Transformer";
const TRANSFORMER_BLOCK: &str = "TransformerBlock";

/// The coordinate axes of a packed row's rotary position: time, height and width.
const ROTARY_AXES: u32 = 3;

/// The tags a timestep group's block vectors are stored for: a block's modulation row is
/// `[TAGS · AdaLnBlock::VECTORS · hidden]`, each tag's vectors one row of the block's table.
pub const TAGS: u32 = 3;

/// The attention of a block, relative to the block: its module, the input projections in the
/// order the attention reads them (the compression gate is a sparse block's alone), and the
/// output projection.
pub(crate) const ATTENTION: &str = "attn";
pub(crate) const INPUTS: [&str; 4] = ["to_q", "to_k", "to_v", "to_gate_compress"];
pub(crate) const ATTENTION_OUTPUT: &str = "to_out.0";

/// The index of the compression gate among [`INPUTS`].
const GATE: usize = 3;

/// The module the stack's blocks are stored under, `transformer_blocks.<i>`.
pub(crate) const BLOCKS: &str = "transformer_blocks";

/// The two projections of a block's feed-forward, relative to the block: the one that fuses
/// the value half and the gate half, and the one back to the hidden width.
pub(crate) const FEED_FORWARD: [&str; 2] = ["ff.net.0.proj", "ff.net.2"];

/// The value-first SwiGLU of a block, its gated width split by `tensor`: the checkpoint fuses
/// the feed-forward's value half first and its gate half second, and `silu(gate)` and the
/// product each round once, as the reference's activation rounds.
pub(crate) fn mlp_config(config: &TransformerConfig, tensor: Split) -> MlpConfig {
    let [gate_up, down] = FEED_FORWARD;

    MlpConfig {
        hidden: config.hidden_size,
        intermediate: config.intermediate_size,
        activation: Activation::Silu,
        rounding: Rounding::Stepwise,
        gate_up: GateUp::ValueFirst(gate_up),
        down,
        bias: false,
        split: tensor,
    }
}

/// The rotary embedding of the stack: `rope_frequency_dim` frequencies repeated over the three
/// axes, each axis rotating its own contiguous block of pairs by its own coordinate, and the
/// trailing channels of a head passing through.
///
/// The inverse frequencies are FP32, `1 / theta ** (arange(0, 2f, 2) / 2f)`
/// (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:331-355`).
pub fn rotary_config(config: &TransformerConfig) -> RotaryConfig {
    RotaryConfig {
        theta: config.rope_theta,
        dim: config.rotated(),
        spectrum: Spectrum::Repeated { axes: ROTARY_AXES },
        precision: Precision::F32,
    }
}

/// One block's attention over this rank's heads of its Ulysses group's rows, inside the
/// sequence-parallel projections; under an all-gather degree over the whole sequence's keys.
///
/// The queries and keys of every head are normalized by the block's gains and their leading
/// channels rotated; H3 rotates 96 of a head's 128 channels over the three coordinates. The
/// dense attention reads the used rows of the packed sequence as its keys; the sparse one reads
/// the tiles its selection keeps and folds its compressed branch in by the block's gate
/// projection ([`VsaAttention`]).
pub struct Attention {
    ulysses: UlyssesAttention,
    /// This rank's place in its all-gather group: which interval of the sequence its Ulysses
    /// group's rows are.
    gather: Split,
    /// The `[head_dim]` gains of the queries' and of the keys' normalization.
    query_gain: Tensor,
    key_gain: Tensor,
    /// The video sparse attention of a FastH3 student; `None` attends densely.
    vsa: Option<VsaAttention>,
    head_dim: usize,
    eps: f32,
    scale: f32,
    /// The deployment's bound on the keys one dense call reads.
    max_rows: u32,
}

impl Attention {
    /// The attention the module at `vb` states (`transformer_blocks.<i>.attn`), at `split`'s
    /// place. `max_rows` is the deployment's bound on the packed rows of a layout.
    ///
    /// # Errors
    ///
    /// A tensor and Ulysses split that do not divide the heads, or a weight the loader refuses.
    pub fn load(
        config: &DenoiserConfig,
        split: DenoiserSplit,
        max_rows: u32,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let transformer = &config.transformer;
        let head_dim = transformer.head_dim as usize;
        let shares = split.ulysses.split.size() * split.tensor.size();
        let local_heads = transformer.num_attention_heads / shares.max(1);
        let rotary = rotary_config(transformer);

        let inputs: &'static [&'static str] = match config.vsa {
            Some(_) => &INPUTS,
            None => &INPUTS[..GATE],
        };
        let ulysses = UlyssesAttention::load(
            &UlyssesAttentionConfig {
                hidden: transformer.hidden_size,
                heads: transformer.num_attention_heads,
                head_dim: transformer.head_dim,
                inputs,
                output: ATTENTION_OUTPUT,
                split: split.ulysses.split,
                projection: split.ulysses.projection,
                tensor: split.tensor,
                axis: ULYSSES_AXIS,
            },
            vb.clone(),
        )?;

        // The sparse attention prepares its tiles with the same gains and the same rotation the
        // dense one applies, and writes the rows the projections produce, in the stack's type.
        let vsa = match config.vsa {
            Some(vsa) => Some(VsaAttention::load(
                &VsaAttentionConfig {
                    tile_rows: vsa.tile_rows,
                    heads: local_heads,
                    head_dim: transformer.head_dim,
                    qk_norm: Some(("norm_q", "norm_k")),
                    eps: transformer.qk_norm_eps,
                    rotated: transformer.rotated(),
                    max_tiles: max_rows / vsa.tile_rows,
                    max_keys: max_rows,
                    output: ulysses.dtype(),
                },
                vb.clone(),
            )?),
            None => None,
        };

        Ok(Self {
            query_gain: vb.pp("norm_q").get(head_dim, "weight")?,
            key_gain: vb.pp("norm_k").get(head_dim, "weight")?,
            ulysses,
            gather: split.gather,
            vsa,
            head_dim,
            eps: transformer.qk_norm_eps,
            scale: rotary.score_scale(transformer.head_dim)?,
            max_rows,
        })
    }

    /// The element type of the attention's projections: the stack's.
    pub fn dtype(&self) -> DType {
        self.ulysses.dtype()
    }

    /// The first row and the rows of this rank's Ulysses group's interval of a sequence of
    /// `rows` packed rows: the whole sequence without an all-gather degree.
    fn interval(&self, rows: usize) -> (usize, usize) {
        let own = rows / self.gather.size() as usize;
        (self.gather.rank() as usize * own, own)
    }

    /// The rotary tables `factors` of the whole sequence narrowed to the interval of `rows`
    /// rows from `first`.
    fn narrowed(
        factors: (&Tensor, &Tensor),
        (first, rows): (usize, usize),
    ) -> Result<(Tensor, Tensor)> {
        Ok((
            factors.0.narrow(0, first, rows)?.contiguous()?,
            factors.1.narrow(0, first, rows)?.contiguous()?,
        ))
    }

    /// `rows`, `[rows, local heads · head_dim]`, as `[rows, local heads, head_dim]`.
    fn headed(&self, rows: &Tensor) -> Result<Tensor> {
        let heads = self.ulysses.local_heads() as usize;

        Ok(rows.reshape((rows.dims()[0], heads, self.head_dim))?)
    }

    /// Attends `normalized`, this rank's `[rows, hidden]`, and returns its `[rows, hidden]`
    /// update.
    ///
    /// `factors` is the whole sequence's `(cos, sin)` rotary tables. `tables` is the layout's
    /// attention tables on the device: for a dense attention the query and the key offsets of
    /// the one sequence (`[2]` `U32` each: every row a query, the used rows the keys), for a
    /// sparse one each tile's valid rows, its segment and each segment's kept tiles
    /// ([`crate::minimax_h3::packing::tile_table`]).
    ///
    /// Under an all-gather degree the projections hold this rank's Ulysses group's interval of
    /// the sequence: its queries attend the whole sequence's keys and values, gathered along the
    /// all-gather group, and the dense query offsets are the interval's own.
    ///
    /// # Errors
    ///
    /// Another number of tables than the attention reads, or what a module refuses.
    pub fn forward(
        &self,
        normalized: &Tensor,
        factors: (&Tensor, &Tensor),
        tables: &[Tensor],
    ) -> Result<Tensor> {
        let ulysses = &self.ulysses;
        let rows = ulysses.rows(normalized)?;
        let gathers = self.gather.size() > 1;

        let Some(vsa) = &self.vsa else {
            let [queries, keys] = tables else {
                return Err(
                    ID.request("a dense attention reads the query and the key offsets".to_owned())
                );
            };
            let query = self.headed(&ulysses.project(0, &rows)?)?;
            let key = self.headed(&ulysses.project(1, &rows)?)?;
            let value = self.headed(&ulysses.project(2, &rows)?)?;

            // The interval's own rows of the sequence, whose rotation they take.
            let sequence = factors.0.dims()[0];
            let interval = self.interval(sequence);
            let narrowed = match gathers {
                true => Some(Self::narrowed(factors, interval)?),
                false => None,
            };
            let (cos, sin) = narrowed.as_ref().map_or(factors, |(cos, sin)| (cos, sin));

            let (query, key) = functional::qk_norm_rope(
                &query,
                &key,
                Some((&self.query_gain, &self.key_gain)),
                cos,
                sin,
                self.eps,
                false,
            )?;

            // Every interval's keys and values in sequence order, and the interval's own rows as
            // the one sequence of queries.
            let (key, value, queries) = match gathers {
                true => {
                    let (key, value) = collective::along(GATHER_AXIS, || {
                        Ok((
                            collective::gathered(&key.contiguous()?)?,
                            collective::gathered(&value.contiguous()?)?,
                        ))
                    })?;
                    let offsets = Tensor::new(&[0_u32, interval.1 as u32], &Device::Cpu)?
                        .to_device(query.device())?;
                    (key, value, offsets)
                }
                false => (key, value, queries.clone()),
            };
            let attended = functional::dense_attention_varlen(
                &query,
                &key,
                &value,
                &queries,
                keys,
                None,
                self.max_rows,
                self.scale,
                false,
                None,
            )?;

            return ulysses.return_rows(&attended.flatten_from(1)?);
        };

        let query = ulysses.project(0, &rows)?;
        let key = ulysses.project(1, &rows)?;
        let value = ulysses.project(2, &rows)?;
        let gate = ulysses.project(GATE, &rows)?;

        let (fine, compressed) = match gathers {
            false => vsa.attend((query, key, value), Some(factors), tiles(tables)?)?,
            true => {
                let interval = self.interval(factors.0.dims()[0]);
                let (cos, sin) = Self::narrowed(factors, interval)?;
                let first = interval.0 / vsa.tile_rows() as usize;

                let tables = tiles(tables)?;
                collective::along(GATHER_AXIS, || {
                    vsa.attend_gathered((query, key, value), Some((&cos, &sin)), tables, first)
                })?
            }
        };
        let folded = vsa.add_gated_(fine, &compressed, &gate)?;

        ulysses.return_rows(&folded)
    }
}

/// A sparse attention's three tile tables: each tile's valid rows, its segment, and each
/// segment's kept tiles.
fn tiles(tables: &[Tensor]) -> Result<(&Tensor, &Tensor, &Tensor)> {
    match tables {
        [sizes, segments, keep] => Ok((sizes, segments, keep)),
        _ => Err(ID.request(
            "a sparse attention reads the tiles' sizes, their segments and the kept tiles"
                .to_owned(),
        )),
    }
}

/// One block of the stack: an AdaLN block around the block's attention.
pub struct Block {
    /// The modulated norms and the value-first SwiGLU.
    adaln: AdaLnBlock,
    attention: Attention,
    /// The tensor split of the block's heads and feed-forward width.
    tensor: Split,
    /// The block's path in the checkpoint, which names its profiler range.
    path: String,
}

impl Block {
    /// The block the module at `vb` states (`transformer_blocks.<i>`).
    ///
    /// # Errors
    ///
    /// A weight the loader refuses.
    pub fn load(
        config: &DenoiserConfig,
        split: DenoiserSplit,
        max_rows: u32,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let transformer = &config.transformer;

        Ok(Self {
            adaln: AdaLnBlock::load(
                &AdaLnBlockConfig {
                    hidden: transformer.hidden_size,
                    eps: transformer.norm_eps,
                    norm1: "norm1",
                    norm2: "norm2",
                    mlp: mlp_config(transformer, split.tensor),
                },
                vb.clone(),
            )?,
            attention: Attention::load(config, split, max_rows, vb.pp(ATTENTION))?,
            tensor: split.tensor,
            path: vb.path().to_owned(),
        })
    }

    /// The block over `hidden`, this rank's `[rows, hidden]`, conditioned by `table`, the
    /// step's `[3 · groups, 6 · hidden]` modulation, at each row's modulation row `rows`.
    ///
    /// # Errors
    ///
    /// What a module refuses.
    pub fn forward(
        &self,
        table: &Tensor,
        rows: &Tensor,
        hidden: &Tensor,
        factors: (&Tensor, &Tensor),
        tables: &[Tensor],
    ) -> Result<Tensor> {
        // The output projection and the feed-forward's down projection reduce their partial
        // sums along the tensor group.
        let forward = || {
            self.adaln.forward(table, rows, hidden, |normalized| {
                self.attention.forward(normalized, factors, tables)
            })
        };
        match self.tensor.size() {
            1 => Ok(forward()?),
            _ => Ok(collective::along(TENSOR_AXIS, forward)?),
        }
    }

    /// A sparse block across ranks: the attended rows return to their owners interval by
    /// interval, and everything after the attention reads a row alone, so an interval's output
    /// projection and feed-forward run while the later intervals travel.
    ///
    /// `table` and `rows` are the block's modulation and each row's modulation row, as in
    /// [`Self::forward`]. Under the gather projection `ahead` holds the whole sequence's
    /// normalized rows
    /// (`[ranks · rows, hidden]`, this rank's block already written) and the next block with
    /// its modulation table; `arriving` holds the gathers of those rows still travelling. As
    /// each interval of this block's output is complete, the next block's normalized rows of it
    /// are set into this rank's block of the sequence and start travelling beside the rest of
    /// this block.
    ///
    /// # Errors
    ///
    /// A dense block, a forward that is not split over the block's group, or what a module
    /// refuses.
    fn forward_exchanged(
        &self,
        (table, rows): (&Tensor, &Tensor),
        hidden: &Tensor,
        factors: (&Tensor, &Tensor),
        tables: &[Tensor],
        ahead: Option<(&Tensor, Option<(&Block, &Tensor)>)>,
        arriving: &mut Vec<usize>,
    ) -> Result<Tensor> {
        let attention = &self.attention;
        let ulysses = &attention.ulysses;
        let Some(vsa) = &attention.vsa else {
            return Err(ID.unsupported("a dense block returns its rows in one exchange".to_owned()));
        };
        let own = hidden.dims()[0];
        let split = ulysses.split();

        // The rows the projections read: the whole sequence's, once every gather of them has
        // arrived, or this rank's own under the exchange projection.
        let input = match ahead {
            Some((sequence, _)) => {
                for started in arriving.drain(..) {
                    ulysses.finish(started)?;
                }
                sequence.clone()
            }
            None => self.adaln.attention_input(table, rows, hidden)?,
        };

        // Under the exchange projection each projection's blocks travel while the next one
        // multiplies; the gate is projected last and travels beside the attention. A send
        // buffer is the exchange's until it is finished, so each is kept until then.
        let exchanged = ulysses.projection() == Projection::Exchange;
        let mut sent = Vec::new();
        let mut travelling = None;
        let mut projections = Vec::with_capacity(INPUTS.len());
        for index in 0..INPUTS.len() {
            if !exchanged {
                projections.push(ulysses.project(index, &input)?);
                continue;
            }

            let send = ulysses.member_blocks(index, &input)?;
            let (received, started) = ulysses.start_exchange(&send)?;
            if let Some(previous) = travelling.replace(started) {
                ulysses.finish(previous)?;
            }
            sent.push(send);
            projections.push(received);
        }
        let [query, key, value, gate]: [Tensor; 4] = projections
            .try_into()
            .map_err(|_| ID.unsupported("a sparse block projects four inputs".to_owned()))?;

        let (fine, compressed) = vsa.attend((query, key, value), Some(factors), tiles(tables)?)?;
        if let Some(started) = travelling {
            ulysses.finish(started)?;
        }

        // Every rank's rows of this rank's heads leave in intervals of each owner's rows, one
        // exchange after another. The composition folds one tile's compression into each kernel
        // block of an owner's rows, so the intervals are whole blocks of rows the layout made
        // whole blocks on every rank.
        let slots: Vec<u32> = (0..split.size()).collect();
        let unit = functional::VSA_BLOCK_ROWS as usize;
        let mut returning = Vec::new();
        for interval in UlyssesAttention::intervals(own, unit) {
            let payload = vsa.compose(&fine, &compressed, &gate, interval.clone(), &slots)?;
            let (received, started) = ulysses.start_exchange(&payload)?;

            returning.push((interval, payload, received, started));
        }

        let mut outputs = Vec::with_capacity(returning.len());
        for (interval, _payload, received, started) in returning {
            let (first, count) = (interval.start, interval.len());
            let update = ulysses.receive(started, &received)?;

            let interval_rows = rows.narrow(0, first, count)?;
            let output = self.adaln.feed_forward(
                table,
                &interval_rows,
                &hidden.narrow(0, first, count)?,
                update,
            )?;

            // These rows are the next block's input, and its attention reads the whole
            // sequence, so they leave now.
            if let Some((sequence, Some((next, next_table)))) = ahead {
                let normalized = next
                    .adaln
                    .attention_input(next_table, &interval_rows, &output)?;

                sequence.slice_set(&normalized, 0, own * split.rank() as usize + first)?;
                arriving.push(ulysses.start_gather(sequence, own, interval)?);
            }
            outputs.push(output);
        }

        Ok(Tensor::cat(&outputs, 0)?)
    }
}

/// One modality's output head: the FP32 projection of the final norm's rows into the
/// modality's velocity rows.
///
/// A parallel-decoding student stacks one head per fine interval of its schedule's grid and
/// predicts one block's velocity with the block's weighted sum of its heads, weights and sum
/// in FP32 (`refs/vllm-omni/…/minimax_h3/pdd.py:41-70`). The sums are tables of the schedule,
/// written once after the load ([`Self::initialize`]); a forward gathers its step's row, so one
/// capture serves every step.
pub struct OutputHead {
    /// The projection a forward reads: `[out, hidden]`, or a student's `[blocks, out · hidden]`
    /// fused table.
    weight: Tensor,
    /// `[out]`, or a student's `[blocks, out]`.
    bias: Tensor,
    /// A student's stacked heads `[heads · out, hidden]` and `[heads · out]`, and its
    /// `[blocks, heads]` fusion weights.
    stacked: Option<(Tensor, Tensor, Tensor)>,
    out: usize,
    hidden: usize,
}

impl OutputHead {
    /// The head the module at `vb` states (`proj_out`, `audio_proj_out`), `out` wide, with the
    /// fusion weights `weights` of a student's stacked heads (`[blocks, heads]` row-major) and
    /// `None` for one head.
    ///
    /// The head is an FP32 parameter at every preset.
    ///
    /// # Errors
    ///
    /// A weight the loader refuses, or weights that are not one row per block.
    pub fn load(
        config: &TransformerConfig,
        out: u32,
        weights: Option<&[f32]>,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let vb = vb.to_dtype(DType::F32);
        let (out, hidden) = (out as usize, config.hidden_size as usize);
        let heads = config.output_heads as usize;

        let weight = vb.get((heads * out, hidden), "weight")?;
        let bias = vb.get(heads * out, "bias")?;
        let Some(weights) = weights else {
            return Ok(Self {
                weight,
                bias,
                stacked: None,
                out,
                hidden,
            });
        };

        if weights.is_empty() || !weights.len().is_multiple_of(heads) {
            return Err(ID.checkpoint(format!(
                "{} fusion weights for {heads} stacked heads",
                weights.len()
            )));
        }
        let blocks = weights.len() / heads;

        Ok(Self {
            weight: vb.writable_zeros("fused_weight", DType::F32, (blocks, out * hidden))?,
            bias: vb.writable_zeros("fused_bias", DType::F32, (blocks, out))?,
            stacked: Some((
                weight,
                bias,
                vb.constant("fusion_weights", DType::F32, (blocks, heads), weights)?,
            )),
            out,
            hidden,
        })
    }

    /// Fuses a student's stacked heads into its per-block tables:
    /// `fused[b] = Σ_n weights[b, n] · head[n]`, in FP32. One head has nothing to fuse.
    ///
    /// # Errors
    ///
    /// What a tensor operation refuses.
    pub fn initialize(&self) -> Result<()> {
        let Some((weight, bias, weights)) = &self.stacked else {
            return Ok(());
        };
        let heads = weights.dims()[1];

        let fused = weights.matmul(&weight.reshape((heads, self.out * self.hidden))?)?;
        self.weight.slice_set(&fused, 0, 0)?;

        let fused = weights.matmul(&bias.reshape((heads, self.out))?)?;
        Ok(self.bias.slice_set(&fused, 0, 0)?)
    }

    /// The velocity rows of `hidden`, `[rows, hidden]` FP32, at evaluation `step`: `[rows, out]`
    /// FP32.
    ///
    /// # Errors
    ///
    /// An evaluation past a student's blocks, or what the projection refuses.
    pub fn forward(&self, hidden: &Tensor, step: u32) -> Result<Tensor> {
        if self.stacked.is_none() {
            return functional::linear(hidden, &self.weight, Some(&self.bias), WeightOrder::NxK);
        }

        let blocks = self.weight.dims()[0];
        if step as usize >= blocks {
            return Err(ID.request(format!(
                "evaluation {step} is past the {blocks} fused heads of the student"
            )));
        }

        // The step is a host value of the forward: its index is moved to the device and the
        // block's row gathered there.
        let index = Tensor::new(&[step], &Device::Cpu)?.to_device(self.weight.device())?;
        let weight = self
            .weight
            .index_select(&index, 0)?
            .reshape((self.out, self.hidden))?;
        let bias = self.bias.index_select(&index, 0)?.reshape(self.out)?;

        functional::linear(hidden, &weight, Some(&bias), WeightOrder::NxK)
    }
}

/// One network of the checkpoint: the input projections, the stack and the output heads.
///
/// A stage of a layer split holds its consecutive blocks of the stack alone, and every stage
/// the end projections and heads, which are small beside a block: every stage projects the
/// latents and reads the velocities, and only the first stage's projected rows enter the stack.
pub struct Transformer {
    /// The projection of a video patch row into the hidden width, FP32.
    proj_in: Linear,
    /// The projection of an audio latent row into the hidden width, FP32.
    audio_proj_in: Linear,
    /// The stack's one rotary embedding, whose factors every block turns by.
    rotary: Rotary,
    /// This stage's blocks of the stack, the first of them the stack's block `first_block`.
    blocks: Vec<Block>,
    first_block: usize,
    /// The layer split: this rank's stage among the stack's.
    layers: Split,
    /// Whether the stack runs its blocks across a Ulysses split alone, returning a sparse
    /// block's rows interval by interval ([`Self::forward_exchanged`]).
    overlapped: bool,
    norm_out: AdaLnOut,
    video_out: OutputHead,
    audio_out: OutputHead,
    hidden: usize,
}

impl Transformer {
    /// The network the checkpoint at `vb` states, at `split`'s place: this stage's blocks of
    /// the stack. `max_rows` is the deployment's bound on the packed rows of a layout.
    ///
    /// # Errors
    ///
    /// A weight the loader refuses, a schedule whose head weights do not fit the heads, or a
    /// layer split of more stages than blocks.
    pub fn load(
        config: &DenoiserConfig,
        split: DenoiserSplit,
        max_rows: u32,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let transformer = &config.transformer;
        let hidden = transformer.hidden_size;

        // The end projections are FP32 parameters at every preset: they touch the trajectory.
        let project_in = |name: &str, inputs: u32| {
            Linear::load(
                &LinearConfig {
                    inputs,
                    outputs: hidden,
                    bias: true,
                    parallel: Parallel::None,
                },
                vb.pp(name).to_dtype(DType::F32),
            )
        };
        let head = |name: &str, out: u32, shift: f32| -> Result<OutputHead> {
            let weights = (transformer.output_heads > 1)
                .then(|| config.schedule.head_weights(shift))
                .transpose()?;

            OutputHead::load(transformer, out, weights.as_deref(), vb.pp(name))
        };

        if split.layers.size() > transformer.num_hidden_layers {
            return Err(ID.config(format!(
                "a layer split of {} stages over a stack of {} blocks",
                split.layers.size(),
                transformer.num_hidden_layers
            )));
        }
        let stage = split.stage_blocks(transformer.num_hidden_layers);
        let blocks = stage
            .clone()
            .map(|index| Block::load(config, split, max_rows, vb.pp(BLOCKS).pp(index)))
            .collect::<Result<Vec<_>>>()?;
        let overlapped = config.vsa.is_some()
            && split.ulysses.split.size() > 1
            && split.gather.size() == 1
            && split.tensor.size() == 1
            && split.layers.size() == 1;

        Ok(Self {
            proj_in: project_in("proj_in", transformer.patch_width())?,
            audio_proj_in: project_in("audio_proj_in", transformer.audio_channels)?,
            rotary: Rotary::load(&rotary_config(transformer), vb.pp("rotary_emb"))?,
            blocks,
            first_block: stage.start as usize,
            layers: split.layers,
            overlapped,
            norm_out: AdaLnOut::load(
                &AdaLnOutConfig {
                    hidden,
                    eps: transformer.norm_eps,
                },
                vb.pp("norm_out").pp("norm"),
            )?,
            video_out: head(
                "proj_out",
                transformer.patch_width(),
                config.schedule.video_shift,
            )?,
            audio_out: head(
                "audio_proj_out",
                transformer.audio_channels,
                config.schedule.audio_shift,
            )?,
            hidden: hidden as usize,
        })
    }

    /// The element type the stack's weights and activations are in.
    ///
    /// The witness is a block's norm gain, never an end projection: those are FP32 at every
    /// preset.
    pub fn dtype(&self) -> DType {
        self.blocks
            .first()
            .map_or(DType::F32, |block| block.adaln.dtype())
    }

    /// Fills the tables the network computes from its weights: a student's fused heads.
    ///
    /// # Errors
    ///
    /// What a tensor operation refuses.
    pub fn initialize(&self) -> Result<()> {
        self.video_out.initialize()?;
        self.audio_out.initialize()
    }

    /// The `(cos, sin)` rotary tables of `positions`, `[3, rows]` FP32 axis-major: FP32
    /// `[rows, 48]` each.
    ///
    /// # Errors
    ///
    /// Positions that are not one coordinate per axis and row.
    pub fn rotary_factors(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        self.rotary.forward(positions)
    }

    /// A video latent `[channels, frames, height, width]` as the network's `1 x 2 x 2` patch
    /// rows: `[frames · height/2 · width/2, 4 · channels]`, row-major over the patch grid and
    /// channel-major inside a row (`refs/vllm-omni/…/minimax_h3/packed_tokens.py:23-41`).
    ///
    /// # Errors
    ///
    /// A latent that is not whole patches.
    pub fn patchify(latent: &Tensor) -> Result<Tensor> {
        let &[channels, frames, height, width] = latent.dims() else {
            return Err(ID.request(format!(
                "a video latent is [channels, frames, height, width], not {:?}",
                latent.dims()
            )));
        };
        if !height.is_multiple_of(2) || !width.is_multiple_of(2) {
            return Err(ID.request(format!(
                "a {height} by {width} latent is not whole 2 x 2 patches"
            )));
        }
        let (rows, columns) = (height / 2, width / 2);

        Ok(latent
            .reshape((channels, frames, rows, 2, columns, 2))?
            .permute((1, 2, 4, 0, 3, 5))?
            .reshape((frames * rows * columns, channels * 4))?)
    }

    /// Patch rows back into a video latent of `dims`, `[channels, frames, height, width]`: the
    /// inverse of [`Self::patchify`] (`packed_tokens.py:44-71`).
    ///
    /// # Errors
    ///
    /// Rows that are not `dims`' patches.
    pub fn unpatchify(rows: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let &[channels, frames, height, width] = dims else {
            return Err(ID.request(format!(
                "a video latent is [channels, frames, height, width], not {dims:?}"
            )));
        };
        let (patch_rows, patch_columns) = (height / 2, width / 2);

        Ok(rows
            .reshape((frames, patch_rows, patch_columns, channels, 2, 2))?
            .permute((3, 0, 1, 4, 2, 5))?
            .reshape((channels, frames, height, width))?)
    }

    /// Video patch rows projected into the hidden width, in FP32.
    ///
    /// # Errors
    ///
    /// Rows the projection refuses.
    pub fn project_video(&self, rows: &Tensor) -> Result<Tensor> {
        self.proj_in.forward(&rows.to_dtype(DType::F32)?)
    }

    /// Audio latent rows projected into the hidden width, in FP32.
    ///
    /// # Errors
    ///
    /// Rows the projection refuses.
    pub fn project_audio(&self, rows: &Tensor) -> Result<Tensor> {
        self.audio_proj_in.forward(&rows.to_dtype(DType::F32)?)
    }

    /// The modulation of this stage's block `index` at `entries`: the stack block's gathered
    /// `[groups, 18 · hidden]` rows read as `[3 · groups, 6 · hidden]`, group `g`'s row holding
    /// its three tags' six vectors each
    /// (`refs/vllm-omni/…/minimax_h3/minimax_h3_blocks.py:736-745`).
    fn modulation(&self, table: &TimestepTable, index: usize, entries: &Tensor) -> Result<Tensor> {
        let groups = entries.elem_count();

        Ok(table.layer(self.first_block + index, entries)?.reshape((
            groups * TAGS as usize,
            AdaLnBlock::VECTORS as usize * self.hidden,
        ))?)
    }

    /// The stack over `hidden`, this rank's `[rows, hidden]` packed rows, at one step: the rows
    /// after the last block.
    ///
    /// `table` is the schedule's modulation and `entries` the `[groups]` entries the step reads
    /// ([`TimestepTable::entries_of`]); `rows` is this rank's `[rows]` modulation row of each
    /// packed row, `3 · group + tag`; `factors` the whole sequence's rotary tables; `tables`
    /// the layout's attention tables ([`Attention::forward`]).
    ///
    /// # Errors
    ///
    /// What a module refuses.
    pub fn forward(
        &self,
        table: &TimestepTable,
        entries: &Tensor,
        hidden: &Tensor,
        rows: &Tensor,
        factors: (&Tensor, &Tensor),
        tables: &[Tensor],
    ) -> Result<Tensor> {
        nn::profiler::range(TRANSFORMER, "transformer", || {
            if self.overlapped {
                return self.forward_exchanged(table, entries, hidden, rows, factors, tables);
            }

            let blocks = |hidden: Tensor| -> Result<Tensor> {
                self.blocks
                    .iter()
                    .enumerate()
                    .try_fold(hidden, |hidden, (index, block)| {
                        nn::profiler::range(TRANSFORMER_BLOCK, &block.path, || {
                            let modulation = self.modulation(table, index, entries)?;
                            block.forward(&modulation, rows, &hidden, factors, tables)
                        })
                    })
            };
            let (stages, stage) = (self.layers.size(), self.layers.rank());
            if stages == 1 {
                return blocks(hidden.clone());
            }

            // The stages pass the rows on as one collective of every stage per stage, in stage
            // order: each stage's output is broadcast from it, so a stage holds its input once
            // the stage before broadcast, and every stage the last stage's output at the end.
            // Every stage posts the same collectives in the same order, as a sequence split's
            // exchanges are, and none waits for a pair of its own.
            let mut held = hidden.clone();
            for root in 0..stages {
                if root == stage {
                    held = blocks(held)?;
                }
                held = collective::along(LAYER_AXIS, || {
                    collective::broadcast(&held.contiguous()?, root)
                })?;
            }
            Ok(held)
        })
    }

    /// The stack of sparse blocks across ranks ([`Block::forward_exchanged`]).
    ///
    /// Under the gather projection the whole sequence's normalized rows live across the
    /// blocks: each block's are gathered while the block before it computes, and the first
    /// block's before the stack starts.
    fn forward_exchanged(
        &self,
        table: &TimestepTable,
        entries: &Tensor,
        hidden: &Tensor,
        rows: &Tensor,
        factors: (&Tensor, &Tensor),
        tables: &[Tensor],
    ) -> Result<Tensor> {
        let first = &self.blocks[0];
        let ulysses = &first.attention.ulysses;
        let own = hidden.dims()[0];
        let mut arriving = Vec::new();

        let sequence = if ulysses.projection() == Projection::Gather {
            let modulation = self.modulation(table, 0, entries)?;
            let normalized = first.adaln.attention_input(&modulation, rows, hidden)?;

            let sequence = nn::collective::sequence_like(&normalized)?;
            sequence.slice_set(&normalized, 0, own * ulysses.split().rank() as usize)?;
            arriving.push(ulysses.start_gather(&sequence, own, 0..own)?);

            Some(sequence)
        } else {
            None
        };

        let mut hidden = hidden.clone();
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = nn::profiler::range(TRANSFORMER_BLOCK, &block.path, || {
                let modulation = self.modulation(table, index, entries)?;

                // The next block's rows are gathered ahead only under the gather projection.
                let next_modulation = match (&sequence, self.blocks.get(index + 1)) {
                    (Some(_), Some(_)) => Some(self.modulation(table, index + 1, entries)?),
                    _ => None,
                };
                let next = self.blocks.get(index + 1).zip(next_modulation.as_ref());
                let ahead = sequence.as_ref().map(|sequence| (sequence, next));

                block.forward_exchanged(
                    (&modulation, rows),
                    &hidden,
                    factors,
                    tables,
                    ahead,
                    &mut arriving,
                )
            })?;
        }
        Ok(hidden)
    }

    /// The two modalities' velocity rows of `hidden`, the stack's rows: `(video, audio)`,
    /// `[rows, 4 · channels]` and `[rows, audio channels]` in FP32.
    ///
    /// The final norm modulates every row by its timestep group's shift and scale
    /// (`groups`, `[rows]` `U32`), the rows widen to FP32, and both heads read every row, as
    /// the reference's final layer does
    /// (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:578-598`); a caller reads each
    /// modality's rows out of its head's.
    ///
    /// # Errors
    ///
    /// What a module refuses.
    pub fn velocities(
        &self,
        table: &TimestepTable,
        entries: &Tensor,
        hidden: &Tensor,
        groups: &Tensor,
        step: u32,
    ) -> Result<(Tensor, Tensor)> {
        let normalized = self
            .norm_out
            .forward(&table.output(entries)?, groups, hidden)?
            .to_dtype(DType::F32)?;

        Ok((
            self.video_out.forward(&normalized, step)?,
            self.audio_out.forward(&normalized, step)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patch_row_holds_its_channels_patches_in_raster_order() {
        // Two channels of one frame of 2 x 4 cells: element `c · 100 + row · 10 + column`.
        let (channels, frames, height, width) = (2, 1, 2, 4);
        let cells: Vec<f32> = (0..channels * frames * height * width)
            .map(|index| {
                let (channel, cell) = (index / (height * width), index % (height * width));
                (channel * 100 + (cell / width) * 10 + cell % width) as f32
            })
            .collect();
        let latent =
            Tensor::from_vec(cells, (channels, frames, height, width), &Device::Cpu).unwrap();

        let rows = Transformer::patchify(&latent).unwrap();
        assert_eq!(rows.dims(), [2, 8]);

        // A row is channel-major, and each channel's patch runs row by row
        // (`packed_tokens.py:39-41`): the second patch starts at column two.
        assert_eq!(
            rows.to_vec2::<f32>().unwrap(),
            [
                [0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0],
                [2.0, 3.0, 12.0, 13.0, 102.0, 103.0, 112.0, 113.0],
            ]
        );

        let restored = Transformer::unpatchify(&rows, latent.dims()).unwrap();
        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            latent.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // A raster that is not whole patches is refused, not truncated.
        let odd = Tensor::zeros((2, 1, 3, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(Transformer::patchify(&odd).is_err());
    }
}
