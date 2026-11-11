//! The diffusion transformer: the packed rows in, one evaluation's velocity out.
//!
//! One `Transformer` is a checkpoint's DiT partition: a projection of each modality's patchified
//! latent into the hidden width, the stack of layers, and the output projections back to each
//! modality's velocity. A layer is an [`AdaLnBlock`] over H3's attention, conditioned by the step's
//! AdaLN rows, and the final normalization is an [`AdaLnOut`].
//!
//! The step's AdaLN rows are gathered on the device from the [`TimestepTable`] in H3's row layout
//! ([`crate::modulation`]), so one captured stack serves every step of the schedule.
//!
//! ## The two crossings
//!
//! The trajectory is FP32 ([`crate::denoiser::SAMPLE_DTYPE`]) and the stack computes in the
//! checkpoint's type. The four end projections are FP32 parameters at every preset, as the
//! reference builds them, so the sample crosses to the stack's type once after the input projection
//! and back once before the output head ([`Transformer::wide`]).

use nn::blocks::{
    AdaLnBlock, AdaLnOut, AdaLnSpec, GateUp, GatedMlpSpec, ModulationTable, RowModulation,
};
use nn::diffusion::TimestepTable;
use nn::family::{SequenceSplit, Split};
use nn::layers::{
    Bias, Linear, Parallel, PatchEmbed, PatchEmbedSpec, Precision, QkNorm, Rotary, RotarySpec,
    Rounding, Spectrum, StepIndexed,
};
use nn::weights::WeightLoader;
use nn::{DType, Forward, Operand, Shape};

use crate::attention::{self, Visibility};
use crate::config::{AttentionKind, TransformerConfig};
use crate::modulation::{TAGS, VECTORS};
use crate::schedule::{Modality, Schedule};
use crate::{Error, Result};

/// The fence classes of the denoiser's stack and of one of its layers
/// (`profiling_config.fence_components`).
const TRANSFORMER: &str = "models.minimax_h3.denoiser.Transformer";
const TRANSFORMER_LAYER: &str = "models.minimax_h3.denoiser.TransformerLayer";

/// The coordinate axes of a packed row's rotary position: time, height and width.
const ROTARY_AXES: u32 = 3;

/// The DiT's rotary spectrum: `rope_frequency_dim` frequencies repeated over the three axes.
///
/// The reference builds its rotary embedding as `RotaryEmbedding(2 * rope_frequency_dim)` and hands
/// it one axis spanning the whole head, so the same spectrum is evaluated at each of the three
/// coordinates and the axes' factors are three contiguous blocks of one table; the trailing
/// channels of a head pass through. The inverse frequencies are FP32,
/// `1 / theta ** (arange(0, 2f, 2, float32) / 2f)` (vllm-omni `minimax_h3_transformer.py`, the
/// Diffusers load path; sglang `dits/minimax_h3.py` keeps the same FP32 `rope.inv_freq` buffer).
pub fn spectrum(config: &TransformerConfig) -> Spectrum {
    Spectrum::Repeated {
        theta: config.rope_theta,
        pairs: config.rope_frequency_dim,
        axes: ROTARY_AXES,
        precision: Precision::F32,
    }
}

/// One layer of the stack: an AdaLN block over the layer's dense or sparse attention.
struct Layer {
    /// The modulated norms and the value-first SwiGLU, `silu(gate)` and the product evaluated with
    /// one rounding each (`Rounding::Stepwise`), as the reference's activation rounds.
    block: AdaLnBlock,
    attention: attention::Layer,
}

impl Layer {
    /// Read layer `index` of a partition, turning by the stack's `rotary`.
    fn load(
        loader: &mut WeightLoader<'_>,
        stack: (&TransformerConfig, AttentionKind),
        index: u32,
        rotary: &Rotary,
        bounds: (u32, u32),
        sequence: SequenceSplit,
    ) -> Result<Self> {
        let config = stack.0;
        let block = AdaLnBlock::load(
            &mut loader.scope(format!("transformer_blocks.{index}")),
            AdaLnSpec {
                hidden: config.hidden_size,
                eps: config.norm_eps,
                norm1: "norm1",
                norm2: "norm2",
                // The checkpoint fuses the feed-forward's value half first and its gate half second.
                mlp: GatedMlpSpec {
                    hidden: config.hidden_size,
                    intermediate: config.intermediate_size,
                    activation: ops::activation::Activation::Silu,
                    rounding: Rounding::Stepwise,
                    gate_up: GateUp::ValueFirst("ff.net.0.proj"),
                    down: "ff.net.2",
                    bias: Bias::Absent,
                },
            },
            Split::WHOLE,
        )?;
        Ok(Self {
            block,
            attention: attention::Layer::load(loader, stack, index, rotary, bounds, sequence)?,
        })
    }

    /// One layer over `hidden`, this rank's `[rows, width]` stream, in place, turned by
    /// `rotation` and attending what `visibility` lets a query see.
    ///
    /// Across ranks a sparse layer's attended rows return interval by interval and everything after
    /// the attention reads a row alone, so an interval's output projection and feed-forward run
    /// while the later intervals travel. Under the gather projection `ahead` holds the whole
    /// sequence's normalized rows (`gathered`), the gathers of them still travelling (`arriving`),
    /// and the next layer with its modulation: as each interval of this layer's output is
    /// complete, the next layer's normalized rows of it are written into this rank's block of
    /// `gathered` and start travelling beside the rest of this layer.
    fn forward(
        &self,
        f: &Forward,
        modulation: RowModulation<'_>,
        hidden: Operand,
        (rotation, visibility): ((&Rotary, nn::layers::Factors), &Visibility),
        (gathered, next): (Option<Operand>, Option<(&Layer, RowModulation<'_>)>),
        arriving: &mut Vec<nn::Transfer>,
    ) -> Result<()> {
        let (attention::Layer::Sparse(sparse), Visibility::Sparse(tables)) =
            (&self.attention, visibility)
        else {
            return Ok(self.block.forward(f, modulation, hidden, |normalized| {
                Ok(self
                    .attention
                    .forward(f, normalized, rotation, visibility)?)
            })?);
        };
        if !sparse.exchanges() {
            return Ok(self.block.forward(f, modulation, hidden, |normalized| {
                Ok(sparse.forward(f, normalized, rotation.1, tables)?)
            })?);
        }
        let own = hidden.rows();
        let rows = match gathered {
            Some(gathered) => {
                for transfer in arriving.drain(..) {
                    sparse.finish(f, transfer)?;
                }
                gathered
            }
            None => self.block.attention_input(f, modulation, hidden)?,
        };
        let attended = sparse.attend(f, rows, rotation.1, tables)?;
        let update = f.empty(hidden.dtype, hidden.shape)?;
        for returning in sparse.deliver(f, attended, own)? {
            let (first, count) = (returning.0.first, returning.0.count);
            sparse.receive(f, returning, update.narrow(0, first, count)?)?;
            let interval = |modulation| interval_rows(modulation, (first, count));
            let stream = hidden.narrow(0, first, count)?;
            self.block.feed_forward(
                f,
                interval(modulation)?,
                stream,
                update.narrow(0, first, count)?,
            )?;
            // These rows are the next layer's input, and its attention reads the whole sequence,
            // so they leave now, beside the rest of this layer.
            if let (Some(gathered), Some((layer, next))) = (gathered, next) {
                let block = gathered.narrow(0, own * u64::from(sparse.rank()) + first, count)?;
                layer
                    .block
                    .attention_input_into(f, interval(next)?, stream, block)?;
                arriving.push(sparse.start_gather(f, gathered, own, (first, count))?);
            }
        }
        Ok(())
    }
}

/// `modulation` over rows `first..first + count` of the rows it modulates.
fn interval_rows(
    modulation: RowModulation<'_>,
    (first, count): (u64, u64),
) -> Result<RowModulation<'_>> {
    Ok(RowModulation {
        table: modulation.table,
        rows: modulation.rows.narrow(0, first, count)?,
    })
}

/// A parallel-decoding student's output heads, fused per evaluation block at load: the video
/// head and the audio head.
///
/// A DMD/PDD student stacks one head per fine interval (`pdd_steps`, e.g. 32) and predicts one
/// block's velocity from the weighted sum of its intervals' heads (weights and accumulation in
/// FP32). The fusion is arithmetic over the checkpoint's own numbers, so it is done once, on the
/// host, at the loading boundary; what a forward does is pick one row of a [`StepIndexed`] table,
/// `[blocks, out * hidden]` FP32, indexed by the step's evaluation (`[1]` int32), so one capture
/// serves every step of a layout.
///
/// FP32 because the reference builds a student's head FP32: the fused row is a sum of the
/// checkpoint's BF16 rows and rounding it back would round a number the reference keeps exact. The
/// projection reads the widened FP32 rows (`Transformer::wide`), so its two operands are one type
/// and the only rounding is the store into the velocity.
///
/// The checkpoint stacks `output_heads` heads along the first axis of `proj_out` and
/// `audio_proj_out` (`[output_heads * out, hidden]`, `[output_heads * out]`). Block `b`'s fused
/// weight is `sum_n w[n] * head[start + n]` with `w` the block's normalized integration weights.
pub fn fuse_heads(
    source: &nn::Checkpoint,
    schedule: &Schedule,
    config: &TransformerConfig,
    loader: &mut WeightLoader<'_>,
) -> Result<(StepIndexed, StepIndexed)> {
    let hidden = u64::from(config.hidden_size);
    let video_out = u64::from(config.video_channels) * 4;
    let audio_out = u64::from(config.audio_channels);
    let video = fuse_modality(
        source,
        schedule,
        (Modality::Video, "proj_out", video_out),
        (hidden, config.output_heads),
        loader,
    )?;
    let audio = fuse_modality(
        source,
        schedule,
        (Modality::Audio, "audio_proj_out", audio_out),
        (hidden, config.output_heads),
        loader,
    )?;
    Ok((video, audio))
}

/// Fuse one modality's stacked heads: `(modality, checkpoint module, output width)` and
/// `(hidden width, stacked heads)`.
fn fuse_modality(
    source: &nn::Checkpoint,
    schedule: &Schedule,
    (modality, name, out): (Modality, &str, u64),
    (hidden, heads): (u64, u32),
    loader: &mut WeightLoader<'_>,
) -> Result<StepIndexed> {
    let heads = u64::from(heads);
    let block_count = schedule.num_steps()?;
    let weight = source.read(&format!("{name}.weight"))?;
    let bias = source.read(&format!("{name}.bias"))?;
    let rows = heads * out;
    if weight.shape.dims() != [rows, hidden] {
        return Err(Error::checkpoint(format!(
            "{name}.weight is {:?}, [{rows}, {hidden}] expected for {heads} stacked heads",
            weight.shape.dims()
        )));
    }
    if bias.shape.dims() != [rows] {
        return Err(Error::checkpoint(format!(
            "{name}.bias is {:?}, [{rows}] expected for {heads} stacked heads",
            bias.shape.dims()
        )));
    }
    let dtype = weight.dtype;
    if bias.dtype != dtype {
        return Err(Error::checkpoint(format!(
            "{name}'s weight is {dtype} and its bias {}",
            ops::abi::name(bias.dtype),
            dtype = ops::abi::name(dtype)
        )));
    }
    let values = ops::values::decode(dtype, &weight.bytes).map_err(nn::Error::from)?;
    let biases = ops::values::decode(dtype, &bias.bytes).map_err(nn::Error::from)?;
    let blocks = u64::from(block_count);
    let per_head = out * hidden;
    let mut fused = vec![0_f32; (blocks * per_head) as usize];
    let mut fused_bias = vec![0_f32; (blocks * out) as usize];
    for block in 0..block_count {
        let (start, weights) = schedule.block_span(modality, block)?;
        let start = u64::from(start);
        let weights = weights
            .iter()
            .map(|weight| *weight as f32)
            .collect::<Vec<_>>();
        let (block_weight, block_bias) = fuse_block(&values, &biases, out, hidden, start, &weights);
        let target = (u64::from(block) * per_head) as usize;
        fused[target..][..block_weight.len()].copy_from_slice(&block_weight);
        let target = (u64::from(block) * out) as usize;
        fused_bias[target..][..block_bias.len()].copy_from_slice(&block_bias);
    }
    let mut heads = loader.scope(format!("denoiser.{}.student_heads", modality.as_str()));
    let width = |value: u64, what: &str| {
        u32::try_from(value).map_err(|_| Error::checkpoint(format!("{name}'s {what} exceeds u32")))
    };
    Ok(StepIndexed::new(
        heads.constant_f32("weight", &[blocks, per_head], &fused)?,
        heads.constant_f32("bias", &[blocks, out], &fused_bias)?,
        width(hidden, "hidden width")?,
        width(out, "width")?,
    )?)
}

/// One block's fused head and bias: `sum_n w[n] * head[start + n]`, in FP32.
///
/// The stacked heads are `[heads * out, hidden]` and `[heads * out]`, head-major, so head `h`'s
/// weight row `o` starts at `(h * out + o) * hidden` and its bias at `h * out + o`. The products and
/// the sum stay FP32, and so does the table the caller stores them in.
fn fuse_block(
    values: &[f32],
    biases: &[f32],
    out: u64,
    hidden: u64,
    start: u64,
    weights: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let per_head = out * hidden;
    let mut weight = vec![0.0_f32; per_head as usize];
    let mut bias = vec![0.0_f32; out as usize];
    for (index, scale) in weights.iter().enumerate() {
        let head = start + index as u64;
        let source = (head * per_head) as usize;
        for (entry, target) in weight.iter_mut().enumerate() {
            *target += scale * values[source + entry];
        }
        let source = (head * out) as usize;
        for (channel, target) in bias.iter_mut().enumerate() {
            *target += scale * biases[source + channel];
        }
    }
    (weight, bias)
}

impl Transformer {
    /// The element type the stack's weights and activations are in.
    ///
    /// The witness is a block's norm gain, never an end projection: those four are pinned FP32
    /// at every preset, so they would report the pin rather than the stack.
    pub fn dtype(&self) -> DType {
        self.layers
            .first()
            .map_or_else(|| self.patch.weight().dtype, |layer| layer.block.dtype())
    }

    /// Read one DiT partition of the checkpoint.
    ///
    /// `bounds` is the deployment's `(packed rows, key tiles)`: the keys one attention call may
    /// read over every layout the deployment serves, and the tiles a sparse compression reduces
    /// over (zero for a dense stack). `heads` are a parallel-decoding student's fused heads, fused
    /// by the loading boundary when the partition stacks more than one.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        stack: (&TransformerConfig, AttentionKind),
        bounds: (u32, u32),
        sequence: SequenceSplit,
        heads: Option<(StepIndexed, StepIndexed)>,
    ) -> Result<Self> {
        let config = stack.0;
        match (config.output_heads, heads.is_some()) {
            (1, false) | (2.., true) => {}
            (count, _) => {
                return Err(Error::checkpoint(format!(
                    "the H3 partition predicts {count} output heads per evaluation and {} fused \
                     student heads were handed in: one head reads its projection, a student its \
                     fused per-block heads",
                    if heads.is_some() { "" } else { "no" }
                )));
            }
        }
        let hidden = u64::from(config.hidden_size);
        let video_in = u64::from(config.video_channels) * 4;
        let audio_in = u64::from(config.audio_channels);
        // The rotation reads this rank's heads: under Ulysses each rank attends `heads / degree` of
        // them over the whole sequence, and every head's query and key are normalized by the
        // layer's gains before they turn.
        let local_heads = config.num_attention_heads / sequence.split.size.max(1);
        let rotary = Rotary::load(
            &mut loader.scope("transformer.rotary_emb"),
            spectrum(config),
            RotarySpec {
                head_dim: config.head_dim,
                heads: local_heads,
                kv_heads: local_heads,
                qk_norm: QkNorm::Fused {
                    epsilon: config.qk_norm_eps,
                    gain: nn::layers::Gain::Plain,
                },
            },
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| Layer::load(loader, stack, index, &rotary, bounds, sequence))
            .collect::<Result<Vec<_>>>()?;
        // The four end projections are FP32 parameters at every preset, as the reference builds
        // them: they are what touches the trajectory.
        let end = |loader: &mut WeightLoader<'_>, name: &str, inputs: u64, outputs: u64| {
            Ok::<_, Error>(Linear::load(
                &mut loader.scope(name).pinned(DType::F32),
                attention::width(inputs)?,
                attention::width(outputs)?,
                Bias::Present,
                Parallel::None,
            )?)
        };
        let audio_proj_in = end(loader, "audio_proj_in", audio_in, hidden)?;
        let outputs = match heads {
            Some((video, audio)) => Heads::Fused { video, audio },
            None => Heads::Plain {
                video: end(loader, "proj_out", hidden, video_in)?,
                audio: end(loader, "audio_proj_out", hidden, audio_in)?,
            },
        };
        Ok(Self {
            patch: PatchEmbed::load(
                &mut loader.scope("proj_in").pinned(DType::F32),
                PatchEmbedSpec {
                    channels: config.video_channels,
                    temporal: 1,
                    spatial: 2,
                    hidden: config.hidden_size,
                },
            )?,
            audio_proj_in,
            rotary,
            layers,
            norm_out: AdaLnOut::load(
                &mut loader.scope("norm_out.norm"),
                config.hidden_size,
                config.norm_eps,
            )?,
            outputs,
        })
    }

    /// The stack's rotary: its spectrum, and the factor tables every layer turns by.
    pub const fn rotary(&self) -> &Rotary {
        &self.rotary
    }

    /// `source` widened to FP32, the type the FP32 end projections read; an FP32 view as it is.
    ///
    /// Every BF16 value is an FP32 value, so the widening is exact; it is where the reference's
    /// `Linear` casts the activation up before the product.
    fn wide(&self, f: &Forward, source: Operand) -> Result<Operand> {
        if source.dtype == DType::F32 {
            return Ok(source);
        }
        Ok(nn::layers::Cast.forward(f, source, DType::F32)?)
    }

    /// A video latent `[channels, frames, rows, columns]` as the DiT's `1x2x2` patch tokens.
    pub fn patchify(&self, f: &Forward, latent: Operand) -> Result<Operand> {
        Ok(self.patch.patchify(f, latent)?)
    }

    /// Patch tokens back into a video latent of extents `raster`.
    pub fn unpatchify(&self, f: &Forward, tokens: Operand, raster: Shape) -> Result<Operand> {
        Ok(self.patch.unpatchify(f, tokens, raster)?)
    }

    /// Video patch tokens projected into the hidden width, FP32.
    pub fn project_video(&self, f: &Forward, tokens: Operand) -> Result<Operand> {
        let wide = self.wide(f, tokens)?;
        Ok(self.patch.project(f, wide)?)
    }

    /// Audio latent rows projected into the hidden width, FP32.
    pub fn project_audio(&self, f: &Forward, rows: Operand) -> Result<Operand> {
        let wide = self.wide(f, rows)?;
        Ok(self.audio_proj_in.forward(f, wide)?)
    }

    /// The final normalization and one modality's output head: the stack's rows in, the FP32
    /// velocity rows out.
    ///
    /// `modulation` is the step's final-norm table and each row's group; `step` is the evaluation
    /// `[1]` i32 a student's fused heads are selected by, and `block` its host value.
    pub fn velocity(
        &self,
        f: &Forward,
        hidden: Operand,
        modality: Modality,
        modulation: RowModulation<'_>,
        (block, step): (u32, Operand),
    ) -> Result<Operand> {
        let normalized = self.norm_out.forward(f, modulation, hidden)?;
        let wide = self.wide(f, normalized)?;
        match &self.outputs {
            Heads::Fused { video, audio } => {
                let fused = match modality {
                    Modality::Video => video,
                    Modality::Audio => audio,
                };
                if u64::from(block) >= fused.steps() {
                    return Err(Error::checkpoint(format!(
                        "H3 evaluation {block} is outside the {} fused student heads",
                        fused.steps()
                    )));
                }
                Ok(fused.forward(f, step, wide)?)
            }
            Heads::Plain { video, audio } => Ok(match modality {
                Modality::Video => video.forward(f, wide)?,
                Modality::Audio => audio.forward(f, wide)?,
            }),
        }
    }

    /// The stack over `hidden`, this rank's `[rows, hidden_size]` packed rows, in place, at one
    /// step.
    ///
    /// `adaln` is the AdaLN table, `entries` the `[1, groups]` entries the step reads
    /// ([`TimestepTable::entries_of`]), and `rows` this rank's `[rows]` AdaLN row of each token.
    pub fn forward(
        &self,
        f: &Forward,
        (adaln, entries): (&TimestepTable, Operand),
        hidden: Operand,
        rows: Operand,
        (factors, visibility): (nn::layers::Factors, &Visibility),
    ) -> Result<()> {
        let width = hidden.shape.last();
        let groups = entries.shape.numel();
        // Block `index`'s `[3 · groups, 6 · width]` modulation: the gathered `[groups, 18 · width]`
        // rows, group `g`'s row holding its three tags' six vectors each.
        let table = |f: &Forward, index: usize| -> Result<ModulationTable> {
            let gathered = adaln.layer(f, index, entries)?;
            Ok(ModulationTable::block(
                gathered.reshape(Shape::new(&[
                    groups * u64::from(TAGS),
                    VECTORS as u64 * width,
                ])),
                width,
            )?)
        };
        // Under the gather projection the whole sequence's normalized rows live across the layers:
        // each layer's are gathered while the layer before it computes.
        let gathered = match self.layers.first().map(|layer| &layer.attention) {
            Some(attention::Layer::Sparse(sparse)) if sparse.gathers_ahead() => {
                let members = u64::from(sparse.members());
                Some((
                    sparse,
                    f.empty(hidden.dtype, Shape::new(&[members * hidden.rows(), width]))?,
                ))
            }
            _ => None,
        };
        let mut arriving = Vec::new();
        let rotation = (&self.rotary, factors);
        f.fenced(TRANSFORMER, format_args!("transformer"), || {
            if let Some((sparse, gathered)) = gathered {
                // The first layer has no layer before it to start its gather.
                f.scoped(|f| {
                    let own = hidden.rows();
                    let block = gathered.narrow(0, own * u64::from(sparse.rank()), own)?;
                    let first = &self.layers[0];
                    let table = table(f, 0)?;
                    first.block.attention_input_into(
                        f,
                        RowModulation {
                            table: &table,
                            rows,
                        },
                        hidden,
                        block,
                    )?;
                    arriving.push(sparse.start_gather(f, gathered, own, (0, own))?);
                    Ok::<_, Error>(())
                })?;
            }
            for (index, layer) in self.layers.iter().enumerate() {
                f.fenced(TRANSFORMER_LAYER, format_args!("layers.{index}"), || {
                    f.scoped(|f| {
                        let own_table = table(f, index)?;
                        let modulation = RowModulation {
                            table: &own_table,
                            rows,
                        };
                        let next_table = match (&gathered, self.layers.get(index + 1)) {
                            (Some(_), Some(_)) => Some(table(f, index + 1)?),
                            _ => None,
                        };
                        let gathered = gathered.map(|(_, gathered)| gathered);
                        // The next layer's rows are gathered ahead only under the gather
                        // projection.
                        let next = gathered.and(
                            self.layers
                                .get(index + 1)
                                .zip(next_table.as_ref())
                                .map(|(next, table)| (next, RowModulation { table, rows })),
                        );
                        layer.forward(
                            f,
                            modulation,
                            hidden,
                            (rotation, visibility),
                            (gathered, next),
                            &mut arriving,
                        )
                    })
                })?;
            }
            Ok(())
        })
    }
}

/// The output heads of a partition: one projection per modality, or a student's fused heads.
enum Heads {
    Plain {
        video: Linear,
        audio: Linear,
    },
    Fused {
        video: StepIndexed,
        audio: StepIndexed,
    },
}

/// One H3 DiT partition: the projections, the stack and the two output heads.
pub struct Transformer {
    /// The video latent's `1x2x2` patch tokens and their projection into the hidden width.
    patch: PatchEmbed,
    audio_proj_in: Linear,
    /// The stack's one rotary embedding, published once and applied by every layer.
    rotary: Rotary,
    layers: Vec<Layer>,
    norm_out: AdaLnOut,
    outputs: Heads,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blocks_head_is_the_weighted_sum_of_its_stacked_heads() {
        // Two heads of one 2x3 output, stacked head-major: head 0 at rows 0..2, head 1 at 2..4.
        let values = vec![
            1.0, 2.0, 3.0, // head 0, row 0
            4.0, 5.0, 6.0, // head 0, row 1
            10.0, 20.0, 30.0, // head 1, row 0
            40.0, 50.0, 60.0, // head 1, row 1
        ];
        let biases = vec![0.5, -0.5, 5.0, -5.0];
        let (weight, bias) = fuse_block(&values, &biases, 2, 3, 0, &[0.25, 0.75]);
        assert_eq!(weight[0], 0.25 * 1.0 + 0.75 * 10.0);
        assert_eq!(weight[5], 0.25 * 6.0 + 0.75 * 60.0);
        assert_eq!(bias[0], 0.25 * 0.5 + 0.75 * 5.0);
        assert_eq!(bias[1], 0.25 * -0.5 + 0.75 * -5.0);
        assert_eq!(weight.len(), 6, "one block's 2x3 head");
    }

    #[test]
    fn a_block_of_the_second_interval_reads_only_its_own_heads() {
        let values = vec![1.0, 2.0, 100.0, 200.0];
        let biases = vec![0.0, 9.0];
        let (weight, bias) = fuse_block(&values, &biases, 1, 2, 1, &[1.0]);
        assert_eq!(weight, vec![100.0, 200.0]);
        assert_eq!(bias, vec![9.0]);
    }
}
