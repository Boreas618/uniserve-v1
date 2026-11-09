//! The diffusion transformer: the packed rows in, one evaluation's velocity out.
//!
//! One `Transformer` is a checkpoint's DiT partition: a projection of each modality's patchified
//! latent into the hidden width, the stack of dilated layers, and the output projections back to
//! each modality's velocity. A layer is an [`AdaLnBlock`] over H3's attention, conditioned by the
//! step's AdaLN rows, and the final normalization is an [`AdaLnOut`].
//!
//! The step's AdaLN rows are device data gathered from the [`TimestepTable`] in H3's row layout
//! ([`crate::modulation`]), so one captured stack serves every step of the schedule.

use crate::attention::{self, Factors};
use crate::attention_sparse::Segments;
use crate::config::{AttentionKind, TransformerConfig};
use crate::modulation::{TAGS, VECTORS};
use crate::partition::Partition;
use crate::schedule::{Modality, Schedule};
use crate::{Error, Result};

use nn::blocks::{AdaLnBlock, AdaLnOut, AdaLnSpec, GateUp, GatedMlpSpec, ModulationTable};
use nn::diffusion::TimestepTable;
use nn::family::Split;
use nn::layers::{
    Bias, Linear, Parallel, PatchEmbed, PatchEmbedSpec, Precision, QkNorm, Rotary, RotarySpec,
    Rounding, Spectrum, StepIndexed, Varlen,
};
use nn::weights::WeightLoader;
use nn::{Forward, Shape, TensorView};

/// The components a deployment's fence probes name (`profiling_config.fence_components`): the
/// denoiser's stack and one of its layers.
const TRANSFORMER: &str = "models.minimax_h3.denoiser.Transformer";
const TRANSFORMER_LAYER: &str = "models.minimax_h3.denoiser.TransformerLayer";

/// The coordinate axes of a packed row's rotary position: time, height and width.
const ROTARY_AXES: u32 = 3;

/// The DiT's rotary spectrum: `rope_frequency_dim` frequencies repeated over the three axes.
///
/// The reference builds its rotary embedding as `RotaryEmbedding(2 * rope_frequency_dim)` and hands
/// it one axis spanning the whole head, so the same spectrum is evaluated at each of the three
/// coordinates and the axes' factors are three contiguous blocks of one table; the trailing
/// channels of a head pass through. The table is evaluated in f64.
pub fn spectrum(config: &TransformerConfig) -> Spectrum {
    Spectrum::Repeated {
        theta: config.rope_theta,
        pairs: config.rope_frequency_dim,
        axes: ROTARY_AXES,
        precision: Precision::F64,
    }
}

/// One dilated layer of the stack: an AdaLN block over the layer's dense or sparse attention.
pub struct Layer {
    /// The modulated norms and the value-first SwiGLU, `silu(gate)` and the product evaluated with
    /// one rounding each (`Rounding::Stepwise`), as the reference's activation rounds.
    block: AdaLnBlock,
    attention: attention::Layer,
}

impl Layer {
    /// Read layer `index` of a partition, turning by the stack's `rotary`.
    #[allow(
        clippy::too_many_arguments,
        reason = "a layer is its index, the stack's configuration, rotary and bounds, the partition \
                  and the loader"
    )]
    fn load(
        kind: AttentionKind,
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotary: &Rotary,
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
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
            attention: attention::Layer::load(
                kind, index, config, rows, rotary, tile_bound, partition, loader,
            )?,
        })
    }

    /// One layer from `hidden` into `out`, modulated by the step's gathered rows.
    ///
    /// `modulation` is this layer's `[3 · groups, 6 · hidden]` table and `rows` the `[tokens]`
    /// AdaLN row each token reads. The layer's output lands in `hidden`, and `out` holds the
    /// attention's residual in between, so no copy of the residual is made.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        f: &Forward,
        hidden: TensorView,
        out: TensorView,
        modulation: &ModulationTable,
        rows: TensorView,
        factors: Factors<'_>,
        visible: Option<&Varlen>,
        segments: Option<&Segments>,
        scratch: &Scratch,
        next: Option<(&Layer, &ModulationTable)>,
        arrived: Vec<nn::Transfer>,
    ) -> Result<Vec<nn::Transfer>> {
        let normalized = scratch.normalized;
        let update = scratch.update;

        // Attention: normalize, attend, and fold the gated update into `out` while normalizing the
        // feed-forward's input.
        // Under Ulysses the attended rows return interval by interval, and what follows attention
        // reads a row alone, so an interval's output projection and feed-forward run while the
        // later intervals travel.
        if let (attention::Layer::Sparse(sparse), attention::LayerScratch::Sparse(views)) =
            (&self.attention, &scratch.attention)
            && sparse.exchanges()
        {
            let tables = segments.ok_or_else(|| {
                Error::Unsupported("a sparse layer was handed no segment views".to_owned())
            })?;
            let own = hidden.rows();
            let ahead = sparse.gathers_ahead();
            let mut arrived = arrived;
            // The stack's first layer has no layer before it to start its gather.
            if !ahead {
                self.block
                    .attention_input(f, modulation, rows, hidden, normalized)?;
            } else if arrived.is_empty() {
                let rows_here = sparse.gathered_rows(views, own, 0, own);
                self.block
                    .attention_input(f, modulation, rows, hidden, rows_here)?;
                arrived.push(sparse.start_gather(f, views, own, 0, own)?);
            }
            let gate = sparse.attend(
                f,
                normalized,
                factors,
                tables.sizes,
                tables.segments,
                tables.keep,
                views,
                arrived,
            )?;
            let mut gathering = Vec::new();
            for interval in sparse.deliver(f, gate, views, own)? {
                let (first, count) = (interval.first, interval.count);
                let part = |view: TensorView| row_interval(view, first, count);
                let selected = rows
                    .at(rows.offset_elements(first))
                    .reshape(nn::Shape::new(&[count]));
                sparse.receive(f, interval, views, part(update))?;
                self.block.feed_forward(
                    f,
                    modulation,
                    selected,
                    part(hidden),
                    part(update),
                    part(out),
                    part(normalized),
                )?;
                // These rows are the next layer's input, and its attention reads the whole
                // sequence, so they leave now, beside the rest of this layer.
                if ahead && let Some((layer, modulation)) = next {
                    let rows_here = sparse.gathered_rows(views, own, first, count);
                    layer.block.attention_input(
                        f,
                        modulation,
                        selected,
                        part(hidden),
                        rows_here,
                    )?;
                    gathering.push(sparse.start_gather(f, views, own, first, count)?);
                }
            }
            return Ok(gathering);
        }
        self.block.forward(
            f,
            modulation,
            rows,
            hidden,
            normalized,
            update,
            out,
            |normalized, update| {
                Ok(self.attention.forward(
                    f,
                    normalized,
                    factors,
                    visible,
                    segments,
                    &scratch.attention,
                    update,
                )?)
            },
        )?;
        Ok(Vec::new())
    }
}

/// Rows `first..first + count` of a `[rows, width]` view.
fn row_interval(view: TensorView, first: u64, count: u64) -> TensorView {
    let width = view.shape.last();
    view.at(view.offset_elements(first * width))
        .reshape(nn::Shape::new(&[count, width]))
}

/// A parallel-decoding student's output heads, fused per evaluation block at load.
///
/// A DMD/PDD student stacks one head per fine interval (`pdd_steps`, e.g. 32) and predicts one
/// block's velocity from the weighted sum of its intervals' heads (weights and accumulation in
/// FP32, rounded to the parameter's dtype). The fusion is arithmetic over the checkpoint's own
/// numbers, so it is done once, on the host, at the loading boundary; what a forward does is pick
/// one row.
///
/// The table is laid out `[blocks, out * hidden]` so that one evaluation's head is **one table
/// row**: `IndexRows::gather` indexed by the worker's `step` view (`[1]` int32) gives `[1, out *
/// hidden]`, and a projection over that view's element type is the ordinary output projection. The
/// gathered buffers sit at fixed arena addresses with contents that change per evaluation, so the
/// graph does not change: one capture serves every step of a layout.
pub struct StudentHeads {
    video: FusedHeads,
    audio: Option<FusedHeads>,
}

/// One modality's fused heads: one projection per evaluation block, `[blocks, out * hidden]`
/// FP32 weights and `[blocks, out]` biases.
///
/// FP32 because the reference builds a student's head FP32: the fused row is a sum of the
/// checkpoint's BF16 rows and rounding it back would round a number the reference keeps exact. The
/// projection reads the widened FP32 rows (`Transformer::wide`), so its two operands are one type
/// and the only rounding is the store into the velocity.
struct FusedHeads {
    heads: StepIndexed,
}

impl StudentHeads {
    /// Fuse a checkpoint's stacked interval heads into one head per evaluation block.
    ///
    /// The checkpoint stacks `output_heads` heads along the first axis of `proj_out` and
    /// `audio_proj_out` (`[output_heads * out, hidden]`, `[output_heads * out]`). Block `b`'s fused
    /// weight is `sum_n w[n] * head[start + n]` with `w` the block's normalized integration weights,
    /// the products and the sum in FP32, and the result rounded to the checkpoint's own type -
    /// exactly `fuse_heads`. One derivation per block
    /// per modality, once per load, published as a constant of the loader's store.
    pub fn fuse(
        source: &nn::Checkpoint,
        schedule: &Schedule,
        config: &TransformerConfig,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let hidden = u64::from(config.hidden_size);
        let video_out = u64::from(config.video_channels) * 4;
        let audio_out = u64::from(config.audio_channels);
        let video = FusedHeads::fuse(
            source,
            schedule,
            Modality::Video,
            "proj_out",
            video_out,
            hidden,
            config.output_heads,
            loader,
        )?;
        let audio = FusedHeads::fuse(
            source,
            schedule,
            Modality::Audio,
            "audio_proj_out",
            audio_out,
            hidden,
            config.output_heads,
            loader,
        )?;
        Ok(Self {
            video,
            audio: Some(audio),
        })
    }
}

impl FusedHeads {
    #[allow(clippy::too_many_arguments)]
    fn fuse(
        source: &nn::Checkpoint,
        schedule: &Schedule,
        modality: Modality,
        name: &str,
        out: u64,
        hidden: u64,
        heads: u32,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
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
                bias.dtype
            )));
        }
        let values = dtype.decode(&weight.bytes).map_err(nn::Error::from)?;
        let biases = dtype.decode(&bias.bytes).map_err(nn::Error::from)?;
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
            let (block_weight, block_bias) =
                fuse_block(&values, &biases, out, hidden, start, &weights);
            let target = (u64::from(block) * per_head) as usize;
            fused[target..][..block_weight.len()].copy_from_slice(&block_weight);
            let target = (u64::from(block) * out) as usize;
            fused_bias[target..][..block_bias.len()].copy_from_slice(&block_bias);
        }
        let mut heads = loader.scope(format!("denoiser.{}.student_heads", modality.as_str()));
        let width = |value: u64, what: &str| {
            u32::try_from(value)
                .map_err(|_| Error::checkpoint(format!("{name}'s {what} exceeds u32")))
        };
        Ok(Self {
            heads: StepIndexed::new(
                heads.constant_f32("weight", &[blocks, per_head], &fused)?,
                heads.constant_f32("bias", &[blocks, out], &fused_bias)?,
                width(hidden, "hidden width")?,
                width(out, "width")?,
            )?,
        })
    }
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
    /// Project one modality's patchified audio rows into the hidden width.
    pub fn project_audio(&self, f: &Forward, rows: TensorView, out: TensorView) -> Result<()> {
        let audio = self.audio_proj_in.as_ref().ok_or_else(|| {
            Error::Unsupported("this H3 partition has no audio input projection".to_owned())
        })?;
        // The audio input projection reads an FP32 activation in the reference like the video's, so
        // the rows widen first when they are not FP32 already: the audio latent's own width (32) is
        // a declared `scaled_add` column count, which is what makes the widening launchable. The
        // weight is the checkpoint's own - it is never widened at load - so the product is an FP32
        // activation against a BF16 weight with an FP32 output, and the caller's `out` decides the
        // one rounding that follows it.
        let rows = self.wide(f, rows)?;
        Ok(audio.forward(f, rows, out)?)
    }

    /// The stack's rotary: its spectrum, and the factor tables every layer turns by.
    pub const fn rotary(&self) -> &Rotary {
        &self.rotary
    }

    /// The rotary factor tables of the packed rows, from their `[3, rows]` axis-major positions.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        Ok(self.rotary.factors(f, positions, cosine, sine)?)
    }

    /// The final normalization and one modality's output projection: `hidden` in, velocity out.
    ///
    /// The final normalization is modulated like every other one: `[groups, 2 * hidden]` of shift
    /// and scale, gathered at the AdaLN entry the step reads. `rows` selects each row's group.
    #[allow(clippy::too_many_arguments)]
    pub fn velocity(
        &self,
        f: &Forward,
        hidden: TensorView,
        modality: Modality,
        modulation: &ModulationTable,
        rows: TensorView,
        block: u32,
        step: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let normalized = self.norm_out.apply(f, modulation, rows, hidden)?;
        // The second crossing (`crate::denoiser::SAMPLE_DTYPE`): the modulated rows are the
        // stack's type and widen here, between the norm and the FP32 head.
        let wide = self.wide(f, normalized)?;
        self.project_head(f, wide, modality, block, step, out)
    }
}

/// One H3 DiT partition: the projections, the stack and the two output heads.
pub struct Transformer {
    /// The video latent's `1x2x2` patch tokens and their projection into the hidden width.
    patch: PatchEmbed,
    audio_proj_in: Option<Linear>,
    /// The stack's one rotary embedding, published once and applied by every layer.
    rotary: Rotary,
    layers: Vec<Layer>,
    norm_out: AdaLnOut,
    /// The base form's one output head per modality. A student fuses its heads instead.
    video_out: Option<Linear>,
    audio_out: Option<Linear>,
    /// A parallel-decoding student's fused heads, when the checkpoint stacks more than one.
    heads: Option<StudentHeads>,
}

impl Transformer {
    /// The element type the checkpoint's weights and the activations they produce are in.
    ///
    /// One type for the whole stack: the precision rule keeps the checkpoint's own representation,
    /// and every layer's operands are of it.
    ///
    /// The witness is a **block's** norm gain and not an end projection. Those four are pinned FP32
    /// at every preset, so `proj_in` reports the pin rather than the
    /// stack: at `auto` a BF16 deployment would claim FP32, a caller would allocate FP32
    /// activations against BF16 blocks, and the sampler's first layer gemm would be the mixed pair
    /// this branch has already spent rounds removing (`input_f32_weight_bf16`). A block's gain is
    /// never pinned, so it is the deployment's type by construction.
    pub fn dtype(&self) -> nn::DType {
        self.layers
            .first()
            .map_or_else(|| self.patch.weight().dtype, |layer| layer.block.dtype())
    }

    /// Read one DiT partition of the checkpoint.
    ///
    /// `rows` bounds one call's keys: the largest packed row count the deployment's layouts hold,
    /// not a request's count, so one partition serves every layout.
    pub fn load(
        config: &TransformerConfig,
        kind: AttentionKind,
        rows: u32,
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
        heads: Option<StudentHeads>,
    ) -> Result<Self> {
        // A sparse deployment's key-tile bound is a statement about the deployment (`Layouts::
        // tiles_bound`), and every layer that compresses reduces over it. Zero is legal only for a
        // dense stack, where nothing reads it: a sparse stack holding zero would hand `VsaAttention` a
        // bound of nothing, and the descriptor a plan reads would name a reduction over no tiles
        // rather than refuse. A caller that passes zero for a sparse deployment is refused here
        // instead of one layer later.
        if matches!(kind, AttentionKind::Sparse { .. }) && tile_bound == 0 {
            return Err(Error::config(
                "a sparse deployment states its key-tile bound and this stack was handed zero: \
                 `Layouts::tiles_bound` is the deployment's own, and a compression reduces over it"
                    .to_owned(),
            ));
        }
        match (config.output_heads, heads.is_some()) {
            (1, false) | (2.., true) => {}
            (0, _) => {
                return Err(Error::checkpoint(
                    "the H3 partition predicts no output heads".to_owned(),
                ));
            }
            (1, true) => {
                return Err(Error::checkpoint(
                    "the H3 partition predicts one output head per evaluation, and fused student \
                     heads were handed in"
                        .to_owned(),
                ));
            }
            (count, false) => {
                return Err(Error::Unsupported(format!(
                    "the H3 partition predicts {count} output heads per evaluation (a \
                     parallel-decoding student), and the loading boundary fused none: a student is \
                     served from its fused per-block heads, never from a plain projection over a \
                     [count * out, hidden] table"
                )));
            }
        }
        let hidden = u64::from(config.hidden_size);
        let video_in = u64::from(config.video_channels) * 4;
        let audio_in = u64::from(config.audio_channels);
        // The rotation reads *this rank's* heads: under Ulysses each rank attends `heads / degree`
        // of them over the whole sequence (`Partition::denoiser_heads`), and every head's query and
        // key are normalized by the layer's gains before they turn.
        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        let rotary = Rotary::load(
            &mut loader.scope("transformer.rotary_emb"),
            spectrum(config),
            RotarySpec {
                head_dim: config.head_dim,
                heads: local_heads,
                kv_heads: local_heads,
                qk_norm: QkNorm::Fused {
                    epsilon: config.qk_norm_eps,
                },
            },
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                Layer::load(
                    kind, index, config, rows, &rotary, tile_bound, partition, loader,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            // **The four end projections are FP32 parameters at every preset**, because they are
            // what touches the trajectory: `video_input`, `audio_input`, `video_output` and
            // `audio_output` are pinned at FP32 whatever the attention, MLP and VAE tiers are. So
            // the widening `wide` performs meets an FP32 weight and the product is FP32 on both
            // sides, which is what makes the released BF16 stack run a FP32 sample at all.
            patch: PatchEmbed::load(
                &mut loader.scope("proj_in").pinned(nn::DType::F32),
                PatchEmbedSpec {
                    channels: config.video_channels,
                    temporal: 1,
                    spatial: 2,
                    hidden: config.hidden_size,
                },
            )?,
            audio_proj_in: Some(Linear::load(
                &mut loader.scope("audio_proj_in").pinned(nn::DType::F32),
                attention::width(audio_in)?,
                attention::width(hidden)?,
                Bias::Present,
                Parallel::None,
            )?),
            rotary,
            layers,
            norm_out: AdaLnOut::load(
                &mut loader.scope("norm_out.norm"),
                config.hidden_size,
                config.norm_eps,
            )?,
            // A student's checkpoint holds the stacked interval heads, so the plain projection is
            // read only when the partition has one head.
            video_out: match heads {
                Some(_) => None,
                None => Some(Linear::load(
                    &mut loader.scope("proj_out").pinned(nn::DType::F32),
                    attention::width(hidden)?,
                    attention::width(video_in)?,
                    Bias::Present,
                    Parallel::None,
                )?),
            },
            audio_out: match heads {
                Some(_) => None,
                None => Some(Linear::load(
                    &mut loader.scope("audio_proj_out").pinned(nn::DType::F32),
                    attention::width(hidden)?,
                    attention::width(audio_in)?,
                    Bias::Present,
                    Parallel::None,
                )?),
            },
            heads,
        })
    }

    /// The FP32 view an FP32 projection reads when `source` is not FP32 already.
    ///
    /// The reference builds all four of the transformer's end projections as FP32 parameters
    /// and its `Linear` casts the activation up before the
    /// product, so the projection reads FP32 whatever the stack's own element type is. The
    /// checkpoint's weights need no conversion: a BF16 value is exactly representable in FP32, so a
    /// product of FP32 inputs against them is the same arithmetic as against widened FP32 weights -
    /// widening changes no value and the accumulation is FP32 either way. What the copy buys is the
    /// input's rounding, which is the last place the stack's element type would otherwise reach the
    /// prediction.
    fn wide(&self, f: &Forward, source: TensorView) -> Result<TensorView> {
        if source.dtype == nn::DType::F32 {
            return Ok(source);
        }
        // A widening: every BF16 value is an FP32 value, so the conversion is exact.
        Ok(nn::layers::Cast.apply(f, source, nn::DType::F32)?)
    }

    /// Cut a video latent `[channels, frames, rows, columns]` into the DiT's `1x2x2` patch tokens
    /// `[tokens, channels · 4]`.
    pub fn patchify(&self, f: &Forward, latent: TensorView, tokens: TensorView) -> Result<()> {
        Ok(self.patch.patchify(f, latent, tokens)?)
    }

    /// Unpack patch tokens back into the video latent's raster; `out`'s extents state the grid, so
    /// a clip of another length replays the same launch.
    pub fn unpatchify(&self, f: &Forward, tokens: TensorView, out: TensorView) -> Result<()> {
        Ok(self.patch.unpatchify(f, tokens, out)?)
    }

    /// Project video patch tokens into the hidden width.
    pub fn project(&self, f: &Forward, tokens: TensorView, out: TensorView) -> Result<()> {
        let wide = self.wide(f, tokens)?;
        Ok(self.patch.project(f, wide, out)?)
    }

    /// The output projection of `hidden` at one evaluation, for either form of the heads.
    fn project_head(
        &self,
        f: &Forward,
        hidden: TensorView,
        modality: Modality,
        block: u32,
        step: TensorView,
        out: TensorView,
    ) -> Result<()> {
        if let Some(heads) = &self.heads {
            let fused = match modality {
                Modality::Video => &heads.video,
                Modality::Audio => heads.audio.as_ref().ok_or_else(|| {
                    Error::Unsupported("this H3 partition predicts no audio velocity".to_owned())
                })?,
            };
            if u64::from(block) >= fused.heads.steps() {
                return Err(Error::checkpoint(format!(
                    "H3 evaluation {block} is outside the {} fused student heads",
                    fused.heads.steps()
                )));
            }
            return Ok(fused.heads.forward(f, step, hidden, out)?);
        }
        match modality {
            Modality::Video => {
                let head = self.video_out.as_ref().ok_or_else(|| {
                    Error::Unsupported("this H3 partition predicts no video velocity".to_owned())
                })?;
                Ok(head.forward(f, hidden, out)?)
            }
            Modality::Audio => {
                let head = self.audio_out.as_ref().ok_or_else(|| {
                    Error::Unsupported("this H3 partition predicts no audio velocity".to_owned())
                })?;
                Ok(head.forward(f, hidden, out)?)
            }
        }
    }

    /// The stack over `hidden` `[tokens, hidden_size]`, in place, at one step.
    ///
    /// `adaln` is the cache the worker bound for this request, `step` the `[1]` i32 step index
    /// the worker wrote before the replay, and `rows` the `[tokens]` AdaLN row of each token.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        adaln: &TimestepTable,
        step: TensorView,
        hidden: TensorView,
        rows: TensorView,
        factors: Factors<'_>,
        visible: Option<&Varlen>,
        segments: Option<&Segments>,
        total_rows: u32,
    ) -> Result<()> {
        f.fenced(TRANSFORMER, format_args!("transformer"), || {
            self.stack(
                f, adaln, step, hidden, rows, factors, visible, segments, total_rows,
            )
        })
    }

    /// [`Self::forward`]'s work, between the stack's fence probes.
    #[allow(clippy::too_many_arguments)]
    fn stack(
        &self,
        f: &Forward,
        adaln: &TimestepTable,
        step: TensorView,
        hidden: TensorView,
        rows: TensorView,
        factors: Factors<'_>,
        visible: Option<&Varlen>,
        segments: Option<&Segments>,
        total_rows: u32,
    ) -> Result<()> {
        let groups = adaln.groups();
        let width = hidden.shape.last();
        let tokens = hidden.shape.dim(0);
        let arena = f.arena();
        let entries = arena.tensor(nn::DType::I32, Shape::new(&[1, groups]))?;
        adaln.entries_of(f, step, entries)?;
        // **The gather's own arrangement: one row per entry, the whole 18-vector table across the
        // columns** (`IndexRows::gather` derives its output from the table's columns and the index count,
        // so a buffer of the same bytes in another shape is refused by name:
        // `out is [12, 32256], [4, 96768] expected`). The block cuts three intervals per group out
        // of it, and that is a view of the same flat order - group `g`'s row holds its three
        // intervals' six vectors each, in order - so the block reads `[3 · groups, 6 · width]` over
        // these bytes and nothing is copied.
        let modulation = arena.tensor(
            hidden.dtype,
            Shape::new(&[groups, u64::from(TAGS) * VECTORS as u64 * width]),
        )?;
        // Two tables, because a layer's last rows are normalized for the next layer while the
        // layer still reads its own.
        let following = arena.tensor(modulation.dtype, modulation.shape)?;
        let tables = [modulation, following];
        let [current, following] = tables.map(|table| {
            ModulationTable::block(
                table.reshape(Shape::new(&[
                    groups * u64::from(TAGS),
                    VECTORS as u64 * width,
                ])),
                width,
            )
        });
        let per_layer = [current?, following?];
        let alternate = arena.tensor(hidden.dtype, Shape::new(&[tokens, width]))?;
        let scratch = Scratch::reserve(self, f, hidden, total_rows)?;
        adaln.layer(f, 0, entries, tables[0])?;
        let mut arrived = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let next = self.layers.get(index + 1);
            if next.is_some() {
                adaln.layer(f, index + 1, entries, tables[(index + 1) % 2])?;
            }
            arrived = f.fenced(TRANSFORMER_LAYER, format_args!("layers.{index}"), || {
                layer.forward(
                    f,
                    hidden,
                    alternate,
                    &per_layer[index % 2],
                    rows,
                    factors,
                    visible,
                    segments,
                    &scratch,
                    next.map(|layer| (layer, &per_layer[(index + 1) % 2])),
                    arrived,
                )
            })?;
        }
        Ok(())
    }
}

/// Every temporary view the stack reads, reserved once for the whole forward.
///
/// The worker's arena is a bump allocator reset once per forward, so scratch that does not depend
/// on the layer is reserved once: reserving it per layer would multiply the arena by the layer
/// count, which a 50-layer stack cannot afford at a served canvas.
pub struct Scratch {
    normalized: TensorView,
    update: TensorView,
    attention: attention::LayerScratch,
}

impl Scratch {
    /// Reserve the stack's scratch for `hidden` `[tokens, width]` with a key set of `total_rows`.
    pub fn reserve(
        transformer: &Transformer,
        f: &Forward,
        hidden: TensorView,
        total_rows: u32,
    ) -> Result<Self> {
        let arena = f.arena();
        let dtype = hidden.dtype;
        let tokens = hidden.shape.dim(0);
        let width = hidden.shape.last();
        let layer = transformer.layers.first().ok_or_else(|| {
            Error::Unsupported("a stack with no layers has no scratch".to_owned())
        })?;
        Ok(Self {
            normalized: arena.tensor(dtype, Shape::new(&[tokens, width]))?,
            update: arena.tensor(dtype, Shape::new(&[tokens, width]))?,
            attention: layer.attention.reserve(f, hidden, total_rows)?,
        })
    }
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
