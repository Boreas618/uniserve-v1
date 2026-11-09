//! The diffusion transformer: the packed rows in, one evaluation's velocity out.
//!
//! One `Transformer` is a checkpoint's DiT partition: a projection of each modality's patchified
//! latent into the hidden width, the stack of dilated layers, and the output projections back to
//! each modality's velocity. A layer is conditioned
//! by the step's AdaLN rows - a modulated normalization of the attention's input, the attention, the
//! gated residual into the feed-forward's normalization, and the feed-forward's own gated residual:
//!
//! ```text
//! normalized = modulated_rms_norm(hidden, norm1, shift_a, scale_a, rows)
//! attended   = attention(normalized)
//! residual   = hidden + gate_a[rows] * attended            (one launch, with the normalization)
//! normalized = modulated_rms_norm(residual, norm2, shift_m, scale_m, rows)
//! hidden     = residual + gate_m[rows] * mlp(normalized)
//! ```
//!
//! The step's AdaLN rows are device data ([`crate::modulation`]), so one captured stack serves
//! every step of the schedule. The tables themselves are the worker's: the model reads the cache
//! through the borrow it is handed, never through state of its own.

use crate::attention::{self, Factors, Visible};
use crate::attention_sparse::Segments;
use crate::config::{AttentionKind, TransformerConfig};
use crate::modulation::AdaLnCache;
use crate::partition::Partition;
use crate::schedule::{Modality, Schedule};
use crate::{Error, Result};

use nn::layers::{
    Bias, Epilogue, GatedActivation, Linear, Modulation, Parallel, Residual, ResidualStream,
    RmsNorm, Scale,
};
use nn::tensor::Backing;
use nn::weights::{Weight, WeightLoader};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

/// The components a deployment's fence probes name (`profiling_config.fence_components`): the
/// denoiser's stack and one of its layers.
const TRANSFORMER: &str = "models.minimax_h3.denoiser.Transformer";
const TRANSFORMER_LAYER: &str = "models.minimax_h3.denoiser.TransformerLayer";

/// The two halves of a packed `[value, gate]` projection, as strided views without a copy.
///
/// The checkpoint fuses the feed-forward's value half first and its gate half second, and the
/// reference projects it once and then reads `branches["up"]` and `branches["gate"]`.
/// The stepwise activation takes the halves as separate operands, so
/// they are views of the packed buffer: `value` at element zero, `gate` one half along, and both
/// rows striding over the packed row.
pub(crate) fn projection_halves(projection: TensorView) -> (TensorView, TensorView) {
    let width = projection.last() / 2;
    let shape = Shape::new(&[projection.rows(), width]);
    let strides = projection.strides;
    let value = TensorView::with_strides(
        projection.element(&[0, 0]),
        projection.dtype,
        shape,
        strides,
    );
    let gate = TensorView::with_strides(
        projection.element(&[0, width]),
        projection.dtype,
        shape,
        strides,
    );
    (value, gate)
}

/// The rotary width of one head's three axes: `2 * rope_frequency_dim` channels each.
///
/// The width is the checkpoint's `rope_freq_dim`, not a constant of this module: the reference
/// builds its rotary embedding as `RotaryEmbedding(2 * rope_frequency_dim)`
/// and applies that one spectrum to each of the
/// three coordinates. Three is the model's own count: `position_ids` is
/// `[3, rows]`, one coordinate for time, height and width.
pub fn rotated_axes(config: &TransformerConfig) -> [u32; 3] {
    [2 * config.rope_frequency_dim; 3]
}

/// One dilated layer of the stack.
pub struct Layer {
    norm1: Weight,
    norm2: Weight,
    attention: attention::Layer,
    gate_up: Linear,
    down: Linear,
    /// The stepwise gated activation (`GatedActivation::with_stepwise`): `silu(gate)` and the
    /// product are evaluated with one rounding each.
    gated: GatedActivation,
    eps: f32,
}

impl Layer {
    /// Read layer `index` of a partition.
    fn load(
        kind: AttentionKind,
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let prefix = format!("transformer_blocks.{index}");
        let hidden = u64::from(config.hidden_size);
        let ffn = u64::from(config.intermediate_size);
        Ok(Self {
            norm1: loader
                .scope(format!("{prefix}.norm1"))
                .tensor("weight", &[hidden])?,
            norm2: loader
                .scope(format!("{prefix}.norm2"))
                .tensor("weight", &[hidden])?,
            attention: attention::Layer::load(
                kind,
                index,
                config,
                rows,
                &rotated_axes(config),
                tile_bound,
                partition,
                loader,
            )?,
            // The checkpoint fuses the feed-forward's value half first and its gate half second.
            gate_up: Linear::load(
                &mut loader.scope(format!("{prefix}.ff.net.0.proj")),
                attention::width(hidden)?,
                attention::width(2 * ffn)?,
                Bias::Absent,
                Parallel::None,
            )?,
            down: Linear::load(
                &mut loader.scope(format!("{prefix}.ff.net.2")),
                attention::width(ffn)?,
                attention::width(hidden)?,
                Bias::Absent,
                Parallel::None,
            )?,
            gated: GatedActivation::new(ops::activation::Activation::Silu).with_stepwise(),
            eps: config.norm_eps,
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
        modulation: TensorView,
        rows: TensorView,
        factors: Factors,
        visible: Option<&Visible>,
        segments: Option<&Segments>,
        scratch: &Scratch,
        next: Option<(&Layer, TensorView)>,
        arrived: Vec<nn::Transfer>,
    ) -> Result<Vec<nn::Transfer>> {
        let width = hidden.shape.last();
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
                self.attention_norm(modulation, width, rows)?
                    .forward(f, hidden, None, normalized)?;
            } else if arrived.is_empty() {
                let rows_here = sparse.gathered_rows(views, own, 0, own);
                self.attention_norm(modulation, width, rows)?
                    .forward(f, hidden, None, rows_here)?;
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
                self.feed_forward(
                    f,
                    modulation,
                    part(hidden),
                    part(update),
                    part(out),
                    selected,
                    part(normalized),
                    part(scratch.packed),
                    part(scratch.mlp),
                )?;
                // These rows are the next layer's input, and its attention reads the whole
                // sequence, so they leave now, beside the rest of this layer.
                if ahead && let Some((layer, modulation)) = next {
                    let rows_here = sparse.gathered_rows(views, own, first, count);
                    layer.attention_norm(modulation, width, selected)?.forward(
                        f,
                        part(hidden),
                        None,
                        rows_here,
                    )?;
                    gathering.push(sparse.start_gather(f, views, own, first, count)?);
                }
            }
            return Ok(gathering);
        }
        self.attention_norm(modulation, width, rows)?
            .forward(f, hidden, None, normalized)?;
        self.attention.forward(
            f,
            normalized,
            factors,
            visible,
            segments,
            &scratch.attention,
            update,
        )?;
        self.feed_forward(
            f,
            modulation,
            hidden,
            update,
            out,
            rows,
            normalized,
            scratch.packed,
            scratch.mlp,
        )?;
        Ok(Vec::new())
    }

    /// The normalization of this layer's attention input, under its `modulation` table at the
    /// AdaLN row `rows` names for each activation row.
    fn attention_norm(
        &self,
        modulation: TensorView,
        width: u64,
        rows: TensorView,
    ) -> Result<RmsNorm> {
        let (shift, scale, _) = chunks(modulation, width, 0)?;
        let modulation = Modulation {
            shift,
            scale,
            gate: None,
            rows,
            retained: None,
        };
        Ok(RmsNorm::new(
            self.norm1,
            self.eps,
            Epilogue::Modulated(modulation),
        )?)
    }

    /// The block after attention, over any rows of the layer: `update` is the attention's output
    /// for them, `rows` their AdaLN rows in `modulation`, and the layer's output lands in `hidden`.
    #[allow(clippy::too_many_arguments)]
    fn feed_forward(
        &self,
        f: &Forward,
        modulation: TensorView,
        hidden: TensorView,
        update: TensorView,
        out: TensorView,
        rows: TensorView,
        normalized: TensorView,
        packed: TensorView,
        mlp: TensorView,
    ) -> Result<()> {
        // The MLP norm folds the attention's gated update into the residual and copies the result
        // out: the block's second residual reads it after the down projection has overwritten
        // `update`, and `retain` is what lets one launch produce both the normalized feed-forward
        // input and that residual base (`crates/ops/kernels/modulated_rms_norm`: with `update` the
        // kernel stores the gated sum, and with `retain` it copies the same value out).
        let width = hidden.shape.last();
        let (_, _, gate_attention) = chunks(modulation, width, 0)?;
        let (shift_mlp, scale_mlp, gate_mlp) = chunks(modulation, width, 3)?;
        let mlp_norm = RmsNorm::new(
            self.norm2,
            self.eps,
            Epilogue::Modulated(Modulation {
                shift: shift_mlp,
                scale: scale_mlp,
                gate: Some(gate_attention),
                rows,
                retained: Some(out),
            }),
        )?;
        let stream = ResidualStream {
            input: hidden,
            output: update,
        };
        mlp_norm.forward(f, update, Some(stream), normalized)?;

        // Feed-forward: the packed value and gate halves, the gated activation, the down
        // projection, and the second gated residual into the retained residual.
        self.gate_up.forward(f, normalized, packed)?;
        // The reference's activation is stepwise here, so the packed
        // projection is read as its two halves and gated with one rounding each - the video
        // decoder's SwiGLU is the same projection with the fused recipe, and that is why the
        // rounding is stated per site rather than per operator.
        let (value, gate) = projection_halves(packed);
        self.gated.forward(f, gate, value, mlp)?;
        // `hidden` is spent by now, so the down projection lands there, and the gated residual
        // writes over its `update` operand: the layer's output is `hidden`, in place.
        self.down.forward(f, mlp, hidden)?;
        let residual = Residual::new(Scale::Row(gate_mlp), None)?;
        Ok(residual.forward(f, out, hidden, Some(rows), hidden)?)
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
/// row**: `gather_rows` indexed by the worker's `step` view (`[1]` int32) gives `[1, out *
/// hidden]`, and a projection over that view's element type is the ordinary output projection. The
/// gathered buffers sit at fixed arena addresses with contents that change per evaluation, so the
/// graph does not change: one capture serves every step of a layout.
pub struct StudentHeads {
    video: FusedHeads,
    audio: Option<FusedHeads>,
}

/// One modality's fused heads: the weight and bias rows of every evaluation block.
struct FusedHeads {
    /// `[blocks, out * hidden]` FP32: block `b`'s fused weight.
    ///
    /// FP32 because the reference builds a student's head FP32: the fused
    /// row is a sum of the checkpoint's BF16 rows and rounding it back would round a number the
    /// reference keeps exact. The projection reads the widened FP32 rows (`Transformer::wide`), so
    /// its two operands are one type and the only rounding is the store into the velocity.
    weight: Weight,
    /// `[blocks, out]`: block `b`'s fused bias.
    bias: Weight,
    /// The velocity's width for this modality.
    out: u32,
    /// The partition's hidden width.
    hidden: u32,
    /// The blocks the schedule fused into the table.
    blocks: u32,
}

/// One evaluation's head, as the two weights a projection reads.
struct Selected {
    weight: Weight,
    bias: Weight,
}

impl FusedHeads {
    /// Pick `block`'s row of each table on the device and answer the two weights over it.
    fn select(
        &self,
        f: &Forward,
        block: u32,
        step: TensorView,
        arena: &nn::Arena,
    ) -> Result<Selected> {
        if block >= self.blocks {
            return Err(Error::checkpoint(format!(
                "H3 evaluation {block} is outside the {} fused student heads",
                self.blocks
            )));
        }
        let dtype = self.weight.dtype();
        let wide = arena.tensor(
            dtype,
            Shape::new(&[1, u64::from(self.out) * u64::from(self.hidden)]),
        )?;
        nn::layers::gather_rows(f, self.weight.view(), step, wide)?;
        let narrow = arena.tensor(dtype, Shape::new(&[1, u64::from(self.out)]))?;
        nn::layers::gather_rows(f, self.bias.view(), step, narrow)?;
        Ok(Selected {
            weight: staged(
                wide.reshape(Shape::new(&[u64::from(self.out), u64::from(self.hidden)])),
            )?,
            bias: staged(narrow.reshape(Shape::new(&[u64::from(self.out)])))?,
        })
    }
}

/// A weight over a device buffer the caller of this module owns for the launch's lifetime.
///
/// The circled views a step gathers into are arena buffers: they live as long as the forward that
/// wrote them, which is exactly as long as the projection that reads them, so a weight can address
/// one without a loader having published it.
fn staged(view: TensorView) -> Result<Weight> {
    let element = view.dtype.bytes().ok_or_else(|| {
        Error::checkpoint(format!(
            "a staged weight of {} has no whole-byte element size",
            view.dtype
        ))
    })?;
    Ok(Weight {
        backing: nn::Backing::new(view.ptr, view.shape.numel() * element),
        view,
    })
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
        Ok(Self {
            weight: heads.constant_f32("weight", &[blocks, per_head], &fused)?,
            bias: heads.constant_f32("bias", &[blocks, out], &fused_bias)?,
            out: u32::try_from(out)
                .map_err(|_| Error::checkpoint(format!("{name}'s width exceeds u32")))?,
            hidden: u32::try_from(hidden)
                .map_err(|_| Error::checkpoint(format!("{name}'s hidden width exceeds u32")))?,
            blocks: block_count,
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

    /// The rotary factor tables of the packed rows, from their axis-major positions.
    ///
    /// Every layer rotates with the same table, so the first layer's rotary computes them.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        let layer = self.layers.first().ok_or_else(|| {
            Error::Unsupported("an H3 partition with no layers has no rotary".to_owned())
        })?;
        layer.attention.factors(f, positions, cosine, sine)
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
        modulation: TensorView,
        rows: TensorView,
        block: u32,
        step: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let width = hidden.shape.last();
        let (shift, scale) = halves(modulation, width)?;
        let modulation = Modulation {
            shift,
            scale,
            gate: None,
            rows,
            retained: None,
        };
        let norm = RmsNorm::new(
            borrowed(self.norm_out.weight())?,
            self.config.norm_eps,
            Epilogue::Modulated(modulation),
        )?;
        let normalized = norm.apply(f, hidden, None)?;
        // The final modulation is the checkpoint's own type (the BF16 norm times the BF16 table),
        // and the projection that follows is FP32 - so the widening is
        // here, between them, and not inside the norm.
        let wide = self.wide(f, normalized)?;
        self.project_head(f, wide, modality, block, step, out)
    }
}

/// The shift and scale halves of a `[rows, 2 * width]` modulation table, as row-strided weights.
fn halves(table: TensorView, width: u64) -> Result<(Weight, Weight)> {
    if table.shape.rank() != 2 || table.shape.last() != 2 * width {
        return Err(Error::checkpoint(format!(
            "an output modulation of {:?} is not [rows, 2 * {width}]",
            table.shape.dims()
        )));
    }
    let rows = table.shape.dim(0);
    let stride = table.strides.get(0);
    let element = table.dtype.bytes().unwrap_or(1);
    let chunk = |index: u64| -> Result<Weight> {
        let view = TensorView::with_strides(
            nn::DevicePtr::new(table.ptr.address() + index * width * element),
            table.dtype,
            Shape::new(&[rows, width]),
            nn::Strides::new(&[stride, 1]),
        );
        borrowed(view)
    };
    Ok((chunk(0)?, chunk(1)?))
}

/// The six affine vectors one layer's modulation table cuts into.
///
/// The table is `[3 · groups, 6 · hidden]`: shift, scale and gate of the attention block, then of
/// the feed-forward block, in that order, so chunk `k` begins `k · hidden` elements into a row. The
/// chunks are weights over the caller's gathered table - nothing is copied or retained here.
fn chunks(table: TensorView, width: u64, first: u64) -> Result<(Weight, Weight, Weight)> {
    if table.shape.rank() != 2 || table.shape.last() != 6 * width {
        return Err(Error::checkpoint(format!(
            "a modulation table of {:?} is not [rows, {}]",
            table.shape.dims(),
            6 * width
        )));
    }
    let rows = table.shape.dim(0);
    let stride = table.strides.get(0);
    let chunk = |index: u64| -> Result<Weight> {
        let elements = (first + index) * width;
        let bytes = elements * table.dtype.bytes().unwrap_or(1);
        let ptr = nn::DevicePtr::new(table.ptr.address() + bytes);
        let view = TensorView::with_strides(
            ptr,
            table.dtype,
            Shape::new(&[rows, width]),
            nn::Strides::new(&[stride, 1]),
        );
        borrowed(view)
    };
    Ok((chunk(0)?, chunk(1)?, chunk(2)?))
}

/// A weight over memory the caller owns: a table gathered for one step.
///
/// Nothing is retained: the worker keeps the table alive for the length of the capture and its
/// replays, which is the same guarantee every other request table has.
fn borrowed(view: TensorView) -> Result<Weight> {
    let bytes = view.numel() * view.dtype.bytes().unwrap_or(1);
    Ok(Weight {
        backing: Backing::new(view.ptr, bytes),
        view,
    })
}

/// One H3 DiT partition: the projections, the stack and the two output heads.
pub struct Transformer {
    proj_in: Linear,
    audio_proj_in: Option<Linear>,
    layers: Vec<Layer>,
    norm_out: RmsNorm,
    /// The base form's one output head per modality. A student fuses its heads instead.
    video_out: Option<Linear>,
    audio_out: Option<Linear>,
    /// A parallel-decoding student's fused heads, when the checkpoint stacks more than one.
    heads: Option<StudentHeads>,
    config: TransformerConfig,
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
            .map_or_else(|| self.proj_in.weight().dtype, |layer| layer.norm1.dtype())
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
        // dense stack, where nothing reads it: a sparse stack holding zero would hand `VsaTiles` a
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
        let layers = (0..config.num_hidden_layers)
            .map(|index| Layer::load(kind, index, config, rows, tile_bound, partition, loader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            // **The four end projections are FP32 parameters at every preset**, because they are
            // what touches the trajectory: `video_input`, `audio_input`, `video_output` and
            // `audio_output` are pinned at FP32 whatever the attention, MLP and VAE tiers are. So
            // the widening `wide` performs meets an FP32 weight and the product is FP32 on both
            // sides, which is what makes the released BF16 stack run a FP32 sample at all.
            proj_in: Linear::load(
                &mut loader.scope("proj_in").pinned(nn::DType::F32),
                attention::width(video_in)?,
                attention::width(hidden)?,
                Bias::Present,
                Parallel::None,
            )?,
            audio_proj_in: Some(Linear::load(
                &mut loader.scope("audio_proj_in").pinned(nn::DType::F32),
                attention::width(audio_in)?,
                attention::width(hidden)?,
                Bias::Present,
                Parallel::None,
            )?),
            layers,
            norm_out: RmsNorm::load(&mut loader.scope("norm_out.norm"), hidden, config.norm_eps)?,
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
            config: *config,
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

    /// Project one modality's packed rows into the hidden width.
    pub fn project(&self, f: &Forward, rows: TensorView, out: TensorView) -> Result<()> {
        let wide = self.wide(f, rows)?;
        Ok(self.proj_in.forward(f, wide, out)?)
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
            let arena = f.arena();
            let fused = match modality {
                Modality::Video => &heads.video,
                Modality::Audio => heads.audio.as_ref().ok_or_else(|| {
                    Error::Unsupported("this H3 partition predicts no audio velocity".to_owned())
                })?,
            };
            let selected = fused.select(f, block, step, arena)?;
            return Ok(
                Linear::new(selected.weight, Some(selected.bias), WeightOrder::NxK)?
                    .forward(f, hidden, out)?,
            );
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
        adaln: &AdaLnCache,
        step: TensorView,
        hidden: TensorView,
        rows: TensorView,
        factors: Factors,
        visible: Option<&Visible>,
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
        adaln: &AdaLnCache,
        step: TensorView,
        hidden: TensorView,
        rows: TensorView,
        factors: Factors,
        visible: Option<&Visible>,
        segments: Option<&Segments>,
        total_rows: u32,
    ) -> Result<()> {
        let groups = adaln.groups();
        let width = hidden.shape.last();
        let tokens = hidden.shape.dim(0);
        let arena = f.arena();
        let entries = arena.tensor(nn::DType::I32, Shape::new(&[1, u64::from(groups.count())]))?;
        adaln.entries_of(f, step, entries)?;
        // **The gather's own arrangement: one row per entry, the whole 18-vector table across the
        // columns** (`gather_rows` derives its output from the table's columns and the index count,
        // so a buffer of the same bytes in another shape is refused by name:
        // `out is [12, 32256], [4, 96768] expected`). The block cuts three intervals per group out
        // of it, and that is a view of the same flat order - group `g`'s row holds its three
        // intervals' six vectors each, in order - so the block reads `[3 · groups, 6 · width]` over
        // these bytes and nothing is copied.
        let modulation = arena.tensor(
            hidden.dtype,
            Shape::new(&[u64::from(groups.count()), adaln.layer_width()]),
        )?;
        // Two tables, because a layer's last rows are normalized for the next layer while the
        // layer still reads its own.
        let following = arena.tensor(modulation.dtype, modulation.shape)?;
        let tables = [modulation, following];
        let per_layer = tables
            .map(|table| table.reshape(Shape::new(&[u64::from(groups.count()) * 3, 6 * width])));
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
                    per_layer[index % 2],
                    rows,
                    factors,
                    visible,
                    segments,
                    &scratch,
                    next.map(|layer| (layer, per_layer[(index + 1) % 2])),
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
    packed: TensorView,
    mlp: TensorView,
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
        let ffn = u64::from(transformer.config.intermediate_size);
        let layer = transformer.layers.first().ok_or_else(|| {
            Error::Unsupported("a stack with no layers has no scratch".to_owned())
        })?;
        Ok(Self {
            normalized: arena.tensor(dtype, Shape::new(&[tokens, width]))?,
            update: arena.tensor(dtype, Shape::new(&[tokens, width]))?,
            packed: arena.tensor(dtype, Shape::new(&[tokens, 2 * ffn]))?,
            mlp: arena.tensor(dtype, Shape::new(&[tokens, ffn]))?,
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
