//! The diffusion transformer: the packed rows in, one evaluation's velocity out.
//!
//! One `Transformer` is a checkpoint's DiT partition: a projection of each modality's patchified
//! latent into the hidden width, the stack of dilated layers, and the output projections back to
//! each modality's velocity (`uniserve/models/minimax_h3/transformer.py`). A layer is conditioned
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
//! every step of the schedule. The tables themselves are the executor's: the model reads the cache
//! through the borrow it is handed, never through state of its own.

use crate::attention::{self, Factors, Visible};
use crate::attention_sparse::Segments;
use crate::config::{AttentionKind, TransformerConfig};
use crate::modulation::{AdaLnCache, Groups};
use crate::partition::Partition;
use crate::schedule::{Modality, Schedule};
use crate::{Error, Result};

use nn::layers::{GatedActivation, GatedResidual, Linear, ModulatedRmsNorm, RmsNorm, Silu};
use nn::tensor::Backing;
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

/// The two halves of a packed `[value, gate]` projection, as strided views without a copy.
///
/// The checkpoint fuses the feed-forward's value half first and its gate half second, and the
/// reference projects it once and then reads `branches["up"]` and `branches["gate"]`
/// (`transformer.py:141-153`). The stepwise activation takes the halves as separate operands, so
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
/// (`uniserve/models/minimax_h3/denoiser.py:231`) and applies that one spectrum to each of the
/// three coordinates (`attention.py:94-105`). Three is the model's own count: `position_ids` is
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
    /// The stepwise gated activation: the reference evaluates `silu(gate)` and the product
    /// with one rounding each (`transformer.py:84-88`, `Rounding.STEPWISE`).
    gated: GatedActivation,
    eps: f32,
}

impl Layer {
    /// Every checkpoint tensor one stack layer reads, as complete names.
    ///
    /// The layer's own load requests, in its order: the two norms, the feed-forward's two
    /// projections, then the attention's.
    pub fn keys(index: u32, kind: AttentionKind) -> Vec<String> {
        let prefix = format!("transformer_blocks.{index}");
        let mut keys: Vec<String> = ["norm1.weight", "norm2.weight"]
            .iter()
            .map(|suffix| format!("{prefix}.{suffix}"))
            .collect();
        for name in ["ff.net.0.proj.weight", "ff.net.2.weight"] {
            keys.push(format!("{prefix}.{name}"));
        }
        keys.extend(attention::Layer::keys(index, kind));
        keys
    }

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
            norm1: gain(loader, &format!("{prefix}.norm1.weight"), hidden)?,
            norm2: gain(loader, &format!("{prefix}.norm2.weight"), hidden)?,
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
            // `projection` appends the `.weight` itself: a name that already ends in it asks the
            // checkpoint for `…weight.weight`, which is the name the index does not hold.
            gate_up: projection(loader, &format!("{prefix}.ff.net.0.proj"), 2 * ffn, hidden)?,
            down: projection(loader, &format!("{prefix}.ff.net.2"), hidden, ffn)?,
            gated: GatedActivation::new(ops::activation::Activation::Silu).with_stepwise(),
            eps: config.norm_eps,
        })
    }

    /// One layer from `hidden` into `out`, modulated by the step's gathered rows.
    ///
    /// `modulation` is this layer's `[3 · groups, 6 · hidden]` table and `rows` the `[tokens]`
    /// AdaLN row each token reads. `out` receives the layer's output, so the stack alternates two
    /// buffers and never copies the residual.
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
    ) -> Result<()> {
        let width = hidden.shape.last();
        let (shift_attention, scale_attention, gate_attention) = chunks(modulation, width, 0)?;
        let (shift_mlp, scale_mlp, gate_mlp) = chunks(modulation, width, 3)?;
        let normalized = scratch.normalized;
        let update = scratch.update;

        // Attention: normalize, attend, and fold the gated update into `out` while normalizing the
        // feed-forward's input.
        let attention_norm =
            ModulatedRmsNorm::new(self.norm1, shift_attention, scale_attention, None, self.eps)?;
        attention_norm.forward(f, hidden, None, rows, normalized, None, None)?;
        self.attention.forward(
            f,
            normalized,
            factors,
            visible,
            segments,
            &scratch.attention,
            update,
        )?;
        // The MLP norm folds the attention's gated update into the residual and copies the result
        // out: the block's second residual reads it after the down projection has overwritten
        // `update`, and `retain` is what lets one launch produce both the normalized feed-forward
        // input and that residual base (`crates/ops/kernels/modulated_rms_norm`: with `update` the
        // kernel stores the gated sum, and with `retain` it copies the same value out).
        let mlp_norm = ModulatedRmsNorm::new(
            self.norm2,
            shift_mlp,
            scale_mlp,
            Some(gate_attention),
            self.eps,
        )?
        .with_retain();
        mlp_norm.forward(f, hidden, Some(update), rows, normalized, None, Some(out))?;

        // Feed-forward: the packed value and gate halves, the gated activation, the down
        // projection, and the second gated residual into the retained residual.
        let packed = scratch.packed;
        let mlp = scratch.mlp;
        self.gate_up.forward(f, normalized, packed)?;
        // The reference's activation is stepwise here (`transformer.py:84-88`), so the packed
        // projection is read as its two halves and gated with one rounding each - the video
        // decoder's SwiGLU is the same projection with the fused recipe, and that is why the
        // rounding is stated per site rather than per operator.
        let (value, gate) = projection_halves(packed);
        self.gated.forward(f, gate, value, mlp)?;
        self.down.forward(f, mlp, update)?;
        Ok(GatedResidual::new(gate_mlp)?.forward(f, out, update, rows)?)
    }
}

/// One evaluation's selected layer outputs, as the caller's own diagnostic views.
///
/// The design's numeric diagnosis (`specs/worker.md` 10) compares one denoiser evaluation layer by
/// layer. The model holds no host-to-device path and reads no environment, so the comparison's
/// buffers are the caller's: this struct borrows them, and a forward with a `Trace` copies each
/// named layer's packed output into its view. It is a diagnosis and never a gate: a forward with no
/// `Trace` is byte-identical to the served one, because the copy is the only difference.
pub struct Trace<'a> {
    /// The layer indices to write, each below the stack's layer count.
    pub layers: &'a [u32],
    /// One `[tokens, hidden]` view per entry of `layers`, in the same order.
    pub views: &'a [TensorView],
    /// `[tokens]` int32 identities (`0..tokens`): the rows a copy of the output reads.
    pub index: TensorView,
}

impl Trace<'_> {
    /// Check the views against one forward's shape, so a mismatch is refused before the stack runs.
    fn check(&self, layers: usize, tokens: u64, width: u64, dtype: nn::DType) -> Result<()> {
        if self.layers.len() != self.views.len() {
            return Err(Error::config(format!(
                "the trace names {} layers and {} views",
                self.layers.len(),
                self.views.len()
            )));
        }
        for (layer, view) in self.layers.iter().zip(self.views) {
            if *layer as usize >= layers {
                return Err(Error::config(format!(
                    "the trace asks for layer {layer} of {layers}"
                )));
            }
            if view.shape.dims() != [tokens, width] || view.dtype != dtype {
                return Err(Error::Config(format!(
                    "the trace's layer {layer} view is {:?} {}, [{tokens}, {width}] {dtype:?} \
                     expected",
                    view.shape.dims(),
                    view.dtype
                )));
            }
        }
        if self.index.numel() != tokens {
            return Err(Error::Config(format!(
                "the trace's identity index covers {} rows and the pack has {tokens}",
                self.index.numel()
            )));
        }
        Ok(())
    }
}

/// A parallel-decoding student's output heads, fused per evaluation block at load.
///
/// A DMD/PDD student stacks one head per fine interval (`pdd_steps`, e.g. 32) and predicts one
/// block's velocity from the weighted sum of its intervals' heads
/// (`uniserve/diffusion/schedule.py:297-326`: weights and accumulation in FP32, rounded to the
/// parameter's dtype). The fusion is arithmetic over the checkpoint's own numbers, so it is done
/// once, on the host, at the loading boundary; what a forward does is pick one row.
///
/// The table is laid out `[blocks, out * hidden]` so that one evaluation's head is **one table
/// row**: `gather_rows` indexed by the executor's `step` view (`[1]` int32) gives `[1, out *
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
    /// FP32 because the reference builds a student's head FP32 (`transformer.py:307-311`): the fused
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
    /// exactly `fuse_heads` (`uniserve/diffusion/schedule.py:297-326`). One derivation per block
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
        let values = host_f32(dtype, &weight.bytes, &format!("{name}.weight"))?;
        let biases = host_f32(dtype, &bias.bytes, &format!("{name}.bias"))?;
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
        let modality_name = modality.as_str();
        Ok(Self {
            weight: loader.constant(
                &format!("denoiser.{modality_name}.student_heads.weight"),
                nn::DType::F32,
                Shape::new(&[blocks, per_head]),
                &host_bytes(nn::DType::F32, &fused)?,
                "minimax-h3",
            )?,
            bias: loader.constant(
                &format!("denoiser.{modality_name}.student_heads.bias"),
                nn::DType::F32,
                Shape::new(&[blocks, out]),
                &host_bytes(nn::DType::F32, &fused_bias)?,
                "minimax-h3",
            )?,
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
/// the sum stay FP32, and so does the table the caller stores them in (`transformer.py:307-311`).
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

/// A checkpoint tensor's values as FP32, for a derivation that must be exact on the host.
///
/// Shared with the decoders' load-boundary folds, which read the checkpoint's own bytes for the same
/// reason: a host product is only exact if nothing rounds on the way in.
pub(crate) fn host_f32(dtype: nn::DType, bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    match dtype {
        nn::DType::BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
            .collect()),
        nn::DType::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()),
        other => Err(Error::Unsupported(format!(
            "{name} is {other}; a parallel-decoding student's heads are fused on the host in FP32, \
             which this build does for BF16 and F32 only"
        ))),
    }
}

/// FP32 values as the little-endian bytes of `dtype`, rounded the way the checkpoint's own type
/// rounds (BF16: round to nearest even).
pub(crate) fn host_bytes(dtype: nn::DType, values: &[f32]) -> Result<Vec<u8>> {
    match dtype {
        nn::DType::BF16 => {
            let mut bytes = Vec::with_capacity(values.len() * 2);
            for value in values {
                let bits = value.to_bits();
                // Round to nearest even on the eight dropped bits before truncating.
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                bytes.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
            }
            Ok(bytes)
        }
        nn::DType::F32 => {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(bytes)
        }
        other => Err(Error::Unsupported(format!(
            "a fused head cannot be stored as {other}"
        ))),
    }
}

impl Transformer {
    /// The programs this DiT stack launches, from its configuration and the deployment's partition.
    ///
    /// Stated without weights, because a deployment compiles its store before it opens the
    /// checkpoint. `kind` decides the attention's programs; `rows` is the largest packed row count the
    /// deployment's layouts hold, which is the dense attention's key bound; and `tiles` is the
    /// deployment's tile bound — the sparse attention's reduction length — which the layouts state
    /// and a sparse deployment carries. **No collective is a program**: the Ulysses exchange issues a
    /// communication, so a partition's degree changes the shapes the attention reads and adds no row.
    #[allow(clippy::too_many_arguments)]
    pub fn programs(
        config: &TransformerConfig,
        kind: AttentionKind,
        dtype: nn::DType,
        boundary: nn::DType,
        rows: u32,
        groups: Groups,
        partition: &Partition,
        tiles: u32,
    ) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use nn::layers::gather_rows_descriptor;
        use nn::layers::{Attention, Rotary};
        use ops::activation::ActAndMul as ActAndMulOp;
        use ops::activation::Silu as SiluOp;
        use ops::attention::SparseAttention;
        use ops::attention::dense::DenseAttention;
        use ops::gemm::Gemm;
        use ops::modulation::GatedResidual as GatedResidualOp;
        use ops::modulation::ModulatedRmsNorm as ModulatedOp;
        use ops::norm::RmsNorm as RmsNormOp;
        use ops::rope::Rotation;
        use ops::rope::{QkNormRope, RotaryFactors};

        let hidden = config.hidden_size;
        let head = config.head_dim;
        let heads = partition.denoiser_heads(config.num_attention_heads);
        let rotated: u32 = rotated_axes(config).iter().sum();
        let interaction = 6 * u64::from(hidden);
        let mut programs = Programs::default();

        // A sparse deployment's attention adds the video-sparse selection's six programs and the fine
        // attention over the tiles it selects. Its projections are the shared `gemm` row and its
        // rotation the shared rope rows, so only the selection and the tile attention are new.
        if let AttentionKind::Sparse { tile, .. } = kind {
            let tiles_layer = nn::layers::VsaTiles::new(tile, heads, head, config.qk_norm_eps)?
                .with_rotation(rotated)?
                .with_tile_bound(tiles)?;
            // `SegmentAttention`'s fourth argument is the key **row** bound, not the tile bound:
            // `keys` is "the largest key extent the compiled program serves, from the same
            // packing", and the layer refuses a value that is not whole `tile`-row tiles. `tiles`
            // is the reduction length in tiles, and passing it here compiled a program for
            // `tiles` rows - 43x short of the deployment's packed bound - while passing the
            // constructor's divisibility check whenever the tile bound happened to divide by the
            // tile. The other call site (`attention_sparse.rs`) already passes `rows`.
            let fine = nn::layers::SegmentAttention::new(
                u64::from(tile),
                u64::from(heads),
                u64::from(head),
                u64::from(rows),
                dtype,
            )?;
            // The prepare launch compiles its strides in, and the buffers are the model's own:
            // `[rows, heads, width]` contiguous, and a factor row of **rotated pairs, not rotated
            // channels**. The arena allocates the two tables `[tokens, pairs]` with
            // `pairs = rope_frequency_dim * 3` (`denoiser.rs`, the two `arena.tensor` calls), which
            // is half the `rotated` this operator counts: 48 against 96. Stating the channel count
            // here named a key no launch asks for, and the launch would have refused it - the
            // descriptor's stride is the table's own (`nn/src/layers/segments.rs`).
            let pairs = u64::from(rotated) / 2;
            programs.launch::<ops::segments::VsaPrepareTiles>(
                &tiles_layer.contiguous_prepare_descriptor(dtype, pairs),
            );
            // `VsaZeroPadding` is launched by the fine attention (`SegmentAttention::forward`,
            // `nn/src/layers/segment.rs:174`) to zero the rows past each tile's valid size in the
            // rows it just wrote, so its descriptor is that layer's: the **output's** element type
            // and the tile geometry the attention's own packing has, not the stack's type. The
            // `VsaTiles` chain zeroes nothing - `prepare` writes the zeros it pools - so an
            // accessor on it would state a row no launch reads, at a type no launch passes.
            programs.launch::<ops::segments::VsaZeroPadding>(&fine.zero_descriptor());
            programs.launch::<ops::segments::VsaScores>(&tiles_layer.scores_descriptor());
            programs.launch::<ops::segments::VsaCompressPooled>(
                &tiles_layer.compress_descriptor(dtype),
            );
            programs.launch::<ops::segments::VsaAddGatedTiles>(
                &tiles_layer.add_gated_tiles_descriptor(dtype),
            );
            programs.launch::<ops::segments::VsaSelectTiles>(
                &tiles_layer.select_descriptor(u64::from(tiles))?,
            );
            programs.launch::<SparseAttention>(&fine.descriptor_of(dtype));
        }

        // Every projection of the stack: the video and audio entrances, the four of each block's
        // attention, its feed-forward pair, and the head per modality. None carries a bias.
        programs.launch::<Gemm>(&Linear::descriptor_of(
            hidden,
            dtype,
            dtype,
            WeightOrder::NxK,
            false,
            dtype,
        ));
        // **And the boundary products, which cross the two types.** The boundary tensors are the
        // F32 side and the block weights are the stack's, so the product reads an activation of the
        // boundary type over a weight of the stack's: the forward's own record says
        // `input_f32_weight_bf16`, and a statement keyed from the stack alone names that pair's
        // mirror - a cell the runtime never asks for, while the one it does ask for stops the run
        // (`the refused gemm is input f32, weight bf16`). A single-typed component has
        // `boundary == dtype` and states nothing here.
        if boundary != dtype {
            // Four products, and the recorder states all four: a boundary **weight** over an
            // activation of either type, and a boundary activation over a stack weight. The
            // checkpoint's four projections (`proj_in`, `proj_out`, `audio_proj_in`,
            // `audio_proj_out`) and the timestep pair are F32 in weight *and* bias, so the product
            // that reads them takes the stack's activation and writes the stack's type; the output
            // projections carry no bias, which is the `epilogue_none` row.
            programs.launch::<Gemm>(&Linear::descriptor_of(
                hidden,
                dtype,
                boundary,
                WeightOrder::NxK,
                true,
                dtype,
            ));
            programs.launch::<Gemm>(&Linear::descriptor_of(
                hidden,
                dtype,
                boundary,
                WeightOrder::NxK,
                false,
                dtype,
            ));
            programs.launch::<Gemm>(&Linear::descriptor_of(
                hidden,
                boundary,
                boundary,
                WeightOrder::NxK,
                true,
                dtype,
            ));
            programs.launch::<Gemm>(&Linear::descriptor_of(
                hidden,
                boundary,
                dtype,
                WeightOrder::NxK,
                true,
                dtype,
            ));
        }
        programs.launch::<RmsNormOp>(&RmsNorm::descriptor_of(hidden, dtype));

        // The timestep embedding's activation: `Linear → SiLU → Linear` over one row per group at the
        // modulation's hidden width. The projections are the shared `gemm` row, whose key is the element
        // types and the epilogue, not the width.
        // **At the boundary's type, not the stack's**: the activation reads the timestep embedder's
        // output, and both of that pair's tensors are stored at the boundary type. A statement that
        // named the stack's type was a row the forward never launches and, in the other direction,
        // the row the forward does launch was missing.
        programs.launch::<SiluOp>(&Silu::descriptor_of(boundary, config.time_hidden_dim));
        programs.launch::<SiluOp>(&Silu::descriptor_of(boundary, config.time_dim));

        // The modulation: three row programs, one per gathered table width — the step's entry per
        // group, a layer's six vectors per tag, and the final normalization's two. The first width is
        // the group count, so a conditioned deployment's cache is a second program.
        //
        // **The first gather is the only i32 one.** It reads each row's group tag, which the packing
        // builds as i32; the other two read the AdaLN cache, and `forward` gathers them from the
        // table's own view - `blocks.N.adaln_proj.linear.weight`, whose type is the checkpoint's under
        // `dtype = "auto"` and is the boundary's here. So those two are stated at the **boundary's**
        // type: `dtype = "auto"` narrows the activations a projection consumes, not the weight a gather
        // reads. Stating i32 for all three named two keys no launch binds; stating the stack's type for
        // these two named two more.
        for (table, columns) in [
            (nn::DType::I32, u64::from(groups.count())),
            (boundary, 18 * u64::from(hidden)),
            (boundary, 2 * u64::from(hidden)),
        ] {
            programs.launch::<ops::logits::LogitsGather>(&gather_rows_descriptor(table, columns)?);
        }

        // The block's two modulated normalizations: the attention's (no update, **no gate**) and the
        // feed-forward's (both, retaining the normalized row for the projection).
        //
        // The gate stride is `0` for the first because `forward` builds that norm with `None` for its
        // gate - there is no gate projection on the attention's normalization, which is what the
        // reference's `modulation` omits. Stating the cache row's stride there named a key no launch
        // binds while the key the launch binds had no row: one wrong number on each side of the same
        // defect, and a store built from the statement would have missed the row a run needs.
        for (update, retain, gate) in [(false, false, 0), (true, true, interaction)] {
            programs.launch::<ModulatedOp>(&ModulatedRmsNorm::descriptor_of(
                dtype,
                dtype,
                nn::DType::I32,
                hidden,
                config.norm_eps,
                gate,
                interaction,
                interaction,
                update,
                retain,
            ));
        }
        programs.launch::<GatedResidualOp>(&GatedResidual::descriptor_of(
            dtype,
            nn::DType::I32,
            u64::from(hidden),
            interaction,
        )?);

        // The rotation and the attention, both over this rank's heads.
        //
        // **The factors are built contiguous and the rotation applies them interleaved**, and the two
        // are separate keys for that reason: one spectrum over the three axes is written as three
        // blocks (`188b8b1d5`), while the per-head rotation pairs the halves of a block. Stating the
        // factors as `interleaved` named a key no launcher binds - the deployment list's contiguous row
        // was the launch's, and this statement was the one that disagreed.
        //
        // The positions are the **integer** table: `position_ids` is declared `i32` beside the tables
        // it is indexed with (`denoiser.rs`), a coordinate being an index rather than a measured value,
        // so a statement typing them F32 named a key no launch binds - and the failure that produced
        // says so in its own words (`generation 1 has no kernel for rotary_factors descriptor
        // dtype_f32_positions_f32_columns_48`). The row the file holds is `positions_i32`.
        let sections = nn::layers::Sections::new(&rotated_axes(config)).map_err(Error::from)?;
        programs.launch::<RotaryFactors>(&Rotary::factors_descriptor_of(
            rotated,
            1.0,
            Some(sections),
            nn::DType::I32,
        ));
        programs.launch::<QkNormRope>(&Rotary::apply_descriptor_of(
            head,
            heads,
            heads,
            Rotation::Half,
            config.qk_norm_eps,
            rotated,
            dtype,
            dtype,
            // The checkpoint stores `norm_q` and `norm_k` per head.
            true,
        ));
        // A sparse stack attends through the selection over tiles, stated above; a dense one through
        // the single attention over every packed row. The two are different backends, not a subset.
        if matches!(kind, AttentionKind::Dense) {
            let attention = Attention::dense(
                head,
                heads,
                heads,
                rows,
                false,
                None,
                1.0 / (head as f32).sqrt(),
            )
            .map_err(Error::from)?;
            programs.launch::<DenseAttention>(
                &attention.dense_descriptor(dtype, dtype, dtype, rows, true),
            );
        }

        // The feed-forward's stepwise gated activation: one rounding each, which is a different
        // program from the fused recipe the video decoder's SwiGLU uses.
        //
        // **Both row strides are twice the width, because the two halves come from one packed
        // projection.** `forward` reads them through `projection_halves`, which keeps the packed
        // tensor's strides: a `[rows, 2 * intermediate]` buffer gives each half a row stride of
        // `2 * intermediate`, and that is the number the kernel's descriptor carries - so this
        // statement, the launch, and the deployment list's row are one value, not three. Stating the
        // width here instead would have been a second, wrong answer to a question the launch already
        // answers, and the walk that records the launch is what a store is built from.
        let packed_stride = 2 * u64::from(config.intermediate_size);
        // **Two rows, because `wide` is chosen by the launch's own element count and both classes
        // are reachable.** The bound crosses `u32::MAX` at this stride (304,976 rows x 14,336), so
        // the widest request a store serves launches the wide form - and every request below
        // 299,593 packed rows launches the narrow one. A store that states only the bound is short
        // the cell every ordinary request asks for, which is what the launch record reported.
        for elements in [u64::from(rows) * u64::from(config.intermediate_size), 1] {
            programs.launch::<ActAndMulOp>(&GatedActivation::descriptor_of(
                dtype,
                u64::from(config.intermediate_size),
                nn::layers::Activation::Silu,
                packed_stride,
                packed_stride,
                elements,
                true,
            )?);
        }

        Ok(programs.finish())
    }

    /// Every checkpoint tensor one DiT partition of the stack reads, as complete names.
    ///
    /// The four end projections and the output norm are the block's own; the per-layer tensors and
    /// the attention's come from [`Layer::keys`]. A student's stacked heads are the same `proj_out`
    /// and `audio_proj_out` tensors, fused at load, so the list is one list for both forms.
    pub fn keys(config: &TransformerConfig, kind: AttentionKind) -> Vec<String> {
        let mut keys = Vec::new();
        for name in ["proj_in", "audio_proj_in", "proj_out", "audio_proj_out"] {
            keys.push(format!("{name}.weight"));
            keys.push(format!("{name}.bias"));
        }
        keys.push("norm_out.norm.weight".to_owned());
        for index in 0..config.num_hidden_layers {
            keys.extend(Layer::keys(index, kind));
        }
        keys
    }

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
        let norm = ModulatedRmsNorm::new(
            borrowed(self.norm_out.weight())?,
            shift,
            scale,
            None,
            self.config.norm_eps,
        )?;
        let arena = f.arena();
        let dtype = hidden.dtype;
        let normalized = arena.tensor(dtype, hidden.shape)?;
        norm.forward(f, hidden, None, rows, normalized, None, None)?;
        // The final modulation is the checkpoint's own type (`modulation.py:62-75` multiplies the
        // BF16 norm by the BF16 table), and the projection that follows is FP32 - so the widening is
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
/// Nothing is retained: the executor keeps the table alive for the length of the capture and its
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
    /// at every preset (`precision.py:106-117`), so `proj_in` reports the pin rather than the
    /// stack: at `auto` a BF16 deployment would claim FP32, a harness would allocate FP32
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
        // rather than refuse. A harness that passes zero for a sparse deployment is refused here
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
            // **The four end projections are FP32 parameters at every preset**, because they are what
            // touches the trajectory: `precision.py:106-117` pins `video_input`, `audio_input`,
            // `video_output` and `audio_output` at FP32 whatever the attention, MLP and VAE tiers
            // are. So the widening `wide` performs meets an FP32 weight and the product is FP32 on
            // both sides, which is what makes the released BF16 stack run a FP32 sample at all.
            proj_in: projection_with_bias(
                loader,
                "proj_in",
                hidden,
                video_in,
                Some(nn::DType::F32),
            )?,
            audio_proj_in: Some(projection_with_bias(
                loader,
                "audio_proj_in",
                hidden,
                audio_in,
                Some(nn::DType::F32),
            )?),
            layers,
            norm_out: RmsNorm::new(
                gain(loader, "norm_out.norm.weight", hidden)?,
                config.norm_eps,
            )?,
            // A student's checkpoint holds the stacked interval heads, so the plain projection is
            // read only when the partition has one head.
            video_out: match heads {
                Some(_) => None,
                None => Some(projection_with_bias(
                    loader,
                    "proj_out",
                    video_in,
                    hidden,
                    Some(nn::DType::F32),
                )?),
            },
            audio_out: match heads {
                Some(_) => None,
                None => Some(projection_with_bias(
                    loader,
                    "audio_proj_out",
                    audio_in,
                    hidden,
                    Some(nn::DType::F32),
                )?),
            },
            heads,
            config: *config,
        })
    }

    /// The configuration this partition was read with.
    pub const fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// The FP32 view an FP32 projection reads when `source` is not FP32 already.
    ///
    /// The reference builds all four of the transformer's end projections as FP32 parameters
    /// (`transformer.py:366-372`, `:404-421`) and its `Linear` casts the activation up before the
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
        let out = f.arena().tensor(nn::DType::F32, source.shape)?;
        // `target <- 1 * source + 0 * target`: a widening, and the zero beta means the target is
        // never read, so an uninitialised buffer is the right operand.
        nn::layers::ScaledAdd.forward(f, out, source, 1.0, 0.0)?;
        Ok(out)
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
    /// `adaln` is the cache the executor bound for this request, `step` the `[1]` i32 step index
    /// the executor wrote before the replay, and `rows` the `[tokens]` AdaLN row of each token.
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
        trace: Option<&Trace<'_>>,
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
        let per_layer = modulation.reshape(Shape::new(&[u64::from(groups.count()) * 3, 6 * width]));
        if let Some(trace) = trace {
            trace.check(self.layers.len(), tokens, width, hidden.dtype)?;
        }
        let mut alternate = arena.tensor(hidden.dtype, Shape::new(&[tokens, width]))?;
        let scratch = Scratch::reserve(self, f, hidden, total_rows)?;
        let mut current = hidden;
        for (index, layer) in self.layers.iter().enumerate() {
            adaln.layer(f, index, entries, modulation)?;
            layer.forward(
                f, current, alternate, per_layer, rows, factors, visible, segments, &scratch,
            )?;
            std::mem::swap(&mut current, &mut alternate);
            // A diagnosis reads the layer's packed output as it stands: the same rows the next
            // layer reads, copied into the caller's view by an identity gather.
            if let Some(trace) = trace
                && let Some(position) = trace.layers.iter().position(|layer| *layer == index as u32)
            {
                nn::layers::gather_rows(f, current, trace.index, trace.views[position])?;
            }
        }
        // An odd number of layers leaves the stack's output one buffer away from the caller's.
        if current.ptr != hidden.ptr {
            return Err(Error::Unsupported(
                "the H3 stack ends in the alternate buffer; the caller must read the returned view"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Every temporary view the stack reads, reserved once for the whole forward.
///
/// The executor's arena is a bump allocator reset once per forward, so scratch that does not depend
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

/// A `[out, in]` projection of the checkpoint, without bias.
fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Linear> {
    let weight = loader.load(&WeightRequest::region(
        format!("{name}.weight"),
        0,
        Shape::new(&[rows, columns]),
    ))?;
    Ok(Linear::new(weight, None, WeightOrder::NxK)?)
}

/// A `[out, in]` projection of the checkpoint with its bias.
fn projection_with_bias(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
    dtype: Option<nn::DType>,
) -> Result<Linear> {
    let mut request =
        WeightRequest::region(format!("{name}.weight"), 0, Shape::new(&[rows, columns]));
    if let Some(dtype) = dtype {
        request = request.pinned(dtype);
    }
    let weight = loader.load(&request)?;
    let mut request = WeightRequest::region(format!("{name}.bias"), 0, Shape::new(&[rows]));
    if let Some(dtype) = dtype {
        request = request.pinned(dtype);
    }
    let bias = loader.load(&request)?;
    Ok(Linear::new(weight, Some(bias), WeightOrder::NxK)?)
}

/// A normalization's or projection's learned gain.
fn gain(loader: &mut WeightLoader<'_>, name: &str, columns: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(name, 0, Shape::new(&[columns])))?)
}

#[cfg(test)]
mod tests {

    /// The stack states its programs from its configuration and its partition: one product, one
    /// output normalization, the three modulation gathers, the block's two modulated shapes, its
    /// rotation and attention, and its stepwise gated activation. The Ulysses degree changes the head
    /// count a key carries and adds no row, because a collective is not a program.
    #[test]
    fn the_stack_states_its_programs_from_its_config() {
        let config = crate::config::tests_support::released_config();
        let (_, denoiser) = config
            .denoisers
            .iter()
            .next()
            .expect("a released DiT partition");
        let transformer = denoiser.transformer;
        let programs = Transformer::programs(
            &transformer,
            AttentionKind::Dense,
            nn::DType::BF16,
            nn::DType::F32,
            32_000,
            Groups::of(true),
            &Partition::single(),
            0,
        )
        .expect("the stack's programs");
        let count = |operator: &str| programs.iter().filter(|(name, _)| name == operator).count();

        // **Five products, not one**: the stack's own (`dtype` over `dtype`, no bias) and the four a
        // deployment whose boundary type differs from the stack's crosses - a boundary weight with and
        // without a bias, and a boundary activation over a stack weight and over a boundary one. The
        // keys are distinct because the pair of element types and the epilogue are what select a
        // module; the widths are launch values. A statement of one product named the direct product
        // only and left every boundary cell unstated, which is a run that stops at its first
        // projection.
        assert_eq!(count("gemm"), 5);
        assert_eq!(count("rms_norm"), 1);
        assert_eq!(count("logits_gather"), 3);
        assert_eq!(count("modulated_rms_norm"), 2);
        assert_eq!(count("gated_residual"), 1);
        assert_eq!(count("qk_norm_rope"), 1);
        assert_eq!(count("dense_attention"), 1);
        assert_eq!(count("rotary_factors"), 1);
        assert_eq!(count("act_and_mul"), 1);
        assert!(
            programs
                .iter()
                .any(|(name, key)| name == "act_and_mul" && key.contains("stepwise_true")),
            "the block's activation is the stepwise recipe: {programs:?}"
        );

        // **The stated keys are the keys a launch binds**, and each of the four below is a defect this
        // test would have caught: a `programs` statement that drifted from the `forward` beside it, so
        // that a store built from the statement either held a program nothing launches or missed the
        // one that runs.
        //
        // The activation reads the two halves of one packed `[rows, 2 * 14336]` projection, so both row
        // strides are twice the width - `projection_halves` keeps the packed tensor's strides, and the
        // launch passes `gate.strides[0]`.
        assert!(
            programs.iter().any(|(name, key)| name == "act_and_mul"
                && key.contains("gate_stride_28672_value_stride_28672")),
            "the block's activation reads a packed projection at twice the width: {programs:?}"
        );
        // The attention's normalization has **no gate** - `forward` builds it with `None` - so its
        // gate stride is `0` while the feed-forward's reads the modulation cache. Stating the cache's
        // stride for both named a key no launch binds.
        assert!(
            programs
                .iter()
                .any(|(name, key)| name == "modulated_rms_norm" && key.contains("gate_stride_0_")),
            "the attention's normalization carries no gate: {programs:?}"
        );
        assert!(
            programs
                .iter()
                .any(|(name, key)| name == "modulated_rms_norm"
                    && key.contains("gate_stride_32256_")),
            "the feed-forward's normalization is gated: {programs:?}"
        );
        // Of the three row gathers, only the per-token **group tag** is i32; the two modulation tables
        // are gathered at the stack's own type, so a statement that typed them i32 named two keys no
        // launch binds - and the element type is the one field a row is written with.
        let gathers = programs
            .iter()
            .filter(|(name, _)| name == "logits_gather")
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            gathers
                .iter()
                .filter(|key| key.contains("dtype_i32"))
                .count(),
            1,
            "only the group tag is i32: {gathers:?}"
        );
        // **The boundary's type, not the stack's.** The gathered table is the modulation projection's
        // weight, and `dtype = "auto"` narrows the activations that projection consumes, not the table
        // the gather reads - the launch derives the key from the table's own view. A statement in the
        // stack's type named two keys no launch binds.
        assert!(
            gathers
                .iter()
                .any(|key| key.contains("dtype_f32_columns_96768"))
                && gathers
                    .iter()
                    .any(|key| key.contains("dtype_f32_columns_10752")),
            "the modulation tables are gathered at the boundary's type: {gathers:?}"
        );

        // The rotation's factor table: **contiguous** sections over the three axes (one spectrum
        // written in blocks) and **integer** positions, because `position_ids` is an `i32` table - a
        // coordinate is an index, not a measured value. Stating either the other way named a key no
        // launch binds, and the store had the launch's row both times.
        let factors = programs
            .iter()
            .find(|(name, _)| name == "rotary_factors")
            .map(|(_, key)| key.clone())
            .expect("the stack states one factor table");
        assert!(
            factors.contains("positions_i32"),
            "the factor table reads the integer position table: {factors}"
        );
        assert!(
            factors.contains("sections_32-32-32") && !factors.contains("interleaved_true"),
            "the factor table is contiguous over three equal axes: {factors}"
        );

        // A conditioned deployment's modulation cache has four groups where a text-only one has two,
        // so the entries gather is a different program rather than the same one reused.
        let generated = Transformer::programs(
            &transformer,
            AttentionKind::Dense,
            nn::DType::BF16,
            nn::DType::F32,
            32_000,
            Groups::of(false),
            &Partition::single(),
            0,
        )
        .expect("a text-only deployment's programs");
        let entries = |set: &[(String, String)]| {
            set.iter()
                .filter(|(name, _)| name == "logits_gather")
                .map(|(_, key)| key.clone())
                .collect::<Vec<_>>()
        };
        assert_ne!(entries(&programs), entries(&generated));

        // A sparse partition states the selection's six programs and the fine attention over the
        // tiles it selects, in place of the dense attention's one: the dense rows are the ones the
        // sparse path does not launch, and the difference is the backend rather than a subset.
        let sparse = Transformer::programs(
            &transformer,
            AttentionKind::Sparse {
                tile: 64,
                sparsity: 0.5,
                reference_keep: Some(0.1),
            },
            nn::DType::BF16,
            nn::DType::F32,
            32_000,
            Groups::of(true),
            &Partition::single(),
            512,
        )
        .expect("a sparse deployment's programs");
        let sparse_has = |operator: &str| sparse.iter().any(|(name, _)| name == operator);
        for operator in [
            "vsa_prepare_tiles",
            "vsa_zero_padding",
            "vsa_scores",
            "vsa_select_tiles",
            "vsa_compress_pooled",
            "vsa_add_gated_tiles",
            "sparse_attention",
        ] {
            assert!(sparse_has(operator), "{operator} is not stated: {sparse:?}");
        }
        // The dense attention's rows are the dense deployment's; the sparse stack projects and rotates
        // through the same rows, so those stay.
        assert!(
            !sparse_has("dense_attention"),
            "a sparse stack launches no dense attention"
        );
        assert!(sparse_has("gemm") && sparse_has("modulated_rms_norm"));
        // A sparse deployment's tile bound is in the compression's descriptor, and the launch refuses a
        // request outside it rather than clamping.
        let compress = sparse
            .iter()
            .find(|(name, _)| name == "vsa_compress_pooled")
            .map(|(_, key)| key.clone())
            .expect("the compression's row");
        assert!(
            compress.contains("max_tiles_512"),
            "the bound the deployment states is the one the row names: {compress}"
        );
    }
    use super::*;
    use nn::DType;

    fn view(tokens: u64, width: u64, dtype: DType) -> TensorView {
        TensorView::new(nn::DevicePtr::new(0), dtype, Shape::new(&[tokens, width]))
    }

    fn index(tokens: u64) -> TensorView {
        TensorView::new(nn::DevicePtr::new(0), DType::I32, Shape::new(&[tokens]))
    }

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

    #[test]
    fn a_bf16_head_round_trips_through_the_host_path() {
        // Every BF16 value is an FP32 value, so a tensor that is already BF16 must come back bit
        // for bit: the fusion reads and writes the checkpoint's own type.
        let values = [0.0_f32, 1.0, -2.5, 0.125, 1.0 / 3.0];
        let bytes = host_bytes(nn::DType::BF16, &values).expect("a BF16 encoding");
        let back = host_f32(nn::DType::BF16, &bytes, "test").expect("a BF16 decoding");
        let again = host_bytes(nn::DType::BF16, &back).expect("a second encoding");
        assert_eq!(bytes, again);
        assert_eq!(back[0], 0.0);
        assert_eq!(back[1], 1.0);
        assert_eq!(back[2], -2.5);
        assert_eq!(back[3], 0.125);
        assert!((back[4] - 1.0 / 3.0).abs() < 0.004, "{} ", back[4]);
    }

    #[test]
    fn a_head_type_the_host_fusion_cannot_read_is_refused() {
        assert!(host_f32(nn::DType::I32, &[0, 0, 0, 0], "test").is_err());
        assert!(host_bytes(nn::DType::I32, &[0.0]).is_err());
    }

    #[test]
    fn a_trace_matching_the_forward_is_accepted() {
        let views = [view(4, 8, DType::BF16), view(4, 8, DType::BF16)];
        let trace = Trace {
            layers: &[0, 49],
            views: &views,
            index: index(4),
        };
        assert!(trace.check(50, 4, 8, DType::BF16).is_ok());
    }

    #[test]
    fn a_trace_of_another_shape_or_dtype_is_refused() {
        let views = [view(4, 8, DType::BF16)];
        assert!(
            Trace {
                layers: &[0],
                views: &views,
                index: index(4),
            }
            .check(50, 4, 9, DType::BF16)
            .is_err(),
            "a view of another width"
        );
        assert!(
            Trace {
                layers: &[0],
                views: &views,
                index: index(4),
            }
            .check(50, 4, 8, DType::F32)
            .is_err(),
            "a view of another dtype"
        );
    }

    #[test]
    fn a_trace_naming_a_layer_the_stack_does_not_have_is_refused() {
        let views = [view(4, 8, DType::BF16)];
        let trace = Trace {
            layers: &[50],
            views: &views,
            index: index(4),
        };
        assert!(trace.check(50, 4, 8, DType::BF16).is_err());
    }

    #[test]
    fn a_trace_whose_lists_disagree_or_whose_index_is_short_is_refused() {
        let views = [view(4, 8, DType::BF16)];
        assert!(
            Trace {
                layers: &[0, 1],
                views: &views,
                index: index(4),
            }
            .check(50, 4, 8, DType::BF16)
            .is_err(),
            "two layers, one view"
        );
        assert!(
            Trace {
                layers: &[0],
                views: &views,
                index: index(3),
            }
            .check(50, 4, 8, DType::BF16)
            .is_err(),
            "a short identity index"
        );
    }
}
