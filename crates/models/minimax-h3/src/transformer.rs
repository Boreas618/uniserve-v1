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
use crate::config::{AttentionKind, TransformerConfig};
use crate::modulation::AdaLnCache;
use crate::partition::Partition;
use crate::schedule::Modality;
use crate::{Error, Result};

use nn::layers::{GatedResidual, Linear, ModulatedRmsNorm, PackedSwiGlu, RmsNorm};
use nn::tensor::Backing;
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

/// The rotary width of one head's three axes: 16 frequencies per axis, 32 channels each.
pub const ROTATED_AXES: [u32; 3] = [32, 32, 32];

/// One dilated layer of the stack.
pub struct Layer {
    norm1: Weight,
    norm2: Weight,
    attention: attention::Dense,
    gate_up: Linear,
    down: Linear,
    swiglu: PackedSwiGlu,
    intermediate_size: u32,
    eps: f32,
}

impl Layer {
    /// Read layer `index` of a partition.
    fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let prefix = format!("transformer_blocks.{index}");
        let hidden = u64::from(config.hidden_size);
        let ffn = u64::from(config.intermediate_size);
        Ok(Self {
            norm1: gain(loader, &format!("{prefix}.norm1.weight"), hidden)?,
            norm2: gain(loader, &format!("{prefix}.norm2.weight"), hidden)?,
            attention: attention::Dense::load(
                index,
                config,
                rows,
                &ROTATED_AXES,
                partition,
                loader,
            )?,
            // The checkpoint fuses the feed-forward's value half first and its gate half second.
            gate_up: projection(
                loader,
                &format!("{prefix}.ff.net.0.proj.weight"),
                2 * ffn,
                hidden,
            )?,
            down: projection(loader, &format!("{prefix}.ff.net.2.weight"), hidden, ffn)?,
            swiglu: PackedSwiGlu::new(None)?,
            intermediate_size: config.intermediate_size,
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
        visible: &Visible,
        total_rows: u32,
    ) -> Result<()> {
        let width = hidden.shape.last();
        let (shift_attention, scale_attention, gate_attention) = chunks(modulation, width, 0)?;
        let (shift_mlp, scale_mlp, gate_mlp) = chunks(modulation, width, 3)?;
        let arena = f.arena();
        let dtype = hidden.dtype;
        let tokens = hidden.shape.dim(0);
        let normalized = arena.tensor(dtype, Shape::new(&[tokens, width]))?;
        let update = arena.tensor(dtype, Shape::new(&[tokens, width]))?;

        // Attention: normalize, attend, and fold the gated update into `out` while normalizing the
        // feed-forward's input.
        let attention_norm = ModulatedRmsNorm::new(
            self.norm1.clone(),
            shift_attention,
            scale_attention,
            None,
            self.eps,
        )?;
        attention_norm.forward(f, hidden, None, rows, normalized, None, None)?;
        self.attention
            .forward(f, normalized, factors, visible, total_rows, update)?;
        let mlp_norm = ModulatedRmsNorm::new(
            self.norm2.clone(),
            shift_mlp,
            scale_mlp,
            Some(gate_attention),
            self.eps,
        )?;
        mlp_norm.forward(f, hidden, Some(update), rows, normalized, None, Some(out))?;

        // Feed-forward: the packed value and gate halves, the gated activation, the down
        // projection, and the second gated residual into the retained residual.
        let ffn = u64::from(self.intermediate_size);
        let packed = arena.tensor(dtype, Shape::new(&[tokens, 2 * ffn]))?;
        let mlp = arena.tensor(dtype, Shape::new(&[tokens, ffn]))?;
        self.gate_up.forward(f, normalized, packed)?;
        self.swiglu.forward(f, packed, mlp, None)?;
        self.down.forward(f, mlp, update)?;
        Ok(GatedResidual::new(gate_mlp)?.forward(f, out, update, rows)?)
    }
}

impl Transformer {
    /// Project one modality's patchified audio rows into the hidden width.
    pub fn project_audio(&self, f: &Forward, rows: TensorView, out: TensorView) -> Result<()> {
        let audio = self.audio_proj_in.as_ref().ok_or_else(|| {
            Error::Unsupported("this H3 partition has no audio input projection".to_owned())
        })?;
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
        Ok(layer.attention.factors(f, positions, cosine, sine)?)
    }

    /// The final normalization and one modality's output projection: `hidden` in, velocity out.
    ///
    /// The final normalization is modulated like every other one: `[groups, 2 * hidden]` of shift
    /// and scale, gathered at the AdaLN entry the step reads. `rows` selects each row's group.
    pub fn velocity(
        &self,
        f: &Forward,
        hidden: TensorView,
        modality: Modality,
        modulation: TensorView,
        rows: TensorView,
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
        match modality {
            Modality::Video => Ok(self.video_out.forward(f, normalized, out)?),
            Modality::Audio => {
                let audio = self.audio_out.as_ref().ok_or_else(|| {
                    Error::Unsupported("this H3 partition predicts no audio velocity".to_owned())
                })?;
                Ok(audio.forward(f, normalized, out)?)
            }
        }
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
        Ok(borrowed(view)?)
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
    video_out: Linear,
    audio_out: Option<Linear>,
    config: TransformerConfig,
}

impl Transformer {
    /// Read one DiT partition of the checkpoint.
    ///
    /// `rows` bounds one call's keys: the largest packed row count the deployment's layouts hold,
    /// not a request's count, so one partition serves every layout.
    pub fn load(
        config: &TransformerConfig,
        kind: AttentionKind,
        rows: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        if config.output_heads != 1 {
            return Err(Error::Unsupported(format!(
                "the H3 partition predicts {} output heads per evaluation (a parallel-decoding \
                 student); its fused per-step heads need a step-indexed gather of a [steps, out, \
                 in] weight, which crates/nn does not express yet",
                config.output_heads
            )));
        }
        if kind != AttentionKind::DENSE {
            return Err(Error::Unsupported(
                "the H3 partition's sparse attention is composed from nn::layers::SegmentAttention \
                 and nn::layers::VsaTiles; this build composes the dense kind only"
                    .to_owned(),
            ));
        }
        let hidden = u64::from(config.hidden_size);
        let video_in = u64::from(config.video_channels) * 4;
        let audio_in = u64::from(config.audio_channels);
        let layers = (0..config.num_hidden_layers)
            .map(|index| Layer::load(index, config, rows, partition, loader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: projection_with_bias(loader, "proj_in", hidden, video_in)?,
            audio_proj_in: Some(projection_with_bias(
                loader,
                "audio_proj_in",
                hidden,
                audio_in,
            )?),
            layers,
            norm_out: RmsNorm::new(
                gain(loader, "norm_out.norm.weight", hidden)?,
                config.norm_eps,
            )?,
            video_out: projection_with_bias(loader, "proj_out", video_in, hidden)?,
            audio_out: Some(projection_with_bias(
                loader,
                "audio_proj_out",
                audio_in,
                hidden,
            )?),
            config: config.clone(),
        })
    }

    /// The configuration this partition was read with.
    pub const fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Project one modality's packed rows into the hidden width.
    pub fn project(&self, f: &Forward, rows: TensorView, out: TensorView) -> Result<()> {
        Ok(self.proj_in.forward(f, rows, out)?)
    }

    /// Project one modality's hidden rows back to its velocity.
    pub fn predict(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        Ok(self.video_out.forward(f, hidden, out)?)
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
        visible: &Visible,
        total_rows: u32,
    ) -> Result<()> {
        let groups = adaln.groups();
        let width = hidden.shape.last();
        let tokens = hidden.shape.dim(0);
        let arena = f.arena();
        let entries = arena.tensor(nn::DType::I32, Shape::new(&[1, u64::from(groups.count())]))?;
        adaln.entries_of(f, step, entries)?;
        let modulation = arena.tensor(
            hidden.dtype,
            Shape::new(&[u64::from(groups.count()) * 3, 6 * width]),
        )?;
        let mut alternate = arena.tensor(hidden.dtype, Shape::new(&[tokens, width]))?;
        let mut current = hidden;
        for (index, layer) in self.layers.iter().enumerate() {
            adaln.layer(f, index, entries, modulation)?;
            layer.forward(
                f, current, alternate, modulation, rows, factors, visible, total_rows,
            )?;
            std::mem::swap(&mut current, &mut alternate);
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

/// A `[out, in]` projection of the checkpoint, without bias.
fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Linear> {
    let weight = loader.load(&WeightRequest::region(
        &format!("{name}.weight"),
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
) -> Result<Linear> {
    let weight = loader.load(&WeightRequest::region(
        &format!("{name}.weight"),
        0,
        Shape::new(&[rows, columns]),
    ))?;
    let bias = gain(loader, &format!("{name}.bias"), rows)?;
    Ok(Linear::new(weight, Some(bias), WeightOrder::NxK)?)
}

/// A normalization's or projection's learned gain.
fn gain(loader: &mut WeightLoader<'_>, name: &str, columns: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(name, 0, Shape::new(&[columns])))?)
}
