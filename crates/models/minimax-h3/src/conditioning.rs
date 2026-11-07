//! H3's dense text refiner: the prompt features the DiT conditions on at every step.
//!
//! The text encoder's Qwen hidden states are `text_dim` wide; the refiner projects them to the
//! transformer's width and refines each document with dense bidirectional attention and a
//! pre-normalized SwiGLU (`uniserve/models/minimax_h3/conditioning.py`). It runs once per request -
//! the executor keeps its output in the request's state and hands it to every denoising step - so
//! it is a plain block of the transformer component, not a schedule.
//!
//! ## Documents in one batch
//!
//! Every document is padded to the same row count, so the queries are the whole packed row range
//! and the keys are only the rows a document's text occupies: a padding row never becomes a key.
//! Both boundary tables are device data (`[documents + 1]` int32 cumulative counts) and the packed
//! row count is the fixed capacity, so one captured call serves every prompt length
//! ([`Documents`]).
//!
//! ## The partition
//!
//! The refiner belongs to the denoiser component, but it is not sharded: it reads at most the
//! prompt's rows once per request, so every rank of a Ulysses group refines the whole prompt with
//! every head and holds the result, which is what every one of its own packed rows gathers from.
//! Sharding two blocks over a thousand rows would save nothing and put two exchanges on the first
//! step of every request.

use nn::layers::{Activation, Attention, GatedActivation, Linear, RmsNorm, ScaledResidual};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

use crate::config::TransformerConfig;
use crate::{Error, Result};

/// The fold's whole reservation from the configuration alone, in reservation order.
///
/// The prompt's refinement runs once per request, inside the denoiser's arena and before the first
/// evaluation (`Denoiser::prepare_conditioning`), so its scratch is part of a stage's floor. The
/// plan that sizes the arena (`crate::denoiser::denoise_arena`) prices it from this function, which
/// is why the extents live here beside the fold that reserves them rather than in the plan.
///
/// The working buffer is the whole refined document (`[rows, hidden]`, the input projection's
/// output); every block then reserves its own [`Scratch`].
pub fn fold_plan(config: &TransformerConfig, rows: u32) -> Vec<(String, Shape)> {
    let mut plan = vec![(
        "refiner.working".to_owned(),
        Shape::new(&[u64::from(rows), u64::from(config.hidden_size)]),
    )];
    for index in 0..config.num_refiner_layers {
        let scratch = Scratch::new(
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.intermediate_size,
            rows,
        );
        for (name, shape) in scratch.entries() {
            plan.push((format!("refiner.{index}.{name}"), shape));
        }
    }
    plan
}

/// The bytes the fold's whole reservation needs, from the configuration alone.
///
/// The denoiser's stage floor takes the larger of this and one evaluation's plan, because the two
/// occupy the same offsets: the fold runs only on the live path, before the first replay, and the
/// captured graph's nodes carry the planning walk's offsets from the region's base.
pub fn fold_bytes(config: &TransformerConfig, rows: u32, dtype: nn::DType) -> u64 {
    let width = dtype.bytes().unwrap_or(4);
    fold_plan(config, rows)
        .iter()
        .map(|(_, shape)| shape.dims().iter().product::<u64>())
        .sum::<u64>()
        * width
}

/// The per-request device tables the refiner's attention reads.
///
/// Both are int32 cumulative counts over the packed rows, `[documents + 1]` long: the queries cover
/// every padded row and the keys stop at each document's text length.
pub struct Documents {
    /// The query boundaries: one packed row per row, so the counts are `0, rows, 2 · rows, ...`.
    pub cu_seq_lens_q: TensorView,
    /// The key boundaries: each document's text rows accumulated, so a padding row is no key.
    pub cu_seq_lens_k: TensorView,
    /// The longest key run of any document: the last entry of `cu_seq_lens_k`, as a launch value.
    pub key_rows: u32,
}

/// One refined document block: pre-normalized attention and a value-first SwiGLU.
pub struct RefinerBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    gate_up: Linear,
    down: Linear,
    /// The stepwise gated activation, as the reference's refiner asks for
    /// (`conditioning.py:75-80`, `Rounding.STEPWISE`).
    gated: GatedActivation,
    attention: Attention,
    /// The identity add of the attention and feed-forward residuals.
    residual: ScaledResidual,
    hidden_size: u32,
    head_dim: u32,
    heads: u32,
    intermediate_size: u32,
    dtype: nn::DType,
}

impl RefinerBlock {
    /// Read block `index` of the checkpoint's token refiner.
    fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let prefix = format!("token_refiner.refiner_blocks.{index}");
        let hidden = u64::from(config.hidden_size);
        let inner = u64::from(config.num_attention_heads) * u64::from(config.head_dim);
        let ffn = u64::from(config.intermediate_size);

        let query = projection(loader, &format!("{prefix}.attn.to_q.weight"), inner, hidden)?;
        let query_dtype = query.view().dtype;
        let key = projection(loader, &format!("{prefix}.attn.to_k.weight"), inner, hidden)?;
        let value = projection(loader, &format!("{prefix}.attn.to_v.weight"), inner, hidden)?;
        let output = projection(
            loader,
            &format!("{prefix}.attn.to_out.0.weight"),
            hidden,
            inner,
        )?;
        // The checkpoint fuses the feed-forward's value half first and its gate half second.
        let gate_up = projection(
            loader,
            &format!("{prefix}.ff.net.0.proj.weight"),
            2 * ffn,
            hidden,
        )?;
        let down = projection(loader, &format!("{prefix}.ff.net.2.weight"), hidden, ffn)?;

        let attention = Attention::dense(
            config.head_dim,
            config.num_attention_heads,
            config.num_attention_heads,
            rows,
            false,
            None,
            scaling(config.head_dim),
        )?;

        let stream = query.dtype();
        Ok(Self {
            dtype: query_dtype,
            intermediate_size: config.intermediate_size,
            norm1: norm(
                loader,
                &format!("{prefix}.norm1.weight"),
                hidden,
                config.norm_eps,
            )?,
            norm2: norm(
                loader,
                &format!("{prefix}.norm2.weight"),
                hidden,
                config.norm_eps,
            )?,
            query_norm: norm(
                loader,
                &format!("{prefix}.attn.norm_q.weight"),
                u64::from(config.head_dim),
                config.qk_norm_eps,
            )?,
            key_norm: norm(
                loader,
                &format!("{prefix}.attn.norm_k.weight"),
                u64::from(config.head_dim),
                config.qk_norm_eps,
            )?,
            query: Linear::new(query, None, WeightOrder::NxK)?,
            key: Linear::new(key, None, WeightOrder::NxK)?,
            value: Linear::new(value, None, WeightOrder::NxK)?,
            output: Linear::new(output, None, WeightOrder::NxK)?,
            gate_up: Linear::new(gate_up, None, WeightOrder::NxK)?,
            down: Linear::new(down, None, WeightOrder::NxK)?,
            gated: GatedActivation::new(ops::activation::Activation::Silu).with_stepwise(),
            attention,
            residual: identity(loader, u64::from(config.hidden_size), stream)?,
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            heads: config.num_attention_heads,
        })
    }

    /// Refine `hidden`, in place.
    fn forward(&self, f: &Forward, hidden: TensorView, visible: &Documents) -> Result<()> {
        let row_count = u32::try_from(hidden.shape.dim(0))
            .map_err(|_| Error::checkpoint("a refiner row count above u32::MAX".to_owned()))?;
        let rows = Rows::reserve(self, f, row_count)?;

        // Attention: normalize, project every head, normalize each head's query and key, attend.
        self.norm1.forward(f, hidden, rows.normed)?;
        self.query.forward(f, rows.normed, rows.query)?;
        self.key.forward(f, rows.normed, rows.key)?;
        self.value.forward(f, rows.normed, rows.value)?;
        // The query and key norms are **per head**: a projection writes `[rows, heads · head]` and
        // the normalization is over one head's channels, so it reads the same buffer as
        // `[rows · heads, head]`. Handing it the flat projection is a width the norm refuses
        // (`RmsNorm::forward` checks the row's extent) - and it refused the whole refiner, which is
        // why no forward of it had ever completed.
        let head = self.head_dim;
        let heads = self.heads;
        self.query_norm.forward(
            f,
            rows.query.reshape(per_head(rows.query, head)),
            rows.query_normed.reshape(per_head(rows.query, head)),
        )?;
        self.key_norm.forward(
            f,
            rows.key.reshape(per_head(rows.key, head)),
            rows.key_normed.reshape(per_head(rows.key, head)),
        )?;

        let (query, key, value) = (rows.query_normed, rows.key_normed, rows.value);
        // The attention's operands are rank 3 - one axis of tokens, then the heads, then a head's
        // channels (`ops/src/attention/dense.rs`) - while every buffer here is `[rows, inner]`.
        self.attention.attend_dense(
            f,
            query.reshape(with_heads(query, heads, head)),
            key.reshape(with_heads(key, heads, head)),
            value.reshape(with_heads(value, heads, head)),
            Some(visible.cu_seq_lens_q),
            Some(visible.cu_seq_lens_k),
            // One document per request over packed rows: every query sees the whole key set.
            visible.key_rows,
            rows.attended
                .reshape(with_heads(rows.attended, heads, head)),
        )?;
        self.output.forward(f, rows.attended, rows.update)?;
        self.residual.forward(f, hidden, rows.update)?;

        // Feed-forward: normalize, project the packed value and gate halves, gate, project back.
        self.norm2.forward(f, hidden, rows.normed)?;
        self.gate_up.forward(f, rows.normed, rows.packed)?;
        let (value, gate) = crate::transformer::projection_halves(rows.packed);
        self.gated.forward(f, gate, value, rows.ffn)?;
        self.down.forward(f, rows.ffn, rows.update)?;
        self.residual.forward(f, hidden, rows.update)?;
        Ok(())
    }
}

/// The refiner: the projector from the text encoder's width, its blocks, and the final norm.
pub struct TokenRefiner {
    input: Linear,
    blocks: Vec<RefinerBlock>,
    norm: RmsNorm,
    hidden_size: u32,
    text_dim: u32,
    rows: u32,
}

impl TokenRefiner {
    /// The programs the prompt's refinement launches, from the configuration alone.
    ///
    /// The refiner is a stack of its own: per-head RMS normalizations with learned weights (no
    /// rotation — the prompt's refinement has no positional table), one dense product, the stepwise
    /// gated activation, and the residual each block folds its two updates through.
    /// `boundary` is the type the refiner's stream carries.
    pub fn programs(
        config: &TransformerConfig,
        boundary: nn::DType,
    ) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use ops::activation::ActAndMul as ActAndMulOp;
        use ops::attention::dense::DenseAttention;
        use ops::gemm::Gemm;
        use ops::norm::RmsNorm as RmsNormOp;
        use ops::residual::ScaledResidual as ResidualOp;

        let hidden = config.hidden_size;
        let inner = config.num_attention_heads * config.head_dim;
        let head = config.head_dim;
        let heads = config.num_attention_heads;
        // The refiner's stream is the boundary's type: it is handed the prompt's features where
        // they are widened into the network. The identity scale is the one exception - it is
        // applied at the type the blocks' output has, which is the stack's.
        let stream = boundary;
        let mut programs = Programs::default();

        // Every projection: the input one, the four of each block's attention and its feed-forward
        // pair. None carries a bias.
        programs.launch::<Gemm>(&Linear::descriptor_of(
            hidden,
            stream,
            stream,
            WeightOrder::NxK,
            false,
            stream,
        ));

        // Two normalization widths: the hidden state's and one head's, which the query and the key
        // normalize over.
        for columns in [hidden, head] {
            programs.launch::<RmsNormOp>(&RmsNorm::descriptor_of(columns, stream));
        }

        // The identity residual each block folds both of its updates through.
        programs.launch::<ResidualOp>(&ScaledResidual::descriptor_of(
            stream,
            u64::from(hidden),
            false,
        )?);

        // The attention over the request's document, and the stepwise gated activation.
        let attention = Attention::dense(
            head,
            heads,
            heads,
            hidden,
            false,
            None,
            1.0 / (head as f32).sqrt(),
        )
        .map_err(Error::from)?;
        programs.launch::<DenseAttention>(
            &attention.dense_descriptor(stream, stream, stream, hidden, true),
        );
        // **Both row strides are twice the width**: the refiner reads its two halves through the
        // same packed projection the transformer does, so each half's row stride is `2 *
        // intermediate` - the launch record states `gate_stride_28672` where this stated 14336.
        // And both width classes are reachable, as at the transformer's packed site.
        let packed_stride = 2 * u64::from(config.intermediate_size);
        for elements in [u64::from(inner), 1] {
            programs.launch::<ActAndMulOp>(&GatedActivation::descriptor_of(
                stream,
                u64::from(config.intermediate_size),
                Activation::Silu,
                packed_stride,
                packed_stride,
                elements,
                true,
            )?);
        }

        Ok(programs.finish())
    }

    /// Every checkpoint tensor the refiner reads, as complete names.
    ///
    /// [`TokenRefiner::load`]'s requests, in its order: the input projection, each block's ten
    /// weights, and the final norm.
    pub fn keys(config: &TransformerConfig) -> Vec<String> {
        let mut keys = vec![
            "context_embedder.weight".to_owned(),
            "context_embedder.bias".to_owned(),
        ];
        for index in 0..config.num_refiner_layers {
            let prefix = format!("token_refiner.refiner_blocks.{index}");
            for name in [
                "norm1.weight",
                "norm2.weight",
                "attn.norm_q.weight",
                "attn.norm_k.weight",
                "attn.to_q.weight",
                "attn.to_k.weight",
                "attn.to_v.weight",
                "attn.to_out.0.weight",
                "ff.net.0.proj.weight",
                "ff.net.2.weight",
            ] {
                keys.push(format!("{prefix}.{name}"));
            }
        }
        keys.push("token_refiner.final_norm.weight".to_owned());
        keys
    }

    /// Read the checkpoint's token refiner, bounded to `rows` packed rows per request.
    ///
    /// `rows` is the deployment layout's padded row count: the attention's key extent and the width
    /// every captured call is shaped for.
    pub fn load(
        config: &TransformerConfig,
        rows: u32,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        if config.num_refiner_layers == 0 {
            return Err(Error::config(
                "an H3 denoiser's token refiner has at least one layer".to_owned(),
            ));
        }
        let hidden = u64::from(config.hidden_size);
        let text = u64::from(config.text_dim);
        let weight = matrix(loader, "context_embedder.weight", hidden, text)?;
        let bias = vector(loader, "context_embedder.bias", hidden)?;
        let input = Linear::new(weight, Some(bias), WeightOrder::NxK)?;
        let blocks = (0..config.num_refiner_layers)
            .map(|index| RefinerBlock::load(index, config, rows, loader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input,
            blocks,
            norm: norm(
                loader,
                "token_refiner.final_norm.weight",
                hidden,
                config.norm_eps,
            )?,
            hidden_size: config.hidden_size,
            text_dim: config.text_dim,
            rows,
        })
    }

    /// How many packed rows the refiner is bounded to.
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Refine `features` `[rows, text_dim]` into `out` `[rows, hidden_size]`.
    pub fn forward(
        &self,
        f: &Forward,
        features: TensorView,
        visible: &Documents,
        out: TensorView,
    ) -> Result<()> {
        let rows = features.shape.dim(0);
        if rows == 0 {
            return Err(Error::checkpoint(
                "a refiner reads at least one row".to_owned(),
            ));
        }
        if features.shape.last() != u64::from(self.text_dim) {
            return Err(Error::checkpoint(format!(
                "refiner features are [rows, {}], not {:?}",
                self.text_dim,
                features.shape.dims()
            )));
        }
        // The whole refined document is the chain of blocks over a working buffer; the input
        // projection is the same dtype as the features it reads.
        let working = f.arena().tensor(
            features.dtype,
            Shape::new(&[rows, u64::from(self.hidden_size)]),
        )?;
        self.input.forward(f, features, working)?;
        for block in &self.blocks {
            block.forward(f, working, visible)?;
        }
        Ok(self.norm.forward(f, working, out)?)
    }
}

/// The scratch one block's forward reads and writes, reserved from the forward's arena.
///
/// One block runs at a time and the arena is reused, so every buffer is written before it is read.
struct Rows {
    normed: TensorView,
    query: TensorView,
    key: TensorView,
    value: TensorView,
    query_normed: TensorView,
    key_normed: TensorView,
    attended: TensorView,
    update: TensorView,
    packed: TensorView,
    ffn: TensorView,
}

/// The extents one refiner block's scratch holds, in the order the block reserves them.
///
/// **One statement for the two readers.** The fold reserves these from the denoiser's arena
/// ([`Rows::reserve`]) and the same arena is sized by the denoiser's plan before any request arrives
/// (`crate::denoiser::denoise_arena`), so a buffer added here is priced there without a second edit.
#[derive(Clone, Copy, Debug)]
pub struct Scratch {
    /// `[rows, hidden]` the normalized rows.
    normed: Shape,
    query: Shape,
    key: Shape,
    value: Shape,
    query_normed: Shape,
    key_normed: Shape,
    attended: Shape,
    update: Shape,
    packed: Shape,
    ffn: Shape,
}

impl Scratch {
    /// The scratch of a block that is `hidden` wide, with `heads` heads of `head_dim` and
    /// `intermediate` feed-forward columns.
    pub fn new(hidden: u32, head_dim: u32, heads: u32, intermediate: u32, rows: u32) -> Self {
        let hidden = u64::from(hidden);
        let head = u64::from(head_dim);
        let inner = u64::from(heads) * head;
        // The gate projection's rows hold the checkpoint's two fused halves of one width.
        let packed_width = 2 * u64::from(intermediate);
        let ffn = u64::from(intermediate);
        let rows = u64::from(rows);
        Self {
            normed: Shape::new(&[rows, hidden]),
            query: Shape::new(&[rows, inner]),
            key: Shape::new(&[rows, inner]),
            value: Shape::new(&[rows, inner]),
            query_normed: Shape::new(&[rows, inner]),
            key_normed: Shape::new(&[rows, inner]),
            attended: Shape::new(&[rows, inner]),
            update: Shape::new(&[rows, hidden]),
            packed: Shape::new(&[rows, packed_width]),
            ffn: Shape::new(&[rows, ffn]),
        }
    }

    /// Every buffer in reservation order, named as a plan names it.
    pub fn entries(&self) -> [(&'static str, Shape); 10] {
        [
            ("normed", self.normed),
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("query_normed", self.query_normed),
            ("key_normed", self.key_normed),
            ("attended", self.attended),
            ("update", self.update),
            ("packed", self.packed),
            ("ffn", self.ffn),
        ]
    }

    /// The elements all ten hold.
    pub fn elements(&self) -> u64 {
        self.entries()
            .iter()
            .map(|(_, shape)| shape.dims().iter().product::<u64>())
            .sum()
    }
}

impl Rows {
    fn reserve(block: &RefinerBlock, f: &Forward, rows: u32) -> Result<Self> {
        let dtype = block.dtype;
        let scratch = Scratch::new(
            block.hidden_size,
            block.head_dim,
            block.heads,
            block.intermediate_size,
            rows,
        );
        let arena = f.arena();
        Ok(Self {
            normed: arena.tensor(dtype, scratch.normed)?,
            query: arena.tensor(dtype, scratch.query)?,
            key: arena.tensor(dtype, scratch.key)?,
            value: arena.tensor(dtype, scratch.value)?,
            query_normed: arena.tensor(dtype, scratch.query_normed)?,
            key_normed: arena.tensor(dtype, scratch.key_normed)?,
            attended: arena.tensor(dtype, scratch.attended)?,
            update: arena.tensor(dtype, scratch.update)?,
            packed: arena.tensor(dtype, scratch.packed)?,
            ffn: arena.tensor(dtype, scratch.ffn)?,
        })
    }
}

/// One flat `[rows, heads · head]` buffer as the shape a per-head norm reads: `[rows · heads, head]`.
fn per_head(view: TensorView, head_dim: u32) -> Shape {
    let head = u64::from(head_dim).max(1);
    Shape::new(&[view.shape.numel() / head, head])
}

/// One flat `[rows, heads · head]` buffer as the shape the attention takes: `[tokens, heads, head]`.
fn with_heads(view: TensorView, heads: u32, head_dim: u32) -> Shape {
    let width = u64::from(heads) * u64::from(head_dim);
    Shape::new(&[
        view.shape.numel() / width.max(1),
        u64::from(heads),
        u64::from(head_dim),
    ])
}

/// The attention scale of one head: `1 / sqrt(head_dim)`.
fn scaling(head_dim: u32) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// The identity residual: every channel's update added unchanged.
///
/// The layer's width is its scale's, so the scale holds one element per channel of the stream.
///
/// The operator reads the scale in the stream's element type, so the ones are stated in it.
fn identity(
    loader: &mut WeightLoader<'_>,
    hidden: u64,
    stream: nn::DType,
) -> Result<ScaledResidual> {
    let one: &[u8] = match stream {
        nn::DType::F32 => &[0x00, 0x00, 0x80, 0x3f],
        nn::DType::BF16 => &[0x80, 0x3f],
        other => {
            return Err(Error::checkpoint(format!(
                "a refiner's stream is F32 or BF16, not {other}"
            )));
        }
    };
    let scale = loader.constant(
        "refiner.identity_scale",
        stream,
        Shape::new(&[hidden]),
        &one.repeat(hidden as usize),
        "minimax-h3",
    )?;
    Ok(ScaledResidual::new(scale, None)?)
}

/// A `[rows, columns]` weight of the checkpoint.
fn matrix(loader: &mut WeightLoader<'_>, name: &str, rows: u64, columns: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(
        name,
        0,
        Shape::new(&[rows, columns]),
    ))?)
}

/// A `[rows]` weight of the checkpoint.
fn vector(loader: &mut WeightLoader<'_>, name: &str, rows: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(name, 0, Shape::new(&[rows])))?)
}

/// A projection of the checkpoint: `[out, in]`.
fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    rows: u64,
    columns: u64,
) -> Result<Weight> {
    matrix(loader, name, rows, columns)
}

/// A normalization's learned gain.
fn norm(loader: &mut WeightLoader<'_>, name: &str, columns: u64, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm::new(vector(loader, name, columns)?, eps)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::tensor::{DevicePtr, Shape};

    /// A view of `shape` at a host address: these tests never launch, so the address is a marker.
    fn view(shape: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), ops::DType::BF16, Shape::new(shape))
    }

    /// The refiner's two flat buffers are read by two readers that want two different ranks.
    ///
    /// A per-head normalization is over one head's channels, so it reads a `[rows, heads · head]`
    /// projection as `[rows · heads, head]`; the attention takes `[tokens, heads, head]`. Handing
    /// either of them the flat projection is what stopped the refiner's forward from ever
    /// completing - and a shape a reader refuses is not a shape a test can leave to a comment.
    #[test]
    fn the_refiners_flat_buffers_are_read_as_a_head_axis_or_as_one_norm_row() {
        let (rows, heads, head) = (5u64, 56u64, 128u64);
        let flat = view(&[rows, heads * head]);
        let normed = per_head(flat, head as u32);
        assert_eq!(normed.dims(), &[rows * heads, head]);
        let attended = with_heads(flat, heads as u32, head as u32);
        assert_eq!(attended.dims(), &[rows, heads, head]);
        // Both views hold the buffer's own elements: a reshape, not a copy or a second allocation.
        assert_eq!(normed.numel(), flat.shape.numel());
        assert_eq!(attended.numel(), flat.shape.numel());
    }
}
