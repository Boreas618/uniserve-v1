//! The capability traits a model implements and the executor calls, and the borrowed inputs and
//! outputs they exchange (specs/worker.md 4.3).
//!
//! The executor dispatches by capability, never by concrete model: a runner owns a
//! `Box<dyn Encoder>`, `Box<dyn Denoiser>`, `Box<dyn Decoder>` or `Box<dyn CausalLm>` and calls only
//! the methods declared here. Every input and output is a borrowed view or a small plain value: no
//! request identity, no sampling state, no pool slot and no execution observation crosses this
//! boundary (`AGENTS.md`, Model Library Boundary).
//!
//! ## What the mechanism writes and what the model reads
//!
//! A batch has two halves. [`BatchRows`] is the host half: the scheduler's mechanism assembles the
//! per-call rows and their column values, the executor uploads them, and they are not visible to a
//! model. [`TokenBatch`] is the device half: the views a model's layers read, laid out exactly as
//! the attention and KV-write kernels read them, so a forward does no host conversion (specs/worker.md
//! 4.3, 4.4, 9).
//!
//! The paged cache is [`KvView`]: one entry per attention group, each with its block table, its
//! page geometry and its per-layer key/value planes. A model with one attention type (Qwen3-MoE) has
//! one group; a model whose blocks attend differently has one per group.

use std::ops::Range;

use ops::PhaseKind;

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::tensor::{DType, TensorView};

/// One host-side call of a batch: which request, which phase, which rows it occupies.
///
/// The mechanism builds these from the scheduler's plan; the executor turns them into the device
/// columns [`BatchRows`] carries and uploads them into a [`TokenBatch`]. A call's `rows` are the
/// half-open row range it owns in the batch's device columns.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BatchRow {
    /// The request slot the call belongs to.
    pub request: u32,
    /// What the call does.
    pub kind: RowKind,
    /// Which attention group the call's keys and values belong to.
    pub cache_group: u32,
    /// The rows this call owns in the batch's device columns.
    pub rows: Range<u32>,
}

/// What one call of a batch computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowKind {
    /// Prompt tokens of a request.
    Prefill,
    /// One generated token per request.
    Decode,
    /// Speculative verification rows: drafted tokens whose logits the model produces.
    Verify,
}

/// The host columns of one batch, as the mechanism writes them and the executor uploads them.
///
/// Every column is already in the layout the device views of [`TokenBatch`] have: the executor
/// copies these into the pinned arena and uploads them before the graph replay. The columns are
/// long-lived buffers the executor reuses every step, so assembling a batch does not allocate.
#[derive(Clone, Default, Debug)]
pub struct BatchRows {
    /// The calls of the batch, in issue order.
    pub calls: Vec<BatchRow>,
    /// `[tokens]` rotary positions.
    pub positions: Vec<i32>,
    /// `[tokens]` cache slots; `-1` writes nothing.
    pub slot_mapping: Vec<i32>,
    /// `[rows]` keys visible to each row's last query.
    pub seq_lens: Vec<i32>,
    /// `[rows + 1]` packed query offsets.
    pub query_offsets: Vec<i32>,
    /// `[rows]` hidden rows whose logits are wanted.
    pub logits_rows: Vec<i32>,
}

impl BatchRows {
    /// Drop every column, keeping the buffers' capacity.
    pub fn clear(&mut self) {
        self.calls.clear();
        self.positions.clear();
        self.slot_mapping.clear();
        self.seq_lens.clear();
        self.query_offsets.clear();
        self.logits_rows.clear();
    }

    /// The number of packed query tokens.
    pub fn tokens(&self) -> usize {
        self.positions.len()
    }

    /// The number of rows (sequences) of the batch.
    pub fn rows(&self) -> usize {
        self.seq_lens.len()
    }
}

/// The device-side layout of one batch, as a model's layers read it.
///
/// Every view addresses a column the mechanism wrote through [`BatchRows`]; a model never converts
/// a host value into a device one. The views live as long as the runner's batch buffers, so a
/// captured graph addresses them unchanged from capture to replay.
#[derive(Clone, Copy)]
pub struct TokenBatch {
    /// `[tokens]` i32 rotary positions.
    pub positions: TensorView,
    /// `[tokens]` i32 cache slots; `-1` writes nothing and initializes nothing.
    pub slot_mapping: TensorView,
    /// `[rows]` i32 keys visible to each row's last query.
    pub seq_lens: TensorView,
    /// `[rows + 1]` i32 packed query offsets.
    pub query_offsets: TensorView,
    /// `[rows]` i32 hidden rows whose logits are produced.
    pub logits_rows: TensorView,
    /// Which phase this batch is.
    pub phase: PhaseKind,
    /// Tokens per cache page.
    pub page_size: u32,
    /// Query tokens packed into this batch.
    pub num_tokens: u32,
    /// The bound on keys any row of this batch reads: the attention descriptor's reduction length.
    pub max_kv_len: u32,
    /// `[rows]` i32 drafted-token counts, when the batch verifies speculative drafts.
    pub draft_lens: Option<TensorView>,
}

impl TokenBatch {
    /// The number of rows (sequences) of the batch.
    pub fn rows(&self) -> u64 {
        self.seq_lens.numel()
    }

    /// The number of packed query tokens.
    pub fn tokens(&self) -> u64 {
        u64::from(self.num_tokens)
    }

    /// Whether this batch is a single generated token per row.
    pub fn is_decode(&self) -> bool {
        self.phase == PhaseKind::Decode
    }
}

/// The key/value planes of one attention layer of one group.
///
/// The cache layout is `[pages, page_size, kv_heads, head_dim]` per plane with one `i32`
/// initialization flag per page (specs/worker.md 9, KV 写入): a reader must not read a page before
/// its flag is raised, and a page's owner zeroes only storage it created.
#[derive(Clone, Copy)]
pub struct KvPlanes {
    /// `[pages, page_size, kv_heads, head_dim]` cached keys.
    pub key: TensorView,
    /// `[pages, page_size, kv_heads, head_dim]` cached values.
    pub value: TensorView,
    /// `[pages, 1, 1, 1]` i32, one flag per page of `key`.
    pub key_initialized: TensorView,
    /// `[pages, 1, 1, 1]` i32, one flag per page of `value`.
    pub value_initialized: TensorView,
}

/// The cache of one attention group: how rows address pages and where the pages are.
#[derive(Clone, Copy)]
pub struct KvGroup<'a> {
    /// `[rows, max_pages]` i32: each row's page ids.
    pub block_tables: TensorView,
    /// Tokens per page; the same as the batch's `page_size`.
    pub page_size: u32,
    /// Element type of the planes (BF16, or FP8 for an FP8 cache).
    pub dtype: DType,
    /// `[pages, 1, 1, 1]` f32: each page's key scale, required for an FP8 cache and absent
    /// otherwise. The page holds `key / scale`; the attention kernel multiplies it back in FP32.
    pub key_scales: Option<TensorView>,
    /// `[pages, 1, 1, 1]` f32: each page's value scale, as `key_scales`.
    pub value_scales: Option<TensorView>,
    /// One entry per layer that reads this group, in model order.
    pub layers: &'a [KvPlanes],
}

/// The paged cache of a whole model: one entry per attention group.
#[derive(Clone, Copy)]
pub struct KvView<'a> {
    /// The attention groups, in the order the model's attention layers index them.
    pub groups: &'a [KvGroup<'a>],
}

impl<'a> KvView<'a> {
    /// The group numbered `group`.
    pub fn group(&self, group: usize) -> Result<&KvGroup<'a>> {
        self.groups.get(group).ok_or_else(|| {
            Error::shape(
                "KvView",
                format!(
                    "attention group {group} was asked for but the cache has {}",
                    self.groups.len()
                ),
            )
        })
    }

    /// The planes of `layer` in `group`.
    pub fn planes(&self, group: usize, layer: usize) -> Result<&KvPlanes> {
        self.group(group)?.layers.get(layer).ok_or_else(|| {
            Error::shape(
                "KvView",
                format!(
                    "layer {layer} was asked for but group {group} holds {}",
                    self.group(group).map_or(0, |group| group.layers.len())
                ),
            )
        })
    }
}

/// What an encoder reads: token ids, the attention layout of their packed rows, and any media.
pub struct EncoderInputs<'a> {
    /// `[tokens]` token ids, of the dtype the embedding's descriptor names.
    pub ids: TensorView,
    /// The batch's positions, lengths and cache writes.
    pub batch: TokenBatch,
    /// The images of a multimodal tower, empty for a text-only encoder.
    pub images: &'a [ImageInput],
}

/// One image a multimodal tower reads: its pixels and the grid they form.
#[derive(Clone, Copy)]
pub struct ImageInput {
    /// `[tokens, pixels]` packed patches, or `[channels, height, width]` rasters.
    pub pixels: TensorView,
    /// `[1, axes]` i32 patch grid.
    pub grid: TensorView,
}

/// What an encoder writes.
pub struct EncoderOutputs {
    /// `[tokens, width]` encoded hidden states.
    pub hidden: TensorView,
}

/// A video request's numerical layout input: what the denoiser must lay out (specs/worker.md 4.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VideoRequest {
    /// The task the request asks for.
    pub task: VideoTask,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Frames of the request; the model snaps it to a legal count.
    pub frames: u32,
    /// Frames per second of the output.
    pub fps: u32,
    /// Frame-one anchors: each one pins the first generated frame.
    pub first_frames: u32,
    /// Last-frame anchors: each one pins the last generated frame.
    pub last_frames: u32,
    /// Reference images or clips: they condition the generation as a whole.
    ///
    /// The three roles are counted apart because they enter the layout differently — an anchor is
    /// one encoded frame at a fixed position, a reference is an encoded clip — and the reference
    /// implementation enforces it (`uniserve/models/denoiser.py`, `ConditionRole`).
    pub references: u32,
}

/// The tasks a video model serves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VideoTask {
    /// Text to video and audio.
    TextToVideoAudio,
    /// First and last frame to video and audio.
    FirstLastToVideoAudio,
    /// Reference media to video and audio.
    ReferenceToVideoAudio,
}

/// The latent layout a denoiser prepares for one request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    /// The video latent tensor.
    pub video: LatentShape,
    /// The audio latent tensor, when the task produces audio.
    pub audio: Option<LatentShape>,
    /// Text conditioning rows the layout reserves.
    pub text_rows: u32,
    /// Condition rows of the reference media.
    pub condition_rows: u32,
}

/// The extents of one latent tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LatentShape {
    /// Channels of the latent space.
    pub channels: u32,
    /// Latent frames (or audio latent steps).
    pub frames: u32,
    /// Latent rows.
    pub height: u32,
    /// Latent columns.
    pub width: u32,
}

/// The seed a noise draw is addressed by.
///
/// A draw is a function of this seed and the element's index, never of a batch position, so
/// recomputing a request after a migration draws the same noise (specs/worker.md 9, 初始噪声).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Seed(pub u64);

/// The latents of one request, as the denoiser reads and writes them.
#[derive(Clone, Copy)]
pub struct Latents {
    /// `[channels, frames, height, width]` video latents.
    pub video: TensorView,
    /// `[channels, steps]` audio latents, when the task has audio.
    pub audio: Option<TensorView>,
}

/// What a denoiser conditions on at a step.
///
/// The rows of every view follow the order the layout reserves: the encoded text first, then the
/// request's conditions in the order `VideoRequest` counts them (first-frame anchors, last-frame
/// anchors, references). A model that needs the roles reads them from the request it laid out, not
/// from this value, so nothing request-shaped crosses the boundary.
#[derive(Clone, Copy)]
pub struct Conditioning {
    /// `[rows, width]` encoded text.
    pub text: TensorView,
    /// `[rows, …]` encoded reference media, when the task has references.
    pub images: Option<TensorView>,
    /// `[rows, …]` encoded reference audio, when the task has audio references.
    pub audio: Option<TensorView>,
}

/// One denoising step's schedule values.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Step {
    /// The zero-based index of the step.
    pub index: u32,
    /// How many steps the schedule has.
    pub count: u32,
    /// The noise level the step starts from.
    pub sigma: f32,
    /// The noise level the step ends at.
    pub sigma_next: f32,
}

/// What a decoder writes: the decoded video frames and audio samples.
#[derive(Clone, Copy)]
pub struct DecodedMedia {
    /// `[frames, channels, height, width]` decoded frames.
    pub video: Option<TensorView>,
    /// `[samples]` decoded audio.
    pub audio: Option<TensorView>,
}

/// A tower that turns inputs into conditioning (a text encoder, a vision tower, a video encoder).
pub trait Encoder {
    /// Encode `inputs` into `out`.
    fn encode(&self, f: &Forward, inputs: EncoderInputs<'_>, out: EncoderOutputs) -> Result<()>;
}

/// A diffusion model that denoises latents step by step.
pub trait Denoiser {
    /// The latent layout this request needs.
    fn prepare_layout(&self, request: &VideoRequest) -> Layout;

    /// Draw the initial noise of `out` from `seed`.
    fn draw_noise(&self, f: &Forward, seed: Seed, out: Latents) -> Result<()>;

    /// Fold `cond` into the latents the schedule starts from.
    fn prepare_latents(&self, f: &Forward, cond: Conditioning, out: Latents) -> Result<()>;

    /// One denoising step: update `latents` in place with the model's velocity.
    fn denoise(&self, f: &Forward, step: Step, latents: Latents, cond: Conditioning) -> Result<()>;
}

/// A decoder that turns latents back into media.
pub trait Decoder {
    /// Decode `latents` into `out`.
    fn decode(&self, f: &Forward, latents: TensorView, out: DecodedMedia) -> Result<()>;
}

/// An autoregressive token model.
///
/// The batch is already in the layout the kernels read (the mechanism writes [`BatchRows`], the
/// executor uploads the [`TokenBatch`] views), so no method here converts or gathers on the host.
pub trait CausalLm {
    /// Write the embeddings of `ids` into `out` (`[tokens, hidden]`).
    fn embed_input_ids(&self, f: &Forward, ids: TensorView, out: TensorView) -> Result<()>;

    /// Run the decoder stack over `hidden` in place, reading and writing `cache`.
    fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        hidden: TensorView,
    ) -> Result<()>;

    /// Project the rows of `hidden` onto the vocabulary, writing the logits to `out`.
    ///
    /// `hidden` is the `[rows, hidden]` view of the rows named by `TokenBatch::logits_rows`; the
    /// executor gathered them. `out` is `[rows, vocabulary]`.
    fn logits(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()>;
}
