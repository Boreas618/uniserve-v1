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
use crate::tensor::{DType, Shape, TensorView};

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
    /// Query tokens packed into this batch, including the padding a bucket appends.
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

    /// Check the batch's own parts against each other: the shapes a layer reads by pointer.
    ///
    /// The four index tensors are read by arithmetic inside a kernel, so a batch whose slot mapping
    /// is one entry short, whose offsets end at another token count or whose `max_kv_len` is shorter
    /// than a page reads the wrong pages with nothing to complain. The layers cannot see the batch's
    /// intent — only its parts — so the check is here, it is shapes and types only (reading the
    /// values would need the device), and it is the one place an assembled batch is validated.
    ///
    /// `rows` is derived from `seq_lens`, so a batch that states the wrong number of sequences is
    /// caught by the two vectors that must agree with it rather than by a separate field.
    pub fn check(&self) -> Result<()> {
        if self.rows() == 0 || self.tokens() == 0 {
            return Err(Error::shape(
                "TokenBatch",
                "a batch of no rows or no tokens has nothing to run".to_owned(),
            ));
        }
        if self.page_size == 0 {
            return Err(Error::shape(
                "TokenBatch",
                "a page of zero tokens cannot be addressed".to_owned(),
            ));
        }
        if self.max_kv_len < self.page_size {
            return Err(Error::shape(
                "TokenBatch",
                format!(
                    "a key bound of {} is shorter than the {} tokens of a page",
                    self.max_kv_len, self.page_size
                ),
            ));
        }
        let rows = self.rows();
        let tokens = self.tokens();
        for (name, view, expected, dtype) in [
            ("positions", self.positions, tokens, DType::I32),
            ("slot_mapping", self.slot_mapping, tokens, DType::I32),
            ("seq_lens", self.seq_lens, rows, DType::I32),
            ("query_offsets", self.query_offsets, rows + 1, DType::I32),
        ] {
            if view.numel() != expected {
                return Err(Error::shape(
                    "TokenBatch",
                    format!(
                        "{name} is {} elements, {expected} expected for {rows} rows of {tokens} \
                         tokens",
                        view.numel()
                    ),
                ));
            }
            if view.dtype != dtype {
                return Err(Error::shape(
                    "TokenBatch",
                    format!("{name} is {}, not {dtype}", view.dtype),
                ));
            }
        }
        // Each row names at most one hidden row for its logits, and a row that names none is a row
        // the sampler does not see.
        if self.logits_rows.numel() > rows || self.logits_rows.dtype != DType::I32 {
            return Err(Error::shape(
                "TokenBatch",
                format!(
                    "logits_rows is {} {} elements for {rows} rows, not at most one i32 per row",
                    self.logits_rows.numel(),
                    self.logits_rows.dtype
                ),
            ));
        }
        if let Some(draft_lens) = self.draft_lens
            && (draft_lens.numel() != rows || draft_lens.dtype != DType::I32)
        {
            return Err(Error::shape(
                "TokenBatch",
                format!(
                    "draft_lens is {} {} elements for {rows} rows",
                    draft_lens.numel(),
                    draft_lens.dtype
                ),
            ));
        }
        Ok(())
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
    /// `[pages, 1, 1, 1]` **u8**, one flag per page of `key`.
    ///
    /// One byte, not four: the plane is the layout and the Storage Server carries one
    /// byte per page, so `cache_copy_region` refuses anything but `DType::U8`
    /// (`prefix_cache.rs`). An `i32` view of this plane writes four bytes where one
    /// belongs and clobbers three neighbouring pages' flags with zeros.
    pub key_initialized: TensorView,
    /// `[pages, 1, 1, 1]` **u8**, one flag per page of `value`; one byte for the same reason.
    pub value_initialized: TensorView,
    /// `[pages, 1, 1, 1]` f32: each page's key scale, present exactly when `key` is FP8.
    ///
    /// An FP8 cache stores a page divided by its own scale and the attention kernel multiplies it
    /// back in FP32. The scale is one *layer's*: two layers of one group keep their caches apart and
    /// each carries its own, which is why it is here and not with the group's geometry.
    pub key_scale: Option<TensorView>,
    /// `[pages, 1, 1, 1]` f32: each page's value scale, present exactly when `value` is FP8.
    pub value_scale: Option<TensorView>,
}

/// The cache of one attention group: how rows address pages and where the pages are.
#[derive(Clone, Copy)]
pub struct KvGroup<'a> {
    /// `[rows, max_pages]` i32: each row's page ids.
    pub block_tables: TensorView,
    /// Tokens per page; the same as the batch's `page_size`.
    pub page_size: u32,
    /// Element type of the group's pages: BF16, or FP8 for a group whose every layer is FP8.
    ///
    /// The scales are per layer ([`KvPlanes::key_scale`], [`KvPlanes::value_scale`]); a mixed group
    /// states its pages' type here and each layer states its own scales.
    pub dtype: DType,
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
    /// The layout the encode fills, as the executor got it from [`Encoder::layout`].
    ///
    /// The model reads the grids and counts from it: a device table cannot be read back, so without
    /// the layout a tower could not tell which of its per-layout rows belong to which image. It is
    /// a host value whose shape the encode was asked for, and it carries no request identity.
    pub layout: &'a EncoderLayout,
    /// The encoder's tables for this layout, in [`Encoder::state_tables`]' order.
    pub state: &'a [TensorView],
    /// The rows an encoder reports back for its caller's inspection, in the model's order.
    ///
    /// Diagnostics are outputs of the encode and never an input of the model's mathematics: a
    /// model that reports nothing passes an empty slice.
    pub diagnostics: &'a [TensorView],
}

/// One image a multimodal tower reads: its pixels.
///
/// The image's geometry and its tables are the **layout's**, not the image's: two images of one
/// grid need the same tap tables, and the executor cannot produce any of them. They are therefore
/// [`Encoder::state_tables`]' and reach the encode through [`EncoderInputs::state`], while this
/// value carries only what the caller decoded.
#[derive(Clone, Copy)]
pub struct ImageInput {
    /// `[tokens, pixels]` packed patches, or `[channels, height, width]` rasters.
    pub pixels: TensorView,
}

/// One image's patch grid, as a layout names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImageShape {
    /// The grid's three extents: time, rows, columns, in patches.
    pub grid: [u32; 3],
}

/// The prompt an encode's tables are derived from.
///
/// The tokens are host values, exactly as the request carried them: no ids are spliced, because the
/// splice is part of the layout's arithmetic and not of the input's transport.
#[derive(Clone, Copy)]
pub struct EncoderPrompt<'a> {
    /// The request's prompt token ids, in order.
    pub ids: &'a [u32],
}

/// The shape of one encode: the rows it fills and the grids it merges.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EncoderLayout {
    /// The packed rows the encode fills.
    pub rows: u32,
    /// The token rows that feed it.
    pub tokens: u32,
    /// The images it merges, in request order.
    pub images: Vec<ImageShape>,
}

/// What an encoder writes.
pub struct EncoderOutputs {
    /// `[tokens, width]` encoded hidden states.
    pub hidden: TensorView,
}

/// A video request's numerical layout input: what the denoiser must lay out (specs/worker.md 4.3).
///
/// Every field is a request fact the executor already holds — it decoded the media, encoded the
/// prompt and knows the canvas — and none is derivable from the others: the packed layout needs the
/// canvas, the roles' counts, the prompt's token count and each condition's raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VideoRequest<'a> {
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
    /// Prompt tokens the request presents, vision tokens included.
    ///
    /// The packed sequence's rotary clock starts at the prompt's length, so a layout cannot place a
    /// row without it.
    pub text_tokens: u32,
    /// The encoded condition media, in request order: anchors, then references.
    pub conditions: &'a [ConditionShape],
    /// Per-image prompt spans, in request order: the prompt rows a reference's own rows replace.
    ///
    /// Those rows take the image's rotary coordinates rather than the text clock's, so a layout
    /// that placed them on the prompt's clock would be silently wrong rather than refusing.
    pub vision_spans: &'a [(u32, u32)],
    /// The condition rows the request's layout holds, as one rung of the deployment's ladder;
    /// `None` is the deployment's whole capacity.
    ///
    /// A layout's rows are real rows of every projection, attention and feed-forward of every step,
    /// so the rung is what a step costs, and the conditions a request carries fill it from its
    /// start. The caller chooses it, because a warm request carries no conditions to count.
    pub condition_rows: Option<u32>,
}

/// One condition's raster, as a layout needs it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConditionShape {
    /// How the condition relates to the generated video.
    pub role: ConditionRole,
    /// Encoded frames; an image is one frame, and audio alone has none.
    pub frames: u32,
    /// Raster rows; zero without pixels.
    pub height: u32,
    /// Raster columns; zero without pixels.
    pub width: u32,
    /// Samples of the encoded track: a reference's soundtrack, zero without one.
    pub audio_samples: u32,
}

/// How a conditioning input relates to the generated video.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConditionRole {
    /// Anchors the first generated frame.
    FirstFrame,
    /// Anchors the last generated frame.
    LastFrame,
    /// Conditions the generation as a whole.
    Reference,
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
///
/// Ordered and hashed as a plain value of numbers: it is a bucket's extent (`executor::plan::Extent`),
/// and a captured graph is keyed by the bucket it holds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Layout {
    /// The video latent tensor.
    pub video: LatentShape,
    /// The audio latent tensor, when the task produces audio.
    pub audio: Option<LatentShape>,
    /// Text conditioning rows the layout reserves.
    pub text_rows: u32,
    /// Condition rows of the reference media.
    pub condition_rows: u32,
    /// Elements of every condition's initial-noise draw, in request order.
    ///
    /// The reference draws one chained stream — conditions, then video, then audio — and a model
    /// can derive the video and audio counts from [`Self::video`] and [`Self::audio`] but not the
    /// conditions': those are drawn at each condition's own unpacked latent shape before any
    /// packing, which is not what [`Self::condition_rows`] counts. Zero for a request with no
    /// conditions.
    pub condition_elements: u32,
}

/// The extents of one latent tensor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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

/// A table a model needs on the device but does not own.
///
/// The executor allocates; the model says what it is and fills it. Four kinds exist, they differ
/// in **who writes them, when, and how many times**, and that is the whole of the protocol:
///
/// - **constants** ([`Denoiser::constant_tables`]) are device memory written by
///   [`Denoiser::prepare_constants`], one forward *after* loading and *before* any capture. They
///   hold values that never change for a model — an AdaLN cache folded once from the weights — so
///   they are filled once per load and read by every capture. The forward is capture-safe: it reads
///   nothing back and synchronizes nothing.
/// - **layout** ([`Denoiser::layout_tables`]) is per *layout* and needs no request: the video VAE's
///   rotary positions of its raster, the packed row of each clip's latent voxel. These are the
///   model's mathematical layout, which the executor cannot compute, and they are the same for
///   every request of one shape — so they are filled once per layout
///   ([`Denoiser::prepare_layout`], host bytes only) and shared.
/// - **state** ([`Denoiser::state_tables`]) is per *request* of a layout: a table derived from the
///   request itself (its prompt clock, its spans, its mask) is one table per request.
///   [`Denoiser::prepare_state`] writes **host bytes only**; the executor allocates each table and
///   uploads those bytes. The list is deterministic for a layout, and the executor keys its storage
///   by `(layout, name)` beside the layout constants'.
/// - **evaluation** ([`Denoiser::prepare_evaluation`]) is per *evaluation index*: one solver
///   coefficient set, again host bytes the executor uploads, so a request's schedule is chosen
///   without capturing a graph per step.
///
/// No `prepare_*` allocates, takes an allocator, synchronizes or reads anything back; allocation is
/// the executor's and uploads are the executor's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StateTable {
    /// The table's name: unique within its kind and layout, and what the executor keys storage by.
    pub name: &'static str,
    /// The element type the model writes and the forward reads.
    pub dtype: DType,
    /// The extents.
    pub shape: Shape,
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

/// Where a piece of raw media anchors in the generated clip.
///
/// The reference's three roles, in the order the packed conditioning writes them: first-frame
/// anchors, last-frame anchors, then references.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MediaRole {
    /// Anchors the first generated frame.
    FirstFrame,
    /// Anchors the last generated frame.
    LastFrame,
    /// A reference the request describes.
    Reference,
}

/// One piece of raw media a request's conditions are built from.
///
/// This is what [`Denoiser::prepare_conditioning`] encodes when the model owns its condition
/// encoders: pixels before they are latents, samples before they are rows. A task with no conditions
/// passes an empty slice.
#[derive(Clone, Copy)]
pub struct MediaInput {
    /// Where the piece anchors, which decides where its rows sit in the packed conditioning.
    pub role: MediaRole,
    /// The packed rows the layout reserves for the piece's visual condition, as the request's own
    /// layout counted them: one row per `1×2×2` patch of its latent baked.
    ///
    /// The conditioner checks the encoder's output against this rather than recomputing it. The
    /// packing that the request was laid out under is the one statement of the arithmetic; a second
    /// statement in the encoder's caller is how two counts of one thing drift apart.
    pub rows: u32,
    /// The packed rows the layout reserves for the piece's soundtrack, in the same sense.
    pub audio_rows: u32,
    /// `[channels, frames, height, width]` pixels, in the encoder's own channel order and
    /// normalisation, when the condition is visual. One frame for a still, which is one latent
    /// frame.
    pub pixels: Option<TensorView>,
    /// Samples in the encoder's shape, when the condition carries a soundtrack.
    pub audio: Option<TensorView>,
    /// The condition's anchoring draw: `[channels, frames, height, width]` at the piece's own latent
    /// raster, FP32, when the piece is visual.
    ///
    /// The runner owns it, as it owns every draw of the request's noise stream - and it is drawn at
    /// latent resolution, so the conditioner patchifies it rather than reading rows.
    pub noise: Option<TensorView>,
}

/// One piece of a request's conditions, as the packed conditioning holds it.
///
/// A request states its conditions in its own order; the model packs them in another (keyframes
/// first), and only the model knows how many rows each takes and where its draw sits in the
/// request's noise stream. So the runner asks for this list and builds its [`MediaInput`]s in this
/// order, rather than deriving either fact beside the packing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConditionPiece {
    /// The condition's place in the request, which is how the raw media is addressed.
    pub condition: u32,
    /// Where the piece anchors.
    pub role: MediaRole,
    /// The packed rows of its visual condition; zero without pixels.
    pub rows: u32,
    /// The packed rows of its soundtrack; zero without one.
    pub audio_rows: u32,
    /// The extents of its [`MediaInput::pixels`], `None` for a piece with no pixels.
    pub pixels: Option<[u64; 5]>,
    /// The extents of its [`MediaInput::audio`], `None` for a piece with no soundtrack.
    pub track: Option<[u64; 3]>,
    /// The shape of its anchoring draw, `None` for a piece with no pixels: an audio reference is
    /// conditioned at its posterior's mean and draws nothing.
    pub draw: Option<LatentShape>,
    /// The first element of the draw in the request's noise stream.
    pub draw_offset: u64,
}

/// How a model's encoders read the media a request's conditions decode to, on the host.
///
/// The media reader decodes bytes into RGB frames and stereo PCM, which are the media's own forms.
/// What an encoder reads is the model's arithmetic: its channel order and normalisation, its patch
/// rows, the padding of a track. The executor owns the decoding and the buffers and asks the model
/// for these conversions, so no model constant reaches it.
pub trait MediaFormat: Send + Sync {
    /// Writes `[frames, height, width, 3]` RGB bytes as `out`, the pixels
    /// [`MediaInput::pixels`] carries: `3 * frames * height * width` elements.
    fn condition_pixels(
        &self,
        rgb: &[u8],
        frames: u32,
        height: u32,
        width: u32,
        out: &mut [f32],
    ) -> Result<()>;

    /// The element type of a vision tower's patch rows.
    fn patch_dtype(&self) -> DType;

    /// The elements of one patch row.
    fn patch_width(&self) -> u32;

    /// The pixels one patch spans along each side of a frame.
    ///
    /// The frames [`Self::patches`] reads are resized to `grid` times this.
    fn patch_side(&self) -> u32;

    /// Writes `[frames, 3, height, width]` planar RGB bytes, already at the grid's raster, as `out`:
    /// the `grid[0] * grid[1] * grid[2]` patch rows [`ImageInput::pixels`] carries, in
    /// [`Self::patch_dtype`].
    fn patches(&self, planes: &[u8], frames: u32, grid: [u32; 3], out: &mut [u8]) -> Result<()>;

    /// The samples an encoder reads of a track of `samples` samples, padding included.
    fn track_samples(&self, samples: u64) -> Result<u64>;

    /// Writes `[samples, 2]` interleaved stereo as `out`, the view [`MediaInput::audio`] carries:
    /// `2 * self.track_samples(samples)` elements.
    fn track(&self, pcm: &[f32], samples: u64, out: &mut [f32]) -> Result<()>;
}

/// What a denoiser conditions on at a step.
///
/// The rows of every view follow the order the layout reserves: the encoded text first, then the
/// request's conditions in the order `VideoRequest` counts them (first-frame anchors, last-frame
/// anchors, references).
///
/// The one request-shaped thing that crosses here is [`Conditioning::media`], the raw media a
/// conditioner encodes. It has to: [`Denoiser::prepare_conditioning`] runs once per request, before
/// any capture, where no request reaches it, and condition rows cannot be built from views that are
/// already latents. Everything else a request carries stays out.
#[derive(Clone, Copy)]
pub struct Conditioning<'a> {
    /// `[rows, width]` encoded text.
    pub text: TensorView,
    /// `[rows, width]` the text encoder's own features, when the model has a conditioner between the
    /// encoder and the denoiser: [`Denoiser::prepare_conditioning`] folds this into `text`.
    ///
    /// Both buffers are the runner's, sized to the request's text capacity, and the conditioner runs
    /// in place over `text`. `None` when the model has no conditioner, or for a task whose
    /// conditioning is nothing but the encoder's features.
    pub source: Option<TensorView>,
    /// The raw media the request's conditions are built from, in the request's own order.
    ///
    /// [`Denoiser::prepare_conditioning`] encodes these and writes the condition rows of `text`;
    /// empty when the task has no conditions, which is every `t2va` request. The runner owns the
    /// buffers and the encoders' arena is its own forward, reset after the call.
    pub media: &'a [MediaInput],
    /// `[rows, …]` encoded reference media, when the task has references.
    pub images: Option<TensorView>,
    /// `[rows, …]` encoded reference audio, when the task has audio references.
    pub audio: Option<TensorView>,
    /// The tables [`Denoiser::state_tables`] named for this layout, in that order.
    pub state: &'a [TensorView],
    /// `[1]` i64: the step the schedule evaluates, when the model selects coefficients on the
    /// device. Absent when the model's coefficients arrive as launch scalars.
    pub step: Option<TensorView>,
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

/// One clip's decoded raster extent, as the runtime states it.
///
/// Frames, rows and columns of the **decoded** media, which is what a runtime bucket is keyed by.
/// A decoder's tables and its arena are keyed by the latents the raster decodes from, and the
/// ratios between the two are the checkpoint's - so the map is the decoder's to state and the
/// runtime never learns it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Clip {
    /// Frames of the decoded raster.
    pub frames: u32,
    /// Rows of one decoded frame.
    pub height: u32,
    /// Columns of one decoded frame.
    pub width: u32,
}

/// What a decoder writes: the decoded video frames and audio samples.
#[derive(Clone, Copy)]
pub struct DecodedMedia {
    /// `[channels, frames, height, width]` decoded frames, in the decoder's own channel order and
    /// element type.
    ///
    /// Channels first, as the media this boundary carries is stated everywhere else: the latents, the
    /// conditioning's pixels and its anchoring draw are all `[channels, frames, height, width]`, and
    /// the layers that place a decoded tile - the unpatchify, the seam blends and the byte step - read
    /// a planar raster. A decoder writes this layout and does not transpose; the caller that needs
    /// `[frames, height, width, channels]` for a file converts once, where it writes.
    pub video: Option<TensorView>,
    /// `[samples, 2]` interleaved stereo PCM, `i16`, at the decoder's own sample rate.
    ///
    /// Two channels, interleaved: one row per sample frame, the left channel's value before the
    /// right's. The wire format and the file format both want that order, so the decoder writes it and
    /// no caller transposes. The channel count is fixed - `output::Pcm::CHANNELS` is two - because this
    /// checkpoint's soundtrack is stereo.
    pub audio: Option<TensorView>,
}

/// A tower that turns inputs into conditioning (a text encoder, a vision tower, a video encoder).
///
/// The protocol is the denoiser's, in the denoiser's words: the model says what shape an encode
/// has, which tables that shape needs and writes their host bytes; the executor allocates and
/// uploads them. A tower's per-grid tables (its rotary clocks, its tap tables, the prompt rows its
/// images replace) are mathematics the executor cannot reproduce, and they are the same for every
/// request of one shape.
pub trait Encoder {
    /// The shape an encode of `batch`'s rows and `images`' grids has.
    fn layout(&self, batch: &TokenBatch, images: &[ImageShape]) -> EncoderLayout;

    /// The tables an encode of `layout` reads, in [`Self::prepare_state`]'s order.
    ///
    /// Deterministic for a layout, so the executor keys storage by `(layout, name)` and shares one
    /// set between every request of that shape.
    fn state_tables(&self, layout: &EncoderLayout) -> Vec<StateTable>;

    /// Write the host bytes of each table of `layout`, in [`Self::state_tables`]' order.
    ///
    /// `prompt` is the request's own tokens, on the host and unspliced: two of a tower's tables (the
    /// interleaved M-RoPE positions and the prompt rows its images replace) are arithmetic over the
    /// token ids, and that arithmetic belongs to the model rather than to the runner. The ids are
    /// deliberately **not** part of the layout: every table's shape follows from the layout's
    /// counts, only the bytes follow from the ids, so two prompts of one shape share one capture.
    ///
    /// `out[i]` is `state_tables(layout)[i]`'s bytes, row-major in its element type, already the
    /// right length; the executor uploads it. No allocation, no synchronization, no read-back.
    fn prepare_state(
        &self,
        prompt: &EncoderPrompt<'_>,
        layout: &EncoderLayout,
        out: &mut [Vec<u8>],
    ) -> Result<()>;

    /// Encode `inputs` into `out`.
    fn encode(&self, f: &Forward, inputs: EncoderInputs<'_>, out: EncoderOutputs) -> Result<()>;
}

/// A diffusion model that denoises latents step by step.
pub trait Denoiser {
    /// The latent layout this request needs.
    fn layout(&self, request: &VideoRequest<'_>) -> Layout;

    /// The bytes this request's forward needs, as the model's own plan computes them.
    ///
    /// This is the one number a caller cannot derive and must not transcribe: it is the total of
    /// the model's plan for the layout this request names, so a caller allocating from a literal
    /// beside the plan holds a second statement of a value only the model owns. The plan is derived
    /// here from `request` in the order the executor already uses - [`layout`](Self::layout) and
    /// then the model's own plan - so a caller never has two ways to name the same plan.
    ///
    /// It is a pure function of the request: two requests with one layout answer the same bytes,
    /// because the executor caches a capture by bucket and a plan that varied inside a layout would
    /// be an allocation that fits one request and not the next. It allocates nothing and touches no
    /// device: the model states what its forward needs and the runtime owner allocates to it.
    ///
    /// A request outside the deployment's bound - a canvas it does not serve, a row count past the
    /// packed ladder - is refused by name, never answered with a number for a layout the forward
    /// cannot run. A denoiser that states no plan refuses, as it does for a schedule it does not
    /// have: a zero would be the same value as an arena of no bytes.
    fn arena_bytes(&self, request: &VideoRequest<'_>) -> Result<u64> {
        Err(Error::Unsupported(format!(
            "a denoiser that states no arena plan for a layout of {:?}",
            self.layout(request)
        )))
    }

    /// The denoising steps one request of `task` runs, in order.
    ///
    /// The runner executes one step per call and cannot derive the schedule: the steer belongs to
    /// the model's checkpoint, and `Step::index` and `Step::count` are what a solver checks itself
    /// against, so a runner holding its own sigma table would be a second source of truth for a
    /// value the model states.
    ///
    /// The task is the key rather than a positional index because a request names its task: a
    /// vector of schedules whose first element is the one a request starts with says which
    /// schedule to take by position, and a second modality inserted at the front changes every
    /// caller silently.
    ///
    /// A model that states no schedule for `task` refuses by name. Empty would be the same value
    /// as a schedule that exists and has no steps, which is the opposite fact.
    fn schedule(&self, task: VideoTask) -> Result<Vec<Step>> {
        Err(Error::Unsupported(format!(
            "a denoiser that states no schedule for {task:?}"
        )))
    }

    /// The width of [`Conditioning::text`]'s rows, when it is not the encoder's.
    ///
    /// `text` is what the denoiser reads and `source` is what the text encoder wrote, and a model
    /// whose conditioner **widens** its input has two widths: H3's refiner projects the encoder's
    /// 5120 columns into the stack's 5376, so a runner that sized both buffers from the encoder's
    /// width hands the conditioner one too narrow for its own output - which no launch refuses until
    /// the packed rows are gathered, one operator later, as `out is [rows, 5376], [rows, 5120]
    /// expected`.
    ///
    /// `None` means one width: no conditioner, or a conditioner that keeps the encoder's.
    fn text_width(&self) -> Option<u32> {
        None
    }

    /// Draw the initial noise of `out` from `seed`, addressed within `layout`.
    ///
    /// The layout is a parameter because the draw's element offsets are functions of the request's
    /// condition noise shapes, which only a layout names: a model that had to remember a
    /// layout-to-offsets map would hold mutable request state, and the seed addressing is a pure
    /// function of `(seed, layout, element index)`.
    fn draw_noise(&self, f: &Forward, seed: Seed, layout: Layout, out: Latents) -> Result<()>;

    /// The pieces of `request`'s conditions, in the order [`Conditioning::media`] holds them.
    ///
    /// A request with no conditions has none. A model that takes conditions states them here, and
    /// one that takes none refuses a request that carries some, by name, rather than answering an
    /// empty list that would condition the clip on nothing.
    fn condition_pieces(&self, request: &VideoRequest<'_>) -> Result<Vec<ConditionPiece>> {
        if request.conditions.is_empty() {
            return Ok(Vec::new());
        }
        Err(Error::Unsupported(format!(
            "a denoiser that takes no conditions, asked for {}",
            request.conditions.len()
        )))
    }

    /// Draw `piece`'s anchoring noise into `out`, at its place in the request's stream.
    ///
    /// The conditions' draws lead the stream [`Self::draw_noise`] continues, so one seed names both.
    fn draw_condition_noise(
        &self,
        f: &Forward,
        seed: Seed,
        piece: &ConditionPiece,
        out: TensorView,
    ) -> Result<()> {
        let _ = (f, seed, out);
        Err(Error::Unsupported(format!(
            "a denoiser that draws no condition noise, asked for condition {}",
            piece.condition
        )))
    }

    /// Fold `cond` into the latents the schedule starts from.
    fn prepare_latents<'a>(&self, f: &Forward, cond: Conditioning<'a>, out: Latents) -> Result<()>;

    /// Fold the conditioner's output into `cond.text`, before the request's first evaluation.
    ///
    /// The executor calls this once per request, before it captures anything and outside a captured
    /// replay: the launches are captured with everything else, so the forward must be capture-safe -
    /// no host read, no synchronization, no allocation. `cond.source` is the buffer the conditioner
    /// reads; `cond.text` is written in place over its own rows.
    ///
    /// Prepares what the denoiser reads from a request's conditioning, in place.
    ///
    /// Required, not defaulted: a model whose text encoder's features are everything its denoiser
    /// reads has no conditioner and says so with an explicit `Ok(())`, while a model that has one
    /// must write it here. A no-op default would make forgetting the call — or forgetting the
    /// override — succeed silently, and a model whose conditioning is prepared nowhere generates a
    /// plausible clip from zeros; the absence of this method's work is not observable in the output.
    fn prepare_conditioning(&self, f: &Forward, cond: Conditioning<'_>) -> Result<()>;

    /// One denoising step: update `latents` in place with the model's velocity.
    fn denoise<'a>(
        &self,
        f: &Forward,
        step: Step,
        latents: Latents,
        cond: Conditioning<'a>,
    ) -> Result<()>;

    /// How many solver evaluations one request makes: the coefficient sets
    /// [`Self::prepare_evaluation`] can be asked for.
    fn evaluations(&self) -> u32;

    /// The tables that never change for this model, in the order [`Self::prepare_constants`] fills
    /// them.
    fn constant_tables(&self) -> Vec<StateTable>;

    /// Fill `out` with the constants, in [`Self::constant_tables`]' order.
    ///
    /// The executor calls this once after loading and before it captures anything: the launches are
    /// captured with everything else, so the forward must be capture-safe — no host read, no
    /// synchronization, no allocation.
    fn prepare_constants(&self, f: &Forward, out: &[TensorView]) -> Result<()>;

    /// The constants a layout needs, in the order [`Self::prepare_layout`] fills them.
    ///
    /// These are the layout's alone: the video VAE's per-voxel rotary positions, the packed row of
    /// each clip's voxel. They are filled once per layout, not once per request, so the executor
    /// keys its storage by `(layout, name)` beside the request's state and shares one set between
    /// every request of that shape.
    fn layout_tables(&self, layout: &Layout) -> Result<Vec<StateTable>>;

    /// Write the host bytes of each layout constant of `layout`, in [`Self::layout_tables`]' order.
    ///
    /// Same rule as [`Self::prepare_state`]: host bytes only, already the right length, uploaded by
    /// the executor, and deterministic for a layout so two requests of one shape agree.
    fn prepare_layout(&self, layout: &Layout, out: &mut [Vec<u8>]) -> Result<()>;

    /// The tables a request of `layout` needs, in the order [`Self::prepare_state`] fills them.
    ///
    /// Deterministic for a layout: the executor keys its storage by `(layout, name)`, so two
    /// requests of the same shape share one set.
    fn state_tables(&self, layout: &Layout) -> Vec<StateTable>;

    /// Write the host bytes of each table of `layout`, in [`Self::state_tables`]' order.
    ///
    /// `out[i]` is `state_tables(layout)[i]`'s bytes, row-major in its element type, and is already
    /// the right length; the executor uploads it.
    fn prepare_state(
        &self,
        request: &VideoRequest<'_>,
        layout: &Layout,
        out: &mut [Vec<u8>],
    ) -> Result<()>;

    /// Write the host bytes of evaluation `index`'s coefficient tables, in
    /// [`Self::constant_tables`]' layout order.
    fn prepare_evaluation(&self, index: u32, out: &mut [Vec<u8>]) -> Result<()>;
}

/// One clip's place on a timeline, as the runner states it.
///
/// A decoder decodes one clip and stops; the clip's frames become the request's video only once they are
/// placed, faded where they overlap the clip before them. These three numbers are what the runner knows
/// and the model does not: how many of the clip's frames the timeline keeps, how many of those fade against
/// the predecessor, and where the first kept frame goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Splice {
    /// The frames of the clip's own decode the timeline keeps, counted from the clip's first frame.
    ///
    /// A clip a successor follows keeps the frames before its tail; the clip that ends the timeline keeps
    /// its tail too, so this is the larger number for the last clip.
    pub keep: u32,
    /// How many of the kept frames fade against the predecessor, counted at the end of `keep`.
    ///
    /// Zero for the timeline's first clip, which has nothing to fade against.
    pub overlap: u32,
    /// The destination frame index of the clip's first kept frame.
    pub first: u32,
}

/// A decoder that turns latents back into media.
pub trait Decoder {
    /// The tables a decode of `layout` reads, in [`Self::prepare_layout`]'s order.
    ///
    /// Every decoder table is layout-keyed: the copy blocks' offsets, the per-voxel rotary
    /// positions and the packed row of each clip's voxel all follow from the layout's counts, not
    /// from a request's seed or schedule. Deterministic for a layout, so the executor keys storage
    /// by `(layout, name)` and shares one set between every request of that shape.
    ///
    /// A layout the block cannot serve is a **named error**, not an empty list: a block with no
    /// tables and a block that cannot answer are different facts, and only one of them is the
    /// caller's to act on. Returning `Ok(vec![])` for a contradictory layout is the shape a silent
    /// failure takes - the runner uploads nothing, the decode reads nothing, and nothing said so.
    fn layout_tables(&self, layout: &Layout) -> Result<Vec<StateTable>>;

    /// Write the host bytes of each table of `layout`, in [`Self::layout_tables`]' order.
    ///
    /// `out[i]` is `layout_tables(layout)[i]`'s bytes, row-major in its element type, and is
    /// already the right length; the executor uploads it. Host bytes only: no allocation, no
    /// synchronization and no read-back, so a decode captured after this call is capture-safe.
    fn prepare_layout(&self, layout: &Layout, out: &mut [Vec<u8>]) -> Result<()>;

    /// The element type this decoder reads its latents in and writes its decoded media in.
    ///
    /// The runner allocates the latents column and the raster column before a graph exists, and a
    /// buffer's width in bytes is irreversible into an element type: a `[voxels, channels]` column
    /// of `u16` and of `bf16` are the same bytes. The design's `dtype = "auto"` reads this from the
    /// checkpoint rather than choosing it, so the model states it here and no runner infers it from
    /// a byte count or from its deployment's configuration.
    ///
    /// Required rather than defaulted: a decoder that does not say has no honest answer to give, and
    /// the point of this boundary is that the alternative - guessing - is silent.
    fn element_type(&self) -> DType;

    /// The element type of the latent block [`Self::decode`] reads, when it is not
    /// [`Self::element_type`].
    ///
    /// A decoder whose reference restores the latents in a wider type than it decodes them to says
    /// so here, and the runner stages the block in that type rather than narrowing it on the way in.
    fn latent_type(&self) -> DType {
        self.element_type()
    }

    /// Write the latent block of the clip starting at latent frame `first`, on the forward's stream.
    ///
    /// `trajectory` is the timeline's latents as a denoiser publishes them, `[channels, voxels]`, and
    /// `block` is the buffer [`Self::decode`] reads, `[rows, channels]` in [`Self::latent_rows`]'
    /// order. A decoder that reads a trajectory says how its rows lie in it; the runner stages the
    /// trajectory and asks, so a clip never crosses to the host.
    fn gather_latents(
        &self,
        f: &crate::Forward,
        layout: &Layout,
        trajectory: TensorView,
        first: u32,
        block: TensorView,
    ) -> Result<()> {
        let _ = (f, layout, trajectory, first, block);
        Err(Error::Unsupported(
            "this decoder reads no denoiser trajectory".to_owned(),
        ))
    }

    /// For each row of the latent block [`Self::decode`] reads, the clip's voxel it holds, counted
    /// frame by frame, row by row.
    ///
    /// `None` reads each voxel once, in that order. A decoder that works in overlapping tiles lists
    /// its tiles' voxels instead, so a voxel two tiles share has two rows and the block is longer
    /// than the clip.
    fn latent_rows(&self, layout: &Layout) -> Result<Option<Vec<u32>>> {
        let _ = layout;
        Ok(None)
    }

    /// The latent frame `clip`, counted from the timeline's first clip, starts at.
    ///
    /// A clip reads a window of the trajectory, and clips overlap by the frames the next one also
    /// reads, so where a window starts is the decoder's arithmetic. A timeline of one clip starts at
    /// zero.
    fn clip_start(&self, clip: u32) -> u32 {
        let _ = clip;
        0
    }

    /// The latent layout one clip's raster decodes from.
    ///
    /// [`Clip`] is the runtime's own extent - frames, rows and columns of the decoded raster - and
    /// this decoder's tables, scratch and arena are keyed by its latents. The ratios between them
    /// belong to the checkpoint, so the decoder states the map: a runtime that computed it would be
    /// holding model knowledge, and the first deployment with another temporal or spatial
    /// compression would find it in two places.
    ///
    /// A decoder whose decoded raster is not its latent layout scaled by ratios it knows refuses by
    /// name, the way [`Decoder::splice`] does, rather than answering with a layout that reads as
    /// one.
    fn layout_of(&self, clip: Clip) -> Result<Layout> {
        let _ = clip;
        Err(Error::shape(
            "Decoder",
            "this decoder states no latent layout for a clip's raster".to_owned(),
        ))
    }

    /// The bytes one decode of `layout` needs, as the model's own plan computes them.
    ///
    /// The same contract as [`Denoiser::arena_bytes`], for the subject a decoder already has: it is
    /// the total of the model's plan for the layout, a pure function of it - two clips of one layout
    /// answer the same bytes, because the executor keys a capture and its arena by layout - and it
    /// allocates nothing and touches no device.
    ///
    /// It takes a layout rather than a request because a decoder is handed a layout:
    /// [`layout_of`](Self::layout_of) is how its runner got one, and
    /// [`layout_tables`](Self::layout_tables) and [`prepare_layout`](Self::prepare_layout) take the
    /// same subject in the same form.
    ///
    /// A layout this block cannot serve - a raster it does not decode to - is refused by name, never
    /// answered with a number the decode would run past. A decoder that states no plan refuses, as it
    /// does for a layout it cannot map: a zero would be the same value as an arena of no bytes.
    fn arena_bytes(&self, layout: &Layout) -> Result<u64> {
        // `Unsupported`, as [`Denoiser::arena_bytes`] answers, and not a shape error: a caller must
        // be able to tell a block that states no plan from one that refuses the layout it was
        // given. The first is a block the seam does not cover yet; the second is a deployment whose
        // layout and block disagree, and a runner that swallowed both would report the second as the
        // first.
        Err(Error::Unsupported(format!(
            "this decoder states no arena plan for a layout of {:?}",
            layout.video
        )))
    }

    /// The interleaved PCM window one decode of `audio` writes: sample frames and channels.
    ///
    /// **A track's window is the model's to state, and the runner sizes a column from it.** The
    /// window is the vocoder's latent steps multiplied by the product of its upsampling rates,
    /// which is checkpoint arithmetic: a runner that computed it would hold model knowledge, and
    /// the first deployment with another rate would find it stated twice.
    ///
    /// Required rather than defaulted to a shape, for the reason [`element_type`](Self::element_type)
    /// is: a decoder that does not say has no honest answer, and the alternative - guessing a window
    /// - is silent, since a short window is a track that ends early and a long one a tail of garbage.
    fn pcm_window(&self, audio: &LatentShape) -> Result<(u64, u32)> {
        let _ = audio;
        Err(Error::Unsupported(
            "this decoder writes no interleaved PCM".to_owned(),
        ))
    }

    /// Decode `latents` into `out`, reading the layout's tables in order.
    fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        state: &[TensorView],
        out: DecodedMedia,
    ) -> Result<()>;

    /// Place one decoded clip on the timeline, fading it against its predecessor.
    ///
    /// `clip` is the clip's decoded raster; `previous` is the predecessor's own decode of the frames the
    /// two share, absent for the timeline's first clip; `out.video` is the whole timeline's raster, which
    /// the splice writes at `splice.first`. A decoder that decodes one clip and knows nothing of timelines
    /// leaves this default, which refuses by the block kind's name so a runner that splices with the wrong
    /// block hears which kind it called.
    fn splice(
        &self,
        f: &Forward,
        previous: Option<TensorView>,
        clip: TensorView,
        splice: Splice,
        out: DecodedMedia,
    ) -> Result<()> {
        let _ = (f, previous, clip, splice, out);
        Err(Error::shape(
            "Decoder",
            "this decoder splices no clips".to_owned(),
        ))
    }

    /// The splices a timeline of `frames` frames is written in, in the order the frames are decoded.
    ///
    /// A timeline is not one clip: the checkpoint decodes it clip by clip, each clip overlapping its
    /// predecessor by a fade, and the last one keeping its own tail. **Which frames each clip writes is
    /// the block's arithmetic, not the runner's** - the frame count a timeline may hold, the body each
    /// clip contributes and the overlap the fade reads all come from the checkpoint's decoding, so a
    /// runner that spliced a timeline asks this and passes each [`Splice`] to [`splice`](Self::splice).
    /// A splice's `first` is the destination frame in the raster handed to `splice`, so a runner that
    /// gives each clip its own raster states `first: 0`.
    ///
    /// The default refuses by the block kind's name for the reason `splice` does: a block that cannot
    /// write a timeline has no schedule to state.
    fn splices(&self, frames: u32) -> Result<Vec<Splice>> {
        Err(Error::shape(
            "Decoder",
            format!(
                "this decoder states no splice schedule, and a timeline of {frames} frame(s) asked for one"
            ),
        ))
    }
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

    /// Runs every expert layer of one step with no tokens of this rank's own.
    ///
    /// A rank of an expert group runs every step of the group, because its experts serve the tokens
    /// its peers route to them. A model without split experts has nothing to join.
    fn join_expert_step(&self, f: &Forward) -> Result<()> {
        let _ = f;
        Err(crate::Error::Unsupported(
            "this model splits no experts across ranks, so it has no expert step to join"
                .to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(channels: u32, frames: u32, height: u32, width: u32) -> LatentShape {
        LatentShape {
            channels,
            frames,
            height,
            width,
        }
    }

    /// "This block states no plan" is `Unsupported`, and a refusal about a layout is not.
    ///
    /// A runner passes over `Unsupported` - the pool is then the only figure there is - and fails on
    /// anything else, because a block that states a plan and refuses the layout it was handed is a
    /// deployment whose block and layout disagree. Both being one variant would make the second
    /// report as the first, which is the silent failure the arena plan exists to remove.
    #[test]
    fn a_block_that_states_no_arena_plan_says_unsupported() {
        struct Silent;

        impl Decoder for Silent {
            fn layout_tables(&self, _layout: &Layout) -> Result<Vec<StateTable>> {
                Ok(Vec::new())
            }

            fn prepare_layout(&self, _layout: &Layout, _out: &mut [Vec<u8>]) -> Result<()> {
                Ok(())
            }

            fn element_type(&self) -> DType {
                DType::F32
            }

            fn decode(
                &self,
                _f: &Forward,
                _latents: TensorView,
                _state: &[TensorView],
                _out: DecodedMedia,
            ) -> Result<()> {
                Ok(())
            }
        }

        let layout = Layout {
            video: latent(16, 2, 3, 4),
            audio: None,
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        };
        let error = Silent
            .arena_bytes(&layout)
            .expect_err("a block with no plan states nothing");
        assert!(
            matches!(error, Error::Unsupported(_)),
            "a block with no plan is Unsupported, not a shape error: {error:?}"
        );
    }

    /// The conditions' draw elements are the sum of their own unpacked latent shapes, which is
    /// what the chained noise stream counts — not the packed rows `condition_rows` names.
    #[test]
    fn a_layout_states_the_condition_elements_of_the_chained_stream() {
        let none = Layout {
            video: latent(16, 13, 24, 40),
            audio: Some(latent(8, 96, 1, 1)),
            text_rows: 512,
            condition_rows: 0,
            condition_elements: 0,
        };
        assert_eq!(none.condition_elements, 0);

        let conditions = [
            ConditionShape {
                role: ConditionRole::FirstFrame,
                frames: 1,
                height: 480,
                width: 832,
                audio_samples: 0,
            },
            ConditionShape {
                role: ConditionRole::Reference,
                frames: 5,
                height: 480,
                width: 832,
                audio_samples: 24_000,
            },
        ];
        // A packed latent's extents: 8× compression spatially, 4 temporally after the first frame.
        let elements: u32 = conditions
            .iter()
            .map(|condition| {
                let frames = condition.frames.div_ceil(4).max(1);
                16 * frames * (condition.height / 8) * (condition.width / 8)
            })
            .sum();
        let layout = Layout {
            video: none.video,
            audio: none.audio,
            text_rows: none.text_rows,
            condition_rows: 6144,
            condition_elements: elements,
        };
        assert_eq!(layout.condition_elements, elements);
        // The stream the draw walks: conditions, then video, then audio.
        let video_elements =
            layout.video.channels * layout.video.frames * layout.video.height * layout.video.width;
        let audio = layout.audio.unwrap();
        let audio_elements = audio.channels * audio.frames * audio.height * audio.width;
        assert_eq!(
            layout.condition_elements + video_elements + audio_elements,
            elements + 16 * 13 * 24 * 40 + 8 * 96
        );
    }
}

#[cfg(test)]
mod batch_tests {
    use ops::{DType, DevicePtr, Shape};

    use super::*;

    fn view(dtype: DType, shape: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x3_0000), dtype, Shape::new(shape))
    }

    /// A batch of `rows` sequences, one token each, over `page_size`-token pages.
    fn batch(rows: u64) -> TokenBatch {
        TokenBatch {
            positions: view(DType::I32, &[rows]),
            slot_mapping: view(DType::I32, &[rows]),
            seq_lens: view(DType::I32, &[rows]),
            query_offsets: view(DType::I32, &[rows + 1]),
            logits_rows: view(DType::I32, &[rows]),
            phase: PhaseKind::Decode,
            page_size: 64,
            num_tokens: u32::try_from(rows).unwrap(),
            max_kv_len: 4_096,
            draft_lens: None,
        }
    }

    #[test]
    fn a_batch_whose_parts_disagree_is_refused_by_the_part() {
        // The consistent one passes, or the rest of this test proves nothing.
        batch(15).check().expect("a consistent batch");

        // A slot short: the write would scatter 14 rows and address a fifteenth slot that is not
        // there — the exact mistake a hand-built example makes.
        let mut short = batch(15);
        short.slot_mapping = view(DType::I32, &[14]);
        let error = short.check().expect_err("a slot mapping of another length");
        assert!(format!("{error}").contains("slot_mapping"), "{error}");

        // Offsets that do not close the batch: one entry per sequence plus the end.
        let mut offsets = batch(15);
        offsets.query_offsets = view(DType::I32, &[15]);
        assert!(offsets.check().is_err());

        // A key bound shorter than a page addresses no page at all.
        let mut unpaged = batch(4);
        unpaged.max_kv_len = 32;
        let error = unpaged.check().expect_err("a bound below one page");
        assert!(format!("{error}").contains("shorter than"), "{error}");

        // More logits rows than rows would have the sampler read past the last row.
        let mut logits = batch(4);
        logits.logits_rows = view(DType::I32, &[8]);
        assert!(logits.check().is_err());

        // Draft counts are one per row, when a batch verifies drafts at all.
        let mut drafts = batch(4);
        drafts.draft_lens = Some(view(DType::I32, &[2]));
        assert!(drafts.check().is_err());

        // And an empty batch is not a batch.
        let mut empty = batch(4);
        empty.num_tokens = 0;
        assert!(empty.check().is_err());
    }
}
