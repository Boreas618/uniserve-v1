//! The capability traits a model implements and execution calls, and the input values of their
//! methods.
//!
//! Execution dispatches by capability, never by concrete model: a runner owns a
//! `dyn Encoder`, `dyn Denoiser`, `dyn Decoder` or `dyn CausalLm` and calls only the methods
//! declared here. Every method takes and returns tensors and small plain values: no request
//! identity, no sampling state, no pool slot, no context and no stream crosses this boundary
//! (`AGENTS.md`, Model Library Boundary). The execution context a method computes under is the
//! thread's, installed by execution around the call.
//!
//! ## What a method takes
//!
//! - **Inputs** are tensors execution lends: a batch's index columns ([`TokenBatch`]), token
//!   ids, hidden rows, media. They are external tensors over a runner's buffers, at the same
//!   addresses from a graph's capture to each of its replays.
//! - **Results** are returned: the hidden rows and the logits of a causal language model are
//!   tensors of the context's allocator, valid until the next forward on the context.
//! - **Persistent state** is a destination: key/value pages ([`KvCache`]), recurrent state
//!   ([`RecurrentState`]), a request's latents, and the product buffers a later stage of the
//!   pipeline reads (an encoder's rows, a decoder's raster and waveform, an image's features)
//!   are execution's memory, lent as external tensors that the model writes in place. These are
//!   the only tensors a method writes without having made them.
//! - **Host values** a forward needs beyond its tensors (positions, packing indices, schedule
//!   coefficients) are computed inside the method as CPU tensors from the plain values it is
//!   given and moved to the device with `to_device`; their shapes follow from the bucket's
//!   extents alone, so every request of one shape replays one captured graph.
//!
//! ## Index tensors
//!
//! Index tensors are `U32`: token ids, cache slots, sequence lengths and offsets, logits rows,
//! page tables, expert ids. Where a column has a row that addresses nothing (a padding token's
//! cache slot, a padding row's request slot), that row holds [`NO_SLOT`]. Rotary positions are
//! coordinates, not indices, and are `I32`.

use proto::module::PhaseKind;
use proto::video::{ConditionRole, LatentShape, Layout};
use tensor::{DType, Shape, Tensor};

use crate::error::{Error, Result};

/// The entry of a slot column that addresses no slot: a padding token writes no cache row, and a
/// padding row reads and writes no recurrent state.
pub const NO_SLOT: u32 = u32::MAX;

/// The index columns of one batch, as a model's modules read them.
///
/// Every tensor is a column the scheduler's batch stated and execution placed on the device; a
/// model never converts a host value into one of them. They are laid out exactly as the
/// attention and KV-write kernels read them, and live as long as the runner's batch buffers, so
/// a captured graph addresses them unchanged from capture to replay.
#[derive(Clone)]
pub struct TokenBatch {
    /// `[tokens]` I32 rotary positions.
    pub positions: Tensor,
    /// `[tokens]` U32 cache slots; [`NO_SLOT`] writes nothing and initializes nothing.
    pub slot_mapping: Tensor,
    /// `[rows]` U32 keys visible to each row's last query.
    pub seq_lens: Tensor,
    /// `[rows + 1]` U32 packed query offsets.
    pub query_offsets: Tensor,
    /// `[rows]` U32 hidden rows whose logits are produced.
    pub logits_rows: Tensor,
    /// Which phase this batch is.
    pub phase: PhaseKind,
    /// Tokens per cache page.
    pub page_size: u32,
    /// Query tokens packed into this batch, including the padding a bucket appends.
    pub num_tokens: u32,
    /// The bound on keys any row of this batch reads: the attention descriptor's reduction
    /// length.
    pub max_kv_len: u32,
    /// `[rows]` U32 drafted-token counts, when the batch verifies speculative drafts.
    pub draft_lens: Option<Tensor>,
    /// `[axes, tokens]` I32 multi-axis rotary coordinates, axis-major, when the batch holds the
    /// rows of an image ([`Vision`]); `None` when every row is text, whose coordinate on every
    /// axis is its entry of `positions`.
    pub rope_positions: Option<Tensor>,
    /// `[1]` U32: the token rows of this batch that are live, when execution states them.
    ///
    /// A captured bucket is wider than the batch that fills it. A module that partitions work by
    /// token row needs to know which rows are real, and it cannot derive that from any shape.
    /// The value is on the device because it changes per replay of one captured graph.
    pub live_tokens: Option<Tensor>,
    /// `[tokens]` U32: the token id of each packed token, as the request's prompt states it (a
    /// vision row holds its placeholder's id), for a model whose layers read token ids beside
    /// the embeddings: an n-gram hash embedding hashes each token with the two before it.
    /// `None` for every other model.
    pub token_ids: Option<Tensor>,
    /// `[rows, 2]` U32: the two token ids before each row's first token, oldest first, with
    /// the vocabulary's end-of-text id where the request has no earlier token. Present exactly
    /// when `token_ids` is: a row's first tokens hash with ids the batch does not hold, which
    /// execution reads from the request's token ids as it reads positions. It is an input,
    /// not state: nothing is kept, checkpointed or committed for it.
    pub ngram_context: Option<Tensor>,
}

impl TokenBatch {
    /// The number of rows (sequences) of the batch.
    pub fn rows(&self) -> usize {
        self.seq_lens.elem_count()
    }

    /// The number of packed query tokens.
    pub fn tokens(&self) -> usize {
        self.num_tokens as usize
    }

    /// Checks the batch's own parts against each other: the shapes and element types a kernel
    /// reads by address.
    ///
    /// The index tensors are read by arithmetic inside a kernel, so a batch whose slot mapping
    /// is one entry short, whose offsets end at another token count or whose `max_kv_len` is
    /// shorter than a page reads the wrong pages with nothing to refuse it. The modules cannot
    /// see the batch's intent, only its parts, so the check is here; it is shapes and types only
    /// (reading the values would need the device), and it is the one place a batch is validated.
    ///
    /// `rows` is derived from `seq_lens`, so a batch that states the wrong number of sequences
    /// is caught by the two columns that must agree with it.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`], naming the column that disagrees.
    pub fn check(&self) -> Result<()> {
        const OWNER: &str = "TokenBatch";

        if self.rows() == 0 || self.tokens() == 0 {
            return Err(Error::shape(
                OWNER,
                "a batch of no rows or no tokens has nothing to run".to_owned(),
            ));
        }
        if self.page_size == 0 {
            return Err(Error::shape(
                OWNER,
                "a page of zero tokens cannot be addressed".to_owned(),
            ));
        }
        if self.max_kv_len < self.page_size {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a key bound of {} is shorter than the {} tokens of a page",
                    self.max_kv_len, self.page_size
                ),
            ));
        }

        let (rows, tokens) = (self.rows(), self.tokens());
        for (name, column, expected, dtype) in [
            ("positions", &self.positions, tokens, DType::I32),
            ("slot_mapping", &self.slot_mapping, tokens, DType::U32),
            ("seq_lens", &self.seq_lens, rows, DType::U32),
            ("query_offsets", &self.query_offsets, rows + 1, DType::U32),
        ] {
            if column.elem_count() != expected {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "{name} is {} elements, {expected} expected for {rows} rows of {tokens} \
                         tokens",
                        column.elem_count()
                    ),
                ));
            }
            if column.dtype() != dtype {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "{name} is {}, not {}",
                        ops::abi::name(column.dtype()),
                        ops::abi::name(dtype)
                    ),
                ));
            }
        }

        // Each row names at most one hidden row for its logits, and a row that names none is a
        // row the sampler does not see.
        if self.logits_rows.elem_count() > rows || self.logits_rows.dtype() != DType::U32 {
            return Err(Error::shape(
                OWNER,
                format!(
                    "logits_rows is {} {} elements for {rows} rows, not at most one u32 per row",
                    self.logits_rows.elem_count(),
                    ops::abi::name(self.logits_rows.dtype())
                ),
            ));
        }

        if let Some(coordinates) = &self.rope_positions
            && (coordinates.rank() != 2
                || coordinates.dims()[1] != tokens
                || coordinates.dtype() != DType::I32)
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "rope_positions is {:?} {}, [axes, {tokens}] i32 expected",
                    coordinates.dims(),
                    ops::abi::name(coordinates.dtype())
                ),
            ));
        }

        if let Some(draft_lens) = &self.draft_lens
            && (draft_lens.elem_count() != rows || draft_lens.dtype() != DType::U32)
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "draft_lens is {} {} elements for {rows} rows",
                    draft_lens.elem_count(),
                    ops::abi::name(draft_lens.dtype())
                ),
            ));
        }

        if let Some(live) = &self.live_tokens
            && (live.elem_count() != 1 || live.dtype() != DType::U32)
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "live_tokens is {} {} elements, one u32 expected",
                    live.elem_count(),
                    ops::abi::name(live.dtype())
                ),
            ));
        }

        match (&self.token_ids, &self.ngram_context) {
            (None, None) => {}
            (Some(ids), Some(context))
                if ids.dims() == [tokens]
                    && ids.dtype() == DType::U32
                    && context.dims() == [rows, 2]
                    && context.dtype() == DType::U32 => {}
            (ids, context) => {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "token_ids is {:?} and ngram_context {:?}: [{tokens}] u32 and \
                         [{rows}, 2] u32 expected, both or neither",
                        ids.as_ref().map(Tensor::dims),
                        context.as_ref().map(Tensor::dims)
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The key/value planes of one attention layer: external tensors over the layer's cache pages.
///
/// A plane is `[pages, page_size, kv_heads, head_dim]` with one initialization flag per page: a
/// reader must not read a page before its flag is raised, and a page's owner zeroes only storage
/// it created.
#[derive(Clone)]
pub struct KvPlanes {
    /// `[pages, page_size, kv_heads, head_dim]` cached keys.
    pub key: Tensor,
    /// `[pages, page_size, kv_heads, head_dim]` cached values.
    pub value: Tensor,
    /// `[pages, 1, 1, 1]` **U8**, one flag per page of `key`.
    ///
    /// One byte, not four: the Storage Server holds one byte per page, and a wider flag would
    /// write over its neighbours.
    pub key_initialized: Tensor,
    /// `[pages, 1, 1, 1]` **U8**, one flag per page of `value`.
    pub value_initialized: Tensor,
    /// `[pages, 1, 1, 1]` F32: each page's key scale, present exactly when `key` is FP8.
    ///
    /// An FP8 cache stores a page divided by its own scale and the attention kernel multiplies
    /// it back in FP32. The scale is one layer's: two layers of one group keep their caches
    /// apart and each has its own.
    pub key_scale: Option<Tensor>,
    /// `[pages, 1, 1, 1]` F32: each page's value scale, present exactly when `value` is FP8.
    pub value_scale: Option<Tensor>,
    /// `[pages, page_size / ratio, width]` index keys, for a layer whose attention reads the
    /// tokens an indexer selects ([`crate::family::CacheShape::index_keys`]): row `g` of a page
    /// is the compressed key of the page's group `g` of `ratio` consecutive tokens, written when
    /// the group completes. It lies on the key/value page table, so a page reused from the
    /// prefix cache brings its groups' keys. `None` for a layer that attends every key.
    pub index_keys: Option<Tensor>,
}

/// The paged cache of one attention group, as execution lends it for one forward: how rows
/// address pages and where each layer's pages are.
///
/// A model with one attention type has one group; a model whose layers attend differently has
/// one per group, in the order its attention layers index them. The element type of a group's
/// pages is its planes' own: BF16, or FP8 for a group whose layers state scales.
#[derive(Clone)]
pub struct KvCache {
    /// `[rows, max_pages]` U32: each row's page ids.
    pub block_tables: Tensor,
    /// Tokens per page; the same as the batch's `page_size`.
    pub page_size: u32,
    /// One entry per layer that reads this group, in model order.
    pub layers: Vec<KvPlanes>,
    /// `[bytes]` U8: the paged-attention kernel's transient workspace, which execution reserves
    /// for the forward at the size the loaded kernel needs. Its head holds the decode's arrival
    /// counters, which the model clears once per forward.
    pub workspace: Tensor,
}

impl KvCache {
    /// The planes of `layer`.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] for a layer the group does not hold.
    pub fn layer(&self, layer: usize) -> Result<&KvPlanes> {
        self.layers.get(layer).ok_or_else(|| {
            Error::shape(
                "KvCache",
                format!(
                    "layer {layer} was asked for but the group holds {}",
                    self.layers.len()
                ),
            )
        })
    }
}

/// The recurrent state a batch's rows read and advance: external tensors over the per-slot state
/// buffers of the layers that keep one ([`crate::family::StateShape`]).
///
/// Its rows are the batch's **state rows**: one request's consecutive tokens each, which advance
/// one slot's state in order. They are the batch's attention rows ([`TokenBatch`]) except in a
/// verification, whose tokens attend as one row each (a decode's shape, so each is evaluated as
/// the decode step at its position would be) while one state row covers them all.
///
/// The live buffers hold every slot's state after its request's last computed token; the
/// checkpoint buffers receive the state a row keeps for the prefix cache. Execution allocates
/// both, restores a slot's live state from the cache before a row that resumes from a cached
/// prefix, and moves a checkpoint into the cache afterwards; a model reads and writes the
/// buffers only through its modules' operators.
#[derive(Clone)]
pub struct RecurrentState {
    /// `[rows + 1]` U32: the packed offsets of each state row's tokens in the batch.
    pub query_offsets: Tensor,
    /// `[rows]` U32: the tokens each state row has seen, its new ones included.
    pub seq_lens: Tensor,
    /// `[rows]` U32: each row's request slot in the buffers; [`NO_SLOT`] for a padding row,
    /// whose state is neither read nor written.
    pub slots: Tensor,
    /// `[rows]` U32: the absolute token count after which each row's state is kept in the
    /// checkpoint buffers, a page boundary inside the row's tokens or at its end; zero for a row
    /// that keeps none.
    pub checkpoints: Tensor,
    /// The live buffers: one entry per recurrent layer, in the model's order of them, each the
    /// layer's fields in the order of its [`crate::family::StateShape`], `[slots, …]` in the
    /// field's element type and extents. A model whose layers keep different fields states
    /// them as runs of layers ([`crate::family::TextModel::state_layers`]), and the entries
    /// follow the runs in order.
    pub live: Vec<Vec<Tensor>>,
    /// The checkpoint buffers, in the same order and extents. Empty for a drafter's state
    /// ([`Drafter::state`]), which execution keeps live alone: its `checkpoints` are then zero.
    pub checkpoint: Vec<Vec<Tensor>>,
    /// The trail buffers of a batch that keeps one, in the same order: `[tokens, …]` per field,
    /// row `t` receiving the layer's state after the batch's token `t`. A speculative
    /// verification keeps one, because only after the forward does it learn after which of a
    /// row's tokens the slot's state stands; `None` for every other batch.
    pub trail: Option<Vec<Vec<Tensor>>>,
    /// The slots of a deployment whose verifications copy no state ([`VerifySlots`]); `None`
    /// for a deployment that keeps one slot per request, with or without a trail.
    pub verify: Option<VerifySlots>,
}

/// The slots a request's rows read and write when acceptance copies no state (vLLM's
/// `state_indices` and `num_accepted`, `refs/vllm/vllm/v1/attention/backends/gdn_attn.py:463-486`).
///
/// A request of a deployment that drafts `n` tokens owns `Q = n + 1` slots of a matrix-valued
/// field and one slot, its first, of every window field, whose windows are `n` rows longer than
/// the layer's history. A decode-phase row (a decode or a verification, at most `Q` tokens)
/// starts from the state `num_accepted` says stands and leaves the state after each of its
/// tokens where the next step can choose it:
///
/// - a matrix-valued field is read from slot `state_indices[r, num_accepted[r] − 1]` and the
///   state after the row's token `t` is written to slot `state_indices[r, t]`, the source read
///   before any slot is written;
/// - a window field is read from row `num_accepted[r] − 1` of the window of slot
///   `state_indices[r, 0]`, and left holding the inputs after the first one read, so the
///   history before the row's token `t + 1` begins at window row `t`.
///
/// A row of any other phase reads the same way and leaves the state after its last token in
/// slot `state_indices[r, 0]` and at window row zero, so the step after it states
/// `num_accepted = 1`. What a commit writes is `num_accepted` alone.
#[derive(Clone)]
pub struct VerifySlots {
    /// `[rows, Q]` U32: each state row's slots; column zero is the row's entry of
    /// [`RecurrentState::slots`]. A padding row's entries are [`NO_SLOT`].
    pub state_indices: Tensor,
    /// `[rows]` U32, in `1..=Q`: how many tokens of the row's previous step stand; one for a
    /// row whose previous step was not of the decode phase.
    pub num_accepted: Tensor,
}

/// The rows of one expert-parallel exchange, as execution lends them to a model whose routed
/// experts are split across ranks.
///
/// A tensor-parallel deployment replicates the router and keeps each rank's experts, so a token
/// routed to another rank travels there. The kernel package owns the transport (its buffers, its
/// ordering, its state); execution creates it once per device and group and sets it on the
/// execution context of the forward, so no value here holds it. What a model reads are the
/// tensors the transport wrote and the extents of the layout they are in. `ranks × rank_rows`
/// rows exist, and `live` states how many of them the running step holds, which the ranks agree
/// on the device.
#[derive(Clone)]
pub struct ExpertExchange {
    /// Ranks in the expert-parallel group; the leading axis of the received layout.
    pub ranks: u32,
    /// Leading ranks of the group that hold no experts.
    pub attention: u32,
    /// Rows of one rank's block of the received tensors: the most one rank may exchange in a
    /// step, which execution sizes the transport for.
    pub rank_rows: u32,
    /// Rows per rank the expert work of this forward is shaped for: the step's graph holds at
    /// most this many, so the received tensors are read to `ranks * step_rows` rows. At most
    /// `rank_rows`.
    pub step_rows: u32,
    /// `[ranks * rank_rows, hidden]` in the activation element type: the dispatched activations.
    pub states: Tensor,
    /// `[ranks * rank_rows, top]` U32: the experts each received row is routed to.
    pub ids: Tensor,
    /// `[ranks * rank_rows, top]` F32: the router weights of each received row.
    pub weights: Tensor,
    /// `[ranks * rank_rows, hidden]` in the activation element type: where this rank's experts
    /// leave each received row's sum, for the combine to return.
    pub combined: Tensor,
    /// `[1]` U32: the leading rows of these tensors the running step holds.
    pub live: Tensor,
    /// `[rank_rows, hidden / 2]` U8: where a fused launch reads this rank's encoded rows.
    pub packed: Tensor,
    /// `[rank_rows, hidden / 16]` block scales of those rows.
    pub packed_scales: Tensor,
    /// One launch runs the whole expert layer, so the module sorts and multiplies nothing
    /// itself.
    pub fused: bool,
    /// This rank holds no experts: it sends routes and takes sums back, and computes nothing.
    pub source_only: bool,
}

impl ExpertExchange {
    /// Checks the received tensors against the extents the exchange laid them out with: the
    /// shapes and element types a kernel reads by address.
    ///
    /// Execution builds this value, so a wrong extent is a wiring error; refusing it here keeps
    /// it out of a launch that would read unrelated memory.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`], naming the tensor that disagrees.
    pub fn check(&self) -> Result<()> {
        const OWNER: &str = "ExpertExchange";

        let (&[_, hidden], &[_, top]) = (self.states.dims(), self.ids.dims()) else {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the received states are {:?} and their expert ids {:?}, not rows of hidden \
                     elements and rows of routes",
                    self.states.dims(),
                    self.ids.dims()
                ),
            ));
        };
        if top == 0 || hidden == 0 {
            return Err(Error::shape(
                OWNER,
                "an exchange of zero hidden elements or zero experts computes nothing".to_owned(),
            ));
        }
        if self.rank_rows == 0 || self.ranks == 0 {
            return Err(Error::shape(
                OWNER,
                "the exchange holds no row: a rank sends at least one".to_owned(),
            ));
        }
        if self.step_rows == 0 || self.step_rows > self.rank_rows || self.attention >= self.ranks {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a step of {} rows per rank in blocks of {}, with {} of {} ranks holding no \
                     experts",
                    self.step_rows, self.rank_rows, self.attention, self.ranks
                ),
            ));
        }

        let rows = self.ranks as usize * self.rank_rows as usize;
        let activation = self.states.dtype();
        for (name, tensor, expected, dtype) in [
            ("states", &self.states, [rows, hidden], activation),
            ("ids", &self.ids, [rows, top], DType::U32),
            ("weights", &self.weights, [rows, top], DType::F32),
            ("combined", &self.combined, [rows, hidden], activation),
        ] {
            if tensor.dims() != expected || tensor.dtype() != dtype {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "the exchange's {name} is {} {:?}, {} {expected:?} expected",
                        ops::abi::name(tensor.dtype()),
                        tensor.dims(),
                        ops::abi::name(dtype)
                    ),
                ));
            }
        }

        if self.live.elem_count() != 1 || self.live.dtype() != DType::U32 {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the exchange's live count is {} {} elements, one u32 expected",
                    self.live.elem_count(),
                    ops::abi::name(self.live.dtype())
                ),
            ));
        }
        Ok(())
    }
}

/// The shape of one encode: the rows it fills and the image grids it merges.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EncoderLayout {
    /// The packed rows the encode fills.
    pub rows: u32,
    /// The token rows that feed it.
    pub tokens: u32,
    /// The patch grids, `[time, rows, columns]`, of the images it merges, in request order.
    pub images: Vec<[u32; 3]>,
}

/// A video request's numerical layout input: what the denoiser must lay out.
///
/// Every field is a request fact execution already holds (it decoded the media, encoded the
/// prompt and knows the resolution), and none is derivable from the others: the packed layout
/// needs the resolution, the roles' counts, the prompt's token count and each condition's
/// raster. A deployment also states one of these per layout it serves, before any request of the
/// layout arrives ([`crate::family::VideoModel::candidates`]): a capture happens before the
/// first request of a shape, and the bucket is derived from the request ([`Denoiser::layout`]),
/// never stated beside it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VideoRequest {
    /// The resolution's width in pixels.
    pub width: u32,
    /// The resolution's height in pixels.
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
    /// The three roles are counted apart because they enter the layout differently: an anchor
    /// is one encoded frame at a fixed position, a reference is an encoded clip.
    pub references: u32,
    /// Prompt tokens the request presents, vision tokens included.
    ///
    /// The packed sequence's rotary clock starts at the prompt's length, so a layout cannot
    /// place a row without it.
    pub text_tokens: u32,
    /// The encoded condition media, in request order: anchors, then references.
    pub conditions: Vec<ConditionShape>,
    /// Per-image prompt spans, in request order: the prompt rows a reference's own rows replace.
    ///
    /// Those rows take the image's rotary coordinates and not the text clock's, so a layout
    /// that placed them on the prompt's clock would be silently wrong.
    pub vision_spans: Vec<(u32, u32)>,
    /// The condition rows the request's layout holds, as one rung of the deployment's ladder;
    /// `None` is the deployment's whole capacity.
    ///
    /// A layout's rows are real rows of every projection, attention and feed-forward of every
    /// step, so the rung is what a step costs, and the conditions a request states fill it from
    /// its start. The caller chooses it, because a request a deployment captures for states no
    /// conditions to count.
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

/// The rows the tensor a request's packed prefix is gathered from holds, for a request of
/// `layout`: the prompt's `text_rows` rows, then `condition_rows` condition rows, then one row
/// of zeros ([`prefix_zero_row`]) that every generated and padding row of the packed sequence
/// gathers.
///
/// Execution allocates and fills that tensor and the denoiser's packing addresses it. The text
/// and condition rows are capacities a request fills from their start.
pub const fn prefix_rows(layout: &Layout) -> u32 {
    prefix_zero_row(layout) + 1
}

/// The row of zeros of a packed prefix's source ([`prefix_rows`]): the row past the text and
/// condition rows, which execution writes, at the same place for every request of a layout.
pub const fn prefix_zero_row(layout: &Layout) -> u32 {
    layout.text_rows + layout.condition_rows
}

/// One piece of raw media a request's conditions are built from.
///
/// This is what [`Denoiser::prepare_conditioning`] encodes when the model owns its condition
/// encoders: pixels before they are latents, samples before they are rows. The role says where
/// the piece anchors, which decides where its rows sit in the packed conditioning: first-frame
/// anchors, last-frame anchors, then references.
#[derive(Clone)]
pub struct MediaInput {
    /// Where the piece anchors.
    pub role: ConditionRole,
    /// The packed rows the layout reserves for the piece's visual condition, as the request's
    /// own layout counted them: one row per `1×2×2` patch of its latent.
    ///
    /// The conditioner checks the encoder's output against this and does not recompute it: the
    /// packing the request was laid out under is the one statement of the arithmetic.
    pub rows: u32,
    /// The packed rows the layout reserves for the piece's soundtrack, in the same sense.
    pub audio_rows: u32,
    /// `[channels, frames, height, width]` pixels, in the encoder's own channel order and
    /// normalization, when the condition is visual. One frame for a still, which is one latent
    /// frame.
    pub pixels: Option<Tensor>,
    /// Samples in the encoder's shape, when the condition has a soundtrack.
    pub audio: Option<Tensor>,
    /// The condition's anchoring draw: `[channels, frames, height, width]` at the piece's own
    /// latent raster, FP32, when the piece is visual.
    ///
    /// Execution owns it, as it owns every draw of the request's noise stream, and it is drawn
    /// at latent resolution, so the conditioner patchifies it.
    pub noise: Option<Tensor>,
}

/// One piece of a request's conditions, as the packed conditioning holds it.
///
/// A request states its conditions in its own order; the model packs them in another (keyframes
/// first), and only the model knows how many rows each takes and where its draw sits in the
/// request's noise stream. So execution asks for this list ([`Denoiser::condition_pieces`]) and
/// builds its [`MediaInput`]s in this order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ConditionPiece {
    /// The condition's place in the request, which is how the raw media is addressed.
    pub condition: u32,
    /// Where the piece anchors.
    pub role: ConditionRole,
    /// The packed rows of its visual condition; zero without pixels.
    pub rows: u32,
    /// The packed rows of its soundtrack; zero without one.
    pub audio_rows: u32,
    /// The shape of its [`MediaInput::pixels`], `None` for a piece with no pixels.
    pub pixels: Option<Shape>,
    /// The shape of its [`MediaInput::audio`], `None` for a piece with no soundtrack.
    pub track: Option<Shape>,
    /// The shape of its anchoring draw, `None` for a piece with no pixels: an audio reference is
    /// conditioned at its posterior's mean and draws nothing.
    pub draw: Option<LatentShape>,
    /// The first element of the draw in the request's noise stream.
    pub draw_offset: u64,
}

/// How a model's encoders read the media a request's conditions decode to.
///
/// The media reader decodes bytes into RGB frames and stereo PCM, which are the media's own
/// forms. What an encoder reads is the model's arithmetic: the normalization of its pixels, its
/// vision tower's patch rows, the padding of a track. The model declares that layout here and
/// execution performs the conversions on its host threads, so no model holds a thread or a
/// media byte format.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MediaFormat {
    /// The normalization of a condition's pixels, which [`MediaInput::pixels`] holds as
    /// `[3, frames, height, width]` FP32, each byte `(byte / 255 - mean) / std`.
    pub pixels: PixelNorm,
    /// The patch rows a vision block presents to the text encoder.
    pub patches: PatchFormat,
    /// Samples of one audio latent frame. A track [`MediaInput::audio`] holds is right-padded
    /// with zeros to a whole number of them, as `[2, 1, padded]` channel rows.
    pub hop: u64,
}

/// The patch rows of a vision tower, as the processor lays them out.
///
/// Each row is one channel-major tubelet of `channels × temporal × side × side` values; rows run
/// over each temporal patch's merge blocks of `merge × merge` patches, a short temporal patch
/// repeating its last frame. A value is `(byte - 255 · mean) / (255 · std)` in FP32, rounded
/// once to `dtype`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PatchFormat {
    /// The pixels one patch spans along each side of a frame; a block's frames are resized to
    /// its grid times this.
    pub side: u32,
    /// Frames one temporal patch spans.
    pub temporal: u32,
    /// Patches one merge block spans along each side.
    pub merge: u32,
    /// The raster's channels.
    pub channels: u32,
    /// The processor's normalization.
    pub norm: PixelNorm,
    /// The element type the rows are stored in: the tower's weights' own.
    pub dtype: DType,
}

impl PatchFormat {
    /// The elements of one patch row: `channels · temporal · side²`.
    pub const fn width(&self) -> u32 {
        self.channels * self.temporal * self.side * self.side
    }

    /// The raster a block of `grid` (`[temporal, rows, columns]` patches) spans,
    /// `(height, width)` in pixels: the size the processor resized the block's frames to when it
    /// laid out the grid, which execution resizes them to again before it lays out their
    /// patches.
    pub const fn raster(&self, grid: [u32; 3]) -> (u32, u32) {
        (grid[1] * self.side, grid[2] * self.side)
    }
}

/// How an RGB raster's values relate to pixel intensities in `[0, 1]`: per channel, red, green,
/// blue, `value = (intensity - mean) / std`.
///
/// One declaration per model serves both directions: an encoder reads condition pixels
/// normalized by it, and a decoder's raster is de-normalized by it into the bytes a caller
/// receives. Execution performs both conversions.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PixelNorm {
    /// The per-channel means.
    pub mean: [f32; 3],
    /// The per-channel standard deviations.
    pub std: [f32; 3],
}

/// What a denoiser conditions on at a step: external tensors over the request's conditioning
/// buffers.
///
/// The rows of every tensor follow the order the layout reserves: the encoded text first, then
/// the request's conditions in the order [`VideoRequest`] counts them (first-frame anchors,
/// last-frame anchors, references).
///
/// The one request-shaped thing here is [`Conditioning::media`], the raw media a conditioner
/// encodes. It has to be: [`Denoiser::prepare_conditioning`] runs once per request, before any
/// capture, and condition rows cannot be built from tensors that are already latents.
#[derive(Clone)]
pub struct Conditioning {
    /// `[rows, width]` encoded text: what the denoiser reads, and the buffer
    /// [`Denoiser::prepare_conditioning`] writes in place.
    pub text: Tensor,
    /// `[rows, width]` the text encoder's own features, when the model has a conditioner
    /// between the encoder and the denoiser: [`Denoiser::prepare_conditioning`] computes `text`
    /// from it. `None` when the model has no conditioner, or for a task whose conditioning is
    /// nothing but the encoder's features.
    pub source: Option<Tensor>,
    /// The raw media the request's conditions are built from, in the order
    /// [`Denoiser::condition_pieces`] states.
    ///
    /// [`Denoiser::prepare_conditioning`] encodes these and writes the condition rows of `text`;
    /// empty when the task has no conditions.
    pub media: Vec<MediaInput>,
    /// `[rows, …]` encoded reference media, when the task has references.
    pub images: Option<Tensor>,
    /// `[rows, …]` encoded reference audio, when the task has audio references.
    pub audio: Option<Tensor>,
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

/// One clip's decoded raster extent, as execution states it.
///
/// Frames, rows and columns of the **decoded** media, which is what a bucket of a decode is
/// keyed by. A decoder's tables are keyed by the latents the raster decodes from, and the ratios
/// between the two are the checkpoint's, so the map is the decoder's to state
/// ([`Decoder::layout_of`]).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Clip {
    /// Frames of the decoded raster.
    pub frames: u32,
    /// Rows of one decoded frame.
    pub height: u32,
    /// Columns of one decoded frame.
    pub width: u32,
}

/// One clip's place on a timeline, as execution states it.
///
/// A decoder decodes one clip and stops; the clip's frames become the request's video only once
/// they are placed, faded where they overlap the clip before them. These three numbers are what
/// execution knows and the model does not: how many of the clip's frames the timeline keeps, how
/// many of those fade against the predecessor, and where the first kept frame goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Splice {
    /// The frames of the clip's own decode the timeline keeps, counted from the clip's first
    /// frame.
    ///
    /// A clip a successor follows keeps the frames before its tail; the clip that ends the
    /// timeline keeps its tail too, so this is the larger number for the last clip.
    pub keep: u32,
    /// How many of the kept frames fade against the predecessor, counted at the end of `keep`.
    ///
    /// Zero for the timeline's first clip, which has nothing to fade against.
    pub overlap: u32,
    /// The destination frame index of the clip's first kept frame.
    pub first: u32,
}

/// A tower that turns inputs into conditioning (a text encoder, a vision tower, a video
/// encoder).
///
/// The model says what shape an encode has ([`Encoder::layout`]), which a runner captures by;
/// the values a shape's encode reads beyond its inputs (rotary clocks, tap tables, the prompt
/// rows its images replace) are computed inside [`Encoder::encode`].
pub trait Encoder {
    /// The shape an encode of `batch`'s rows and of images of the patch grids `images`
    /// (`[time, rows, columns]` each) has.
    fn layout(&self, batch: &TokenBatch, images: &[[u32; 3]]) -> EncoderLayout;

    /// Encodes one prompt into `out`, `[tokens, width]`: the runner's product buffer, which the
    /// stage that reads the conditioning is lent later.
    ///
    /// `prompt` is the request's own token ids on the host, exactly as the request stated them:
    /// the tower's positions and the prompt rows its images replace are arithmetic over them,
    /// which the encode computes as CPU tensors and moves to the device. Their shapes follow
    /// from `layout`'s counts alone, so two prompts of one shape share one capture. `ids` is the
    /// `[tokens]` U32 column of the same ids on the device, `batch` the positions, lengths and
    /// cache writes of their packed rows, and `images` the pixels of the images the prompt
    /// presents, one tensor per image in request order (`[patches, pixels]` packed patches, or
    /// `[channels, height, width]` rasters), empty for a text-only encoder. `layout` is the
    /// layout the encode fills, as execution got it from [`Encoder::layout`]: the model reads
    /// the grids and counts from it, because a device tensor cannot be read back.
    ///
    /// # Errors
    ///
    /// Inputs that do not describe `layout`, or what a module refuses.
    fn encode(
        &self,
        prompt: &[u32],
        ids: &Tensor,
        batch: &TokenBatch,
        images: &[Tensor],
        layout: &EncoderLayout,
        out: &Tensor,
    ) -> Result<()>;
}

/// A diffusion model that denoises latents step by step.
///
/// A request's latents are execution's: `video` is `[channels, frames, height, width]` and
/// `audio`, when the task has audio, `[channels, steps]`, both external tensors over the
/// request's latent buffers, which the methods below write in place.
pub trait Denoiser {
    /// The latent layout this request needs.
    ///
    /// # Errors
    ///
    /// A request whose resolution or frame count the model does not lay out; no layout is
    /// invented for it.
    fn layout(&self, request: &VideoRequest) -> Result<Layout>;

    /// The denoising steps `request` runs, in order.
    ///
    /// A runner executes one step per call and cannot derive the schedule: it belongs to the
    /// model's checkpoint, and `Step::index` and `Step::count` are what a solver checks itself
    /// against, so a runner holding its own sigma table would be a second source for a value
    /// the model states. A model whose schedule depends on what a request conditions on reads
    /// that from `request`.
    ///
    /// # Errors
    ///
    /// A model that states no schedule for `request` refuses by name. Empty would be the same
    /// value as a schedule that exists and has no steps, which is the opposite fact.
    fn schedule(&self, request: &VideoRequest) -> Result<Vec<Step>> {
        Err(Error::Unsupported(format!(
            "a denoiser that states no schedule for a request of {} anchor(s) and {} reference(s)",
            request.first_frames + request.last_frames,
            request.references
        )))
    }

    /// The width of [`Conditioning::text`]'s rows, when it is not the encoder's.
    ///
    /// `text` is what the denoiser reads and `source` is what the text encoder wrote, and a
    /// model whose conditioner **widens** its input has two widths: a runner that sized both
    /// buffers from the encoder's width would lend the conditioner one too narrow for its own
    /// output.
    ///
    /// `None` means one width: no conditioner, or a conditioner that keeps the encoder's.
    fn text_width(&self) -> Option<u32> {
        None
    }

    /// Draws the initial noise of a request's latents from `seed`, addressed within `layout`.
    ///
    /// A draw is a function of the seed and the element's index, never of a batch position, so
    /// recomputing a request after a migration draws the same noise. The layout is a parameter
    /// because the draw's element offsets are functions of the request's condition noise
    /// shapes, which only a layout names.
    ///
    /// # Errors
    ///
    /// Latents that are not `layout`'s, or what a module refuses.
    fn draw_noise(
        &self,
        seed: u64,
        layout: &Layout,
        video: &Tensor,
        audio: Option<&Tensor>,
    ) -> Result<()>;

    /// The pieces of `request`'s conditions, in the order [`Conditioning::media`] holds them.
    ///
    /// # Errors
    ///
    /// A request with no conditions has none. A model that takes conditions states them here,
    /// and one that takes none refuses a request that states some, by name, where an empty list
    /// would condition the clip on nothing.
    fn condition_pieces(&self, request: &VideoRequest) -> Result<Vec<ConditionPiece>> {
        if request.conditions.is_empty() {
            return Ok(Vec::new());
        }
        Err(Error::Unsupported(format!(
            "a denoiser that takes no conditions, asked for {}",
            request.conditions.len()
        )))
    }

    /// Draws `piece`'s anchoring noise into `out`, the piece's [`MediaInput::noise`] buffer, at
    /// its place in the request's stream.
    ///
    /// The conditions' draws lead the stream [`Self::draw_noise`] continues, so one seed names
    /// both.
    ///
    /// # Errors
    ///
    /// A denoiser that draws no condition noise.
    fn draw_condition_noise(&self, seed: u64, piece: &ConditionPiece, out: &Tensor) -> Result<()> {
        let _ = (seed, out);
        Err(Error::Unsupported(format!(
            "a denoiser that draws no condition noise, asked for condition {}",
            piece.condition
        )))
    }

    /// Computes the latents the schedule starts from, in place, from the drawn noise and
    /// `cond`.
    ///
    /// # Errors
    ///
    /// What a module refuses.
    fn prepare_latents(
        &self,
        cond: &Conditioning,
        video: &Tensor,
        audio: Option<&Tensor>,
    ) -> Result<()>;

    /// Prepares what the denoiser reads from a request's conditioning, in place: runs the
    /// conditioner over `cond.source` into `cond.text` and encodes `cond.media` into the
    /// condition rows of `cond.text`.
    ///
    /// Execution calls this once per request, before the request's first evaluation and outside
    /// a captured replay.
    ///
    /// Required, not defaulted: a model whose text encoder's features are everything its
    /// denoiser reads has no conditioner and says so with an explicit `Ok(())`, while a model
    /// that has one must write it here. A default that did nothing would make forgetting the
    /// override succeed silently, and a model whose conditioning is prepared nowhere generates
    /// a plausible clip from zeros.
    ///
    /// # Errors
    ///
    /// Conditioning that does not describe `request`, or what a module refuses.
    fn prepare_conditioning(&self, request: &VideoRequest, cond: &Conditioning) -> Result<()>;

    /// One denoising step of `request`: updates the latents in place with the model's velocity.
    ///
    /// The host values the step reads (packing indices, rotary positions, the step's schedule
    /// coefficients) are computed here from `request` and `step` as CPU tensors and moved to
    /// the device; their shapes follow from the request's layout alone, so every request of one
    /// layout replays one capture.
    ///
    /// # Errors
    ///
    /// What a module refuses.
    fn forward(
        &self,
        step: Step,
        request: &VideoRequest,
        video: &Tensor,
        audio: Option<&Tensor>,
        cond: &Conditioning,
    ) -> Result<()>;

    /// Fills the tables this model computes from its weights once, after the load and before
    /// any capture, and again after its weights were rewritten in place: the tables the load
    /// reserved writable ([`VarBuilder::writable_zeros`](crate::VarBuilder::writable_zeros)),
    /// such as a fixed schedule's timestep modulation. A model with none returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// What a module refuses.
    fn initialize(&self) -> Result<()>;
}

/// A decoder that turns latents back into media.
pub trait Decoder {
    /// The element type this decoder reads its latents in and writes its decoded media in.
    ///
    /// A runner allocates the latent buffer and the raster buffer before a graph exists, and a
    /// buffer's width in bytes does not say its element type, so the model states it here from
    /// its checkpoint and no runner infers it from a byte count or from its deployment's
    /// configuration.
    fn element_type(&self) -> DType;

    /// The element type of the latent block [`Self::decode`] reads, when it is not
    /// [`Self::element_type`].
    ///
    /// A decoder whose reference restores the latents in a wider type than it decodes them to
    /// says so here, and the runner keeps the block in that type.
    fn latent_type(&self) -> DType {
        self.element_type()
    }

    /// Writes the latent block of the clip starting at latent frame `first` into `block`.
    ///
    /// `trajectory` is the timeline's latents as a denoiser publishes them,
    /// `[channels, voxels]`, and `block` is the runner's buffer [`Self::decode`] reads,
    /// `[rows, channels]` in [`Self::latent_rows`]' order. A decoder that reads a trajectory
    /// says how its rows lie in it, so a clip never crosses to the host.
    ///
    /// # Errors
    ///
    /// A decoder that reads no denoiser trajectory.
    fn gather_latents(
        &self,
        layout: &Layout,
        trajectory: &Tensor,
        first: u32,
        block: &Tensor,
    ) -> Result<()> {
        let _ = (layout, trajectory, first, block);
        Err(Error::Unsupported(
            "this decoder reads no denoiser trajectory".to_owned(),
        ))
    }

    /// For each row of the latent block [`Self::decode`] reads, the clip's voxel it holds,
    /// counted frame by frame, row by row: a `[rows]` U32 CPU tensor.
    ///
    /// `None` reads each voxel once, in that order. A decoder that works in overlapping tiles
    /// lists its tiles' voxels instead, so a voxel two tiles share has two rows and the block
    /// is longer than the clip.
    ///
    /// # Errors
    ///
    /// A layout the decoder does not decode.
    fn latent_rows(&self, layout: &Layout) -> Result<Option<Tensor>> {
        let _ = layout;
        Ok(None)
    }

    /// The latent frame `clip`, counted from the timeline's first clip, starts at.
    ///
    /// A clip reads a window of the trajectory, and clips overlap by the frames the next one
    /// also reads, so where a window starts is the decoder's arithmetic. A timeline of one clip
    /// starts at zero.
    fn clip_start(&self, clip: u32) -> u32 {
        let _ = clip;
        0
    }

    /// The latent layout one clip's raster decodes from.
    ///
    /// [`Clip`] is execution's own extent and this decoder's tables are keyed by its latents.
    /// The ratios between them belong to the checkpoint, so the decoder states the map.
    ///
    /// # Errors
    ///
    /// A decoder whose decoded raster is not its latent layout scaled by ratios it knows
    /// refuses by name.
    fn layout_of(&self, clip: Clip) -> Result<Layout> {
        let _ = clip;
        Err(Error::shape(
            "Decoder",
            "this decoder states no latent layout for a clip's raster".to_owned(),
        ))
    }

    /// The waveform one decode of `audio` writes: samples per channel and channels.
    ///
    /// A track's window is the model's to state, and a runner sizes its buffer from it: the
    /// window is the vocoder's latent steps multiplied by the product of its upsampling rates,
    /// which is checkpoint arithmetic.
    ///
    /// # Errors
    ///
    /// A decoder that writes no waveform.
    fn pcm_window(&self, audio: &LatentShape) -> Result<(u64, u32)> {
        let _ = audio;
        Err(Error::Unsupported(
            "this decoder writes no waveform".to_owned(),
        ))
    }

    /// Decodes `latents` of `layout` into the runner's product buffers: `video`,
    /// `[channels, frames, height, width]` decoded frames in the decoder's own channel order
    /// and element type, and `audio`, a `[channels, samples]` FP32 waveform, one row per
    /// channel, at the decoder's own sample rate. A decoder writes the buffers its media has.
    ///
    /// The raster is channels first, as the media this boundary states is everywhere else, and
    /// the waveform planar, as the vocoder computes it: execution converts once, where it
    /// writes a file's bytes, so no decoder holds a media byte format.
    ///
    /// The values a decode of `layout` reads beyond its latents (offsets, rotary positions, the
    /// packed row of each voxel) are computed here as CPU tensors and moved to the device;
    /// their shapes follow from `layout`'s counts alone.
    ///
    /// # Errors
    ///
    /// Buffers that are not `layout`'s, or what a module refuses.
    fn decode(
        &self,
        layout: &Layout,
        latents: &Tensor,
        video: Option<&Tensor>,
        audio: Option<&Tensor>,
    ) -> Result<()>;
}

/// The step that writes decoded clips into one timeline.
///
/// A decoder decodes one clip of its window and stops; a timeline longer than one window is
/// decoded clip by clip, each clip overlapping its predecessor, and written by this step: it
/// fades a clip's leading frames against its predecessor's own decode of the frames the two
/// share and places the frames the timeline keeps. Which frames each clip writes is the
/// checkpoint's decoding arithmetic, so the splicer states the schedule ([`Self::splices`]) and
/// execution passes each [`Splice`] back with its clip.
pub trait Splicer {
    /// The element type of the rasters it reads and places: the decoder's raster type.
    fn element_type(&self) -> DType;

    /// How the rasters [`Self::splice`] places are normalized, which execution undoes as it
    /// writes a timeline's bytes.
    fn pixel_norm(&self) -> PixelNorm;

    /// The splices a timeline of `frames` frames is written in, in the order the frames are
    /// decoded.
    ///
    /// A splice's `first` is the destination frame in the raster given to [`Self::splice`], so
    /// execution that gives each clip its own raster states `first: 0`.
    ///
    /// # Errors
    ///
    /// When `frames` is not a frame count a timeline of this checkpoint holds.
    fn splices(&self, frames: u32) -> Result<Vec<Splice>>;

    /// Places one decoded clip on the timeline, fading it against its predecessor.
    ///
    /// `clip` is the clip's decoded raster; `previous` is the predecessor's own decode of the
    /// frames the two share, absent for the timeline's first clip. Returns the normalized runs
    /// of the splice's kept frames, in order: each a `[3, frames, rows, columns]` raster in the
    /// decoder's element type (a view of the clip's own decode, or a fade computed here) with
    /// the frame of the splice's destination, counted from [`Splice::first`], that the raster's
    /// first frame becomes. Execution de-normalizes them into the timeline's bytes by
    /// [`Self::pixel_norm`] before the next forward on the context.
    ///
    /// # Errors
    ///
    /// When the tensors are not a clip and a predecessor of this checkpoint's raster, or
    /// `splice` fades more frames than the checkpoint overlaps.
    fn splice(
        &self,
        previous: Option<&Tensor>,
        clip: &Tensor,
        splice: Splice,
    ) -> Result<Vec<(Tensor, u32)>>;
}

/// An autoregressive token model.
///
/// The batch is already in the layout the kernels read, so no method here converts or gathers
/// on the host.
pub trait CausalLm {
    /// The embeddings of `ids`, a `[tokens]` U32 column: `[tokens, hidden]` in the model's
    /// activation type.
    ///
    /// # Errors
    ///
    /// Ids the embedding refuses.
    fn embed_input_ids(&self, ids: &Tensor) -> Result<Tensor>;

    /// Runs the decoder stack over `hidden`, the batch's embeddings, and returns the final
    /// normalized hidden state, `[tokens, hidden]`; a model that carries several residual
    /// streams returns them side by side, `[tokens, stream_width]`
    /// ([`crate::family::TextModel::stream_width`]), and reads them out in
    /// [`Self::compute_logits`].
    ///
    /// `cache` holds the model's attention groups, in the order its attention layers index
    /// them, whose pages the batch's slot mapping writes in place. `state` is the recurrent
    /// state of a model whose [`crate::family::TextModel::state`] declares one, advanced in
    /// place, and `None` for a model that keeps none. `exchange` is lent to a model whose
    /// routed experts are split across ranks, and is `None` otherwise.
    ///
    /// # Errors
    ///
    /// A batch or a cache a module refuses, or a recurrent layer run without its state.
    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &[KvCache],
        state: Option<&RecurrentState>,
        exchange: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> Result<Tensor>;

    /// The logits of `hidden`, the rows of [`Self::forward`]'s result named by
    /// [`TokenBatch::logits_rows`], which execution gathered: `[rows, vocabulary]` FP32.
    ///
    /// # Errors
    ///
    /// Rows the head refuses.
    fn compute_logits(&self, hidden: &Tensor) -> Result<Tensor>;

    /// Runs every expert layer of one step with no tokens of this rank's own.
    ///
    /// A rank of an expert group runs every step of the group, because its experts serve the
    /// tokens its peers route to them.
    ///
    /// # Errors
    ///
    /// A model without split experts has nothing to join.
    fn join_expert_step(&self, exchange: &ExpertExchange) -> Result<()> {
        let _ = exchange;
        Err(Error::Unsupported(
            "this model splits no experts across ranks, so it has no expert step to join"
                .to_owned(),
        ))
    }
}

/// A causal language model's own drafter: multi-token-prediction layers that propose the token
/// after the next from the model's hidden state and the next token (DeepSeek-V3's and
/// Qwen3-Next's MTP module, vLLM's and SGLang's `*MTP` drafters).
///
/// The drafter's entry at position `i` reads the model's final hidden state at `i` (the output
/// of its [`CausalLm::forward`]) and the embedding of the token at `i + 1`; its own hidden
/// state there is what [`CausalLm::compute_logits`] turns into the distribution of the token at
/// `i + 2`, and what its next entry reads in place of the model's when it drafts further ahead.
/// The drafter embeds its tokens with the model's [`CausalLm::embed_input_ids`] and projects
/// with the model's head; its own layers attend over a key/value cache of their own, the draft
/// cache, which execution allocates as it does the model's ([`KvCache`]).
///
/// Which positions a step evaluates, which drafts it proposes and how a verification accepts
/// them are execution's: the drafter computes one forward over the rows it is given.
pub trait Drafter: Send + Sync {
    /// The key/value cache its layers read, the draft cache, as this rank holds it.
    fn cache(&self) -> crate::family::CacheShape;

    /// Runs the drafter over `batch`: `embedded [tokens, hidden]` holds the embedding of each
    /// entry's next token and `hidden [tokens, hidden]` the hidden state each entry reads;
    /// returns the drafter's own normalized hidden state, `[tokens, hidden]`. `cache` is the
    /// draft cache's one attention group, which the batch's slot mapping writes in place.
    ///
    /// # Errors
    ///
    /// Tensors a module refuses.
    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> Result<Tensor>;

    /// The state its layers keep per request slot beside the draft cache, in the order
    /// [`Self::forward_stateful`] is lent their buffers; empty for a drafter that keeps none.
    ///
    /// Execution keeps one live row of each field per request slot and nothing else: no field
    /// is stepwise ([`crate::family::StateField::stepwise`]), and the state is neither kept in
    /// the prefix cache's checkpoints nor restored from them. A request that resumes from a
    /// cached prefix, or starts on a slot, runs the drafter from that page boundary over rows
    /// that hold whatever the slot last held, and every forward advances the state over all
    /// its rows, the entries a verification rejects and the proposals past the accepted
    /// position included. A drafter therefore states only fields of which a forward whose
    /// state rows start at a page boundary reads no row from before it, and that a forward
    /// overwrites at a position before any later forward reads them there, such as a ring of
    /// the recent rows of an index group the page divides; a field that summarizes every
    /// earlier token, as a recurrent layer's does, has no place here.
    fn state(&self) -> Vec<crate::family::StateShape> {
        Vec::new()
    }

    /// [`Self::forward`] for a drafter whose layers keep a state ([`Self::state`]): `state`
    /// holds the live buffers of its layers, advanced in place, with no checkpoint buffer, no
    /// trail and no verify slots; its rows are the batch's. `hidden` and the result are rows
    /// of the model's [`crate::family::TextModel::stream_width`]. A drafter that keeps none
    /// ignores `state`.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`], and a stateful drafter run without its state.
    fn forward_stateful(
        &self,
        batch: &TokenBatch,
        cache: &KvCache,
        state: Option<&RecurrentState>,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let _ = state;
        self.forward(batch, cache, embedded, hidden)
    }

    /// The rows [`CausalLm::compute_logits`] reads of `hidden`, rows of the drafter's output:
    /// the output itself, or what the drafter's own read of it computes when the model's head
    /// projects another width than the drafter returns.
    ///
    /// # Errors
    ///
    /// Rows a module refuses.
    fn logits_rows(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(hidden.clone())
    }
}

/// The vision tower of a causal language model: images to rows of the model's hidden width,
/// which a prefill writes into the request's context in place of the image's placeholder.
///
/// Execution reads an image's bytes, sizes it by [`Vision::image_grid`], resizes and lays its
/// pixels out in [`Vision::patch_format`] on its host threads and calls
/// [`Vision::encode_image`]; the features it keeps are the rows a later prefill places, at the
/// rotary coordinates [`Vision::image_positions`] states. Every input is a tensor or a host
/// value of the image's grid: no request identity crosses.
pub trait Vision {
    /// The patch grid `[time, rows, columns]` an image of `height` x `width` pixels is resized
    /// to: the checkpoint's processor rule.
    ///
    /// # Errors
    ///
    /// An image the processor cannot size (an aspect it refuses).
    fn image_grid(&self, height: u32, width: u32) -> Result<[u32; 3]>;

    /// How the tower reads an image's pixels: one patch row per patch of the grid.
    fn patch_format(&self) -> PatchFormat;

    /// The rows one image of `grid` encodes to: the tokens of its block in the context.
    ///
    /// # Errors
    ///
    /// A grid that is not whole merge blocks.
    fn image_rows(&self, grid: [u32; 3]) -> Result<u32>;

    /// The most rows an image the processor sizes encodes to: the block of the largest image.
    fn max_image_rows(&self) -> u32;

    /// The rotary axes of [`TokenBatch::rope_positions`] a batch with image rows states.
    fn position_axes(&self) -> u32;

    /// The rotary coordinates of an image block of `grid` whose first row takes logical
    /// position `start`: an `[axes, rows]` I32 CPU tensor, axis-major, `rows` being
    /// [`Vision::image_rows`]. A text row reads its own position on every axis.
    ///
    /// # Errors
    ///
    /// As [`Vision::image_rows`].
    fn image_positions(&self, grid: [u32; 3], start: u32) -> Result<Tensor>;

    /// Encodes one image: `pixels` (`[patches, width]` in [`Vision::patch_format`]) into `out`,
    /// the buffer execution keeps the image's features in, `[rows, hidden]` in the model's
    /// activation type, `rows` being [`Vision::image_rows`]. The grid's position tables are
    /// computed here as CPU tensors and moved to the device.
    ///
    /// # Errors
    ///
    /// Tensors that do not describe `grid`, or what a module refuses.
    fn encode_image(&self, grid: [u32; 3], pixels: &Tensor, out: &Tensor) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use tensor::{DType, Device, Tensor};

    use super::*;

    /// A CPU column of `elements` zeros of `dtype`.
    fn column(dtype: DType, elements: usize) -> Tensor {
        Tensor::zeros(elements, dtype, &Device::Cpu).unwrap()
    }

    /// A batch of `rows` sequences, one token each, over 64-token pages.
    fn batch(rows: usize) -> TokenBatch {
        TokenBatch {
            positions: column(DType::I32, rows),
            slot_mapping: column(DType::U32, rows),
            seq_lens: column(DType::U32, rows),
            query_offsets: column(DType::U32, rows + 1),
            logits_rows: column(DType::U32, rows),
            phase: PhaseKind::Decode,
            page_size: 64,
            num_tokens: u32::try_from(rows).unwrap(),
            max_kv_len: 4_096,
            draft_lens: None,
            rope_positions: None,
            live_tokens: None,
            token_ids: None,
            ngram_context: None,
        }
    }

    #[test]
    fn a_batch_whose_parts_disagree_is_refused_by_the_part() {
        // The consistent one passes, or the rest of this test proves nothing.
        batch(15).check().expect("a consistent batch");

        // A slot short: the write would scatter 14 rows and address a fifteenth slot that is
        // not there.
        let mut short = batch(15);
        short.slot_mapping = column(DType::U32, 14);
        let error = short.check().expect_err("a slot mapping of another length");
        assert!(format!("{error}").contains("slot_mapping"), "{error}");

        // An index column is unsigned: a signed one is another element type.
        let mut signed = batch(15);
        signed.seq_lens = column(DType::I32, 15);
        let error = signed.check().expect_err("signed sequence lengths");
        assert!(
            format!("{error}").contains("seq_lens is i32, not u32"),
            "{error}"
        );

        // Offsets that do not close the batch: one entry per sequence plus the end.
        let mut offsets = batch(15);
        offsets.query_offsets = column(DType::U32, 15);
        assert!(offsets.check().is_err());

        // A key bound shorter than a page addresses no page at all.
        let mut unpaged = batch(4);
        unpaged.max_kv_len = 32;
        let error = unpaged.check().expect_err("a bound below one page");
        assert!(format!("{error}").contains("shorter than"), "{error}");

        // More logits rows than rows would have the sampler read past the last row.
        let mut logits = batch(4);
        logits.logits_rows = column(DType::U32, 8);
        assert!(logits.check().is_err());

        // Draft counts are one per row, when a batch verifies drafts at all.
        let mut drafts = batch(4);
        drafts.draft_lens = Some(column(DType::U32, 2));
        assert!(drafts.check().is_err());

        // Rotary coordinates are one column per token on every axis.
        let mut coordinates = batch(4);
        coordinates.rope_positions = Some(Tensor::zeros((3, 5), DType::I32, &Device::Cpu).unwrap());
        assert!(coordinates.check().is_err());

        // And an empty batch is not a batch.
        let mut empty = batch(4);
        empty.num_tokens = 0;
        assert!(empty.check().is_err());
    }

    #[test]
    fn a_packed_prefix_holds_the_text_the_conditions_and_one_row_of_zeros() {
        let latent = LatentShape {
            channels: 16,
            frames: 13,
            height: 24,
            width: 40,
        };
        let layout = Layout {
            video: latent,
            audio: None,
            text_rows: 512,
            condition_rows: 6144,
            condition_elements: 0,
        };

        assert_eq!(prefix_zero_row(&layout), 512 + 6144);
        assert_eq!(prefix_rows(&layout), 512 + 6144 + 1);
    }

    #[test]
    fn a_cache_refuses_a_layer_its_group_does_not_hold() {
        let plane = || column(DType::U8, 1);
        let cache = KvCache {
            block_tables: column(DType::U32, 4),
            page_size: 64,
            layers: vec![KvPlanes {
                key: plane(),
                value: plane(),
                key_initialized: plane(),
                value_initialized: plane(),
                key_scale: None,
                value_scale: None,
                index_keys: None,
            }],
            workspace: plane(),
        };

        assert!(cache.layer(0).is_ok());
        let error = cache.layer(2).err().unwrap().to_string();
        assert!(error.contains("layer 2"), "{error}");
    }
}
