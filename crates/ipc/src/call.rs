//! Request identities and call descriptors.
//!
//! A [`Call`] is one immutable call of a request ([`RequestKey`]):
//! its [`CallKind`], the [`CallCoordinates`] it executes at, the products it
//! reads and writes, the [`Bounds`] the scheduler reserved for it and, for a
//! sampling call, its token processor inputs ([`SamplingState`]).
//!
//! Types with invariants carry a `validate` method. The `ensure_valid!` and
//! `invalid_message!` macros come from the crate root.

use super::*;
use proto::config::DEFAULT_BLOCK;

/// `(engine_id, request_id, request_epoch)`.
///
/// The scheduler stamps each admission with the next value of its epoch
/// counter, so a reused request identifier yields a distinct key and no call
/// or product reference aliases across requests or epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestKey {
    /// Scheduler that allocates request identifiers.
    pub engine_id: u64,
    /// Identifier assigned to the request within its scheduler.
    pub request_id: RequestId,
    /// Admission generation that distinguishes reuse of the same request identifier.
    pub request_epoch: u64,
}

impl RequestKey {
    /// Constructs an identity from engine, request, and admission epoch.
    pub const fn new(engine_id: u64, request_id: RequestId, request_epoch: u64) -> Self {
        Self {
            engine_id,
            request_id,
            request_epoch,
        }
    }
}

/// The numerical mode of a token model forward.
///
/// `Prefill`, `Decode` and `Verify` extend a request's prefix cache causally.
/// `TokenDenoising` runs one denoising pass over rows of a token canvas that
/// read the request's cached prefix without writing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ForwardMode {
    Prefill,
    Decode,
    Verify,
    TokenDenoising,
}

impl ForwardMode {}

/// Exactly one call classification. The sum preserves the distinct
/// forward and media contracts without parallel opcode metadata.
///
/// Serde represents a kind by its inner enum's snake_case name alone
/// (`untagged`), which round-trips only while the names of [`ForwardMode`]
/// and [`MediaCall`] stay disjoint. A request's KV cache reaches another
/// worker group by the scheduler's device-direct hand-off, not by a call.
/// `cache_export` named a removed transfer kind and is not reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CallKind {
    Forward(ForwardMode),
    Media(MediaCall),
}

impl CallKind {
    /// Computations accepted as individual scheduled request items. A rank's
    /// reports list its supported calls in this order.
    pub const ALL: [Self; 15] = [
        Self::Forward(ForwardMode::Prefill),
        Self::Forward(ForwardMode::Decode),
        Self::Forward(ForwardMode::Verify),
        Self::Media(MediaCall::MediaReading),
        Self::Forward(ForwardMode::TokenDenoising),
        Self::Media(MediaCall::VisionEncoding),
        Self::Media(MediaCall::LatentEncoding),
        Self::Media(MediaCall::TextEncoding),
        Self::Media(MediaCall::LatentPreparation),
        Self::Media(MediaCall::Denoising),
        Self::Media(MediaCall::VideoDecoding),
        Self::Media(MediaCall::AudioDecoding),
        Self::Media(MediaCall::VideoEncoding),
        Self::Media(MediaCall::AudioEncoding),
        Self::Media(MediaCall::Muxing),
    ];

    /// Returns whether a call of this kind advances its request's state.
    ///
    /// A request's next call follows its latest state-advancing call, and
    /// only an `Ok` result of such a call moves the request's state.
    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::Forward(_) | Self::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
        )
    }
}

/// Hard resource maxima the scheduler reserves before a call runs.
///
/// [`Call::validate`] checks `max_tokens` against `input_token_ids`; it does
/// not check `max_completion_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Bounds {
    /// Maximum tokens the call may process or produce.
    pub max_tokens: u32,
    /// Prefix cache units newly allocated to the request's tables and first
    /// declared to the ranks by this call. The engine's own accounting: it
    /// is not encoded, since a rank learns the units from the batch's
    /// `new_cache_units`, and a decoded call states zero.
    #[serde(skip)]
    pub max_cache_units: u32,
    /// Maximum host-visible completion data in bytes.
    pub max_completion_bytes: u64,
}

/// Branch-local token processor inputs for one sampling call.
///
/// Token ids in every field are strictly increasing: [`Call::validate`]
/// rejects any other order, and [`Self::canonicalize`] produces it.
/// `allowed_token_ids` distinguishes no whitelist (`None`) from a present empty
/// whitelist, which deterministically represents an invalid all-masked
/// distribution. Penalty token counts are not carried here: they are a
/// device-resident committed base plus bounded per-call deltas folded after
/// sampling accepts tokens, so no host token history participates in a
/// successor's sampling input.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SamplingState {
    /// Optional whitelist of token identifiers eligible for sampling.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Token identifiers excluded from sampling.
    pub suppressed_token_ids: Vec<u32>,
    /// Token identifiers that terminate generation when selected.
    pub finish_token_ids: Vec<u32>,
    /// Whether the current grammar state requires immediate termination.
    pub force_finish: bool,
}

impl SamplingState {}

/// The coordinates one call executes at.
///
/// A rank would otherwise derive these by chaining from the calls that
/// preceded it. The engine holds the request state they come from, so it states
/// them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallCoordinates {
    /// Position of this call's first token in the request's logical sequence.
    pub logical_position: u32,
    /// Tokens whose prefix cache a numerical call may attend to at submission.
    pub cache_visible_len: u32,
    /// Tokens whose prefix cache is initialized at submission; never below the visible extent.
    pub cache_computed_len: u32,
}

impl CallCoordinates {
    /// Validates the containment the coordinates must satisfy.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.cache_visible_len <= self.cache_computed_len,
            "call coordinates place visible prefix cache beyond the computed extent"
        );
        Ok(())
    }
}

/// One denoising step of a block-diffusion request's resident canvas.
///
/// The deployment's launch configuration fixes the sampling
/// (`EngineConfig::canvas_sampling`); the worker holds the canvas, its
/// self-conditioning input and its stopping history in the request's slot.
/// `block` counts the blocks already committed to the request's context and
/// `step` the steps already run on this canvas; step zero starts the canvas
/// from random tokens. The step that stops the canvas, by convergence or by
/// reaching the launched step limit, reports the canvas's argmax tokens as
/// its committed tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CanvasStep {
    /// Blocks committed to the request's context before this canvas.
    pub block: u32,
    /// Denoising steps already run on this canvas.
    pub step: u32,
}

/// One image block of a context prefill.
///
/// The block writes the vision-encoder product `feature` into the prefix cache as one
/// attention block, before the call's input token `offset`: the call's
/// context is its input tokens `[0, offset)`, then the block, then the
/// tokens from `offset` on. Blocks sharing an offset follow in list order.
///
/// The block's `rows` take as many cache positions; its rows rotate from the
/// logical position after the context before it, and the text after it
/// continues `positions` logical positions later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionInput {
    /// Input tokens of the call that precede the block.
    pub offset: u32,
    /// Vision-encoder features the block injects.
    pub feature: TensorRef,
    /// Cache rows the block writes: the feature's rows.
    pub rows: u32,
    /// Logical positions the block advances the context by: its image's span when it is the
    /// image's last block, zero otherwise.
    pub positions: u32,
}

/// What a token call does for its request's draft cache, in a deployment
/// that speculates with its model's drafter.
///
/// The draft cache entry at position `i` is the drafter's evaluation of the
/// target's hidden state at `i` with the token at `i + 1`. A forward call
/// writes the entries of its positions whose next token it knows: the
/// tokens it evaluates, its own sampled token, or `lookahead`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftWork {
    /// Draft tokens the call proposes after its last committed token, which
    /// the request's next verification may verify; at most the deployment's
    /// draft limit, and zero for none.
    pub proposals: u32,
    /// The token after the call's last input token, which the draft cache
    /// entry of its last position reads; absent when the call samples that
    /// token itself or knows none.
    pub lookahead: Option<u32>,
    /// The first position of a call that evaluates tokens its request's
    /// cache already holds, its input tokens from there on, to fill the
    /// request's draft cache: it starts from the state checkpoint at that
    /// position (or from the beginning), writes no target K/V and leaves
    /// its request slot's state as it was. Absent for every other call.
    pub refill: Option<u32>,
}

/// One immutable call with its identity, data dependencies, and output limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Call {
    /// Request that owns the call.
    pub request_key: RequestKey,
    /// Logical batch and selection ordinal, preserved when the batch is
    /// projected onto a rank.
    pub call_id: CallId,
    /// Coordinates this call executes at, so a rank does not derive them.
    pub coordinates: CallCoordinates,
    /// Component bound within the selected Worker.
    pub component: String,
    /// Kind performed by this call.
    pub kind: CallKind,
    /// Scheduler-declared limits for the call and its outputs.
    pub bounds: Bounds,
    /// Data dependencies consumed in input order.
    pub inputs: Vec<TensorRef>,
    /// Outputs allocated to this producer.
    pub outputs: Vec<TensorRef>,
    /// Acknowledgment slots of the ranks that read this call's products.
    ///
    /// A producing rank cannot name its readers: it knows its own component,
    /// not which component consumes what it publishes, and a product is read
    /// in a later batch than the one producing it. The engine states them, and
    /// a published product retires once each has acknowledged it. The
    /// executing rank's own slot is never listed.
    pub consumer_slots: Vec<u32>,
    /// Sampled token and continuation bit in one I64 scalar: the token in
    /// the low 31 bits and continuation in bit 31.
    pub token_output: Option<TensorRef>,
    /// Image blocks a prefill injects, in context order: each block's
    /// vision-encoder feature and the input token it precedes.
    pub vision_inputs: Vec<VisionInput>,
    /// VAE features consumed by multimodal forward.
    pub latent_feature_input: Option<TensorRef>,
    /// Features produced by the selected image encoder.
    pub encoder_output: Option<TensorRef>,
    /// Boolean device completion consumed by a dependent call.
    pub completion_output: Option<TensorRef>,
    /// Device predicate that enables execution: a U8 flag or a packed I64
    /// continuation scalar. A false predicate yields [`CallStatus::Predicated`].
    pub predicate: Option<TensorRef>,
    /// Host-side sampler constraints for this call; absent uses admission defaults.
    pub sampling_state: Option<SamplingState>,
    /// Host-known prompt, draft, or decode input tokens in model input order.
    /// Empty when continuation reads its predecessor's device token, and for
    /// a token-denoising call, which denoises its resident canvas.
    pub input_token_ids: Vec<u32>,
    /// The step a token-denoising call runs on its request's generation
    /// canvas; absent for every other call.
    pub canvas: Option<CanvasStep>,
    /// Encoded source image in base64, consumed by a vision or latent encoder.
    /// Dispatch and in-flight matching share these immutable bytes.
    pub input_image: Option<std::sync::Arc<str>>,
    /// Published cache extent consumed by this call (a latent preparation or denoising step).
    pub cache_input: Option<BufferId>,
    /// What the call does for its request's draft cache; the default, for
    /// a deployment that does not speculate, does nothing.
    pub draft: DraftWork,
    /// The stored trajectory a resumed request's latent preparation opens
    /// at, instead of step zero, a video request's; absent for every other
    /// call.
    pub latent_restore: Option<LatentRestore>,
    /// The host encoder pool pages an encoder call (a video request's, or
    /// an input image's encoding) reads its output from instead of computing it, or
    /// writes the output it computed to; absent for every other call.
    pub host_output: Option<HostOutput>,
    /// The posterior draw an image latent encoding samples its latent with:
    /// the encoding's ordinal among its request's latent encodings, its
    /// input images' latent encoders in context order. The
    /// engine derives it from the request alone, so the rank keys the draw on
    /// the request's seed and this ordinal, whichever encodings an
    /// encoder cache spared and whichever engine the request resumed on.
    /// Absent on every other call.
    pub latent_draw: Option<u32>,
}

/// Where an encoder call's output lies in the host encoder pool
/// (`Call::host_output`).
///
/// The pool stores an encoder's whole output, its rows in order, in pages
/// of their own: the `i`-th page holds its bytes `i * page_bytes` up to
/// `(i + 1) * page_bytes`, the last one their remainder. A video encoder
/// call's output holds the stored output's rows `rows[0]` up to `rows[1]` (a
/// latent encoding round's units, or all of them), its row `r` being the
/// stored row `rows[0] + r`, so the call reads or writes the bytes of those
/// rows in the pages they fall on. An image encoding's output is the whole
/// stored output, whose rows the worker prepares from the image: at most
/// `rows[1]`, the cache tokens of the block the features write, which the
/// pages are sized by, from `rows[0] = 0` on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostOutput {
    /// The pool at the call's ranks' host's Storage Server
    /// (`EncoderPoolInfo::pool`).
    pub pool: u64,
    /// Whether every rank of the call reads its output's rows from the
    /// pages instead of computing them; otherwise the call computes its
    /// output and the rank that wrote rows writes them to the pages.
    pub import: bool,
    /// The bytes of one row.
    pub row_bytes: u64,
    /// The stored output's rows the call's output holds: its first and its
    /// end row; for an image encoding, its first row and a bound of its
    /// end.
    pub rows: [u32; 2],
    /// The pool slot of each page of the stored output; `None` for a page
    /// an export leaves unwritten, as the pool holds it already. An import
    /// names every page.
    pub slots: Vec<Option<u64>>,
}

impl HostOutput {
    /// Validates that the rows are nonempty, that every page is named for
    /// an import, and, given the pool's `page_bytes`, that the pages hold
    /// the stored output's rows up to the call's last one.
    pub fn validate(&self, page_bytes: Option<u64>) -> ValidationResult<()> {
        let [first, end] = self.rows;
        ensure_valid!(
            first < end && self.row_bytes > 0 && !self.slots.is_empty(),
            "a host output names no rows or no pages"
        );
        ensure_valid!(
            !self.import || self.slots.iter().all(Option::is_some),
            "an imported host output names every page"
        );
        if let Some(page_bytes) = page_bytes {
            ensure_valid!(
                u64::from(end) * self.row_bytes <= self.slots.len() as u64 * page_bytes,
                "a host output's pages do not hold its rows"
            );
        }
        Ok(())
    }
}

impl Call {
    /// Element types of the scalar outputs one call relays through a relay
    /// entry of its request slot: the token (`token_output`) and the
    /// completion (`completion_output`), which [`Self::validate`] requires of
    /// them.
    pub const RELAY_DTYPES: [DType; 2] = [DType::I64, DType::U8];

    /// Bytes one relay entry of a request slot holds of a call's relayed
    /// scalars.
    ///
    /// A rank sizes its request-relay arenas from it, one row per (request
    /// slot, relay entry) pair.
    pub fn relay_row_bytes() -> u64 {
        Self::RELAY_DTYPES
            .iter()
            .map(|dtype| dtype.element_bytes())
            .sum()
    }

    /// Create a call with no inputs or outputs and zero resource bounds.
    /// The caller fills its numerical coordinates and limits before submission.
    pub fn new(request_key: RequestKey, call_id: CallId, kind: CallKind) -> Self {
        Self {
            request_key,
            call_id,
            kind,
            component: DEFAULT_BLOCK.into(),
            coordinates: CallCoordinates::default(),
            bounds: Bounds::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            consumer_slots: Vec::new(),
            token_output: None,
            vision_inputs: Vec::new(),
            latent_feature_input: None,
            encoder_output: None,
            completion_output: None,
            predicate: None,
            sampling_state: None,
            input_token_ids: Vec::new(),
            canvas: None,
            input_image: None,
            cache_input: None,
            draft: DraftWork::default(),
            latent_restore: None,
            host_output: None,
            latent_draw: None,
        }
    }

    /// Tensor dependencies in the call's signature, excluding its predicate.
    pub fn tensor_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.vision_inputs.iter().map(|input| &input.feature))
            .chain(self.latent_feature_input.iter())
    }

    /// All tensor declarations owned by this call.
    ///
    /// Includes the device scalars (token, completion), which
    /// [`Self::buffer_outputs`] omits.
    pub fn tensor_outputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.outputs
            .iter()
            .chain(self.token_output.iter())
            .chain(self.completion_output.iter())
            .chain(self.encoder_output.iter())
    }

    /// Mutable declarations while assigning call identities before submission.
    pub fn tensor_outputs_mut(&mut self) -> impl Iterator<Item = &mut TensorRef> {
        self.outputs
            .iter_mut()
            .chain(self.token_output.iter_mut())
            .chain(self.completion_output.iter_mut())
            .chain(self.encoder_output.iter_mut())
    }

    /// Tensor inputs that require persistent destination storage.
    pub fn buffer_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.vision_inputs.iter().map(|input| &input.feature))
            .chain(self.latent_feature_input.iter())
    }

    /// Tensor outputs backed by scheduler-allocated persistent buffers.
    ///
    /// Each needs a [`BufferAllocation`] large enough for it.
    pub fn buffer_outputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.outputs.iter().chain(self.encoder_output.iter())
    }

    /// Exact data and predicate dependencies, independent of physical representation.
    pub fn input_buffers(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.tensor_inputs()
            .chain(self.predicate.iter())
            .map(TensorRef::buffer_id)
            .chain(self.cache_input)
    }

    /// Resource identities published by this call.
    pub fn output_buffers(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.tensor_outputs().map(TensorRef::buffer_id)
    }

    /// Returns whether successful execution advances request state.
    pub const fn advances_state(&self) -> bool {
        self.kind.advances_state()
    }

    /// Validates the invariants one call carries on its own.
    ///
    /// Checks identity, component, coordinates, the token bound, canonical
    /// sampling-state sets, the encoded-image source, product ownership, generations, output indices, dtypes,
    /// cross-request inputs, the predicate, and the stored trajectory of a
    /// latent preparation. Relationships to other calls and to the
    /// batch's allocation tables are the scheduler's to keep.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish call identity, component, coordinates, and token capacity.
        ensure_valid!(self.call_id.batch_id > 0, "call id must be positive");
        ensure_valid!(
            !self.component.is_empty(),
            "call component must not be empty"
        );
        self.coordinates.validate()?;

        ensure_valid!(
            self.input_token_ids.len() <= self.bounds.max_tokens as usize,
            "input token count exceeds the call's token bound"
        );
        if let Some(state) = &self.sampling_state {
            for ids in [
                state.allowed_token_ids.as_deref(),
                Some(state.suppressed_token_ids.as_slice()),
                Some(state.finish_token_ids.as_slice()),
            ]
            .into_iter()
            .flatten()
            {
                ensure_valid!(
                    ids.windows(2).all(|pair| pair[0] < pair[1]),
                    "sampling-state token ids are not canonical"
                );
            }
        }

        if let Some(image) = &self.input_image {
            ensure_valid!(
                !image.is_empty()
                    && matches!(
                        self.kind,
                        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
                    ),
                "an encoded image requires an image encoder"
            );
        }
        ensure_valid!(
            self.latent_draw.is_none() || self.kind == CallKind::Media(MediaCall::LatentEncoding),
            "only a latent encoding names a posterior draw"
        );

        // A token-denoising call runs one step of its request's resident
        // generation canvas of `max_tokens` tokens, and produces no sampled
        // token output.
        let denoises = self.kind == CallKind::Forward(ForwardMode::TokenDenoising);
        ensure_valid!(
            self.canvas.is_some() == denoises,
            "a token-denoising call carries a canvas step, and no other call does"
        );
        ensure_valid!(
            !denoises || self.token_output.is_none(),
            "token denoising samples no token output"
        );
        if self.canvas.is_some() {
            ensure_valid!(
                self.input_token_ids.is_empty() && self.bounds.max_tokens > 0,
                "a canvas step denoises its resident canvas of max_tokens tokens"
            );
        }

        // Every output must be uniquely owned by this producer and fit the
        // resource class reserved for the call. The cache output shares the
        // output-index space with the tensor outputs.
        let mut output_indices = HashSet::with_capacity(self.outputs.len());
        let consumes_cache = matches!(
            self.kind,
            CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
        );
        if let Some(input) = self.cache_input {
            input.validate()?;
            ensure_valid!(
                consumes_cache && input.owner == self.request_key,
                "cache input is incompatible with its call or request"
            );
        }

        for output in self.tensor_outputs() {
            output.validate()?;
            ensure_valid!(
                output.request_key == self.request_key && output.producer_call_id == self.call_id,
                "an output product is not owned by its producing call"
            );
            ensure_valid!(
                output.generation > 0,
                "an output product has no logical generation"
            );
            ensure_valid!(
                output_indices.insert(output.output_index),
                "call repeats an output index"
            );
        }

        for tensor in self.token_output.iter() {
            ensure_valid!(
                tensor.dtype == DType::I64 && tensor.shape_bound.max_elements() == 1,
                "device token relay requires one int64 element"
            );
        }
        for tensor in self.completion_output.iter() {
            ensure_valid!(
                tensor.dtype == DType::U8 && tensor.shape_bound.max_elements() == 1,
                "device completion requires one uint8 element"
            );
        }

        // Only encoder features (vision and latent-feature inputs) can be
        // shared across requests.
        for input in self.tensor_inputs() {
            input.validate()?;
            ensure_valid!(
                input.request_key == self.request_key
                    || self
                        .vision_inputs
                        .iter()
                        .any(|vision| &vision.feature == input)
                    || Some(input) == self.latent_feature_input.as_ref(),
                "request-local tensor belongs to another request"
            );
        }
        if let Some(predicate) = &self.predicate {
            predicate.validate()?;
            ensure_valid!(
                predicate.request_key == self.request_key,
                "call predicate belongs to another request"
            );
            ensure_valid!(
                matches!(predicate.dtype, DType::U8 | DType::I64)
                    && predicate.shape_bound.max_elements() == 1,
                "device predicate requires a boolean or packed continuation scalar"
            );
        }

        // Only a latent preparation, which opens a trajectory, opens a
        // stored one.
        if let Some(restore) = &self.latent_restore {
            restore.validate()?;
            ensure_valid!(
                self.kind == CallKind::Media(MediaCall::LatentPreparation),
                "only a latent preparation restores a stored trajectory"
            );
        }
        // Only an encoder call that writes one output has its output in the
        // host encoder pool: a video encoder call's one output, or an input
        // image encoding's features.
        if let Some(output) = &self.host_output {
            output.validate(None)?;
            let one_output = match self.encoder_output {
                Some(_) => {
                    self.outputs.is_empty() && self.kind != CallKind::Media(MediaCall::TextEncoding)
                }
                None => self.outputs.len() == 1,
            };
            ensure_valid!(
                matches!(
                    self.kind,
                    CallKind::Media(
                        MediaCall::VisionEncoding
                            | MediaCall::TextEncoding
                            | MediaCall::LatentEncoding
                    )
                ) && one_output,
                "only an encoder call of one output has a host output"
            );
        }
        Ok(())
    }
}

/// A concrete media-input, encoder, diffusion, decoder, or media-output
/// computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum MediaCall {
    /// Decodes a video request's condition media on a CPU worker into the
    /// inputs of the vision and latent encoders.
    MediaReading,
    VisionEncoding,
    LatentEncoding,
    TextEncoding,
    LatentPreparation,
    Denoising,
    VideoDecoding,
    AudioDecoding,
    VideoEncoding,
    AudioEncoding,
    Muxing,
}

impl MediaCall {
    /// Every media call in declaration order.
    pub const ALL: [Self; 11] = [
        Self::MediaReading,
        Self::VisionEncoding,
        Self::LatentEncoding,
        Self::TextEncoding,
        Self::LatentPreparation,
        Self::Denoising,
        Self::VideoDecoding,
        Self::AudioDecoding,
        Self::VideoEncoding,
        Self::AudioEncoding,
        Self::Muxing,
    ];

    /// The calls that read this call's products in a video request's graph,
    /// in the order the graph consumes them.
    ///
    /// The engine plans a request's consumption from this table, and both
    /// the engine and the worker derive from it which ranks back a
    /// component's persistent products: the ranks that execute the
    /// component and those that execute a reader of its products. A reader
    /// of resident state ([`Self::reads_resident_state`]) reads the
    /// denoiser's latent pages instead of a persistent product.
    pub const fn readers(self) -> &'static [Self] {
        match self {
            Self::MediaReading => &[Self::VisionEncoding, Self::LatentEncoding],
            Self::VisionEncoding => &[Self::TextEncoding],
            Self::TextEncoding | Self::LatentEncoding => &[Self::LatentPreparation],
            Self::LatentPreparation => &[Self::Denoising],
            Self::Denoising => &[Self::Denoising, Self::VideoDecoding, Self::AudioDecoding],
            Self::VideoDecoding => &[Self::VideoEncoding],
            Self::VideoEncoding => &[Self::Muxing],
            Self::AudioDecoding => &[Self::AudioEncoding],
            Self::AudioEncoding | Self::Muxing => &[],
        }
    }

    /// Whether this call reads its predecessor's trajectory from the
    /// denoiser's resident latent pages instead of a persistent product.
    pub const fn reads_resident_state(self) -> bool {
        matches!(self, Self::Denoising)
    }
}

impl From<proto::request::ImageEncoder> for MediaCall {
    /// The call an image encoder of a request runs.
    fn from(encoder: proto::request::ImageEncoder) -> Self {
        match encoder {
            proto::request::ImageEncoder::Vision => Self::VisionEncoding,
            proto::request::ImageEncoder::Latent => Self::LatentEncoding,
        }
    }
}
