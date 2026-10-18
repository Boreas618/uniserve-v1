//! Request identities and call descriptors.
//!
//! A [`Call`] is one immutable computation of a request ([`RequestKey`]):
//! its [`CallKind`], the [`CallCoordinates`] it executes at, the products it
//! reads and writes, the [`Bounds`] the scheduler reserved for it and, for a
//! sampling call, its random-draw coordinates ([`Rng`]) and token processor
//! inputs ([`SamplingState`]). A [`Batch`] carries the calls of one
//! numerical call.
//!
//! Types with invariants carry a `validate` method, which
//! [`Batch::validate`] reaches. The `ensure_valid!` and `invalid_message!`
//! macros come from the crate root.

use super::*;

/// `(engine_id, request_id, request_epoch)`.
///
/// The scheduler stamps each admission with the next value of its epoch
/// counter, so a reused request identifier yields a distinct key and no call
/// or product reference aliases across requests or epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestKey {
    /// Engine instance that allocates request identifiers.
    pub engine_id: u64,
    /// Identifier assigned to the request within its engine instance.
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

impl ForwardMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::TokenDenoising => "token_denoising",
        }
    }
}

/// A concrete media-input, encoder, diffusion, decoder, or media-output
/// computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum MediaCall {
    /// Decodes a video request's condition media on a host rank into the
    /// inputs of the vision and latent encoders.
    MediaReading,
    VisionEncoding,
    LatentEncoding,
    TextEncoding,
    LatentPreparation,
    Denoising,
    ImageDecoding,
    VideoDecoding,
    AudioDecoding,
    VideoEncoding,
    AudioEncoding,
    Muxing,
}

impl MediaCall {
    /// Every media call in declaration order.
    pub const ALL: [Self; 12] = [
        Self::MediaReading,
        Self::VisionEncoding,
        Self::LatentEncoding,
        Self::TextEncoding,
        Self::LatentPreparation,
        Self::Denoising,
        Self::ImageDecoding,
        Self::VideoDecoding,
        Self::AudioDecoding,
        Self::VideoEncoding,
        Self::AudioEncoding,
        Self::Muxing,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MediaReading => "media_reading",
            Self::VisionEncoding => "vision_encoding",
            Self::LatentEncoding => "latent_encoding",
            Self::TextEncoding => "text_encoding",
            Self::LatentPreparation => "latent_preparation",
            Self::Denoising => "denoising",
            Self::ImageDecoding => "image_decoding",
            Self::VideoDecoding => "video_decoding",
            Self::AudioDecoding => "audio_decoding",
            Self::VideoEncoding => "video_encoding",
            Self::AudioEncoding => "audio_encoding",
            Self::Muxing => "muxing",
        }
    }
}

/// The storage action performed between concrete producers and consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum TransferMode {
    Tensor,
    CacheExport,
    CacheInstall,
}

impl TransferMode {
    /// Every transfer mode in declaration order.
    pub const ALL: [Self; 3] = [Self::Tensor, Self::CacheExport, Self::CacheInstall];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::CacheExport => "cache_export",
            Self::CacheInstall => "cache_install",
        }
    }
}

/// Exactly one computation classification. The sum preserves the distinct
/// forward, media, and storage contracts without parallel opcode metadata.
///
/// Serde represents a kind by its inner enum's snake_case name alone
/// (`untagged`), which round-trips only while the names of [`ForwardMode`],
/// [`MediaCall`], and [`TransferMode`] stay disjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CallKind {
    Forward(ForwardMode),
    Media(MediaCall),
    Transfer(TransferMode),
}

impl CallKind {
    /// Computations accepted as individual scheduled request items. A rank's
    /// reports list its supported calls in this order.
    pub const ALL: [Self; 19] = [
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
        Self::Media(MediaCall::ImageDecoding),
        Self::Media(MediaCall::VideoDecoding),
        Self::Media(MediaCall::AudioDecoding),
        Self::Media(MediaCall::VideoEncoding),
        Self::Media(MediaCall::AudioEncoding),
        Self::Media(MediaCall::Muxing),
        Self::Transfer(TransferMode::Tensor),
        Self::Transfer(TransferMode::CacheExport),
        Self::Transfer(TransferMode::CacheInstall),
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

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward(mode) => mode.as_str(),
            Self::Media(call) => call.as_str(),
            Self::Transfer(mode) => mode.as_str(),
        }
    }
}

/// Hard resource maxima the scheduler reserves before a call runs.
///
/// [`Call::validate`] checks `max_tokens` against `input_token_ids` and
/// `max_latent_bytes` against encoder, latent, and image outputs;
/// [`Batch::validate`] checks `max_transfer_bytes` against installed cache
/// inputs. Neither checks `max_cache_units` or `max_completion_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Bounds {
    /// Maximum tokens the call may process or produce.
    pub max_tokens: u32,
    /// Paged-KV units newly allocated to the request's tables and first
    /// declared to workers by this call.
    pub max_cache_units: u32,
    /// Maximum latent storage in bytes.
    pub max_latent_bytes: u64,
    /// Maximum host-visible completion data in bytes.
    pub max_completion_bytes: u64,
    /// Maximum cross-call transfer data in bytes.
    pub max_transfer_bytes: u64,
}

/// How the common sampler consumes deterministic random draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DrawLayout {
    /// One draw used to sample a target token.
    TargetSampling = 0,
    /// Draw sequence used to propose speculative tokens.
    SpeculativeProposal = 1,
    /// Draw sequence used to generate diffusion noise.
    FlowNoise = 2,
}

/// Deterministic random-draw coordinates for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rng {
    /// Request-level random seed.
    pub seed: u64,
    /// First semantic draw index assigned to the call.
    pub semantic_index_base: u64,
    /// Mapping from semantic work to deterministic draws.
    pub draw_layout: DrawLayout,
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
    /// Token identifiers that advance a structured-generation transition.
    pub transition_token_ids: Vec<u32>,
    /// Whether the current grammar state requires immediate termination.
    pub force_finish: bool,
}

impl SamplingState {
    /// Sorts and deduplicates token identifier sets for deterministic encoding.
    pub fn canonicalize(&mut self) {
        if let Some(allowed) = &mut self.allowed_token_ids {
            allowed.sort_unstable();
            allowed.dedup();
        }
        self.suppressed_token_ids.sort_unstable();
        self.suppressed_token_ids.dedup();
        self.finish_token_ids.sort_unstable();
        self.finish_token_ids.dedup();
        self.transition_token_ids.sort_unstable();
        self.transition_token_ids.dedup();
    }
}

/// The coordinates one call executes at.
///
/// A rank would otherwise derive these by chaining from the calls that
/// preceded it. The engine holds the request state they come from, so it states
/// them and the rank asserts its own ledger agrees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallCoordinates {
    /// Position of this call's first token in the request's logical sequence.
    pub logical_position: u32,
    /// Tokens whose prefix cache a numerical call may attend to at submission.
    pub cache_visible_len: u32,
    /// Tokens whose prefix cache is initialized at submission; never below the visible extent.
    pub cache_computed_len: u32,
    /// Denoising steps completed for this request at submission.
    pub flow_step: u32,
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
/// The request's admitted `ArRequestParams::canvas` fixes the sampling; the
/// worker holds the canvas, its self-conditioning input and its stopping
/// history in the request's slot. `block` counts the blocks already
/// committed to the request's context and `step` the steps already run on
/// this canvas; step zero starts the canvas from random tokens. The step
/// that stops the canvas, by convergence or by reaching the admitted step
/// limit, reports the canvas's argmax tokens as its committed tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CanvasStep {
    /// Blocks committed to the request's context before this canvas.
    pub block: u32,
    /// Denoising steps already run on this canvas.
    pub step: u32,
}

/// Candidate log-probabilities a token-denoising call reads at canvas slots.
///
/// The call's `input_token_ids` hold its canvas rows back to back. Slot `i`
/// is canvas token `slot_tokens[i]` of that sequence and reads the candidate
/// token ids `candidate_ids[candidate_offsets[i]..candidate_offsets[i + 1]]`.
/// The worker reports, in `candidate_ids` order, each candidate's natural-log
/// probability under the log-softmax over the full vocabulary of the model's
/// logits at its slot (`RequestOutput::candidate_logprobs`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Readout {
    /// Index of each slot's token in the call's concatenated canvas rows.
    pub slot_tokens: Vec<u32>,
    /// Offsets of each slot's candidates in `candidate_ids`: one more entry
    /// than there are slots, starting at zero and ending at the id count.
    pub candidate_offsets: Vec<u32>,
    /// Candidate token ids of every slot, slot after slot.
    pub candidate_ids: Vec<u32>,
}

impl Readout {
    /// Number of candidate log-probabilities the readout reports.
    pub fn candidate_count(&self) -> usize {
        self.candidate_ids.len()
    }

    /// Validates the slot layout against a call of `canvas_tokens` canvas
    /// tokens.
    ///
    /// Slots address canvas tokens in increasing order, each reads at least
    /// one candidate, and the offsets partition `candidate_ids` in slot
    /// order.
    pub fn validate(&self, canvas_tokens: usize) -> ValidationResult<()> {
        ensure_valid!(!self.slot_tokens.is_empty(), "readout reads no slot");
        ensure_valid!(
            self.candidate_offsets.len() == self.slot_tokens.len() + 1,
            "readout candidate offsets do not bound every slot"
        );
        ensure_valid!(
            self.candidate_offsets.first() == Some(&0)
                && self.candidate_offsets.last().map(|&end| end as usize)
                    == Some(self.candidate_ids.len())
                && self
                    .candidate_offsets
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "readout candidate offsets do not partition the candidates"
        );
        ensure_valid!(
            self.slot_tokens
                .iter()
                .all(|&token| (token as usize) < canvas_tokens),
            "readout slot lies outside the call's canvas rows"
        );
        ensure_valid!(
            self.slot_tokens.windows(2).all(|pair| pair[0] < pair[1]),
            "readout slots are not in canvas order"
        );
        Ok(())
    }
}

/// Component used when a deployment does not partition a model by capability.
pub const DEFAULT_COMPONENT: &str = "model";

/// One image block of a context prefill.
///
/// The block writes the vision-encoder product `feature` into the prefix cache as one
/// attention block, before the call's input token `offset`: the call's
/// context is its input tokens `[0, offset)`, then the block, then the
/// tokens from `offset` on. Blocks sharing an offset follow in list order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionInput {
    /// Input tokens of the call that precede the block.
    pub offset: u32,
    /// Vision-encoder features the block injects.
    pub feature: TensorRef,
}

/// One immutable computation with its identity, data dependencies, and output limits.
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
    pub code: CallKind,
    /// Scheduler-declared limits for the computation and its outputs.
    pub bounds: Bounds,
    /// Data dependencies consumed in input order.
    pub inputs: Vec<TensorRef>,
    /// Outputs allocated to this producer.
    pub outputs: Vec<TensorRef>,
    /// Acknowledgment slots of the ranks that read this call's products.
    ///
    /// A producing rank cannot name its readers: it knows its own component,
    /// not which component consumes what it publishes, and a product is read
    /// in a later batch than the one producing it. The head states them, and
    /// a published product retires once each has acknowledged it. The
    /// executing rank's own slot is never listed.
    pub consumer_slots: Vec<u32>,
    /// Source token scalar transported between components.
    pub token_input: Option<TensorRef>,
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
    /// Diffusion trajectory consumed by this computation.
    pub latent_input: Option<TensorRef>,
    /// Diffusion trajectory produced by this computation.
    pub latent_output: Option<TensorRef>,
    /// Resident image consumed by feedback encoding.
    pub image_input: Option<TensorRef>,
    /// Decoded image retained for feedback.
    pub image_output: Option<TensorRef>,
    /// Boolean device completion consumed by a dependent computation.
    pub completion_output: Option<TensorRef>,
    /// Boolean device decision that selects an image transition.
    pub transition_output: Option<TensorRef>,
    /// Device predicate that enables execution: a U8 flag or a packed I64
    /// continuation scalar. A false predicate yields [`CallStatus::Predicated`].
    pub predicate: Option<TensorRef>,
    /// Deterministic sampling coordinates.
    pub rng: Option<Rng>,
    /// Host-side sampler constraints for this computation; absent uses admission defaults.
    pub sampling_state: Option<SamplingState>,
    /// Host-known prompt, draft, or decode input tokens in model input order,
    /// or a token-denoising call's canvas rows back to back. Empty when
    /// continuation reads its predecessor's device token.
    pub input_token_ids: Vec<u32>,
    /// Candidate slots a token-denoising readout reports; absent for every
    /// other call.
    pub readout: Option<Readout>,
    /// The step a token-denoising call runs on its request's generation
    /// canvas; absent for every other call.
    pub canvas: Option<CanvasStep>,
    /// Encoded source image in base64, consumed by a vision or latent encoder.
    /// Dispatch and in-flight matching share these immutable bytes.
    pub input_image: Option<std::sync::Arc<str>>,
    /// Published or installed cache extent consumed by this computation.
    pub cache_input: Option<BufferId>,
    /// Identity under which this computation publishes or installs cache state.
    pub cache_output: Option<BufferId>,
}

impl Call {
    /// Create a computation with no inputs or outputs and zero resource bounds.
    /// The caller fills its numerical coordinates and limits before submission.
    pub fn new(request_key: RequestKey, call_id: CallId, code: CallKind) -> Self {
        Self {
            request_key,
            call_id,
            code,
            component: DEFAULT_COMPONENT.into(),
            coordinates: CallCoordinates::default(),
            bounds: Bounds::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            consumer_slots: Vec::new(),
            token_input: None,
            token_output: None,
            vision_inputs: Vec::new(),
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,
            predicate: None,
            rng: None,
            sampling_state: None,
            input_token_ids: Vec::new(),
            readout: None,
            canvas: None,
            input_image: None,
            cache_input: None,
            cache_output: None,
        }
    }

    /// A context prefill interleaves prompt tokens and input-image blocks.
    pub fn writes_context(&self) -> bool {
        self.code == CallKind::Forward(ForwardMode::Prefill)
            && !self.vision_inputs.is_empty()
            && self.completion_output.is_none()
    }

    /// Image feedback or a latent input writes one visual-state row.
    /// Context-prefill vision blocks instead belong to its prompt segments.
    pub fn writes_visual_state(&self) -> bool {
        !self.writes_context()
            && (!self.vision_inputs.is_empty() || self.latent_feature_input.is_some())
    }

    /// Tensor dependencies in the computation signature, excluding its predicate.
    ///
    /// Includes `token_input` and `latent_input`, which [`Self::buffer_inputs`]
    /// omits.
    pub fn tensor_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.token_input.iter())
            .chain(self.vision_inputs.iter().map(|input| &input.feature))
            .chain(self.latent_feature_input.iter())
            .chain(self.latent_input.iter())
            .chain(self.image_input.iter())
    }

    /// All tensor declarations owned by this computation.
    ///
    /// Includes the device scalars (token, completion, transition) and
    /// `latent_output`, which [`Self::buffer_outputs`] omits.
    pub fn tensor_outputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.outputs
            .iter()
            .chain(self.token_output.iter())
            .chain(self.completion_output.iter())
            .chain(self.transition_output.iter())
            .chain(self.encoder_output.iter())
            .chain(self.latent_output.iter())
            .chain(self.image_output.iter())
    }

    /// Mutable declarations while assigning computation identities before submission.
    pub fn tensor_outputs_mut(&mut self) -> impl Iterator<Item = &mut TensorRef> {
        self.outputs
            .iter_mut()
            .chain(self.token_output.iter_mut())
            .chain(self.completion_output.iter_mut())
            .chain(self.transition_output.iter_mut())
            .chain(self.encoder_output.iter_mut())
            .chain(self.latent_output.iter_mut())
            .chain(self.image_output.iter_mut())
    }

    /// Tensor inputs that require persistent destination storage.
    pub fn buffer_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.vision_inputs.iter().map(|input| &input.feature))
            .chain(self.latent_feature_input.iter())
            .chain(self.image_input.iter())
    }

    /// Tensor outputs backed by scheduler-allocated persistent buffers.
    ///
    /// [`Batch::validate`] requires a [`BufferAllocation`] large enough for each
    /// of these. Latent trajectories are addressed through [`LatentParams`]
    /// instead.
    pub fn buffer_outputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.outputs
            .iter()
            .chain(self.encoder_output.iter())
            .chain(self.image_output.iter())
    }

    /// Exact data and predicate dependencies, independent of physical representation.
    pub fn input_buffers(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.tensor_inputs()
            .chain(self.predicate.iter())
            .map(TensorRef::buffer_id)
            .chain(self.cache_input)
    }

    /// Resource identities published by this computation.
    pub fn output_buffers(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.tensor_outputs()
            .map(TensorRef::buffer_id)
            .chain(self.cache_output)
    }

    /// Returns whether successful execution advances request state.
    pub const fn advances_state(&self) -> bool {
        self.code.advances_state()
    }

    /// Validates the invariants one call carries on its own.
    ///
    /// Checks identity, component, coordinates, the token bound, canonical
    /// sampling-state sets, the encoded-image source, cache export
    /// identities, product ownership, generations, output indices, dtypes and
    /// byte bounds, cross-request inputs, and the predicate. `rng` is not
    /// checked. Relationships to other calls and to the batch's allocation
    /// tables belong to [`Batch::validate`].
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
            "input token count exceeds the computation token bound"
        );
        if let Some(state) = &self.sampling_state {
            for ids in [
                state.allowed_token_ids.as_deref(),
                Some(state.suppressed_token_ids.as_slice()),
                Some(state.finish_token_ids.as_slice()),
                Some(state.transition_token_ids.as_slice()),
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
                        self.code,
                        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
                    )
                    && self.image_input.is_none(),
                "encoded image requires an image encoder without another image source"
            );
        }

        // A token-denoising call either reads candidate log-probabilities at
        // slots of the canvas rows it carries, or runs one step of its
        // request's resident generation canvas of `max_tokens` tokens. Either
        // way it produces no sampled token output.
        let denoises = self.code == CallKind::Forward(ForwardMode::TokenDenoising);
        ensure_valid!(
            u8::from(self.readout.is_some()) + u8::from(self.canvas.is_some())
                == u8::from(denoises),
            "a token-denoising call carries exactly one readout or canvas step"
        );
        ensure_valid!(
            !denoises || (self.token_output.is_none() && self.transition_output.is_none()),
            "token denoising samples no token output"
        );
        if let Some(readout) = &self.readout {
            ensure_valid!(
                !self.input_token_ids.is_empty(),
                "a readout requires canvas tokens"
            );
            readout.validate(self.input_token_ids.len())?;
        }
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
        let publishes_cache = matches!(
            self.code,
            CallKind::Transfer(TransferMode::CacheExport | TransferMode::CacheInstall)
        );
        ensure_valid!(
            self.cache_output.is_some() == publishes_cache,
            "cache export or installation requires one cache output identity"
        );
        if let Some(output) = self.cache_output {
            output.validate()?;
            ensure_valid!(
                output.owner == self.request_key && output.producer_call_id == self.call_id,
                "cache output is not owned by its producing computation"
            );
            output_indices.insert(output.output_index);
        }
        let consumes_cache = matches!(
            self.code,
            CallKind::Transfer(TransferMode::CacheInstall)
                | CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
        );
        if let Some(input) = self.cache_input {
            input.validate()?;
            ensure_valid!(
                consumes_cache && input.owner == self.request_key,
                "cache input is incompatible with its computation or request"
            );
        }
        ensure_valid!(
            self.code != CallKind::Transfer(TransferMode::CacheInstall)
                || self.cache_input.is_some(),
            "cache installation requires a source export"
        );

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

        for output in self
            .encoder_output
            .iter()
            .chain(self.latent_output.iter())
            .chain(self.image_output.iter())
        {
            ensure_valid!(
                output.max_bytes() <= self.bounds.max_latent_bytes,
                "image or trajectory output exceeds its declared byte capacity"
            );
        }
        if let Some(input) = &self.token_input {
            ensure_valid!(
                self.code == CallKind::Transfer(TransferMode::Tensor),
                "token transfer input requires a tensor-transfer computation"
            );
            ensure_valid!(
                input.dtype == DType::I64 && input.shape_bound.max_elements() == 1,
                "token transfer input requires one int64 element"
            );
        }
        for tensor in self.token_output.iter() {
            ensure_valid!(
                tensor.dtype == DType::I64 && tensor.shape_bound.max_elements() == 1,
                "device token relay requires one int64 element"
            );
        }
        for tensor in self
            .completion_output
            .iter()
            .chain(self.transition_output.iter())
        {
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
                "computation predicate belongs to another request"
            );
            ensure_valid!(
                matches!(predicate.dtype, DType::U8 | DType::I64)
                    && predicate.shape_bound.max_elements() == 1,
                "device predicate requires a boolean or packed continuation scalar"
            );
        }
        Ok(())
    }
}
