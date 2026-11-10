//! Generation lifecycle state and worker-call planning.
//!
//! A request advances through context ingestion, then either understanding
//! decode or block-diffusion canvases of text ([`GenerationPhase`]).
//! [`RequestState`] holds the accepted progress of one admitted token request.
//!
//! Each `plan_*` builder returns one call whose identities are placeholders
//! and whose `Bounds` come from `finish_plan`; `register_call` then stamps the
//! request key and assigns product generations. Every planned output is named
//! by request key (which includes the request epoch), producer call, output
//! index, and generation.
//!
//! On completion, `validate_generation_result` checks a worker result against
//! the submitted call and the request's limits, and
//! `RequestState::process_generation_result` applies its accepted progress.
//! Token semantics (stop conditions, public events) are resolved afterwards
//! in `output`.

use super::{EncoderCachePin, Phase, RequestAllocations, Terminal};
use crate::completion::CallCompletion;
use crate::kv_cache::{BlockTable, CacheAllocation};

use ipc::{ForwardMode, MediaCall};

use crate::{GenerationRequest, ImageEncoderInput, ImageInput};
use ipc::{
    Bounds, Call, CallId, CallKind, CallStatus, CanvasStep, DimBound, RequestKey, SamplingState,
    ShapeBound, TensorRef, VisionInput,
};
use ipc::{CanvasSampling, RequestId, SamplingParams};
use proto::tensor::DType;

/// Builds an unstamped product reference for a planned call output.
///
/// The owning `request_key` and `producer_call_id` are placeholders until
/// [`register_call`] stamps the real identity, and generation zero marks the
/// product for a fresh generation there. The shape bound is empty: it carries
/// identity, not a device geometry. `output_index` must be unique among the
/// call's outputs, including `cache_output`; `Call::validate` rejects repeats.
fn output_tensor(output_index: u16, dtype: DType) -> TensorRef {
    TensorRef {
        request_key: RequestKey::new(0, RequestId(0), 0),
        producer_call_id: CallId::new(0, 0),
        output_index,
        generation: 0,
        dtype,
        shape_bound: ShapeBound::default(),
    }
}

/// Builds a product with an explicit byte bound.
fn bounded_tensor(output_index: u16, dtype: DType, shape_bound: ShapeBound) -> TensorRef {
    let mut product = output_tensor(output_index, dtype);
    product.shape_bound = shape_bound;
    product
}

/// Converts a byte bound into a one-dimensional, device-sized element bound of
/// `dtype`, rounding up.
///
/// Fails with `ProductBoundTooLarge` when the element count exceeds
/// `u32::MAX` and with `MissingProductBound` when it is zero.
fn dynamic_element_bound(bytes: u64, dtype: DType) -> Result<ShapeBound, PlanningError> {
    let elements = bytes.div_ceil(dtype.element_bytes());
    let max = u32::try_from(elements).map_err(|_| PlanningError::ProductBoundTooLarge { bytes })?;
    if max == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    Ok(ShapeBound {
        dims: vec![DimBound::Device { max }],
    })
}

/// Serialized size of one ranked logprob entry (`TokenLogprob` in the IPC
/// schema).
const RANKED_LOGPROB_BYTES: u64 = 12;

/// Computes the maximum serialized log-probability payload required by a request.
///
/// Returns `None` when neither generated nor prompt logprobs are requested.
/// Each reported row costs 4 bytes plus `RANKED_LOGPROB_BYTES` per entry; a
/// row's entries are its own token and the requested top-k candidates
/// (`n_logprobs` or `n_prompt_logprobs`). A call reports at most
/// `generated_rows` generated rows, one per token it may commit (a
/// verification's current token and drafts), and `prompt_positions` prompt
/// rows. The worker leases a batch's completion storage from the sum of its
/// calls' `max_completion_bytes` and refuses a capture beyond its lease.
fn logprob_result_bytes(
    sampling: &SamplingParams,
    generated_rows: u32,
    prompt_positions: u32,
) -> Result<Option<u64>, PlanningError> {
    let generated = sampling.generated_logprobs_requested();
    let prompt = sampling.prompt_logprobs_requested();
    if !generated && !prompt {
        return Ok(None);
    }
    let generated_entries = generated.then_some(1 + u64::from(sampling.n_logprobs));
    let prompt_entries = prompt.then_some(1 + u64::from(sampling.n_prompt_logprobs));
    let generated_bytes = generated_entries
        .unwrap_or(0)
        .checked_mul(RANKED_LOGPROB_BYTES)
        .and_then(|bytes| bytes.checked_add(if generated { 4 } else { 0 }))
        .and_then(|bytes| bytes.checked_mul(u64::from(generated_rows)))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let per_prompt_bytes = 4_u64
        .checked_add(
            prompt_entries
                .unwrap_or(0)
                .checked_mul(RANKED_LOGPROB_BYTES)
                .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
        )
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let prompt_bytes = u64::from(prompt_positions)
        .checked_mul(per_prompt_bytes)
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let bytes = generated_bytes
        .checked_add(prompt_bytes)
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    Ok(Some(bytes))
}

/// Where a request's prefill/decode hand-off stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum HandOff {
    /// No hand-off is under way.
    #[default]
    None,
    /// Its units are being copied into the decode group's planes.
    Copying,
    /// Its units are in the decode group's planes: the request's next call
    /// admits it there with the token it committed, and declares its tables
    /// with no fresh unit, since the copied units hold its pages.
    Copied,
}

/// Lifecycle phase for a canonical generation request.
///
/// `RequestState::phase` records the accepted phase. Scheduling projects the
/// phase of the next call from it and the calls still in flight
/// (`Scheduler::next_generation_phase`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationPhase {
    /// Write the current latent input-image block's encoder feature into cache
    /// (`RequestState::input_image_features`) with a call of its own.
    IngestState,
    /// Ingest the context: prefill prompt tokens and the vision blocks of
    /// input images between them, encoding each block first
    /// (`Scheduler::next_context_call`).
    Prefill,
    /// Move the request's KV cache from the prefill group that prefilled its
    /// prompt to a decode group (`WorkerGroupRole::Decode`): the Storage
    /// Server's Transfer Server copies the request's units from the prefill
    /// group's device planes into the decode group's (`crate::hand_off`),
    /// and no call runs meanwhile (`RequestState::hand_off`). The decode
    /// group then decodes the request from its committed token on.
    HandOff,
    /// Decode understanding (text) tokens.
    DecodeUnd,
    /// Run one denoising step of a canvas-generating request's current block
    /// over its context. The step that stops the block reports the block's
    /// tokens.
    Canvas,
    /// Write a stopped block's tokens into the request's context as a causal
    /// prefill row (`RequestState::canvas_commit`); the next block reads them.
    CommitCanvas,
}

/// Image extension writes one latent input-image block with a call of its
/// own, which reads the latent encoder's feature. The vision blocks of input
/// images are instead attention blocks of a context prefill
/// (`is_prompt_extend`).
pub(super) fn consumes_image_features(call: &Call) -> bool {
    call.kind == CallKind::Forward(ForwardMode::Prefill) && call.latent_feature_input.is_some()
}

/// Prompt extension, a context prefill, writes prompt tokens into the prefix cache together
/// with the vision blocks of the input images between them
/// (`Call::vision_inputs`): a prefill with no latent image extension. It
/// samples the next token unless its request generates canvases, whose
/// prompt only conditions its canvases. A canvas-generating request's block
/// commit is a prompt extension too, of the tokens of its stopped block.
pub(super) fn is_prompt_extend(call: &Call) -> bool {
    call.kind == CallKind::Forward(ForwardMode::Prefill) && !consumes_image_features(call)
}

impl RequestState {
    /// Applies one validated call in request order. Pending identities are
    /// consumed once before this update; inactive successors never reach it.
    ///
    /// Updates the request's cursors, cache extents, encoder indices, denoising
    /// progress, and phase for the call kind. The completion loop in
    /// `execution` calls this, after removing the call from the in-flight
    /// queue, only for a `CallStatus::Ok` result that
    /// `validate_generation_result` accepted, on a request of the call's epoch
    /// whose chain is not invalidated and that has no terminal intent or
    /// deferred finish. Phase changes that depend on token values belong to
    /// `output`.
    ///
    /// Returns an error, before changing any state, for a call id with batch
    /// zero or a context prefill whose blocks disagree with the request's
    /// input images. The caller then finishes the request with an error.
    pub(crate) fn process_generation_result(
        &mut self,
        call: &Call,
        record: &CallCompletion,
    ) -> Result<(), GenerationResultError> {
        if call.call_id.batch_id == 0 {
            return Err(GenerationResultError::MissingCallId);
        }
        if record.status == CallStatus::Predicated {
            return Ok(());
        }

        if matches!(call.kind, CallKind::Forward(_)) {
            // A forward reports the extent it initialized, which a verifier
            // leaves above the prefix it accepted.
            self.cache_computed_len = record.cache_computed_len;
        }

        match call.kind {
            // A block commit is planned only in `CommitCanvas`, and a prompt
            // chunk never is; the committed block extends the context and
            // the next block starts at step zero.
            CallKind::Forward(ForwardMode::Prefill)
                if is_prompt_extend(call) && self.phase == GenerationPhase::CommitCanvas =>
            {
                let count = call.input_token_ids.len().min(u32::MAX as usize) as u32;
                self.logical_position = self.logical_position.saturating_add(count);
                self.cache_visible_len = self.cache_visible_len.saturating_add(count);
                self.canvas_block = self.canvas_block.saturating_add(1);
                self.canvas_step = 0;
                self.canvas_commit.clear();
                self.phase = GenerationPhase::Canvas;
            }
            // A context prefill writes its prompt tokens and the vision
            // blocks between them; an image's positions count once its last
            // block is written.
            CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => {
                let count = call.input_token_ids.len().min(u32::MAX as usize) as u32;
                let advance = self
                    .context_advance(count, call.vision_inputs.len(), self.block_cursor())
                    .ok_or(GenerationResultError::Progress {
                        detail: "context_blocks_mismatch",
                    })?;
                self.num_computed_prompt_tokens =
                    self.num_computed_prompt_tokens.saturating_add(count);
                self.logical_position = self.logical_position.saturating_add(advance.logical);
                self.cache_visible_len =
                    self.cache_visible_len.saturating_add(advance.prefix_cache);
                self.num_ingested_images = advance.cursor.image;
                self.image_encoder_index = advance.cursor.encoder;
                // The blocks it wrote lead the context features.
                let written = call.vision_inputs.len().min(self.context_features.len());
                self.context_features.drain(..written);
            }
            CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                // A latent input-image block: the worker reports the cache
                // extent the feature write reached. Logical positions advance
                // by the image's position count once, after its last encoder;
                // context ingestion continues with the image's next block, or
                // after the image.
                self.cache_visible_len = record.cache_visible_len;
                let image = &self.req.multimodal_inputs.images[self.num_ingested_images];
                self.image_encoder_index = self.image_encoder_index.saturating_add(1);
                self.input_image_features = None;
                if self.image_encoder_index == image.encoders.len() {
                    self.logical_position =
                        self.logical_position.saturating_add(image.num_positions);
                    self.num_ingested_images = self.num_ingested_images.saturating_add(1);
                    self.image_encoder_index = 0;
                }
                self.phase = GenerationPhase::Prefill;
            }
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                // A result that commits no token still consumes one position;
                // `output` resolves it as the primary EOS.
                let count = record.committed_tokens.len().max(1).min(u32::MAX as usize) as u32;
                self.logical_position = self.logical_position.saturating_add(count);
                self.cache_visible_len = self.cache_visible_len.saturating_add(count);
            }
            // A canvas step that reports no tokens leaves its block running;
            // one that reports the block's tokens stopped it, and `output`
            // publishes them before the block is committed.
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                if record.committed_tokens.is_empty() {
                    self.canvas_step = self.canvas_step.saturating_add(1);
                } else {
                    self.canvas_commit = record.committed_tokens.clone();
                    self.phase = GenerationPhase::CommitCanvas;
                }
            }
            _ => {}
        }

        // A sampling call draws once from the request's stream, which is what
        // the next commit's sampling position must state: a decode draws for
        // its token, and a prefill draws when it samples its last row. A
        // verification installs the tokens a drafter proposed without drawing,
        // and a prompt chunk that samples nothing draws nothing either. The
        // call declares that it sampled by carrying the output it sampled into
        // (`Call::token_output`); `consumer_slots` is the worker's, filled after
        // this call was planned and never reported back.
        if matches!(
            call.kind,
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Prefill)
        ) && call.token_output.is_some()
        {
            self.sampling_position = self.sampling_position.saturating_add(1);
        }
        Ok(())
    }
}

/// Builds the immutable auxiliary feature product produced by an encode call.
fn encoder_output(
    encoder: MediaCall,
    limits: &crate::GenerationLimits,
) -> Result<TensorRef, PlanningError> {
    // An image encoder is a latent or a vision encoding call
    // (`GenerationRequest::validate`).
    let bytes = if encoder == MediaCall::LatentEncoding {
        limits.max_latent_feature_bytes
    } else {
        limits.max_vision_feature_bytes
    };
    Ok(bounded_tensor(
        0,
        DType::BF16,
        dynamic_element_bound(bytes, DType::BF16)?,
    ))
}

/// Builds a call of `kind` before assigning storage and execution identities.
///
/// Every optional input and output is unset, the call id and the request
/// key's engine id and epoch are zero placeholders, and the bound is one
/// token; the `plan_*` builders fill in the rest.
fn new_call(request: &GenerationRequest, kind: CallKind) -> Call {
    let mut call = Call::new(
        RequestKey::new(0, request.request_id, 0),
        CallId::new(0, 0),
        kind,
    );
    call.bounds.max_tokens = 1;
    call
}

/// Plans a context prefill: prompt tokens `start..end` and the vision
/// `blocks` of the input images between them, which write `block_tokens` cache
/// tokens together, with a sampled-token output when this call ends the prompt
/// and the request `samples` from its prompt, unless the context ends with a
/// block of this call. A chunk before the prompt's end
/// only writes the cache: it draws nothing from the request's stream, so a
/// prompt split over several batches draws its first token once. A request
/// samples nothing from its prompt on a deployment that generates canvases.
///
/// The prompt of a canvas-generating request only conditions its canvases, so
/// its prefill samples nothing. A context that ends with an
/// image block samples nothing either: understanding decode
/// then starts from BOS, as after a trailing latent block. Fails with
/// `InvalidPromptRange` for a range past the prompt, or an empty one that
/// carries no block.
pub(super) fn plan_prompt(
    request: &GenerationRequest,
    start: u32,
    end: u32,
    blocks: Vec<VisionInput>,
    block_tokens: u32,
    sampling_state: Option<SamplingState>,
    samples: bool,
) -> Result<Call, PlanningError> {
    if start > end
        || (start == end && blocks.is_empty())
        || end as usize > request.prompt_token_ids.len()
    {
        return Err(PlanningError::InvalidPromptRange {
            start,
            end,
            prompt_tokens: request.prompt_token_ids.len(),
        });
    }
    let mut call = new_call(request, CallKind::Forward(ForwardMode::Prefill));
    call.bounds.max_tokens = end.saturating_sub(start).saturating_add(block_tokens);
    call.input_token_ids = request.prompt_token_ids[start as usize..end as usize].to_vec();
    // A block after the call's last prompt token, at the prompt's end.
    let ends_with_block = blocks
        .last()
        .is_some_and(|block| block.offset == end.saturating_sub(start))
        && end as usize == request.prompt_token_ids.len();
    call.vision_inputs = blocks;
    let ends_prompt = end as usize == request.prompt_token_ids.len();
    if !samples || !ends_prompt || ends_with_block {
        return finish_plan(request, call, 0);
    }
    call.token_output = Some(output_tensor(0, DType::I64));
    call.sampling_state = sampling_state;
    // Prompt position zero reports no logprob, so a chunk starting there
    // covers one position fewer; `validate_generation_result` recovers the
    // same offset from the call's coordinates.
    let prompt_positions = if request.sampling.prompt_logprobs_requested() {
        end.saturating_sub(start)
            .saturating_sub(u32::from(start == 0))
    } else {
        0
    };
    finish_plan(request, call, prompt_positions)
}

/// Plans denoising step `step` of the canvas-generating `request`'s block
/// `block`, the number of blocks already committed to its context, under the
/// deployment's `canvas` sampling.
///
/// The call carries no tokens: the worker keeps the block's canvas and
/// sampler state in the request's slot. Its `max_tokens` is the canvas
/// length, which the step denoises and which a stopping step reports as its
/// committed tokens; its completion payload bound covers those tokens at
/// four bytes each. Its completion output is true while the block continues
/// after the step: the predicate of the next step, which the scheduler may
/// queue before this one resolves.
pub(super) fn plan_canvas_step(
    request: &GenerationRequest,
    canvas: &CanvasSampling,
    block: u32,
    step: u32,
) -> Result<Call, PlanningError> {
    let mut call = new_call(request, CallKind::Forward(ForwardMode::TokenDenoising));
    call.bounds.max_tokens = canvas.canvas_length;
    call.canvas = Some(CanvasStep { block, step });
    call.completion_output = Some(output_tensor(0, DType::U8));
    let mut call = finish_plan(request, call, 0)?;
    call.bounds.max_completion_bytes = u64::from(canvas.canvas_length).saturating_mul(4);
    Ok(call)
}

/// Plans the commit of a stopped block's `tokens` to the context of the
/// canvas-generating `request`: a prefill that writes them into the prefix cache, as causal
/// prompt, and samples nothing. Fails with `EmptyCanvasCommit` for an empty
/// block.
pub(super) fn plan_canvas_commit(
    request: &GenerationRequest,
    tokens: &[u32],
) -> Result<Call, PlanningError> {
    if tokens.is_empty() {
        return Err(PlanningError::EmptyCanvasCommit);
    }
    let mut call = new_call(request, CallKind::Forward(ForwardMode::Prefill));
    call.bounds.max_tokens = u32::try_from(tokens.len())
        .map_err(|_| PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    call.input_token_ids = tokens.to_vec();
    finish_plan(request, call, 0)
}

/// Plans a sampled continuation; relay inputs leave host token values empty.
///
/// The call is a decode whose input is `input_token`, or nothing when
/// `relay_input` continues from the predecessor's device-resident token.
pub(super) fn plan_decode(
    request: &GenerationRequest,
    input_token: u32,
    relay_input: bool,
    sampling_state: Option<SamplingState>,
) -> Result<Call, PlanningError> {
    let mut call = new_call(request, CallKind::Forward(ForwardMode::Decode));
    call.input_token_ids = if relay_input {
        Vec::new()
    } else {
        vec![input_token]
    };
    call.token_output = Some(output_tensor(0, DType::I64));
    call.sampling_state = sampling_state;
    finish_plan(request, call, 0)
}

/// Plans the verification of a speculating request's next step: its
/// committed `input_token` followed by `drafts`, the drafts its drafter
/// proposed, of which the call commits the accepted prefix and the token
/// after it, at most one more than their count; with no drafts it decodes
/// one token, a verification row of length one. The call proposes
/// `proposals` drafts for the step after it.
pub(super) fn plan_verify(
    request: &GenerationRequest,
    input_token: u32,
    drafts: &[u32],
    proposals: u32,
    sampling_state: Option<SamplingState>,
) -> Result<Call, PlanningError> {
    let mut call = new_call(request, CallKind::Forward(ForwardMode::Verify));
    call.bounds.max_tokens = (1 + drafts.len()).min(u32::MAX as usize) as u32;
    call.input_token_ids = std::iter::once(input_token)
        .chain(drafts.iter().copied())
        .collect();
    call.draft.proposals = proposals;
    call.token_output = Some(output_tensor(0, DType::I64));
    call.sampling_state = sampling_state;
    finish_plan(request, call, 0)
}

/// The drafts a verification verifies: its input tokens after its current
/// token, or all of them when it continues from its predecessor's device
/// token.
pub(crate) fn verify_drafts(call: &Call) -> &[u32] {
    if call.predicate.is_some() {
        &call.input_token_ids
    } else {
        call.input_token_ids.get(1..).unwrap_or_default()
    }
}

/// Plans a refill of the speculating `request`'s draft cache over `tokens`,
/// which its cache holds from position `start` on: the target model
/// evaluates them again, from the state checkpoint at `start`, and the
/// drafter writes their draft cache entries, the last reading `lookahead`,
/// the token after them.
pub(super) fn plan_refill(
    request: &GenerationRequest,
    start: u32,
    tokens: Vec<u32>,
    lookahead: u32,
) -> Result<Call, PlanningError> {
    let mut call = new_call(request, CallKind::Forward(ForwardMode::Prefill));
    call.bounds.max_tokens = u32::try_from(tokens.len())
        .map_err(|_| PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    call.input_token_ids = tokens;
    call.draft = ipc::DraftWork {
        proposals: 0,
        lookahead: Some(lookahead),
        refill: Some(start),
    };
    finish_plan(request, call, 0)
}

/// Plans the cache write of a latent input-image block, retaining an exact
/// token count only when the encoder supplies one.
///
/// The call is a prefill that reads `feature` and may write up to the
/// encoder's `cache_token_capacity` cache tokens; it samples nothing.
pub(super) fn plan_image_extend(
    limits: &crate::GenerationLimits,
    request: &GenerationRequest,
    encoder: crate::ImageEncoderInput,
    feature: TensorRef,
) -> Result<Call, PlanningError> {
    let capacity = encoder.cache_token_capacity(limits)?;
    let mut call = new_call(request, CallKind::Forward(ForwardMode::Prefill));
    call.bounds.max_tokens = capacity;
    call.latent_feature_input = Some(feature);
    finish_plan(request, call, 0)
}

/// Encodes an input image (`image_base64`) into a feature tensor with
/// encoder `step`; a latent encoding samples its posterior at `draw`
/// (`Call::latent_draw`).
///
/// Fails with `MissingImageInput` for an empty payload.
pub(super) fn plan_encode(
    limits: &crate::GenerationLimits,
    request: &GenerationRequest,
    step: MediaCall,
    image_base64: String,
    draw: Option<u32>,
) -> Result<Call, PlanningError> {
    if image_base64.is_empty() {
        return Err(PlanningError::MissingImageInput);
    }
    let mut call = new_call(request, CallKind::Media(step));
    call.input_image = Some(image_base64.into());
    call.encoder_output = Some(encoder_output(step, limits)?);
    call.latent_draw = (step == MediaCall::LatentEncoding)
        .then_some(draw)
        .flatten();
    finish_plan(request, call, 0)
}

/// The posterior draw of a request's latent encoding (`Call::latent_draw`)
/// of the input-image block at `block`: its ordinal among the request's
/// latent encodings, which count its input images' latent encoders in
/// context order.
pub(super) fn latent_draw(request: &GenerationRequest, block: BlockCursor) -> u32 {
    let latent = |inputs: &[ImageEncoderInput]| {
        inputs
            .iter()
            .filter(|input| input.encoder == crate::ImageEncoder::Latent)
            .count()
    };
    let images = &request.multimodal_inputs.images;
    let draw = images[..block.image.min(images.len())]
        .iter()
        .map(|image| latent(&image.encoders))
        .sum::<usize>()
        + images.get(block.image).map_or(0, |image| {
            latent(&image.encoders[..block.encoder.min(image.encoders.len())])
        });
    u32::try_from(draw).unwrap_or(u32::MAX)
}

/// Derives storage bounds directly from the call's actual inputs and outputs.
///
/// `max_completion_bytes` covers the logprob payload of a token-producing
/// call. `max_cache_units` starts at zero;
/// `Scheduler::plan_call` sets it for forward calls.
fn finish_plan(
    request: &GenerationRequest,
    mut call: Call,
    prompt_positions: u32,
) -> Result<Call, PlanningError> {
    let logprob_bytes = if call.token_output.is_some() {
        // A verification commits up to its bound of tokens, each reporting
        // its row; every other call reports one.
        let generated_rows = if call.kind == CallKind::Forward(ForwardMode::Verify) {
            call.bounds.max_tokens
        } else {
            1
        };
        logprob_result_bytes(&request.sampling, generated_rows, prompt_positions)?.unwrap_or(0)
    } else {
        0
    };
    let max_transfer_bytes = call.bounds.max_transfer_bytes;
    call.bounds = Bounds {
        max_tokens: call.bounds.max_tokens,
        max_cache_units: 0,
        max_completion_bytes: logprob_bytes,
        max_transfer_bytes,
    };
    Ok(call)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Failure while constructing the next worker call.
pub(crate) enum PlanningError {
    #[error(transparent)]
    Capacity(#[from] crate::GenerationResourceError),
    #[error("invalid prompt range {start}..{end} for {prompt_tokens} input tokens")]
    InvalidPromptRange {
        start: u32,
        end: u32,
        prompt_tokens: usize,
    },
    #[error("a canvas commit carries no tokens")]
    EmptyCanvasCommit,
    #[error("image encoding requires image bytes")]
    MissingImageInput,
    #[error("product bound must be nonzero")]
    MissingProductBound,
    #[error("product bound {bytes} bytes exceeds the IPC representation")]
    ProductBoundTooLarge { bytes: u64 },
    #[error("product generation counter exhausted")]
    ProductGenerationExhausted,
}

/// Assigns final identities to the selected call and its allocated outputs.
/// Identity exhaustion is checked before changing the shared generation counter.
///
/// The caller must already have set `call.call_id` to the call's batch and
/// row. Every output with generation zero, `cache_output` first and then the
/// tensor outputs in `Call::tensor_outputs` order, receives the next value of
/// the engine-wide counter; generations are nonzero and fit in `u32`. On
/// error nothing is changed, and the caller treats it as engine-fatal.
pub(super) fn register_call(
    call: &mut Call,
    request_key: RequestKey,
    next_product_generation: &mut u64,
) -> Result<(), PlanningError> {
    let call_id = call.call_id;
    let required_generations = call
        .tensor_outputs()
        .filter(|product| product.generation == 0)
        .count()
        + usize::from(
            call.cache_output
                .is_some_and(|output| output.generation == 0),
        );
    let first_generation = (*next_product_generation).max(1);
    if required_generations > 0 {
        let last_generation = first_generation
            .checked_add(required_generations as u64 - 1)
            .ok_or(PlanningError::ProductGenerationExhausted)?;
        if last_generation > u64::from(u32::MAX) {
            return Err(PlanningError::ProductGenerationExhausted);
        }
    }
    // The preflight above guarantees every generation this call assigns fits.
    let mut acquire_generation = || {
        let generation = u32::try_from((*next_product_generation).max(1))
            .map_err(|_| PlanningError::ProductGenerationExhausted)?;
        *next_product_generation = u64::from(generation) + 1;
        Ok::<_, PlanningError>(generation)
    };
    call.request_key = request_key;
    if let Some(output) = &mut call.cache_output {
        output.owner = request_key;
        output.producer_call_id = call_id;
        if output.generation == 0 {
            output.generation = acquire_generation()?;
        }
    }
    for product in call.tensor_outputs_mut() {
        product.request_key = request_key;
        product.producer_call_id = call_id;
        if product.generation == 0 {
            product.generation = acquire_generation()?;
        }
    }
    Ok(())
}

/// Validates a result against submitted inputs and immutable request limits.
///
/// `image_cache` is the input frozen when the call was submitted
/// (`InflightInput::Generation`). A predicated result is valid only when the
/// call carried a predicate and the result reports no tokens and no products.
/// `canvas` is the deployment's block-diffusion sampling, which bounds a
/// canvas step's block. Returns the first violation found; the caller
/// finishes the request with an error on any `Err`.
pub(crate) fn validate_generation_result(
    call: &Call,
    image_cache: Option<(u32, Option<u32>)>,
    state: &RequestState,
    record: &CallCompletion,
    canvas: Option<&CanvasSampling>,
) -> Result<(), GenerationResultError> {
    let request = &state.req;

    // Identity and status.
    if record.request_key != call.request_key {
        return Err(GenerationResultError::Identity {
            detail: "request_mismatch",
        });
    }
    if call.call_id.batch_id == 0 || record.call_id != call.call_id {
        return Err(GenerationResultError::Identity {
            detail: "call_id_mismatch",
        });
    }
    if record.status == CallStatus::Predicated {
        if call.predicate.as_ref().is_none() || !record.committed_tokens.as_slice().is_empty() {
            return Err(GenerationResultError::Status {
                detail: "invalid_predicated_result",
            });
        }
        return Ok(());
    }

    let call_variant = call.kind;
    if record.status != CallStatus::Ok {
        return Err(GenerationResultError::Status {
            detail: "call_failed",
        });
    }

    // A feature write must reach exactly `start + tokens` when the encoder
    // declared an exact token count, and otherwise stay within the call's
    // token capacity above `start`. A context prefill with vision blocks
    // writes the extent its columns state, its prompt tokens and blocks, which
    // the request's progress advances by itself (`context_advance`): its rows
    // are the batch's, and the rank reports positions rather than an extent
    // of its own choosing.
    if consumes_image_features(call) {
        let (start, num_cache_tokens) = image_cache.ok_or(GenerationResultError::Progress {
            detail: "image_kv_input_missing",
        })?;
        let actual = record.cache_visible_len;
        if num_cache_tokens.is_some_and(|tokens| actual != start.saturating_add(tokens))
            || (num_cache_tokens.is_none()
                && (actual < start || actual > start.saturating_add(call.bounds.max_tokens)))
        {
            return Err(GenerationResultError::Progress {
                detail: "image_kv_mismatch",
            });
        }
    }

    // A canvas step reports no tokens while its block runs and the whole
    // block once it stops; the step at the block's step limit must stop it.
    if let Some(step) = call.canvas {
        let committed = record.committed_tokens.len();
        if committed != 0 && committed != call.bounds.max_tokens as usize {
            return Err(GenerationResultError::Token {
                detail: "canvas_block_length_mismatch",
            });
        }
        let last_step = canvas.is_none_or(|canvas| step.step.saturating_add(1) >= canvas.max_steps);
        if committed == 0 && last_step {
            return Err(GenerationResultError::Progress {
                detail: "canvas_step_limit_exceeded",
            });
        }
        if !record.token_logprobs.is_empty() || !record.prompt_logprobs.is_empty() {
            return Err(GenerationResultError::Logprob {
                detail: "unexpected_canvas_logprobs",
            });
        }
        return Ok(());
    }

    // Sampled tokens: presence, verified-draft prefix and count. The allowed
    // set and the banned tokens are the sampler's to apply, on the device.
    let sampled_tokens = record.committed_tokens.as_slice();
    let produces_token = call.token_output.is_some();
    if !produces_token && !sampled_tokens.is_empty() {
        return Err(GenerationResultError::Token {
            detail: "unexpected_sampled_token",
        });
    }
    if produces_token && sampled_tokens.is_empty() {
        return Err(GenerationResultError::Token {
            detail: "missing_sampled_token",
        });
    }
    // A verify commits its accepted drafts plus the target's own token, except
    // that a finish token among the drafts ends the commit there without an
    // extra token.
    if call_variant == CallKind::Forward(ForwardMode::Verify) {
        let drafts = verify_drafts(call);
        let listed = record.committed_tokens.as_slice();
        let terminal_prefix = !listed.is_empty()
            && listed.len() <= drafts.len()
            && listed == &drafts[..listed.len()]
            && listed
                .last()
                .is_some_and(|token| state.finish_token_ids.contains(token));
        if !terminal_prefix {
            let accepted = listed.len().saturating_sub(1);
            if listed.is_empty()
                || accepted > drafts.len()
                || listed[..accepted] != drafts[..accepted]
            {
                return Err(GenerationResultError::Token {
                    detail: "verified_draft_prefix_mismatch",
                });
            }
        }
    }
    if produces_token {
        let max_tokens = if call_variant == CallKind::Forward(ForwardMode::Verify) {
            call.bounds.max_tokens as usize
        } else {
            1
        };
        if sampled_tokens.len() > max_tokens {
            return Err(GenerationResultError::Token {
                detail: "text_token_count_mismatch",
            });
        }
    }
    // Generated logprobs are expected exactly when a forward call samples a
    // token and the request asked for them, one entry per committed token
    // (a verification's accepted drafts and the token after them alike):
    // each entry's first candidate is its token, and candidates need nonzero
    // ranks, finite values, and distinct ids.
    let reports_logprobs = produces_token
        && request.sampling.generated_logprobs_requested()
        && matches!(
            call_variant,
            CallKind::Forward(ForwardMode::Prefill)
                | CallKind::Forward(ForwardMode::Decode)
                | CallKind::Forward(ForwardMode::Verify)
        );
    let generated = record.token_logprobs.as_slice();
    if !reports_logprobs || sampled_tokens.is_empty() {
        if !generated.is_empty() {
            return Err(GenerationResultError::Logprob {
                detail: "unexpected_generated_logprobs",
            });
        }
    } else if generated.len() != sampled_tokens.len() {
        return Err(GenerationResultError::Logprob {
            detail: "missing_generated_logprobs",
        });
    } else {
        for (candidates, &token_id) in generated.iter().zip(sampled_tokens) {
            if candidates
                .first()
                .is_none_or(|first| first.token_id != token_id)
            {
                return Err(GenerationResultError::Logprob {
                    detail: "generated_logprob_token_mismatch",
                });
            }
            let mut token_ids = std::collections::HashSet::new();
            if candidates.iter().any(|candidate| {
                candidate.rank == 0
                    || !candidate.logprob.is_finite()
                    || !token_ids.insert(candidate.token_id)
            }) {
                return Err(GenerationResultError::Logprob {
                    detail: "invalid_generated_logprob_candidates",
                });
            }
        }
    }

    // Prompt logprobs cover every input token of a prompt-extending prefill
    // except prompt position zero, so a chunk starting there reports one
    // position fewer; this reads only the call's stated coordinates, not
    // request progress, which cancellation may have stopped.
    let prompt_tokens = (is_prompt_extend(call) && request.sampling.prompt_logprobs_requested())
        .then(|| {
            &call.input_token_ids[usize::from(
                call.coordinates.logical_position == 0 && !call.input_token_ids.is_empty(),
            )..]
        });
    match (
        prompt_tokens,
        (!record.prompt_logprobs.is_empty()).then_some(record.prompt_logprobs.as_slice()),
    ) {
        (None, Some(positions)) if !positions.is_empty() => {
            return Err(GenerationResultError::Logprob {
                detail: "unexpected_prompt_logprobs",
            });
        }
        (Some(expected), actual) => {
            let actual = actual.unwrap_or_default();
            if actual.len() != expected.len() {
                return Err(GenerationResultError::Logprob {
                    detail: "prompt_logprob_count_mismatch",
                });
            }
            for (&expected_token, candidates) in expected.iter().zip(actual) {
                let Some(first) = candidates.first() else {
                    return Err(GenerationResultError::Logprob {
                        detail: "empty_prompt_logprob_position",
                    });
                };
                if first.token_id != expected_token {
                    return Err(GenerationResultError::Logprob {
                        detail: "prompt_logprob_token_mismatch",
                    });
                }
                let mut seen = std::collections::HashSet::new();
                if candidates
                    .iter()
                    .any(|candidate| candidate.rank == 0 || !seen.insert(candidate.token_id))
                {
                    return Err(GenerationResultError::Logprob {
                        detail: "invalid_prompt_logprob_candidates",
                    });
                }
            }
        }
        (None, None | Some(_)) => {}
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
/// Invalid execution identity, numerical result, or produced value.
pub(crate) enum GenerationResultError {
    #[error("call has no registered identity")]
    MissingCallId,
    #[error("worker status invalid: {detail}")]
    Status { detail: &'static str },
    #[error("worker identity invalid: {detail}")]
    Identity { detail: &'static str },
    #[error("worker progress invalid: {detail}")]
    Progress { detail: &'static str },
    #[error("worker token result invalid: {detail}")]
    Token { detail: &'static str },
    #[error("worker logprob result invalid: {detail}")]
    Logprob { detail: &'static str },
}

/// Engine-owned state for one admitted request.
pub(crate) struct RequestState {
    pub req: GenerationRequest,
    /// Sorted device finish tokens (see `finish_token_ids` in the scheduler
    /// module).
    pub(crate) finish_token_ids: Vec<u32>,
    /// Storage the request holds, installed at admission: its request-pool
    /// slot and per-group cache block tables, plus the buffer spans its calls
    /// reserve later. Each table owns its physical page
    /// references and therefore has exactly the request's lifetime.
    pub(super) allocations: Option<RequestAllocations>,
    /// Admission generation used to reject results from prior request lifetimes.
    pub(crate) request_epoch: u64,
    /// Most recent accepted state-producing call.
    pub(crate) last_state_call_id: CallId,
    /// Last device token retained until a consumer is registered.
    pub(crate) latest_token: Option<TensorRef>,
    /// A false device predicate invalidated the unresolved successor chain.
    /// Already-submitted descendants must drain before scheduling resumes from
    /// the last host-observed execution result.
    pub(crate) speculative_chain_invalidated: bool,
    /// The request's draft cache, in a deployment that speculates.
    pub(crate) draft: super::speculation::DraftCache,
    pub(super) output: super::output::RequestOutput,
    /// Current accepted call stage. Completion processing and output
    /// resolution advance it; scheduling changes it only on an encoder-cache
    /// hit, which moves the request to `IngestState` without a call.
    pub(super) phase: Phase,
    /// Prompt token range already accepted by the worker.
    pub(super) num_computed_prompt_tokens: u32,
    /// Context images whose every encoder feature has been written into the prefix cache.
    pub(super) num_ingested_images: usize,
    /// Next encoder of the current context image: with `num_ingested_images`,
    /// the block cursor (`RequestState::block_cursor`).
    pub(super) image_encoder_index: usize,
    /// Encoder feature of the current latent block awaiting its cache write.
    pub(super) input_image_features: Option<TensorRef>,
    /// Encoder products of the vision blocks from the accepted block cursor
    /// on, in context order, for the context prefills that inject them. The
    /// blocks of in-flight context prefills lead the list until their
    /// results are accepted, so a planned call that is not submitted leaves
    /// its blocks' products in place.
    pub(super) context_features: Vec<TensorRef>,
    /// Model position of the accepted sequence. Tokens advance it by one and
    /// an image by its position count, so it differs from `cache_visible_len`
    /// when image inputs are present.
    pub(super) logical_position: u32,
    /// Accepted cache length visible to attention, in tokens.
    pub(super) cache_visible_len: u32,
    /// Accepted tokens whose prefix cache is initialized. A verifier initializes its
    /// rejected drafts, so this exceeds the visible extent until the next
    /// forward makes exactly what it initializes visible.
    pub(super) cache_computed_len: u32,
    /// Draws the request's sampler has made. The slot's device state advances
    /// its own sampling position as it installs a call's tokens; this is the
    /// host copy the mechanism states in the commit table, and it advances
    /// exactly as the reference counts draws: once for a decode and once for a
    /// prefill that samples, and not at all for a verification, whose accepted
    /// span installs tokens without drawing, nor for a prompt chunk that
    /// samples nothing.
    pub(super) sampling_position: u32,
    /// Committed token the next decode continues from.
    pub(super) next_token: u32,
    pub(super) num_generated_tokens: usize,
    /// Where the request's hand-off from its prefill group to a decode group
    /// stands (`GenerationPhase::HandOff`).
    pub(super) hand_off: HandOff,
    /// Whether the current epoch is registered with its execution workers.
    pub(super) worker_registered: bool,
    /// Whether the request's slot holds the request's recurrent state at its
    /// accepted position: false until the request's first token call is
    /// planned on the slot, which restores the state from the checkpoint its
    /// reused prefix ends at (`batching::columns`). A preempted or migrated
    /// request is admitted again as a new state, so it restores again.
    pub(crate) state_resident: bool,
    /// Whether the worker's slot holds the request's sampling state: its
    /// penalty counts started from zero and its bias row
    /// (`batching::columns::adjusts_logits`). False until the request's first
    /// token batch on a worker group clears the counts and installs the row,
    /// and again after a hand-off moves the request to the decode group. A
    /// preempted or migrated request is admitted again as a new state, so its
    /// slot starts again.
    pub(crate) sampling_resident: bool,
    /// Allocation serial of the request's cache units already declared to the
    /// workers: units allocated at or after it are fresh for the next
    /// dispatch (`CacheAllocation::allocated_units`).
    pub(super) num_cache_units_sent: u64,
    /// Whether admission reserves the complete multimodal cache requirement.
    pub(super) reserve_worstcase: bool,
    /// Worst-case cache token extent computed at submission; its units are
    /// reserved only when `reserve_worstcase` is set.
    pub(super) max_reserved_cache_tokens: usize,
    /// Per-group page keys of the complete prompt pages, retained for
    /// publishing reusable prompt pages as prefill completes; every cache
    /// level keys the pages by them.
    pub(super) prefix_page_hashes: Vec<Vec<storage::Key>>,
    /// Per group, the leading prompt pages already published to the prefix
    /// cache, or reused from it.
    pub(super) prefix_published: Vec<usize>,
    /// Whether the host KV cache serves the request's prefill worker and the
    /// request reads it: its published pages are offloaded there under
    /// their page keys (`prefix_page_hashes`).
    pub(super) host_keyed: bool,
    /// The prompt tokens through which the request's sliding windows were
    /// considered for offload: the aligned boundaries after it are still to
    /// come.
    pub(super) offload_boundary: usize,
    /// Accepted text and control tokens used by penalties.
    pub(super) generated_token_ids: Vec<u32>,
    /// False after a completed call creates state unavailable from the input.
    pub(super) replayable: bool,
    /// Encoder-cache entries this request holds; released when the request
    /// finishes.
    pub(super) encoder_cache_pins: Vec<EncoderCachePin>,
    /// The image encoding the request looked up in the host encoder pools
    /// and plans next (`Scheduler::encoding_ready`), until its call is
    /// registered with the output it reads or writes.
    pub(super) host_encoding: Option<super::encodings::Encoding>,
    /// Request-local encoder outputs outside the encoder cache, freed by
    /// `Scheduler::free_consumed_products` or when the request finishes.
    pub(super) transient_encoder_products: Vec<TensorRef>,
    /// Blocks a canvas-generating request has committed to its context.
    pub(super) canvas_block: u32,
    /// Accepted denoising steps of the current block.
    pub(super) canvas_step: u32,
    /// Tokens of the stopped block awaiting their commit
    /// (`GenerationPhase::CommitCanvas`); empty otherwise.
    pub(super) canvas_commit: Vec<u32>,
    /// Monotonic microseconds (`clock::now_monotonic_us`) at which the
    /// request state was created: queue waits are measured on the monotonic
    /// clock, and only the public `Scheduled` event states it as Unix time.
    pub queued_us: u64,
    pub(crate) terminal_intent: Option<Terminal>,
    /// Cause of the first rejected worker result, which
    /// `Scheduler::fail_after_inflight` reports as the request's terminal
    /// error and its error finish logs.
    pub(super) failure: Option<String>,
    /// The prompt tokens the request's usage charges: its own prompt, without
    /// the tokens a migrated request extends it with.
    pub(super) prompt_tokens: usize,
}

impl RequestState {
    /// The prompt tokens the request's usage charges: its prompt without the
    /// tokens it generated before it migrated, which its prompt continues
    /// with (`Scheduler::request_state`) and its usage counts as generated.
    pub(crate) fn charged_prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Returns the request allocations, which admission installs.
    pub(super) fn allocations(&self) -> Option<&RequestAllocations> {
        self.allocations.as_ref()
    }

    /// Returns mutable access to the request allocations, which admission installs.
    pub(super) fn allocations_mut(&mut self) -> Option<&mut RequestAllocations> {
        self.allocations.as_mut()
    }

    /// Returns the request-pool row, which admission assigns.
    pub(super) fn request_pool_idx(&self) -> Option<u32> {
        self.allocations().map(RequestAllocations::request_slot)
    }

    /// Returns the request's block tables, one per cache group once admitted.
    pub(super) fn block_tables(&self) -> Option<&[BlockTable]> {
        self.allocations().map(RequestAllocations::block_tables)
    }

    /// Returns mutable access to the request's cache tables once admitted.
    pub(super) fn prefix_cache_mut(&mut self) -> Option<&mut CacheAllocation> {
        self.allocations_mut()
            .map(|allocations| &mut allocations.prefix_cache)
    }

    /// Returns whether the request includes context images.
    pub(super) fn has_context_images(&self) -> bool {
        !self.req.multimodal_inputs.images.is_empty()
    }

    /// Returns whether the request's cache holds its prompt and then the
    /// tokens it generates, in order: a request from a prompt without
    /// context images, whose migration resumes from its generated tokens and
    /// finds their pages as its prompt's.
    pub(super) fn caches_generated_text(&self) -> bool {
        !self.has_context_images() && self.replayable
    }

    /// Returns whether the text request can be replayed.
    pub(crate) fn is_replayable_text(&self) -> bool {
        self.replayable
    }

    /// Returns the accepted block cursor: the next input-image block to
    /// write into the context.
    pub(super) fn block_cursor(&self) -> BlockCursor {
        BlockCursor {
            image: self.num_ingested_images,
            encoder: self.image_encoder_index,
        }
    }

    /// Returns the input-image block at `cursor` with its image, or `None`
    /// past the last block.
    pub(super) fn input_block(
        &self,
        cursor: BlockCursor,
    ) -> Option<(&ImageInput, &ImageEncoderInput)> {
        let image = self.req.multimodal_inputs.images.get(cursor.image)?;
        Some((image, image.encoders.get(cursor.encoder)?))
    }

    /// Returns the number of input-image blocks from `from` up to `to`.
    pub(super) fn blocks_between(&self, from: BlockCursor, to: BlockCursor) -> usize {
        let images = &self.req.multimodal_inputs.images;
        let ordinal = |cursor: BlockCursor| {
            images[..cursor.image.min(images.len())]
                .iter()
                .map(|image| image.encoders.len())
                .sum::<usize>()
                + cursor.encoder
        };
        ordinal(to).saturating_sub(ordinal(from))
    }

    /// Returns the cursor of the block after the one at `cursor`.
    pub(super) fn next_block(&self, cursor: BlockCursor) -> Option<BlockCursor> {
        let image = self.req.multimodal_inputs.images.get(cursor.image)?;
        Some(if cursor.encoder + 1 < image.encoders.len() {
            BlockCursor {
                image: cursor.image,
                encoder: cursor.encoder + 1,
            }
        } else {
            BlockCursor {
                image: cursor.image + 1,
                encoder: 0,
            }
        })
    }

    /// Returns what a context prefill of `text` prompt tokens and `blocks`
    /// vision blocks adds to the context when the block cursor is at `from`.
    ///
    /// Returns `None` when a block is not a vision block of an input image
    /// with an exact cache length; submission validation requires every vision
    /// block of an input image to declare one.
    pub(super) fn context_advance(
        &self,
        text: u32,
        blocks: usize,
        from: BlockCursor,
    ) -> Option<ContextAdvance> {
        let mut advance = ContextAdvance {
            logical: text,
            prefix_cache: text,
            cursor: from,
        };
        for _ in 0..blocks {
            let (image, input) = self.input_block(advance.cursor)?;
            if input.encoder != crate::ImageEncoder::Vision {
                return None;
            }
            advance.prefix_cache = advance.prefix_cache.checked_add(input.num_cache_tokens?)?;
            let next = self.next_block(advance.cursor)?;
            if next.image != advance.cursor.image {
                advance.logical = advance.logical.saturating_add(image.num_positions);
            }
            advance.cursor = next;
        }
        Some(advance)
    }
}

/// A position in a request's sequence of input-image blocks: an image and
/// the next of its encoders to write. Each encoder of an input image writes
/// one block of the image into the context, in encoder order, at the image's
/// prompt position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct BlockCursor {
    /// Index of the image in the request's input images.
    pub(crate) image: usize,
    /// Index of the block's encoder in the image's encoders.
    pub(crate) encoder: usize,
}

/// What a context prefill adds to its request's context
/// (`RequestState::context_advance`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextAdvance {
    /// Logical positions: one per prompt token, plus the positions of each
    /// image whose last block the call writes.
    pub(crate) logical: u32,
    /// Cache tokens: one per prompt token, plus each block's exact cache length.
    pub(crate) prefix_cache: u32,
    /// The block cursor after the call.
    pub(crate) cursor: BlockCursor,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(prompt: u32) -> GenerationRequest {
        GenerationRequest {
            request_id: RequestId(7),
            prompt_token_ids: (0..prompt).collect(),
            multimodal_inputs: Default::default(),
            sampling: Default::default(),
            max_und_tokens: 16,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 3,
            cache: Default::default(),
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
    }

    /// A prompt longer than one batch's budget is split, and only the chunk that ends it draws. The
    /// chunk before it used to declare a token output too: its draw advanced the request's sampling
    /// position, so the chunk that ended the prompt read as a row with a committed token behind it
    /// and the runner refused it (GSM8K at `355a8c564`, `a greedy row of slot 6 ends a prompt that
    /// already holds 1 committed token(s)`).
    #[test]
    fn only_the_chunk_that_ends_a_prompt_samples() {
        let request = request(700);
        let plan = |start, end| {
            plan_prompt(&request, start, end, Vec::new(), 0, None, true).expect("a prompt range")
        };
        assert!(
            plan(0, 512).token_output.is_none(),
            "a chunk before the end draws nothing"
        );
        assert!(
            plan(512, 700).token_output.is_some(),
            "the last chunk draws the first token"
        );
        assert!(
            plan(0, 700).token_output.is_some(),
            "a whole prompt is its own last chunk"
        );
    }
}
