//! Conversions between the request vocabulary and the request interface's
//! messages (`request.proto`).
//!
//! A value becomes its message whole. A message becomes a value only when
//! every enumeration it names is one this build knows and every message the
//! value requires is present; a peer of a later build that sends an
//! enumeration value this build does not know is refused rather than
//! misread.

use std::sync::Arc;

use super::wire;
use super::{
    ArtifactEvent, ArtifactMedia, CachePolicy, CacheResume, ConditionExtent, DiffusionRequest,
    DiffusionSamplingParams, EncoderInputs, Event, FinishReason, GenerationFeatures,
    GenerationLimits, GenerationRequest, ImageEncoder, ImageEncoderInput, ImageInput, MediaKind,
    MigrationCause, MultimodalInputs, PositionLogprobs, RejectionKind, Request, RequestId,
    SamplingParams, StopReason, TokenLogprob, Trajectory,
};

/// A message that states no valid value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the request interface's {0}")]
pub struct WireError(String);

impl WireError {
    fn new(what: impl Into<String>) -> Self {
        Self(what.into())
    }
}

/// The enumeration value `value` of a message field named `field`.
fn known<E: TryFrom<i32>>(value: i32, field: &str) -> Result<E, WireError> {
    E::try_from(value).map_err(|_| WireError::new(format!("{field} {value} is unknown")))
}

/// A message `field` the value requires.
fn present<M>(message: Option<M>, field: &str) -> Result<M, WireError> {
    message.ok_or_else(|| WireError::new(format!("{field} is missing")))
}

/// A count a message carries as `u64`, which this platform holds.
fn count(value: u64, field: &str) -> Result<usize, WireError> {
    usize::try_from(value).map_err(|_| WireError::new(format!("{field} {value} is too large")))
}

// ----------------------------------------------------------------- requests

impl Request {
    /// The request's message.
    pub fn to_wire(&self) -> wire::Request {
        match self {
            Self::Generation(request) => wire::Request {
                id: request.request_id.0,
                priority: request.priority,
                kind: Some(wire::request::Kind::Generation(request.as_ref().into())),
            },
            Self::Diffusion(request) => wire::Request {
                id: request.request_id.0,
                priority: request.priority,
                kind: Some(wire::request::Kind::Diffusion(request.into())),
            },
        }
    }

    /// The request a message states.
    ///
    /// # Errors
    ///
    /// Fails when the message names no request kind, a value this build
    /// does not know, or omits a value the request requires.
    pub fn from_wire(request: wire::Request) -> Result<Self, WireError> {
        let id = RequestId(request.id);
        match present(request.kind, "request kind")? {
            wire::request::Kind::Generation(generation) => {
                let mut generation = GenerationRequest::try_from(generation)?;
                generation.request_id = id;
                generation.priority = request.priority;
                Ok(Self::Generation(Box::new(generation)))
            }
            wire::request::Kind::Diffusion(diffusion) => {
                let mut diffusion = DiffusionRequest::try_from(diffusion)?;
                diffusion.request_id = id;
                diffusion.priority = request.priority;
                Ok(Self::Diffusion(diffusion))
            }
        }
    }
}

impl From<&GenerationRequest> for wire::GenerationRequest {
    fn from(request: &GenerationRequest) -> Self {
        Self {
            prompt_token_ids: request.prompt_token_ids.clone(),
            images: request
                .multimodal_inputs
                .images
                .iter()
                .map(Into::into)
                .collect(),
            sampling: Some((&request.sampling).into()),
            max_und_tokens: request.max_und_tokens as u64,
            stop_strings: request.stop_strings.clone(),
            stop_token_ids: request.stop_token_ids.clone(),
            cache: Some(wire::CachePolicy {
                read: request.cache.read,
                write: request.cache.write,
                isolation_key: request.cache.isolation_key,
                state_seeds: request
                    .cache
                    .state_seeds
                    .iter()
                    .map(|seed| seed.to_vec())
                    .collect(),
                resume: request.cache.resume.map(|resume| wire::CacheResume {
                    epoch: resume.epoch,
                    request_id: resume.request_id,
                }),
            }),
            include_stop_token: request.include_stop_token,
            generated_token_ids: request.generated_token_ids.clone(),
            adapter: request.adapter,
        }
    }
}

impl TryFrom<wire::GenerationRequest> for GenerationRequest {
    type Error = WireError;

    /// The request without its identity and priority, which the enclosing
    /// [`wire::Request`] states.
    fn try_from(request: wire::GenerationRequest) -> Result<Self, WireError> {
        let cache = present(request.cache, "cache policy")?;
        Ok(Self {
            request_id: RequestId::default(),
            prompt_token_ids: request.prompt_token_ids,
            multimodal_inputs: MultimodalInputs {
                images: request
                    .images
                    .into_iter()
                    .map(ImageInput::try_from)
                    .collect::<Result<_, _>>()?,
            },
            sampling: present(request.sampling, "sampling parameters")?.try_into()?,
            max_und_tokens: count(request.max_und_tokens, "max_und_tokens")?,
            stop_strings: request.stop_strings,
            stop_token_ids: request.stop_token_ids,
            priority: 0,
            cache: CachePolicy {
                read: cache.read,
                write: cache.write,
                isolation_key: cache.isolation_key,
                state_seeds: cache
                    .state_seeds
                    .iter()
                    .map(|seed| <[u8; 32]>::try_from(seed.as_slice()))
                    .collect::<Result<_, _>>()
                    .map_err(|_| WireError::new("a state seed is not 32 bytes"))?,
                resume: cache.resume.map(|resume| CacheResume {
                    epoch: resume.epoch,
                    request_id: resume.request_id,
                }),
            },
            include_stop_token: request.include_stop_token,
            generated_token_ids: request.generated_token_ids,
            adapter: request.adapter,
        })
    }
}

impl From<&ImageInput> for wire::ImageInput {
    fn from(image: &ImageInput) -> Self {
        Self {
            digest: image.digest.to_vec(),
            b64: image.b64.clone(),
            position: image.position,
            num_positions: image.num_positions,
            encoders: image.encoders.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<wire::ImageInput> for ImageInput {
    type Error = WireError;

    fn try_from(image: wire::ImageInput) -> Result<Self, WireError> {
        Ok(Self {
            digest: digest(&image.digest, "an input image's digest")?,
            b64: image.b64,
            position: image.position,
            num_positions: image.num_positions,
            encoders: image
                .encoders
                .into_iter()
                .map(ImageEncoderInput::try_from)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl From<&ImageEncoderInput> for wire::ImageEncoderInput {
    fn from(input: &ImageEncoderInput) -> Self {
        let encoder = match input.encoder {
            ImageEncoder::Vision => wire::ImageEncoder::Vision,
            ImageEncoder::Latent => wire::ImageEncoder::Latent,
        };
        Self {
            encoder: encoder as i32,
            num_cache_tokens: input.num_cache_tokens,
            max_cache_tokens: input.max_cache_tokens,
            height: input.height,
            width: input.width,
        }
    }
}

impl TryFrom<wire::ImageEncoderInput> for ImageEncoderInput {
    type Error = WireError;

    fn try_from(input: wire::ImageEncoderInput) -> Result<Self, WireError> {
        let encoder = match known::<wire::ImageEncoder>(input.encoder, "image encoder")? {
            wire::ImageEncoder::Vision => ImageEncoder::Vision,
            wire::ImageEncoder::Latent => ImageEncoder::Latent,
        };
        Ok(Self {
            encoder,
            num_cache_tokens: input.num_cache_tokens,
            max_cache_tokens: input.max_cache_tokens,
            height: input.height,
            width: input.width,
        })
    }
}

impl From<&SamplingParams> for wire::SamplingParams {
    fn from(sampling: &SamplingParams) -> Self {
        Self {
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            ignore_eos: sampling.ignore_eos,
            seed: sampling.seed,
            min_p: sampling.min_p,
            repetition_penalty: sampling.repetition_penalty,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
            logit_bias: sampling
                .logit_bias
                .iter()
                .map(|&(token, bias)| wire::LogitBias { token, bias })
                .collect(),
            min_tokens: sampling.min_tokens as u64,
            return_logprobs: sampling.return_logprobs,
            n_logprobs: sampling.n_logprobs,
            return_prompt_logprobs: sampling.return_prompt_logprobs,
            n_prompt_logprobs: sampling.n_prompt_logprobs,
            bad_words_ids: sampling
                .bad_words_ids
                .iter()
                .map(|tokens| wire::TokenSequence {
                    tokens: tokens.clone(),
                })
                .collect(),
            allowed_token_ids: sampling.allowed_token_ids.as_ref().map(|tokens| {
                wire::TokenSequence {
                    tokens: tokens.clone(),
                }
            }),
            typical_p: sampling.typical_p,
            forced_token_ids: sampling.forced_token_ids.clone(),
        }
    }
}

impl TryFrom<wire::SamplingParams> for SamplingParams {
    type Error = WireError;

    fn try_from(sampling: wire::SamplingParams) -> Result<Self, WireError> {
        Ok(Self {
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            ignore_eos: sampling.ignore_eos,
            seed: sampling.seed,
            min_p: sampling.min_p,
            repetition_penalty: sampling.repetition_penalty,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
            logit_bias: sampling
                .logit_bias
                .into_iter()
                .map(|bias| (bias.token, bias.bias))
                .collect(),
            min_tokens: count(sampling.min_tokens, "min_tokens")?,
            return_logprobs: sampling.return_logprobs,
            n_logprobs: sampling.n_logprobs,
            return_prompt_logprobs: sampling.return_prompt_logprobs,
            n_prompt_logprobs: sampling.n_prompt_logprobs,
            bad_words_ids: sampling
                .bad_words_ids
                .into_iter()
                .map(|sequence| sequence.tokens)
                .collect(),
            allowed_token_ids: sampling.allowed_token_ids.map(|sequence| sequence.tokens),
            typical_p: sampling.typical_p,
            forced_token_ids: sampling.forced_token_ids,
        })
    }
}

impl From<&DiffusionRequest> for wire::DiffusionRequest {
    fn from(request: &DiffusionRequest) -> Self {
        let sampling = &request.sampling;
        Self {
            prompt_token_ids: request.prompt_token_ids.clone(),
            conditioning: request.conditioning.clone(),
            conditions: request
                .conditions
                .iter()
                .map(|condition| wire::ConditionExtent {
                    pixel_bytes: condition.pixel_bytes,
                    audio_samples: condition.audio_samples,
                    vision_patches: condition.vision_patches,
                    vision_tokens: condition.vision_tokens,
                    latent_units: condition.latent_units.clone(),
                    audio_rows: condition.audio_rows,
                })
                .collect(),
            sampling: Some(wire::DiffusionSamplingParams {
                num_frames: sampling.num_frames,
                video_units: sampling.video_units,
                seed: sampling.seed,
                width: sampling.width,
                height: sampling.height,
                condition_rows: sampling.condition_rows,
            }),
            trajectory: request.trajectory.id,
            denoised_steps: request.trajectory.denoised_steps,
            state_seed: request.trajectory.state_seed,
            stored_layout: request.trajectory.stored_layout.clone(),
            encoder_inputs: request
                .encoder_inputs
                .as_ref()
                .map(|inputs| wire::EncoderInputs {
                    text: inputs.text.to_vec(),
                    vision: inputs.vision.map(|digest| digest.to_vec()),
                    video_units: inputs
                        .video_units
                        .iter()
                        .map(|digest| digest.to_vec())
                        .collect(),
                    audio: inputs.audio.map(|digest| digest.to_vec()),
                }),
        }
    }
}

/// A SHA-256 digest from its wire bytes.
fn digest(bytes: &[u8], field: &str) -> Result<[u8; 32], WireError> {
    bytes
        .try_into()
        .map_err(|_| WireError::new(format!("{field} is not a SHA-256 digest")))
}

impl TryFrom<wire::DiffusionRequest> for DiffusionRequest {
    type Error = WireError;

    /// The request without its identity and priority, which the enclosing
    /// [`wire::Request`] states.
    fn try_from(request: wire::DiffusionRequest) -> Result<Self, WireError> {
        let sampling = present(request.sampling, "diffusion sampling")?;
        Ok(Self {
            request_id: RequestId::default(),
            prompt_token_ids: request.prompt_token_ids,
            conditioning: request.conditioning,
            conditions: request
                .conditions
                .into_iter()
                .map(|condition| ConditionExtent {
                    pixel_bytes: condition.pixel_bytes,
                    audio_samples: condition.audio_samples,
                    vision_patches: condition.vision_patches,
                    vision_tokens: condition.vision_tokens,
                    latent_units: condition.latent_units,
                    audio_rows: condition.audio_rows,
                })
                .collect(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: sampling.num_frames,
                video_units: sampling.video_units,
                seed: sampling.seed,
                width: sampling.width,
                height: sampling.height,
                condition_rows: sampling.condition_rows,
            },
            trajectory: Trajectory {
                id: request.trajectory,
                denoised_steps: request.denoised_steps,
                state_seed: request.state_seed,
                stored_layout: request.stored_layout,
            },
            encoder_inputs: request
                .encoder_inputs
                .map(|inputs| -> Result<Box<EncoderInputs>, WireError> {
                    Ok(Box::new(EncoderInputs {
                        text: digest(&inputs.text, "the text encoding's inputs")?,
                        vision: inputs
                            .vision
                            .map(|bytes| digest(&bytes, "the vision encoding's inputs"))
                            .transpose()?,
                        video_units: inputs
                            .video_units
                            .iter()
                            .map(|bytes| digest(bytes, "a condition unit's inputs"))
                            .collect::<Result<_, _>>()?,
                        audio: inputs
                            .audio
                            .map(|bytes| digest(&bytes, "the audio encoding's inputs"))
                            .transpose()?,
                    }))
                })
                .transpose()?,
        })
    }
}

// ------------------------------------------------------------------- events

impl Event {
    /// The message of this event of request `request`, and for an artifact
    /// the media whose descriptor travels beside the message.
    pub fn to_wire(&self, request: RequestId) -> (wire::Event, Option<Arc<dyn ArtifactMedia>>) {
        use wire::event::Kind;

        let mut media = None;
        let kind = match self {
            Self::Scheduled {
                queued_at,
                scheduled_at,
                cached_tokens,
            } => Kind::Scheduled(wire::Scheduled {
                queued_at: *queued_at,
                scheduled_at: *scheduled_at,
                cached_tokens: *cached_tokens as u64,
            }),
            Self::TextToken { id, logprob } => Kind::TextToken(wire::TextToken {
                id: *id,
                logprob: *logprob,
            }),
            Self::TokenLogprobs { id, candidates } => Kind::TokenLogprobs(wire::TokenLogprobs {
                id: *id,
                candidates: candidates.iter().map(Into::into).collect(),
            }),
            Self::PromptLogprobs { positions } => Kind::PromptLogprobs(wire::PromptLogprobs {
                positions: positions
                    .iter()
                    .map(|position| wire::PositionLogprobs {
                        entries: position.entries.iter().map(Into::into).collect(),
                    })
                    .collect(),
            }),
            Self::MediaProgress {
                phase,
                completed_steps,
            } => Kind::MediaProgress(wire::MediaProgress {
                phase: phase.clone(),
                completed_steps: *completed_steps,
            }),
            Self::Artifact(artifact) => {
                let media_kind = match artifact.media_kind {
                    MediaKind::Image => wire::MediaKind::Image,
                    MediaKind::Video => wire::MediaKind::Video,
                    MediaKind::Audio => wire::MediaKind::Audio,
                };
                media = Some(Arc::clone(&artifact.media));
                Kind::Artifact(wire::Artifact {
                    media_kind: media_kind as i32,
                    content_type: artifact.content_type.clone(),
                    bytes: artifact.media.bytes().len() as u64,
                })
            }
            Self::ArtifactUnavailable { message } => {
                Kind::ArtifactUnavailable(wire::ArtifactUnavailable {
                    message: message.clone(),
                })
            }
            Self::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
            } => Kind::Finished(wire::Finished {
                reason: wire::FinishReason::from(reason) as i32,
                stop_reason: stop_reason.as_ref().map(|stop| wire::StopReason {
                    kind: Some(match stop {
                        StopReason::Token(token) => wire::stop_reason::Kind::Token(*token),
                        StopReason::String(text) => wire::stop_reason::Kind::Text(text.clone()),
                    }),
                }),
                prompt_tokens: *prompt_tokens as u64,
                completion_tokens: *completion_tokens as u64,
            }),
            Self::Rejected { kind, message } => Kind::Rejected(wire::Rejected {
                kind: match kind {
                    RejectionKind::Invalid => wire::RejectionKind::Invalid,
                    RejectionKind::Overloaded => wire::RejectionKind::Overloaded,
                } as i32,
                message: message.clone(),
            }),
            Self::Error { message } => Kind::Error(wire::Failed {
                message: message.clone(),
            }),
            Self::Migratable {
                cause,
                denoised_steps,
                stored_layout,
            } => Kind::Migratable(wire::Migratable {
                cause: match cause {
                    MigrationCause::Drain => wire::MigrationCause::Drain,
                    MigrationCause::Loss => wire::MigrationCause::Loss,
                } as i32,
                denoised_steps: *denoised_steps,
                stored_layout: stored_layout.clone(),
            }),
        };
        let event = wire::Event {
            request: request.0,
            kind: Some(kind),
        };
        (event, media)
    }

    /// The event a message states, with the request it belongs to. An
    /// artifact's bytes are `media`, which the descriptor that came with
    /// the message maps.
    ///
    /// # Errors
    ///
    /// Fails when the message names no event, a value this build does not
    /// know, or an artifact without its media.
    pub fn from_wire(
        event: wire::Event,
        media: Option<Arc<dyn ArtifactMedia>>,
    ) -> Result<(RequestId, Self), WireError> {
        use wire::event::Kind;

        let request = RequestId(event.request);
        let event = match present(event.kind, "event kind")? {
            Kind::Scheduled(scheduled) => Self::Scheduled {
                queued_at: scheduled.queued_at,
                scheduled_at: scheduled.scheduled_at,
                cached_tokens: count(scheduled.cached_tokens, "cached tokens")?,
            },
            Kind::TextToken(token) => Self::TextToken {
                id: token.id,
                logprob: token.logprob,
            },
            Kind::TokenLogprobs(logprobs) => Self::TokenLogprobs {
                id: logprobs.id,
                candidates: logprobs.candidates.into_iter().map(Into::into).collect(),
            },
            Kind::PromptLogprobs(logprobs) => Self::PromptLogprobs {
                positions: logprobs
                    .positions
                    .into_iter()
                    .map(|position| PositionLogprobs {
                        entries: position.entries.into_iter().map(Into::into).collect(),
                    })
                    .collect(),
            },
            Kind::MediaProgress(progress) => Self::MediaProgress {
                phase: progress.phase,
                completed_steps: progress.completed_steps,
            },
            Kind::Artifact(artifact) => {
                let media_kind = match known::<wire::MediaKind>(artifact.media_kind, "media kind")?
                {
                    wire::MediaKind::Image => MediaKind::Image,
                    wire::MediaKind::Video => MediaKind::Video,
                    wire::MediaKind::Audio => MediaKind::Audio,
                };
                Self::Artifact(ArtifactEvent {
                    media_kind,
                    content_type: artifact.content_type,
                    media: present(media, "artifact's media")?,
                })
            }
            Kind::ArtifactUnavailable(unavailable) => Self::ArtifactUnavailable {
                message: unavailable.message,
            },
            Kind::Finished(finished) => Self::Finished {
                reason: known::<wire::FinishReason>(finished.reason, "finish reason")?.into(),
                stop_reason: finished
                    .stop_reason
                    .map(|stop| match present(stop.kind, "stop reason kind")? {
                        wire::stop_reason::Kind::Token(token) => Ok(StopReason::Token(token)),
                        wire::stop_reason::Kind::Text(text) => Ok(StopReason::String(text)),
                    })
                    .transpose()?,
                prompt_tokens: count(finished.prompt_tokens, "prompt tokens")?,
                completion_tokens: count(finished.completion_tokens, "completion tokens")?,
            },
            Kind::Rejected(rejected) => Self::Rejected {
                kind: match known::<wire::RejectionKind>(rejected.kind, "rejection kind")? {
                    wire::RejectionKind::Invalid => RejectionKind::Invalid,
                    wire::RejectionKind::Overloaded => RejectionKind::Overloaded,
                },
                message: rejected.message,
            },
            Kind::Error(failed) => Self::Error {
                message: failed.message,
            },
            Kind::Migratable(migratable) => Self::Migratable {
                cause: match known::<wire::MigrationCause>(migratable.cause, "migration cause")? {
                    wire::MigrationCause::Drain => MigrationCause::Drain,
                    wire::MigrationCause::Loss => MigrationCause::Loss,
                },
                denoised_steps: migratable.denoised_steps,
                stored_layout: migratable.stored_layout,
            },
        };
        Ok((request, event))
    }
}

impl From<&TokenLogprob> for wire::TokenLogprob {
    fn from(logprob: &TokenLogprob) -> Self {
        Self {
            token_id: logprob.token_id,
            logprob: logprob.logprob,
            rank: logprob.rank,
        }
    }
}

impl From<wire::TokenLogprob> for TokenLogprob {
    fn from(logprob: wire::TokenLogprob) -> Self {
        Self {
            token_id: logprob.token_id,
            logprob: logprob.logprob,
            rank: logprob.rank,
        }
    }
}

impl From<&FinishReason> for wire::FinishReason {
    fn from(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Completed => Self::Completed,
            FinishReason::Eos => Self::Eos,
            FinishReason::MaxTokens => Self::MaxTokens,
            FinishReason::Stop => Self::Stop,
            FinishReason::Cancelled => Self::Cancelled,
            FinishReason::Aborted => Self::Aborted,
            FinishReason::Repetition => Self::Repetition,
            FinishReason::Error => Self::Error,
            FinishReason::CacheCapacity => Self::CacheCapacity,
        }
    }
}

impl From<wire::FinishReason> for FinishReason {
    fn from(reason: wire::FinishReason) -> Self {
        match reason {
            wire::FinishReason::Completed => Self::Completed,
            wire::FinishReason::Eos => Self::Eos,
            wire::FinishReason::MaxTokens => Self::MaxTokens,
            wire::FinishReason::Stop => Self::Stop,
            wire::FinishReason::Cancelled => Self::Cancelled,
            wire::FinishReason::Aborted => Self::Aborted,
            wire::FinishReason::Repetition => Self::Repetition,
            wire::FinishReason::Error => Self::Error,
            wire::FinishReason::CacheCapacity => Self::CacheCapacity,
        }
    }
}

// ------------------------------------------------------------------- limits

impl From<&GenerationLimits> for wire::GenerationLimits {
    fn from(limits: &GenerationLimits) -> Self {
        let features = limits.features;
        Self {
            understanding: features.contains(GenerationFeatures::UNDERSTANDING),
            vision_encode: features.contains(GenerationFeatures::VISION_ENCODE),
            latent_encode: features.contains(GenerationFeatures::LATENT_ENCODE),
            max_vae_grid_tokens: limits.max_vae_grid_tokens,
            max_vit_grid_tokens: limits.max_vit_grid_tokens,
            max_latent_feature_bytes: limits.max_latent_feature_bytes,
            max_vision_feature_bytes: limits.max_vision_feature_bytes,
            encoder_cache_entries: limits.encoder_cache_entries,
            max_logprobs: limits.max_logprobs,
        }
    }
}

impl From<wire::GenerationLimits> for GenerationLimits {
    fn from(limits: wire::GenerationLimits) -> Self {
        let mut features = GenerationFeatures::empty();
        for (present, feature) in [
            (limits.understanding, GenerationFeatures::UNDERSTANDING),
            (limits.vision_encode, GenerationFeatures::VISION_ENCODE),
            (limits.latent_encode, GenerationFeatures::LATENT_ENCODE),
        ] {
            features.set(feature, present);
        }
        Self {
            features,
            max_vae_grid_tokens: limits.max_vae_grid_tokens,
            max_vit_grid_tokens: limits.max_vit_grid_tokens,
            max_latent_feature_bytes: limits.max_latent_feature_bytes,
            max_vision_feature_bytes: limits.max_vision_feature_bytes,
            encoder_cache_entries: limits.encoder_cache_entries,
            max_logprobs: limits.max_logprobs,
        }
    }
}
