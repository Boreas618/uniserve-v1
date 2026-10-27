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
    ArtifactEvent, ArtifactMedia, CachePolicy, CfgRenorm, ConditionExtent, DiffusionRequest,
    DiffusionSamplingParams, Event, FeedbackNextToken, FeedbackSource, FinishReason,
    GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationRequest, ImageEncoder,
    ImageEncoderInput, ImageGenerationConfig, ImageInput, ImageParams, ImageTrigger, MediaKind,
    MigrationCause, MultimodalInputs, PositionLogprobs, ReadoutRow, ReadoutSlot, RejectionKind,
    Request, RequestId, SamplingParams, StopReason, TokenLogprob, Trajectory,
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
            negative_prompt_token_ids: request.negative_prompt_token_ids.clone(),
            images: request
                .multimodal_inputs
                .images
                .iter()
                .map(Into::into)
                .collect(),
            constraint: wire::GenerationConstraint::from(request.constraint) as i32,
            sampling: Some((&request.sampling).into()),
            image: Some((&request.image).into()),
            max_images: u32::from(request.max_images),
            max_und_tokens: request.max_und_tokens as u64,
            stop_strings: request.stop_strings.clone(),
            stop_token_ids: request.stop_token_ids.clone(),
            cache: Some(wire::CachePolicy {
                read: request.cache.read,
                write: request.cache.write,
                isolation_key: request.cache.isolation_key,
                state_seed: request.cache.state_seed,
            }),
            include_stop_token: request.include_stop_token,
            image_generation: Some((&request.image_generation).into()),
            readout: request
                .readout
                .iter()
                .map(|row| wire::ReadoutRow {
                    token_ids: row.token_ids.clone(),
                    slots: row
                        .slots
                        .iter()
                        .map(|slot| wire::ReadoutSlot {
                            position: slot.position,
                            candidates: slot.candidates.clone(),
                        })
                        .collect(),
                })
                .collect(),
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
            negative_prompt_token_ids: request.negative_prompt_token_ids,
            multimodal_inputs: MultimodalInputs {
                images: request
                    .images
                    .into_iter()
                    .map(ImageInput::try_from)
                    .collect::<Result<_, _>>()?,
            },
            constraint: known::<wire::GenerationConstraint>(request.constraint, "constraint")?
                .into(),
            sampling: present(request.sampling, "sampling parameters")?.try_into()?,
            image: present(request.image, "image parameters")?.try_into()?,
            max_images: u16::try_from(request.max_images)
                .map_err(|_| WireError::new("max_images exceeds 65535"))?,
            max_und_tokens: count(request.max_und_tokens, "max_und_tokens")?,
            stop_strings: request.stop_strings,
            stop_token_ids: request.stop_token_ids,
            priority: 0,
            cache: CachePolicy {
                read: cache.read,
                write: cache.write,
                isolation_key: cache.isolation_key,
                state_seed: cache.state_seed,
            },
            include_stop_token: request.include_stop_token,
            image_generation: present(request.image_generation, "image generation")?.try_into()?,
            readout: request
                .readout
                .into_iter()
                .map(|row| ReadoutRow {
                    token_ids: row.token_ids,
                    slots: row
                        .slots
                        .into_iter()
                        .map(|slot| ReadoutSlot {
                            position: slot.position,
                            candidates: slot.candidates,
                        })
                        .collect(),
                })
                .collect(),
            generated_token_ids: request.generated_token_ids,
            adapter: request.adapter,
        })
    }
}

impl From<GenerationConstraint> for wire::GenerationConstraint {
    fn from(constraint: GenerationConstraint) -> Self {
        match constraint {
            GenerationConstraint::Default => Self::Default,
            GenerationConstraint::UndOnly => Self::UndOnly,
            GenerationConstraint::GenOnly => Self::GenOnly,
        }
    }
}

impl From<wire::GenerationConstraint> for GenerationConstraint {
    fn from(constraint: wire::GenerationConstraint) -> Self {
        match constraint {
            wire::GenerationConstraint::Default => Self::Default,
            wire::GenerationConstraint::UndOnly => Self::UndOnly,
            wire::GenerationConstraint::GenOnly => Self::GenOnly,
        }
    }
}

impl From<&ImageInput> for wire::ImageInput {
    fn from(image: &ImageInput) -> Self {
        Self {
            hash: image.hash,
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
            hash: image.hash,
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

impl From<&ImageParams> for wire::ImageParams {
    fn from(image: &ImageParams) -> Self {
        let renorm = match image.cfg_renorm_type {
            CfgRenorm::Global => wire::CfgRenorm::Global,
            CfgRenorm::None => wire::CfgRenorm::None,
            CfgRenorm::TextChannel => wire::CfgRenorm::TextChannel,
        };
        Self {
            steps: u32::from(image.steps),
            cfg_text_scale: image.cfg_text_scale,
            cfg_img_scale: image.cfg_img_scale,
            cfg_renorm_type: renorm as i32,
            cfg_renorm_min: image.cfg_renorm_min,
            cfg_interval_start: image.cfg_interval.0,
            cfg_interval_end: image.cfg_interval.1,
            timestep_shift: image.timestep_shift,
            height: image.height,
            width: image.width,
            seed: image.seed,
        }
    }
}

impl TryFrom<wire::ImageParams> for ImageParams {
    type Error = WireError;

    fn try_from(image: wire::ImageParams) -> Result<Self, WireError> {
        let cfg_renorm_type = match known::<wire::CfgRenorm>(image.cfg_renorm_type, "cfg renorm")? {
            wire::CfgRenorm::Global => CfgRenorm::Global,
            wire::CfgRenorm::None => CfgRenorm::None,
            wire::CfgRenorm::TextChannel => CfgRenorm::TextChannel,
        };
        Ok(Self {
            steps: u16::try_from(image.steps)
                .map_err(|_| WireError::new("image steps exceed 65535"))?,
            cfg_text_scale: image.cfg_text_scale,
            cfg_img_scale: image.cfg_img_scale,
            cfg_renorm_type,
            cfg_renorm_min: image.cfg_renorm_min,
            cfg_interval: (image.cfg_interval_start, image.cfg_interval_end),
            timestep_shift: image.timestep_shift,
            height: image.height,
            width: image.width,
            seed: image.seed,
        })
    }
}

impl From<&ImageGenerationConfig> for wire::ImageGenerationConfig {
    fn from(config: &ImageGenerationConfig) -> Self {
        let trigger = match &config.trigger {
            ImageTrigger::Disabled => wire::image_trigger::Kind::Disabled(wire::Disabled {}),
            ImageTrigger::Token { token_id } => wire::image_trigger::Kind::Token(*token_id),
            ImageTrigger::Suffix { token_ids } => {
                wire::image_trigger::Kind::Suffix(wire::TokenSequence {
                    tokens: token_ids.clone(),
                })
            }
            ImageTrigger::RoundCloseThenSuffix {
                close_token_ids,
                trigger_token_ids,
            } => wire::image_trigger::Kind::RoundCloseThenSuffix(wire::RoundCloseThenSuffix {
                close_token_ids: close_token_ids.clone(),
                trigger_token_ids: trigger_token_ids.clone(),
            }),
        };
        let feedback_source = match config.feedback_source {
            None => wire::FeedbackSource::None,
            Some(FeedbackSource::DeviceProduct) => wire::FeedbackSource::DeviceProduct,
            Some(FeedbackSource::ArtifactProduct) => wire::FeedbackSource::ArtifactProduct,
        };
        let next = match config.feedback_next_token {
            FeedbackNextToken::None => wire::feedback_next_token::Kind::None(wire::Disabled {}),
            FeedbackNextToken::Bos => wire::feedback_next_token::Kind::Bos(wire::Disabled {}),
            FeedbackNextToken::EndOfImage => {
                wire::feedback_next_token::Kind::EndOfImage(wire::Disabled {})
            }
            FeedbackNextToken::Token { token_id } => {
                wire::feedback_next_token::Kind::Token(token_id)
            }
        };
        Self {
            trigger: Some(wire::ImageTrigger {
                kind: Some(trigger),
            }),
            requires_text_for_image: config.requires_text_for_image,
            feedback_source: feedback_source as i32,
            feedback_encoders: config.feedback_encoders.iter().map(Into::into).collect(),
            num_feedback_positions: config.num_feedback_positions,
            feedback_next_token: Some(wire::FeedbackNextToken { kind: Some(next) }),
            sample_feedback_continuation: config.sample_feedback_continuation,
        }
    }
}

impl TryFrom<wire::ImageGenerationConfig> for ImageGenerationConfig {
    type Error = WireError;

    fn try_from(config: wire::ImageGenerationConfig) -> Result<Self, WireError> {
        let trigger = match present(
            present(config.trigger, "image trigger")?.kind,
            "image trigger kind",
        )? {
            wire::image_trigger::Kind::Disabled(_) => ImageTrigger::Disabled,
            wire::image_trigger::Kind::Token(token_id) => ImageTrigger::Token { token_id },
            wire::image_trigger::Kind::Suffix(sequence) => ImageTrigger::Suffix {
                token_ids: sequence.tokens,
            },
            wire::image_trigger::Kind::RoundCloseThenSuffix(round) => {
                ImageTrigger::RoundCloseThenSuffix {
                    close_token_ids: round.close_token_ids,
                    trigger_token_ids: round.trigger_token_ids,
                }
            }
        };
        let feedback_source =
            match known::<wire::FeedbackSource>(config.feedback_source, "feedback source")? {
                wire::FeedbackSource::None => None,
                wire::FeedbackSource::DeviceProduct => Some(FeedbackSource::DeviceProduct),
                wire::FeedbackSource::ArtifactProduct => Some(FeedbackSource::ArtifactProduct),
            };
        let feedback_next_token = match present(
            present(config.feedback_next_token, "feedback next token")?.kind,
            "feedback next token kind",
        )? {
            wire::feedback_next_token::Kind::None(_) => FeedbackNextToken::None,
            wire::feedback_next_token::Kind::Bos(_) => FeedbackNextToken::Bos,
            wire::feedback_next_token::Kind::EndOfImage(_) => FeedbackNextToken::EndOfImage,
            wire::feedback_next_token::Kind::Token(token_id) => {
                FeedbackNextToken::Token { token_id }
            }
        };
        Ok(Self {
            trigger,
            requires_text_for_image: config.requires_text_for_image,
            feedback_source,
            feedback_encoders: config
                .feedback_encoders
                .into_iter()
                .map(ImageEncoderInput::try_from)
                .collect::<Result<_, _>>()?,
            num_feedback_positions: config.num_feedback_positions,
            feedback_next_token,
            sample_feedback_continuation: config.sample_feedback_continuation,
        })
    }
}

impl From<&DiffusionRequest> for wire::DiffusionRequest {
    fn from(request: &DiffusionRequest) -> Self {
        let sampling = &request.sampling;
        Self {
            task: request.task.clone(),
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
            }),
            trajectory: request.trajectory.id,
            denoised_steps: request.trajectory.denoised_steps,
            state_seed: request.trajectory.state_seed,
        }
    }
}

impl TryFrom<wire::DiffusionRequest> for DiffusionRequest {
    type Error = WireError;

    /// The request without its identity and priority, which the enclosing
    /// [`wire::Request`] states.
    fn try_from(request: wire::DiffusionRequest) -> Result<Self, WireError> {
        let sampling = present(request.sampling, "diffusion sampling")?;
        Ok(Self {
            request_id: RequestId::default(),
            task: request.task,
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
            },
            trajectory: Trajectory {
                id: request.trajectory,
                denoised_steps: request.denoised_steps,
                state_seed: request.state_seed,
            },
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
            } => Kind::Scheduled(wire::Scheduled {
                queued_at: *queued_at,
                scheduled_at: *scheduled_at,
            }),
            Self::TextToken { id, logprob } => Kind::TextToken(wire::TextToken {
                id: *id,
                logprob: *logprob,
            }),
            Self::TextTokens { ids } => Kind::TextTokens(wire::TextTokens { ids: ids.clone() }),
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
            Self::Readout { candidate_logprobs } => Kind::Readout(wire::Readout {
                candidate_logprobs: candidate_logprobs.clone(),
            }),
            Self::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } => Kind::ImageBegin(wire::ImageBegin {
                image_id: *image_id,
                height: *height,
                width: *width,
                steps: u32::from(*steps),
            }),
            Self::ImageStep { image_id, step } => Kind::ImageStep(wire::ImageStep {
                image_id: *image_id,
                step: u32::from(*step),
            }),
            Self::ImageCommit { image_id } => Kind::ImageCommit(wire::ImageCommit {
                image_id: *image_id,
            }),
            Self::ImageDone {
                image_id,
                height,
                width,
                bytes,
                pixels_png_b64,
            } => Kind::ImageDone(wire::ImageDone {
                image_id: *image_id,
                height: *height,
                width: *width,
                bytes: *bytes,
                png_b64: pixels_png_b64.clone(),
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
                images,
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
                images: *images as u64,
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
            } => Kind::Migratable(wire::Migratable {
                cause: match cause {
                    MigrationCause::Drain => wire::MigrationCause::Drain,
                    MigrationCause::Loss => wire::MigrationCause::Loss,
                } as i32,
                denoised_steps: *denoised_steps,
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
            },
            Kind::TextToken(token) => Self::TextToken {
                id: token.id,
                logprob: token.logprob,
            },
            Kind::TextTokens(tokens) => Self::TextTokens { ids: tokens.ids },
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
            Kind::Readout(readout) => Self::Readout {
                candidate_logprobs: readout.candidate_logprobs,
            },
            Kind::ImageBegin(begin) => Self::ImageBegin {
                image_id: begin.image_id,
                height: begin.height,
                width: begin.width,
                steps: u16::try_from(begin.steps)
                    .map_err(|_| WireError::new("image steps exceed 65535"))?,
            },
            Kind::ImageStep(step) => Self::ImageStep {
                image_id: step.image_id,
                step: u16::try_from(step.step)
                    .map_err(|_| WireError::new("an image step exceeds 65535"))?,
            },
            Kind::ImageCommit(commit) => Self::ImageCommit {
                image_id: commit.image_id,
            },
            Kind::ImageDone(done) => Self::ImageDone {
                image_id: done.image_id,
                height: done.height,
                width: done.width,
                bytes: done.bytes,
                pixels_png_b64: done.png_b64,
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
                images: count(finished.images, "images")?,
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
            FinishReason::ImageDone => Self::ImageDone,
            FinishReason::Cancelled => Self::Cancelled,
            FinishReason::Aborted => Self::Aborted,
            FinishReason::Repetition => Self::Repetition,
            FinishReason::Error => Self::Error,
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
            wire::FinishReason::ImageDone => Self::ImageDone,
            wire::FinishReason::Cancelled => Self::Cancelled,
            wire::FinishReason::Aborted => Self::Aborted,
            wire::FinishReason::Repetition => Self::Repetition,
            wire::FinishReason::Error => Self::Error,
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
            image_generation: features.contains(GenerationFeatures::IMAGE_GENERATION),
            token_denoising: features.contains(GenerationFeatures::TOKEN_DENOISING),
            max_latent_units: limits.max_latent_units,
            latent_downsample: limits.latent_downsample,
            max_vae_grid_tokens: limits.max_vae_grid_tokens,
            max_vit_grid_tokens: limits.max_vit_grid_tokens,
            latent_bytes_per_unit: limits.latent_bytes_per_unit,
            max_latent_feature_bytes: limits.max_latent_feature_bytes,
            max_vision_feature_bytes: limits.max_vision_feature_bytes,
            commit_marker_tokens: limits.commit_marker_tokens,
            max_cfg_branches: limits.max_cfg_branches,
            encoder_cache_entries: limits.encoder_cache_entries,
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
            (
                limits.image_generation,
                GenerationFeatures::IMAGE_GENERATION,
            ),
            (limits.token_denoising, GenerationFeatures::TOKEN_DENOISING),
        ] {
            features.set(feature, present);
        }
        Self {
            features,
            max_latent_units: limits.max_latent_units,
            latent_downsample: limits.latent_downsample,
            max_vae_grid_tokens: limits.max_vae_grid_tokens,
            max_vit_grid_tokens: limits.max_vit_grid_tokens,
            latent_bytes_per_unit: limits.latent_bytes_per_unit,
            max_latent_feature_bytes: limits.max_latent_feature_bytes,
            max_vision_feature_bytes: limits.max_vision_feature_bytes,
            commit_marker_tokens: limits.commit_marker_tokens,
            max_cfg_branches: limits.max_cfg_branches,
            encoder_cache_entries: limits.encoder_cache_entries,
        }
    }
}
