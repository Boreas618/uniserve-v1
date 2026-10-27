//! Every request and event value comes back from its message as it went in,
//! and a message naming a value this build does not know is refused.

use std::sync::Arc;

use proto::request::{
    ArtifactEvent, ArtifactMedia, CachePolicy, CfgRenorm, ConditionExtent, DiffusionRequest,
    DiffusionSamplingParams, Event, FeedbackNextToken, FeedbackSource, FinishReason,
    GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationRequest, ImageEncoder,
    ImageEncoderInput, ImageGenerationConfig, ImageInput, ImageParams, ImageTrigger, MediaKind,
    MigrationCause, MultimodalInputs, PositionLogprobs, ReadoutRow, ReadoutSlot, RejectionKind,
    Request, RequestId, SamplingParams, StopReason, TokenLogprob, Trajectory, wire,
};

/// A generation request with every field away from its default.
fn generation() -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(41),
        prompt_token_ids: vec![1, 2, 3, 4],
        negative_prompt_token_ids: vec![9],
        multimodal_inputs: MultimodalInputs {
            images: vec![ImageInput {
                hash: 0xfeed,
                b64: "aW1hZ2U=".to_owned(),
                position: 2,
                num_positions: 6,
                encoders: vec![
                    ImageEncoderInput {
                        encoder: ImageEncoder::Latent,
                        num_cache_tokens: Some(4),
                        max_cache_tokens: None,
                        height: None,
                        width: None,
                    },
                    ImageEncoderInput {
                        encoder: ImageEncoder::Vision,
                        num_cache_tokens: None,
                        max_cache_tokens: Some(16),
                        height: None,
                        width: None,
                    },
                ],
            }],
        },
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.9,
            ignore_eos: true,
            seed: Some(7),
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            logit_bias: vec![(5, -1.5)],
            min_tokens: 3,
            return_logprobs: true,
            n_logprobs: 2,
            return_prompt_logprobs: true,
            n_prompt_logprobs: 1,
            bad_words_ids: vec![vec![6, 7]],
            allowed_token_ids: Some(vec![1, 2]),
            typical_p: 0.95,
            forced_token_ids: vec![8],
        },
        image: ImageParams {
            steps: 12,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.5,
            cfg_renorm_type: CfgRenorm::TextChannel,
            cfg_renorm_min: 0.1,
            cfg_interval: (0.2, 0.8),
            timestep_shift: 3.0,
            height: 512,
            width: 768,
            seed: Some(11),
        },
        max_images: 2,
        max_und_tokens: 128,
        stop_strings: vec!["end".to_owned()],
        stop_token_ids: vec![2],
        priority: -3,
        cache: CachePolicy {
            read: false,
            write: true,
            isolation_key: Some(99),
            state_seed: Some(7),
        },
        include_stop_token: true,
        image_generation: ImageGenerationConfig {
            trigger: ImageTrigger::RoundCloseThenSuffix {
                close_token_ids: vec![10],
                trigger_token_ids: vec![11, 12],
            },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::ArtifactProduct),
            feedback_encoders: vec![ImageEncoderInput {
                encoder: ImageEncoder::Vision,
                num_cache_tokens: Some(8),
                max_cache_tokens: Some(8),
                height: None,
                width: None,
            }],
            num_feedback_positions: 9,
            feedback_next_token: FeedbackNextToken::Token { token_id: 13 },
            sample_feedback_continuation: true,
        },
        readout: vec![ReadoutRow {
            token_ids: vec![20, 21],
            slots: vec![ReadoutSlot {
                position: 1,
                candidates: vec![30, 31],
            }],
        }],
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

#[test]
fn requests_come_back_from_their_messages_unchanged() {
    let requests = [
        Request::from(generation()),
        Request::Diffusion(DiffusionRequest {
            request_id: RequestId(42),
            task: "ref2va".to_owned(),
            prompt_token_ids: vec![5, 6],
            conditioning: b"{\"task\":\"ref2va\"}".to_vec(),
            conditions: vec![ConditionExtent {
                pixel_bytes: 3 * 64 * 32,
                audio_samples: 100,
                vision_patches: 8,
                vision_tokens: 2,
                latent_units: vec![2, 1],
                audio_rows: 4,
            }],
            priority: 5,
            sampling: DiffusionSamplingParams {
                num_frames: 22,
                video_units: 2,
                seed: 3,
                width: 64,
                height: 32,
            },
            // A resumed request names its trajectory, the steps an engine
            // stored and the latent seed it continues with.
            trajectory: Trajectory {
                id: 11,
                denoised_steps: 4,
                state_seed: Some(13),
            },
        }),
    ];
    for request in requests {
        let message = request.to_wire();
        assert_eq!(Request::from_wire(message).unwrap(), request);
    }

    // Each image trigger and continuation token comes back.
    for (trigger, next) in [
        (ImageTrigger::Disabled, FeedbackNextToken::None),
        (ImageTrigger::Token { token_id: 3 }, FeedbackNextToken::Bos),
        (
            ImageTrigger::Suffix {
                token_ids: vec![4, 5],
            },
            FeedbackNextToken::EndOfImage,
        ),
    ] {
        let mut request = generation();
        request.image_generation.trigger = trigger;
        request.image_generation.feedback_next_token = next;
        request.image_generation.feedback_source = None;
        let request = Request::from(request);
        assert_eq!(Request::from_wire(request.to_wire()).unwrap(), request);
    }
}

#[test]
fn a_value_this_build_does_not_know_is_refused() {
    let mut message = Request::from(generation()).to_wire();
    let Some(wire::request::Kind::Generation(generation)) = message.kind.as_mut() else {
        unreachable!("a generation request")
    };
    generation.constraint = 77;
    let error = Request::from_wire(message).unwrap_err();
    assert!(error.to_string().contains("constraint 77"), "{error}");
}

/// Artifact bytes held in this process.
#[derive(Debug)]
struct Bytes(Vec<u8>);

impl ArtifactMedia for Bytes {
    fn bytes(&self) -> &[u8] {
        &self.0
    }

    fn descriptor(&self) -> Option<std::os::fd::BorrowedFd<'_>> {
        None
    }
}

#[test]
fn events_come_back_from_their_messages_unchanged() {
    let candidate = TokenLogprob {
        token_id: 4,
        logprob: -0.5,
        rank: 1,
    };
    let events = [
        Event::Scheduled {
            queued_at: 1.5,
            scheduled_at: 2.5,
        },
        Event::TextToken {
            id: 3,
            logprob: Some(-0.25),
        },
        Event::TextTokens { ids: vec![1, 2] },
        Event::TokenLogprobs {
            id: 3,
            candidates: vec![candidate.clone()],
        },
        Event::PromptLogprobs {
            positions: vec![PositionLogprobs {
                entries: vec![candidate],
            }],
        },
        Event::Readout {
            candidate_logprobs: vec![-1.0, -2.0],
        },
        Event::ImageBegin {
            image_id: 1,
            height: 64,
            width: 32,
            steps: 6,
        },
        Event::ImageStep {
            image_id: 1,
            step: 2,
        },
        Event::ImageCommit { image_id: 1 },
        Event::ImageDone {
            image_id: 1,
            height: 64,
            width: 32,
            bytes: 10,
            pixels_png_b64: "cG5n".to_owned(),
        },
        Event::MediaProgress {
            phase: "denoising".to_owned(),
            completed_steps: 4,
        },
        Event::ArtifactUnavailable {
            message: "gone".to_owned(),
        },
        Event::Finished {
            reason: FinishReason::Stop,
            stop_reason: Some(StopReason::String("end".to_owned())),
            prompt_tokens: 4,
            completion_tokens: 9,
            images: 1,
        },
        Event::Finished {
            reason: FinishReason::Eos,
            stop_reason: Some(StopReason::Token(2)),
            prompt_tokens: 4,
            completion_tokens: 9,
            images: 0,
        },
        Event::Rejected {
            kind: RejectionKind::Overloaded,
            message: "full".to_owned(),
        },
        Event::Error {
            message: "lost".to_owned(),
        },
        Event::Migratable {
            cause: MigrationCause::Drain,
            denoised_steps: 3,
        },
    ];
    for event in events {
        let (message, media) = event.to_wire(RequestId(8));
        assert!(media.is_none());
        let (request, decoded) = Event::from_wire(message, None).unwrap();
        assert_eq!(request, RequestId(8));
        // Events compare by their messages: an artifact's media has no
        // equality of its own.
        assert_eq!(decoded.to_wire(request).0, event.to_wire(request).0);
    }

    // An artifact's message names its kind, type and length, and travels
    // with its media.
    let artifact = Event::Artifact(ArtifactEvent {
        media_kind: MediaKind::Video,
        content_type: "video/mp4".to_owned(),
        media: Arc::new(Bytes(b"mp4 bytes".to_vec())),
    });
    let (message, media) = artifact.to_wire(RequestId(9));
    let Some(wire::event::Kind::Artifact(fields)) = &message.kind else {
        unreachable!("an artifact")
    };
    assert_eq!(fields.bytes, 9);
    let media = media.expect("an artifact travels with its media");
    let (_, decoded) = Event::from_wire(message.clone(), Some(media)).unwrap();
    let Event::Artifact(decoded) = decoded else {
        unreachable!("an artifact")
    };
    assert_eq!(decoded.media_kind, MediaKind::Video);
    assert_eq!(decoded.content_type, "video/mp4");
    assert_eq!(decoded.media.bytes(), b"mp4 bytes");
    assert!(Event::from_wire(message, None).is_err());
}

#[test]
fn limits_come_back_from_their_message_unchanged() {
    let limits = GenerationLimits {
        features: GenerationFeatures::UNDERSTANDING | GenerationFeatures::LATENT_ENCODE,
        max_latent_units: 1,
        latent_downsample: 2,
        max_vae_grid_tokens: 3,
        max_vit_grid_tokens: 4,
        latent_bytes_per_unit: 5,
        max_latent_feature_bytes: 6,
        max_vision_feature_bytes: 7,
        commit_marker_tokens: 8,
        max_cfg_branches: 9,
        encoder_cache_entries: 10,
    };
    let message = wire::GenerationLimits::from(&limits);
    assert_eq!(GenerationLimits::from(message), limits);
}
