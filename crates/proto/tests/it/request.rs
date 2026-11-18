//! Every request and event value comes back from its message as it went in,
//! and a message naming a value this build does not know is refused.

use std::sync::Arc;

use proto::request::{
    ArtifactEvent, ArtifactMedia, CachePolicy, ConditionExtent, DiffusionRequest,
    DiffusionSamplingParams, EncoderInputs, Event, FinishReason, GenerationFeatures,
    GenerationLimits, GenerationRequest, ImageEncoder, ImageEncoderInput, ImageInput, MediaKind,
    MigrationCause, MultimodalInputs, PositionLogprobs, RejectionKind, Request, RequestId,
    SamplingParams, StopReason, TokenLogprob, Trajectory, wire,
};

/// A generation request with every field away from its default.
fn generation() -> GenerationRequest {
    GenerationRequest {
        // The trace id the daemon gave it, which travels beside the request's identity.
        trace: 0x1234_5678_9abc,
        request_id: RequestId(41),
        prompt_token_ids: vec![1, 2, 3, 4],
        multimodal_inputs: MultimodalInputs {
            images: vec![ImageInput {
                digest: [0xfe; 32],
                encoded: b"image".to_vec(),
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
        max_und_tokens: 128,
        stop_strings: vec!["end".to_owned()],
        stop_token_ids: vec![2],
        priority: -3,
        cache: CachePolicy {
            read: false,
            write: true,
            isolation_key: Some(99),
            state_seeds: vec![[7; 32]],
            resume: Some(proto::request::CacheResume {
                epoch: 3,
                request_id: 5,
            }),
        },
        include_stop_token: true,
        generated_token_ids: vec![40, 41, 42],
        adapter: 0,
        hand_off: true,
        lend: Some(proto::request::Lend {
            endpoint: None,
            session: 1,
            owner: 5,
            tokens: 64,
            stored: 0,
        }),
    }
}

#[test]
fn requests_come_back_from_their_messages_unchanged() {
    let requests = [
        Request::from(generation()),
        Request::Diffusion(Box::new(DiffusionRequest {
            trace: 7,
            request_id: RequestId(42),
            prompt_token_ids: vec![5, 6],
            conditioning: proto::video::Conditions {
                prompt_tokens: 2,
                ..proto::video::Conditions::default()
            }
            .encode(),
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
                condition_rows: 7,
            },
            // A resumed request names its trajectory, the steps an engine
            // stored, the latent seed it continues with and where the pages
            // lie.
            trajectory: Trajectory {
                id: std::num::NonZeroU64::new(11),
                denoised_steps: 4,
                state_seed: Some(13),
                stored_layout: b"{\"version\":1}".to_vec(),
            },
            // The digests of what its encoders read, one per condition
            // unit, a vision and an audio digest.
            encoder_inputs: Some(Box::new(EncoderInputs {
                text: [1; 32],
                vision: Some([2; 32]),
                video_units: vec![[3; 32], [4; 32]],
                audio: Some([5; 32]),
            })),
            // The media travel beside the request's frame, not in it.
            media: Vec::new(),
        })),
    ];
    for request in requests {
        let message = request.to_wire();
        assert_eq!(Request::from_wire(message).unwrap(), request);
    }
}

#[test]
fn a_value_this_build_does_not_know_is_refused() {
    let mut message = Request::from(generation()).to_wire();
    let Some(wire::request::Kind::Generation(generation)) = message.kind.as_mut() else {
        unreachable!("a generation request")
    };
    generation.images[0].encoders[0].encoder = 77;
    let error = Request::from_wire(message).unwrap_err();
    assert!(error.to_string().contains("image encoder 77"), "{error}");
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
            cached_tokens: 496,
        },
        Event::TextToken {
            id: 3,
            logprob: Some(-0.25),
        },
        Event::TokenLogprobs {
            id: 3,
            candidates: vec![candidate.clone()],
        },
        Event::PromptLogprobs {
            positions: vec![PositionLogprobs {
                entries: vec![candidate],
            }],
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
        },
        Event::Finished {
            reason: FinishReason::Eos,
            stop_reason: Some(StopReason::Token(2)),
            prompt_tokens: 4,
            completion_tokens: 9,
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
            stored_layout: vec![1, 2, 3],
            lend: None,
        },
        Event::Migratable {
            cause: MigrationCause::Transfer,
            denoised_steps: 0,
            stored_layout: Vec::new(),
            lend: None,
        },
        Event::Migratable {
            cause: MigrationCause::Handoff,
            denoised_steps: 0,
            stored_layout: Vec::new(),
            lend: Some(proto::request::Lend {
                endpoint: Some("gb200-a:29800".to_owned()),
                session: 3,
                owner: 17,
                tokens: 129,
                stored: 2,
            }),
        },
        Event::Rejected {
            kind: RejectionKind::Role,
            message: "a decode replica computes no prompt".to_owned(),
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
        max_vae_grid_tokens: 3,
        max_vit_grid_tokens: 4,
        max_latent_feature_bytes: 6,
        max_vision_feature_bytes: 7,
        encoder_cache_entries: 10,
        max_logprobs: 20,
    };
    let message = wire::GenerationLimits::from(&limits);
    assert_eq!(GenerationLimits::from(message), limits);
}

/// The step values a `Tune` command carries come back as they went in, stated and unstated
/// values alike, and a command of a step of no call, or of a group of no batch in flight, is
/// refused.
#[test]
fn step_values_come_back_from_their_command() {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use proto::config::Tuning;

    let stated = Tuning {
        revision: 42,
        chunked_prefill_size: NonZeroUsize::new(4096),
        max_batch: NonZeroUsize::new(64).unwrap(),
        max_num_batched_tokens: NonZeroUsize::new(8192),
        max_queued_requests: NonZeroUsize::new(32),
        queue_depths: [("model".to_owned(), NonZeroUsize::new(3).unwrap())].into(),
        prefix_caching: true,
        host_tier: false,
        copy_budget: 64 << 20,
        report_period: Duration::from_millis(250),
        watchdog_timeout: Duration::from_secs(90),
        compare_timeout: Duration::from_secs(120),
        nvtx_ranges: true,
        kv_events: proto::config::KvEventsConfig {
            endpoint: Some("tcp://*:5557".to_owned()),
            replay_endpoint: Some("tcp://*:5600".to_owned()),
            topic: "kv".to_owned(),
            buffer_steps: NonZeroUsize::new(64).unwrap(),
        },
        roles: vec![
            proto::config::Role::Prefill,
            proto::config::Role::Mixed,
            proto::config::Role::Decode,
        ],
    };
    let unstated = Tuning {
        revision: 1,
        chunked_prefill_size: None,
        max_batch: NonZeroUsize::MIN,
        max_num_batched_tokens: None,
        max_queued_requests: None,
        queue_depths: Default::default(),
        prefix_caching: false,
        host_tier: true,
        copy_budget: 0,
        report_period: Duration::from_secs(1),
        watchdog_timeout: Duration::from_secs(150),
        compare_timeout: Duration::from_secs(300),
        nvtx_ranges: false,
        kv_events: Default::default(),
        roles: Vec::new(),
    };
    let mut no_depth = stated.command(3);
    no_depth.queue_depths.insert("model".to_owned(), 0);
    assert_eq!(Tuning::of_command(&no_depth), None);
    for tuning in [stated, unstated] {
        let command = tuning.command(7);
        assert_eq!(command.id, 7);
        assert_eq!(Tuning::of_command(&command), Some(tuning));
    }
    let empty = wire::Tune {
        max_batch: 0,
        ..wire::Tune::default()
    };
    assert_eq!(Tuning::of_command(&empty), None);
}
