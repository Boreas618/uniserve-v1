//! MiniMax-H3 condition processing: media sources, probing, request planning
//! and the conditioner presentation.
//!
//! A MiniMax-H3 request must be sized before admission: the engine reserves
//! the conditioner's token count and the worker encodes each condition in
//! the units the plan states, so both are known exactly before any pixel is
//! decoded. One request flows through four stages, which
//! [`VideoProcessor::process`] runs in order:
//!
//! 1. [`plan::check_request`] applies every rule that needs no media
//!    (task, roles, counts, target), so a malformed request is rejected
//!    before anything is fetched.
//! 2. [`sources::MediaFetcher`] resolves each `conditions[i].uri` under the
//!    server's media policy (`data:`, `file://` under the media directory,
//!    `http(s)://` with size, time and address limits) into an owned buffer.
//! 3. [`probe::MediaProber`] reads the facts the plan needs: an image's
//!    displayed size from its header and EXIF orientation; a video's display
//!    aspect, frame rate, decoded frame count and soundtrack, and an audio
//!    file's sample rate and decoded sample count, through `ffprobe`.
//! 4. [`plan::plan_request`] resolves the canvas, the frame counts, how each
//!    condition is prepared, the Qwen3-VL vision grids and the encoder units;
//!    [`presentation::present`] tokenizes the conditioner's presentation with
//!    the checkpoint's tokenizer and tags every token for the denoiser's AdaLN
//!    modulation.
//!
//! The request rules are `uniserve-core`'s ([`plan`], [`presentation`]); the
//! model package reads the canvas rule and the frame counts from them through
//! the rank runtime library, and plans no request.
//!
//! Every rejection is a [`VideoInputError::Invalid`] naming the request field
//! at fault, `conditions[i]` for a condition; it converts into an OpenAI
//! `invalid_request_error`. Failures of the server's own resources (a missing
//! `ffprobe`, scratch storage) are [`VideoInputError::Internal`].

pub mod artifact;
pub mod probe;
pub mod service;
pub mod sources;

pub use service::{ProcessedVideo, VideoService};

use uniserve_core::VideoTask;
use uniserve_core::video::plan::{
    self, ConditionSpec, PlanLimits, ProcessorConfig, RequestPlan, Target, VideoInputError,
};
use uniserve_core::video::presentation::{self, Presentation};

use crate::openai::ApiError;
use crate::profile::tokenizer::DynTokenizer;

use probe::MediaProber;
use sources::{FetchedMedia, MediaFetcher};

impl From<VideoInputError> for ApiError {
    /// A rejection becomes a 400 `invalid_request_error` whose message starts
    /// with the field path; a server failure becomes a 500 `server_error`.
    fn from(error: VideoInputError) -> Self {
        match &error {
            VideoInputError::Invalid { field, .. } => {
                ApiError::invalid_request(error.to_string(), Some(field.param()))
            }
            VideoInputError::Internal(message) => {
                ApiError::server_error(format!("video condition processing failed: {message}"))
            }
        }
    }
}

/// One condition of a request, as the request states it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionInput<'a> {
    /// The condition's type, role, keyframe index and start offset.
    pub spec: ConditionSpec,
    /// Where the media comes from.
    pub uri: &'a str,
}

/// The fields of a MiniMax-H3 request that the processor reads.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoInputs<'a> {
    /// The request's task.
    pub task: VideoTask,
    /// The prompt, presented after the conditions.
    pub prompt: &'a str,
    /// The requested output.
    pub target: Target<'a>,
    /// The conditions, in request order.
    pub conditions: Vec<ConditionInput<'a>>,
}

/// A request ready for admission: its sizes, its presentation and the media
/// bytes of its conditions.
#[derive(Debug, Clone)]
pub struct ProcessedVideoRequest {
    /// Canvas, frame counts, condition preparation and rows.
    pub plan: RequestPlan,
    /// The conditioner's token ids and their AdaLN tags.
    pub presentation: Presentation,
    /// The fetched media of each condition, in request order.
    pub media: Vec<FetchedMedia>,
}

/// Processes MiniMax-H3 video requests for one served checkpoint.
///
/// One instance is shared by all requests; [`VideoProcessor::process`] takes
/// `&self` and runs concurrently.
pub struct VideoProcessor {
    fetcher: MediaFetcher,
    prober: MediaProber,
    vision: ProcessorConfig,
    limits: PlanLimits,
    tokenizer: DynTokenizer,
}

impl VideoProcessor {
    /// Combines the media policy, the prober, the checkpoint's vision
    /// processor geometry and serving limits, and its tokenizer.
    pub fn new(
        fetcher: MediaFetcher,
        prober: MediaProber,
        vision: ProcessorConfig,
        limits: PlanLimits,
        tokenizer: DynTokenizer,
    ) -> Self {
        Self {
            fetcher,
            prober,
            vision,
            limits,
            tokenizer,
        }
    }

    /// What the deployment serves, which every request is planned against.
    pub fn limits(&self) -> &PlanLimits {
        &self.limits
    }

    /// Validates, fetches, probes, plans and presents one request.
    ///
    /// The request-only rules run before any fetch. Fetches and probes of
    /// different conditions run concurrently; the first failure ends the
    /// request.
    ///
    /// # Errors
    ///
    /// Returns [`VideoInputError::Invalid`] naming the field at fault for a
    /// request the server does not serve, and [`VideoInputError::Internal`]
    /// when the server's own processing resources fail.
    pub async fn process(
        &self,
        request: &VideoInputs<'_>,
    ) -> Result<ProcessedVideoRequest, VideoInputError> {
        let specs: Vec<ConditionSpec> = request
            .conditions
            .iter()
            .map(|condition| condition.spec.clone())
            .collect();
        plan::check_request(request.task, &request.target, &specs, &self.limits)?;

        let uris: Vec<&str> = request
            .conditions
            .iter()
            .map(|condition| condition.uri)
            .collect();
        let media = self.fetcher.fetch_all(&uris).await?;
        let probes =
            futures::future::try_join_all(media.iter().enumerate().map(|(index, fetched)| {
                self.prober
                    .probe(index, specs[index].condition_type, fetched.bytes())
            }))
            .await?;

        let plan = plan::plan_request(
            request.task,
            &request.target,
            &specs,
            &probes,
            &self.vision,
            &self.limits,
        )?;
        let presentation = presentation::present(&*self.tokenizer, &plan, request.prompt)?;
        Ok(ProcessedVideoRequest {
            plan,
            presentation,
            media,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use axum::Router;
    use axum::routing::get;
    use base64::Engine as _;
    use image::ImageEncoder as _;
    use serde_json::Value;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
    use uniserve_core::video::plan::{
        ConditionSpec, ConditionType, PlanLimits, Prepared, ProcessorConfig, RequestField, Role,
        Target, VideoInputError,
    };
    use uniserve_core::video::presentation::{TEXT_TAG, VIDEO_TAG};
    use uniserve_core::{Canvas, FrameClips, VideoTask};

    use super::probe::{MediaProber, ProbeConfig};
    use super::sources::{MediaFetcher, MediaLimits, MediaPolicy, RemoteMediaPolicy};
    use super::{ConditionInput, VideoInputs, VideoProcessor};
    use crate::openai::ApiError;
    use crate::profile::assets::PipelineCheckpoint;
    use crate::profile::tokenizer::HuggingFaceTokenizer;
    use crate::serving::model::pipeline_tokenizer;

    /// The planning vectors the request rules are tested against.
    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../../tests/python/fixtures/minimax_h3_plan.json"
        ))
        .unwrap()
    }

    /// The MiniMax-H3 checkpoint's vision processor geometry.
    pub(in crate::serving::video) fn vision() -> ProcessorConfig {
        serde_json::from_value(fixture()["vision"].clone()).unwrap()
    }

    /// The canvases of the vectors' accepted requests: what a deployment that
    /// prepares them all serves.
    pub(in crate::serving::video) fn served_canvases() -> Vec<Canvas> {
        let mut canvases = Vec::new();
        for case in fixture()["requests"].as_array().unwrap() {
            if let Some(expected) = case.get("expected") {
                let pair = &expected["canvas"];
                let canvas = Canvas {
                    width: pair[0].as_u64().unwrap() as u32,
                    height: pair[1].as_u64().unwrap() as u32,
                };
                if !canvases.contains(&canvas) {
                    canvases.push(canvas);
                }
            }
        }
        canvases
    }

    /// Limits that serve every task and duration the model does, at the
    /// canvases of the vectors.
    fn limits() -> PlanLimits {
        PlanLimits {
            tasks: VideoTask::ALL.to_vec(),
            max_video_seconds: 15.0,
            frame_rate: 24,
            frames: FrameClips {
                min_frames: 22,
                clip_frames: 17,
                clip_latent_frames: 5,
                tail_latent_frames: 2,
            },
            canvases: served_canvases(),
            // The fixture's requests follow the released pipeline.
            references: uniserve_core::video::plan::ReferenceConfig::RELEASED,
        }
    }

    /// A deterministic tokenizer: one token per character, plus the vision
    /// markers and placeholders as special tokens.
    pub(in crate::serving::video) fn character_tokenizer() -> HuggingFaceTokenizer {
        let mut vocab: Vocab = (1_u32..=127)
            .map(|id| (char::from_u32(id).unwrap().to_string(), id))
            .collect();
        vocab.insert("<unk>".to_owned(), 0);
        let specials = [
            ("<|vision_start|>", 128),
            ("<|vision_end|>", 129),
            ("<|image_pad|>", 130),
            ("<|video_pad|>", 131),
        ];
        for (token, id) in specials {
            vocab.insert(token.to_owned(), id);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = TokenizerBuilder::new(model);
        tokenizer.add_special_tokens(
            &specials
                .iter()
                .map(|(token, _)| AddedToken::from(*token, true))
                .collect::<Vec<_>>(),
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        HuggingFaceTokenizer::new(&path).unwrap()
    }

    fn processor(scratch: &std::path::Path) -> VideoProcessor {
        let fetcher = MediaFetcher::new(MediaPolicy {
            media_directory: None,
            remote: RemoteMediaPolicy {
                allow_private_addresses: true,
                ..RemoteMediaPolicy::default()
            },
            limits: MediaLimits::new(64 << 20),
        })
        .unwrap();
        // Image conditions are probed in process; `ffprobe` is never run.
        let prober = MediaProber::new(ProbeConfig {
            ffprobe: scratch.join("ffprobe"),
            timeout: Duration::from_secs(5),
            scratch_directory: scratch.to_path_buf(),
            max_concurrent_probes: 2,
        });
        VideoProcessor::new(
            fetcher,
            prober,
            vision(),
            limits(),
            Arc::new(character_tokenizer()),
        )
    }

    fn png_data_uri(width: u32, height: u32) -> String {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &vec![90; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(encoded)
        )
    }

    const REFERENCE: ConditionSpec = ConditionSpec {
        condition_type: ConditionType::Image,
        role: Role::Reference,
        frame_index: None,
        start_seconds: None,
    };

    /// A request is fetched, probed, planned and presented, and keeps its
    /// media bytes for the worker.
    #[tokio::test]
    async fn a_request_is_prepared_end_to_end() {
        let scratch = tempfile::tempdir().unwrap();
        let processor = processor(scratch.path());
        let uri = png_data_uri(160, 90);
        let request = VideoInputs {
            task: VideoTask::Ref2va,
            prompt: "a fox",
            target: Target {
                short_edge: 768,
                aspect_ratio: "auto",
                duration_seconds: Some(5.0),
            },
            conditions: vec![ConditionInput {
                spec: REFERENCE,
                uri: &uri,
            }],
        };
        let processed = processor.process(&request).await.unwrap();

        // A 16:9 image reference is encoded at 3648x2048: 7296 rows and as
        // many vision tokens.
        let plan = &processed.plan;
        assert_eq!((plan.canvas.width, plan.canvas.height), (1344, 768));
        assert_eq!(plan.num_frames, 124);
        let Prepared::Image(size) = plan.conditions[0].prepared else {
            panic!("not an image reference");
        };
        assert_eq!((size.width, size.height), (3648, 2048));
        assert_eq!(plan.conditions[0].latent_units(plan.canvas, 1), [7296]);

        let label = "<Picture 1>: ".chars().count();
        let tags = &processed.presentation.tags;
        assert_eq!(tags.len(), label + 7296 + 2 + "a fox".len());
        assert!(tags[..label].iter().all(|&tag| tag == TEXT_TAG));
        assert!(
            tags[label..label + 7298]
                .iter()
                .all(|&tag| tag == VIDEO_TAG)
        );
        assert!(tags[label + 7298..].iter().all(|&tag| tag == TEXT_TAG));
        assert_eq!(processed.media.len(), 1);
        assert!(processed.media[0].bytes.starts_with(b"\x89PNG"));
    }

    /// A request that breaks a request-only rule is rejected before any of
    /// its media is fetched.
    #[tokio::test]
    async fn request_rules_apply_before_fetching() {
        let fetches = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&fetches);
        let router = Router::new().route(
            "/image.png",
            get(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                async { Vec::<u8>::new() }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let scratch = tempfile::tempdir().unwrap();
        let processor = processor(scratch.path());
        let uri = format!("http://{address}/image.png");
        let request = VideoInputs {
            task: VideoTask::Ref2va,
            prompt: "a fox",
            target: Target {
                short_edge: 768,
                aspect_ratio: "auto",
                duration_seconds: Some(3.0),
            },
            conditions: vec![ConditionInput {
                spec: REFERENCE,
                uri: &uri,
            }],
        };
        let error = processor.process(&request).await.unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetDuration));
        assert_eq!(fetches.load(Ordering::Relaxed), 0);
    }

    /// A rejection is an invalid request that names the condition at fault
    /// in its message and the condition list as its parameter; a server
    /// failure is a server error.
    #[test]
    fn errors_map_to_openai_categories() {
        let rejected: ApiError = VideoInputError::condition(2, "is not an image").into();
        assert_eq!(
            rejected,
            ApiError::invalid_request("conditions[2]: is not an image", Some("conditions"))
        );

        let rejected: ApiError =
            VideoInputError::invalid(RequestField::TargetDuration, "is required").into();
        assert_eq!(
            rejected,
            ApiError::invalid_request(
                "target.duration_seconds: is required",
                Some("target.duration_seconds")
            )
        );

        let failed: ApiError = VideoInputError::internal("ffprobe did not start").into();
        assert_eq!(failed.code(), "server_error");
    }

    /// The checkpoint's own tokenizer, loaded as the server loads it
    /// (`UNISERVE_MINIMAX_H3_MODEL` names the checkpoint root), encodes every
    /// text the vectors present to the reference's token ids and resolves
    /// the vision markers and placeholders to the reference's ids.
    #[tokio::test]
    async fn checkpoint_tokenizer_matches_the_reference() {
        use uniserve_core::video::presentation::Tokenizer as _;

        let Ok(root) = std::env::var("UNISERVE_MINIMAX_H3_MODEL") else {
            eprintln!("UNISERVE_MINIMAX_H3_MODEL is not set; skipping the tokenizer parity test");
            return;
        };
        let pipeline = PipelineCheckpoint::resolve(&root).await.unwrap().unwrap();
        let tokenizer = pipeline_tokenizer(&pipeline).await.unwrap();
        let fixture = fixture();
        for (token, id) in fixture["tokens"].as_object().unwrap() {
            assert_eq!(
                tokenizer.token_id(token).map(u64::from),
                id.as_u64(),
                "{token}"
            );
        }
        for case in fixture["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            for segment in expected["segments"].as_array().unwrap() {
                let Some(text) = segment.get("text") else {
                    continue;
                };
                let ids: Vec<u64> = uniserve_core::video::presentation::Tokenizer::encode(
                    &tokenizer,
                    text.as_str().unwrap(),
                )
                .unwrap()
                .into_iter()
                .map(u64::from)
                .collect();
                assert_eq!(Value::from(ids), segment["token_ids"], "{}", case["name"]);
            }
        }
    }
}
