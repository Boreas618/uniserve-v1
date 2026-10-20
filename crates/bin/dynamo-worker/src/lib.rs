// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental in-process Dynamo worker for UniServe FastH3.
//!
//! This reference integration embeds UniServe's Rust scheduler and engine in
//! the Dynamo worker process. The preferred production boundary remains the
//! native HTTP sidecar, which does not couple either project's dependencies.
//!
//! The Dynamo frontend forwards each `/v1/videos` request to
//! `RawEngine::generate`, nesting unknown client fields under `extra_args`.
//! The worker takes `uniserve serve`'s options, `--config` file and `--set`
//! included, and refuses before startup the ones it cannot honor: the HTTP
//! server's, data and expert parallelism, condition media and asynchronous
//! job storage, which a Dynamo `t2va` request never uses. It serves FastH3
//! text-to-video-and-audio at the canvases its
//! `video_config.resolutions` and `video_config.aspect_ratios` prepare, as the started
//! deployment declares them, and refuses at startup a deployment that serves
//! no `t2va`, such as a reference denoiser, or a canvas name its model does
//! not generate: `prepare_request` maps the Dynamo request onto
//! a `t2va` request of UniServe's video API and refuses any control FastH3
//! does not implement rather than ignoring it. A refused request's stream
//! holds only its invalid-argument error, so the worker stays routable; a
//! frontend at the pinned Dynamo revision answers it with a 400, while the
//! 1.5.0 release answers every error of a video stream with a 500. The
//! request then enters
//! `Engine::generate_video`, the path the HTTP `/v1/videos` route uses, and a
//! successful request yields one terminal response object
//! carrying the single MP4 artifact.
//! UniServe's own HTTP listener is never started.
//!
//! # Running it end to end
//!
//! No test of this repository launches the worker: it needs Dynamo's
//! frontend (the `ai-dynamo` 1.5.0 Python package, the release of the pinned
//! `dynamo-backend-common`), a complete FastH3 checkpoint and four GPUs.
//! `docs/minimax_h3/dynamo.md` installs the frontend environment and builds
//! the worker. On one host, file discovery and Dynamo's default TCP request
//! plane and ZMQ event plane need neither etcd nor NATS. From the repository
//! root:
//!
//! ```bash
//! export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=$PWD/artifacts/dynamo/discovery
//! artifacts/dynamo/frontend-env/bin/python -m dynamo.frontend \
//!   --http-host 127.0.0.1 --http-port 18090 \
//!   --namespace uniserve-fasth3 --discovery-backend file &
//! target/release/uniserve-dynamo-worker "$UNISERVE_FAST_H3_MODEL" \
//!   --namespace uniserve-fasth3 --served-model-name FastH3 \
//!   --worker-python "$PWD/.venv/bin/python" \
//!   --workers configs/minimax_h3/ulysses4.toml \
//!   --max-model-len 16384 --max-running-requests 2 \
//!   --set video_config.max_seconds=15 --set quantization_config.mode=quality &
//! # Once `curl -fsS http://127.0.0.1:18090/v1/models` lists FastH3:
//! curl --fail-with-body --max-time 600 http://127.0.0.1:18090/v1/videos \
//!   -H 'Content-Type: application/json' -d '{"model": "FastH3",
//!   "prompt": "A red fox trots across a snowy meadow at sunrise, its breath steaming, while birds call from the pines.",
//!   "seconds": 5, "size": "1344x768", "response_format": "b64_json",
//!   "output_format": "mp4", "nvext": {"seed": 42}}'
//! ```
//!
//! The response's `data[0].b64_json` decodes to a 1344x768 24 fps H.264
//! video of 124 frames with 32 kHz AAC audio. This request maps onto the
//! same `t2va` request as a `/v1/videos` request to `uniserve serve` with the
//! same deployment, prompt, seed, canvas and duration, and returns the same
//! MP4. SIGINT to the worker unregisters it from discovery, shuts the engine
//! and its ranks down and exits with status 0.

use std::collections::{BTreeMap, HashMap};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use clap::Parser;
use dynamo_backend_common::{
    BackendError, CommonArgs, DisaggregationMode, DynamoError, EngineConfig, ErrorType,
    GenerateContext, ModelInput, RawEngine, WorkerConfig as DynamoWorkerConfig,
};
use futures::StreamExt as _;
use futures::stream::BoxStream;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;
use uniserve_core::{Canvas, MediaKind, ServedCanvas, VideoTask};
use uniserve_engine::VideoDenoiserInfo;
use uniserve_server::options::{ServeArgs, ServeConfig, Stated, printed};
use uniserve_server::profile::ModelDescription;
use uniserve_server::profile::assets::PipelineCheckpoint;
use uniserve_server::{
    Engine,
    openai::{ApiError, VideoGenerationRequest, VideoTarget},
    serving::{ServeRequestId, video::artifact::ArtifactCollector},
};

/// The duration of a Dynamo request that omits `seconds`, at most the
/// deployment's capacity.
const DEFAULT_SECONDS: f64 = 5.0;

/// Configuration paths of `uniserve serve` this worker cannot honor: it
/// serves no HTTP API (so it neither listens nor logs HTTP requests), runs
/// one replica without expert parallelism, receives requests without
/// condition media and keeps no asynchronous jobs.
const UNSUPPORTED_PATHS: [&str; 9] = [
    "server_config",
    "logging_config.log_requests",
    "data_parallel_size",
    "moe_config",
    "placement_config.transfer",
    "mm_processor_kwargs",
    "video_config.max_condition_rows",
    "media_config",
    "video_job_config",
];

#[derive(Clone, Parser)]
#[command(
    name = "uniserve-dynamo-worker",
    about = "Experimental in-process Dynamo worker for UniServe FastH3."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    // The serving command line of `uniserve serve`; `UNSUPPORTED_PATHS`
    // lists the options this worker refuses.
    #[command(flatten)]
    serve: ServeArgs,
}

/// Dynamo `RawEngine` serving FastH3 video generation from an in-process
/// UniServe engine.
pub struct DynamoFastH3Engine {
    /// The UniServe configuration the engine starts with.
    config: ServeConfig,
    /// The option paths the operator stated, against which startup refuses
    /// options of another model family.
    stated: Stated,
    /// The single model name requests name and responses carry.
    served_model_name: String,
    /// The UniServe engine, set once by `start` after it verified that the
    /// engine serves a `t2va` video denoiser. Every other trait method reads
    /// it and treats an unset engine as not started.
    state: OnceCell<Arc<Engine>>,
    /// Cancelled by `cleanup`; every in-flight `generate` stream then drops
    /// its UniServe request and ends without a response.
    cancel: CancellationToken,
}

impl DynamoFastH3Engine {
    /// Parses the process command line into an unstarted engine and the
    /// Dynamo worker registration that advertises it as a `videos` endpoint.
    ///
    /// The serving options resolve as for `uniserve serve`: a `--config`
    /// file, its flags and `--set` assignments. With `--print-config` the
    /// resolved configuration is printed and the process exits. Clap exits
    /// the process on a malformed command line. Settings this worker cannot
    /// honor (disaggregation, encoder routing, RL routes, a path in
    /// `UNSUPPORTED_PATHS`, a missing deployment file) and a configuration
    /// that does not resolve or that `ServeConfig::validate` refuses
    /// return an invalid-argument error. No rank starts until
    /// `RawEngine::start`, whose workers refuse a video canvas selection the
    /// model does not generate.
    pub fn from_args() -> Result<(Self, DynamoWorkerConfig), DynamoError> {
        let args = <Args as Parser>::parse();
        if args.serve.print_config {
            let (config, _) = args
                .serve
                .resolve()
                .map_err(|error| invalid_argument(format!("{error:#}")))?;
            let printed = printed(&config).map_err(|error| invalid_argument(error.to_string()))?;
            print!("{printed}");
            std::process::exit(0);
        }
        Self::try_from_args(args)
    }

    fn try_from_args(args: Args) -> Result<(Self, DynamoWorkerConfig), DynamoError> {
        if args.common.disaggregation_mode != DisaggregationMode::Aggregated {
            return Err(invalid_argument(
                "UniServe FastH3 supports only aggregated Dynamo workers",
            ));
        }
        if args.common.route_to_encoder {
            return Err(invalid_argument("route-to-encoder is not supported"));
        }
        if args.common.enable_rl {
            return Err(invalid_argument("RL engine routes are not supported"));
        }

        let (serve, stated) = args
            .serve
            .resolve()
            .map_err(|error| invalid_argument(format!("{error:#}")))?;
        if let Some(path) = UNSUPPORTED_PATHS
            .iter()
            .find(|path| stated.contains_within(path))
        {
            return Err(invalid_argument(format!(
                "`{path}` is not supported by the Dynamo worker"
            )));
        }
        if serve.workers.is_none() {
            return Err(invalid_argument(
                "`workers` is required: a deployment file places FastH3's components",
            ));
        }

        let mut config = serve;
        config
            .validate(&stated)
            .map_err(|error| invalid_argument(format!("{error:#}")))?;
        // The Dynamo request form carries no conditions, so the workers
        // provision no condition rows.
        config.video_config.max_condition_rows = Some(0);
        let served_model_name = config.served_model_name().to_owned();

        let registration = DynamoWorkerConfig {
            namespace: args.common.namespace.clone(),
            component: args.common.component.clone(),
            endpoint: args.common.endpoint.clone(),
            endpoint_types: "videos".to_string(),
            model_name: served_model_name.clone(),
            served_model_name: Some(served_model_name.clone()),
            model_input: ModelInput::Text,
            custom_jinja_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: true,
            enable_local_indexer: false,
            enable_kv_routing: false,
            disaggregation_mode: DisaggregationMode::Aggregated,
            route_to_encoder: false,
            enable_rl: false,
            ..Default::default()
        };
        Ok((
            Self {
                config,
                stated,
                served_model_name,
                state: OnceCell::new(),
                cancel: CancellationToken::new(),
            },
            registration,
        ))
    }

    /// Starts UniServe's engine and worker ranks for the configured FastH3
    /// checkpoint.
    ///
    /// Fails with an invalid argument when the checkpoint's pipeline index
    /// cannot be resolved or does not name MiniMax H3, or when the started
    /// deployment's video denoiser serves no `t2va`, such as a reference
    /// denoiser. It fails with an engine error when UniServe cannot start,
    /// including a canvas selection its workers refuse. A refusal after
    /// startup shuts the engine down before returning.
    async fn build_state(&self) -> Result<Arc<Engine>, DynamoError> {
        // A MiniMax H3 checkpoint names its pipeline class in a root index in
        // place of config.json; refuse anything else before a rank starts.
        let pipeline = PipelineCheckpoint::resolve(self.config.model())
            .await
            .map_err(|error| invalid_argument(error.to_string()))?;
        if pipeline
            .and_then(|pipeline| ModelDescription::from_pipeline_class(pipeline.class_name()))
            != Some(ModelDescription::MiniMaxH3)
        {
            return Err(invalid_argument(
                "MODEL is not a FastH3 video-generation checkpoint",
            ));
        }

        // Only `uniserve_server::http` binds the configuration's listener, and
        // this worker never serves UniServe's HTTP API.
        let state = Arc::new(
            Engine::start(&self.config, &self.stated)
                .await
                .map_err(|error| engine_error(format!("failed to start UniServe: {error:#}")))?,
        );

        // The worker sends only `t2va` requests. The engine validated the
        // denoiser's handshake, which declares positive frame and audio rates
        // and at least one served canvas.
        let tasks = state
            .video_denoiser()
            .map(|denoiser| denoiser.tasks.clone())
            .unwrap_or_default();
        if !tasks.iter().any(|task| task == VideoTask::T2va.as_str()) {
            let _ = state.shutdown().await;
            return Err(invalid_argument(format!(
                "the Dynamo worker sends text-to-video requests; this deployment serves tasks {tasks:?}"
            )));
        }
        Ok(state)
    }

    /// The started engine and the video denoiser it serves.
    fn started(&self, call: &str) -> Result<(&Arc<Engine>, &VideoDenoiserInfo), DynamoError> {
        let engine = self
            .state
            .get()
            .ok_or_else(|| engine_error(format!("{call} called before start")))?;
        // `start` sets only an engine that serves a video denoiser.
        let denoiser = engine
            .video_denoiser()
            .ok_or_else(|| engine_error("the started engine serves no video denoiser"))?;
        Ok((engine, denoiser))
    }
}

#[async_trait]
impl RawEngine for DynamoFastH3Engine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        // The check refuses a second start before it launches ranks; `set`
        // refuses the second of two starts that raced past it.
        if self.state.initialized() {
            return Err(engine_error("UniServe engine already started"));
        }
        let state = self.build_state().await?;
        self.state
            .set(state)
            .map_err(|_| engine_error("UniServe engine already started"))?;
        let (_, denoiser) = self.started("start")?;
        let runtime_data = runtime_data(self.config.video_config.max_seconds, denoiser);
        Ok(EngineConfig {
            model: self.served_model_name.clone(),
            served_model_name: Some(self.served_model_name.clone()),
            runtime_data,
            llm: None,
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        let (state, denoiser) = self.started("generate")?;
        // A refusal of this request is the stream's only item. Dynamo's
        // router reads an error returned before the stream as a worker that
        // cannot serve: it answers 500 and takes the instance out of routing
        // for its inhibited period, while an in-stream invalid argument
        // leaves it routable and is answered by the frontend (a 400 from the
        // pinned Dynamo revision on). Only failures of the engine itself end
        // `generate` before the stream.
        let prepared = match prepare_request(
            request,
            &self.served_model_name,
            self.config.video_config.max_seconds,
            &denoiser.canvases,
        ) {
            Ok(prepared) => prepared,
            Err(error) => return Ok(refusal(error)),
        };
        // The Dynamo context id names the request in UniServe's logs and the
        // response.
        let request_id = ServeRequestId::new(ctx.id().to_string());

        // The engine runs the path the HTTP video route uses: prompt
        // preprocessing, submission and request accounting.
        let (_, mut events) = match state
            .generate_video(request_id.clone(), prepared.request)
            .await
        {
            Ok(started) => started,
            // The video service refused the request, for example a duration
            // outside the deployment's range.
            Err(error) if error.status_code().is_client_error() => {
                return Ok(refusal(api_error(error)));
            }
            Err(error) => return Err(api_error(error)),
        };
        let cancel = self.cancel.clone();
        let served_model_name = self.served_model_name.clone();
        let response_format = prepared.response_format;
        let denoiser = denoiser.clone();

        // Unless stopped, the stream yields exactly one item: the response
        // object, or the error that ended the request. A Dynamo stop or
        // `cleanup` instead ends the stream without an item, and dropping the
        // UniServe stream with it cancels the request in its engine core.
        Ok(Box::pin(async_stream::stream! {
            let started_at = Instant::now();
            let mut collector = ArtifactCollector::default();
            let artifact = loop {
                // `biased` polls the stop and shutdown signals first, so they
                // take precedence over an event that is ready at the same time.
                let event = tokio::select! {
                    biased;
                    _ = ctx.stopped() => return,
                    _ = cancel.cancelled() => return,
                    event = events.next() => event,
                };
                let event = match event {
                    Some(Err(error)) => {
                        yield Err(api_error(ApiError::from(error)));
                        return;
                    }
                    Some(Ok(event)) => Some(event),
                    None => None,
                };
                if let ControlFlow::Break(outcome) = collector.push(event) {
                    match outcome {
                        Ok(artifact) => break artifact,
                        Err(error) => {
                            yield Err(api_error(ApiError::from(error)));
                            return;
                        }
                    }
                }
            };
            if artifact.media_kind != MediaKind::Video || artifact.content_type != "video/mp4" {
                yield Err(engine_error(format!(
                    "UniServe produced {:?} with content type {:?}, expected video/mp4",
                    artifact.media_kind, artifact.content_type
                )));
                return;
            }
            yield Ok(video_response(
                &request_id,
                &served_model_name,
                response_format,
                artifact.media.as_bytes(),
                &denoiser,
                started_at.elapsed(),
            ));
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        if let Some(state) = self.state.get() {
            state
                .shutdown()
                .await
                .map_err(|error| engine_error(format!("UniServe shutdown failed: {error}")))?;
        }
        Ok(())
    }
}

/// The request body Dynamo's frontend forwards: the fields of its
/// `NvCreateVideoRequest` as dispatched to a worker.
///
/// Unknown fields are refused. The frontend moves unknown top-level client
/// fields under `extra_args["media_passthrough"]`; `prepare_request` reads
/// `resolution` and `aspect_ratio` from there and refuses any other extra
/// argument.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamoVideoRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    input_reference: Option<String>,
    #[serde(default)]
    seconds: Option<i32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    nvext: Option<VideoNvExt>,
    #[serde(default)]
    extra_args: Option<Map<String, Value>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseFormat {
    #[default]
    Url,
    B64Json,
}

/// Dynamo's `nvext` video extensions. `prepare_request` accepts only the
/// fields FastH3 can honor, `seed` at any non-negative value. The checkpoint
/// fixes the denoising schedule and the frame rate, and the duration states
/// the frame count, so `num_inference_steps`, `fps` and `num_frames` are
/// refused like the other controls FastH3 does not take.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoNvExt {
    #[serde(default)]
    annotations: Option<Vec<String>>,
    #[serde(default)]
    fps: Option<i32>,
    #[serde(default)]
    num_frames: Option<i32>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    num_inference_steps: Option<i32>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    boundary_ratio: Option<f32>,
    #[serde(default)]
    guidance_scale_2: Option<f32>,
}

#[derive(Debug)]
struct PreparedRequest {
    request: VideoGenerationRequest,
    response_format: ResponseFormat,
}

/// Maps a Dynamo video request onto a `t2va` `VideoGenerationRequest` whose
/// target is the selected raster and the request's duration.
///
/// The canvas is selected by `size` (`WIDTHxHEIGHT`, one of the served
/// canvases), by the passthrough `resolution` (`768p`) and `aspect_ratio`
/// fields, or by both when they name the same canvas; omitted, it is the
/// deployment's default, the first it declares. Its resolution class names
/// the target's short edge.
/// Omitted fields default to seed 0, a `url` response, and 5 seconds, capped
/// at `max_video_seconds`; the video service bounds the duration and derives
/// the frame count when it plans the target.
/// Every refusal is an invalid-argument error.
fn prepare_request(
    value: Value,
    served_model_name: &str,
    max_video_seconds: f64,
    canvases: &[ServedCanvas],
) -> Result<PreparedRequest, DynamoError> {
    let request: DynamoVideoRequest = serde_json::from_value(value)
        .map_err(|error| invalid_argument(format!("invalid video request: {error}")))?;
    if request.model != served_model_name {
        return Err(invalid_argument(format!(
            "model {:?} is not served; expected {:?}",
            request.model, served_model_name
        )));
    }
    if request.prompt.trim().is_empty() {
        return Err(invalid_argument("prompt must not be empty"));
    }
    reject_present("input_reference", request.input_reference.as_ref())?;
    reject_present("user", request.user.as_ref())?;
    if request.stream == Some(true) {
        return Err(invalid_argument("stream=true is not supported"));
    }
    if request
        .output_format
        .as_deref()
        .is_some_and(|format| !format.eq_ignore_ascii_case("mp4"))
    {
        return Err(invalid_argument("output_format must be mp4"));
    }
    let (mut short_edge, mut aspect_ratio) = passthrough_canvas(request.extra_args)?;
    if let Some(size) = request.size.as_deref() {
        let served = size
            .split_once('x')
            .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
            .and_then(|(width, height)| {
                canvases
                    .iter()
                    .find(|served| served.canvas == Canvas { width, height })
            })
            .ok_or_else(|| {
                invalid_argument(format!(
                    "size must be one of {}",
                    sizes(canvases).join(", ")
                ))
            })?;
        if short_edge.is_some_and(|value| value != served.short_edge)
            || aspect_ratio
                .as_deref()
                .is_some_and(|value| value != served.aspect_ratio)
        {
            return Err(invalid_argument(format!(
                "size {size} conflicts with resolution or aspect_ratio"
            )));
        }
        short_edge = Some(served.short_edge);
        aspect_ratio = Some(served.aspect_ratio.clone());
    }
    // Refuse an unprovisioned canvas here; the video service refuses it
    // again when it plans the target.
    let served =
        select_canvas(canvases, short_edge, aspect_ratio.as_deref()).map_err(invalid_argument)?;

    // `seconds` is an integer in Dynamo's API; the video service refuses one
    // outside the deployment's duration range when it plans the target.
    let seconds = request
        .seconds
        .map_or_else(|| DEFAULT_SECONDS.min(max_video_seconds), f64::from);
    let nvext = request.nvext.unwrap_or_default();
    reject_present("nvext.annotations", nvext.annotations.as_ref())?;
    reject_present("nvext.negative_prompt", nvext.negative_prompt.as_ref())?;
    reject_present(
        "nvext.num_inference_steps",
        nvext.num_inference_steps.as_ref(),
    )?;
    reject_present("nvext.guidance_scale", nvext.guidance_scale.as_ref())?;
    reject_present("nvext.boundary_ratio", nvext.boundary_ratio.as_ref())?;
    reject_present("nvext.guidance_scale_2", nvext.guidance_scale_2.as_ref())?;
    reject_present("nvext.fps", nvext.fps.as_ref())?;
    reject_present("nvext.num_frames", nvext.num_frames.as_ref())?;
    // Dynamo's seed is signed and UniServe's unsigned.
    let seed = match nvext.seed {
        Some(seed) if seed < 0 => return Err(invalid_argument("nvext.seed must not be negative")),
        Some(seed) => seed as u64,
        None => 0,
    };
    Ok(PreparedRequest {
        request: VideoGenerationRequest {
            model: request.model,
            prompt: request.prompt,
            task: VideoTask::T2va,
            conditions: Vec::new(),
            target: VideoTarget {
                short_edge: served.short_edge,
                aspect_ratio: served.aspect_ratio.clone(),
                duration_seconds: Some(seconds),
            },
            seed,
        },
        response_format: request.response_format.unwrap_or_default(),
    })
}

/// Selects a served canvas; an omitted field takes the default's, the first
/// canvas the deployment names.
///
/// # Errors
///
/// Returns a message naming the served values when the deployment does not
/// serve the requested resolution, aspect ratio or their pair.
fn select_canvas<'a>(
    canvases: &'a [ServedCanvas],
    short_edge: Option<u32>,
    aspect_ratio: Option<&str>,
) -> Result<&'a ServedCanvas, String> {
    // The validated handshake serves at least one canvas.
    let default = &canvases[0];
    let short_edge = short_edge.unwrap_or(default.short_edge);
    let aspect_ratio = aspect_ratio.unwrap_or(&default.aspect_ratio);
    if !canvases
        .iter()
        .any(|served| served.short_edge == short_edge)
    {
        return Err(format!(
            "resolution must be one of {}",
            resolutions(canvases).join(", ")
        ));
    }
    if !canvases
        .iter()
        .any(|served| served.aspect_ratio == aspect_ratio)
    {
        return Err(format!(
            "aspect_ratio must be one of {}",
            aspect_ratios(canvases).join(", ")
        ));
    }
    canvases
        .iter()
        .find(|served| served.short_edge == short_edge && served.aspect_ratio == aspect_ratio)
        .ok_or_else(|| format!("no canvas is served for {short_edge}p {aspect_ratio}"))
}

/// Served canvases as `WIDTHxHEIGHT`, the default first.
fn sizes(canvases: &[ServedCanvas]) -> Vec<String> {
    canvases
        .iter()
        .map(|served| format!("{}x{}", served.canvas.width, served.canvas.height))
        .collect()
}

/// Served resolution classes, as `768p`, the default first.
fn resolutions(canvases: &[ServedCanvas]) -> Vec<String> {
    let mut classes: Vec<String> = Vec::new();
    for served in canvases {
        let class = format!("{}p", served.short_edge);
        if !classes.contains(&class) {
            classes.push(class);
        }
    }
    classes
}

/// Served aspect ratios, the default first.
fn aspect_ratios(canvases: &[ServedCanvas]) -> Vec<String> {
    let mut ratios: Vec<String> = Vec::new();
    for served in canvases {
        if !ratios.contains(&served.aspect_ratio) {
            ratios.push(served.aspect_ratio.clone());
        }
    }
    ratios
}

fn reject_present<T>(field: &str, value: Option<&T>) -> Result<(), DynamoError> {
    match value {
        Some(_) => Err(invalid_argument(format!(
            "{field} is not supported by UniServe FastH3"
        ))),
        None => Ok(()),
    }
}

/// Builds the terminal response body in the shape of Dynamo's
/// `NvVideosResponse`.
///
/// The MP4 is embedded in the response: `Url` returns it as a `data:` URL and
/// `B64Json` as bare base64. `fps` and `audio_sample_rate` report the rates
/// the deployment's denoiser declares for its video and audio, not values
/// read from the artifact.
fn video_response(
    id: &str,
    model: &str,
    response_format: ResponseFormat,
    media: &[u8],
    denoiser: &VideoDenoiserInfo,
    elapsed: Duration,
) -> Value {
    let encoded = base64::engine::general_purpose::STANDARD.encode(media);
    let (url, b64_json) = match response_format {
        ResponseFormat::Url => (Some(format!("data:video/mp4;base64,{encoded}")), None),
        ResponseFormat::B64Json => (None, Some(encoded)),
    };
    json!({
        "id": id,
        "object": "video",
        "model": model,
        "status": "completed",
        "progress": 100,
        "created": uniserve_core::now_unix_secs_u64() as i64,
        "data": [{
            "output_format": "mp4",
            "url": url,
            "b64_json": b64_json,
            "fps": denoiser.frame_rate,
            "audio_sample_rate": denoiser.audio_sample_rate,
        }],
        "inference_time_s": elapsed.as_secs_f64(),
    })
}

/// Reads `resolution` and `aspect_ratio` from the frontend's
/// `extra_args["media_passthrough"]`, refusing every other extra argument.
fn passthrough_canvas(
    extra_args: Option<Map<String, Value>>,
) -> Result<(Option<u32>, Option<String>), DynamoError> {
    let mut extra_args = extra_args.unwrap_or_default();
    let passthrough = match extra_args.remove("media_passthrough") {
        None => Map::new(),
        Some(Value::Object(passthrough)) => passthrough,
        Some(_) => return Err(invalid_argument("extra video fields are not supported")),
    };
    if !extra_args.is_empty()
        || passthrough
            .keys()
            .any(|key| key != "resolution" && key != "aspect_ratio")
    {
        return Err(invalid_argument("extra video fields are not supported"));
    }
    let field = |name: &str| -> Result<Option<String>, DynamoError> {
        match passthrough.get(name) {
            None => Ok(None),
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(_) => Err(invalid_argument(format!("{name} must be a string"))),
        }
    };
    // A resolution class `768p` names the short edge 768.
    let short_edge = field("resolution")?
        .map(|value| {
            value
                .strip_suffix('p')
                .filter(|digits| !digits.starts_with('+'))
                .and_then(|digits| digits.parse::<u32>().ok())
                .ok_or_else(|| {
                    invalid_argument(format!(
                        "resolution must be a class such as 768p, got {value:?}"
                    ))
                })
        })
        .transpose()?;
    Ok((short_edge, field("aspect_ratio")?))
}

/// Registration metadata Dynamo copies into the model's runtime config.
fn runtime_data(max_video_seconds: f64, denoiser: &VideoDenoiserInfo) -> HashMap<String, Value> {
    let canvases = &denoiser.canvases;
    let default = select_canvas(canvases, None, None).ok();
    BTreeMap::from([
        ("backend".to_string(), json!("uniserve-inprocess")),
        ("task".to_string(), json!("t2va")),
        ("fps".to_string(), json!(denoiser.frame_rate)),
        (
            "width".to_string(),
            json!(default.map(|served| served.canvas.width)),
        ),
        (
            "height".to_string(),
            json!(default.map(|served| served.canvas.height)),
        ),
        ("sizes".to_string(), json!(sizes(canvases))),
        ("resolutions".to_string(), json!(resolutions(canvases))),
        ("aspect_ratios".to_string(), json!(aspect_ratios(canvases))),
        ("max_video_seconds".to_string(), json!(max_video_seconds)),
    ])
    .into_iter()
    .collect()
}

/// Maps a UniServe API error to Dynamo's error type: a 4xx status becomes an
/// invalid argument and any other status an unknown backend error.
fn api_error(error: ApiError) -> DynamoError {
    let kind = if error.status_code().is_client_error() {
        BackendError::InvalidArgument
    } else {
        BackendError::Unknown
    };
    dynamo_error(kind, error.to_error_response().error.message)
}

/// A response stream whose only item is `error`, the refusal of its request.
fn refusal(error: DynamoError) -> BoxStream<'static, Result<Value, DynamoError>> {
    futures::stream::once(std::future::ready(Err(error))).boxed()
}

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    dynamo_error(BackendError::InvalidArgument, message)
}

fn engine_error(message: impl Into<String>) -> DynamoError {
    dynamo_error(BackendError::Unknown, message)
}

fn dynamo_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses a worker command line as `from_args` does.
    fn parsed(flags: &[&str]) -> Result<(DynamoFastH3Engine, DynamoWorkerConfig), DynamoError> {
        let args = Args::try_parse_from(["uniserve-dynamo-worker"].iter().chain(flags))
            .expect("the command line parses");
        DynamoFastH3Engine::try_from_args(args)
    }

    /// The worker takes `uniserve serve`'s options and refuses, by path,
    /// those a Dynamo `t2va` worker cannot honor, and a missing deployment.
    #[test]
    fn serve_options_configure_the_worker_and_unsupported_ones_are_refused() {
        let deployment = std::env::temp_dir().join(format!(
            "uniserve-dynamo-deployment-{}.toml",
            std::process::id()
        ));
        std::fs::write(
            &deployment,
            "[[workers]]\nid = \"model\"\nranks = [\"cuda:0\"]\nqueue_depth = 6\n\
             [workers.components.transformer]\n",
        )
        .unwrap();
        let workers = deployment.to_str().unwrap();

        let (engine, registration) = parsed(&[
            "/models/FastH3",
            "--workers",
            workers,
            "--served-model-name",
            "FastH3",
            "--set",
            "video_config.max_seconds=5",
        ])
        .unwrap();
        assert_eq!(registration.model_name, "FastH3");
        assert_eq!(engine.config.video_config.max_seconds, 5.0);
        assert_eq!(engine.config.video_config.max_condition_rows, Some(0));

        for flags in [
            vec!["/models/FastH3", "--workers", workers, "--port", "9000"],
            vec![
                "/models/FastH3",
                "--workers",
                workers,
                "--data-parallel-size",
                "2",
            ],
            vec![
                "/models/FastH3",
                "--workers",
                workers,
                "--set",
                "video_config.max_condition_rows=4",
            ],
            vec![
                "/models/FastH3",
                "--workers",
                workers,
                "--set",
                "media_config.allow_remote=false",
            ],
            vec!["/models/FastH3"],
        ] {
            let error = parsed(&flags).err().expect("refused");
            assert!(error.to_string().contains('`'), "{flags:?}: {error}");
        }
    }

    fn request() -> Value {
        json!({
            "model": "FastH3",
            "prompt": "A stream in a forest",
            "seconds": 5,
            "response_format": "b64_json",
            "nvext": {"seed": 1000}
        })
    }

    /// A served canvas, named by its resolution class and aspect ratio.
    fn canvas(short_edge: u32, aspect_ratio: &str, width: u32, height: u32) -> ServedCanvas {
        ServedCanvas {
            short_edge,
            aspect_ratio: aspect_ratio.to_owned(),
            canvas: Canvas { width, height },
        }
    }

    /// A 24 fps text-to-video denoiser with 32 kHz audio serving `canvases`.
    fn denoiser(canvases: Vec<ServedCanvas>) -> VideoDenoiserInfo {
        VideoDenoiserInfo {
            tasks: vec!["t2va".to_owned(), "fl2va".to_owned()],
            schedule_points: 9,
            video_shift: 5.0,
            audio_shift: 5.0,
            frame_rate: 24,
            audio_sample_rate: 32_000,
            frames: uniserve_core::FrameClips {
                min_frames: 4,
                clip_frames: 120,
                clip_latent_frames: 15,
                tail_latent_frames: 1,
            },
            canvases,
            references: uniserve_core::ReferenceLimits::default(),
        }
    }

    /// The default deployment's canvases: 768p 16:9 and 9:16.
    fn served() -> Vec<ServedCanvas> {
        vec![
            canvas(768, "16:9", 1344, 768),
            canvas(768, "9:16", 768, 1344),
        ]
    }

    #[test]
    fn request_maps_to_uniserve_contract() {
        let prepared = match prepare_request(request(), "FastH3", 15.0, &served()) {
            Ok(prepared) => prepared,
            Err(error) => panic!("supported request was rejected: {error}"),
        };
        assert_eq!(prepared.request.task, VideoTask::T2va);
        // The deployment's default raster: 768p 16:9.
        assert_eq!(prepared.request.target.short_edge, 768);
        assert_eq!(prepared.request.target.aspect_ratio, "16:9");
        assert_eq!(prepared.request.target.duration_seconds, Some(5.0));
        assert_eq!(prepared.request.seed, 1000);
        assert!(matches!(prepared.response_format, ResponseFormat::B64Json));
    }

    #[test]
    fn size_or_passthrough_fields_select_the_target() {
        // 768p and 480p, each at 16:9 and 21:9.
        let denoiser = denoiser(vec![
            canvas(768, "16:9", 1344, 768),
            canvas(768, "21:9", 1536, 672),
            canvas(480, "16:9", 832, 480),
            canvas(480, "21:9", 992, 416),
        ]);
        let prepare = |request: Value| prepare_request(request, "FastH3", 15.0, &denoiser.canvases);
        for (size, short_edge, aspect_ratio) in
            [("1344x768", 768, "16:9"), ("992x416", 480, "21:9")]
        {
            let by_size = match prepare(json!({"model": "FastH3", "prompt": "x", "size": size})) {
                Ok(prepared) => prepared.request,
                Err(error) => panic!("size {size} was rejected: {error}"),
            };
            assert_eq!(by_size.target.short_edge, short_edge);
            assert_eq!(by_size.target.aspect_ratio, aspect_ratio);
            let resolution = format!("{short_edge}p");
            let passthrough = json!({"resolution": resolution, "aspect_ratio": aspect_ratio});
            let by_name = match prepare(json!({
                "model": "FastH3",
                "prompt": "x",
                "extra_args": {"media_passthrough": passthrough.clone()},
            })) {
                Ok(prepared) => prepared.request,
                Err(error) => panic!("{passthrough} was rejected: {error}"),
            };
            assert_eq!(by_name, by_size);
            assert!(
                prepare(json!({
                    "model": "FastH3",
                    "prompt": "x",
                    "size": size,
                    "extra_args": {"media_passthrough": passthrough},
                }))
                .is_ok()
            );
        }
        for request in [
            json!({"model": "FastH3", "prompt": "x", "size": "768x1344"}),
            json!({"model": "FastH3", "prompt": "x", "size": "1344x768",
                "extra_args": {"media_passthrough": {"resolution": "480p"}}}),
            json!({"model": "FastH3", "prompt": "x",
                "extra_args": {"media_passthrough": {"aspect_ratio": "1:1"}}}),
            json!({"model": "FastH3", "prompt": "x",
                "extra_args": {"media_passthrough": {"resolution": 480}}}),
        ] {
            assert!(prepare(request).is_err());
        }
        let runtime = runtime_data(15.0, &denoiser);
        assert_eq!(runtime["fps"], 24);
        assert_eq!(runtime["width"], 1344);
        assert_eq!(runtime["height"], 768);
        assert_eq!(
            runtime["sizes"],
            json!(["1344x768", "1536x672", "832x480", "992x416"])
        );
    }

    /// An omitted duration is 5 seconds, or the deployment's maximum when
    /// that is shorter.
    #[test]
    fn an_omitted_duration_defaults_to_five_seconds() {
        for (max_video_seconds, expected) in [(15.0, 5.0), (4.5, 4.5)] {
            let request = json!({"model": "FastH3", "prompt": "A stream in a forest"});
            let prepared = match prepare_request(request, "FastH3", max_video_seconds, &served()) {
                Ok(prepared) => prepared,
                Err(error) => panic!("max {max_video_seconds}s refused the default: {error}"),
            };
            assert_eq!(prepared.request.target.duration_seconds, Some(expected));
        }
    }

    // Each request carries one field or value FastH3 does not support.
    #[test]
    fn unsupported_control_is_rejected() {
        for request in [
            json!({"model":"FastH3", "prompt":"x", "input_reference":"image"}),
            json!({"model":"FastH3", "prompt":"x", "stream":true}),
            json!({"model":"FastH3", "prompt":"x", "size":"832x480"}),
            json!({"model":"FastH3", "prompt":"x", "nvext":{"seed":-1}}),
            json!({"model":"FastH3", "prompt":"x", "nvext":{"fps":24}}),
            json!({"model":"FastH3", "prompt":"x", "nvext":{"num_frames":124}}),
            json!({"model":"FastH3", "prompt":"x", "nvext":{"num_inference_steps":4}}),
            json!({"model":"FastH3", "prompt":"x", "extra_args":{"media_passthrough":{"foo":1}}}),
        ] {
            assert!(prepare_request(request, "FastH3", 15.0, &served()).is_err());
        }
    }

    #[test]
    fn response_formats_embed_the_mp4() {
        let response = video_response(
            "id",
            "FastH3",
            ResponseFormat::B64Json,
            b"mp4",
            &denoiser(served()),
            Duration::from_millis(1_500),
        );
        assert_eq!(response["data"][0]["b64_json"], "bXA0");
        assert_eq!(response["data"][0]["fps"], 24);
        assert_eq!(response["data"][0]["audio_sample_rate"], 32_000);
        assert!(response["data"][0]["url"].is_null());
        assert_eq!(response["inference_time_s"], 1.5);

        let response = video_response(
            "id",
            "FastH3",
            ResponseFormat::Url,
            b"mp4",
            &denoiser(served()),
            Duration::ZERO,
        );
        assert_eq!(response["data"][0]["url"], "data:video/mp4;base64,bXA0");
    }
}
