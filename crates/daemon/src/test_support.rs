//! Simulated components shared by the crate's tests, one fixture each: the
//! byte tokenizer, in-storage tokenizer loading, the engine over simulated
//! schedulers, the video service, and the server's router.
//!
//! Requests travel through the server's router (`http::router`) into a real
//! `Leader`, whose client submits them over the request interface to
//! schedulers that drive `SimEngine` through `SimExecutor`; the schedulers
//! serve the interface on a thread of this process, as an engine process
//! serves it. The default simulator provides no video media components, so
//! the engine refuses every video submission as an invalid request.

use std::collections::{BTreeSet, HashMap};
use std::os::unix::net::UnixStream;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use engine::{EngineConfig, SchedulerHandle, SimEngine, SimExecutor};
use models::video::plan::{MAX_VIDEO_SECONDS, ProcessorConfig, canvas};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;

use crate::engine::client::{EngineClient, EngineInfo};
use crate::input::chat::HfChatRenderer;
use crate::input::media::MediaConnector;
use crate::input::video::VideoService;
use crate::input::{InputProcessor, RequestRenderer};
use crate::leader::Leader;
use crate::model::{ModelConfig, ModelParameters, SamplingDefaults};
use crate::openai::video_jobs::VideoJobs;
use crate::options::ChatTemplateContentFormatOption;
use crate::options::{
    ImageFetchOptions, LoggingOptions, ServeConfig, ServerOptions, VideoJobOptions,
};
use crate::tokenizer::DynTokenizer;

/// Served model name the simulated engines answer to.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// The application crate's byte tokenizer, which the serving tests decode
/// with (see `frontend::test_support::configured_tokenizer`).
pub(crate) use frontend::test_support::{configured_tokenizer, load_tokenizer};

/// The server's router over `engine` under the default server options.
pub(crate) fn router(engine: Leader) -> Router {
    router_with(engine, &ServerOptions::default())
}

/// The daemon's router over `engine` under `server`, with the HTTP body limit
/// for the video submission routes and default video job options.
pub(crate) fn router_with(engine: Leader, server: &ServerOptions) -> Router {
    let jobs = VideoJobs::new(VideoJobOptions::default(), CancellationToken::new());
    crate::http::router(
        Arc::new(engine),
        Arc::new(jobs),
        server,
        crate::http::BODY_LIMIT,
        crate::http::admin::Admin::new(
            CancellationToken::new(),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        ),
        None,
    )
}

/// Builds an engine for a model with `parameters` over one default
/// `SimEngine`.
///
/// The tokenizer is [`configured_tokenizer`], whose
/// `<|im_end|>` (ID 2) is the end-of-sequence token, and the chat template
/// renders only the first message's content. A MiniMax H3 model serves the
/// API's longest duration capacity, as a default video deployment does.
pub(crate) fn sim_engine(parameters: ModelParameters) -> Leader {
    let model = sim_model(parameters);
    let core = sim_core(EngineConfig::sim(SERVED_MODEL), SimEngine::new());
    engine_over(model, configured_tokenizer(), vec![core])
}

/// Starts a scheduler under `config` that drives `worker` in process, with the
/// epoch's policy installed ([`policy_on`]): a scheduler with no policy decides
/// nothing, and a fixture's engine has no Program Server to give it one.
pub(crate) fn sim_core(config: EngineConfig, worker: SimEngine) -> SchedulerHandle {
    let core = SchedulerHandle::with_executor(config, Box::new(SimExecutor::new(worker)))
        .expect("start a sim scheduler");
    policy_on(&core);
    core
}

/// Installs the tree's `fcfs` policy on `core` through the mechanism's own
/// steps (§7.3 steps 2 and 3): `Precopy`, then the switchover at a batch
/// boundary.
///
/// A deployed scheduler takes its policy from the Program Server while its
/// epoch starts; a simulated one has none, so every fixture that submits a
/// request needs this. The program is the real module this tree builds, with
/// the arguments of `core`'s own batch capacity — the limits it declares fit
/// the ceilings the mechanism checks them against, so the precopy is answered
/// with success rather than a refusal that would hide behind a fixture.
///
/// Requires `SCHED_POLICY_WASM`; see [`engine::packaged_policy`], whose panic
/// states the command that builds the module.
pub(crate) fn policy_on(core: &SchedulerHandle) {
    let capacity = engine::BatchCapacity::of(core.info());
    let (program, module) = packaged_policy(capacity);
    let source: std::sync::Arc<dyn engine::PolicySource + Send + Sync> =
        std::sync::Arc::new(engine::PolicyBytes::new(module));
    core.precopy_policy(program, source)
        .recv_timeout(POLICY_ANSWER)
        .expect("the policy precopy is answered within the bound")
        .expect("the packaged policy's limits fit the scheduler's capacity");
    core.switchover_policy()
        .recv_timeout(POLICY_ANSWER)
        .expect("the switchover is answered within the bound")
        .expect("the switchover applies at the next batch boundary");
}

/// How long a fixture waits for an answer it cannot proceed without.
///
/// A scheduler that does not answer must state that, not hang: an unbounded
/// `recv` in a fixture is a suite that stops with no line saying which answer it
/// never got, which is exactly how a 53-second run became a 42-minute one.
pub(crate) const POLICY_ANSWER: std::time::Duration = std::time::Duration::from_secs(120);

/// How long the packaged policy module may take to be built on first use.
///
/// The build runs `cargo` inside the test process, so it can wait on a package
/// cache another cargo holds. Its own name is what says so: without a bound the
/// fixture looks like a deadlock in the daemon.
pub(crate) const POLICY_BUILD: std::time::Duration = std::time::Duration::from_secs(600);

/// [`engine::packaged_policy`] under a bound, so a stalled build names itself.
///
/// The build is blocking and its thread cannot be cancelled; the thread is left
/// to finish rather than joined, which is what a test fixture can do and a
/// service cannot.
pub(crate) fn packaged_policy(capacity: engine::BatchCapacity) -> (engine::PolicyProgram, Vec<u8>) {
    let (answered, answer) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = answered.send(engine::packaged_policy(capacity));
    });
    answer.recv_timeout(POLICY_BUILD).unwrap_or_else(|_| {
        panic!(
            "the fcfs policy module was not built within {} s: a cargo build inside this              process is waiting on a package cache or a wasm32 toolchain another process holds.              Set SCHED_POLICY_WASM to a module built with `cargo build -p scheduling-policy-fcfs              --target wasm32-unknown-unknown --release` to run without the build.",
            POLICY_BUILD.as_secs()
        )
    })
}

/// Builds an engine for a model with `parameters` over started schedulers,
/// one per data-parallel rank; see [`sim_engine`].
pub(crate) fn sim_engine_over(parameters: ModelParameters, cores: Vec<SchedulerHandle>) -> Leader {
    engine_over(sim_model(parameters), configured_tokenizer(), cores)
}

/// A client of `cores`, which serve the request interface on a thread of
/// this process over a socket pair, labelling their statistics with their
/// model.
pub(crate) fn sim_client(cores: Vec<SchedulerHandle>) -> EngineClient {
    let state = if cores.iter().any(SchedulerHandle::starts_on_trial) {
        crate::engine::client::EngineState::Standby
    } else {
        crate::engine::client::EngineState::Ready
    };
    let first = &cores[0];
    let info = EngineInfo {
        limits: first.generation_limits(),
        video_denoiser: first
            .info()
            .video_denoiser
            .as_ref()
            .map(|denoiser| denoiser.description.clone())
            .unwrap_or_default(),
        latent_encoding_units: first
            .media_units(ipc::MediaCall::LatentEncoding)
            .unwrap_or(0),
        schedulers: cores.len() as u32,
        segments: Vec::new(),
        prefix_seeds: Vec::new(),
        capacity: None,
        latent_seed: 0,
    };
    let model_name = first.model_name().to_owned();
    // The control thread routes every submission by an instance of the same
    // program the schedulers carry: `policy_on` installed it on each core, and
    // the engine installs it here for the route. Without one the engine has no
    // policy to route a request with and refuses every submission.
    let policy = {
        let (program, module) = packaged_policy(engine::BatchCapacity::of(first.info()));
        engine::EpochPolicy {
            program,
            module: std::sync::Arc::new(engine::PolicyBytes::new(module)),
        }
    };
    let (daemon, engine) = UnixStream::pair().expect("a socket pair");
    std::thread::Builder::new()
        .name("sim-engine".into())
        .spawn(move || {
            if let Err(error) = engine::serve_requests(engine, cores, Some(policy)) {
                // A simulated engine that ends is the whole of a test's failure and
                // otherwise leaves no line: the client reports only that its process
                // ended, and the reason - a frame the engine refused, a scheduler that
                // did not answer - is lost with the thread. Standard error, so a test
                // run with `--nocapture` carries it beside the assertion.
                eprintln!("the simulated engine ended: {error:#}");
            }
        })
        .expect("start the simulated engine");
    EngineClient::new(
        proto::Connection::from_unix(daemon.try_clone().expect("clone the request socket"))
            .expect("a reader"),
        proto::Connection::from_unix(daemon).expect("a writer"),
        info,
        state,
        crate::leader::stats_sink(model_name, std::sync::Arc::default(), 0),
        None,
    )
    .expect("a client")
}

/// The simulated model's facts.
pub(crate) fn sim_model(parameters: ModelParameters) -> ModelConfig {
    ModelConfig {
        parameters,
        sampling_defaults: SamplingDefaults::default(),
        max_model_len: 4096,
        primary_eos_token_id: Some(2),
        eos_token_ids: BTreeSet::from([2]),
    }
}

/// Serves `model` with `tokenizer` over `cores` without request or stats
/// logging. A MiniMax H3 model serves [`sim_video_service`]; any other
/// model renders chat requests with a template of the first message's
/// content.
pub(crate) fn engine_over(
    model: ModelConfig,
    tokenizer: DynTokenizer,
    cores: Vec<SchedulerHandle>,
) -> Leader {
    let model_name = cores[0].model_name().to_owned();
    let client = std::sync::Arc::new(sim_client(cores));
    leader_over(model, tokenizer, model_name, vec![client])
}

/// Binds the public leader to independent engine connections.
pub(crate) fn leader_over(
    model: ModelConfig,
    tokenizer: DynTokenizer,
    model_name: String,
    clients: Vec<std::sync::Arc<EngineClient>>,
) -> Leader {
    let renderer = match &model.parameters {
        ModelParameters::MiniMaxH3 => RequestRenderer::Video(sim_video_service(MAX_VIDEO_SECONDS)),
        _ => RequestRenderer::Chat(
            HfChatRenderer::new(
                Some("{{ messages[0].content }}".to_string()),
                HashMap::new(),
                ChatTemplateContentFormatOption::String,
            )
            .unwrap(),
        ),
    };
    let processor = InputProcessor::new(
        model,
        tokenizer,
        renderer,
        clients[0].info().limits.clone(),
        false,
    )
    .unwrap();
    let images = MediaConnector::from_image_options(&ImageFetchOptions::default()).unwrap();
    let options = ServeConfig {
        logging_config: LoggingOptions {
            log_requests: false,
            log_stats: false,
        },
        ..ServeConfig::default()
    };
    Leader::new(
        SERVED_MODEL.to_owned(),
        model_name,
        clients,
        processor,
        images,
        &options,
        test_updates(),
    )
}

/// The video service of a four-step text-to-video denoiser that generates
/// every canvas of the canvas rule, with the released checkpoint's vision
/// processor geometry and the default media policy.
pub(crate) fn sim_video_service(max_video_seconds: f64) -> VideoService {
    VideoService::new(
        models::video::DenoiserInfo {
            tasks: vec!["t2va".to_owned()],
            schedule_points: 5,
            video_shift: 12.0,
            audio_shift: 3.0,
            frame_rate: 24,
            audio_sample_rate: 32_000,
            frames: models::video::FrameClips {
                min_frames: 22,
                clip_frames: 17,
                clip_latent_frames: 5,
                tail_latent_frames: 2,
            },
            // The base checkpoint's canvas of each named ratio at 768p.
            canvases: [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)]
                .map(|(width, height): (u32, u32)| models::video::ServedCanvas {
                    short_edge: 768,
                    aspect_ratio: format!("{width}:{height}"),
                    canvas: canvas(f64::from(width), f64::from(height)).unwrap(),
                })
                .to_vec(),
            references: models::video::ReferenceLimits::default(),
        },
        ProcessorConfig {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_min_pixels: 65_536,
            image_max_pixels: 16_777_216,
            video_min_pixels: 4_096,
            video_max_pixels: 25_165_824,
        },
        max_video_seconds,
        1,
        &crate::options::MediaOptions::default(),
        // A text-to-video denoiser takes no references to size.
        models::video::plan::ReferenceConfig::new(&models::video::ReferenceLimits::default()),
        configured_tokenizer(),
    )
    .unwrap()
}

/// How long a fixture's request may take to answer before it is a stall.
///
/// A handler that never answers, or a body that never ends, would otherwise
/// hold the whole run open with nothing said: every step of a fixture is
/// driven by the test and a simulated engine answers at once, so a wait that
/// outlives this is a defect and must name itself.
pub(crate) const STALL: std::time::Duration = std::time::Duration::from_secs(30);

/// Sends `request` through `router` and returns the status, the headers, and
/// the body parsed as JSON (`Null` for an empty body).
pub(crate) async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, HeaderMap, serde_json::Value) {
    let waited = format!("{} {}", request.method(), request.uri());
    let response = match tokio::time::timeout(STALL, router.clone().oneshot(request)).await {
        Ok(response) => response.unwrap(),
        Err(_) => panic!("{waited} answered nothing within {} s", STALL.as_secs()),
    };
    let (parts, body) = response.into_parts();
    let bytes = match tokio::time::timeout(STALL, axum::body::to_bytes(body, usize::MAX)).await {
        Ok(bytes) => bytes.unwrap(),
        Err(_) => panic!("{waited}'s body did not end within {} s", STALL.as_secs()),
    };
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (parts.status, parts.headers, json)
}

/// Builds a JSON `POST` request to `path`.
pub(crate) fn post_json(path: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// A driver over fresh epoch records and a roster that commands nothing: the
/// leaders the crate's tests build never run a switch, and the adapter's own
/// tests drive the real one.
pub(crate) fn test_updates() -> std::sync::Arc<crate::updates::driver::Driver> {
    std::sync::Arc::new(crate::updates::driver::Driver::new(
        std::sync::Arc::new(crate::epoch::Epochs::new()),
        Box::new(Uncommanded),
    ))
}

/// A roster that answers every command and reports nothing: the switch is not
/// the subject of a leader test.
struct Uncommanded;

impl crate::updates::switch::Roster for Uncommanded {
    fn claim(
        &mut self,
        _host: &str,
        _world: u64,
        _scopes: &[proto::process::Scope],
        _retire: &[u64],
    ) -> Result<(), String> {
        Ok(())
    }

    fn spawn(
        &mut self,
        _spawn: proto::process::Spawn,
        _launch: &crate::engine::launch::Launch,
    ) -> Result<crate::updates::switch::Spawned, String> {
        Ok(crate::updates::switch::Spawned {
            generation: _spawn.generation,
            pid: 1,
        })
    }

    fn stop(&mut self, _host: &str, _group: &str, _generation: Option<u64>) {}

    fn command(&mut self, _host: &str, _generation: u64, _command: proto::process::Command) {}

    fn serve(&mut self, _epoch: u64, _requests: &str) {}

    fn resume(&mut self, _host: &str, _generation: u64, _reason: &str) {}

    fn events(&mut self) -> Vec<crate::updates::switch::Event> {
        Vec::new()
    }
}
