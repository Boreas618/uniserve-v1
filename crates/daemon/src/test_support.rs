//! Simulated components shared by the crate's tests, one fixture each: the
//! byte tokenizer, in-storage tokenizer loading, the engine over simulated
//! schedulers, the video service, and the server's router.
//!
//! Requests travel through the server's router (`http::router`) into a real
//! `Leader`, whose client submits them over the request interface to
//! schedulers that drive `SimEngine` through `SimWorker`; the schedulers
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
use scheduler::{SchedulerHandle, SimEngine, SimWorker};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;

use crate::engine::client::{EngineClient, EngineInfo};
use crate::input::chat::HfChatRenderer;
use crate::input::media::MediaConnector;
use crate::input::video::VideoService;
use crate::input::{InputProcessor, RequestRenderer};
use crate::leader::Leader;
use crate::model::{ModelConfig, SamplingDefaults};
use crate::openai::video_jobs::VideoJobs;
use crate::options::ChatTemplateContentFormat;
use crate::options::{ImageFetchConfig, LoggingConfig, ServerArgs, ServerConfig, VideoJobConfig};
use crate::tokenizer::DynTokenizer;

/// Served model name the simulated engines answer to.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// The application crate's byte tokenizer, which the serving tests decode
/// with (see `frontend::test_support::configured_tokenizer`).
pub(crate) use frontend::test_support::{configured_tokenizer, load_tokenizer};

/// The server's router over `engine` under the default server options.
pub(crate) fn router(engine: Leader) -> Router {
    router_with(engine, &ServerConfig::default())
}

/// The daemon's router over `engine` under `server`, with the HTTP body limit
/// for the video submission routes and default video job options.
pub(crate) fn router_with(engine: Leader, server: &ServerConfig) -> Router {
    let jobs = VideoJobs::new(VideoJobConfig::default(), CancellationToken::new());
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

/// Builds an engine for a model of `family` over one default
/// `SimEngine`.
///
/// The tokenizer is [`configured_tokenizer`], whose
/// `<|im_end|>` (ID 2) is the end-of-sequence token, and the chat template
/// renders only the first message's content. A MiniMax H3 model serves the
/// API's longest duration capacity, as a default video deployment does.
pub(crate) fn sim_engine(family: processing::Family) -> Leader {
    let model = sim_model(family);
    let core = sim_core(scheduler::sim_args(SERVED_MODEL), SimEngine::new());
    engine_over(model, configured_tokenizer(), vec![core])
}

/// Starts a scheduler under `config` that drives `worker` in process, with the
/// epoch's policy installed ([`policy_on`]): a scheduler with no policy decides
/// nothing, and a fixture's engine has no Program Server to give it one.
pub(crate) fn sim_core(args: proto::config::EngineArgs, worker: SimEngine) -> SchedulerHandle {
    let core = SchedulerHandle::with_worker(
        &args,
        scheduler::SchedulerConfig::default(),
        Box::new(SimWorker::new(worker)),
    )
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
/// Requires `SCHED_POLICY_WASM`; see [`scheduler::packaged_policy`], whose panic
/// states the command that builds the module.
pub(crate) fn policy_on(core: &SchedulerHandle) {
    let capacity = scheduler::BatchCapacity::of(core.info());
    let (program, module) = packaged_policy(capacity);
    let source: std::sync::Arc<dyn scheduler::PolicySource + Send + Sync> =
        std::sync::Arc::new(scheduler::PolicyBytes::new(module));
    core.precopy_policy(program, source)
        .recv_timeout(POLICY_ANSWER)
        .expect("the policy precopy is answered within the bound")
        .expect("the packaged policy's limits fit the scheduler's capacity");
    core.switchover_policy()
        .recv_timeout(POLICY_ANSWER)
        .expect("the switchover is answered within the bound")
        .expect("the switchover applies at the next batch boundary");
    // A scheduler that fatal-stops leaves its handle dead, and a dead handle is what
    // an engine reports as "no live scheduler answered the state the route policy is
    // instantiated on" - a sentence about the wrong subject. Its own reason is here,
    // and standard error is where a test run carries it.
    if let Some(reason) = core.failure() {
        eprintln!("the simulated scheduler stopped after its policy was installed: {reason}");
    }
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

/// [`scheduler::packaged_policy`] under a bound, so a stalled build names itself.
///
/// The build is blocking and its thread cannot be cancelled; the thread is left
/// to finish rather than joined, which is what a test fixture can do and a
/// service cannot.
pub(crate) fn packaged_policy(
    capacity: scheduler::BatchCapacity,
) -> (scheduler::PolicyProgram, Vec<u8>) {
    let (answered, answer) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = answered.send(scheduler::packaged_policy(capacity));
    });
    answer.recv_timeout(POLICY_BUILD).unwrap_or_else(|_| {
        panic!(
            "the fcfs policy module was not built within {} s: a cargo build inside this              process is waiting on a package cache or a wasm32 toolchain another process holds.              Set SCHED_POLICY_WASM to a module built with `cargo build -p scheduling-policy-fcfs              --target wasm32-unknown-unknown --release` to run without the build.",
            POLICY_BUILD.as_secs()
        )
    })
}

/// Builds an engine for a model of `family` over started schedulers,
/// one per data-parallel rank; see [`sim_engine`].
pub(crate) fn sim_engine_over(family: processing::Family, cores: Vec<SchedulerHandle>) -> Leader {
    engine_over(sim_model(family), configured_tokenizer(), cores)
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
        let (program, module) = packaged_policy(scheduler::BatchCapacity::of(first.info()));
        scheduler::EpochPolicy {
            program,
            module: std::sync::Arc::new(scheduler::PolicyBytes::new(module)),
        }
    };
    let (daemon, engine) = UnixStream::pair().expect("a socket pair");
    std::thread::Builder::new()
        .name("sim-engine".into())
        .spawn(move || {
            if let Err(error) = scheduler::serve_requests(engine, cores, Some(policy)) {
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

/// A chat family whose outputs are parsed in the Qwen3 formats.
pub(crate) const CHAT: processing::Family = processing::Family {
    id: "chat",
    model_types: &["chat"],
    pipeline_classes: &[],
    serving: processing::Serving::Chat(processing::ChatProfile {
        reasoning: Some(processing::ReasoningFormat::Qwen3),
        tool_calls: Some(processing::ToolCallFormat::Qwen3),
    }),
};

/// A video family whose processor is [`TestProcessor`].
pub(crate) const VIDEO: processing::Family = processing::Family {
    id: "video",
    model_types: &[],
    pipeline_classes: &["TestVideoPipeline"],
    serving: processing::Serving::Video(&TestVideo),
};

/// The video family [`VIDEO`] serves.
struct TestVideo;

impl processing::VideoFamily for TestVideo {
    fn default_prompt_tokens(&self) -> u32 {
        4096
    }

    fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn bind(
        &self,
        _: processing::VideoBinding<'_>,
    ) -> Result<Box<dyn processing::VideoProcessor>, String> {
        Ok(Box::new(TestProcessor))
    }
}

/// A video model of one rule and plain plans, for the daemon's own behavior: a request shorter
/// than a second is refused before its media is fetched, and every image reference is one
/// condition whose rows are its height.
pub(crate) struct TestProcessor;

impl processing::VideoProcessor for TestProcessor {
    fn capabilities(&self) -> serde_json::Value {
        serde_json::json!({"sizes": ["64x64"]})
    }

    fn denoise_steps(&self) -> u32 {
        4
    }

    fn check(&self, request: &processing::VideoSpec) -> Result<(), processing::InputError> {
        match request.length {
            Some(processing::Length::Seconds(seconds)) if seconds < 1.0 => Err(
                processing::InputError::invalid("seconds", "a video lasts at least a second"),
            ),
            _ => Ok(()),
        }
    }

    fn plan(
        &self,
        request: &processing::VideoSpec,
        media: &[processing::MediaMetadata],
        vocabulary: &dyn processing::Vocabulary,
    ) -> Result<processing::VideoPlan, processing::InputError> {
        use proto::video::{Canvas, ConditionMedia, ConditionRole, ImageFit};

        let conditions = request
            .references
            .iter()
            .zip(media)
            .map(|(reference, media)| {
                let processing::MediaMetadata::Image(image) = media else {
                    return Err(processing::InputError::Internal(
                        "the test processor plans images".to_owned(),
                    ));
                };
                let canvas = Canvas {
                    width: image.width,
                    height: image.height,
                };
                Ok(processing::ConditionPlan {
                    role: match reference.frame_index {
                        Some(0) => ConditionRole::FirstFrame,
                        Some(_) => ConditionRole::LastFrame,
                        None => ConditionRole::Reference,
                    },
                    media: ConditionMedia::Image(ImageFit {
                        resized: canvas,
                        left: 0,
                        top: 0,
                        size: canvas,
                    }),
                    vision: None,
                    latent_units: vec![image.height],
                    audio_rows: 0,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prompt_token_ids = vocabulary
            .encode(&request.prompt)
            .map_err(|error| processing::InputError::Internal(error.to_string()))?;
        Ok(processing::VideoPlan {
            size: request.size.unwrap_or(Canvas {
                width: 64,
                height: 64,
            }),
            num_frames: 22,
            fps: 24,
            seed: request.seed.unwrap_or(42),
            prompt_token_ids,
            vision_spans: Vec::new(),
            condition_rows: conditions
                .iter()
                .flat_map(|condition| condition.latent_units.iter())
                .sum(),
            conditions,
            video_units: 2,
        })
    }
}

/// The video service of [`TestProcessor`] under the default media policy.
pub(crate) fn test_video_service() -> VideoService {
    VideoService::new(
        Box::new(TestProcessor),
        &crate::options::MediaConfig::default(),
        configured_tokenizer(),
    )
    .unwrap()
}

/// The simulated model's facts.
pub(crate) fn sim_model(family: processing::Family) -> ModelConfig {
    ModelConfig {
        family,
        sampling_defaults: SamplingDefaults::default(),
        max_model_len: 4096,
        primary_eos_token_id: Some(2),
        eos_token_ids: BTreeSet::from([2]),
    }
}

/// Serves `model` with `tokenizer` over `cores` without request or stats
/// logging. A video model serves [`test_video_service`]; a chat model
/// renders chat requests with a template of the first message's content.
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
    let renderer = match model.video() {
        Some(_) => RequestRenderer::Video(test_video_service()),
        None => RequestRenderer::Chat(
            HfChatRenderer::new(
                Some("{{ messages[0].content }}".to_string()),
                HashMap::new(),
                Some(ChatTemplateContentFormat::String),
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
    let images = MediaConnector::from_image_options(&ImageFetchConfig::default()).unwrap();
    let options = ServerArgs {
        logging_config: LoggingConfig {
            log_requests: false,
            log_stats: false,
        },
        ..ServerArgs::default()
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
    // A response answered as one document carries its body beside the response, and a connection
    // reads it from there: the status of a failure is the body's outcome, so it is read here too.
    let document = response
        .extensions()
        .get::<crate::http::stream::HandedBody>()
        .and_then(crate::http::stream::HandedBody::take)
        .filter(|handed| handed.stream.head_held());
    let (mut parts, body) = response.into_parts();
    let bytes = if let Some(mut handed) = document {
        use crate::http::stream::Opened;
        let opened = tokio::time::timeout(STALL, handed.stream.opened()).await;
        let opened = opened.unwrap_or_else(|_| {
            panic!(
                "{waited}'s document did not open within {} s",
                STALL.as_secs()
            )
        });
        match opened {
            Opened::Frame(frame) => frame.to_vec(),
            Opened::Failed(error, frames) => {
                parts.status = axum::response::IntoResponse::into_response(error).status();
                frames.concat()
            }
            Opened::Ended => Vec::new(),
        }
    } else {
        match tokio::time::timeout(STALL, axum::body::to_bytes(body, usize::MAX)).await {
            Ok(bytes) => bytes.unwrap().to_vec(),
            Err(_) => panic!("{waited}'s body did not end within {} s", STALL.as_secs()),
        }
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
        _epoch: u64,
        _scopes: &[proto::process::Scope],
        _retire: &[u64],
    ) -> Result<(), String> {
        Ok(())
    }

    fn spawn(
        &mut self,
        _spawn: proto::process::Spawn,
        _launch: &proto::config::EngineArgs,
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
