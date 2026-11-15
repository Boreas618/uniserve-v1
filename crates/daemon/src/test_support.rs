//! Fixtures shared by the crate's tests, one each: the byte
//! tokenizer, in-storage tokenizer loading, the engine over fakes below its
//! schedulers, the video service, and the server's router.
//!
//! Requests travel through the server's router (`http::router`) into a real
//! `Leader`, whose client submits them over the request interface to the
//! engine's own control thread and schedulers (`engine::serve`), run on a
//! thread of this process over socket pairs; below the schedulers, the
//! executors and the buffer pool are fakes (`testkit::engine`). The default
//! fake serves text alone, so the engine refuses every video submission as an
//! invalid request. The fake engine holds no rank: an operation on the ranks
//! answers with no rank's result, and one on the KV cache's segments (sleep,
//! wake, making room, a weight update) is refused, since the engine holds no
//! session with a Storage Server whose segments it would change.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use proto::config::EngineArgs;
use proto::request::wire;
use testkit::{FakeExecutor, Fixture};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;

use crate::api::input::chat::HfChatRenderer;
use crate::api::input::{Processor, RequestRenderer};
use crate::api::model::{ModelConfig, SamplingDefaults};
use crate::api::openai::video_jobs::VideoJobs;
use crate::api::tokenizer::DynTokenizer;
use crate::engine::client::EngineClient;
use crate::leader::Leader;
use crate::options::ChatTemplateContentFormat;
use crate::options::{LoggingConfig, ServerArgs, ServerConfig, VideoJobConfig};

/// Served model name the test engines serve under.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// The application crate's byte tokenizer, which the serving tests decode
/// with (see `crate::api::test_support::configured_tokenizer`).
pub(crate) use crate::api::test_support::{CHAT, VIDEO, configured_tokenizer, test_video_service};

/// The server's router over `engine` under the default server options.
pub(crate) fn router(engine: impl Into<Arc<Leader>>) -> Router {
    router_with(engine, &ServerConfig::default())
}

/// The daemon's router over `engine` under `server`, with the HTTP body limit
/// for the video submission routes and default video job options.
pub(crate) fn router_with(engine: impl Into<Arc<Leader>>, server: &ServerConfig) -> Router {
    let jobs = VideoJobs::new(VideoJobConfig::default(), CancellationToken::new());
    crate::http::router(
        engine.into(),
        Arc::new(jobs),
        server,
        crate::http::BODY_LIMIT,
        None,
    )
}

/// The state of the admin routes over `engine`, with a new event log.
fn admin_state(engine: impl Into<Arc<Leader>>) -> crate::http::admin::AdminState {
    let leader = engine.into();
    let admin = crate::http::admin::Admin::with_epochs(
        CancellationToken::new(),
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        Arc::new(crate::epoch::Epochs::new()),
    );
    admin.watch(&leader);
    crate::http::admin::AdminState {
        leader,
        admin,
        jobs: Arc::new(VideoJobs::new(
            VideoJobConfig::default(),
            CancellationToken::new(),
        )),
        restarter: None,
    }
}

/// The operator's admin routes over `engine`, as the admin listener serves
/// them.
pub(crate) fn admin_router(engine: impl Into<Arc<Leader>>) -> Router {
    crate::http::admin::router(admin_state(engine))
}

/// The client API and the operator's admin routes over one `engine`, for a
/// test that needs both: a daemon serves them on two listeners.
pub(crate) fn daemon_routers(engine: impl Into<Arc<Leader>>) -> Router {
    let engine = engine.into();
    router(Arc::clone(&engine)).merge(admin_router(engine))
}

/// The harness's admin routes over `engine`, as the harness listener serves
/// them.
pub(crate) fn harness_router(engine: impl Into<Arc<Leader>>) -> Router {
    crate::http::admin::harness_router(admin_state(engine))
}

/// Builds an engine for a model of `family` over the default fake
/// ([`text_fake`]) of one scheduler.
///
/// The tokenizer is [`configured_tokenizer`], whose
/// `<|im_end|>` (ID 2) is the end-of-sequence token, and the chat template
/// renders only the first message's content. A MiniMax H3 model serves the
/// API's longest duration capacity, as a default video deployment does.
pub(crate) fn served_engine(family: processing::Family) -> Leader {
    engine_of(family, testkit::args(SERVED_MODEL), Fixture::new(text_fake))
}

/// How many tokens a default fake's request generates before its end of text,
/// wherever its prompt ends ([`FakeExecutor::with_text_len`]).
pub(crate) const TEXT_LEN: u64 = 8;

/// The token every default fake samples: `a` of [`configured_tokenizer`], which
/// the serving tests restrict their requests to (`allowed_token_ids`) and read
/// back as text.
pub(crate) const LETTER: u32 = b'a' as u32;

/// The default fake: it samples [`LETTER`], and each request's text ends after
/// [`TEXT_LEN`] tokens.
pub(crate) fn text_fake(fake: FakeExecutor) -> FakeExecutor {
    letter_fake(fake).with_text_len(TEXT_LEN)
}

/// A fake sampling [`LETTER`] whose text never ends: a request generates until
/// its own limit.
pub(crate) fn letter_fake(fake: FakeExecutor) -> FakeExecutor {
    fake.with_token(LETTER)
}

/// How long the packaged policy module may take to be built on first use.
///
/// The build runs `cargo` inside the test process, so it can wait on a package
/// cache another cargo holds. Its own name is what says so: without a bound the
/// fixture looks like a deadlock in the daemon.
const POLICY_BUILD: std::time::Duration = std::time::Duration::from_secs(600);

/// Builds the packaged policy module a test engine's schedulers instantiate
/// ([`testkit::packaged_module`]) under a bound, so a stalled build names
/// itself; the engines' starts then load it built.
///
/// The build is blocking and its thread cannot be cancelled; the thread is left
/// to finish rather than joined, which is what a test fixture can do and a
/// service cannot.
fn build_policy() {
    let (built, module) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = built.send(testkit::packaged_module());
    });
    if module.recv_timeout(POLICY_BUILD).is_err() {
        panic!(
            "the arrival policy module was not built within {} s: a cargo build inside this \
             process is waiting on a package cache or a wasm32 toolchain another process holds. \
             Set SCHED_POLICY_WASM to a module built with `cargo build -p \
             scheduling-policy-arrival --target wasm32-unknown-unknown --release` to run \
             without the build.",
            POLICY_BUILD.as_secs()
        )
    }
}

/// Builds an engine for a model of `family` over the engine of `launch` and
/// `fixture` ([`fake_engine`]); see [`served_engine`].
pub(crate) fn engine_of(
    family: processing::Family,
    launch: EngineArgs,
    fixture: Fixture,
) -> Leader {
    engine_over(test_model(family), configured_tokenizer(), launch, fixture)
}

/// A client of the engine of `launch` over the fakes of `fixture`
/// ([`fake_engine`]).
pub(crate) fn fake_client(launch: EngineArgs, fixture: Fixture) -> EngineClient {
    fake_engine(launch, fixture).0
}

/// A client of the engine of `launch`, whose start builds it over the fakes of
/// `fixture` (`testkit::fake_start`), with the engine's process link: the
/// commands a test sends it as its daemon (`Drain`, `Promote`), and the reports
/// it sends back. The engine runs its control thread on a thread of this
/// process (`testkit::spawn`); the client welcomes it as this daemon's engine
/// control does ([`crate::engine::control::welcome`]) and is made once its
/// start ended, labelling its statistics with its model.
///
/// The engine ends once the client and the commands are dropped: its daemon
/// closed both of its interfaces.
pub(crate) fn fake_engine(
    launch: EngineArgs,
    fixture: Fixture,
) -> (
    EngineClient,
    std::sync::mpsc::Sender<proto::process::Command>,
    std::sync::mpsc::Receiver<proto::process::Report>,
) {
    build_policy();
    let engine = launch.engine;
    let model_name = launch.engine_config.model.model.clone();
    let (mut reader, process, _) =
        testkit::spawn(launch, testkit::fake_start(fixture)).expect("start the test's engine");

    // The daemon's half of the handshake, then the state the start ends in.
    let mut writer = reader.writer().expect("a write half of the request link");
    let hello: wire::Hello = reader
        .recv(STALL)
        .expect("the request link reads")
        .expect("the engine says hello in time");
    crate::engine::control::welcome(&mut writer, &hello, engine).expect("the engine is welcome");
    let deadline = Instant::now() + STALL;
    let (info, state) = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        assert!(
            !left.is_zero(),
            "the test's engine did not end its start within {} s",
            STALL.as_secs()
        );
        let frame: Option<wire::FromEngine> = reader.recv(left).expect("the request link reads");
        if let Some(wire::FromEngine {
            kind: Some(wire::from_engine::Kind::State(state)),
        }) = frame
            && let Some(started) =
                crate::engine::control::start_state(state).expect("the engine starts")
        {
            break started;
        }
    };

    let (commands, reports) = process_link(process);
    let client = EngineClient::new(
        reader,
        writer,
        info,
        state,
        crate::leader::stats_sink(
            model_name,
            std::sync::Arc::new(crate::observe::Observer::new(crate::observe::span(
                &crate::options::UpdateConfig::default(),
            ))),
            0,
        ),
        None,
    )
    .expect("a client");
    (client, commands, reports)
}

/// The daemon's end of an engine's process interface `connection`, as a test
/// plays it: each command sent on the returned sender is written as the
/// daemon's `Command` event, and each report the engine writes is received on
/// the returned receiver. The interface closes once the sender is dropped.
fn process_link(
    connection: service::Connection,
) -> (
    std::sync::mpsc::Sender<proto::process::Command>,
    std::sync::mpsc::Receiver<proto::process::Report>,
) {
    use proto::process::{Downstream, Event, Upstream, downstream, event, upstream};

    let (commands, commanded) = std::sync::mpsc::channel::<proto::process::Command>();
    let (reports, reported) = std::sync::mpsc::channel();
    let mut writer = connection
        .writer()
        .expect("a write half of the process interface");
    std::thread::Builder::new()
        .name("process-commands".into())
        .spawn(move || {
            for command in commanded {
                let sent = writer.deliver(&Downstream {
                    kind: Some(downstream::Kind::Event(Event {
                        kind: Some(event::Kind::Command(command)),
                    })),
                });
                if sent.is_err() {
                    break;
                }
            }
            // The daemon closes the interface both ways, which ends the reader
            // below as well as the engine's.
            writer.shutdown();
        })
        .expect("start the process interface's writer");

    let mut reader = connection;
    std::thread::Builder::new()
        .name("process-reports".into())
        .spawn(move || {
            loop {
                match reader.recv::<Upstream>(STALL) {
                    Ok(Some(Upstream {
                        kind: Some(upstream::Kind::Report(report)),
                    })) => {
                        // A test that let go of the reports reads none of them.
                        let _ = reports.send(report);
                    }
                    Ok(_) => {}
                    Err(_) => return,
                }
            }
        })
        .expect("start the process interface's reader");
    (commands, reported)
}

/// The test model's facts.
pub(crate) fn test_model(family: processing::Family) -> ModelConfig {
    ModelConfig {
        family,
        sampling_defaults: SamplingDefaults::default(),
        max_model_len: 4096,
        primary_eos_token_id: Some(2),
        eos_token_ids: BTreeSet::from([2]),
    }
}

/// Serves `model` with `tokenizer` over the engine of `launch` and `fixture`
/// ([`fake_engine`]) without request or stats logging. A video model serves
/// [`test_video_service`]; a chat model renders chat requests with a template
/// of the first message's content.
pub(crate) fn engine_over(
    model: ModelConfig,
    tokenizer: DynTokenizer,
    launch: EngineArgs,
    fixture: Fixture,
) -> Leader {
    let model_name = launch.engine_config.model.model.clone();
    let client = std::sync::Arc::new(fake_client(launch, fixture));
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
    let processor = Processor::new(
        model,
        tokenizer,
        renderer,
        clients[0].info().limits.clone(),
        false,
    )
    .unwrap();
    let options = ServerArgs {
        model: Some(model_name),
        served_model_name: Some(SERVED_MODEL.to_owned()),
        logging_config: LoggingConfig {
            log_requests: false,
            log_stats: false,
        },
        ..ServerArgs::default()
    };
    let observer = std::sync::Arc::new(crate::observe::Observer::new(crate::observe::span(
        &options.update_config,
    )));
    Leader::new(
        processing::Registry::default(),
        clients,
        processor,
        &options,
        observer,
        test_updater(),
    )
    .unwrap()
}

/// How long a fixture's request may wait for its response before it is a stall.
///
/// A handler that never responds, or a body that never ends, would otherwise
/// hold the whole run open with nothing said: every step of a fixture is
/// driven by the test and a simulated engine responds at once, so a wait that
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
        Err(_) => panic!("{waited} got no response within {} s", STALL.as_secs()),
    };
    read_response(response, &waited).await
}

/// The status, the headers, and the body parsed as JSON (`Null` for an empty
/// body) that a connection writes for `response`, the router's answer to the
/// request `waited` names.
pub(crate) async fn read_response(
    response: axum::response::Response,
    waited: &str,
) -> (StatusCode, HeaderMap, serde_json::Value) {
    // A response written as one document carries its body beside the response, and a connection
    // reads it from there: the status of a failure is the body's result, so it is read here too.
    let document = response
        .extensions()
        .get::<crate::http::stream::HandedBody>()
        .and_then(crate::http::stream::HandedBody::take)
        .filter(|held| held.stream.head_held());
    let (mut parts, body) = response.into_parts();
    let bytes = if let Some(mut held) = document {
        use crate::http::stream::Opened;
        let opened = tokio::time::timeout(STALL, held.stream.opened()).await;
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

/// Sends `request`, whose response is a streamed body, through `router` and
/// returns the body's frames as the connection reads them; dropping the
/// handle is the client going away.
pub(crate) async fn send_streamed(
    router: &Router,
    request: Request<Body>,
) -> crate::http::stream::ResponseStream {
    let waited = format!("{} {}", request.method(), request.uri());
    let response = match tokio::time::timeout(STALL, router.clone().oneshot(request)).await {
        Ok(response) => response.unwrap(),
        Err(_) => panic!("{waited} got no response within {} s", STALL.as_secs()),
    };
    assert_eq!(response.status(), StatusCode::OK, "{waited}");
    response
        .extensions()
        .get::<crate::http::stream::HandedBody>()
        .and_then(crate::http::stream::HandedBody::take)
        .unwrap_or_else(|| panic!("{waited} returned no streamed body"))
        .stream
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

/// The engines' and the node daemons' ends of a process link, played by
/// `testkit::process`: every request the leader makes of a node gets the reply
/// a node gives it, every engine the leader spawns is a fake engine that
/// reports as the process interface states, and a test reads what each was
/// sent. An engine's process link is filed under the launch generation it was
/// launched under, as the leader's service files it.
pub(crate) struct Engines {
    /// Every request the leader made of the nodes, in arrival order.
    requests: Arc<std::sync::Mutex<Vec<proto::process::request::Kind>>>,
    /// The commands the daemon sent, by launch generation.
    commands: std::sync::mpsc::Receiver<(u64, proto::process::Command)>,
    /// The reports the engines send up, by launch generation.
    reports: std::sync::mpsc::Sender<(u64, proto::process::Report)>,
    exits: std::sync::mpsc::Sender<proto::process::Exit>,
    greetings: std::sync::mpsc::Sender<(u64, String)>,
    /// The fake engines by launch generation.
    engines: std::collections::BTreeMap<u64, testkit::process::Engine>,
    /// How many of the recorded spawns were started as engines.
    started: usize,
    /// The waker the link's senders fire, which the leader's service would wait on.
    _waker: service::Waker,
}

impl Engines {
    /// The fakes behind a new process link, whose nodes refuse what
    /// `refusals` names.
    pub(crate) fn link(
        refusals: testkit::process::Refusals,
    ) -> (Self, crate::supervision::processes::ProcessLink) {
        type Asked = (
            proto::process::Request,
            std::sync::mpsc::Sender<proto::process::Reply>,
        );
        let (requests, asked) = std::sync::mpsc::channel::<Asked>();
        let (exits, exited) = std::sync::mpsc::channel();
        let (greetings, greeted) = std::sync::mpsc::channel();
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&log);
        // The leader waits for each reply, so a request is recorded before the
        // call that made it returns.
        std::thread::spawn(move || {
            while let Ok((request, reply)) = asked.recv() {
                if let Some(kind) = &request.kind {
                    recorded.lock().unwrap().push(kind.clone());
                }
                let _ = reply.send(testkit::process::node_reply(&request, &refusals));
            }
        });
        let (commands, sent) = std::sync::mpsc::channel();
        let (reports, reported) = std::sync::mpsc::channel();
        let waker = service::Waker::new().expect("a waker");
        let link = crate::supervision::processes::ProcessLink {
            requests: crate::supervision::ToService::new(requests, waker.wake()),
            greetings: greeted,
            exits: exited,
            commands: crate::supervision::ProcessCommands {
                commands: crate::supervision::ToService::new(commands, waker.wake()),
                reports: Arc::new(std::sync::Mutex::new(reported)),
            },
        };
        let engines = Self {
            requests: log,
            commands: sent,
            reports,
            exits,
            greetings,
            engines: std::collections::BTreeMap::new(),
            started: 0,
            _waker: waker,
        };
        (engines, link)
    }

    /// Adds the serving engine of launch generation `generation`, which the
    /// daemon's engine control started: it states its launch generation as
    /// its connection number.
    pub(crate) fn serve(&mut self, generation: u64) {
        self.engines
            .insert(generation, testkit::process::Engine::serving(generation));
    }

    /// Starts every engine the leader spawned since the last call: each
    /// connects, greets with `build` and reports that it parked, or that it
    /// serves, as its launch file states. Returns whether one started.
    pub(crate) fn start_spawned(&mut self, build: &str) -> bool {
        let spawns = self.spawns();
        let new = &spawns[self.started.min(spawns.len())..];
        for spawn in new {
            let launch: proto::config::EngineArgs =
                serde_json::from_slice(&spawn.launch).expect("a launch file");
            let engine = if launch.parked() {
                testkit::process::Engine::parked(spawn.generation)
            } else {
                testkit::process::Engine::serving(spawn.generation)
            };
            let _ = self.greetings.send((spawn.generation, build.to_owned()));
            let _ = self.reports.send((spawn.generation, engine.started()));
            self.engines.insert(spawn.generation, engine);
        }
        let started = !new.is_empty();
        self.started = spawns.len();
        started
    }

    /// Delivers every command the daemon sent to the engine it names, and
    /// sends the engines' reports up. Returns whether one was delivered.
    pub(crate) fn deliver(&mut self) -> bool {
        let mut delivered = false;
        while let Ok((peer, command)) = self.commands.try_recv() {
            delivered = true;
            let Some(engine) = self.engines.get_mut(&peer) else {
                continue;
            };
            for report in engine.command(command) {
                let _ = self.reports.send((peer, report));
            }
        }
        delivered
    }

    /// Runs the update of `updater` as far as the fakes take it: the daemon
    /// reads the engines' reports, the engines the daemon's commands, until
    /// neither has anything left.
    pub(crate) fn run(&mut self, updater: &crate::updates::Updater, build: &str) {
        loop {
            updater.advance();
            let started = self.start_spawned(build);
            if !(self.deliver() || started) {
                return;
            }
        }
    }

    /// The fake engine of launch generation `generation`.
    pub(crate) fn engine(&mut self, generation: u64) -> &mut testkit::process::Engine {
        self.engines
            .get_mut(&generation)
            .expect("an engine of that generation")
    }

    /// The commands the engine of launch generation `generation` was sent.
    pub(crate) fn received(&mut self, generation: u64) -> Vec<proto::process::command::Kind> {
        self.engine(generation).received().to_vec()
    }

    /// Sends `report` up as the engine of launch generation `generation`.
    pub(crate) fn report(&self, generation: u64, report: proto::process::Report) {
        let _ = self.reports.send((generation, report));
    }

    /// The node `host` reports that the engine of launch generation
    /// `generation` exited with `status`.
    pub(crate) fn exit(&self, host: &str, generation: u64, status: &str) {
        let _ = self.exits.send(proto::process::Exit {
            host: host.to_owned(),
            group: crate::engine::control::ENGINE_GROUP.to_owned(),
            rank: 0,
            generation,
            status: status.to_owned(),
            last_line: String::new(),
        });
    }

    /// Every request the leader made of the nodes, in arrival order.
    pub(crate) fn requests(&self) -> Vec<proto::process::request::Kind> {
        self.requests.lock().unwrap().clone()
    }

    /// The spawns among them.
    pub(crate) fn spawns(&self) -> Vec<proto::process::Spawn> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                proto::process::request::Kind::Spawn(spawn) => Some(spawn),
                _ => None,
            })
            .collect()
    }

    /// The claims among them.
    pub(crate) fn claims(&self) -> Vec<proto::process::Claim> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                proto::process::request::Kind::Claim(claim) => Some(claim),
                _ => None,
            })
            .collect()
    }

    /// The launch generations the leader stopped, in order.
    pub(crate) fn stops(&self) -> Vec<Option<u64>> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                proto::process::request::Kind::Stop(stop) => Some(stop.generation),
                _ => None,
            })
            .collect()
    }

    /// The weight retentions among them.
    pub(crate) fn retentions(&self) -> Vec<proto::process::KeepWeights> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                proto::process::request::Kind::KeepWeights(keep) => Some(keep),
                _ => None,
            })
            .collect()
    }
}

/// An updater over fresh epoch records whose engines and nodes are fakes
/// nobody plays: the leaders the crate's tests build start no update, and
/// `updates::tests` drives one.
pub(crate) fn test_updater() -> std::sync::Arc<crate::updates::Updater> {
    let epochs = std::sync::Arc::new(crate::epoch::Epochs::new());
    let (_engines, link) = Engines::link(testkit::process::Refusals::default());
    let processes = crate::supervision::processes::Processes::new(
        Arc::clone(&epochs),
        "deployment".to_owned(),
        link,
    );
    std::sync::Arc::new(crate::updates::Updater::new(epochs, processes))
}
