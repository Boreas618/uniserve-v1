//! Fixtures shared by the crate's tests, one each: the byte tokenizer, in-storage tokenizer
//! loading, the engine over fakes below its schedulers, an engine a test plays itself, the video
//! service, the server's router and the connection requests are served over.
//!
//! Requests travel over an HTTP/1.1 connection the daemon's own connection code serves
//! (`http::conn`) into the server's router (`http::router`) and a real `Leader`, whose client
//! submits them over the request interface to the engine's own control thread and schedulers
//! (`engine::serve`), run on a thread of this process over socket pairs; below the schedulers,
//! the executors and the buffer pool are fakes (`testkit::engine`). The default fake serves text
//! alone, so the engine refuses every video submission as an invalid request. The fake engine
//! holds no rank: an operation on the ranks answers with no rank's result, and one on the KV
//! cache's segments (sleep, wake, making room, a weight update) is refused, since the engine
//! holds no session with a Storage Server whose segments it would change.
//!
//! A test that states an engine's every answer itself plays the engine over a socket pair
//! ([`scripted_client`]).

use std::collections::{BTreeSet, HashMap};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use http_body_util::BodyExt as _;
use proto::config::EngineArgs;
use proto::request::{Event, RequestId, wire};
use service::Connection;
use testkit::{FakeExecutor, Fixture};
use tokio_util::sync::CancellationToken;

use crate::api::input::chat::HfChatRenderer;
use crate::api::input::{Processor, RequestRenderer};
use crate::api::model::{ModelConfig, SamplingDefaults};
use crate::api::openai::video_jobs::VideoJobs;
use crate::api::tokenizer::DynTokenizer;
use crate::engine::client::{EngineClient, EngineInfo, EngineState};
use crate::leader::Leader;
use crate::options::ChatTemplateContentFormat;
use crate::options::{LoggingConfig, ServerArgs, ServerConfig, VideoJobConfig};

/// Served model name the test engines serve under.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// The chat template of the test chat models: the first message's content, as the prompt.
const FIRST_MESSAGE: &str = include_str!("first_message.jinja");

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
        server.max_request_body,
        None,
    )
}

/// The state of the admin routes over `engine`, with a new event log.
fn admin_state(engine: impl Into<Arc<Leader>>) -> crate::http::admin::AdminState {
    let leader = engine.into();
    let admin = crate::http::admin::Admin::with_epochs(
        CancellationToken::new(),
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        Arc::new(crate::epoch::Epochs::default()),
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
/// The tokenizer is [`configured_tokenizer`], whose `<|im_end|>` (ID 2) is the end-of-sequence
/// token, and the chat template renders only the first message's content. A video family serves
/// the longest output the family declares, as a deployment that states none does.
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
/// process (`testkit::spawn`); the client welcomes it and waits for its start as this daemon's
/// engine control does ([`crate::engine::control::welcome`],
/// [`crate::engine::control::await_start`]), labelling its statistics with its model.
///
/// The engine ends once the client and the commands are dropped: its daemon
/// closed both of its interfaces.
pub(crate) fn fake_engine(
    launch: EngineArgs,
    fixture: Fixture,
) -> (
    EngineClient,
    Sender<proto::process::Command>,
    Receiver<proto::process::Report>,
) {
    let (engine, epoch) = (launch.engine, launch.epoch);
    let model_name = launch.engine_config.model.model.clone();
    let max_model_len = launch.engine_config.model.max_model_len;
    // The policy module the engine's schedulers instantiate is made on the test's own thread, so
    // that a module that cannot be made fails the test rather than the engine's start.
    testkit::packaged_module();

    let (mut reader, process, _) =
        testkit::spawn(launch, testkit::fake_start(fixture)).expect("start the test's engine");

    // The daemon's half of the handshake, then the state the start ends in.
    let mut writer = reader.writer().expect("a write half of the request link");
    let (hello, _) = reader
        .recv_next_with::<wire::Hello>()
        .expect("the engine says hello");
    crate::engine::control::welcome(&mut writer, &hello, engine).expect("the engine is welcome");
    let (info, state) =
        crate::engine::control::await_start(&mut reader).expect("the engine starts");

    let (commands, reports) = process_link(process);
    let client = EngineClient::new(
        reader,
        writer,
        epoch,
        info,
        state,
        crate::leader::stats_sink(
            model_name,
            std::sync::Arc::new(crate::observe::Observer::new(crate::observe::span(
                &crate::options::UpdateConfig::default(),
            ))),
            std::sync::Arc::new(crate::metrics::Metrics::new(max_model_len)),
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
    connection: Connection,
) -> (
    Sender<proto::process::Command>,
    Receiver<proto::process::Report>,
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
            // The read ends with the interface, which either side closes.
            while let Ok((upstream, _)) = reader.recv_next_with::<Upstream>() {
                if let Some(upstream::Kind::Report(report)) = upstream.kind {
                    // A test that let go of the reports reads none of them.
                    let _ = reports.send(report);
                }
            }
        })
        .expect("start the process interface's reader");
    (commands, reported)
}

/// What a test's engine is given to answer ([`scripted_client`]).
pub(crate) enum Arrival<'a> {
    /// A submitted request, which the engine accepts on scheduler 0 before its events.
    Submitted(&'a proto::request::Request),
    /// A takeover of the stream of the request it names, which needs no acceptance.
    TakenOver(RequestId),
}

/// What a test's engine received ([`scripted_client`]); each receiver ends once the engine's
/// connection closed.
pub(crate) struct ScriptedEngine {
    /// The requests submitted to it, in arrival order.
    pub(crate) submitted: Receiver<proto::request::Request>,
    /// The commands for its requests (consumed positions, acknowledgements, cancellations,
    /// takeovers), in arrival order.
    pub(crate) commands: Receiver<wire::Command>,
}

/// What a test's engine serves: [`GenerationLimits`](proto::request::GenerationLimits)'
/// defaults on one scheduler, a batch capacity of one request, no page or trajectory seed, and
/// statistics every second, which a test's engine never sends.
pub(crate) fn engine_info() -> EngineInfo {
    EngineInfo {
        limits: proto::request::GenerationLimits::default(),
        video_denoiser: Vec::new(),
        schedulers: 1,
        segments: Vec::new(),
        capacity: wire::BatchCapacity {
            max_batch: 1,
            max_num_batched_tokens: 1,
            max_running_requests: 1,
        },
        prefix_seeds: Vec::new(),
        latent_seed: 0,
        stats_period: std::time::Duration::from_secs(1),
    }
}

/// A client of a serving engine of epoch `epoch` that the test plays over a socket pair, serving
/// [`engine_info`]; see [`scripted_client_with`].
pub(crate) fn scripted_client(
    epoch: u64,
    script: impl Fn(Arrival<'_>) -> Vec<Event> + Send + 'static,
) -> (EngineClient, ScriptedEngine) {
    scripted_client_with(epoch, engine_info(), script)
}

/// A client of a serving engine of epoch `epoch`, which serves `info`, that the test plays over a
/// socket pair: for each arrival the engine writes the events `script` states for it, after the
/// acceptance of a submitted request, and it hands what it received to the returned
/// [`ScriptedEngine`]. The engine's thread ends with the connection, which the client closes as
/// it drops.
pub(crate) fn scripted_client_with(
    epoch: u64,
    info: EngineInfo,
    script: impl Fn(Arrival<'_>) -> Vec<Event> + Send + 'static,
) -> (EngineClient, ScriptedEngine) {
    let (daemon, engine) = UnixStream::pair().expect("a socket pair");
    let reader = Connection::new(daemon).expect("the daemon's end of the request link");
    let writer = reader.writer().expect("a write half of the request link");
    let mut engine = Connection::new(engine).expect("the engine's end of the request link");

    let (submissions, submitted) = std::sync::mpsc::channel();
    let (commanded, commands) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("scripted-engine".into())
        .spawn(move || {
            let write = |engine: &mut Connection, kind: wire::from_engine::Kind| {
                engine.deliver(&wire::FromEngine { kind: Some(kind) })
            };
            let events = |engine: &mut Connection, id: RequestId, events: Vec<Event>| {
                events.into_iter().try_for_each(|event| {
                    write(engine, wire::from_engine::Kind::Event(event.to_wire(id).0))
                })
            };
            // The read ends with the link, which the client closes as it drops.
            while let Ok((frame, _)) = engine.recv_next_with::<wire::ToEngine>() {
                let written = match frame.kind {
                    Some(wire::to_engine::Kind::Submit(submit)) => {
                        submit.requests.into_iter().try_for_each(|request| {
                            let id = RequestId(request.id);
                            let request = proto::request::Request::from_wire(request)
                                .expect("the daemon submits a request the interface states");
                            let accepted = wire::Accepted {
                                request: id.0,
                                scheduler: 0,
                            };
                            write(&mut engine, wire::from_engine::Kind::Accepted(accepted))?;
                            events(&mut engine, id, script(Arrival::Submitted(&request)))?;
                            let _ = submissions.send(request);
                            Ok(())
                        })
                    }
                    Some(wire::to_engine::Kind::Command(command)) => {
                        let id = RequestId(command.request);
                        let taken_over =
                            matches!(command.kind, Some(wire::command::Kind::Takeover(_)));
                        let _ = commanded.send(command);
                        if taken_over {
                            events(&mut engine, id, script(Arrival::TakenOver(id)))
                        } else {
                            Ok(())
                        }
                    }
                    // A request's condition media and the engine commands play no part in a
                    // script.
                    Some(
                        wire::to_engine::Kind::Engine(_) | wire::to_engine::Kind::ConditionPart(_),
                    )
                    | None => Ok(()),
                };
                if written.is_err() {
                    break;
                }
            }
        })
        .expect("start the scripted engine");

    let client = EngineClient::new(
        reader,
        writer,
        epoch,
        info,
        EngineState::Serving,
        Box::new(|_| {}),
        None,
    )
    .expect("a client");
    (
        client,
        ScriptedEngine {
            submitted,
            commands,
        },
    )
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
    let client = Arc::new(fake_client(launch, fixture));
    leader_over(model, tokenizer, model_name, vec![client])
}

/// Binds the public leader to independent engine connections.
pub(crate) fn leader_over(
    model: ModelConfig,
    tokenizer: DynTokenizer,
    model_name: String,
    clients: Vec<Arc<EngineClient>>,
) -> Leader {
    let renderer = match model.video() {
        Some(_) => RequestRenderer::Video(test_video_service()),
        None => RequestRenderer::Chat(
            HfChatRenderer::new(
                Some(FIRST_MESSAGE.to_owned()),
                HashMap::new(),
                Some(ChatTemplateContentFormat::String),
            )
            .unwrap(),
        ),
    };
    let metrics = std::sync::Arc::new(crate::metrics::Metrics::new(model.max_model_len));
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
            ..LoggingConfig::default()
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
        metrics,
        test_updater(),
    )
    .unwrap()
}

/// A response as a client reads it off an HTTP/1.1 connection.
pub(crate) type Received = hyper::Response<hyper::body::Incoming>;

/// Sends `request` to `router` over a new HTTP/1.1 connection that the daemon's own connection
/// code serves (`http::conn::serve_connection`), and returns the response as the client reads
/// it, its body still streaming. The connection closes once the response is dropped.
pub(crate) async fn exchange(router: &Router, mut request: Request<Body>) -> Received {
    let (server, client) = tokio::net::UnixStream::pair().expect("a socket pair");
    // The connection reads requests under the default server options' limits, as a daemon's.
    let served = crate::http::conn::Server {
        app: router.clone(),
        limits: crate::http::conn::Limits::of(&ServerConfig::default()),
        admitting: crate::http::handoff::Admissions::default(),
    };
    tokio::spawn(crate::http::conn::serve_connection(
        tokio_util::either::Either::Right(server),
        served,
        CancellationToken::new(),
    ));

    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(client))
            .await
            .expect("an HTTP/1.1 connection");
    tokio::spawn(connection);
    // An HTTP/1.1 request names its host (RFC 9112 §3.2).
    request
        .headers_mut()
        .entry(axum::http::header::HOST)
        .or_insert(axum::http::HeaderValue::from_static("daemon"));
    sender
        .send_request(request)
        .await
        .expect("the connection answers")
}

/// Sends `request` to `router` over a connection ([`exchange`]) and returns the status, the
/// headers, and the body parsed as JSON (`Null` for an empty body).
pub(crate) async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, HeaderMap, serde_json::Value) {
    let (parts, body) = exchange(router, request).await.into_parts();
    let bytes = body
        .collect()
        .await
        .expect("the response's body ends")
        .to_bytes();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (parts.status, parts.headers, json)
}

/// The JSON documents of a server-sent event stream's `text`: every event's `data` payload but
/// the stream's closing `[DONE]`, in order.
pub(crate) fn sse_documents(text: &str) -> Vec<serde_json::Value> {
    text.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload)
                .unwrap_or_else(|error| panic!("an event's data is JSON ({error}): {payload}"))
        })
        .collect()
}

/// A streamed response's body as its client reads it: dropping it closes the connection, which
/// is the client going away.
pub(crate) struct Streamed(hyper::body::Incoming);

impl Streamed {
    /// The body bytes the next read takes off the connection; `None` once the body ended.
    pub(crate) async fn next_frame(&mut self) -> Option<Bytes> {
        loop {
            let frame = self.0.frame().await?.expect("the body streams");
            if let Ok(data) = frame.into_data() {
                return Some(data);
            }
        }
    }
}

/// Sends `request`, whose response is a streamed body, to `router` over a connection
/// ([`exchange`]), and returns the body as its client reads it.
pub(crate) async fn send_streamed(router: &Router, request: Request<Body>) -> Streamed {
    let waited = format!("{} {}", request.method(), request.uri());
    let response = exchange(router, request).await;
    assert_eq!(response.status(), StatusCode::OK, "{waited}");
    Streamed(response.into_body())
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
        type Asked = (proto::process::Request, Sender<proto::process::Reply>);
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
                    service::lock(&recorded).push(kind.clone());
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
    /// connects on a connection of its own, greets with `build` and reports that it parked, or
    /// that it serves, as its launch states. Returns whether one started.
    pub(crate) fn start_spawned(&mut self, build: &str) -> bool {
        let spawns = self.spawns();
        let new = &spawns[self.started.min(spawns.len())..];
        for spawn in new {
            let launch: EngineArgs =
                serde_json::from_slice(&spawn.launch).expect("a launch the interface states");
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
        });
    }

    /// Every request the leader made of the nodes, in arrival order.
    pub(crate) fn requests(&self) -> Vec<proto::process::request::Kind> {
        service::lock(&self.requests).clone()
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
pub(crate) fn test_updater() -> Arc<crate::updates::Updater> {
    let epochs = Arc::new(crate::epoch::Epochs::default());
    let (_engines, link) = Engines::link(testkit::process::Refusals::default());
    let processes = crate::supervision::processes::Processes::new(
        Arc::clone(&epochs),
        "deployment".to_owned(),
        link,
    );
    Arc::new(crate::updates::Updater::new(epochs, processes))
}
