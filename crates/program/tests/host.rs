//! The program host runs real programs: the crates under `tests/programs`, compiled to
//! `wasm32-unknown-unknown` by the cargo that built this test, a control program (`chunk`), a
//! routing program (`fastest`) and a KV program (`kv`), with test embedders answering their host
//! functions.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use program::{
    Budget, Call, Config, Control, Error, Filter, Function, Host, Instance, Level, Module, Query,
    Reason, Record, Report, State,
};
use scheduling::kv::{
    Directory, Found, Handoff, Holding, Hook, Key, Location, Move, Node, Path, Tier, Tiering,
    Transport,
};
use scheduling::routing::{Hit, Replica, Request, Role, Route, Routing};
use scheduling::time::{Duration, Instant, millis, mins, secs};
use serde_json::json;

/// The wasm32 module of every program crate under `tests/programs`, built once per test process
/// into a target directory of the test's own, so the build does not wait for the lock of the
/// build that runs the tests, and read from the artifacts cargo's JSON messages name.
fn wasm(name: &str) -> Vec<u8> {
    static MODULES: OnceLock<BTreeMap<String, Vec<u8>>> = OnceLock::new();

    let modules = MODULES.get_or_init(|| {
        let packages = ["program-chunk", "program-fastest", "program-kv"];
        let mut build = Command::new(env!("CARGO"));
        build
            .args(["build", "--release", "--locked", "--offline", "--lib"])
            .args(["--message-format=json-render-diagnostics"])
            .args(["--target", "wasm32-unknown-unknown"])
            .arg("--target-dir")
            .arg(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("programs"));
        for package in packages {
            build.args(["--package", package]);
        }
        let build = build.output().unwrap();
        assert!(
            build.status.success(),
            "building the programs: {}",
            String::from_utf8_lossy(&build.stderr)
        );

        build
            .stdout
            .split(|byte| *byte == b'\n')
            .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
            .filter(|message| message["reason"] == "compiler-artifact")
            .flat_map(|message| message["filenames"].as_array().cloned().unwrap_or_default())
            .filter_map(|file| file.as_str().map(PathBuf::from))
            .filter(|file| {
                file.extension()
                    .is_some_and(|extension| extension == "wasm")
            })
            .map(|file| {
                let stem = file.file_stem().unwrap().to_str().unwrap().to_owned();
                (stem, std::fs::read(&file).unwrap())
            })
            .collect()
    });
    modules[&format!("program_{name}")].clone()
}

/// A budget far above what the programs spend.
const BUDGET: Budget = Budget {
    fuel: 50_000_000,
    per_byte: 200,
    memory: 64 << 20,
};

/// What a test embedder was asked.
#[derive(Debug, Default)]
struct Calls {
    logs: Vec<(Level, String)>,
    stats: Vec<Query>,
    events: Vec<Filter>,
    validated: Vec<Config>,
}

/// A leader daemon's embedder: the `stats` table holds one p99 per series, the `events` table
/// the records given, and validation reports `report`.
#[derive(Clone, Default)]
struct Leader {
    calls: Arc<Mutex<Calls>>,
    p99: BTreeMap<String, f64>,
    records: Vec<Record>,
    report: Option<Result<Report, String>>,
}

impl Leader {
    fn with(tpot: f64, ttft: f64) -> Leader {
        Leader {
            p99: BTreeMap::from([
                ("uniserve_tpot_seconds".to_owned(), tpot),
                ("uniserve_ttft_seconds".to_owned(), ttft),
            ]),
            ..Leader::default()
        }
    }
}

impl Host for Leader {
    fn log(&mut self, level: Level, text: &str) {
        self.calls
            .lock()
            .unwrap()
            .logs
            .push((level, text.to_owned()));
    }
}

impl Control for Leader {
    fn stats(&mut self, query: &Query) -> f64 {
        self.calls.lock().unwrap().stats.push(query.clone());
        match query.function {
            Function::Quantile(0.99) => self.p99.get(&query.name).copied().unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }

    fn events(&mut self, filter: &Filter) -> Vec<Record> {
        self.calls.lock().unwrap().events.push(filter.clone());
        self.records
            .iter()
            .filter(|record| record.kind == filter.kind)
            .filter(|record| {
                filter
                    .fields
                    .iter()
                    .all(|(key, value)| record.fields.get(key) == Some(value))
            })
            .cloned()
            .collect()
    }

    fn validate(&mut self, config: &Config) -> Result<Report, String> {
        self.calls.lock().unwrap().validated.push(config.clone());
        self.report.clone().unwrap_or(Ok(Report {
            effect: Some("step".into()),
            interrupts: false,
        }))
    }
}

/// An embedder that answers only `uniserve.log`: the leader's routing host or a Storage
/// Server.
#[derive(Clone, Default)]
struct Logger {
    logs: Arc<Mutex<Vec<(Level, String)>>>,
}

impl Host for Logger {
    fn log(&mut self, level: Level, text: &str) {
        self.logs.lock().unwrap().push((level, text.to_owned()));
    }
}

fn chunk_args() -> serde_json::Value {
    json!({ "tpot": "50ms", "ttft": "1s", "low": 512, "high": 8192, "cooldown": "1m" })
}

fn state(chunk: u32) -> State {
    State::new(
        Instant(1_000_000_000_000),
        Config(json!({ "scheduler_config": { "chunked_prefill_size": chunk } })),
    )
}

fn chunk_of(config: &Config) -> Option<u32> {
    config.get("scheduler_config.chunked_prefill_size")
}

/// The control program reads the last ten seconds' p99 TPOT and TTFT through `uniserve.stats`,
/// halves the chunk while TPOT is over its target, doubles it while TTFT is, and otherwise
/// leaves the configuration as it is, without asking for its revisions or validating.
#[test]
fn a_control_program_steps_on_the_stats_its_host_computes() {
    let module = Module::<Leader>::control(&wasm("chunk")).unwrap();
    let leader = Leader::with(0.080, 0.5);
    let calls = Arc::clone(&leader.calls);
    let mut chunk = Instance::new(&module, leader, &BUDGET, &chunk_args()).unwrap();

    // Creating the program reads nothing.
    assert!(calls.lock().unwrap().stats.is_empty());

    let output: Option<Config> = chunk.step(&state(2048)).unwrap();
    assert_eq!(output.as_ref().and_then(chunk_of), Some(1024));
    {
        let calls = calls.lock().unwrap();
        let read: Vec<(&str, Duration, Function)> = calls
            .stats
            .iter()
            .map(|query| (query.name.as_str(), query.window, query.function))
            .collect();
        assert_eq!(
            read,
            [
                ("uniserve_tpot_seconds", secs(10), Function::Quantile(0.99)),
                ("uniserve_ttft_seconds", secs(10), Function::Quantile(0.99)),
            ]
        );
        assert_eq!(calls.events.len(), 1);
        assert_eq!(calls.events[0].kind, "revision");
        assert_eq!(calls.events[0].fields["writer"], json!("chunk"));
        assert_eq!(calls.validated.len(), 1);
        assert_eq!(chunk_of(&calls.validated[0]), Some(1024));
        assert_eq!(
            calls.logs,
            [(
                Level::Info,
                "chunk 2048 -> 1024 (p99 tpot 0.080s, ttft 0.500s)".to_owned()
            )]
        );
    }

    // TTFT over its target while TPOT has room: double, up to `high`.
    chunk.host_mut().p99 = Leader::with(0.020, 1.5).p99;
    let output: Option<Config> = chunk.step(&state(6000)).unwrap();
    assert_eq!(output.as_ref().and_then(chunk_of), Some(8192));

    // Both within their targets: no change, and nothing read beyond the two series.
    let before = {
        let calls = calls.lock().unwrap();
        (calls.stats.len(), calls.events.len(), calls.validated.len())
    };
    chunk.host_mut().p99 = Leader::with(0.020, 0.5).p99;
    let output: Option<Config> = chunk.step(&state(2048)).unwrap();
    assert_eq!(output, None);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.stats.len(), before.0 + 2);
    assert_eq!(
        (calls.events.len(), calls.validated.len()),
        (before.1, before.2)
    );
}

/// A rolled-back change of the program's own holds it back for its cooldown, read from its
/// revisions; a change validation reports as interrupting is not made.
#[test]
fn a_control_program_reads_its_revisions_and_validates() {
    let module = Module::<Leader>::control(&wasm("chunk")).unwrap();
    let now = state(2048).time;
    let rolled_back = |time: Instant| Record {
        time,
        kind: "revision".into(),
        labels: BTreeMap::new(),
        fields: BTreeMap::from([
            ("writer".to_owned(), json!("chunk")),
            ("result".to_owned(), json!("rolled_back")),
        ]),
        revision: 41,
    };

    let leader = Leader {
        records: vec![rolled_back(now - secs(30))],
        ..Leader::with(0.080, 0.5)
    };
    let mut chunk = Instance::new(&module, leader, &BUDGET, &chunk_args()).unwrap();
    let output: Option<Config> = chunk.step(&state(2048)).unwrap();
    assert_eq!(output, None);

    chunk.host_mut().records = vec![rolled_back(now - mins(2))];
    let output: Option<Config> = chunk.step(&state(2048)).unwrap();
    assert_eq!(output.as_ref().and_then(chunk_of), Some(1024));

    chunk.host_mut().report = Some(Ok(Report {
        effect: Some("epoch".into()),
        interrupts: true,
    }));
    let output: Option<Config> = chunk.step(&state(2048)).unwrap();
    assert_eq!(output, None);

    chunk.host_mut().report = Some(Err("above max_num_batched_tokens".into()));
    let output: Option<Config> = chunk.step(&state(2048)).unwrap();
    assert_eq!(output, None);
    let logs = &chunk.host().calls.lock().unwrap().logs;
    assert_eq!(
        logs.last(),
        Some(&(
            Level::Warn,
            "chunk 1024 is rejected: above max_num_batched_tokens".to_owned()
        ))
    );
}

/// A saved state restores into a new instance that continues where the saved one stopped; a
/// state that is not the program's type, or arguments that are not its `Args`, are reported.
#[test]
fn state_is_saved_and_restored_and_a_foreign_state_is_reported() {
    let module = Module::<Leader>::control(&wasm("chunk")).unwrap();
    let leader = Leader::with(0.080, 0.5);
    let mut chunk = Instance::new(&module, leader.clone(), &BUDGET, &chunk_args()).unwrap();
    let _: Option<Config> = chunk.step(&state(2048)).unwrap();
    let saved = chunk.save().unwrap();

    let mut restored = Instance::restore(&module, leader.clone(), &BUDGET, &saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
    let output: Option<Config> = restored.step(&state(1024)).unwrap();
    assert_eq!(output.as_ref().and_then(chunk_of), Some(512));
    assert_ne!(restored.save().unwrap(), saved);

    // A state saved by another program: the embedder creates the program anew.
    let fastest = Module::<Logger>::new(&wasm("fastest")).unwrap();
    let mut router = Instance::new(&fastest, Logger::default(), &BUDGET, &routing_args()).unwrap();
    let foreign = router.save().unwrap();
    let error = Instance::restore(&module, leader.clone(), &BUDGET, &foreign).err();
    assert!(matches!(error, Some(Error::State(_))), "{error:?}");
    let error = Instance::restore(&module, leader.clone(), &BUDGET, b"not json").err();
    assert!(matches!(error, Some(Error::State(_))), "{error:?}");

    let error = Instance::new(&module, leader, &BUDGET, &json!({ "tpot": "50ms" })).err();
    assert!(matches!(error, Some(Error::Args(_))), "{error:?}");
}

fn routing_args() -> serde_json::Value {
    json!({ "rate": 10000.0, "page": 64 })
}

fn replica(role: Role, queued: u64, remaining: Duration, running: u32, hit: u32) -> Replica {
    Replica {
        id: format!("{role:?}-{queued}"),
        role,
        queued,
        remaining,
        running,
        usage: Vec::new(),
        reservable: 1000,
        hits: vec![Hit {
            tier: Tier::Device,
            tokens: hit,
        }],
    }
}

fn routing(replicas: Vec<Replica>) -> Routing {
    Routing {
        request: Request {
            prompt: 4096,
            prefix: vec![Key([7; 32])],
            session: None,
        },
        replicas,
    }
}

/// The routing program routes on the replicas it is handed and calls no host function; its
/// module links with `uniserve.log` alone, which a control program's does not.
#[test]
fn a_routing_program_routes_with_log_alone() {
    let module = Module::<Logger>::new(&wasm("fastest")).unwrap();
    let logger = Logger::default();
    let logs = Arc::clone(&logger.logs);
    let mut router = Instance::new(&module, logger, &BUDGET, &routing_args()).unwrap();

    // The second prefill replica's queue is longer but it holds most of the prompt; the decode
    // replica with the fewest running requests decodes.
    let replicas = vec![
        replica(Role::Prefill, 0, millis(100), 3, 0),
        replica(Role::Prefill, 512, millis(20), 3, 3072),
        replica(Role::Decode, 0, millis(5), 9, 0),
        replica(Role::Decode, 0, millis(5), 2, 0),
    ];
    let route: Route = router.step(&routing(replicas)).unwrap();
    assert_eq!(
        route,
        Route {
            prefill: 1,
            decode: 3
        }
    );

    // Without disaggregation both are the mixed replica that finishes first.
    let replicas = vec![
        replica(Role::Mixed, 4096, millis(10), 1, 0),
        replica(Role::Mixed, 0, millis(10), 1, 0),
    ];
    let route: Route = router.step(&routing(replicas)).unwrap();
    assert_eq!(
        route,
        Route {
            prefill: 1,
            decode: 1
        }
    );
    assert!(logs.lock().unwrap().is_empty());

    // A control program imports `uniserve.stats`, which a routing host does not offer.
    let error = Module::<Logger>::new(&wasm("chunk")).err();
    assert!(matches!(error, Some(Error::Load(_))), "{error:?}");
}

/// A panic ends the instance with the panic's text, and every later call reports it; so does
/// disabling it.
#[test]
fn a_panic_ends_the_instance_with_its_text() {
    let module = Module::<Logger>::new(&wasm("fastest")).unwrap();
    let mut router = Instance::new(&module, Logger::default(), &BUDGET, &routing_args()).unwrap();

    let error = router.step::<_, Route>(&routing(Vec::new())).unwrap_err();
    let Error::Ended {
        call: Call::Step,
        reason: Reason::Panic(text),
    } = &error
    else {
        panic!("{error:?}");
    };
    assert!(text.contains("no replica computes prompts"), "{text}");

    let replicas = vec![replica(Role::Mixed, 0, millis(1), 0, 0)];
    assert_eq!(
        router.step::<_, Route>(&routing(replicas)).unwrap_err(),
        error
    );
    assert_eq!(router.save().unwrap_err(), error);

    let mut router = Instance::new(&module, Logger::default(), &BUDGET, &routing_args()).unwrap();
    router.disable();
    let replicas = vec![replica(Role::Mixed, 0, millis(1), 0, 0)];
    let error = router.step::<_, Route>(&routing(replicas)).unwrap_err();
    assert!(
        matches!(
            &error,
            Error::Ended {
                reason: Reason::Disabled,
                ..
            }
        ),
        "{error:?}"
    );
}

/// A call that runs past its fuel ends the instance; a call's budget grows with its message,
/// so the same input fits a budget with fuel per byte.
#[test]
fn a_call_past_its_fuel_ends_the_instance() {
    let module = Module::<Logger>::new(&wasm("fastest")).unwrap();
    let replicas: Vec<Replica> = (0..20_000)
        .map(|index| replica(Role::Mixed, index, millis(index), 1, 0))
        .collect();
    let large = routing(replicas);

    let lean = Budget {
        fuel: 2_000_000,
        per_byte: 0,
        ..BUDGET
    };
    let mut router = Instance::new(&module, Logger::default(), &lean, &routing_args()).unwrap();
    let error = router.step::<_, Route>(&large).unwrap_err();
    assert_eq!(
        error,
        Error::Ended {
            call: Call::Step,
            reason: Reason::Fuel
        }
    );

    let metered = Budget {
        per_byte: 200,
        ..lean
    };
    let mut router = Instance::new(&module, Logger::default(), &metered, &routing_args()).unwrap();
    let route: Route = router.step(&large).unwrap();
    assert_eq!(
        route,
        Route {
            prefill: 0,
            decode: 0
        }
    );
    assert!(router.usage().fuel > lean.fuel);

    let starved = Budget {
        fuel: 100,
        per_byte: 0,
        ..BUDGET
    };
    let error = Instance::new(&module, Logger::default(), &starved, &routing_args()).err();
    assert!(
        matches!(
            &error,
            Some(Error::Ended {
                reason: Reason::Fuel,
                ..
            })
        ),
        "{error:?}"
    );
}

/// The KV program answers both hooks of one instance: a handoff's path, and the tiering
/// hook's demotions.
#[test]
fn a_kv_program_answers_both_hooks() {
    let module = Module::<Logger>::new(&wasm("kv")).unwrap();
    let mut kv = Instance::new(
        &module,
        Logger::default(),
        &BUDGET,
        &json!({ "watermark": 0.5 }),
    )
    .unwrap();

    let mut handoff = Handoff {
        pages: 4,
        tail: 10,
        found: vec![
            Found {
                tier: Tier::Store,
                pages: 4,
            },
            Found {
                tier: Tier::Host,
                pages: 4,
            },
        ],
        local: false,
        transports: vec![Transport::Mooncake],
    };
    let path: Path = kv.step(&Hook::<_, &Tiering>::Handoff(&handoff)).unwrap();
    assert_eq!(path, Path::Pull(Tier::Host));

    handoff.found.clear();
    let path: Path = kv.step(&Hook::<_, &Tiering>::Handoff(&handoff)).unwrap();
    assert_eq!(path, Path::Push(Transport::Mooncake));

    let keys: Vec<Key> = (0..5).map(|byte| Key([byte; 32])).collect();
    let tiering = Tiering {
        time: Instant(0),
        directory: Directory {
            nodes: vec![Node {
                name: "gb200-a".into(),
                tiers: vec![Holding {
                    tier: Tier::Device,
                    keys: keys.clone(),
                    capacity: 6,
                }],
            }],
        },
    };
    let moves: Vec<Move> = kv.step(&Hook::<&Handoff, _>::Tiering(&tiering)).unwrap();
    let device = Location {
        node: 0,
        tier: Tier::Device,
    };
    let host = Location {
        node: 0,
        tier: Tier::Host,
    };
    assert_eq!(
        moves,
        keys[..2]
            .iter()
            .map(|&key| Move {
                key,
                from: device,
                to: host,
                copy: false
            })
            .collect::<Vec<_>>()
    );

    // The state of both hooks survives a restart.
    let saved = kv.save().unwrap();
    let mut restored = Instance::restore(&module, Logger::default(), &BUDGET, &saved).unwrap();
    assert_eq!(restored.save().unwrap(), saved);
}

/// Bytes that are not a program module are refused when loaded.
#[test]
fn bytes_that_are_not_a_program_do_not_load() {
    let error = Module::<Logger>::new(b"\0asm but not really").err();
    assert!(matches!(error, Some(Error::Load(_))), "{error:?}");

    // A module without the program interface's exports.
    let error = Module::<Logger>::new(b"\0asm\x01\0\0\0").err();
    assert!(matches!(error, Some(Error::Load(_))), "{error:?}");
}
