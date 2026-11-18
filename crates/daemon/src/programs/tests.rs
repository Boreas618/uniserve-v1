//! The leader's programs run real programs: the crates under `crates/program/tests/programs`,
//! compiled to `wasm32-unknown-unknown` by the cargo that builds these tests, over a real
//! observation store and `program-state/` directory. The node's apply is the receiving end of
//! the outputs channel, which each test answers.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use program::Kind;
use scheduling::routing::{Replica, Request, Role, Route, Routing};
use sha2::{Digest as _, Sha256};
use tokio::sync::mpsc;

use super::*;
use crate::observation::{Event, Labels, Retention, System};

/// The wasm32 module of the program crate `program-<name>`, built once per test process into a
/// target directory beside this test's own, read from the artifact cargo's JSON messages name.
fn wasm(name: &str) -> Arc<[u8]> {
    static MODULES: OnceLock<BTreeMap<String, Arc<[u8]>>> = OnceLock::new();

    let modules = MODULES.get_or_init(|| {
        // The test executable is `<target>/<profile>/deps/<name>`.
        let target = std::env::current_exe()
            .unwrap()
            .ancestors()
            .nth(3)
            .unwrap()
            .join("programs");
        let build = Command::new(env!("CARGO"))
            .args(["build", "--release", "--locked", "--offline", "--lib"])
            .args(["--message-format=json-render-diagnostics"])
            .args(["--target", "wasm32-unknown-unknown"])
            .arg("--target-dir")
            .arg(&target)
            .args(["--package", "program-chunk", "--package", "program-fastest"])
            .output()
            .unwrap();
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
                (stem, Arc::from(std::fs::read(&file).unwrap()))
            })
            .collect()
    });
    Arc::clone(&modules[&format!("program_{name}")])
}

/// Program `name` of the package, from the module of the crate `program-<module>`, with
/// `entry` as its table in `config.toml`.
fn loaded(name: &str, module: &str, entry: &str) -> Loaded {
    let bytes = wasm(module);
    let (kind, _) = program::kind(&bytes).unwrap();
    Loaded {
        name: name.to_owned(),
        entry: toml::from_str(entry).unwrap(),
        kind,
        digest: format!("{:x}", Sha256::digest(&bytes)),
        bytes,
    }
}

/// The `chunk` control program, stepping every 50 ms: it halves the prefill chunk while the p99
/// TPOT is over 50 ms.
fn chunk(args: &str) -> Loaded {
    loaded(
        "chunk",
        "chunk",
        &format!("source = \"programs/chunk\"\nperiod = \"50ms\"\nargs = {args}\n"),
    )
}

const CHUNK_ARGS: &str =
    "{ tpot = \"50ms\", ttft = \"1s\", low = 512, high = 16384, cooldown = \"1h\" }";

/// A deployment at revision `revision` whose prefill chunk is `chunk` tokens.
fn current(revision: u64, chunk: usize) -> Current {
    let mut config = crate::config::Config {
        model: Some("org/model".to_owned()),
        ..Default::default()
    };
    config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(chunk);
    Current {
        revision,
        package: Package::of(config, BTreeMap::new()).unwrap(),
    }
}

/// The programs of a deployment at revision 1 with a chunk of 8192, over a fresh store and state
/// directory, and the outputs they hand the node.
fn programs(
    states: &tempfile::TempDir,
) -> (
    Programs,
    Store,
    mpsc::UnboundedReceiver<Output>,
    Arc<Router>,
) {
    let store = Store::new(Arc::new(System), Retention::default()).unwrap();
    let (outputs, written) = mpsc::unbounded_channel();
    let router = Arc::new(Router::default());
    let programs = Programs::new(
        store.clone(),
        States::new(states.path().to_owned()),
        current(1, 8192),
        outputs,
        Arc::clone(&router),
    );
    (programs, store, written, router)
}

/// The `log` events program `name` wrote.
fn logs(store: &Store, name: &str) -> Vec<Event> {
    store
        .events()
        .kind(crate::observation::Kind::Log)
        .writer(name)
        .window(Duration::from_secs(3600))
}

/// Whether any of program `name`'s log events has a text containing `part`.
fn logged(store: &Store, name: &str, part: &str) -> bool {
    logs(store, name).iter().any(|event| {
        event.fields["text"]
            .as_str()
            .is_some_and(|text| text.contains(part))
    })
}

/// A control program steps every period on the observations of the store and the current
/// configuration; its output reaches the node as its write, and its state is saved after the
/// step and restored by the next instance of the same program.
#[tokio::test(flavor = "multi_thread")]
async fn a_control_program_writes_from_what_it_observes_and_keeps_its_state() {
    let states = tempfile::tempdir().unwrap();
    let (programs, store, mut written, _) = programs(&states);
    // Built first: the observations are the last ten seconds' when the program reads them.
    let program = chunk(CHUNK_ARGS);
    // The last ten seconds' TPOT is over the program's 50 ms target.
    for _ in 0..20 {
        store
            .observe("uniserve_tpot_seconds", &Labels::new(), 1, 0.2)
            .unwrap();
    }

    programs.replace(&[program]).await;
    let output = tokio::time::timeout(Duration::from_secs(60), written.recv())
        .await
        .unwrap_or_else(|_| panic!("no write; the program logged {:?}", logs(&store, "chunk")))
        .unwrap();
    assert_eq!(output.writer, "chunk");
    assert_eq!(
        output.input.scheduler_config.chunked_prefill_size,
        NonZeroUsize::new(8192)
    );
    assert_eq!(
        output.output.scheduler_config.chunked_prefill_size,
        NonZeroUsize::new(4096)
    );
    // Only the chunk changed.
    let changes = proto::reload::changes(Some(&output.input), &output.output);
    assert_eq!(changes.len(), 1, "{changes:?}");
    output.done.send(Ok(2)).unwrap();

    // The node converged: the program reads revision 2 from its next step on.
    programs.set_current(current(2, 4096));
    let output = tokio::time::timeout(Duration::from_secs(60), written.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        output.input.scheduler_config.chunked_prefill_size,
        NonZeroUsize::new(4096)
    );
    assert_eq!(
        output.output.scheduler_config.chunked_prefill_size,
        NonZeroUsize::new(2048)
    );
    output
        .done
        .send(Err("refused for the test".to_owned()))
        .unwrap();
    programs.stop().await;

    // The refusal is the program's log event, and its `info!` lines are too.
    assert!(logged(&store, "chunk", "refused for the test"));
    assert!(logged(&store, "chunk", "8192 -> 4096"));

    // The state the program saved counts the changes it made, and the next instance restores it.
    let args = chunk(CHUNK_ARGS).entry.args;
    let saved = States::new(states.path().to_owned())
        .load("chunk", &args)
        .unwrap()
        .unwrap();
    let saved: serde_json::Value = serde_json::from_slice(&saved).unwrap();
    // A third step may have begun before the stop.
    assert!(saved["changes"].as_u64().unwrap() >= 2, "{saved}");
    let (programs, store, _written, _) = self::programs(&states);
    programs.replace(&[chunk(CHUNK_ARGS)]).await;
    programs.stop().await;
    assert!(logged(&store, "chunk", "restored from its saved state"));
}

/// A change keeps the instance of a program whose source, arguments and module are unchanged;
/// a program given other arguments is created from them rather than restored from the state of
/// the old ones; a program whose arguments are not its `Args` does not start, and its failure is
/// recorded.
#[tokio::test(flavor = "multi_thread")]
async fn a_change_replaces_only_the_programs_it_changes() {
    let states = tempfile::tempdir().unwrap();
    let (programs, store, _written, _) = programs(&states);
    let restored = || logged(&store, "chunk", "restored from its saved state");

    programs.replace(&[chunk(CHUNK_ARGS)]).await;
    programs.replace(&[chunk(CHUNK_ARGS)]).await;
    // One instance, kept: none ended and saved for another to restore.
    assert!(!restored());

    // Another module of the same arguments restores what the first one saved when it ended.
    let mut updated = chunk(CHUNK_ARGS);
    updated.digest = "0".repeat(64);
    programs.replace(&[updated]).await;
    assert!(restored(), "{:?}", logs(&store, "chunk"));

    // Other arguments: created from them.
    let wider = CHUNK_ARGS.replace("16384", "32768");
    programs.replace(&[chunk(&wider)]).await;
    let restores = logs(&store, "chunk")
        .iter()
        .filter(|event| event.fields["text"] == "restored from its saved state")
        .count();
    assert_eq!(restores, 1);

    programs
        .replace(&[chunk("{ low = \"not a number\" }")])
        .await;
    programs.stop().await;
    assert!(logged(&store, "chunk", "did not start"));
}

/// The replicas a request is routed among: a mixed replica, a prefill replica and a decode
/// replica.
fn routing() -> Routing {
    let replica = |id: &str, role, queued, running| Replica {
        id: id.to_owned(),
        role,
        queued,
        remaining: scheduling::time::Duration(0),
        running,
        usage: Vec::new(),
        reservable: 1 << 20,
        hits: Vec::new(),
    };
    Routing {
        request: Request {
            prompt: 1024,
            prefix: Vec::new(),
            session: None,
        },
        replicas: vec![
            replica("0.0", Role::Mixed, 8192, 4),
            replica("0.1", Role::Prefill, 0, 0),
            replica("0.2", Role::Decode, 0, 1),
        ],
    }
}

/// The routing program chooses each request's replicas; a route of a replica of the wrong role
/// is refused and recorded, and a program that panics ends, after which no route comes from it.
#[tokio::test(flavor = "multi_thread")]
async fn the_routing_program_routes_and_its_wrong_routes_are_refused() {
    let states = tempfile::tempdir().unwrap();
    let (programs, store, _written, router) = programs(&states);
    let fastest = loaded(
        "fastest",
        "fastest",
        "source = \"programs/fastest\"\nargs = { rate = 10000.0, page = 64 }\n",
    );
    assert_eq!(fastest.kind, Kind::Routing);
    assert_eq!(router.route(&routing()), None, "no program routes yet");

    programs.replace(&[fastest]).await;
    // The idle prefill replica computes the prompt and the decode replica generates.
    assert_eq!(
        router.route(&routing()),
        Some(Route {
            prefill: 1,
            decode: 2
        })
    );

    // With no replica that generates tokens, the program names the prefill replica for both,
    // which the leader refuses.
    let mut prefill_only = routing();
    prefill_only
        .replicas
        .retain(|replica| replica.role == Role::Prefill);
    assert_eq!(router.route(&prefill_only), None);
    assert!(logged(&store, "fastest", "generates no tokens"));

    // With no replica that computes prompts, the program panics and ends.
    let mut decode_only = routing();
    decode_only
        .replicas
        .retain(|replica| replica.role == Role::Decode);
    assert_eq!(router.route(&decode_only), None);
    assert!(logged(&store, "fastest", "no replica computes prompts"));
    assert_eq!(
        router.route(&routing()),
        None,
        "an ended program routes no more"
    );
    programs.stop().await;
}

/// A program's write changes what it changed of the configuration it read and nothing else:
/// another writer's change since its read survives, and an option it returned to its
/// default leaves the statement.
#[test]
fn an_output_merges_by_the_options_it_changed() {
    let mut input = Config::default();
    input.scheduler_config.chunked_prefill_size = NonZeroUsize::new(8192);
    input.max_num_batched_tokens = NonZeroUsize::new(16384);

    // Another writer changed the batch's tokens since the program read.
    let mut current = input.clone();
    current.max_num_batched_tokens = NonZeroUsize::new(32768);

    let mut output = input.clone();
    output.scheduler_config.chunked_prefill_size = NonZeroUsize::new(4096);
    let merged = merge(&input, &output, &current).unwrap();
    assert_eq!(
        merged.scheduler_config.chunked_prefill_size,
        NonZeroUsize::new(4096)
    );
    assert_eq!(merged.max_num_batched_tokens, NonZeroUsize::new(32768));

    // Back to the default: the option is no longer stated.
    let mut output = input.clone();
    output.scheduler_config.chunked_prefill_size = None;
    let merged = merge(&input, &output, &current).unwrap();
    assert_eq!(merged.scheduler_config.chunked_prefill_size, None);
    assert_eq!(merged.max_num_batched_tokens, NonZeroUsize::new(32768));
}
