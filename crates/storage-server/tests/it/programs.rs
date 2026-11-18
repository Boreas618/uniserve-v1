//! The deployment's KV program on the Storage Server: real programs, compiled to
//! `wasm32-unknown-unknown` by the cargo that built this test, set by the deployment's head and
//! stepped at the handoff and tiering hooks.
//!
//! `program-kv` pulls a handoff's full pages from a tier that holds them all and pushes them
//! otherwise; `program-kv-read` reads every handoff over Mooncake Transfer Engine and moves
//! every page one tier down, and its state takes the pull count `program-kv` saves.

use std::collections::BTreeMap;
use std::path::{Path as FilePath, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ::storage::protocol::{ErrorKind, KvProgram, ProgramEvent, ProgramRecord, ProgramStart};
use ::storage::{Client, Error};
use scheduling::kv::{Found, Handoff, Path, Tier, Transport};
use scheduling::program::Level;

use crate::PATIENCE;
use crate::server::{Running, head};
use crate::sessions::host_copy;

/// The wasm32 module of the program crate `package`, built once per test process into a target
/// directory of the test's own, read from the artifact cargo's JSON messages name.
fn wasm(package: &str) -> Vec<u8> {
    static MODULES: OnceLock<BTreeMap<String, Vec<u8>>> = OnceLock::new();

    let modules = MODULES.get_or_init(|| {
        let mut build = Command::new(env!("CARGO"));
        build
            .args(["build", "--release", "--locked", "--offline", "--lib"])
            .args(["--message-format=json-render-diagnostics"])
            .args(["--target", "wasm32-unknown-unknown"])
            .arg("--target-dir")
            .arg(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("programs"))
            .args(["--package", "program-kv", "--package", "program-kv-read"]);
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
            .filter_map(|message| {
                let name = message["target"]["name"].as_str()?.to_owned();
                let files = message["filenames"].as_array()?.clone();
                let wasm = files
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .find(|file| file.ends_with(".wasm"))?
                    .to_owned();
                Some((name, std::fs::read(wasm).unwrap()))
            })
            .collect()
    });
    modules[&package.replace('-', "_")].clone()
}

/// The KV program `package` named `kv`, saving its state in `state`, stepping its tiering hook
/// every `period` when `tiering`.
fn program(package: &str, state: &FilePath, period: Duration, tiering: bool) -> KvProgram {
    KvProgram {
        name: "kv".to_owned(),
        module: wasm(package),
        args: serde_json::json!({ "watermark": 0.5 }),
        period_ms: u64::try_from(period.as_millis()).unwrap(),
        tiering,
        state: state.to_owned(),
    }
}

/// A handoff of three full pages, the host tier holding `host` of them, between sides that
/// have `transports`.
fn handoff(host: u32, transports: Vec<Transport>) -> Handoff {
    Handoff {
        pages: 3,
        tail: 8,
        found: vec![Found {
            tier: Tier::Host,
            pages: host,
        }],
        local: false,
        transports,
    }
}

/// What the program records, waiting until `done` holds of what was drained so far.
fn records_until(head: &mut Client, done: impl Fn(&[ProgramRecord]) -> bool) -> Vec<ProgramRecord> {
    let started = Instant::now();
    let mut records = Vec::new();
    loop {
        records.extend(head.program_records().unwrap());
        if done(&records) {
            return records;
        }
        assert!(started.elapsed() < PATIENCE, "the records were {records:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The deployment's KV program decides each handoff's path where the handoff can take it. The
/// same module and arguments set again keep the instance; new arguments are a new instance
/// created from them, and another module under the same arguments a new instance restored from
/// the state the one it replaces saved. An output the handoff cannot take, a
/// read over a transport the sides lack, is refused and recorded, and the default rule pushes
/// over the sides' first transport; a module that is not one leaves the program as it was.
#[test]
fn a_kv_program_chooses_each_handoff_path_and_its_state_passes_to_its_replacement() {
    let server = Running::with_children(None);
    let state = tempfile::tempdir().unwrap();
    let mut head = head(&server.socket);
    let hour = Duration::from_secs(3600);

    // With no program, the default rule.
    let nixl = vec![Transport::Nixl];
    assert_eq!(
        head.handoff_path(handoff(3, nixl.clone())).unwrap(),
        Path::Push(Transport::Nixl)
    );

    let mut kv = program("program-kv", state.path(), hour, false);
    assert_eq!(
        head.program(Some(kv.clone())).unwrap(),
        Some(ProgramStart::Created)
    );
    // New arguments: a new instance created from them, though the one it replaces saved its
    // state, which was saved under the others.
    kv.args = serde_json::json!({ "watermark": 0.9 });
    assert_eq!(
        head.program(Some(kv.clone())).unwrap(),
        Some(ProgramStart::Created)
    );
    // The host tier holds every full page: a pull. It holds two of three: a push.
    for _ in 0..2 {
        assert_eq!(
            head.handoff_path(handoff(3, nixl.clone())).unwrap(),
            Path::Pull(Tier::Host)
        );
    }
    assert_eq!(
        head.handoff_path(handoff(2, nixl.clone())).unwrap(),
        Path::Push(Transport::Nixl)
    );
    // Sides without a common transport cannot hand the tail over.
    let refused = head.handoff_path(handoff(3, Vec::new()));
    assert!(
        matches!(
            refused,
            Err(Error::Server {
                kind: ErrorKind::Invalid,
                ..
            })
        ),
        "{refused:?}"
    );
    assert_eq!(
        head.program(Some(kv.clone())).unwrap(),
        Some(ProgramStart::Kept)
    );

    // Another module under the same arguments: a new instance, restored from the two pulls the
    // one it replaces saved.
    let mut read = program("program-kv-read", state.path(), hour, false);
    read.args = kv.args.clone();
    assert_eq!(
        head.program(Some(read)).unwrap(),
        Some(ProgramStart::Restored)
    );
    assert_eq!(
        head.handoff_path(handoff(3, nixl.clone())).unwrap(),
        Path::Push(Transport::Nixl)
    );
    let records: Vec<ProgramEvent> = head
        .program_records()
        .unwrap()
        .into_iter()
        .map(|record| record.event)
        .collect();
    assert_eq!(records.len(), 2, "{records:?}");
    assert_eq!(
        records[0],
        ProgramEvent::Log {
            level: Level::Info,
            text: "read 1 after 2 pulls".to_owned()
        }
    );
    assert!(
        matches!(&records[1], ProgramEvent::Refused { hook, .. } if hook == "handoff"),
        "{records:?}"
    );
    // The records were drained.
    assert!(head.program_records().unwrap().is_empty());

    // Bytes that are no program change nothing.
    let mut broken = program("program-kv-read", state.path(), hour, false);
    broken.module = b"\0asm\x01\0\0\0".to_vec();
    let refused = head.program(Some(broken));
    assert!(
        matches!(
            refused,
            Err(Error::Server {
                kind: ErrorKind::Invalid,
                ..
            })
        ),
        "{refused:?}"
    );
    head.handoff_path(handoff(3, nixl.clone())).unwrap();
    let logged = head.program_records().unwrap();
    assert!(
        logged.iter().any(|record| record.event
            == ProgramEvent::Log {
                level: Level::Info,
                text: "read 2 after 2 pulls".to_owned()
            }),
        "{logged:?}"
    );

    // No program: the instance leaves its state, and the default rule decides.
    assert_eq!(head.program(None).unwrap(), None);
    assert!(state.path().join("kv.state").is_file());
    assert_eq!(
        head.handoff_path(handoff(3, nixl)).unwrap(),
        Path::Push(Transport::Nixl)
    );
}

/// On the leader's node the program's tiering hook is stepped every period over the directory
/// of the node's buffer pool, and its state is saved after each step. A move of a device page
/// is its engine's to make and is skipped; a host page's copy into Mooncake Store needs the
/// connector, which this server does not run, and is skipped with that reason.
#[test]
fn the_leader_s_kv_program_steps_its_tiering_hook_over_the_node_s_pool_every_period() {
    let server = Running::with_children(None);
    // Key 7 is cached on the device and on the host.
    let (_engine, _schedulers, _opened, _parts) = host_copy(&server);
    let state = tempfile::tempdir().unwrap();
    let mut head = head(&server.socket);

    let read = program(
        "program-kv-read",
        state.path(),
        Duration::from_millis(50),
        true,
    );
    assert_eq!(
        head.program(Some(read)).unwrap(),
        Some(ProgramStart::Created)
    );

    let skipped = |records: &[ProgramRecord], reason: &str| {
        records.iter().any(|record| {
            matches!(&record.event, ProgramEvent::Skipped { reason: text } if text.contains(reason))
        })
    };
    let records = records_until(&mut head, |records| {
        skipped(records, "device memory") && skipped(records, "no Mooncake connector")
    });
    assert!(
        records.iter().any(|record| record.event
            == ProgramEvent::Log {
                level: Level::Info,
                text: "2 move(s) over 1 node(s)".to_owned()
            }),
        "{records:?}"
    );
    assert!(state.path().join("kv.state").is_file());
}
