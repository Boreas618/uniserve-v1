//! The readers of `profile` at their public boundary: an Nsight Systems export's
//! hints (`nsys::analyze`), a cProfile capture's call edges (`calls::analyze`),
//! and a profiling session's file read through `uniserve-profile`.

#![cfg(test)]

use std::path::{Path, PathBuf};
use std::process::Command;

use profile::{calls, nsys};
use serde_json::Value;

/// A fragment of an Nsight Systems export: two threads of one process share a
/// correlation id; the first thread's Runtime wait has a backtrace record of its
/// own and runs inside a batch range of the workers' domain (batch 7) and a
/// `model.` range, the second's Driver wait inside a `model.` range alone.
fn export(directory: &Path) -> PathBuf {
    let path = directory.join("trace.sqlite");
    let db = rusqlite::Connection::open(&path).expect("an export");
    db.execute_batch(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
         INSERT INTO StringIds VALUES
             (1, 'cudaEventSynchronize_v3020'), (2, 'cudaEventSynchronize'), (3, 'cuEventSynchronize');
         CREATE TABLE ENUM_NSYS_EVENT_CLASS (id INTEGER, name TEXT);
         INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES
             (0, 'TRACE_PROCESS_EVENT_CUDA_RUNTIME'), (1, 'TRACE_PROCESS_EVENT_CUDA_DRIVER'),
             (67, 'TRACE_PROCESS_EVENT_CUDABACKTRACE');
         CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
             start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER, correlationId INTEGER,
             nameId INTEGER, returnValue INTEGER, callchainId INTEGER);
         INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
             (125, 130, 0, 16777217, 17, 1, 0, NULL),
             (124, 131, 67, 16777217, 17, 2, 0, 5),
             (125, 130, 1, 16777218, 17, 3, 0, NULL);
         CREATE TABLE NVTX_EVENTS (
             start INTEGER, end INTEGER, eventType INTEGER, rangeId INTEGER, category INTEGER,
             color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER,
             domainId INTEGER, uint64Value INTEGER);
         INSERT INTO NVTX_EVENTS (start, end, eventType, text, globalTid, uint64Value) VALUES
             (90, 210, 59, 'uniserve.worker.execute', 16777217, 7),
             (100, 200, 59, 'model.forward', 16777217, NULL),
             (100, 200, 59, 'model.forward', 16777218, NULL);",
    )
    .expect("the export's tables");
    path
}

/// Each host wait inside a selected range is one hint, a backtrace record
/// being no call of its own, and a call keeps the batch of the batch range
/// around its range; a prefix no range has selects nothing, and says so.
#[test]
fn the_scope_keeps_batch_context_and_counts_each_call_once() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let path = export(directory.path());

    let report = nsys::analyze(&path, "model.").expect("the export is read");
    let waits: Vec<&nsys::Hint> = report
        .hints
        .iter()
        .filter(|hint| hint.rule == "host-sync")
        .collect();
    assert_eq!(waits.len(), 2);
    let mut batches: Vec<(i64, Option<i64>)> = waits
        .iter()
        .map(|hint| (hint.evidence[0].global_tid, hint.evidence[0].scope.batch_id))
        .collect();
    batches.sort_unstable();
    assert_eq!(batches, [(16_777_217, Some(7)), (16_777_218, None)]);
    assert_eq!(report.coverage["scoped_calls"], 2);
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("No GPU copy activities"))
    );

    let missing = nsys::analyze(&path, "uncaptured.").expect("the export is read");
    assert!(missing.hints.is_empty());
    assert!(
        missing
            .notes
            .iter()
            .any(|note| note.contains("No closed NVTX ranges match"))
    );
}

/// A wait overlapping the GIL held on its own thread is a hint, which keeps
/// that hold and the waits for the GIL of other threads of its process that
/// overlap both: not a waiter of another process, of another domain, or one
/// outside the overlap, and not an unclosed range, which coverage counts.
#[test]
fn the_gil_evidence_is_the_hold_on_the_waiting_thread_and_its_process_waiters() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let path = export(directory.path());
    rusqlite::Connection::open(&path)
        .expect("the export")
        .execute_batch(
            "INSERT INTO NVTX_EVENTS (start, eventType, text, globalTid, domainId) VALUES
                 (0, 75, 'GIL Trace', 16777217, 3), (0, 75, 'GIL Trace', 33554433, 3);
             INSERT INTO NVTX_EVENTS (start, end, eventType, text, globalTid, domainId) VALUES
                 (120, 135, 59, 'Holding GIL', 16777217, 3),
                 (126, 129, 59, 'Waiting for GIL', 16777218, 3),
                 (124, 131, 59, 'Waiting for GIL', 33554433, 3),
                 (100, 125, 59, 'Waiting for GIL', 16777219, 3),
                 (130, 140, 59, 'Waiting for GIL', 16777220, 3),
                 (124, 131, 59, 'Holding GIL', 16777218, 4),
                 (124, NULL, 59, 'Waiting for GIL', 16777221, 3);",
        )
        .expect("the GIL ranges");

    let report = nsys::analyze(&path, "model.").expect("the export is read");
    let hints: Vec<&nsys::Hint> = report
        .hints
        .iter()
        .filter(|hint| hint.rule == "gil-held-wait")
        .collect();
    assert_eq!(hints.len(), 1);
    assert!(
        hints[0].observation.contains("for 5 ns"),
        "{}",
        hints[0].observation
    );
    assert!(
        hints[0].observation.contains("1 other thread(s)"),
        "{}",
        hints[0].observation
    );
    let event = &hints[0].evidence[0];
    assert_eq!(event.global_tid, 16_777_217);
    let gil: Vec<(&str, i64)> = event
        .gil
        .iter()
        .map(|state| (state.name.as_str(), state.global_tid))
        .collect();
    assert_eq!(
        gil,
        [("Holding GIL", 16_777_217), ("Waiting for GIL", 16_777_218)]
    );
    assert_eq!(report.coverage["gil_holds"], 1);
    assert_eq!(report.coverage["gil_open_ranges"], 1);
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("Unclosed GIL ranges"))
    );
}

/// Without the profiler's GIL domain, ownership is unknown and no GIL hint is
/// given.
#[test]
fn the_gil_is_unknown_without_the_profilers_evidence() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let report = nsys::analyze(&export(directory.path()), "model.").expect("the export is read");
    assert!(!report.hints.iter().any(|hint| hint.rule == "gil-held-wait"));
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("GIL ownership during waits is unknown"))
    );
}

/// Calls from Python into a selected builtin and its callbacks into Python are
/// edges of their own direction (`tests/fixtures/sorted.py` sorts twice, each
/// sort calling its Python key three times); a selection the capture holds no
/// builtin of reports unknown coverage.
#[test]
fn native_callbacks_and_python_calls_have_separate_directions() {
    let capture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sorted.pstats");

    let report = calls::analyze(&capture, &["sorted".to_owned()]).expect("the capture is read");
    let edge = |direction: &str| {
        report
            .hints
            .iter()
            .find(|hint| hint.direction == direction)
            .unwrap_or_else(|| panic!("no {direction} edge: {:?}", report.hints))
    };
    assert_eq!(edge("python-to-native").calls, 2);
    assert_eq!(edge("native-to-python").calls, 6);
    assert!(edge("native-to-python").callee.ends_with("(key)"));
    assert!(report.notes.iter().any(|note| note.contains("no timeline")));

    let missing =
        calls::analyze(&capture, &["not_captured".to_owned()]).expect("the capture is read");
    assert!(missing.hints.is_empty());
    assert!(
        missing
            .notes
            .iter()
            .any(|note| note.contains("coverage is unknown"))
    );
}

/// A session file in the engine's schema (`crates/engine/src/profiling/schema.sql`)
/// of rank 0: one kernel, and one probed module call of step 3 on device 0
/// whose `%globaltimer` clock sample aligns it 1000 ns later on CUPTI's clock.
fn session(directory: &Path) -> PathBuf {
    let path = directory.join("profile-1-epoch1-rank0.sqlite");
    let db = rusqlite::Connection::open(&path).expect("a session file");
    db.execute_batch(include_str!(
        "../../../crates/engine/src/profiling/schema.sql"
    ))
    .expect("the schema");
    db.execute_batch(
        "INSERT INTO metadata VALUES ('rank', '0'), ('session', '1'), ('epoch', '1'), ('device', '0');
         INSERT INTO strings VALUES (1, 'gemm_kernel');
         INSERT INTO collectors VALUES ('timeline', 'prepared', 1, ''), ('timeline', 'ended', 9, '');
         INSERT INTO kernel (start, end, device, context, stream, correlation, name)
             VALUES (5000, 6000, 0, 1, 7, 11, 1);
         INSERT INTO clocks (source, device, gpu, realtime, cupti, uncertainty)
             VALUES ('globaltimer', 0, 0, 0, 1000, 10);
         INSERT INTO fence_probe VALUES (0, 0, 'forward', 'model.layers.0', 'TransformerLayer');
         INSERT INTO fence VALUES (0, 2000, 0, 3), (0, 2500, 1, 3);",
    )
    .expect("the session's records");
    path
}

/// `summary` states the file's worker, collectors and rows by table, and
/// `perfetto` writes a trace with the kernel on its stream's track and the
/// probed module's call on its device's fence track, aligned to CUPTI's clock.
#[test]
fn a_session_file_is_summarized_and_converted() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let path = session(directory.path());
    let command = env!("CARGO_BIN_EXE_uniserve-profile");

    let output = Command::new(command)
        .arg("summary")
        .arg(&path)
        .output()
        .expect("the reader runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value = serde_json::from_slice(&output.stdout).expect("a JSON summary");
    assert_eq!(summary[0]["rank"], 0);
    assert_eq!(summary[0]["session"], 1);
    assert_eq!(summary[0]["rows"]["kernel"], 1);
    assert_eq!(summary[0]["rows"]["fence"], 2);
    assert_eq!(summary[0]["collectors"].as_array().map(Vec::len), Some(2));

    let trace = directory.path().join("trace.json");
    let status = Command::new(command)
        .arg("perfetto")
        .arg(&path)
        .arg("-o")
        .arg(&trace)
        .status()
        .expect("the converter runs");
    assert!(status.success());
    let trace: Value =
        serde_json::from_str(&std::fs::read_to_string(&trace).expect("a trace")).expect("JSON");
    let events = trace["traceEvents"].as_array().expect("events");
    let slice = |name: &str| {
        events
            .iter()
            .find(|event| event["ph"] == "X" && event["name"] == name)
            .unwrap_or_else(|| panic!("no slice {name}"))
    };
    // Times start at the earliest event (the module's call, at 2000 + 1000 ns),
    // in microseconds.
    let layer = slice("TransformerLayer");
    assert_eq!(layer["ts"], 0.0);
    assert_eq!(layer["dur"], 0.5);
    assert_eq!(layer["args"]["path"], "model.layers.0");
    let kernel = slice("gemm_kernel");
    assert_eq!(kernel["ts"], 2.0);
    assert_eq!(kernel["dur"], 1.0);
}
