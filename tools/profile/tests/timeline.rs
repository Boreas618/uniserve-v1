//! A deployment's timeline at its public boundary: `span` rows as
//! `/admin/query` renders them, read through an admin socket
//! (`timeline::read`) and converted into a trace (`timeline::convert`).

#![cfg(test)]

use std::collections::BTreeMap;

use profile::admin::Admin;
use profile::timeline::{self, Filter, Span};
use serde_json::{Value, json};

/// The span rows `rows` describes, as `/admin/query` renders them.
fn spans(rows: &Value) -> Vec<Span> {
    serde_json::from_value(rows.clone()).expect("span rows")
}

/// A request's path: the daemon's stages, the scheduler's membership record
/// of the request in batch 7, and batch 7's kernels on two worker ranks (the
/// rank 1 kernel listed first, and its rank 0 kernels one inside the
/// other). Times are nanoseconds since the Unix epoch.
fn request_path() -> Value {
    json!([
        {"labels": {"process": "worker", "replica": "0", "rank": "1", "op": "gemm", "descriptor": "bf16"},
         "fields": {"start": "1000004000", "end": "1000006000", "batch": "7", "kernel": "gemm"}},
        {"labels": {"process": "daemon"},
         "fields": {"start": "1000000000", "end": "1000001000", "request": "r1", "stage": "tokenize"}},
        {"labels": {"process": "scheduler", "replica": "0"},
         "fields": {"start": "1000002000", "end": "1000002500", "request": "r1", "batch": "7"}},
        {"labels": {"process": "worker", "replica": "0", "rank": "0"},
         "fields": {"start": "1000003000", "end": "1000008000", "batch": "7", "stage": "launch"}},
        {"labels": {"process": "worker", "replica": "0", "rank": "0", "op": "gemm", "descriptor": "bf16"},
         "fields": {"start": "1000004000", "end": "1000005000", "batch": "7", "kernel": "gemm"}},
        {"labels": {"process": "daemon"},
         "fields": {"start": "1000009000", "end": "1000009500", "request": "r1", "stage": "detokenize"}}
    ])
}

fn events(trace: &Value) -> &Vec<Value> {
    trace["traceEvents"].as_array().expect("trace events")
}

/// The processes of a trace by name, with their sort index.
fn processes(trace: &Value) -> BTreeMap<String, i64> {
    let mut names = BTreeMap::new();
    let mut order = BTreeMap::new();
    for event in events(trace) {
        let pid = event["pid"].as_i64().expect("a pid");
        match event["name"].as_str() {
            Some("process_name") => {
                names.insert(
                    pid,
                    event["args"]["name"].as_str().expect("a name").to_owned(),
                );
            }
            Some("process_sort_index") => {
                order.insert(pid, event["args"]["sort_index"].as_i64().expect("an index"));
            }
            _ => {}
        }
    }
    names
        .into_iter()
        .map(|(pid, name)| (name, order[&pid]))
        .collect()
}

/// The pid of the process named `name`.
fn pid(trace: &Value, name: &str) -> i64 {
    events(trace)
        .iter()
        .find(|event| event["name"] == "process_name" && event["args"]["name"] == name)
        .and_then(|event| event["pid"].as_i64())
        .unwrap_or_else(|| panic!("no process {name}"))
}

/// The name of each track of process `pid`, by tid.
fn tracks(trace: &Value, pid: i64) -> BTreeMap<i64, String> {
    events(trace)
        .iter()
        .filter(|event| event["name"] == "thread_name" && event["pid"] == pid)
        .map(|event| {
            (
                event["tid"].as_i64().expect("a tid"),
                event["args"]["name"].as_str().expect("a name").to_owned(),
            )
        })
        .collect()
}

/// The slices of process `pid` as (track name, slice name, start, duration),
/// in microseconds.
fn slices(trace: &Value, pid: i64) -> Vec<(String, String, f64, f64)> {
    let names = tracks(trace, pid);
    events(trace)
        .iter()
        .filter(|event| event["ph"] == "X" && event["pid"] == pid)
        .map(|event| {
            (
                names[&event["tid"].as_i64().expect("a tid")].clone(),
                event["name"].as_str().expect("a name").to_owned(),
                event["ts"].as_f64().expect("a start"),
                event["dur"].as_f64().expect("a duration"),
            )
        })
        .collect()
}

/// A request's timeline is a process per place in the request's path, in
/// that order; the request's spans share its track; the membership record
/// is a slice on that track whose arrows reach batch 7's first span on each
/// rank; a kernel keeps its operator and descriptor; times start at the
/// earliest span, in microseconds.
#[test]
fn a_request_timeline_links_the_request_to_its_batch_on_every_rank() {
    let trace = timeline::convert(&spans(&request_path())).expect("a trace");

    assert_eq!(
        processes(&trace),
        BTreeMap::from([
            ("daemon".to_owned(), 1),
            ("scheduler (replica 0)".to_owned(), 2),
            ("worker (replica 0, rank 0)".to_owned(), 3),
            ("worker (replica 0, rank 1)".to_owned(), 4),
        ])
    );

    let daemon = pid(&trace, "daemon");
    let mut stages = slices(&trace, daemon);
    stages.sort_by(|a, b| a.2.total_cmp(&b.2));
    assert_eq!(
        stages,
        vec![
            ("request r1".into(), "tokenize".into(), 0.0, 1.0),
            ("request r1".into(), "detokenize".into(), 9.0, 0.5),
        ]
    );

    let scheduler = pid(&trace, "scheduler (replica 0)");
    assert_eq!(
        slices(&trace, scheduler),
        vec![("request r1".into(), "batch 7".into(), 2.0, 0.5)]
    );

    // Both rank 0 spans nest on one track.
    let rank0 = pid(&trace, "worker (replica 0, rank 0)");
    let rank1 = pid(&trace, "worker (replica 0, rank 1)");
    let mut kernels = slices(&trace, rank0);
    kernels.sort_by(|a, b| a.2.total_cmp(&b.2));
    assert_eq!(
        kernels,
        vec![
            ("batches".into(), "launch".into(), 3.0, 5.0),
            ("batches".into(), "gemm".into(), 4.0, 1.0),
        ]
    );
    let gemm = events(&trace)
        .iter()
        .find(|event| event["ph"] == "X" && event["pid"] == rank1)
        .expect("the rank 1 kernel");
    assert_eq!(
        gemm["args"],
        json!({"op": "gemm", "descriptor": "bf16", "batch": "7", "kernel": "gemm"})
    );

    // Each arrow starts on the membership record and ends on the batch's
    // first span of a rank.
    let membership_track = tracks(&trace, scheduler)
        .into_iter()
        .find(|(_, name)| name == "request r1")
        .map(|(tid, _)| tid)
        .expect("the request's track");
    let mut arrows = BTreeMap::new();
    for event in events(&trace) {
        if let Some(id) = event["id"].as_i64() {
            arrows
                .entry(id)
                .or_insert_with(Vec::new)
                .push(event.clone());
        }
    }
    let mut ends = Vec::new();
    for pair in arrows.values() {
        let [begin, end] = &pair[..] else {
            panic!("an arrow of two events: {pair:?}");
        };
        assert_eq!(begin["ph"], "s");
        assert_eq!(begin["pid"], scheduler);
        assert_eq!(begin["tid"], membership_track);
        assert_eq!(begin["ts"], 2.0);
        assert_eq!(end["ph"], "f");
        assert_eq!(end["bp"], "e");
        ends.push((
            end["pid"].as_i64().expect("a pid"),
            end["ts"].as_f64().expect("a time"),
        ));
    }
    ends.sort_by_key(|end| end.0);
    assert_eq!(ends, vec![(rank0, 3.0), (rank1, 4.0)]);
}

/// Spans of one track that overlap without nesting take further tracks of
/// the same name; a span that starts as another ends reuses the first.
#[test]
fn overlapping_spans_that_do_not_nest_take_another_track() {
    let rows = json!([
        {"labels": {"process": "worker", "replica": "0", "rank": "0"},
         "fields": {"start": "100", "end": "300", "batch": "1", "stage": "launch"}},
        {"labels": {"process": "worker", "replica": "0", "rank": "0"},
         "fields": {"start": "200", "end": "400", "batch": "2", "stage": "launch"}},
        {"labels": {"process": "worker", "replica": "0", "rank": "0"},
         "fields": {"start": "300", "end": "500", "batch": "3", "stage": "launch"}}
    ]);
    let trace = timeline::convert(&spans(&rows)).expect("a trace");
    let worker = pid(&trace, "worker (replica 0, rank 0)");

    let mut placed: Vec<(String, f64)> = slices(&trace, worker)
        .into_iter()
        .map(|(track, _, start, _)| (track, start))
        .collect();
    placed.sort_by(|a, b| a.1.total_cmp(&b.1));
    assert_eq!(
        placed,
        vec![
            ("batches".into(), 0.0),
            ("batches #2".into(), 0.1),
            ("batches".into(), 0.2),
        ]
    );
}

/// A span without its times names the span and the missing field.
#[test]
fn a_span_without_its_end_is_refused() {
    let rows = json!([
        {"labels": {"process": "daemon"}, "fields": {"start": "1", "request": "r1", "stage": "parse"}}
    ]);
    let error = timeline::convert(&spans(&rows)).expect_err("a span without `end`");
    let message = format!("{error:#}");
    assert!(message.contains("span 0"), "{message}");
    assert!(message.contains("`end`"), "{message}");
}

/// An admin socket that answers `POST /admin/query` as the daemon does: the
/// rows of a statement given as the form field `q`, and a refusal of an
/// empty one.
fn serve(socket: std::path::PathBuf, rows: Value) {
    use axum::Router;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;

    let (ready, started) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        runtime.block_on(async move {
            let listener = tokio::net::UnixListener::bind(&socket).expect("the socket");
            let router = Router::new().route(
                "/admin/query",
                post(move |body: String| async move {
                    if body.strip_prefix("q=").is_some_and(|sql| !sql.is_empty()) {
                        axum::Json(rows).into_response()
                    } else {
                        let refusal = json!({"error": {"message": "no statement"}});
                        (StatusCode::BAD_REQUEST, axum::Json(refusal)).into_response()
                    }
                }),
            );
            ready.send(()).expect("the test waits");
            axum::serve(listener, router).await.expect("the server");
        });
    });
    started.recv().expect("the server starts");
}

/// `timeline::read` sends the filter's statement to the admin socket and
/// reads the rows it answers with into spans the trace converts.
#[test]
fn spans_are_read_through_the_admin_socket() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let socket = directory.path().join("admin.sock");
    serve(socket.clone(), request_path());

    let admin = Admin::new(&format!("unix:{}", socket.display())).expect("a client");
    let filter = Filter {
        request: Some("r1".into()),
        ..Filter::default()
    };
    let read = timeline::read(&admin, &filter).expect("the spans");
    assert_eq!(read.len(), 6);
    assert_eq!(read[1].fields["stage"], "tokenize");

    let trace = timeline::convert(&read).expect("a trace");
    assert_eq!(processes(&trace).len(), 4);
}

/// A daemon's refusal reaches the caller with the daemon's message.
#[test]
fn a_refused_statement_states_the_daemons_message() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let socket = directory.path().join("admin.sock");
    serve(socket.clone(), json!([]));

    let admin = Admin::new(&format!("unix:{}", socket.display())).expect("a client");
    let error = admin
        .query::<Value>("")
        .map(|_| ())
        .expect_err("an empty statement is refused");
    let message = format!("{error:#}");
    assert!(
        message.contains("400 Bad Request: no statement"),
        "{message}"
    );
}
