//! `uniserve-eval run` over its public command line, against a stand-in of the
//! OpenAI chat completions API served in this process, and a stand-in of a
//! deployment's admin socket (`/admin/policy`, `/admin/query`).
//!
//! The evaluator starts each server a profile names and waits for its address to
//! answer `ready_path`, or for the readiness record a `ready_fd` server writes;
//! the stand-ins answer on the address and the socket, and the profile's command
//! is a process that only reports its record and stays alive until the evaluator
//! stops it.

#![cfg(test)]

use std::io::Read as _;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse as _, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};

/// The stand-in's answer to one streamed chat completion: one token of content
/// per event, as many as the request's `max_completion_tokens`, then the usage the request
/// asked for with `stream_options.include_usage`, then `[DONE]`.
async fn chat(axum::Json(request): axum::Json<Value>) -> Response {
    let tokens = request["max_completion_tokens"].as_u64().unwrap_or(1);
    let mut events = String::new();
    for index in 0..tokens {
        let chunk = json!({
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": format!("t{index} ")}, "finish_reason": null}],
        });
        events.push_str(&format!("data: {chunk}\n\n"));
    }
    let end = json!({
        "id": "chat-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": 5 + tokens},
    });
    events.push_str(&format!("data: {end}\n\ndata: [DONE]\n\n"));
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(events))
        .unwrap_or_default()
}

/// Serves the stand-in on a free local port from a thread of its own, and
/// returns its address.
fn serve() -> SocketAddr {
    let runtime = tokio::runtime::Runtime::new().expect("a runtime");
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("a free port");
    let address = listener.local_addr().expect("the listener's address");
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(chat));
    std::thread::spawn(move || runtime.block_on(async move { axum::serve(listener, app).await }));
    address
}

/// What the admin stand-in observed and answers: the `config.toml` of each
/// package applied, and how many counter totals were read.
#[derive(Default)]
struct Admin {
    applied: Mutex<Vec<String>>,
    applied_at: tokio::sync::Notify,
    totals_read: Mutex<u32>,
}

/// The wall clock in nanoseconds since the Unix epoch.
fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as i64)
        .unwrap_or_default()
}

/// `PUT /admin/policy`: the archive's `config.toml` is recorded, and the change
/// converges at once.
async fn apply(State(admin): State<Arc<Admin>>, body: Bytes) -> Response {
    let mut archive = tar::Archive::new(body.as_ref());
    let mut config = None;
    for entry in archive.entries().expect("a tar archive") {
        let mut entry = entry.expect("an archive entry");
        if entry.path().expect("a path").as_ref() == Path::new("config.toml") {
            let mut text = String::new();
            entry.read_to_string(&mut text).expect("a text file");
            config = Some(text);
        }
    }
    let Some(config) = config else {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(r#"{"error": {"message": "no config.toml"}}"#))
            .unwrap_or_default();
    };
    admin
        .applied
        .lock()
        .expect("the stand-in's lock")
        .push(config);
    admin.applied_at.notify_waiters();
    axum::Json(json!({"revision": 2, "result": "converged", "effect": "step"})).into_response()
}

/// `POST /admin/query`: the store's counter totals, which advance by three
/// requests and two observations after the first reading, or its buckets.
async fn query(State(admin): State<Arc<Admin>>, body: Bytes) -> Response {
    let form: Vec<(String, String)> = serde_urlencoded::from_bytes(&body).expect("a form");
    let sql = &form
        .iter()
        .find(|(key, _)| key == "q")
        .expect("a statement")
        .1;
    if sql.contains("GROUP BY") {
        let read = {
            let mut read = admin.totals_read.lock().expect("the stand-in's lock");
            *read += 1;
            *read
        };
        let (requests, count, sum) = if read == 1 {
            (5.0, 2, 0.25)
        } else {
            (8.0, 4, 0.75)
        };
        return axum::Json(json!([
            {"name": "uniserve_requests_total", "labels": {"outcome": "length", "replica": "0"},
             "total": requests, "count": null, "sum": null},
            {"name": "uniserve_time_to_first_token_seconds", "labels": {},
             "total": 0.1, "count": count, "sum": sum},
        ]))
        .into_response();
    }
    axum::Json(json!([
        {"time": now_ns(), "name": "uniserve_requests_total",
         "labels": {"outcome": "length", "replica": "0"}, "value": 3.0, "revision": 2},
    ]))
    .into_response()
}

/// `GET /admin/query` as server-sent events: once a package was applied, the
/// update's event reporting its trial.
async fn subscribe(State(admin): State<Arc<Admin>>, headers: HeaderMap) -> Response {
    assert_eq!(headers[header::ACCEPT], "text/event-stream");
    let stream = futures::stream::unfold(Some(admin), |admin| async move {
        let admin = admin?;
        let applied = admin.applied_at.notified();
        if admin
            .applied
            .lock()
            .expect("the stand-in's lock")
            .is_empty()
        {
            applied.await;
        }
        let rows = json!([{
            "time": now_ns(), "kind": "log", "labels": {"source": "update", "update": "2"},
            "fields": {"phase": "trial", "message": "update 2 serves on trial"}, "revision": 2,
        }]);
        let message = format!("id: 7\ndata: {rows}\n\n");
        Some((Ok::<_, std::io::Error>(Bytes::from(message)), None))
    });
    // The stream stays open after its message, as the store's does.
    let open = futures::StreamExt::chain(stream, futures::stream::pending());
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(open))
        .unwrap_or_default()
}

/// Serves the admin stand-in on a Unix socket at `socket` from a thread of its
/// own.
fn serve_admin(socket: &Path) -> Arc<Admin> {
    let admin = Arc::new(Admin::default());
    let runtime = tokio::runtime::Runtime::new().expect("a runtime");
    let listener = runtime
        .block_on(async { tokio::net::UnixListener::bind(socket) })
        .expect("the admin socket");
    let app = Router::new()
        .route("/admin/policy", axum::routing::put(apply))
        .route("/admin/query", post(query).get(subscribe))
        .with_state(Arc::clone(&admin));
    std::thread::spawn(move || runtime.block_on(async move { axum::serve(listener, app).await }));
    admin
}

/// A profile of one text point over `rows`, sent to the stand-in at `address`.
fn profile(directory: &Path, address: SocketAddr, rows: &Path) -> std::path::PathBuf {
    let path = directory.join("profile.toml");
    let text = format!(
        r#"[servers.standin]
host = "127.0.0.1"
port = {port}
ready_path = "/health"
command = ["sleep", "600"]

[benchmarks.text-standin]
server = "standin"
task = "text"
model = "standin"
dataset = "jsonl"
dataset_path = "{rows}"
load = {{ num_prompts = 3, request_rate = inf, max_concurrency = 2, warmup_requests = 1, seed = 42 }}
sampling = {{ temperature = 0.0, max_tokens = 8, ignore_eos = true, stream = true }}
metrics = {{ output_throughput = "higher" }}
"#,
        port = address.port(),
        rows = rows.display(),
    );
    std::fs::write(&path, text).expect("the profile is written");
    path
}

/// A run measures every row once, each stream carrying the tokens its request
/// asked for with the server's usage, and writes a valid bundle that records
/// the point it measured.
#[test]
fn a_text_point_measures_every_row_and_writes_its_bundle() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let rows = directory.path().join("rows.jsonl");
    let lines: Vec<String> = (0..3)
        .map(|index| {
            json!({"id": format!("row-{index}"), "prompt": format!("question {index}")}).to_string()
        })
        .collect();
    std::fs::write(&rows, lines.join("\n") + "\n").expect("the rows are written");
    let config = profile(directory.path(), serve(), &rows);
    let output = directory.path().join("results");

    let status = Command::new(env!("CARGO_BIN_EXE_uniserve-eval"))
        .args(["run", "--config"])
        .arg(&config)
        .arg("text-standin")
        .arg("--output-root")
        .arg(&output)
        .env("TMPDIR", directory.path())
        .status()
        .expect("the evaluator runs");
    let bundle = output.join("text-standin");
    let summary = std::fs::read_to_string(bundle.join("summary.json")).unwrap_or_default();
    assert!(
        status.success(),
        "uniserve-eval run exited with {status}: {summary}"
    );

    let records: Vec<Value> = std::fs::read_to_string(bundle.join("requests.jsonl"))
        .expect("the requests are recorded")
        .lines()
        .map(|line| serde_json::from_str(line).expect("a JSON record"))
        .collect();
    assert_eq!(records.len(), 3);
    for record in &records {
        assert_eq!(record["success"], true, "{record}");
        assert_eq!(record["output_len"], 8, "{record}");
        assert_eq!(record["output_len_source"], "server_usage", "{record}");
    }
    let mut rows: Vec<&str> = records
        .iter()
        .filter_map(|record| record["example"]["id"].as_str())
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, ["row-0", "row-1", "row-2"]);

    let summary: Value = serde_json::from_str(&summary).expect("a JSON summary");
    assert_eq!(summary["validation"]["valid"], true, "{summary}");
    let run: Value = serde_json::from_str(
        &std::fs::read_to_string(bundle.join("run.json")).expect("a run state"),
    )
    .expect("JSON");
    assert_eq!(run["status"], "completed", "{run}");
}

/// A timeline point on a server that reports its readiness record applies its
/// change's package through the admin socket the record names, waits for the
/// update's phase among the deployment's events, and reports how far the
/// store's counters advanced in its window.
#[test]
fn a_timeline_point_changes_and_observes_the_deployment_through_its_admin_socket() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let rows = directory.path().join("rows.jsonl");
    std::fs::write(
        &rows,
        json!({"id": "row-0", "prompt": "question"}).to_string() + "\n",
    )
    .expect("the rows are written");
    let package = directory.path().join("next");
    std::fs::create_dir_all(package.join("programs")).expect("the package's directory");
    std::fs::write(package.join("config.toml"), "paused = true\n").expect("the package's config");
    let socket: PathBuf = directory.path().join("admin.sock");
    let admin = serve_admin(&socket);
    let address = serve();

    // The server's command writes its serving record on the descriptor the
    // evaluator appends as `--ready-fd N` ($2 to the script).
    let record = json!({
        "state": "serving", "pid": 1, "instance": null,
        "endpoints": {"admin": format!("unix:{}", socket.display())}, "settings": null,
    });
    let config = directory.path().join("profile.toml");
    let text = format!(
        r#"[servers.standin]
host = "127.0.0.1"
port = {port}
ready_fd = true
command = ["bash", "-c", 'printf "%s\n" "$RECORD" >&"$2"; exec sleep 600', "standin"]
environment = {{ RECORD = '{record}' }}

[benchmarks.timeline-standin]
server = "standin"
task = "text"
model = "standin"
dataset = "jsonl"
dataset_path = "{rows}"
load = {{ num_prompts = 1, request_rate = inf, max_concurrency = 1, warmup_requests = 0, seed = 42, duration_s = 0.5 }}
sampling = {{ temperature = 0.0, max_tokens = 4, ignore_eos = true, stream = true }}
metrics = {{ output_throughput = "higher" }}

[[benchmarks.timeline-standin.changes]]
name = "pause"
at_s = 0.1
policy = "next"

[[benchmarks.timeline-standin.changes]]
name = "after-trial"
after = "pause"
phase = "trial"
command = ["true"]
"#,
        port = address.port(),
        rows = rows.display(),
    );
    std::fs::write(&config, text).expect("the profile is written");
    let output = directory.path().join("results");

    let status = Command::new(env!("CARGO_BIN_EXE_uniserve-eval"))
        .args(["run", "--config"])
        .arg(&config)
        .arg("timeline-standin")
        .arg("--output-root")
        .arg(&output)
        .env("TMPDIR", directory.path())
        .status()
        .expect("the evaluator runs");
    let bundle = output.join("timeline-standin");
    let summary = std::fs::read_to_string(bundle.join("summary.json")).unwrap_or_default();
    let timeline = std::fs::read_to_string(bundle.join("timeline.json")).unwrap_or_default();
    assert!(
        status.success(),
        "uniserve-eval run exited with {status}: {summary} {timeline}"
    );

    assert_eq!(
        *admin.applied.lock().expect("the stand-in's lock"),
        ["paused = true\n"]
    );
    let timeline: Value = serde_json::from_str(&timeline).expect("a JSON timeline");
    let changes = timeline["changes"].as_array().expect("the changes");
    assert_eq!(changes[0]["status"], 200, "{timeline}");
    assert_eq!(changes[0]["answer"]["result"], "converged", "{timeline}");
    assert!(changes[1].get("error").is_none(), "{timeline}");
    let phase_at = timeline["events"]
        .as_array()
        .expect("the events")
        .iter()
        .find(|event| event["fields"]["phase"] == "trial")
        .and_then(|event| event["at_s"].as_f64())
        .expect("the update's trial among the events");
    assert!(
        changes[1]["issued_s"]
            .as_f64()
            .expect("the second change's issue")
            >= phase_at,
        "{timeline}"
    );

    let summary: Value = serde_json::from_str(&summary).expect("a JSON summary");
    assert_eq!(summary["validation"]["valid"], true, "{summary}");
    let origin = format!("http://127.0.0.1:{}", address.port());
    let observed = &summary["server_metrics"][&origin];
    assert_eq!(observed["available"], true, "{summary}");
    assert_eq!(
        observed["counter_deltas"],
        json!({
            r#"uniserve_requests_total{outcome="length",replica="0"}"#: 3.0,
            "uniserve_time_to_first_token_seconds_count": 2.0,
            "uniserve_time_to_first_token_seconds_sum": 0.5,
        }),
        "{summary}"
    );
    let buckets: Vec<Value> = std::fs::read_to_string(bundle.join("stats.jsonl"))
        .expect("the store's buckets are kept")
        .lines()
        .map(|line| serde_json::from_str(line).expect("a JSON row"))
        .collect();
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0]["origin"], origin.as_str());
    assert_eq!(buckets[0]["name"], "uniserve_requests_total");
}
