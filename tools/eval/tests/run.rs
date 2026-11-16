//! `uniserve-eval run` over its public command line, against a stand-in of the
//! OpenAI chat completions API served in this process.
//!
//! The evaluator starts each server a profile names and waits for its address to
//! answer `ready_path`; the stand-in answers on the address, and the profile's
//! command is a process that only stays alive until the evaluator stops it.

#![cfg(test)]

use std::net::SocketAddr;
use std::path::Path;
use std::process::Command;

use axum::Router;
use axum::body::Body;
use axum::http::header;
use axum::response::Response;
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
