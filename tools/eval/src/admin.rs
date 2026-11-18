//! A deployment's admin API, an HTTP/1.1 server on a Unix socket: the harness
//! writes through `PUT /admin/policy` alone and observes through
//! `/admin/query` alone (`docs/admin.md`).
//!
//! The observation store has two tables, `stats` (one metric value per time
//! bucket) and `events` (one row per thing that happened); a statement is
//! DataFusion SQL over them. Its rows arrive as a JSON array of objects, a map
//! column rendered as an object. A subscription streams a statement over every
//! `events` row appended to the store as server-sent events, each message the
//! statement's rows over the events appended since the previous one.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use serde_json::Value;

/// The route of the deployment's policy package.
const POLICY: &str = "/admin/policy";
/// The route of the observation store's queries.
const QUERY: &str = "/admin/query";

/// What answered a request: its status and its whole body.
struct Answer {
    status: u16,
    body: Bytes,
}

impl Answer {
    /// The body as text, invalid UTF-8 replaced.
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Opens an HTTP/1.1 connection on `socket` and sends one request with the
/// given headers and body; the response's body is left to the caller.
async fn send(
    socket: &Path,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> Result<hyper::Response<hyper::body::Incoming>> {
    let stream = tokio::net::UnixStream::connect(socket)
        .await
        .with_context(|| format!("connecting to {}", socket.display()))?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    // The connection's driver ends with the connection, once the body is read
    // or dropped.
    tokio::spawn(connection);
    let mut builder = hyper::Request::builder()
        .method(method)
        .uri(path)
        .header(hyper::header::HOST, "admin");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    Ok(sender
        .send_request(builder.body(Full::new(Bytes::from(body)))?)
        .await?)
}

/// One request whose whole body is read.
async fn exchange(
    socket: &Path,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> Result<Answer> {
    let response = send(socket, method, path, headers, body).await?;
    let status = response.status().as_u16();
    let body = response.into_body().collect().await?.to_bytes();
    Ok(Answer { status, body })
}

/// The message of an admin error body (`{"error": {"message": ...}}`), else
/// the body as it reads.
fn refusal(answer: &Answer) -> String {
    serde_json::from_slice::<Value>(&answer.body)
        .ok()
        .and_then(|body| body["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| answer.text())
}

/// The rows of one SQL statement over the deployment's observation store,
/// each a JSON object by column name (`POST /admin/query`, the statement
/// form-encoded as `q`).
///
/// # Errors
///
/// When the socket does not answer, the statement is refused (with the
/// store's reason) or the answer is no JSON array.
pub async fn query(socket: &Path, sql: &str) -> Result<Vec<Value>> {
    let form = serde_urlencoded::to_string([("q", sql)])?;
    let answer = exchange(
        socket,
        "POST",
        QUERY,
        &[(
            hyper::header::CONTENT_TYPE.as_str(),
            "application/x-www-form-urlencoded",
        )],
        form.into_bytes(),
    )
    .await?;
    ensure!(
        answer.status == 200,
        "the query was answered HTTP {}: {}",
        answer.status,
        refusal(&answer)
    );
    match serde_json::from_slice(&answer.body).context("the query's rows")? {
        Value::Array(rows) => Ok(rows),
        other => bail!("the query answered {other}, not an array of rows"),
    }
}

/// The tar archive of the policy package in `directory`, as `PUT
/// /admin/policy` carries it: its `config.toml`, then every file under its
/// `programs/`, by its path in the package. Nothing else of the directory is
/// part of the package.
///
/// # Errors
///
/// When the directory holds no `config.toml` or a file cannot be read.
pub fn archive(directory: &Path) -> Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    let config = directory.join("config.toml");
    ensure!(
        config.is_file(),
        "{} holds no config.toml: it is no policy package",
        directory.display()
    );
    builder
        .append_path_with_name(&config, "config.toml")
        .with_context(|| format!("archiving {}", config.display()))?;
    let programs = directory.join("programs");
    if programs.is_dir() {
        builder
            .append_dir_all("programs", &programs)
            .with_context(|| format!("archiving {}", programs.display()))?;
    }
    builder
        .into_inner()
        .context("finishing the package archive")
}

/// What `PUT /admin/policy` answered: its status and its body, the revision's
/// record when the change converged (200) or its successor converges the rest
/// (202), else the reason it was refused or rolled back.
#[derive(Debug, Clone)]
pub struct Applied {
    pub status: u16,
    pub answer: Value,
}

/// Applies the policy package in `directory` to the deployment (`PUT
/// /admin/policy`, with `?force=true` when `force`), answered once the change
/// ended.
///
/// # Errors
///
/// When the package does not archive or the socket does not answer; a refused
/// change is an answer.
pub async fn apply(socket: &Path, directory: &Path, force: bool) -> Result<Applied> {
    let body = archive(directory)?;
    let path = if force {
        format!("{POLICY}?force=true")
    } else {
        POLICY.to_owned()
    };
    let answer = exchange(
        socket,
        "PUT",
        &path,
        &[(hyper::header::CONTENT_TYPE.as_str(), "application/x-tar")],
        body,
    )
    .await?;
    let text = answer.text();
    Ok(Applied {
        status: answer.status,
        answer: serde_json::from_str(&text).unwrap_or(Value::String(text)),
    })
}

/// One message of a subscription: the id of the last event it covers and the
/// statement's rows over the events it covers.
#[derive(Debug, Clone)]
pub struct Message {
    pub id: u64,
    pub rows: Vec<Value>,
}

/// A statement over every `events` row appended to the store, as a stream of
/// server-sent events (`GET /admin/query` with `Accept: text/event-stream`).
pub struct Subscription {
    body: hyper::body::Incoming,
    lines: crate::transport::sse::Lines,
    /// The fields of the message being read: its `id:` and its `data:` lines.
    id: Option<u64>,
    data: Vec<String>,
    /// Messages a chunk completed that the caller has not taken yet.
    completed: std::collections::VecDeque<Message>,
}

impl Subscription {
    /// Opens the stream after event `after` (the id of the last message read
    /// before a reconnection), or from the oldest event the store keeps.
    ///
    /// # Errors
    ///
    /// When the socket does not answer or the statement is refused.
    pub async fn open(socket: &Path, sql: &str, after: Option<u64>) -> Result<Self> {
        let path = format!("{QUERY}?{}", serde_urlencoded::to_string([("q", sql)])?);
        let resume = after.map(|after| after.to_string());
        let mut headers = vec![(hyper::header::ACCEPT.as_str(), "text/event-stream")];
        if let Some(resume) = &resume {
            headers.push(("last-event-id", resume.as_str()));
        }
        let response = send(socket, "GET", &path, &headers, Vec::new()).await?;
        let status = response.status().as_u16();
        if status != 200 {
            let body = response.into_body().collect().await?.to_bytes();
            let answer = Answer { status, body };
            bail!(
                "the subscription was answered HTTP {status}: {}",
                refusal(&answer)
            );
        }
        Ok(Self {
            body: response.into_body(),
            lines: crate::transport::sse::Lines::default(),
            id: None,
            data: Vec::new(),
            completed: std::collections::VecDeque::new(),
        })
    }

    /// The next message; `None` once the stream ended.
    ///
    /// # Errors
    ///
    /// When the stream breaks or a message's data is no array of rows.
    pub async fn next(&mut self) -> Result<Option<Message>> {
        loop {
            if let Some(message) = self.completed.pop_front() {
                return Ok(Some(message));
            }
            let Some(frame) = self.body.frame().await else {
                return Ok(None);
            };
            let Ok(chunk) = frame?.into_data() else {
                continue;
            };
            for line in self.lines.push(&chunk) {
                if let Some(message) = self.feed(&line)? {
                    self.completed.push_back(message);
                }
            }
        }
    }

    /// Consumes one line; a blank line completes the message whose fields
    /// preceded it.
    fn feed(&mut self, line: &str) -> Result<Option<Message>> {
        if line.is_empty() {
            let id = self.id.take();
            let data = std::mem::take(&mut self.data);
            if data.is_empty() {
                return Ok(None);
            }
            let rows = match serde_json::from_str(&data.join("\n"))
                .context("a subscription message's rows")?
            {
                Value::Array(rows) => rows,
                other => bail!("a subscription message carried {other}, not an array of rows"),
            };
            let id = id.context("a subscription message without its id")?;
            return Ok(Some(Message { id, rows }));
        }
        if let Some(id) = line.strip_prefix("id:") {
            self.id = Some(id.trim().parse().context("a subscription message's id")?);
        } else if let Some(data) = line.strip_prefix("data:") {
            self.data
                .push(data.strip_prefix(' ').unwrap_or(data).to_owned());
        }
        Ok(None)
    }
}
