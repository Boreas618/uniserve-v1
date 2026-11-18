//! One node's admin API, as the manager calls it: the deployment's policy package
//! (`/admin/policy`) and SQL over its observation store (`/admin/query`), whose `events` table
//! the manager also follows as a server-sent event stream.

use std::time::Duration;

use anyhow::{Context as _, Result};
use serde_json::Value;

/// Why a request to a node failed: the node was not reached, or it answered with an error.
#[derive(Debug)]
pub(crate) enum Failure {
    Unreached(anyhow::Error),
    Answered(anyhow::Error),
}

impl From<Failure> for anyhow::Error {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Unreached(error) | Failure::Answered(error) => error,
        }
    }
}

/// One node's admin API.
#[derive(Clone)]
pub(crate) struct Node {
    /// The node's address as the operator stated it, which names it in every
    /// line the manager prints.
    pub(crate) base: String,
    /// The URL prefix of its routes: the address itself, or a placeholder
    /// host for a Unix socket, which the client connects by path.
    url: String,
    http: reqwest::Client,
}

impl Node {
    /// The node at `address`: `unix:<path>` for its admin socket, else a base
    /// URL.
    pub(crate) fn new(address: &str) -> Result<Self> {
        let (http, url) = match address.strip_prefix("unix:") {
            Some(path) => (
                reqwest::Client::builder().unix_socket(path),
                "http://admin".to_owned(),
            ),
            None => (
                reqwest::Client::builder(),
                address.trim_end_matches('/').to_owned(),
            ),
        };
        // Every request opens a connection of its own: a hot restart's successor binds over the
        // node's socket while its predecessor finishes the connections it accepted, so a pooled
        // connection would reach the daemon that no longer serves the node.
        let http = http.pool_max_idle_per_host(0);
        Ok(Self {
            base: address.trim_end_matches('/').to_owned(),
            url,
            http: http.build().context("building the HTTP client")?,
        })
    }

    /// Sends `request` to `path` and reads its successful response's body.
    async fn exchange(
        &self,
        request: reqwest::RequestBuilder,
        path: &str,
    ) -> Result<bytes::Bytes, Failure> {
        let unreached = |error: reqwest::Error| {
            Failure::Unreached(anyhow::Error::new(error).context(format!("reaching {}", self.base)))
        };
        let response = request.send().await.map_err(unreached)?;
        let status = response.status();
        let body = response.bytes().await.map_err(unreached)?;
        if !status.is_success() {
            return Err(Failure::Answered(anyhow::anyhow!(
                "{}{path} responded {status}: {}",
                self.base,
                String::from_utf8_lossy(&body)
            )));
        }
        Ok(body)
    }

    /// The rows of the SQL statement `sql` over the node's observation store, each a JSON
    /// object of its columns (`POST /admin/query`).
    pub(crate) async fn query(&self, sql: &str) -> Result<Vec<Value>> {
        let request = self
            .http
            .post(format!("{}/admin/query", self.url))
            .form(&[("q", sql)]);
        let body = self.exchange(request, "/admin/query").await?;
        serde_json::from_slice(&body)
            .with_context(|| format!("reading {}/admin/query's rows", self.base))
    }

    /// `PUT /admin/policy` of the package `archive` (a tar archive of its directory), forcing a
    /// target a rollback marked unavailable when `force`: the revision's record as the node
    /// answers it once the change converged, or why it did not.
    pub(crate) async fn put_policy(&self, archive: Vec<u8>, force: bool) -> Result<Value> {
        let path = if force {
            "/admin/policy?force=true"
        } else {
            "/admin/policy"
        };
        let request = self
            .http
            .put(format!("{}{path}", self.url))
            .header(reqwest::header::CONTENT_TYPE, "application/x-tar")
            .body(archive);
        let body = self.exchange(request, path).await?;
        serde_json::from_slice(&body).with_context(|| format!("reading {}{path}", self.base))
    }

    /// The node's clock, in nanoseconds since the Unix epoch: where a statement over the rows
    /// appended from now on starts.
    pub(crate) async fn now(&self) -> Result<i64> {
        let rows = self.query("SELECT CAST(now() AS BIGINT) AS now").await?;
        rows.first()
            .and_then(|row| row["now"].as_i64())
            .with_context(|| format!("{} answered no clock", self.base))
    }

    /// Follows the rows of the SQL statement `sql` over every `events` row the node keeps and
    /// appends, as its server-sent event stream delivers them: each row is handed to `row` as it
    /// arrives. A stream that ends or a node that cannot be reached is asked again every
    /// `retry`, each failure printed, resuming after the last message's id (`Last-Event-ID`), so
    /// no row is read twice.
    ///
    /// # Errors
    ///
    /// Fails when the node refuses the statement.
    pub(crate) async fn follow(
        &self,
        sql: &str,
        retry: Duration,
        mut row: impl FnMut(&Value),
    ) -> Result<()> {
        let mut last: Option<u64> = None;
        loop {
            let mut request = self
                .http
                .post(format!("{}/admin/query", self.url))
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .form(&[("q", sql)]);
            if let Some(id) = last {
                request = request.header("last-event-id", id.to_string());
            }
            match self.stream(request, &mut last, &mut row).await {
                Ok(()) => println!("{}: the event stream ended; following again", self.base),
                Err(Failure::Unreached(error)) => {
                    println!("{}: {error:#}; asking again in {retry:?}", self.base);
                    tokio::time::sleep(retry).await;
                }
                Err(Failure::Answered(error)) => return Err(error),
            }
        }
    }

    /// Reads one server-sent event stream to its end: every `data:` line's rows go to `row`, and
    /// every `id:` line is the position a reconnection resumes after.
    async fn stream(
        &self,
        request: reqwest::RequestBuilder,
        last: &mut Option<u64>,
        row: &mut impl FnMut(&Value),
    ) -> Result<(), Failure> {
        let unreached = |error: reqwest::Error| {
            Failure::Unreached(anyhow::Error::new(error).context(format!("reaching {}", self.base)))
        };
        let mut response = request.send().await.map_err(unreached)?;
        let status = response.status();
        if !status.is_success() {
            let body = response.bytes().await.map_err(unreached)?;
            return Err(Failure::Answered(anyhow::anyhow!(
                "{}/admin/query responded {status}: {}",
                self.base,
                String::from_utf8_lossy(&body)
            )));
        }
        let mut pending = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(unreached)? {
            pending.extend_from_slice(&chunk);
            while let Some(end) = pending.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = pending.drain(..=end).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end();
                if let Some(id) = line.strip_prefix("id:") {
                    *last = id.trim().parse().ok().or(*last);
                } else if let Some(data) = line.strip_prefix("data:") {
                    let rows: Vec<Value> = serde_json::from_str(data.trim()).map_err(|error| {
                        Failure::Answered(
                            anyhow::Error::new(error)
                                .context(format!("reading {}'s event stream", self.base)),
                        )
                    })?;
                    rows.iter().for_each(&mut *row);
                }
            }
        }
        Ok(())
    }
}
