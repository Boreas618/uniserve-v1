//! One node's admin API, as the manager calls it: typed requests and responses of the interface
//! crate's schema (`proto::admin`), and the node's event log read by long poll.

use std::time::Duration;

use anyhow::{Context as _, Result};
use proto::admin::{EventsPage, EventsQuery, Status};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// How the manager reads a node's event log.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Polling {
    /// How long one request of the event log waits for an event before the manager asks
    /// again; the node waits at most its `admin_config.event_log.longest_wait`.
    pub(crate) wait: Duration,
    /// How long the manager waits before it asks again a node it could not reach.
    pub(crate) retry: Duration,
}

/// Why a request to a node failed: the node was not reached, or it answered with an error or
/// with what does not read as the route's response.
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

    /// Sends `request` to `path` and reads its successful response as `T`.
    async fn exchange<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
        path: &str,
    ) -> Result<T, Failure> {
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
        serde_json::from_slice(&body).map_err(|error| {
            Failure::Answered(
                anyhow::Error::new(error).context(format!("reading {}{path}", self.base)),
            )
        })
    }

    /// `GET path`, read as `T`.
    pub(crate) async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let request = self.http.get(format!("{}{path}", self.url));
        Ok(self.exchange(request, path).await?)
    }

    /// `method path` with the JSON body `body`, its response read as `T`.
    pub(crate) async fn send<B: Serialize, T: DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let request = self
            .http
            .request(method, format!("{}{path}", self.url))
            .json(body);
        Ok(self.exchange(request, path).await?)
    }

    /// `GET /admin/status`: what the node serves.
    pub(crate) async fn status(&self) -> Result<Status> {
        self.get("/admin/status").await
    }

    /// `PUT /admin/policy` of the package `archive` (a tar archive of its directory), forcing a
    /// target a rollback marked unavailable when `force`: the revision's record as the node
    /// answers it once the change converged, or why it did not.
    pub(crate) async fn put_policy(
        &self,
        archive: Vec<u8>,
        force: bool,
    ) -> Result<serde_json::Value> {
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
        Ok(self.exchange(request, path).await?)
    }

    /// The sequence number of the node's last event: a reader from it reads what happens from
    /// now on.
    pub(crate) async fn last_event(&self) -> Result<u64> {
        Ok(self.status().await?.next_event.saturating_sub(1))
    }

    /// The node's events after `after`, waiting for one as `polling` states. A node the manager
    /// cannot reach is asked again every `polling.retry`, each failure printed, until it
    /// answers.
    pub(crate) async fn events(&self, after: u64, polling: Polling) -> Result<EventsPage> {
        let query = EventsQuery {
            after,
            wait: polling.wait.as_secs_f64(),
        };
        loop {
            let request = self
                .http
                .get(format!("{}/admin/events", self.url))
                .query(&query);
            match self.exchange(request, "/admin/events").await {
                Ok(page) => return Ok(page),
                Err(Failure::Unreached(error)) => {
                    println!(
                        "{}: {error:#}; asking again in {:?}",
                        self.base, polling.retry
                    );
                    tokio::time::sleep(polling.retry).await;
                }
                Err(Failure::Answered(error)) => return Err(error),
            }
        }
    }
}
