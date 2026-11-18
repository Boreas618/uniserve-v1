//! A deployment's admin API as `uniserve-profile` reads it: SQL statements
//! over the observation store's tables through `/admin/query` (`docs/admin.md`).

use anyhow::{Context, Result, anyhow};
use serde::de::DeserializeOwned;

/// One daemon's admin API, reached at the address its readiness record states.
pub struct Admin {
    /// The address as the operator stated it, which names the daemon in errors.
    address: String,
    /// The URL prefix of the routes: a placeholder host for a Unix socket,
    /// which the client connects by path, or the address itself.
    url: String,
    http: reqwest::Client,
}

impl Admin {
    /// The daemon at `address`: `unix:<path>` for its admin socket, else a base
    /// URL.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(address: &str) -> Result<Self> {
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
        Ok(Self {
            address: address.to_owned(),
            url,
            http: http.build().context("building the HTTP client")?,
        })
    }

    /// The rows `sql` selects, each read as `T` from the JSON object the
    /// daemon renders it as (a map column is an object of strings).
    ///
    /// The statement travels as the form body of `POST /admin/query`, so its
    /// length is bounded by the daemon's request body limit, not a URL's.
    /// The call blocks the calling thread until the whole answer is read.
    ///
    /// # Errors
    ///
    /// When the daemon cannot be reached, refuses the statement (its error
    /// message is in the error), or answers with rows that are not `T`.
    pub fn query<T: DeserializeOwned>(&self, sql: &str) -> Result<Vec<T>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("starting the HTTP client's runtime")?;

        let (status, body) = runtime
            .block_on(async {
                let response = self
                    .http
                    .post(format!("{}/admin/query", self.url))
                    .form(&[("q", sql)])
                    .send()
                    .await?;
                let status = response.status();
                Ok::<_, reqwest::Error>((status, response.bytes().await?))
            })
            .with_context(|| format!("reaching {}", self.address))?;

        if !status.is_success() {
            // A refusal is `{"error": {"message": ...}}`; anything else is
            // shown as the daemon sent it.
            let message = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|refusal| refusal["error"]["message"].as_str().map(str::to_owned))
                .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
            return Err(anyhow!(
                "{}/admin/query responded {status}: {message}",
                self.address
            ));
        }

        serde_json::from_slice(&body)
            .with_context(|| format!("reading the rows of {}/admin/query", self.address))
    }
}
