//! Loopback stand-in for the Hugging Face Hub endpoints a [`Hub`] reaches.
//!
//! The stub serves a revision's listing (`/api/models/{repo}/revision/{revision}`) and whole or
//! byte-range file downloads (`/{repo}/resolve/{revision}/{file}`) of one commit's files,
//! answering an unpublished file with 404 as the Hub does, or answers either kind of request with
//! a fixed failure status. Tests drive the real Hub client against it, so only the network
//! boundary is replaced; the cache layout the tests seed is computed through `hf_hub`'s own cache
//! API.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::Router;
use axum::http::{HeaderMap, HeaderName, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use hf_hub::{Cache, Repo, RepoType};

use crate::api::model::assets::hub::{Hub, HubSettings};

/// The commit the stub reports for every revision unless told otherwise.
pub(super) const COMMIT: &str = "5f3b0c1e9d2a4b6c8e0f1a3b5c7d9e1f2a4b6c8d";

/// Repository contents and failure modes served by the stub.
#[derive(Clone, Default)]
pub(super) struct HubStub {
    /// Published file contents by repository-relative path.
    pub(super) files: BTreeMap<String, Vec<u8>>,
    /// The files the listing names in place of `files`' paths, for a test whose cache holds a
    /// listed file the stub does not serve.
    pub(super) listed: Option<Vec<String>>,
    /// Status answered for the listing in place of the listing.
    pub(super) listing_status: Option<StatusCode>,
    /// Status answered for every file download in place of its content.
    pub(super) download_status: Option<StatusCode>,
    /// Commit reported for every revision in place of [`COMMIT`].
    pub(super) commit: Option<&'static str>,
}

impl HubStub {
    /// Publishes `files` as `(path, content)` pairs; contents must be non-empty.
    pub(super) fn with_files<C: AsRef<[u8]>>(files: &[(&str, C)]) -> Self {
        Self {
            files: files
                .iter()
                .map(|(name, content)| ((*name).to_owned(), content.as_ref().to_vec()))
                .collect(),
            ..Self::default()
        }
    }

    /// Lists `names` as the revision's files in place of the published files' paths.
    pub(super) fn listing(self, names: &[&str]) -> Self {
        Self {
            listed: Some(names.iter().map(|&name| name.to_owned()).collect()),
            ..self
        }
    }

    /// Serves the stub on a loopback port for the rest of the test and returns a Hub that reaches
    /// it, with its cache under `cache_root` (the directory [`seed_snapshot`] writes).
    pub(super) async fn serve(self, cache_root: &Path) -> Hub {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new().fallback(move |uri: Uri, headers: HeaderMap| {
            let stub = self.clone();
            async move { stub.respond(&uri, &headers) }
        });
        tokio::spawn(async move { axum::serve(listener, router).await });

        // A cache under the test directory also keeps the client from reading a token file
        // outside it.
        Hub::new(HubSettings {
            cache: cache_root.join("hub"),
            endpoint: Some(endpoint),
            token: None,
            offline: false,
        })
        .unwrap()
    }

    /// The commit the stub reports.
    fn commit(&self) -> &'static str {
        self.commit.unwrap_or(COMMIT)
    }

    /// Answers one request by its path and `Range` header.
    fn respond(&self, uri: &Uri, headers: &HeaderMap) -> Response {
        let path = uri.path();
        if path.starts_with("/api/models/") {
            if let Some(status) = self.listing_status {
                return status.into_response();
            }
            let names = match &self.listed {
                Some(names) => names.clone(),
                None => self.files.keys().cloned().collect(),
            };
            let siblings = names
                .iter()
                .map(|name| serde_json::json!({ "rfilename": name }))
                .collect::<Vec<_>>();
            return axum::Json(serde_json::json!({ "siblings": siblings, "sha": self.commit() }))
                .into_response();
        }

        let Some((_, filename)) = path
            .split_once("/resolve/")
            .and_then(|(_, rest)| rest.split_once('/'))
        else {
            return StatusCode::NOT_FOUND.into_response();
        };
        if let Some(status) = self.download_status {
            return status.into_response();
        }
        let Some(content) = self.files.get(filename) else {
            return StatusCode::NOT_FOUND.into_response();
        };

        // A `bytes=start-end` range is answered as partial content with the end clamped to the
        // file, as HTTP servers do; the Hub client probes metadata with `bytes=0-0` and requests
        // past the end when it downloads. A request without a range receives the whole file.
        let last = content.len() - 1;
        let (status, first, end) = match headers
            .get(header::RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("bytes="))
            .and_then(|value| value.split_once('-'))
            .and_then(|(start, end)| {
                Some((start.parse::<usize>().ok()?, end.parse::<usize>().ok()?))
            }) {
            Some((start, end)) => (StatusCode::PARTIAL_CONTENT, start, end.min(last)),
            None => (StatusCode::OK, 0, last),
        };

        // The client sizes the download from Content-Range, names the cached blob after the ETag
        // and files it under the reported commit.
        (
            status,
            [
                (header::ETAG, format!("\"{}\"", filename.replace('/', "-"))),
                (
                    header::CONTENT_RANGE,
                    format!("bytes {first}-{end}/{}", content.len()),
                ),
                (
                    HeaderName::from_static("x-repo-commit"),
                    self.commit().to_owned(),
                ),
            ],
            content[first..=end].to_vec(),
        )
            .into_response()
    }
}

/// Writes `files` into the Hub cache under `cache_root` as the snapshot of `repo_id` at
/// `commit`, with no ref naming it, the layout `huggingface_hub` leaves when it downloads at a
/// commit. Returns the snapshot's directory.
pub(super) fn seed_snapshot(
    cache_root: &Path,
    repo_id: &str,
    commit: &str,
    files: &[(&str, &str)],
) -> PathBuf {
    let snapshot = Cache::new(cache_root.join("hub"))
        .repo(Repo::with_revision(
            repo_id.to_owned(),
            RepoType::Model,
            commit.to_owned(),
        ))
        .pointer_path(commit);
    for (name, content) in files {
        let path = snapshot.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
    snapshot
}
