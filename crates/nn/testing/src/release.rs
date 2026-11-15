//! Released checkpoints by identifier: a Hugging Face model repository at a pinned commit,
//! resolved through the standard Hugging Face cache.
//!
//! The cache is the one every Hugging Face client shares, configured by the standard
//! environment: `HF_HUB_CACHE`, or `HF_HOME/hub`, or `~/.cache/huggingface/hub`; `HF_ENDPOINT`
//! for a mirror; `HF_TOKEN`, or the token file under `HF_HOME`, for a gated repository; and
//! `HF_HUB_OFFLINE` for a host that reads the cache alone, which its operator populated
//! beforehand (`hf download <repository> --revision <commit>`). Online, the files of the pinned
//! commit the cache lacks are fetched into it first.
//!
//! A release that cannot be resolved fails the test, naming the repository and the commit: a
//! test that reads a release never passes without it.

use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use nn::Checkpoint;

/// A released checkpoint: a Hugging Face model repository at a pinned commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    /// The repository's id, `owner/name`.
    pub repository: &'static str,
    /// The commit the release is pinned at, its full hash: a branch or a tag would let the
    /// tested files change under the test.
    pub revision: &'static str,
}

impl std::fmt::Display for Release {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}@{}", self.repository, self.revision)
    }
}

impl Release {
    /// The release's snapshot directory in the Hugging Face cache: every file of the
    /// repository at the pinned commit. Online, the files the cache lacks are fetched first;
    /// with `HF_HUB_OFFLINE` set the cache alone is read.
    ///
    /// # Panics
    ///
    /// Naming the repository and the commit, when the hub cannot list or serve the commit's
    /// files, or, offline, when the cache holds no snapshot of the commit.
    pub fn snapshot(&self) -> PathBuf {
        let cache = cache();
        let repo = Repo::with_revision(
            self.repository.to_owned(),
            RepoType::Model,
            self.revision.to_owned(),
        );
        let snapshot = cache.repo(repo.clone()).pointer_path(self.revision);

        if offline() {
            assert!(
                snapshot.is_dir(),
                "{self} is not in the Hugging Face cache at {}, and HF_HUB_OFFLINE forbids \
                 fetching it",
                cache.path().display()
            );
            return snapshot;
        }

        let mut builder = ApiBuilder::from_cache(cache).with_token(token());
        if let Some(endpoint) = std::env::var_os("HF_ENDPOINT") {
            builder = builder.with_endpoint(endpoint.to_string_lossy().into_owned());
        }
        let api = builder
            .build()
            .unwrap_or_else(|error| panic!("{self}: a Hugging Face client: {error}"));
        let remote = api.repo(repo);

        // The listing at the pinned commit is the release's file set, and the commit it
        // resolves to is the pin itself.
        let listing = remote
            .info()
            .unwrap_or_else(|error| panic!("{self}: the hub's listing of the commit: {error}"));
        assert_eq!(
            listing.sha, self.revision,
            "{self}: the pin names commit {}, not a commit hash",
            listing.sha
        );

        for sibling in &listing.siblings {
            if !snapshot.join(&sibling.rfilename).exists() {
                remote.download(&sibling.rfilename).unwrap_or_else(|error| {
                    panic!("{self}: fetching {}: {error}", sibling.rfilename)
                });
            }
        }
        snapshot
    }

    /// The checkpoint of the release's snapshot ([`Release::snapshot`]), opened by its
    /// declaration.
    pub fn checkpoint(&self) -> Checkpoint {
        let snapshot = self.snapshot();
        Checkpoint::open(&snapshot).unwrap_or_else(|error| panic!("{self}: {error}"))
    }
}

/// The Hugging Face cache the environment configures: `HF_HUB_CACHE`, or `HF_HOME/hub`, or
/// the user's default.
fn cache() -> Cache {
    match std::env::var_os("HF_HUB_CACHE") {
        Some(path) => Cache::new(PathBuf::from(path)),
        None => Cache::from_env(),
    }
}

/// The token of gated repositories: `HF_TOKEN`, or the token file `hf auth login` writes under
/// `HF_HOME`.
fn token() -> Option<String> {
    match std::env::var("HF_TOKEN") {
        Ok(token) if !token.is_empty() => Some(token),
        _ => Cache::from_env().token(),
    }
}

/// Whether `HF_HUB_OFFLINE` states the hub unreachable, as the Hugging Face clients read it.
fn offline() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| {
        matches!(
            value.to_ascii_uppercase().as_str(),
            "1" | "ON" | "YES" | "TRUE"
        )
    })
}
