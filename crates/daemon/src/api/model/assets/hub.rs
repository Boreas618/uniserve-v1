//! The Hugging Face Hub as this process reaches it, and the snapshots of its repositories.
//!
//! A checkpoint on the Hub is a model repository at one revision ([`HubAsset`]). [`Hub`] resolves
//! it as `huggingface_hub`'s `snapshot_download` does: the revision names one commit, whose files
//! lie in the standard cache's snapshot of the commit ([`Snapshot`]), downloaded into it when the
//! cache lacks them. Which of the commit's files are downloaded is the reader's choice
//! ([`Snapshot::fetch`]): a model's every file, or a diffusers pipeline's root files and the
//! folders of the stages it declares, as `diffusers` downloads a pipeline. A process reaches the
//! Hub through one [`Hub`], configured from the standard Hugging Face environment
//! ([`HubSettings::from_env`]): an offline host pre-populates the cache and sets
//! `HF_HUB_OFFLINE`, and a mirror is named by `HF_ENDPOINT`.

use std::fmt;
use std::path::{Path, PathBuf};

use hf_hub::api::tokio::{Api, ApiBuilder, ApiError};
use hf_hub::{Cache, Repo, RepoType};
use thiserror_ext::AsReport as _;

use crate::api::model::assets::error::{Error, Result};

/// A checkpoint in a Hugging Face model repository at one revision (a branch, a tag or a commit):
/// the repository's root, or the folder of it that holds the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubAsset {
    /// The repository, `<owner>/<name>`.
    repository: String,
    /// The revision the repository is read at.
    revision: String,
    /// The repository-relative folder that holds the checkpoint; `None` for the repository's
    /// root.
    subfolder: Option<String>,
}

impl HubAsset {
    /// Parses `<owner>/<name>[@<revision>][/<folder>]`, the form `huggingface_hub`'s file system
    /// names a path of a repository in, as a deployment states a checkpoint: without a revision
    /// the repository is read at the Hub's default branch, and without a folder the checkpoint
    /// is the repository's root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when the repository is not `<owner>/<name>` in the characters
    /// the Hub admits in a repository name, the revision after `@` is empty, or the folder has an
    /// empty, `.` or `..` component.
    pub fn parse(text: &str) -> Result<Self> {
        Self::parsed(text, false)
    }

    /// Parses `hf://<owner>/<name>@<revision>[/<folder>]`, the form one checkpoint pins another
    /// at, which always names its revision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `text` is not of that form.
    pub fn parse_pin(text: &str) -> Result<Self> {
        let pinned = text.strip_prefix("hf://").ok_or_else(|| {
            Error::invalid(format!("{text:?} is not hf://<owner>/<name>@<revision>"))
        })?;
        Self::parsed(pinned, true)
    }

    /// The asset `text` names; `pinned` requires it to name its revision.
    fn parsed(text: &str, pinned: bool) -> Result<Self> {
        let refused = || {
            Error::invalid(format!(
                "{text:?} names no Hugging Face checkpoint: state <owner>/<name>, with @<revision> \
                 for a branch, tag or commit other than the default branch, and /<folder> for a \
                 checkpoint in a folder of the repository"
            ))
        };
        // The Hub names a repository `<owner>/<name>`, each part of ASCII letters, digits, `-`,
        // `_` and `.`.
        let named = |part: &str| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
        };
        let (owner, rest) = text.split_once('/').ok_or_else(refused)?;
        let (name, subfolder) = match rest.split_once('/') {
            Some((name, subfolder)) => (name, Some(subfolder)),
            None => (rest, None),
        };
        let (name, revision) = match name.split_once('@') {
            Some((name, revision)) => (name, Some(revision)),
            None => (name, None),
        };
        let folder = |folder: &str| {
            folder
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != "..")
        };
        if !named(owner)
            || !named(name)
            || revision.is_some_and(str::is_empty)
            || (pinned && revision.is_none())
            || subfolder.is_some_and(|subfolder| !folder(subfolder))
        {
            return Err(refused());
        }
        let repository = format!("{owner}/{name}");
        Ok(Self {
            revision: match revision {
                Some(revision) => revision.to_owned(),
                None => Repo::model(repository.clone()).revision().to_owned(),
            },
            repository,
            subfolder: subfolder.map(str::to_owned),
        })
    }

    /// The repository at this asset's revision, as `hf_hub` names it.
    fn repo(&self) -> Repo {
        self.at(&self.revision)
    }

    /// The repository at `revision`, as `hf_hub` names it.
    fn at(&self, revision: &str) -> Repo {
        Repo::with_revision(
            self.repository.clone(),
            RepoType::Model,
            revision.to_owned(),
        )
    }

    /// The asset at `commit`, as it is stated: `<owner>/<name>@<commit>[/<folder>]`.
    fn at_commit(&self, commit: &str) -> String {
        Self {
            revision: commit.to_owned(),
            ..self.clone()
        }
        .to_string()
    }

    /// The checkpoint's directory in the snapshot directory `snapshot` of its commit.
    fn within(&self, snapshot: PathBuf) -> PathBuf {
        match &self.subfolder {
            Some(subfolder) => snapshot.join(subfolder),
            None => snapshot,
        }
    }
}

impl fmt::Display for HubAsset {
    /// Formats the asset as it is stated: `<owner>/<name>@<revision>[/<folder>]`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}@{}", self.repository, self.revision)?;
        if let Some(subfolder) = &self.subfolder {
            write!(formatter, "/{subfolder}")?;
        }
        Ok(())
    }
}

/// What the Hub is reached with: the standard Hugging Face environment's settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubSettings {
    /// The cache snapshots are read from and downloaded into: `$HF_HOME/hub`, or
    /// `~/.cache/huggingface/hub` when `HF_HOME` is unset.
    pub cache: PathBuf,
    /// The Hub's address (`HF_ENDPOINT`), for a mirror; `None` reaches the Hub itself.
    pub endpoint: Option<String>,
    /// The access token (`HF_TOKEN`); `None` uses the token `hf auth login` wrote into the
    /// cache's directory, when there is one.
    pub token: Option<String>,
    /// Whether no request may reach the Hub (`HF_HUB_OFFLINE`): every snapshot is read from the
    /// cache, which the host has populated.
    pub offline: bool,
}

impl HubSettings {
    /// The settings the process environment states, read once for the process's [`Hub`]: the
    /// cache by `hf_hub`'s own rule, and `HF_ENDPOINT`, `HF_TOKEN` and `HF_HUB_OFFLINE` by
    /// `huggingface_hub`'s, which reads an empty variable as unset.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `HF_HUB_OFFLINE` holds a value that is neither true
    /// (`1`, `ON`, `YES`, `TRUE`) nor false (`0`, `OFF`, `NO`, `FALSE`), in any case.
    pub fn from_env() -> Result<Self> {
        let variable = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let offline = match variable("HF_HUB_OFFLINE")
            .map(|value| value.to_ascii_uppercase())
            .as_deref()
        {
            None | Some("0" | "OFF" | "NO" | "FALSE") => false,
            Some("1" | "ON" | "YES" | "TRUE") => true,
            Some(other) => {
                return Err(Error::invalid(format!(
                    "HF_HUB_OFFLINE is {other:?}: state 1 to read only the cache, or 0"
                )));
            }
        };
        Ok(Self {
            cache: Cache::from_env().path().clone(),
            endpoint: variable("HF_ENDPOINT"),
            token: variable("HF_TOKEN"),
            offline,
        })
    }
}

/// The Hub as this process reaches it: the client, the cache it reads and fills, and whether it
/// may reach the network.
///
/// The client and the cache are built from one [`HubSettings`], so the cache a snapshot is read
/// from is always the one the client downloads into.
pub struct Hub {
    /// The client of the Hub's API and file downloads, without progress output.
    api: Api,
    /// The cache `api` fills.
    cache: Cache,
    /// Whether every snapshot is read from `cache` alone.
    offline: bool,
}

impl Hub {
    /// The Hub `settings` describe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when the client cannot be built, such as for a token that is
    /// not a valid header value.
    pub fn new(settings: HubSettings) -> Result<Self> {
        let cache = Cache::new(settings.cache);
        let mut builder = ApiBuilder::from_cache(cache.clone()).with_progress(false);
        if let Some(endpoint) = settings.endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(token) = settings.token {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build().map_err(|error| {
            Error::invalid(format!(
                "the Hugging Face Hub client cannot be built: {}",
                error.as_report()
            ))
        })?;
        Ok(Self {
            api,
            cache,
            offline: settings.offline,
        })
    }

    /// The Hub the process environment describes ([`HubSettings::from_env`]).
    ///
    /// # Errors
    ///
    /// As [`HubSettings::from_env`] and [`Hub::new`].
    pub fn from_env() -> Result<Self> {
        Self::new(HubSettings::from_env()?)
    }

    /// The cache's snapshot of `asset`: the directory of the commit its revision names, which
    /// [`Snapshot::fetch`] fills with the files its reader reads.
    ///
    /// Online, the Hub's listing of the revision names its commit and files, and the revision is
    /// recorded in the cache as naming that commit, so an offline host reads the same snapshot.
    /// Offline, the revision is the commit when it is one (40 lowercase hexadecimal digits, as
    /// the Hub writes a commit) and otherwise the commit the cache recorded for it, and the cache
    /// must hold that commit's snapshot, which the host populated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Remote`] when the listing fails, including for a repository or revision
    /// the Hub does not serve, and [`Error::NotCached`] when an offline cache holds no snapshot
    /// of the revision.
    pub async fn snapshot(&self, asset: &HubAsset) -> Result<Snapshot> {
        if self.offline {
            return self.cached_snapshot(asset);
        }

        let listing = self
            .api
            .repo(asset.repo())
            .info()
            .await
            .map_err(|error| Error::Remote {
                model: asset.to_string(),
                message: error.to_report_string(),
            })?;
        let commit = listing.sha;
        let pinned = asset.at(&commit);
        self.cache
            .repo(asset.repo())
            .create_ref(&commit)
            .map_err(|source| Error::Io {
                path: self.cache.path().clone(),
                source,
            })?;
        // The checkpoint's files are the commit's files in its folder, named relative to it.
        let folder = asset
            .subfolder
            .as_ref()
            .map(|subfolder| format!("{subfolder}/"));
        let files = listing
            .siblings
            .into_iter()
            .filter_map(|file| match &folder {
                Some(folder) => file
                    .rfilename
                    .strip_prefix(folder.as_str())
                    .map(str::to_owned),
                None => Some(file.rfilename),
            })
            .collect();
        Ok(Snapshot {
            directory: asset.within(self.cache.repo(pinned.clone()).pointer_path(&commit)),
            listed: Some(Listed {
                api: self.api.clone(),
                asset: asset.at_commit(&commit),
                repo: pinned,
                folder,
                files,
            }),
        })
    }

    /// The cache's snapshot of `asset`, without a request; see [`Hub::snapshot`].
    fn cached_snapshot(&self, asset: &HubAsset) -> Result<Snapshot> {
        let commit = if is_commit(&asset.revision) {
            asset.revision.clone()
        } else {
            // The cache records the commit a revision named in `refs/<revision>` beside the
            // repository's snapshots.
            let reference = self
                .cache
                .path()
                .join(asset.repo().folder_name())
                .join("refs")
                .join(&asset.revision);
            match std::fs::read_to_string(&reference) {
                Ok(commit) => commit.trim().to_owned(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Err(Error::NotCached {
                        asset: asset.to_string(),
                        path: reference,
                    });
                }
                Err(source) => {
                    return Err(Error::Io {
                        path: reference,
                        source,
                    });
                }
            }
        };
        let directory = asset.within(self.cache.repo(asset.at(&commit)).pointer_path(&commit));
        if !directory.is_dir() {
            return Err(Error::NotCached {
                asset: asset.to_string(),
                path: directory,
            });
        }
        Ok(Snapshot {
            directory,
            listed: None,
        })
    }
}

/// A checkpoint's local directory: a stated directory as it is, or the cache's snapshot of a Hub
/// revision with the files the revision lists, which [`Snapshot::fetch`] downloads into it.
pub struct Snapshot {
    /// The directory the checkpoint is read from.
    directory: PathBuf,
    /// The revision's listing, for a snapshot resolved online; `None` for a stated directory and
    /// for an offline snapshot, whose directory holds every file it has.
    listed: Option<Listed>,
}

/// The files a Hub revision lists, and the client that downloads them.
struct Listed {
    /// The client.
    api: Api,
    /// The checkpoint at its commit, as failures name it.
    asset: String,
    /// The repository at its commit, which every file is downloaded at.
    repo: Repo,
    /// The checkpoint's folder in the repository, with its trailing `/`; `None` for the root.
    folder: Option<String>,
    /// The checkpoint-relative paths of the commit's files.
    files: Vec<String>,
}

impl Snapshot {
    /// The checkpoint stated as the local directory `directory`, read in place.
    pub fn local(directory: PathBuf) -> Self {
        Self {
            directory,
            listed: None,
        }
    }

    /// The directory the checkpoint is read from.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Downloads every file of the revision whose checkpoint-relative path `read` selects and
    /// the snapshot lacks; a file the snapshot holds is read without a request, since a commit's
    /// files never change. A stated directory and an offline snapshot fetch nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Remote`] when a download fails.
    pub async fn fetch(&self, read: impl Fn(&str) -> bool) -> Result<()> {
        let Some(listed) = &self.listed else {
            return Ok(());
        };
        let downloads = listed.api.repo(listed.repo.clone());
        for file in listed.files.iter().filter(|file| read(file)) {
            if self.directory.join(file).is_file() {
                continue;
            }
            let path = match &listed.folder {
                Some(folder) => format!("{folder}{file}"),
                None => file.clone(),
            };
            downloads
                .download(&path)
                .await
                .map_err(|error: ApiError| Error::Remote {
                    model: listed.asset.clone(),
                    message: format!("downloading {file}: {}", error.as_report()),
                })?;
        }
        Ok(())
    }
}

/// Whether `revision` is a commit: 40 lowercase hexadecimal digits, the form the Hub and
/// `huggingface_hub` give one.
fn is_commit(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use axum::http::StatusCode;
    use tempfile::tempdir;

    use super::*;
    use crate::api::model::assets::hub_stub::{COMMIT, HubStub, seed_snapshot};

    const REPO: &str = "org/model";

    #[test]
    fn an_asset_names_a_repository_at_a_revision() {
        assert_eq!(
            HubAsset::parse("org/model").unwrap().to_string(),
            "org/model@main"
        );
        assert_eq!(
            HubAsset::parse("org/model@v1.0").unwrap().to_string(),
            "org/model@v1.0"
        );
        assert_eq!(
            HubAsset::parse("org/adapters@v2/student/dense")
                .unwrap()
                .to_string(),
            "org/adapters@v2/student/dense"
        );
        assert_eq!(
            HubAsset::parse("org/adapters/student").unwrap().to_string(),
            "org/adapters@main/student"
        );
        assert_eq!(
            HubAsset::parse_pin("hf://org/base@9bfb6693").unwrap(),
            HubAsset::parse("org/base@9bfb6693").unwrap()
        );
        for refused in [
            "model",
            "/srv/checkpoint",
            "org/model/../other",
            "org/model//extra",
            "org/model@",
            "or g/model",
            "/org/model",
        ] {
            assert!(HubAsset::parse(refused).is_err(), "{refused}");
        }
        for refused in [
            "org/base@main",
            "hf://org/base",
            "hf://base@main",
            "hf://org/base@",
        ] {
            assert!(HubAsset::parse_pin(refused).is_err(), "{refused}");
        }
    }

    /// A snapshot is the directory of the revision's commit, filled with the files its reader
    /// selects: a selected file the cache holds is read without a download (the stub does not
    /// publish it), the other selected files are downloaded, and an unselected file is not.
    #[tokio::test]
    async fn a_snapshot_fetches_the_files_its_reader_selects() {
        let cache = tempdir().unwrap();
        let seeded = seed_snapshot(cache.path(), REPO, COMMIT, &[("config.json", "cached")]);
        let hub = HubStub::with_files(&[("tokenizer.json", "downloaded"), ("weights/a", "a")])
            .listing(&["config.json", "tokenizer.json", "weights/a"])
            .serve(cache.path())
            .await;

        let snapshot = hub.snapshot(&HubAsset::parse(REPO).unwrap()).await.unwrap();
        snapshot.fetch(|file| !file.contains('/')).await.unwrap();

        assert_eq!(snapshot.directory(), seeded);
        let read = |name: &str| fs::read_to_string(snapshot.directory().join(name)).unwrap();
        assert_eq!(read("config.json"), "cached");
        assert_eq!(read("tokenizer.json"), "downloaded");
        assert!(!snapshot.directory().join("weights/a").exists());
    }

    /// A checkpoint in a folder of its repository is that folder of the snapshot, and its files
    /// are named relative to it: only the folder's files are listed and fetched.
    #[tokio::test]
    async fn a_checkpoint_in_a_folder_is_that_folder_of_the_snapshot() {
        let cache = tempdir().unwrap();
        let hub = HubStub::with_files(&[
            ("student/adapter_model.safetensors", "student"),
            ("other/adapter_model.safetensors", "other"),
        ])
        .serve(cache.path())
        .await;

        let snapshot = hub
            .snapshot(&HubAsset::parse("org/adapters/student").unwrap())
            .await
            .unwrap();
        snapshot
            .fetch(|file| file == "adapter_model.safetensors")
            .await
            .unwrap();

        assert!(snapshot.directory().ends_with("student"));
        assert_eq!(
            fs::read_to_string(snapshot.directory().join("adapter_model.safetensors")).unwrap(),
            "student"
        );
        assert!(!snapshot.directory().join("../other").exists());
    }

    /// An offline host reads the snapshot a revision named when it was last resolved, or the
    /// commit's own, and refuses a revision its cache holds no snapshot of.
    #[tokio::test]
    async fn an_offline_host_reads_the_snapshot_its_cache_recorded() {
        let cache = tempdir().unwrap();
        let online = HubStub::with_files(&[("config.json", "{}")])
            .serve(cache.path())
            .await;
        let asset = HubAsset::parse(REPO).unwrap();
        let resolved = online.snapshot(&asset).await.unwrap();
        resolved.fetch(|_| true).await.unwrap();
        let offline = Hub::new(HubSettings {
            cache: cache.path().join("hub"),
            endpoint: None,
            token: None,
            offline: true,
        })
        .unwrap();

        let snapshot = offline.snapshot(&asset).await.unwrap();
        assert_eq!(snapshot.directory(), resolved.directory());
        let commit = HubAsset::parse(&format!("{REPO}@{COMMIT}")).unwrap();
        assert_eq!(
            offline.snapshot(&commit).await.unwrap().directory(),
            resolved.directory()
        );
        assert!(matches!(
            offline
                .snapshot(&HubAsset::parse("org/other").unwrap())
                .await,
            Err(Error::NotCached { .. })
        ));
    }

    /// A listing the Hub refuses is a remote failure, never an empty snapshot, and so is a
    /// download it refuses.
    #[tokio::test]
    async fn a_refused_request_is_a_remote_error() {
        let cache = tempdir().unwrap();
        let listing = HubStub {
            listing_status: Some(StatusCode::UNAUTHORIZED),
            ..HubStub::with_files(&[("config.json", "{}")])
        }
        .serve(cache.path())
        .await;
        let download = HubStub {
            download_status: Some(StatusCode::FORBIDDEN),
            ..HubStub::with_files(&[("config.json", "{}")])
        }
        .serve(cache.path())
        .await;
        let asset = HubAsset::parse(REPO).unwrap();

        assert!(matches!(
            listing.snapshot(&asset).await,
            Err(Error::Remote { .. })
        ));
        let snapshot = download.snapshot(&asset).await.unwrap();
        assert!(matches!(
            snapshot.fetch(|_| true).await,
            Err(Error::Remote { .. })
        ));
    }
}
