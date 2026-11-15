//! The served checkpoint's files: where they lie and what they state.
//!
//! A deployment states each of its checkpoints, a model or an adapter, as a [`Source`]: a
//! Hugging Face repository at a revision, resolved through the process's [`Hub`] into the
//! standard cache's snapshot of the revision's commit, or a local directory it names explicitly.
//! Either way a checkpoint is read from a local directory, which is also what the engines open
//! ([`Checkpoint`]).
//!
//! Which files of that directory are read is what the checkpoint's family declares: the root
//! files that declare the family ([`declared_family`]); a chat checkpoint's tokenizer,
//! configuration and template ([`ChatFiles`]); a diffusers pipeline's stage folders, over the
//! release a layered checkpoint pins ([`PipelineCheckpoint`]). [`config`] deserializes the JSON
//! metadata. Every failure is this module's [`Error`].

use std::fmt;
use std::path::{Path, PathBuf};

pub mod config;
pub mod error;
mod files;
mod hub;
#[cfg(test)]
mod hub_stub;
mod pipeline;
mod pipeline_index;

pub use config::{
    GenerationConfig, HfSpecialTokens, HfTokenizerConfig, NamedSpecialToken, OneOrManyTokenIds,
    load_generation_config, load_tokenizer_config, read_json,
};
pub use error::{Error, Result};
pub use files::ChatFiles;
pub use hub::{Hub, HubAsset, HubSettings, Snapshot};
pub use pipeline::PipelineCheckpoint;

/// Where a deployment states one of its checkpoints lies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A Hugging Face repository at a revision, read from the cache's snapshot of it.
    Hub(HubAsset),
    /// A local directory, read in place.
    Directory(PathBuf),
}

impl Source {
    /// The local directory the checkpoint is read from: the snapshot `hub` resolves for a
    /// repository ([`Hub::snapshot`]), or the stated directory.
    ///
    /// # Errors
    ///
    /// The errors of [`Hub::snapshot`], and [`Error::NotDirectory`] for a stated directory that
    /// is not one.
    pub async fn locate(&self, hub: &Hub) -> Result<Snapshot> {
        match self {
            Self::Hub(asset) => hub.snapshot(asset).await,
            Self::Directory(directory) if directory.is_dir() => {
                Ok(Snapshot::local(directory.clone()))
            }
            Self::Directory(directory) => Err(Error::NotDirectory {
                model: directory.display().to_string(),
            }),
        }
    }

    /// The local directory of the checkpoint with every file it lists: a model's or an
    /// adapter's, whose engine reads the checkpoint whole.
    ///
    /// # Errors
    ///
    /// As [`Source::locate`] and [`Snapshot::fetch`].
    pub async fn directory(&self, hub: &Hub) -> Result<PathBuf> {
        let snapshot = self.locate(hub).await?;
        snapshot.fetch(|_| true).await?;
        Ok(snapshot.directory().to_path_buf())
    }
}

impl fmt::Display for Source {
    /// Formats the source as it is stated: `<owner>/<name>@<revision>` or the directory.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hub(asset) => asset.fmt(formatter),
            Self::Directory(directory) => directory.display().fmt(formatter),
        }
    }
}

/// A deployment's checkpoint as its engines open it, in local directories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    /// The checkpoint's directory.
    pub directory: PathBuf,
    /// The snapshot of the release a layered checkpoint continues, whose stage folders the
    /// checkpoint omits ([`PipelineCheckpoint::base`]); `None` for a checkpoint that holds every
    /// stage.
    pub base: Option<PathBuf>,
    /// The directory of the weight adapter fused into the checkpoint at load, when the
    /// deployment states one.
    pub adapter: Option<PathBuf>,
}

/// The file `name` of the checkpoint in `directory`, which the checkpoint must hold.
///
/// # Errors
///
/// Returns [`Error::MissingFile`] when the checkpoint has no such file.
pub fn required_file(directory: &Path, name: &str) -> Result<PathBuf> {
    optional_file(directory, name).ok_or_else(|| Error::MissingFile {
        model: directory.display().to_string(),
        file: name.to_owned(),
    })
}

/// The file `name` of the checkpoint in `directory`, when the checkpoint publishes it.
pub fn optional_file(directory: &Path, name: &str) -> Option<PathBuf> {
    let path = directory.join(name);
    path.is_file().then_some(path)
}

/// Whether the checkpoint-relative `file` lies at the checkpoint's root, where a checkpoint
/// declares what it is ([`declared_family`]).
pub fn root_file(file: &str) -> bool {
    !file.contains('/')
}

/// The text of the checkpoint's root file `name`, when the checkpoint publishes it.
fn optional_text(directory: &Path, name: &str) -> Result<Option<String>> {
    optional_file(directory, name)
        .map(|path| std::fs::read_to_string(&path).map_err(|source| Error::Io { path, source }))
        .transpose()
}

/// The family of `registry` the checkpoint in `directory` declares in the root files the
/// registry's families declare their classes in ([`processing::Registry::root_files`]), under
/// request processing's one rule ([`processing::Registry::resolve`]).
///
/// # Errors
///
/// Returns [`Error::Family`] when the checkpoint declares no family `registry` serves, and the
/// errors of reading the root files.
pub fn declared_family(
    directory: &Path,
    registry: &processing::Registry,
) -> Result<processing::Family> {
    let mut files = Vec::new();
    for file in registry.root_files() {
        if let Some(text) = optional_text(directory, file.name())? {
            files.push((file, text));
        }
    }
    Ok(registry.resolve(files.iter().map(|(file, text)| (*file, text.as_str())))?)
}
