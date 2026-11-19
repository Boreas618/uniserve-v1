//! The files of a chat checkpoint that request processing reads, as its family declares them.
//!
//! Every chat checkpoint publishes its tokenizer definition at its root (`tokenizer.json`, with
//! the tokens `tokenizer_config.json` declares, as `transformers` saves a tokenizer), which the
//! daemon loads itself. Every other file request processing reads is the family's declaration
//! (`scheduling::processing::Serving::Chat`): the processing program names each file its binding
//! reads and whether a checkpoint must publish it, and the daemon reads exactly those, so
//! nothing is found by trying the places other families use.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use scheduling::processing::CheckpointFile;

use crate::api::model::assets::error::{Error, Result};
use crate::api::model::assets::{optional_file, required_file};

/// The tokenizer definition every chat checkpoint publishes at its root.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The files of one chat checkpoint request processing reads.
#[derive(Debug, Clone)]
pub struct ChatFiles {
    /// The checkpoint's directory, which holds its tokenizer.
    directory: PathBuf,
    /// Every declared file the checkpoint publishes, by its path relative to the root.
    files: BTreeMap<String, PathBuf>,
}

impl ChatFiles {
    /// The files of the chat checkpoint in `directory` whose family's binding reads `declared`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingFile`] when the checkpoint lacks its tokenizer definition or a
    /// file its family requires.
    pub fn locate(directory: &Path, declared: &[CheckpointFile]) -> Result<Self> {
        required_file(directory, TOKENIZER_FILE)?;
        let mut files = BTreeMap::new();
        for file in declared {
            let path = if file.required {
                Some(required_file(directory, &file.name)?)
            } else {
                optional_file(directory, &file.name)
            };
            if let Some(path) = path {
                files.insert(file.name.clone(), path);
            }
        }
        Ok(Self {
            directory: directory.to_path_buf(),
            files,
        })
    }

    /// The directory of the checkpoint's tokenizer (`HuggingFaceTokenizer::load`).
    pub fn tokenizer_directory(&self) -> &Path {
        &self.directory
    }

    /// Every file request processing reads of the checkpoint, the tokenizer definition first,
    /// each by its path relative to the root and where it lies: what decides how its requests
    /// are processed.
    pub fn processing_files(&self) -> Vec<(String, PathBuf)> {
        let tokenizer = (
            TOKENIZER_FILE.to_owned(),
            self.directory.join(TOKENIZER_FILE),
        );
        std::iter::once(tokenizer)
            .chain(
                self.files
                    .iter()
                    .map(|(name, path)| (name.clone(), path.clone())),
            )
            .collect()
    }

    /// The text of every declared file the checkpoint publishes, by its path: what the family is
    /// bound with.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] for a file that cannot be read as UTF-8.
    pub fn texts(&self) -> Result<BTreeMap<String, String>> {
        self.files
            .iter()
            .map(|(name, path)| {
                let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
                Ok((name.clone(), text))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    /// The declaration of a family that reads its configuration and tokenizer metadata, and
    /// generation defaults when a checkpoint publishes them.
    fn declared() -> Vec<CheckpointFile> {
        let file = |name: &str, required| CheckpointFile {
            name: name.to_owned(),
            required,
        };
        vec![
            file("config.json", true),
            file("tokenizer_config.json", true),
            file("generation_config.json", false),
        ]
    }

    /// A checkpoint's declared files are read where they lie, an optional one it does not
    /// publish left out; a required one it lacks, like its tokenizer, is refused.
    #[test]
    fn a_checkpoint_is_read_as_its_family_declares() {
        let directory = tempdir().unwrap();
        for (name, text) in [
            ("tokenizer.json", "{}"),
            ("config.json", "{\"a\": 1}"),
            ("tokenizer_config.json", "{}"),
        ] {
            fs::write(directory.path().join(name), text).unwrap();
        }

        let files = ChatFiles::locate(directory.path(), &declared()).unwrap();
        let texts = files.texts().unwrap();
        assert_eq!(
            texts.keys().map(String::as_str).collect::<Vec<_>>(),
            ["config.json", "tokenizer_config.json"]
        );
        assert_eq!(texts["config.json"], "{\"a\": 1}");

        fs::remove_file(directory.path().join("tokenizer_config.json")).unwrap();
        assert!(matches!(
            ChatFiles::locate(directory.path(), &declared()),
            Err(Error::MissingFile { .. })
        ));
        fs::remove_file(directory.path().join("tokenizer.json")).unwrap();
        assert!(matches!(
            ChatFiles::locate(directory.path(), &[]),
            Err(Error::MissingFile { .. })
        ));
    }
}
