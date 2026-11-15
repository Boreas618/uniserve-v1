//! The files of a chat checkpoint that request processing reads, where its family keeps them.
//!
//! Every chat checkpoint publishes, at its root, the configuration that declares its family
//! ([`processing::RootFile::Config`]) and its tokenizer (`tokenizer.json` with `tokenizer_config.json`,
//! as `transformers` saves a tokenizer). Its generation defaults (`generation_config.json`) are
//! optional in the format. Where it states its context bound and keeps its chat template is the
//! family's declaration ([`processing::ChatCheckpoint`]), so nothing is found by trying the places
//! other families use.

use std::path::{Path, PathBuf};

use processing::{ChatCheckpoint, ChatTemplate};
use serde_json::Value;

use crate::api::model::assets::config::{load_tokenizer_config, read_json};
use crate::api::model::assets::error::{Error, Result};
use crate::api::model::assets::{optional_file, required_file};

/// The tokenizer definition every chat checkpoint publishes at its root.
const TOKENIZER_FILE: &str = "tokenizer.json";

/// The tokenizer metadata every chat checkpoint publishes beside its definition.
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";

/// The generation defaults a checkpoint may publish.
const GENERATION_CONFIG_FILE: &str = "generation_config.json";

/// The files of one chat checkpoint request processing reads.
#[derive(Debug, Clone)]
pub struct ChatFiles {
    /// The checkpoint's directory, which holds its tokenizer.
    directory: PathBuf,
    /// The configuration that declares the family.
    config: PathBuf,
    /// The tokenizer metadata.
    tokenizer_config: PathBuf,
    /// The generation defaults, when the checkpoint publishes them.
    generation_config: Option<PathBuf>,
    /// Where the family keeps what else request processing reads.
    checkpoint: ChatCheckpoint,
}

impl ChatFiles {
    /// The files of the chat checkpoint in `directory`, whose family keeps its context bound and
    /// template as `checkpoint` states.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingFile`] when the checkpoint lacks its configuration, either
    /// tokenizer file, or the template file its family declares.
    pub fn locate(directory: &Path, checkpoint: ChatCheckpoint) -> Result<Self> {
        required_file(directory, TOKENIZER_FILE)?;
        if let ChatTemplate::File(name) = checkpoint.chat_template {
            required_file(directory, name)?;
        }
        Ok(Self {
            directory: directory.to_path_buf(),
            config: required_file(directory, processing::RootFile::Config.name())?,
            tokenizer_config: required_file(directory, TOKENIZER_CONFIG_FILE)?,
            generation_config: optional_file(directory, GENERATION_CONFIG_FILE),
            checkpoint,
        })
    }

    /// The directory of the checkpoint's tokenizer (`HuggingFaceTokenizer::load`).
    pub fn tokenizer_directory(&self) -> &Path {
        &self.directory
    }

    /// The checkpoint's tokenizer metadata (`tokenizer_config.json`).
    pub fn tokenizer_config(&self) -> &Path {
        &self.tokenizer_config
    }

    /// The checkpoint's generation defaults (`generation_config.json`); `None` for a checkpoint
    /// that states none.
    pub fn generation_config(&self) -> Option<&Path> {
        self.generation_config.as_deref()
    }

    /// The checkpoint's chat template, read where its family keeps it.
    ///
    /// # Errors
    ///
    /// Fails when the file that holds it cannot be read, or a tokenizer configuration that
    /// should hold it states none.
    pub fn chat_template(&self) -> Result<String> {
        match self.checkpoint.chat_template {
            ChatTemplate::TokenizerConfig => load_tokenizer_config(&self.tokenizer_config)?
                .chat_template
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "{} states no chat_template, where the checkpoint's family keeps it",
                        self.tokenizer_config.display()
                    ))
                }),
            ChatTemplate::File(name) => {
                let path = self.directory.join(name);
                std::fs::read_to_string(&path).map_err(|source| Error::Io { path, source })
            }
        }
    }

    /// The longest sequence the model's positions cover, in tokens, stated where the family's
    /// checkpoints state it.
    ///
    /// # Errors
    ///
    /// Fails when the configuration cannot be read, or states no count of 32-bit range at the
    /// family's field.
    pub fn context_length(&self) -> Result<u32> {
        let field = self.checkpoint.context_length;
        let config: Value = read_json(&self.config)?;
        field
            .iter()
            .try_fold(&config, |value, key| value.get(key))
            .and_then(Value::as_u64)
            .and_then(|length| u32::try_from(length).ok())
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{} states no context length at `{}`, where the checkpoint's family states it",
                    self.config.display(),
                    field.join(".")
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    /// A family that keeps its template in its tokenizer configuration and its context bound at
    /// the root of its configuration.
    const ROOT: ChatCheckpoint = ChatCheckpoint {
        context_length: &["max_position_embeddings"],
        chat_template: ChatTemplate::TokenizerConfig,
    };

    /// A family whose language model is a section of its configuration and whose template is a
    /// file of its own.
    const SECTION: ChatCheckpoint = ChatCheckpoint {
        context_length: &["text_config", "max_position_embeddings"],
        chat_template: ChatTemplate::File("chat_template.jinja"),
    };

    /// A checkpoint directory with a tokenizer, `config` and `tokenizer_config`.
    fn checkpoint(config: &str, tokenizer_config: &str) -> tempfile::TempDir {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join(TOKENIZER_FILE), "{}").unwrap();
        fs::write(
            directory.path().join(processing::RootFile::Config.name()),
            config,
        )
        .unwrap();
        fs::write(
            directory.path().join(TOKENIZER_CONFIG_FILE),
            tokenizer_config,
        )
        .unwrap();
        directory
    }

    #[test]
    fn a_checkpoint_states_its_bound_and_template_where_its_family_keeps_them() {
        let directory = checkpoint(
            r#"{"max_position_embeddings": 40960}"#,
            r#"{"chat_template": "{{ messages }}"}"#,
        );

        let files = ChatFiles::locate(directory.path(), ROOT).unwrap();

        assert_eq!(files.context_length().unwrap(), 40960);
        assert_eq!(files.chat_template().unwrap(), "{{ messages }}");
        assert_eq!(files.generation_config(), None);
    }

    /// A family's declaration is the one place read: a bound at the root of a sectioned
    /// configuration is not its language model's, and a template the tokenizer configuration
    /// also states is read from the family's file, verbatim.
    #[test]
    fn a_sectioned_checkpoint_is_read_where_its_family_says() {
        let directory = checkpoint(
            r#"{"max_position_embeddings": 1, "text_config": {"max_position_embeddings": 262144}}"#,
            r#"{"chat_template": "from the configuration"}"#,
        );
        assert!(matches!(
            ChatFiles::locate(directory.path(), SECTION),
            Err(Error::MissingFile { .. })
        ));
        fs::write(directory.path().join("chat_template.jinja"), "{{ x }}\\n\n").unwrap();
        fs::write(directory.path().join(GENERATION_CONFIG_FILE), "{}").unwrap();

        let files = ChatFiles::locate(directory.path(), SECTION).unwrap();

        assert_eq!(files.context_length().unwrap(), 262144);
        assert_eq!(files.chat_template().unwrap(), "{{ x }}\\n\n");
        assert!(files.generation_config().is_some());
    }

    /// A checkpoint that does not state what its family says it does is refused rather than
    /// served with a substitute.
    #[test]
    fn an_unstated_bound_or_template_is_refused() {
        let directory = checkpoint(r#"{"text_config": {}}"#, "{}");

        let files = ChatFiles::locate(directory.path(), ROOT).unwrap();

        assert!(matches!(files.context_length(), Err(Error::Invalid(_))));
        assert!(matches!(files.chat_template(), Err(Error::Invalid(_))));
    }
}
