//! Deserialized tokenizer and generation metadata of a checkpoint.
//!
//! Each loader reads one file the checkpoint publishes ([`super::ChatFiles`] locates them). A
//! field the file does not state is `None`, as the format makes every one of these fields
//! optional, and fields the daemon does not read are ignored.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::api::model::assets::error::{Error, Result};

/// Tokenizer metadata consumed by configured chat rendering and stop handling.
///
/// Read from `tokenizer_config.json` by `ModelConfig::from_files` (for the
/// primary EOS), by `HfChatRenderer::load` (for the special tokens exposed to
/// its template) and by [`super::ChatFiles::chat_template`] (for a family that
/// keeps its template there).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HfTokenizerConfig {
    /// Named tokenizer special tokens.
    #[serde(flatten)]
    pub special_tokens: HfSpecialTokens,
    /// Embedded default chat template, when configured.
    pub chat_template: Option<String>,
}

/// A named Hugging Face special token represented as text or an object.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum NamedSpecialToken {
    /// Token represented directly as text.
    Text(String),
    /// Token represented by an object containing its text.
    WithContent {
        /// Token text.
        content: String,
    },
}

impl Serialize for NamedSpecialToken {
    /// Serializes either form as its plain token text, which is the value a
    /// chat template sees for `bos_token` and the other special tokens.
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl From<NamedSpecialToken> for String {
    /// Extracts the token text from either form.
    fn from(value: NamedSpecialToken) -> Self {
        match value {
            NamedSpecialToken::Text(string) => string,
            NamedSpecialToken::WithContent { content } => content,
        }
    }
}

impl NamedSpecialToken {
    /// Returns the token text.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(value) | Self::WithContent { content: value } => value,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
/// Special-token strings resolved from tokenizer metadata.
///
/// Serialization omits unset tokens, so a chat template sees them as
/// undefined rather than null.
pub struct HfSpecialTokens {
    /// Beginning-of-sequence token.
    pub bos_token: Option<NamedSpecialToken>,
    /// End-of-sequence token.
    pub eos_token: Option<NamedSpecialToken>,
    /// Unknown-token marker.
    pub unk_token: Option<NamedSpecialToken>,
    /// Padding token.
    pub pad_token: Option<NamedSpecialToken>,
}

impl HfSpecialTokens {
    /// Returns whether no special-token value is configured.
    pub fn is_empty(&self) -> bool {
        self.bos_token.is_none()
            && self.eos_token.is_none()
            && self.unk_token.is_none()
            && self.pad_token.is_none()
    }
}

/// Generation defaults consumed by configured request lowering.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    /// One or more token identifiers that terminate generation.
    pub eos_token_id: Option<OneOrManyTokenIds>,
    /// Default sampling temperature.
    pub temperature: Option<f32>,
    /// Default nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Default candidate-token limit.
    pub top_k: Option<u32>,
    /// Default minimum relative token probability.
    pub min_p: Option<f32>,
    /// Default repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Default maximum number of generated tokens.
    pub max_new_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
/// Token-id field accepted as one identifier or a list, as
/// `generation_config.json` writes `eos_token_id`.
pub enum OneOrManyTokenIds {
    /// One token identifier.
    One(u32),
    /// Ordered token identifier list.
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    /// Converts one-or-many identifiers into a deduplicated ordered set.
    pub fn into_set(self) -> BTreeSet<u32> {
        match self {
            Self::One(id) => BTreeSet::from([id]),
            Self::Many(ids) => ids.into_iter().collect(),
        }
    }
}

/// Loads a checkpoint's tokenizer metadata (`tokenizer_config.json`).
///
/// # Errors
///
/// As [`read_json`].
pub fn load_tokenizer_config(path: &Path) -> Result<HfTokenizerConfig> {
    read_json(path)
}

/// Loads a checkpoint's generation defaults (`generation_config.json`).
///
/// # Errors
///
/// As [`read_json`].
pub fn load_generation_config(path: &Path) -> Result<GenerationConfig> {
    read_json(path)
}

/// Deserializes the JSON file at `path`.
///
/// # Errors
///
/// Returns [`Error::Io`] for an unreadable file and [`Error::Json`] for
/// content that does not deserialize into `T`, including a missing required
/// field.
pub fn read_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let content = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}
