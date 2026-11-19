//! The tokenizer and generation metadata of a chat checkpoint, as its binding reads them from
//! the files' text.
//!
//! A field the file does not state is `None`, as the format makes every one of these fields
//! optional, and fields request processing does not read are ignored.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// The file every chat checkpoint publishes beside its tokenizer definition.
pub const TOKENIZER_CONFIG: &str = "tokenizer_config.json";

/// The generation defaults a checkpoint may publish.
pub const GENERATION_CONFIG: &str = "generation_config.json";

/// Tokenizer metadata consumed by chat rendering and stop handling (`tokenizer_config.json`):
/// the special tokens a template sees, the primary EOS, and the template a family keeps here.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct TokenizerConfig {
    /// Named tokenizer special tokens.
    #[serde(flatten)]
    pub special_tokens: SpecialTokens,
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

impl NamedSpecialToken {
    /// Returns the token text.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(value) | Self::WithContent { content: value } => value,
        }
    }
}

/// Special-token strings resolved from tokenizer metadata.
///
/// Serialization omits unset tokens, so a chat template sees them as
/// undefined rather than null.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token.
    pub bos_token: Option<NamedSpecialToken>,
    /// End-of-sequence token.
    pub eos_token: Option<NamedSpecialToken>,
    /// Unknown-token marker.
    pub unk_token: Option<NamedSpecialToken>,
    /// Padding token.
    pub pad_token: Option<NamedSpecialToken>,
}

impl SpecialTokens {
    /// Returns whether no special-token value is configured.
    pub fn is_empty(&self) -> bool {
        self.bos_token.is_none()
            && self.eos_token.is_none()
            && self.unk_token.is_none()
            && self.pad_token.is_none()
    }
}

/// Generation defaults (`generation_config.json`).
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

/// Token-id field accepted as one identifier or a list, as `generation_config.json` writes
/// `eos_token_id`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
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

/// The JSON `text` of the checkpoint's file `name`, deserialized.
///
/// # Errors
///
/// Names the file and why its content does not deserialize, including a missing required
/// field.
pub fn parse<T: for<'de> Deserialize<'de>>(name: &str, text: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|error| format!("{name} does not parse: {error}"))
}
