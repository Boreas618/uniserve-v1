//! Hugging Face tokenizer loading and the unified tokenizer implementation.

use std::path::Path;
use std::sync::Arc;

use fastokens::{NewToken, Tokenizer as FastokensTokenizer};
use indexmap::IndexMap;
use serde::Deserialize;
use thiserror_ext::AsReport as _;
use tracing::info;

use crate::api::tokenizer::Result;
use crate::api::tokenizer::incremental::IncrementalDecoder;

/// The tokenizer definition of a tokenizer directory, as `transformers` saves a tokenizer.
const TOKENIZER_FILE: &str = "tokenizer.json";

/// The tokenizer metadata saved beside the definition.
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";

/// Tokenizer loaded from a Hugging Face tokenizer directory.
///
/// One instance is shared through `DynTokenizer` by every request of a
/// resolved model; all methods take `&self`.
pub struct HuggingFaceTokenizer {
    tokenizer: Box<FastokensTokenizer>,
}

impl HuggingFaceTokenizer {
    /// Loads the tokenizer in `directory` the way `transformers` loads a
    /// checkpoint's tokenizer: its `tokenizer.json`, with every token its
    /// `tokenizer_config.json` declares and the definition lacks.
    ///
    /// `fastokens` reads the definition and registers the configuration's
    /// `added_tokens_decoder` at the ids it states; the special tokens the
    /// configuration declares by name or in its lists ([`DeclaredTokens`]) are
    /// registered through `fastokens`' own registration, which assigns ids as
    /// `tokenizers` does. A prompt containing such a token then encodes it as
    /// one token, as the checkpoint's own processor encodes it.
    ///
    /// # Errors
    ///
    /// Returns a tokenizer error when either file cannot be read or parsed,
    /// or when `fastokens` cannot build or extend the definition.
    pub fn load(directory: &Path) -> Result<Self> {
        let definition = directory.join(TOKENIZER_FILE);
        let config = directory.join(TOKENIZER_CONFIG_FILE);
        info!(directory = %directory.display(), "loading the Hugging Face tokenizer");
        let mut tokenizer = FastokensTokenizer::from_file(&definition).map_err(|error| {
            tokenizer_error!(
                "failed to load {}: {}",
                definition.display(),
                error.as_report()
            )
        })?;
        let text = std::fs::read_to_string(&config).map_err(|error| {
            tokenizer_error!("failed to read {}: {}", config.display(), error.as_report())
        })?;
        let declared: DeclaredTokens = serde_json::from_str(&text).map_err(|error| {
            tokenizer_error!(
                "failed to parse {}: {}",
                config.display(),
                error.as_report()
            )
        })?;

        // A token the definition or the decoder entries registered keeps its entry as it is.
        let registered: std::collections::HashSet<String> = tokenizer
            .added_token_configs()
            .iter()
            .map(|token| token.content.clone())
            .collect();
        let missing: Vec<NewToken> = declared
            .tokens()
            .into_iter()
            .filter(|token| !registered.contains(&token.content))
            .collect();
        tokenizer.add_tokens(&missing).map_err(|error| {
            tokenizer_error!(
                "failed to register the tokens {} declares: {}",
                config.display(),
                error.as_report()
            )
        })?;
        Ok(Self::from_fastokens(tokenizer))
    }

    /// Builds a tokenizer from a parsed `tokenizer.json` definition.
    ///
    /// Returns a tokenizer error when `fastokens` cannot build the definition.
    pub fn from_json(definition: serde_json::Value) -> Result<Self> {
        let tokenizer = FastokensTokenizer::from_json(definition)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        Ok(Self::from_fastokens(tokenizer))
    }

    /// Wraps a loaded tokenizer.
    fn from_fastokens(tokenizer: FastokensTokenizer) -> Self {
        Self {
            tokenizer: Box::new(tokenizer),
        }
    }

    /// The `fastokens` tokenizer, which the incremental decoder steps with.
    pub(super) fn inner(&self) -> &FastokensTokenizer {
        &self.tokenizer
    }

    /// Encodes one prompt string into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))
    }

    /// Decodes token identifiers with optional special-token filtering, through
    /// `fastokens`' decoder, which gives an ID outside the vocabulary no text as
    /// Hugging Face `tokenizers` does.
    ///
    /// Fails only when the tokenizer's decoder does.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| tokenizer_error!("decoding failed: {}", error.as_report()))
    }

    /// Resolves one vocabulary token to its identifier.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Returns whether an identifier belongs to the special-token set.
    pub fn is_special_id(&self, token_id: u32) -> bool {
        self.tokenizer.is_special_token(token_id)
    }

    /// Creates a stateful incremental decoder primed with the given prompt tokens.
    ///
    /// `min_bytes_to_buffer` is the number of trailing output bytes that
    /// `IncrementalDecoder::next_chunk` withholds; the serving output stages
    /// pass the stop-string holdback computed by `stop_string_holdback_bytes`.
    pub fn create_decode_stream(
        self: &Arc<Self>,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> IncrementalDecoder {
        IncrementalDecoder::new(
            Arc::clone(self),
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        )
    }
}

/// The special tokens a `tokenizer_config.json` declares beside its
/// `added_tokens_decoder`, which `transformers` registers when it loads the
/// tokenizer and `fastokens` does not.
///
/// `fastokens` 0.3 merges only the configuration's `added_tokens_decoder`
/// into the definition; `transformers` also registers the named special tokens
/// (`SPECIAL_TOKENS_ATTRIBUTES`), the model-specific ones a table of
/// `extra_special_tokens` names, and the listed extra special tokens. A
/// `transformers` 5 configuration states that list as `extra_special_tokens`
/// and a `transformers` 4 configuration as `additional_special_tokens`, which
/// `transformers` 5 reads when its own key is absent, as here.
#[derive(Debug, Deserialize)]
struct DeclaredTokens {
    bos_token: Option<DeclaredToken>,
    eos_token: Option<DeclaredToken>,
    unk_token: Option<DeclaredToken>,
    sep_token: Option<DeclaredToken>,
    pad_token: Option<DeclaredToken>,
    cls_token: Option<DeclaredToken>,
    mask_token: Option<DeclaredToken>,
    /// The extra special tokens of a `transformers` 5 configuration: a list,
    /// or a table naming the model-specific ones.
    extra_special_tokens: Option<ExtraTokens>,
    /// The extra special tokens of a `transformers` 4 configuration.
    #[serde(default)]
    additional_special_tokens: Vec<DeclaredToken>,
}

/// The extra special tokens a `transformers` 5 configuration states.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExtraTokens {
    /// Tokens listed without names.
    Listed(Vec<DeclaredToken>),
    /// Model-specific special tokens by attribute name (`image_token`, ...), in
    /// the order the file states them.
    Named(IndexMap<String, DeclaredToken>),
}

/// One token a `tokenizer_config.json` declares: its text, which
/// `transformers` makes a special token, or an `AddedToken` object.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum DeclaredToken {
    Text(String),
    Token {
        content: String,
        #[serde(default)]
        single_word: bool,
        #[serde(default)]
        lstrip: bool,
        #[serde(default)]
        rstrip: bool,
        normalized: Option<bool>,
        special: Option<bool>,
    },
}

impl DeclaredToken {
    /// The token to register: `named` forces the special flag of a named
    /// token; an unset `normalized` is `!special`, as `AddedToken` sets it.
    fn new_token(self, named: bool) -> NewToken {
        match self {
            Self::Text(content) => NewToken {
                content,
                special: true,
                ..NewToken::default()
            },
            Self::Token {
                content,
                single_word,
                lstrip,
                rstrip,
                normalized,
                special,
            } => {
                let special = named || special.unwrap_or(false);
                NewToken {
                    content,
                    single_word,
                    lstrip,
                    rstrip,
                    normalized: normalized.unwrap_or(!special),
                    special,
                }
            }
        }
    }
}

impl DeclaredTokens {
    /// The declared tokens in the order `transformers` (5.x) registers them
    /// after the `added_tokens_decoder` entries: the named special tokens,
    /// then the model-specific named ones, then the listed extra special
    /// tokens.
    fn tokens(self) -> Vec<NewToken> {
        let named = [
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.sep_token,
            self.pad_token,
            self.cls_token,
            self.mask_token,
        ]
        .into_iter()
        .flatten();
        let (model_specific, listed) = match self.extra_special_tokens {
            Some(ExtraTokens::Named(table)) => (table.into_values().collect(), Vec::new()),
            Some(ExtraTokens::Listed(list)) => (Vec::new(), list),
            None => (Vec::new(), self.additional_special_tokens),
        };
        named
            .chain(model_specific)
            .map(|token| token.new_token(true))
            .chain(listed.into_iter().map(|token| token.new_token(false)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::HuggingFaceTokenizer;

    fn tiny_bpe_tokenizer() -> TokenizerBuilder {
        let model = BPE::builder()
            .vocab_and_merges(
                [
                    ("<unk>".to_string(), 0),
                    ("h".to_string(), 1),
                    ("e".to_string(), 2),
                    ("l".to_string(), 3),
                    ("o".to_string(), 4),
                    ("he".to_string(), 5),
                    ("ll".to_string(), 6),
                    ("hell".to_string(), 7),
                    ("hello".to_string(), 8),
                ],
                vec![
                    ("h".to_string(), "e".to_string()),
                    ("l".to_string(), "l".to_string()),
                    ("he".to_string(), "ll".to_string()),
                    ("hell".to_string(), "o".to_string()),
                ],
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("build bpe tokenizer");
        TokenizerBuilder::new(model)
    }

    #[test]
    fn configured_tokenizer_encodes_decodes_and_identifies_special_tokens() {
        let mut tokenizer = tiny_bpe_tokenizer();
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
        let directory = tempdir().expect("create tokenizer directory");
        tokenizer
            .save(directory.path().join("tokenizer.json"), false)
            .expect("save tokenizer");
        std::fs::write(directory.path().join("tokenizer_config.json"), "{}").unwrap();
        let configured =
            HuggingFaceTokenizer::load(directory.path()).expect("load configured tokenizer");

        let encoded = configured.encode("hello", false).expect("encode text");
        assert_eq!(configured.decode(&encoded, false).unwrap(), "hello");
        let special_id = configured
            .token_to_id("<|im_end|>")
            .expect("resolve special token");
        assert!(configured.is_special_id(special_id));
    }

    const BYTE_LEVEL_TOKENIZER: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "single_word": false,
             "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                          "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": {
                "<|endoftext|>": 0,
                "H": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7,
                "Ġ": 8, "!": 9, "｜": 10
            },
            "merges": []
        }
    }"#;

    /// A tokenizer directory of [`BYTE_LEVEL_TOKENIZER`] and `config`.
    fn byte_level_directory(config: &str) -> tempfile::TempDir {
        let directory = tempdir().expect("create tokenizer directory");
        std::fs::write(
            directory.path().join("tokenizer.json"),
            BYTE_LEVEL_TOKENIZER,
        )
        .expect("write tokenizer");
        std::fs::write(directory.path().join("tokenizer_config.json"), config)
            .expect("write tokenizer config");
        directory
    }

    /// Models sample over a vocabulary dimension that can exceed the tokenizer
    /// vocabulary, so an ID in that gap can reach decoding. Like Hugging Face
    /// `tokenizers`, decoding skips it instead of failing.
    #[test]
    fn decoding_skips_unknown_token_ids() {
        let directory = byte_level_directory("{}");
        let configured = HuggingFaceTokenizer::load(directory.path()).unwrap();
        assert_eq!(configured.decode(&[999], false).unwrap(), "");
        assert_eq!(configured.decode(&[1, 999, 2], false).unwrap(), "He");
        assert_eq!(configured.decode(&[1, 8, 5], false).unwrap(), "H w");
    }

    /// The special tokens a `tokenizer_config.json` declares but the
    /// definition lacks are registered as `transformers` registers them:
    /// once each, a vocabulary text at its vocabulary id, any other text
    /// after the largest id in declaration order (named tokens before the
    /// extra ones). The expected ids are those `transformers` 5.14 assigns
    /// when it loads the same two files.
    #[test]
    fn configured_special_tokens_are_registered_like_transformers() {
        let directory = byte_level_directory(
            r#"{
                "added_tokens_decoder": {
                    "0": {"content": "<|endoftext|>", "lstrip": false, "normalized": false,
                          "rstrip": false, "single_word": false, "special": true}
                },
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "add_bos_token": false,
                "additional_special_tokens": ["<|endoftext|>", "<d>", "</d>", "<d>", "H"]
            }"#,
        );
        let configured =
            HuggingFaceTokenizer::load(directory.path()).expect("load configured tokenizer");

        for (token, id) in [("<|pad|>", 11), ("<d>", 12), ("</d>", 13), ("H", 1)] {
            assert_eq!(configured.token_to_id(token), Some(id), "{token}");
            assert!(configured.is_special_id(id), "{token}");
        }
        assert_eq!(
            configured
                .encode("Hello<d>world</d>!<|pad|>", false)
                .unwrap(),
            [1, 2, 3, 3, 4, 12, 5, 4, 6, 3, 7, 13, 9, 11]
        );
    }

    /// A `transformers` 5 configuration names model-specific special tokens in
    /// a table of `extra_special_tokens`, which are registered in the file's
    /// order after the named ones, and its own key stands for its list.
    #[test]
    fn named_extra_special_tokens_are_registered_in_order() {
        let directory = byte_level_directory(
            r#"{
                "eos_token": "<|endoftext|>",
                "extra_special_tokens": {"video_token": "<|video|>", "image_token": "<|image|>"},
                "additional_special_tokens": ["<|unread|>"]
            }"#,
        );
        let configured = HuggingFaceTokenizer::load(directory.path()).unwrap();

        assert_eq!(configured.token_to_id("<|video|>"), Some(11));
        assert_eq!(configured.token_to_id("<|image|>"), Some(12));
        assert_eq!(configured.token_to_id("<|unread|>"), None);
    }
}
