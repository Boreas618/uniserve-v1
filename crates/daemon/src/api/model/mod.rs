// The served checkpoint's files and JSON metadata.
//
// [`assets`] locates a checkpoint in a local directory, the local Hugging
// Face Hub cache, or the Hub itself; reads the root `config.json` or diffusers
// pipeline index that selects the model family; reads the tokenizer's and
// generation's metadata; and reads tensor shapes from a safetensors header
// without loading weights. The resolved model, its family profile and its
// output parsers are the frontend's as well; they move here from the daemon
// with the rest of the application layer.

pub mod assets;
pub mod generation;

// Model-profile selection and load-time serving capabilities.
//
// The server identifies the served model family from the checkpoint itself,
// by what its root `config.json` or, for a diffusers pipeline, its root index
// declares (`processing::Registry::resolve`). The result is one
// [`ModelConfig`], built at startup by `ModelConfig::load` in
// `input::processor`. The leader's startup reads it, and `Processor::new` then
// binds the worker's capabilities onto it before request preprocessing and
// discovery share it immutably.
//
// The checkpoint's files are resolved and read here ([`assets`]); which family
// a checkpoint is, and what that family serves, is the executable's registry
// of families (`processing::Registry`), whose request processing reads the
// checkpoint's tokenizer (`crate::api::tokenizer`) through `processing::Vocabulary`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use processing::Family;
use processing::{ChatProfile, Serving, VideoFamily};

use crate::api::model::assets::{
    GenerationConfig, HfTokenizerConfig, ResolvedModelFiles, load_generation_config,
    load_model_config, load_tokenizer_config,
};
use crate::api::tokenizer::HuggingFaceTokenizer;

/// Model-provided sampling defaults. `None` remains distinct from an explicit zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingDefaults {
    /// Temperature used when the request omits it; zero selects greedy sampling.
    pub temperature: Option<f32>,
    /// Default nucleus probability mass.
    pub top_p: Option<f32>,
    /// Default candidate limit; zero disables top-k filtering.
    pub top_k: Option<u32>,
    /// Default minimum probability relative to the most likely token.
    pub min_p: Option<f32>,
    /// Default multiplicative penalty for repeated tokens.
    pub repetition_penalty: Option<f32>,
    /// Default generation length from `max_new_tokens`, used only when the
    /// request omits `max_tokens`; an explicit request value replaces it.
    pub max_output_tokens: Option<u32>,
}

/// Immutable model facts shared by startup, request preprocessing, and discovery.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// The checkpoint's family, which states what it serves. What a video
    /// family's placed denoiser serves belongs to the loaded workers, whose
    /// startup handshake reports it.
    pub family: Family,
    /// Checkpoint defaults applied only to omitted request fields.
    pub sampling_defaults: SamplingDefaults,
    /// Combined input/output token ceiling: the configured `--max-model-len`,
    /// else the checkpoint's own bound. `build_state` in the crate root passes
    /// it to the workers, and request preprocessing enforces it.
    pub max_model_len: u32,
    /// Tokenizer-config `eos_token` resolved through the vocabulary.
    ///
    /// `special_token_ids` in the crate root places it first in the engine's
    /// EOS list unless the profile's `GenerationControls` carry a nonzero
    /// `eos`, which takes that place instead.
    pub primary_eos_token_id: Option<u32>,
    /// Complete EOS set resolved from tokenizer and generation metadata.
    pub eos_token_ids: BTreeSet<u32>,
}

impl ModelConfig {
    /// Context length of a model whose configuration declares none and for
    /// which no `max_model_len` is configured.
    pub const DEFAULT_MAX_MODEL_LEN: u32 = 8192;
    /// Resolves vocabulary and checkpoint defaults for `family`, the chat
    /// family the checkpoint declares.
    ///
    /// A configured `max_model_len` takes precedence over the checkpoint's
    /// `max_position_embeddings`, and [`ModelConfig::DEFAULT_MAX_MODEL_LEN`]
    /// applies when neither is set.
    ///
    /// # Errors
    ///
    /// Fails when a present metadata file cannot be read or parsed.
    pub fn from_files(
        family: Family,
        files: &ResolvedModelFiles,
        max_model_len: Option<u32>,
        tokenizer: &HuggingFaceTokenizer,
    ) -> assets::Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let (primary_eos_token_id, eos_token_ids) =
            stop_token_ids(&tokenizer_config, &generation_config, tokenizer);

        Ok(Self {
            family,
            sampling_defaults: generation_defaults(&generation_config),
            max_model_len: max_model_len
                .or(model_config.max_position_embeddings())
                .unwrap_or(Self::DEFAULT_MAX_MODEL_LEN),
            primary_eos_token_id,
            eos_token_ids,
        })
    }

    /// Resolves a family that ships as a diffusers pipeline.
    ///
    /// A pipeline checkpoint has no root generation or tokenizer metadata, so
    /// the result carries no sampling defaults and no EOS tokens. The
    /// deployment states `max_model_len`, the longest prompt it admits: a
    /// pipeline keeps no key/value cache, so nothing in the checkpoint implies
    /// one.
    ///
    /// # Errors
    ///
    /// Returns [`assets::Error::UnsupportedPipeline`] for a family that does
    /// not serve video, and [`assets::Error::MissingPromptLength`] when the
    /// deployment states no `max_model_len`.
    pub fn from_pipeline(family: Family, max_model_len: Option<u32>) -> assets::Result<Self> {
        let Serving::Video(_) = family.serving else {
            return Err(assets::Error::UnsupportedPipeline {
                class_name: family.id.to_owned(),
            });
        };
        Ok(Self {
            family,
            sampling_defaults: SamplingDefaults::default(),
            max_model_len: max_model_len.ok_or(assets::Error::MissingPromptLength)?,
            primary_eos_token_id: None,
            eos_token_ids: BTreeSet::new(),
        })
    }

    /// The family's stable identifier.
    pub const fn id(&self) -> &'static str {
        self.family.id
    }

    /// The output formats of a chat family; `None` for a family that serves
    /// no chat.
    pub const fn chat(&self) -> Option<ChatProfile> {
        match self.family.serving {
            Serving::Chat(profile) => Some(profile),
            Serving::Video(_) => None,
        }
    }

    /// The video family; `None` for a family that serves no video.
    pub fn video(&self) -> Option<&'static dyn VideoFamily> {
        match self.family.serving {
            Serving::Video(video) => Some(video),
            Serving::Chat(_) => None,
        }
    }
}

/// Extracts serving generation defaults from repository configuration.
fn generation_defaults(config: &GenerationConfig) -> SamplingDefaults {
    SamplingDefaults {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        min_p: config.min_p,
        repetition_penalty: config.repetition_penalty,
        max_output_tokens: config.max_new_tokens,
    }
}

/// Combines tokenizer and generation metadata into one canonical termination policy.
///
/// Returns the tokenizer-config `eos_token` as the primary id and the union of
/// the generation-config `eos_token_id` values with that primary. An
/// `eos_token` absent from the vocabulary yields no primary rather than an
/// error.
fn stop_token_ids(
    tokenizer_config: &HfTokenizerConfig,
    generation_config: &GenerationConfig,
    tokenizer: &HuggingFaceTokenizer,
) -> (Option<u32>, BTreeSet<u32>) {
    let primary = tokenizer_config
        .special_tokens
        .eos_token
        .as_ref()
        .and_then(|token| tokenizer.token_to_id(token.as_str()));
    let mut ids = generation_config
        .eos_token_id
        .clone()
        .map(|ids| ids.into_set())
        .unwrap_or_default();
    if let Some(primary) = primary {
        ids.insert(primary);
    }
    (primary, ids)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use processing::{
        ChatProfile, Family, ReasoningFormat, Registry, Serving, ToolCallFormat, VideoBinding,
        VideoFamily, VideoProcessor,
    };

    use super::ModelConfig;
    use crate::api::model::assets::ResolvedModelFiles;
    use crate::api::tokenizer::HuggingFaceTokenizer;

    const SPECIAL_TOKENS: &[&str] = &[
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<img>",
        "</img>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ];

    /// Writes a minimal checkpoint for `model_type`: a character-level BPE
    /// tokenizer over ASCII plus [`SPECIAL_TOKENS`], and root, tokenizer, and
    /// generation configs. Dropping the returned directory deletes the files.
    fn configured_files(model_type: &str) -> (tempfile::TempDir, ResolvedModelFiles) {
        let directory = tempdir().expect("create model directory");
        let mut vocab = Vocab::from_iter([("<unk>".to_string(), 0_u32)]);
        for codepoint in 1_u32..=127 {
            vocab.insert(char::from_u32(codepoint).unwrap().to_string(), codepoint);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build tokenizer model");
        let mut tokenizer = TokenizerBuilder::new(model);
        tokenizer.add_special_tokens(
            &SPECIAL_TOKENS
                .iter()
                .map(|token| AddedToken::from(*token, true))
                .collect::<Vec<_>>(),
        );
        let tokenizer_path = directory.path().join("tokenizer.json");
        tokenizer
            .save(&tokenizer_path, false)
            .expect("save tokenizer");

        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            format!(
                r#"{{"model_type":"{model_type}","max_position_embeddings":4096,"num_attention_heads":8}}"#
            ),
        )
        .expect("write model config");
        let tokenizer_config_path = directory.path().join("tokenizer_config.json");
        fs::write(&tokenizer_config_path, r#"{"eos_token":"<|im_end|>"}"#)
            .expect("write tokenizer config");
        let generation_config_path = directory.path().join("generation_config.json");
        fs::write(
            &generation_config_path,
            r#"{"eos_token_id":2,"temperature":0.6,"top_p":0.95,"top_k":20,"max_new_tokens":512}"#,
        )
        .expect("write generation config");

        let files = ResolvedModelFiles {
            tokenizer_path,
            tokenizer_config_path: Some(tokenizer_config_path),
            generation_config_path: Some(generation_config_path),
            chat_template_path: None,
            config_path: Some(config_path),
        };
        (directory, files)
    }

    /// The family the checkpoint [`configured_files`] writes for `model_type` declares, as
    /// startup loads it with a registry of [`CHAT`] and [`VIDEO`].
    async fn loaded(
        model_type: &str,
    ) -> Result<ModelConfig, crate::api::input::ModelResolutionError> {
        let (directory, _files) = configured_files(model_type);
        let registry = Registry::new([CHAT, VIDEO]).unwrap();
        let loaded = ModelConfig::load(
            directory.path().to_str().unwrap(),
            None,
            &proto::config::ChatConfig::default(),
            &registry,
        )
        .await;
        loaded.map(|loaded| loaded.config)
    }

    /// A chat family of the two `model_type`s these checkpoints declare.
    const CHAT: Family = Family {
        id: "chat",
        model_types: &["qwen3", "qwen3_moe"],
        classes: &[],
        serving: Serving::Chat(ChatProfile {
            reasoning: Some(ReasoningFormat::Think),
            tool_calls: Some(ToolCallFormat::Json),
            images: None,
        }),
    };

    /// A video family that binds nothing: these tests read only how it is resolved.
    struct Video;

    impl VideoFamily for Video {
        fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
            &[]
        }

        fn default_sizes(&self) -> &'static [&'static str] {
            &["768x768"]
        }

        fn default_max_seconds(&self) -> f64 {
            4.0
        }

        fn bind(&self, _: VideoBinding<'_>) -> Result<Box<dyn VideoProcessor>, String> {
            Err("this family binds no processor".to_owned())
        }
    }

    const VIDEO: Family = Family {
        id: "video",
        model_types: &["video"],
        classes: &["VideoModularPipeline"],
        serving: Serving::Video(&Video),
    };

    /// A chat checkpoint's facts are the files it publishes: its context bound and its
    /// generation defaults.
    #[test]
    fn a_chat_checkpoint_states_its_bound_and_defaults() {
        let (_directory, files) = configured_files("qwen3");
        let tokenizer = HuggingFaceTokenizer::load(
            &files.tokenizer_path,
            files.tokenizer_config_path.as_deref(),
        )
        .unwrap();
        let profile = ModelConfig::from_files(CHAT, &files, None, &tokenizer).unwrap();
        assert_eq!(profile.family, CHAT);
        assert_eq!(profile.max_model_len, 4096);
        assert_eq!(profile.sampling_defaults.max_output_tokens, Some(512));

        let bounded = ModelConfig::from_files(CHAT, &files, Some(1024), &tokenizer).unwrap();
        assert_eq!(bounded.max_model_len, 1024);
    }

    /// Startup serves the family the checkpoint declares; a video family declared by a root
    /// configuration with no pipeline index beside it, and a `model_type` no family claims, are
    /// refused.
    #[tokio::test]
    async fn startup_serves_the_family_the_checkpoint_declares() {
        for model_type in ["qwen3", "qwen3_moe"] {
            assert_eq!(loaded(model_type).await.unwrap().family, CHAT);
        }
        assert!(matches!(
            loaded("video").await,
            Err(crate::api::input::ModelResolutionError::NoPipeline { family: "video" })
        ));
        assert!(matches!(
            loaded("unsupported_architecture").await,
            Err(crate::api::input::ModelResolutionError::Family(
                processing::ResolveError::Unserved { .. }
            ))
        ));
    }

    /// A pipeline admits the prompt length its deployment states, and a
    /// deployment that states none is refused: no checkpoint figure implies
    /// one.
    #[test]
    fn a_pipeline_admits_the_prompt_length_its_deployment_states() {
        let profile = ModelConfig::from_pipeline(VIDEO, Some(2048)).unwrap();
        assert_eq!(profile.family, VIDEO);
        assert_eq!(profile.max_model_len, 2048);
        assert!(matches!(
            ModelConfig::from_pipeline(VIDEO, None),
            Err(super::assets::Error::MissingPromptLength)
        ));
        assert!(ModelConfig::from_pipeline(CHAT, Some(2048)).is_err());
    }
}
