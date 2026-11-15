// The served checkpoint's files and JSON metadata, and the model facts startup resolves.
//
// [`assets`] resolves where a checkpoint lies (a Hugging Face repository's snapshot or a local
// directory) and reads the files its family declares. The server identifies the served model
// family from the checkpoint itself, by what its root `config.json` or, for a diffusers pipeline,
// its root index declares (`processing::Registry::resolve`). The result is one [`ModelConfig`],
// built at startup by `ModelConfig::load` in `input::processor`. The leader's startup reads it,
// and `Processor::new` then binds the worker's capabilities onto it before request preprocessing
// and discovery share it immutably.
//
// Which family a checkpoint is, and what that family serves, is the executable's registry of
// families (`processing::Registry`), whose request processing reads the checkpoint's tokenizer
// (`crate::api::tokenizer`) through `processing::Vocabulary`.

pub mod assets;

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use processing::Family;
use processing::{ChatProfile, Serving, VideoFamily};

use crate::api::model::assets::{
    ChatFiles, GenerationConfig, HfTokenizerConfig, OneOrManyTokenIds, load_generation_config,
    load_tokenizer_config,
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
    /// else the bound the checkpoint states where its family declares it. The
    /// leader passes it to the workers, and request preprocessing enforces it.
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
    /// Resolves vocabulary and checkpoint defaults for `family`, the chat
    /// family the checkpoint declares, from its `files`.
    ///
    /// A configured `max_model_len` takes precedence over the context length
    /// the checkpoint states where its family declares it
    /// ([`ChatFiles::context_length`]); a checkpoint that states no generation
    /// defaults leaves every sampling default unset.
    ///
    /// # Errors
    ///
    /// Fails when a metadata file cannot be read or parsed, or no
    /// `max_model_len` is configured and the checkpoint states no context
    /// length.
    pub fn from_files(
        family: Family,
        files: &ChatFiles,
        max_model_len: Option<u32>,
        tokenizer: &HuggingFaceTokenizer,
    ) -> assets::Result<Self> {
        let generation_config = files
            .generation_config()
            .map(load_generation_config)
            .transpose()?;
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config())?;
        let (primary_eos_token_id, eos_token_ids) =
            stop_token_ids(&tokenizer_config, generation_config.as_ref(), tokenizer);

        Ok(Self {
            family,
            sampling_defaults: generation_config
                .as_ref()
                .map(generation_defaults)
                .unwrap_or_default(),
            max_model_len: match max_model_len {
                Some(configured) => configured,
                None => files.context_length()?,
            },
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

    /// The generation bounds the model's family states before its workers report their own,
    /// which the engine lowers to its workers' capacities (each limit is the smaller of the
    /// two).
    ///
    /// No family states a capacity of its own: a text family bounds no position's candidates
    /// (its workers' samplers do), and a family that takes images bounds none of their
    /// encodings (its workers' feature storage and encoder cache do), which the engine's
    /// interface states as the largest value of each limit. A family that takes no images
    /// states none of their limits.
    pub fn generation_limits(&self) -> proto::request::GenerationLimits {
        let mut limits = proto::request::GenerationLimits::default();
        if self.chat().is_some() {
            limits.max_logprobs = u32::MAX;
        }
        if self.chat().is_some_and(|profile| profile.images.is_some()) {
            limits.max_vision_feature_bytes = u64::MAX;
            limits.max_vit_grid_tokens = u32::MAX;
            limits.encoder_cache_entries = u32::MAX;
        }
        limits
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
/// the generation-config `eos_token_id` values, when the checkpoint states
/// generation defaults, with that primary. An `eos_token` absent from the
/// vocabulary yields no primary rather than an error.
fn stop_token_ids(
    tokenizer_config: &HfTokenizerConfig,
    generation_config: Option<&GenerationConfig>,
    tokenizer: &HuggingFaceTokenizer,
) -> (Option<u32>, BTreeSet<u32>) {
    let primary = tokenizer_config
        .special_tokens
        .eos_token
        .as_ref()
        .and_then(|token| tokenizer.token_to_id(token.as_str()));
    let mut ids = generation_config
        .and_then(|config| config.eos_token_id.clone())
        .map(OneOrManyTokenIds::into_set)
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
        BasePin, ChatCheckpoint, ChatProfile, ChatTemplate, Family, ReasoningFormat, Registry,
        RootFile, Serving, ToolCallFormat, VideoBinding, VideoFamily, VideoProcessor,
    };

    use super::ModelConfig;
    use crate::api::input::ModelResolutionError;
    use crate::api::model::assets::{ChatFiles, Hub, HubSettings, Source};
    use crate::api::tokenizer::HuggingFaceTokenizer;

    const SPECIAL_TOKENS: &[&str] = &["<|start|>", "<|end|>", "<think>", "</think>"];

    /// Where [`CHAT`]'s checkpoints keep what request processing reads.
    const CHECKPOINT: ChatCheckpoint = ChatCheckpoint {
        context_length: &["max_position_embeddings"],
        chat_template: ChatTemplate::TokenizerConfig,
    };

    /// Writes a minimal checkpoint of the model class `class`: a character-level BPE
    /// tokenizer over ASCII plus [`SPECIAL_TOKENS`], and root, tokenizer, and
    /// generation configs. Dropping the returned directory deletes the files.
    fn checkpoint(class: &str) -> tempfile::TempDir {
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
        tokenizer
            .save(directory.path().join("tokenizer.json"), false)
            .expect("save tokenizer");

        fs::write(
            directory.path().join("config.json"),
            format!(r#"{{"architectures":["{class}"],"max_position_embeddings":4096}}"#),
        )
        .expect("write model config");
        fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"eos_token":"<|end|>","chat_template":"{{ messages }}"}"#,
        )
        .expect("write tokenizer config");
        fs::write(
            directory.path().join("generation_config.json"),
            r#"{"eos_token_id":2,"temperature":0.6,"top_p":0.95,"top_k":20,"max_new_tokens":512}"#,
        )
        .expect("write generation config");
        directory
    }

    /// A Hub that reaches no network: these tests state local checkpoints.
    fn offline_hub(cache: &std::path::Path) -> Hub {
        Hub::new(HubSettings {
            cache: cache.to_path_buf(),
            endpoint: None,
            token: None,
            offline: true,
        })
        .unwrap()
    }

    /// The family the checkpoint [`checkpoint`] writes for `class` declares, as startup loads
    /// it with a registry of [`CHAT`] and [`VIDEO`].
    async fn loaded(class: &str) -> Result<ModelConfig, ModelResolutionError> {
        let (directory, cache) = (checkpoint(class), tempdir().unwrap());
        let registry = Registry::new([CHAT, VIDEO]).unwrap();
        let loaded = ModelConfig::load(
            &offline_hub(cache.path()),
            &Source::Directory(directory.path().to_path_buf()),
            None,
            None,
            &proto::config::ChatConfig::default(),
            &serde_json::Map::new(),
            &registry,
        )
        .await;
        loaded.map(|loaded| loaded.config)
    }

    /// A chat family of the class these checkpoints declare.
    const CHAT: Family = Family {
        id: "chat",
        root: RootFile::Config,
        classes: &["ChatForCausalLM"],
        serving: Serving::Chat(ChatProfile {
            reasoning: Some(ReasoningFormat {
                open: "<think>",
                close: "</think>",
            }),
            tool_calls: Some(ToolCallFormat::Json),
            images: None,
            checkpoint: CHECKPOINT,
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

        fn processing_parts(&self) -> &'static [&'static str] {
            &[]
        }

        fn base_pin(&self) -> Option<BasePin> {
            None
        }

        fn bind(&self, _: VideoBinding<'_>) -> Result<Box<dyn VideoProcessor>, String> {
            Err("this family binds no processor".to_owned())
        }
    }

    const VIDEO: Family = Family {
        id: "video",
        root: RootFile::ModularIndex,
        classes: &["VideoModularPipeline"],
        serving: Serving::Video(&Video),
    };

    /// A chat checkpoint's facts are the files it publishes: its context bound and its
    /// generation defaults; one that publishes no generation defaults states none.
    #[test]
    fn a_chat_checkpoint_states_its_bound_and_defaults() {
        let directory = checkpoint("ChatForCausalLM");
        let files = ChatFiles::locate(directory.path(), CHECKPOINT).unwrap();
        let tokenizer = HuggingFaceTokenizer::load(files.tokenizer_directory()).unwrap();
        let profile = ModelConfig::from_files(CHAT, &files, None, &tokenizer).unwrap();
        assert_eq!(profile.family, CHAT);
        assert_eq!(profile.max_model_len, 4096);
        assert_eq!(profile.sampling_defaults.max_output_tokens, Some(512));

        let bounded = ModelConfig::from_files(CHAT, &files, Some(1024), &tokenizer).unwrap();
        assert_eq!(bounded.max_model_len, 1024);

        fs::remove_file(directory.path().join("generation_config.json")).unwrap();
        let files = ChatFiles::locate(directory.path(), CHECKPOINT).unwrap();
        let undefaulted = ModelConfig::from_files(CHAT, &files, None, &tokenizer).unwrap();
        assert_eq!(
            undefaulted.sampling_defaults,
            super::SamplingDefaults::default()
        );
        assert_eq!(
            undefaulted.eos_token_ids,
            [tokenizer.token_to_id("<|end|>").unwrap()].into()
        );
    }

    /// Startup serves the family the checkpoint declares; a video family's class named in a
    /// root configuration rather than its pipeline index, and a class no family claims, are
    /// refused.
    #[tokio::test]
    async fn startup_serves_the_family_the_checkpoint_declares() {
        assert_eq!(loaded("ChatForCausalLM").await.unwrap().family, CHAT);
        assert!(matches!(
            loaded("VideoModularPipeline").await,
            Err(ModelResolutionError::Assets(super::assets::Error::Family(
                processing::ResolveError::Unserved { .. }
            )))
        ));
        assert!(matches!(
            loaded("unsupported_architecture").await,
            Err(ModelResolutionError::Assets(super::assets::Error::Family(
                processing::ResolveError::Unserved { .. }
            )))
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
