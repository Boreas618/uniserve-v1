// The served checkpoint's files, and the model facts startup resolves.
//
// [`assets`] resolves where a checkpoint lies (a Hugging Face repository's snapshot or a local
// directory) and reads the files its family declares. The server identifies the served model
// family from the checkpoint itself, by what its root `config.json` or, for a diffusers pipeline,
// its root index declares, among the families the epoch's processing program declares
// (`crate::api::program::Program::resolve`). The result is one [`ModelConfig`], built at startup
// by `ModelConfig::load` in `input::processor` from what the program's binding of the family
// states. The leader's startup reads it, and `Processor::new` then binds the worker's
// capabilities onto it before request preprocessing and discovery share it immutably.

pub mod assets;

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use scheduling::processing::video::Pipeline;
use scheduling::processing::{ChatFacts, Family, Serving};

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

/// What a chat family's request processing parses and takes, as its binding states it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChatFeatures {
    /// Whether the family's tool calls are parsed from its output.
    pub tools: bool,
    /// Whether the family delimits reasoning in its output.
    pub reasoning: bool,
    /// Whether the family takes image parts.
    pub images: bool,
}

/// Immutable model facts shared by startup, request preprocessing, and discovery.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// The checkpoint's family, as the epoch's processing program declares it: what it serves.
    /// What a video family's placed denoiser serves belongs to the loaded workers, whose
    /// startup handshake reports it.
    pub family: Family,
    /// Checkpoint defaults applied only to omitted request fields.
    pub sampling_defaults: SamplingDefaults,
    /// Combined input/output token ceiling: the configured `--max-model-len`,
    /// else the bound the checkpoint states where its family keeps it. The
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
    /// What a chat family parses and takes; `None` for a family that serves no chat.
    pub chat: Option<ChatFeatures>,
}

impl ModelConfig {
    /// The facts of a chat `family` its binding states (`facts`).
    pub fn of_chat(family: Family, facts: &ChatFacts) -> Self {
        let sampling = &facts.sampling;
        Self {
            family,
            sampling_defaults: SamplingDefaults {
                temperature: sampling.temperature,
                top_p: sampling.top_p,
                top_k: sampling.top_k,
                min_p: sampling.min_p,
                repetition_penalty: sampling.repetition_penalty,
                max_output_tokens: sampling.max_output_tokens,
            },
            max_model_len: facts.max_model_len,
            primary_eos_token_id: facts.primary_eos,
            eos_token_ids: facts.eos.iter().copied().collect(),
            chat: Some(ChatFeatures {
                tools: facts.tools,
                reasoning: facts.reasoning,
                images: facts.images,
            }),
        }
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
                class_name: family.id,
            });
        };
        Ok(Self {
            family,
            sampling_defaults: SamplingDefaults::default(),
            max_model_len: max_model_len.ok_or(assets::Error::MissingPromptLength)?,
            primary_eos_token_id: None,
            eos_token_ids: BTreeSet::new(),
            chat: None,
        })
    }

    /// The family's stable identifier.
    pub fn id(&self) -> &str {
        &self.family.id
    }

    /// The video family's pipeline; `None` for a family that serves no video.
    pub fn video(&self) -> Option<&Pipeline> {
        match &self.family.serving {
            Serving::Video(pipeline) => Some(pipeline),
            Serving::Chat { .. } => None,
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
        if self.chat.is_some() {
            limits.max_logprobs = u32::MAX;
        }
        if self.chat.is_some_and(|chat| chat.images) {
            limits.max_vision_feature_bytes = u64::MAX;
            limits.max_vit_grid_tokens = u32::MAX;
            limits.encoder_cache_entries = u32::MAX;
        }
        limits
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use scheduling::processing::ResolveError;
    use tempfile::tempdir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::ModelConfig;
    use crate::api::input::ModelResolutionError;
    use crate::api::model::assets::{Hub, HubSettings, Source};

    const SPECIAL_TOKENS: &[&str] = &["<|start|>", "<|end|>", "<think>", "</think>"];

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

    /// The model facts startup loads the checkpoint in `directory` with, through the tests'
    /// processing program, under a configured `max_model_len`.
    async fn loaded(
        directory: &std::path::Path,
        max_model_len: Option<u32>,
    ) -> Result<ModelConfig, ModelResolutionError> {
        let cache = tempdir().unwrap();
        ModelConfig::load(
            &offline_hub(cache.path()),
            &Source::Directory(directory.to_path_buf()),
            None,
            max_model_len.and_then(std::num::NonZeroU32::new),
            &proto::config::ChatConfig::default(),
            &serde_json::Map::new(),
            &crate::api::test_support::program(),
        )
        .await
        .map(|loaded| loaded.config)
    }

    /// A chat checkpoint resolves to the program's family its configuration declares, whose
    /// binding states the checkpoint's facts: its context bound, its generation defaults and
    /// its EOS ids; a deployment's `max_model_len` takes the bound's place, and a checkpoint that
    /// publishes no generation defaults states none.
    #[tokio::test]
    async fn a_chat_checkpoint_binds_to_the_facts_its_files_state() {
        let directory = checkpoint("ChatForCausalLM");
        let config = loaded(directory.path(), None).await.unwrap();
        assert_eq!(config.id(), "chat");
        assert_eq!(config.max_model_len, 4096);
        assert_eq!(config.sampling_defaults.max_output_tokens, Some(512));
        assert_eq!(config.sampling_defaults.temperature, Some(0.6));
        let chat = config.chat.expect("a chat family");
        assert!(chat.tools && chat.reasoning && !chat.images);

        assert_eq!(
            loaded(directory.path(), Some(1024))
                .await
                .unwrap()
                .max_model_len,
            1024
        );

        fs::remove_file(directory.path().join("generation_config.json")).unwrap();
        let undefaulted = loaded(directory.path(), None).await.unwrap();
        assert_eq!(
            undefaulted.sampling_defaults,
            super::SamplingDefaults::default()
        );
        // `<|end|>` follows `<unk>` and the 127 ASCII tokens in the vocabulary.
        assert_eq!(undefaulted.eos_token_ids, [129].into());
        assert_eq!(undefaulted.primary_eos_token_id, Some(129));
    }

    /// A video family's class named in a root configuration rather than its pipeline index, and
    /// a class no family of the program claims, are refused, naming what the program serves.
    #[tokio::test]
    async fn a_checkpoint_no_family_of_the_program_claims_is_refused() {
        for class in ["TestVideoPipeline", "UnsupportedForCausalLM"] {
            let directory = checkpoint(class);
            let refused = loaded(directory.path(), None).await;
            let Err(ModelResolutionError::Assets(super::assets::Error::Family(
                ResolveError::Unserved { served, .. },
            ))) = refused
            else {
                panic!("{class} is refused: {refused:?}");
            };
            assert!(served.contains("ChatForCausalLM"), "{served}");
        }
    }

    /// A pipeline admits the prompt length its deployment states, and a deployment that states
    /// none is refused: no checkpoint figure implies one; a chat family has no pipeline form.
    #[test]
    fn a_pipeline_admits_the_prompt_length_its_deployment_states() {
        let program = crate::api::test_support::program();
        let family = |id: &str| {
            program
                .families()
                .iter()
                .find(|family| family.id == id)
                .cloned()
                .unwrap()
        };
        let profile = ModelConfig::from_pipeline(family("video"), Some(2048)).unwrap();
        assert_eq!(profile.id(), "video");
        assert_eq!(profile.max_model_len, 2048);
        assert!(profile.video().is_some());
        assert!(matches!(
            ModelConfig::from_pipeline(family("video"), None),
            Err(super::assets::Error::MissingPromptLength)
        ));
        assert!(ModelConfig::from_pipeline(family("chat"), Some(2048)).is_err());
    }
}
