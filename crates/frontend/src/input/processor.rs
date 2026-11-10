//! Load-time model resolution and model-owned request tokenization.
//!
//! `ModelConfig::load` resolves checkpoint facts, the tokenizer, and what renders requests
//! at startup, by the family of the executable's catalogue the checkpoint declares: a diffusers
//! pipeline index selects a video family, whose processor reads the pipeline files the family
//! names, and otherwise the root `config.json` selects a chat family with its chat template. [`InputProcessor::new`] then binds the
//! generation limits the started workers report and the request
//! renderer they complete ([`RequestRenderer`]), and the result is shared immutably by the
//! serving runtime.
//!
//! [`InputProcessor`] binds tokenizer assets, generation limits, sampling policy, and
//! output processing. [`InputProcessor::preprocess`] tokenizes a `GenerateRequest` into a
//! [`GenerationRequest`] and the `OutputParams` (`crate::output`) the daemon retains.
//! A video deployment's `VideoService` produces its `DiffusionRequest`s.

use std::sync::Arc;

use crate::model::ModelConfig;
use crate::model::assets::{PipelineCheckpoint, ResolvedModelFiles};
use crate::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use processing::Catalogue;
use processing::{
    ImageFamily, ImageProcessor, ReasoningFormat, Serving, ToolCallFormat, VideoFamily,
};
use proto::request::{CachePolicy, GenerationFeatures, GenerationLimits, GenerationRequest};
use proto::request::{RequestId, SamplingParams as EngineSamplingParams};
use thiserror::Error;

use crate::input::chat::template::MultimodalRenderInfo;
use crate::input::chat::{ChatRequest, HfChatRenderer};
use crate::input::request::GenerateRequest;
use crate::input::sampling::resolve_max_tokens;
use crate::input::video::VideoService;
use crate::input::{Result, ServeError};
use crate::output::OutputParams;
use crate::output::chat::ChatOutputProcessor;

/// Owns the tokenizer and template resources used to produce final engine inputs.
/// Runtime capabilities are fixed at construction, before this value is shared.
///
/// Preprocessing methods are synchronous and CPU-bound; the daemon's served model runs its request
/// preprocessing on the blocking pool.
pub struct InputProcessor {
    pub(super) config: ModelConfig,
    pub(super) tokenizer: DynTokenizer,
    renderer: RequestRenderer,
    // Worker-advertised generation features and resource limits.
    pub(super) limits: GenerationLimits,
    pub(super) parse_reasoning: bool,
    // Whether the chat family writes tool calls and its checkpoint's template
    // asks for them in that format.
    tool_calls: bool,
    // The family's image processor, when the family takes images and the workers encode them.
    images: Option<Arc<dyn ImageProcessor>>,
}

#[derive(Debug, Error)]
/// Failure while resolving a configured model and its assets.
pub enum ModelResolutionError {
    /// A file a video family's processor reads cannot be read.
    #[error("reading {}", path.display())]
    ProcessorFile {
        /// The file.
        path: std::path::PathBuf,
        /// The read's failure.
        #[source]
        source: std::io::Error,
    },
    /// Model files or profile metadata cannot be resolved.
    #[error(transparent)]
    Assets(#[from] crate::model::assets::Error),
    /// The model tokenizer cannot be loaded.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    /// The model chat renderer cannot be initialized.
    #[error(transparent)]
    Chat(#[from] crate::output::chat::Error),
    /// The checkpoint's image processor files are not its family's.
    #[error("binding the image processor: {0}")]
    ImageProcessor(String),
    /// The running worker lacks a feature required by the selected profile.
    #[error("configured model description `{description}` requires worker feature `{feature}`")]
    MissingFeature {
        /// Stable model-profile identifier.
        description: &'static str,
        /// Required runtime feature.
        feature: GenerationFeatures,
    },
}

/// The model facts and preprocessing resources startup resolves.
pub struct LoadedModel {
    /// Immutable model facts.
    pub config: ModelConfig,
    /// The prompt tokenizer.
    pub tokenizer: DynTokenizer,
    /// What renders the checkpoint's requests once its engine has started.
    pub renderer: LoadedRenderer,
}

/// What renders a loaded checkpoint's requests, fixed by its layout.
pub enum LoadedRenderer {
    /// A token model's chat template, and the image processor of a family that takes images.
    Chat {
        renderer: Box<HfChatRenderer>,
        /// The family's image processor, bound to the checkpoint's processor files; `None` for
        /// a family that serves text alone.
        images: Option<Arc<dyn ImageProcessor>>,
    },
    /// A video family and the contents of the pipeline files its processor
    /// reads; the processor is bound from the denoiser handshake.
    Video {
        /// The family.
        family: &'static dyn VideoFamily,
        /// The files [`VideoFamily::processor_files`] names, in that order.
        files: Vec<String>,
    },
}

/// What renders the served model's API requests into engine requests.
#[expect(
    clippy::large_enum_variant,
    reason = "a deployment builds one renderer and keeps it for its lifetime; boxing the template \
              saves nothing"
)]
pub enum RequestRenderer {
    /// A token model's chat template.
    Chat(HfChatRenderer),
    /// The video service of a video pipeline, bound to what its denoiser
    /// serves.
    Video(VideoService),
}

/// Loads a diffusers pipeline's `tokenizer` component as the pipeline's own
/// `transformers` tokenizer loads it: `tokenizer.json` with the special
/// tokens its `tokenizer_config.json`, when published, declares.
///
/// The conditioner reads token ids, so a declared token missing here would
/// split into text tokens the reference never presents (MiniMax-H3 declares
/// its `<d>` dialogue marker only in the configuration).
pub async fn pipeline_tokenizer(
    pipeline: &PipelineCheckpoint,
) -> std::result::Result<HuggingFaceTokenizer, ModelResolutionError> {
    let path = pipeline
        .component_file("tokenizer", "tokenizer.json")
        .await?;
    let config = match pipeline
        .component_file("tokenizer", "tokenizer_config.json")
        .await
    {
        Ok(config) => Some(config),
        Err(crate::model::assets::Error::MissingFile { .. }) => None,
        Err(error) => return Err(error.into()),
    };
    let tokenizer = HuggingFaceTokenizer::load(&path, config.as_deref())?;
    Ok(tokenizer)
}

impl ModelConfig {
    /// Loads the checkpoint `model` names: its facts and the tokenizer and
    /// template resources request processing needs, with the family of
    /// `catalogue` that claims it.
    ///
    /// A diffusers pipeline has no chat template and carries its conditioner's
    /// vision processor geometry instead. The returned `max_model_len` is the
    /// configured `max_model_len`, else, for a root configuration, its
    /// `max_position_embeddings` or `ModelConfig::DEFAULT_MAX_MODEL_LEN`; a
    /// pipeline's deployment must state it. `chat` configures a chat
    /// checkpoint's template.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint's files cannot be resolved or read, or no
    /// family of `catalogue` serves it.
    pub async fn load(
        model: &str,
        max_model_len: Option<std::num::NonZeroU32>,
        chat: &proto::config::ChatConfig,
        catalogue: &Catalogue,
    ) -> std::result::Result<LoadedModel, ModelResolutionError> {
        let max_model_len = max_model_len.map(std::num::NonZeroU32::get);
        // A diffusers pipeline declares its class and component folders in a
        // root index rather than a root `config.json`, so the index selects the
        // family and locates the tokenizer component before any other asset
        // is resolved. A component export reads the components its pinned
        // base supplies from the Hub cache at the pinned revision, as its
        // workers do. A family that a root `config.json` describes is
        // selected by that file alone, even when the checkpoint publishes an
        // index beside it.
        if let Some(pipeline) = PipelineCheckpoint::resolve(model).await?
            && let Some(family) = catalogue.by_class(pipeline.class_name())
            && let Serving::Video(video) = family.serving
        {
            let tokenizer: DynTokenizer = Arc::new(pipeline_tokenizer(&pipeline).await?);
            let mut files = Vec::with_capacity(video.processor_files().len());
            for (component, name) in video.processor_files() {
                let path = pipeline.component_file(component, name).await?;
                let text = std::fs::read_to_string(&path).map_err(|source| {
                    ModelResolutionError::ProcessorFile {
                        path: path.clone(),
                        source,
                    }
                })?;
                files.push(text);
            }
            return Ok(LoadedModel {
                config: Self::from_pipeline(family, max_model_len)?,
                tokenizer,
                renderer: LoadedRenderer::Video {
                    family: video,
                    files,
                },
            });
        }

        let files = ResolvedModelFiles::new(model).await?;
        let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::load(
            &files.tokenizer_path,
            files.tokenizer_config_path.as_deref(),
        )?);
        let config = Self::from_files(catalogue, &files, max_model_len, tokenizer.as_ref()).await?;
        // A family that takes images binds its processor to the checkpoint's processor files; a
        // template that reads content parts writes its own image markup, and one that reads
        // flattened content is given the family's.
        let images = match config.chat().and_then(|profile| profile.images) {
            Some(family) => Some(bind_images(model, family).await?),
            None => None,
        };
        let multimodal = images.as_ref().map(|images| MultimodalRenderInfo {
            placeholder_token: images.markup().to_owned(),
        });
        let renderer = HfChatRenderer::load(&files, chat, multimodal)?;
        Ok(LoadedModel {
            config,
            tokenizer,
            renderer: LoadedRenderer::Chat {
                renderer: Box::new(renderer),
                images,
            },
        })
    }
}

/// Binds a chat family's image processor to the processor files of the checkpoint `model` names.
async fn bind_images(
    model: &str,
    family: &'static dyn ImageFamily,
) -> std::result::Result<Arc<dyn ImageProcessor>, ModelResolutionError> {
    let source = crate::model::assets::model_files::ModelSource::from_model_id(model)?;
    let mut files = Vec::with_capacity(family.processor_files().len());
    for name in family.processor_files() {
        let path = source.file(name).await?;
        let text = std::fs::read_to_string(&path)
            .map_err(|source| ModelResolutionError::ProcessorFile { path, source })?;
        files.push(text);
    }
    family
        .bind(&files)
        .map(Arc::from)
        .map_err(ModelResolutionError::ImageProcessor)
}

impl InputProcessor {
    /// The model's tokenizer, which a process continuing a response builds
    /// its incremental decoder from.
    pub fn tokenizer(&self) -> DynTokenizer {
        Arc::clone(&self.tokenizer)
    }

    /// Binds model resources to the generation limits the started workers
    /// report in their handshake, without rebuilding model data.
    ///
    /// `renderer` is the checkpoint's chat template, or the video service a video pipeline's
    /// denoiser handshake built.
    ///
    /// # Errors
    ///
    /// Returns [`ModelResolutionError::MissingFeature`] when `limits` do not cover the features
    /// the configured model needs.
    pub fn new(
        config: ModelConfig,
        tokenizer: DynTokenizer,
        renderer: RequestRenderer,
        limits: GenerationLimits,
        parse_reasoning: bool,
    ) -> std::result::Result<Self, ModelResolutionError> {
        let needs = match config.chat() {
            Some(_) => GenerationFeatures::UNDERSTANDING,
            None => GenerationFeatures::empty(),
        };
        validate_runtime_features(&config, &limits, needs)?;

        // A chat checkpoint serves tool calls only when its family writes them
        // and its template asks for them in that format; otherwise requests
        // with tools are refused rather than returned with unparsed calls.
        let tool_calls = match (config.chat(), &renderer) {
            (Some(profile), RequestRenderer::Chat(renderer)) => match profile.tool_calls {
                Some(ToolCallFormat::Qwen3) => {
                    crate::input::chat::template::template_instructs_json_tool_calls(renderer)
                }
                None => false,
            },
            _ => false,
        };
        Ok(Self {
            config,
            tokenizer,
            renderer,
            limits,
            parse_reasoning,
            tool_calls,
            images: None,
        })
    }

    /// Takes image parts through `images`, the family's bound image processor, when the workers
    /// encode images ([`GenerationFeatures::VISION_ENCODE`]); a deployment whose workers do not
    /// keeps refusing them.
    #[must_use]
    pub fn with_images(mut self, images: Option<Arc<dyn ImageProcessor>>) -> Self {
        self.images = images.filter(|_| {
            self.limits
                .covers(GenerationFeatures::VISION_ENCODE)
                .is_ok()
        });
        self
    }

    /// Immutable facts of the loaded model.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The chat template of a token model; `None` for a video pipeline.
    pub(super) fn chat_renderer(&self) -> Option<&HfChatRenderer> {
        match &self.renderer {
            RequestRenderer::Chat(renderer) => Some(renderer),
            RequestRenderer::Video(_) => None,
        }
    }

    /// The video service of a video deployment; `None` for other models.
    pub fn video_service(&self) -> Option<&VideoService> {
        match &self.renderer {
            RequestRenderer::Video(video) => Some(video),
            RequestRenderer::Chat(_) => None,
        }
    }

    /// The served video contract, as `GET /v1/capabilities` reports it;
    /// `Value::Null` for a model that serves no video. See
    /// `VideoService::capabilities`.
    pub fn video_capabilities(&self) -> serde_json::Value {
        self.video_service()
            .map_or(serde_json::Value::Null, |video| {
                video.capabilities(self.config.max_model_len)
            })
    }

    /// Returns whether the model supports image input: its family takes images and its workers
    /// encode them.
    pub fn supports_image_input(&self) -> bool {
        self.images.is_some()
    }

    /// Returns whether the model parses tool calls from its output.
    pub fn supports_tool_calling(&self) -> bool {
        self.tool_calls
    }

    /// Returns whether the model produces structured reasoning output.
    pub fn supports_reasoning(&self) -> bool {
        self.config
            .chat()
            .is_some_and(|profile| profile.reasoning.is_some())
    }

    /// Validates the prompt and requested modalities against the loaded model.
    ///
    /// Every refusal is `ServeError::UnsupportedFeature`. The final check asks the worker
    /// limits to cover the features this request reaches.
    fn validate_generation_features(
        &self,
        request_id: &str,
        prompt: &ChatRequest,
        has_input_image: bool,
        image_output: bool,
    ) -> Result<()> {
        let reject = |feature| ServeError::UnsupportedFeature {
            request_id: request_id.to_owned(),
            feature,
        };
        if self.config.chat().is_none() {
            return Err(reject("generation_endpoint"));
        }
        if has_input_image && !self.supports_image_input() {
            return Err(reject("image_input"));
        }
        if image_output {
            return Err(reject("image_output"));
        }
        if prompt.uses_tools() && !self.supports_tool_calling() {
            return Err(reject("tool_calling"));
        }
        if prompt.chat_options.reasoning_effort.is_some() && !self.supports_reasoning() {
            return Err(reject("reasoning"));
        }
        self.limits
            .covers(GenerationFeatures::UNDERSTANDING)
            .map_err(|feature| reject(feature.name()))
    }

    /// Tokenizes one generation request into its engine request and the
    /// output parameters the daemon keeps.
    ///
    /// `images` are the resolved `image_url` parts of a chat prompt, in
    /// `ChatRequest::image_urls` order; a text prompt has none.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::UnsupportedFeature` from feature validation and
    /// `ServeError::Tokenize` for every
    /// failure during tokenization, validation, and resource sizing.
    pub fn preprocess(
        &self,
        request: GenerateRequest,
        images: Vec<crate::input::ImageInput>,
    ) -> Result<(GenerationRequest, OutputParams)> {
        let GenerateRequest {
            request_id,
            prompt,
            image_output,
            sampling,
            max_tokens,
            stop_token_ids,
            decode,
            priority,
            cache_salt,
            adapter,
        } = request;
        let has_input_image = !images.is_empty() || prompt.has_multimodal();
        self.validate_generation_features(&request_id, &prompt, has_input_image, image_output)?;
        let mut generation = GenerationRequest {
            // The daemon's submission stamps the engine's identifier for the request.
            request_id: RequestId::default(),
            prompt_token_ids: Vec::new(),
            multimodal_inputs: Default::default(),
            sampling,
            max_und_tokens: 0,
            include_stop_token: decode.include_stop_str_in_output,
            stop_strings: decode.stop_strings.clone().unwrap_or_default(),
            stop_token_ids,
            priority,
            cache: CachePolicy {
                isolation_key: crate::input::cache_isolation_key(
                    cache_salt.as_deref(),
                    adapter.as_ref().map(|adapter| adapter.identity.as_str()),
                ),
                ..CachePolicy::default()
            },
            generated_token_ids: Vec::new(),
            adapter: adapter.as_ref().map_or(0, |adapter| adapter.slot),
        };

        // Each model fills its actual computation inputs in the same request.
        // No partially prepared request is exposed to the submission boundary.
        // The closure gathers every `TokenizeError` so it maps once to `ServeError::Tokenize`.
        let processor = (|| -> std::result::Result<_, crate::input::TokenizeError> {
            let processor =
                self.preprocess_chat_input(prompt, max_tokens, &images, &mut generation)?;
            generation.validate()?;
            // The workers compute a position's candidates in their graphs, as many as they state.
            self.limits.covers_logprobs(&generation.sampling)?;
            Ok(processor)
        })()
        .map_err(|source| ServeError::Tokenize {
            request_id: request_id.clone(),
            source,
        })?;

        let output = OutputParams {
            tokenizer: Arc::clone(&self.tokenizer),
            // Detokenization needs the prompt after ownership moves to the engine.
            prompt_token_ids: generation.prompt_token_ids.clone(),
            decode,
            prompt_logprobs: generation.sampling.prompt_logprobs_requested(),
            logprobs: generation.sampling.generated_logprobs_requested(),
            processor,
        };
        Ok((generation, output))
    }
}

/// Refuses to bind a model whose required features the worker limits do not cover,
/// reporting the uncovered features as `ModelResolutionError::MissingFeature`.
fn validate_runtime_features(
    config: &ModelConfig,
    limits: &GenerationLimits,
    needs: GenerationFeatures,
) -> std::result::Result<(), ModelResolutionError> {
    limits
        .covers(needs)
        .map_err(|feature| ModelResolutionError::MissingFeature {
            description: config.id(),
            feature,
        })
}

impl InputProcessor {
    /// Renders a chat prompt and fills its token, sampling, and cache requirements; the output
    /// is parsed in the formats the chat family writes.
    fn preprocess_chat_input(
        &self,
        chat_request: ChatRequest,
        max_tokens: Option<u32>,
        images: &[crate::input::ImageInput],
        generation: &mut GenerationRequest,
    ) -> std::result::Result<ChatOutputProcessor, crate::input::TokenizeError> {
        chat_request.validate()?;
        let profile = self.config.chat().unwrap_or(processing::ChatProfile {
            reasoning: None,
            tool_calls: None,
            images: None,
        });
        let tools = (profile.tool_calls.is_some() && chat_request.tool_parsing_enabled())
            .then_some(chat_request.tools.as_slice());
        let processor = match (profile.reasoning, profile.tool_calls) {
            (Some(ReasoningFormat::Qwen3), _) | (None, Some(ToolCallFormat::Qwen3)) => {
                ChatOutputProcessor::qwen3(
                    tools,
                    std::sync::Arc::clone(&self.tokenizer),
                    self.parse_reasoning && profile.reasoning.is_some(),
                )?
            }
            (None, None) => ChatOutputProcessor::new(None, None),
        };
        let rendered_text = self
            .chat_renderer()
            .ok_or(crate::output::chat::Error::MissingChatTemplate)?
            .render(&chat_request)?;
        let mut prompt_token_ids = self.tokenizer.encode(&rendered_text, false)?;
        generation.multimodal_inputs.images = self.lower_images(
            &mut prompt_token_ids,
            images,
            generation.cache.isolation_key,
        )?;

        // The context holds the prompt's tokens and every image block's rows.
        let image_tokens: u32 = generation
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| &image.encoders)
            .filter_map(|encoder| encoder.num_cache_tokens)
            .sum();
        let prompt_len = prompt_token_ids.len() as u32 + image_tokens;
        let max_tokens = self.resolve_length(generation, max_tokens, prompt_len)?;

        generation.prompt_token_ids = prompt_token_ids;
        generation.max_und_tokens = max_tokens as usize;
        // Prompt logprobs require computation for every prompt token, which a prefix-cache
        // read would skip.
        generation.cache.read &= !generation.sampling.prompt_logprobs_requested();
        Ok(processor)
    }

    /// Lowers the fetched `images` of a rendered prompt into the request's image inputs.
    ///
    /// The template wrote the family's placeholder token once per image, in the order the images
    /// were fetched. Each placeholder is removed from `prompt` and its image becomes a vision block
    /// written before the token that followed it, of the rows and positions the family's processor
    /// states for the image's size. An image's digest, which keys its encodings, covers its bytes
    /// and the request's cache partition `isolation`, so requests of other partitions share no
    /// encoding.
    fn lower_images(
        &self,
        prompt: &mut Vec<u32>,
        images: &[crate::input::ImageInput],
        isolation: Option<u64>,
    ) -> std::result::Result<Vec<proto::request::ImageInput>, crate::input::TokenizeError> {
        use base64::Engine as _;

        let Some(processor) = self.images.as_ref() else {
            return Ok(Vec::new());
        };
        let placeholder = self
            .tokenizer
            .token_to_id(processor.placeholder())
            .ok_or_else(|| {
                crate::input::TokenizeError::Invalid(format!(
                    "the vocabulary has no image placeholder `{}`",
                    processor.placeholder()
                ))
            })?;
        let found = prompt.iter().filter(|id| **id == placeholder).count();
        if found != images.len() {
            return Err(crate::input::TokenizeError::Invalid(format!(
                "the prompt renders {found} image placeholder(s) for {} image(s)",
                images.len()
            )));
        }

        let mut inputs = Vec::with_capacity(images.len());
        let mut kept = Vec::with_capacity(prompt.len());
        let mut next = images.iter();
        for &id in prompt.iter() {
            if id != placeholder {
                kept.push(id);
                continue;
            }
            let image = next.next().ok_or_else(|| {
                crate::input::TokenizeError::Invalid("an image placeholder has no image".into())
            })?;
            let block = processor
                .block(image.width(), image.height())
                .map_err(crate::input::TokenizeError::Invalid)?;
            let mut digest = storage::key::Digest::new(b"chat-image");
            digest.bytes(&image.digest());
            match isolation {
                Some(key) => {
                    digest.number(1);
                    digest.number(key);
                }
                None => digest.number(0),
            }
            inputs.push(proto::request::ImageInput {
                digest: digest.finish(),
                b64: base64::engine::general_purpose::STANDARD.encode(image.bytes()),
                position: kept.len() as u32,
                num_positions: block.positions,
                encoders: vec![proto::request::ImageEncoderInput {
                    encoder: proto::request::ImageEncoder::Vision,
                    num_cache_tokens: Some(block.tokens),
                    max_cache_tokens: None,
                    height: None,
                    width: None,
                }],
            });
        }
        *prompt = kept;
        Ok(inputs)
    }

    /// The engine's sampling parameters of a chat `request`: its controls
    /// over the model's sampling defaults, lowered once, with its logit
    /// biases keyed by token ID.
    ///
    /// # Errors
    ///
    /// Fails when a bad word cannot be tokenized or the parameters violate
    /// the engine's sampling contract.
    pub fn sampling(
        &self,
        request: &crate::openai::ChatCompletionRequest,
        logit_bias: Option<std::collections::HashMap<u32, f32>>,
    ) -> std::result::Result<EngineSamplingParams, crate::input::TokenizeError> {
        super::sampling::lower(
            request,
            logit_bias,
            &self.config.sampling_defaults,
            &self.tokenizer,
        )
    }

    /// Resolves the request's generation length for a prompt of
    /// `prompt_len` tokens, and adds the checkpoint's EOS tokens other than
    /// the primary one to `generation`'s stop tokens unless it ignores EOS.
    ///
    /// # Errors
    ///
    /// Fails when the prompt leaves no room in the context, or the request's
    /// `min_tokens` exceeds the resolved length.
    fn resolve_length(
        &self,
        generation: &mut GenerationRequest,
        max_tokens: Option<u32>,
        prompt_len: u32,
    ) -> std::result::Result<u32, crate::input::TokenizeError> {
        let max_tokens = resolve_max_tokens(
            max_tokens,
            self.config.sampling_defaults.max_output_tokens,
            Some(self.config.max_model_len),
            prompt_len,
        )?;
        let min_tokens = generation.sampling.min_tokens as u32;

        // Unless the request ignores EOS, the checkpoint's EOS ids other than the primary one
        // become explicit stop tokens.
        if !generation.sampling.ignore_eos {
            for token_id in self
                .config
                .eos_token_ids
                .iter()
                .filter(|token| Some(**token) != self.config.primary_eos_token_id)
            {
                if !generation.stop_token_ids.contains(token_id) {
                    generation.stop_token_ids.push(*token_id);
                }
            }
        }

        if min_tokens > max_tokens {
            return Err(crate::input::TokenizeError::MinTokensExceedsMaximum {
                min_tokens,
                max_tokens,
            });
        }
        Ok(max_tokens)
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::*;

    /// A chat checkpoint's tokenizer registers the tokens its
    /// `tokenizer_config.json` declares and its `tokenizer.json` lacks, as
    /// `transformers` loads it: a declared `<d>` encodes as one token.
    #[tokio::test]
    async fn a_chat_tokenizer_registers_the_tokens_its_configuration_declares() {
        let directory = tempfile::tempdir().unwrap();
        let vocab = ["<unk>", "a", "b", "<", ">", "d", "<|im_end|>"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect::<Vocab>();
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = TokenizerBuilder::new(model);
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
        tokenizer
            .save(directory.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            r#"{"model_type": "chat", "max_position_embeddings": 64}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {
                    "7": {"content": "<d>", "lstrip": false, "normalized": false,
                          "rstrip": false, "single_word": false, "special": true}
                }
            }"#,
        )
        .unwrap();

        let catalogue = processing::Catalogue::new([crate::test_support::CHAT]).unwrap();
        let loaded = ModelConfig::load(
            directory.path().to_str().unwrap(),
            None,
            &proto::config::ChatConfig::default(),
            &catalogue,
        )
        .await
        .unwrap();

        let marker = loaded
            .tokenizer
            .token_to_id("<d>")
            .expect("<d> is registered");
        assert_eq!(marker, 7);
        assert_eq!(loaded.tokenizer.encode("a<d>b", false).unwrap(), [1, 7, 2]);
    }
}
