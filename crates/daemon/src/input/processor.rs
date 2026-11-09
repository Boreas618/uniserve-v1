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

use crate::model::assets::{PipelineCheckpoint, ResolvedModelFiles};
use crate::model::ModelConfig;
use crate::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use processing::{Catalogue, ReasoningFormat, Serving, ToolCallFormat, VideoFamily};
use proto::request::{
    CachePolicy, GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationRequest,
    ImageGenerationConfig,
};
use proto::request::{ImageParams, RequestId, SamplingParams as EngineSamplingParams};
use thiserror::Error;

use crate::input::chat::{ChatRequest, HfChatRenderer};
use crate::input::request::{GenerateRequest, Prompt, SamplingParams};
use crate::input::sampling::resolve_max_tokens;
use crate::input::video::VideoService;
use crate::input::{Result, ServeError};
use crate::output::OutputParams;
use crate::output::chat::ChatOutputProcessor;
use crate::output::text::TextDecodeOptions;

/// Owns the tokenizer and template resources used to produce final engine inputs.
/// Runtime capabilities are fixed at construction, before this value is shared.
///
/// Preprocessing methods are synchronous and CPU-bound; `Leader` runs its request
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
pub(crate) struct LoadedModel {
    /// Immutable model facts.
    pub config: ModelConfig,
    /// The prompt tokenizer.
    pub tokenizer: DynTokenizer,
    /// What renders the checkpoint's requests once its engine has started.
    pub renderer: LoadedRenderer,
}

/// What renders a loaded checkpoint's requests, fixed by its layout.
pub(crate) enum LoadedRenderer {
    /// A token model's chat template.
    Chat { renderer: Box<HfChatRenderer> },
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
pub(crate) async fn pipeline_tokenizer(
    pipeline: &PipelineCheckpoint,
) -> std::result::Result<HuggingFaceTokenizer, ModelResolutionError> {
    let path = pipeline
        .component_file("tokenizer", "tokenizer.json")
        .await?;
    let tokenizer = match pipeline
        .component_file("tokenizer", "tokenizer_config.json")
        .await
    {
        Ok(config) => HuggingFaceTokenizer::with_config(&path, &config)?,
        Err(crate::model::assets::Error::MissingFile { .. }) => HuggingFaceTokenizer::new(&path)?,
        Err(error) => return Err(error.into()),
    };
    Ok(tokenizer)
}

/// How this deployment loads a checkpoint: its tokenizer, its template
/// resources and the facts its engine is configured with.
///
/// The model is the frontend's (delivery item 2); loading it is the request
/// processing's, which moves with `input/` in stage A — until then it is an
/// extension trait here, so the daemon names no inherent impl of a foreign
/// type.
pub(crate) trait LoadModel {
    fn load(
        config: &crate::options::ServerArgs,
        catalogue: &Catalogue,
    ) -> impl std::future::Future<Output = std::result::Result<LoadedModel, ModelResolutionError>>;
}

impl LoadModel for ModelConfig {
    /// Loads model facts and the tokenizer/template resources needed by preprocessing.
    ///
    /// A diffusers pipeline has no chat template and carries its conditioner's
    /// vision processor geometry instead. The returned `max_model_len` is the
    /// configured one, else, for a root configuration, its
    /// `max_position_embeddings` or `ModelConfig::DEFAULT_MAX_MODEL_LEN`,
    /// and for a pipeline the model's default prompt limit.
    async fn load(
        config: &crate::options::ServerArgs,
        catalogue: &Catalogue,
    ) -> std::result::Result<LoadedModel, ModelResolutionError> {
        let max_model_len = config.max_model_len.map(std::num::NonZeroU32::get);
        // A diffusers pipeline declares its class and component folders in a
        // root index rather than a root `config.json`, so the index selects the
        // family and locates the tokenizer component before any other asset
        // is resolved. A component export reads the components its pinned
        // base supplies from the Hub cache at the pinned revision, as its
        // workers do. A family that a root `config.json` describes is
        // selected by that file alone, even when the checkpoint publishes an
        // index beside it.
        if let Some(pipeline) = PipelineCheckpoint::resolve(config.model()).await?
            && let Some(family) = catalogue.by_pipeline_class(pipeline.class_name())
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

        let files = ResolvedModelFiles::new(config.model()).await?;
        let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path)?);
        let model = Self::from_files(catalogue, &files, max_model_len, tokenizer.as_ref()).await?;
        // No served model's template renders image parts: the families whose
        // prompts carried images are not served by this daemon.
        let renderer = HfChatRenderer::load(&files, &config.chat_config, None)?;
        Ok(LoadedModel {
            config: model,
            tokenizer,
            renderer: LoadedRenderer::Chat {
                renderer: Box::new(renderer),
            },
        })
    }
}

impl InputProcessor {
    /// The model's tokenizer, which a process continuing a response builds
    /// its incremental decoder from (design `specs/worker.md` §7.5.1).
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
        })
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

    /// Returns whether the model supports image input.
    ///
    /// No model this daemon serves accepts image input, for the same reason:
    /// the checkpoint that read an image part is not served.
    pub fn supports_image_input(&self) -> bool {
        false
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
        prompt: &Prompt,
        has_input_image: bool,
        constraint: GenerationConstraint,
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
        if constraint != GenerationConstraint::UndOnly {
            return Err(reject("image_output"));
        }
        if let Prompt::Chat(chat) = prompt {
            // Tool definitions, prior assistant tool calls, and tool responses all require a
            // model that parses tool calls.
            let uses_tools = !chat.tools.is_empty()
                || chat.messages.iter().any(|message| match message {
                    crate::input::chat::ChatMessage::Developer { tools, .. } => {
                        tools.as_ref().is_some_and(|tools| !tools.is_empty())
                    }
                    crate::input::chat::ChatMessage::Assistant { content } => {
                        content.has_tool_calls()
                    }
                    crate::input::chat::ChatMessage::ToolResponse { .. } => true,
                    _ => false,
                });
            if uses_tools && !self.supports_tool_calling() {
                return Err(reject("tool_calling"));
            }
            if chat.chat_options.reasoning_effort.is_some() && !self.supports_reasoning() {
                return Err(reject("reasoning"));
            }
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
    /// Returns `ServeError::UnsupportedFeature` from feature validation,
    /// `ServeError::UnsupportedSamplingControl` for a control the model's
    /// generation does not define, and `ServeError::Tokenize` for every
    /// failure during tokenization, validation, and resource sizing.
    pub fn preprocess(
        &self,
        request: GenerateRequest,
        images: Vec<crate::input::ImageInput>,
    ) -> Result<(GenerationRequest, OutputParams)> {
        let GenerateRequest {
            request_id,
            prompt,
            constraint,
            sampling_params,
            priority,
            cache_salt,
            adapter,
        } = request;
        let has_input_image =
            !images.is_empty() || matches!(&prompt, Prompt::Chat(chat) if chat.has_multimodal());
        self.validate_generation_features(&request_id, &prompt, has_input_image, constraint)?;
        let mut generation = GenerationRequest {
            // `Leader::submit` stamps the engine's identifier for the request.
            request_id: RequestId::default(),
            prompt_token_ids: Vec::new(),
            negative_prompt_token_ids: Vec::new(),
            multimodal_inputs: Default::default(),
            constraint,
            sampling: EngineSamplingParams {
                // The text sampling seed's signed bit pattern seeds the engine.
                seed: sampling_params.seed.map(|seed| seed as u64),
                ..EngineSamplingParams::default()
            },
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: 0,
            include_stop_token: false,
            stop_strings: sampling_params.stop.clone(),
            stop_token_ids: sampling_params.stop_token_ids.clone(),
            priority,
            cache: CachePolicy {
                isolation_key: crate::input::cache_isolation_key(
                    cache_salt.as_deref(),
                    adapter.as_ref().map(|adapter| adapter.identity.as_str()),
                ),
                ..CachePolicy::default()
            },
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: adapter.as_ref().map_or(0, |adapter| adapter.slot),
            generated_images: Vec::new(),
            trajectory: Default::default(),
        };
        let mut decode = sampling_params.decode_options();

        // Each model fills its actual computation inputs in the same request.
        // No partially prepared request is exposed to the submission boundary.
        // The closure gathers every `TokenizeError` so it maps once to `ServeError::Tokenize`.
        let processor = (|| -> std::result::Result<_, crate::input::TokenizeError> {
            let processor = self.preprocess_chat_input(
                chat_only(prompt),
                &sampling_params,
                &mut generation,
                &mut decode,
            )?;
            generation.validate()?;
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

/// Returns the chat conversation of a prompt for a model that serves chat only.
///
/// Only an image-generation request sends a text prompt, and feature validation
/// refuses it for a model without image output.
fn chat_only(prompt: Prompt) -> ChatRequest {
    match prompt {
        Prompt::Chat(chat) => chat,
        Prompt::Text { .. } => {
            unreachable!("a text prompt is image generation, refused above for a chat-only model")
        }
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
        sampling_params: &SamplingParams,
        generation: &mut GenerationRequest,
        decode: &mut TextDecodeOptions,
    ) -> std::result::Result<ChatOutputProcessor, crate::input::TokenizeError> {
        chat_request.validate()?;
        let profile = self.config.chat().unwrap_or(processing::ChatProfile {
            reasoning: None,
            tool_calls: None,
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
        let prompt_token_ids = self.tokenizer.encode(&rendered_text, false)?;

        let prompt_len = prompt_token_ids.len() as u32;
        let (sampling, max_tokens, stop_token_ids) =
            self.resolve_sampling(sampling_params, generation.sampling.seed, prompt_len)?;

        generation.prompt_token_ids = prompt_token_ids;
        generation.sampling = sampling;
        generation.stop_token_ids = stop_token_ids;
        generation.max_und_tokens = max_tokens as usize;
        generation.include_stop_token = decode.include_stop_str_in_output;
        // Prompt logprobs require computation for every prompt token, which a prefix-cache
        // read would skip.
        generation.cache.read &= !generation.sampling.prompt_logprobs_requested();
        Ok(processor)
    }

    /// Resolves model defaults and request controls into validated engine
    /// sampling parameters under the request's resolved `seed`, with the
    /// generation length and the stop tokens.
    fn resolve_sampling(
        &self,
        sampling: &SamplingParams,
        seed: Option<u64>,
        prompt_len: u32,
    ) -> std::result::Result<(EngineSamplingParams, u32, Vec<u32>), crate::input::TokenizeError>
    {
        let defaults = &self.config.sampling_defaults;

        let max_tokens = resolve_max_tokens(
            sampling.max_tokens,
            defaults.max_output_tokens,
            Some(self.config.max_model_len),
            prompt_len,
        )?;
        let min_tokens = sampling.min_tokens.unwrap_or(0);

        // Unless the request ignores EOS, the checkpoint's EOS ids other than the primary one
        // become explicit stop tokens.
        let mut stop_token_ids = sampling.stop_token_ids.clone();
        if !sampling.ignore_eos {
            for token_id in self
                .config
                .eos_token_ids
                .iter()
                .filter(|token| Some(**token) != self.config.primary_eos_token_id)
            {
                if !stop_token_ids.contains(token_id) {
                    stop_token_ids.push(*token_id);
                }
            }
        }

        if min_tokens > max_tokens {
            return Err(crate::input::TokenizeError::MinTokensExceedsMaximum {
                min_tokens,
                max_tokens,
            });
        }

        let core = super::sampling::resolve_sampling(&self.tokenizer, defaults, sampling, seed)?;

        Ok((core, max_tokens, stop_token_ids))
    }
}
