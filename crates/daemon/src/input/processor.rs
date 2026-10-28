//! Load-time model resolution and model-owned request tokenization.
//!
//! `ModelConfig::load` resolves checkpoint facts, the tokenizer, and what renders requests
//! at startup: a diffusers pipeline index selects a video model, planned against its
//! conditioner's vision processor, and otherwise the root `config.json` selects a
//! token-generating model with its chat template. [`InputProcessor::new`] then binds the
//! generation limits the started workers report and the request
//! renderer they complete ([`RequestRenderer`]), and the result is shared immutably by the
//! serving runtime.
//!
//! [`InputProcessor`] binds tokenizer assets, generation limits, sampling policy, and
//! output processing. [`InputProcessor::preprocess`] tokenizes a `GenerateRequest` into a
//! [`GenerationRequest`] and the `OutputParams` (`crate::output`) the daemon retains, and delegates
//! SenseNova and Bagel inputs to the sibling `omni` module; `omni` reads the `pub(super)`
//! fields.
//! A MiniMax H3 deployment's `VideoService` produces its `DiffusionRequest`s.

use std::sync::Arc;

use crate::model::assets::{PipelineCheckpoint, ResolvedModelFiles};
use crate::model::{ModelConfig, ModelParameters};
use crate::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use models::ModelDescription;
use models::omni::bagel::BagelProfile;
use models::video::plan::ProcessorConfig;
use proto::request::{
    CachePolicy, GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationRequest,
    ImageGenerationConfig,
};
use proto::request::{ImageParams, RequestId, SamplingParams as EngineSamplingParams};
use thiserror::Error;

use crate::input::chat::template::MultimodalRenderInfo;
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
    // Whether a Qwen3-family chat template instructs the tool-call format
    // `Qwen25ToolParser` reads; false for every other family.
    qwen3_tool_calls: bool,
}

#[derive(Debug, Error)]
/// Failure while resolving a configured model and its assets.
pub enum ModelResolutionError {
    /// A conditioner's processor config cannot be read.
    #[error("reading {}", path.display())]
    ProcessorFile {
        /// The config file.
        path: std::path::PathBuf,
        /// The read's failure.
        #[source]
        source: std::io::Error,
    },
    /// The conditioner's processor configs are not processor configs or
    /// disagree.
    #[error("invalid media checkpoint contract")]
    ProcessorConfig(#[source] models::video::plan::ProcessorConfigError),
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
    /// A token model's chat template, with the checkpoint files it was read
    /// from, which the System One readout renders its own prompts from.
    Chat {
        renderer: Box<HfChatRenderer>,
        files: ResolvedModelFiles,
    },
    /// A video pipeline's conditioner vision processor geometry; the video
    /// service built from the denoiser handshake plans requests against it.
    Video(ProcessorConfig),
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

impl ModelConfig {
    /// Loads model facts and the tokenizer/template resources needed by preprocessing.
    ///
    /// A diffusers pipeline has no chat template and carries its conditioner's
    /// vision processor geometry instead. The returned `max_model_len` is the
    /// configured one, else, for a root configuration, its
    /// `max_position_embeddings` or `ModelConfig::DEFAULT_MAX_MODEL_LEN`,
    /// and for a pipeline the model's default prompt limit.
    pub(crate) async fn load(
        config: &crate::options::ServeConfig,
    ) -> std::result::Result<LoadedModel, ModelResolutionError> {
        let max_model_len = config.max_model_len.map(std::num::NonZeroU32::get);
        // A diffusers pipeline declares its class and component folders in a
        // root index rather than a root `config.json`, so the index selects the
        // profile and locates the tokenizer component before any other asset
        // is resolved. A component export reads the components its pinned
        // base supplies from the Hub cache at the pinned revision, as its
        // workers do. A family that a root `config.json` describes is
        // selected by that file alone, even when the checkpoint publishes an
        // index beside it.
        if let Some(pipeline) = PipelineCheckpoint::resolve(config.model()).await? {
            let description = ModelDescription::from_pipeline_class(pipeline.class_name())
                .ok_or_else(|| crate::model::assets::Error::UnsupportedPipeline {
                    class_name: pipeline.class_name().to_owned(),
                })?;
            if description.is_pipeline() {
                let tokenizer: DynTokenizer = Arc::new(pipeline_tokenizer(&pipeline).await?);
                // The conditioner's Qwen3-VL processor fixes how condition media
                // is patched; requests are planned against it.
                let mut processor = Vec::with_capacity(2);
                for name in ["preprocessor_config.json", "video_preprocessor_config.json"] {
                    processor.push(pipeline.component_file("processor", name).await?);
                }
                let read = |path: &std::path::PathBuf| {
                    std::fs::read_to_string(path).map_err(|source| {
                        ModelResolutionError::ProcessorFile {
                            path: path.clone(),
                            source,
                        }
                    })
                };
                let vision = ProcessorConfig::parse(&read(&processor[0])?, &read(&processor[1])?)
                    .map_err(ModelResolutionError::ProcessorConfig)?;
                let model = Self::from_pipeline(description, max_model_len)?;
                return Ok(LoadedModel {
                    config: model,
                    tokenizer,
                    renderer: LoadedRenderer::Video(vision),
                });
            }
        }

        let files = ResolvedModelFiles::new(config.model()).await?;
        let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path)?);
        let model =
            Self::from_files(config.model(), &files, max_model_len, tokenizer.as_ref()).await?;
        // DiffusionGemma's template renders each image part itself, as one
        // image token that preprocessing expands; the omni models replace
        // image parts before rendering.
        let multimodal = match &model.parameters {
            ModelParameters::DiffusionGemma(profile) => tokenizer
                .id_to_token(profile.tokens.image)
                .map(|placeholder_token| MultimodalRenderInfo { placeholder_token }),
            _ => None,
        };
        let renderer = HfChatRenderer::load(&files, &config.chat_config, multimodal)?;
        Ok(LoadedModel {
            config: model,
            tokenizer,
            renderer: LoadedRenderer::Chat {
                renderer: Box::new(renderer),
                files,
            },
        })
    }

    /// Vocabulary markers used by multimodal scheduling and prompt construction.
    pub(crate) fn generation_controls(&self) -> Option<&models::omni::GenerationControls> {
        match &self.parameters {
            ModelParameters::SenseNova(value) => Some(&value.controls),
            ModelParameters::Bagel(value) => Some(&value.controls),
            ModelParameters::Qwen3
            | ModelParameters::MiniMaxH3
            | ModelParameters::DiffusionGemma(_) => None,
        }
    }

    /// Model facts used at startup before the loaded workers' capacities and
    /// served features are applied (`resolve_generation_limits`).
    pub(crate) fn generation_limits(
        &self,
        model_dtype: proto::request::ModelDtype,
    ) -> proto::request::GenerationLimits {
        match &self.parameters {
            ModelParameters::Qwen3 | ModelParameters::MiniMaxH3 => {
                proto::request::GenerationLimits {
                    latent_downsample: 1,
                    max_cfg_branches: 1,
                    ..Default::default()
                }
            }
            ModelParameters::SenseNova(profile) => profile.runtime_limits(model_dtype),
            ModelParameters::Bagel(_) => BagelProfile::runtime_limits(model_dtype),
            // A readout prefills text and encoded images and denoises its
            // canvases once over them. An image writes one cache entry per soft
            // token; the worker's encoder entries bound the feature bytes and
            // the number of cached encodings, so the model states no bound of
            // its own for either.
            ModelParameters::DiffusionGemma(profile) => proto::request::GenerationLimits {
                latent_downsample: 1,
                max_vit_grid_tokens: profile.images.max_soft_tokens,
                max_vision_feature_bytes: u64::MAX,
                max_cfg_branches: 1,
                encoder_cache_entries: u32::MAX,
                ..Default::default()
            },
        }
    }
}

impl InputProcessor {
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
        let needs = match &config.parameters {
            ModelParameters::Qwen3 => GenerationFeatures::UNDERSTANDING,
            // The System One readout denoises canvases over its prompt.
            ModelParameters::DiffusionGemma(_) => GenerationFeatures::TOKEN_DENOISING,
            ModelParameters::SenseNova(profile) => {
                configured_omni_needs(&profile.image_generation, &profile.image_encoders)
            }
            ModelParameters::Bagel(profile) => {
                configured_omni_needs(&profile.image_generation, &profile.image_encoders)
            }
            ModelParameters::MiniMaxH3 => GenerationFeatures::empty(),
        };
        validate_runtime_features(&config, &limits, needs)?;

        // A Qwen3-family checkpoint serves tool calls only when its template
        // asks for the format the Qwen3 parser reads; otherwise requests with
        // tools are refused rather than returned with unparsed calls.
        let qwen3_tool_calls = match (&config.parameters, &renderer) {
            (ModelParameters::Qwen3, RequestRenderer::Chat(renderer)) => {
                crate::input::chat::template::template_instructs_json_tool_calls(renderer)
            }
            _ => false,
        };
        Ok(Self {
            config,
            tokenizer,
            renderer,
            limits,
            parse_reasoning,
            qwen3_tool_calls,
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

    /// The video service of a MiniMax H3 deployment; `None` for other models.
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

    /// Returns whether the model supports image output.
    pub fn supports_image_output(&self) -> bool {
        matches!(
            self.config.parameters,
            ModelParameters::SenseNova(_) | ModelParameters::Bagel(_)
        )
    }

    /// Returns whether the model supports image input.
    pub fn supports_image_input(&self) -> bool {
        matches!(
            self.config.parameters,
            ModelParameters::SenseNova(_)
                | ModelParameters::Bagel(_)
                | ModelParameters::DiffusionGemma(_)
        )
    }

    /// Returns whether the model parses tool calls from its output.
    pub fn supports_tool_calling(&self) -> bool {
        match self.config.parameters {
            ModelParameters::Qwen3 => self.qwen3_tool_calls,
            ModelParameters::DiffusionGemma(_) => true,
            ModelParameters::SenseNova(_)
            | ModelParameters::Bagel(_)
            | ModelParameters::MiniMaxH3 => false,
        }
    }

    /// Returns whether the model produces structured reasoning output.
    pub fn supports_reasoning(&self) -> bool {
        matches!(
            self.config.parameters,
            ModelParameters::Qwen3
                | ModelParameters::SenseNova(_)
                | ModelParameters::DiffusionGemma(_)
        )
    }

    /// Validates the prompt and requested modalities against the loaded model.
    ///
    /// Every refusal is `ServeError::UnsupportedFeature`. The final check asks the worker
    /// limits to cover the features this request reaches; for SenseNova and Bagel these depend
    /// on the selected modalities, and for them and DiffusionGemma on whether the request
    /// carries an input image.
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
        if matches!(self.config.parameters, ModelParameters::MiniMaxH3) {
            return Err(reject("generation_endpoint"));
        }
        if has_input_image && !self.supports_image_input() {
            return Err(reject("image_input"));
        }
        if constraint != GenerationConstraint::UndOnly && !self.supports_image_output() {
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
        let needs = match &self.config.parameters {
            ModelParameters::Qwen3 => GenerationFeatures::UNDERSTANDING,
            ModelParameters::SenseNova(profile) => omni_required_features(
                &profile.image_generation,
                &profile.image_encoders,
                has_input_image,
                constraint,
            ),
            ModelParameters::Bagel(profile) => omni_required_features(
                &profile.image_generation,
                &profile.image_encoders,
                has_input_image,
                constraint,
            ),
            // Block diffusion denoises canvases over a prompt whose images the
            // vision encoder writes into it.
            ModelParameters::DiffusionGemma(_) => {
                GenerationFeatures::TOKEN_DENOISING
                    | if has_input_image {
                        GenerationFeatures::VISION_ENCODE
                    } else {
                        GenerationFeatures::empty()
                    }
            }
            ModelParameters::MiniMaxH3 => {
                unreachable!("video generation refused above")
            }
        };
        self.limits
            .covers(needs)
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
            image_params,
            priority,
            cache_salt,
            adapter,
        } = request;
        let has_input_image =
            !images.is_empty() || matches!(&prompt, Prompt::Chat(chat) if chat.has_multimodal());
        self.validate_generation_features(&request_id, &prompt, has_input_image, constraint)?;
        if matches!(self.config.parameters, ModelParameters::DiffusionGemma(_)) {
            crate::input::diffusion_gemma::refuse_token_sampling_controls(
                &request_id,
                &sampling_params,
            )?;
        }

        let mut generation = GenerationRequest {
            // `Leader::submit` stamps the engine's identifier for the request.
            request_id: RequestId::default(),
            prompt_token_ids: Vec::new(),
            negative_prompt_token_ids: Vec::new(),
            multimodal_inputs: Default::default(),
            constraint,
            sampling: EngineSamplingParams {
                // The request's one seed: an image-generation seed takes
                // precedence over the text sampling seed, whose signed bit
                // pattern seeds the engine.
                seed: image_params
                    .as_ref()
                    .and_then(|image| image.seed)
                    .or_else(|| sampling_params.seed.map(|seed| seed as u64)),
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
        };
        let mut decode = sampling_params.decode_options();

        // Each model fills its actual computation inputs in the same request.
        // No partially prepared request is exposed to the submission boundary.
        // The closure gathers every `TokenizeError` so it maps once to `ServeError::Tokenize`.
        let processor = (|| -> std::result::Result<_, crate::input::TokenizeError> {
            let processor = match &self.config.parameters {
                ModelParameters::Qwen3 => self.preprocess_qwen3_input(
                    chat_only(prompt),
                    &sampling_params,
                    &mut generation,
                    &mut decode,
                )?,
                ModelParameters::SenseNova(profile) => crate::input::omni::preprocess_omni(
                    profile,
                    self,
                    &request_id,
                    prompt,
                    images,
                    image_params,
                    &mut generation,
                )?,
                ModelParameters::Bagel(profile) => crate::input::omni::preprocess_omni(
                    profile,
                    self,
                    &request_id,
                    prompt,
                    images,
                    image_params,
                    &mut generation,
                )?,
                ModelParameters::DiffusionGemma(profile) => self.preprocess_diffusion_gemma_input(
                    profile,
                    chat_only(prompt),
                    &images,
                    &sampling_params,
                    &mut generation,
                    &mut decode,
                )?,
                ModelParameters::MiniMaxH3 => {
                    unreachable!("generation features checked above")
                }
            };
            if matches!(
                self.config.parameters,
                ModelParameters::Qwen3 | ModelParameters::DiffusionGemma(_)
            ) {
                generation.validate()?;
            } else {
                crate::input::omni::prepare_generation_resources(
                    self,
                    &sampling_params,
                    &mut generation,
                )?;
            }
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

/// Returns the multimodal resources required by the active profile.
///
/// Used at load time: evaluates `GenerationConstraint::Default` (both understanding and
/// generation branches enabled) with every configured image encoder. Per-request checks use
/// `omni_required_features`.
fn configured_omni_needs(
    policy: &ImageGenerationConfig,
    image_encoders: &[proto::request::ImageEncoderInput],
) -> GenerationFeatures {
    policy.required_features(
        GenerationConstraint::Default,
        image_encoders.iter().map(|input| input.encoder),
    )
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
            description: config.description().id(),
            feature,
        })
}

/// Returns the runtime features required for multimodal serving.
///
/// The profile's context image encoders count only when the request carries an input image;
/// `ImageGenerationConfig::required_features` adds feedback encoders independently of it.
fn omni_required_features(
    policy: &ImageGenerationConfig,
    image_encoders: &[proto::request::ImageEncoderInput],
    has_input_image: bool,
    constraint: GenerationConstraint,
) -> GenerationFeatures {
    let context_steps = if has_input_image {
        image_encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    policy.required_features(constraint, context_steps)
}

impl InputProcessor {
    /// Renders a Qwen chat prompt and fills its token, sampling, and cache requirements.
    fn preprocess_qwen3_input(
        &self,
        chat_request: ChatRequest,
        sampling_params: &SamplingParams,
        generation: &mut GenerationRequest,
        decode: &mut TextDecodeOptions,
    ) -> std::result::Result<ChatOutputProcessor, crate::input::TokenizeError> {
        chat_request.validate()?;
        let processor = ChatOutputProcessor::qwen3(
            chat_request
                .tool_parsing_enabled()
                .then_some(chat_request.tools.as_slice()),
            std::sync::Arc::clone(&self.tokenizer),
            self.parse_reasoning,
        )?;
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

#[cfg(test)]
mod tests {
    use std::fs;

    use proto::request::CanvasSampling;
    use proto::request::{GenerationFeatures, GenerationLimits};
    use serde_json::json;
    use tempfile::TempDir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::{LoadedModel, LoadedRenderer};
    use crate::input::{InputProcessor, RequestRenderer};
    use crate::model::{ModelConfig, ModelParameters};
    use crate::options::ServeConfig;
    use crate::tokenizer::HuggingFaceTokenizer;
    use models::ModelDescription;

    /// Writes a checkpoint laid out as DiffusionGemma publishes it: a root
    /// transformers `config.json` beside a `DiffusionGemmaPipeline` diffusers
    /// index. The tokenizer maps
    /// each ASCII character to one token and adds the Gemma control tokens,
    /// including the thought-channel delimiters its reasoning parser reads.
    fn indexed_checkpoint() -> TempDir {
        let directory = tempfile::tempdir().unwrap();
        let mut vocab = Vocab::from_iter([("<unk>".to_owned(), 0_u32)]);
        for code in 1_u32..=127 {
            vocab.insert(char::from_u32(code).unwrap().to_string(), code);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut builder = TokenizerBuilder::new(model);
        builder.add_special_tokens(
            &[
                "<pad>",
                "<eos>",
                "<bos>",
                "<mask>",
                "<turn|>",
                "<|image>",
                "<|image|>",
                "<image|>",
                "<|channel>",
                "<channel|>",
            ]
            .map(|token| AddedToken::from(token, true)),
        );
        let tokenizer_path = directory.path().join("tokenizer.json");
        builder.save(&tokenizer_path, false).unwrap();
        let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).unwrap();
        let id = |token: &str| tokenizer.token_to_id(token).unwrap();

        let write = |name: &str, value: serde_json::Value| {
            fs::write(directory.path().join(name), value.to_string()).unwrap();
        };
        write(
            "tokenizer_config.json",
            json!({"bos_token": "<bos>", "eos_token": "<eos>",
                   "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}"}),
        );
        write(
            "config.json",
            json!({
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "model_type": "diffusion_gemma",
                "canvas_length": 256,
                "image_token_id": id("<|image|>"),
                "boi_token_id": id("<|image>"),
                "eoi_token_id": id("<image|>"),
                "vision_soft_tokens_per_image": 280,
                "vision_config": {"patch_size": 16, "pooling_kernel_size": 3},
                "text_config": {"model_type": "diffusion_gemma_text", "max_position_embeddings": 262144},
            }),
        );
        write(
            "generation_config.json",
            json!({
                "eos_token_id": [1, 106, 50],
                "max_new_tokens": 256,
                "max_denoising_steps": 48,
                "sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1},
                "t_min": 0.4,
                "t_max": 0.8,
                "confidence_threshold": 0.005,
                "stability_threshold": 1,
            }),
        );
        write(
            "model_index.json",
            json!({
                "_class_name": "DiffusionGemmaPipeline",
                "model": ["transformers", "DiffusionGemmaForBlockDiffusion"],
                "scheduler": ["diffusers", "BlockRefinementScheduler"],
            }),
        );
        directory
    }

    fn config(directory: &TempDir) -> ServeConfig {
        ServeConfig {
            model: Some(directory.path().to_str().unwrap().to_owned()),
            served_model_name: Some("diffusion-gemma".to_owned()),
            ..ServeConfig::default()
        }
    }

    /// The diffusers index beside the root configuration does not make the
    /// checkpoint a pipeline: the root configuration resolves the model, its
    /// text limit from `text_config`, and its block-diffusion settings.
    #[tokio::test]
    async fn an_indexed_diffusion_gemma_checkpoint_resolves_from_its_root_config() {
        let directory = indexed_checkpoint();

        let LoadedModel {
            config: model,
            tokenizer,
            renderer,
            ..
        } = ModelConfig::load(&config(&directory)).await.unwrap();

        assert_eq!(model.description(), ModelDescription::DiffusionGemma);
        assert_eq!(model.max_model_len, 262_144);
        assert!(model.eos_token_ids.is_superset(&[1, 50, 106].into()));
        assert!(matches!(renderer, LoadedRenderer::Chat { .. }));
        let ModelParameters::DiffusionGemma(profile) = &model.parameters else {
            unreachable!("the description is DiffusionGemma");
        };
        assert_eq!(profile.sampling.canvas_length, 256);
        assert_eq!(
            profile.tokens.mask,
            tokenizer.token_to_id("<mask>").unwrap()
        );
        assert_eq!(
            profile.tokens.turn_end,
            tokenizer.token_to_id("<turn|>").unwrap()
        );
        assert_eq!(profile.sampling.max_steps, 48);
        assert!(!profile.quantized);
    }

    /// Binds a DiffusionGemma checkpoint whose context holds `max_model_len`
    /// tokens to a block-diffusion worker.
    async fn diffusion_gemma_processor(directory: &TempDir, max_model_len: u32) -> InputProcessor {
        let LoadedModel {
            config: model,
            tokenizer,
            renderer,
            ..
        } = ModelConfig::load(&ServeConfig {
            max_model_len: std::num::NonZeroU32::new(max_model_len),
            ..config(directory)
        })
        .await
        .unwrap();
        let LoadedRenderer::Chat { renderer, .. } = renderer else {
            unreachable!("a DiffusionGemma checkpoint renders chat");
        };
        InputProcessor::new(
            model,
            tokenizer,
            RequestRenderer::Chat(*renderer),
            GenerationLimits {
                features: GenerationFeatures::TOKEN_DENOISING | GenerationFeatures::VISION_ENCODE,
                latent_downsample: 1,
                max_cfg_branches: 1,
                max_vit_grid_tokens: 280,
                max_vision_feature_bytes: u64::MAX,
                encoder_cache_entries: u32::MAX,
                ..Default::default()
            },
            true,
        )
        .unwrap()
    }

    /// Preprocesses a chat completion as its route lowers it, answering a
    /// refusal as the route does.
    fn preprocess_chat(
        processor: &InputProcessor,
        request_id: String,
        request: crate::openai::ChatCompletionRequest,
        images: Vec<crate::input::ImageInput>,
    ) -> Result<
        (
            proto::request::GenerationRequest,
            crate::output::OutputParams,
        ),
        crate::openai::ApiError,
    > {
        let (request, _) = crate::openai::chat_completions::lower_chat_request(
            request_id,
            "diffusion-gemma",
            request,
        )?;
        processor
            .preprocess(request, images)
            .map_err(crate::openai::ApiError::from)
    }

    fn chat_request(fields: serde_json::Value) -> crate::openai::ChatCompletionRequest {
        let mut request = json!({
            "model": "diffusion-gemma",
            "messages": [{"role": "user", "content": "Hello"}],
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    /// A chat request generates its reply in canvases under the deployment's
    /// block-diffusion sampling, seeded by the request or, without a seed,
    /// by a fresh random one, and truncated at `max_completion_tokens`.
    #[tokio::test]
    async fn a_diffusion_gemma_chat_reply_is_generated_in_canvases() {
        let directory = indexed_checkpoint();
        let processor = diffusion_gemma_processor(&directory, 65_536).await;

        let (generation, _) = preprocess_chat(
            &processor,
            String::from("chat"),
            chat_request(json!({
                "seed": 5,
                "max_completion_tokens": 300,
                "stop": ["END"],
            })),
            Vec::new(),
        )
        .unwrap();
        // Every canvas the deployment generates runs under the profile's
        // sampling, with which the engine and its ranks are launched.
        let ModelParameters::DiffusionGemma(profile) = &processor.config().parameters else {
            unreachable!("the processor serves DiffusionGemma");
        };
        assert_eq!(
            profile.sampling,
            CanvasSampling {
                canvas_length: 256,
                max_steps: 48,
                entropy_bound: 0.1,
                t_min: 0.4,
                t_max: 0.8,
                confidence_threshold: 0.005,
                stability_threshold: 1,
            }
        );
        assert_eq!(generation.sampling.seed, Some(5));
        assert_eq!(generation.max_und_tokens, 300);
        assert_eq!(generation.stop_strings, ["END"]);
        let tokenizer =
            HuggingFaceTokenizer::new(&directory.path().join("tokenizer.json")).unwrap();
        assert_eq!(
            generation.prompt_token_ids,
            tokenizer.encode("<bos>Hello", false).unwrap()
        );
        assert!(generation.multimodal_inputs.images.is_empty());

        let (unseeded, _) = preprocess_chat(
            &processor,
            String::from("unseeded"),
            chat_request(json!({})),
            Vec::new(),
        )
        .unwrap();
        assert!(unseeded.sampling.seed.is_some());
        // The checkpoint's `max_new_tokens` is the default length.
        assert_eq!(unseeded.max_und_tokens, 256);
    }

    /// A reply fills only whole canvases of the context the prompt leaves.
    #[tokio::test]
    async fn a_diffusion_gemma_reply_fits_whole_canvases_of_the_context() {
        let directory = indexed_checkpoint();
        // `<bos>Hello` is six tokens, which leave 594 positions: two canvases.
        let processor = diffusion_gemma_processor(&directory, 600).await;
        let (generation, _) = preprocess_chat(
            &processor,
            String::from("long"),
            chat_request(json!({"max_completion_tokens": 1000})),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(generation.max_und_tokens, 512);

        let processor = diffusion_gemma_processor(&directory, 200).await;
        let error = preprocess_chat(
            &processor,
            String::from("full"),
            chat_request(json!({})),
            Vec::new(),
        )
        .err()
        .unwrap();
        assert_eq!(error.status_code().as_u16(), 400);
        let message = error.to_error_response().error.message;
        assert!(message.contains("256-token canvas"), "{message}");
    }

    /// An image part renders as the image token and expands into the image's
    /// soft-token run, where the engine writes the image's vision features.
    #[tokio::test]
    async fn a_diffusion_gemma_chat_image_enters_at_its_part() {
        let directory = indexed_checkpoint();
        let processor = diffusion_gemma_processor(&directory, 65_536).await;
        let mut png = Vec::new();
        image::RgbImage::new(64, 48)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let image = crate::input::ImageInput::from_bytes(png).unwrap();

        let request = chat_request(json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "What is this?"},
            ]}],
        }));
        let (generation, _) =
            preprocess_chat(&processor, String::from("image"), request, vec![image]).unwrap();

        // The image's soft-token run leaves the prompt between `<|image>` and
        // `<image|>`; the image enters there with one position and one cache
        // entry per soft token, as many as the Gemma-4 processor sizes.
        let tokenizer =
            HuggingFaceTokenizer::new(&directory.path().join("tokenizer.json")).unwrap();
        let image_start = tokenizer.token_to_id("<|image>").unwrap();
        let [input] = generation.multimodal_inputs.images.as_slice() else {
            panic!("{:?}", generation.multimodal_inputs.images);
        };
        let ModelParameters::DiffusionGemma(profile) = &processor.config().parameters else {
            unreachable!("the checkpoint is DiffusionGemma");
        };
        let soft_tokens = profile.images.soft_tokens(64, 48).unwrap();
        assert_eq!(input.num_positions, soft_tokens);
        assert_eq!(input.encoders[0].num_cache_tokens, Some(soft_tokens));
        let position = input.position as usize;
        assert_eq!(generation.prompt_token_ids[position - 1], image_start);
        assert_eq!(
            generation.prompt_token_ids[position],
            tokenizer.token_to_id("<image|>").unwrap()
        );
        assert!(
            !generation
                .prompt_token_ids
                .contains(&tokenizer.token_to_id("<|image|>").unwrap())
        );
    }

    /// Token-sampling controls and grammar constraints have no meaning for a
    /// denoised canvas; each is refused with a 400 that names it. Grammar
    /// constraints (`response_format` and guided decoding) have no field in
    /// the chat schema, so the request body itself is refused, naming the
    /// field.
    #[tokio::test]
    async fn diffusion_gemma_refuses_token_sampling_controls() {
        for field in ["response_format", "guided_json", "structured_outputs"] {
            let error = serde_json::from_value::<crate::openai::ChatCompletionRequest>(json!({
                "model": "diffusion-gemma",
                "messages": [{"role": "user", "content": "Hello"}],
                field: {"type": "json_object"},
            }))
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("unknown field `{field}`")),
                "{error}"
            );
        }

        let directory = indexed_checkpoint();
        let processor = diffusion_gemma_processor(&directory, 65_536).await;
        for (control, value) in [
            ("temperature", json!(0.7)),
            ("top_p", json!(0.9)),
            ("top_k", json!(5)),
            ("min_p", json!(0.1)),
            ("frequency_penalty", json!(0.5)),
            ("presence_penalty", json!(0.5)),
            ("repetition_penalty", json!(1.1)),
            ("logit_bias", json!({"5": 1.0})),
            ("allowed_token_ids", json!([5])),
            ("bad_words", json!(["x"])),
            ("logprobs", json!(true)),
            ("prompt_logprobs", json!(1)),
            ("min_tokens", json!(3)),
            ("ignore_eos", json!(true)),
        ] {
            let error = preprocess_chat(
                &processor,
                String::from("refused"),
                chat_request(json!({ control: value })),
                Vec::new(),
            )
            .err()
            .unwrap();
            assert_eq!(error.status_code().as_u16(), 400, "{control}");
            let body = error.to_error_response().error;
            assert_eq!(body.param.as_deref(), Some(control));
            assert!(body.message.contains(control), "{}", body.message);
        }
    }
}
