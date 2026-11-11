//! Load-time model resolution and the lowering of requests into engine requests.
//!
//! `ModelConfig::load` resolves checkpoint facts, the tokenizer, and what renders requests
//! at startup, by the family of the executable's catalogue the checkpoint declares: a diffusers
//! pipeline index selects a video family, whose processor reads the pipeline files the family
//! names, and otherwise the root `config.json` selects a chat family with its chat template.
//! [`Processor::new`] then binds the generation limits the started workers report and the request
//! renderer they complete ([`RequestRenderer`]), and the result is shared immutably by the
//! serving runtime.
//!
//! [`Processor`] is the deployment's input processing, as vLLM's `Processor` is: it binds the
//! tokenizer, the chat template, the generation limits and the sampling defaults, and
//! [`Processor::process`] turns one parsed chat request into the engine's
//! [`GenerationRequest`] and the lowering its output processing is built from
//! (`proto::handoff::Lowering`). A video deployment's `VideoService` produces its
//! `DiffusionRequest`s.

use std::sync::Arc;

use crate::model::ModelConfig;
use crate::model::assets::{PipelineCheckpoint, ResolvedModelFiles};
use crate::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use processing::Catalogue;
use processing::{
    ImageFamily, ImageProcessor, ReasoningFormat, Serving, ToolCallFormat, VideoFamily,
};
use proto::handoff;
use proto::request::{
    CachePolicy, GenerationFeatures, GenerationLimits, GenerationRequest, RequestId,
};
use thiserror::Error;

use crate::input::chat::template::{MultimodalRenderInfo, RESERVED_TEMPLATE_KWARGS};
use crate::input::chat::{GenerationPromptMode, HfChatRenderer};
use crate::input::sampling::resolve_max_tokens;
use crate::input::video::VideoService;
use crate::input::{Result, ServeError, TokenizeError};
use crate::protocol::{
    ChatCompletionRequest, ChatMessage, ChatModality, ChatToolChoice, ContentPart, MessageContent,
};

/// The deployment's input processing: the tokenizer and template resources that turn a parsed
/// request into the engine's request. Runtime capabilities are fixed at construction, before
/// this value is shared.
///
/// Its methods are synchronous and CPU-bound; the daemon runs them on its blocking thread
/// pool.
pub struct Processor {
    pub(super) config: ModelConfig,
    pub(super) tokenizer: DynTokenizer,
    renderer: RequestRenderer,
    // Worker-advertised generation features and resource limits.
    pub(super) limits: GenerationLimits,
    pub(super) parse_reasoning: bool,
    // The family's image processor, when the family takes images and the workers encode them.
    images: Option<Arc<dyn ImageProcessor>>,
}

/// The LoRA adapter a request computes with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterSelection {
    /// The slot of the workers' adapter pool it lies in.
    pub slot: u32,
    /// What keeps its cache entries apart from every other adapter's, one
    /// loaded into the same slot before included.
    pub identity: String,
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
    Chat(#[from] crate::input::chat::Error),
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

impl Processor {
    /// The model's tokenizer, which output processing builds a request's incremental decoder
    /// from.
    pub fn tokenizer(&self) -> DynTokenizer {
        Arc::clone(&self.tokenizer)
    }

    /// Binds model resources to the generation limits the started workers
    /// report in their handshake, without rebuilding model data.
    ///
    /// `renderer` is the checkpoint's chat template, or the video service a video pipeline's
    /// denoiser handshake built. `parse_reasoning` is the deployment's choice
    /// (`chat_config.reasoning_parser`) to answer reasoning apart from content.
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

        Ok(Self {
            config,
            tokenizer,
            renderer,
            limits,
            parse_reasoning,
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

    /// Returns whether the model's tool calls are parsed from its output: whether its
    /// family's chat profile names the format it writes them in.
    pub fn supports_tool_calling(&self) -> bool {
        self.config
            .chat()
            .is_some_and(|profile| profile.tool_calls.is_some())
    }

    /// Returns whether the model produces structured reasoning output.
    pub fn supports_reasoning(&self) -> bool {
        self.config
            .chat()
            .is_some_and(|profile| profile.reasoning.is_some())
    }

    /// Validates the features `request` reaches against the loaded model.
    ///
    /// Every refusal is `ServeError::UnsupportedFeature`. The final check asks the worker
    /// limits to cover the features this request reaches.
    fn validate_generation_features(
        &self,
        request_id: &str,
        request: &ChatCompletionRequest,
        has_input_image: bool,
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
        // No served model generates images.
        if request.modalities.contains(&ChatModality::Image) {
            return Err(reject("image_output"));
        }
        if request.uses_tools() && !self.supports_tool_calling() {
            return Err(reject("tool_calling"));
        }
        if request.reasoning_effort.is_some() && !self.supports_reasoning() {
            return Err(reject("reasoning"));
        }
        self.limits
            .covers(GenerationFeatures::UNDERSTANDING)
            .map_err(|feature| reject(feature.name()))
    }

    /// Lowers one chat completion request into its engine request and the lowering its output
    /// processing is built from.
    ///
    /// The engine request carries the rendered and tokenized prompt, the sampling parameters
    /// (each of the request's controls over the checkpoint's defaults), the generation length,
    /// the stop conditions and the cache partition of `cache_salt` and `adapter`. The lowering
    /// states what the response needs of the same request: the prompt, the decode options, the
    /// parsers of the model family's formats and the response's OpenAI fields, named
    /// `request_id`.
    ///
    /// `images` are the fetched `image_url` parts of the request, in
    /// `ChatCompletionRequest::image_urls` order; a text request has none. `adapter` is the
    /// LoRA adapter the request's `model` names, when it names one.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::UnsupportedFeature` for a feature the deployment does not serve,
    /// and `ServeError::Tokenize` for every failure of lowering, rendering, tokenization,
    /// validation and resource sizing.
    pub fn process(
        &self,
        request_id: &str,
        request: &ChatCompletionRequest,
        adapter: Option<&AdapterSelection>,
        images: Vec<crate::input::ImageInput>,
    ) -> Result<(GenerationRequest, handoff::Lowering)> {
        let has_input_image = !images.is_empty() || !request.image_urls().is_empty();
        self.validate_generation_features(request_id, request, has_input_image)?;

        // The closure gathers every `TokenizeError` so it maps once to `ServeError::Tokenize`.
        let generation = (|| -> std::result::Result<_, TokenizeError> {
            validate_request(request)?;
            let mut generation = GenerationRequest {
                // The daemon's submission stamps the engine's identifier for the request.
                request_id: RequestId::default(),
                prompt_token_ids: Vec::new(),
                multimodal_inputs: Default::default(),
                sampling: super::sampling::lower(
                    request,
                    &self.config.sampling_defaults,
                    &self.tokenizer,
                )?,
                max_und_tokens: 0,
                include_stop_token: request.include_stop_str_in_output,
                stop_strings: request
                    .stop
                    .as_ref()
                    .map(|stop| stop.as_slice().to_vec())
                    .unwrap_or_default(),
                stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
                priority: request.priority.unwrap_or(0),
                cache: CachePolicy {
                    isolation_key: crate::input::cache_isolation_key(
                        request.cache_salt.as_deref(),
                        adapter.map(|adapter| adapter.identity.as_str()),
                    ),
                    ..CachePolicy::default()
                },
                generated_token_ids: Vec::new(),
                adapter: adapter.map_or(0, |adapter| adapter.slot),
            };

            self.tokenize(request, &images, &mut generation)?;
            generation.validate()?;
            // The workers compute a position's candidates in their graphs, as many as they state.
            self.limits.covers_logprobs(&generation.sampling)?;
            Ok(generation)
        })()
        .map_err(|source| ServeError::Tokenize {
            request_id: request_id.to_owned(),
            source,
        })?;

        let lowering = handoff::Lowering {
            // Detokenization needs the prompt after ownership moves to the engine.
            prompt_token_ids: generation.prompt_token_ids.clone(),
            decode_options: Some(handoff::TextDecode {
                skip_special_tokens: request.skip_special_tokens,
                include_stop_str_in_output: request.include_stop_str_in_output,
                stop_strings: generation.stop_strings.clone(),
                min_tokens: request.min_tokens.unwrap_or(0),
            }),
            prompt_logprobs: generation.sampling.prompt_logprobs_requested(),
            logprobs: generation.sampling.generated_logprobs_requested(),
            parser: Some(self.parser(request)),
            response: Some(response_options(request_id, request)),
            // The daemon states both when it submits the request.
            compile_us: 0,
            started_unix_us: 0,
        };
        Ok((generation, lowering))
    }

    /// The parsers `request`'s output is read with: the formats the model family's chat profile
    /// names, reasoning when the deployment answers it apart from content, tool calls when the
    /// model may call the request's tools.
    fn parser(&self, request: &ChatCompletionRequest) -> handoff::Parser {
        let Some(profile) = self.config.chat() else {
            return handoff::Parser::default();
        };
        // The kind names the tool-call format; a family that writes none has its reasoning
        // parsed under the first kind, which then parses no tool call.
        let kind = match (profile.reasoning, profile.tool_calls) {
            (None, None) => handoff::parser::Kind::ParserKindNone,
            (_, Some(ToolCallFormat::Xml)) => handoff::parser::Kind::ParserKindXml,
            (_, Some(ToolCallFormat::Json)) | (Some(ReasoningFormat::Think), None) => {
                handoff::parser::Kind::ParserKindJson
            }
        };
        let tools = request
            .callable_tools()
            .filter(|_| profile.tool_calls.is_some());

        handoff::Parser {
            kind: kind as i32,
            parse_reasoning: self.parse_reasoning && profile.reasoning.is_some(),
            parse_tools: tools.is_some(),
            // A request's tools serialize as it stated them.
            tools: tools
                .and_then(|tools| serde_json::to_string(tools).ok())
                .unwrap_or_default(),
        }
    }
}

/// The OpenAI fields the response to `request` carries, named `request_id`.
///
/// The route has checked that `request.model` names the served model or a loaded adapter, so
/// the response names it back.
fn response_options(
    request_id: &str,
    request: &ChatCompletionRequest,
) -> handoff::ChatResponseOptions {
    handoff::ChatResponseOptions {
        request_id: request_id.to_owned(),
        response_model: request.model.clone(),
        include_usage: request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        requested_logprobs: request.logprobs,
        // Only a response answered as one document carries prompt logprobs.
        include_prompt_logprobs: request.prompt_logprobs.is_some(),
        include_reasoning: request.include_reasoning,
        return_token_ids: request.return_token_ids.unwrap_or(false),
        return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
        // One timestamp for the whole response, shared by every streamed chunk.
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        // How many of a generated position's candidates `top_logprobs` lists, the sampled
        // token first: the request's count, zero when unset and every one for `-1`, as
        // vLLM's `chat_top_logprobs_limit`.
        top_logprobs: request
            .top_logprobs
            .map_or(0, |count| u32::try_from(count).unwrap_or(u32::MAX)),
    }
}

/// Checks the rules of a chat request that its schema does not state.
///
/// # Errors
///
/// Refuses a `prompt_logprobs` value below `-1` or a positive or `-1` value on a streamed
/// request, a `tool_choice` that makes the model call a tool, template arguments that set a
/// variable the request itself determines, an assistant
/// message with no reasoning, content, or tool calls or with image content, and a non-function
/// tool call.
fn validate_request(request: &ChatCompletionRequest) -> std::result::Result<(), TokenizeError> {
    let invalid = |param: &'static str, message: &str| TokenizeError::InvalidParameter {
        param: Some(param).filter(|param| !param.is_empty()),
        message: message.to_owned(),
    };

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        // Any non-negative count, or `-1` for the full vocabulary.
        if prompt_logprobs < -1 {
            return Err(invalid(
                "prompt_logprobs",
                "prompt_logprobs must be a non-negative value or -1.",
            ));
        }

        // Stream chunks have no `prompt_logprobs` field, so a streamed response could not
        // deliver them. A streamed `0` is accepted, as vLLM accepts it, and the sampling
        // lowering drops it: scoring every prompt position would disable prefix-cache reads
        // for output the client never receives.
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            return Err(invalid(
                "prompt_logprobs",
                "prompt_logprobs are not available when stream=true.",
            ));
        }
    }

    // Holding a model to a tool call takes constrained decoding, as vLLM serves both of these
    // choices, and this server has none: a model left free would answer in text under a
    // response that promised a call.
    let unserved = match &request.tool_choice {
        Some(ChatToolChoice::Required) => Some("`required`".to_owned()),
        Some(ChatToolChoice::Function(name)) => Some(format!("the function `{name}`")),
        Some(ChatToolChoice::Auto | ChatToolChoice::None) | None => None,
    };
    if let Some(choice) = unserved {
        return Err(invalid(
            "tool_choice",
            &format!(
                "tool_choice {choice} is not served: it needs constrained decoding, which this \
                 server does not provide. Use `auto` or `none`."
            ),
        ));
    }

    // Template arguments may not replace the variables the renderer derives from the request
    // itself.
    if let Some(kwargs) = request.chat_template_kwargs.as_ref()
        && let Some(name) = RESERVED_TEMPLATE_KWARGS
            .into_iter()
            .find(|name| kwargs.contains_key(*name))
    {
        return Err(invalid(
            "chat_template_kwargs",
            &format!(
                "chat_template_kwargs may not set `{name}`, which the request itself determines."
            ),
        ));
    }

    for message in &request.messages {
        let ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            ..
        } = message
        else {
            continue;
        };
        if let Some(MessageContent::Parts(parts)) = content
            && parts
                .iter()
                .any(|part| matches!(part, ContentPart::ImageUrl { .. }))
        {
            return Err(invalid("", "Assistant message content must be text."));
        }
        let tool_calls = tool_calls.as_deref().unwrap_or_default();
        if tool_calls.iter().any(|call| call.tool_type != "function") {
            return Err(invalid("", "Only function tool calls are supported."));
        }
        let has_reasoning = reasoning.as_ref().is_some_and(|text| !text.is_empty());
        let has_content = match content {
            None => false,
            Some(MessageContent::Text(_)) => true,
            Some(MessageContent::Parts(parts)) => !parts.is_empty(),
        };
        if !has_reasoning && !has_content && tool_calls.is_empty() {
            return Err(invalid(
                "",
                "Assistant messages must contain content or tool_calls.",
            ));
        }
    }
    Ok(())
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

impl Processor {
    /// Renders and tokenizes `request`'s prompt into `generation`, with its image blocks, its
    /// generation length and its stop tokens.
    fn tokenize(
        &self,
        request: &ChatCompletionRequest,
        images: &[crate::input::ImageInput],
        generation: &mut GenerationRequest,
    ) -> std::result::Result<(), TokenizeError> {
        let rendered_text = self
            .chat_renderer()
            .ok_or(crate::input::chat::Error::MissingChatTemplate)?
            .render(request, GenerationPromptMode::StartNewAssistant)?;
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
        let max_tokens =
            self.resolve_length(generation, request.max_output_tokens(), prompt_len)?;

        generation.prompt_token_ids = prompt_token_ids;
        generation.max_und_tokens = max_tokens as usize;
        // Prompt logprobs require computation for every prompt token, which a prefix-cache
        // read would skip.
        generation.cache.read &= !generation.sampling.prompt_logprobs_requested();
        Ok(())
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
    ) -> std::result::Result<Vec<proto::request::ImageInput>, TokenizeError> {
        use base64::Engine as _;

        let Some(processor) = self.images.as_ref() else {
            return Ok(Vec::new());
        };
        let placeholder = self
            .tokenizer
            .token_to_id(processor.placeholder())
            .ok_or_else(|| {
                TokenizeError::Invalid(format!(
                    "the vocabulary has no image placeholder `{}`",
                    processor.placeholder()
                ))
            })?;
        let found = prompt.iter().filter(|id| **id == placeholder).count();
        if found != images.len() {
            return Err(TokenizeError::Invalid(format!(
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
                TokenizeError::Invalid("an image placeholder has no image".into())
            })?;
            let block = processor
                .block(image.width(), image.height())
                .map_err(TokenizeError::Invalid)?;
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
    ) -> std::result::Result<u32, TokenizeError> {
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
            return Err(TokenizeError::MinTokensExceedsMaximum {
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
