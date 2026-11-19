//! Load-time model resolution and the lowering of requests into engine requests.
//!
//! `ModelConfig::load` resolves the checkpoint's family among the families the epoch's
//! processing program declares (`crate::api::program::Program::resolve`), reads the files the
//! family declares and the tokenizer, and binds the family to them in the program: a chat
//! family's binding states the model's facts; a video family binds once its workers described
//! the placed denoiser ([`LoadedBinding`]). [`Processor::new`] then binds the generation limits
//! the started workers report, and the result is shared immutably by the serving runtime.
//!
//! [`Processor`] is the deployment's input processing, as vLLM's `Processor` is: it binds the
//! tokenizer, the program's binding, the generation limits and the sampling defaults, and
//! [`Processor::process`] turns one parsed chat request into the engine's
//! [`GenerationRequest`], the lowering its output processing is built from
//! (`proto::handoff::Lowering`), and the request's instance of the program
//! ([`Session`]), which rendered its prompt and parses its response. A video deployment's
//! `VideoService` produces its `DiffusionRequest`s through the same program.

use std::sync::Arc;

use crate::api::model::ModelConfig;
use crate::api::model::assets::{self, ChatFiles, Checkpoint, Hub, PipelineCheckpoint, Source};
use crate::api::program::{Binding, Program, Session};
use crate::api::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use proto::handoff;
use proto::request::{
    CachePolicy, GenerationFeatures, GenerationLimits, GenerationRequest, RequestId,
};
use scheduling::processing::{Bind, Bound, ChatOptions, ContentFormat, Open, Segment, Serving};
use service::span::{self, Stage};
use thiserror::Error;

use crate::api::input::sampling::resolve_max_tokens;
use crate::api::input::video::VideoService;
use crate::api::input::{Result, ServeError, TokenizeError};
use crate::api::protocol::{
    ChatCompletionRequest, ChatMessage, ChatModality, ChatToolChoice, ContentPart, MessageContent,
};

/// The deployment's input processing: the tokenizer and the program binding that turn a parsed
/// request into the engine's request. Runtime capabilities are fixed at construction, before
/// this value is shared.
///
/// Its methods are synchronous and CPU-bound; the daemon runs them on its blocking thread
/// pool.
pub struct Processor {
    pub(super) config: ModelConfig,
    pub(super) tokenizer: DynTokenizer,
    /// The family bound to the checkpoint in the epoch's processing program, whose instances
    /// requests check out.
    binding: Arc<Binding>,
    /// The video service of a video deployment.
    video: Option<VideoService>,
    // Worker-advertised generation features and resource limits.
    pub(super) limits: GenerationLimits,
    /// The candidates a request may ask for at a position: the workers' `max_logprobs`, which
    /// a kernel generation of other candidates moves while the processor serves
    /// ([`Processor::set_max_logprobs`]).
    max_logprobs: std::sync::atomic::AtomicU32,
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
    /// A file the family's binding reads cannot be read.
    #[error("reading {}", path.display())]
    ProcessorFile {
        /// The file.
        path: std::path::PathBuf,
        /// The read's failure.
        #[source]
        source: std::io::Error,
    },
    /// Model files or profile metadata cannot be resolved, or the checkpoint declares no family
    /// the epoch's processing program serves.
    #[error(transparent)]
    Assets(#[from] crate::api::model::assets::Error),
    /// A family that serves video is declared by a checkpoint with no pipeline index.
    #[error("the {family} family serves video, and this checkpoint publishes no pipeline index")]
    NoPipeline {
        /// The declared family.
        family: String,
    },
    /// The model tokenizer cannot be loaded.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    /// The processing program refuses the checkpoint or the options, or fails.
    #[error(transparent)]
    Program(#[from] crate::api::program::Error),
    /// A binding states facts of another kind than its family serves.
    #[error("the {family} family's binding states facts of another kind than it serves")]
    Facts {
        /// The family.
        family: String,
    },
    /// The running worker lacks a feature required by the selected profile.
    #[error("configured model description `{description}` requires worker feature `{feature}`")]
    MissingFeature {
        /// Stable model-profile identifier.
        description: String,
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
    /// The family bound, or to bind, in the epoch's processing program.
    pub binding: LoadedBinding,
    /// The checkpoint in the local directories its engines open.
    pub checkpoint: Checkpoint,
}

/// A loaded checkpoint's family in the epoch's processing program.
pub enum LoadedBinding {
    /// A chat family, bound to the checkpoint.
    Chat(Arc<Binding>),
    /// A video family, with what it binds with beside the description its workers state once
    /// they started.
    Video {
        /// The program.
        program: Arc<Program>,
        /// The pipeline files and options it binds with.
        bind: Box<Bind>,
    },
}

/// Loads a diffusers pipeline's `tokenizer` stage as the pipeline's own `transformers` tokenizer
/// loads it: the stage's tokenizer directory, with the tokens its configuration declares.
///
/// The conditioner reads token ids, so a declared token missing here would split into text
/// tokens the reference never presents.
pub fn pipeline_tokenizer(
    pipeline: &PipelineCheckpoint,
) -> std::result::Result<HuggingFaceTokenizer, ModelResolutionError> {
    let directory = pipeline.stage_directory("tokenizer")?;
    Ok(HuggingFaceTokenizer::load(&directory)?)
}

/// The deployment's chat options as the processing program reads them: the configured template's
/// source (a file's read exactly as written), its arguments, its content shape, and whether
/// reasoning is answered apart from content.
///
/// # Errors
///
/// [`ModelResolutionError::ProcessorFile`] when the configured template file cannot be read.
fn chat_options(
    chat: &proto::config::ChatConfig,
) -> std::result::Result<ChatOptions, ModelResolutionError> {
    use proto::config::{ChatTemplateContentFormat, ChatTemplateSource, ReasoningParser};

    let template = match &chat.template {
        Some(ChatTemplateSource::File(path)) => {
            tracing::info!(path = %path.display(), "rendering with the configured chat template file");
            Some(std::fs::read_to_string(path).map_err(|source| {
                ModelResolutionError::ProcessorFile {
                    path: path.clone(),
                    source,
                }
            })?)
        }
        Some(ChatTemplateSource::Inline(source)) => {
            tracing::info!("rendering with the configured chat template");
            Some(source.clone())
        }
        None => None,
    };
    Ok(ChatOptions {
        template,
        template_kwargs: chat
            .template_kwargs
            .iter()
            .flatten()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        content_format: chat.template_content_format.map(|format| match format {
            ChatTemplateContentFormat::String => ContentFormat::String,
            ChatTemplateContentFormat::OpenAi => ContentFormat::OpenAi,
            ChatTemplateContentFormat::Preserve => ContentFormat::Preserve,
        }),
        reasoning: chat.reasoning_parser == ReasoningParser::Auto,
    })
}

impl ModelConfig {
    /// Loads the checkpoint `model` states, with the adapter `adapter` states: its facts, the
    /// tokenizer, its family bound in the epoch's processing `program`, and the local
    /// directories its engines open.
    ///
    /// The checkpoint's root files declare its family among the program's, which decides what
    /// else is read: a chat checkpoint's every file (its engines read it whole), its tokenizer
    /// and the files its family declares, which the family is bound to with the deployment's
    /// chat options (`chat`) and processor options (`mm_processor_kwargs`); a diffusers
    /// pipeline's stage folders, over the release a layered checkpoint pins, and the files its
    /// video family's processor reads, which it binds to once its workers described the placed
    /// denoiser. A Hugging Face repository is fetched through `hub` into its cache's snapshot as
    /// each is read. The returned `max_model_len` is the configured `max_model_len`, else, for a
    /// chat checkpoint, the bound its binding states; a pipeline's deployment must state it.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint's files cannot be resolved or read, no family of `program`
    /// serves it, or the family refuses the checkpoint or the options.
    pub async fn load(
        hub: &Hub,
        model: &Source,
        adapter: Option<&Source>,
        max_model_len: Option<std::num::NonZeroU32>,
        chat: &proto::config::ChatConfig,
        processor_options: &serde_json::Map<String, serde_json::Value>,
        program: &Arc<Program>,
    ) -> std::result::Result<LoadedModel, ModelResolutionError> {
        let max_model_len = max_model_len.map(std::num::NonZeroU32::get);
        let snapshot = model.locate(hub).await?;
        snapshot.fetch(assets::root_file).await?;
        let directory = snapshot.directory();
        let family = assets::declared_family(directory, program)?;
        let adapter = match adapter {
            Some(adapter) => Some(adapter.directory(hub).await?),
            None => None,
        };

        let declared = match &family.serving {
            Serving::Chat { files } => files.clone(),
            // A diffusers pipeline names its stage folders in its root index, which locates the
            // tokenizer stage before any other asset is read.
            Serving::Video(video) => {
                let pipeline =
                    PipelineCheckpoint::open(&snapshot, family.root, video.base_pin.as_ref(), hub)
                        .await?
                        .ok_or_else(|| ModelResolutionError::NoPipeline {
                            family: family.id.clone(),
                        })?;
                let tokenizer: DynTokenizer = Arc::new(pipeline_tokenizer(&pipeline)?);
                let mut files = std::collections::BTreeMap::new();
                for (stage, name) in &video.processor_files {
                    let path = pipeline.stage_file(stage, name)?;
                    let text = std::fs::read_to_string(&path).map_err(|source| {
                        ModelResolutionError::ProcessorFile {
                            path: path.clone(),
                            source,
                        }
                    })?;
                    files.insert(format!("{stage}/{name}"), text);
                }
                let bind = Bind {
                    family: family.id.clone(),
                    files,
                    processor_options: processor_options.clone(),
                    max_model_len,
                    ..Bind::default()
                };
                return Ok(LoadedModel {
                    config: Self::from_pipeline(family, max_model_len)?,
                    tokenizer,
                    binding: LoadedBinding::Video {
                        program: Arc::clone(program),
                        bind: Box::new(bind),
                    },
                    checkpoint: Checkpoint {
                        directory: directory.to_path_buf(),
                        base: pipeline.base().map(std::path::Path::to_path_buf),
                        adapter,
                    },
                });
            }
        };

        snapshot.fetch(|_| true).await?;
        let files = ChatFiles::locate(directory, &declared)?;
        let tokenizer: DynTokenizer =
            Arc::new(HuggingFaceTokenizer::load(files.tokenizer_directory())?);
        let bind = Bind {
            family: family.id.clone(),
            files: files.texts()?,
            chat: chat_options(chat)?,
            processor_options: processor_options.clone(),
            max_model_len,
            video: None,
        };
        let binding = program.bind(bind, Arc::clone(&tokenizer))?;
        let Bound::Chat(facts) = binding.facts() else {
            return Err(ModelResolutionError::Facts { family: family.id });
        };
        let config = Self::of_chat(family, facts);
        Ok(LoadedModel {
            config,
            tokenizer,
            binding: LoadedBinding::Chat(binding),
            checkpoint: Checkpoint {
                directory: directory.to_path_buf(),
                base: None,
                adapter,
            },
        })
    }
}

impl Processor {
    /// The model's tokenizer, which output processing builds a request's incremental decoder
    /// from.
    pub fn tokenizer(&self) -> DynTokenizer {
        Arc::clone(&self.tokenizer)
    }

    /// The family's binding in the epoch's processing program, whose instances its requests
    /// check out.
    pub fn binding(&self) -> &Arc<Binding> {
        &self.binding
    }

    /// The epoch's processing program.
    pub fn program(&self) -> &Arc<Program> {
        self.binding.program()
    }

    /// Binds model resources to the generation limits the started workers
    /// report in their handshake, without rebuilding model data.
    ///
    /// `binding` is the family bound in the epoch's processing program; `video` the video
    /// service a video pipeline's denoiser handshake built.
    ///
    /// # Errors
    ///
    /// Returns [`ModelResolutionError::MissingFeature`] when `limits` do not cover the features
    /// the configured model needs.
    pub fn new(
        config: ModelConfig,
        tokenizer: DynTokenizer,
        binding: Arc<Binding>,
        video: Option<VideoService>,
        limits: GenerationLimits,
    ) -> std::result::Result<Self, ModelResolutionError> {
        let needs = match config.chat {
            Some(_) => GenerationFeatures::UNDERSTANDING,
            None => GenerationFeatures::empty(),
        };
        validate_runtime_features(&config, &limits, needs)?;
        Ok(Self {
            config,
            tokenizer,
            binding,
            video,
            max_logprobs: std::sync::atomic::AtomicU32::new(limits.max_logprobs),
            limits,
        })
    }

    /// The candidates a request may ask for at a position from now on: the serving engines'
    /// graphs compute this many, of a kernel generation that captured them for another count.
    pub fn set_max_logprobs(&self, max_logprobs: u32) {
        self.max_logprobs
            .store(max_logprobs, std::sync::atomic::Ordering::Release);
    }

    /// The candidates a request may ask for at a position now ([`Processor::set_max_logprobs`]).
    pub fn max_logprobs(&self) -> u32 {
        self.max_logprobs.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Immutable facts of the loaded model.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The video service of a video deployment; `None` for other models.
    pub fn video_service(&self) -> Option<&VideoService> {
        self.video.as_ref()
    }

    /// What the deployment serves of video, as `GET /v1/capabilities` reports it;
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
        self.config.chat.is_some_and(|chat| chat.images)
            && self
                .limits
                .covers(GenerationFeatures::VISION_ENCODE)
                .is_ok()
    }

    /// Returns whether the model's tool calls are parsed from its output.
    pub fn supports_tool_calling(&self) -> bool {
        self.config.chat.is_some_and(|chat| chat.tools)
    }

    /// Returns whether the model produces structured reasoning output.
    pub fn supports_reasoning(&self) -> bool {
        self.config.chat.is_some_and(|chat| chat.reasoning)
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
        if self.config.chat.is_none() {
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

    /// Lowers one chat completion request into its engine request, the lowering its output
    /// processing is built from, and the request's instance of the processing program.
    ///
    /// The engine request carries the rendered and tokenized prompt, the sampling parameters
    /// (each of the request's controls over the checkpoint's defaults), the generation length,
    /// the stop conditions and the cache partition of `cache_salt` and `adapter`. The lowering
    /// states what the response needs of the same request: the prompt, the decode options and
    /// the response's OpenAI fields, named `request_id`. The session rendered the prompt and
    /// holds the response's parser, opened for the prompt.
    ///
    /// `images` are the fetched `image_url` parts of the request, in
    /// `ChatCompletionRequest::image_urls` order; a text request has none. `adapter` is the
    /// LoRA adapter the request's `model` names, when it names one. `trace` is the request's
    /// trace id, which the generation request carries and its template, tokenize and
    /// preprocess spans name.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::UnsupportedFeature` for a feature the deployment does not serve,
    /// `ServeError::Processing` when the program fails, and `ServeError::Tokenize` for every
    /// failure of lowering, rendering, tokenization, validation and resource sizing.
    pub fn process(
        &self,
        request_id: &str,
        request: &ChatCompletionRequest,
        adapter: Option<&AdapterSelection>,
        images: Vec<crate::api::input::ImageInput>,
        trace: u64,
    ) -> Result<(GenerationRequest, handoff::Lowering, Session)> {
        let has_input_image = !images.is_empty() || !request.image_urls().is_empty();
        self.validate_generation_features(request_id, request, has_input_image)?;
        let tokenize = |source| ServeError::Tokenize {
            request_id: request_id.to_owned(),
            source,
        };

        validate_request(request).map_err(tokenize)?;
        let mut session = self.binding.checkout()?;
        let mut generation = GenerationRequest {
            trace,
            // The daemon's submission stamps the engine's identifier for the request.
            request_id: RequestId::default(),
            prompt_token_ids: Vec::new(),
            multimodal_inputs: Default::default(),
            sampling: super::sampling::lower(
                request,
                &self.config.sampling_defaults,
                &self.tokenizer,
                self.max_logprobs(),
            )
            .map_err(tokenize)?,
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
                isolation_key: storage::key::isolation(
                    request.cache_salt.as_deref(),
                    adapter.map(|adapter| adapter.identity.as_str()),
                ),
                ..CachePolicy::default()
            },
            generated_token_ids: Vec::new(),
            adapter: adapter.map_or(0, |adapter| adapter.slot),
            hand_off: false,
            lend: None,
        };

        self.tokenize(request_id, &mut session, request, &images, &mut generation)?;
        (|| -> std::result::Result<(), TokenizeError> {
            generation.validate()?;
            // The workers compute a position's candidates in their graphs, as many as they state.
            GenerationLimits {
                max_logprobs: self.max_logprobs(),
                ..GenerationLimits::default()
            }
            .covers_logprobs(&generation.sampling)?;
            Ok(())
        })()
        .map_err(tokenize)?;

        // The response's parser starts where the prompt leaves generation: its tail from its
        // last special token on decides whether generation starts inside a reasoning section.
        let open = Open {
            tools: super::chat::callable_tools(request),
            prompt: prompt_tail(&self.tokenizer, &generation.prompt_token_ids).to_vec(),
        };
        session
            .open(&open)?
            .map_err(|message| tokenize(TokenizeError::Parser(message)))?;

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
            response: Some(response_options(
                request_id,
                request,
                generation.sampling.n_logprobs,
            )),
        };
        Ok((generation, lowering, session))
    }
}

/// The prompt's ids from its last special token on, the whole prompt when it holds none: what
/// decides the region a response's generation starts in, as the processing program reads it
/// (`scheduling::processing::Open::prompt`).
pub fn prompt_tail<'p>(tokenizer: &HuggingFaceTokenizer, prompt: &'p [u32]) -> &'p [u32] {
    let start = prompt
        .iter()
        .rposition(|id| tokenizer.is_special_id(*id))
        .unwrap_or(0);
    &prompt[start..]
}

/// The OpenAI fields the response to `request` carries, named `request_id`.
///
/// The route has checked that `request.model` names the served model or a loaded adapter, so
/// the response names it back. `top_logprobs` is the count of a generated position's candidates
/// the engine request computes, which the response lists.
fn response_options(
    request_id: &str,
    request: &ChatCompletionRequest,
    top_logprobs: u32,
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
        created: crate::api::since_epoch().as_secs(),
        // How many of a generated position's candidates `top_logprobs` lists, the sampled
        // token first: the request's count, zero when unset, as vLLM's
        // `chat_top_logprobs_limit`.
        top_logprobs,
    }
}

/// Checks the rules of a chat request that its schema does not state.
///
/// # Errors
///
/// Refuses a `prompt_logprobs` value below `-1` or a positive or `-1` value on a streamed
/// request, a `tool_choice` that makes the model call a tool, an assistant message with no
/// reasoning, content, or tool calls or with image content, and a non-function tool call. The
/// renderer refuses template arguments that set a variable the request itself determines
/// (`HfChatRenderer::render`).
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
            description: config.id().to_owned(),
            feature,
        })
}

impl Processor {
    /// Renders `request`'s prompt in `session` and tokenizes it into `generation`, with its
    /// image blocks, its generation length and its stop tokens.
    ///
    /// The program states the prompt as text segments, each tokenized here without special
    /// tokens (the tokenizer splits the added tokens a segment's text names), and the blocks of
    /// the request's images, each written where its segment stands, in the order the images were
    /// fetched. An image's digest, which keys its encodings, covers its bytes and the request's
    /// cache partition, so requests of other partitions share no encoding.
    fn tokenize(
        &self,
        request_id: &str,
        session: &mut Session,
        request: &ChatCompletionRequest,
        images: &[crate::api::input::ImageInput],
        generation: &mut GenerationRequest,
    ) -> Result<()> {
        let trace = generation.trace;
        let tokenize = |source| ServeError::Tokenize {
            request_id: request_id.to_owned(),
            source,
        };
        let prompt = {
            let _template = span::enter(Stage::Template, trace);
            session
                .render(&super::chat::render(request, images))?
                .map_err(|refusal| tokenize(TokenizeError::Render(refusal)))?
        };

        let _tokenize = span::enter(Stage::Tokenize, trace);
        let isolation = generation.cache.isolation_key;
        let mut prompt_token_ids = Vec::new();
        let mut inputs = Vec::with_capacity(images.len());
        let mut next = images.iter();
        for segment in prompt.segments {
            match segment {
                Segment::Text(text) => {
                    let ids = self
                        .tokenizer
                        .encode(&text, false)
                        .map_err(|error| tokenize(error.into()))?;
                    prompt_token_ids.extend(ids);
                }
                Segment::Image(block) => {
                    let image = next.next().ok_or_else(|| {
                        tokenize(TokenizeError::Invalid(
                            "the prompt places more images than the request fetched".into(),
                        ))
                    })?;
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
                        encoded: image.bytes().to_vec(),
                        position: prompt_token_ids.len() as u32,
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
            }
        }
        if next.next().is_some() {
            return Err(tokenize(TokenizeError::Invalid(format!(
                "the prompt places {} of the request's {} images",
                inputs.len(),
                images.len()
            ))));
        }
        generation.multimodal_inputs.images = inputs;

        // The context holds the prompt's tokens and every image block's rows.
        let image_tokens: u32 = generation
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| &image.encoders)
            .filter_map(|encoder| encoder.num_cache_tokens)
            .sum();
        let prompt_len = prompt_token_ids.len() as u32 + image_tokens;
        let max_tokens = self
            .resolve_length(generation, request.max_output_tokens(), prompt_len)
            .map_err(tokenize)?;

        generation.prompt_token_ids = prompt_token_ids;
        generation.max_und_tokens = max_tokens as usize;
        // Prompt logprobs require computation for every prompt token, which a prefix-cache
        // read would skip.
        generation.cache.read &= !generation.sampling.prompt_logprobs_requested();
        Ok(())
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
            self.config.max_model_len,
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
            r#"{"architectures": ["ChatForCausalLM"], "max_position_embeddings": 64}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{
                "eos_token": "<|im_end|>",
                "chat_template": "{{ messages }}",
                "added_tokens_decoder": {
                    "7": {"content": "<d>", "lstrip": false, "normalized": false,
                          "rstrip": false, "single_word": false, "special": true}
                }
            }"#,
        )
        .unwrap();

        let cache = tempfile::tempdir().unwrap();
        let hub = Hub::new(assets::HubSettings {
            cache: cache.path().to_path_buf(),
            endpoint: None,
            token: None,
            offline: true,
        })
        .unwrap();
        let loaded = ModelConfig::load(
            &hub,
            &Source::Directory(directory.path().to_path_buf()),
            None,
            None,
            &proto::config::ChatConfig::default(),
            &serde_json::Map::new(),
            &crate::api::test_support::program(),
        )
        .await
        .unwrap();

        let marker = loaded
            .tokenizer
            .token_to_id("<d>")
            .expect("<d> is registered");
        assert_eq!(marker, 7);
        assert_eq!(loaded.tokenizer.encode("a<d>b", false).unwrap(), [1, 7, 2]);
        assert_eq!(loaded.config.primary_eos_token_id, Some(6));
    }

    /// The prompt's tail starts at its last special token: a reasoning delimiter before it
    /// belongs to an earlier turn.
    #[test]
    fn the_prompt_tail_starts_at_the_last_special_token() {
        let tokenizer = crate::api::test_support::configured_tokenizer();
        // `<think>` is 5, `<|im_end|>` 2 and `<|im_start|>` 1, all special; `a` is 97.
        assert_eq!(prompt_tail(&tokenizer, &[97, 5, 2, 97, 1, 97]), [1, 97]);
        assert_eq!(prompt_tail(&tokenizer, &[97, 97]), [97, 97]);
        assert_eq!(prompt_tail(&tokenizer, &[]), [] as [u32; 0]);
    }
}
