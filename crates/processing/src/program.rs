//! The processing program: a build's families served through the processing interface
//! (`scheduling::processing`), as the leader daemon loads it per epoch.
//!
//! [`Program`] implements the interface over a list of families ([`Families`]), which a program
//! crate exports: the release's (`crates/processing/program`) over [`Release`], `crate::FAMILIES`,
//! and the daemon tests' (`crates/processing/testing`) over its test families. An instance is
//! bound to one family and one checkpoint ([`Program::bind_with`]): a chat family's binding reads
//! the checkpoint's files the family declared ([`declare`]), compiles its template and states the
//! model's facts; a video family's binds the family's processor to what the workers described.
//! The bound instance renders chat requests, parses one response at a time and applies a video
//! family's rules.
//!
//! The vocabulary of the bound checkpoint's tokenizer is the daemon's, which a module reaches
//! through its host functions ([`HostVocabulary`]); the calls that read it take it as a value,
//! so a native build of the crate drives the same code with a vocabulary of its own.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use scheduling::processing::video::{self, Facts, InputError, Process, VideoInput, VideoSpec};
use scheduling::processing::{
    self as interface, Bind, Bound, ChatFacts, CheckpointFile, Event, ImageSize, Open, Processing,
    Prompt, Refusal, Render, SamplingDefaults, Segment,
};
use serde_json::Value;

use crate::chat::metadata::{self, GenerationConfig, TokenizerConfig};
use crate::chat::{GenerationPromptMode, HfChatRenderer, MultimodalRenderInfo};
use crate::family::{Family, RootFile};
use crate::image::ImageProcessor;
use crate::output::{self, Delimiters, Options, Parser};
use crate::serving::{ChatProfile, ChatTemplate, Serving};
use crate::video::{VideoBinding, VideoProcessor};
use crate::vocabulary::{Vocabulary, VocabularyError};

/// The families a processing program serves.
pub trait Families: 'static {
    /// The families, in the order the program declares them.
    fn families() -> &'static [Family];
}

/// The release's families (`crate::FAMILIES`), which the release's program
/// (`crates/processing/program`) exports and the native tests and bench drive.
pub struct Release;

impl Families for Release {
    fn families() -> &'static [Family] {
        &crate::FAMILIES
    }
}

/// One instance of the processing program over the families `F`: the family it is bound to and
/// the parser of the response it serves.
pub struct Program<F> {
    binding: Binding,
    parser: Option<Parser>,
    families: PhantomData<F>,
}

/// What an instance is bound to.
enum Binding {
    /// A chat family, bound to a checkpoint.
    Chat(Box<Chat>),
    /// A video family's processor, bound to the placed denoiser.
    Video(Box<dyn VideoProcessor>),
}

/// A chat family bound to one checkpoint.
struct Chat {
    profile: ChatProfile,
    renderer: HfChatRenderer,
    /// The family's image processor, bound to the checkpoint's processor files; `None` for a
    /// family that serves text alone.
    images: Option<Box<dyn ImageProcessor>>,
    /// Whether the deployment answers reasoning apart from content.
    reasoning: bool,
    /// The family's reasoning delimiters with their ids, or why the vocabulary lacks one; `None`
    /// for a family that delimits no reasoning.
    delimiters: Option<Result<Delimiters, String>>,
}

impl Chat {
    /// The reasoning delimiters a parser is built with: `Ok(None)` for a family without
    /// reasoning, an error naming the token the vocabulary lacks.
    fn delimiters(&self) -> output::Result<Option<Delimiters>> {
        match &self.delimiters {
            None => Ok(None),
            Some(Ok(delimiters)) => Ok(Some(*delimiters)),
            Some(Err(message)) => {
                let format = self.profile.reasoning.map_or_else(String::new, |format| {
                    format!("{} {}", format.open, format.close)
                });
                Err(output::Error::Parser {
                    kind: "reasoning",
                    name: format,
                    message: message.clone(),
                })
            }
        }
    }
}

/// The declaration of `family` the program states (`families`): its identity, its claims, and
/// the files of a checkpoint its binding reads.
pub fn declare(family: &Family) -> interface::Family {
    let serving = match family.serving {
        Serving::Chat(profile) => interface::Serving::Chat {
            files: chat_files(profile),
        },
        Serving::Video(video) => interface::Serving::Video(video::Pipeline {
            processor_files: video
                .processor_files()
                .iter()
                .map(|(stage, file)| ((*stage).to_owned(), (*file).to_owned()))
                .collect(),
            processing_parts: video
                .processing_parts()
                .iter()
                .map(|part| (*part).to_owned())
                .collect(),
            base_pin: video.base_pin().map(|pin| video::BasePin {
                file: pin.file.to_owned(),
                field: pin.field.to_owned(),
            }),
            default_sizes: video
                .default_sizes()
                .iter()
                .map(|size| (*size).to_owned())
                .collect(),
            default_max_seconds: video.default_max_seconds(),
        }),
    };
    interface::Family {
        id: family.id.to_owned(),
        root: family.root,
        classes: family
            .classes
            .iter()
            .map(|class| (*class).to_owned())
            .collect(),
        serving,
    }
}

/// The files at a chat checkpoint's root its binding reads, beside the tokenizer definition:
/// the configuration, the tokenizer metadata, the generation defaults when published, the
/// template's file (required of a checkpoint only when it is the one place the family keeps
/// its template), and the image processor's files of a family that takes images.
fn chat_files(profile: ChatProfile) -> Vec<CheckpointFile> {
    let file = |name: &str, required: bool| CheckpointFile {
        name: name.to_owned(),
        required,
    };
    let mut files = vec![
        file(RootFile::Config.name(), true),
        file(metadata::TOKENIZER_CONFIG, true),
        file(metadata::GENERATION_CONFIG, false),
    ];
    match profile.checkpoint.chat_template {
        ChatTemplate::File(name) => files.push(file(name, true)),
        // A checkpoint without the file keeps its template in the tokenizer configuration,
        // which every chat checkpoint publishes.
        ChatTemplate::FileOrTokenizerConfig(name) => files.push(file(name, false)),
    }
    if let Some(images) = profile.images {
        files.extend(images.processor_files().iter().map(|name| file(name, true)));
    }
    files
}

impl<F: Families> Program<F> {
    /// Binds `bind.family` to the checkpoint `bind` states, resolving the control tokens the
    /// family names through `vocabulary`, the checkpoint's tokenizer; and the facts the daemon
    /// serves the model with.
    ///
    /// # Errors
    ///
    /// Names the family when the program serves none of its identity, a file the family reads
    /// that the checkpoint does not publish or that does not parse, a context length the
    /// configuration does not state where the family keeps it, a template that is missing or
    /// does not compile, processor options the family does not take, and a video family's
    /// description or processor files that are not its own.
    pub fn bind_with(bind: Bind, vocabulary: &dyn Vocabulary) -> Result<(Self, Bound), String> {
        let family = F::families()
            .iter()
            .find(|family| family.id == bind.family)
            .ok_or_else(|| format!("this program serves no family {:?}", bind.family))?;
        let (binding, bound) = match family.serving {
            Serving::Chat(profile) => {
                let (chat, facts) = bind_chat(family, profile, &bind, vocabulary)?;
                (Binding::Chat(Box::new(chat)), Bound::Chat(facts))
            }
            Serving::Video(video) => {
                let binding = bind
                    .video
                    .as_ref()
                    .ok_or("a video family binds once its workers described the denoiser")?;
                let files = video
                    .processor_files()
                    .iter()
                    .map(|(stage, file)| {
                        let path = format!("{stage}/{file}");
                        bind.files
                            .get(&path)
                            .cloned()
                            .ok_or_else(|| format!("the pipeline publishes no {path}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let processor = video.bind(VideoBinding {
                    description: &binding.description,
                    max_seconds: binding.max_seconds,
                    processor_options: &bind.processor_options,
                    files: &files,
                })?;
                let facts = Facts {
                    capabilities: processor.capabilities(),
                    denoise_steps: processor.denoise_steps(),
                };
                (Binding::Video(processor), Bound::Video(facts))
            }
        };
        Ok((
            Self {
                binding,
                parser: None,
                families: PhantomData,
            },
            bound,
        ))
    }

    /// Processes a video request whose references were probed, tokenizing its prompt with
    /// `vocabulary`.
    ///
    /// # Errors
    ///
    /// The bound family's refusal, and an internal failure for an instance bound to a chat
    /// family.
    pub fn process_with(
        &self,
        process: &Process,
        vocabulary: &dyn Vocabulary,
    ) -> Result<VideoInput, InputError> {
        self.video()?
            .process(&process.spec, &process.media, vocabulary)
    }

    /// The bound video family's processor.
    fn video(&self) -> Result<&dyn VideoProcessor, InputError> {
        match &self.binding {
            Binding::Video(processor) => Ok(processor.as_ref()),
            Binding::Chat(_) => Err(InputError::Internal(
                "the bound family serves chat, not video".to_owned(),
            )),
        }
    }

    /// The bound chat family.
    fn chat(&self) -> Result<&Chat, String> {
        match &self.binding {
            Binding::Chat(chat) => Ok(chat),
            Binding::Video(_) => Err("the bound family serves video, not chat".to_owned()),
        }
    }

    /// The open parser.
    fn parser(&mut self) -> Result<&mut Parser, String> {
        self.parser
            .as_mut()
            .ok_or_else(|| "no parser is open: the response's parser opens first".to_owned())
    }
}

/// Binds the chat `family`, whose profile is `profile`, to the checkpoint `bind` states.
fn bind_chat(
    family: &Family,
    profile: ChatProfile,
    bind: &Bind,
    vocabulary: &dyn Vocabulary,
) -> Result<(Chat, ChatFacts), String> {
    let file = |name: &str| bind.files.get(name).map(String::as_str);
    let required = |name: &str| {
        file(name).ok_or_else(|| {
            format!(
                "the checkpoint publishes no {name}, which the {} family reads",
                family.id
            )
        })
    };

    let max_model_len = match bind.max_model_len {
        Some(configured) => configured,
        None => context_length(required(RootFile::Config.name())?, profile)?,
    };
    let tokenizer: TokenizerConfig = metadata::parse(
        metadata::TOKENIZER_CONFIG,
        required(metadata::TOKENIZER_CONFIG)?,
    )?;
    let generation: Option<GenerationConfig> = file(metadata::GENERATION_CONFIG)
        .map(|text| metadata::parse(metadata::GENERATION_CONFIG, text))
        .transpose()?;

    // The primary EOS is the tokenizer configuration's `eos_token`, resolved through the
    // vocabulary; an `eos_token` the vocabulary lacks yields no primary rather than an error.
    let primary_eos = tokenizer
        .special_tokens
        .eos_token
        .as_ref()
        .and_then(|token| vocabulary.token_id(token.as_str()));
    let mut eos: BTreeSet<u32> = generation
        .as_ref()
        .and_then(|config| config.eos_token_id.clone())
        .map(metadata::OneOrManyTokenIds::into_set)
        .unwrap_or_default();
    eos.extend(primary_eos);

    // A deployment's template replaces the checkpoint's in every layout. The checkpoint's own is
    // read where its family's checkpoints keep it, a standalone file before the tokenizer
    // configuration's field, as `transformers` loads a checkpoint's template.
    let template = match &bind.chat.template {
        Some(template) => template.clone(),
        None => match profile.checkpoint.chat_template {
            ChatTemplate::File(name) => required(name)?.to_owned(),
            ChatTemplate::FileOrTokenizerConfig(name) => match file(name) {
                Some(source) => source.to_owned(),
                None => tokenizer.chat_template.clone().ok_or_else(|| {
                    format!(
                        "the checkpoint publishes no {name} and its {} states no \
                         chat_template, the places the {} family's checkpoints keep their \
                         template",
                        metadata::TOKENIZER_CONFIG,
                        family.id
                    )
                })?,
            },
        },
    };

    // A family that takes images binds its processor to the checkpoint's processor files; a
    // template that reads content parts writes its own image markup, and one that reads
    // flattened content is given the family's.
    let images = match profile.images {
        Some(images) => {
            let files = images
                .processor_files()
                .iter()
                .map(|name| required(name).map(str::to_owned))
                .collect::<Result<Vec<_>, _>>()?;
            let processor = images
                .bind(&files, &bind.processor_options)
                .map_err(|error| format!("binding the image processor: {error}"))?;
            Some(processor)
        }
        None if bind.processor_options.is_empty() => None,
        None => {
            return Err(format!(
                "binding the image processor: the {} family binds no image processor, which \
                 `mm_processor_kwargs` configures",
                family.id
            ));
        }
    };
    let multimodal = images.as_ref().map(|images| MultimodalRenderInfo {
        placeholder_token: images.markup().to_owned(),
    });
    let special_tokens = Some(tokenizer.special_tokens).filter(|tokens| !tokens.is_empty());
    let renderer = HfChatRenderer::new(
        Some(template),
        bind.chat.template_kwargs.clone(),
        bind.chat.content_format,
    )
    .map_err(|error| error.to_string())?
    .with_special_tokens(special_tokens)
    .with_multimodal(multimodal);

    // Each delimiter is one token of the vocabulary, whose id decides from the prompt where
    // generation starts.
    let delimiters = profile.reasoning.map(|format| {
        let id = |token: &str| {
            vocabulary
                .token_id(token)
                .ok_or_else(|| format!("tokenizer is missing reasoning delimiter token `{token}`"))
        };
        Ok(Delimiters {
            format,
            open: id(format.open)?,
            close: id(format.close)?,
        })
    });

    let facts = ChatFacts {
        max_model_len,
        primary_eos,
        eos: eos.into_iter().collect(),
        sampling: generation
            .as_ref()
            .map(sampling_defaults)
            .unwrap_or_default(),
        tools: profile.tool_calls.is_some(),
        reasoning: profile.reasoning.is_some(),
        images: profile.images.is_some(),
    };
    let chat = Chat {
        profile,
        renderer,
        images,
        reasoning: bind.chat.reasoning,
        delimiters,
    };
    Ok((chat, facts))
}

/// The longest sequence the model's positions cover, in tokens, stated in `config` (the text of
/// `config.json`) where the family's checkpoints state it.
///
/// # Errors
///
/// Fails when the configuration does not parse, or states no count of 32-bit range at the
/// family's field.
fn context_length(config: &str, profile: ChatProfile) -> Result<u32, String> {
    let field = profile.checkpoint.context_length;
    let config: Value = metadata::parse(RootFile::Config.name(), config)?;
    field
        .iter()
        .try_fold(&config, |value, key| value.get(key))
        .and_then(Value::as_u64)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| {
            format!(
                "{} states no context length at `{}`, where the checkpoint's family states it",
                RootFile::Config.name(),
                field.join(".")
            )
        })
}

/// The serving generation defaults a checkpoint's generation configuration states.
fn sampling_defaults(config: &GenerationConfig) -> SamplingDefaults {
    SamplingDefaults {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        min_p: config.min_p,
        repetition_penalty: config.repetition_penalty,
        max_output_tokens: config.max_new_tokens,
    }
}

/// The prompt of `text`, a rendered request, for a family whose image processor is `images`:
/// the text cut at each image's placeholder token, where the block of the next of `sizes`
/// stands.
///
/// # Errors
///
/// Refuses a prompt whose placeholders are not one per image, and an image the processor
/// cannot size.
fn prompt(
    text: String,
    sizes: &[ImageSize],
    images: Option<&dyn ImageProcessor>,
) -> Result<Prompt, Refusal> {
    let Some(images) = images else {
        if !sizes.is_empty() {
            return Err(Refusal::new("the served family takes no images"));
        }
        return Ok(Prompt {
            segments: vec![Segment::Text(text)],
        });
    };

    let pieces: Vec<&str> = text.split(images.placeholder()).collect();
    let found = pieces.len() - 1;
    if found != sizes.len() {
        return Err(Refusal::new(format!(
            "the prompt renders {found} image placeholder(s) for {} image(s)",
            sizes.len()
        )));
    }
    let mut segments = Vec::with_capacity(pieces.len() + sizes.len());
    for (index, piece) in pieces.iter().enumerate() {
        if !piece.is_empty() {
            segments.push(Segment::Text((*piece).to_owned()));
        }
        if let Some(size) = sizes.get(index) {
            let block = images
                .block(size.width, size.height)
                .map_err(Refusal::new)?;
            segments.push(Segment::Image(interface::ImageBlock {
                tokens: block.tokens,
                positions: block.positions,
            }));
        }
    }
    Ok(Prompt { segments })
}

impl<F: Families> Processing for Program<F> {
    fn families() -> Vec<interface::Family> {
        F::families().iter().map(declare).collect()
    }

    fn bind(bind: Bind) -> Result<(Self, Bound), String> {
        Self::bind_with(bind, &HostVocabulary)
    }

    fn render(&mut self, render: Render) -> Result<Prompt, Refusal> {
        let chat = self.chat().map_err(Refusal::new)?;
        let text = chat
            .renderer
            .render(&render, GenerationPromptMode::StartNewAssistant)?;
        prompt(text, &render.images, chat.images.as_deref())
    }

    fn open(&mut self, open: Open) -> Result<(), String> {
        let chat = self.chat()?;
        let options = Options {
            reasoning: chat.reasoning && chat.profile.reasoning.is_some(),
            tools: open.tools.filter(|_| chat.profile.tool_calls.is_some()),
        };
        let parser = Parser::new(
            options,
            chat.delimiters(),
            chat.profile.tool_calls,
            &open.prompt,
        )
        .map_err(|error| error.report())?;
        self.parser = Some(parser);
        Ok(())
    }

    fn push(&mut self, text: &str) -> Result<Vec<Event>, String> {
        self.parser()?.push(text).map_err(|error| error.report())
    }

    fn finish(&mut self) -> Result<Vec<Event>, String> {
        let events = self.parser()?.finish().map_err(|error| error.report());
        self.parser = None;
        events
    }

    fn save(&self) -> Vec<u8> {
        let state = self.parser.as_ref().map(Parser::save);
        // A state of strings, flags and counts always encodes.
        serde_json::to_vec(&state).unwrap_or_default()
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), String> {
        let state: Option<output::State> = serde_json::from_slice(state)
            .map_err(|error| format!("the parser's state does not decode: {error}"))?;
        let Some(state) = state else {
            self.parser = None;
            return Ok(());
        };
        let chat = self.chat()?;
        let parser = Parser::restore(state, chat.delimiters(), chat.profile.tool_calls)
            .map_err(|error| error.report())?;
        self.parser = Some(parser);
        Ok(())
    }

    fn check_video(&self, spec: &VideoSpec) -> Result<(), InputError> {
        self.video()?.check(spec)
    }

    fn process_video(&self, process: Process) -> Result<VideoInput, InputError> {
        self.process_with(&process, &HostVocabulary)
    }
}

/// The vocabulary of the bound checkpoint's tokenizer, which the daemon holds and a module
/// reaches through `uniserve.token` and `uniserve.encode`.
pub struct HostVocabulary;

impl Vocabulary for HostVocabulary {
    fn token_id(&self, token: &str) -> Option<u32> {
        interface::guest::token(token)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, VocabularyError> {
        interface::guest::encode_text(text).map_err(|error| VocabularyError(error.into()))
    }
}

#[cfg(test)]
mod tests;
