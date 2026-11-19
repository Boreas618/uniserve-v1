//! The processing program's interface: how the leader daemon asks an epoch's request processing
//! (`crates/processing`, built as a wasm32 module) what the epoch's model families are, binds one
//! to a checkpoint, renders chat requests into prompts, parses completions into reasoning,
//! content and tool calls, and applies a video family's request rules.
//!
//! The daemon holds what is common to every model (the OpenAI protocol, tokenization and
//! detokenization, stop strings, sampling, media fetching and decoding, response assembly); a
//! family's request side is this program's, fixed with the epoch's module set from the same
//! source tree as the epoch's engine. A module states [`Kind::Processing`](crate::program::Kind)
//! and [`INTERFACE_VERSION`] in its [`SECTION`](crate::program::SECTION), which the daemon checks
//! before it links it.
//!
//! # Exports
//!
//! A processing program implements [`Processing`] and invokes
//! [`export_processing!`](crate::export_processing) once. Every message and reply is JSON of the
//! named type in a buffer of the module's memory, as for every program
//! ([`crate::program`]); `alloc` and `free` are the same exports. A processing export is called
//! once per decoded update of every streaming response, so its convention takes one entry into
//! the module: the host writes a message of at most the module's message buffer's length into
//! that buffer (the `inbox` export, `() -> u64`, its place and length packed, which the host reads
//! once per instance), and a longer one into a buffer of `alloc`, which the export frees once it
//! decoded it; its reply is the module's one reply buffer, which lives until the module's next
//! call and which the host reads in place and does not free.
//!
//! | Export | Message | Reply | Called |
//! | --- | --- | --- | --- |
//! | `families` | | `Vec<`[`Family`]`>` | once per module |
//! | `bind` | [`Bind`] | `Result<`[`Bound`]`, String>` | once per instance, before every other call |
//! | `render` | [`Render`] | `Result<`[`Prompt`]`, `[`Refusal`]`>` | once per chat request |
//! | `parse_open` | [`Open`] | `Result<(), String>` | once per chat response |
//! | `parse_push` | `String`, decoded text | `Result<Vec<`[`Event`]`>, String>` | every decoded update |
//! | `parse_finish` | | `Result<Vec<`[`Event`]`>, String>` | at the end of generation |
//! | `parse_save` | | `Vec<u8>`, the parser's state | at a handoff |
//! | `parse_restore` | `Vec<u8>` | `Result<(), String>` | in the successor, in place of `parse_open` |
//! | `video_check` | [`video::VideoSpec`] | `Result<(), `[`video::InputError`]`>` | before a video request's media are fetched |
//! | `video_process` | [`video::Process`] | `Result<`[`video::VideoInput`]`, `[`video::InputError`]`>` | once its media are probed |
//!
//! An instance serves one bound family and holds at most one parser: the daemon checks an
//! instance out of its epoch's pool for one request and keeps it to the request's end, so a
//! parser's state needs no identity, and a trap in a family's code fails the request the
//! instance served alone. `parse_open` and `parse_restore` replace the parser the instance held.
//!
//! # Host functions
//!
//! Beside `uniserve.log`, a processing program may import the vocabulary of the bound
//! checkpoint's tokenizer, which the daemon holds: `uniserve.token` (`(ptr, len) -> i64`, the id
//! of the token whose text is the UTF-8 bytes at `ptr`, or -1) and `uniserve.encode`
//! (`(ptr, len) -> u64`: the message is the text as a JSON string, the reply
//! `Result<Vec<u32>, String>` its encoding without special tokens, in a buffer of the module's
//! `alloc`). A video family's prompt is tokenized through them ([`guest::token`],
//! [`guest::encode_text`]); a chat prompt is returned as text, which the daemon tokenizes
//! ([`Prompt`]).

pub mod chat;
pub mod guest;
pub mod video;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub use chat::{
    Content, ContentPart, Event, Function, FunctionCall, ImageBlock, ImageSize, Message, Open,
    Prompt, Refusal, Render, Segment, Tool, ToolCall,
};

/// The version of the processing interface: the exports, the host functions and the messages of
/// this module. Every change that is not an addition raises it, and a daemon loads only a
/// program of its own version.
pub const INTERFACE_VERSION: u32 = 1;

/// The names of the exports and of the host functions only a processing program has.
pub mod name {
    pub const FAMILIES: &str = "families";
    pub const BIND: &str = "bind";
    pub const RENDER: &str = "render";
    pub const PARSE_OPEN: &str = "parse_open";
    pub const PARSE_PUSH: &str = "parse_push";
    pub const PARSE_FINISH: &str = "parse_finish";
    pub const PARSE_SAVE: &str = "parse_save";
    pub const PARSE_RESTORE: &str = "parse_restore";
    pub const VIDEO_CHECK: &str = "video_check";
    pub const VIDEO_PROCESS: &str = "video_process";
    pub const INBOX: &str = "inbox";

    pub const TOKEN: &str = "token";
    pub const ENCODE: &str = "encode";
}

/// The functions a processing module exports beside its `memory`.
pub const EXPORTS: [&str; 13] = [
    crate::program::name::ALLOC,
    crate::program::name::FREE,
    name::INBOX,
    name::FAMILIES,
    name::BIND,
    name::RENDER,
    name::PARSE_OPEN,
    name::PARSE_PUSH,
    name::PARSE_FINISH,
    name::PARSE_SAVE,
    name::PARSE_RESTORE,
    name::VIDEO_CHECK,
    name::VIDEO_PROCESS,
];

/// A file at a checkpoint's root that names the checkpoint's class.
///
/// Each checkpoint format declares its class in one file under one key, and a family names the
/// format its checkpoints are published in: no file is read in place of another, and no second
/// key stands in for the class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootFile {
    /// A transformers checkpoint's `config.json`, whose `architectures[0]` names the model
    /// class.
    Config,
    /// A diffusers modular pipeline's `modular_model_index.json`, whose `_class_name` names the
    /// pipeline class.
    ModularIndex,
}

impl RootFile {
    /// The file's name at the checkpoint's root.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Config => "config.json",
            Self::ModularIndex => "modular_model_index.json",
        }
    }

    /// The class the file's JSON object names, if it names one.
    pub fn class(self, object: &Map<String, Value>) -> Option<&str> {
        match self {
            Self::Config => object
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|architectures| architectures.first())
                .and_then(Value::as_str),
            Self::ModularIndex => object.get("_class_name").and_then(Value::as_str),
        }
    }
}

/// One model family as a processing program declares it (`families`): its identity, the classes
/// its checkpoints declare, and what the daemon reads of a checkpoint to bind it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Family {
    /// The family's stable identifier, as the daemon reports it.
    pub id: String,
    /// The root file the family's checkpoints name their class in.
    pub root: RootFile,
    /// The classes a checkpoint of this family names in [`Self::root`].
    pub classes: Vec<String>,
    /// What the family serves and what of a checkpoint its binding reads.
    pub serving: Serving,
}

/// What a family serves, with the checkpoint files its binding reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Serving {
    /// Chat and completions over a causal language model whose tokenizer (`tokenizer.json`
    /// with `tokenizer_config.json`) lies at the checkpoint's root.
    Chat {
        /// The files at the checkpoint's root its binding reads, beside the tokenizer
        /// definition; every one of them decides how a request is processed.
        files: Vec<CheckpointFile>,
    },
    /// Video generation over a diffusers pipeline, whose prompt tokenizer is its `tokenizer`
    /// stage.
    Video(video::Pipeline),
}

/// A file a family's binding reads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointFile {
    /// The file's path relative to the checkpoint's root.
    pub name: String,
    /// Whether a checkpoint of the family must publish it; an optional file is bound without
    /// when the checkpoint publishes none.
    pub required: bool,
}

/// Why a checkpoint resolves to no family ([`resolve`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// A root file the checkpoint publishes is not a JSON object.
    #[error("the checkpoint's {} is not a JSON object: {message}", file.name())]
    Malformed {
        /// The file.
        file: RootFile,
        /// What is wrong with it.
        message: String,
    },
    /// No root file the families declare in names a class.
    #[error(
        "the checkpoint declares no model class: none of {files} it publishes names one",
        files = files.join(", ")
    )]
    Undeclared {
        /// The root files the families declare their classes in.
        files: Vec<&'static str>,
    },
    /// No family claims what the checkpoint declares.
    #[error(
        "the checkpoint declares {declared}, which this build cannot serve: it serves {served}"
    )]
    Unserved {
        /// Each class the checkpoint declares, with the file that declares it.
        declared: String,
        /// Every class the families claim.
        served: String,
    },
    /// Two root files declare classes of two families.
    #[error("the checkpoint declares {first} and {second}, two families' classes")]
    Ambiguous {
        /// The first family's class, with the file that declares it.
        first: String,
        /// The second family's class, with the file that declares it.
        second: String,
    },
}

/// A family as [`resolve`] reads it: the root file it declares in and the classes it claims
/// there. Both a program's declarations ([`Family`]) and the native families a build links
/// implement it, so one rule resolves a checkpoint wherever it is resolved.
pub trait Claim {
    /// The family's identity.
    fn id(&self) -> &str;
    /// The root file its checkpoints name their class in.
    fn root(&self) -> RootFile;
    /// The classes it claims.
    fn classes(&self) -> Vec<&str>;
}

impl Claim for Family {
    fn id(&self) -> &str {
        &self.id
    }

    fn root(&self) -> RootFile {
        self.root
    }

    fn classes(&self) -> Vec<&str> {
        self.classes.iter().map(String::as_str).collect()
    }
}

/// The root files `families` declare their classes in, each once, in their order: the files a
/// checkpoint is resolved from.
pub fn root_files<C: Claim>(families: &[C]) -> Vec<RootFile> {
    let mut files = Vec::new();
    for family in families {
        if !files.contains(&family.root()) {
            files.push(family.root());
        }
    }
    files
}

/// The family of `families` a checkpoint declares in its root files: `files` holds the text of
/// each of [`root_files`] the checkpoint publishes.
///
/// Every published file is read under its own key, and a family claims the checkpoint when the
/// file it declares in names one of its classes. Exactly one family must claim it.
///
/// # Errors
///
/// [`ResolveError::Malformed`] for a file that is not a JSON object,
/// [`ResolveError::Undeclared`] when no file names a class, [`ResolveError::Unserved`] when no
/// family claims a class the files name, and [`ResolveError::Ambiguous`] when two files name
/// classes of two families.
pub fn resolve<'f, 't, C: Claim>(
    families: &'f [C],
    files: impl IntoIterator<Item = (RootFile, &'t str)>,
) -> Result<&'f C, ResolveError> {
    let mut declared = Vec::new();
    let mut claimed: Option<(&C, String)> = None;
    for (file, text) in files {
        let malformed = |message: String| ResolveError::Malformed { file, message };
        let object = match serde_json::from_str(text) {
            Ok(Value::Object(object)) => object,
            Ok(_) => return Err(malformed("it is not an object".to_owned())),
            Err(error) => return Err(malformed(error.to_string())),
        };
        let Some(class) = file.class(&object) else {
            continue;
        };
        let stated = format!("class {class} ({})", file.name());
        let family = families
            .iter()
            .find(|family| family.root() == file && family.classes().contains(&class));
        match (family, &claimed) {
            (None, _) => declared.push(stated),
            (Some(family), None) => claimed = Some((family, stated)),
            (Some(family), Some((first, _))) if family.id() == first.id() => {}
            (Some(_), Some((_, first))) => {
                return Err(ResolveError::Ambiguous {
                    first: first.clone(),
                    second: stated,
                });
            }
        }
    }

    if let Some((family, _)) = claimed {
        return Ok(family);
    }
    if declared.is_empty() {
        return Err(ResolveError::Undeclared {
            files: root_files(families)
                .into_iter()
                .map(RootFile::name)
                .collect(),
        });
    }
    Err(ResolveError::Unserved {
        declared: declared.join(" and "),
        served: families
            .iter()
            .flat_map(|family| family.classes())
            .collect::<Vec<_>>()
            .join(", "),
    })
}

/// What the daemon binds an instance's family with: the checkpoint's files the family declared
/// and the deployment's options of request processing. The control tokens a family names (its
/// EOS, its reasoning delimiters) resolve through the vocabulary's host functions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bind {
    /// The family to bind, by its identity.
    pub family: String,
    /// The text of every file the family declared that the checkpoint publishes, by its path:
    /// a chat family's [`Serving::Chat`] files relative to the checkpoint's root; a video
    /// family's processor files as `<stage>/<file>` of its pipeline.
    pub files: std::collections::BTreeMap<String, String>,
    /// The deployment's chat options.
    pub chat: ChatOptions,
    /// The deployment's processor options (`mm_processor_kwargs`), which the family alone reads
    /// and validates.
    pub processor_options: Map<String, Value>,
    /// The deployment's `max_model_len`, when it states one.
    pub max_model_len: Option<u32>,
    /// What a video family binds its processor to, once the workers described the placed
    /// denoiser; `None` for a chat family.
    pub video: Option<video::Binding>,
}

/// The deployment's chat options (`chat_config`), as request processing reads them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    /// The template every request renders with in place of the checkpoint's own, as the
    /// deployment states it.
    pub template: Option<String>,
    /// Keyword arguments bound into every render, which a request's own replace by name.
    pub template_kwargs: Map<String, Value>,
    /// The shape message content renders in; detected from the template when unset.
    pub content_format: Option<ContentFormat>,
    /// Whether reasoning is answered apart from content (`chat_config.reasoning_parser`).
    pub reasoning: bool,
}

/// The shape a chat template takes message `content` in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentFormat {
    /// Content is one string.
    String,
    /// Content is a list of typed parts, as OpenAI states it.
    OpenAi,
    /// String content stays a string and a list stays a list.
    Preserve,
}

/// What a binding states of the bound checkpoint: the facts the daemon serves the model with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// A chat family's facts.
    Chat(ChatFacts),
    /// A video family's facts.
    Video(video::Facts),
}

/// The facts of a bound chat checkpoint.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatFacts {
    /// The longest sequence a request may reach, in tokens: the deployment's `max_model_len`,
    /// else the context length the checkpoint states where its family keeps it.
    pub max_model_len: u32,
    /// The tokenizer configuration's `eos_token`, as a vocabulary id.
    pub primary_eos: Option<u32>,
    /// Every id that ends generation: the generation defaults' `eos_token_id` and the primary.
    pub eos: Vec<u32>,
    /// The checkpoint's generation defaults.
    pub sampling: SamplingDefaults,
    /// Whether the family's tool calls are parsed from its output.
    pub tools: bool,
    /// Whether the family delimits reasoning in its output.
    pub reasoning: bool,
    /// Whether the family takes image parts.
    pub images: bool,
}

/// A checkpoint's generation defaults (`generation_config.json`); `None` where it states none.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingDefaults {
    /// Temperature; zero selects greedy sampling.
    pub temperature: Option<f32>,
    /// Nucleus probability mass.
    pub top_p: Option<f32>,
    /// Candidate limit; zero disables top-k filtering.
    pub top_k: Option<u32>,
    /// Minimum probability relative to the most likely token.
    pub min_p: Option<f32>,
    /// Multiplicative penalty for repeated tokens.
    pub repetition_penalty: Option<f32>,
    /// Generation length (`max_new_tokens`), used when a request states none.
    pub max_output_tokens: Option<u32>,
}

/// A family's request processing, bound to one checkpoint: what a processing program implements
/// and [`export_processing!`](crate::export_processing) exports.
///
/// A program fails a call by returning its error, which fails the request the call served; a
/// panic traps and ends the instance, which also fails only that request.
pub trait Processing: Sized {
    /// The families the program serves.
    fn families() -> Vec<Family>;

    /// Binds `bind.family` to the checkpoint `bind` states, and the facts the daemon serves it
    /// with.
    ///
    /// # Errors
    ///
    /// Why the checkpoint or the options are not the family's.
    fn bind(bind: Bind) -> Result<(Self, Bound), String>;

    /// The prompt of one chat request.
    ///
    /// # Errors
    ///
    /// Refuses a request the family's prompt cannot be made of.
    fn render(&mut self, render: Render) -> Result<Prompt, Refusal>;

    /// Opens the parser of one response, replacing the one the instance held.
    ///
    /// # Errors
    ///
    /// Why the request's tools cannot be parsed against.
    fn open(&mut self, open: Open) -> Result<(), String>;

    /// The events one decoded update parses into.
    ///
    /// # Errors
    ///
    /// Fails the response, naming the malformed output.
    fn push(&mut self, text: &str) -> Result<Vec<Event>, String>;

    /// The events the parser held back, at the end of generation.
    ///
    /// # Errors
    ///
    /// Fails the response when the output ended inside a structure.
    fn finish(&mut self) -> Result<Vec<Event>, String>;

    /// The open parser's state, which [`Self::restore`] takes back.
    fn save(&self) -> Vec<u8>;

    /// Restores a parser from the state [`Self::save`] returned.
    ///
    /// # Errors
    ///
    /// Why the state is not one this program saved.
    fn restore(&mut self, state: &[u8]) -> Result<(), String>;

    /// Applies the bound video family's rules that need no media.
    ///
    /// # Errors
    ///
    /// Refuses the request, naming the parameter at fault.
    fn check_video(&self, spec: &video::VideoSpec) -> Result<(), video::InputError>;

    /// Processes a video request whose references were probed.
    ///
    /// # Errors
    ///
    /// Refuses the request, naming the parameter at fault, or fails processing valid input.
    fn process_video(
        &self,
        process: video::Process,
    ) -> Result<video::VideoInput, video::InputError>;
}

/// Generates the exports of the [`Processing`] type `$program` (see [`crate::processing`]).
///
/// A processing program crate invokes it once, at its root; the exports exist only in the
/// wasm32 module, as [`export!`](crate::export)'s do.
#[macro_export]
macro_rules! export_processing {
    ($program:ty) => {
        const _: fn() = || {
            fn program<P: $crate::processing::Processing>() {}
            program::<$program>();
        };

        // The module's statement of its kind and interface version, which its host reads before
        // it links it.
        #[cfg(target_arch = "wasm32")]
        #[used]
        #[unsafe(link_section = "uniserve.program")]
        static PROGRAM_SECTION: [u8; 5] = $crate::program::section(
            $crate::program::Kind::Processing,
            $crate::processing::INTERFACE_VERSION,
        );

        #[cfg(target_arch = "wasm32")]
        const _: () = {
            // The bound family of this instance, which `bind` fills. wasm32 without threads runs
            // one export at a time and the host binds an instance before any other call of it,
            // which makes the one mutable slot sound.
            static mut BOUND: ::core::option::Option<$program> = ::core::option::Option::None;

            #[unsafe(no_mangle)]
            extern "C" fn alloc(len: u32) -> u32 {
                $crate::abi::allocate(len)
            }

            #[unsafe(no_mangle)]
            unsafe extern "C" fn free(ptr: u32, len: u32) {
                // SAFETY: the host hands back a buffer `alloc` or a reply made.
                unsafe { $crate::abi::release(ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn inbox() -> u64 {
                $crate::processing::guest::inbox()
            }

            #[unsafe(no_mangle)]
            extern "C" fn families() -> u64 {
                $crate::processing::guest::families::<$program>()
            }

            #[unsafe(no_mangle)]
            extern "C" fn bind(ptr: u32, len: u32) -> u64 {
                // SAFETY: the host's message lives for this call, and the slot is this module's.
                unsafe { $crate::processing::guest::bind::<$program>(&raw mut BOUND, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn render(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::render::<$program>(&raw mut BOUND, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn parse_open(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::open::<$program>(&raw mut BOUND, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn parse_push(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::push::<$program>(&raw mut BOUND, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn parse_finish() -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::finish::<$program>(&raw mut BOUND) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn parse_save() -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::save::<$program>(&raw mut BOUND) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn parse_restore(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::restore::<$program>(&raw mut BOUND, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn video_check(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::check::<$program>(&raw mut BOUND, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn video_process(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `bind`.
                unsafe { $crate::processing::guest::process::<$program>(&raw mut BOUND, ptr, len) }
            }
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A family declared by a `config.json` class.
    fn chat() -> Family {
        Family {
            id: "chat".to_owned(),
            root: RootFile::Config,
            classes: vec!["ChatForCausalLM".to_owned()],
            serving: Serving::Chat { files: Vec::new() },
        }
    }

    /// A family declared by a modular pipeline's class.
    fn video() -> Family {
        Family {
            id: "video".to_owned(),
            root: RootFile::ModularIndex,
            classes: vec!["VideoModularPipeline".to_owned()],
            serving: Serving::Chat { files: Vec::new() },
        }
    }

    const CONFIG: &str = r#"{"architectures": ["ChatForCausalLM"], "model_type": "chat_model"}"#;
    const INDEX: &str =
        r#"{"_class_name": "VideoModularPipeline", "transformer": ["diffusers", "X"]}"#;

    /// A checkpoint resolves by the class the file its family declares in names, whichever other
    /// files it publishes.
    #[test]
    fn a_checkpoint_resolves_by_its_familys_root_file() {
        let families = [chat(), video()];
        assert_eq!(
            root_files(&families),
            [RootFile::Config, RootFile::ModularIndex]
        );
        assert_eq!(
            resolve(&families, [(RootFile::Config, CONFIG)]).map(|family| &family.id),
            Ok(&"chat".to_owned())
        );
        assert_eq!(
            resolve(
                &families,
                [(RootFile::Config, "{}"), (RootFile::ModularIndex, INDEX)]
            )
            .map(|family| &family.id),
            Ok(&"video".to_owned())
        );
    }

    /// A class no family claims is refused naming what the checkpoint declares and everything
    /// the families serve; no class, two families' classes and a file that is no JSON object
    /// are each refused as such.
    #[test]
    fn an_unserved_undeclared_or_ambiguous_checkpoint_is_refused() {
        let families = [chat(), video()];
        let unserved = r#"{"architectures": ["OtherForCausalLM"]}"#;
        let message = resolve(&families, [(RootFile::Config, unserved)])
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("OtherForCausalLM (config.json)"),
            "{message}"
        );
        assert!(
            message.contains("ChatForCausalLM, VideoModularPipeline"),
            "{message}"
        );

        assert!(matches!(
            resolve(&families, []),
            Err(ResolveError::Undeclared { .. })
        ));
        assert!(matches!(
            resolve(
                &families,
                [(RootFile::Config, CONFIG), (RootFile::ModularIndex, INDEX)]
            ),
            Err(ResolveError::Ambiguous { .. })
        ));
        assert!(matches!(
            resolve(&families, [(RootFile::Config, "[1]")]),
            Err(ResolveError::Malformed {
                file: RootFile::Config,
                ..
            })
        ));
    }
}
