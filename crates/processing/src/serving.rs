//! What a model family serves, as request processing serves it: chat in a family's output formats,
//! or video through the processor a video family binds. A family states it in its
//! request-processing description ([`crate::Family::serving`]).

use crate::image::ImageFamily;
use crate::video::VideoFamily;

/// What a family serves.
#[derive(Clone, Copy)]
pub enum Serving {
    /// Chat and completions over a causal language model.
    Chat(ChatProfile),
    /// Video generation over a diffusion pipeline.
    Video(&'static dyn VideoFamily),
}

/// The output formats a chat family's checkpoints write, and the image input it takes.
///
/// A format is stated for what the model writes, never for a model: request processing holds one
/// parser per tool-call grammar and one reasoning parser, which reads the delimiters a family
/// states, and several families may write one format. What a checkpoint's template instructs is
/// the family's to state here; request processing does not read a template to find out.
#[derive(Clone, Copy)]
pub struct ChatProfile {
    /// How the model delimits its reasoning, when it reasons.
    pub reasoning: Option<ReasoningFormat>,
    /// How the model writes tool calls, when it calls tools.
    pub tool_calls: Option<ToolCallFormat>,
    /// How the family's checkpoints take images, when they do; a family without it serves text
    /// alone and refuses an image part.
    pub images: Option<&'static dyn ImageFamily>,
    /// What the family's checkpoints state beside their tokenizer that request processing reads.
    pub checkpoint: ChatCheckpoint,
}

/// What a chat family's checkpoints state at their root beside their tokenizer (`tokenizer.json`
/// and `tokenizer_config.json`), as request processing reads it: each fact where the family's
/// checkpoints keep it, so nothing is found by trying the places other families use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatCheckpoint {
    /// The field of the configuration ([`crate::RootFile::Config`]) that states the longest
    /// sequence the model's positions cover, as the keys from the document's root to it:
    /// `["max_position_embeddings"]`, or `["text_config", "max_position_embeddings"]` for a
    /// checkpoint whose language model is one section of its configuration.
    pub context_length: &'static [&'static str],
    /// Where the checkpoints keep their chat template.
    pub chat_template: ChatTemplate,
}

/// Where a chat family's checkpoints keep their chat template.
///
/// `transformers` stored a tokenizer's template as the `chat_template` field of
/// `tokenizer_config.json` until it saved templates as files of Jinja source of their own
/// (`chat_template.jinja`), and it loads a checkpoint's template from that file when the
/// checkpoint publishes it, else from the field. A family states the places its published
/// checkpoints keep it in, which request processing reads by that precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    /// A file of Jinja source at the checkpoint's root, such as `chat_template.jinja`, which
    /// every checkpoint of the family publishes: the family's checkpoints were all saved since
    /// `transformers` saves templates as files.
    File(&'static str),
    /// The file of Jinja source at the checkpoint's root when the checkpoint publishes it, else
    /// the `chat_template` field of `tokenizer_config.json`: the family's checkpoints were saved
    /// both before and since, so one keeps its template in the field and another, such as a
    /// later quantized export, in the file alone.
    FileOrTokenizerConfig(&'static str),
}

impl std::fmt::Debug for ChatProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatProfile")
            .field("reasoning", &self.reasoning)
            .field("tool_calls", &self.tool_calls)
            .field("images", &self.images.is_some())
            .field("checkpoint", &self.checkpoint)
            .finish()
    }
}

impl PartialEq for ChatProfile {
    /// Equal formats and checkpoint layouts, and image input of the same family or of none.
    fn eq(&self, other: &Self) -> bool {
        let images = match (self.images, other.images) {
            (Some(left), Some(right)) => std::ptr::addr_eq(left, right),
            (left, right) => left.is_none() && right.is_none(),
        };
        self.reasoning == other.reasoning
            && self.tool_calls == other.tool_calls
            && self.checkpoint == other.checkpoint
            && images
    }
}

impl Eq for ChatProfile {}

/// A reasoning output format: the reasoning a model writes between two delimiters, each one token
/// of the vocabulary.
///
/// The delimiters are the family's data, and request processing reads every family's reasoning
/// with one parser parameterized by them (`crate::output::reasoning`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReasoningFormat {
    /// The text that opens a reasoning section.
    pub open: &'static str,
    /// The text that closes it.
    pub close: &'static str,
}

/// A tool-call output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// One JSON object per call, naming the function and its arguments, between `<tool_call>`
    /// and `</tool_call>`:
    ///
    /// ```text
    /// <tool_call>
    /// {"name": NAME, "arguments": {ARG: VALUE}}
    /// </tool_call>
    /// ```
    Json,
    /// Function and parameter tags between `<tool_call>` and `</tool_call>`, each value as
    /// text that the tool's JSON schema types:
    ///
    /// ```text
    /// <tool_call>
    /// <function=NAME>
    /// <parameter=ARG>
    /// VALUE
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    Xml,
}

impl ToolCallFormat {
    /// The format's name, which a malformed call's failure states it by.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Xml => "xml",
        }
    }

    /// The text that opens one tool-call region and the text that closes it.
    ///
    /// The JSON format's newlines belong to its delimiters, as its families' chat templates
    /// instruct them, so a bare `<tool_call>` mentioned in prose stays text. The tag format
    /// writes its tags on lines of their own, and whitespace between its tags carries no
    /// meaning, so its delimiters are the tags alone.
    pub const fn delimiters(self) -> (&'static str, &'static str) {
        match self {
            Self::Json => ("<tool_call>\n", "\n</tool_call>"),
            Self::Xml => ("<tool_call>", "</tool_call>"),
        }
    }
}
