//! Chat-template sources, the Jinja environment a template compiles in, and compiled-template
//! rendering.
//!
//! A template's source is read exactly as written: [`read_jinja`] for a file of Jinja source,
//! [`read_template_json`] for a checkpoint's `chat_template.json`, whose one schema is an object
//! with a `chat_template` string. [`CompiledChatTemplate`] compiles the source once, in an
//! environment configured as Hugging Face templates expect, together with the content format it
//! renders message content in (stated, or detected from the parsed template in `format`), and
//! [`TemplateContext`] is the set of global variables a render sees.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use minijinja::Environment;
use minijinja::machinery::{Token, WhitespaceConfig, tokenize};
use minijinja::syntax::SyntaxConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

use super::format::{ChatTemplateContentFormat, detect_chat_template_content_format};
use super::{TemplateMessage, TemplateTool};
use crate::api::model::assets::HfSpecialTokens;
use crate::api::protocol::ReasoningEffort;

/// Why a chat template cannot be read, compiled or rendered.
#[derive(Debug, ThisError)]
pub enum TemplateError {
    /// MiniJinja failed to tokenize, compile or render the template.
    #[error("the chat template fails")]
    Jinja(#[from] minijinja::Error),
    /// A template file cannot be read.
    #[error("reading the chat template {}", path.display())]
    Read {
        /// The file.
        path: PathBuf,
        /// The read's failure.
        #[source]
        source: std::io::Error,
    },
    /// A `chat_template.json` is not an object with a string `chat_template`.
    #[error("{} is not an object with a string `chat_template`", path.display())]
    TemplateJson {
        /// The file.
        path: PathBuf,
        /// The parse's failure.
        #[source]
        source: serde_json::Error,
    },
    /// The template's source is empty, so it renders every request to an empty prompt.
    #[error("the chat template is empty")]
    Empty,
}

type Result<T> = std::result::Result<T, TemplateError>;

/// The whitespace handling Hugging Face chat templates are written for: Jinja environments with
/// `trim_blocks` and `lstrip_blocks` enabled, on which their output's whitespace depends. Every
/// read of a template's source (compiling it, locating its tags, detecting its content format)
/// uses it, so each reads the template the environment compiles.
const WHITESPACE: WhitespaceConfig = WhitespaceConfig {
    keep_trailing_newline: false,
    lstrip_blocks: true,
    trim_blocks: true,
};

/// The Jinja source in the file at `path`, exactly as written.
///
/// # Errors
///
/// [`TemplateError::Read`] when the file cannot be read as UTF-8.
pub(super) fn read_jinja(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|source| TemplateError::Read {
        path: path.to_owned(),
        source,
    })
}

/// The template a checkpoint's `chat_template.json` holds: the string of its `chat_template`
/// field, as the processor configuration of a multimodal checkpoint states it.
///
/// # Errors
///
/// [`TemplateError::Read`] when the file cannot be read, and [`TemplateError::TemplateJson`] when
/// it is not an object with a string `chat_template`.
pub(super) fn read_template_json(path: &Path) -> Result<String> {
    #[derive(Deserialize)]
    struct TemplateFile {
        chat_template: String,
    }

    let text = read_jinja(path)?;
    serde_json::from_str::<TemplateFile>(&text)
        .map(|file| file.chat_template)
        .map_err(|source| TemplateError::TemplateJson {
            path: path.to_owned(),
            source,
        })
}

/// `source` with Hugging Face's `{% generation %}` and `{% endgeneration %}` tags as the
/// `{% if true %}` and `{% endif %}` of a block MiniJinja compiles.
///
/// Transformers renders chat templates with an `AssistantTracker` extension whose
/// `{% generation %} … {% endgeneration %}` marks the assistant's spans for the training masks
/// `return_assistant_tokens_mask` asks for, and renders its body unchanged. MiniJinja parses a
/// closed set of statements and has no interface for an extension's statement (an unknown name
/// is a syntax error), so the tags are mapped onto a built-in block with the same body. The tags
/// are found by MiniJinja's own lexer, which tells statements from raw blocks, comments and
/// string literals, and only each tag's name is replaced: its delimiters and their
/// whitespace-control markers stay as the template states them.
///
/// # Errors
///
/// [`TemplateError::Jinja`] when the source does not tokenize.
fn generation_blocks(source: &str) -> Result<String> {
    let tokens = tokenize(source, false, SyntaxConfig {}, WHITESPACE)
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut out = String::with_capacity(source.len());
    let mut copied = 0;
    for window in tokens.windows(3) {
        let [
            (Token::BlockStart, _),
            (Token::Ident(name), span),
            (Token::BlockEnd, _),
        ] = window
        else {
            continue;
        };
        let block = match *name {
            "generation" => "if true",
            "endgeneration" => "endif",
            _ => continue,
        };
        let (start, end) = (span.start_offset as usize, span.end_offset as usize);
        out.push_str(&source[copied..start]);
        out.push_str(block);
        copied = end;
    }
    out.push_str(&source[copied..]);
    Ok(out)
}

/// Builds the environment that holds `source`, compiled as `"chat"`.
///
/// # Errors
///
/// [`TemplateError::Jinja`] when the template does not compile.
fn build_environment(source: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    env.set_trim_blocks(WHITESPACE.trim_blocks);
    env.set_lstrip_blocks(WHITESPACE.lstrip_blocks);
    env.set_keep_trailing_newline(WHITESPACE.keep_trailing_newline);
    super::python::install(&mut env);
    env.add_template_owned("chat".to_owned(), source)?;
    Ok(env)
}

#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
/// Values installed into the Jinja environment for one render.
///
/// Serializes as one flat map in field order: the flattened request variables, special tokens
/// and template kwargs become top-level variables, `None` fields are omitted (so templates see
/// them as undefined), and on a key collision the later entry wins.
pub(super) struct TemplateContext<'a> {
    #[serde(flatten)]
    pub(super) request: RequestVariables<'a>,
    #[serde(flatten)]
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    #[serde(flatten)]
    pub(super) template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
    // Declared after `template_kwargs` so that a request's `reasoning_effort`
    // overrides a default kwarg of the same name; when the request sets none,
    // the field is omitted and the kwarg stays visible.
    pub(super) reasoning_effort: Option<ReasoningEffort>,
}

/// The variables a render takes from the request itself, which no template argument may replace.
#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
pub(super) struct RequestVariables<'a> {
    pub(super) messages: &'a [TemplateMessage],
    pub(super) add_generation_prompt: bool,
    pub(super) continue_final_message: bool,
    pub(super) tools: Option<&'a [TemplateTool]>,
    pub(super) documents: Option<&'a [serde_json::Value]>,
}

impl RequestVariables<'_> {
    /// The names of the variables, as the template sees them: each field's serialized name.
    ///
    /// They are read from a value in which every variable is present, so a field added to this
    /// type is a reserved name without a list to extend.
    #[expect(
        clippy::expect_used,
        reason = "a struct of slices and booleans serializes as a JSON object"
    )]
    pub(super) fn names() -> Vec<String> {
        let every = RequestVariables {
            messages: &[],
            add_generation_prompt: false,
            continue_final_message: false,
            tools: Some(&[]),
            documents: Some(&[]),
        };
        let variables: serde_json::Map<String, serde_json::Value> = serde_json::to_value(every)
            .and_then(serde_json::from_value)
            .expect("the request variables serialize as an object");
        variables.into_iter().map(|(name, _)| name).collect()
    }
}

/// One compiled chat template with its Jinja environment and resolved content format.
pub(super) struct CompiledChatTemplate {
    /// Fully configured environment holding the template under the name `"chat"`.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
}

impl CompiledChatTemplate {
    /// Compiles `source` in a fully configured environment.
    ///
    /// The content format is `content_format` when one is stated; otherwise it is detected from
    /// the template the environment compiles, its generation tags mapped
    /// ([`generation_blocks`]).
    ///
    /// # Errors
    ///
    /// [`TemplateError::Empty`] for an empty source, and [`TemplateError::Jinja`] when the
    /// template does not tokenize, parse or compile.
    pub(super) fn new(
        source: String,
        content_format: Option<ChatTemplateContentFormat>,
    ) -> Result<Self> {
        if source.is_empty() {
            return Err(TemplateError::Empty);
        }
        let source = generation_blocks(&source)?;
        let content_format = match content_format {
            Some(format) => format,
            None => detect_chat_template_content_format(&source, WHITESPACE)?,
        };
        let env = build_environment(source)?;
        Ok(Self {
            env,
            content_format,
        })
    }

    /// Applies the compiled template to the given context and returns the rendered prompt.
    pub(super) fn apply(&self, ctx: TemplateContext<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(ctx).map_err(TemplateError::from)
    }

    /// Returns the resolved message-content shape expected by the template.
    pub(super) fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn render(source: &str) -> String {
        CompiledChatTemplate::new(source.to_owned(), None)
            .unwrap()
            .apply(TemplateContext::default())
            .unwrap()
    }

    /// A generation block renders its body with the whitespace control its tags state, and a
    /// tag's name in a string, a comment or a raw block, or a longer identifier, is left alone.
    #[test]
    fn a_generation_block_renders_its_body_with_its_whitespace_control() {
        assert_eq!(
            render("a\n{%- generation %}b{% endgeneration -%}\n c"),
            "abc"
        );
        assert_eq!(
            render("{{ 'generation' }}{# {% generation %} #}{% raw %}{% generation %}{% endraw %}"),
            "generation{% generation %}"
        );
        assert_eq!(render("{% set generations = 1 %}{{ generations }}"), "1");
    }

    /// The content format of a template whose generation tags enclose a loop over a message's
    /// content parts is detected from the template the environment compiles.
    #[test]
    fn a_generation_template_s_content_format_is_detected_from_its_compiled_form() {
        let template = CompiledChatTemplate::new(
            "{% for message in messages %}{% generation %}{% for part in message.content %}\
             {{ part.text }}{% endfor %}{% endgeneration %}{% endfor %}"
                .to_owned(),
            None,
        )
        .unwrap();
        assert_eq!(template.content_format(), ChatTemplateContentFormat::OpenAi);
    }

    #[test]
    fn a_template_renders_its_context() {
        assert_eq!(render("{{ messages }}"), "[]");
    }

    /// A template that does not parse, or an empty one, is refused when it is compiled.
    #[test]
    fn an_invalid_or_empty_template_is_refused() {
        let error = CompiledChatTemplate::new("{% invalid".to_owned(), None)
            .err()
            .unwrap();
        assert!(matches!(error, TemplateError::Jinja(_)), "{error}");
        let error = CompiledChatTemplate::new(String::new(), None)
            .err()
            .unwrap();
        assert!(matches!(error, TemplateError::Empty), "{error}");
    }

    #[test]
    fn special_tokens_are_undefined_when_not_provided() {
        assert_eq!(
            render("{% if bos_token is defined %}{{ bos_token }}{% endif %}hello"),
            "hello"
        );
    }

    /// A Jinja file is read exactly as written, its escapes and surrounding whitespace kept.
    #[test]
    fn a_jinja_file_reads_verbatim() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat.jinja");
        fs::write(&path, "  {{ messages }}\\n\n").unwrap();

        assert_eq!(read_jinja(&path).unwrap(), "  {{ messages }}\\n\n");
    }

    /// A `chat_template.json` holds its template in a `chat_template` field; any other shape is
    /// refused.
    #[test]
    fn a_template_json_holds_its_template_in_one_field() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, r#"{"chat_template":"{{ messages }}"}"#).unwrap();
        assert_eq!(read_template_json(&path).unwrap(), "{{ messages }}");

        fs::write(&path, "\"{{ messages }}\"").unwrap();
        assert!(matches!(
            read_template_json(&path),
            Err(TemplateError::TemplateJson { .. })
        ));
    }
}
