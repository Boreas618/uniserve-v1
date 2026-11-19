//! Fixtures the crate's tests share: chat messages and requests as a client states them
//! ([`chat`]), tool parsers driven over chunked text ([`tool_parsing`]), and a vocabulary of
//! single-character tokens beside a few control tokens ([`Characters`]).

use crate::vocabulary::{Vocabulary, VocabularyError};

/// Messages and requests of the processing interface, each as a client sends it.
pub(crate) mod chat {
    use scheduling::processing::{
        Content, ContentPart, Function, FunctionCall, Message, Render, Tool, ToolCall,
    };

    /// A request of `messages`, with no tools, template arguments or images.
    pub(crate) fn request(messages: Vec<Message>) -> Render {
        Render {
            messages,
            ..Render::default()
        }
    }

    /// The function tool `name`, taking arguments of the JSON schema `parameters`.
    pub(crate) fn function(name: &str, parameters: serde_json::Value) -> Tool {
        Tool::Function {
            function: Function {
                name: name.to_owned(),
                description: None,
                parameters,
                strict: None,
            },
        }
    }

    /// A system message of `text`.
    pub(crate) fn system(text: impl Into<String>) -> Message {
        Message::System {
            content: Content::Text(text.into()),
        }
    }

    /// A user message of `text`.
    pub(crate) fn user(text: impl Into<String>) -> Message {
        Message::User {
            content: Content::Text(text.into()),
        }
    }

    /// A user message of content `parts`.
    pub(crate) fn user_parts(parts: Vec<ContentPart>) -> Message {
        Message::User {
            content: Content::Parts(parts),
        }
    }

    /// A text content part.
    pub(crate) fn text_part(text: impl Into<String>) -> ContentPart {
        ContentPart::Text { text: text.into() }
    }

    /// An image content part; the fixture's `_url` stands for the client's reference, which the
    /// daemon fetched.
    pub(crate) fn image_part(_url: &str) -> ContentPart {
        ContentPart::Image
    }

    /// A developer message of `text`, declaring `tools`.
    pub(crate) fn developer(text: impl Into<String>, tools: Option<Vec<Tool>>) -> Message {
        Message::Developer {
            content: Content::Text(text.into()),
            tools,
        }
    }

    /// An assistant message of `text`.
    pub(crate) fn assistant(text: impl Into<String>) -> Message {
        assistant_turn("", text, Vec::new())
    }

    /// The assistant message a client sends back for a turn that reasoned `reasoning`, said
    /// `text` and called `tool_calls`; an empty part is absent from the message.
    pub(crate) fn assistant_turn(
        reasoning: impl Into<String>,
        text: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Message {
        let (reasoning, text) = (reasoning.into(), text.into());
        Message::Assistant {
            content: (!text.is_empty()).then_some(Content::Text(text)),
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls,
        }
    }

    /// The function call `id` of `name` with the JSON text `arguments`.
    pub(crate) fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            function: FunctionCall {
                name: name.to_owned(),
                arguments: Some(arguments.to_owned()),
            },
        }
    }

    /// A tool's response of `content` to the call `tool_call_id`.
    pub(crate) fn tool_response(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> Message {
        Message::Tool {
            content: Content::Text(content.into()),
            tool_call_id: tool_call_id.into(),
        }
    }
}

/// A vocabulary whose tokens are single characters, by their code point, beside the control
/// tokens it names: what the bindings' tests resolve delimiters and EOS tokens with.
pub(crate) struct Characters {
    /// The control tokens and their ids, which no character's id equals.
    pub(crate) controls: Vec<(&'static str, u32)>,
}

impl Characters {
    /// The ChatML and thinking control tokens of the serving tests: `<|im_start|>` 1,
    /// `<|im_end|>` 2, `<|vision_start|>` 3, `<|vision_end|>` 4, `<think>` 5 and `</think>` 6.
    pub(crate) fn chat() -> Self {
        Self {
            controls: vec![
                ("<|im_start|>", 1),
                ("<|im_end|>", 2),
                ("<|vision_start|>", 3),
                ("<|vision_end|>", 4),
                ("<think>", 5),
                ("</think>", 6),
            ],
        }
    }
}

impl Vocabulary for Characters {
    fn token_id(&self, token: &str) -> Option<u32> {
        self.controls
            .iter()
            .find(|(text, _)| *text == token)
            .map(|(_, id)| *id)
            .or_else(|| {
                let mut chars = token.chars();
                let first = chars.next()?;
                chars.next().is_none().then_some(u32::from(first))
            })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, VocabularyError> {
        Ok(text.chars().map(u32::from).collect())
    }
}

/// Drives tool parsers over text cut into chunks, as decoded text reaches them.
pub(crate) mod tool_parsing {
    use crate::output::Result;
    use crate::output::tools::{ToolParser, ToolParserOutput};

    /// The output of `chunks` pushed through `parser` in order and the parser finished, with
    /// each call's deltas merged.
    pub(crate) fn parse_chunks<'a>(
        parser: &mut dyn ToolParser,
        chunks: impl IntoIterator<Item = &'a str>,
    ) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output)?;
        }
        output.append(parser.finish()?);
        Ok(output.coalesce_calls())
    }

    /// The output of the whole of `text` pushed through `parser` as one chunk.
    pub(crate) fn parse_whole(parser: &mut dyn ToolParser, text: &str) -> Result<ToolParserOutput> {
        parse_chunks(parser, [text])
    }

    /// `text` cut into chunks of `size` characters, the last one shorter.
    pub(crate) fn chunks_of(text: &str, size: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chars
            .chunks(size)
            .map(|chunk| chunk.iter().collect())
            .collect()
    }
}
