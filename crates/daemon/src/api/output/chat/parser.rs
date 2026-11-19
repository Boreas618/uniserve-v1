//! Assistant output processing for chat generation.
//!
//! [`ChatParser`] parses one request's decoded text through the request's instance of the
//! epoch's processing program, which holds the parsers of the formats its family writes: the
//! program splits reasoning from visible text and extracts tool calls from the visible text, and
//! states each update as events (`scheduling::processing::Event`). The parser frames them as the
//! [`AssistantEvent`]s the request's `OutputProcessor` assembles into content blocks and tool
//! calls (`structured`), naming each call as it opens.

use std::sync::Arc;

use proto::handoff;
use scheduling::processing::Event;
use uuid::Uuid;

use crate::api::error::{Error, Result};
use crate::api::output::chat::AssistantBlockKind;
use crate::api::program::{Binding, Session};

/// Parsed assistant output of one decoded update.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantEvent {
    /// Parsed text of kind `Text` or `Reasoning`.
    TextDelta {
        /// Whether the text is visible or reasoning.
        kind: AssistantBlockKind,
        /// The text.
        delta: String,
    },
    /// Opens a tool call; subsequent argument deltas belong to it.
    ToolCallStart {
        /// Tool-call identifier from `generate_tool_call_id`.
        id: String,
        /// Function name selected by the model.
        name: String,
    },
    /// Argument text for the most recently started tool call.
    ToolCallArgumentsDelta {
        /// Next chunk of the JSON arguments text.
        delta: String,
    },
}

/// The parser of one chat request's decoded text: the request's instance of the processing
/// program, whose parser `Processor::process` opened for the request's prompt.
pub struct ChatParser {
    session: Session,
}

impl ChatParser {
    /// The parser of the request whose instance is `session`.
    pub fn new(session: Session) -> Self {
        Self { session }
    }

    /// The parser a handed-over response's `state` states, restored in an instance of
    /// `binding`, the family the successor binds the request's checkpoint to.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Lowering`] when the state names another program than `binding`'s or
    /// states none, and with [`Error::Output`] when the program does not restore it.
    pub fn resume(binding: &Arc<Binding>, state: &handoff::Processor) -> Result<Self> {
        if state.program != binding.program().digest() {
            return Err(Error::Lowering {
                part: "parser state of this deployment's processing program",
            });
        }
        let mut session = binding.checkout()?;
        session.restore(&state.state)?.map_err(Error::output)?;
        Ok(Self { session })
    }

    /// The parser's state, for the process that goes on writing the response after a handoff:
    /// the program's opaque bytes and the program that saved them.
    ///
    /// # Errors
    ///
    /// Fails when the program's instance ended.
    pub fn state(&mut self) -> Result<handoff::Processor> {
        Ok(handoff::Processor {
            state: self.session.save()?,
            program: self.session.binding().program().digest().to_owned(),
        })
    }

    /// Parses one decoded text delta into assistant events; its reasoning text precedes its
    /// visible text.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Output`] for output the family's parsers refuse, such as a malformed
    /// tool call, and with [`Error::Processing`] when the program's instance ended.
    pub fn push(&mut self, delta: &str) -> Result<Vec<AssistantEvent>> {
        let events = self.session.push(delta)?.map_err(Error::output)?;
        Ok(events.into_iter().map(assistant_event).collect())
    }

    /// Releases the text the parsers hold back at the end of generation.
    ///
    /// # Errors
    ///
    /// As [`Self::push`], and [`Error::Output`] when the text ended inside a tool call.
    pub fn finish(&mut self) -> Result<Vec<AssistantEvent>> {
        let events = self.session.finish()?.map_err(Error::output)?;
        Ok(events.into_iter().map(assistant_event).collect())
    }
}

/// The assistant event of one parsed event; a call that opens is given its identifier.
fn assistant_event(event: Event) -> AssistantEvent {
    match event {
        Event::Reasoning(delta) => AssistantEvent::TextDelta {
            kind: AssistantBlockKind::Reasoning,
            delta,
        },
        Event::Content(delta) => AssistantEvent::TextDelta {
            kind: AssistantBlockKind::Text,
            delta,
        },
        Event::ToolCall { name } => AssistantEvent::ToolCallStart {
            id: generate_tool_call_id(),
            name,
        },
        Event::Arguments(delta) => AssistantEvent::ToolCallArgumentsDelta { delta },
    }
}

/// Generates an OpenAI-compatible tool-call identifier: `call_` followed by
/// 24 hex digits of a random UUID.
fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
