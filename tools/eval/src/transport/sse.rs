//! Server-sent events and the chat-completion values they and complete
//! messages carry.
//!
//! Reasoning text (`reasoning_content`, else `reasoning`) counts as generated
//! text beside visible content, ahead of it.

use serde_json::{Map, Value};

/// One decoded event of a stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The `[DONE]` marker.
    Done,
    /// A JSON payload.
    Json(Value),
    /// A payload that is not JSON, with why.
    ParseError(String),
}

/// Assembles a stream's `data:` lines into events: a blank line dispatches the
/// lines buffered since the last one, other lines complete nothing.
#[derive(Debug, Default)]
pub struct Parser {
    lines: Vec<String>,
}

impl Parser {
    /// Consumes one line without its terminator; returns the event a blank line
    /// completes.
    pub fn feed(&mut self, line: &str) -> Option<Event> {
        if line.is_empty() {
            return self.flush();
        }
        let data = line.strip_prefix("data:")?;
        let data = data.strip_prefix(' ').unwrap_or(data);
        self.lines.push(data.to_owned());
        None
    }

    /// The final event whose dispatching blank line never arrived.
    pub fn finish(&mut self) -> Option<Event> {
        self.flush()
    }

    fn flush(&mut self) -> Option<Event> {
        if self.lines.is_empty() {
            return None;
        }
        let data = self.lines.join("\n");
        self.lines.clear();
        if data == "[DONE]" {
            return Some(Event::Done);
        }
        Some(match serde_json::from_str::<Value>(&data) {
            Ok(value) => Event::Json(value),
            Err(error) => {
                Event::ParseError(format!("invalid SSE JSON payload: {data:?} ({error})"))
            }
        })
    }
}

/// Splits a body's bytes into lines as they arrive: at each line feed, a
/// preceding carriage return removed, of any length (an event carrying an
/// encoded image exceeds any fixed buffer). Splitting at a line feed never cuts
/// a UTF-8 sequence.
#[derive(Debug, Default)]
pub struct Lines {
    buffer: Vec<u8>,
}

impl Lines {
    /// The complete lines `chunk` ends.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=newline).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            lines.push(String::from_utf8_lossy(&line).into_owned());
        }
        lines
    }

    /// The final unterminated line.
    pub fn finish(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let mut line = std::mem::take(&mut self.buffer);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Some(String::from_utf8_lossy(&line).into_owned())
    }
}

/// How a stream ended, gathered one event at a time as they arrive (no event is
/// kept): how many arrived, whether one was not a JSON object, the first error
/// an event carried, whether the stream terminated, whether any carried output.
#[derive(Debug, Default)]
pub struct Outcome {
    events: usize,
    invalid: bool,
    pub error: Option<String>,
    terminal: bool,
    output: bool,
}

impl Outcome {
    pub fn observe(&mut self, event: &Event) {
        self.events += 1;
        let payload = match event {
            Event::Done => {
                self.terminal = true;
                return;
            }
            Event::ParseError(_) => {
                self.invalid = true;
                return;
            }
            Event::Json(Value::Object(payload)) => payload,
            Event::Json(_) => {
                self.invalid = true;
                return;
            }
        };
        if self.error.is_none()
            && let Some(Value::Object(error)) = payload.get("error")
        {
            self.error = Some(error_text(error));
        }
        if !self.terminal {
            self.terminal = has_finish_reason(payload);
        }
        if !self.output {
            self.output = !delta_text(payload).is_empty() || !delta_images(payload).is_empty();
        }
    }

    /// The stream's classification; the first failing condition, in this order,
    /// wins.
    pub fn classify(&self) -> (bool, &'static str) {
        if self.events == 0 {
            (false, "response_empty_response")
        } else if self.invalid {
            (false, "response_invalid_event")
        } else if self.error.is_some() {
            (false, "model_error")
        } else if !self.terminal {
            (false, "response_missing_terminal")
        } else if !self.output {
            (false, "response_empty_output")
        } else {
            (true, "ok")
        }
    }
}

/// An OpenAI-style error object as `type: message`, or the whole object when it
/// states no message.
fn error_text(error: &Map<String, Value>) -> String {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty());
    let kind = error
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| !kind.is_empty());
    match (message, kind) {
        (Some(message), Some(kind)) => format!("{kind}: {message}"),
        (Some(message), None) => message.to_owned(),
        (None, _) => {
            let sorted: std::collections::BTreeMap<&String, &Value> = error.iter().collect();
            serde_json::to_string(&sorted).unwrap_or_default()
        }
    }
}

fn reasoning(message: &Map<String, Value>) -> &str {
    message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| message.get("reasoning").and_then(Value::as_str))
        .unwrap_or_default()
}

fn is_image_part(part: &Map<String, Value>) -> bool {
    part.get("type").and_then(Value::as_str) == Some("image_url")
        || part.contains_key("image_url")
        || part.contains_key("b64_json")
}

fn choices(event: &Map<String, Value>) -> impl Iterator<Item = &Map<String, Value>> {
    event
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
}

fn has_finish_reason(event: &Map<String, Value>) -> bool {
    choices(event).any(|choice| {
        choice
            .get("finish_reason")
            .is_some_and(|reason| !reason.is_null())
    })
}

/// The text of every choice of an event: its delta's reasoning, then its string
/// content, then a completion-style `text`.
pub fn delta_text(event: &Map<String, Value>) -> String {
    let mut text = String::new();
    for choice in choices(event) {
        if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
            text.push_str(reasoning(delta));
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                text.push_str(content);
            }
        }
        if let Some(completion) = choice.get("text").and_then(Value::as_str) {
            text.push_str(completion);
        }
    }
    text
}

/// The image parts of every choice's delta: its `images`, then the image parts
/// of a list-valued content.
pub fn delta_images(event: &Map<String, Value>) -> Vec<Map<String, Value>> {
    let mut images = Vec::new();
    for choice in choices(event) {
        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            continue;
        };
        if let Some(Value::Array(parts)) = delta.get("images") {
            images.extend(parts.iter().filter_map(Value::as_object).cloned());
        }
        if let Some(Value::Array(parts)) = delta.get("content") {
            images.extend(
                parts
                    .iter()
                    .filter_map(Value::as_object)
                    .filter(|part| is_image_part(part))
                    .cloned(),
            );
        }
    }
    images
}

/// A complete message's reasoning, then its content: a string, or the `text`
/// parts of a list.
pub fn message_text(message: Option<&Map<String, Value>>) -> String {
    let Some(message) = message else {
        return String::new();
    };
    let mut text = reasoning(message).to_owned();
    match message.get("content") {
        Some(Value::String(content)) => text.push_str(content),
        Some(Value::Array(parts)) => {
            for part in parts.iter().filter_map(Value::as_object) {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
                }
            }
        }
        _ => {}
    }
    text
}

/// A complete message's images: its `images`, then the image parts of a
/// list-valued content.
pub fn message_images(message: Option<&Map<String, Value>>) -> Vec<Map<String, Value>> {
    let Some(message) = message else {
        return Vec::new();
    };
    let mut images = Vec::new();
    if let Some(Value::Array(parts)) = message.get("images") {
        images.extend(parts.iter().filter_map(Value::as_object).cloned());
    }
    if let Some(Value::Array(parts)) = message.get("content") {
        images.extend(
            parts
                .iter()
                .filter_map(Value::as_object)
                .filter(|part| is_image_part(part))
                .cloned(),
        );
    }
    images
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Data lines join across a blank line's dispatch; `[DONE]` and bad JSON
    /// are events of their own, and a stream's last event needs no blank line.
    #[test]
    fn events_dispatch_at_blank_lines() {
        let mut parser = Parser::default();
        assert_eq!(parser.feed("data: {\"a\":"), None);
        assert_eq!(parser.feed("data: 1}"), None);
        assert_eq!(parser.feed(": comment"), None);
        assert_eq!(
            parser.feed(""),
            Some(Event::Json(serde_json::json!({"a": 1})))
        );
        assert_eq!(parser.feed("data: [DONE]"), None);
        assert_eq!(parser.feed(""), Some(Event::Done));
        parser.feed("data: {oops");
        assert!(matches!(parser.finish(), Some(Event::ParseError(_))));

        let mut lines = Lines::default();
        assert_eq!(lines.push(b"data: x\r\nda"), ["data: x"]);
        assert_eq!(lines.push(b"ta: y\n"), ["data: y"]);
        assert_eq!(lines.push(b"tail"), Vec::<String>::new());
        assert_eq!(lines.finish().as_deref(), Some("tail"));
    }
}
