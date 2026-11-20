//! The encoder against the reference encoder's cases and vLLM's renders of requests
//! (`vectors.json`).

use scheduling::processing::Render;
use serde_json::{Map, Value};

use super::encoding::{self, Block, Call, Message, Mode, Options, Task};
use super::{Encoder, vectors};
use crate::chat::ChatEncoder;
use crate::testing::chat;

/// The text of a reference message's `content`: a string, or the text of its text blocks.
fn text_of(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block["type"] == "text")
            .map(|block| block["text"].as_str().unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n\n"),
        other => panic!("content of no known shape: {other}"),
    }
}

/// The task a reference message asks for.
fn task_of(message: &Value) -> Option<Task> {
    Some(match message.get("task")?.as_str()? {
        "action" => Task::Action,
        "query" => Task::Query,
        "authority" => Task::Authority,
        "domain" => Task::Domain,
        "title" => Task::Title,
        "read_url" => Task::ReadUrl,
        other => panic!("no task {other}"),
    })
}

/// The schemas of the tools a reference message declares: each tool's function object.
fn tools_of(message: &Value) -> Vec<Value> {
    message
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| tools.iter().map(|tool| tool["function"].clone()).collect())
        .unwrap_or_default()
}

/// A message in the reference encoder's format (`encoding.py`'s message dictionaries).
fn reference_message(message: &Value) -> Message {
    let content = || text_of(&message["content"]);
    match message["role"].as_str().expect("a role") {
        "system" => Message::System {
            content: content(),
            tools: tools_of(message),
        },
        "user" => Message::User {
            content: match &message["content"] {
                Value::Array(blocks) => blocks
                    .iter()
                    .map(|block| match block["type"].as_str() {
                        Some("text") => Block::Text(block["text"].as_str().unwrap().to_owned()),
                        Some("image_url") => Block::Image,
                        other => panic!("no block {other:?}"),
                    })
                    .collect(),
                other => vec![Block::Text(text_of(other))],
            },
            task: task_of(message),
        },
        "latest_reminder" => Message::LatestReminder { content: content() },
        "assistant" => Message::Assistant {
            reasoning: text_of(&message["reasoning_content"]),
            content: content(),
            tool_calls: message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|call| Call {
                    id: call["id"].as_str().unwrap_or_default().to_owned(),
                    name: call["function"]["name"].as_str().unwrap().to_owned(),
                    arguments: call["function"]["arguments"].as_str().unwrap().to_owned(),
                })
                .collect(),
            task: task_of(message),
        },
        "tool" => Message::Tool {
            tool_call_id: message["tool_call_id"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            content: content(),
        },
        other => panic!("no role {other}"),
    }
}

/// The options a reference case states.
fn reference_options(case: &Value) -> Options {
    let effort = match &case["reasoning_effort"] {
        Value::Null => 75,
        Value::String(name) => match name.as_str() {
            "low" => 50,
            "high" => 75,
            "max" => 100,
            other => panic!("no effort {other}"),
        },
        Value::Number(budget) => u8::try_from(budget.as_u64().unwrap()).unwrap(),
        other => panic!("no effort {other}"),
    };
    Options {
        mode: match case["thinking_mode"].as_str() {
            Some("thinking") => Mode::Thinking,
            _ => Mode::Chat,
        },
        drop_thinking: case["drop_thinking"].as_bool().unwrap(),
        effort,
    }
}

/// Every reference case, the checkpoint's own `encoding/tests` among them, encodes to the text
/// the reference encoder writes, byte for byte, and what the reference refuses is refused.
#[test]
fn conversations_encode_as_the_reference_encodes_them() {
    let vectors = vectors();
    let cases = vectors["reference"].as_array().unwrap();
    assert_eq!(
        cases
            .iter()
            .filter(|case| case["name"]
                .as_str()
                .unwrap()
                .starts_with("encoding/tests/"))
            .count(),
        5,
        "the checkpoint's five cases"
    );
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let messages = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(reference_message)
            .collect();
        let encoded = encoding::encode(messages, &reference_options(case));
        match case.get("prompt").and_then(Value::as_str) {
            Some(prompt) => assert_eq!(encoded.as_deref(), Ok(prompt), "{name}"),
            None => assert!(encoded.is_err(), "{name}: {encoded:?}"),
        }
    }
}

/// Every request encodes to the text vLLM's `deepseek_v41` tokenizer mode renders for it, over
/// the deployment's template arguments, and what vLLM refuses is refused.
#[test]
fn requests_encode_as_vllm_renders_them() {
    let vectors = vectors();
    for case in vectors["requests"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let render: Render = serde_json::from_value(case["render"].clone()).unwrap();
        let defaults: Map<String, Value> =
            serde_json::from_value(case["defaults"].clone()).unwrap();
        let encoded = Encoder.encode(&render, &defaults);
        match case.get("prompt").and_then(Value::as_str) {
            Some(prompt) => assert_eq!(encoded.as_deref(), Ok(prompt), "{name}"),
            None => assert!(encoded.is_err(), "{name}: {encoded:?}"),
        }
    }
}

/// An image part is refused by its type, since the family serves text, and so is a reasoning
/// effort the format does not take, naming the field that stated it.
#[test]
fn image_parts_and_unknown_efforts_are_refused_by_name() {
    let images = chat::request(vec![chat::user_parts(vec![
        chat::text_part("see"),
        chat::image_part("https://example.com/a.png"),
    ])]);
    let refusal = Encoder.encode(&images, &Map::new()).unwrap_err();
    assert_eq!(refusal.param.as_deref(), Some("messages"));
    assert!(refusal.message.contains("image_url"), "{refusal:?}");

    let effort = Render {
        reasoning_effort: Some("medium".to_owned()),
        ..chat::request(vec![chat::user("q")])
    };
    let refusal = Encoder.encode(&effort, &Map::new()).unwrap_err();
    assert_eq!(refusal.param.as_deref(), Some("reasoning_effort"));
    assert!(refusal.message.contains("\"medium\""), "{refusal:?}");
}
