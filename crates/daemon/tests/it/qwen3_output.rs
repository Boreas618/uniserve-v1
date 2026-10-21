//! Qwen3 reasoning and tool-call output behavior across stream boundaries.

use std::sync::Arc;

use daemon::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use daemon::serving::chat::output::AssistantEvent;
use daemon::serving::chat::{
    AssistantBlockKind, ChatOutputProcessor, ChatRequest, ChatToolChoice, Tool,
};
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

/// Builds a tokenizer whose only special tokens are the reasoning delimiters
/// that `Qwen3ReasoningParser` requires and the `<|im_end|>` turn delimiter.
fn qwen_tokenizer() -> Result<DynTokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let model = BPE::builder()
        .vocab_and_merges([("<unk>".to_string(), 0), ("a".to_string(), 1)], Vec::new())
        .unk_token("<unk>".to_string())
        .build()?;
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
        AddedToken::from("<|im_end|>", true),
    ]);
    let definition = serde_json::from_str(&tokenizer.to_string(false)?)?;
    Ok(Arc::new(HuggingFaceTokenizer::from_json(definition)?))
}

/// A chat request that declares the `lookup` tool with `Auto` tool choice,
/// which enables tool-call parsing.
fn request_with_lookup() -> ChatRequest {
    let mut request = ChatRequest::for_test();
    request.tools = vec![Tool {
        name: "lookup".to_string(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
        strict: None,
    }];
    request.tool_choice = ChatToolChoice::Auto;
    request
}

/// Runs `processor` over `prompt` and `deltas`, then finishes it.
fn parse(
    mut processor: ChatOutputProcessor,
    prompt: &[u32],
    deltas: &[&str],
) -> Result<Vec<AssistantEvent>, Box<dyn std::error::Error + Send + Sync>> {
    processor.initialize(prompt);
    let mut events = Vec::new();
    for delta in deltas {
        events.extend(processor.push((*delta).to_string())?);
    }
    events.extend(processor.finish()?);
    Ok(events)
}

/// Assistant output in stream order, with adjacent text deltas merged and
/// each call's argument deltas joined, so the comparison does not depend on
/// how deltas are split.
#[derive(Debug, PartialEq, Eq)]
enum OrderedOutput {
    Text(AssistantBlockKind, String),
    Call { name: String, arguments: String },
}

/// Folds assistant events into their ordered output.
fn ordered(events: Vec<AssistantEvent>) -> Vec<OrderedOutput> {
    let mut ordered = Vec::<OrderedOutput>::new();
    for event in events {
        match event {
            AssistantEvent::TextDelta { kind, delta } => match ordered.last_mut() {
                Some(OrderedOutput::Text(last, text)) if *last == kind => text.push_str(&delta),
                _ => ordered.push(OrderedOutput::Text(kind, delta)),
            },
            AssistantEvent::ToolCallStart { name, .. } => ordered.push(OrderedOutput::Call {
                name,
                arguments: String::new(),
            }),
            AssistantEvent::ToolCallArgumentsDelta { delta } => {
                let Some(OrderedOutput::Call { arguments, .. }) = ordered.last_mut() else {
                    panic!("arguments delta {delta:?} does not follow its tool call");
                };
                arguments.push_str(&delta);
            }
        }
    }
    ordered
}

fn text(value: &str) -> OrderedOutput {
    OrderedOutput::Text(AssistantBlockKind::Text, value.to_string())
}

fn lookup(q: &str) -> OrderedOutput {
    OrderedOutput::Call {
        name: "lookup".to_string(),
        arguments: format!(r#"{{"q":"{q}"}}"#),
    }
}

/// A single decoded delta containing a reasoning block, text, and two tool
/// calls separates into reasoning, visible text with the tool-call markup
/// removed, and one call per `<tool_call>` block in order.
#[test]
fn qwen3_processor_emits_reasoning_text_and_tool_calls() -> TestResult {
    let tokenizer = qwen_tokenizer()?;
    let prompt_end = tokenizer
        .token_to_id("<|im_end|>")
        .ok_or_else(|| std::io::Error::other("configured prompt-end token is unavailable"))?;
    // The final argument enables reasoning parsing.
    let processor = ChatOutputProcessor::qwen3(&request_with_lookup(), tokenizer, true)?;

    // The prompt ends with `<|im_end|>` and has no reasoning delimiter after
    // it, so generation starts outside a reasoning section and the `<think>`
    // in the output opens one.
    let events = parse(
        processor,
        &[prompt_end],
        &[concat!(
            "<think>hidden</think>before",
            "<tool_call>\n",
            r#"{"name":"lookup","arguments":{"q":"x"}}"#,
            "\n</tool_call>",
            "between",
            "<tool_call>\n",
            r#"{"name":"lookup","arguments":{"q":"y"}}"#,
            "\n</tool_call>",
            "after"
        )],
    )?;

    assert_eq!(
        ordered(events),
        vec![
            OrderedOutput::Text(AssistantBlockKind::Reasoning, "hidden".to_string()),
            text("before"),
            lookup("x"),
            text("between"),
            lookup("y"),
            text("after"),
        ]
    );
    Ok(())
}

/// Text and tool calls keep their stream order when one decoded delta spans
/// call boundaries, both when the delta starts outside a call and when it
/// starts inside a call's arguments.
#[test]
fn qwen3_processor_keeps_text_and_tool_call_order_within_a_delta() -> TestResult {
    let processor = ChatOutputProcessor::qwen3(&request_with_lookup(), qwen_tokenizer()?, false)?;

    // The first delta starts outside any call and ends inside the second
    // call's arguments; the second delta starts inside those arguments.
    let events = parse(
        processor,
        &[],
        &[
            concat!(
                "before",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"x"}}"#,
                "\n</tool_call>",
                "between",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"#,
            ),
            concat!(
                r#""y"}}"#,
                "\n</tool_call>",
                "after",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"z"}}"#,
                "\n</tool_call>",
            ),
        ],
    )?;

    assert_eq!(
        ordered(events),
        vec![
            text("before"),
            lookup("x"),
            text("between"),
            lookup("y"),
            text("after"),
            lookup("z"),
        ]
    );
    Ok(())
}

/// With reasoning parsing disabled, `<think>` delimiters stream verbatim as
/// assistant content instead of opening a reasoning block, including when the
/// delimiters arrive in separate deltas.
#[test]
fn disabled_reasoning_parsing_streams_delimiters_as_content() -> TestResult {
    let processor = ChatOutputProcessor::qwen3(&ChatRequest::for_test(), qwen_tokenizer()?, false)?;

    let events = parse(processor, &[], &["<think>", "hi</think>after"])?;

    assert_eq!(ordered(events), vec![text("<think>hi</think>after")]);
    Ok(())
}
