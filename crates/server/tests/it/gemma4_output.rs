//! Gemma-4 reasoning and tool-call output behavior on realistic token streams.
//!
//! `fixtures/gemma4_chat_output.json` holds DiffusionGemma replies as prompt
//! token ids and generated `(token id, text)` pairs, where the text is what
//! incremental decoding with special tokens kept yields for that token under
//! the checkpoint tokenizer. The `reference_generation_*` samples are
//! Transformers 5.14.1 generations of diffusiongemma-26B-A4B-it with thinking
//! enabled, cut before their end-of-turn token. The other samples render
//! assistant turns through the checkpoints' chat templates: tool calls after
//! reasoning, the empty thought channel that opens a reply without thinking,
//! and a thinking continuation after a tool response under each checkpoint's
//! template. The expected reasoning, content, and tool calls come from
//! Transformers' Gemma-4 response parser, except for the continuation whose
//! thought channel the BF16 template opens in the prompt, where they are the
//! rendered parts themselves.
//!
//! `fixtures/gemma4_tool_replies.json` holds every distinct reply that
//! Transformers 5.14.1 generated for one tool prompt (the weather in Paris
//! with a declared `get_weather(city)` tool, thinking off) on the BF16 and
//! NVFP4 checkpoints over seeds 0..63, with the seeds that produced each.
//! They include the malformed shapes the model writes: a call without its
//! opening token, a thought channel whose name lacks its newline or that
//! never closes, and a call without `call:`. The expected values are SGLang's
//! non-streaming chat parse (`--reasoning-parser gemma4 --tool-call-parser
//! gemma4`), except that the `<|channel>` token and channel name SGLang keeps
//! in the reasoning of a channel whose name lacks its newline are stripped
//! (the reply's `sglang` entry then records SGLang's own parse). A reply that
//! follows the chat template's assistant grammar also records Transformers'
//! `parse_response` of it (`hf`).

use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;
use server::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use server::serving::chat::output::AssistantEvent;
use server::serving::chat::{
    AssistantBlockKind, ChatOutputProcessor, ChatRequest, ChatToolChoice, Tool,
};
use server::serving::text::TextDecodeOptions;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

/// DiffusionGemma commits generated tokens in blocks of this many tokens.
const COMMITTED_BLOCK_TOKENS: usize = 256;

/// Special tokens of the Gemma-4 checkpoint tokenizer with their ids, in
/// ascending id order.
const SPECIAL_TOKENS: [(u32, &str); 17] = [
    (0, "<pad>"),
    (1, "<eos>"),
    (2, "<bos>"),
    (3, "<unk>"),
    (4, "<mask>"),
    (46, "<|tool>"),
    (47, "<tool|>"),
    (48, "<|tool_call>"),
    (49, "<tool_call|>"),
    (50, "<|tool_response>"),
    (51, "<tool_response|>"),
    (52, "<|\"|>"),
    (98, "<|think|>"),
    (100, "<|channel>"),
    (101, "<channel|>"),
    (105, "<|turn>"),
    (106, "<turn|>"),
];

#[derive(Debug, Deserialize)]
struct Fixture {
    samples: Vec<Sample>,
}

#[derive(Debug, Deserialize)]
struct Sample {
    name: String,
    prompt_token_ids: Vec<u32>,
    /// Generated tokens with the text each one adds to the decoded output.
    output: Vec<(u32, String)>,
    reasoning: String,
    content: String,
    tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ToolCall {
    name: String,
    arguments: Value,
}

/// Assistant output in stream order, with adjacent text of one kind merged
/// and each call's argument deltas parsed together as JSON.
#[derive(Debug, PartialEq)]
enum Block {
    Reasoning(String),
    Text(String),
    Call(ToolCall),
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/gemma4_chat_output.json"))
        .expect("fixture is valid JSON")
}

#[derive(Debug, Deserialize)]
struct ToolReplyFixture {
    prompt_token_ids: Vec<u32>,
    replies: Vec<ToolReply>,
}

#[derive(Debug, Deserialize)]
struct ToolReply {
    name: String,
    output: Vec<(u32, String)>,
    reasoning: String,
    content: String,
    tool_calls: Vec<ToolCall>,
    /// Transformers' parse, present for a reply in the template's grammar.
    hf: Option<Parse>,
}

#[derive(Debug, Deserialize)]
struct Parse {
    reasoning: String,
    content: String,
    tool_calls: Vec<ToolCall>,
}

/// The replies of `fixtures/gemma4_tool_replies.json`, each as a `Sample`
/// expecting the reference serving parse, with Transformers' parse when the
/// reply follows the template's grammar.
fn tool_replies() -> Vec<(Sample, Option<Sample>)> {
    let fixture: ToolReplyFixture =
        serde_json::from_str(include_str!("../fixtures/gemma4_tool_replies.json"))
            .expect("fixture is valid JSON");
    fixture
        .replies
        .into_iter()
        .map(|reply| {
            let sample = |name: &str, expected: Parse| Sample {
                name: name.to_string(),
                prompt_token_ids: fixture.prompt_token_ids.clone(),
                output: reply.output.clone(),
                reasoning: expected.reasoning,
                content: expected.content,
                tool_calls: expected.tool_calls,
            };
            let hf = reply
                .hf
                .map(|parse| sample(&format!("{} (Transformers)", reply.name), parse));
            let serving = Parse {
                reasoning: reply.reasoning,
                content: reply.content,
                tool_calls: reply.tool_calls,
            };
            (sample(&reply.name, serving), hf)
        })
        .collect()
}

/// Builds a tokenizer that holds the checkpoint's special tokens at their
/// checkpoint ids, which is all the processor reads from its tokenizer. The
/// remaining ids up to the last special token hold ordinary placeholder
/// tokens, and ids beyond the vocabulary count as ordinary tokens.
fn gemma4_tokenizer() -> DynTokenizer {
    let (last_special_id, _) = SPECIAL_TOKENS[SPECIAL_TOKENS.len() - 1];
    let mut vocab: Vocab = (0..=last_special_id)
        .map(|id| (format!("<placeholder_{id}>"), id))
        .collect();
    for (id, token) in SPECIAL_TOKENS {
        vocab.retain(|_, existing_id| *existing_id != id);
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("build tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    let added: Vec<AddedToken> = SPECIAL_TOKENS
        .into_iter()
        .map(|(_, token)| AddedToken::from(token, true))
        .collect();
    tokenizer.add_special_tokens(&added);

    let definition = serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
    Arc::new(HuggingFaceTokenizer::from_json(definition).expect("load tokenizer"))
}

/// A request that declares the fixture's tools and lets the model call them.
fn request_with_tools() -> ChatRequest {
    let tool = |name: &str| Tool {
        name: name.to_string(),
        description: None,
        parameters: serde_json::json!({"type": "object"}),
        strict: None,
    };
    let mut request = ChatRequest::for_test();
    request.tools = vec![tool("get_weather"), tool("plan_trip")];
    request.tool_choice = ChatToolChoice::Auto;
    request
}

/// The decoded text of `sample`'s generated tokens, in deltas of at most
/// `tokens_per_delta` tokens.
fn deltas(sample: &Sample, tokens_per_delta: usize) -> Vec<String> {
    sample
        .output
        .chunks(tokens_per_delta)
        .map(|chunk| chunk.iter().map(|(_, text)| text.as_str()).collect())
        .collect()
}

/// Runs `processor` over `sample`'s prompt and `deltas`, then finishes it,
/// and returns the assistant blocks.
fn assistant_blocks(
    mut processor: ChatOutputProcessor,
    sample: &Sample,
    deltas: Vec<String>,
) -> Vec<Block> {
    processor.initialize(&sample.prompt_token_ids);
    let mut events = Vec::new();
    for delta in deltas {
        events.extend(processor.push(delta).expect("processing succeeds"));
    }
    events.extend(processor.finish().expect("processing succeeds"));

    let mut blocks = Vec::new();
    let mut open_arguments: Option<(String, String)> = None;
    for event in events {
        if !matches!(event, AssistantEvent::ToolCallArgumentsDelta { .. })
            && let Some((name, arguments)) = open_arguments.take()
        {
            blocks.push(Block::Call(ToolCall {
                name,
                arguments: serde_json::from_str(&arguments).expect("arguments are JSON"),
            }));
        }

        match event {
            AssistantEvent::TextDelta { kind, delta } => match (kind, blocks.last_mut()) {
                (AssistantBlockKind::Reasoning, Some(Block::Reasoning(text)))
                | (AssistantBlockKind::Text, Some(Block::Text(text))) => text.push_str(&delta),
                (AssistantBlockKind::Reasoning, _) => blocks.push(Block::Reasoning(delta)),
                (AssistantBlockKind::Text, _) => blocks.push(Block::Text(delta)),
            },
            AssistantEvent::ToolCallStart { name, .. } => {
                open_arguments = Some((name, String::new()));
            }
            AssistantEvent::ToolCallArgumentsDelta { delta } => {
                open_arguments
                    .as_mut()
                    .expect("arguments follow a started call")
                    .1
                    .push_str(&delta);
            }
        }
    }
    if let Some((name, arguments)) = open_arguments {
        blocks.push(Block::Call(ToolCall {
            name,
            arguments: serde_json::from_str(&arguments).expect("arguments are JSON"),
        }));
    }
    blocks
}

/// The blocks a sample's expected parse describes, in the Gemma-4 reply
/// order: thought channel, tool calls, then visible text.
fn expected_blocks(sample: &Sample) -> Vec<Block> {
    let mut blocks = Vec::new();
    if !sample.reasoning.is_empty() {
        blocks.push(Block::Reasoning(sample.reasoning.clone()));
    }
    blocks.extend(sample.tool_calls.iter().cloned().map(Block::Call));
    if !sample.content.is_empty() {
        blocks.push(Block::Text(sample.content.clone()));
    }
    blocks
}

/// Every fixture reply separates into its reasoning, tool calls, and
/// visible text, both when tokens arrive one at a time and when each decoded
/// delta carries a whole committed block.
#[test]
fn gemma4_processor_parses_replies_token_by_token_and_by_committed_block() {
    let tokenizer = gemma4_tokenizer();

    for sample in fixture().samples {
        for tokens_per_delta in [1, COMMITTED_BLOCK_TOKENS] {
            let processor = ChatOutputProcessor::gemma4(
                &request_with_tools(),
                &mut TextDecodeOptions::default(),
                Arc::clone(&tokenizer),
                true,
            )
            .expect("build processor");

            let blocks = assistant_blocks(processor, &sample, deltas(&sample, tokens_per_delta));

            assert_eq!(
                blocks,
                expected_blocks(&sample),
                "{} with {tokens_per_delta} tokens per delta",
                sample.name
            );
        }
    }
}

/// Every reply the model wrote to a tool prompt, malformed ones included,
/// parses as the reference serving system parses it, and as Transformers
/// does when the reply follows the template's grammar, token by token and by
/// committed block. No channel markup reaches the reasoning or the content.
#[test]
fn gemma4_processor_parses_real_tool_replies_as_the_reference_does() {
    let tokenizer = gemma4_tokenizer();

    for (serving, hf) in tool_replies() {
        for sample in std::iter::once(&serving).chain(hf.as_ref()) {
            for tokens_per_delta in [1, COMMITTED_BLOCK_TOKENS] {
                let request = request_with_tools();
                let processor = ChatOutputProcessor::gemma4(
                    &request,
                    &mut TextDecodeOptions::default(),
                    Arc::clone(&tokenizer),
                    true,
                )
                .expect("build processor");

                let blocks = assistant_blocks(processor, sample, deltas(sample, tokens_per_delta));

                assert_eq!(
                    blocks,
                    expected_blocks(sample),
                    "{} with {tokens_per_delta} tokens per delta",
                    sample.name
                );
            }
        }
    }
}

/// The processor decodes with special tokens kept, so without reasoning or
/// tool parsing the channel and tool-call delimiters stream verbatim as
/// visible text.
#[test]
fn gemma4_processor_without_parsers_streams_delimiters_as_text() {
    let sample = fixture()
        .samples
        .into_iter()
        .find(|sample| sample.name == "tool_calls_after_reasoning")
        .expect("fixture has a tool-call sample");
    let mut decode = TextDecodeOptions::default();
    let processor = ChatOutputProcessor::gemma4(
        &ChatRequest::for_test(),
        &mut decode,
        gemma4_tokenizer(),
        false,
    )
    .expect("build processor");

    let blocks = assistant_blocks(processor, &sample, deltas(&sample, 1));

    assert!(!decode.skip_special_tokens);
    let raw: String = sample
        .output
        .iter()
        .map(|(_, text)| text.as_str())
        .collect();
    assert!(raw.starts_with("<|channel>thought\n") && raw.contains("<|tool_call>call:"));
    assert_eq!(blocks, [Block::Text(raw)]);
}
