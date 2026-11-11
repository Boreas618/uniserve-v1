//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use frontend::input::chat::HfChatRenderer;
use frontend::input::media::ImageInput;
use frontend::input::{ModelResolutionError, Processor, RequestRenderer};
use frontend::model::ModelConfig;
use frontend::model::assets::ResolvedModelFiles;
use frontend::openai::ApiError;
use frontend::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use processing::{Catalogue, Family};
use processing::{ChatProfile, ReasoningFormat, Serving, ToolCallFormat};
use proto::config::ChatTemplateContentFormat;
use proto::handoff::Lowering;
use proto::request::{GenerationLimits, GenerationRequest};
use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

/// The model name every test request names and the routes check.
const SERVED_MODEL: &str = "served-model";

/// Minimal ChatML template: one `<|im_start|>role\ncontent<|im_end|>` turn per
/// message, then an open assistant turn when a generation prompt is requested.
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
/// Special tokens added to the synthetic tokenizer: ChatML turn delimiters,
/// image-marker tokens, and reasoning and answer delimiters.
const SPECIAL_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<img>",
    "</img>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
];

/// A chat family that declares neither a reasoning nor a tool-call format.
const PLAIN: Family = Family {
    id: "plain",
    model_types: &["plain"],
    classes: &[],
    serving: Serving::Chat(ChatProfile {
        reasoning: None,
        tool_calls: None,
        images: None,
    }),
};

/// A chat family of Qwen3's `model_type`s whose outputs are parsed in its formats.
const CHAT: Family = Family {
    id: "chat",
    model_types: &["qwen3", "qwen3_moe"],
    classes: &[],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Think),
        tool_calls: Some(ToolCallFormat::Json),
        images: None,
    }),
};

/// Resolves a chat checkpoint of `model_type` over a synthetic checkpoint with
/// worker limits that cover every generation feature; see
/// `try_resolved_model`.
fn resolved_model(model_type: &str) -> (tempfile::TempDir, DynTokenizer, Processor) {
    try_resolved_model(model_type, runtime_limits()).unwrap()
}

/// Writes a synthetic checkpoint to a temporary directory and binds an
/// `Processor` for it with the given worker limits.
///
/// The tokenizer maps each ASCII code point from 1 to 127 to one token whose
/// ID is the code point (no merges), plus `SPECIAL_TOKENS`, so outside the
/// special tokens one character is one prompt token. `generation_config.json`
/// sets a 128-token default generation length. The returned directory owns
/// the checkpoint files.
///
/// # Errors
///
/// Returns the `Processor::new` error, for example when `limits` do not
/// cover a feature the profile needs. Other setup failures panic.
fn try_resolved_model(
    model_type: &str,
    limits: GenerationLimits,
) -> Result<(tempfile::TempDir, DynTokenizer, Processor), ModelResolutionError> {
    let directory = tempdir().unwrap();

    let mut vocab = Vocab::from_iter([("<unk>".to_string(), 0_u32)]);
    for codepoint in 1_u32..=127 {
        vocab.insert(char::from_u32(codepoint).unwrap().to_string(), codepoint);
    }
    let tokenizer_model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer_builder = TokenizerBuilder::new(tokenizer_model);
    tokenizer_builder.add_special_tokens(
        &SPECIAL_TOKENS
            .iter()
            .map(|token| AddedToken::from(*token, true))
            .collect::<Vec<_>>(),
    );
    let tokenizer_path = directory.path().join("tokenizer.json");
    tokenizer_builder.save(&tokenizer_path, false).unwrap();

    let config_path = directory.path().join("config.json");
    fs::write(
        &config_path,
        format!(
            r#"{{"model_type":"{model_type}","max_position_embeddings":4096,"num_attention_heads":8}}"#
        ),
    )
    .unwrap();
    let tokenizer_config_path = directory.path().join("tokenizer_config.json");
    fs::write(
        &tokenizer_config_path,
        format!(
            "{{\"eos_token\":\"<|im_end|>\",\"chat_template\":{}}}",
            serde_json::to_string(CHAT_TEMPLATE).unwrap()
        ),
    )
    .unwrap();
    let generation_config_path = directory.path().join("generation_config.json");
    fs::write(
        &generation_config_path,
        r#"{"eos_token_id":2,"max_new_tokens":128}"#,
    )
    .unwrap();
    let files = ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path: Some(tokenizer_config_path),
        generation_config_path: Some(generation_config_path),
        chat_template_path: None,
        config_path: Some(config_path),
    };
    let tokenizer: DynTokenizer = Arc::new(
        HuggingFaceTokenizer::load(
            &files.tokenizer_path,
            files.tokenizer_config_path.as_deref(),
        )
        .unwrap(),
    );
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap();
    let config = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(ModelConfig::from_files(
            &Catalogue::new([CHAT, PLAIN, processing::qwen::QWEN4_EXP]).unwrap(),
            &files,
            Some(4096),
            tokenizer.as_ref(),
        ))
        .unwrap();
    let renderer = RequestRenderer::Chat(renderer);
    let model = Processor::new(config, Arc::clone(&tokenizer), renderer, limits, true)?;
    Ok((directory, tokenizer, model))
}

/// Worker limits that cover every generation feature with capacities no test
/// request reaches.
fn runtime_limits() -> GenerationLimits {
    GenerationLimits {
        features: proto::request::GenerationFeatures::all(),
        max_vae_grid_tokens: 1_000_000,
        max_vit_grid_tokens: 1_000_000,
        max_latent_feature_bytes: u64::MAX,
        max_vision_feature_bytes: u64::MAX,
        encoder_cache_entries: 32,
        max_logprobs: 20,
    }
}

/// Lowers a chat completion as the deployment's `Processor` does for its
/// route, answering a refusal as the route does.
fn preprocess_chat(
    processor: &Processor,
    request_id: String,
    request: frontend::protocol::ChatCompletionRequest,
    images: Vec<ImageInput>,
) -> Result<(GenerationRequest, Lowering), ApiError> {
    processor
        .process(&request_id, &request, None, images)
        .map_err(ApiError::from)
}

/// A single-turn text chat request to the served model, with the wire
/// `fields` set on top.
fn chat_request(fields: serde_json::Value) -> frontend::protocol::ChatCompletionRequest {
    let mut request = serde_json::json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["text"],
    });
    request
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    serde_json::from_value(request).unwrap()
}

/// `Processor::new` refuses to bind a profile whose worker limits lack a
/// feature the profile's configured branches need, and names the missing
/// feature by its `GenerationFeatures::name` diagnostic.
#[test]
fn model_resolution_refuses_a_missing_runtime_branch() {
    let mut limits = runtime_limits();
    limits
        .features
        .remove(proto::request::GenerationFeatures::UNDERSTANDING);
    let error = match try_resolved_model("qwen3", limits) {
        Ok(_) => panic!("incomplete worker limits must fail model resolution"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("runtime_und_execution"),
        "got: {error}"
    );
}

#[test]
fn context_length_limits_preprocessed_requests() {
    let (_directory, tokenizer, loaded) = resolved_model("qwen3");
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap();
    let mut config = loaded.config().clone();
    config.max_model_len = 8;
    let processor = Processor::new(
        config,
        tokenizer,
        RequestRenderer::Chat(renderer),
        runtime_limits(),
        true,
    )
    .unwrap();

    let error = preprocess_chat(
        &processor,
        String::from("context-capacity"),
        chat_request(serde_json::json!({"messages": [{
            "role": "user",
            "content": "This prompt exceeds eight tokens"
        }]})),
        Vec::new(),
    )
    .err()
    .unwrap();
    assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    let message = error.to_error_response().error.message;
    assert!(message.contains("8"), "{message}");
}

/// Defaults fill only omitted controls: an explicit zero temperature stays
/// zero rather than taking the default 1.0, while an explicit zero completion
/// length is rejected rather than replaced by the 128-token checkpoint default.
#[test]
fn sampling_defaults_preserve_explicit_zero_controls() {
    let (_directory, _tokenizer, model) = resolved_model("qwen3");
    let mut request: frontend::protocol::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": SERVED_MODEL, "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    let (default, _) = preprocess_chat(
        &model,
        String::from("sampling-defaults"),
        request.clone(),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(default.max_und_tokens, 128);
    assert_eq!(default.sampling.temperature, 1.0);

    request.temperature = Some(0.0);
    let (greedy, _) = preprocess_chat(
        &model,
        String::from("sampling-defaults"),
        request.clone(),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(greedy.sampling.temperature, 0.0);
    assert_eq!(greedy.max_und_tokens, 128);

    request.max_completion_tokens = Some(0);
    assert!(matches!(
        preprocess_chat(
            &model,
            String::from("sampling-defaults"),
            request,
            Vec::new()
        ),
        Err(frontend::openai::ApiError::InvalidRequest { .. })
    ));
}

/// The cache salt selects a stable isolation partition, and prompt logprobs
/// disable prefix-cache reads only, because a prefix-cache hit would skip the
/// prompt positions whose logprobs are requested.
#[test]
fn cache_salt_isolates_and_prompt_logprobs_disable_cache_reads() {
    let (_directory, _tokenizer, processor) = resolved_model("qwen3");
    let preprocess = |fields: serde_json::Value| {
        preprocess_chat(
            &processor,
            String::from("cache-controls"),
            chat_request(fields),
            Vec::new(),
        )
        .unwrap()
    };

    let (shared, _) = preprocess(serde_json::json!({}));
    assert_eq!(shared.cache.isolation_key, None);
    assert!(shared.cache.read && shared.cache.write);

    // The same salt always produces the same key, and another salt another.
    let (isolated, _) = preprocess(serde_json::json!({"cache_salt": "ab"}));
    assert!(isolated.cache.isolation_key.is_some());
    assert!(isolated.cache.read && isolated.cache.write);
    let (repeat, _) = preprocess(serde_json::json!({"cache_salt": "ab"}));
    assert_eq!(repeat.cache.isolation_key, isolated.cache.isolation_key);
    let (other, _) = preprocess(serde_json::json!({"cache_salt": "ba"}));
    assert_ne!(other.cache.isolation_key, isolated.cache.isolation_key);

    let (logprobs, response) = preprocess(serde_json::json!({"prompt_logprobs": 0}));
    assert!(!logprobs.cache.read && logprobs.cache.write);
    assert!(response.prompt_logprobs);
}

/// A streamed chat response has no field for prompt logprobs. A streamed
/// request with `prompt_logprobs: 0`, which the route accepts like vLLM does,
/// therefore requests no prompt scoring and keeps prefix-cache reads, while
/// the same buffered request still requests them.
#[test]
fn streamed_chat_does_not_request_undeliverable_prompt_logprobs() {
    let (_directory, _tokenizer, processor) = resolved_model("qwen3");
    for stream in [true, false] {
        let request: frontend::protocol::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "modalities": ["text"],
                "stream": stream,
                "prompt_logprobs": 0
            }))
            .unwrap();

        let (generation, response) = preprocess_chat(
            &processor,
            String::from("stream-prompt-logprobs"),
            request,
            Vec::new(),
        )
        .unwrap();

        assert_eq!(response.prompt_logprobs, !stream, "stream={stream}");
        assert_eq!(
            generation.sampling.prompt_logprobs_requested(),
            !stream,
            "stream={stream}"
        );
        assert_eq!(generation.cache.read, stream, "stream={stream}");
    }
}

/// Qwen3 chat template shipped by Qwen3 and Qwen3-MoE checkpoints; it asks for
/// JSON tool calls inside `<tool_call>` lines.
const QWEN3_TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/templates/qwen3.jinja"
));
/// A template whose tool instruction shows function and parameter tags
/// instead of JSON objects.
const XML_CALL_TEMPLATE: &str = "{%- if tools -%}<|im_start|>system\n# Tools\n<tools>{%- for tool in tools -%}{{ tool | tojson }}{%- endfor -%}</tools>\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter>\nvalue\n</parameter>\n</function>\n</tool_call><|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

/// Rebinds a resolved Qwen3-family model to `template`.
fn with_template(model_type: &str, template: &str) -> Processor {
    let (_directory, tokenizer, loaded) = resolved_model(model_type);
    let renderer = HfChatRenderer::new(
        Some(template.to_string()),
        HashMap::new(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap();
    Processor::new(
        loaded.config().clone(),
        tokenizer,
        RequestRenderer::Chat(renderer),
        runtime_limits(),
        true,
    )
    .unwrap()
}

/// A chat request that asks for image output is refused by name: no served
/// profile generates images, so asking for one alone — or beside text — is a
/// refusal, not a text answer.
#[test]
fn a_chat_request_for_image_output_is_refused_by_name() {
    let (_directory, _tokenizer, processor) = resolved_model("qwen3");
    for modalities in [vec!["image"], vec!["text", "image"]] {
        let request: frontend::protocol::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "draw a bird"}],
                "modalities": modalities,
            }))
            .unwrap();
        let error = preprocess_chat(
            &processor,
            String::from("image-output"),
            request,
            Vec::new(),
        )
        .err()
        .unwrap_or_else(|| panic!("{modalities:?} was admitted"));
        let message = error.to_error_response().error.message;
        assert!(
            message.contains("image_output"),
            "{modalities:?}: {message}"
        );
    }
}

/// A family's chat profile declares its tool-call format, and that alone
/// decides whether requests with tools are served: whatever the checkpoint's
/// template text instructs, a family that declares a format has its calls
/// parsed in it, and the lowering states the format and the request's tools
/// to output processing.
#[test]
fn the_profile_declares_the_tool_call_format() {
    let tool_request = serde_json::json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {}}
        }}]
    });
    for model_type in ["qwen3", "qwen3_moe"] {
        for template in [QWEN3_TEMPLATE, XML_CALL_TEMPLATE, CHAT_TEMPLATE] {
            let processor = with_template(model_type, template);
            assert!(processor.supports_tool_calling(), "{model_type}");
            let (_, lowering) = preprocess_chat(
                &processor,
                String::from("tools"),
                serde_json::from_value(tool_request.clone()).unwrap(),
                Vec::new(),
            )
            .unwrap();

            let parser = lowering.parser.unwrap();
            assert_eq!(
                parser.kind,
                proto::handoff::parser::Kind::ParserKindJson as i32
            );
            assert!(parser.parse_tools && parser.parse_reasoning);
            let tools: serde_json::Value = serde_json::from_str(&parser.tools).unwrap();
            assert_eq!(tools[0]["function"]["name"], "weather");
        }
    }

    // `tool_choice: none` withholds the tools from the model, so no call is parsed.
    let mut withheld = tool_request.clone();
    withheld["tool_choice"] = serde_json::json!("none");
    let (_, lowering) = preprocess_chat(
        &with_template("qwen3", QWEN3_TEMPLATE),
        String::from("no-tools"),
        serde_json::from_value(withheld).unwrap(),
        Vec::new(),
    )
    .unwrap();
    let parser = lowering.parser.unwrap();
    assert!(!parser.parse_tools);
    assert!(parser.tools.is_empty());
}

/// A family whose profile declares no tool-call format refuses a request
/// with tools by name, rather than returning its calls as unparsed text, and
/// one that declares function and parameter tags has them stated to output
/// processing.
#[test]
fn tool_calling_follows_the_familys_declared_format() {
    let tool_request = serde_json::json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {}}
        }}]
    });

    let (_directory, _, plain) = resolved_model("plain");
    assert!(!plain.supports_tool_calling());
    let error = preprocess_chat(
        &plain,
        String::from("plain-tools"),
        serde_json::from_value(tool_request.clone()).unwrap(),
        Vec::new(),
    )
    .err()
    .unwrap();
    assert_eq!(error.status_code().as_u16(), 400);
    let message = error.to_error_response().error.message;
    assert!(message.contains("tool_calling"), "{message}");

    let (_directory, _, tagged) = resolved_model("qwen4_exp");
    assert!(tagged.supports_tool_calling());
    let (_, lowering) = preprocess_chat(
        &tagged,
        String::from("xml-tools"),
        serde_json::from_value(tool_request).unwrap(),
        Vec::new(),
    )
    .unwrap();
    let parser = lowering.parser.unwrap();
    assert_eq!(
        parser.kind,
        proto::handoff::parser::Kind::ParserKindXml as i32
    );
    assert!(parser.parse_tools);
}

/// What an agent harness sends turn after turn is lowered as it sends it: tool
/// definitions with OpenAI's `strict`, `tool_choice: auto`, `max_tokens`, an
/// assistant turn sent back as it was received (`reasoning_content`, null
/// content, tool calls) and the tools' results. Each turn's prompt begins with
/// the prompt of the turn before it, under the same cache salt and so in the
/// same cache partition, which is what lets the prefix cache serve it.
#[test]
fn a_harness_conversation_grows_by_its_new_turns() {
    let processor = with_template("qwen3", QWEN3_TEMPLATE);
    let turn = |messages: serde_json::Value| {
        let request = serde_json::json!({
            "model": SERVED_MODEL,
            "messages": messages,
            "tools": [{"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }}],
            "tool_choice": "auto",
            "max_tokens": 64,
            "cache_salt": "harness-session",
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        preprocess_chat(
            &processor,
            String::from("harness"),
            serde_json::from_value(request).unwrap(),
            Vec::new(),
        )
        .unwrap()
    };

    let first = serde_json::json!([
        {"role": "system", "content": "You improve the server."},
        {"role": "user", "content": "Read the scheduler."},
    ]);
    let mut second = first.clone();
    second.as_array_mut().unwrap().extend([
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "reasoning_content": "I need the file first.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\": \"a.rs\"}"},
            }],
        }),
        serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"}),
    ]);

    let (first_turn, first_lowering) = turn(first);
    let (second_turn, _) = turn(second);

    assert_eq!(first_turn.max_und_tokens, 64);
    assert!(first_lowering.parser.unwrap().parse_tools);
    assert!(first_lowering.response.unwrap().include_usage);
    assert!(first_turn.cache.isolation_key.is_some());
    assert_eq!(
        first_turn.cache.isolation_key,
        second_turn.cache.isolation_key
    );
    assert!(second_turn.cache.read && second_turn.cache.write);
    assert!(
        second_turn.prompt_token_ids.len() > first_turn.prompt_token_ids.len()
            && second_turn
                .prompt_token_ids
                .starts_with(&first_turn.prompt_token_ids),
        "the second turn's prompt does not begin with the first turn's"
    );
}

/// A `tool_choice` that makes the model call a tool, `required` or one named
/// function, is refused by name: vLLM serves both through guided decoding,
/// which a deployment here does not have.
#[test]
fn a_tool_choice_that_needs_constrained_decoding_is_refused() {
    let processor = with_template("qwen3", QWEN3_TEMPLATE);
    for choice in [
        serde_json::json!("required"),
        serde_json::json!({"type": "function", "function": {"name": "weather"}}),
    ] {
        let request = serde_json::json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {}}
            }}],
            "tool_choice": choice,
        });
        let error = preprocess_chat(
            &processor,
            String::from("forced"),
            serde_json::from_value(request).unwrap(),
            Vec::new(),
        )
        .err()
        .unwrap_or_else(|| panic!("tool_choice {choice} was admitted"));

        assert_eq!(error.status_code().as_u16(), 400, "{choice}");
        let response = error.to_error_response().error;
        assert_eq!(response.param.as_deref(), Some("tool_choice"), "{choice}");
        assert!(
            response.message.contains("constrained decoding"),
            "{}",
            response.message
        );
    }
}

/// Negative seeds are accepted and reinterpreted as their two's-complement
/// `u64` value.
#[test]
fn negative_seeds_are_reinterpreted() {
    let (_directory, _, model) = resolved_model("qwen3");
    for seed in [-1, -2] {
        let (request, _) = preprocess_chat(
            &model,
            String::from("sampling"),
            chat_request(serde_json::json!({"seed": seed})),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(request.sampling.seed, Some(seed as u64));
    }
}

/// A request may ask for as many candidates at a position as the deployment's
/// `max_logprobs` (20 here), and no more, as vLLM rules: a larger count is
/// refused, and so is `-1`, every candidate of the vocabulary, which vLLM
/// serves only where `max_logprobs` itself is `-1`. A deployment's workers
/// compute `max_logprobs` candidates a position, a count, so no deployment
/// here serves the whole vocabulary.
#[test]
fn a_logprob_count_above_max_logprobs_is_refused() {
    let (_directory, _, model) = resolved_model("qwen3");
    let lowered = |fields: serde_json::Value| {
        preprocess_chat(
            &model,
            String::from("logprobs"),
            chat_request(fields),
            Vec::new(),
        )
    };

    for count in [0, 5, 20] {
        let (generated, _) =
            lowered(serde_json::json!({"logprobs": true, "top_logprobs": count})).unwrap();
        assert_eq!(generated.sampling.n_logprobs, count);
        let (prompt, _) = lowered(serde_json::json!({"prompt_logprobs": count})).unwrap();
        assert_eq!(prompt.sampling.n_prompt_logprobs, count);
    }

    for count in [21, -1] {
        for (fields, parameter) in [
            (
                serde_json::json!({"logprobs": true, "top_logprobs": count}),
                "logprobs",
            ),
            (
                serde_json::json!({"prompt_logprobs": count}),
                "prompt_logprobs",
            ),
        ] {
            let error = lowered(fields)
                .err()
                .unwrap_or_else(|| panic!("{parameter} of {count} was admitted"));
            assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
            let message = error.to_error_response().error.message;
            assert!(message.contains(parameter), "{count}: {message}");
            assert!(message.contains("20"), "{count}: {message}");
        }
    }
}

/// An image of a family that takes images: one merged token per 32-pixel square, spanning the
/// larger side's tokens, as a Qwen-VL processor states a block.
struct TestImages;

impl processing::ImageProcessor for TestImages {
    fn placeholder(&self) -> &str {
        "<|image_pad|>"
    }

    fn markup(&self) -> &str {
        "<|vision_start|><|image_pad|><|vision_end|>"
    }

    fn block(&self, width: u32, height: u32) -> Result<processing::ImageBlock, String> {
        Ok(processing::ImageBlock {
            tokens: (width / 32) * (height / 32),
            positions: (width / 32).max(height / 32),
        })
    }
}

/// A fetched `width` x `height` PNG.
fn png(width: u32, height: u32) -> ImageInput {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::RgbImage::new(width, height)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    ImageInput::from_bytes(bytes.into_inner()).unwrap()
}

/// A chat request whose user turn is the text `before`, an image, then `after`.
fn image_chat(before: &str, after: &str) -> frontend::protocol::ChatCompletionRequest {
    chat_request(serde_json::json!({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": before},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": after},
        ]}],
    }))
}

/// A deployment whose family serves text alone refuses an image part at admission, naming the
/// feature.
#[test]
fn a_text_only_deployment_refuses_an_image_part() {
    let (_directory, _tokenizer, processor) = resolved_model("qwen3");
    assert!(!processor.supports_image_input());
    let error = preprocess_chat(
        &processor,
        String::from("text-only"),
        image_chat("look ", " now"),
        vec![png(64, 32)],
    )
    .expect_err("an image part reached a text-only deployment");
    let message = error.to_error_response().error.message;
    assert!(message.contains("image_input"), "{message}");
}

/// A family that takes images lowers each image part into a vision block of the rows and
/// positions its processor states, written where the template's placeholder stood, with a
/// digest that keys its encodings within the request's cache partition.
#[test]
fn an_image_part_becomes_a_vision_block_in_place_of_its_placeholder() {
    use base64::Engine as _;

    let (_directory, tokenizer, loaded) = resolved_model("qwen3");
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap()
    .with_multimodal(Some(
        frontend::input::chat::template::MultimodalRenderInfo {
            placeholder_token: "<|vision_start|><|image_pad|><|vision_end|>".to_owned(),
        },
    ));
    let processor = Processor::new(
        loaded.config().clone(),
        Arc::clone(&tokenizer),
        RequestRenderer::Chat(renderer),
        runtime_limits(),
        true,
    )
    .unwrap()
    .with_images(Some(Arc::new(TestImages)));
    assert!(processor.supports_image_input());

    let lower = |salt: Option<&str>| {
        let mut fields = serde_json::json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look "},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": " now"},
            ]}],
        });
        if let Some(salt) = salt {
            fields["cache_salt"] = serde_json::json!(salt);
        }
        preprocess_chat(
            &processor,
            String::from("image"),
            chat_request(fields),
            vec![png(64, 32)],
        )
        .unwrap()
        .0
    };
    let generation = lower(None);
    let pad = tokenizer.token_to_id("<|image_pad|>").unwrap();
    let start = tokenizer.token_to_id("<|vision_start|>").unwrap();
    let end = tokenizer.token_to_id("<|vision_end|>").unwrap();
    assert!(!generation.prompt_token_ids.contains(&pad));
    let [image] = generation.multimodal_inputs.images.as_slice() else {
        panic!("one image part lowers to one image input");
    };
    // The block is written after `<|vision_start|>`, before `<|vision_end|>`.
    let position = image.position as usize;
    assert_eq!(generation.prompt_token_ids[position - 1], start);
    assert_eq!(generation.prompt_token_ids[position], end);
    assert_eq!(image.num_positions, 2);
    let [encoder] = image.encoders.as_slice() else {
        panic!("one vision encoder");
    };
    assert_eq!(encoder.encoder, proto::request::ImageEncoder::Vision);
    assert_eq!(encoder.num_cache_tokens, Some(2));
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(&image.b64)
            .unwrap(),
        png(64, 32).bytes().to_vec()
    );
    // Requests of another cache partition share no encoding of the image.
    assert_ne!(
        lower(Some("tenant")).multimodal_inputs.images[0].digest,
        image.digest
    );
    assert_eq!(lower(None).multimodal_inputs.images[0].digest, image.digest);
}
