//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use daemon::input::chat::HfChatRenderer;
use daemon::input::media::ImageInput;
use daemon::input::{InputProcessor, ModelResolutionError, RequestRenderer};
use daemon::model::ModelConfig;
use daemon::model::assets::ResolvedModelFiles;
use daemon::openai::ApiError;
use daemon::options::ChatTemplateContentFormat;
use daemon::output::OutputParams;
use daemon::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use processing::{Catalogue, ChatProfile, Family, ReasoningFormat, Serving, ToolCallFormat};
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
    "<img>",
    "</img>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
];

/// A chat family of Qwen3's `model_type`s whose outputs are parsed in its formats.
const CHAT: Family = Family {
    id: "chat",
    model_types: &["qwen3", "qwen3_moe"],
    pipeline_classes: &[],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Qwen3),
        tool_calls: Some(ToolCallFormat::Qwen3),
    }),
};

/// Resolves a chat checkpoint of `model_type` over a synthetic checkpoint with
/// worker limits that cover every generation feature; see
/// `try_resolved_model`.
fn resolved_model(model_type: &str) -> (tempfile::TempDir, DynTokenizer, InputProcessor) {
    try_resolved_model(model_type, runtime_limits()).unwrap()
}

/// Writes a synthetic checkpoint to a temporary directory and binds an
/// `InputProcessor` for it with the given worker limits.
///
/// The tokenizer maps each ASCII code point from 1 to 127 to one token whose
/// ID is the code point (no merges), plus `SPECIAL_TOKENS`, so outside the
/// special tokens one character is one prompt token. `generation_config.json`
/// sets a 128-token default generation length. The returned directory owns
/// the checkpoint files.
///
/// # Errors
///
/// Returns the `InputProcessor::new` error, for example when `limits` do not
/// cover a feature the profile needs. Other setup failures panic.
fn try_resolved_model(
    model_type: &str,
    limits: GenerationLimits,
) -> Result<(tempfile::TempDir, DynTokenizer, InputProcessor), ModelResolutionError> {
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
    let tokenizer: DynTokenizer =
        Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap());
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
            &Catalogue::new([CHAT]).unwrap(),
            &files,
            Some(4096),
            tokenizer.as_ref(),
        ))
        .unwrap();
    let renderer = RequestRenderer::Chat(renderer);
    let model = InputProcessor::new(config, Arc::clone(&tokenizer), renderer, limits, true)?;
    Ok((directory, tokenizer, model))
}

/// Worker limits that cover every generation feature with capacities no test
/// request reaches.
fn runtime_limits() -> GenerationLimits {
    GenerationLimits {
        features: proto::request::GenerationFeatures::all(),
        max_latent_units: 1_000_000,
        latent_downsample: 16,
        max_vae_grid_tokens: 1_000_000,
        max_vit_grid_tokens: 1_000_000,
        latent_bytes_per_unit: u64::MAX.div_ceil(1_000_000),
        max_latent_feature_bytes: u64::MAX,
        max_vision_feature_bytes: u64::MAX,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 32,
    }
}

/// Preprocesses a chat completion as its route lowers it, answering a
/// refusal as the route does.
fn preprocess_chat(
    processor: &InputProcessor,
    request_id: String,
    request: daemon::openai::ChatCompletionRequest,
    images: Vec<ImageInput>,
) -> Result<(GenerationRequest, OutputParams), ApiError> {
    let (request, _) =
        daemon::openai::chat_completions::lower_chat_request(request_id, SERVED_MODEL, request)?;
    processor
        .preprocess(request, images)
        .map_err(ApiError::from)
}

/// A single-turn text chat request to the served model, with the wire
/// `fields` set on top.
fn chat_request(fields: serde_json::Value) -> daemon::openai::ChatCompletionRequest {
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

/// `InputProcessor::new` refuses to bind a profile whose worker limits lack a
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
    let processor = InputProcessor::new(
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
    let mut request: daemon::openai::ChatCompletionRequest =
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
        Err(daemon::openai::ApiError::InvalidRequest { .. })
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
        let request: daemon::openai::ChatCompletionRequest =
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
/// Tool instruction in the style of Qwen3-Coder templates, which ask for XML
/// `<function=...>` calls instead of JSON objects.
const XML_CALL_TEMPLATE: &str = "{%- if tools -%}<|im_start|>system\n# Tools\n<tools>{%- for tool in tools -%}{{ tool | tojson }}{%- endfor -%}</tools>\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter>\nvalue\n</parameter>\n</function>\n</tool_call><|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

/// Rebinds a resolved Qwen3-family model to `template`.
fn with_template(model_type: &str, template: &str) -> InputProcessor {
    let (_directory, tokenizer, loaded) = resolved_model(model_type);
    let renderer = HfChatRenderer::new(
        Some(template.to_string()),
        HashMap::new(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap();
    InputProcessor::new(
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
        let request: daemon::openai::ChatCompletionRequest =
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

/// Dense and mixture-of-experts Qwen3 checkpoints serve tool calls only when
/// their chat template asks for the JSON calls the Qwen3 parser reads; a
/// template asking for another call format refuses requests with tools
/// instead of returning their calls as unparsed text.
#[test]
fn qwen3_family_declares_tool_calling_only_for_json_calls() {
    let tool_request = serde_json::json!({
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {}}
        }}]
    });
    for model_type in ["qwen3", "qwen3_moe"] {
        let json_calls = with_template(model_type, QWEN3_TEMPLATE);
        assert!(json_calls.supports_tool_calling(), "{model_type}");
        preprocess_chat(
            &json_calls,
            String::from("json-tools"),
            serde_json::from_value(tool_request.clone()).unwrap(),
            Vec::new(),
        )
        .unwrap();

        for template in [XML_CALL_TEMPLATE, CHAT_TEMPLATE] {
            let other = with_template(model_type, template);
            assert!(!other.supports_tool_calling(), "{model_type}");
            let error = preprocess_chat(
                &other,
                String::from("other-tools"),
                serde_json::from_value(tool_request.clone()).unwrap(),
                Vec::new(),
            )
            .err()
            .unwrap();
            assert_eq!(error.status_code().as_u16(), 400, "{model_type}");
            let message = error.to_error_response().error.message;
            assert!(message.contains("tool_calling"), "{message}");
        }
    }
}

/// Negative seeds are accepted and reinterpreted as their two's-complement
/// `u64` value, and a logprob count of `-1` requests every candidate
/// (`u32::MAX`).
#[test]
fn negative_seeds_and_logprob_counts_are_reinterpreted() {
    let (_directory, _, model) = resolved_model("qwen3");
    for seed in [-1, -2] {
        let (request, _) = preprocess_chat(
            &model,
            String::from("sampling"),
            chat_request(serde_json::json!({"seed": seed, "logprobs": true, "top_logprobs": -1})),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(request.sampling.seed, Some(seed as u64));
        assert_eq!(request.sampling.n_logprobs, u32::MAX);
    }
}
