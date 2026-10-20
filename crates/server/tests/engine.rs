//! Requests through the engine with real preprocessing and output assembly:
//! admission rejections, request timing, and the request metrics.

#![cfg(test)]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use futures::{StreamExt, poll};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer};
use uniserve_engine::{EngineConfig, EngineCore, SimEngine, SimExecutor};
use uniserve_server::Engine;
use uniserve_server::openai::ChatCompletionRequest;
use uniserve_server::options::{ImageFetchOptions, LoggingOptions};
use uniserve_server::profile::tokenizer::HuggingFaceTokenizer;
use uniserve_server::profile::{ModelConfig, ModelParameters, SamplingDefaults};
use uniserve_server::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
use uniserve_server::serving::media::MediaConnector;
use uniserve_server::serving::{FinishReason, InputProcessor, RequestOutput, RequestRenderer};

/// Builds an engine over a default `SimEngine`.
/// `request` as the chat route lowers it for the served `sim-model`.
fn lowered(id: &str, request: ChatCompletionRequest) -> uniserve_server::serving::GenerateRequest {
    uniserve_server::openai::chat_completions::lower_chat_request(
        id.to_owned(),
        "sim-model",
        request,
    )
    .unwrap()
    .0
}

fn runtime() -> Engine {
    runtime_with_worker(SimEngine::new())
}

/// Builds a Qwen3 engine whose engine core drives `worker` in process through
/// `SimExecutor`.
fn runtime_with_worker(worker: SimEngine) -> Engine {
    runtime_for_engine_model(worker, "sim-model")
}

/// Builds a Qwen3 engine whose engine core, named `engine_model`, drives
/// `worker` in process through `SimExecutor`.
///
/// The engine's model name labels the process-wide request metrics, so a test
/// that reads them names its own engine. The tokenizer maps each code point
/// below 128 to one token whose ID is the code point; `<|im_end|>` is added
/// as ID 128, which the model config names as its end-of-sequence token. The
/// chat template renders only the first message's content.
fn runtime_for_engine_model(worker: SimEngine, engine_model: &str) -> Engine {
    let vocabulary = (0_u32..128)
        .map(|id| (char::from_u32(id).unwrap().to_string(), id))
        .collect::<Vocab>();
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false).unwrap();
    let tokenizer = Arc::new(HuggingFaceTokenizer::new(&path).unwrap());

    let core = EngineCore::with_executor(
        EngineConfig::sim(engine_model),
        Box::new(SimExecutor::new(worker)),
    )
    .unwrap();
    let processor = InputProcessor::new(
        ModelConfig {
            parameters: ModelParameters::Qwen3,
            sampling_defaults: SamplingDefaults::default(),
            max_model_len: 4096,
            primary_eos_token_id: Some(128),
            eos_token_ids: BTreeSet::from([128]),
        },
        tokenizer,
        RequestRenderer::Chat(
            HfChatRenderer::new(
                Some("{{ messages[0].content }}".to_string()),
                HashMap::new(),
                ChatTemplateContentFormatOption::String,
            )
            .unwrap(),
        ),
        core.generation_limits(),
        false,
    )
    .unwrap();

    let images = MediaConnector::from_image_options(&ImageFetchOptions::default()).unwrap();
    let logging = LoggingOptions {
        log_requests: false,
        log_stats: false,
    };
    Engine::new(
        "sim-model".to_owned(),
        vec![core],
        processor,
        images,
        None,
        &logging,
    )
    .unwrap()
}

/// Greedy chat request of at most `max_completion_tokens` tokens that may
/// sample only `a`, stopping on each of `stop`.
fn request_with(max_completion_tokens: u32, stop: &[&str]) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "sim-model",
        "messages": [{"role": "user", "content": "prompt"}],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.0,
        "allowed_token_ids": [u32::from(b'a')],
        "stop": stop,
    }))
    .unwrap()
}

/// A request that deterministically stops on the `aa` stop string after two
/// output tokens.
fn request() -> ChatCompletionRequest {
    request_with(8, &["aa"])
}

/// A chat prompt that exceeds the total capacity of a two-unit prefix cache is
/// returned as a stream and then rejected by engine admission. Collecting the
/// completion must report that rejection as HTTP 400.
#[tokio::test]
async fn chat_admission_rejection_remains_a_bad_request() {
    let mut worker = SimEngine::new();
    // Unit zero is the sentinel: two allocatable 64-token pages remain.
    worker.set_num_units(3);
    let runtime = runtime_with_worker(worker);
    let request = serde_json::from_value(serde_json::json!({
        "model": "sim-model",
        "messages": [{"role": "user", "content": "a".repeat(1024)}],
        "max_completion_tokens": 8
    }))
    .unwrap();

    let stream = runtime
        .generate(lowered("rejected-chat", request))
        .await
        .unwrap();
    let context = uniserve_server::openai::chat_completions::ChatResponseContext {
        request_id: "rejected-chat".to_string(),
        response_model: "sim-model".to_string(),
        include_usage: false,
        requested_logprobs: false,
        include_prompt_logprobs: false,
        include_reasoning: false,
        return_token_ids: false,
        return_tokens_as_token_ids: false,
    };
    let error =
        uniserve_server::openai::chat_completions::collect_chat_completion(stream, context, 0)
            .await
            .unwrap_err();

    assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    runtime.shutdown().await.unwrap();
}

/// `compile_us` is measured from the start of `generate`, so it includes
/// time spent queued for the blocking pool; the reported total latency must
/// cover that wait as well.
#[test]
fn successful_total_includes_time_waiting_for_preprocessing() {
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    executor.block_on(async {
        let runtime = runtime();

        // Hold the only blocking thread so preprocessing waits in the queue
        // for a measurable interval before it runs.
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let occupied = tokio::task::spawn_blocking(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx.await.unwrap();
        let mut pending = Box::pin(runtime.generate(lowered("timed", request())));
        assert!(poll!(pending.as_mut()).is_pending());
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        release_tx.send(()).unwrap();
        occupied.await.unwrap();

        let mut stream = pending.await.unwrap();
        let mut timing = None;
        while let Some(event) = stream.next().await {
            if let RequestOutput::Finished { usage, .. } = event.unwrap() {
                timing = Some(usage.timings);
            }
        }
        let timing = timing.unwrap();
        assert!(timing.total_us >= timing.compile_us, "{timing:?}");
        runtime.shutdown().await.unwrap();
    });
}

/// A completed request is recorded once in the request metrics under its
/// engine's model name: the token counters and token histograms carry the
/// counts of its usage report, its finish status is counted, and each latency
/// histogram whose input it carries holds one observation.
#[tokio::test]
async fn completed_request_is_recorded_in_request_metrics() {
    let engine_model = "request-metrics-model";
    let runtime = runtime_for_engine_model(SimEngine::new(), engine_model);

    let mut stream = runtime
        .generate(lowered("metrics", request_with(3, &[])))
        .await
        .unwrap();
    let mut usage = None;
    let mut finish = None;
    while let Some(event) = stream.next().await {
        if let RequestOutput::Finished {
            reason,
            usage: used,
        } = event.unwrap()
        {
            usage = Some((used.prompt_tokens, used.output_tokens));
            finish = Some(reason);
        }
    }
    assert_eq!(finish, Some(FinishReason::Length));
    let (prompt_tokens, generation_tokens) = usage.unwrap();
    assert_eq!(generation_tokens, 3);

    let rendered = uniserve_observability::METRICS.render().unwrap();
    let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
    for expected in [
        format!("uniserve:request_success_total{{{labels},finished_reason=\"length\"}} 1"),
        format!("uniserve:prompt_tokens_total{{{labels}}} {prompt_tokens}"),
        format!("uniserve:generation_tokens_total{{{labels}}} {generation_tokens}"),
        format!("uniserve:request_prompt_tokens_count{{{labels}}} 1"),
        format!("uniserve:request_generation_tokens_count{{{labels}}} 1"),
        format!("uniserve:request_max_num_generation_tokens_count{{{labels}}} 1"),
        format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
        format!("uniserve:request_queue_time_seconds_count{{{labels}}} 1"),
        format!("uniserve:time_to_first_token_seconds_count{{{labels}}} 1"),
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing `{expected}` in:\n{rendered}"
        );
    }
    runtime.shutdown().await.unwrap();
}

/// A client that disconnects mid-generation is counted as an aborted request
/// with its latency, while its partial output, which never reached a usage
/// report, adds nothing to the token metrics.
#[tokio::test]
async fn disconnected_request_is_recorded_as_aborted_without_token_usage() {
    let engine_model = "disconnect-metrics-model";
    let mut worker = SimEngine::new();
    worker.set_text_len(1_000_000);
    let runtime = runtime_for_engine_model(worker, engine_model);

    let mut stream = runtime
        .generate(lowered("disconnect", request_with(1_000, &[])))
        .await
        .unwrap();
    loop {
        match stream.next().await.unwrap().unwrap() {
            RequestOutput::TextDelta { text } if !text.is_empty() => break,
            _ => {}
        }
    }
    drop(stream);

    let rendered = uniserve_observability::METRICS.render().unwrap();
    let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
    for expected in [
        format!("uniserve:request_success_total{{{labels},finished_reason=\"abort\"}} 1"),
        format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing `{expected}` in:\n{rendered}"
        );
    }
    for family in [
        "uniserve:generation_tokens_total",
        "uniserve:request_generation_tokens_count",
    ] {
        let series = format!("{family}{{{labels}}}");
        assert!(
            !rendered.lines().any(|line| line.starts_with(&series)),
            "unexpected `{series}` in:\n{rendered}"
        );
    }
    runtime.shutdown().await.unwrap();
}
