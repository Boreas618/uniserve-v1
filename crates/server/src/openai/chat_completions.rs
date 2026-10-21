//! `POST /v1/chat/completions`: validation, the response options, and the
//! buffered and streaming responses.
//!
//! The handler validates and lowers the body (the schema's `validator` rules
//! in `ValidatedJson`, then [`lower_chat_request`]), capturing the response
//! options ([`ChatResponseContext`]), and runs the request through
//! `Engine::generate`. Both response paths consume the `RequestOutput`
//! stream it returns: [`collect_chat_completion`] folds it into one
//! `ChatCompletionResponse`; [`chat_completion_chunk_stream`] maps it to
//! `ChatCompletionStreamResponse` chunks, and [`chat_completion_sse_stream`]
//! frames those chunks as SSE events ending in `data: [DONE]`.
//!
//! One decoded update from the engine becomes several events: semantic events
//! (reasoning and text deltas, block and tool-call boundaries) followed by the
//! update's `TokenDelta`. Events flushed when generation finishes follow the
//! final update's tokens. The streaming path relies on this order to attach
//! token metadata to the chunk of the same update and to drop the metadata of
//! updates that contain hidden reasoning.

use std::collections::HashMap;
use std::convert::Infallible;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Json, Router};
use engine::GenerationConstraint;
use futures::{Stream, StreamExt as _, pin_mut};
use itertools::Itertools as _;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, trace};
use tracing_futures::Instrument as _;

use crate::frontend::Engine;
use crate::http::RequestId;
use crate::openai::error::{bail_invalid_request, bail_server_error, server_error};
use crate::openai::{
    ApiError, AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatImageConfig, ChatLogProbs, ChatLogProbsContent, ChatMessage, ChatMessageDelta,
    ChatModality, ContentPart, FunctionCallDelta, FunctionCallResponse, ImageUrl, MessageContent,
    Tool, ToolCall, ToolCallDelta, TopLogProb, Usage, ValidatedJson, check_model_served,
    convert_logit_bias,
};
use crate::profile::tools;
use crate::serving::chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatContent,
    ChatContentPart, ChatMessage as ServingChatMessage, ChatRequest, ChatToolChoice,
};
use crate::serving::text::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};
use crate::serving::{
    FinishReason, GenerateRequest, ImageSamplingParams, Prompt, RequestOutput, SamplingParams,
    StopReason,
};

/// The chat completions route.
pub(crate) fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(engine)
}

/// Validates one chat completion request and runs it through the engine.
///
/// Returns an SSE stream when `stream` is set and one JSON completion
/// otherwise. Errors from submission, and from collection in the
/// non-streaming case, become `ApiError` responses; errors after an SSE
/// stream has started are reported inside the stream.
async fn chat_completions(
    State(engine): State<Arc<Engine>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_id = format!("chatcmpl-{base_id}");
    let (request, response) = match lower_chat_request(request_id, engine.served_model_name(), body)
    {
        Ok(lowered) => lowered,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %response.request_id,
        engine_request_id = tracing::field::Empty,
    );

    // One timestamp for the whole response, shared by every streamed chunk.
    let created = crate::openai::unix_time();

    let serve_stream = match engine
        .generate(request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(error).into_response(),
    };

    if stream {
        let chunk_stream = chat_completion_chunk_stream(serve_stream, response, created);
        let sse_stream = chat_completion_sse_stream(chunk_stream).instrument(request_span);

        // Emit periodic SSE keep-alive comments so long prefills (no chunks
        // yet) do not get torn down by idle-timeout proxies between client
        // and server.
        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect_chat_completion(serve_stream, response, created)
            .instrument(request_span)
            .await
        {
            Ok(response) => Json(response).into_response(),
            Err(error) => error.into_response(),
        }
    }
}

/// Validates the rules of a chat request that depend on the served model.
///
/// Rejects a `model` other than `served_model_name` with
/// [`ApiError::ModelNotFound`], and rejects with [`ApiError::InvalidRequest`]
/// a `prompt_logprobs` value below `-1` or a positive or `-1` value on a
/// streamed request.
pub fn validate_request_compat(
    request: &ChatCompletionRequest,
    served_model_name: &str,
) -> Result<(), ApiError> {
    check_model_served(&request.model, served_model_name)?;

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        // Any non-negative count, or `-1` for the full vocabulary.
        if prompt_logprobs < -1 {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs must be a non-negative value or -1."
            );
        }

        // Stream chunks (`ChatCompletionStreamResponse`) have no
        // `prompt_logprobs` field, so a streamed response could not deliver
        // them. A streamed `0` is accepted, as vLLM accepts it, and
        // `lower_chat_request` drops it.
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs are not available when stream=true."
            );
        }
    }

    Ok(())
}

/// The response options of one chat request, captured when the request is
/// lowered.
///
/// The handler passes it to [`collect_chat_completion`] or
/// [`chat_completion_chunk_stream`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponseContext {
    /// OpenAI-compatible response identifier.
    pub request_id: String,
    /// Model name reported in response objects.
    pub response_model: String,
    /// Whether streaming responses include a terminal usage chunk
    /// (`stream_options.include_usage`). Non-streaming responses always
    /// carry usage.
    pub include_usage: bool,
    /// Whether generated-token logprobs were requested.
    pub requested_logprobs: bool,
    /// Whether prompt-token logprobs were requested. Only non-streaming
    /// responses carry them; [`validate_request_compat`] rejects a streamed
    /// request that asks for a positive or `-1` count, and
    /// [`lower_chat_request`] drops a streamed `0`.
    pub include_prompt_logprobs: bool,
    /// Whether parsed reasoning is exposed in responses.
    pub include_reasoning: bool,
    /// Whether token identifiers are exposed in responses.
    pub return_token_ids: bool,
    /// Whether textual tokens use token-identifier placeholders.
    pub return_tokens_as_token_ids: bool,
}

impl ChatResponseContext {
    /// Captures the response options of `request`.
    pub fn from_request(
        request: &ChatCompletionRequest,
        request_id: String,
        served_model_name: &str,
    ) -> Self {
        Self {
            request_id,
            response_model: served_model_name.to_string(),
            include_usage: request
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
            requested_logprobs: request.logprobs,
            include_prompt_logprobs: request.prompt_logprobs.is_some(),
            include_reasoning: request.include_reasoning,
            return_token_ids: request.return_token_ids.unwrap_or(false),
            return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
        }
    }
}

/// Collects a chat event stream into one non-streaming response.
///
/// `context` holds the response options the request chose; `created` is the
/// response timestamp in UNIX seconds.
///
/// When reasoning is present but `include_reasoning` is false, the response
/// omits output logprobs and token IDs as well as the reasoning text, because
/// that metadata covers every generated token, reasoning tokens included.
///
/// # Errors
///
/// Returns `ApiError::rejected` for a `Rejected` event and a server error
/// when the stream yields an error or a `Failed` event, closes without a
/// terminal event, lacks output logprobs the response must include or prompt
/// logprobs the request asked for, or fails to convert output logprobs.
pub async fn collect_chat_completion(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    context: ChatResponseContext,
    created: u64,
) -> Result<ChatCompletionResponse, ApiError> {
    // Usage is part of every non-streaming response.
    let ChatResponseContext {
        request_id,
        response_model,
        include_usage: _,
        requested_logprobs,
        include_prompt_logprobs,
        include_reasoning,
        return_token_ids,
        return_tokens_as_token_ids,
    } = context;
    let collected = collect_chat_events(stream).await?;
    let CollectedChatOutput {
        message,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        token_ids,
        images,
        image_steps_per_image,
        reason,
        usage,
    } = collected;

    let stop_reason = stop_reason_value(&reason);
    let saw_tool_calls = message.tool_calls().next().is_some();
    let reasoning = message.reasoning();
    // Output logprobs and token IDs cover the complete generated token stream.
    // When reasoning is hidden, omit them rather than leaking hidden reasoning
    // tokens through per-token metadata.
    let include_output_metadata = include_reasoning || reasoning.is_none();
    let finish_reason = openai_finish_reason(&reason, saw_tool_calls).to_string();
    let tool_calls = message
        .tool_calls()
        .map(|call| ToolCall {
            id: call.id.clone(),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: call.name.clone(),
                arguments: Some(call.arguments.clone()),
            },
        })
        .collect::<Vec<_>>();
    let logprobs = if requested_logprobs && include_output_metadata {
        Some(decoded_logprobs_to_openai_chat(
            logprobs.as_ref().ok_or_else(|| {
                server_error!("chat response requested logprobs but generation returned none")
            })?,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(decoded_prompt_logprobs_to_maps(
            prompt_logprobs.as_ref().ok_or_else(|| {
                server_error!(
                    "chat response requested prompt_logprobs but generation returned none"
                )
            })?,
            return_tokens_as_token_ids,
        ))
    } else {
        None
    };
    let usage = Usage::new(
        usage.prompt_tokens,
        usage.output_tokens,
        usage.image_count,
        usage.image_steps,
        image_steps_per_image,
    );

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: Some(message.text()).filter(|text| !text.is_empty()),
                tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
                reasoning: if include_reasoning { reasoning } else { None },
                images: Some(images).filter(|images| !images.is_empty()),
            },
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            token_ids: (return_token_ids && include_output_metadata).then_some(token_ids),
        }],
        usage: Some(usage),
        prompt_logprobs,
        prompt_token_ids: return_token_ids.then(|| prompt_token_ids.to_vec()),
    })
}

/// Everything `collect_chat_events` accumulates from one request's stream.
#[derive(Debug, Clone, PartialEq)]
struct CollectedChatOutput {
    message: AssistantMessage,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: Option<crate::serving::text::DecodedPromptLogprobs>,
    /// Output logprobs concatenated across `TokenDelta` updates; `None` when
    /// no update carried any.
    logprobs: Option<DecodedLogprobs>,
    token_ids: Vec<u32>,
    images: Vec<ContentPart>,
    /// `ImageStep` counts per completed image, in completion order. Images
    /// that began but never completed are not listed.
    image_steps_per_image: Vec<u32>,
    reason: FinishReason,
    usage: crate::serving::RequestUsage,
}

/// Collects semantic serving events into one non-streaming chat response payload.
///
/// The closed content blocks and tool calls carry the complete message, so
/// the deltas inside them are skipped.
async fn collect_chat_events(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
) -> Result<CollectedChatOutput, ApiError> {
    pin_mut!(stream);
    let mut message = AssistantMessage::default();
    let mut prompt_token_ids = Vec::new();
    let mut prompt_logprobs = None;
    let mut logprobs: Option<DecodedLogprobs> = None;
    let mut token_ids = Vec::new();
    let mut images = Vec::new();
    let mut image_step_counts = HashMap::<u32, u32>::new();
    let mut completed_image_ids = Vec::<u32>::new();

    while let Some(next) = stream.next().await {
        match next {
            Ok(RequestOutput::Accepted(prompt)) => {
                prompt_token_ids = prompt.token_ids.to_vec();
                prompt_logprobs = prompt.logprobs;
            }
            Ok(RequestOutput::TokenDelta {
                token_ids: delta_token_ids,
                logprobs: delta_logprobs,
            }) => {
                token_ids.extend(delta_token_ids);
                if let Some(mut delta_logprobs) = delta_logprobs {
                    logprobs
                        .get_or_insert_with(|| DecodedLogprobs {
                            positions: Vec::new(),
                        })
                        .positions
                        .append(&mut delta_logprobs.positions);
                }
            }
            Ok(RequestOutput::OutputBlockEnd { block, .. }) => message.push_block(block),
            Ok(RequestOutput::ToolCallEnd { call, .. }) => {
                message.push_block(AssistantContentBlock::ToolCall(call));
            }
            Ok(RequestOutput::ImageStep { image_id }) => {
                *image_step_counts.entry(image_id).or_default() += 1;
            }
            Ok(RequestOutput::ImageDone(image)) => {
                image_step_counts.entry(image.image_id).or_default();
                completed_image_ids.push(image.image_id);
                images.push(image_content_part(image.png_base64));
            }
            Ok(RequestOutput::Finished { reason, usage }) => {
                return Ok(CollectedChatOutput {
                    message,
                    prompt_token_ids,
                    prompt_logprobs,
                    logprobs,
                    token_ids,
                    images,
                    image_steps_per_image: completed_image_ids
                        .iter()
                        .map(|image_id| {
                            image_step_counts.get(image_id).copied().unwrap_or_default()
                        })
                        .collect(),
                    reason,
                    usage,
                });
            }
            Ok(RequestOutput::Rejected { kind, message }) => {
                return Err(ApiError::rejected(kind, message));
            }
            Ok(RequestOutput::Failed { message }) => {
                // The failure detail is logged but not returned to the client.
                error!(%message, "chat completion failed");
                bail_server_error!("Internal server error");
            }
            Ok(_) => {}
            Err(error) => {
                return Err(server_error!(
                    "chat completion stream failed: {}",
                    error.to_report_string()
                ));
            }
        }
    }

    Err(server_error!(
        "chat completion stream closed before terminal finish event"
    ))
}

/// Converts one serving event stream into OpenAI chat-completion chunks.
///
/// `context` holds the response options the request chose. Prompt logprobs are never
/// streamed (`ChatResponseContext::include_prompt_logprobs`). The chunk sequence is an
/// assistant-role start chunk, content, reasoning, tool-call, logprob, and
/// image deltas, one chunk carrying `finish_reason`, and, with
/// `include_usage`, a final usage chunk.
///
/// When logprobs or token IDs are requested, the semantic deltas of one
/// decoded update are buffered in a `PendingChatChunk` and emitted together
/// with that update's metadata. With reasoning hidden, the metadata of an
/// update that touches hidden reasoning is dropped along with the reasoning
/// text.
///
/// # Errors
///
/// Yields `ApiError::rejected` for a `Rejected` event, and a server error for
/// a stream error, a `Failed` event, or a logprob conversion failure. The
/// stream ends after the first error.
#[try_stream]
pub async fn chat_completion_chunk_stream(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    context: ChatResponseContext,
    created: u64,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    let ChatResponseContext {
        request_id,
        response_model,
        include_usage,
        requested_logprobs,
        include_prompt_logprobs: _,
        include_reasoning,
        return_token_ids,
        return_tokens_as_token_ids,
    } = context;
    pin_mut!(stream);
    let mut saw_tool_calls = false;
    let mut image_step_counts = HashMap::<u32, u32>::new();
    let mut completed_image_ids = Vec::<u32>::new();
    // Token metadata arrives after all semantic deltas of one decoded update
    // (see the module docs). If that update contains hidden reasoning,
    // including a delimiter-only start or end of a hidden reasoning block,
    // its metadata is omitted along with the reasoning text; with reasoning
    // visible, every update keeps its metadata. `inside_hidden_reasoning`
    // stays set across updates while a hidden reasoning block is open.
    // `suppress_current_update_metadata` marks the current update as touching
    // hidden reasoning and is cleared when its metadata arrives.
    let mut inside_hidden_reasoning = false;
    let mut suppress_current_update_metadata = false;

    // With logprobs or token IDs requested, semantic deltas are buffered until
    // the update's `TokenDelta` arrives, so one chunk carries both.
    let mut pending_chunk =
        (requested_logprobs || return_token_ids).then(PendingChatChunk::default);

    while let Some(next) = stream.next().await {
        match next {
            Ok(RequestOutput::Accepted(prompt)) => {
                let mut chunk = start_chunk(&request_id, &response_model, created);
                if return_token_ids {
                    chunk.prompt_token_ids = Some(prompt.token_ids.to_vec());
                }
                y.yield_ok(chunk).await;
            }
            Ok(RequestOutput::TextDelta { text }) => {
                if !text.is_empty() {
                    let kind = AssistantBlockKind::Text;
                    if let Some(pending_chunk) = pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, text);
                    } else {
                        let chunk =
                            block_delta_chunk(&request_id, &response_model, created, kind, text);
                        y.yield_ok(chunk).await;
                    }
                }
            }
            Ok(RequestOutput::TokenDelta {
                token_ids,
                logprobs,
            }) => {
                let include_metadata =
                    !suppress_current_update_metadata && !inside_hidden_reasoning;
                suppress_current_update_metadata = false;
                let openai_logprobs = if include_metadata {
                    logprobs
                        .as_ref()
                        .map(|lp| decoded_logprobs_to_openai_chat(lp, return_tokens_as_token_ids))
                        .transpose()?
                } else {
                    None
                };
                let openai_token_ids = include_metadata
                    .then_some(token_ids)
                    .and_then(|token_ids| return_token_ids.then_some(token_ids))
                    .filter(|t| !t.is_empty());
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.logprobs = openai_logprobs;
                    pending_chunk.token_ids = openai_token_ids;
                    if let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                    {
                        y.yield_ok(chunk).await;
                    }
                } else if let Some(logprobs) = openai_logprobs {
                    y.yield_ok(logprobs_only_chunk(
                        &request_id,
                        &response_model,
                        created,
                        logprobs,
                    ))
                    .await;
                }
            }
            Ok(RequestOutput::ReasoningDelta { text: delta }) => {
                let kind = AssistantBlockKind::Reasoning;
                let include_delta =
                    include_reasoning || !matches!(kind, AssistantBlockKind::Reasoning);
                if include_delta {
                    if let Some(pending_chunk) = pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, delta);
                    } else {
                        let chunk =
                            block_delta_chunk(&request_id, &response_model, created, kind, delta);
                        y.yield_ok(chunk).await;
                    }
                } else {
                    suppress_current_update_metadata = true;
                }
            }
            Ok(RequestOutput::OutputBlockStart { kind, .. }) => {
                debug!(?kind, "starting new block");
                if !include_reasoning && matches!(kind, AssistantBlockKind::Reasoning) {
                    inside_hidden_reasoning = true;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(RequestOutput::OutputBlockEnd { .. }) => {
                debug!("ending current block");
                // Every block end follows its own start, so this is the end
                // of the hidden reasoning block that the start arm opened.
                if inside_hidden_reasoning {
                    inside_hidden_reasoning = false;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(RequestOutput::ToolCallStart {
                index, id, name, ..
            }) => {
                let tool_index = index as u32;
                saw_tool_calls = true;
                debug!(
                    tool_call_id = %id,
                    tool_call_name = %name,
                    "starting new tool call"
                );
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_start(tool_index, id, name);
                } else {
                    y.yield_ok(tool_call_start_chunk(
                        &request_id,
                        &response_model,
                        created,
                        tool_index,
                        id,
                        name,
                    ))
                    .await;
                }
            }
            Ok(RequestOutput::ToolCallArgumentsDelta { index, delta, .. }) => {
                let tool_index = index as u32;
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_arguments(tool_index, delta);
                } else {
                    y.yield_ok(tool_call_arguments_chunk(
                        &request_id,
                        &response_model,
                        created,
                        tool_index,
                        delta,
                    ))
                    .await;
                }
            }
            Ok(RequestOutput::ToolCallEnd { .. }) => {
                debug!("ending current tool call");
            }
            Ok(RequestOutput::ImageStep { image_id }) => {
                *image_step_counts.entry(image_id).or_default() += 1;
            }
            Ok(RequestOutput::ImageDone(image)) => {
                // Flush buffered deltas first so the image chunk keeps its
                // position after the text generated before it.
                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                {
                    y.yield_ok(chunk).await;
                }
                let chunk =
                    image_delta_chunk(&request_id, &response_model, created, image.png_base64);
                y.yield_ok(chunk).await;
                image_step_counts.entry(image.image_id).or_default();
                completed_image_ids.push(image.image_id);
            }
            Ok(RequestOutput::Finished { reason, usage }) => {
                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                {
                    y.yield_ok(chunk).await;
                }

                y.yield_ok(final_chunk(
                    &request_id,
                    &response_model,
                    created,
                    &reason,
                    saw_tool_calls,
                ))
                .await;

                if include_usage {
                    y.yield_ok(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::new(
                            usage.prompt_tokens,
                            usage.output_tokens,
                            usage.image_count,
                            usage.image_steps,
                            completed_image_ids
                                .iter()
                                .map(|image_id| {
                                    image_step_counts.get(image_id).copied().unwrap_or_default()
                                })
                                .collect(),
                        ),
                    ))
                    .await;
                }

                return Ok(());
            }
            Ok(RequestOutput::Rejected { kind, message }) => {
                return Err(ApiError::rejected(kind, message));
            }
            Ok(RequestOutput::Failed { message }) => {
                // As in `collect_chat_completion`, the failure detail is
                // logged but not returned to the client.
                error!(%message, "chat completion failed");
                bail_server_error!("Internal server error");
            }
            Ok(_) => {}
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "chat stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

/// Builds a streaming usage chunk.
fn usage_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
}

/// Builds a streaming image-delta chunk.
fn image_delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    png_b64: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            images: Some(vec![image_content_part(png_b64)]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Builds an image content part for a chat response.
fn image_content_part(png_b64: String) -> ContentPart {
    ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: format!("data:image/png;base64,{png_b64}"),
            detail: None,
        },
        uuid: None,
    }
}

/// One in-flight chat-completions SSE chunk being assembled at the route layer.
///
/// Serving output processing splits one decoded update into several semantic
/// events and delivers the update's token IDs and logprobs on a `TokenDelta`
/// after them (see the module docs). The OpenAI chat API, though, wants one
/// streamed chunk to optionally carry both the delta and its logprobs.
///
/// This small buffer accumulates the semantic delta first, then attaches the
/// following metadata and flushes one combined chunk. It relies on the
/// output-processing invariant that all semantic events from one decoded
/// update are emitted before that update's metadata. Events flushed when
/// generation finishes are buffered after the final metadata and emitted by
/// the flush before the `finish_reason` chunk.
#[derive(Debug, Default)]
struct PendingChatChunk {
    /// The currently buffered OpenAI delta payload assembled from one or more
    /// chat semantic events belonging to the same decoded update.
    delta: ChatMessageDelta,
    /// The token-aligned logprobs for that same decoded update.
    logprobs: Option<ChatLogProbs>,
    /// Per-update output token IDs for the same decoded update.
    token_ids: Option<Vec<u32>>,
}

impl PendingChatChunk {
    /// Appends one assistant text/reasoning block delta to the buffered OpenAI
    /// delta payload.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: String) {
        match kind {
            AssistantBlockKind::Text => append_delta_text(&mut self.delta.content, delta),
            AssistantBlockKind::Reasoning => append_delta_text(&mut self.delta.reasoning, delta),
        }
    }

    /// Appends the OpenAI tool-call-start representation to the buffered delta.
    fn push_tool_call_start(&mut self, index: u32, id: String, name: String) {
        self.delta
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(ToolCallDelta {
                index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            });
    }

    /// Appends one incremental tool-call arguments update to the buffered delta.
    fn push_tool_call_arguments(&mut self, index: u32, delta: String) {
        self.delta
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(ToolCallDelta {
                index,
                id: None,
                tool_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(delta),
                }),
            });
    }

    /// Finalizes the currently buffered SSE chunk when it contains either a
    /// semantic delta or a logprobs payload.
    ///
    /// This may produce:
    /// - a combined delta + logprobs chunk
    /// - a delta-only chunk
    /// - a logprobs-only chunk
    ///
    /// Metadata-only chunks are intentional: token metadata belongs to a
    /// decoded update, and an update need not produce a visible delta, for
    /// example while the decoder or the output parsers hold back incomplete
    /// text.
    fn take_chunk(
        &mut self,
        request_id: &str,
        response_model: &str,
        created: u64,
    ) -> Option<ChatCompletionStreamResponse> {
        let has_delta = self.delta.content.is_some()
            || self.delta.reasoning.is_some()
            || self.delta.tool_calls.is_some();
        let logprobs = self.logprobs.take();
        let token_ids = self.token_ids.take();
        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            return None;
        }

        let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
        chunk.choices.push(ChatCompletionStreamChoice {
            delta: self.take_delta(),
            logprobs,
            token_ids,
            ..Default::default()
        });
        Some(chunk)
    }

    /// Takes the currently buffered OpenAI delta payload and leaves this pending
    /// chunk empty for the next decoded update.
    fn take_delta(&mut self) -> ChatMessageDelta {
        ChatMessageDelta {
            role: self.delta.role.take(),
            content: self.delta.content.take(),
            tool_calls: self.delta.tool_calls.take(),
            reasoning: self.delta.reasoning.take(),
            images: self.delta.images.take(),
        }
    }
}

/// Appends one text fragment to an optional OpenAI delta string field.
fn append_delta_text(slot: &mut Option<String>, delta: String) {
    match slot {
        Some(existing) => existing.push_str(&delta),
        None => *slot = Some(delta),
    }
}

/// Converts one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error":...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
pub async fn chat_completion_sse_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamResponse, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(chunk) => y.yield_ok(to_sse_event(&chunk)).await,
            Err(error) => {
                y.yield_ok(to_error_sse_event(&error)).await;
                break;
            }
        }
    }

    y.yield_ok(done_sse_event()).await;
    Ok(())
}

/// Serializes one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &ChatCompletionStreamResponse) -> Event {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize chat completion chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting chunk");
    json_sse_event(payload)
}

/// Serializes one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response()).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting error");
    json_sse_event(payload)
}

/// Wraps a serialized JSON payload in a server-sent event.
///
/// Compact `serde_json` output already escapes line breaks inside strings;
/// the replacement guarantees that one payload stays on one `data:` line.
/// axum's `Event::data` splits a raw `\n` across several `data:` fields and
/// panics on a raw `\r`.
fn json_sse_event(payload: String) -> Event {
    Event::default().data(payload.replace('\r', "\\r").replace('\n', "\\n"))
}

/// Builds the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("chat completion emitting done");
    Event::default().data("[DONE]")
}

/// Builds the initial assistant-role SSE chunk required by OpenAI streaming.
fn start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            role: Some(AssistantRole),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Builds one content-delta SSE chunk from one internal assistant block delta.
fn block_delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    kind: AssistantBlockKind,
    delta: String,
) -> ChatCompletionStreamResponse {
    let delta = match kind {
        AssistantBlockKind::Text => ChatMessageDelta {
            content: Some(delta),
            ..Default::default()
        },
        AssistantBlockKind::Reasoning => ChatMessageDelta {
            reasoning: Some(delta),
            ..Default::default()
        },
    };

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta,
        ..Default::default()
    });
    chunk
}

/// Builds the streaming chunk that opens one assistant function call.
fn tool_call_start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    id: String,
    name: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Builds a streaming chunk containing incremental function arguments.
fn tool_call_arguments_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    delta: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: None,
                tool_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(delta),
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Builds a chunk containing only log-probability updates.
fn logprobs_only_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    logprobs: ChatLogProbs,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        logprobs: Some(logprobs),
        ..Default::default()
    });
    chunk
}

/// Builds the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    reason: &FinishReason,
    saw_tool_calls: bool,
) -> ChatCompletionStreamResponse {
    let stop_reason = stop_reason_value(reason);
    let finish_reason = openai_finish_reason(reason, saw_tool_calls);

    debug!(
        finish_reason = %finish_reason,
        stop_reason = ?stop_reason,
        "chat stream finished"
    );

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        stop_reason,
        ..Default::default()
    });
    chunk
}

/// Maps a finish reason to the OpenAI `finish_reason` string.
///
/// A stop after any tool call reports `tool_calls`, and a repetition stop
/// reports `stop`.
fn openai_finish_reason(reason: &FinishReason, saw_tool_calls: bool) -> &'static str {
    match reason {
        FinishReason::Stop(_) if saw_tool_calls => "tool_calls",
        FinishReason::Stop(_) | FinishReason::Repetition => "stop",
        reason => reason.as_str(),
    }
}

/// Returns the `stop_reason` response value: the matched stop token ID or
/// stop string, or `None` for a stop without one and every other reason.
fn stop_reason_value(reason: &FinishReason) -> Option<Value> {
    match reason {
        FinishReason::Stop(Some(StopReason::Token(id))) => Some(serde_json::json!(id)),
        FinishReason::Stop(Some(StopReason::String(text))) => Some(serde_json::json!(text)),
        _ => None,
    }
}

/// Validates and lowers one chat completion request into the serving
/// layer's generate request, capturing its response options.
///
/// # Errors
///
/// Fails as [`validate_request_compat`] does, and with an invalid request
/// for template kwargs that set a variable the request itself determines, an
/// assistant message with no reasoning, content, or tool calls or with image
/// content, a non-function tool call, or a `logit_bias` key that is not a
/// token ID.
pub fn lower_chat_request(
    request_id: String,
    served_model_name: &str,
    request: ChatCompletionRequest,
) -> Result<(GenerateRequest, ChatResponseContext), ApiError> {
    validate_request_compat(&request, served_model_name)?;
    let context =
        ChatResponseContext::from_request(&request, request_id.clone(), served_model_name);

    // Template kwargs may not replace the variables the renderer derives
    // from the request itself.
    let template_kwargs = request.chat_template_kwargs.unwrap_or_default();
    if let Some(name) = crate::serving::chat::ChatOptions::RESERVED_TEMPLATE_KWARGS
        .into_iter()
        .find(|name| template_kwargs.contains_key(*name))
    {
        bail_invalid_request!(
            param = "chat_template_kwargs",
            "chat_template_kwargs may not set `{name}`, which the request itself determines."
        );
    }
    // Stream chunks have no `prompt_logprobs` field. Validation accepts a
    // streamed `prompt_logprobs: 0` for vLLM compatibility, but lowering it
    // would score every prompt position and disable prefix-cache reads for
    // output the client never receives, so it is dropped here.
    let prompt_logprobs = request.prompt_logprobs.filter(|_| !request.stream);
    let messages = request
        .messages
        .into_iter()
        .map(convert_message)
        .try_collect()?;

    let lowered = GenerateRequest {
        request_id,
        prompt: Prompt::Chat(ChatRequest {
            messages,
            tools: convert_tools(request.tools)?,
            // An absent choice lets the model decide, as OpenAI's default does.
            tool_choice: request.tool_choice.unwrap_or(ChatToolChoice::Auto),
            chat_options: crate::serving::chat::ChatOptions {
                reasoning_effort: request.reasoning_effort,
                template_kwargs,
                ..Default::default()
            },
        }),
        constraint: convert_modalities(&request.modalities),
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            seed: request.seed,
            max_tokens: request.max_completion_tokens,
            min_tokens: request.min_tokens,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            ignore_eos: request.ignore_eos,
            stop: request.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
            stop_token_ids: request.stop_token_ids.unwrap_or_default(),
            bad_words: request.bad_words.unwrap_or_default(),
            allowed_token_ids: request.allowed_token_ids,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            logprobs: request
                .logprobs
                .then_some(request.top_logprobs.unwrap_or(0)),
            prompt_logprobs,
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
        },
        image_params: request.image_config.as_ref().map(convert_image_config),
        priority: request.priority.unwrap_or(0),
        cache_salt: request.cache_salt,
    };
    Ok((lowered, context))
}

/// Converts requested response modalities into the generation constraint.
///
/// An empty list selects both text and image output, like an explicit request
/// for both.
fn convert_modalities(modalities: &[ChatModality]) -> GenerationConstraint {
    match (
        modalities.contains(&ChatModality::Text),
        modalities.contains(&ChatModality::Image),
    ) {
        (true, false) => GenerationConstraint::UndOnly,
        (false, true) => GenerationConstraint::GenOnly,
        (true, true) | (false, false) => GenerationConstraint::Default,
    }
}

/// Maps chat `image_config` fields onto serving image controls.
///
/// Wire names differ from the serving names: `guidance_scale` and
/// `image_guidance_scale` are the text and image CFG scales, `cfg_norm` is the
/// renormalization type, and `num_images` is `max_images`.
fn convert_image_config(config: &ChatImageConfig) -> ImageSamplingParams {
    ImageSamplingParams {
        resolution: config.resolution,
        width: config.width,
        height: config.height,
        steps: config.steps,
        cfg_text_scale: config.guidance_scale,
        cfg_img_scale: config.image_guidance_scale,
        cfg_interval: config.cfg_interval,
        cfg_renorm_type: config.cfg_norm,
        cfg_renorm_min: config.cfg_renorm_min,
        timestep_shift: config.timestep_shift,
        seed: config.seed,
        max_images: config.num_images,
    }
}

/// Converts one wire chat message into the serving layer's structured message model.
///
/// Assistant content becomes ordered blocks: non-empty reasoning first, then
/// text content, then tool calls.
///
/// # Errors
///
/// Rejects an assistant message with no reasoning, content, or tool calls, an
/// assistant message with image content, and a non-function tool call.
fn convert_message(message: ChatMessage) -> Result<ServingChatMessage, ApiError> {
    match message {
        ChatMessage::System { content, .. } => {
            Ok(ServingChatMessage::system(convert_content(content)?))
        }
        ChatMessage::User { content, .. } => {
            Ok(ServingChatMessage::user(convert_content(content)?))
        }
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            ..
        } => {
            let mut blocks = Vec::new();
            if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
                blocks.push(AssistantContentBlock::Reasoning { text: reasoning });
            }
            if let Some(content) = content {
                blocks.extend(convert_assistant_content(content)?);
            }
            if let Some(tool_calls) = tool_calls {
                blocks.extend(convert_assistant_tool_calls(tool_calls)?);
            }
            if blocks.is_empty() {
                bail_invalid_request!("Assistant messages must contain content or tool_calls.");
            }
            Ok(ServingChatMessage::assistant_blocks(blocks))
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => Ok(ServingChatMessage::tool_response(
            convert_content(content)?,
            tool_call_id,
        )),
        ChatMessage::Developer { content, tools, .. } => Ok(ServingChatMessage::developer(
            convert_content(content)?,
            convert_message_tools(tools)?,
        )),
    }
}

/// Converts OpenAI message content into serving chat content.
fn convert_content(content: MessageContent) -> Result<ChatContent, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(ChatContent::Text(text)),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(ChatContentPart::text(text)),
                ContentPart::ImageUrl { image_url, uuid } => Ok(ChatContentPart::ImageUrl {
                    image_url: image_url.url,
                    detail: image_url.detail,
                    uuid,
                }),
            })
            .try_collect()
            .map(ChatContent::Parts),
    }
}

/// Converts assistant message content into text blocks, rejecting image parts.
fn convert_assistant_content(
    content: MessageContent,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(vec![AssistantContentBlock::Text { text }]),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(AssistantContentBlock::Text { text }),
                ContentPart::ImageUrl { .. } => {
                    bail_invalid_request!("Assistant message content must be text.")
                }
            })
            .collect(),
    }
}

/// Validates and converts assistant function calls into structured content blocks.
///
/// Missing arguments become the empty JSON object `{}`.
fn convert_assistant_tool_calls(
    tool_calls: Vec<ToolCall>,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    tool_calls
        .into_iter()
        .map(|tool_call| {
            if tool_call.tool_type != "function" {
                bail_invalid_request!("Only function tool calls are supported.");
            }
            Ok(AssistantContentBlock::ToolCall(AssistantToolCall {
                id: tool_call.id,
                name: tool_call.function.name,
                arguments: tool_call
                    .function
                    .arguments
                    .unwrap_or_else(|| "{}".to_string()),
            }))
        })
        .collect()
}

/// Converts OpenAI function tools into serving tool definitions.
fn convert_tools(tools: Option<Vec<Tool>>) -> Result<Vec<tools::Tool>, ApiError> {
    tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| {
            let Tool::Function { function } = tool;
            Ok(tools::Tool {
                name: function.name,
                description: function.description,
                parameters: function.parameters,
                strict: None,
            })
        })
        .collect()
}

/// Converts developer-message tools, mapping an absent or empty list to `None`.
fn convert_message_tools(tools: Option<Vec<Tool>>) -> Result<Option<Vec<tools::Tool>>, ApiError> {
    let tools = convert_tools(tools)?;
    Ok((!tools.is_empty()).then_some(tools))
}

// Logprob values: the text layer's decoded candidates, where each position
// lists its candidate tokens in engine order and a generated position lists
// the sampled token first. With `return_tokens_as_token_ids`, every token
// string is rendered as `token_id:<id>` instead of its decoded text.

/// Converts decoded prompt positions into token-to-logprob maps.
///
/// Index `i` of the result describes prompt position `i`. Entry `0` is `None`
/// because the first prompt token has no preceding context to be scored
/// against; entry `i` comes from `scored_positions[i - 1]`. Candidates whose
/// rendered token strings are equal share one map key, and the later
/// candidate's logprob is kept.
pub fn decoded_prompt_logprobs_to_maps(
    prompt_logprobs: &DecodedPromptLogprobs,
    return_tokens_as_token_ids: bool,
) -> Vec<Option<HashMap<String, f32>>> {
    std::iter::once(None)
        .chain(prompt_logprobs.scored_positions.iter().map(|position| {
            Some(position_top_logprobs_map(
                position,
                return_tokens_as_token_ids,
            ))
        }))
        .collect()
}

/// Converts decoded generated-token candidates into chat logprob content.
///
/// # Errors
///
/// Returns a server error when a position has no candidates, since the
/// sampled token is read from the first candidate.
pub fn decoded_logprobs_to_openai_chat(
    logprobs: &DecodedLogprobs,
    return_tokens_as_token_ids: bool,
) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(|position| position_to_chat_logprobs_content(position, return_tokens_as_token_ids))
        .try_collect()?;
    Ok(ChatLogProbs {
        content: Some(content),
    })
}

/// Formats a token for an OpenAI log-probability response.
fn format_token(entry: &DecodedTokenLogprob, as_token_id: bool) -> String {
    if as_token_id {
        format!("token_id:{}", entry.token_id)
    } else {
        entry.token.clone()
    }
}

/// Returns the top-log-probability map for one token position.
fn position_top_logprobs_map(
    position: &DecodedPositionLogprobs,
    return_tokens_as_token_ids: bool,
) -> HashMap<String, f32> {
    position
        .entries
        .iter()
        .map(|entry| {
            (
                format_token(entry, return_tokens_as_token_ids),
                clamp_logprob(entry.logprob),
            )
        })
        .collect()
}

/// Converts one decoded candidate distribution into OpenAI chat log-probability content.
///
/// The first candidate is the sampled token. `top_logprobs` lists every
/// candidate at the position, including the sampled token. `bytes` holds the
/// UTF-8 encoding of the rendered token string, which is the `token_id:<id>`
/// placeholder when `return_tokens_as_token_ids` is set.
fn position_to_chat_logprobs_content(
    position: &DecodedPositionLogprobs,
    return_tokens_as_token_ids: bool,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        server_error!("decoded chat logprobs position unexpectedly had no token candidates")
    })?;
    let token = format_token(chosen, return_tokens_as_token_ids);
    Ok(ChatLogProbsContent {
        bytes: Some(token.as_bytes().to_vec()),
        token,
        logprob: clamp_logprob(chosen.logprob),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| {
                let token = format_token(entry, return_tokens_as_token_ids);
                TopLogProb {
                    bytes: Some(token.as_bytes().to_vec()),
                    token,
                    logprob: clamp_logprob(entry.logprob),
                }
            })
            .collect(),
    })
}

/// Floors a log probability at `-9999.0`, so `-inf` (and NaN, which
/// `f32::max` discards) reaches the response as a finite number.
fn clamp_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};

    use super::ChatResponseContext;
    use super::chat_completion_chunk_stream;
    use crate::serving::chat::{AssistantBlockKind, AssistantContentBlock};
    use crate::serving::text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};
    use crate::serving::{FinishReason, PromptInfo, RequestOutput, RequestUsage};

    /// Single-candidate logprobs for one generated token.
    fn logprobs(token_id: u32) -> Option<DecodedLogprobs> {
        Some(DecodedLogprobs {
            positions: vec![DecodedPositionLogprobs {
                entries: vec![DecodedTokenLogprob {
                    token_id,
                    token: format!("t{token_id}"),
                    logprob: -0.1,
                    rank: 1,
                }],
            }],
        })
    }

    /// Serving events of a Qwen3 thinking response, in the order the
    /// assembler emits them: the update of token 10 produces reasoning, and
    /// the update of token 11 closes the reasoning block and starts the
    /// answer before its `TokenDelta` arrives.
    fn thinking_response() -> Vec<crate::serving::Result<RequestOutput>> {
        vec![
            Ok(RequestOutput::Accepted(PromptInfo {
                token_ids: vec![1].into(),
                logprobs: None,
            })),
            Ok(RequestOutput::OutputBlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            }),
            Ok(RequestOutput::ReasoningDelta {
                text: "think".to_string(),
            }),
            Ok(RequestOutput::TokenDelta {
                token_ids: vec![10],
                logprobs: logprobs(10),
            }),
            Ok(RequestOutput::OutputBlockEnd {
                index: 0,
                block: AssistantContentBlock::Reasoning {
                    text: "think".to_string(),
                },
            }),
            Ok(RequestOutput::OutputBlockStart {
                index: 1,
                kind: AssistantBlockKind::Text,
            }),
            Ok(RequestOutput::TextDelta {
                text: "Hi".to_string(),
            }),
            Ok(RequestOutput::TokenDelta {
                token_ids: vec![11],
                logprobs: logprobs(11),
            }),
            Ok(RequestOutput::Finished {
                reason: FinishReason::Stop(None),
                usage: RequestUsage::default(),
            }),
        ]
    }

    /// Streams `thinking_response` with logprobs and token IDs requested and
    /// returns the streamed token IDs and the number of logprob positions.
    async fn streamed_metadata(include_reasoning: bool) -> (Vec<u32>, usize) {
        let context = ChatResponseContext {
            request_id: "r".to_string(),
            response_model: "m".to_string(),
            include_usage: false,
            requested_logprobs: true,
            include_prompt_logprobs: false,
            include_reasoning,
            return_token_ids: true,
            return_tokens_as_token_ids: false,
        };
        let chunks = chat_completion_chunk_stream(stream::iter(thinking_response()), context, 0)
            .collect::<Vec<_>>()
            .await;

        let mut token_ids = Vec::new();
        let mut logprob_positions = 0;
        for chunk in chunks {
            for choice in chunk.unwrap().choices {
                token_ids.extend(choice.token_ids.unwrap_or_default());
                logprob_positions += choice
                    .logprobs
                    .and_then(|logprobs| logprobs.content)
                    .map_or(0, |content| content.len());
            }
        }
        (token_ids, logprob_positions)
    }

    /// With reasoning visible, the stream carries the metadata of every
    /// generated token, like the buffered response, including the update
    /// that closes the reasoning block.
    #[tokio::test]
    async fn visible_reasoning_streams_metadata_of_every_update() {
        assert_eq!(streamed_metadata(true).await, (vec![10, 11], 2));
    }

    /// With reasoning hidden, updates that contain hidden reasoning, including
    /// the update that only closes the hidden block, stream no metadata.
    #[tokio::test]
    async fn hidden_reasoning_streams_no_metadata_of_reasoning_updates() {
        assert_eq!(streamed_metadata(false).await, (Vec::new(), 0));
    }
}
