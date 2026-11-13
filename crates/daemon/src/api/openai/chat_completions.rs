//! `POST /v1/chat/completions`: the route, and the buffered and streaming
//! responses.
//!
//! The handler parses the body (the schema's `validator` rules in
//! `ValidatedJson`), checks the model it names, submits the request
//! (`Leader::generate`: the deployment's `Processor`, the hold, an engine)
//! and answers with the body `run::GenerationRun` drives one engine event at
//! a time through the request's `OutputProcessor`. Its sink writes each
//! `RequestOutput`: [`ChatCollector`] folds them into one
//! `ChatCompletionResponse`; [`ChatStreamEmitter`] maps them to
//! `ChatCompletionStreamResponse` chunks. Which fields a response carries is
//! the request's response options (`handoff::ChatResponseOptions`), stated
//! when the request was lowered.
//!
//! One decoded update from the engine becomes several outputs: semantic ones
//! (reasoning and text deltas, block and tool-call boundaries) followed by the
//! update's `TokenDelta`. Outputs flushed when generation finishes follow the
//! final update's tokens. The streaming path relies on this order to attach
//! token metadata to the chunk of the same update and to drop the metadata of
//! updates that contain hidden reasoning.

use std::collections::HashMap;
use std::result::Result;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Router};
use itertools::Itertools as _;
use proto::handoff;
use proto::handoff::assistant_block::Block;
use proto::handoff::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
};
use serde_json::Value;
use tracing::{debug, error};
use tracing_futures::Instrument as _;

use crate::api::body::{Opening, Session};
use crate::api::openai::error::{bail_server_error, server_error};
use crate::api::openai::run::{ChatBodySink, GenerationRun};
use crate::api::openai::{ApiError, RequestId, ValidatedJson, check_model_served};
use crate::api::output::chat::AssistantBlockKind;
use crate::api::output::{
    FinishReason, RequestOutput, RequestUsage, StopReason, finish_reason_name,
};
use crate::api::protocol::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse, ChatLogProbs,
    ChatLogProbsContent, ChatMessageDelta, FunctionCallDelta, FunctionCallResponse, ToolCall,
    ToolCallDelta, TopLogProb, Usage,
};
use crate::http::served::Served;

/// The chat completions route.
///
/// Every response's body is attached to the daemon's connection
/// ([`Served::attach`]), so a hot restart freezes it at an engine event
/// boundary and the successor goes on writing it.
pub(crate) fn router(served: Arc<Served>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(served)
}

/// Runs one chat completion request through the engine.
///
/// Returns an SSE stream when `stream` is set and one JSON completion
/// otherwise. A request the deployment refuses, and a failure of a response
/// answered as one document, become `ApiError` responses; errors after an SSE
/// stream has started are reported inside the stream.
async fn chat_completions(
    State(served): State<Arc<Served>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_id = format!("chatcmpl-{base_id}");

    // A request names the served model, or a loaded LoRA adapter, which it
    // then computes with and its response names.
    let adapter = served.leader.adapter(&body.model);
    if adapter.is_none()
        && let Err(error) = check_model_served(&body.model, served.leader.served_model_name())
    {
        return error.into_response();
    }

    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %request_id,
        engine_request_id = tracing::field::Empty,
    );
    let (events, output) = match served
        .leader
        .generate(request_id.clone(), body, adapter)
        .instrument(request_span.clone())
        .await
    {
        Ok(generation) => generation,
        Err(error) => return error.into_response(),
    };

    // A streamed response writes its head at once and answers as server-sent
    // events; a response answered as one document holds its head until the
    // document exists, so its status is its result.
    let (session, content_type, opening): (Box<dyn Session>, _, _) = if stream {
        let sink = ChatStreamEmitter::new(output.response());
        (
            Box::new(GenerationRun::new(events, output, sink)),
            "text/event-stream",
            Opening::Streamed { keep_alive: true },
        )
    } else {
        let sink = ChatBodySink::new(output.response());
        (
            Box::new(GenerationRun::new(events, output, sink)),
            "application/json",
            Opening::Held,
        )
    };

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type);
    if opening != Opening::Held {
        builder = builder
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive");
    }
    let mut response = builder
        .body(Body::empty())
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
    // The response registers before it writes a frame, so a hot restart
    // pauses it at its next engine event boundary.
    match served.attach(&mut response, session, opening, request_id) {
        Ok(()) => response,
        Err(error) => error.into_response(),
    }
}

/// The non-streaming response `collector` collected, with the fields
/// `options` asks for.
///
/// # Errors
///
/// Fails when the request's terminal output has not arrived, when the request
/// asked for logprobs generation did not return, or when a logprob cannot be
/// converted.
pub fn chat_response(
    collector: &ChatCollector,
    options: &handoff::ChatResponseOptions,
) -> Result<ChatCompletionResponse, ApiError> {
    let Some((reason, stop_reason, usage)) = collector.finished.clone() else {
        return Err(server_error!(
            "chat completion stream closed before terminal finish event"
        ));
    };
    let blocks = || {
        collector
            .blocks
            .iter()
            .filter_map(|block| block.block.as_ref())
    };
    let handoff::ChatResponseOptions {
        requested_logprobs,
        top_logprobs,
        include_prompt_logprobs,
        include_reasoning,
        return_token_ids,
        return_tokens_as_token_ids,
        ..
    } = *options;

    let stop_reason = stop_reason_value(stop_reason.as_ref());
    // The message's visible text, its reasoning and its tool calls, each the
    // concatenation of its blocks in order.
    let text: String = blocks()
        .filter_map(|block| match block {
            Block::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let reasoning: String = blocks()
        .filter_map(|block| match block {
            Block::Reasoning(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let reasoning = Some(reasoning).filter(|text| !text.is_empty());
    let tool_calls = blocks()
        .filter_map(|block| match block {
            Block::ToolCall(call) => Some(ToolCall {
                id: call.id.clone(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: call.name.clone(),
                    arguments: Some(call.arguments.clone()),
                },
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let saw_tool_calls = !tool_calls.is_empty();
    // Output logprobs and token IDs cover the complete generated token stream.
    // When reasoning is hidden, omit them rather than leaking hidden reasoning
    // tokens through per-token metadata.
    let include_output_metadata = include_reasoning || reasoning.is_none();
    let finish_reason = openai_finish_reason(&reason, saw_tool_calls).to_string();
    let logprobs = if requested_logprobs && include_output_metadata {
        Some(decoded_logprobs_to_openai_chat(
            collector.logprobs.as_ref().ok_or_else(|| {
                server_error!("chat response requested logprobs but generation returned none")
            })?,
            return_tokens_as_token_ids,
            top_logprobs,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(decoded_prompt_logprobs_to_maps(
            collector.prompt_logprobs.as_ref().ok_or_else(|| {
                server_error!(
                    "chat response requested prompt_logprobs but generation returned none"
                )
            })?,
            return_tokens_as_token_ids,
        ))
    } else {
        None
    };
    let usage = Usage::new(usage.prompt_tokens, usage.output_tokens);

    Ok(ChatCompletionResponse {
        id: options.request_id.clone(),
        object: "chat.completion".to_string(),
        created: options.created,
        model: options.response_model.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: Some(text).filter(|text| !text.is_empty()),
                tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
                reasoning: if include_reasoning { reasoning } else { None },
            },
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            token_ids: (return_token_ids && include_output_metadata)
                .then(|| collector.token_ids.clone()),
        }],
        usage: Some(usage),
        prompt_logprobs,
        prompt_token_ids: return_token_ids.then(|| collector.prompt_token_ids.clone()),
    })
}

/// One non-streaming chat response, collected from the request's outputs.
///
/// The closed content blocks and tool calls carry the complete message, so
/// the deltas inside them are skipped. A hot restart hands this stage over
/// too (`handoff.ChatCollection`, design §7.5.1 step 3): the successor goes on
/// collecting from where the predecessor stopped and writes the one body the
/// response has.
#[derive(Default)]
pub struct ChatCollector {
    /// The closed blocks of the assistant message, in order.
    blocks: Vec<handoff::AssistantBlock>,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: Option<DecodedPromptLogprobs>,
    /// Output logprobs concatenated across `TokenDelta` updates; `None` when
    /// no update carried any.
    logprobs: Option<DecodedLogprobs>,
    token_ids: Vec<u32>,
    /// The request's terminal output: why it finished, the stop token or
    /// string that matched, and what it consumed.
    finished: Option<(FinishReason, Option<StopReason>, RequestUsage)>,
}

impl ChatCollector {
    /// A collector before any output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one output into the collected response.
    ///
    /// # Errors
    ///
    /// Fails as the response does: a rejected or failed request.
    pub fn push(&mut self, output: &RequestOutput) -> Result<(), ApiError> {
        match output {
            RequestOutput::Accepted(prompt) => {
                self.prompt_token_ids = prompt.token_ids.to_vec();
                self.prompt_logprobs = prompt.logprobs.clone();
            }
            RequestOutput::TokenDelta {
                token_ids,
                logprobs,
            } => {
                self.token_ids.extend(token_ids.iter().copied());
                if let Some(delta_logprobs) = logprobs {
                    self.logprobs
                        .get_or_insert_with(|| DecodedLogprobs {
                            positions: Vec::new(),
                        })
                        .positions
                        .extend(delta_logprobs.positions.iter().cloned());
                }
            }
            RequestOutput::OutputBlockEnd { block, .. } => {
                self.blocks.push(handoff::AssistantBlock {
                    block: Some(block.clone()),
                });
            }
            RequestOutput::ToolCallEnd { call, .. } => {
                self.blocks.push(handoff::AssistantBlock {
                    block: Some(Block::ToolCall(call.clone())),
                });
            }
            RequestOutput::Finished {
                reason,
                stop_reason,
                usage,
            } => {
                self.finished = Some((reason.clone(), stop_reason.clone(), usage.clone()));
            }
            RequestOutput::Rejected { kind, message } => {
                return Err(ApiError::rejected(*kind, message.clone()));
            }
            RequestOutput::Failed { message } => {
                // The failure detail is logged but not returned to the client.
                error!(%message, "chat completion failed");
                bail_server_error!("Internal server error");
            }
            _ => {}
        }
        Ok(())
    }

    /// Whether the request's terminal output arrived.
    pub fn done(&self) -> bool {
        self.finished.is_some()
    }

    /// The collector's state, for the process that goes on collecting after a
    /// hot restart (design §7.5.1 step 3).
    pub fn state(&self) -> handoff::ChatCollection {
        let finish_reason = self
            .finished
            .as_ref()
            .map(|(reason, _, _)| finish_reason_name(reason).to_owned())
            .unwrap_or_default();
        handoff::ChatCollection {
            blocks: self.blocks.clone(),
            prompt_token_ids: self.prompt_token_ids.clone(),
            // The logprobs the response already collected are the response:
            // the collector concatenated them as each update carried them, and
            // no later update repeats an earlier position (§7.5.1 step 3).
            prompt_logprobs: self.prompt_logprobs.clone(),
            logprobs: self.logprobs.clone(),
            token_ids: self.token_ids.clone(),
            finish_reason,
            usage: self.finished.as_ref().map(|(_, _, usage)| handoff::Usage {
                prompt_tokens: usage.prompt_tokens,
                output_tokens: usage.output_tokens,
            }),
            stop_reason: None,
        }
    }

    /// Takes over a state this collector produced.
    pub fn resume(&mut self, state: &handoff::ChatCollection) {
        self.blocks.clone_from(&state.blocks);
        self.prompt_token_ids.clone_from(&state.prompt_token_ids);
        self.prompt_logprobs.clone_from(&state.prompt_logprobs);
        self.logprobs.clone_from(&state.logprobs);
        self.token_ids.clone_from(&state.token_ids);
        self.finished = None;
    }
}

/// The OpenAI chunks of one streamed chat completion: converts one serving
/// event stream into OpenAI chat-completion chunks.
///
/// Prompt logprobs are never streamed: a chunk has no field for them. The
/// chunk sequence is an assistant-role start chunk, content, reasoning,
/// tool-call, logprob, and image deltas, one chunk carrying `finish_reason`,
/// and, with `include_usage`, a final usage chunk.
///
/// When logprobs or token IDs are requested, the semantic deltas of one
/// decoded update are buffered in a `PendingChatChunk` and emitted together
/// with that update's metadata. With reasoning hidden, the metadata of an
/// update that touches hidden reasoning is dropped along with the reasoning
/// text.
///
/// A hot restart hands this stage over too (`handoff.ChatChunks`): the
/// successor writes the same chunk sequence, including the chunk an update
/// had buffered when the response stopped.
pub struct ChatStreamEmitter {
    /// The response's OpenAI fields; every chunk repeats its `created`.
    response: handoff::ChatResponseOptions,
    /// Whether a tool call was published, which the finish reason follows.
    saw_tool_calls: bool,
    /// Whether a reasoning block is open that the client does not see.
    inside_hidden_reasoning: bool,
    /// Whether the current update's metadata belongs to hidden reasoning.
    suppress_current_update_metadata: bool,
    /// The chunk being buffered until the update's metadata arrives.
    pending_chunk: Option<PendingChatChunk>,
    /// Body frames produced in all.
    frames: u64,
    /// Whether the terminal chunk was produced; nothing follows it.
    done: bool,
}

impl ChatStreamEmitter {
    /// The emitter of the streamed response `response` describes.
    pub fn new(response: &handoff::ChatResponseOptions) -> Self {
        // With logprobs or token IDs requested, semantic deltas are buffered
        // until the update's `TokenDelta` arrives, so one chunk carries both.
        let pending_chunk = (response.requested_logprobs || response.return_token_ids)
            .then(PendingChatChunk::default);
        Self {
            response: response.clone(),
            saw_tool_calls: false,
            inside_hidden_reasoning: false,
            suppress_current_update_metadata: false,
            pending_chunk,
            frames: 0,
            done: false,
        }
    }

    /// The chunks `output` produces, in order.
    ///
    /// Token metadata arrives after all semantic deltas of one decoded update
    /// (see the module docs). If that update contains hidden reasoning,
    /// including a delimiter-only start or end of a hidden reasoning block,
    /// its metadata is omitted along with the reasoning text; with reasoning
    /// visible, every update keeps its metadata.
    ///
    /// # Errors
    ///
    /// Fails as the response does: a rejected or failed request, or logprobs
    /// that cannot be converted.
    pub fn push(
        &mut self,
        output: &RequestOutput,
    ) -> Result<Vec<ChatCompletionStreamResponse>, ApiError> {
        let mut chunks = Vec::new();
        match output {
            RequestOutput::Accepted(prompt) => {
                let mut chunk = start_chunk(
                    &self.response.request_id,
                    &self.response.response_model,
                    self.response.created,
                );
                if self.response.return_token_ids {
                    chunk.prompt_token_ids = Some(prompt.token_ids.to_vec());
                }
                chunks.push(chunk);
            }
            RequestOutput::TextDelta { text } => {
                if !text.is_empty() {
                    let kind = AssistantBlockKind::Text;
                    if let Some(pending_chunk) = self.pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, text.clone());
                    } else {
                        chunks.push(block_delta_chunk(
                            &self.response.request_id,
                            &self.response.response_model,
                            self.response.created,
                            kind,
                            text.clone(),
                        ));
                    }
                }
            }
            RequestOutput::TokenDelta {
                token_ids,
                logprobs,
            } => {
                let include_metadata =
                    !self.suppress_current_update_metadata && !self.inside_hidden_reasoning;
                self.suppress_current_update_metadata = false;
                let openai_logprobs = if include_metadata {
                    logprobs
                        .as_ref()
                        .map(|lp| {
                            decoded_logprobs_to_openai_chat(
                                lp,
                                self.response.return_tokens_as_token_ids,
                                self.response.top_logprobs,
                            )
                        })
                        .transpose()?
                } else {
                    None
                };
                let openai_token_ids = include_metadata
                    .then_some(token_ids)
                    .and_then(|token_ids| self.response.return_token_ids.then_some(token_ids))
                    .filter(|tokens| !tokens.is_empty());
                if let Some(pending_chunk) = self.pending_chunk.as_mut() {
                    pending_chunk.logprobs = openai_logprobs;
                    pending_chunk.token_ids = openai_token_ids.cloned();
                    if let Some(chunk) = pending_chunk.take_chunk(
                        &self.response.request_id,
                        &self.response.response_model,
                        self.response.created,
                    ) {
                        chunks.push(chunk);
                    }
                } else if let Some(logprobs) = openai_logprobs {
                    chunks.push(logprobs_only_chunk(
                        &self.response.request_id,
                        &self.response.response_model,
                        self.response.created,
                        logprobs,
                    ));
                }
            }
            RequestOutput::ReasoningDelta { text: delta } => {
                let kind = AssistantBlockKind::Reasoning;
                let include_delta = self.response.include_reasoning
                    || !matches!(kind, AssistantBlockKind::Reasoning);
                if include_delta {
                    if let Some(pending_chunk) = self.pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, delta.clone());
                    } else {
                        chunks.push(block_delta_chunk(
                            &self.response.request_id,
                            &self.response.response_model,
                            self.response.created,
                            kind,
                            delta.clone(),
                        ));
                    }
                } else {
                    self.suppress_current_update_metadata = true;
                }
            }
            RequestOutput::OutputBlockStart { kind, .. } => {
                debug!(?kind, "starting new block");
                if !self.response.include_reasoning && matches!(kind, AssistantBlockKind::Reasoning)
                {
                    self.inside_hidden_reasoning = true;
                    self.suppress_current_update_metadata = true;
                }
            }
            RequestOutput::OutputBlockEnd { .. } => {
                debug!("ending current block");
                // Every block end follows its own start, so this is the end of
                // the hidden reasoning block that the start arm opened.
                if self.inside_hidden_reasoning {
                    self.inside_hidden_reasoning = false;
                    self.suppress_current_update_metadata = true;
                }
            }
            RequestOutput::ToolCallStart {
                index, id, name, ..
            } => {
                let tool_index = *index as u32;
                self.saw_tool_calls = true;
                debug!(tool_call_id = %id, tool_call_name = %name, "starting new tool call");
                if let Some(pending_chunk) = self.pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_start(tool_index, id.clone(), name.clone());
                } else {
                    chunks.push(tool_call_start_chunk(
                        &self.response.request_id,
                        &self.response.response_model,
                        self.response.created,
                        tool_index,
                        id.clone(),
                        name.clone(),
                    ));
                }
            }
            RequestOutput::ToolCallArgumentsDelta { index, delta, .. } => {
                let tool_index = *index as u32;
                if let Some(pending_chunk) = self.pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_arguments(tool_index, delta.clone());
                } else {
                    chunks.push(tool_call_arguments_chunk(
                        &self.response.request_id,
                        &self.response.response_model,
                        self.response.created,
                        tool_index,
                        delta.clone(),
                    ));
                }
            }
            RequestOutput::ToolCallEnd { .. } => {
                debug!("ending current tool call");
            }
            RequestOutput::Finished {
                reason,
                stop_reason,
                usage,
            } => {
                if let Some(pending_chunk) = self.pending_chunk.as_mut()
                    && let Some(chunk) = pending_chunk.take_chunk(
                        &self.response.request_id,
                        &self.response.response_model,
                        self.response.created,
                    )
                {
                    chunks.push(chunk);
                }
                chunks.push(final_chunk(
                    &self.response.request_id,
                    &self.response.response_model,
                    self.response.created,
                    reason,
                    stop_reason.as_ref(),
                    self.saw_tool_calls,
                ));
                if self.response.include_usage {
                    chunks.push(usage_chunk(
                        &self.response.request_id,
                        &self.response.response_model,
                        self.response.created,
                        Usage::new(usage.prompt_tokens, usage.output_tokens),
                    ));
                }
                self.done = true;
            }
            RequestOutput::Rejected { kind, message } => {
                return Err(ApiError::rejected(*kind, message.clone()));
            }
            RequestOutput::Failed { message } => {
                // As in the collected response, the failure detail is
                // logged but not returned to the client.
                error!(%message, "chat completion failed");
                bail_server_error!("Internal server error");
            }
            _ => {}
        }
        self.frames = self.frames.saturating_add(chunks.len() as u64);
        Ok(chunks)
    }

    /// Whether the terminal chunk was produced.
    pub fn done(&self) -> bool {
        self.done
    }

    /// The stage's state, for the process that goes on writing the response
    /// after a hot restart.
    ///
    /// A response freezes at an engine event boundary, and the buffered chunk
    /// only ever holds the deltas of one decoded update, whose metadata
    /// arrives in the same engine event: a step boundary is always a point
    /// with nothing buffered. A buffered chunk here is a defect, and the
    /// stage reports that it cannot be handed over rather than losing it.
    pub fn state(&self) -> Option<handoff::ChatChunks> {
        if self.has_pending_chunk() {
            return None;
        }
        Some(handoff::ChatChunks {
            saw_tool_calls: self.saw_tool_calls,
            inside_hidden_reasoning: self.inside_hidden_reasoning,
            suppress_current_update_metadata: self.suppress_current_update_metadata,
            frames: self.frames,
            done: self.done,
            created: self.response.created,
        })
    }

    /// Takes over a state this stage produced.
    pub fn resume(&mut self, state: &handoff::ChatChunks) {
        self.saw_tool_calls = state.saw_tool_calls;
        self.inside_hidden_reasoning = state.inside_hidden_reasoning;
        self.suppress_current_update_metadata = state.suppress_current_update_metadata;
        self.frames = state.frames;
        self.done = state.done;
    }

    /// Whether a chunk is buffered from an update whose metadata has not
    /// arrived yet.
    fn has_pending_chunk(&self) -> bool {
        self.pending_chunk.as_ref().is_some_and(|pending| {
            pending.delta.content.is_some()
                || pending.delta.reasoning.is_some()
                || pending.delta.tool_calls.is_some()
                || pending.logprobs.is_some()
                || pending.token_ids.is_some()
        })
    }
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
                // The call opens with its name and empty arguments, as OpenAI's first
                // chunk of a call does, so a client that concatenates `arguments` has
                // a string to start from.
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: Some(String::new()),
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
                // The call opens with its name and empty arguments, as OpenAI's first
                // chunk of a call does, so a client that concatenates `arguments` has
                // a string to start from.
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: Some(String::new()),
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
    stop_reason: Option<&StopReason>,
    saw_tool_calls: bool,
) -> ChatCompletionStreamResponse {
    let stop_reason = stop_reason_value(stop_reason);
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
    match finish_reason_name(reason) {
        "stop" if saw_tool_calls => "tool_calls",
        "repetition" => "stop",
        name => name,
    }
}

/// Returns the `stop_reason` response value: the matched stop token ID or
/// stop string, or `None` for a finish without one.
fn stop_reason_value(reason: Option<&StopReason>) -> Option<Value> {
    match reason {
        Some(StopReason::Token(id)) => Some(serde_json::json!(id)),
        Some(StopReason::String(text)) => Some(serde_json::json!(text)),
        None => None,
    }
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

/// Converts decoded generated-token candidates into chat logprob content, each position listing
/// its first `top_logprobs` candidates.
///
/// # Errors
///
/// Returns a server error when a position has no candidates, since the
/// sampled token is read from the first candidate.
pub fn decoded_logprobs_to_openai_chat(
    logprobs: &DecodedLogprobs,
    return_tokens_as_token_ids: bool,
    top_logprobs: u32,
) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(|position| {
            position_to_chat_logprobs_content(position, return_tokens_as_token_ids, top_logprobs)
        })
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
/// The first candidate is the sampled token, then the position's most likely tokens without it.
/// `top_logprobs` lists the first `limit` of those candidates, the sampled token first, as vLLM's
/// chat response does. `bytes` holds the UTF-8 encoding of the rendered token string, which is the
/// `token_id:<id>` placeholder when `return_tokens_as_token_ids` is set.
fn position_to_chat_logprobs_content(
    position: &DecodedPositionLogprobs,
    return_tokens_as_token_ids: bool,
    limit: u32,
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
            .take(usize::try_from(limit).unwrap_or(usize::MAX))
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
    use proto::handoff::ChatResponseOptions;

    use super::ChatCollector;
    use super::ChatStreamEmitter;
    use super::chat_response;
    use crate::api::output::chat::AssistantBlockKind;
    use crate::api::output::{FinishReason, PromptInfo, RequestOutput, RequestUsage};
    use crate::api::protocol::ChatCompletionStreamResponse;
    use proto::handoff::assistant_block::Block;
    use proto::handoff::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

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
    fn thinking_response() -> Vec<Result<RequestOutput, crate::api::output::OutputError>> {
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
                block: Block::Reasoning("think".to_string()),
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
                reason: FinishReason::Stop,
                stop_reason: None,
                usage: RequestUsage::default(),
            }),
        ]
    }

    /// Streams `thinking_response` with logprobs and token IDs requested and
    /// returns the streamed token IDs and the number of logprob positions.
    async fn streamed_metadata(include_reasoning: bool) -> (Vec<u32>, usize) {
        let context = ChatResponseOptions {
            request_id: "r".to_string(),
            response_model: "m".to_string(),
            include_usage: false,
            requested_logprobs: true,
            include_reasoning,
            return_token_ids: true,
            created: 7,
            ..Default::default()
        };
        let mut emitter = ChatStreamEmitter::new(&context);
        let chunks: Vec<ChatCompletionStreamResponse> = thinking_response()
            .into_iter()
            .flat_map(|output| emitter.push(&output.unwrap()).unwrap())
            .collect();

        let mut token_ids = Vec::new();
        let mut logprob_positions = 0;
        for chunk in chunks {
            for choice in chunk.choices {
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

    /// The chunk as the client receives it, for comparing two runs.
    fn chunks_of(chunks: &[ChatCompletionStreamResponse]) -> Vec<serde_json::Value> {
        chunks
            .iter()
            .map(|chunk| serde_json::to_value(chunk).expect("a chunk serializes"))
            .collect()
    }

    #[tokio::test]
    async fn a_handed_over_chunk_stage_writes_the_same_chunks() {
        let context = ChatResponseOptions {
            request_id: "r".to_string(),
            response_model: "m".to_string(),
            include_usage: true,
            requested_logprobs: true,
            include_reasoning: true,
            return_token_ids: true,
            created: 7,
            ..Default::default()
        };
        let outputs: Vec<RequestOutput> = thinking_response()
            .into_iter()
            .map(|output| output.expect("a served event"))
            .collect();

        // The same response written in one process: the reference sequence.
        let mut uninterrupted = ChatStreamEmitter::new(&context);
        let mut expected = Vec::new();
        for output in &outputs {
            expected.extend(uninterrupted.push(output).expect("a chunk"));
        }
        let expected = chunks_of(&expected);

        // The same response handed over at every position: where the stage can
        // state itself, the successor writes exactly the rest of the same
        // chunks; where it cannot, it says so instead of losing one.
        let mut handable = 0;
        for split in 1..outputs.len() {
            let mut first = ChatStreamEmitter::new(&context);
            let mut chunks = Vec::new();
            for output in &outputs[..split] {
                chunks.extend(first.push(output).expect("a chunk"));
            }
            let state = first.state();
            let Some(state) = state else {
                // Only a buffered update keeps the stage from being stated.
                continue;
            };
            handable += 1;
            let mut second = ChatStreamEmitter::new(&context);
            second.resume(&state);
            for output in &outputs[split..] {
                chunks.extend(second.push(output).expect("a chunk"));
            }
            assert_eq!(
                chunks_of(&chunks),
                expected,
                "a handoff after {split} outputs writes different chunks"
            );
        }
        assert!(
            handable > 0,
            "no position in the response can be handed over"
        );
    }

    /// The collected one-body response continues where a handoff left it: a
    /// collector handed over mid-response collects exactly the response the
    /// uninterrupted one does.
    #[tokio::test]
    async fn a_handed_over_collector_writes_the_same_response() {
        let context = ChatResponseOptions {
            request_id: "r".to_string(),
            response_model: "m".to_string(),
            include_usage: false,
            requested_logprobs: true,
            include_reasoning: true,
            return_token_ids: true,
            created: 7,
            ..Default::default()
        };
        let outputs: Vec<RequestOutput> = thinking_response()
            .into_iter()
            .map(|output| output.expect("a served event"))
            .collect();

        let mut uninterrupted = ChatCollector::new();
        for output in &outputs {
            uninterrupted.push(output).expect("a collected output");
        }
        let expected =
            serde_json::to_value(chat_response(&uninterrupted, &context).expect("a response"))
                .expect("the response serializes");

        for split in 1..outputs.len() {
            let mut first = ChatCollector::new();
            for output in &outputs[..split] {
                first.push(output).expect("a collected output");
            }
            // The blocks, token ids and images travel; the request's terminal
            // output arrives after the handoff.
            let mut second = ChatCollector::new();
            second.resume(&first.state());
            for output in &outputs[split..] {
                second.push(output).expect("a collected output");
            }
            let response =
                serde_json::to_value(chat_response(&second, &context).expect("a response"))
                    .expect("the response serializes");
            assert_eq!(
                response, expected,
                "a handoff after {split} outputs differs"
            );
        }
    }

    /// A stage holding a buffered chunk reports that it cannot be handed over,
    /// rather than handing over a state that would drop it.
    #[tokio::test]
    async fn a_buffered_chunk_cannot_be_handed_over() {
        let context = ChatResponseOptions {
            request_id: "r".to_string(),
            response_model: "m".to_string(),
            include_usage: false,
            requested_logprobs: true,
            include_reasoning: true,
            return_token_ids: true,
            created: 7,
            ..Default::default()
        };
        // The deltas of one update are buffered until its metadata arrives.
        let mut emitter = ChatStreamEmitter::new(&context);
        emitter
            .push(&RequestOutput::TextDelta {
                text: "Hi".to_string(),
            })
            .expect("a buffered delta");
        assert!(
            emitter.state().is_none(),
            "a buffered chunk is never stated as handed over"
        );
        emitter
            .push(&RequestOutput::TokenDelta {
                token_ids: vec![11],
                logprobs: logprobs(11),
            })
            .expect("the update's metadata");
        assert!(
            emitter.state().is_some(),
            "after the update's metadata the stage can be handed over again"
        );
    }
}
