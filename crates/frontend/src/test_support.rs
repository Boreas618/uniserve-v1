//! Fixtures shared by the crate's tests.
//!
//! `configured_tokenizer` builds the deterministic byte tokenizer the serving
//! tests decode with: every printable ASCII character is one token whose ID is
//! its code point, and the chat, vision and thinking special tokens encode
//! atomically. [`CHAT`] and [`VIDEO`] are the families those tests serve, and
//! [`TestProcessor`] processes the video family's requests. It is compiled for the
//! crate's own tests and, through the `testing` feature, for the tests of the
//! crates that serve over it.

#![expect(
    clippy::expect_used,
    reason = "a fixture that cannot be built is the test's failure report"
)]

use std::sync::Arc;

use tokenizers::AddedToken;
use tokenizers::Tokenizer as TokenizerBuilder;
use tokenizers::models::bpe::{BPE, Vocab};

use crate::input::video::VideoService;
use crate::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

/// Loads a `tokenizers` definition in storage, as the frontend loads a
/// checkpoint's `tokenizer.json`.
pub fn load_tokenizer(definition: &TokenizerBuilder) -> HuggingFaceTokenizer {
    let json = definition.to_string(false).expect("serialize tokenizer");
    HuggingFaceTokenizer::from_json(serde_json::from_str(&json).expect("parse tokenizer"))
        .expect("load tokenizer")
}

/// Builds the deterministic byte-oriented tokenizer the serving tests use.
///
/// Every printable ASCII character encodes to the token whose ID equals its
/// code point, so tests write IDs as `b'a' as u32`. ID 0 is the `<unk>` token,
/// and IDs 1 through 6 become the chat, vision, and thinking special tokens.
/// The BPE model has no merges, so encoding is one token per character.
pub fn configured_tokenizer() -> DynTokenizer {
    let mut vocab = (0_u32..=127)
        .map(|id| {
            let token = if id == 0 {
                "<unk>".to_string()
            } else {
                char::from_u32(id).expect("ASCII code point").to_string()
            };
            (token, id)
        })
        .collect::<Vocab>();
    for (id, token) in [
        (1, "<|im_start|>"),
        (2, "<|im_end|>"),
        (3, "<|vision_start|>"),
        (4, "<|vision_end|>"),
        (5, "<think>"),
        (6, "</think>"),
        (97, "a"),
        (98, "b"),
    ] {
        vocab.retain(|_, existing_id| *existing_id != id);
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("build configured tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
        AddedToken::from("<|vision_start|>", true),
        AddedToken::from("<|vision_end|>", true),
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
    ]);
    Arc::new(load_tokenizer(&tokenizer))
}

/// A chat family whose outputs are parsed in the Qwen3 formats.
pub const CHAT: processing::Family = processing::Family {
    id: "chat",
    model_types: &["chat"],
    classes: &[],
    serving: processing::Serving::Chat(processing::ChatProfile {
        reasoning: Some(processing::ReasoningFormat::Qwen3),
        tool_calls: Some(processing::ToolCallFormat::Qwen3),
        images: None,
    }),
};

/// A video family whose processor is [`TestProcessor`].
pub const VIDEO: processing::Family = processing::Family {
    id: "video",
    model_types: &[],
    classes: &["TestVideoPipeline"],
    serving: processing::Serving::Video(&TestVideo),
};

/// The video family [`VIDEO`] serves.
struct TestVideo;

impl processing::VideoFamily for TestVideo {
    fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn bind(
        &self,
        _: processing::VideoBinding<'_>,
    ) -> Result<Box<dyn processing::VideoProcessor>, String> {
        Ok(Box::new(TestProcessor))
    }
}

/// A video model of one rule and plain inputs, for request processing's own behavior: a request shorter
/// than a second is refused before its media is fetched, and every image reference is one
/// condition whose rows are its height.
pub struct TestProcessor;

impl processing::VideoProcessor for TestProcessor {
    fn capabilities(&self) -> serde_json::Value {
        serde_json::json!({"sizes": ["64x64"]})
    }

    fn denoise_steps(&self) -> u32 {
        4
    }

    fn check(&self, request: &processing::VideoSpec) -> Result<(), processing::InputError> {
        match request.length {
            Some(processing::Length::Seconds(seconds)) if seconds < 1.0 => Err(
                processing::InputError::invalid("seconds", "a video lasts at least a second"),
            ),
            _ => Ok(()),
        }
    }

    fn process(
        &self,
        request: &processing::VideoSpec,
        media: &[processing::MediaMetadata],
        vocabulary: &dyn processing::Vocabulary,
    ) -> Result<processing::VideoInput, processing::InputError> {
        use proto::video::{Canvas, ConditionMedia, ConditionRole, ImageFit};

        let conditions = request
            .references
            .iter()
            .zip(media)
            .map(|(reference, media)| {
                let processing::MediaMetadata::Image(image) = media else {
                    return Err(processing::InputError::Internal(
                        "the test processor processes images".to_owned(),
                    ));
                };
                let canvas = Canvas {
                    width: image.width,
                    height: image.height,
                };
                Ok(processing::ConditionInput {
                    role: match reference.frame_index {
                        Some(0) => ConditionRole::FirstFrame,
                        Some(_) => ConditionRole::LastFrame,
                        None => ConditionRole::Reference,
                    },
                    media: ConditionMedia::Image(ImageFit {
                        resized: canvas,
                        left: 0,
                        top: 0,
                        size: canvas,
                    }),
                    vision: None,
                    latent_units: vec![image.height],
                    audio_rows: 0,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prompt_token_ids = vocabulary
            .encode(&request.prompt)
            .map_err(|error| processing::InputError::Internal(error.to_string()))?;
        Ok(processing::VideoInput {
            size: request.size.unwrap_or(Canvas {
                width: 64,
                height: 64,
            }),
            num_frames: 22,
            fps: 24,
            seed: request.seed.unwrap_or(42),
            prompt_token_ids,
            vision_spans: Vec::new(),
            condition_rows: conditions
                .iter()
                .flat_map(|condition| condition.latent_units.iter())
                .sum(),
            conditions,
            video_units: 2,
        })
    }
}

/// The video service of [`TestProcessor`] under the default media policy.
pub fn test_video_service() -> VideoService {
    VideoService::new(
        Box::new(TestProcessor),
        &proto::config::MediaConfig::default(),
        configured_tokenizer(),
    )
    .expect("the default media policy builds a video service")
}

/// A request's engine events sent on a channel, as an engine that never
/// migrates the request delivers them: the engine boundary of the frontend's
/// own tests.
pub struct ChannelEvents(tokio::sync::mpsc::UnboundedReceiver<proto::request::Event>);

impl ChannelEvents {
    /// The events the channel's sender sends.
    pub fn new(events: tokio::sync::mpsc::UnboundedReceiver<proto::request::Event>) -> Self {
        Self(events)
    }
}

impl crate::events::Events for ChannelEvents {
    fn poll_step(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<crate::events::Next> {
        self.0.poll_recv(cx).map(|event| match event {
            Some(event) => crate::events::Next::Event(event),
            None => crate::events::Next::Ended,
        })
    }

    fn recv(
        &mut self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Option<proto::request::Event>> + Send + '_>,
    > {
        Box::pin(self.0.recv())
    }

    fn migrate_pending(
        &mut self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), proto::request::Event>> + Send + '_>,
    > {
        Box::pin(std::future::ready(Ok(())))
    }

    fn acknowledge_consumed_prefix(&mut self) {}

    fn stop_at_consumed_prefix(&mut self) {
        self.0.close();
    }

    fn cancel(&mut self) {
        self.0.close();
    }

    fn hand_over(&mut self) {}

    fn engine_request_id(&self) -> Option<proto::request::RequestId> {
        None
    }

    fn progress(&self) -> (u64, u64) {
        (0, 0)
    }

    fn submitted(&self) -> Option<Vec<u8>> {
        None
    }
}

/// A request whose statistics no one reads, which entered the server at the
/// instant it holds.
pub struct Unrecorded(pub std::time::Instant);

impl crate::events::Outcome for Unrecorded {
    fn started(&self) -> std::time::Instant {
        self.0
    }

    fn engine(&self) -> u32 {
        0
    }

    fn observe(&mut self, _: &crate::output::RequestOutput) {}

    fn abandoned(&mut self) {}
}

/// Wire chat messages and requests for the tests that render or lower chat:
/// each message as a client sends it.
pub mod chat {
    use crate::input::chat::{ChatConfig, ChatRequest, ChatToolChoice};
    use crate::openai::{
        ChatMessage, ContentPart, FunctionCallResponse, ImageUrl, MessageContent, ToolCall,
    };
    use crate::output::chat::AssistantContentBlock;
    use crate::output::tools::Tool;

    /// A request of `messages` with default render options and no tools.
    pub fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            messages,
            chat_options: ChatConfig::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
        }
    }

    /// A system message of `text`.
    pub fn system(text: impl Into<String>) -> ChatMessage {
        ChatMessage::System {
            content: MessageContent::Text(text.into()),
            name: None,
        }
    }

    /// A user message of `text`.
    pub fn user(text: impl Into<String>) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Text(text.into()),
            name: None,
        }
    }

    /// A user message of content `parts`.
    pub fn user_parts(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Parts(parts),
            name: None,
        }
    }

    /// A text content part.
    pub fn text_part(text: impl Into<String>) -> ContentPart {
        ContentPart::Text { text: text.into() }
    }

    /// An image content part of `url`.
    pub fn image_part(url: impl Into<String>) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
            uuid: None,
        }
    }

    /// A developer message of `text`, declaring `tools`.
    pub fn developer(text: impl Into<String>, tools: Option<Vec<Tool>>) -> ChatMessage {
        ChatMessage::Developer {
            content: MessageContent::Text(text.into()),
            tools: tools.map(|tools| {
                tools
                    .into_iter()
                    .map(|tool| crate::openai::Tool::Function {
                        function: crate::openai::Function {
                            name: tool.name,
                            description: tool.description,
                            parameters: tool.parameters,
                        },
                    })
                    .collect()
            }),
            name: None,
        }
    }

    /// An assistant message of `text`.
    pub fn assistant(text: impl Into<String>) -> ChatMessage {
        assistant_blocks(vec![AssistantContentBlock::Text { text: text.into() }])
    }

    /// The assistant message a client sends back for a turn whose output
    /// was `blocks`: its reasoning, its text and its tool calls.
    pub fn assistant_blocks(blocks: Vec<AssistantContentBlock>) -> ChatMessage {
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for block in blocks {
            match block {
                AssistantContentBlock::Reasoning { text: part } => reasoning.push_str(&part),
                AssistantContentBlock::Text { text: part } => text.push_str(&part),
                AssistantContentBlock::ToolCall(call) => tool_calls.push(ToolCall {
                    id: call.id,
                    tool_type: "function".to_owned(),
                    function: FunctionCallResponse {
                        name: call.name,
                        arguments: Some(call.arguments),
                    },
                }),
            }
        }
        ChatMessage::Assistant {
            content: (!text.is_empty()).then_some(MessageContent::Text(text)),
            name: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
        }
    }

    /// A tool's response of `content` to the call `tool_call_id`.
    pub fn tool_response(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> ChatMessage {
        ChatMessage::Tool {
            content: MessageContent::Text(content.into()),
            tool_call_id: tool_call_id.into(),
        }
    }
}
