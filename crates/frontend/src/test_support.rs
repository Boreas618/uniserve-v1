//! Fixtures shared by the crate's tests.
//!
//! `configured_tokenizer` builds the deterministic byte tokenizer the serving
//! tests decode with: every printable ASCII character is one token whose ID is
//! its code point, and the chat, vision and thinking special tokens encode
//! atomically. [`CHAT`] and [`VIDEO`] are the families those tests serve, and
//! [`TestProcessor`] plans the video family's requests. It is compiled for the
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
    fn default_prompt_tokens(&self) -> u32 {
        4096
    }

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

/// A video model of one rule and plain plans, for request processing's own behavior: a request shorter
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

    fn plan(
        &self,
        request: &processing::VideoSpec,
        media: &[processing::MediaMetadata],
        vocabulary: &dyn processing::Vocabulary,
    ) -> Result<processing::VideoPlan, processing::InputError> {
        use proto::video::{Canvas, ConditionMedia, ConditionRole, ImageFit};

        let conditions = request
            .references
            .iter()
            .zip(media)
            .map(|(reference, media)| {
                let processing::MediaMetadata::Image(image) = media else {
                    return Err(processing::InputError::Internal(
                        "the test processor plans images".to_owned(),
                    ));
                };
                let canvas = Canvas {
                    width: image.width,
                    height: image.height,
                };
                Ok(processing::ConditionPlan {
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
        Ok(processing::VideoPlan {
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
