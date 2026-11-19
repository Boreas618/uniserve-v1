//! A processing program of the daemon tests' families, served by the processing crate's own
//! machinery ([`processing::program::Program`]):
//!
//! - [`CHAT`] (`ChatForCausalLM` in `config.json`): `<think>` reasoning and JSON tool calls,
//!   its template in `tokenizer_config.json`, its context length at `max_position_embeddings`;
//! - [`PLAIN`] (`PlainForCausalLM`): a chat family that declares neither format;
//! - [`TAGS`] (`TagsForCausalLM`): tool calls as function and parameter tags, no reasoning;
//! - [`VISION`] (`VisionForCausalLM`): [`CHAT`]'s formats, taking images as blocks of one merged
//!   token per 32-pixel square ([`TestImages`]);
//! - [`VIDEO`] (`TestVideoPipeline` in `modular_model_index.json`): a video family of one rule
//!   and plain inputs ([`TestProcessor`]).
//!
//! The daemon's tests build the crate for wasm32 and load its module as an epoch's processing
//! program, as they would the release's.

use processing::program::{Families, Program};
use processing::{
    BasePin, ChatCheckpoint, ChatProfile, ChatTemplate, ConditionInput, Family, ImageBlock,
    ImageFamily, ImageProcessor, InputError, Length, MediaMetadata, ReasoningFormat, RootFile,
    Serving, ToolCallFormat, VideoBinding, VideoFamily, VideoInput, VideoProcessor, VideoSpec,
    Vocabulary,
};
use scheduling::processing::video::{Canvas, ConditionMedia, ConditionRole, ImageFit};

/// A chat family whose outputs are parsed as `<think>` reasoning and JSON tool calls.
pub const CHAT: Family = Family {
    id: "chat",
    root: RootFile::Config,
    classes: &["ChatForCausalLM"],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat {
            open: "<think>",
            close: "</think>",
        }),
        tool_calls: Some(ToolCallFormat::Json),
        images: None,
        checkpoint: ChatCheckpoint {
            context_length: &["max_position_embeddings"],
            chat_template: ChatTemplate::TokenizerConfig,
        },
    }),
};

/// A chat family of plain text: no reasoning, no tool calls.
pub const PLAIN: Family = Family {
    id: "plain",
    root: RootFile::Config,
    classes: &["PlainForCausalLM"],
    serving: Serving::Chat(ChatProfile {
        reasoning: None,
        tool_calls: None,
        images: None,
        checkpoint: ChatCheckpoint {
            context_length: &["max_position_embeddings"],
            chat_template: ChatTemplate::TokenizerConfig,
        },
    }),
};

/// A chat family that writes tool calls as function and parameter tags and delimits no
/// reasoning.
pub const TAGS: Family = Family {
    id: "tags",
    root: RootFile::Config,
    classes: &["TagsForCausalLM"],
    serving: Serving::Chat(ChatProfile {
        reasoning: None,
        tool_calls: Some(ToolCallFormat::Xml),
        ..chat_profile()
    }),
};

/// A chat family of [`CHAT`]'s formats that takes images through [`TestImages`].
pub const VISION: Family = Family {
    id: "vision",
    root: RootFile::Config,
    classes: &["VisionForCausalLM"],
    serving: Serving::Chat(ChatProfile {
        images: Some(&TestImages),
        ..chat_profile()
    }),
};

/// [`CHAT`]'s profile.
const fn chat_profile() -> ChatProfile {
    let Serving::Chat(profile) = CHAT.serving else {
        panic!("the chat family serves chat");
    };
    profile
}

/// A video family whose processor is [`TestProcessor`].
pub const VIDEO: Family = Family {
    id: "video",
    root: RootFile::ModularIndex,
    classes: &["TestVideoPipeline"],
    serving: Serving::Video(&TestVideo),
};

/// The families of the tests' program.
pub struct Tested;

impl Families for Tested {
    fn families() -> &'static [Family] {
        &[CHAT, PLAIN, TAGS, VISION, VIDEO]
    }
}

scheduling::export_processing!(Program<Tested>);

/// The image side of [`VISION`]: an image is a block of one merged token per 32-pixel square,
/// spanning the larger side's tokens, as a Qwen-VL processor states a block, which replaces the
/// `<|image_pad|>` its template writes between `<|vision_start|>` and `<|vision_end|>`.
pub struct TestImages;

impl ImageFamily for TestImages {
    fn processor_files(&self) -> &'static [&'static str] {
        &[]
    }

    fn bind(
        &self,
        _: &[String],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Box<dyn ImageProcessor>, String> {
        match options.keys().next() {
            Some(name) => Err(format!("the test images take no option {name:?}")),
            None => Ok(Box::new(TestImages)),
        }
    }
}

impl ImageProcessor for TestImages {
    fn placeholder(&self) -> &str {
        "<|image_pad|>"
    }

    fn markup(&self) -> &str {
        "<|vision_start|><|image_pad|><|vision_end|>"
    }

    fn block(&self, width: u32, height: u32) -> Result<ImageBlock, String> {
        Ok(ImageBlock {
            tokens: (width / 32) * (height / 32),
            positions: (width / 32).max(height / 32),
        })
    }
}

/// The video family [`VIDEO`] serves.
struct TestVideo;

impl VideoFamily for TestVideo {
    fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn default_sizes(&self) -> &'static [&'static str] {
        &["768x768"]
    }

    fn default_max_seconds(&self) -> f64 {
        4.0
    }

    fn processing_parts(&self) -> &'static [&'static str] {
        // The part whose files decide how its requests are processed, which an update of a
        // pipeline's weights compares.
        &["processor"]
    }

    fn base_pin(&self) -> Option<BasePin> {
        None
    }

    fn bind(&self, _: VideoBinding<'_>) -> Result<Box<dyn VideoProcessor>, String> {
        Ok(Box::new(TestProcessor))
    }
}

/// A video model of one rule and plain inputs, for request processing's own behavior: a request
/// shorter than a second is refused before its media is fetched, and every image reference is
/// one condition whose rows are its height.
pub struct TestProcessor;

impl VideoProcessor for TestProcessor {
    fn capabilities(&self) -> serde_json::Value {
        serde_json::json!({"sizes": ["64x64"]})
    }

    fn denoise_steps(&self) -> u32 {
        4
    }

    fn check(&self, request: &VideoSpec) -> Result<(), InputError> {
        match request.length {
            Some(Length::Seconds(seconds)) if seconds < 1.0 => Err(InputError::invalid(
                "seconds",
                "a video lasts at least a second",
            )),
            _ => Ok(()),
        }
    }

    fn process(
        &self,
        request: &VideoSpec,
        media: &[MediaMetadata],
        vocabulary: &dyn Vocabulary,
    ) -> Result<VideoInput, InputError> {
        let conditions = request
            .references
            .iter()
            .zip(media)
            .map(|(reference, media)| {
                let MediaMetadata::Image(image) = media else {
                    return Err(InputError::Internal(
                        "the test processor processes images".to_owned(),
                    ));
                };
                let canvas = Canvas {
                    width: image.width,
                    height: image.height,
                };
                Ok(ConditionInput {
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
            .map_err(|error| InputError::Internal(error.to_string()))?;
        Ok(VideoInput {
            size: request.size.unwrap_or(Canvas {
                width: 64,
                height: 64,
            }),
            num_frames: 22,
            fps: 24,
            seed: request.seed,
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
