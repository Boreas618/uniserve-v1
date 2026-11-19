//! The DeepSeek-V4.1 chat family (`DeepseekV41ForCausalLM`), as request processing serves it.
//!
//! A DeepSeek-V4.1 checkpoint publishes no chat template: its `encoding/encoding.py` is the
//! reference of its prompt format, which [`encoding`] ports, and [`request`] reads a chat request
//! into it as vLLM's `deepseek_v41` tokenizer mode does. Its language model is the `text_config`
//! section of its configuration, which states its context length.
//!
//! The model reasons between `<think>` and `</think>`, each one token of its vocabulary. In
//! thinking mode, the default, a request's prompt ends with `<think>`, so its response starts in
//! reasoning; in chat mode it ends with `</think>`. The model writes its tool calls as one DSML
//! `calls` block at the end of its turn ([`ToolCallFormat::Dsml`]), which also ends reasoning that
//! the model did not close.
//!
//! The checkpoint holds a vision tower, which this family does not serve: a request's image parts
//! are refused.

pub mod encoding;
mod request;

pub use request::Encoder;

use crate::family::{Family, RootFile};
use crate::serving::{
    ChatCheckpoint, ChatProfile, ChatTemplate, ReasoningFormat, Serving, ToolCallFormat,
};

/// The model class of a DeepSeek-V4.1 checkpoint.
pub const CLASS: &str = "DeepseekV41ForCausalLM";

/// DeepSeek-V4.1 (`DeepseekV41ForCausalLM`): a text chat family with `<think>` reasoning and DSML
/// tool calls, whose prompts the family encodes.
pub const FAMILY: Family = Family {
    id: "deepseek_v41",
    root: RootFile::Config,
    classes: &[CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat {
            open: encoding::THINK_START,
            close: encoding::THINK_END,
        }),
        tool_calls: Some(ToolCallFormat::Dsml),
        images: None,
        checkpoint: ChatCheckpoint {
            context_length: &["text_config", "max_position_embeddings"],
            chat_template: ChatTemplate::Encoder(&Encoder),
        },
    }),
};

/// The vectors the encoder is tested against (`vectors.json`, whose `sources` state how they were
/// generated): the checkpoint's own encoding cases, and requests encoded by vLLM's
/// `deepseek_v41` tokenizer mode.
#[cfg(test)]
pub(crate) fn vectors() -> serde_json::Value {
    serde_json::from_str(include_str!("vectors.json")).expect("the encoding vectors are JSON")
}

#[cfg(test)]
mod tests;
