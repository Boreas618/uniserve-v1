//! Qwen3 as request processing serves it ([`processing`]).
//!
//! A checkpoint whose root `config.json` declares a Qwen3 `model_type` serves chat. Its outputs
//! delimit reasoning between `<think>` and `</think>` and write tool calls as JSON between
//! `<tool_call>` and `</tool_call>`, when its chat template asks for them.

use processing::{ChatProfile, Family, ReasoningFormat, Serving, ToolCallFormat};

/// Qwen3 as request processing serves it.
pub const FAMILY: Family = Family {
    id: "qwen3",
    model_types: &["qwen3", "qwen3_moe"],
    pipeline_classes: &[],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Qwen3),
        tool_calls: Some(ToolCallFormat::Qwen3),
    }),
};
