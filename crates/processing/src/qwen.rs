//! Chat profiles of the Qwen families: the output formats their checkpoints write.
//!
//! Every Qwen chat model since Qwen3 delimits its reasoning with `<think>` and `</think>`.
//! Their tool calls differ by generation, as each generation's chat template instructs them:
//!
//! | Family | `model_type` | Tool calls |
//! | --- | --- | --- |
//! | Qwen 2.5, Qwen3 | `qwen3`, `qwen3_moe` | [`ToolCallFormat::Json`] |
//! | Qwen3.6 | `qwen3_5_moe` | [`ToolCallFormat::Xml`] |
//! | Qwen3.8 Flash Next | `qwen4_exp` | [`ToolCallFormat::Xml`] |
//!
//! A family whose model crate serves it registers its own [`Family`] with the profile here and
//! the image side its tower provides. [`QWEN4_EXP`] is stated here whole, because request
//! processing serves its formats before a model crate registers the family for execution.

use crate::family::Family;
use crate::serving::{ChatProfile, ReasoningFormat, Serving, ToolCallFormat};

/// The profile of Qwen 2.5 and Qwen3 checkpoints: JSON tool calls.
pub const QWEN3: ChatProfile = ChatProfile {
    reasoning: Some(ReasoningFormat::Think),
    tool_calls: Some(ToolCallFormat::Json),
    images: None,
};

/// The profile of Qwen3.6 checkpoints (`Qwen3_5MoeForConditionalGeneration`): function and
/// parameter tags. The model crate adds the image side.
pub const QWEN3_5_MOE: ChatProfile = ChatProfile {
    reasoning: Some(ReasoningFormat::Think),
    tool_calls: Some(ToolCallFormat::Xml),
    images: None,
};

/// Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`), as request processing serves it:
/// `<think>` reasoning and function and parameter tags for tool calls. Its registration for
/// execution adds the image side its model crate provides.
pub const QWEN4_EXP: Family = Family {
    id: "qwen4_exp",
    model_types: &["qwen4_exp"],
    classes: &["Qwen4ExpForConditionalGeneration"],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Think),
        tool_calls: Some(ToolCallFormat::Xml),
        images: None,
    }),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Registry;

    /// A Qwen3.8 Flash Next checkpoint resolves to its family by the `model_type` or the class
    /// its `config.json` declares, and the family's profile names the formats its output is
    /// parsed in.
    #[test]
    fn a_qwen4_exp_checkpoint_resolves_to_its_profile() {
        let registry = Registry::new([QWEN4_EXP]).unwrap();

        assert_eq!(registry.by_model_type("qwen4_exp"), Some(QWEN4_EXP));
        assert_eq!(
            registry.by_class("Qwen4ExpForConditionalGeneration"),
            Some(QWEN4_EXP)
        );
        let Serving::Chat(profile) = QWEN4_EXP.serving else {
            panic!("Qwen3.8 Flash Next serves chat");
        };
        assert_eq!(profile.reasoning, Some(ReasoningFormat::Think));
        assert_eq!(profile.tool_calls, Some(ToolCallFormat::Xml));
    }

    /// The generations differ in their tool-call format and share their reasoning format.
    #[test]
    fn the_generations_declare_their_tool_call_formats() {
        assert_eq!(QWEN3.tool_calls, Some(ToolCallFormat::Json));
        assert_eq!(QWEN3_5_MOE.tool_calls, Some(ToolCallFormat::Xml));
        assert_eq!(QWEN3.reasoning, QWEN3_5_MOE.reasoning);
        assert_eq!(ReasoningFormat::Think.delimiters(), ("<think>", "</think>"));
    }
}
