//! The Qwen chat families, as request processing serves them.
//!
//! The daemon serves a family by what is stated here and links no model: how the family's
//! checkpoints declare it, the output formats they write and, for a family that takes images,
//! the processor that sizes their blocks ([`crate::qwen_vl`]). The model crate (`qwen3`) holds
//! the numerical side of the families it loads and refers to these values.
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
//! [`QWEN4_EXP`] states the formats of a family no model crate loads yet, so its image side is
//! absent until one does.

use crate::family::Family;
use crate::image::{ImageFamily, ImageProcessor};
use crate::qwen_vl::ProcessorConfig;
use crate::serving::{ChatProfile, ReasoningFormat, Serving, ToolCallFormat};

/// The model class of a Qwen3-MoE checkpoint.
pub const QWEN3_MOE_CLASS: &str = "Qwen3MoeForCausalLM";

/// The model class of a Qwen3.6 checkpoint.
pub const QWEN3_5_MOE_CLASS: &str = "Qwen3_5MoeForConditionalGeneration";

/// Qwen 2.5 and Qwen3 (`Qwen3MoeForCausalLM`): a text chat family with JSON tool calls.
pub const QWEN3: Family = Family {
    id: "qwen3",
    model_types: &["qwen3", "qwen3_moe"],
    classes: &[QWEN3_MOE_CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Think),
        tool_calls: Some(ToolCallFormat::Json),
        images: None,
    }),
};

/// Qwen3.6 (`Qwen3_5MoeForConditionalGeneration`): a chat family that takes images and writes
/// tool calls as function and parameter tags.
pub const QWEN3_5_MOE: Family = Family {
    id: "qwen3_5_moe",
    model_types: &["qwen3_5_moe"],
    classes: &[QWEN3_5_MOE_CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Think),
        tool_calls: Some(ToolCallFormat::Xml),
        images: Some(&Images),
    }),
};

/// Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`): `<think>` reasoning and function and
/// parameter tags for tool calls.
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

/// How request processing binds the image processor of a Qwen checkpoint whose vision tower
/// serves: the Qwen-VL processor rules, read from the checkpoint's two processor files.
struct Images;

impl ImageFamily for Images {
    fn processor_files(&self) -> &'static [&'static str] {
        &ProcessorConfig::FILES
    }

    fn bind(&self, files: &[String]) -> Result<Box<dyn ImageProcessor>, String> {
        let [image, video] = files else {
            return Err(format!("{} processor files, two expected", files.len()));
        };

        let processor = ProcessorConfig::parse(image, video)?;
        Ok(Box::new(processor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Registry;

    /// The chat profile a Qwen family serves.
    fn profile(family: Family) -> ChatProfile {
        let Serving::Chat(profile) = family.serving else {
            panic!("{} serves chat", family.id);
        };
        profile
    }

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
        assert_eq!(profile(QWEN4_EXP).reasoning, Some(ReasoningFormat::Think));
        assert_eq!(profile(QWEN4_EXP).tool_calls, Some(ToolCallFormat::Xml));
    }

    /// The generations differ in their tool-call format and share their reasoning format, and
    /// only the generation with a vision tower takes images.
    #[test]
    fn the_generations_declare_their_tool_call_formats() {
        assert_eq!(profile(QWEN3).tool_calls, Some(ToolCallFormat::Json));
        assert_eq!(profile(QWEN3_5_MOE).tool_calls, Some(ToolCallFormat::Xml));
        assert_eq!(profile(QWEN3).reasoning, profile(QWEN3_5_MOE).reasoning);
        assert_eq!(ReasoningFormat::Think.delimiters(), ("<think>", "</think>"));

        assert!(profile(QWEN3).images.is_none());
        assert!(profile(QWEN3_5_MOE).images.is_some());
    }
}
