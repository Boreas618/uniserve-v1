//! The Qwen3 chat families, as request processing serves them.
//!
//! The daemon serves a family by what is stated here and links no model: how
//! the family's checkpoints declare it, the output formats they write, and,
//! for a family that takes images, the processor that sizes their blocks
//! ([`crate::qwen_vl`]). The model crate (`qwen3`) holds the numerical side of
//! the same families and refers to these values.

use crate::family::Family;
use crate::image::{ImageFamily, ImageProcessor};
use crate::qwen_vl::ProcessorConfig;
use crate::serving::{ChatProfile, ReasoningFormat, Serving, ToolCallFormat};

/// The model class of a Qwen3-MoE checkpoint.
pub const MOE_CLASS: &str = "Qwen3MoeForCausalLM";

/// The model class of a Qwen3.5-MoE checkpoint.
pub const HYBRID_CLASS: &str = "Qwen3_5MoeForConditionalGeneration";

/// `Qwen3MoeForCausalLM`: a text chat family.
pub const MOE: Family = Family {
    id: "qwen3",
    model_types: &["qwen3", "qwen3_moe"],
    classes: &[MOE_CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Qwen3),
        tool_calls: Some(ToolCallFormat::Qwen3),
        images: None,
    }),
};

/// `Qwen3_5MoeForConditionalGeneration`: a chat family that takes images.
pub const HYBRID: Family = Family {
    id: "qwen3_5_moe",
    model_types: &["qwen3_5_moe"],
    classes: &[HYBRID_CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(ReasoningFormat::Qwen3),
        tool_calls: None,
        images: Some(&Images),
    }),
};

/// How request processing binds the image processor of a Qwen3 checkpoint
/// whose vision tower serves: the Qwen-VL processor rules, read from the
/// checkpoint's two processor files.
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
