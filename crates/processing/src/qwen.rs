//! The Qwen chat families, as request processing serves them.
//!
//! The processing program serves a family by what is stated here: how the family's checkpoints
//! declare it, the output formats they write and, for a family that takes images, the processor
//! that sizes their blocks ([`crate::qwen_vl`]). The model crate (`qwen3`) holds
//! the numerical side of the families it loads and refers to these values.
//!
//! Every Qwen chat model since Qwen3 delimits its reasoning with `<think>` and `</think>`.
//! Their tool calls differ by generation, as each generation's chat template instructs them. Each
//! family claims the model class its checkpoints' `config.json` names:
//!
//! | Family | Model class | Tool calls |
//! | --- | --- | --- |
//! | Qwen3-MoE | `Qwen3MoeForCausalLM` | [`ToolCallFormat::Json`] |
//! | Qwen3.6 | `Qwen3_5MoeForConditionalGeneration` | [`ToolCallFormat::Xml`] |
//! | Qwen3.8 Flash Next | `Qwen4ExpForConditionalGeneration` | [`ToolCallFormat::Xml`] |
//!
//! Qwen3.6 and Qwen3.8 Flash Next hold the Qwen3-VL vision tower and take images, sized by the
//! Qwen-VL processor rules their checkpoints state.

use crate::family::{Family, RootFile};
use crate::image::{ImageFamily, ImageProcessor};
use crate::qwen_vl::ProcessorConfig;
use crate::serving::{
    ChatCheckpoint, ChatProfile, ChatTemplate, ReasoningFormat, Serving, ToolCallFormat,
};

/// How every Qwen chat model since Qwen3 delimits its reasoning.
const THINK: ReasoningFormat = ReasoningFormat {
    open: "<think>",
    close: "</think>",
};

/// The model class of a Qwen3-MoE checkpoint.
pub const QWEN3_MOE_CLASS: &str = "Qwen3MoeForCausalLM";

/// The model class of a Qwen3.6 checkpoint.
pub const QWEN3_5_MOE_CLASS: &str = "Qwen3_5MoeForConditionalGeneration";

/// Qwen3-MoE (`Qwen3MoeForCausalLM`): a text chat family with JSON tool calls.
pub const QWEN3: Family = Family {
    id: "qwen3",
    root: RootFile::Config,
    classes: &[QWEN3_MOE_CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(THINK),
        tool_calls: Some(ToolCallFormat::Json),
        images: None,
        // A Qwen3 checkpoint states its context bound at the root of its configuration. Qwen's
        // releases keep the template in their tokenizer configuration, and exports saved since
        // `transformers` writes templates as files keep it in `chat_template.jinja` alone, as
        // NVIDIA's ModelOpt NVFP4 export of Qwen3-30B-A3B does.
        checkpoint: ChatCheckpoint {
            context_length: &["max_position_embeddings"],
            chat_template: ChatTemplate::FileOrTokenizerConfig("chat_template.jinja"),
        },
    }),
};

/// Qwen3.6 (`Qwen3_5MoeForConditionalGeneration`): a chat family that takes images and writes
/// tool calls as function and parameter tags.
pub const QWEN3_5_MOE: Family = Family {
    id: "qwen3_5_moe",
    root: RootFile::Config,
    classes: &[QWEN3_5_MOE_CLASS],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(THINK),
        tool_calls: Some(ToolCallFormat::Xml),
        images: Some(&Images),
        checkpoint: VISION_CHAT_CHECKPOINT,
    }),
};

/// Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`): a chat family that takes images,
/// with `<think>` reasoning and function and parameter tags for tool calls.
pub const QWEN4_EXP: Family = Family {
    id: "qwen4_exp",
    root: RootFile::Config,
    classes: &["Qwen4ExpForConditionalGeneration"],
    serving: Serving::Chat(ChatProfile {
        reasoning: Some(THINK),
        tool_calls: Some(ToolCallFormat::Xml),
        images: Some(&Images),
        checkpoint: VISION_CHAT_CHECKPOINT,
    }),
};

/// Where a Qwen checkpoint with a vision tower states what request processing reads: its language
/// model is the `text_config` section of its configuration, and its template a file of its own,
/// which every checkpoint of these families publishes (some also copy it into their tokenizer
/// configuration), as they were all saved since `transformers` writes templates as files.
const VISION_CHAT_CHECKPOINT: ChatCheckpoint = ChatCheckpoint {
    context_length: &["text_config", "max_position_embeddings"],
    chat_template: ChatTemplate::File("chat_template.jinja"),
};

/// How request processing binds the image processor of a Qwen checkpoint whose vision tower
/// serves: the Qwen-VL processor rules, read from the checkpoint's two processor files.
struct Images;

impl ImageFamily for Images {
    fn processor_files(&self) -> &'static [&'static str] {
        &ProcessorConfig::FILES
    }

    fn bind(
        &self,
        files: &[String],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Box<dyn ImageProcessor>, String> {
        // The Qwen-VL processor rules read every value from the checkpoint's processor files.
        if let Some(name) = options.keys().next() {
            return Err(format!(
                "the Qwen-VL image processor takes no option {name:?}"
            ));
        }
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

    /// A Qwen3.8 Flash Next checkpoint resolves to its family by the class its `config.json`
    /// declares, and the family's profile names the formats its output is parsed in.
    #[test]
    fn a_qwen4_exp_checkpoint_resolves_to_its_profile() {
        let registry = Registry::new([QWEN4_EXP]).unwrap();

        let config = r#"{"architectures": ["Qwen4ExpForConditionalGeneration"]}"#;
        assert_eq!(
            registry.resolve([(crate::RootFile::Config, config)]),
            Ok(QWEN4_EXP)
        );
        assert_eq!(profile(QWEN4_EXP).reasoning, Some(THINK));
        assert_eq!(profile(QWEN4_EXP).tool_calls, Some(ToolCallFormat::Xml));
    }

    /// The generations differ in their tool-call format and share their reasoning format, and
    /// only the generation with a vision tower takes images.
    #[test]
    fn the_generations_declare_their_tool_call_formats() {
        assert_eq!(profile(QWEN3).tool_calls, Some(ToolCallFormat::Json));
        assert_eq!(profile(QWEN3_5_MOE).tool_calls, Some(ToolCallFormat::Xml));
        assert_eq!(profile(QWEN3).reasoning, profile(QWEN3_5_MOE).reasoning);
        assert_eq!((THINK.open, THINK.close), ("<think>", "</think>"));

        assert!(profile(QWEN3).images.is_none());
        assert!(profile(QWEN3_5_MOE).images.is_some());
        assert!(profile(QWEN4_EXP).images.is_some());
    }

    /// The Qwen-VL image processor reads every value from the checkpoint's processor files, so it
    /// binds without options and refuses any, naming it.
    #[test]
    fn the_qwen_vl_image_processor_takes_no_option() {
        let images = profile(QWEN3_5_MOE)
            .images
            .expect("a family that takes images");
        let file = r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "size": {"shortest_edge": 4096, "longest_edge": 16777216}}"#;
        let files = [file.to_owned(), file.to_owned()];

        assert!(images.bind(&files, &serde_json::Map::new()).is_ok());
        let options = serde_json::Map::from_iter([("max_pixels".to_owned(), 1.into())]);
        let error = images
            .bind(&files, &options)
            .err()
            .expect("an option is refused");
        assert!(error.contains("max_pixels"), "{error}");
    }
}
