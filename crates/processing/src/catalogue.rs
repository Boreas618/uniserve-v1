//! The model families request processing serves, and the family a checkpoint resolves to.

use crate::video::VideoFamily;

/// One model family as request processing serves it.
#[derive(Clone, Copy)]
pub struct Family {
    /// The family's stable identifier, as the daemon reports it.
    pub id: &'static str,
    /// The `model_type` values of a root `config.json` that declare this family; empty for a
    /// family that ships only as a diffusers pipeline.
    pub model_types: &'static [&'static str],
    /// The `_class_name` values of a diffusers pipeline index that declare this family.
    pub pipeline_classes: &'static [&'static str],
    /// What the family serves.
    pub serving: Serving,
}

impl std::fmt::Debug for Family {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Family")
            .field("id", &self.id)
            .field("model_types", &self.model_types)
            .field("pipeline_classes", &self.pipeline_classes)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Family {
    /// Families are one value each: equal when they are the same family.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Family {}

/// What a family serves.
#[derive(Clone, Copy)]
pub enum Serving {
    /// Chat and completions over a causal language model.
    Chat(ChatProfile),
    /// Video generation over a diffusion pipeline.
    Video(&'static dyn VideoFamily),
}

/// The output formats a chat family's checkpoints write.
///
/// A format is a shared parser keyed by its name, as other serving systems key theirs; several
/// families may write one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatProfile {
    /// How the model delimits its reasoning, when it reasons.
    pub reasoning: Option<ReasoningFormat>,
    /// How the model writes tool calls, when its chat template asks for them.
    pub tool_calls: Option<ToolCallFormat>,
}

/// A reasoning output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// Reasoning between `<think>` and `</think>`, as Qwen3 writes it.
    Qwen3,
}

/// A tool call output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// One JSON object per call between `<tool_call>` and `</tool_call>`, as Qwen3 writes it.
    Qwen3,
}

/// The families an executable serves.
#[derive(Clone, Debug, Default)]
pub struct Catalogue {
    families: Vec<Family>,
}

impl Catalogue {
    /// A catalogue of `families`.
    ///
    /// # Errors
    ///
    /// Fails, naming the class, when two families claim one `model_type` or pipeline class.
    pub fn new(families: impl IntoIterator<Item = Family>) -> Result<Self, String> {
        let families: Vec<Family> = families.into_iter().collect();
        for (index, family) in families.iter().enumerate() {
            let earlier = &families[..index];
            for class in family.model_types.iter().chain(family.pipeline_classes) {
                if earlier.iter().any(|other| {
                    other.model_types.contains(class) || other.pipeline_classes.contains(class)
                }) {
                    return Err(format!("two model families claim the class {class}"));
                }
            }
        }
        Ok(Self { families })
    }

    /// The family a root `config.json` declares by its `model_type`.
    pub fn by_model_type(&self, model_type: &str) -> Option<Family> {
        self.families
            .iter()
            .copied()
            .find(|family| family.model_types.contains(&model_type))
    }

    /// The family a diffusers pipeline index declares by its `_class_name`.
    pub fn by_pipeline_class(&self, class_name: &str) -> Option<Family> {
        self.families
            .iter()
            .copied()
            .find(|family| family.pipeline_classes.contains(&class_name))
    }

    /// Every family, in registration order.
    pub fn families(&self) -> &[Family] {
        &self.families
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT: Family = Family {
        id: "chat",
        model_types: &["chat_model"],
        pipeline_classes: &[],
        serving: Serving::Chat(ChatProfile {
            reasoning: None,
            tool_calls: None,
        }),
    };

    /// A checkpoint resolves by the class its own file declares, and a class no family claims
    /// resolves to none.
    #[test]
    fn a_family_is_found_by_the_class_it_claims() {
        let catalogue = Catalogue::new([CHAT]).unwrap();
        assert_eq!(catalogue.by_model_type("chat_model"), Some(CHAT));
        assert_eq!(catalogue.by_model_type("other"), None);
        assert_eq!(catalogue.by_pipeline_class("chat_model"), None);
    }

    /// Two families that claim one class are refused.
    #[test]
    fn two_families_may_not_claim_one_class() {
        let twin = Family { id: "twin", ..CHAT };
        let error = Catalogue::new([CHAT, twin]).unwrap_err();
        assert!(error.contains("chat_model"), "{error}");
    }
}
