//! The families a build serves, as request processing resolves and serves them.
//!
//! A model crate states one [`Family`] beside the numerical model execution loads
//! (`nn::family::Family` holds it), and the executable registers each once: request processing
//! reads the [`Catalogue`] of these values, execution the catalogue of numerical families built
//! from the same list. A checkpoint resolves by what its own files declare: the `model_type` of a
//! root `config.json`, or its model class.

use crate::serving::Serving;

/// One model family, as its checkpoints declare it and request processing serves it.
#[derive(Clone, Copy)]
pub struct Family {
    /// The family's stable identifier, as the daemon reports it.
    pub id: &'static str,
    /// The `model_type` values of a root `config.json` that declare this family; empty for a
    /// family that ships only as a diffusers pipeline.
    pub model_types: &'static [&'static str],
    /// The model classes a checkpoint of this family declares: a transformers checkpoint's
    /// `config.json` `architectures[0]`, or a diffusers pipeline's `_class_name`.
    pub classes: &'static [&'static str],
    /// What request processing serves.
    pub serving: Serving,
}

impl std::fmt::Debug for Family {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Family")
            .field("id", &self.id)
            .field("model_types", &self.model_types)
            .field("classes", &self.classes)
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

/// The request-processing side of every model family this tree serves: what a daemon registers,
/// without linking a model. The engine executable registers the numerical side of the same
/// families, each of which holds its value of this list.
pub const FAMILIES: [Family; 3] = [
    crate::qwen3::MOE,
    crate::qwen3::HYBRID,
    crate::minimax_h3::FAMILY,
];

/// Two families claim one `model_type` or class: a checkpoint of it would resolve to either.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("two model families claim the class {0}")]
pub struct ClaimError(pub &'static str);

/// The families an executable serves, each claiming the `model_type`s and classes its checkpoints
/// declare.
#[derive(Clone, Debug, Default)]
pub struct Catalogue {
    families: Vec<Family>,
}

impl Catalogue {
    /// A catalogue of `families`.
    ///
    /// # Errors
    ///
    /// Fails, naming the class, when two families claim one `model_type` or class.
    pub fn new(families: impl IntoIterator<Item = Family>) -> Result<Self, ClaimError> {
        let families: Vec<Family> = families.into_iter().collect();
        for (index, family) in families.iter().enumerate() {
            for class in family.model_types.iter().chain(family.classes) {
                if families[..index].iter().any(|earlier| {
                    earlier.model_types.contains(class) || earlier.classes.contains(class)
                }) {
                    return Err(ClaimError(class));
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

    /// The family a checkpoint declares by its class: a transformers checkpoint's
    /// `architectures[0]`, or a diffusers pipeline's `_class_name`.
    pub fn by_class(&self, class: &str) -> Option<Family> {
        self.families
            .iter()
            .copied()
            .find(|family| family.classes.contains(&class))
    }

    /// Every family, in registration order.
    pub fn families(&self) -> &[Family] {
        &self.families
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::ChatProfile;

    const CHAT: Family = Family {
        id: "chat",
        model_types: &["chat_model"],
        classes: &["ChatForCausalLM"],
        serving: Serving::Chat(ChatProfile {
            reasoning: None,
            tool_calls: None,
            images: None,
        }),
    };

    /// A checkpoint resolves by the `model_type` or the class its own file declares, and one no
    /// family claims resolves to none.
    #[test]
    fn a_family_is_found_by_what_its_checkpoints_declare() {
        let catalogue = Catalogue::new([CHAT]).unwrap();
        assert_eq!(catalogue.by_model_type("chat_model"), Some(CHAT));
        assert_eq!(catalogue.by_class("ChatForCausalLM"), Some(CHAT));
        assert_eq!(catalogue.by_model_type("other"), None);
        assert_eq!(catalogue.by_class("chat_model"), None);
    }

    /// Two families that claim one `model_type` or class are refused: a checkpoint of it would
    /// resolve to either.
    #[test]
    fn two_families_may_not_claim_one_class() {
        let twin = Family { id: "twin", ..CHAT };
        assert_eq!(
            Catalogue::new([CHAT, twin]).unwrap_err(),
            ClaimError("chat_model")
        );

        let class = Family {
            id: "class",
            model_types: &[],
            ..CHAT
        };
        assert_eq!(
            Catalogue::new([CHAT, class]).unwrap_err(),
            ClaimError("ChatForCausalLM")
        );
    }
}
