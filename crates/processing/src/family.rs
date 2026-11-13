//! The families a build serves, as request processing resolves and serves them.
//!
//! A model crate states one [`Family`] beside the numerical model execution loads
//! (`nn::family::Family` holds it), and the executable registers each once: request processing
//! reads the [`Registry`] of these values, execution the registry of numerical families built
//! from the same list. A checkpoint resolves by what its own root files declare
//! ([`Registry::resolve`]): the daemon and the engine read those files where the checkpoint is and
//! resolve them by this one rule.

use serde_json::Value;

use crate::serving::Serving;

/// The file a transformers checkpoint declares its family in, at its root: `architectures[0]`
/// names its model class and `model_type` its configuration's kind.
pub const CONFIG_FILE: &str = "config.json";

/// The files a diffusers pipeline checkpoint declares its family in, at its root, in the order
/// they are read: the classic pipeline index, then the modular one. Each names the pipeline's
/// class in `_class_name`; a checkpoint's index is the first of them it publishes.
pub const PIPELINE_INDEX_FILES: [&str; 2] = ["model_index.json", "modular_model_index.json"];

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
pub const FAMILIES: [Family; 4] = [
    crate::qwen::QWEN3,
    crate::qwen::QWEN3_5_MOE,
    crate::qwen::QWEN4_EXP,
    crate::minimax_h3::FAMILY,
];

/// Two families claim one `model_type` or class: a checkpoint of it would resolve to either.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("two model families claim the class {0}")]
pub struct RegistryError(pub &'static str);

/// Why a checkpoint resolves to no family of a [`Registry`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// A root file the checkpoint publishes is not a JSON object.
    #[error("the checkpoint's {file} is not a JSON object: {message}")]
    Malformed {
        /// The file, [`CONFIG_FILE`] or the pipeline index.
        file: &'static str,
        /// What is wrong with it.
        message: String,
    },
    /// The checkpoint declares neither a class nor a `model_type`.
    #[error(
        "the checkpoint declares no model class: it has neither a {CONFIG_FILE} naming \
         `architectures` or `model_type` nor a pipeline index ({}) naming `_class_name`",
        PIPELINE_INDEX_FILES.join(" or ")
    )]
    Undeclared,
    /// No family claims what the checkpoint declares.
    #[error(
        "the checkpoint declares {declared}, which this build cannot serve: it serves {served}"
    )]
    Unserved {
        /// The class, `model_type` or both the checkpoint declares.
        declared: String,
        /// Every class and `model_type` the registry's families claim.
        served: String,
    },
}

/// The families an executable serves, each claiming the `model_type`s and classes its checkpoints
/// declare.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    families: Vec<Family>,
}

impl Registry {
    /// A registry of `families`.
    ///
    /// # Errors
    ///
    /// Fails, naming the class, when two families claim one `model_type` or class.
    pub fn new(families: impl IntoIterator<Item = Family>) -> Result<Self, RegistryError> {
        let families: Vec<Family> = families.into_iter().collect();
        for (index, family) in families.iter().enumerate() {
            for class in family.model_types.iter().chain(family.classes) {
                if families[..index].iter().any(|earlier| {
                    earlier.model_types.contains(class) || earlier.classes.contains(class)
                }) {
                    return Err(RegistryError(class));
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

    /// The family a checkpoint declares in its root files: `config` is the text of its
    /// [`CONFIG_FILE`] and `index` that of its pipeline index (the first of
    /// [`PIPELINE_INDEX_FILES`] it publishes), each `None` when the checkpoint publishes none.
    ///
    /// A `config.json` decides when it declares anything: the family claiming its class
    /// (`architectures[0]`), else the family claiming its `model_type`. A checkpoint whose root has
    /// no `config.json`, or one that declares neither, is a diffusers pipeline, decided by the
    /// class its index names (`_class_name`).
    ///
    /// # Errors
    ///
    /// [`ResolveError::Malformed`] for a file that is not a JSON object, [`ResolveError::Undeclared`]
    /// when neither file declares anything, and [`ResolveError::Unserved`] when no family claims
    /// what the deciding file declares.
    pub fn resolve(
        &self,
        config: Option<&str>,
        index: Option<&str>,
    ) -> Result<Family, ResolveError> {
        if let Some(config) = config {
            let config = object(CONFIG_FILE, config)?;
            let class = config
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|architectures| architectures.first())
                .and_then(Value::as_str);
            let model_type = config.get("model_type").and_then(Value::as_str);

            let claimed = class
                .and_then(|class| self.by_class(class))
                .or_else(|| model_type.and_then(|model_type| self.by_model_type(model_type)));
            match (claimed, class, model_type) {
                (Some(family), _, _) => return Ok(family),
                (None, None, None) => {}
                (None, class, model_type) => {
                    let declared = [
                        class.map(|class| format!("class {class}")),
                        model_type.map(|model_type| format!("model_type {model_type}")),
                    ];
                    return Err(self.unserved(declared.into_iter().flatten().collect()));
                }
            }
        }

        let Some(index) = index else {
            return Err(ResolveError::Undeclared);
        };
        let index = object("pipeline index", index)?;
        let class = index
            .get("_class_name")
            .and_then(Value::as_str)
            .ok_or(ResolveError::Undeclared)?;
        self.by_class(class)
            .ok_or_else(|| self.unserved(vec![format!("class {class}")]))
    }

    /// The refusal of a checkpoint that declares `declared`, which names everything this registry
    /// serves.
    fn unserved(&self, declared: Vec<String>) -> ResolveError {
        let claimed: Vec<&str> = self
            .families
            .iter()
            .flat_map(|family| family.classes.iter().chain(family.model_types))
            .copied()
            .collect();
        ResolveError::Unserved {
            declared: declared.join(" and "),
            served: claimed.join(", "),
        }
    }
}

/// `text`, the content of the checkpoint's `file`, as a JSON object.
fn object(file: &'static str, text: &str) -> Result<serde_json::Map<String, Value>, ResolveError> {
    let malformed = |message: String| ResolveError::Malformed { file, message };
    match serde_json::from_str(text) {
        Ok(Value::Object(object)) => Ok(object),
        Ok(_) => Err(malformed("it is not an object".to_owned())),
        Err(error) => Err(malformed(error.to_string())),
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
        let registry = Registry::new([CHAT]).unwrap();
        assert_eq!(registry.by_model_type("chat_model"), Some(CHAT));
        assert_eq!(registry.by_class("ChatForCausalLM"), Some(CHAT));
        assert_eq!(registry.by_model_type("other"), None);
        assert_eq!(registry.by_class("chat_model"), None);
    }

    /// Two families that claim one `model_type` or class are refused: a checkpoint of it would
    /// resolve to either.
    #[test]
    fn two_families_may_not_claim_one_class() {
        let twin = Family { id: "twin", ..CHAT };
        assert_eq!(
            Registry::new([CHAT, twin]).unwrap_err(),
            RegistryError("chat_model")
        );

        let class = Family {
            id: "class",
            model_types: &[],
            ..CHAT
        };
        assert_eq!(
            Registry::new([CHAT, class]).unwrap_err(),
            RegistryError("ChatForCausalLM")
        );
    }

    /// A video family declared by a pipeline's class alone.
    const VIDEO: Family = Family {
        id: "video",
        model_types: &[],
        classes: &["VideoModularPipeline"],
        ..CHAT
    };

    /// A checkpoint resolves by what its root files declare: a `config.json` by its class, else
    /// by its `model_type`, and a checkpoint with no `config.json` by its pipeline index's class.
    #[test]
    fn a_checkpoint_resolves_by_what_its_root_files_declare() {
        let registry = Registry::new([CHAT, VIDEO]).unwrap();
        let index = r#"{"_class_name": "VideoModularPipeline", "transformer": ["diffusers", "X"]}"#;

        assert_eq!(registry.resolve(None, Some(index)), Ok(VIDEO));
        assert_eq!(
            registry.resolve(
                Some(r#"{"architectures": ["ChatForCausalLM"]}"#),
                Some(index)
            ),
            Ok(CHAT)
        );
        assert_eq!(
            registry.resolve(Some(r#"{"model_type": "chat_model"}"#), None),
            Ok(CHAT)
        );
        // A class no family claims falls back to the `model_type` beside it.
        assert_eq!(
            registry.resolve(
                Some(r#"{"architectures": ["OtherForCausalLM"], "model_type": "chat_model"}"#),
                None
            ),
            Ok(CHAT)
        );
        // A `config.json` that declares nothing leaves the decision to the pipeline index.
        assert_eq!(registry.resolve(Some("{}"), Some(index)), Ok(VIDEO));
    }

    /// What no family claims is refused naming what the checkpoint declares and everything the
    /// build serves; a checkpoint that declares nothing, and a file that is no JSON object, are
    /// refused as such.
    #[test]
    fn an_unserved_or_undeclared_checkpoint_is_refused_by_name() {
        let registry = Registry::new([CHAT, VIDEO]).unwrap();

        let refused = registry
            .resolve(Some(r#"{"architectures": ["OtherForCausalLM"]}"#), None)
            .unwrap_err();
        let message = refused.to_string();
        assert!(message.contains("OtherForCausalLM"), "{message}");
        assert!(
            message.contains("ChatForCausalLM, chat_model, VideoModularPipeline"),
            "{message}"
        );

        let refused = registry
            .resolve(None, Some(r#"{"_class_name": "OtherPipeline"}"#))
            .unwrap_err();
        assert!(
            matches!(refused, ResolveError::Unserved { .. }),
            "{refused}"
        );

        assert_eq!(registry.resolve(None, None), Err(ResolveError::Undeclared));
        assert_eq!(
            registry.resolve(Some("{}"), None),
            Err(ResolveError::Undeclared)
        );
        assert!(
            registry
                .resolve(None, Some("{}"))
                .unwrap_err()
                .to_string()
                .contains("modular_model_index.json")
        );
        assert!(matches!(
            registry.resolve(Some("[1]"), None),
            Err(ResolveError::Malformed {
                file: "config.json",
                ..
            })
        ));
    }
}
