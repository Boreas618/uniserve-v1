//! The families a build serves, as request processing resolves and serves them.
//!
//! A model crate states one [`Family`] beside the numerical model execution loads
//! (`nn::family::Family` holds it), and the executable registers each once: request processing
//! reads the [`Registry`] of these values, execution the registry of numerical families built
//! from the same list. A family declares the one root file its checkpoints name their class in
//! ([`RootFile`]) and the classes it claims there; a checkpoint resolves by what those files
//! declare ([`Registry::resolve`]). The engine reads the root files the registry names
//! ([`Registry::root_files`]) where the checkpoint is and resolves them by the processing
//! interface's one rule (`scheduling::processing::resolve`), by which the daemon resolves a
//! checkpoint among the families the processing program declares.

pub use scheduling::processing::{ResolveError, RootFile};

use crate::serving::Serving;

/// One model family, as its checkpoints declare it and request processing serves it.
#[derive(Clone, Copy)]
pub struct Family {
    /// The family's stable identifier, as the daemon reports it.
    pub id: &'static str,
    /// The root file the family's checkpoints name their class in.
    pub root: RootFile,
    /// The classes a checkpoint of this family names in [`Self::root`].
    pub classes: &'static [&'static str],
    /// What request processing serves.
    pub serving: Serving,
}

impl std::fmt::Debug for Family {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Family")
            .field("id", &self.id)
            .field("root", &self.root)
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

/// The request-processing side of every model family this tree serves: what the tree's processing
/// program serves (`crate::program::Release`), which the leader daemon loads without linking a
/// model. The engine executable registers the numerical side of the same families, each of which
/// holds its value of this list.
pub const FAMILIES: [Family; 5] = [
    crate::qwen::QWEN3,
    crate::qwen::QWEN3_5_MOE,
    crate::qwen::QWEN4_EXP,
    crate::minimax_h3::FAMILY,
    crate::deepseek_v41::FAMILY,
];

/// Two families claim one class: a checkpoint of it would resolve to either.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("two model families claim the class {0}")]
pub struct RegistryError(pub &'static str);

/// The families an executable serves, each claiming the classes its checkpoints declare in its
/// root file.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    families: Vec<Family>,
}

impl Registry {
    /// A registry of `families`.
    ///
    /// # Errors
    ///
    /// Fails, naming the class, when two families claim one class.
    pub fn new(families: impl IntoIterator<Item = Family>) -> Result<Self, RegistryError> {
        let families: Vec<Family> = families.into_iter().collect();
        for (index, family) in families.iter().enumerate() {
            for class in family.classes {
                if families[..index]
                    .iter()
                    .any(|earlier| earlier.classes.contains(class))
                {
                    return Err(RegistryError(class));
                }
            }
        }
        Ok(Self { families })
    }

    /// The family that claims `class`: a transformers checkpoint's `architectures[0]`, or a
    /// diffusers pipeline's `_class_name`.
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

    /// The root files the families declare their classes in, each once, in registration order:
    /// the files a checkpoint is resolved from.
    pub fn root_files(&self) -> Vec<RootFile> {
        scheduling::processing::root_files(&self.families)
    }

    /// The family a checkpoint declares in its root files: `files` holds the text of each of
    /// [`Self::root_files`] the checkpoint publishes, resolved by the processing interface's
    /// one rule ([`scheduling::processing::resolve`]).
    ///
    /// # Errors
    ///
    /// [`ResolveError::Malformed`] for a file that is not a JSON object,
    /// [`ResolveError::Undeclared`] when no file names a class, [`ResolveError::Unserved`] when no
    /// family claims a class the files name, and [`ResolveError::Ambiguous`] when two files name
    /// classes of two families.
    pub fn resolve<'a>(
        &self,
        files: impl IntoIterator<Item = (RootFile, &'a str)>,
    ) -> Result<Family, ResolveError> {
        scheduling::processing::resolve(&self.families, files).copied()
    }
}

impl scheduling::processing::Claim for Family {
    fn id(&self) -> &str {
        self.id
    }

    fn root(&self) -> RootFile {
        self.root
    }

    fn classes(&self) -> Vec<&str> {
        self.classes.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::ChatProfile;

    const CHAT: Family = Family {
        id: "chat",
        root: RootFile::Config,
        classes: &["ChatForCausalLM"],
        serving: Serving::Chat(ChatProfile {
            reasoning: None,
            tool_calls: None,
            images: None,
            checkpoint: crate::ChatCheckpoint {
                context_length: &["max_position_embeddings"],
                chat_template: crate::ChatTemplate::FileOrTokenizerConfig("chat_template.jinja"),
            },
        }),
    };

    /// A video family declared by a modular pipeline's class.
    const VIDEO: Family = Family {
        id: "video",
        root: RootFile::ModularIndex,
        classes: &["VideoModularPipeline"],
        ..CHAT
    };

    const CONFIG: &str = r#"{"architectures": ["ChatForCausalLM"], "model_type": "chat_model"}"#;
    const INDEX: &str =
        r#"{"_class_name": "VideoModularPipeline", "transformer": ["diffusers", "X"]}"#;

    /// A family is found by the classes it claims, and two families may not claim one class.
    #[test]
    fn a_family_is_found_by_its_one_claimed_class() {
        let registry = Registry::new([CHAT, VIDEO]).unwrap();
        assert_eq!(registry.by_class("ChatForCausalLM"), Some(CHAT));
        assert_eq!(registry.by_class("chat_model"), None);
        assert_eq!(
            registry.root_files(),
            [RootFile::Config, RootFile::ModularIndex]
        );

        let twin = Family {
            id: "twin",
            ..VIDEO
        };
        assert_eq!(
            Registry::new([CHAT, VIDEO, twin]).unwrap_err(),
            RegistryError("VideoModularPipeline")
        );
    }

    /// A checkpoint resolves by the class the file its family declares in names, whichever other
    /// files it publishes.
    #[test]
    fn a_checkpoint_resolves_by_its_familys_root_file() {
        let registry = Registry::new([CHAT, VIDEO]).unwrap();
        assert_eq!(registry.resolve([(RootFile::Config, CONFIG)]), Ok(CHAT));
        assert_eq!(
            registry.resolve([(RootFile::ModularIndex, INDEX)]),
            Ok(VIDEO)
        );
        // A `config.json` that names no class leaves the decision to the index, as a file the
        // video family does not declare in.
        assert_eq!(
            registry.resolve([(RootFile::Config, "{}"), (RootFile::ModularIndex, INDEX)]),
            Ok(VIDEO)
        );
    }

    /// A class no family claims is refused, whatever its `model_type` says, naming what the
    /// checkpoint declares and everything the build serves; a checkpoint whose files name no
    /// class, two files naming two families' classes, and a file that is no JSON object are each
    /// refused as such.
    #[test]
    fn an_unserved_undeclared_or_ambiguous_checkpoint_is_refused() {
        let registry = Registry::new([CHAT, VIDEO]).unwrap();

        let unserved = r#"{"architectures": ["OtherForCausalLM"], "model_type": "chat_model"}"#;
        let message = registry
            .resolve([(RootFile::Config, unserved)])
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("OtherForCausalLM (config.json)"),
            "{message}"
        );
        assert!(
            message.contains("ChatForCausalLM, VideoModularPipeline"),
            "{message}"
        );

        assert!(matches!(
            registry.resolve([]),
            Err(ResolveError::Undeclared { .. })
        ));
        let undeclared = registry
            .resolve([(RootFile::ModularIndex, "{}")])
            .unwrap_err();
        assert!(
            undeclared.to_string().contains("modular_model_index.json"),
            "{undeclared}"
        );

        assert!(matches!(
            registry.resolve([(RootFile::Config, CONFIG), (RootFile::ModularIndex, INDEX)]),
            Err(ResolveError::Ambiguous { .. })
        ));
        assert!(matches!(
            registry.resolve([(RootFile::Config, "[1]")]),
            Err(ResolveError::Malformed {
                file: RootFile::Config,
                ..
            })
        ));
    }
}
