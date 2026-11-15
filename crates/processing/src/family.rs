//! The families a build serves, as request processing resolves and serves them.
//!
//! A model crate states one [`Family`] beside the numerical model execution loads
//! (`nn::family::Family` holds it), and the executable registers each once: request processing
//! reads the [`Registry`] of these values, execution the registry of numerical families built
//! from the same list. A family declares the one root file its checkpoints name their class in
//! ([`RootFile`]) and the classes it claims there; a checkpoint resolves by what those files
//! declare ([`Registry::resolve`]). The daemon and the engine read the root files the registry
//! names ([`Registry::root_files`]) where the checkpoint is and resolve them by this one rule.

use serde_json::Value;

use crate::serving::Serving;

/// A file at a checkpoint's root that names the checkpoint's class.
///
/// Each checkpoint format declares its class in one file under one key, and a family names the
/// format its checkpoints are published in: no file is read in place of another, and no second
/// key stands in for the class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RootFile {
    /// A transformers checkpoint's `config.json`, whose `architectures[0]` names the model
    /// class.
    Config,
    /// A diffusers modular pipeline's `modular_model_index.json`, whose `_class_name` names the
    /// pipeline class.
    ModularIndex,
}

impl RootFile {
    /// The file's name at the checkpoint's root.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Config => "config.json",
            Self::ModularIndex => "modular_model_index.json",
        }
    }

    /// The class the file's JSON object names, if it names one.
    fn class(self, object: &serde_json::Map<String, Value>) -> Option<&str> {
        match self {
            Self::Config => object
                .get("architectures")
                .and_then(Value::as_array)
                .and_then(|architectures| architectures.first())
                .and_then(Value::as_str),
            Self::ModularIndex => object.get("_class_name").and_then(Value::as_str),
        }
    }
}

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

/// The request-processing side of every model family this tree serves: what a daemon registers,
/// without linking a model. The engine executable registers the numerical side of the same
/// families, each of which holds its value of this list.
pub const FAMILIES: [Family; 4] = [
    crate::qwen::QWEN3,
    crate::qwen::QWEN3_5_MOE,
    crate::qwen::QWEN4_EXP,
    crate::minimax_h3::FAMILY,
];

/// Two families claim one class: a checkpoint of it would resolve to either.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("two model families claim the class {0}")]
pub struct RegistryError(pub &'static str);

/// Why a checkpoint resolves to no family of a [`Registry`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// A root file the checkpoint publishes is not a JSON object.
    #[error("the checkpoint's {} is not a JSON object: {message}", file.name())]
    Malformed {
        /// The file.
        file: RootFile,
        /// What is wrong with it.
        message: String,
    },
    /// No root file the registry's families declare in names a class.
    #[error(
        "the checkpoint declares no model class: none of {files} it publishes names one",
        files = files.join(", ")
    )]
    Undeclared {
        /// The root files the registry's families declare their classes in.
        files: Vec<&'static str>,
    },
    /// No family claims what the checkpoint declares.
    #[error(
        "the checkpoint declares {declared}, which this build cannot serve: it serves {served}"
    )]
    Unserved {
        /// Each class the checkpoint declares, with the file that declares it.
        declared: String,
        /// Every class the registry's families claim.
        served: String,
    },
    /// Two root files declare classes of two families.
    #[error("the checkpoint declares {first} and {second}, two families' classes")]
    Ambiguous {
        /// The first family's class, with the file that declares it.
        first: String,
        /// The second family's class, with the file that declares it.
        second: String,
    },
}

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
        let mut files = Vec::new();
        for family in &self.families {
            if !files.contains(&family.root) {
                files.push(family.root);
            }
        }
        files
    }

    /// The family a checkpoint declares in its root files: `files` holds the text of each of
    /// [`Self::root_files`] the checkpoint publishes.
    ///
    /// Every published file is read under its own key, and a family claims the checkpoint when
    /// the file it declares in names one of its classes. Exactly one family must claim it.
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
        let mut declared = Vec::new();
        let mut claimed: Option<(Family, String)> = None;
        for (file, text) in files {
            let object = object(file, text)?;
            let Some(class) = file.class(&object) else {
                continue;
            };
            let stated = format!("class {class} ({})", file.name());
            let family = self
                .families
                .iter()
                .copied()
                .find(|family| family.root == file && family.classes.contains(&class));
            match (family, &claimed) {
                (None, _) => declared.push(stated),
                (Some(family), None) => claimed = Some((family, stated)),
                (Some(family), Some((first, _))) if family == *first => {}
                (Some(_), Some((_, first))) => {
                    return Err(ResolveError::Ambiguous {
                        first: first.clone(),
                        second: stated,
                    });
                }
            }
        }

        if let Some((family, _)) = claimed {
            return Ok(family);
        }
        if declared.is_empty() {
            return Err(ResolveError::Undeclared {
                files: self.root_files().into_iter().map(RootFile::name).collect(),
            });
        }
        Err(ResolveError::Unserved {
            declared: declared.join(" and "),
            served: self
                .families
                .iter()
                .flat_map(|family| family.classes.iter().copied())
                .collect::<Vec<_>>()
                .join(", "),
        })
    }
}

/// `text`, the content of the checkpoint's `file`, as a JSON object.
fn object(file: RootFile, text: &str) -> Result<serde_json::Map<String, Value>, ResolveError> {
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
        root: RootFile::Config,
        classes: &["ChatForCausalLM"],
        serving: Serving::Chat(ChatProfile {
            reasoning: None,
            tool_calls: None,
            images: None,
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
