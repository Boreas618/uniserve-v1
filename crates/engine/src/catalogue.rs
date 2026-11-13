//! The family a checkpoint resolves to.
//!
//! The executable registers each family once ([`nn::family::Registry`]); a main process resolves
//! its checkpoint by what the checkpoint's root files declare, under request processing's one rule
//! ([`processing::Registry::resolve`]), and from then on dispatches by the family's model and the
//! capabilities it implements. Nothing else in this crate names a model.

use std::path::Path;

use anyhow::{Context as _, Result};
use nn::family::{Family, Registry};

/// The family of the checkpoint at `checkpoint`, by what its root `config.json` or pipeline index
/// declares ([`processing::Registry::resolve`]).
///
/// # Errors
///
/// Fails when a root file cannot be read, or the checkpoint declares nothing a family of
/// `registry` claims; the refusal names what the build serves.
pub fn resolve(registry: &Registry, checkpoint: &Path) -> Result<Family> {
    let config = root_file(checkpoint, processing::CONFIG_FILE)?;
    let mut index = None;
    for file in processing::PIPELINE_INDEX_FILES {
        index = root_file(checkpoint, file)?;
        if index.is_some() {
            break;
        }
    }

    let claims =
        processing::Registry::new(registry.families().iter().map(|family| family.request))?;
    let declared = claims
        .resolve(config.as_deref(), index.as_deref())
        .with_context(|| format!("resolving the model family of {}", checkpoint.display()))?;
    registry
        .families()
        .iter()
        .copied()
        .find(|family| family.request == declared)
        .with_context(|| format!("no family of this build is {}", declared.id))
}

/// The text of the file `name` at the checkpoint's root; `None` when the checkpoint has none.
fn root_file(checkpoint: &Path, name: &str) -> Result<Option<String>> {
    let path = checkpoint.join(name);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

#[cfg(test)]
mod tests {

    use nn::family::{Model, TextCheckpoint, TextFamily, VideoCheckpoint, VideoFamily};
    use processing::{ChatProfile, Serving};

    use super::*;

    /// A model that opens nothing: resolution reads only a family's classes.
    struct Unopened;

    impl TextFamily for Unopened {
        fn open(&self, _: &nn::Checkpoint) -> nn::Result<Box<dyn TextCheckpoint>> {
            Err(nn::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    impl VideoFamily for Unopened {
        fn open(&self, _: &nn::Checkpoint) -> nn::Result<Box<dyn VideoCheckpoint>> {
            Err(nn::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    /// The families of a text class and a video class.
    fn registry() -> Registry {
        let chat = Serving::Chat(ChatProfile {
            reasoning: None,
            tool_calls: None,
            images: None,
        });
        Registry::new([
            Family {
                request: processing::Family {
                    id: "text",
                    model_types: &[],
                    classes: &["TextForCausalLM"],
                    serving: chat,
                },
                model: Model::Text(&Unopened),
            },
            Family {
                request: processing::Family {
                    id: "video",
                    model_types: &[],
                    classes: &["VideoModularPipeline"],
                    serving: chat,
                },
                model: Model::Video(&Unopened),
            },
        ])
        .unwrap()
    }

    /// A checkpoint resolves by the root file its format declares it in: a diffusers modular
    /// pipeline's `modular_model_index.json`, and a transformers checkpoint's `config.json`, which
    /// decides when both are present.
    #[test]
    fn a_checkpoint_resolves_by_the_class_it_declares() {
        let registry = registry();
        let root = tempfile::tempdir().unwrap();

        std::fs::write(
            root.path().join("modular_model_index.json"),
            r#"{"_class_name": "VideoModularPipeline"}"#,
        )
        .unwrap();
        assert_eq!(resolve(&registry, root.path()).unwrap().id(), "video");

        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures": ["TextForCausalLM"]}"#,
        )
        .unwrap();
        assert_eq!(resolve(&registry, root.path()).unwrap().id(), "text");
    }
}
