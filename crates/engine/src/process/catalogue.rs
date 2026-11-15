//! The family a checkpoint resolves to.
//!
//! The executable registers each family once ([`nn::family::Registry`]); a main process resolves
//! its checkpoint by what the checkpoint's root files declare, under request processing's one rule
//! ([`processing::Registry::resolve`]), and from then on dispatches by the family's model and the
//! capabilities it implements. Nothing else in this crate names a model.

use std::path::Path;

use anyhow::{Context as _, Result};
use nn::family::{Family, Registry};

/// The family of the checkpoint at `checkpoint`, by what the root files the families declare in
/// name ([`processing::Registry::resolve`]).
///
/// # Errors
///
/// Fails when a root file cannot be read, or the checkpoint declares nothing a family of
/// `registry` claims; the refusal names what the build serves.
pub(crate) fn resolve(registry: &Registry, checkpoint: &Path) -> Result<Family> {
    let claims =
        processing::Registry::new(registry.families().iter().map(|family| family.request))?;
    let mut files = Vec::new();
    for file in claims.root_files() {
        if let Some(text) = root_file(checkpoint, file.name())? {
            files.push((file, text));
        }
    }

    let declared = claims
        .resolve(files.iter().map(|(file, text)| (*file, text.as_str())))
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
        fn open(
            &self,
            _: &nn::Checkpoint,
            _: &proto::config::Quantization,
        ) -> nn::Result<Box<dyn TextCheckpoint>> {
            Err(nn::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    impl VideoFamily for Unopened {
        fn open(
            &self,
            _: &nn::Checkpoint,
            _: &proto::config::Quantization,
        ) -> nn::Result<Box<dyn VideoCheckpoint>> {
            Err(nn::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    /// The families of a text class and a video class.
    fn registry() -> Registry {
        let chat = Serving::Chat(ChatProfile {
            reasoning: None,
            tool_calls: None,
            images: None,
            checkpoint: processing::ChatCheckpoint {
                context_length: &["max_position_embeddings"],
                chat_template: processing::ChatTemplate::TokenizerConfig,
            },
        });
        Registry::new([
            Family {
                request: processing::Family {
                    id: "text",
                    root: processing::RootFile::Config,
                    classes: &["TextForCausalLM"],
                    serving: chat,
                },
                model: Model::Text(&Unopened),
                step_stack: "TextTransformer",
            },
            Family {
                request: processing::Family {
                    id: "video",
                    root: processing::RootFile::ModularIndex,
                    classes: &["VideoModularPipeline"],
                    serving: chat,
                },
                model: Model::Video(&Unopened),
                step_stack: "VideoTransformer",
            },
        ])
        .unwrap()
    }

    /// A checkpoint resolves by the root file its family declares in: a diffusers modular
    /// pipeline's `modular_model_index.json`, a transformers checkpoint's `config.json`; one whose
    /// two files name two families' classes is refused.
    #[test]
    fn a_checkpoint_resolves_by_the_class_it_declares() {
        let registry = registry();
        let (pipeline, text) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let index = r#"{"_class_name": "VideoModularPipeline"}"#;
        let config = r#"{"architectures": ["TextForCausalLM"]}"#;

        std::fs::write(pipeline.path().join("modular_model_index.json"), index).unwrap();
        assert_eq!(resolve(&registry, pipeline.path()).unwrap().id(), "video");

        std::fs::write(text.path().join("config.json"), config).unwrap();
        assert_eq!(resolve(&registry, text.path()).unwrap().id(), "text");

        std::fs::write(pipeline.path().join("config.json"), config).unwrap();
        assert!(resolve(&registry, pipeline.path()).is_err());
    }
}
