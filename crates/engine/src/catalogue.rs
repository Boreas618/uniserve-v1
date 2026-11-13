//! The family a checkpoint resolves to.
//!
//! The executable registers each family once ([`nn::family::Registry`]); a main process resolves
//! its checkpoint by the class the checkpoint declares ([`resolve`]) and from then on dispatches by
//! the family's model and the capabilities it implements. Nothing else in this crate names a model.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use nn::family::{Family, Registry};

/// The family of the checkpoint at `checkpoint`, by the class it declares.
///
/// # Errors
///
/// Fails when the checkpoint declares no class, or one no family of `registry` claims; the
/// refusal names the classes the build serves.
pub fn resolve(registry: &Registry, checkpoint: &Path) -> Result<Family> {
    let declared = architectural_class(checkpoint)?;
    registry.by_class(&declared).with_context(|| {
        format!(
            "{} declares model class {declared}, which this build cannot serve: it serves {}",
            checkpoint.display(),
            registry
                .families()
                .iter()
                .flat_map(|family| family.request.classes.iter().copied())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

/// The model class a checkpoint declares, as the checkpoint itself states it.
///
/// A transformers checkpoint states it in `config.json`'s `architectures`; a diffusers modular
/// pipeline states it in `modular_model_index.json`'s `_class_name` and has no `config.json` at
/// its root at all. Both are read here, so a family is resolved by the declared class rather than
/// by which file a checkpoint happens to carry.
fn architectural_class(checkpoint: &Path) -> Result<String> {
    let config_path = checkpoint.join("config.json");
    if config_path.is_file() {
        let text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", config_path.display()))?;
        let class = config
            .get("architectures")
            .and_then(|architectures| architectures.as_array())
            .and_then(|architectures| architectures.first())
            .and_then(|class| class.as_str())
            .ok_or_else(|| anyhow::anyhow!("{} declares no model class", config_path.display()))?;
        return Ok(class.to_owned());
    }

    let index_path = checkpoint.join("modular_model_index.json");
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        let class = index
            .get("_class_name")
            .and_then(|class| class.as_str())
            .ok_or_else(|| anyhow::anyhow!("{} declares no `_class_name`", index_path.display()))?;
        return Ok(class.to_owned());
    }

    bail!(
        "{} declares no model class: it has neither {} (a transformers checkpoint's \
         `architectures`) nor {} (a modular pipeline's `_class_name`)",
        checkpoint.display(),
        config_path.display(),
        index_path.display()
    )
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

    /// A checkpoint resolves by the class it declares, read from whichever file its format states
    /// it in: a transformers checkpoint's `config.json`, a diffusers modular pipeline's
    /// `modular_model_index.json`.
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

    /// A class no family claims is refused naming the classes the build serves, and a checkpoint
    /// that declares none is refused naming both files a class is read from.
    #[test]
    fn an_unknown_or_missing_class_is_refused_by_name() {
        let registry = registry();
        let root = tempfile::tempdir().unwrap();

        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures": ["OtherForCausalLM"]}"#,
        )
        .unwrap();
        let error = format!("{:#}", resolve(&registry, root.path()).unwrap_err());
        assert!(error.contains("OtherForCausalLM"), "{error}");
        assert!(
            error.contains("TextForCausalLM, VideoModularPipeline"),
            "{error}"
        );

        std::fs::remove_file(root.path().join("config.json")).unwrap();
        let error = format!("{:#}", resolve(&registry, root.path()).unwrap_err());
        assert!(error.contains("declares no model class"), "{error}");
        assert!(error.contains("modular_model_index.json"), "{error}");
    }
}
