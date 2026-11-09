//! The model families this executable serves, and the family a checkpoint resolves to.
//!
//! The executable registers each family once ([`Catalogue::new`]); a main process resolves its
//! checkpoint by the class the checkpoint declares and from then on dispatches by the family's kind
//! and the capabilities its model implements. Nothing else in this crate names a model.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use nn::family::Family;

/// The model families a build serves, each claiming the classes its checkpoints declare.
#[derive(Clone, Debug)]
pub struct Catalogue {
    families: Vec<Family>,
}

impl Catalogue {
    /// A catalogue of `families`.
    ///
    /// # Errors
    ///
    /// Fails when two families claim one class: a checkpoint of it would resolve to either.
    pub fn new(families: impl IntoIterator<Item = Family>) -> Result<Self> {
        let families: Vec<Family> = families.into_iter().collect();
        for (index, family) in families.iter().enumerate() {
            for class in family.classes() {
                if families[..index]
                    .iter()
                    .any(|earlier| earlier.classes().contains(class))
                {
                    bail!("two model families claim the class {class}");
                }
            }
        }
        Ok(Self { families })
    }

    /// The family of the checkpoint at `checkpoint`, by the class it declares.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint declares no class, or one no family of this build claims; the
    /// refusal names the classes the build serves.
    pub fn resolve(&self, checkpoint: &Path) -> Result<Family> {
        let declared = architectural_class(checkpoint)?;
        self.families
            .iter()
            .copied()
            .find(|family| family.classes().contains(&declared.as_str()))
            .with_context(|| {
                format!(
                    "{} declares model class {declared}, which this build cannot serve: it serves {}",
                    checkpoint.display(),
                    self.families
                        .iter()
                        .flat_map(|family| family.classes().iter().copied())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
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
    use std::path::PathBuf;

    use nn::family::{TextCheckpoint, TextFamily, VideoCheckpoint, VideoFamily};

    use super::*;

    /// A family that claims one class and opens nothing: the catalogue reads only its classes.
    struct Claims(&'static [&'static str]);

    impl TextFamily for Claims {
        fn classes(&self) -> &'static [&'static str] {
            self.0
        }

        fn open(&self, _: &Path) -> nn::Result<Box<dyn TextCheckpoint>> {
            Err(nn::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    impl VideoFamily for Claims {
        fn classes(&self) -> &'static [&'static str] {
            self.0
        }

        fn open(&self, _: &[PathBuf]) -> nn::Result<Box<dyn VideoCheckpoint>> {
            Err(nn::Error::Unsupported("no checkpoint".to_owned()))
        }
    }

    static TEXT: Claims = Claims(&["TextForCausalLM"]);
    static VIDEO: Claims = Claims(&["VideoModularPipeline"]);

    /// A checkpoint resolves by the class it declares, read from whichever file its format states
    /// it in: a transformers checkpoint's `config.json`, a diffusers modular pipeline's
    /// `modular_model_index.json`.
    #[test]
    fn a_checkpoint_resolves_by_the_class_it_declares() {
        let catalogue = Catalogue::new([Family::Text(&TEXT), Family::Video(&VIDEO)]).unwrap();
        let root = tempfile::tempdir().unwrap();

        std::fs::write(
            root.path().join("modular_model_index.json"),
            r#"{"_class_name": "VideoModularPipeline"}"#,
        )
        .unwrap();
        assert!(matches!(
            catalogue.resolve(root.path()).unwrap(),
            Family::Video(_)
        ));

        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures": ["TextForCausalLM"]}"#,
        )
        .unwrap();
        assert!(matches!(
            catalogue.resolve(root.path()).unwrap(),
            Family::Text(_)
        ));
    }

    /// A class no family claims is refused naming the classes the build serves, and a checkpoint
    /// that declares none is refused naming both files a class is read from.
    #[test]
    fn an_unknown_or_missing_class_is_refused_by_name() {
        let catalogue = Catalogue::new([Family::Text(&TEXT), Family::Video(&VIDEO)]).unwrap();
        let root = tempfile::tempdir().unwrap();

        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures": ["OtherForCausalLM"]}"#,
        )
        .unwrap();
        let error = format!("{:#}", catalogue.resolve(root.path()).unwrap_err());
        assert!(error.contains("OtherForCausalLM"), "{error}");
        assert!(
            error.contains("TextForCausalLM, VideoModularPipeline"),
            "{error}"
        );

        std::fs::remove_file(root.path().join("config.json")).unwrap();
        let error = format!("{:#}", catalogue.resolve(root.path()).unwrap_err());
        assert!(error.contains("declares no model class"), "{error}");
        assert!(error.contains("modular_model_index.json"), "{error}");
    }

    /// Two families that claim one class are refused: a checkpoint of it would resolve to either.
    #[test]
    fn two_families_may_not_claim_one_class() {
        let error = Catalogue::new([Family::Text(&TEXT), Family::Video(&TEXT)]).unwrap_err();
        assert!(error.to_string().contains("TextForCausalLM"), "{error}");
    }
}
