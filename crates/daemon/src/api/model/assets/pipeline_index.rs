//! Root stage index of a diffusers pipeline checkpoint.
//!
//! A pipeline checkpoint names its pipeline class and stage folders in a
//! root index instead of a root `config.json`: a modular pipeline's
//! `modular_model_index.json`, where each stage lives in the folder named by
//! its key unless its specification names its folder explicitly.
//!
//! The class selects the family ([`super::declared_family`]); `PipelineCheckpoint` then reads
//! the index for the folder of each stage it loads.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::api::model::assets::error::{Error, Result};

/// Stage folders declared by a checkpoint's root index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PipelineIndex {
    /// Stage name mapped to the checkpoint-relative folder holding its files.
    stages: BTreeMap<String, String>,
}

impl PipelineIndex {
    /// Parses a root index.
    ///
    /// Keys starting with `_` are index metadata. Every other key whose value is an array (a
    /// stage specification `[library, class, options?]`) names a stage; its folder is the string
    /// `subfolder` option when present and the key otherwise. Keys with non-array values are
    /// ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `content` is not JSON or not a JSON object.
    pub(super) fn parse(content: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(content)
            .map_err(|error| Error::invalid(format!("pipeline index is not JSON: {error}")))?;
        let index = value
            .as_object()
            .ok_or_else(|| Error::invalid("pipeline index must be a JSON object"))?;
        let stages = index
            .iter()
            .filter(|(name, specification)| !name.starts_with('_') && specification.is_array())
            .map(|(name, specification)| {
                let folder = specification
                    .get(2)
                    .and_then(|options| options.get("subfolder"))
                    .and_then(Value::as_str)
                    .unwrap_or(name);
                (name.clone(), folder.to_owned())
            })
            .collect();
        Ok(Self { stages })
    }

    /// The checkpoint-relative folders of the stages the index declares.
    pub(super) fn folders(&self) -> impl Iterator<Item = &str> {
        self.stages.values().map(String::as_str)
    }

    /// Whether the checkpoint-relative `file` lies in the folder of a declared stage.
    pub(super) fn holds(&self, file: &str) -> bool {
        within(file, self.folders())
    }

    /// The checkpoint-relative folder that holds `stage`'s files.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] for a stage the index does not declare.
    pub(super) fn stage_folder(&self, stage: &str) -> Result<&str> {
        self.stages
            .get(stage)
            .map(String::as_str)
            .ok_or_else(|| Error::invalid(format!("the pipeline declares no {stage} stage")))
    }
}

/// Whether the checkpoint-relative `file` lies in one of `folders`.
pub(super) fn within<'a>(file: &str, mut folders: impl Iterator<Item = &'a str>) -> bool {
    folders.any(|folder| {
        file.strip_prefix(folder)
            .is_some_and(|rest| rest.starts_with('/'))
    })
}

/// The text of the root index of the pipeline checkpoint in `directory`, which its family
/// declares in `index` (a modular pipeline's `modular_model_index.json`), or `None` for a
/// checkpoint that publishes none and so is no pipeline.
///
/// # Errors
///
/// Fails when a published index cannot be read.
pub(super) fn index_text(
    directory: &Path,
    index: scheduling::processing::RootFile,
) -> Result<Option<String>> {
    super::optional_text(directory, index.name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_modular_stage_resolves_to_its_declared_subfolder() {
        let index = PipelineIndex::parse(
            r#"{
                "_class_name": "ExamplePipeline",
                "_diffusers_version": "0.36.0",
                "tokenizer": ["transformers", "ExampleTokenizerFast",
                              {"subfolder": "text_tokenizer", "revision": null}]
            }"#,
        )
        .unwrap();

        assert_eq!(index.stage_folder("tokenizer").unwrap(), "text_tokenizer");
    }

    #[test]
    fn a_stage_without_a_subfolder_resolves_to_its_key() {
        let index = PipelineIndex::parse(
            r#"{"_class_name": "ExamplePipeline", "tokenizer": ["transformers", "ExampleTokenizer"]}"#,
        )
        .unwrap();

        assert_eq!(index.stage_folder("tokenizer").unwrap(), "tokenizer");
        assert!(index.stage_folder("vae").is_err());
    }

    /// A checkpoint is a pipeline when it publishes the index its family declares, and no other
    /// file stands in for it.
    #[test]
    fn a_checkpoint_is_a_pipeline_when_it_publishes_its_index() {
        let directory = tempfile::tempdir().unwrap();
        let index = scheduling::processing::RootFile::ModularIndex;
        std::fs::write(directory.path().join("model_index.json"), "classic").unwrap();
        assert_eq!(index_text(directory.path(), index).unwrap(), None);

        std::fs::write(directory.path().join("modular_model_index.json"), "modular").unwrap();
        assert_eq!(
            index_text(directory.path(), index).unwrap().as_deref(),
            Some("modular")
        );
    }
}
