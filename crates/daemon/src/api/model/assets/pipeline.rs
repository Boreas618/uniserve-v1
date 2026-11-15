//! Stage files of a diffusers pipeline checkpoint, over the release a layered checkpoint pins.
//!
//! A pipeline checkpoint holds its stages in the folders its root index names
//! ([`PipelineIndex`]). A family whose checkpoints may hold only some of them declares where they
//! pin the release whose folders supply the others ([`processing::BasePin`]); such a checkpoint
//! holds the stage folders it replaces, and every other stage is read from the release's
//! snapshot. An engine opens the checkpoint and the release as one checkpoint
//! ([`super::Checkpoint::base`]).

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::api::model::assets::error::{Error, Result};
use crate::api::model::assets::hub::{Hub, HubAsset, Snapshot};
use crate::api::model::assets::pipeline_index::{self, PipelineIndex, within};
use crate::api::model::assets::{optional_text, required_file, root_file};

/// A diffusers pipeline checkpoint with the source of each stage's folder.
pub struct PipelineCheckpoint {
    /// The checkpoint's root index.
    index: PipelineIndex,
    /// The checkpoint's directory.
    directory: PathBuf,
    /// The snapshot of the release that supplies the stage folders the checkpoint omits; `None`
    /// for a checkpoint that pins none.
    base: Option<PathBuf>,
}

impl PipelineCheckpoint {
    /// Opens the pipeline checkpoint `snapshot` holds, of the video family `family`, whose
    /// checkpoints publish their stage index in `index`; `None` for a checkpoint that publishes
    /// no index.
    ///
    /// The checkpoint's root files and the folders of the stages its index declares are fetched
    /// into the snapshot, as `diffusers` downloads a pipeline. When the family declares a pin
    /// and the checkpoint states one, the pinned release is resolved through `hub`, and its root
    /// files and the folders of the declared stages the checkpoint omits are fetched into the
    /// release's snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] for a malformed index or pin, and the errors of resolving and
    /// fetching either snapshot.
    pub async fn open(
        snapshot: &Snapshot,
        index: processing::RootFile,
        family: &dyn processing::VideoFamily,
        hub: &Hub,
    ) -> Result<Option<Self>> {
        let directory = snapshot.directory();
        let Some(index) = pipeline_index::index_text(directory, index)? else {
            return Ok(None);
        };
        let index = PipelineIndex::parse(&index)?;
        snapshot
            .fetch(|file| root_file(file) || index.holds(file))
            .await?;

        let pinned = match family.base_pin() {
            Some(pin) => pinned_release(directory, pin)?,
            None => None,
        };
        let base = match pinned {
            Some(release) => {
                let release = hub.snapshot(&release).await?;
                let omitted: Vec<&str> = index
                    .folders()
                    .filter(|folder| !directory.join(folder).is_dir())
                    .collect();
                release
                    .fetch(|file| root_file(file) || within(file, omitted.iter().copied()))
                    .await?;
                Some(release.directory().to_path_buf())
            }
            None => None,
        };
        Ok(Some(Self {
            index,
            directory: directory.to_path_buf(),
            base,
        }))
    }

    /// The directory of `stage`'s files: the checkpoint's folder of the stage when the
    /// checkpoint holds it, and the pinned release's otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] for a stage the index does not declare, and
    /// [`Error::MissingFile`] when neither the checkpoint nor its release holds the stage's
    /// folder.
    pub fn stage_directory(&self, stage: &str) -> Result<PathBuf> {
        let folder = self.index.stage_folder(stage)?;
        let own = self.directory.join(folder);
        if own.is_dir() {
            return Ok(own);
        }
        match &self.base {
            Some(base) if base.join(folder).is_dir() => Ok(base.join(folder)),
            _ => Err(Error::MissingFile {
                model: self.directory.display().to_string(),
                file: folder.to_owned(),
            }),
        }
    }

    /// The file `filename` of `stage`'s directory ([`Self::stage_directory`]).
    ///
    /// # Errors
    ///
    /// As [`Self::stage_directory`], and [`Error::MissingFile`] when the stage's directory has
    /// no such file.
    pub fn stage_file(&self, stage: &str, filename: &str) -> Result<PathBuf> {
        required_file(&self.stage_directory(stage)?, filename)
    }

    /// The snapshot of the release the checkpoint continues; `None` for a checkpoint that pins
    /// none.
    pub fn base(&self) -> Option<&Path> {
        self.base.as_deref()
    }
}

/// The release the pipeline checkpoint in `directory` pins where `pin` says its family's
/// checkpoints do; `None` for a checkpoint that publishes no such file or states no such field.
///
/// # Errors
///
/// Fails when the file cannot be read, is not JSON, or states a field that is not
/// `hf://<owner>/<name>@<revision>`.
fn pinned_release(directory: &Path, pin: processing::BasePin) -> Result<Option<HubAsset>> {
    let Some(text) = optional_text(directory, pin.file)? else {
        return Ok(None);
    };
    let path = directory.join(pin.file);
    let description: Value = serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.clone(),
        source,
    })?;
    match description.get(pin.field) {
        None => Ok(None),
        Some(Value::String(release)) => HubAsset::parse_pin(release).map(Some),
        Some(_) => Err(Error::invalid(format!(
            "{} states a {} that is not a string",
            path.display(),
            pin.field
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use processing::{BasePin, VideoBinding, VideoFamily, VideoProcessor};
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::api::model::assets::hub_stub::{COMMIT, HubStub, seed_snapshot};

    const BASE: &str = "org/base";

    /// A family whose checkpoints pin their release in `release.json`.
    struct Layered;

    impl VideoFamily for Layered {
        fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
            &[]
        }

        fn default_sizes(&self) -> &'static [&'static str] {
            &["64x64"]
        }

        fn default_max_seconds(&self) -> f64 {
            1.0
        }

        fn processing_parts(&self) -> &'static [&'static str] {
            &["tokenizer", "processor"]
        }

        fn base_pin(&self) -> Option<BasePin> {
            Some(BasePin {
                file: "release.json",
                field: "base",
            })
        }

        fn bind(
            &self,
            _: VideoBinding<'_>,
        ) -> std::result::Result<Box<dyn VideoProcessor>, String> {
            Err("this family binds no processor".to_owned())
        }
    }

    /// The index of a layered export: three stages, of which the export holds `transformer`.
    const INDEX: &str = r#"{
        "_class_name": "ExamplePipeline",
        "tokenizer": ["transformers", "ExampleTokenizer", {"subfolder": "tokenizer"}],
        "processor": ["transformers", "ExampleProcessor", {"subfolder": "processor"}],
        "transformer": ["diffusers", "ExampleTransformer", {"subfolder": "transformer"}]
    }"#;

    fn write(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    /// An export pinning `BASE` at `COMMIT`.
    fn export() -> TempDir {
        let root = tempdir().unwrap();
        write(&root.path().join("modular_model_index.json"), INDEX);
        write(
            &root.path().join("release.json"),
            &format!(r#"{{"base": "hf://{BASE}@{COMMIT}"}}"#),
        );
        write(&root.path().join("transformer/config.json"), "export DiT");
        root
    }

    /// The stage folders the export omits come from the snapshot of the pinned release: a file
    /// the cache holds is read without a request (the stub does not publish it), the others are
    /// downloaded, and the release's own copy of a stage the export holds, like a file of no
    /// stage, is never fetched. The export's own stage is read from the export.
    #[tokio::test]
    async fn an_export_reads_the_stages_it_omits_from_its_release() {
        let (export, cache) = (export(), tempdir().unwrap());
        let seeded = seed_snapshot(
            cache.path(),
            BASE,
            COMMIT,
            &[("tokenizer/tokenizer.json", "cached")],
        );
        let hub = HubStub::with_files(&[
            ("processor/config.json", "downloaded"),
            ("transformer/config.json", "release DiT"),
            ("docs/notes.md", "notes"),
        ])
        .listing(&[
            "tokenizer/tokenizer.json",
            "processor/config.json",
            "transformer/config.json",
            "docs/notes.md",
        ])
        .serve(cache.path())
        .await;

        let checkpoint = PipelineCheckpoint::open(
            &Snapshot::local(export.path().to_path_buf()),
            processing::RootFile::ModularIndex,
            &Layered,
            &hub,
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(checkpoint.base(), Some(seeded.as_path()));
        let read = |path: PathBuf| fs::read_to_string(path).unwrap();
        let tokenizer = checkpoint
            .stage_file("tokenizer", "tokenizer.json")
            .unwrap();
        assert_eq!(tokenizer, seeded.join("tokenizer/tokenizer.json"));
        assert_eq!(read(tokenizer), "cached");
        assert_eq!(
            read(checkpoint.stage_file("processor", "config.json").unwrap()),
            "downloaded"
        );
        assert_eq!(
            read(checkpoint.stage_file("transformer", "config.json").unwrap()),
            "export DiT"
        );
        assert!(!seeded.join("transformer").exists());
        assert!(!seeded.join("docs").exists());
    }

    /// A checkpoint that states no pin reads every stage from itself, and a stage folder it
    /// lacks is a missing file.
    #[tokio::test]
    async fn a_checkpoint_without_a_pin_reads_its_own_stages() {
        let (export, cache) = (export(), tempdir().unwrap());
        fs::remove_file(export.path().join("release.json")).unwrap();
        let hub = HubStub::default().serve(cache.path()).await;

        let checkpoint = PipelineCheckpoint::open(
            &Snapshot::local(export.path().to_path_buf()),
            processing::RootFile::ModularIndex,
            &Layered,
            &hub,
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(checkpoint.base(), None);
        assert!(checkpoint.stage_file("transformer", "config.json").is_ok());
        assert!(matches!(
            checkpoint.stage_file("tokenizer", "tokenizer.json"),
            Err(Error::MissingFile { .. })
        ));
    }
}
