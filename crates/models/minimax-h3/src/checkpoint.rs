//! The H3 checkpoint's layout: which DiT partitions it holds, which tasks each serves, the FastH3
//! contract that states a student's schedule and sparse attention, and the JSON sidecars the
//! loading boundary reads.
//!
//! A checkpoint is one of two things (specs/worker.md 9, `docs/minimax_h3/minimax_h3.md`):
//!
//! - the base release, a pipeline index (`model_index.json`) over the directories
//!   `transformer`, `transformer_ref`, `text_encoder`, `processor`, `vae` and `audio_vae`, of which
//!   the two DiT directories are optional and a deployment places one;
//! - a FastH3 export, which states a `fastvideo_inference.json` contract naming its task (so the
//!   base DiT partition it replaces and the tasks it serves), its schedule (DMD rungs or
//!   parallel-decoding blocks), its sparse attention, and the base revision it reads every
//!   directory it omits from.
//!
//! Nothing here builds a network: [`Layout`] is what the checkpoint statement says, and
//! [`crate::config::Config`] is the typed architecture the layers consume.

use std::collections::BTreeMap;
use std::path::Path;

use nn::VideoTask;
use serde_json::{Map, Value};

use crate::{Error, Result};

/// One diffusion transformer a checkpoint directory holds.
///
/// The base release holds the base network (`transformer`) for `t2va` and `fl2va` and the
/// reference-conditioned one (`transformer_ref`) for `ref2va`; a FastH3 export holds the single
/// partition it distils, and places one per deployment.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Dit {
    /// The base network, directory `transformer`.
    Base,
    /// The reference-conditioned network, directory `transformer_ref`.
    Reference,
}

impl Dit {
    /// The checkpoint directory of this network.
    pub const fn directory(self) -> &'static str {
        match self {
            Self::Base => "transformer",
            Self::Reference => "transformer_ref",
        }
    }

    /// Both networks, in the order a checkpoint's sidecars are read.
    pub const ALL: [Self; 2] = [Self::Base, Self::Reference];
}

/// The DiT partition of the base release that serves each task.
pub const TASK_DITS: [(VideoTask, Dit); 3] = [
    (VideoTask::TextToVideoAudio, Dit::Base),
    (VideoTask::FirstLastToVideoAudio, Dit::Base),
    (VideoTask::ReferenceToVideoAudio, Dit::Reference),
];

/// The module roots a packed ModelOpt export's manifest names, and the network each belongs to.
pub const MANIFEST_ROOTS: [(&str, Dit); 2] = [
    ("denoiser", Dit::Base),
    ("reference_denoiser", Dit::Reference),
];

/// The pipeline class a load of this crate takes (`model_index.json`'s `_class_name`).
///
/// A task export such as `MiniMax-H3/FL2VA` states `MiniMaxH3Pipeline` and names a `video_vae`
/// directory; it is not a pipeline root, and the engine's dispatch on the checkpoint's class name
/// does not reach this crate with one.
pub const PIPELINE_CLASS: &str = "MiniMaxH3ModularPipeline";

/// The tasks `dit` serves, in canonical order.
pub fn tasks_of(dit: Dit) -> Vec<VideoTask> {
    TASK_DITS
        .iter()
        .filter_map(|(task, owner)| (*owner == dit).then_some(*task))
        .collect()
}

/// What a FastH3 export's `fastvideo_inference.json` states that the architecture needs.
///
/// The contract also carries provenance (checkpoint hashes, the FastVideo commit, the kernel
/// backend), which no numerical layer reads and which is therefore not part of this value.
#[derive(Clone, PartialEq, Debug)]
pub struct Contract {
    /// The task the student was distilled for, and so the tasks it serves.
    pub task: VideoTask,
    /// The base DiT partition the student replaces.
    pub dit: Dit,
    /// The base revision the export reads every directory it omits from, as the export states it
    /// (`hf://<repository>@<revision>` or a path).
    pub base_revision: Option<String>,
    /// The video noise level shift of the student's schedule.
    pub video_shift: f32,
    /// The audio noise level shift of the student's schedule.
    pub audio_shift: f32,
    /// The student's schedule.
    pub schedule: ContractSchedule,
    /// The student's sparse attention.
    pub attention: SparseContract,
}

/// The schedule a FastH3 export states, one of the two kinds a student is distilled with.
#[derive(Clone, PartialEq, Debug)]
pub enum ContractSchedule {
    /// A DMD student's trained rungs on the `num_train_timesteps`-step training schedule.
    Rungs(Vec<u32>),
    /// A parallel-decoding student's block boundaries on its fine grid, and the grid's maximum
    /// unshifted time.
    Nodes {
        /// Fine-grid indices of the block boundaries, strictly increasing from zero to the interval
        /// count.
        nodes: Vec<u32>,
        /// The fixed point of the modality shifts, in `(0, 1]`.
        max_t: f32,
    },
}

/// The sparse attention a FastH3 export states.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SparseContract {
    /// Rows of one spatiotemporal tile.
    pub tile: u32,
    /// The fraction of key tiles a generated-video query tile does not read.
    pub sparsity: f64,
    /// The fraction of its own key tiles a reference-video query keeps, when the student has
    /// reference videos.
    pub reference_keep: Option<f64>,
}

/// One recognized checkpoint: the DiT partitions it holds with the tasks each serves, the contract
/// an export states, and whether its weights are a packed ModelOpt manifest.
#[derive(Clone, PartialEq, Debug)]
pub struct Layout {
    /// Each DiT partition the checkpoint holds, with the tasks it serves in canonical order.
    pub dits: Vec<(Dit, Vec<VideoTask>)>,
    /// The FastH3 contract, for an export that states one.
    pub contract: Option<Contract>,
    /// Whether the checkpoint holds a `modelopt_manifest.json`, whose roots name the packed
    /// modules the weight loader reads at the checkpoint's own representation.
    pub packed: bool,
}

impl Layout {
    /// Classify the checkpoint at `root`.
    ///
    /// A FastH3 export is recognized by its contract; anything else by a pipeline index and the
    /// DiT directories it names and the root holds.
    pub fn detect(root: &Path) -> Result<Self> {
        let contract = Contract::read(root)?;
        let packed = root.join("modelopt_manifest.json").is_file();
        if let Some(contract) = contract {
            let dits = vec![(contract.dit, tasks_of(contract.dit))];
            return Ok(Self {
                dits,
                contract: Some(contract),
                packed,
            });
        }

        let index = read_json(&root.join("model_index.json")).map_err(|error| match error {
            Error::Io { .. } => Error::checkpoint(format!(
                "{} holds neither fastvideo_inference.json nor model_index.json",
                root.display()
            )),
            other => other,
        })?;
        let index = index.as_object().ok_or_else(|| {
            Error::checkpoint(format!(
                "{}/model_index.json must hold a JSON object",
                root.display()
            ))
        })?;
        if let Some(class) = index.get("_class_name").and_then(Value::as_str) {
            if class != PIPELINE_CLASS {
                return Err(Error::checkpoint(format!(
                    "{} is a {class} checkpoint, not a {PIPELINE_CLASS} pipeline root; the task \
                     exports under FL2VA/ are not pipeline roots",
                    root.display()
                )));
            }
        }
        let mut dits = Vec::new();
        for dit in Dit::ALL {
            if index.contains_key(dit.directory())
                && root.join(dit.directory()).join("config.json").is_file()
            {
                dits.push((dit, tasks_of(dit)));
            }
        }
        if dits.is_empty() {
            return Err(Error::checkpoint(format!(
                "{}/model_index.json names no DiT partition the root holds",
                root.display()
            )));
        }
        Ok(Self {
            dits,
            contract: None,
            packed,
        })
    }
}

impl Contract {
    /// Read the contract at `root`, or `None` when the checkpoint states none.
    pub fn read(root: &Path) -> Result<Option<Self>> {
        let path = root.join("fastvideo_inference.json");
        if !path.is_file() {
            return Ok(None);
        }
        let value = read_json(&path)?;
        let object = value.as_object().ok_or_else(|| {
            Error::checkpoint(format!(
                "{}/fastvideo_inference.json must hold a JSON object",
                root.display()
            ))
        })?;
        Self::parse(object).map(Some)
    }

    /// Parse a contract's JSON object.
    pub fn parse(object: &Map<String, Value>) -> Result<Self> {
        // A student states its task as `task` (`t2av` for text-to-video-and-audio) or, in the
        // OmniRef exports, as `model_type`; a task no partition serves is refused.
        let stated = object
            .get("task")
            .or_else(|| object.get("model_type"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::checkpoint(
                    "fastvideo_inference.json states no task or model_type".to_owned(),
                )
            })?;
        let task = match stated {
            "t2av" => VideoTask::TextToVideoAudio,
            "ref2va" => VideoTask::ReferenceToVideoAudio,
            other => {
                return Err(Error::checkpoint(format!(
                    "unsupported FastH3 checkpoint: its task must be t2av or ref2va, got {other:?}"
                )));
            }
        };
        let dit = TASK_DITS
            .iter()
            .find_map(|(served, dit)| (*served == task).then_some(*dit))
            .ok_or_else(|| {
                Error::checkpoint(format!("no DiT partition serves the task {stated:?}"))
            })?;
        let base_revision = match object.get("base_model_revision") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(Error::checkpoint(
                    "fastvideo_inference.json base_model_revision must be a string".to_owned(),
                ));
            }
        };
        let video_shift = single(object, "video_scheduler_shift")?;
        let audio_shift = single(object, "audio_scheduler_shift")?;
        let schedule = Self::schedule(object)?;
        let attention = SparseContract {
            tile: integer(object, "vsa_tile_size")?,
            sparsity: number(object, "vsa_sparsity")?,
            reference_keep: match object.get("vsa_ref_keep_rate") {
                None | Some(Value::Null) => None,
                Some(value) => Some(value.as_f64().ok_or_else(|| {
                    Error::checkpoint(
                        "fastvideo_inference.json vsa_ref_keep_rate must be a number".to_owned(),
                    )
                })?),
            },
        };
        Ok(Self {
            task,
            dit,
            base_revision,
            video_shift,
            audio_shift,
            schedule,
            attention,
        })
    }

    /// The schedule the contract states: a DMD ladder or a parallel-decoding block grid.
    fn schedule(object: &Map<String, Value>) -> Result<ContractSchedule> {
        match (
            object.get("dmd_denoising_steps"),
            object.get("pdd_step_indices"),
        ) {
            (Some(steps), None) => {
                let rungs = steps
                    .as_array()
                    .ok_or_else(|| {
                        Error::checkpoint(
                            "fastvideo_inference.json dmd_denoising_steps must be a list"
                                .to_owned(),
                        )
                    })?
                    .iter()
                    .map(|rung| {
                        rung.as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .ok_or_else(|| {
                                Error::checkpoint(
                                    "fastvideo_inference.json dmd_denoising_steps must hold \
                                     integers"
                                        .to_owned(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ContractSchedule::Rungs(rungs))
            }
            (None, Some(indices)) => {
                let nodes = indices
                    .as_array()
                    .ok_or_else(|| {
                        Error::checkpoint(
                            "fastvideo_inference.json pdd_step_indices must be a list".to_owned(),
                        )
                    })?
                    .iter()
                    .map(|node| {
                        node.as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .ok_or_else(|| {
                                Error::checkpoint(
                                    "fastvideo_inference.json pdd_step_indices must hold integers"
                                        .to_owned(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ContractSchedule::Nodes {
                    nodes,
                    max_t: single(object, "grid_max_t")?,
                })
            }
            (Some(_), Some(_)) => Err(Error::checkpoint(
                "fastvideo_inference.json states both dmd_denoising_steps and pdd_step_indices"
                    .to_owned(),
            )),
            (None, None) => Err(Error::checkpoint(
                "unsupported FastH3 checkpoint: fastvideo_inference.json states neither \
                 dmd_denoising_steps nor pdd_step_indices"
                    .to_owned(),
            )),
        }
    }
}

/// The parsed JSON of every sidecar the loading boundary reads.
///
/// [`crate::config::Config::normalize`] turns these into the typed architecture without touching
/// the filesystem, so a test states a checkpoint as JSON and exercises the same validation a
/// deployment does.
pub struct Sidecars {
    /// `text_encoder/config.json`: the Qwen3-VL text model and vision tower.
    pub text_encoder: Value,
    /// `processor/preprocessor_config.json`: the pixel normalization of the vision tower.
    pub processor: Value,
    /// `processor/video_preprocessor_config.json`, when the checkpoint states one.
    pub video_processor: Option<Value>,
    /// `vae/config.json`: the video VAE.
    pub video_vae: Value,
    /// `audio_vae/config.json`: the audio VAE.
    pub audio_vae: Value,
    /// `scheduler/scheduler_config.json`: the video noise level shift.
    pub scheduler: Value,
    /// `audio_scheduler/scheduler_config.json`: the audio noise level shift.
    pub audio_scheduler: Value,
    /// Each DiT partition the layout holds: `<directory>/config.json`.
    pub dits: BTreeMap<Dit, Value>,
}

impl Sidecars {
    /// Read every sidecar of the checkpoint at `root`.
    ///
    /// A sidecar the root omits is read from `base`, the snapshot of the base revision a FastH3
    /// export pins (`Contract::base_revision`), which is how an export that holds only its DiT
    /// partition and schedulers reaches the text encoder, processor and VAEs it uses. `base` is
    /// `None` for a checkpoint that holds every directory, and a sidecar both roots lack is an
    /// error naming it.
    pub fn read(root: &Path, base: Option<&Path>, layout: &Layout) -> Result<Self> {
        let read = |relative: &str| -> Result<Value> { read_sidecar(root, base, relative) };
        let text_encoder = read("text_encoder/config.json")?;
        let processor = read("processor/preprocessor_config.json")?;
        let video_processor =
            optional_sidecar(root, base, "processor/video_preprocessor_config.json")?;
        let video_vae = read("vae/config.json")?;
        let audio_vae = read("audio_vae/config.json")?;
        let scheduler = read("scheduler/scheduler_config.json")?;
        let audio_scheduler = read("audio_scheduler/scheduler_config.json")?;
        let mut dits = BTreeMap::new();
        for (dit, _) in &layout.dits {
            let relative = format!("{}/config.json", dit.directory());
            dits.insert(*dit, read(&relative)?);
        }
        Ok(Self {
            text_encoder,
            processor,
            video_processor,
            video_vae,
            audio_vae,
            scheduler,
            audio_scheduler,
            dits,
        })
    }
}

/// Read one sidecar of the checkpoint at `root`, falling back to `base` when the root omits it.
fn read_sidecar(root: &Path, base: Option<&Path>, relative: &str) -> Result<Value> {
    let own = root.join(relative);
    if own.is_file() {
        return read_json(&own);
    }
    if let Some(base) = base {
        let inherited = base.join(relative);
        if inherited.is_file() {
            return read_json(&inherited);
        }
    }
    Err(Error::checkpoint(format!(
        "the checkpoint at {} holds no {relative}",
        root.display()
    )))
}

/// Read one sidecar that a checkpoint may omit, from the root or from `base`.
fn optional_sidecar(root: &Path, base: Option<&Path>, relative: &str) -> Result<Option<Value>> {
    let own = root.join(relative);
    if own.is_file() {
        return read_json(&own).map(Some);
    }
    match base {
        Some(base) if base.join(relative).is_file() => read_json(&base.join(relative)).map(Some),
        _ => Ok(None),
    }
}

/// Read and parse one JSON file.
fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// A JSON object's `key` as an integer, refusing a value that is absent or not an integer.
///
/// The contract's fields are counts and indices, so a negative or fractional value is refused
/// rather than truncated.
fn integer(object: &Map<String, Value>, key: &str) -> Result<u32> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            Error::checkpoint(format!("fastvideo_inference.json states no integer {key}"))
        })
}

/// A JSON object's `key` as a finite number.
/// One number a contract states, in the FP64 the sidecar writes and the reference computes in.
///
/// The sparsity's `ceil((1 - sparsity) * tiles)` is a tile count, so narrowing to FP32 would
/// change it: FP32 0.9 widened back is greater than 0.9, and a ten-tile segment then keeps two
/// tiles instead of one.
fn number(object: &Map<String, Value>, key: &str) -> Result<f64> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            Error::checkpoint(format!("fastvideo_inference.json states no number {key}"))
        })
}

/// One number a contract states as an FP32 quantity (the schedule's shifts).
fn single(object: &Map<String, Value>, key: &str) -> Result<f32> {
    Ok(number(object, key)? as f32)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use nn::VideoTask;

    use super::*;

    fn write(root: &Path, relative: &str, body: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("a sidecar has a parent")).expect("create");
        fs::write(&path, body).expect("write");
    }

    fn base_index(dits: &[&str]) -> String {
        let entries = dits
            .iter()
            .map(|dit| format!("\"{dit}\": [\"diffusers\", \"Model\"]"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{{entries}}}")
    }

    #[test]
    fn a_task_export_is_not_a_pipeline_root() {
        let root = tempfile::tempdir().expect("tempdir");
        // `MiniMax-H3/FL2VA/model_index.json`: the pipeline class a task export states, with its
        // `video_vae` directory rather than the modular pipeline's `vae`.
        write(
            &root.path(),
            "model_index.json",
            r#"{"_class_name": "MiniMaxH3Pipeline", "transformer": ["diffusers", "Model"]}"#,
        );
        write(&root.path(), "transformer/config.json", "{}");
        write(&root.path(), "video_vae/config.json", "{}");
        let error = Layout::detect(root.path())
            .err()
            .expect("a task export is refused");
        assert!(
            matches!(error, Error::Checkpoint(ref problem)
                if problem.contains("MiniMaxH3Pipeline") && problem.contains("pipeline root")),
            "{error}"
        );

        // The modular pipeline's own class name is taken, and an index that states none is not
        // refused for it: a hand-built export need not name one.
        write(
            &root.path(),
            "model_index.json",
            &format!(
                r#"{{"_class_name": "{PIPELINE_CLASS}", "transformer": ["diffusers", "Model"]}}"#
            ),
        );
        assert!(Layout::detect(root.path()).is_ok());
    }

    #[test]
    fn a_base_release_lists_the_dits_its_index_names_and_its_root_holds() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            &root.path(),
            "model_index.json",
            &base_index(&["transformer", "transformer_ref"]),
        );
        for dit in Dit::ALL {
            write(
                &root.path(),
                &format!("{}/config.json", dit.directory()),
                "{}",
            );
        }
        let layout = Layout::detect(root.path()).expect("a base release is recognized");
        assert_eq!(
            layout.dits,
            vec![
                (
                    Dit::Base,
                    vec![
                        VideoTask::TextToVideoAudio,
                        VideoTask::FirstLastToVideoAudio
                    ]
                ),
                (Dit::Reference, vec![VideoTask::ReferenceToVideoAudio]),
            ]
        );
        assert!(layout.contract.is_none());
        assert!(!layout.packed);
    }

    #[test]
    fn a_dit_the_index_names_but_the_root_lacks_is_not_a_layout() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            &root.path(),
            "model_index.json",
            &base_index(&["transformer", "transformer_ref"]),
        );
        write(&root.path(), "transformer/config.json", "{}");
        let layout = Layout::detect(root.path()).expect("the base DiT is present");
        assert_eq!(layout.dits.len(), 1);

        let empty = tempfile::tempdir().expect("tempdir");
        write(
            &empty.path(),
            "model_index.json",
            &base_index(&["transformer"]),
        );
        let error = Layout::detect(empty.path()).expect_err("no DiT is present");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("no DiT partition"))
        );
    }

    #[test]
    fn a_root_with_neither_contract_nor_index_is_refused_by_name() {
        let root = tempfile::tempdir().expect("tempdir");
        let error = Layout::detect(root.path()).expect_err("an empty root is not a checkpoint");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("model_index.json")),
            "{error}"
        );
    }

    #[test]
    fn a_dmd_export_states_its_rungs_and_serves_its_partitions_tasks() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            &root.path(),
            "fastvideo_inference.json",
            r#"{
                "task": "t2av",
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.8
            }"#,
        );
        let layout = Layout::detect(root.path()).expect("an export is recognized");
        assert_eq!(
            layout.dits,
            vec![(
                Dit::Base,
                vec![
                    VideoTask::TextToVideoAudio,
                    VideoTask::FirstLastToVideoAudio
                ]
            )]
        );
        let contract = layout.contract.expect("the export states a contract");
        assert_eq!(contract.task, VideoTask::TextToVideoAudio);
        assert_eq!(contract.dit, Dit::Base);
        assert_eq!(contract.video_shift, 10.0);
        assert_eq!(contract.audio_shift, 3.0);
        assert_eq!(
            contract.schedule,
            ContractSchedule::Rungs(vec![999, 874, 749, 624, 500, 375, 250, 125])
        );
        assert_eq!(
            contract.attention,
            SparseContract {
                tile: 64,
                sparsity: 0.8,
                reference_keep: None
            }
        );
        assert!(contract.base_revision.is_none());
    }

    #[test]
    fn an_omniref_export_states_its_block_grid_its_keep_rate_and_its_base_revision() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            &root.path(),
            "fastvideo_inference.json",
            r#"{
                "model_type": "ref2va",
                "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693",
                "video_scheduler_shift": 12.0,
                "audio_scheduler_shift": 3.0,
                "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
                "grid_max_t": 0.999,
                "vsa_tile_size": 128,
                "vsa_sparsity": 0.9,
                "vsa_ref_keep_rate": 0.1
            }"#,
        );
        let layout = Layout::detect(root.path()).expect("the export is recognized");
        assert_eq!(
            layout.dits,
            vec![(Dit::Reference, vec![VideoTask::ReferenceToVideoAudio])]
        );
        let contract = layout.contract.expect("the export states a contract");
        assert_eq!(contract.dit, Dit::Reference);
        assert_eq!(
            contract.base_revision.as_deref(),
            Some("hf://MiniMaxAI/MiniMax-H3@9bfb6693")
        );
        assert_eq!(
            contract.schedule,
            ContractSchedule::Nodes {
                nodes: vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
                max_t: 0.999
            }
        );
        assert_eq!(contract.attention.reference_keep, Some(0.1));
    }

    #[test]
    fn a_contract_is_refused_when_it_states_no_supported_task() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            &root.path(),
            "fastvideo_inference.json",
            r#"{"task": "t2v", "video_scheduler_shift": 1.0, "audio_scheduler_shift": 1.0,
                "dmd_denoising_steps": [10], "vsa_tile_size": 64, "vsa_sparsity": 0.0}"#,
        );
        let error = Layout::detect(root.path()).expect_err("t2v is not served");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("t2v")),
            "{error}"
        );
    }

    #[test]
    fn a_contract_states_exactly_one_schedule() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            &root.path(),
            "fastvideo_inference.json",
            r#"{"task": "t2av", "video_scheduler_shift": 1.0, "audio_scheduler_shift": 1.0,
                "vsa_tile_size": 64, "vsa_sparsity": 0.0}"#,
        );
        let error = Layout::detect(root.path()).expect_err("no schedule is stated");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("dmd_denoising_steps")),
            "{error}"
        );
    }

    #[test]
    fn the_sidecars_of_an_export_fall_back_to_the_base_revision() {
        let base = tempfile::tempdir().expect("tempdir");
        let export = tempfile::tempdir().expect("tempdir");
        for (relative, body) in [
            ("text_encoder/config.json", r#"{"from": "base"}"#),
            ("processor/preprocessor_config.json", r#"{"from": "base"}"#),
            ("vae/config.json", r#"{"from": "base"}"#),
            ("audio_vae/config.json", r#"{"from": "base"}"#),
            ("scheduler/scheduler_config.json", r#"{"shift": 12.0}"#),
            ("audio_scheduler/scheduler_config.json", r#"{"shift": 3.0}"#),
        ] {
            write(&base.path(), relative, body);
        }
        write(
            &export.path(),
            "transformer/config.json",
            r#"{"num_layers": 50}"#,
        );
        let layout = Layout {
            dits: vec![(Dit::Base, tasks_of(Dit::Base))],
            contract: None,
            packed: false,
        };
        let sidecars =
            Sidecars::read(export.path(), Some(base.path()), &layout).expect("the base fills in");
        assert_eq!(sidecars.text_encoder["from"], "base");
        assert_eq!(sidecars.dits[&Dit::Base]["num_layers"], 50);
        assert!(sidecars.video_processor.is_none());

        let error = Sidecars::read(export.path(), None, &layout)
            .err()
            .expect("without the base the text encoder is missing");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("text_encoder/config.json")),
            "{error}"
        );
    }
}
