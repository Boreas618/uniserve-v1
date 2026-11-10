//! The H3 checkpoint's layout: which DiT partitions it holds, which tasks each serves, the FastH3
//! contract that states a student's schedule and sparse attention, and the JSON sidecars the
//! loading boundary reads.
//!
//! A checkpoint is one of two things (`docs/minimax_h3/minimax_h3.md`):
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

use nn::config::Fields;
use serde_json::Value;

use crate::VideoTask;

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
    /// The student's attention backend, as the export states it.
    pub backend: AttentionBackend,
    /// The student's sparse attention, stated exactly when the backend is sparse.
    pub attention: Option<SparseContract>,
}

/// The attention backend a FastH3 export states for its student.
///
/// The checkpoints say so themselves (`attention_backend`), and it is the fact that decides whether a
/// dense or a tiled attention is served: inferring it from the presence of the sparse knobs would make a
/// student that states dense attention but keeps its distillation's `vsa_*` values read as sparse, and the
/// inference would be invisible until a device disagreed with it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttentionBackend {
    /// Full bidirectional attention over the packed row set: what the base release runs.
    Dense,
    /// VSA sparse attention over spatiotemporal tiles.
    VideoSparseH3,
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
/// an export states.
#[derive(Clone, PartialEq, Debug)]
pub struct Layout {
    /// Each DiT partition the checkpoint holds, with the tasks it serves in canonical order.
    pub dits: Vec<(Dit, Vec<VideoTask>)>,
    /// The FastH3 contract, for an export that states one.
    pub contract: Option<Contract>,
}

impl Layout {
    /// Classify `checkpoint`.
    ///
    /// A FastH3 export is recognized by its contract; anything else by a pipeline index and the
    /// DiT directories it names and the checkpoint holds.
    pub fn detect(checkpoint: &nn::Checkpoint) -> Result<Self> {
        let root = checkpoint.root().display();
        if let Some(contract) = Contract::read(checkpoint)? {
            let dits = vec![(contract.dit, tasks_of(contract.dit))];
            return Ok(Self {
                dits,
                contract: Some(contract),
            });
        }
        // A base release with a FastH3 adapter fused into its base partition is that student: it
        // serves the partition it was distilled from on the student's contract.
        if let Some(fusion) = checkpoint
            .component(Dit::Base.directory())
            .ok()
            .and_then(nn::Checkpoint::adapter)
        {
            let contract = Contract::of_adapter(fusion, checkpoint)?;
            return Ok(Self {
                dits: vec![(contract.dit, tasks_of(contract.dit))],
                contract: Some(contract),
            });
        }

        let index = checkpoint.sidecar("model_index.json")?.ok_or_else(|| {
            Error::checkpoint(format!(
                "{root} holds neither fastvideo_inference.json nor model_index.json"
            ))
        })?;
        let index = Fields::of("model_index.json", &index)?;
        if let Some(class) = index.optional_text("_class_name")?
            && class != PIPELINE_CLASS
        {
            return Err(Error::checkpoint(format!(
                "{root} is a {class} checkpoint, not a {PIPELINE_CLASS} pipeline root; the task \
                 exports under FL2VA/ are not pipeline roots"
            )));
        }
        let mut dits = Vec::new();
        for dit in Dit::ALL {
            if index.contains(dit.directory())
                && checkpoint
                    .sidecar(&format!("{}/config.json", dit.directory()))?
                    .is_some()
            {
                dits.push((dit, tasks_of(dit)));
            }
        }
        if dits.is_empty() {
            return Err(Error::checkpoint(format!(
                "{root}/model_index.json names no DiT partition the checkpoint holds"
            )));
        }
        Ok(Self {
            dits,
            contract: None,
        })
    }
}

impl Contract {
    /// Read `checkpoint`'s contract, or `None` when it states none.
    pub fn read(checkpoint: &nn::Checkpoint) -> Result<Option<Self>> {
        match checkpoint.sidecar("fastvideo_inference.json")? {
            None => Ok(None),
            Some(value) => Self::parse(&Fields::of("fastvideo_inference.json", &value)?).map(Some),
        }
    }

    /// Parse a contract's JSON object.
    pub fn parse(object: &Fields<'_>) -> Result<Self> {
        // A student states its task as `task` (`t2av` for text-to-video-and-audio) or, in the
        // OmniRef exports, as `model_type`; a task no partition serves is refused.
        let stated = match object.optional_text("task")? {
            Some(task) => task,
            None => object.optional_text("model_type")?.ok_or_else(|| {
                Error::checkpoint(
                    "fastvideo_inference.json states no task or model_type".to_owned(),
                )
            })?,
        };
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
        let base_revision = object
            .optional_text("base_model_revision")?
            .map(str::to_owned);
        // The shifts are FP32 quantities of the schedule, read in the FP64 the sidecar writes.
        let video_shift = object.number("video_scheduler_shift")? as f32;
        let audio_shift = object.number("audio_scheduler_shift")? as f32;
        let schedule = Self::schedule(object)?;
        // The backend is the export's own statement. A contract that names none attends densely: the
        // sparse knobs are the distillation's, and a dense student need carry none of them.
        let backend = match object.optional_text("attention_backend")? {
            None | Some("DENSE") => AttentionBackend::Dense,
            Some("VIDEO_SPARSE_ATTN_H3") => AttentionBackend::VideoSparseH3,
            Some(other) => {
                return Err(Error::checkpoint(format!(
                    "fastvideo_inference.json states the attention backend {other:?}, which this \
                     build does not serve: it serves DENSE and VIDEO_SPARSE_ATTN_H3"
                )));
            }
        };
        // The sparsity stays in the FP64 the sidecar writes: its `ceil((1 - sparsity) * tiles)` is
        // a tile count, and FP32 0.9 widened back is greater than 0.9, so a ten-tile segment would
        // keep two tiles instead of one.
        let attention = match backend {
            AttentionBackend::Dense => None,
            AttentionBackend::VideoSparseH3 => Some(SparseContract {
                tile: object.index("vsa_tile_size")?,
                sparsity: object.number("vsa_sparsity")?,
                reference_keep: object.optional_number("vsa_ref_keep_rate")?,
            }),
        };
        Ok(Self {
            task,
            dit,
            base_revision,
            video_shift,
            audio_shift,
            schedule,
            backend,
            attention,
        })
    }

    /// The contract of the FastH3 adapter `fusion`, fused into `checkpoint`'s base partition.
    ///
    /// A FastH3 adapter ([`fasth3_adapter`]) is the rank-64 extraction of a FastH3 Preview v1
    /// student from the base release, and it ships no `fastvideo_inference.json`: its contract is
    /// the one that release's full exports state, the four DMD rungs `[999, 749, 500, 250]` of the
    /// `t2av` partition at the base release's own scheduler shifts. An adapter that assigns the
    /// layers' VSA compression gates (`attn.to_gate_compress.weight`, which the base does not
    /// hold) is the VSA student, distilled at 64-row tiles and 0.9 sparsity
    /// (`FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`); one that assigns none is the dense
    /// student.
    ///
    /// # Errors
    ///
    /// Fails when the scheduler sidecars cannot be read.
    pub fn of_adapter(fusion: &nn::Fusion, checkpoint: &nn::Checkpoint) -> Result<Self> {
        let sidecar = |relative: &str| -> Result<f32> {
            let value = checkpoint.sidecar(relative)?.ok_or_else(|| {
                Error::checkpoint(format!(
                    "no root of the checkpoint at {} holds {relative}",
                    checkpoint.root().display()
                ))
            })?;
            crate::config::shift(&value, relative)
        };
        let gated = fusion
            .assigned()
            .any(|name| name.ends_with(".attn.to_gate_compress.weight"));
        let (backend, attention) = if gated {
            (
                AttentionBackend::VideoSparseH3,
                Some(SparseContract {
                    tile: 64,
                    sparsity: 0.9,
                    reference_keep: None,
                }),
            )
        } else {
            (AttentionBackend::Dense, None)
        };
        Ok(Self {
            task: VideoTask::TextToVideoAudio,
            dit: Dit::Base,
            base_revision: None,
            video_shift: sidecar("scheduler/scheduler_config.json")?,
            audio_shift: sidecar("audio_scheduler/scheduler_config.json")?,
            schedule: ContractSchedule::Rungs(vec![999, 749, 500, 250]),
            backend,
            attention,
        })
    }

    /// The schedule the contract states: a DMD ladder or a parallel-decoding block grid.
    fn schedule(object: &Fields<'_>) -> Result<ContractSchedule> {
        match (
            object.optional_indices("dmd_denoising_steps")?,
            object.optional_indices("pdd_step_indices")?,
        ) {
            (Some(rungs), None) => Ok(ContractSchedule::Rungs(rungs)),
            (None, Some(nodes)) => Ok(ContractSchedule::Nodes {
                nodes,
                max_t: object.number("grid_max_t")? as f32,
            }),
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

/// The base model a FastH3 adapter states it edits.
pub const FASTH3_BASE_MODEL: &str = "MiniMaxAI/MiniMax-H3";

/// Whether an adapter's header `metadata` identifies a FastH3 release.
///
/// FastVideo writes its container for ordinary H3 adapters too, so the container alone does not
/// claim one: the adapter must name a student of FastVideo's own namespace that is FastH3
/// (`finetuned_model`, such as `FastVideo/FastVideo-FastH3-4-step-v1`) over the base release
/// (`base_model`). Claiming another adapter would put it on a four-step schedule it was never
/// distilled for.
pub fn fasth3_adapter(metadata: &BTreeMap<String, String>) -> bool {
    let student = metadata
        .get("finetuned_model")
        .map(|name| name.to_lowercase())
        .unwrap_or_default();
    let base = metadata.get("base_model").map(String::as_str);
    metadata.get("format").map(String::as_str) == Some(nn::adapter::FASTVIDEO_FORMAT)
        && student.starts_with("fastvideo/")
        && student.contains("fasth3")
        && base.is_some_and(|base| base.eq_ignore_ascii_case(FASTH3_BASE_MODEL))
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
    /// Read every sidecar `layout` needs from `checkpoint`.
    ///
    /// A pipeline's components do not have to lie in one directory: an export that continues a base
    /// release holds its own DiT partition and schedulers, and the base holds the text encoder, the
    /// processor and the VAEs, which the engine opens as one checkpoint. Each sidecar is read from
    /// the first root that holds it, the export's before the base's, and a sidecar no root holds is
    /// an error naming it.
    pub fn read(checkpoint: &nn::Checkpoint, layout: &Layout) -> Result<Self> {
        let read = |relative: &str| -> Result<Value> {
            checkpoint.sidecar(relative)?.ok_or_else(|| {
                Error::checkpoint(format!(
                    "no root of the checkpoint at {} holds {relative}",
                    checkpoint.root().display()
                ))
            })
        };
        let text_encoder = read("text_encoder/config.json")?;
        let processor = read("processor/preprocessor_config.json")?;
        let video_processor = checkpoint.sidecar("processor/video_preprocessor_config.json")?;
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use crate::VideoTask;

    use super::*;

    fn write(root: &Path, relative: &str, body: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("a sidecar has a parent")).expect("create");
        fs::write(&path, body).expect("write");
    }

    /// The checkpoint `roots` hold, first root first.
    ///
    /// A layout is read from sidecars alone, so each root is given one empty shard set of its own,
    /// under a directory no sidecar names, which is what makes it a checkpoint to open.
    fn open(roots: &[&Path]) -> nn::Checkpoint {
        for root in roots {
            let shards = root.join("weights");
            fs::create_dir_all(&shards).expect("a shard directory");
            // A safetensors file is its header's length, the header, and no payload.
            let header = b"{}      ";
            let mut file = (header.len() as u64).to_le_bytes().to_vec();
            file.extend_from_slice(header);
            fs::write(shards.join("model.safetensors"), file).expect("an empty shard");
        }
        nn::Checkpoint::open_roots(roots).expect("the roots open")
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
            root.path(),
            "model_index.json",
            r#"{"_class_name": "MiniMaxH3Pipeline", "transformer": ["diffusers", "Model"]}"#,
        );
        write(root.path(), "transformer/config.json", "{}");
        write(root.path(), "video_vae/config.json", "{}");
        let error = Layout::detect(&open(&[root.path()])).expect_err("a task export is refused");
        assert!(
            matches!(error, Error::Checkpoint(ref problem)
                if problem.contains("MiniMaxH3Pipeline") && problem.contains("pipeline root")),
            "{error}"
        );

        // The modular pipeline's own class name is taken, and an index that states none is not
        // refused for it: a hand-built export need not name one.
        write(
            root.path(),
            "model_index.json",
            &format!(
                r#"{{"_class_name": "{PIPELINE_CLASS}", "transformer": ["diffusers", "Model"]}}"#
            ),
        );
        assert!(Layout::detect(&open(&[root.path()])).is_ok());
    }

    #[test]
    fn a_base_release_lists_the_dits_its_index_names_and_its_root_holds() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "model_index.json",
            &base_index(&["transformer", "transformer_ref"]),
        );
        for dit in Dit::ALL {
            write(
                root.path(),
                &format!("{}/config.json", dit.directory()),
                "{}",
            );
        }
        let layout = Layout::detect(&open(&[root.path()])).expect("a base release is recognized");
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
    }

    #[test]
    fn a_dit_the_index_names_but_the_root_lacks_is_not_a_layout() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "model_index.json",
            &base_index(&["transformer", "transformer_ref"]),
        );
        write(root.path(), "transformer/config.json", "{}");
        let layout = Layout::detect(&open(&[root.path()])).expect("the base DiT is present");
        assert_eq!(layout.dits.len(), 1);

        let empty = tempfile::tempdir().expect("tempdir");
        write(
            empty.path(),
            "model_index.json",
            &base_index(&["transformer"]),
        );
        let error = Layout::detect(&open(&[empty.path()])).expect_err("no DiT is present");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("no DiT partition"))
        );
    }

    #[test]
    fn a_root_with_neither_contract_nor_index_is_refused_by_name() {
        let root = tempfile::tempdir().expect("tempdir");
        let error =
            Layout::detect(&open(&[root.path()])).expect_err("an empty root is not a checkpoint");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("model_index.json")),
            "{error}"
        );
    }

    #[test]
    fn a_dmd_export_states_its_rungs_and_serves_its_partitions_tasks() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "fastvideo_inference.json",
            r#"{
                "task": "t2av",
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.8
            }"#,
        );
        let layout = Layout::detect(&open(&[root.path()])).expect("an export is recognized");
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
        assert_eq!(contract.backend, AttentionBackend::VideoSparseH3);
        assert_eq!(
            contract.attention,
            Some(SparseContract {
                tile: 64,
                sparsity: 0.8,
                reference_keep: None
            })
        );
        assert!(contract.base_revision.is_none());
    }

    #[test]
    fn an_export_states_its_attention_backend_and_an_unknown_one_is_refused() {
        let dense: serde_json::Value = serde_json::from_str(
            r#"{
                "task": "t2av",
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "dmd_denoising_steps": [10],
                "attention_backend": "DENSE",
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.9
            }"#,
        )
        .expect("json");
        let contract =
            Contract::parse(&Fields::of("fastvideo_inference.json", &dense).expect("object"))
                .expect("a dense export");
        assert_eq!(contract.backend, AttentionBackend::Dense);
        // The knobs are the distillation's, not the backend's: a dense student carries none.
        assert_eq!(contract.attention, None);

        // A contract that names no backend attends densely, which is what a base release does.
        let silent: serde_json::Value = serde_json::from_str(
            r#"{
                "task": "t2av",
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "dmd_denoising_steps": [10]
            }"#,
        )
        .expect("json");
        let contract =
            Contract::parse(&Fields::of("fastvideo_inference.json", &silent).expect("object"))
                .expect("a base release");
        assert_eq!(contract.backend, AttentionBackend::Dense);
        assert_eq!(contract.attention, None);

        let unknown: serde_json::Value = serde_json::from_str(
            r#"{
                "task": "t2av",
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "dmd_denoising_steps": [10],
                "attention_backend": "FLEX_ATTENTION"
            }"#,
        )
        .expect("json");
        let error =
            Contract::parse(&Fields::of("fastvideo_inference.json", &unknown).expect("object"))
                .expect_err("an unserved backend is refused by name");
        assert!(error.to_string().contains("FLEX_ATTENTION"), "{error}");
    }

    #[test]
    fn an_omniref_export_states_its_block_grid_its_keep_rate_and_its_base_revision() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "fastvideo_inference.json",
            r#"{
                "model_type": "ref2va",
                "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693",
                "video_scheduler_shift": 12.0,
                "audio_scheduler_shift": 3.0,
                "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
                "grid_max_t": 0.999,
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "vsa_tile_size": 128,
                "vsa_sparsity": 0.9,
                "vsa_ref_keep_rate": 0.1
            }"#,
        );
        let layout = Layout::detect(&open(&[root.path()])).expect("the export is recognized");
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
        let SparseContract {
            tile,
            sparsity,
            reference_keep,
        } = contract
            .attention
            .expect("an OmniRef export states its sparse attention");
        assert_eq!((tile, sparsity, reference_keep), (128, 0.9, Some(0.1)));
    }

    /// Write a safetensors file of `tensors`, each `(name, extents)` of BF16 zeros, under the
    /// header metadata `metadata`.
    fn safetensors(path: &Path, metadata: &[(&str, &str)], tensors: &[(&str, &[u64])]) {
        let mut entries = vec![format!(
            r#""__metadata__":{{{}}}"#,
            metadata
                .iter()
                .map(|(key, value)| format!(r#""{key}":"{value}""#))
                .collect::<Vec<_>>()
                .join(",")
        )];
        let mut bytes = 0;
        for (name, shape) in tensors {
            let length = shape.iter().product::<u64>() * 2;
            let extents = shape.iter().map(u64::to_string).collect::<Vec<_>>();
            entries.push(format!(
                r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{bytes},{}]}}"#,
                extents.join(","),
                bytes + length
            ));
            bytes += length;
        }
        let mut header = format!("{{{}}}", entries.join(","));
        header.push_str(&" ".repeat(header.len().div_ceil(8) * 8 - header.len()));
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.resize(file.len() + bytes as usize, 0);
        fs::write(path, file).expect("write");
    }

    /// A base release with a FastH3 adapter fused into its `transformer` reads as that student:
    /// the four-step DMD ladder at the base's shifts, sparse exactly when the adapter assigns the
    /// compression gates, serving the base partition's tasks only.
    #[test]
    fn a_fasth3_adapter_over_the_base_release_is_its_student() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "model_index.json",
            &base_index(&["transformer", "transformer_ref"]),
        );
        for dit in Dit::ALL {
            write(
                root.path(),
                &format!("{}/config.json", dit.directory()),
                "{}",
            );
            safetensors(
                &root.path().join(dit.directory()).join("model.safetensors"),
                &[],
                &[("blocks.0.attn.to_q.weight", &[4, 4])],
            );
        }
        write(
            root.path(),
            "scheduler/scheduler_config.json",
            r#"{"shift": 12.0}"#,
        );
        write(
            root.path(),
            "audio_scheduler/scheduler_config.json",
            r#"{"shift": 3.0}"#,
        );

        let adapters = tempfile::tempdir().expect("tempdir");
        fn student(gates: &str) -> [(&str, &str); 7] {
            [
                ("format", "fastvideo-lora-v2"),
                ("finetuned_model", "FastVideo/FastVideo-FastH3-4-step-v1"),
                ("base_model", "MiniMaxAI/MiniMax-H3"),
                ("rank", "2"),
                ("low_rank_tensors", "2"),
                ("diff_tensors", "0"),
                ("set_weight_tensors", gates),
            ]
        }
        let low_rank: [(&str, &[u64]); 2] = [
            ("blocks.0.attn.to_q.lora_A.weight", &[2, 4]),
            ("blocks.0.attn.to_q.lora_B.weight", &[4, 2]),
        ];
        let gate: (&str, &[u64]) = ("blocks.0.attn.to_gate_compress.set_weight", &[4, 4]);

        let detect = |metadata: &[(&str, &str)], tensors: &[(&str, &[u64])]| {
            let file = adapters.path().join("adapter_model.safetensors");
            safetensors(&file, metadata, tensors);
            let adapter = nn::Adapter::open(&file)?;
            let checkpoint = nn::Checkpoint::open(root.path())?;
            let nn::family::Model::Video(family) = crate::family::FAMILY.model else {
                unreachable!("MiniMax-H3 is a video family")
            };
            let component = family.adapter_component(&checkpoint, &adapter)?;
            assert_eq!(component, Some("transformer"));
            Layout::detect(&checkpoint.attach(component, adapter)?)
                .map_err(|error| nn::Error::Unsupported(error.to_string()))
        };

        let mut sparse = low_rank.to_vec();
        sparse.push(gate);
        let layout = detect(&student("1"), &sparse).expect("the VSA student");
        assert_eq!(layout.dits, vec![(Dit::Base, tasks_of(Dit::Base))]);
        let contract = layout
            .contract
            .expect("an adapter states the student's contract");
        assert_eq!(
            contract.schedule,
            ContractSchedule::Rungs(vec![999, 749, 500, 250])
        );
        assert_eq!((contract.video_shift, contract.audio_shift), (12.0, 3.0));
        assert_eq!(contract.backend, AttentionBackend::VideoSparseH3);
        assert_eq!(
            contract.attention,
            Some(SparseContract {
                tile: 64,
                sparsity: 0.9,
                reference_keep: None
            })
        );

        let dense = detect(&student("0"), &low_rank).expect("the dense student");
        let contract = dense.contract.expect("a contract");
        assert_eq!(contract.backend, AttentionBackend::Dense);
        assert_eq!(contract.attention, None);

        // FastVideo's container over another model is not claimed as FastH3.
        let mut other = student("0");
        other[1] = ("finetuned_model", "someone/H3-style-lora");
        let error = detect(&other, &low_rank).expect_err("not a FastH3 release");
        assert!(
            error.to_string().contains("not a FastH3 adapter"),
            "{error}"
        );
    }

    #[test]
    fn a_contract_is_refused_when_it_states_no_supported_task() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "fastvideo_inference.json",
            r#"{"task": "t2v", "video_scheduler_shift": 1.0, "audio_scheduler_shift": 1.0,
                "dmd_denoising_steps": [10], "vsa_tile_size": 64, "vsa_sparsity": 0.0}"#,
        );
        let error = Layout::detect(&open(&[root.path()])).expect_err("t2v is not served");
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("t2v")),
            "{error}"
        );
    }

    #[test]
    fn a_contract_states_exactly_one_schedule() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "fastvideo_inference.json",
            r#"{"task": "t2av", "video_scheduler_shift": 1.0, "audio_scheduler_shift": 1.0,
                "vsa_tile_size": 64, "vsa_sparsity": 0.0}"#,
        );
        let error = Layout::detect(&open(&[root.path()])).expect_err("no schedule is stated");
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
            write(base.path(), relative, body);
        }
        write(
            export.path(),
            "transformer/config.json",
            r#"{"num_layers": 50}"#,
        );
        let layout = Layout {
            dits: vec![(Dit::Base, tasks_of(Dit::Base))],
            contract: None,
        };
        let sidecars = Sidecars::read(&open(&[export.path(), base.path()]), &layout)
            .expect("the base fills in");
        assert_eq!(sidecars.text_encoder["from"], "base");
        assert_eq!(sidecars.dits[&Dit::Base]["num_layers"], 50);
        assert!(sidecars.video_processor.is_none());

        // Without the base the export holds no text encoder: the reader says which sidecar it
        // looked for. `Sidecars` is not `Debug`, so the error is matched rather than unwrapped.
        let error = match Sidecars::read(&open(&[export.path()]), &layout) {
            Ok(_) => panic!("without the base the text encoder is missing"),
            Err(error) => error,
        };
        assert!(
            matches!(error, Error::Checkpoint(ref problem) if problem.contains("text_encoder/config.json")),
            "{error}"
        );
    }
}
