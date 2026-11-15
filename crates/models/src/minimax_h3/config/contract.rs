//! A FastH3 student's inference contract: `fastvideo_inference.json`, the file FastVideo publishes
//! beside every FastH3 student it releases, as an export's root holds it and as a deployment ships
//! it beside an adapter.
//!
//! The contract is the student's statement of what it is: the network it distils (its task), the
//! base release it continues when it holds only that network, its schedule and its attention.
//! The family composes a checkpoint by it ([`crate::minimax_h3::family`]) and reads the student's
//! denoising from it ([`super::Config::read`]).
//!
//! ## Schema `fasth3-inference-contract-v1`
//!
//! The released v1 contracts take two forms under the one `schema_version`, and both are the
//! schema's documented fields:
//!
//! | | `t2av` students (`FastVideo-FastH3-8-Step-V2`, `-4-step-Preview-v1-VSA-DataFree`) | `ref2va` students (`FastVideo-FastH3-Omni-8-Step-V1`, the OmniRef exports) |
//! | --- | --- | --- |
//! | Task | `task` | `model_type` |
//! | Network | the task's | `transformer_component`, which agrees with the task |
//! | Base release | none: the export holds the whole pipeline | `base_model_revision`, `hf://<repository>@<revision>` |
//! | Schedule | `dmd_denoising_steps` | `pdd_step_indices` with `grid_max_t` |
//!
//! A contract states its task under exactly one of the two keys and one schedule; both forms
//! state `attention_backend` (`DENSE`, or `VIDEO_SPARSE_ATTN_H3` with `vsa_tile_size`,
//! `vsa_sparsity` and an optional `vsa_ref_keep_rate`) and may state the shifts their schedule is
//! taken at (`video_scheduler_shift`, `audio_scheduler_shift`).

use nn::Source;
use nn::config::Fields;
use nn::family::Stage;
use serde_json::Value;

use super::{Component, VsaConfig, network};
use crate::minimax_h3::{Error, Result, VideoTask};

/// What a FastH3 student's contract states.
#[derive(Clone, Debug, PartialEq)]
pub struct Contract {
    /// The network the student distils, by the deployment stage that places it.
    pub network: Stage,
    /// The base release the student continues, `None` for a student that holds the whole
    /// pipeline.
    pub base: Option<BaseRevision>,
    /// The student's schedule.
    pub schedule: StudentSchedule,
    /// The video and audio scheduler shifts the contract states, each `None` when it states none.
    pub shifts: [Option<f64>; 2],
    /// The student's video sparse attention, `None` for dense attention.
    pub vsa: Option<VsaConfig>,
}

/// A Hugging Face repository at one commit, as a contract pins its base release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseRevision {
    /// The repository id, `<owner>/<name>`.
    pub repository: String,
    /// The commit.
    pub revision: String,
}

/// A student's schedule, as its contract states it.
#[derive(Clone, Debug, PartialEq)]
pub enum StudentSchedule {
    /// A DMD ladder: the training-schedule timesteps of its rungs.
    Ladder(Vec<u32>),
    /// A parallel-decoding block grid: its step indices and the grid's largest time.
    Blocks {
        /// The grid's step indices.
        nodes: Vec<u32>,
        /// The grid's largest time.
        grid_max_t: f64,
    },
}

impl Contract {
    /// The file a student's contract is published as.
    pub const FILE: &'static str = "fastvideo_inference.json";

    /// The schema version this reader reads.
    pub const SCHEMA: &'static str = "fasth3-inference-contract-v1";

    /// The contract the directory `source` holds, `None` when it holds none.
    ///
    /// # Errors
    ///
    /// When the file cannot be read or is not a contract this family serves ([`Self::parse`]).
    pub fn read(source: &Source) -> Result<Option<Self>> {
        source
            .sidecar(Self::FILE)?
            .map(|value| Self::parse(&value))
            .transpose()
    }

    /// The contract `value` states.
    ///
    /// # Errors
    ///
    /// When it states another schema, its task under neither key or both, a task no network of
    /// this family serves, a network that disagrees with its task, a malformed base revision,
    /// no schedule or two, or an attention backend this family does not serve.
    pub fn parse(value: &Value) -> Result<Self> {
        let fields = Fields::of(Self::FILE, value)?;

        let schema = fields.text("schema_version")?;
        if schema != Self::SCHEMA {
            return Err(Error::checkpoint(format!(
                "{} states the schema {schema:?}; this family reads {}",
                Self::FILE,
                Self::SCHEMA
            )));
        }

        let task = match (
            fields.optional_text("task")?,
            fields.optional_text("model_type")?,
        ) {
            (Some(task), None) | (None, Some(task)) => task,
            (Some(_), Some(_)) => {
                return Err(Error::checkpoint(format!(
                    "{} states its task twice, as `task` and as `model_type`",
                    Self::FILE
                )));
            }
            (None, None) => {
                return Err(Error::checkpoint(format!(
                    "{} states no task: a {} contract states it as `task` or as `model_type`",
                    Self::FILE,
                    Self::SCHEMA
                )));
            }
        };
        let network = match task {
            "t2av" => network(VideoTask::TextToVideoAudio),
            "ref2va" => network(VideoTask::ReferenceToVideoAudio),
            other => {
                return Err(Error::checkpoint(format!(
                    "unsupported FastH3 checkpoint: its task must be t2av or ref2va, got {other:?}"
                )));
            }
        };
        if let Some(component) = fields.optional_text("transformer_component")?
            && Component::named(component).and_then(Component::network) != Some(network)
        {
            return Err(Error::checkpoint(format!(
                "{} names the network `{component}`, and its task {task} is served by `{}`",
                Self::FILE,
                network.name()
            )));
        }

        let base = fields
            .optional_text("base_model_revision")?
            .map(BaseRevision::parse)
            .transpose()?;

        let schedule = match (
            fields.optional_indices("dmd_denoising_steps")?,
            fields.optional_indices("pdd_step_indices")?,
        ) {
            (Some(rungs), None) => StudentSchedule::Ladder(rungs),
            (None, Some(nodes)) => StudentSchedule::Blocks {
                nodes,
                grid_max_t: fields.number("grid_max_t")?,
            },
            (Some(_), Some(_)) => {
                return Err(Error::checkpoint(format!(
                    "{} states both dmd_denoising_steps and pdd_step_indices",
                    Self::FILE
                )));
            }
            (None, None) => {
                return Err(Error::checkpoint(format!(
                    "unsupported FastH3 checkpoint: {} states neither dmd_denoising_steps nor \
                     pdd_step_indices",
                    Self::FILE
                )));
            }
        };

        let vsa = match fields.text("attention_backend")? {
            "DENSE" => None,
            "VIDEO_SPARSE_ATTN_H3" => Some(VsaConfig::new(
                fields.index("vsa_tile_size")?,
                fields.number("vsa_sparsity")?,
                fields.optional_number("vsa_ref_keep_rate")?,
            )?),
            other => {
                return Err(Error::checkpoint(format!(
                    "{} states the attention backend {other:?}, which this build does not \
                     serve: it serves DENSE and VIDEO_SPARSE_ATTN_H3",
                    Self::FILE
                )));
            }
        };

        Ok(Self {
            network,
            base,
            schedule,
            shifts: [
                fields.optional_number("video_scheduler_shift")?,
                fields.optional_number("audio_scheduler_shift")?,
            ],
            vsa,
        })
    }
}

impl BaseRevision {
    /// The revision `pinned`, written `hf://<repository>@<revision>`.
    ///
    /// # Errors
    ///
    /// When `pinned` is not of that form with an `<owner>/<name>` repository and a revision.
    pub fn parse(pinned: &str) -> Result<Self> {
        let malformed = || {
            Error::checkpoint(format!(
                "{}'s base_model_revision must be hf://<owner>/<name>@<revision>, got {pinned:?}",
                Contract::FILE
            ))
        };
        let (repository, revision) = pinned
            .strip_prefix("hf://")
            .and_then(|pinned| pinned.rsplit_once('@'))
            .ok_or_else(malformed)?;
        let named = repository.split_once('/').is_some_and(|(owner, name)| {
            !owner.is_empty() && !name.is_empty() && !name.contains('/')
        });
        if !named || revision.is_empty() {
            return Err(malformed());
        }

        Ok(Self {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
        })
    }
}

impl std::fmt::Display for BaseRevision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "hf://{}@{}", self.repository, self.revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract of a reference student as the OmniRef exports publish it: its task as
    /// `model_type`, its network, its base and a block grid.
    fn reference() -> Value {
        serde_json::json!({
            "schema_version": Contract::SCHEMA,
            "model_type": "ref2va",
            "transformer_component": "transformer_ref",
            "base_model_revision":
                "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
            "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
            "grid_max_t": 0.999,
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "vsa_tile_size": 128,
            "vsa_sparsity": 0.9,
            "vsa_ref_keep_rate": 0.1,
        })
    }

    #[test]
    fn a_reference_student_states_its_network_and_the_base_it_continues() {
        let contract = Contract::parse(&reference()).unwrap();

        assert_eq!(contract.network, Stage::TransformerRef);
        assert_eq!(
            contract.base,
            Some(BaseRevision {
                repository: "MiniMaxAI/MiniMax-H3".to_owned(),
                revision: "9bfb6693f2cf6de171db46d1aa586f67d773a1da".to_owned(),
            })
        );
        assert!(matches!(
            contract.schedule,
            StudentSchedule::Blocks { ref nodes, .. } if nodes.len() == 9
        ));
        assert_eq!(
            contract.vsa,
            Some(VsaConfig::new(128, 0.9, Some(0.1)).unwrap())
        );
    }

    /// A v1 contract states its task under `task` or under `model_type`: one of the two, and a
    /// network that agrees with it.
    #[test]
    fn a_contract_states_its_task_once_and_its_network_agrees() {
        let mut both = reference();
        both["task"] = serde_json::json!("ref2va");
        let error = Contract::parse(&both).unwrap_err();
        assert!(error.to_string().contains("twice"), "{error}");

        let mut neither = reference();
        neither.as_object_mut().unwrap().remove("model_type");
        let error = Contract::parse(&neither).unwrap_err();
        assert!(error.to_string().contains("states no task"), "{error}");

        let mut disagreeing = reference();
        disagreeing["transformer_component"] = serde_json::json!("transformer");
        let error = Contract::parse(&disagreeing).unwrap_err();
        assert!(error.to_string().contains("`transformer`"), "{error}");
    }

    /// Another schema, an unpinned base and an unstated attention are refused rather than read
    /// as something they do not state.
    #[test]
    fn a_contract_outside_its_schema_is_refused() {
        let mut later = reference();
        later["schema_version"] = serde_json::json!("fasth3-inference-contract-v2");
        let error = Contract::parse(&later).unwrap_err();
        assert!(error.to_string().contains("v2"), "{error}");

        let mut unpinned = reference();
        unpinned["base_model_revision"] = serde_json::json!("MiniMaxAI/MiniMax-H3");
        let error = Contract::parse(&unpinned).unwrap_err();
        assert!(error.to_string().contains("base_model_revision"), "{error}");

        let mut unstated = reference();
        unstated
            .as_object_mut()
            .unwrap()
            .remove("attention_backend");
        assert!(Contract::parse(&unstated).is_err());
    }
}
