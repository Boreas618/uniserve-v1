//! A recorded start: the kernels a deployment's main process launches, read from the start a
//! serving process makes, with nothing served (`uniserve packages`).
//!
//! [`record`] takes the launch a serving main process is started with and runs the first half of
//! each of its ranks' starts, to the runner's profile run, on the meta device
//! (`specs/engine-worker-port.md` §1.1, steps 4 to 8): the same family resolution, rank placement,
//! model construction, cache layout and runner construction, through the same function
//! ([`load_ranks`] onto [`Backing::Meta`]). It starts nothing the profile run does not read:
//!
//! - no daemon connection: the epoch is the one the launch states;
//! - no Storage Server: each weight's element type and extents come from the checkpoint's shard
//!   headers ([`nn::WeightLoader::meta`]), no payload is read and nothing is published;
//! - no device memory and no communicator: a profile run launches nothing, and a parallel group
//!   states its split alone;
//! - no session with the buffer pool, Build Server, scheduler or worker.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Result;
use proto::config::EngineArgs;

use super::ranks::{Backing, Meeting, load_ranks};
use crate::process::ranks::rank::ModelRunner;

/// Every kernel a main process launches: descriptor keys by operator name (`ops::Op::NAME`).
pub type Recorded = BTreeMap<String, BTreeSet<String>>;

/// Records the kernels the main process `launch` describes would launch: the union of its ranks'
/// profiled keys. Its checkpoint is served by the family of `registry` that claims the class it
/// declares.
///
/// # Errors
///
/// Fails where a serving start of the same launch fails before its device steps: the family, the
/// placement, the model's load from its shard headers, the cache's layout, the runner, or a
/// forward of the profile run.
pub fn record(launch: &EngineArgs, registry: &nn::family::Registry) -> Result<Recorded> {
    let family =
        crate::process::catalogue::resolve(registry, Path::new(&launch.engine_config.model.model))?;
    let model_runner = ModelRunner::of(launch, family)?;
    let ranks = load_ranks(model_runner.ranks()?, Backing::Meta, launch, Meeting::Local)?;

    let mut recorded = Recorded::new();
    for rank in &ranks {
        for (operator, key) in &rank.keys {
            recorded
                .entry((*operator).to_owned())
                .or_default()
                .insert(key.clone());
        }
        tracing::info!(
            rank = rank.model_runner.rank(),
            keys = rank.keys.len(),
            "the rank's profile run recorded its kernels"
        );
    }
    Ok(recorded)
}
