//! A recorded start: the programs a deployment's main process launches, read from the start a
//! serving process makes, with nothing served (`uniserve programs`).
//!
//! [`record`] takes the launch a serving main process is started with and runs its start to the
//! point where every device's worker has recorded its runners' walk ([`Worker::recorded`]): the
//! same family resolution, rank placement, capacity plan, model construction and runner
//! construction, through the same functions. It starts nothing the walk does not read:
//!
//! - no daemon connection: the epoch is the one the launch states ([`Epoch::of`]);
//! - no Storage Server: each weight's element type and extents come from the checkpoint's shard
//!   headers ([`nn::WeightLoader::meta`]), no payload is read and nothing is published, and the
//!   resident memory is storage of this process's own;
//! - no NCCL communicator: a parallel group states its split ([`nn::Group::recording`]), which is
//!   all a recording forward reads of it;
//! - no Program Server, scheduler, shadow region or warmup.
//!
//! The devices are used: the resident memory, the graph pools and the runners' columns are device
//! allocations, as they are when a serving process starts. Because no weight is mapped, the
//! capacity plan sees the whole device, so a cache the deployment does not bound is planned larger
//! than a serving process plans it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Result;
use proto::config::EngineArgs;
use worker::blocking::ThreadPool;
use worker::device::Worker;
use worker::runners::products::GroupProducts;

use super::launch::Epoch;
use super::ranks::{Backing, load_ranks};
use crate::rank::ModelRunner;

/// Every program a main process launches: descriptor keys by operator name (`ops::Op::NAME`).
pub type Recorded = BTreeMap<String, BTreeSet<String>>;

/// Records the programs the main process `launch` describes would launch, the union of its
/// devices' recorded walks, and releases everything it started. Its checkpoint is served by the
/// family of `catalogue` that claims the class it declares.
///
/// # Errors
///
/// Fails where a serving start of the same launch fails before its workers run: the family, the
/// placement, the model's load from its shard headers, the capacity plan, a runner, or a walk.
pub fn record(launch: &EngineArgs, catalogue: &crate::Catalogue) -> Result<Recorded> {
    let family = catalogue.resolve(Path::new(&launch.engine_config.model.model))?;
    let model_runner = ModelRunner::of(launch, family)?;
    let epoch = Epoch::of(launch)?;
    let mut ranks = load_ranks(model_runner.ranks()?, Backing::Recording, launch, &epoch)?;
    let blocking = ThreadPool::new(model_runner.blocking_limits());
    let products = GroupProducts::new(ranks.len());
    let count = ranks.len();

    let mut recorded = Recorded::new();
    let mut workers: Vec<Worker> = Vec::new();
    let mut walk = || -> Result<()> {
        for (index, rank) in ranks.iter_mut().enumerate() {
            let spawned =
                rank.spawn_workers((index, count), &products, epoch.generation, &blocking)?;
            for (device, worker, _) in spawned {
                for (operator, keys) in worker.recorded() {
                    recorded
                        .entry(operator.clone())
                        .or_default()
                        .extend(keys.iter().cloned());
                }
                tracing::info!(
                    rank = rank.model_runner.rank(),
                    device = device.0,
                    operators = worker.recorded().len(),
                    keys = worker.recorded().values().map(BTreeSet::len).sum::<usize>(),
                    "the device's runners recorded their walk"
                );
                workers.push(worker);
            }
        }
        Ok(())
    };
    let walked = walk();

    // Released in the order a serving process closes: the expert union's waits, the device
    // threads, the blocking lanes, then the resident owners the runners borrowed.
    if let Some((union, _)) = ranks.first().and_then(|rank| rank.model_runner.union()) {
        union.end();
    }
    for worker in workers {
        if let Err(error) = worker.close() {
            tracing::warn!(%error, "closing a device worker failed");
        }
    }
    for error in blocking.close() {
        tracing::warn!(%error, "closing a blocking lane failed");
    }
    for rank in ranks {
        if let Err(error) = rank.memory.close() {
            tracing::warn!(%error, "closing the resident owners failed");
        }
    }
    walked?;
    Ok(recorded)
}
