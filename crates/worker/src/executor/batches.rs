//! `Executors`: the engine's `ipc::execution::Batches` over a node's executors and its peers.
//!
//! A scheduler addresses a batch by its worker group and phase, not by device. A node's devices
//! are each a rank of one worker group and hold the runners of the stages placed on them; this
//! is the one place a batch is routed to them, so a scheduler never needs to know how many
//! devices it drives:
//!
//! - A phase one device of the group serves is that device's. A phase several devices serve is
//!   a parallel group's: each runs its part of one batch, so each is submitted the batch, and
//!   the submissions are joined into one that completes when every rank's did.
//! - A group with ranks on other hosts runs its batches there too, through the peer link of
//!   each engine process that holds them ([`Remote`]). A group whose ranks are all another
//!   host's is served through its links alone.
//! - A video decode or encode is one unit of a timeline and needs no other unit, so the devices
//!   that serve the phase take the units in turn instead of each computing the same one.
//!
//! The scheduler's thread is the one thread that submits, so every executor's queue holds the
//! group's batches in one order, which is the order their collectives need. The groups and
//! phases are fixed when the executors start: the route holds no lock and makes no retry.

use std::sync::Arc;

use ipc::PhaseId;
use ipc::execution::{Batch, Batches, Computation, Extent, Fault, Inventory, Submission};

use super::peers::Remote;
use super::{Executor, Operation};
use crate::buckets::Bucket;
use crate::{Error, Result};

/// A node's executors and peer links, as the scheduler submits batches to them.
pub struct Executors {
    /// The executors, each with the worker group it is a rank of.
    executors: Vec<(String, Arc<Executor>)>,
    /// The engine processes of other hosts that hold ranks of these groups, each its peer link
    /// with the groups it holds ranks of.
    remotes: Vec<(Vec<String>, Arc<Remote>)>,
    /// The pools this node serves, as the deployment launched them: what the scheduler builds
    /// its view of the workers from. A scheduler refuses a node that states none.
    inventory: Vec<Inventory>,
}

impl Executors {
    /// The route over `executors`, each with the worker group it is a rank of, and `remotes`,
    /// each with the groups whose ranks its engine process holds; `inventory` states the pools
    /// they serve.
    ///
    /// The executors are shared with the control thread, which reads their jobs in flight and
    /// their counters through the same handles.
    pub fn new(
        executors: Vec<(String, Arc<Executor>)>,
        remotes: Vec<(Vec<String>, Arc<Remote>)>,
        inventory: Vec<Inventory>,
    ) -> Self {
        Self {
            executors,
            remotes,
            inventory,
        }
    }

    /// The executors of `group` whose runners serve `phase`, by their place among this node's.
    fn serving(&self, group: &str, phase: PhaseId) -> Vec<(usize, &Arc<Executor>)> {
        self.executors
            .iter()
            .enumerate()
            .filter(|(_, (held, executor))| held == group && executor.serves(phase))
            .map(|(index, (_, executor))| (index, executor))
            .collect()
    }

    /// The peer links of the processes that hold ranks of `group`, by their place after this
    /// node's executors.
    fn remotes_of(&self, group: &str) -> Vec<(usize, &Arc<Remote>)> {
        let executors = self.executors.len();

        self.remotes
            .iter()
            .enumerate()
            .filter(|(_, (groups, _))| groups.iter().any(|held| held == group))
            .map(|(index, (_, remote))| (executors + index, remote))
            .collect()
    }

    /// The refusal of `batch` when neither an executor of this node nor a peer serves it.
    fn unrouted(&self, batch: &Batch) -> Error {
        let served: Vec<PhaseId> = self
            .executors
            .iter()
            .filter(|(group, _)| *group == batch.group)
            .flat_map(|(_, executor)| executor.phases().iter().copied())
            .collect();

        route_refusal(&batch.group, batch.phase, &served)
    }

    /// The one executor a unit-measured call runs on, when `batch` is one, with its place among
    /// this node's executors.
    ///
    /// A video decode is one unit of the timeline, so the devices that serve the phase take the
    /// units in turn. A video encode follows its unit: the device that decoded a clip encodes
    /// it.
    fn executor_of_unit(&self, batch: &Batch) -> Result<Option<(usize, &Arc<Executor>)>> {
        let Some(media) = &batch.media else {
            return Ok(None);
        };
        if !matches!(
            media.computation.call,
            ipc::MediaCall::VideoDecoding | ipc::MediaCall::VideoEncoding
        ) {
            return Ok(None);
        }

        let serving = self.serving(&batch.group, batch.phase);
        if serving.is_empty() {
            return Err(self.unrouted(batch));
        }
        let unit = media.request(0)?.unit as usize;
        Ok(Some(serving[unit % serving.len()]))
    }

    /// Submits `batch` to every executor of its group that serves its phase (`submit`) and to
    /// the group's ranks on other hosts (`remote`), as one submission.
    fn on_every_rank(
        &self,
        batch: Batch,
        submit: impl Fn(usize, &Executor, Batch) -> Result<Submission>,
        remote: impl Fn(&Remote, Batch) -> Result<Submission>,
    ) -> Result<Submission> {
        let executors = self.serving(&batch.group, batch.phase);
        let peers = self.remotes_of(&batch.group);
        if executors.is_empty() && peers.is_empty() {
            return Err(self.unrouted(&batch));
        }

        // The other hosts' ranks take the batch first: it travels while this node's executors
        // prepare their inputs, and the ranks' collectives meet in the batch's forward.
        let mut parts = Vec::with_capacity(executors.len() + peers.len());
        for (_, peer) in peers {
            parts.push(remote(peer, batch.clone())?);
        }
        for (index, executor) in executors {
            parts.push(submit(index, executor, batch.clone())?);
        }

        // The last part is the submission the others join: this node's last executor, or the
        // last peer of a group whose ranks are all another host's.
        let last = parts.pop().ok_or_else(|| self.unrouted(&batch))?;
        Ok(last.joined(parts))
    }

    /// The bucket the executor serving `phase` would serve `computation` in, or `None` when it
    /// states no bucket for it. A media deployment's devices are one group, so the first
    /// executor that serves the phase is asked.
    ///
    /// # Errors
    ///
    /// When no executor of this node serves `phase`.
    pub fn media(&self, phase: PhaseId, computation: &Computation) -> Result<Option<Bucket>> {
        let executor = self
            .executors
            .iter()
            .map(|(_, executor)| executor)
            .find(|executor| executor.serves(phase))
            .ok_or_else(|| route_refusal("", phase, &[]))?;

        Ok(executor.media(computation))
    }

    /// The executors `batch` runs on, by their place among this node's executors and, after
    /// them, its peer links: a group whose ranks are all another host's runs on its links.
    ///
    /// # Errors
    ///
    /// When nothing here serves the batch's group and phase.
    pub fn devices(&self, batch: &Batch) -> Result<Vec<usize>> {
        // The same two routes `submit` takes.
        if let Some((index, _)) = self.executor_of_unit(batch)? {
            return Ok(vec![index]);
        }

        let mut devices: Vec<usize> = self
            .serving(&batch.group, batch.phase)
            .into_iter()
            .map(|(index, _)| index)
            .collect();
        if devices.is_empty() {
            devices = self
                .remotes_of(&batch.group)
                .into_iter()
                .map(|(index, _)| index)
                .collect();
        }
        if devices.is_empty() {
            return Err(self.unrouted(batch));
        }
        Ok(devices)
    }

    /// How many executors and peer links run this node's batches, one queue each.
    pub fn device_count(&self) -> usize {
        self.executors.len() + self.remotes.len()
    }

    /// Submits `batch` to the executors of its group and phase and to the group's ranks on
    /// other hosts, or to its unit's one executor.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when nothing here serves the batch's group and phase, or what a queue
    /// or a peer link refuses.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        if let Some((_, executor)) = self.executor_of_unit(&batch)? {
            return executor.submit(batch);
        }

        self.on_every_rank(
            batch,
            |index, executor, mut batch| {
                // A page copy names the rank that makes it: its place among the group's ranks on
                // this node.
                let group = &self.executors[index].0;
                let rank = self.executors[..index]
                    .iter()
                    .filter(|(held, _)| held == group)
                    .count();
                batch.page_loads.retain(|copy| copy.rank as usize == rank);
                batch
                    .page_offloads
                    .retain(|copy| copy.rank as usize == rank);

                executor.submit(batch)
            },
            Remote::submit,
        )
    }

    /// Submits the shadow batch `batch` to every rank of its group and phase.
    ///
    /// # Errors
    ///
    /// As [`Executors::submit`].
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.on_every_rank(
            batch,
            |_, executor, batch| executor.shadow(batch),
            Remote::shadow,
        )
    }

    /// Submits one control operation to every executor of this node, as one submission:
    /// `operation` makes each executor's from its place among the node's executors, which is
    /// the rank a page copy or a storage operation names.
    ///
    /// # Errors
    ///
    /// When the node holds no executor, or a queue is closed.
    pub fn control(
        &self,
        label: &str,
        operation: impl Fn(usize) -> Operation,
    ) -> Result<Submission> {
        let mut submissions = self
            .executors
            .iter()
            .enumerate()
            .map(|(rank, (_, executor))| executor.control(label, operation(rank)))
            .collect::<Result<Vec<_>>>()?;

        let last = submissions
            .pop()
            .ok_or_else(|| Error::Invalid("this node holds no executor".to_owned()))?;
        Ok(last.joined(submissions))
    }
}

/// The refusal of a batch whose group and phase no executor of this node serves.
///
/// The scheduler numbers the phases of its batches and the deployment places stages by the
/// same numbers; when the two disagree the batch arrives here and nothing can serve it. Both
/// are named, the phase asked for and the phases this node's executors of the group serve,
/// because one figure alone cannot show the disagreement. The refusal is `Invalid`: the batch
/// is not servable here, which is not a broken device.
fn route_refusal(group: &str, phase: PhaseId, served: &[PhaseId]) -> Error {
    Error::Invalid(format!(
        "no executor of this node in worker group {group:?} serves phase {}; its executors of \
         the group serve {:?}",
        phase.0,
        served.iter().map(|served| served.0).collect::<Vec<_>>()
    ))
}

impl Batches for Executors {
    fn submit(&self, batch: Batch) -> std::result::Result<Submission, Box<Fault>> {
        Ok(Executors::submit(self, batch)?)
    }

    fn shadow(&self, batch: Batch) -> std::result::Result<Submission, Box<Fault>> {
        Ok(Executors::shadow(self, batch)?)
    }

    fn media(
        &self,
        phase: PhaseId,
        computation: &Computation,
    ) -> std::result::Result<Option<Extent>, Box<Fault>> {
        // The scheduler reads the extent the bucket states, and nothing else of the bucket.
        Ok(Executors::media(self, phase, computation)?.map(|bucket| bucket.extent))
    }

    fn devices(&self, batch: &Batch) -> std::result::Result<Vec<usize>, Box<Fault>> {
        Ok(Executors::devices(self, batch)?)
    }

    fn device_count(&self) -> usize {
        Executors::device_count(self)
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_no_executor_serves_is_refused_by_both_numbers() {
        let refusal = route_refusal("decode", PhaseId(3), &[PhaseId(0), PhaseId(1)]);
        assert!(matches!(refusal, Error::Invalid(_)), "{refusal:?}");

        let text = refusal.to_string();
        assert!(
            text.contains("phase 3") && text.contains("[0, 1]") && text.contains("decode"),
            "{text}"
        );
    }
}
