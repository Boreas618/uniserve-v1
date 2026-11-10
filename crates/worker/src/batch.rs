//! The worker's half of the crossing from the scheduling mechanism (`ipc::execution`).
//!
//! The batch the mechanism submits, its columns, its output and the [`Batches`] contract are
//! `ipc::execution`'s, which the scheduler and the worker both read. This module is what the worker
//! adds to them: [`Workers`], its [`Batches`] over a node's devices, and the [`Jobs`] a runner asks
//! its device worker to issue for a batch.
//!
//! A submission is one or more jobs because a program-layer update interleaves work with serving:
//! a batch is a `Replay` of its bucket's graph, a precopy is the `Capture`/`Rebind` jobs of the
//! buckets a changed operator uses, and switchover and purge are submissions of their own so they
//! land on a batch boundary (7.3).

use std::sync::Arc;

use ipc::PhaseId;
use ipc::execution::{
    Batch, BatchId, Batches, CallId, CallOutput, Computation, Extent, Fault, HostPools, Inventory,
    RowState, Submission,
};

use crate::plan::Bucket;
use crate::{Error, Result};

/// The answers to a replay that served `calls`, leaving `calls` empty.
///
/// A media call's result is a product, which its runner published under the request's name, so a
/// call reports that it ran and nothing else.
pub(crate) fn answered(calls: &mut Vec<CallId>) -> Vec<CallOutput> {
    calls
        .drain(..)
        .map(|call| CallOutput::completed(call, Vec::new(), RowState::default()))
        .collect()
}

/// The work a runner asks its device worker to issue for one submission.
///
/// The runner stages what the batch needs before it returns; a job is the device work that follows
/// the staging, in the order this list holds it. Each runner job names the runner that performs it,
/// because only the runner owns the model and the arena.
#[derive(Default)]
pub struct Jobs {
    jobs: Vec<Job>,
}

impl Jobs {
    /// A submission of `jobs`, in order.
    pub fn new(jobs: impl IntoIterator<Item = Job>) -> Self {
        Self {
            jobs: jobs.into_iter().collect(),
        }
    }

    /// Every job, in order.
    pub fn iter(&self) -> impl Iterator<Item = &Job> {
        self.jobs.iter()
    }
}

/// One job the device worker issues, in submission order.
#[derive(Clone)]
pub enum Job {
    /// Takes this rank's part in the expert step a peer opened, with no tokens of its own.
    Join {
        /// The runner that joins.
        runner: crate::runner::RunnerId,
    },
    /// Replays the captured graph of `bucket`: the runner staged the batch's inputs already.
    Replay {
        /// The runner that serves the batch.
        runner: crate::runner::RunnerId,
        /// The bucket whose graph is replayed.
        bucket: Bucket,
        /// The generation the bucket's graph belongs to.
        generation: u64,
        /// The batch the job serves, for the stall report (2.4).
        batch: BatchId,
    },
    /// Captures `bucket` for `generation`, between serving batches (4.5 recapture).
    Capture {
        /// The runner that captures the bucket.
        runner: crate::runner::RunnerId,
        /// The bucket to capture.
        bucket: Bucket,
        /// The generation the capture is for.
        generation: u64,
    },
    /// Runs the paired forward of one changed operator and compares its two results (7.3).
    Shadow {
        /// The runner that walks the paired forward and compares.
        runner: crate::runner::RunnerId,
        /// The operator whose new kernel runs beside its old one.
        operator: &'static str,
        /// The operator's descriptor key, which names the launch's kernel (4.2).
        key: String,
        /// The bucket whose static activations the pair uses.
        bucket: Bucket,
        /// The generation the new kernel belongs to.
        generation: u64,
    },
    /// Rebinds `bucket`'s launches for `generation` from that generation's precopy walk.
    Rebind {
        /// The runner that walks the forward and rebinds the graph.
        runner: crate::runner::RunnerId,
        /// The bucket to rebind.
        bucket: Bucket,
        /// The generation the rebind is for.
        generation: u64,
    },
    /// Serves one call on the host: a media call whose work is encoding, muxing or reading, and
    /// whose answer is bytes rather than a device result (2.4's fourth kind of blocking work).
    ///
    /// The job carries no bucket because a host call captures no graph: the runner staged the bytes
    /// this job reads when it submitted the batch, and performs the work on the process's media
    /// lane, so nothing here reaches a device stream. A `CallOutput` from such a job carries the
    /// artifact (`media_output`) and no tokens, which is what the mechanism's completion reads.
    Host {
        /// The runner that serves the call.
        runner: crate::runner::RunnerId,
        /// The call this job answers.
        call: CallId,
    },
    /// Performs one latent preparation on the denoiser that will step the trajectory: the call
    /// states how the denoiser's pages hold the request's samples and, for a resumed request,
    /// copies the stored pages the engine pinned into the trajectory on the denoiser's stream.
    ///
    /// The job carries no bucket because a preparation captures no graph; the runner noted the
    /// request's layout when it submitted the batch.
    Prepare {
        /// The runner that prepares the trajectory.
        runner: crate::runner::RunnerId,
        /// The call this job answers.
        call: CallId,
    },
}

impl Job {
    /// What this job is doing, for the report of a failed submission.
    ///
    /// The device thread knows the job it was performing and the caller does not, and the failure
    /// crosses a thread boundary as a bare error, so the description is written where the job is
    /// performed and carried beside that error.
    pub fn describe(&self) -> String {
        match self {
            Self::Join { .. } => "joining an expert step".to_owned(),
            Self::Replay {
                bucket,
                generation,
                batch,
                ..
            } => format!(
                "replaying batch {} from the bucket of {} tokens and {} rows of generation {generation}",
                batch.0, bucket.tokens, bucket.rows
            ),
            Self::Capture {
                bucket, generation, ..
            } => format!("capturing bucket {bucket} of generation {generation}"),
            Self::Rebind {
                bucket, generation, ..
            } => format!("rebinding bucket {bucket} of generation {generation}"),
            Self::Shadow {
                operator,
                key,
                bucket,
                generation,
                ..
            } => format!(
                "shadowing {operator} {key} in the bucket of {} tokens and {} rows of generation {generation}",
                bucket.tokens, bucket.rows
            ),
            Self::Host { call, .. } => format!("serving call {} on the host", call.0),
            Self::Prepare { call, .. } => format!("preparing the trajectory of call {}", call.0),
        }
    }

    /// The runner that performs the job, if a runner does.
    pub fn runner(&self) -> Option<crate::runner::RunnerId> {
        match self {
            Self::Join { runner }
            | Self::Replay { runner, .. }
            | Self::Capture { runner, .. }
            | Self::Rebind { runner, .. }
            | Self::Shadow { runner, .. }
            | Self::Host { runner, .. }
            | Self::Prepare { runner, .. } => Some(*runner),
        }
    }
}

impl std::fmt::Debug for Job {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Join { runner } => formatter
                .debug_struct("Join")
                .field("runner", runner)
                .finish(),
            Self::Replay {
                runner,
                bucket,
                generation,
                batch,
            } => formatter
                .debug_struct("Replay")
                .field("runner", runner)
                .field("bucket", bucket)
                .field("generation", generation)
                .field("batch", batch)
                .finish(),
            Self::Capture {
                runner,
                bucket,
                generation,
            } => formatter
                .debug_struct("Capture")
                .field("runner", runner)
                .field("bucket", bucket)
                .field("generation", generation)
                .finish(),
            Self::Shadow {
                runner,
                operator,
                key,
                bucket,
                generation,
            } => formatter
                .debug_struct("Shadow")
                .field("runner", runner)
                .field("operator", operator)
                .field("key", key)
                .field("bucket", bucket)
                .field("generation", generation)
                .finish(),
            Self::Rebind {
                runner,
                bucket,
                generation,
            } => formatter
                .debug_struct("Rebind")
                .field("runner", runner)
                .field("bucket", bucket)
                .field("generation", generation)
                .finish(),
            Self::Host { runner, call } => formatter
                .debug_struct("Host")
                .field("runner", runner)
                .field("call", call)
                .finish(),
            Self::Prepare { runner, call } => formatter
                .debug_struct("Prepare")
                .field("runner", runner)
                .field("call", call)
                .finish(),
        }
    }
}

/// Where the host KV cache of a rank is reached: the rank's pages are keyed by the page identities
/// it reports (`PrefixCacheInfo::identities`), so an epoch that computes the same pages finds what
/// another epoch of the deployment left there.
#[derive(Clone, Debug, Default)]
pub struct HostCacheEndpoint {
    /// Where an engine on another host reaches this rank's Storage Server.
    pub endpoint: Option<String>,
}

/// The mechanism's `Batches` over a node's devices (2.4, 4.4): a batch goes to every device whose
/// runners serve its phase.
///
/// A node's workers are its devices, and a device's runners are the blocks of the model placed
/// on it; the mechanism addresses a batch by phase, not by device. This is the one place that
/// translation happens, so a scheduler never needs to know how many devices it drives. A phase one
/// device serves is that device's. A phase several devices serve is a parallel group's: each runs
/// its part of one batch, so each must see it. The scheduler's thread is the one thread that submits,
/// so every device's queue holds the group's batches in one order, which is the order their
/// collectives need. It is a plain route on purpose: the phase list is fixed at spawn, so no lock,
/// no schedule, no retry.
pub struct Workers {
    workers: Vec<Arc<crate::device::Worker>>,
    /// The pools this rank serves, as the deployment launched them (4.2): the scheduler builds its
    /// worker facts from this and refuses to start on an empty answer, so a rank states its pools
    /// before it is handed over rather than after the first batch finds none.
    inventory: Vec<Inventory>,
    /// What keys this rank's pages in the host KV cache, when the deployment keeps one.
    host_cache: Option<HostCacheEndpoint>,
}

impl Workers {
    /// The rank's devices, as the handles the rest of the process already holds.
    ///
    /// A main process holds one handle per device from the moment it spawns them: the control
    /// thread reads their running jobs for the stall rule and their counters for the metrics,
    /// and a scheduler submits through the same handles. Sharing them rather than moving them in
    /// keeps one device one owner.
    pub fn shared(workers: Vec<Arc<crate::device::Worker>>) -> Result<Self> {
        Ok(Self {
            workers,
            inventory: Vec::new(),
            host_cache: None,
        })
    }

    /// States that this rank keeps a host KV cache, with what keys its pages.
    #[must_use]
    pub fn keeping_host_cache(mut self, keys: HostCacheEndpoint) -> Self {
        self.host_cache = Some(keys);
        self
    }

    /// States the pools this rank serves, which the scheduler's worker facts are built from.
    ///
    /// The facts are the deployment's: its pool's name, its ranks as they reported at handshake,
    /// what the launch placed there and the model's maximum length. A rank that serves batches must
    /// state them - `Submissions::over_batches` refuses a worker that states no pool - so this is
    /// called where the launch facts are known, before the scheduler is built.
    #[must_use]
    pub fn stating(mut self, inventory: Vec<Inventory>) -> Self {
        self.inventory = inventory;
        self
    }

    /// The device that serves `phase`.
    fn route(&self, phase: PhaseId) -> Result<&Arc<crate::device::Worker>> {
        self.workers
            .iter()
            .find(|worker| worker.serves(phase))
            .ok_or_else(|| {
                route_refusal(
                    phase,
                    &self
                        .workers
                        .iter()
                        .flat_map(|worker| worker.phases().iter().copied())
                        .collect::<Vec<_>>(),
                )
            })
    }

    /// The one device a unit-measured call runs on, when `batch` is one.
    ///
    /// A video decode is one unit of the timeline and needs no other unit, so the devices that serve
    /// the phase take the units in turn instead of each decoding the same one: a device's clip is
    /// independent of its neighbours' but for the tail the decoder hands across. A video encode
    /// follows its unit, so the device that decoded a clip encodes it from its own column.
    fn device_of_unit(&self, batch: &Batch) -> Result<Option<&Arc<crate::device::Worker>>> {
        let Some(media) = &batch.media else {
            return Ok(None);
        };
        if !matches!(
            media.computation.call,
            ipc::MediaCall::VideoDecoding | ipc::MediaCall::VideoEncoding
        ) {
            return Ok(None);
        }
        let serving: Vec<&Arc<crate::device::Worker>> = self
            .workers
            .iter()
            .filter(|worker| worker.serves(batch.phase))
            .collect();
        if serving.is_empty() {
            return Err(route_refusal(batch.phase, &[]));
        }
        let unit = media.request(0)?.unit as usize;
        Ok(Some(serving[unit % serving.len()]))
    }

    /// Hands `batch` to every device that serves its phase, as one submission.
    fn on_every_device(
        &self,
        batch: Batch,
        submit: impl Fn(&crate::device::Worker, Batch) -> Result<Submission>,
    ) -> Result<Submission> {
        let first = self.route(batch.phase)?;
        let mut peers = Vec::new();
        for peer in self
            .workers
            .iter()
            .filter(|worker| worker.serves(batch.phase) && !Arc::ptr_eq(worker, first))
        {
            peers.push(submit(peer, batch.clone())?);
        }
        Ok(submit(first, batch)?.joined(peers))
    }
}

/// The refusal of a batch whose phase no device of this rank serves.
///
/// A phase number is a plan's, and the plan that numbers the batches and the plan that splits the
/// model must agree on it; when they do not, the batch reaches this rank and nothing here can serve
/// it. Both numbers are named - the phase asked for and the phases this rank's devices serve -
/// because the disagreement is between two plans and one figure alone cannot show it. The answer is
/// `Invalid`: the batch is not servable here, which is a refusal rather than a broken device. What a
/// reader of the number does with that distinction is the caller's - the in-process submission path
/// currently wraps every worker error alike, so an `Invalid` and a broken context read the same
/// there.
fn route_refusal(phase: PhaseId, served: &[PhaseId]) -> Error {
    Error::Invalid(format!(
        "no device of this rank serves phase {}; its devices serve {:?}",
        phase.0,
        served.iter().map(|served| served.0).collect::<Vec<_>>()
    ))
}

/// The routes themselves, in the worker's own error: [`Batches`] answers them across the crossing,
/// and the main process, which holds the concrete `Workers`, reads the worker's error directly.
impl Workers {
    /// The bucket the device serving `phase` would serve `computation` in.
    pub fn media(&self, phase: PhaseId, computation: &Computation) -> Result<Option<Bucket>> {
        Ok(self.route(phase)?.media(computation))
    }

    /// The devices `batch` runs on, by their place among this node's devices.
    pub fn devices(&self, batch: &Batch) -> Result<Vec<usize>> {
        // The same two routes `submit` takes: a unit-measured call's one device, or every device
        // that serves the phase, refused by name when none does.
        if let Some(worker) = self.device_of_unit(batch)? {
            return Ok(self
                .workers
                .iter()
                .position(|held| Arc::ptr_eq(held, worker))
                .into_iter()
                .collect());
        }
        self.route(batch.phase)?;
        Ok(self
            .workers
            .iter()
            .enumerate()
            .filter(|(_, worker)| worker.serves(batch.phase))
            .map(|(index, _)| index)
            .collect())
    }

    /// Serves `batch` on the devices of its phase, or on its unit's one device.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        match self.device_of_unit(&batch)? {
            Some(worker) => worker.submit(batch),
            None => self.on_every_device(batch, |worker, mut batch| {
                // A page copy names the rank that makes it, and a rank is a device here.
                if let Some(rank) = self
                    .workers
                    .iter()
                    .position(|held| std::ptr::eq(held.as_ref(), worker))
                {
                    batch.page_loads.retain(|copy| copy.rank as usize == rank);
                    batch
                        .page_offloads
                        .retain(|copy| copy.rank as usize == rank);
                }
                worker.submit(batch)
            }),
        }
    }

    /// Runs `batch`'s paired forwards on every device of its phase.
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.on_every_device(batch, |worker, batch| worker.shadow(batch))
    }

    /// Maps the host KV cache's page pools on every device.
    pub fn map_host_cache(&self, tokens: u64) -> Result<HostPools> {
        let Some(keys) = &self.host_cache else {
            return Ok(Vec::new());
        };
        let ranks = self
            .workers
            .iter()
            .map(|worker| worker.map_host_cache(tokens, keys))
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .inventory
            .iter()
            .map(|pool| (pool.group.clone(), ranks.clone()))
            .collect())
    }

    /// Copies each device's share of `offloads` to the host KV cache.
    pub fn copy_pages(&self, offloads: Vec<ipc::PageCopy>) -> Result<Submission> {
        let mut submissions = self
            .workers
            .iter()
            .enumerate()
            .map(|(rank, worker)| {
                let own = offloads
                    .iter()
                    .filter(|copy| copy.rank as usize == rank)
                    .cloned()
                    .collect();
                worker.copy_pages(own)
            })
            .collect::<Result<Vec<_>>>()?;
        let first = submissions
            .pop()
            .ok_or_else(|| crate::Error::Invalid("this rank holds no device".to_owned()))?;
        Ok(first.joined(submissions))
    }

    /// Copies each device's share of the preempted trajectories to the host latent pools.
    pub fn offload_latents(&self, offloads: Vec<ipc::LatentOffload>) -> Result<Submission> {
        // A page copy names the rank that makes it, and a rank is a device here.
        let mut submissions = self
            .workers
            .iter()
            .enumerate()
            .map(|(rank, worker)| {
                let own = offloads
                    .iter()
                    .map(|offload| ipc::LatentOffload {
                        request_key: offload.request_key,
                        pages: offload
                            .pages
                            .iter()
                            .filter(|copy| copy.rank as usize == rank)
                            .cloned()
                            .collect(),
                    })
                    .filter(|offload| !offload.pages.is_empty())
                    .collect();
                worker.offload_latents(own)
            })
            .collect::<Result<Vec<_>>>()?;
        let first = submissions
            .pop()
            .ok_or_else(|| crate::Error::Invalid("this rank holds no device".to_owned()))?;
        Ok(first.joined(submissions))
    }
}

impl Batches for Workers {
    fn submit(&self, batch: Batch) -> std::result::Result<Submission, Box<Fault>> {
        Ok(Workers::submit(self, batch)?)
    }

    fn shadow(&self, batch: Batch) -> std::result::Result<Submission, Box<Fault>> {
        Ok(Workers::shadow(self, batch)?)
    }

    fn media(
        &self,
        phase: PhaseId,
        computation: &Computation,
    ) -> std::result::Result<Option<Extent>, Box<Fault>> {
        // The mechanism states the extent the planned bucket was captured for, which is all it
        // reads of the bucket: the bucket itself is the worker's.
        Ok(Workers::media(self, phase, computation)?.map(|bucket| bucket.extent))
    }

    fn devices(&self, batch: &Batch) -> std::result::Result<Vec<usize>, Box<Fault>> {
        Ok(Workers::devices(self, batch)?)
    }

    fn device_count(&self) -> usize {
        self.workers.len()
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }

    fn map_host_cache(&self, tokens: u64) -> std::result::Result<HostPools, Box<Fault>> {
        Ok(Workers::map_host_cache(self, tokens)?)
    }

    fn copy_pages(
        &self,
        offloads: Vec<ipc::PageCopy>,
    ) -> std::result::Result<Submission, Box<Fault>> {
        Ok(Workers::copy_pages(self, offloads)?)
    }

    fn offload_latents(
        &self,
        offloads: Vec<ipc::LatentOffload>,
    ) -> std::result::Result<Submission, Box<Fault>> {
        Ok(Workers::offload_latents(self, offloads)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch whose phase no device of this rank serves is refused where it is routed, and the
    /// refusal names the phase the plan asked for and the phases this rank's devices serve: the
    /// disagreement is between two plans, and one figure alone cannot show it. It is a refusal and
    /// not a device failure - the batch is not servable here, which is not the same as a broken
    /// device - and it is stated as such, because a reader that can tell the two apart can end one
    /// batch instead of the epoch.
    #[test]
    fn a_phase_no_device_serves_is_refused_by_both_numbers() {
        let refusal = route_refusal(ipc::PhaseId(3), &[ipc::PhaseId(0), ipc::PhaseId(1)]);
        assert!(
            matches!(refusal, Error::Invalid(_)),
            "a batch nothing here can serve was refused as a device failure: {refusal:?}"
        );
        let text = refusal.to_string();
        assert!(
            text.contains("phase 3") && text.contains("[0, 1]"),
            "the refusal names neither the phase asked for nor the ones served: {text}"
        );
    }
}
