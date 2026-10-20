//! Request lifecycle commands and their retirement.
//!
//! A batch carries ordered commands with its calls:
//!
//! - `Start` installs a request epoch at its scheduler-assigned slot before
//!   the batch resolves predecessors; newly admitted slots are reset.
//! - `Finish` closes an epoch. The epoch retires (its slot becomes
//!   reusable and its storage is released) once its admitted calls have
//!   completed and every physical reader of its storage has retired,
//!   except for the products it retains.
//! - `Free` releases one persistent product at admission, so an explicit
//!   release never waits behind an allocation-dependent batch; existing
//!   readers keep their leases.
//!
//! Request storage (tensor store, prefix cache, latent pool, block tables,
//! decode and diffusion LM state) is owned by the storage and capability runtimes
//! and takes part through [`RequestStorage`].
//!
//! Lifecycle commands run on the control thread, which issues no stream
//! work itself: what an owner's command needs on a device (a slot's reset,
//! the clearing of released rows, what follows a released buffer's
//! fences) is recorded into the job of the lane whose stream it belongs to
//! ([`LaneJobs`]), and the backend submits those jobs after the command.

use std::collections::{HashMap, HashSet};

use uniserve_ipc::{Batch, BatchCommand, BufferId, CallKind, NewRequest, RequestKey};

use super::backend::Runtimes;
use crate::lane::Job;
use crate::request::RequestPool;
use crate::{Error, Result};

/// The jobs a lifecycle command records: one per call kind the rank serves,
/// performed by the lane serving that kind. An owner records work on a
/// stream into the job of the kind whose lane owns the stream.
pub(crate) struct LaneJobs {
    jobs: HashMap<CallKind, Job>,
    /// The kind whose job takes work that belongs to no lane's stream
    /// (deferred work behind an event): the first the rank serves.
    any: Option<CallKind>,
    /// That job on a rank that serves no call kind and so has no lane.
    unserved: Job,
}

impl LaneJobs {
    /// Empty jobs for a rank serving `kinds`.
    pub(crate) fn new(kinds: &[CallKind]) -> Self {
        Self {
            jobs: kinds.iter().map(|&kind| (kind, Job::new())).collect(),
            any: kinds.first().copied(),
            unserved: Job::new(),
        }
    }

    /// The job of the lane serving `kind`.
    pub(crate) fn job(&self, kind: CallKind) -> Result<&Job> {
        self.jobs.get(&kind).ok_or_else(|| {
            Error::Invariant(format!(
                "lifecycle work names {}, which this rank does not serve",
                kind.as_str()
            ))
        })
    }

    /// Record the work no lane's stream carries for the lane serving
    /// `kind` from now on: the lane of the forward in flight. The job that
    /// waits for an event one of its jobs records is then ordered behind
    /// that record, since a lane performs its jobs in order.
    pub(crate) fn follow(&mut self, kind: CallKind) {
        if self.jobs.contains_key(&kind) {
            self.any = Some(kind);
        }
    }

    /// The job for work no lane's stream carries.
    pub(crate) fn any(&self) -> &Job {
        self.any
            .and_then(|kind| self.jobs.get(&kind))
            .unwrap_or(&self.unserved)
    }

    /// The jobs that hold work, leaving every job empty for the next
    /// command: each with the kind whose lane performs it, or `None` for
    /// the job of a rank without a lane, which its submitter performs.
    pub(crate) fn take(&mut self) -> Vec<(Option<CallKind>, Job)> {
        let mut taken: Vec<_> = self
            .jobs
            .iter_mut()
            .filter(|(_, job)| !job.is_empty())
            .map(|(&kind, job)| (Some(kind), std::mem::take(job)))
            .collect();
        if !self.unserved.is_empty() {
            taken.push((None, std::mem::take(&mut self.unserved)));
        }
        taken
    }
}

/// A request slot a batch's `Start` admitted, and the admission it holds.
pub(crate) struct AdmittedSlot<'a> {
    pub(crate) slot: usize,
    pub(crate) request: &'a NewRequest,
}

/// Storage that request epochs and products own, driven by lifecycle
/// commands. Methods run on the executor thread.
pub(crate) trait RequestStorage {
    /// Reset per-slot state of newly admitted request slots; the resets a
    /// device makes are recorded into `jobs`.
    fn admit_slots(&mut self, admitted: &[AdmittedSlot<'_>], jobs: &LaneJobs) -> Result<()>;

    /// Revoke explicitly freed products and those of finished requests.
    /// Existing readers keep their physical leases. What follows a
    /// released product's outstanding fences is recorded into `jobs`.
    fn release_buffers(&mut self, buffers: &HashSet<BufferId>, jobs: &LaneJobs) -> Result<()>;

    /// Begin releasing finished requests' storage, keeping `retained`
    /// products readable until their own `Free`. Device work is recorded
    /// into `jobs` as for [`Self::release_buffers`].
    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
        jobs: &LaneJobs,
    ) -> Result<()>;

    /// Whether every physical reader of these products and requests has
    /// retired.
    fn retirement_ready(
        &mut self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<bool>;

    /// Return retired requests' slots for reuse; what clears them on a
    /// device is recorded into `jobs`.
    fn retire_slots(&mut self, slots: &[usize], jobs: &LaneJobs) -> Result<()>;

    /// Progress asynchronous completions (event pools, remote readers).
    fn reap(&mut self) -> Result<()>;

    /// Whether a remote acknowledgment without a wake source is outstanding.
    fn awaiting_acknowledgment(&self) -> bool {
        false
    }

    /// Install (or clear) the service wake fired by asynchronous
    /// completions while the rank serves.
    fn set_wake(&self, _wake: Option<uniserve_ipc::Wake>) {}

    /// Release the owner's storage at shutdown, after the executor drained.
    /// An aborted close keeps whatever device work may still address.
    fn close(&mut self, _aborted: bool) -> Result<()> {
        Ok(())
    }
}

/// One batch's Finish and Free commands.
pub(crate) struct Retirement {
    requests: HashSet<RequestKey>,
    buffers: HashSet<BufferId>,
    retained: HashSet<BufferId>,
    /// Finished epochs that hold a slot on this rank.
    local: Vec<RequestKey>,
    complete: bool,
}

impl Retirement {
    pub(crate) fn new(batch: &Batch) -> Self {
        let mut requests = HashSet::new();
        let mut buffers = HashSet::new();
        let mut retained = HashSet::new();
        for command in &batch.commands {
            match command {
                BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                } => {
                    requests.insert(*request_key);
                    retained.extend(retained_buffers.iter().copied());
                }
                BatchCommand::Free { buffer } => {
                    buffers.insert(*buffer);
                }
                BatchCommand::Start { .. } => {}
            }
        }
        // A product freed in the same batch is not retained.
        retained.retain(|buffer| !buffers.contains(buffer));
        Self {
            requests,
            buffers,
            retained,
            local: Vec::new(),
            complete: false,
        }
    }

    /// Revoke explicitly freed products before the batch waits for storage.
    pub(crate) fn admit(&self, runtimes: &mut Runtimes, jobs: &LaneJobs) -> Result<()> {
        if self.buffers.is_empty() {
            return Ok(());
        }
        for owner in runtimes.owners() {
            owner.release_buffers(&self.buffers, jobs)?;
        }
        Ok(())
    }

    /// Start retiring after the batch launched.
    pub(crate) fn begin(
        &mut self,
        requests: &RequestPool,
        runtimes: &mut Runtimes,
        jobs: &LaneJobs,
    ) -> Result<()> {
        if self.requests.is_empty() && self.buffers.is_empty() {
            self.complete = true;
            return Ok(());
        }
        // Remote epochs can own products without holding a slot here.
        self.local = self
            .requests
            .iter()
            .copied()
            .filter(|key| {
                requests
                    .peek(key.request_id.0)
                    .is_some_and(|request| request.key() == *key)
            })
            .collect();
        for owner in runtimes.owners() {
            owner.release_requests(&self.requests, &self.retained, jobs)?;
            owner.release_buffers(&self.buffers, jobs)?;
        }
        Ok(())
    }

    /// Retire the finished epochs once their work and readers have drained.
    /// Returns whether retirement is complete.
    pub(crate) fn poll(
        &mut self,
        requests: &mut RequestPool,
        runtimes: &mut Runtimes,
        jobs: &LaneJobs,
    ) -> Result<bool> {
        if self.complete {
            return Ok(true);
        }
        for key in &self.local {
            if !requests.retirement_ready(*key)? {
                return Ok(false);
            }
        }
        for owner in runtimes.owners() {
            if !owner.retirement_ready(&self.buffers, &self.requests, &self.retained)? {
                return Ok(false);
            }
        }

        let mut slots = Vec::new();
        let mut retired = Vec::new();
        for key in &self.local {
            if let Some(request) = requests.peek(key.request_id.0)
                && request.key() == *key
                && !request.retired()?
            {
                slots.push(request.slot());
                retired.push(key.request_id.0);
            }
        }
        if !slots.is_empty() {
            for owner in runtimes.owners() {
                owner.retire_slots(&slots, jobs)?;
            }
        }
        for request_id in retired {
            requests.retire(request_id)?;
        }
        self.complete = true;
        Ok(true)
    }
}
