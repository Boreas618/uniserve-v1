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
//! Request storage (the product store, the prefix cache and its block
//! tables, the lanes' output leases and latent ledger, the transfer
//! runtime's output leases) takes part through [`RequestStorage`]; slot
//! state the executor keeps is released through its `release`.
//!
//! Lifecycle commands run on the control thread, which issues no stream
//! work itself: what an owner's command needs on a device (a slot's reset,
//! the clearing of released rows, what follows a released buffer's
//! fences) is recorded into the command's job ([`LaneJobs`]), and the
//! backend submits it to the lane after the command.

use std::collections::HashSet;

use ipc::{Batch, BatchCommand, BufferId, RequestKey};

use super::backend::Owners;
use crate::Result;
use crate::lane::Job;
use crate::request::RequestPool;

/// The job a lifecycle command records, performed by the rank's lane (by
/// the builder thread on a rank without one). An owner records the work
/// its command needs on a stream into it.
#[derive(Default)]
pub(crate) struct LaneJobs {
    job: Job,
}

impl LaneJobs {
    /// The command's job.
    pub(crate) fn job(&self) -> &Job {
        &self.job
    }

    /// The job when it holds work, leaving it empty for the next command.
    pub(crate) fn take(&mut self) -> Option<Job> {
        (!self.job.is_empty()).then(|| std::mem::take(&mut self.job))
    }
}

/// Storage that request epochs and products own, driven by lifecycle
/// commands. Methods run on the executor thread.
pub(crate) trait RequestStorage {
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
    pub(crate) fn admit(&self, owners: &mut Owners, jobs: &LaneJobs) -> Result<()> {
        if self.buffers.is_empty() {
            return Ok(());
        }
        for owner in owners.request_storage() {
            owner.release_buffers(&self.buffers, jobs)?;
        }
        Ok(())
    }

    /// Start retiring after the batch launched.
    pub(crate) fn begin(
        &mut self,
        requests: &RequestPool,
        owners: &mut Owners,
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
        for owner in owners.request_storage() {
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
        owners: &mut Owners,
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
        for owner in owners.request_storage() {
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
            for owner in owners.request_storage() {
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
