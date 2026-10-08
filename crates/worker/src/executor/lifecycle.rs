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
//! Request storage (tensor store, KV cache, latent pool, block tables,
//! decode and canvas state) is owned by the storage and capability runtimes
//! and takes part through [`RequestStorage`].

use std::collections::HashSet;

use uniserve_ipc::{Batch, BatchCommand, BufferId, RequestKey};

use crate::Result;
use crate::request::RequestPool;

/// Storage that request epochs and products own, driven by lifecycle
/// commands. Methods run on the executor thread.
pub trait RequestStorage {
    /// Reset per-slot state of newly admitted request slots.
    fn admit_slots(&mut self, slots: &[usize]) -> Result<()>;

    /// Revoke explicitly freed products and those of finished requests.
    /// Existing readers keep their physical leases.
    fn release_buffers(&mut self, buffers: &HashSet<BufferId>) -> Result<()>;

    /// Begin releasing finished requests' storage, keeping `retained`
    /// products readable until their own `Free`.
    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()>;

    /// Whether every physical reader of these products and requests has
    /// retired.
    fn retirement_ready(
        &mut self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<bool>;

    /// Return retired requests' slots for reuse.
    fn retire_slots(&mut self, slots: &[usize]) -> Result<()>;

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
pub struct Retirement {
    requests: HashSet<RequestKey>,
    buffers: HashSet<BufferId>,
    retained: HashSet<BufferId>,
    /// Finished epochs that hold a slot on this rank.
    local: Vec<RequestKey>,
    complete: bool,
}

impl Retirement {
    pub fn new(batch: &Batch) -> Self {
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
    pub fn admit(&self, storage: &mut [&mut dyn RequestStorage]) -> Result<()> {
        if self.buffers.is_empty() {
            return Ok(());
        }
        for owner in storage.iter_mut() {
            owner.release_buffers(&self.buffers)?;
        }
        Ok(())
    }

    /// Start retiring after the batch launched.
    pub fn begin(
        &mut self,
        requests: &RequestPool,
        storage: &mut [&mut dyn RequestStorage],
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
        for owner in storage.iter_mut() {
            owner.release_requests(&self.requests, &self.retained)?;
            owner.release_buffers(&self.buffers)?;
        }
        Ok(())
    }

    /// Retire the finished epochs once their work and readers have drained.
    /// Returns whether retirement is complete.
    pub fn poll(
        &mut self,
        requests: &mut RequestPool,
        storage: &mut [&mut dyn RequestStorage],
    ) -> Result<bool> {
        if self.complete {
            return Ok(true);
        }
        for key in &self.local {
            if !requests.retirement_ready(*key)? {
                return Ok(false);
            }
        }
        for owner in storage.iter_mut() {
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
            for owner in storage.iter_mut() {
                owner.retire_slots(&slots)?;
            }
        }
        for request_id in retired {
            requests.retire(request_id)?;
        }
        self.complete = true;
        Ok(true)
    }
}
