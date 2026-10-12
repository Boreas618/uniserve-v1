//! Backing storage and retirement of registered transport exports.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_cuda::Event;

use super::vmm_pool::{PoolChunk, VmmPool};
use super::{
    Capacity, DRAIN_LIMIT, DescriptorGrants, Retirement, SharedBuffer, Spans, drain_event, notify,
};
use crate::host::{HostAction, HostTask};
use crate::storage::memory::Memory;
use crate::{Error, EventPool, Outcome, Result};
use std::time::Instant;

/// A device's VMM export pool, shared by the exports placed in it.
pub(crate) type SharedPool = Arc<Mutex<VmmPool<Arc<Memory>>>>;

/// A CUDA VMM export: either the producer's own exportable storage, or a
/// copy into a chunk of the device's export pool.
pub(crate) struct CudaSource {
    pub(crate) source: Spans,
    pub(crate) event: Arc<Event>,
    /// The producer's storage of a pooled export, retained until the copy
    /// into the chunk completed.
    pub(crate) copied: Option<Spans>,
    pub(crate) pool: Option<(SharedPool, Arc<PoolChunk>)>,
    pub(crate) consumers: Vec<usize>,
    pub(crate) grants: Option<Arc<DescriptorGrants>>,
    pub(crate) export: String,
    /// The process descriptor an in-place POSIX export created; pooled
    /// exports share the pool's descriptor instead.
    pub(crate) descriptor: Option<OwnedFd>,
}

enum Backing {
    /// Spans read in place by readers in this process, fenced by the
    /// producer's event when they are device memory.
    Local {
        source: Spans,
        event: Option<Arc<Event>>,
    },
    Cuda(CudaSource),
    Shared(SharedBuffer),
}

/// One export's physical backing and byte reservation. Reader admission
/// belongs to its registry; the backing returns its bytes to the transfer
/// capacity only after physical retirement.
pub(crate) struct TransportBuffer {
    capacity: Arc<Capacity>,
    nbytes: u64,
    backing: Mutex<Backing>,
}

impl TransportBuffer {
    pub(crate) fn local(
        source: Spans,
        event: Option<Arc<Event>>,
        nbytes: u64,
        capacity: Arc<Capacity>,
    ) -> Self {
        Self {
            capacity,
            nbytes,
            backing: Mutex::new(Backing::Local { source, event }),
        }
    }

    pub(crate) fn shared(storage: SharedBuffer, capacity: Arc<Capacity>) -> Self {
        Self {
            capacity,
            nbytes: storage.nbytes() as u64,
            backing: Mutex::new(Backing::Shared(storage)),
        }
    }

    pub(crate) fn cuda(source: CudaSource, nbytes: u64, capacity: Arc<Capacity>) -> Self {
        Self {
            capacity,
            nbytes,
            backing: Mutex::new(Backing::Cuda(source)),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Backing> {
        self.backing.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The exported spans a process-local reader borrows.
    pub(crate) fn spans(&self) -> Result<Spans> {
        match &*self.lock() {
            Backing::Local { source, .. } => Ok(source.clone()),
            Backing::Cuda(source) => Ok(source.source.clone()),
            Backing::Shared(_) => Err(Error::Invariant(
                "shared buffers expose their mapping through a shared read".into(),
            )),
        }
    }

    /// The producer fence a reader waits on before reading the spans.
    pub(crate) fn event(&self) -> Option<Arc<Event>> {
        match &*self.lock() {
            Backing::Local { event, .. } => event.clone(),
            Backing::Cuda(source) => Some(Arc::clone(&source.event)),
            Backing::Shared(_) => None,
        }
    }

    /// Whether remote readers released a shared buffer. Producer events and
    /// VMM reader words are observed by the event pool and the VMM pool.
    pub(crate) fn settled(&self) -> bool {
        match &*self.lock() {
            Backing::Shared(storage) => storage.settled(),
            _ => true,
        }
    }

    /// Shared buffers of CUDA producers unregister their mapping, which may
    /// wait for the device; they retire on a host lane.
    pub(crate) fn asynchronous(&self) -> bool {
        matches!(&*self.lock(), Backing::Shared(storage) if storage.is_cuda())
    }

    /// Wait on the host for the producer's last write, at most
    /// [`DRAIN_LIMIT`] (shutdown only: no executor is left to route a
    /// completion).
    pub(crate) fn drain(&self) -> Result<()> {
        if let Some(event) = self.event() {
            drain_event(&event, Instant::now() + DRAIN_LIMIT)?;
        } else if let Backing::Shared(storage) = &*self.lock() {
            storage.synchronize()?;
        }
        Ok(())
    }

    /// Begin physical retirement once no reader holds the buffer: revoke a
    /// pooled chunk's reader admission, then release the backing after the
    /// producer fence completes (immediately without one).
    pub(crate) fn reclaim(
        buffer: Arc<Self>,
        events: &EventPool,
        completion: Option<Arc<Retirement>>,
    ) -> Result<()> {
        {
            let backing = buffer.lock();
            if let Backing::Cuda(source) = &*backing
                && let Some((pool, chunk)) = &source.pool
            {
                let grant = source
                    .grants
                    .as_ref()
                    .map(|grants| (Arc::clone(grants), source.export.clone()));
                pool.lock().unwrap_or_else(PoisonError::into_inner).retire(
                    chunk,
                    &source.consumers,
                    &source.event,
                    grant,
                )?;
            }
        }

        let event = buffer.event();
        let retirement = SourceRetirement { buffer, completion };
        match event {
            Some(event) => {
                // The producer's fence was recorded when the buffer was
                // exported, by a job submitted since or at once.
                events.defer(vec![event], Box::new(move || retirement.finish()))
            }
            None => retirement.finish(),
        }
    }

    /// Release the backing after physical access ended and return its bytes.
    fn finish(&self) -> Result<()> {
        let copied = {
            let mut backing = self.lock();
            match &mut *backing {
                Backing::Local { .. } => None,
                Backing::Shared(storage) => {
                    storage.close()?;
                    None
                }
                Backing::Cuda(source) => {
                    if source.pool.is_none() {
                        if let Some(grants) = &source.grants {
                            grants.release(&source.export);
                        }
                        source.descriptor.take();
                    }
                    source.copied.take()
                }
            }
        };
        drop(copied);
        self.capacity.release(self.nbytes)
    }
}

/// The release of one export's backing and the completion of its physical
/// retirement signal. It runs after the producer fence, or on a host lane
/// when unregistering a shared mapping may wait for the device.
pub(crate) struct SourceRetirement {
    pub(crate) buffer: Arc<TransportBuffer>,
    pub(crate) completion: Option<Arc<Retirement>>,
}

impl SourceRetirement {
    pub(crate) fn finish(&self) -> Result<()> {
        let result = self.buffer.finish();
        if let Some(completion) = &self.completion {
            let outcome = match &result {
                Ok(()) => Ok(()),
                Err(error) => Err(Error::Transport(format!(
                    "transport storage retirement failed: {error}"
                ))),
            };
            notify(completion.complete(outcome)?);
        }
        result
    }
}

impl HostAction for SourceRetirement {
    type Output = ();
    type Error = Error;
    type Callback = Box<dyn FnOnce() + Send>;
    type Wake = uniserve_ipc::Wake;

    fn ready(&self) -> Result<bool> {
        Ok(true)
    }

    fn run(&self) -> Result<()> {
        self.finish()
    }

    fn release(&self) -> Result<()> {
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<Error>> {
        // A failed reclamation remains owned by its registry and its failed
        // completion. The host task has no separate input lease.
        Some(Outcome::Success(()))
    }

    fn defer_release(_: Arc<HostTask<Self>>) -> Result<()> {
        Err(Error::Invariant(
            "transport reclamation has no deferred input".into(),
        ))
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for callback in callbacks {
            callback();
        }
    }

    fn wake(wake: &uniserve_ipc::Wake) {
        wake.wake();
    }

    fn report(error: Error) {
        tracing::warn!(target: "uniserve_worker.transfer", "transport storage retirement failed: {error}");
    }

    fn error(error: Error) -> Error {
        error
    }

    fn note_cleanup(_: &mut Error, cleanup: Error) {
        tracing::warn!(target: "uniserve_worker.transfer", "transport retirement cleanup also failed: {cleanup}");
    }
}
