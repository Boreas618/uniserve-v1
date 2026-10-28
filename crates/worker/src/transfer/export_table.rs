//! Registered transport exports of one endpoint and their reclamation.
//!
//! Every registering transport (local, shared memory, CUDA VMM) owns one
//! [`ExportTable`]: a [`BufferRegistry`] of its exports, keyed by the
//! locator's transport identity (the endpoint name and the local key, the
//! segment name or the CUDA VMM export id). A locator reaches a consumer only
//! through the engine, which binds only locators a producer reported, so the
//! key alone finds an export. Endpoints are also published in a process-wide
//! map by name, so a reader in the same address space resolves a local or
//! CUDA VMM locator to the producer's storage directly.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Weak};

use ipc::{Locator, TransferTransport, Wake};

use super::backing::{SourceRetirement, TransportBuffer};
use super::registry::{BufferRegistry, RegisteredBuffer};
use super::{Retirement, notify};
use crate::host::HostPool;
use crate::{Error, EventPool, Result, lock};

/// A registered export's identity within its endpoint.
#[derive(Clone, Eq, Hash, PartialEq)]
pub(crate) enum SourceKey {
    Local(u64),
    PosixShm(String),
    Cuda(String),
}

/// A registered export's physical backing.
pub(crate) struct Source {
    buffer: Arc<TransportBuffer>,
}

impl Source {
    pub(crate) fn new(buffer: Arc<TransportBuffer>) -> Self {
        Self { buffer }
    }
}

type Table = BufferRegistry;

static ENDPOINTS: LazyLock<Mutex<HashMap<String, Weak<ExportTable>>>> =
    LazyLock::new(Mutex::default);

/// A process-local reader's admission to a registered export. Releasing it
/// lets the producer reclaim the export once its other readers finished.
pub(crate) struct SourceGrant {
    owner: Arc<ExportTable>,
    key: SourceKey,
}

impl SourceGrant {
    pub(crate) fn release(&self) -> Result<()> {
        self.owner.release_reader(&self.key)
    }
}

/// One transport's registered exports.
pub(crate) struct ExportTable {
    name: String,
    events: Arc<EventPool>,
    state: Mutex<Table>,
    /// Shared-storage retirements that unregister CUDA host mappings run on this
    /// host pool, created on first use.
    retirements: Mutex<Option<Arc<HostPool<SourceRetirement>>>>,
    wake: Mutex<Option<Wake>>,
}

impl ExportTable {
    pub(crate) fn new(name: String, events: Arc<EventPool>) -> Arc<Self> {
        let endpoint = Arc::new(Self {
            name,
            events,
            state: Mutex::new(Table::new()),
            retirements: Mutex::new(None),
            wake: Mutex::new(None),
        });
        lock(&ENDPOINTS).insert(endpoint.name.clone(), Arc::downgrade(&endpoint));
        endpoint
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    fn lock(&self) -> MutexGuard<'_, Table> {
        lock(&self.state)
    }

    fn key(&self, locator: &Locator) -> Result<SourceKey> {
        let (endpoint, key) = match &locator.transport {
            TransferTransport::Local { endpoint, key } => (endpoint, SourceKey::Local(*key)),
            TransferTransport::PosixShm { endpoint, name } => {
                (endpoint, SourceKey::PosixShm(name.clone()))
            }
            TransferTransport::CudaVmm {
                endpoint,
                export_id,
                ..
            } => (endpoint, SourceKey::Cuda(export_id.clone())),
            TransferTransport::Channel { .. } => {
                return Err(Error::Invalid(
                    "transport does not register source storage".into(),
                ));
            }
        };
        if *endpoint != self.name {
            return Err(Error::Invalid("buffer belongs to another endpoint".into()));
        }
        Ok(key)
    }

    fn buffer<'a>(state: &'a mut Table, key: &SourceKey) -> Result<&'a mut RegisteredBuffer> {
        state
            .get_mut(key)
            .ok_or_else(|| Error::Invalid("buffer is no longer registered".into()))
    }

    /// Register an export's backing under `locator`'s key; the backing
    /// retires through the registry once the export is released and its
    /// readers finished.
    pub(crate) fn register(&self, locator: &Locator, buffer: Arc<TransportBuffer>) -> Result<()> {
        let key = self.key(locator)?;
        self.remove_finished();
        self.lock()
            .register(key, Source::new(buffer), Arc::new(Retirement::default()))
    }

    /// Revoke future reads. The returned signal completes after physical
    /// retirement; `None` when the buffer already left the registry.
    pub(crate) fn release(&self, locator: &Locator) -> Result<Option<Arc<Retirement>>> {
        let key = self.key(locator)?;
        let retirement = {
            let mut state = self.lock();
            if state.get_mut(&key).is_none() {
                return Ok(None);
            }
            let buffer = Self::buffer(&mut state, &key)?;
            buffer.release();
            Arc::clone(buffer.retirement())
        };
        self.reclaim(&key)?;
        Ok(Some(retirement))
    }

    /// The physical retirement signal of a registered export.
    pub(crate) fn retirement(&self, locator: &Locator) -> Result<Arc<Retirement>> {
        let key = self.key(locator)?;
        let mut state = self.lock();
        Ok(Arc::clone(Self::buffer(&mut state, &key)?.retirement()))
    }

    /// Admit a process-local reader of `locator`'s export, wherever its
    /// producing endpoint lives in this process.
    pub(crate) fn acquire_read(locator: &Locator) -> Result<(Arc<TransportBuffer>, SourceGrant)> {
        let endpoint = match &locator.transport {
            TransferTransport::Local { endpoint, .. }
            | TransferTransport::CudaVmm { endpoint, .. } => endpoint,
            _ => {
                return Err(Error::Invalid(
                    "only local and CUDA VMM exports are read in place".into(),
                ));
            }
        };
        let owner = lock(&ENDPOINTS)
            .get(endpoint)
            .and_then(Weak::upgrade)
            .ok_or_else(|| Error::Invalid("local buffer has no live owner".into()))?;
        let key = owner.key(locator)?;
        let buffer = {
            let mut state = owner.lock();
            Arc::clone(&Self::buffer(&mut state, &key)?.acquire()?.buffer)
        };
        Ok((buffer, SourceGrant { owner, key }))
    }

    fn release_reader(&self, key: &SourceKey) -> Result<()> {
        self.lock()
            .get_mut(key)
            .ok_or_else(|| Error::Invariant("active read lost its source buffer".into()))?
            .release_reader()?;
        self.reclaim(key)
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> bool {
        self.lock().awaiting_acknowledgment()
    }

    /// Sweep remote acknowledgments and discard successfully retired storage.
    pub(crate) fn reap(&self) -> Result<()> {
        let keys: Vec<_> = self.lock().revoked().cloned().collect();
        for key in keys {
            self.reclaim(&key)?;
        }
        self.remove_finished();
        Ok(())
    }

    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        let pool = lock(&self.retirements).clone();
        if let Some(pool) = pool {
            pool.set_wake(wake.clone());
        }
        let previous = std::mem::replace(&mut *lock(&self.wake), wake);
        drop(previous);
    }

    /// Revoke every export, wait for producers on the host, and retire.
    /// Fails, retaining storage, while remote readers still hold exports.
    pub(crate) fn close(&self) -> Result<()> {
        self.unregister();
        self.lock().close();
        let mut result = self.reap();
        let buffers: Vec<_> = self
            .lock()
            .buffers()
            .map(|buffer| Arc::clone(&buffer.source().buffer))
            .collect();
        for buffer in buffers {
            result = result.and(buffer.drain());
        }
        // No service loop is left to take the completions the deferred
        // releases wait for: the pool waits for their events on the host.
        result = result.and(self.events.drain()).and(self.reap());
        result = result.and(self.events.drain());

        // Joining the pool must not hold the registry lock: retirement
        // observers may inspect returned storage.
        let pool = lock(&self.retirements).take();
        if let Some(pool) = pool {
            for error in pool.close() {
                result = result.and(Err(error));
            }
        }
        result = result.and(self.reap()).and(self.events.drain());
        result.and(self.lock().require_retired())
    }

    fn unregister(&self) {
        let mut endpoints = lock(&ENDPOINTS);
        if endpoints
            .get(&self.name)
            .is_some_and(|entry| std::ptr::eq(entry.as_ptr(), self))
        {
            endpoints.remove(&self.name);
        }
    }

    fn reclaim(&self, key: &SourceKey) -> Result<()> {
        let (buffer, retirement) = {
            let mut state = self.lock();
            let Some(registered) = state.get_mut(key) else {
                return Ok(());
            };
            let Some((source, retirement)) =
                registered.begin_reclaim(|source| Ok(source.buffer.settled()))?
            else {
                return Ok(());
            };
            (Arc::clone(&source.buffer), Arc::clone(retirement))
        };

        let result = if buffer.asynchronous() {
            self.retire_on_pool(SourceRetirement {
                buffer,
                completion: Some(Arc::clone(&retirement)),
            })
        } else {
            TransportBuffer::reclaim(buffer, &self.events, Some(Arc::clone(&retirement)))
        };
        if let Err(error) = result {
            if !retirement.done() {
                notify(retirement.complete(Err(Error::transport(format!(
                    "transport reclamation failed: {error}"
                ))))?);
            }
            return Err(error);
        }
        Ok(())
    }

    fn retire_on_pool(&self, action: SourceRetirement) -> Result<()> {
        let pool = {
            let mut installed = lock(&self.retirements);
            match &*installed {
                Some(pool) => Arc::clone(pool),
                None => {
                    let pool = HostPool::new(256, 1, "uniserve-posix-shm-retire")?;
                    pool.set_wake(lock(&self.wake).clone());
                    let pool = Arc::new(pool);
                    *installed = Some(Arc::clone(&pool));
                    pool
                }
            }
        };
        let task = pool.reserve()?;
        if let Err(error) = task.configure(action).and_then(|()| task.submit()) {
            if let Err(cleanup) = task.cancel(true) {
                tracing::warn!("failed to withdraw a transport retirement: {cleanup}");
            }
            return Err(error);
        }
        Ok(())
    }

    fn remove_finished(&self) {
        let retired = self.lock().take_finished();
        drop(retired);
    }
}

impl Drop for ExportTable {
    fn drop(&mut self) {
        self.unregister();
    }
}
