//! Cache import destinations and conversion workspaces retained through physical reads.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use ipc::{BufferId, RequestKey};

use super::read::ReadRef;
use crate::{Error, Result, lock};

mod backend;
mod copy;

pub(crate) use backend::{ImportCopy, ImportEntry, ImportWorkspace, Retired};

struct ImportState {
    reads: Vec<Arc<ReadRef>>,
    workspace: Option<Arc<ImportWorkspace>>,
    started: bool,
    finished: bool,
    drained: bool,
    cancelled: bool,
    released: bool,
}

/// One destination held until adoption or abandonment and all physical access
/// finishes: the reads writing it and the workspace converting into it.
pub(crate) struct CacheImport {
    pub(crate) buffer: BufferId,
    pub(crate) request_pool_idx: usize,
    state: Mutex<ImportState>,
}

impl CacheImport {
    pub(crate) fn new(buffer: BufferId, request_pool_idx: usize, copy: bool) -> Self {
        Self {
            buffer,
            request_pool_idx,
            state: Mutex::new(ImportState {
                reads: Vec::new(),
                workspace: None,
                started: false,
                finished: !copy,
                drained: !copy,
                cancelled: false,
                released: false,
            }),
        }
    }

    pub(crate) fn cancelled(&self) -> bool {
        self.lock().cancelled
    }

    pub(crate) fn released(&self) -> bool {
        self.lock().released
    }

    fn drained(&self) -> bool {
        self.lock().drained
    }

    fn start(&self) {
        self.lock().started = true;
    }

    /// A copy that could not drain its stream keeps all backing retained.
    fn finish(&self, drained: bool) {
        let mut state = self.lock();
        state.finished = true;
        state.drained = drained;
    }

    /// Cancelling a queued host task never enters its action. Its destination
    /// has no device access to drain, but still needs normal import retirement.
    fn task_done(&self) {
        let mut state = self.lock();
        if !state.started {
            state.finished = true;
            state.drained = true;
        }
    }

    /// Register every started read, including one racing cancellation. The
    /// caller cancels it after unlocking when this returns true.
    pub(crate) fn retain(&self, read: Arc<ReadRef>) -> bool {
        let mut state = self.lock();
        state.reads.push(read);
        state.cancelled
    }

    fn lock(&self) -> MutexGuard<'_, ImportState> {
        lock(&self.state)
    }
}

struct PoolState {
    imports: HashMap<BufferId, ImportEntry>,
    available: VecDeque<Arc<ImportWorkspace>>,
    closed: bool,
}

/// Share a bounded set of import workspaces among host tasks, and retain
/// every reserved import until its physical accesses ended.
pub(crate) struct CacheImporter {
    state: Mutex<PoolState>,
    available: Condvar,
}

impl CacheImporter {
    pub(crate) fn new(workspaces: Vec<Arc<ImportWorkspace>>) -> Self {
        Self {
            state: Mutex::new(PoolState {
                imports: HashMap::new(),
                available: workspaces.into(),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    pub(crate) fn require_retired(&self) -> Result<()> {
        if !self.lock().imports.is_empty() {
            return Err(Error::Resource("cache imports still own physical storage"));
        }
        Ok(())
    }

    fn lock(&self) -> MutexGuard<'_, PoolState> {
        lock(&self.state)
    }

    /// Reserve native cache intervals and register their owner in one critical
    /// section. The reservation callback performs no foreign or device work.
    pub(crate) fn reserve(
        &self,
        write: ImportEntry,
        reserve_destination: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let mut state = self.lock();
        if state.closed || state.imports.contains_key(&write.buffer) {
            return Err(Error::Invalid(
                "cache import destination is closed or already reserved".into(),
            ));
        }
        reserve_destination()?;
        state.imports.insert(write.buffer, write);
        Ok(())
    }

    /// The backend first observes successful copy-task completion. Physical
    /// transport accesses may outlive that result and continue to hold pages.
    pub(crate) fn adopt(&self, write: &CacheImport) -> Result<()> {
        let state = self.lock();
        let owned = state
            .imports
            .get(&write.buffer)
            .is_some_and(|active| std::ptr::eq(&**active, write));
        let mut write = write.lock();
        if !owned || write.cancelled {
            return Err(Error::Invalid(
                "cache import destination is no longer active".into(),
            ));
        }
        write.released = true;
        Ok(())
    }

    /// Wait only on an import lane, never on the request execution thread.
    /// Cancellation and shutdown wake blocked workspace borrowers.
    pub(crate) fn acquire(&self, write: &CacheImport) -> Result<Arc<ImportWorkspace>> {
        let mut state = self.lock();
        loop {
            if state.closed || write.cancelled() {
                return Err(Error::Resource("cache import was cancelled"));
            }
            if let Some(workspace) = state.available.pop_front() {
                write.lock().workspace = Some(Arc::clone(&workspace));
                return Ok(workspace);
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub(crate) fn require_active(&self, write: &CacheImport) -> Result<()> {
        if self.lock().closed || write.cancelled() {
            return Err(Error::Resource("cache import was cancelled"));
        }
        Ok(())
    }

    pub(crate) fn abandon(&self, write: &CacheImport) -> Vec<Arc<ReadRef>> {
        let state = self.lock();
        let reads = if state
            .imports
            .get(&write.buffer)
            .is_some_and(|active| std::ptr::eq(&**active, write))
        {
            cancel(write)
        } else {
            Vec::new()
        };
        drop(state);
        self.available.notify_all();
        reads
    }

    pub(crate) fn release(&self, buffers: &HashSet<BufferId>) -> Vec<Arc<ReadRef>> {
        self.cancel_matching(|buffer| buffers.contains(&buffer), false)
    }

    pub(crate) fn cancel_requests(
        &self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Vec<Arc<ReadRef>> {
        self.cancel_matching(
            |buffer| requests.contains(&buffer.owner) && !retained.contains(&buffer),
            false,
        )
    }

    pub(crate) fn stop(&self) -> Vec<Arc<ReadRef>> {
        self.cancel_matching(|_| true, true)
    }

    fn cancel_matching(
        &self,
        selected: impl Fn(BufferId) -> bool,
        close: bool,
    ) -> Vec<Arc<ReadRef>> {
        let mut state = self.lock();
        state.closed |= close;
        let reads = state
            .imports
            .iter()
            .filter(|(buffer, _)| selected(**buffer))
            .flat_map(|(_, write)| cancel(write))
            .collect();
        drop(state);
        self.available.notify_all();
        reads
    }

    /// Return retired owners for completion notification after unlocking.
    /// Failed or cancelled physical completions continue to retain their reads.
    pub(crate) fn reap(&self) -> Vec<ImportEntry> {
        let mut retired_reads = Vec::new();
        let mut retired = Vec::new();
        {
            let mut state = self.lock();
            let mut buffers = Vec::new();
            let mut workspaces = Vec::new();
            for (&buffer, write) in &state.imports {
                let mut write = write.lock();
                let mut index = 0;
                while index < write.reads.len() {
                    if write.reads[index].succeeded() {
                        retired_reads.push(write.reads.swap_remove(index));
                    } else {
                        index += 1;
                    }
                }
                if !write.finished || !write.drained || !write.reads.is_empty() {
                    continue;
                }
                if let Some(workspace) = write.workspace.take() {
                    workspaces.push(workspace);
                }
                if write.released {
                    buffers.push(buffer);
                }
            }
            state.available.extend(workspaces);
            for buffer in buffers {
                if let Some(write) = state.imports.remove(&buffer) {
                    retired.push(write);
                }
            }
        }
        self.available.notify_all();
        drop(retired_reads);
        retired
    }
}

fn cancel(write: &CacheImport) -> Vec<Arc<ReadRef>> {
    let mut state = write.lock();
    state.cancelled = true;
    state.released = true;
    state.reads.clone()
}
