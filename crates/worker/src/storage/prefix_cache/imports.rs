//! Imports of other ranks' cache exports into this rank's pages: the
//! reserved import ([`PendingImport`]) and the import lane, whose bounded
//! workspaces' host tasks copy the exports into assigned pages.

use std::sync::Arc;

use uniserve_ipc::{BufferId, CacheTransfer};

use super::block_tables::GroupTable;
use crate::host::HostLane;
use crate::transfer::{
    CacheImporter, ImportEntry, ImportTask, ImportWorkspace, Retired, Retirement,
};
use crate::{Error, HostTask, Outcome, Result};

/// A reserved cache import: the destination tables and units it resets, the
/// export it installs, and its copy task (none when nothing is copied).
pub struct PendingImport {
    pub(super) entry: ImportEntry,
    pub(super) tables: Vec<Arc<GroupTable>>,
    pub(super) export: Arc<CacheTransfer>,
    pub(super) task: Option<Arc<HostTask<ImportTask>>>,
}

impl PendingImport {
    pub fn buffer(&self) -> BufferId {
        self.entry.buffer
    }

    pub fn slot(&self) -> usize {
        self.entry.request_pool_idx
    }

    pub fn export(&self) -> &Arc<CacheTransfer> {
        &self.export
    }

    /// Whether an install call adopted the import as its slot's prefix.
    pub fn adopted(&self) -> bool {
        self.entry.write().released()
    }

    /// Whether the copy task finished (successfully or not).
    pub fn done(&self) -> bool {
        self.task.as_ref().is_none_or(|task| task.completion.done())
    }

    /// The import's physical retirement: it completes once the import was
    /// adopted or abandoned and every read and the workspace stream drained.
    pub fn retirement(&self) -> &Arc<Retirement> {
        self.entry.retirement()
    }

    /// Run `callback` once the copy task finished.
    pub fn subscribe(&self, callback: Box<dyn FnOnce() + Send>) {
        match &self.task {
            Some(task) => {
                if let Some(callback) = task.completion.subscribe(callback) {
                    callback();
                }
            }
            None => callback(),
        }
    }

    /// The copy's result, without waiting.
    pub(super) fn result(&self) -> Result<()> {
        let Some(task) = &self.task else {
            return Ok(());
        };
        match task.completion.outcome() {
            Some(Outcome::Success(_)) => Ok(()),
            Some(Outcome::Failed(error)) => {
                Err(Error::Transport(format!("cache import failed: {error}")))
            }
            Some(Outcome::Cancelled) => Err(Error::Resource("cache import was cancelled")),
            None => Err(Error::State(
                "cache import was observed before input readiness",
            )),
        }
    }
}

/// Import workspaces, their copy lane and the retirement observers.
pub(super) struct Imports {
    pub(super) importer: Arc<CacheImporter<ImportEntry, ImportWorkspace>>,
    pub(super) lane: HostLane<ImportTask>,
    pub(super) retired: Arc<Retired>,
}

impl Imports {
    pub(super) fn cancel(&self, reads: Vec<Arc<crate::transfer::ReadRef>>) -> Result<()> {
        let mut result = Ok(());
        for read in reads {
            result = result.and(read.read().cancel());
        }
        result.and(self.retired.reap())
    }
}
