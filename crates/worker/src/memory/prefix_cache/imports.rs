//! Imports of other ranks' cache exports into this rank's pages: the import
//! pool, whose bounded workspaces' host tasks copy the exports into assigned
//! pages.

use std::sync::Arc;


use crate::host::HostPool;
use crate::transfer::{CacheImporter, ImportCopy, Retired};
use crate::Result;



/// Import workspaces, the host pool their copies run on and the retirement
/// observers.
pub(super) struct Imports {
    pub(super) importer: Arc<CacheImporter>,
    pub(super) pool: HostPool<ImportCopy>,
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
