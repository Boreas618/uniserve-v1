//! A rank's view of one host page pool of its host's Storage Server.
//!
//! The server keeps page pools of host storage: a memfd of equal slots per
//! page layout and NUMA node, shared by every process of the host that asks
//! for the same layout (`storage::protocol::Request::PagePool`). The host
//! KV cache maps one per cache group, a denoising rank one for its latent
//! trajectories, and an encoder rank its host's encoder pool; each maps it
//! here, writable and registered for device copies, and addresses its slots
//! through it. The engine decides which slots hold what; the rank copies.

use std::sync::Arc;

use ::storage::protocol::{PageShape, Reply, Request};
use platform::{DeviceAttribute, DeviceId};

use crate::memory::allocation::{Backing, ElementType, Storage, StorageView};
use crate::memory::retained::Retained;
use crate::{Error, Result};

/// The host NUMA node closest to `device`, on which a pool the device copies
/// to and from lies, so the copies cross no inter-node link; `None` when the
/// device does not state one.
pub(crate) fn numa_node(device: DeviceId) -> Option<u32> {
    device
        .attribute(DeviceAttribute::HostNumaId)
        .ok()
        .and_then(|node| u32::try_from(node).ok())
}

/// The pages a pool of request state holds when the requests in flight
/// hold `running` pages: theirs and as many again for preempted requests
/// waiting to resume. A request that resumes is admitted before every
/// request that starts (`Scheduler::queue_media`), so the requests still to
/// resume hold at most what the latest preemption stored, and an epoch on
/// trial stores its own while the epoch it succeeds keeps its for a
/// rollback.
pub(crate) fn resuming_pages(running: u64) -> u64 {
    2 * running
}

/// One page pool, mapped into the rank and registered with its device.
pub(crate) struct HostPages {
    /// The pool's identity at the Storage Server.
    pool: u64,
    storage: Arc<Storage>,
    slots: u64,
    page_bytes: u64,
}

impl HostPages {
    /// Map the pool of `shape` on `numa_node` at the rank's host's Storage
    /// Server, registered with `device`; the server creates it with `bytes`
    /// when it holds none, and answers the pool it holds otherwise, at its
    /// size. `what` names the pool in errors.
    ///
    /// # Errors
    ///
    /// When the server refuses the pool, answers without its storage, or
    /// the storage cannot be mapped.
    pub(crate) fn map(
        retained: &Retained,
        device: DeviceId,
        shape: PageShape,
        numa_node: Option<u32>,
        bytes: Option<u64>,
        what: &'static str,
    ) -> Result<Self> {
        let answer = retained.call(what, |client| {
            client.call(
                &Request::PagePool {
                    shape,
                    numa_node,
                    bytes,
                },
                &[],
            )
        })?;
        let Reply::PagePool {
            pool,
            slots,
            page_bytes,
        } = answer.reply
        else {
            return Err(Error::Interface(format!(
                "the storage server answered {what} with {:?}",
                answer.reply
            )));
        };
        let descriptor = answer.descriptors.into_iter().next().ok_or_else(|| {
            Error::Interface(format!("{what}: the pool arrived without its storage"))
        })?;
        let storage = Storage::map_host(descriptor, (slots * page_bytes) as usize, device, true)?;
        tracing::info!(
            pool,
            slots,
            page_bytes,
            numa_node,
            what,
            "mapped a host page pool"
        );
        Ok(Self {
            pool,
            storage,
            slots,
            page_bytes,
        })
    }

    /// The pool's identity at the Storage Server.
    pub(crate) fn pool(&self) -> u64 {
        self.pool
    }

    /// The bytes of one page.
    pub(crate) fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    /// The host address of host page `page`.
    ///
    /// # Errors
    ///
    /// When the page lies outside the pool.
    pub(crate) fn address(&self, page: u64) -> Result<u64> {
        if page >= self.slots {
            return Err(Error::Invalid(format!(
                "host page {page} lies outside pool {} of {} pages",
                self.pool, self.slots
            )));
        }
        Ok(self.storage.address() + page * self.page_bytes)
    }

    /// The `uint8 [page_bytes]` view of host page `page`.
    ///
    /// # Errors
    ///
    /// When the page lies outside the pool.
    pub(crate) fn view(&self, page: u64) -> Result<StorageView> {
        self.address(page)?;
        StorageView::contiguous(
            Arc::clone(&self.storage) as Arc<dyn Backing>,
            (page * self.page_bytes) as usize,
            ElementType::U8,
            &[self.page_bytes as i64],
        )
    }
}
