//! A rank's mapping of one host page pool of its host's Storage Server.
//!
//! The server keeps page pools of host storage: a memfd of equal slots per page shape and NUMA
//! node, shared by every process of the host that asks for the same shape
//! (`storage::protocol::Request::Pool`). The host KV cache maps one per cache group, a
//! denoising rank one for its latent trajectories, and an encoder rank its host's encoder pool;
//! each maps it here, writable and registered for device copies, and addresses its slots
//! through it. The engine decides which slots hold what; the rank copies.

use std::sync::Arc;

use platform::{DeviceAttribute, DeviceId};
use storage::protocol::{PageShape, Request, Response};

use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::{Error, Result};

/// The host NUMA node closest to `device`, on which a pool the device copies to and from lies,
/// so the copies cross no link between nodes; `None` when the device does not state one.
pub fn numa_node(device: DeviceId) -> Option<u32> {
    device
        .attribute(DeviceAttribute::HostNumaId)
        .ok()
        .and_then(|node| u32::try_from(node).ok())
}

/// The pages a pool of request state holds when the requests in flight hold `running` pages:
/// theirs and as many again for preempted requests waiting to resume.
///
/// A request that resumes is admitted before every request that starts, so the requests still
/// to resume hold at most what the latest preemption stored, and an epoch on trial stores its
/// own while the epoch it succeeds keeps its pages for a rollback.
pub fn resuming_pages(running: u64) -> u64 {
    2 * running
}

/// One page pool, mapped into the rank and registered with its device.
pub struct HostPages {
    /// The pool's identity at the Storage Server.
    pool: u64,
    storage: Arc<Storage>,
    /// How many pages the pool holds.
    slots: u64,
    /// The shape of the pool's pages, with the page bytes the server holds them at.
    shape: PageShape,
}

impl HostPages {
    /// Maps the pool of `shape` on `numa_node` at the rank's host's Storage Server, registered
    /// with `device`. The server creates the pool with `bytes` when it holds none, and returns
    /// the pool it holds otherwise, at its size. `what` names the pool in errors.
    ///
    /// # Errors
    ///
    /// When the server refuses the pool, responds without its storage, or the storage cannot
    /// be mapped.
    pub fn map(
        client: &StorageClient,
        device: DeviceId,
        shape: PageShape,
        numa_node: Option<u32>,
        bytes: Option<u64>,
        what: &'static str,
    ) -> Result<Self> {
        let (response, descriptors) = client.call(what, |client| {
            client.call(
                &Request::Pool {
                    shape,
                    numa_node,
                    bytes,
                },
                &[],
            )
        })?;
        let Response::Pool { pool, slots } = response else {
            return Err(Error::Interface(format!(
                "the storage server responded to {what} with {response:?}"
            )));
        };
        let descriptor = descriptors.into_iter().next().ok_or_else(|| {
            Error::Interface(format!("{what}: the pool arrived without its storage"))
        })?;

        let page_bytes = shape.page_bytes;
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
            shape,
        })
    }

    /// The pool's identity at the Storage Server.
    pub fn pool(&self) -> u64 {
        self.pool
    }

    /// The bytes of one page.
    pub fn page_bytes(&self) -> u64 {
        self.shape.page_bytes
    }

    /// The shape of the pool's pages: their layout, the KV heads a page holds and its bytes.
    pub fn shape(&self) -> PageShape {
        self.shape
    }

    /// The host address of page `page`, the first of [`Self::page_bytes`] bytes a device copy
    /// reads or writes.
    ///
    /// # Errors
    ///
    /// When the page lies outside the pool.
    pub fn address(&self, page: u64) -> Result<u64> {
        if page >= self.slots {
            return Err(Error::Invalid(format!(
                "host page {page} lies outside pool {} of {} pages",
                self.pool, self.slots
            )));
        }
        Ok(self.storage.address() + page * self.shape.page_bytes)
    }
}
