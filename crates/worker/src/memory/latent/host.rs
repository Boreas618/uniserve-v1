//! The host latent pool of a video denoising rank: the Storage Server page
//! pool its preempted requests' trajectories are written to
//! (`ipc::LatentOffload`), and the pools of the stored ranks a resumed
//! request's trajectory is restored from (`ipc::LatentRestore`).
//!
//! A pool page is one page of a request slot's latent banks: `page_units`
//! float32 values of a stored rank's own share of the trajectory, in the
//! order its banks held them. The pool is the rank's: the same rank (its
//! worker group and its rank) of every world of the deployment whose pages
//! have that size maps it, whatever its parallel configuration, so a world
//! that succeeds another finds the trajectories its predecessor stored, and
//! it is sized for the trajectories that rank can have stored at once
//! ([`pool_pages`]). A restore reads the pages of every stored rank on the
//! host, so a rank also maps, on first use, the pools of the other ranks a
//! trajectory was stored by ([`LatentHostPool::source`]). The engine
//! decides which pages move and keys them; the rank only copies, on a copy
//! stream of its device.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cuda::{DeviceAttribute, DeviceId};
use ipc::{LatentPageCopy, LatentPoolInfo, LatentPoolRef};
use sha2::{Digest, Sha256};
use storage::Key;
use storage::protocol::{PageShape, Reply, Request};

use crate::ffi::{Backing, DType};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::retained::Retained;
use crate::{Error, Result};

/// The version of the page encoding above: pages of another version are
/// another shape.
const PAGE_ENCODING: u32 = 1;

/// The pages a rank's pool holds: every trajectory the rank can have
/// stored at once, `slots.pages` pages each.
///
/// - One preemption (a park, a sleep, a weight update) stores the
///   trajectory of each request the rank's slots hold: at most
///   `slots.slots` trajectories.
/// - A stored trajectory is needed until the request's latent preparation
///   restores it. A request that resumes one is admitted before every
///   request that starts one (`Scheduler::queue_media`), so the slots take
///   the resumed requests first, and the trajectories a world still has to
///   restore are at most those of its latest preemption: `slots.slots`.
/// - A world on trial stores its own trajectories while the world it
///   succeeds keeps the ones it stored, which a rollback restores: as many
///   again.
///
/// A trajectory the pool evicted beyond that, as when requests reach the
/// engine in another order, is a miss: its request starts from noise.
fn pool_pages(slots: SlotPages) -> u64 {
    2 * slots.slots as u64 * u64::from(slots.pages)
}

/// The latent pages of a rank's request slots.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotPages {
    /// The float32 values of one page.
    pub(crate) page_units: u32,
    /// The pages one request slot owns.
    pub(crate) pages: u32,
    /// The request slots.
    pub(crate) slots: usize,
}

/// The rank's host latent pool, mapped and registered for device copies,
/// and the pools of other ranks it read stored pages from.
pub(crate) struct LatentHostPool {
    pool: LatentPoolRef,
    storage: Arc<Storage>,
    slots: u64,
    retained: Arc<Retained>,
    device: DeviceId,
    /// The other ranks' pools mapped for restores, by pool, with their
    /// slots and page bytes.
    sources: Mutex<HashMap<u64, (Arc<Storage>, u64)>>,
}

/// The digest of the byte layout of a pool's pages: the page encoding and
/// the page's values.
fn layout(page_units: u32) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-page-layout\0");
    digest.update(PAGE_ENCODING.to_le_bytes());
    digest.update(DType::F32.name().as_bytes());
    digest.update([0]);
    digest.update(page_units.to_le_bytes());
    digest.finalize().into()
}

/// The digest of what determines the rank's share of every request's
/// samples (`LatentPoolInfo::region`): the page encoding, the slot's page
/// count and the executor's statement of the video plan and the rank's
/// share of the denoiser (`region`, from its manifest).
fn rank_region(page_units: u32, slot_pages: u32, region: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-region\0");
    digest.update(layout(page_units));
    digest.update(slot_pages.to_le_bytes());
    digest.update((region.len() as u64).to_le_bytes());
    digest.update(region.as_bytes());
    digest.finalize().into()
}

/// The digest that identifies the rank's pool (`PageShape::layout`): the
/// byte layout of its pages, and the worker group and rank it serves, so
/// that each rank maps a pool of its own, sized for its own trajectories,
/// which the same rank of a succeeding world maps again whatever its
/// parallel configuration.
fn rank_pool(page_units: u32, scope: &storage::Scope) -> [u8; 32] {
    let mut key = storage::key::Digest::new(b"latent-pool");
    key.bytes(&layout(page_units));
    key.text(&scope.group);
    key.number(u64::from(scope.rank));
    key.finish()
}

/// Map the page pool of `shape` on `numa_node` at the rank's host's
/// Storage Server, creating it with `bytes` when the server holds none,
/// registered with `device`. Returns its identity, its storage and its
/// slots.
fn map_pool(
    retained: &Retained,
    device: DeviceId,
    shape: PageShape,
    numa_node: Option<u32>,
    bytes: Option<u64>,
) -> Result<(u64, Arc<Storage>, u64)> {
    let answer = retained.call("mapping a host latent pool", |client| {
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
            "the storage server answered the latent pool with {:?}",
            answer.reply
        )));
    };
    let descriptor =
        answer.descriptors.into_iter().next().ok_or_else(|| {
            Error::Interface("the latent pool arrived without its storage".into())
        })?;
    let storage = Storage::map_host(descriptor, (slots * page_bytes) as usize, device, true)?;
    Ok((pool, storage, slots))
}

impl LatentHostPool {
    /// Map the rank's latent pool for pages of `slots.page_units` float32
    /// values at its host's Storage Server, registered with `device` and
    /// sized by [`pool_pages`]. `identity` keys what the rank's denoising
    /// computes (`Fingerprints::pages`), `region` states what determines the
    /// rank's share of a request's samples, and `endpoint` is where an
    /// engine on another host reaches the server. Returns the pool and what
    /// the rank reports of it.
    pub(crate) fn start(
        retained: &Arc<Retained>,
        device: DeviceId,
        slots: SlotPages,
        identity: &str,
        region: &str,
        endpoint: Option<&str>,
    ) -> Result<(Self, LatentPoolInfo)> {
        let SlotPages {
            page_units,
            pages: slot_pages,
            ..
        } = slots;
        let page_bytes = u64::from(page_units) * DType::F32.bytes() as u64;
        // The pool lies on the host NUMA node closest to the device, so the
        // copies between them cross no inter-node link.
        let numa_node = device
            .attribute(DeviceAttribute::HostNumaId)
            .ok()
            .and_then(|node| u32::try_from(node).ok());
        let shape = PageShape {
            layout: Key(rank_pool(page_units, retained.scope())),
            kv_head_offset: 0,
            kv_head_count: 0,
            page_bytes,
        };
        let (pool, storage, pool_slots) = map_pool(
            retained,
            device,
            shape,
            numa_node,
            Some(pool_pages(slots) * page_bytes),
        )?;
        tracing::info!(
            pool,
            slots = pool_slots,
            page_bytes,
            numa_node,
            "mapped the host latent pool"
        );
        let pool = LatentPoolRef {
            pool,
            shape: shape.layout.0,
            numa_node,
            page_bytes,
        };
        let info = LatentPoolInfo {
            pool,
            slot_pages,
            region: rank_region(page_units, slot_pages, region),
            identity: identity.to_owned(),
            endpoint: endpoint.map(str::to_owned),
        };
        Ok((
            Self {
                pool,
                storage,
                slots: pool_slots,
                retained: Arc::clone(retained),
                device,
                sources: Mutex::new(HashMap::new()),
            },
            info,
        ))
    }

    /// The host address of the pool slot `copy` names, after checking that
    /// it is this pool's.
    pub(crate) fn address(&self, copy: &LatentPageCopy) -> Result<u64> {
        if copy.pool != self.pool.pool || copy.slot >= self.slots {
            return Err(Error::Invalid(format!(
                "latent pool slot {} of pool {} is not a slot of this rank's pool {} of {} slots",
                copy.slot, copy.pool, self.pool.pool, self.slots
            )));
        }
        Ok(self.storage.address() + copy.slot * self.pool.page_bytes)
    }

    /// The bytes of one page of the rank's pool.
    pub(crate) fn page_bytes(&self) -> u64 {
        self.pool.page_bytes
    }

    /// The `uint8 [page_bytes]` view of slot `slot` of `pool`, a pool of
    /// the rank's host that holds stored pages a restore reads: the rank's
    /// own, or another rank's, which is mapped on first use and kept for
    /// later restores.
    ///
    /// Fails when the server holds no such pool or the slot lies outside
    /// it.
    pub(crate) fn source(&self, pool: &LatentPoolRef, slot: u64) -> Result<StorageView> {
        let (storage, slots) = if pool.pool == self.pool.pool {
            (Arc::clone(&self.storage), self.slots)
        } else {
            let mut sources = self
                .sources
                .lock()
                .map_err(|_| Error::State("the latent source pools' lock is poisoned"))?;
            match sources.get(&pool.pool) {
                Some((storage, slots)) => (Arc::clone(storage), *slots),
                None => {
                    let shape = PageShape {
                        layout: Key(pool.shape),
                        kv_head_offset: 0,
                        kv_head_count: 0,
                        page_bytes: pool.page_bytes,
                    };
                    // The pool holds pinned pages, so the server finds it;
                    // a pool of another identity is not the one named.
                    let (found, storage, slots) =
                        map_pool(&self.retained, self.device, shape, pool.numa_node, None)?;
                    if found != pool.pool {
                        return Err(Error::Invalid(format!(
                            "latent pool {} is no longer at the storage server",
                            pool.pool
                        )));
                    }
                    tracing::info!(
                        pool = found,
                        slots,
                        "mapped another rank's host latent pool for a restore"
                    );
                    sources.insert(found, (Arc::clone(&storage), slots));
                    (storage, slots)
                }
            }
        };
        if slot >= slots {
            return Err(Error::Invalid(format!(
                "latent pool slot {slot} lies outside pool {} of {slots} slots",
                pool.pool
            )));
        }
        StorageView::contiguous(
            storage as Arc<dyn Backing>,
            (slot * pool.page_bytes) as usize,
            DType::U8,
            &[pool.page_bytes as i64],
        )
    }
}
