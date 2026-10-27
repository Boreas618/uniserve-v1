//! The host latent pool of a video denoising rank: the Storage Server page
//! pool its preempted requests' trajectories are written to and restored
//! from (`ipc::LatentOffload`, `ipc::LatentRestore`).
//!
//! A pool page is one page of a request slot's latent banks: `page_units`
//! float32 values, the rank's own share of the trajectory, in the order the
//! banks hold them. The pool is shared by every denoising rank of the
//! deployment on the host's NUMA node whose pages have that size, in every
//! world of the deployment, so a world that succeeds another restores the
//! trajectories its predecessor stored. The engine decides which pages move
//! and keys them; the rank only copies, on a copy stream of its device.

use std::sync::Arc;

use cuda::{DeviceAttribute, DeviceId};
use ipc::{LatentPageCopy, LatentPoolInfo};
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

/// How many trajectories of every request slot the pool holds: those of
/// the requests one preemption ends, and as many again for a preemption
/// while the first ones' pages are still pinned by their restores.
const SLOT_TRAJECTORIES: u64 = 2;

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

/// The rank's host latent pool, mapped and registered for device copies.
pub(crate) struct LatentHostPool {
    pool: u64,
    memory: Arc<Storage>,
    slots: u64,
    page_bytes: u64,
    page_units: u32,
}

/// The digest of the byte layout of a pool's pages, which identifies the
/// pool (`PageShape::layout`): the page encoding and the page's values.
fn layout(page_units: u32) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-page-layout\0");
    digest.update(PAGE_ENCODING.to_le_bytes());
    digest.update(DType::F32.name().as_bytes());
    digest.update([0]);
    digest.update(page_units.to_le_bytes());
    digest.finalize().into()
}

/// The digest of what one of the rank's pages means besides the numerics
/// of the denoising: the page encoding, the slot's page count and the
/// executor's statement of the video plan and the rank's share of the
/// denoiser (`region`, from its manifest).
fn rank_region(page_units: u32, slot_pages: u32, region: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-latent-region\0");
    digest.update(layout(page_units));
    digest.update(slot_pages.to_le_bytes());
    digest.update((region.len() as u64).to_le_bytes());
    digest.update(region.as_bytes());
    digest.finalize().into()
}

impl LatentHostPool {
    /// Map the latent pool for pages of `slots.page_units` float32 values
    /// at the rank's host's Storage Server, registered with `device`, sized
    /// for [`SLOT_TRAJECTORIES`] trajectories of each of the rank's request
    /// slots on each of the host's devices, since every denoising rank on
    /// the device's NUMA node shares it. `identity` keys what the rank's
    /// denoising computes (`Fingerprints::pages`), `region` what its pages
    /// mean, and `endpoint` is where an engine on another host reaches the
    /// server. Returns the pool and what the rank reports of it.
    pub(crate) fn start(
        retained: &Retained,
        device: DeviceId,
        slots: SlotPages,
        identity: &str,
        region: &str,
        endpoint: Option<&str>,
    ) -> Result<(Self, LatentPoolInfo)> {
        let SlotPages {
            page_units,
            pages: slot_pages,
            slots: request_slots,
        } = slots;
        let page_bytes = u64::from(page_units) * DType::F32.bytes() as u64;
        // The pool lies on the host NUMA node closest to the device, so the
        // copies between them cross no inter-node link.
        let numa_node = device
            .attribute(DeviceAttribute::HostNumaId)
            .ok()
            .and_then(|node| u32::try_from(node).ok());
        let devices = DeviceId::count()? as u64;
        let slots =
            SLOT_TRAJECTORIES * request_slots as u64 * u64::from(slot_pages) * devices.max(1);
        let shape = PageShape {
            layout: Key(layout(page_units)),
            kv_head_offset: 0,
            kv_head_count: 0,
            page_bytes,
        };
        let answer = retained.call("mapping the host latent pool", |client| {
            client.call(
                &Request::PagePool {
                    shape,
                    numa_node,
                    bytes: Some(slots * page_bytes),
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
                Error::Interface("the latent pool arrived without its memory".into())
            })?;
        let memory = Storage::map_host(descriptor, (slots * page_bytes) as usize, device, true)?;
        tracing::info!(
            pool,
            slots,
            page_bytes,
            numa_node,
            "mapped the host latent pool"
        );
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
                memory,
                slots,
                page_bytes,
                page_units,
            },
            info,
        ))
    }

    /// The host address of the pool slot `copy` names, after checking that
    /// it is this pool's.
    pub(crate) fn address(&self, copy: &LatentPageCopy) -> Result<u64> {
        if copy.pool != self.pool || copy.slot >= self.slots {
            return Err(Error::Invalid(format!(
                "latent pool slot {} of pool {} is not a slot of this rank's pool {} of {} slots",
                copy.slot, copy.pool, self.pool, self.slots
            )));
        }
        Ok(self.memory.address() + copy.slot * self.page_bytes)
    }

    /// The bytes of one page.
    pub(crate) fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    /// The `[page_units]` float32 view of the pool slot `copy` names, which
    /// a restore reads.
    pub(crate) fn view(&self, copy: &LatentPageCopy) -> Result<StorageView> {
        self.address(copy)?;
        StorageView::contiguous(
            Arc::clone(&self.memory) as Arc<dyn Backing>,
            (copy.slot * self.page_bytes) as usize,
            DType::F32,
            &[i64::from(self.page_units)],
        )
    }
}
