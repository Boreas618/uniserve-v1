//! The host KV cache (L2) of the rank's prefix cache.
//!
//! Each cache group copies its pages through a page pool of the host's Storage
//! Server: a memfd of equal slots shared by every worker of the host whose
//! pages have the same shape. The rank maps each pool writable and registers it
//! for device copies; the engine decides which pages to copy and carries them
//! in batches (`Batch::page_loads`, `Batch::page_offloads`), and the copies run
//! on the rank's copy stream as soon as the batch arrives
//! (`PrefixCache::issue_pages`).
//!
//! A page is one page of one group on this rank, layer-major:
//!
//! 1. keys `[layers, block_size, kv_heads, head_dim]`;
//! 2. values, in the same shape;
//! 3. for FP8 pools, the scales `[2, layers]` as `f32`.
//!
//! One layer of one page fills exactly one plane of the unit pool (a
//! group's block size is the plane over its row bytes), so a page copies as
//! one plane-sized run per field and layer, and one scale per field and
//! layer.

use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::{CopyBatch, DeviceAttribute, DeviceId, SourceOrder};
use ipc::{PageCopy, PagePoolInfo};
use sha2::{Digest, Sha256};
use storage::Key;
use storage::protocol::{PageShape, Reply, Request};

use super::plan::{CacheGroupPlan, CachePlan};
use super::planes::{CachePlanes, UnitColumn};
use crate::ffi::Backing;
use crate::memory::allocation::Storage;
use crate::memory::retained::Retained;
use crate::stream::Issue;
use crate::{Error, Result};

/// The version of the page encoding above: pages of another version are
/// another content.
const PAGE_ENCODING: u32 = 1;

/// One group's page pool, mapped into the rank.
struct GroupPool {
    pool: u64,
    memory: Arc<Storage>,
    slots: u64,
    page_bytes: u64,
    layers: u32,
}

/// The rank's page pools, by cache group.
pub(crate) struct HostKvCache {
    pools: BTreeMap<u32, GroupPool>,
}

/// The bytes of one page of `group` on a pool of `plan`'s layout.
fn page_bytes(plan: &CachePlan, group: &CacheGroupPlan) -> u64 {
    let layers = group.layer_ids.len() as u64;
    let scales = if plan.quantized { 2 * layers * 4 } else { 0 };
    2 * layers * plan.plane_bytes + scales
}

/// The digest of the byte layout of `group`'s pages: the page encoding,
/// the cache dtype and every group's shape, and the group itself. It
/// identifies the group's page pool (`PageShape::layout`).
fn layout(plan: &CachePlan, group: usize) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-page-layout\0");
    digest.update(PAGE_ENCODING.to_le_bytes());
    digest.update([u8::from(plan.quantized)]);
    digest.update(plan.plane_bytes.to_le_bytes());
    digest.update((plan.groups.len() as u64).to_le_bytes());
    for entry in &plan.groups {
        digest.update(entry.window.unwrap_or(0).to_le_bytes());
        digest.update(entry.block_size.to_le_bytes());
        digest.update((entry.layer_ids.len() as u64).to_le_bytes());
        for layer in &entry.layer_ids {
            digest.update(layer.to_le_bytes());
        }
        digest.update(entry.total_kv_heads.to_le_bytes());
        digest.update(entry.head_dim.to_le_bytes());
        digest.update(entry.dtype.name().as_bytes());
        digest.update([0]);
    }
    digest.update((group as u64).to_le_bytes());
    digest.finalize().into()
}

/// The digest of what determines the values of `group`'s pages: the model
/// fingerprint and the pages' layout. It seeds the group's page keys.
fn content(fingerprint: &str, layout: &[u8; 32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-page-content\0");
    digest.update((fingerprint.len() as u64).to_le_bytes());
    digest.update(fingerprint.as_bytes());
    digest.update(layout);
    digest.finalize().into()
}

/// The key seed of each of `plan`'s groups' page pools for a model of
/// `fingerprint` (`HostKvCache::start` reports them).
pub(crate) fn page_contents(plan: &CachePlan, fingerprint: &str) -> Vec<[u8; 32]> {
    (0..plan.groups.len())
        .map(|group| content(fingerprint, &layout(plan, group)))
        .collect()
}

impl HostKvCache {
    /// Map a page pool for every cache group of `plan` at the rank's host's
    /// Storage Server, registered with `device`. A sliding-window group's
    /// pool holds the windows the engine offloads before aligned
    /// boundaries, page by page, as a full-attention group's holds every
    /// published page. `fingerprint`
    /// is the pages' key digest (`Fingerprints::pages`).
    /// The pools share `bytes` so that each holds as many pages as the
    /// others; each takes the server's size when `None`.
    /// Returns the host KV cache and what the rank reports of each pool.
    pub(crate) fn start(
        retained: &Retained,
        plan: &CachePlan,
        device: DeviceId,
        fingerprint: &str,
        endpoint: Option<&str>,
        bytes: Option<u64>,
    ) -> Result<(Self, Vec<PagePoolInfo>)> {
        let mut pools = BTreeMap::new();
        let mut infos = Vec::new();
        // One page of every group per slot of the shared budget.
        let slots = match bytes {
            Some(bytes) => {
                let page_set: u64 = plan
                    .groups
                    .iter()
                    .map(|group| page_bytes(plan, group))
                    .sum();
                let slots = bytes.checked_div(page_set).unwrap_or(0);
                if slots == 0 {
                    return Err(Error::Resource(
                        "the host KV cache holds no page of each cache group",
                    ));
                }
                Some(slots)
            }
            None => None,
        };
        for (index, group) in plan.groups.iter().enumerate() {
            let layout = layout(plan, index);
            let content = content(fingerprint, &layout);
            // The pool lies on the host NUMA node closest to the device, so
            // the copies between them cross no inter-node link.
            let numa_node = device
                .attribute(DeviceAttribute::HostNumaId)
                .ok()
                .and_then(|node| u32::try_from(node).ok());
            let shape = PageShape {
                layout: Key(layout),
                kv_head_offset: group.kv_head_offset,
                kv_head_count: group.num_kv_heads,
                page_bytes: page_bytes(plan, group),
            };
            let bytes = slots.map(|slots| slots * shape.page_bytes);
            let answer = retained.call("mapping a page pool", |client| {
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
                    "the storage server answered a page pool with {:?}",
                    answer.reply
                )));
            };
            let descriptor =
                answer.descriptors.into_iter().next().ok_or_else(|| {
                    Error::Interface("a page pool arrived without its memory".into())
                })?;
            let memory =
                Storage::map_host(descriptor, (slots * page_bytes) as usize, device, true)?;
            tracing::info!(
                group = index,
                pool,
                slots,
                page_bytes,
                numa_node,
                "mapped the host page pool"
            );
            pools.insert(
                index as u32,
                GroupPool {
                    pool,
                    memory,
                    slots,
                    page_bytes,
                    layers: group.layer_ids.len() as u32,
                },
            );
            infos.push(PagePoolInfo {
                group_id: index as u32,
                pool,
                content,
                endpoint: endpoint.map(str::to_owned),
            });
        }
        Ok((Self { pools }, infos))
    }

    /// One page's copies between its slot and its units: `(device, host,
    /// bytes)` for every field and layer, then every scale.
    fn runs(&self, planes: &CachePlanes, copy: &PageCopy) -> Result<Vec<(u64, u64, usize)>> {
        let pool = self
            .pools
            .get(&copy.group_id)
            .filter(|pool| pool.pool == copy.pool)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "page pool {} is not the pool of cache group {}",
                    copy.pool, copy.group_id
                ))
            })?;
        if copy.slot >= pool.slots {
            return Err(Error::Invalid(format!(
                "page slot {} is outside the pool of {} slots",
                copy.slot, pool.slots
            )));
        }
        let plan = planes.plan();
        let columns = plan.columns;
        if copy.units.len() as u32 * columns != pool.layers {
            return Err(Error::Invalid(format!(
                "a page of {} layers spans {} units of {columns} columns, not {}",
                pool.layers,
                pool.layers.div_ceil(columns),
                copy.units.len()
            )));
        }
        let plane = plan.plane_bytes;
        let slot = pool.memory.address() + copy.slot * pool.page_bytes;
        let layers = u64::from(pool.layers);
        let mut runs = Vec::with_capacity(2 * pool.layers as usize * 2);
        for field in 0..2 {
            for layer in 0..pool.layers {
                let unit = copy.units[(layer / columns) as usize].0;
                let device = planes.plane_address(field, layer % columns, unit)?;
                let host = slot + (field as u64 * layers + u64::from(layer)) * plane;
                runs.push((device, host, plane as usize));
            }
        }
        if plan.quantized {
            for field in 0..2 {
                for layer in 0..pool.layers {
                    let unit = copy.units[(layer / columns) as usize].0;
                    let device = planes.scale_address(field, layer % columns, unit)?;
                    let host =
                        slot + 2 * layers * plane + (field as u64 * layers + u64::from(layer)) * 4;
                    runs.push((device, host, 4));
                }
            }
        }
        Ok(runs)
    }

    /// Load the units of `loads` from their slots in `stream` order. The
    /// units are new units of the batch, which the copies overwrite whole
    /// (every column of both fields, and an FP8 pool's scales); an FP8 pool
    /// then marks them initialized with `column`, bound to `stream`.
    pub(crate) fn load(
        &self,
        planes: &CachePlanes,
        loads: &[PageCopy],
        column: &UnitColumn,
        stream: impl Issue,
    ) -> Result<()> {
        if loads.is_empty() {
            return Ok(());
        }
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in loads {
            for (device, host, bytes) in self.runs(planes, copy)? {
                batch.push(device, host, bytes);
            }
        }
        tracing::debug!(
            pages = loads.len(),
            first_slot = loads[0].slot,
            first_unit = loads[0].units.first().map(|unit| unit.0),
            "loading pages from the host KV cache"
        );
        // SAFETY: the slots are pinned by the engine until the batch's output
        // arrives, and the units are the batch's new units, which no other
        // operation reads or writes before the batch's forward, which waits
        // for these copies.
        unsafe { stream.copy(&batch) }?;
        if planes.plan().quantized {
            let units: Vec<u32> = loads
                .iter()
                .flat_map(|copy| copy.units.iter().map(|unit| unit.0))
                .collect();
            column.mark_initialized(planes, &units, stream)?;
        }
        Ok(())
    }

    /// Offload the units of the published pages of `offloads` into their
    /// slots in `stream` order; the batch's output waits for the copies.
    pub(crate) fn offload(
        &self,
        planes: &CachePlanes,
        offloads: &[PageCopy],
        stream: impl Issue,
    ) -> Result<()> {
        if offloads.is_empty() {
            return Ok(());
        }
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in offloads {
            for (device, host, bytes) in self.runs(planes, copy)? {
                batch.push(host, device, bytes);
            }
        }
        tracing::debug!(
            pages = offloads.len(),
            first_slot = offloads[0].slot,
            first_unit = offloads[0].units.first().map(|unit| unit.0),
            "offloading pages to the host KV cache"
        );
        // SAFETY: the slots are reserved for this rank until the engine
        // commits them after the batch's output, and the engine holds the
        // published units, which no batch writes, until then.
        unsafe { stream.copy(&batch) }
    }
}
