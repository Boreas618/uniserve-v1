//! Storage hazards of a batch's writes: every cache span the batch writes and
//! the physical accesses still overlapping it (executions of earlier batches,
//! imports, and exports other ranks are reading), and for a batch that writes a
//! latent bank the exports of the versions in it. The batch's imports are
//! admitted once these retired; a failed access keeps its storage, so the batch
//! fails instead ([`require_storage`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CacheTransfer, Call, CallKind, MediaCall, TransferMode,
};

use super::{BatchInputs, Hazard};
use crate::executor::backend::WorkerBackend;
use crate::storage::PrefixCache;
use crate::transfer::Retirement;
use crate::{Error, Outcome, Result};

/// What a batch's cache transfers and writes depend on.
pub(super) struct Resolved {
    /// Resident exports the batch's install calls read.
    pub(super) resident: Vec<Arc<CacheTransfer>>,
    /// Physical accesses overlapping the cache storage the batch writes.
    pub(super) hazards: Vec<Arc<Retirement>>,
}

/// Resolve the resident exports install calls read and the physical
/// accesses the batch's cache writes overlap. Neither reserves storage.
pub(super) fn resolve(backend: &mut WorkerBackend, plan: &BatchPlan) -> Result<Resolved> {
    let parts = backend.parts();
    let Some(prefix_cache) = parts
        .storage
        .and_then(|storage| storage.prefix_cache.as_mut())
    else {
        return Ok(Resolved {
            resident: Vec::new(),
            hazards: Vec::new(),
        });
    };

    // Supplied transfers take precedence; other install sources must be
    // exports resident on this rank.
    let mut supplied: HashSet<BufferId> = plan
        .cache_inputs
        .iter()
        .map(|transfer| transfer.source)
        .collect();
    let mut resident = Vec::new();
    for call in installs(plan) {
        let source = call
            .cache_input
            .ok_or_else(|| Error::Invalid("cache installation requires a source export".into()))?;
        if supplied.insert(source) {
            let export = prefix_cache
                .resident(source)
                .ok_or_else(|| Error::Invalid("cache export buffer is not resident".into()))?;
            resident.push(Arc::clone(export));
        }
    }

    if !prefix_cache.has_pending_accesses() {
        return Ok(Resolved {
            resident,
            hazards: Vec::new(),
        });
    }
    let spans = write_spans(prefix_cache, parts.requests, plan, &resident)?;
    let hazards = prefix_cache.write_dependencies(&spans);
    Ok(Resolved { resident, hazards })
}

/// The exports of trajectory versions in the banks the batch's latent
/// preparation and denoising calls write: other ranks still reading a
/// version keep its bank from being overwritten.
///
/// A denoising step writes the bank of the version two steps behind the
/// one it reads, which its trajectory has moved past. That version is
/// revoked here (`LatentPool::supersede`) rather than
/// left published until its Free: the step then waits only for the reads
/// other ranks already claimed, which finish by themselves, and never for
/// a later batch of the head that would carry the Free.
pub(super) fn latent_hazards(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
) -> Result<Vec<Arc<Retirement>>> {
    let writes: Vec<(usize, &uniserve_ipc::LatentParams, bool)> = plan
        .latent_params
        .iter()
        .filter_map(|params| {
            let call = plan.calls.iter().find(|call| {
                call.request_key == params.request_key && call.call_id == params.call_id
            })?;
            matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
            )
            .then_some((params, call.code == CallKind::Media(MediaCall::Denoising)))
        })
        .map(|(params, denoising)| {
            let slot = backend
                .requests()
                .get(params.request_key.request_id.0)?
                .slot();
            Ok((slot, params, denoising))
        })
        .collect::<Result<_>>()?;
    if writes.is_empty() {
        return Ok(Vec::new());
    }
    let parts = backend.parts();
    let Some(mut image) = parts.image else {
        return Ok(Vec::new());
    };
    let Ok((pool, _)) = image.latent_pool() else {
        return Ok(Vec::new());
    };
    pool.reap();
    let mut hazards = Vec::new();
    let mut superseded = Vec::new();
    for (slot, params, denoising) in writes {
        let slot = pool.slot(slot as i64)?;
        let pages: Vec<usize> = params
            .page_table
            .iter()
            .map(|&page| page as usize)
            .collect();
        if denoising {
            superseded.extend(pool.supersede(slot, params.request_key, &pages));
        }
        hazards.extend(pool.write_dependencies(slot, &pages));
    }
    if !superseded.is_empty() {
        // A version is published through the product store's locations;
        // revoking them stops new claims and lets each export retire once
        // its claimed reads finished.
        parts
            .storage
            .ok_or_else(|| Error::Invariant("published latents need storage owners".into()))?
            .tensors
            .revoke_buffer_exports(&superseded)?;
    }
    Ok(hazards)
}

/// The install calls of a batch.
pub(super) fn installs(plan: &BatchPlan) -> impl Iterator<Item = &Call> {
    plan.calls
        .iter()
        .filter(|call| call.code == CallKind::Transfer(TransferMode::CacheInstall))
}

/// Every cache span the batch writes: whole newly assigned units (a recycled
/// unit may have served another group's page size), forward rows that
/// write their queries, and the extent each install copies.
fn write_spans(
    prefix_cache: &mut PrefixCache,
    requests: &crate::request::RequestPool,
    plan: &BatchPlan,
    resident: &[Arc<CacheTransfer>],
) -> Result<Vec<(u32, u32, u32)>> {
    let unit_tokens = prefix_cache.plan().unit_tokens();
    let tables = prefix_cache.tables();
    let mut spans = Vec::new();
    for allocation in &plan.new_cache_units {
        spans.extend(
            allocation
                .unit_ids
                .iter()
                .map(|unit| (unit.0, 0, unit_tokens)),
        );
    }
    let forward = &plan.forward;
    for (row, &write) in forward.write_cache.iter().enumerate() {
        if !write {
            continue;
        }
        let length = u64::from(forward.query_lens[row]);
        let start = u64::from(forward.seq_lens[row]).saturating_sub(length);
        for table in tables.for_batch(forward.request_pool_indices[row], &plan.block_tables)? {
            spans.extend(table.spans(start, length)?);
        }
    }
    let transfers: HashMap<BufferId, &CacheTransfer> = plan
        .cache_inputs
        .iter()
        .chain(resident.iter().map(AsRef::as_ref))
        .map(|transfer| (transfer.source, transfer))
        .collect();
    for call in installs(plan) {
        let Some(transfer) = call.cache_input.and_then(|source| transfers.get(&source)) else {
            continue;
        };
        let slot = requests.get(call.request_key.request_id.0)?.slot() as u32;
        for table in tables.for_batch(slot, &plan.block_tables)? {
            let start = u64::from(transfer.base_extent)
                .max(u64::from(table.start_page) * u64::from(table.shape.page_tokens));
            let end = u64::from(transfer.exported_extent);
            if start < end {
                spans.extend(table.spans(start, end - start)?);
            }
        }
    }
    Ok(spans)
}

/// Fail when an overlapping access ended without retiring its storage.
pub(super) fn require_storage(state: &BatchInputs) -> Result<()> {
    for Hazard(retirement) in &state.dependencies {
        match retirement.outcome() {
            Some(Outcome::Success(())) => {}
            Some(Outcome::Failed(error)) => {
                return Err(Error::transport_source(
                    "a cache access the batch overwrites failed to retire",
                    error,
                ));
            }
            Some(Outcome::Cancelled) | None => {
                return Err(Error::Resource(
                    "a cache access the batch overwrites did not retire",
                ));
            }
        }
    }
    Ok(())
}
