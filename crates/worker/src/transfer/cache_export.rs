//! Native export of a slot's visible cache extent.
//!
//! An export to one `(request, destination)` carries only tokens after the
//! destination's accepted base, and a sliding-window group only its window.
//! Each group publishes key and value tensors `[tokens, layers, kv heads,
//! head dim]` over its whole transfer layer axis; this rank contributes one
//! locator set per unit position of a page (its `columns` consecutive
//! layers) at its layer and head offset. FP8 pools also publish one scale
//! row per touched page, `[pages, 2, layers, head groups]`.
//!
//! Dense exports alias the live pages, which the cache manager protects from
//! writes until the export retires. FP8 exports are frozen into staging
//! first, because a later append can grow a page's scale and re-encode it:
//! one allocation per cache group holds every value and scale view the
//! group freezes.

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder};
use uniserve_ipc::{BufferId, CacheGroupTransfer, CacheTransfer, TensorTransfer};

use super::exports::{self, Exports, Location};
use super::{Spans, Transports, copy_planner};
use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{CachePlanes, GroupTable, TransferBases, UnitAccesses};
use crate::stream::Issue;
use crate::{Error, Result};

/// The pages a token interval touches: `(absolute page, offset, length)`.
fn pages(start: u64, end: u64, page_tokens: u64) -> Vec<(u64, u32, u32)> {
    let mut pages = Vec::new();
    let mut position = start;
    while position < end {
        let (page, offset) = (position / page_tokens, position % page_tokens);
        let length = (end - position).min(page_tokens - offset);
        pages.push((page, offset as u32, length as u32));
        position += length;
    }
    pages
}

/// Byte alignment of each view frozen into a group's staging.
const FROZEN_ALIGNMENT: usize = 256;

fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(FROZEN_ALIGNMENT) * FROZEN_ALIGNMENT
}

/// One cache group's staging for frozen (FP8) views of an export: a single
/// allocation, sized for every view the group freezes and carved in
/// order. Exportable memory of the pool's device, which a CUDA VMM export
/// names where it lies and local peer devices may address, or host memory
/// for a CPU pool. The export holds it until its issuer performed the copies
/// into and out of it ([`CacheExport`]); an export that reads it in place
/// retains a carved view of it until that export retires.
struct Frozen {
    memory: Arc<Memory>,
    used: usize,
}

impl Frozen {
    fn allocate(planes: &CachePlanes, bytes: usize) -> Result<Self> {
        let memory = match planes.device() {
            Some(device) => Memory::exportable(device, bytes.max(1), &Memory::peers(device)?)?,
            None => Memory::host(bytes.max(1)),
        };
        Ok(Self { memory, used: 0 })
    }

    /// The offset of the next `bytes` of the staging.
    fn carve(&mut self, bytes: usize) -> Result<usize> {
        let offset = self.used;
        if offset + bytes > self.memory.bytes() {
            return Err(Error::Invariant(
                "frozen cache views exceed their staging".into(),
            ));
        }
        self.used = aligned(offset + bytes);
        Ok(offset)
    }

    /// A contiguous view of the next bytes of the staging.
    fn view(&mut self, dtype: DType, shape: &[i64]) -> Result<Spans> {
        let elements: i64 = shape.iter().product();
        let offset = self.carve(elements as usize * dtype.bytes())?;
        Spans::single(StorageView::contiguous(
            Arc::clone(&self.memory) as Arc<dyn Backing>,
            offset,
            dtype,
            shape,
        )?)
    }

    /// Copy `source` into the next view of the staging, in `stream`'s order.
    fn freeze(
        &mut self,
        planes: &CachePlanes,
        source: &Spans,
        stream: impl Issue,
    ) -> Result<Spans> {
        let shape: Vec<i64> = source.shape().iter().map(|&extent| extent as i64).collect();
        let target = self.view(source.dtype(), &shape)?;
        let mut plan = copy_planner(planes.device(), SourceOrder::Stream)?;
        target.plan_copy(source, &mut plan)?;
        // SAFETY: the staging is retained by the export's spans until it
        // retires; the pages are protected by the export reservation.
        // Without a stream both are host operands of this export and the
        // pool.
        unsafe { stream.copy_strided(&plan) }?;
        Ok(target)
    }
}

/// One group's export inputs.
struct GroupExport<'a> {
    index: usize,
    table: &'a GroupTable,
    start: u32,
    visible: u32,
}

/// A slot's export while its issuer's work is outstanding ([`export`]): the
/// transfer record without its locations, and the export of every view.
pub struct CacheExport {
    buffer: BufferId,
    transfer: CacheTransfer,
    /// The exports of each tensor of each group, in the transfer's order.
    exports: Vec<Vec<Vec<Exports>>>,
    /// The staging FP8 views were frozen into. The issuer's work copies
    /// into it and, for a mechanism that carries the bytes (shared memory,
    /// the rank channel), out of it; a job performs those copies later, and
    /// such an export does not retain the staging. The staging is released
    /// with this value, once that work was performed.
    staging: Vec<Arc<Memory>>,
}

impl CacheExport {
    /// The transfer record with its locations, and every location.
    ///
    /// The caller keeps this value until the issuer the export was given
    /// performed its work and calls this only then: at once after a stream,
    /// from the job's completion on after a recorder. On failure every
    /// location is revoked and the reservation released. The returned
    /// transfer becomes resident only when the batch commits it
    /// (`PrefixCache::validate_transfers` then `apply_transfers`).
    pub(crate) fn locate(
        self,
        accesses: &mut UnitAccesses,
    ) -> Result<(CacheTransfer, Vec<Location>)> {
        let Self {
            buffer,
            mut transfer,
            exports,
            staging,
        } = self;
        let mut locations: Vec<Location> = Vec::new();
        let mut result = Ok(());
        // The exports a failure leaves without locations.
        let mut unlocated = Vec::new();

        let tensors = transfer
            .groups
            .iter_mut()
            .flat_map(|group| group.tensors.iter_mut());
        for (tensor, exports) in tensors.zip(exports.into_iter().flatten()) {
            for export in exports {
                if result.is_err() {
                    unlocated.push(export);
                    continue;
                }
                match export.locate(|retirement| accesses.retain_export(buffer, retirement)) {
                    Ok(located) => {
                        tensor
                            .locations
                            .extend(located.iter().map(|(_, locator)| locator.clone()));
                        locations.extend(located);
                    }
                    Err(error) => result = Err(error),
                }
            }
        }

        if let Err(error) = result {
            unlocated.into_iter().for_each(Exports::revoke);
            if let Err(cleanup) = exports::release(&locations) {
                tracing::warn!(target: "uniserve_worker.transfer", "failed to revoke a failed KV export: {cleanup}");
            }
            accesses.release_exports(&[buffer]);
            return Err(error);
        }
        // The issuer's copies out of the staging were performed; exports
        // that read it in place retain their own views of it.
        drop(staging);
        Ok((transfer, locations))
    }

    /// Revoke an export that is not published and release its reservation.
    /// As for [`Self::locate`], only once the issuer performed its work:
    /// the copies into an export's storage ran before it is given back.
    pub(crate) fn revoke(self, accesses: &mut UnitAccesses) {
        self.exports
            .into_iter()
            .flatten()
            .flatten()
            .for_each(Exports::revoke);
        accesses.release_exports(&[self.buffer]);
        drop(self.staging);
    }
}

/// Export `visible` tokens of `slot` to `destination` under `buffer`, in
/// the order of `stream` (the stream that wrote the pages; none for a CPU
/// pool).
///
/// The exported intervals are reserved before any view is exported. On
/// failure every accepted export is revoked and the reservation released.
/// [`CacheExport::locate`] gives the transfer and its locations once the
/// issuer's work was performed.
#[expect(clippy::too_many_arguments, reason = "one export's complete inputs")]
pub(crate) fn export(
    accesses: &mut UnitAccesses,
    bases: &TransferBases,
    tables: &crate::storage::BlockTables,
    planes: &CachePlanes,
    slot: u32,
    visible: u32,
    destination: &str,
    buffer: BufferId,
    transports: &Transports,
    consumers: &[usize],
    stream: impl Issue,
) -> Result<CacheExport> {
    let (base, base_extent) = bases
        .destination_base(buffer.owner, destination)
        .map_or((None, 0), |(buffer, extent)| (Some(buffer), extent));
    if visible > tables.allocated_length(slot) {
        return Err(Error::Invalid(
            "cache export exceeds its scheduler block table".into(),
        ));
    }
    if visible < base_extent {
        return Err(Error::Invalid(
            "cache export destination is ahead of its source".into(),
        ));
    }

    let mut selected = Vec::new();
    let mut groups = Vec::new();
    let mut spans = Vec::new();
    for group in 0..tables.groups().len() {
        let table = tables.table(slot, group as u32)?;
        let start = table.shape.window.map_or(base_extent, |window| {
            base_extent.max(visible.saturating_sub(window))
        });
        spans.extend(table.spans(u64::from(start), u64::from(visible - start))?);
        if visible > base_extent {
            groups.push(CacheGroupTransfer {
                start,
                page_tokens: table.shape.page_tokens,
                tensors: Vec::new(),
            });
        }
        selected.push(table);
    }
    if visible > base_extent {
        accesses.reserve_export(buffer, &spans)?;
    }

    let mut transfer = CacheTransfer {
        source: buffer,
        destination: destination.to_owned(),
        base,
        base_extent,
        exported_extent: visible,
        compute_dtype: planes.plan().compute_dtype().name().to_owned(),
        groups,
    };

    // A transport can accept an export before a later group fails: keep
    // every export so a failure revokes all of them. The views are one
    // attempt: the copies a job recorded for those accepted before the
    // failure address storage their revocation gives back.
    let mut exports = Vec::new();
    let mut staging = Vec::new();
    let result = stream.attempt(|| -> Result<()> {
        for (index, (group, table)) in transfer.groups.iter_mut().zip(&selected).enumerate() {
            let export = GroupExport {
                index,
                table,
                start: group.start,
                visible,
            };
            exports.push(Vec::new());
            group.tensors = export_group(
                accesses,
                planes,
                &export,
                buffer,
                transports,
                consumers,
                stream,
                &mut exports[index],
                &mut staging,
            )?;
        }
        Ok(())
    });
    let export = CacheExport {
        buffer,
        transfer,
        exports,
        staging,
    };
    if let Err(error) = result {
        export.revoke(accesses);
        return Err(error);
    }
    Ok(export)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one group's complete export inputs"
)]
fn export_group(
    accesses: &mut UnitAccesses,
    planes: &CachePlanes,
    export: &GroupExport<'_>,
    buffer: BufferId,
    transports: &Transports,
    consumers: &[usize],
    stream: impl Issue,
    exports: &mut Vec<Vec<Exports>>,
    staging: &mut Vec<Arc<Memory>>,
) -> Result<Vec<TensorTransfer>> {
    let count = export.visible - export.start;
    if count == 0 {
        return Ok(Vec::new());
    }
    let plan = planes.plan();
    let group = &plan.groups[export.index];
    let table = export.table;
    let columns = plan.columns;
    let pages = pages(
        u64::from(export.start),
        u64::from(export.visible),
        u64::from(table.shape.page_tokens),
    );
    let mut retain = |retirement| accesses.retain_export(buffer, retirement);

    // The layers in unit position `row` of every page are one consecutive
    // run of `columns` layers, exported as one tensor per mechanism: the
    // locator count follows the group's units per page, never the model's
    // depth.
    let rows = table.shape.units_per_page;
    let mut sources = Vec::new();
    for field in 0..2 {
        for row in 0..rows {
            let units = table.row(row as usize);
            let views = pages
                .iter()
                .map(|&(page, offset, length)| {
                    let unit = units[(page - u64::from(table.start_page)) as usize];
                    planes.page(field, export.index, unit, offset, length)
                })
                .collect::<Result<Vec<_>>>()?;
            sources.push(Spans::new(views)?);
        }
    }
    // One scale per (unit, column, field) and touched page.
    let scale_bytes = pages.len() * columns as usize * 4;
    let mut frozen = if plan.quantized {
        let bytes = sources
            .iter()
            .map(|source| aligned(source.nbytes() as usize))
            .sum::<usize>()
            + 2 * rows as usize * aligned(scale_bytes);
        let frozen = Frozen::allocate(planes, bytes)?;
        staging.push(Arc::clone(&frozen.memory));
        Some(frozen)
    } else {
        None
    };

    let mut tensors = Vec::new();
    let mut sources = sources.into_iter();
    // Keys, then values, in the order `sources` holds them.
    for _ in 0..2 {
        exports.push(Vec::new());
        let field_exports = exports.len() - 1;
        for row in 0..rows {
            let source = sources
                .next()
                .ok_or_else(|| Error::Invariant("a cache row has no source".into()))?;
            let source = match &mut frozen {
                Some(frozen) => frozen.freeze(planes, &source, stream)?,
                None => source,
            };
            // The offset places this run inside the group's transfer at the
            // run's first layer and this rank's first KV head.
            let offset = [
                0,
                u64::from(group.axis.offset + row * columns),
                u64::from(group.kv_head_offset),
                0,
            ];
            exports[field_exports].push(exports::export(
                transports,
                &source,
                Some(&offset),
                consumers,
                false,
                stream,
                &mut retain,
            )?);
        }
        tensors.push(TensorTransfer {
            shape: vec![
                u64::from(count),
                u64::from(group.axis.total),
                u64::from(group.total_kv_heads),
                u64::from(group.head_dim),
            ],
            locations: Vec::new(),
        });
    }

    if let Some(frozen) = &mut frozen {
        // Published rows are [page, K/V, layers, head group], frozen like
        // the values. A rank's head group is its KV-head offset divided by
        // its local head count.
        exports.push(Vec::new());
        let scale_exports = exports.len() - 1;
        for field in 0..2 {
            for row in 0..rows {
                let units = table.row(row as usize);
                let source =
                    frozen.view(DType::F32, &[pages.len() as i64, 1, i64::from(columns), 1])?;
                let base = source.first().address();
                let mut batch = CopyBatch::new(SourceOrder::Stream);
                for (index, &(page, _, _)) in pages.iter().enumerate() {
                    let unit = units[(page - u64::from(table.start_page)) as usize];
                    for column in 0..columns {
                        batch.push(
                            base + ((index * columns as usize + column as usize) * 4) as u64,
                            planes.scale_address(field, column, unit)?,
                            4,
                        );
                    }
                }
                // SAFETY: the staging stays retained by the export; the
                // scales are protected by the export reservation. Without a
                // stream both are host operands of a CPU pool.
                unsafe { stream.copy(&batch) }?;
                let offset = [
                    0,
                    field as u64,
                    u64::from(group.axis.offset + row * columns),
                    u64::from(group.kv_head_offset / group.num_kv_heads),
                ];
                exports[scale_exports].push(exports::export(
                    transports,
                    &source,
                    Some(&offset),
                    consumers,
                    false,
                    stream,
                    &mut retain,
                )?);
            }
        }
        tensors.push(TensorTransfer {
            shape: vec![
                pages.len() as u64,
                2,
                u64::from(group.axis.total),
                u64::from(group.total_kv_heads / group.num_kv_heads),
            ],
            locations: Vec::new(),
        });
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_split_intervals_at_page_boundaries() {
        assert_eq!(pages(6, 6, 4), vec![]);
        assert_eq!(pages(6, 13, 4), vec![(1, 2, 2), (2, 0, 4), (3, 0, 1)]);
        assert_eq!(pages(8, 12, 4), vec![(2, 0, 4)]);
    }
}
