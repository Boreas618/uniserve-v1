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
//! first, because a later append can grow a page's scale and re-encode it;
//! the export keeps that staging until its issuer's work was performed.

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder};
use uniserve_ipc::{BufferId, CacheGroupTransfer, CacheTransfer, TensorTransfer};

use super::exports::{self, Exports, Location};
use super::{Spans, Transports, copy_planner};
use crate::ffi::{Backing, DType};
use crate::storage::memory::{Storage, StorageView};
use crate::storage::{CachePlanes, GroupTable, TransferBases, UnitAccesses};
use crate::stream::Issue;
use crate::{Error, Result};

/// The pages a token interval touches: `(absolute page, offset, length)`.
fn pages(start: u64, end: u64, block_size: u64) -> Vec<(u64, u32, u32)> {
    let mut pages = Vec::new();
    let mut position = start;
    while position < end {
        let (page, offset) = (position / block_size, position % block_size);
        let length = (end - position).min(block_size - offset);
        pages.push((page, offset as u32, length as u32));
        position += length;
    }
    pages
}

/// Staging for frozen exports: exportable memory of the pool's device, which
/// a CUDA VMM export names where it lies and local peer devices may address,
/// or host memory for a CPU pool.
fn staging(planes: &CachePlanes, bytes: usize) -> Result<Arc<Storage>> {
    match planes.device() {
        Some(device) => Storage::export_staging(device, bytes.max(1), &Storage::peers(device)?),
        None => Ok(Storage::host(bytes.max(1))),
    }
}

/// Copy `source` into a new contiguous staging tensor, in `stream`'s order.
/// The staging joins `frozen`, which the export keeps until located or
/// revoked.
fn freeze(
    planes: &CachePlanes,
    source: &Spans,
    stream: impl Issue,
    frozen: &mut Vec<Arc<Storage>>,
) -> Result<Spans> {
    let shape: Vec<i64> = source.shape().iter().map(|&extent| extent as i64).collect();
    let memory = staging(planes, source.nbytes() as usize)?;
    frozen.push(Arc::clone(&memory));
    let target = Spans::single(StorageView::contiguous(
        memory as Arc<dyn Backing>,
        0,
        source.dtype(),
        &shape,
    )?)?;
    let mut plan = copy_planner(planes.device(), SourceOrder::Stream)?;
    target.plan_copy(source, &mut plan)?;
    // SAFETY: the export retains the staging until its issuer performed
    // this copy; the pages are protected by the export reservation. Without
    // a stream both are host operands of this export and the pool.
    unsafe { stream.copy_strided(&plan) }?;
    Ok(target)
}

/// One group's export inputs.
struct GroupExport<'a> {
    index: usize,
    table: &'a GroupTable,
    start: u32,
    visible: u32,
}

/// A slot's export while its issuer's work is outstanding (`export`): the
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
    staging: Vec<Arc<Storage>>,
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
                tracing::warn!("failed to revoke a failed KV export: {cleanup}");
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
                block_size: table.shape.block_size,
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
    frozen: &mut Vec<Arc<Storage>>,
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
        u64::from(table.shape.block_size),
    );
    let mut retain = |retirement| accesses.retain_export(buffer, retirement);

    let mut tensors = Vec::new();
    for field in 0..2 {
        exports.push(Vec::new());
        let field_exports = exports.len() - 1;
        // The layers in unit position `row` of every page are one
        // consecutive run of `columns` layers, exported as one tensor per
        // mechanism: the locator count follows the group's units per page,
        // never the model's depth.
        for row in 0..table.shape.units_per_page {
            let units = table.row(row as usize);
            let views = pages
                .iter()
                .map(|&(page, offset, length)| {
                    let unit = units[(page - u64::from(table.start_page)) as usize];
                    planes.page(field, export.index, unit, offset, length)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut source = Spans::new(views)?;
            if plan.quantized {
                source = freeze(planes, &source, stream, frozen)?;
            }
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

    if plan.quantized {
        // One scale per (unit, column, field): published rows are
        // [page, K/V, layers, head group], frozen like the values. A rank's
        // head group is its KV-head offset divided by its local head count.
        exports.push(Vec::new());
        let scale_exports = exports.len() - 1;
        for field in 0..2 {
            for row in 0..table.shape.units_per_page {
                let units = table.row(row as usize);
                let memory = staging(planes, pages.len() * columns as usize * 4)?;
                frozen.push(Arc::clone(&memory));
                let mut batch = CopyBatch::new(SourceOrder::Stream);
                for (index, &(page, _, _)) in pages.iter().enumerate() {
                    let unit = units[(page - u64::from(table.start_page)) as usize];
                    for column in 0..columns {
                        batch.push(
                            memory.address()
                                + ((index * columns as usize + column as usize) * 4) as u64,
                            planes.scale_address(field, column, unit)?,
                            4,
                        );
                    }
                }
                // SAFETY: the staging stays retained by the export; the
                // scales are protected by the export reservation. Without a
                // stream both are host operands of a CPU pool.
                unsafe { stream.copy(&batch) }?;
                let source = Spans::single(StorageView::contiguous(
                    memory as Arc<dyn Backing>,
                    0,
                    DType::F32,
                    &[pages.len() as i64, 1, i64::from(columns), 1],
                )?)?;
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
    use std::collections::HashSet;

    use uniserve_core::{CallId, PrefixCacheDtype, RequestId, UnitId};
    use uniserve_ipc::{BlockTable, RequestKey, TransferTransport, WorkerEndpoint};

    use super::*;
    use crate::EventPool;
    use crate::lane::Job;
    use crate::storage::{BlockTables, CacheGroupPlan, CachePlan, GroupAxis};
    use crate::transfer::{TransferCapacity, TransportKind, TransportOptions};

    #[test]
    fn pages_split_intervals_at_page_boundaries() {
        assert_eq!(pages(6, 6, 4), vec![]);
        assert_eq!(pages(6, 13, 4), vec![(1, 2, 2), (2, 0, 4), (3, 0, 1)]);
        assert_eq!(pages(8, 12, 4), vec![(2, 0, 4)]);
    }

    /// Set in the child process that runs the recorded FP8 export.
    const FROZEN_CHILD: &str = "UNISERVE_TEST_FROZEN_EXPORT_CHILD";

    /// The byte of FP8 element `[token, column, head, dim]` of `field`.
    fn value(field: usize, index: [usize; 4]) -> u8 {
        let [token, column, head, dim] = index;
        ((field * 101 + token * 31 + column * 17 + head * 7 + dim) % 251) as u8
    }

    /// The scale of `field` in `column`.
    fn scale(field: usize, column: usize) -> f32 {
        1.5 + (field * 2 + column) as f32
    }

    /// A host FP8 cache export recorded into a job carries the values and
    /// scales its job froze, though no export mechanism of the rank retains
    /// the frozen views: the rank channel copies them out, in the job.
    ///
    /// The export runs in a child process in which every allocation of
    /// 64 KiB or more is its own mapping, returned to the system when it is
    /// freed, so a job that copied into staging released before it ran
    /// faults instead of writing memory that is no longer the export's.
    #[test]
    fn a_recorded_fp8_export_carries_what_its_job_froze() -> Result<()> {
        if std::env::var_os(FROZEN_CHILD).is_none() {
            let executable = std::env::current_exe()
                .map_err(|error| Error::Invalid(format!("no test executable: {error}")))?;
            let status = std::process::Command::new(executable)
                .args([
                    "--exact",
                    "transfer::cache_export::tests::a_recorded_fp8_export_carries_what_its_job_froze",
                    "--nocapture",
                ])
                .env(FROZEN_CHILD, "1")
                .env("MALLOC_MMAP_THRESHOLD_", "65536")
                .status()
                .map_err(|error| Error::Invalid(format!("cannot run the export: {error}")))?;
            assert!(status.success(), "the recorded FP8 export failed: {status}");
            return Ok(());
        }

        // One group of two layers in two columns: a unit holds an 8-token
        // page of 2 KV heads of 4096 FP8 elements per layer, so each frozen
        // field is 128 KiB.
        let (tokens, columns, heads, dims) = (8_usize, 2_usize, 2_usize, 4096_usize);
        let plane_bytes = (tokens * heads * dims) as u64;
        let plan = Arc::new(CachePlan {
            columns: columns as u32,
            plane_bytes,
            quantized: true,
            unit_bytes: 2 * columns as u64 * (plane_bytes + 1 + 4),
            dtype: PrefixCacheDtype::Float8E4m3Fn,
            groups: vec![CacheGroupPlan {
                window: None,
                block_size: tokens as u32,
                units_per_page: 1,
                layers: vec!["a".into(), "b".into()],
                layer_ids: vec![0, 1],
                axis: GroupAxis {
                    offset: 0,
                    total: 2,
                },
                num_kv_heads: heads as u32,
                total_kv_heads: heads as u32,
                kv_head_offset: 0,
                head_dim: dims as u32,
                dtype: DType::F8E4M3,
                compute_dtype: DType::BF16,
            }],
            handle: crate::ffi::Handle::none(),
        });
        let planes = CachePlanes::allocate(Arc::clone(&plan), None, 2, None)?;
        let mut tables = BlockTables::new(plan.group_shapes()?, 1, 1, 2)?;
        let update = tables.prepare(&[BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 0,
            unit_ids: vec![UnitId(1)],
            allocated_tokens: tokens as u32,
        }])?;
        tables.commit(update);

        // Unit 1's page and scales, written through the pool's own views.
        for field in 0..2 {
            let page = planes.page(field, 0, 1, 0, tokens as u32)?;
            let strides = page.strides().to_vec();
            for token in 0..tokens {
                for column in 0..columns {
                    for head in 0..heads {
                        for dim in 0..dims {
                            let offset = [token, column, head, dim]
                                .iter()
                                .zip(&strides)
                                .map(|(&index, &stride)| index as i64 * stride)
                                .sum::<i64>();
                            // SAFETY: the element lies in the pool's host
                            // values, which nothing else accesses here.
                            unsafe {
                                *((page.address() as i64 + offset) as *mut u8) =
                                    value(field, [token, column, head, dim]);
                            }
                        }
                    }
                }
            }
            for column in 0..columns {
                let address = planes.scale_address(field, column as u32, 1)?;
                // SAFETY: one f32 of the pool's host scales.
                unsafe { *(address as *mut f32) = scale(field, column) };
            }
        }

        // The rank reaches consumer 1 only over the rank channel.
        let transports = Transports::new(
            &[TransportKind::Channel],
            Arc::new(TransferCapacity::new(1 << 24, 8)?),
            EventPool::new(),
            &TransportOptions {
                source: WorkerEndpoint {
                    host: "host".into(),
                    address_space: "space".into(),
                    incarnation: "incarnation".into(),
                },
                acknowledgment_slot: 0,
                host_slots: HashSet::new(),
                cross_host_consumers: true,
                vmm_pool_bytes: 1 << 20,
            },
        )?;
        let buffer = BufferId {
            owner: RequestKey::new(1, RequestId(7), 1),
            producer_call_id: CallId::new(2, 0),
            output_index: 0,
            generation: 2,
        };
        let mut accesses = UnitAccesses::default();

        // The freeze and the channel's copies are recorded; the job
        // performs them after the export returned.
        let job = Job::new();
        let recorded = export(
            &mut accesses,
            &TransferBases::default(),
            &tables,
            &planes,
            1,
            tokens as u32,
            "target",
            buffer,
            &transports,
            &[1],
            job.on(None),
        )?;
        job.perform().check().map_err(|failure| failure.error)?;
        let (transfer, _) = recorded.locate(&mut accesses)?;

        let payloads = |tensor: usize| -> Vec<Vec<u8>> {
            transfer.groups[0].tensors[tensor]
                .locations
                .iter()
                .filter_map(|locator| match &locator.transport {
                    TransferTransport::Channel { payload, .. } => Some(payload.clone()),
                    _ => None,
                })
                .collect()
        };
        for field in 0..2 {
            let mut expected = Vec::with_capacity(tokens * columns * heads * dims);
            for token in 0..tokens {
                for column in 0..columns {
                    for head in 0..heads {
                        for dim in 0..dims {
                            expected.push(value(field, [token, column, head, dim]));
                        }
                    }
                }
            }
            assert_eq!(payloads(field), [expected], "field {field} values");
        }
        // One scale row [page, K/V, layers, head group] per field.
        let scales: Vec<Vec<u8>> = (0..2)
            .map(|field| {
                (0..columns)
                    .flat_map(|column| scale(field, column).to_ne_bytes())
                    .collect()
            })
            .collect();
        assert_eq!(payloads(2), scales);
        Ok(())
    }
}
