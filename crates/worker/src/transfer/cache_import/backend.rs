//! Native cache import copies: direct page reads and converted reads.
//!
//! An import reads another rank's cache export into this rank's pages on an
//! import thread. When the source stores the destination's representation
//! (same dtype; for FP8 also page-aligned with the same page size, compute
//! dtype and one source scale group per destination head range), transports
//! write the live pages directly, one layer at a time to bound the read
//! fan-out. Otherwise each destination page span is read into the
//! workspace's scratch and converted by the bound numerical function.
//!
//! Ordering is stream-based: resets and conversions run on the workspace's
//! stream, and every transport read records its handoff on that stream, so a
//! transport never writes a page or the scratch before earlier work on the
//! workspace stream that reads or resets it.

use std::ops::{Deref, Range};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::Duration;

use tvm_ffi::{Any, AnyView};
use uniserve_cuda::Stream;
use uniserve_ipc::{CacheTransfer, Locator, Wake};

use super::{CacheImport, CacheImporter};
use crate::ffi::{Backing, DType, Function, Handle, Interpreter};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{CachePlanes, GroupTable, UnitColumn};
use crate::transfer::fetch;
use crate::transfer::read::{Read, ReadRef};
use crate::transfer::{Notify, Retirement, Spans, Transports, notify};
use crate::{Error, Result};

/// One import thread's stream, scratch and bound numerical operations.
pub(crate) struct ImportWorkspace {
    stream: Option<Arc<Stream>>,
    scratch: Arc<Memory>,
    elements: u64,
    units: UnitColumn,
    /// The library's `PageConversion` over the scratch and the pool.
    conversion: Handle,
    convert: Function,
}

/// The library class converting imported pages.
const PAGE_CONVERSION: &str = "uniserve.prefix_cache.conversion.PageConversion";

impl ImportWorkspace {
    /// Scratch bytes of one workspace: two raw source pages of up to 8-byte
    /// elements, one FP32 conversion page and the FP32 source scales:
    /// `20 E + 8 S` for `E` page elements and `S` page scale entries.
    pub(crate) fn scratch_bytes(elements: u64, scales: u64) -> u64 {
        20 * elements + 8 * scales
    }

    /// Allocate a workspace for `planes` on its own ordinary stream and
    /// construct the unit resets and the page conversion on it.
    pub(crate) fn bind(planes: &CachePlanes) -> Result<Self> {
        let (elements, scales) = planes.plan().page_extents();
        let bytes = Self::scratch_bytes(elements, scales) as usize;
        let (stream, scratch) = match planes.device() {
            Some(device) => (
                Some(Arc::new(Stream::new(device)?)),
                Memory::device(device, bytes.max(1))?,
            ),
            None => (None, Memory::host(bytes.max(1))),
        };
        let units = UnitColumn::bind(planes, stream.as_deref())?;

        let scratch_view = StorageView::contiguous(
            Arc::clone(&scratch) as Arc<dyn Backing>,
            0,
            DType::U8,
            &[bytes as i64],
        )?;
        let mut arguments = vec![
            planes.plan().handle.to_any(),
            Any::from(i64::from(planes.num_units())),
            scratch_view.handle()?.to_any(),
            Any::from(elements as i64),
            Any::from(scales as i64),
            Any::from(stream.as_ref().map_or(0, |stream| stream.raw() as i64)),
        ];
        arguments.extend(planes.field_arguments()?);
        let conversion =
            Handle::new(Interpreter::import(PAGE_CONVERSION)?.call_values(&arguments)?)?;
        Ok(Self {
            stream,
            scratch,
            elements,
            units,
            conversion,
            convert: Interpreter::import(&format!("{PAGE_CONVERSION}.convert"))?,
        })
    }

    fn stream(&self) -> Option<&Stream> {
        self.stream.as_deref()
    }

    /// A contiguous scratch view at `offset` bytes.
    fn view(&self, offset: u64, dtype: DType, shape: &[i64]) -> Result<Spans> {
        Spans::single(StorageView::contiguous(
            Arc::clone(&self.scratch) as Arc<dyn Backing>,
            offset as usize,
            dtype,
            shape,
        )?)
    }
}

/// A reserved import as the importer retains it.
#[derive(Clone)]
pub(crate) struct ImportEntry {
    write: Arc<CacheImport>,
    retirement: Arc<Retirement>,
}

impl ImportEntry {
    pub(crate) fn new(write: Arc<CacheImport>) -> Self {
        Self {
            write,
            retirement: Arc::new(Retirement::default()),
        }
    }

    pub(crate) fn write(&self) -> &Arc<CacheImport> {
        &self.write
    }

    pub(crate) fn retirement(&self) -> &Arc<Retirement> {
        &self.retirement
    }
}

impl Deref for ImportEntry {
    type Target = CacheImport;

    fn deref(&self) -> &Self::Target {
        &self.write
    }
}

/// Notifies retired imports' observers and the service wake.
pub(crate) struct Retired {
    pub(crate) importer: Weak<CacheImporter>,
    pub(crate) wake: Mutex<Option<Wake>>,
}

impl Retired {
    pub(crate) fn notify(&self, imports: Vec<ImportEntry>) -> Result<()> {
        let changed = !imports.is_empty();
        for import in imports {
            if !import.retirement.done() {
                notify(import.retirement.complete(Ok(()))?);
            }
        }
        if changed
            && let Some(wake) = self
                .wake
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
        {
            wake.wake();
        }
        Ok(())
    }

    pub(crate) fn reap(&self) -> Result<()> {
        match self.importer.upgrade() {
            Some(importer) => self.notify(importer.reap()),
            None => Ok(()),
        }
    }
}

/// One import's copy work, the action of a host task of the cache's import
/// lane ([`HostAction`](crate::host::HostAction) in `copy.rs`). Workspace
/// admission, cancellation and physical retirement belong to the importer.
pub(crate) struct ImportCopy {
    pub(crate) write: Arc<CacheImport>,
    pub(crate) export: Arc<CacheTransfer>,
    pub(crate) tables: Vec<Arc<GroupTable>>,
    pub(crate) initialized: Vec<u32>,
    pub(crate) planes: Arc<CachePlanes>,
    pub(crate) transports: Arc<Transports>,
    pub(crate) importer: Arc<CacheImporter>,
    pub(crate) retired: Arc<Retired>,
}

impl ImportCopy {
    /// Reset newly allocated destination units on the workspace's stream.
    pub(super) fn reset(&self, workspace: &ImportWorkspace) -> Result<()> {
        workspace
            .units
            .zero(&self.planes, &self.initialized, workspace.stream())
    }

    /// Copy cache values and retain every physical transport read on the
    /// import.
    pub(super) fn copy(&self, workspace: &ImportWorkspace) -> Result<()> {
        let plan = self.planes.plan();
        for (index, group) in self.export.groups.iter().enumerate() {
            let Some(first) = group.tensors.first() else {
                continue;
            };
            let target = &plan.groups[index];
            let layer = u64::from(target.axis.offset);
            let head = u64::from(target.kv_head_offset);
            let axes = [
                layer..layer + target.layers.len() as u64,
                head..head + u64::from(target.num_kv_heads),
                0..u64::from(target.head_dim),
            ];
            let source_dtype = DType::parse(
                &first
                    .locations
                    .first()
                    .ok_or_else(|| Error::Invalid("cache transfer tensor has no locations".into()))?
                    .dtype,
            )?;

            let mut direct = source_dtype == target.dtype;
            let scale_heads = if source_dtype == DType::F8E4M3 {
                let groups = group
                    .tensors
                    .get(2)
                    .and_then(|scales| scales.shape.get(3))
                    .copied()
                    .filter(|&groups| groups > 0)
                    .ok_or_else(|| Error::Invalid("FP8 cache transfer has no scales".into()))?;
                first.shape[2] / groups
            } else {
                0
            };
            if direct && plan.quantized {
                // Partial pages keep their destination scale; an intact
                // source scale can only serve one destination head group.
                let page_tokens = self.tables[index].shape.page_tokens;
                direct = group.page_tokens == page_tokens
                    && group.start.is_multiple_of(page_tokens)
                    && self.export.compute_dtype == target.compute_dtype.name()
                    && head / scale_heads
                        == (head + u64::from(target.num_kv_heads) - 1) / scale_heads;
            }

            let mut locators: Vec<Vec<Option<Arc<Locator>>>> = group
                .tensors
                .iter()
                .map(|tensor| vec![None; tensor.locations.len()])
                .collect();
            let context = GroupCopy {
                backend: self,
                workspace,
                index,
                axes,
                scale_heads,
                source_dtype,
            };
            if direct {
                context.direct(&mut locators)?;
            } else {
                context.converted(&mut locators)?;
            }
        }
        Ok(())
    }

    /// Wait for the workspace stream's accesses. An error leaves their
    /// completion unknown.
    pub(super) fn drain(&self, workspace: &ImportWorkspace) -> Result<()> {
        if let Some(stream) = workspace.stream() {
            stream.synchronize()?;
        }
        Ok(())
    }
}

/// The copy of one transferred group.
struct GroupCopy<'a> {
    backend: &'a ImportCopy,
    workspace: &'a ImportWorkspace,
    index: usize,
    /// This rank's global layers, KV heads and head width in the transfer.
    axes: [Range<u64>; 3],
    /// Source KV heads per scale group (0 for unquantized sources).
    scale_heads: u64,
    source_dtype: DType,
}

impl GroupCopy<'_> {
    fn group(&self) -> &uniserve_ipc::CacheGroupTransfer {
        &self.backend.export.groups[self.index]
    }

    /// Read `region` of transfer tensor `tensor` into `destination` and
    /// retain each read on the import.
    ///
    /// When the rank's read credits are taken, the import thread waits for
    /// returns and retries: earlier reads retire on their own transport
    /// threads, and nothing is submitted before the whole fan-out fits.
    fn fetch(
        &self,
        tensor: usize,
        destination: &Spans,
        region: &[Range<u64>],
        locators: &mut [Vec<Option<Arc<Locator>>>],
    ) -> Result<Vec<Arc<Read>>> {
        let backend = self.backend;
        let reads = fetch::plan(
            &self.group().tensors[tensor],
            destination,
            &backend.transports,
            region,
            &mut locators[tensor],
        )?;
        loop {
            backend.importer.require_active(&backend.write)?;
            match self.submit(&reads) {
                Err(Error::ReadBackpressure { returns }) => self.wait_for_credits(returns)?,
                result => return result,
            }
        }
    }

    fn submit(&self, reads: &[fetch::PlannedRead]) -> Result<Vec<Arc<Read>>> {
        let backend = self.backend;
        let retired = Arc::downgrade(&backend.retired);
        // Issued on the workspace's stream at once, so the reads start
        // here.
        let reads = fetch::submit(reads, self.workspace.stream(), |read| {
            let cancelled = backend.write.retain(Arc::new(ReadRef::new(read)));
            // Retired reads may let a finished import retire.
            let retired = Weak::clone(&retired);
            read.add_retirement_callback(Arc::new(move || {
                if let Some(retired) = retired.upgrade()
                    && let Err(error) = retired.reap()
                {
                    tracing::warn!(target: "uniserve_worker.transfer", "failed to retire KV imports: {error}");
                }
            }));
            if cancelled {
                read.cancel()?;
            }
            Ok(())
        })?;
        fetch::start(&reads)?;
        Ok(reads)
    }

    /// Block until read credits returned after the refused submission
    /// (`returns` is the return count it observed), or the import is
    /// cancelled.
    fn wait_for_credits(&self, returns: u64) -> Result<()> {
        let signal = Arc::new((Mutex::new(false), Condvar::new()));
        let notified = Arc::clone(&signal);
        let callback: Notify = Arc::new(move || {
            *notified.0.lock().unwrap_or_else(PoisonError::into_inner) = true;
            notified.1.notify_all();
        });
        let capacity = self.backend.transports.capacity();
        if let Some(callback) = capacity.notify_reads_returned(callback, returns) {
            callback();
        }
        let mut returned = signal.0.lock().unwrap_or_else(PoisonError::into_inner);
        while !*returned {
            self.backend.importer.require_active(&self.backend.write)?;
            returned = signal
                .1
                .wait_timeout(returned, Duration::from_millis(10))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        Ok(())
    }

    /// Wait for each read's value on the host, then order the workspace
    /// stream after its copy.
    fn consume(&self, reads: &[Arc<Read>]) -> Result<()> {
        for read in reads {
            read.wait_ready();
            read.result(self.workspace.stream())?;
        }
        Ok(())
    }

    fn close(reads: Vec<Arc<Read>>) -> Result<()> {
        reads.iter().try_for_each(Read::close)
    }

    fn direct(&self, locators: &mut [Vec<Option<Arc<Locator>>>]) -> Result<()> {
        let backend = self.backend;
        let planes = &backend.planes;
        let columns = planes.plan().columns;
        let group = self.group();
        let table = &backend.tables[self.index];
        let carried = u64::from(backend.export.exported_extent - group.start);
        let spans = table.spans(u64::from(group.start), carried)?;
        let rows = table.shape.units_per_page as usize;
        let pages = (spans.len() / rows) as u64;

        // Bound physical fan-out by one layer, regardless of model depth.
        for (column, layer) in self.axes[0].clone().enumerate() {
            let selected: Vec<_> = spans
                .iter()
                .skip(column / columns as usize)
                .step_by(rows)
                .copied()
                .collect();
            let plane_column = column as u32 % columns;
            let mut reads = Vec::new();
            for field in 0..2 {
                let destination = Spans::new(
                    selected
                        .iter()
                        .map(|&(unit, offset, count)| {
                            planes.layer_tokens(
                                field,
                                self.index,
                                plane_column,
                                unit,
                                offset,
                                count,
                            )
                        })
                        .collect::<Result<_>>()?,
                )?;
                let region = [
                    0..carried,
                    layer..layer + 1,
                    self.axes[1].clone(),
                    self.axes[2].clone(),
                ];
                reads.extend(self.fetch(field, &destination, &region, locators)?);

                if let Some(head) = self.axes[1].start.checked_div(self.scale_heads) {
                    let destination = Spans::new(
                        selected
                            .iter()
                            .map(|&(unit, _, _)| planes.scale(field, plane_column, unit))
                            .collect::<Result<_>>()?,
                    )?;
                    let field = field as u64;
                    let region = [0..pages, field..field + 1, layer..layer + 1, head..head + 1];
                    reads.extend(self.fetch(2, &destination, &region, locators)?);
                }
            }
            self.consume(&reads)?;
            Self::close(reads)?;
        }

        backend.importer.require_active(&backend.write)?;
        let units: Vec<u32> = spans.iter().map(|&(unit, _, _)| unit).collect();
        self.workspace
            .units
            .mark_initialized(planes, &units, self.workspace.stream())
    }

    fn converted(&self, locators: &mut [Vec<Option<Arc<Locator>>>]) -> Result<()> {
        let backend = self.backend;
        let group = self.group();
        let table = &backend.tables[self.index];
        let workspace = self.workspace;
        let elements = workspace.elements;
        let page_tokens = u64::from(table.shape.page_tokens);
        let source_page_tokens = u64::from(group.page_tokens);
        let start = u64::from(group.start);
        let end = u64::from(backend.export.exported_extent);
        let extent = |axis: &Range<u64>| (axis.end - axis.start) as i64;
        let (layers, heads, width) = (
            extent(&self.axes[0]),
            extent(&self.axes[1]),
            extent(&self.axes[2]),
        );
        let rows = table.shape.units_per_page as usize;

        let mut position = start;
        while position < end {
            let page = position / page_tokens;
            let offset = position % page_tokens;
            let count = (end - position).min(page_tokens - offset);
            let source_offset = position % source_page_tokens;
            let shape = [count as i64, layers, heads, width];
            let region = [
                position - start..position - start + count,
                self.axes[0].clone(),
                self.axes[1].clone(),
                self.axes[2].clone(),
            ];

            // Raw source pages: field f at byte 8 E f of the scratch.
            let mut reads = Vec::new();
            for field in 0..2 {
                let destination =
                    workspace.view(8 * elements * field as u64, self.source_dtype, &shape)?;
                reads.extend(self.fetch(field, &destination, &region, locators)?);
            }
            let mut pages = 0;
            if self.scale_heads > 0 {
                let first = position / source_page_tokens - start / source_page_tokens;
                pages = (source_offset + count).div_ceil(source_page_tokens);
                let groups = self.axes[1].end.div_ceil(self.scale_heads)
                    - self.axes[1].start / self.scale_heads;
                let region = [
                    first..first + pages,
                    0..2,
                    self.axes[0].clone(),
                    self.axes[1].start / self.scale_heads
                        ..self.axes[1].end.div_ceil(self.scale_heads),
                ];
                let destination = workspace.view(
                    20 * elements,
                    DType::F32,
                    &[pages as i64, 2, layers, groups as i64],
                )?;
                reads.extend(self.fetch(2, &destination, &region, locators)?);
            }
            self.consume(&reads)?;

            let first_unit = (page - u64::from(table.start_page)) as usize * rows;
            let units: tvm_ffi::Array<i64> = table.units[first_unit..first_unit + rows]
                .iter()
                .map(|unit| i64::from(unit.0))
                .collect();
            let arguments = (
                self.index as i64,
                tvm_ffi::String::from(self.source_dtype.name()),
                tvm_ffi::String::from(backend.export.compute_dtype.as_str()),
                count as i64,
                offset as i64,
                source_offset as i64,
                source_page_tokens as i64,
                self.scale_heads as i64,
                pages as i64,
            );
            // The conversion enqueues on the workspace stream. The next
            // span's reads record their handoff after it, so no transport
            // overwrites the scratch before the conversion consumed it.
            workspace.convert.invoke(&[
                workspace.conversion.view(),
                AnyView::from(&arguments.0),
                AnyView::from(&arguments.1),
                AnyView::from(&arguments.2),
                AnyView::from(&arguments.3),
                AnyView::from(&arguments.4),
                AnyView::from(&arguments.5),
                AnyView::from(&arguments.6),
                AnyView::from(&arguments.7),
                AnyView::from(&arguments.8),
                AnyView::from(&units),
            ])?;
            Self::close(reads)?;
            position += count;
        }
        Ok(())
    }
}
