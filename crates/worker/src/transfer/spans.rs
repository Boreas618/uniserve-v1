//! Logical tensors stored as ordered first-axis spans, and their copies.
//!
//! An exported product or a read destination is one view or several views
//! that concatenate along axis 0 into one logical tensor, such as the pages
//! of a paged allocation. [`Spans`] selects regions of that logical tensor
//! and plans copies between two partitions without packing either.

use std::ops::Range;

use cuda::{CopyBatch, DeviceId, PitchedBatch, SourceOrder, Stream, StridedCopy, memcpy_async};

use crate::ffi::{DType, MemoryKind};
use crate::memory::allocation::StorageView;
use crate::{Error, Result};

/// Ordered first-axis spans of one logical tensor. Every span has the same
/// rank (at least one), element type, device and trailing extents, and a
/// positive leading extent.
#[derive(Clone, Debug)]
pub struct Spans {
    views: Vec<StorageView>,
}

impl Spans {
    pub(crate) fn new(views: Vec<StorageView>) -> Result<Self> {
        let Some(first) = views.first() else {
            return Err(Error::Invalid("a tensor needs at least one span".into()));
        };
        if first.shape().is_empty()
            || views.iter().any(|view| {
                view.shape().len() != first.shape().len()
                    || view.dtype() != first.dtype()
                    || view.device() != first.device()
                    || view.shape()[1..] != first.shape()[1..]
                    || view.shape().iter().any(|&extent| extent < 1)
            })
        {
            return Err(Error::Invalid(
                "tensor spans disagree on their representation".into(),
            ));
        }
        Ok(Self { views })
    }

    pub fn single(view: StorageView) -> Result<Self> {
        Self::new(vec![view])
    }

    pub fn views(&self) -> &[StorageView] {
        &self.views
    }

    pub fn first(&self) -> &StorageView {
        &self.views[0]
    }

    pub(crate) fn dtype(&self) -> DType {
        self.first().dtype()
    }

    pub(crate) fn device(&self) -> MemoryKind {
        self.first().device()
    }

    /// The logical shape: summed leading extents and the shared trailing axes.
    pub(crate) fn shape(&self) -> Vec<u64> {
        let mut shape: Vec<u64> = self
            .first()
            .shape()
            .iter()
            .map(|&extent| extent as u64)
            .collect();
        shape[0] = self.views.iter().map(|view| view.shape()[0] as u64).sum();
        shape
    }

    /// Bytes of the addressed elements, excluding strided gaps.
    pub fn nbytes(&self) -> u64 {
        self.views.iter().map(StorageView::nbytes).sum()
    }

    /// The pieces of each span that `region` (in logical coordinates)
    /// touches, each in its own span's coordinates. No data moves.
    pub(crate) fn region(&self, region: &[Range<u64>]) -> Result<Self> {
        let shape = self.shape();
        if region.len() != shape.len()
            || region
                .iter()
                .zip(&shape)
                .any(|(range, &extent)| range.start > range.end || range.end > extent)
        {
            return Err(Error::Invalid(
                "tensor region exceeds its destination spans".into(),
            ));
        }

        let mut pieces = Vec::new();
        let mut position = 0;
        for view in &self.views {
            let extent = view.shape()[0] as u64;
            let start = region[0].start.max(position);
            let end = region[0].end.min(position + extent);
            if start < end {
                let mut local = region.to_vec();
                local[0] = start - position..end - position;
                pieces.push(view.region(&local)?);
            }
            position += extent;
        }
        if pieces.is_empty() {
            return Err(Error::Invalid("a tensor region selects no elements".into()));
        }
        Self::new(pieces)
    }

    /// Validate a complete destination before it is split into reads: its
    /// representation must match, and no two destination elements may share
    /// storage, since separate shard reads write them concurrently.
    pub(crate) fn validate_destination(
        &self,
        shape: &[u64],
        dtype: DType,
        device: MemoryKind,
    ) -> Result<()> {
        if self.dtype() != dtype || self.device() != device || self.shape() != shape {
            return Err(Error::Invalid(
                "transfer destination disagrees with the exported representation".into(),
            ));
        }

        let mut intervals: Vec<_> = self.views.iter().flat_map(StorageView::intervals).collect();
        intervals.sort_unstable_by_key(|interval| interval.start);
        let elements: u64 = self.views.iter().map(StorageView::numel).sum();
        let covered: u64 = intervals
            .iter()
            .map(|interval| interval.end - interval.start)
            .sum();
        if covered != elements * dtype.bytes() as u64
            || intervals.windows(2).any(|pair| pair[0].end > pair[1].start)
        {
            return Err(Error::Invalid("transfer destination spans overlap".into()));
        }
        Ok(())
    }

    /// Plan copies of `source` into `self`, walking both partitions together
    /// and splitting at span boundaries on either side.
    pub(crate) fn plan_copy(&self, source: &Self, plan: &mut StridedCopy) -> Result<()> {
        if self.shape() != source.shape() || self.dtype() != source.dtype() {
            return Err(Error::Invalid(
                "transfer partitions have different logical representations".into(),
            ));
        }

        let (mut target_index, mut source_index) = (0, 0);
        let (mut target_start, mut source_start) = (0_i64, 0_i64);
        while target_index < self.views.len() && source_index < source.views.len() {
            let target = &self.views[target_index];
            let value = &source.views[source_index];
            let count = (target.shape()[0] - target_start).min(value.shape()[0] - source_start);
            let target_piece = target.narrow(0, target_start, count)?;
            let source_piece = value.narrow(0, source_start, count)?;
            plan.plan_within(
                &target_piece.operand().strided(),
                &source_piece.operand().strided(),
                [target.copy_span(), value.copy_span()],
            )?;

            target_start += count;
            source_start += count;
            if target_start == target.shape()[0] {
                target_index += 1;
                target_start = 0;
            }
            if source_start == value.shape()[0] {
                source_index += 1;
                source_start = 0;
            }
        }
        Ok(())
    }
}

/// A planner for copies between two operands of `device`'s address space,
/// or of host memory when `device` is `None`.
pub(crate) fn copy_planner(device: Option<DeviceId>, order: SourceOrder) -> Result<StridedCopy> {
    match device {
        Some(device) => Ok(StridedCopy::for_device(device, order)?),
        None => Ok(StridedCopy::new(order, usize::MAX)),
    }
}

/// Plan and submit the copy of `source` into `destination` on `stream`.
///
/// Operands on one device (or a device and host memory) go through the
/// batched copy APIs. Operands on two different devices are copied run by
/// run with `cuMemcpyAsync`: the batched APIs reject peer operands unless
/// peer access is enabled, while unified-address copies between devices
/// are always served (directly or staged by the driver).
///
/// # Safety
/// Both operands stay valid, and the destination is not accessed by other
/// work, until `stream` reaches the copies.
pub(crate) unsafe fn submit_copy(
    destination: &Spans,
    source: &Spans,
    stream: &Stream,
) -> Result<()> {
    match (destination.device(), source.device()) {
        (MemoryKind::Cuda(target), MemoryKind::Cuda(origin)) if target != origin => {
            // Without a pitch every planned copy is one contiguous run.
            let mut plan = StridedCopy::new(SourceOrder::Stream, 0);
            destination.plan_copy(source, &mut plan)?;
            for (target, origin, bytes) in plan.linear().iter() {
                // SAFETY: forwarded from the caller.
                unsafe { memcpy_async(target, origin, bytes, stream) }?;
            }
            Ok(())
        }
        _ => {
            let mut plan = copy_planner(Some(stream.device()), SourceOrder::Stream)?;
            destination.plan_copy(source, &mut plan)?;
            // SAFETY: forwarded from the caller.
            unsafe { plan.submit(stream) }.map_err(Error::from)
        }
    }
}

/// Execute planned copies between host operands on the calling thread.
///
/// # Safety
/// Every planned operand is host memory that stays valid, and that no other
/// thread or device accesses, for the duration of the call.
pub(crate) unsafe fn copy_on_host(plan: &StridedCopy) {
    // SAFETY: forwarded from the caller.
    unsafe {
        copy_linear(plan.linear());
        copy_pitched(plan.pitched());
    }
}

unsafe fn copy_linear(batch: &CopyBatch) {
    for (destination, source, bytes) in batch.iter() {
        // SAFETY: the caller guarantees valid, disjoint host operands.
        unsafe {
            std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, bytes);
        }
    }
}

unsafe fn copy_pitched(batch: &PitchedBatch) {
    for copy in batch.copies() {
        for layer in 0..copy.depth {
            for row in 0..copy.height {
                let destination = copy.destination
                    + ((layer * copy.destination_layer + row) * copy.destination_pitch) as u64;
                let source =
                    copy.source + ((layer * copy.source_layer + row) * copy.source_pitch) as u64;
                // SAFETY: the caller guarantees valid, disjoint host operands.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source as *const u8,
                        destination as *mut u8,
                        copy.width,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ffi::Backing;
    use crate::memory::allocation::Storage;

    fn write_floats(memory: &Arc<dyn Backing>, values: &[f32]) {
        // SAFETY: test memory is host memory owned by the caller.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                memory.address() as *mut f32,
                values.len(),
            );
        }
    }

    fn read_floats(memory: &Arc<dyn Backing>, count: usize) -> Vec<f32> {
        // SAFETY: test memory is host memory owned by the caller.
        unsafe { std::slice::from_raw_parts(memory.address() as *const f32, count) }.to_vec()
    }

    #[test]
    fn regions_split_across_spans_in_local_coordinates() -> Result<()> {
        let memory: Arc<dyn Backing> = Storage::host(12 * 4);
        let pages = Spans::new(vec![
            StorageView::contiguous(Arc::clone(&memory), 0, DType::F32, &[2, 3])?,
            StorageView::contiguous(Arc::clone(&memory), 24, DType::F32, &[2, 3])?,
        ])?;
        assert_eq!(pages.shape(), vec![4, 3]);

        let middle = pages.region(&[1..3, 1..3])?;
        assert_eq!(middle.views().len(), 2);
        assert_eq!(middle.shape(), vec![2, 2]);
        assert_eq!(middle.views()[1].address(), memory.address() + 28);
        assert!(pages.region(&[0..5, 0..3]).is_err());
        Ok(())
    }

    #[test]
    fn destinations_reject_aliasing_but_accept_interleaved_pages() -> Result<()> {
        let memory: Arc<dyn Backing> = Storage::host(16 * 4);
        // Columns of a [4, 4] grid viewed as rows: `[columns, 4]` with
        // strides `[1, 4]`, starting at the first column's element.
        let columns = |first: usize, count: i64| {
            StorageView::strided(
                Arc::clone(&memory),
                first * 4,
                DType::F32,
                &[count, 4],
                &[1, 4],
            )
        };
        let interleaved = Spans::new(vec![columns(0, 2)?])?;
        interleaved.validate_destination(&[2, 4], DType::F32, MemoryKind::Host)?;

        let even = columns(0, 1)?;
        let odd = columns(1, 1)?;
        Spans::new(vec![even.clone(), odd])?.validate_destination(
            &[2, 4],
            DType::F32,
            MemoryKind::Host,
        )?;
        let aliased = Spans::new(vec![even.clone(), even])?;
        assert!(
            aliased
                .validate_destination(&[2, 4], DType::F32, MemoryKind::Host)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn host_copies_walk_both_partitions() -> Result<()> {
        let source_memory: Arc<dyn Backing> = Storage::host(12 * 4);
        let target_memory: Arc<dyn Backing> = Storage::host(12 * 4);
        write_floats(
            &source_memory,
            &(0..12).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let source = Spans::single(StorageView::contiguous(
            Arc::clone(&source_memory),
            0,
            DType::F32,
            &[4, 3],
        )?)?;
        // Destination pages store rows in reverse page order.
        let target = Spans::new(vec![
            StorageView::contiguous(Arc::clone(&target_memory), 24, DType::F32, &[1, 3])?,
            StorageView::contiguous(Arc::clone(&target_memory), 0, DType::F32, &[2, 3])?,
            StorageView::contiguous(Arc::clone(&target_memory), 36, DType::F32, &[1, 3])?,
        ])?;

        let mut plan = copy_planner(None, SourceOrder::Stream)?;
        target.plan_copy(&source, &mut plan)?;
        // SAFETY: both operands are live host allocations owned by this test.
        unsafe { copy_on_host(&plan) };
        let values = read_floats(&target_memory, 12);
        assert_eq!(&values[6..9], &[0.0, 1.0, 2.0]);
        assert_eq!(&values[0..6], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&values[9..12], &[9.0, 10.0, 11.0]);
        Ok(())
    }

    /// A cache page of 4 tokens of 6 columns read across segmented planes
    /// whose tail segment of every column was given back: the copy plan keeps
    /// each copy inside a segment, and the copy moves every column (one copy
    /// across the columns is refused by the driver; `cuda`'s
    /// `a_pitched_copy_across_a_released_segment_reads_its_rows`).
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_page_across_the_columns_of_planes_given_back_in_part_copies() -> Result<()> {
        use cuda::{HandleType, PinnedBuffer, memset_d8};

        let device = DeviceId(0);
        let stream = Stream::new(device)?;
        let granularity = HandleType::Descriptor.granularity(device)?;
        let (tokens, columns, row) = (4_usize, 6_usize, 64_usize);
        // Every column is two segments, back to back.
        let planes = Storage::segmented(device, &vec![granularity; 2 * columns], &[])?;
        for segment in 0..2 * columns {
            drop(planes.create_segment(segment)?);
        }
        for column in 0..columns {
            planes.release_segment(2 * column + 1)?;
            // SAFETY: the column's first segment is mapped; nothing else
            // accesses it.
            unsafe {
                memset_d8(
                    planes.address() + (2 * column * granularity) as u64,
                    column as u8 + 1,
                    tokens * row,
                    &stream,
                )
            }?;
        }
        let page = StorageView::strided(
            Arc::clone(&planes) as Arc<dyn Backing>,
            0,
            DType::U8,
            &[tokens as i64, columns as i64, row as i64],
            &[row as i64, (2 * granularity) as i64, 1],
        )?;
        let staging = Storage::device(device, tokens * columns * row)?;
        let frozen = StorageView::contiguous(
            Arc::clone(&staging) as Arc<dyn Backing>,
            0,
            DType::U8,
            &[tokens as i64, columns as i64, row as i64],
        )?;
        let host = PinnedBuffer::new(device, tokens * columns * row)?;
        // SAFETY: the planes, the staging and the host buffer outlive the
        // synchronization below.
        unsafe {
            submit_copy(&Spans::single(frozen)?, &Spans::single(page)?, &stream)?;
            memcpy_async(host.address(), staging.address(), host.len(), &stream)?;
        }
        stream.synchronize()?;
        // SAFETY: the stream that wrote the buffer is synchronized.
        let bytes = unsafe { host.as_slice() };
        for (index, chunk) in bytes.chunks(row).enumerate() {
            let column = index % columns;
            assert!(
                chunk.iter().all(|&byte| byte == column as u8 + 1),
                "token {} column {column}",
                index / columns
            );
        }
        Ok(())
    }
}
