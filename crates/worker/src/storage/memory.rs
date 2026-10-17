//! Rust-owned allocations and the typed views Python receives as torch
//! tensors.
//!
//! Runtime storage (cache planes, graph inputs and outputs, workspaces, staging)
//! is allocated here. A [`Memory`] value owns one allocation; a
//! [`StorageView`] is a typed, strided window into it that retains the
//! allocation. Python receives a view as one torch tensor, made once from
//! the view's DLPack export and passed by every call that takes the view
//! ([`StorageView::handle`]). Dropping the last Rust and Python reference
//! releases the memory through `uniserve-cuda`'s reclaimer, which never
//! waits for device work: owners must retire all device accesses to an
//! allocation before dropping it.
//!
//! Exportable device allocations (`cuMemCreate` with a shareable handle) are
//! recorded in a process-wide registry by address, so a transport can export
//! any view of them in place ([`Memory::exportable_containing`]).

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use tvm_ffi::{Any, AnyView};
use uniserve_cuda::{
    CopyBatch, DeviceAllocation, DeviceId, PeerMapping, PinnedBuffer, SourceOrder, Strided,
};

use crate::ffi::{
    Backing, DType, Function, Handle, Interpreter, MemoryKind, Tensor, contiguous_strides, export,
};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};

/// One allocation owned by the worker.
pub struct Memory {
    kind: Kind,
}

enum Kind {
    Device(DeviceAllocation),
    Pinned(PinnedBuffer),
    /// Pageable host memory for CPU lanes. Boxed so its address is stable.
    Host(Box<[u8]>),
    /// Another process's (or peer device's) exported allocation mapped into
    /// this process.
    Mapped(PeerMapping),
}

/// Exportable allocations by base address. Entries are weak: the registry
/// never extends an allocation's lifetime, and a dropped allocation removes
/// its own entry.
static EXPORTABLE: Mutex<BTreeMap<u64, Weak<Memory>>> = Mutex::new(BTreeMap::new());

impl Memory {
    /// Device memory on `device`. Contents are uninitialized.
    pub fn device(device: DeviceId, bytes: usize) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::linear(device, bytes)?;
        Ok(Arc::new(Self {
            kind: Kind::Device(allocation),
        }))
    }

    /// Device memory that can be exported in place to other processes, with
    /// read-write access for `device` and every device in `access`. The
    /// length is rounded up to the allocation granularity. Contents are
    /// uninitialized.
    pub(crate) fn exportable(
        device: DeviceId,
        bytes: usize,
        access: &[DeviceId],
    ) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::exportable(device, bytes, access)?;
        let address = allocation.address();
        let memory = Arc::new(Self {
            kind: Kind::Device(allocation),
        });
        EXPORTABLE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(address, Arc::downgrade(&memory));
        Ok(memory)
    }

    /// Every other visible device that can address `device`'s memory: the
    /// devices an exportable allocation grants access to, so in-process
    /// readers on those devices can copy from it.
    pub(crate) fn peers(device: DeviceId) -> Result<Vec<DeviceId>> {
        let mut peers = Vec::new();
        for peer in (0..DeviceId::count()?).map(DeviceId) {
            if peer != device && peer.can_access_peer(device)? {
                peers.push(peer);
            }
        }
        Ok(peers)
    }

    /// Page-locked host memory usable as a CUDA copy source or destination
    /// for `device`. Contents are uninitialized.
    pub fn pinned(device: DeviceId, bytes: usize) -> Result<Arc<Self>> {
        let buffer = PinnedBuffer::new(device, bytes)?;
        Ok(Arc::new(Self {
            kind: Kind::Pinned(buffer),
        }))
    }

    /// Zeroed pageable host memory.
    pub fn host(bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            kind: Kind::Host(vec![0; bytes].into_boxed_slice()),
        })
    }

    /// Adopt a mapping of exported device memory.
    pub(crate) fn mapped(mapping: PeerMapping) -> Arc<Self> {
        Arc::new(Self {
            kind: Kind::Mapped(mapping),
        })
    }

    /// The device allocation, `None` for host memory and mappings.
    pub(crate) fn allocation(&self) -> Option<&DeviceAllocation> {
        match &self.kind {
            Kind::Device(allocation) => Some(allocation),
            _ => None,
        }
    }

    /// The live exportable allocation containing `address`, with the
    /// address's byte offset inside it.
    pub(crate) fn exportable_containing(address: u64) -> Option<(Arc<Self>, u64)> {
        let registry = EXPORTABLE.lock().unwrap_or_else(PoisonError::into_inner);
        let (&base, memory) = registry.range(..=address).next_back()?;
        let memory = memory.upgrade()?;
        (address < base + memory.bytes() as u64).then(|| (memory, address - base))
    }

    /// The allocation's host-visible bytes. `None` for device memory.
    ///
    /// # Safety
    ///
    /// No device or other host access may write the returned range while the
    /// slice is alive.
    pub unsafe fn host_bytes(&self) -> Option<&[u8]> {
        match &self.kind {
            Kind::Device(_) | Kind::Mapped(_) => None,
            // SAFETY: the buffer is owned by `self` and the caller excludes
            // concurrent writers.
            Kind::Pinned(buffer) => {
                Some(unsafe { std::slice::from_raw_parts(buffer.as_ptr(), buffer.len()) })
            }
            Kind::Host(bytes) => Some(bytes),
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        if let Kind::Device(allocation) = &self.kind
            && allocation.handle_type().is_some()
        {
            let mut registry = EXPORTABLE.lock().unwrap_or_else(PoisonError::into_inner);
            // A new allocation may already reuse the address; remove only a
            // dead entry.
            if registry
                .get(&allocation.address())
                .is_some_and(|entry| entry.strong_count() == 0)
            {
                registry.remove(&allocation.address());
            }
        }
    }
}

impl Backing for Memory {
    fn device(&self) -> MemoryKind {
        match &self.kind {
            Kind::Device(allocation) => MemoryKind::Cuda(allocation.device()),
            Kind::Mapped(mapping) => MemoryKind::Cuda(mapping.device()),
            Kind::Pinned(_) => MemoryKind::PinnedHost,
            Kind::Host(_) => MemoryKind::Host,
        }
    }

    fn address(&self) -> u64 {
        match &self.kind {
            Kind::Device(allocation) => allocation.address(),
            Kind::Mapped(mapping) => mapping.address(),
            Kind::Pinned(buffer) => buffer.address(),
            Kind::Host(bytes) => bytes.as_ptr() as u64,
        }
    }

    fn bytes(&self) -> usize {
        match &self.kind {
            Kind::Device(allocation) => allocation.len(),
            Kind::Mapped(mapping) => mapping.len(),
            Kind::Pinned(buffer) => buffer.len(),
            Kind::Host(bytes) => bytes.len(),
        }
    }
}

/// A typed, strided window into an allocation.
///
/// `offset` is in bytes; `shape` and `strides` are in elements. Construction
/// checks that the window lies inside its allocation.
///
/// A view and its clones are one tensor to Python: they share the torch
/// tensor the view is wrapped as ([`Self::handle`]). A view derived from it
/// (a prefix, a row, a region, a reshape) is another tensor, wrapped on its
/// own; narrowing an axis to its whole extent returns the view itself.
#[derive(Clone)]
pub struct StorageView {
    memory: Arc<dyn Backing>,
    offset: usize,
    dtype: DType,
    shape: Vec<i64>,
    strides: Vec<i64>,
    /// The torch tensor of the view, made by the first call that passes it.
    /// Its last clone's drop releases it on the Python thread.
    tensor: Arc<OnceLock<Handle>>,
}

/// An owned copy operand: a view's address, element width, shape and
/// element strides in the form [`uniserve_cuda::StridedCopy`] plans.
pub(crate) struct CopyOperand {
    address: u64,
    element_bytes: usize,
    shape: Vec<u64>,
    strides: Vec<i64>,
}

impl CopyOperand {
    pub(crate) fn strided(&self) -> Strided<'_> {
        Strided {
            address: self.address,
            element_bytes: self.element_bytes,
            shape: &self.shape,
            strides: &self.strides,
        }
    }
}

impl StorageView {
    /// A contiguous view of `shape` starting `offset` bytes into `memory`.
    pub fn contiguous(
        memory: Arc<dyn Backing>,
        offset: usize,
        dtype: DType,
        shape: &[i64],
    ) -> Result<Self> {
        Self::strided(memory, offset, dtype, shape, &contiguous_strides(shape))
    }

    pub(crate) fn strided(
        memory: Arc<dyn Backing>,
        offset: usize,
        dtype: DType,
        shape: &[i64],
        strides: &[i64],
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(Error::Invariant(
                "a view needs one stride per dimension".to_owned(),
            ));
        }
        let view = Self {
            memory,
            offset,
            dtype,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            tensor: Arc::default(),
        };
        view.byte_len()?;
        Ok(view)
    }

    /// Another window of the same allocation and dtype, inside this one:
    /// a different tensor, wrapped on its own.
    fn derive(&self, offset: usize, shape: Vec<i64>, strides: Vec<i64>) -> Self {
        Self {
            memory: Arc::clone(&self.memory),
            offset,
            dtype: self.dtype,
            shape,
            strides,
            tensor: Arc::default(),
        }
    }

    /// A whole allocation viewed as `[len / dtype.bytes()]` elements.
    pub fn whole(memory: Arc<dyn Backing>, dtype: DType) -> Result<Self> {
        let elements = memory.bytes() / dtype.bytes();
        Self::contiguous(memory, 0, dtype, &[elements as i64])
    }

    /// The allocation this view retains.
    pub fn memory(&self) -> &Arc<dyn Backing> {
        &self.memory
    }

    /// Byte offset of the first element inside the allocation.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn device(&self) -> MemoryKind {
        self.memory.device()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub(crate) fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Address of the first element.
    pub fn address(&self) -> u64 {
        self.memory.address() + self.offset as u64
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Elements addressed by the view.
    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|&extent| extent as u64).product()
    }

    /// Bytes of the addressed elements, excluding strided gaps.
    pub(crate) fn nbytes(&self) -> u64 {
        self.numel() * self.dtype.bytes() as u64
    }

    /// Bytes from the first to one past the last element.
    pub fn byte_len(&self) -> Result<usize> {
        if self.shape.contains(&0) {
            return Ok(0);
        }
        if self
            .shape
            .iter()
            .chain(&self.strides)
            .any(|&value| value < 0)
        {
            return Err(Error::Invariant(
                "views require non-negative extents and strides".to_owned(),
            ));
        }
        let last: i64 = self
            .shape
            .iter()
            .zip(&self.strides)
            .map(|(&extent, &stride)| (extent - 1) * stride)
            .sum();
        let bytes = (last as usize + 1) * self.dtype.bytes();
        if self.offset + bytes > self.memory.bytes() {
            return Err(Error::Invariant(format!(
                "a view of {bytes} bytes at offset {} exceeds its {}-byte allocation",
                self.offset,
                self.memory.bytes()
            )));
        }
        Ok(bytes)
    }

    /// The leading `count` entries along dimension 0.
    pub(crate) fn prefix(&self, count: i64) -> Result<Self> {
        let extent = *self
            .shape
            .first()
            .ok_or_else(|| Error::Invariant("a scalar view has no leading dimension".to_owned()))?;
        if count > extent {
            return Err(Error::Invariant(format!(
                "prefix of {count} exceeds the leading extent {extent}"
            )));
        }
        self.narrow(0, 0, count)
    }

    /// Elements `[start, start + length)` of `axis`. The whole axis is this
    /// view itself.
    pub(crate) fn narrow(&self, axis: usize, start: i64, length: i64) -> Result<Self> {
        let extent = *self
            .shape
            .get(axis)
            .ok_or_else(|| Error::Invariant(format!("view has no axis {axis}")))?;
        if start < 0 || length < 0 || start + length > extent {
            return Err(Error::Invariant(format!(
                "range {start}..{} exceeds axis {axis} of extent {extent}",
                start + length
            )));
        }
        if start == 0 && length == extent {
            return Ok(self.clone());
        }
        let offset = self.offset + (start * self.strides[axis]) as usize * self.dtype.bytes();
        let mut shape = self.shape.clone();
        shape[axis] = length;
        Ok(self.derive(offset, shape, self.strides.clone()))
    }

    /// The view at `index` of `axis`, with that axis removed.
    pub(crate) fn select(&self, axis: usize, index: i64) -> Result<Self> {
        let row = self.narrow(axis, index, 1)?;
        let (mut shape, mut strides) = (row.shape, row.strides);
        shape.remove(axis);
        strides.remove(axis);
        Ok(self.derive(row.offset, shape, strides))
    }

    /// The per-axis `[start, end)` region of this view.
    pub(crate) fn region(&self, region: &[Range<u64>]) -> Result<Self> {
        if region.len() != self.shape.len() {
            return Err(Error::Invalid(
                "a region needs one range per view axis".to_owned(),
            ));
        }
        let mut view = self.clone();
        for (axis, range) in region.iter().enumerate() {
            if range.start > range.end {
                return Err(Error::Invalid("a region range is reversed".to_owned()));
            }
            view = view.narrow(axis, range.start as i64, (range.end - range.start) as i64)?;
        }
        Ok(view)
    }

    /// The same bytes reinterpreted as a contiguous `shape` of `dtype`.
    pub(crate) fn reshape(&self, dtype: DType, shape: &[i64]) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(Error::Invariant(
                "only contiguous views can be reshaped".to_owned(),
            ));
        }
        let view = Self::contiguous(Arc::clone(&self.memory), self.offset, dtype, shape)?;
        if view.byte_len()? > self.byte_len()? {
            return Err(Error::Invariant(
                "a reshape cannot grow its view".to_owned(),
            ));
        }
        Ok(view)
    }

    /// The view as a strided copy operand.
    pub(crate) fn operand(&self) -> CopyOperand {
        CopyOperand {
            address: self.address(),
            element_bytes: self.dtype.bytes(),
            shape: self.shape.iter().map(|&extent| extent as u64).collect(),
            strides: self.strides.clone(),
        }
    }

    /// The maximal contiguous byte intervals the view addresses, in address
    /// order. Axes whose stride equals the run below them fold into the run.
    pub(crate) fn intervals(&self) -> Vec<Range<u64>> {
        if self.shape.contains(&0) {
            return Vec::new();
        }
        let element = self.dtype.bytes() as u64;
        let mut axes: Vec<(u64, u64)> = self
            .strides
            .iter()
            .zip(&self.shape)
            .filter(|(_, extent)| **extent > 1)
            .map(|(&stride, &extent)| (stride as u64, extent as u64))
            .collect();
        axes.sort_unstable();

        let mut width = 1;
        let mut outer = Vec::new();
        for (stride, extent) in axes {
            if stride == width && outer.is_empty() {
                width *= extent;
            } else {
                outer.push((stride, extent));
            }
        }

        let base = self.address();
        let mut starts = vec![0_u64];
        for (stride, extent) in outer {
            starts = starts
                .iter()
                .flat_map(|&start| (0..extent).map(move |index| start + index * stride))
                .collect();
        }
        let mut intervals: Vec<_> = starts
            .into_iter()
            .map(|start| {
                let start = base + start * element;
                start..start + width * element
            })
            .collect();
        intervals.sort_unstable_by_key(|interval| interval.start);
        intervals
    }

    /// Write `bytes` into the leading bytes of this contiguous view, after
    /// the work `stream` issued before (it has a stream exactly for device
    /// memory). The source is read before the call returns, so the caller
    /// may reuse it at once.
    ///
    /// # Safety
    ///
    /// The write runs when `stream` runs it: during the call for host
    /// memory without a stream, after the stream's earlier work on a
    /// stream, when the job runs for a job's recorder. Until it ran, nothing
    /// reads or writes the destination bytes except work ordered after it
    /// (behind it on `stream`, or behind a fence of `stream`), and the
    /// view's allocation keeps a reference: the caller retires the write
    /// before dropping the allocation's last reference, as for any device
    /// access.
    pub(crate) unsafe fn upload(&self, bytes: &[u8], stream: impl Issue) -> Result<()> {
        if !self.is_contiguous() || bytes.len() as u64 > self.nbytes() {
            return Err(Error::Invariant(
                "an upload writes a prefix of a contiguous view".to_owned(),
            ));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if matches!(self.device(), MemoryKind::Cuda(_)) != stream.stream().is_some() {
            return Err(Error::Invariant(
                "uploads use a stream exactly for device memory".to_owned(),
            ));
        }

        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        batch.push(self.address(), bytes.as_ptr() as u64, bytes.len());
        // SAFETY: the destination lies in this view; the caller keeps it
        // allocated and unaccessed until the copy ran. The pageable source
        // is consumed during the call.
        unsafe { stream.copy(&batch) }
    }

    /// Zero this contiguous view, in the order of what `stream` issues for
    /// device memory (a stream at once, or a job's recorder).
    ///
    /// # Safety
    ///
    /// As for [`Self::upload`], over the whole view: until the fill ran, no
    /// access that is not ordered after it reads or writes the view, and
    /// the view's allocation keeps a reference.
    pub(crate) unsafe fn zero(&self, stream: impl Issue) -> Result<()> {
        if !self.is_contiguous() {
            return Err(Error::Invariant(
                "only a contiguous view is zeroed".to_owned(),
            ));
        }
        if matches!(self.device(), MemoryKind::Cuda(_)) != stream.stream().is_some() {
            return Err(Error::Invariant(
                "fills use a stream exactly for device memory".to_owned(),
            ));
        }
        // SAFETY: the bytes lie in this view, which the caller keeps
        // allocated and unaccessed until the fill ran.
        unsafe {
            stream.fill(Fill {
                address: self.address(),
                value: 0,
                width: 1,
                count: self.nbytes() as usize,
            })
        }
    }

    /// The view as a TVM-FFI tensor: a DLPack export retaining the
    /// allocation. Python callables receive the view's torch tensor instead
    /// ([`Self::handle`]); a harness returns this export to its Python
    /// caller.
    pub fn to_tensor(&self) -> Result<Tensor> {
        export(
            Arc::clone(&self.memory),
            self.offset,
            self.dtype,
            &self.shape,
            &self.strides,
        )
    }

    /// The view as the torch tensor every call that takes it passes.
    ///
    /// The first call that passes the view wraps its DLPack export with
    /// `torch.from_dlpack`, called as it is; every later call through any
    /// clone of the view passes that same Python object, which costs no
    /// conversion. The tensor retains the allocation through its DLPack
    /// deleter, so the memory outlives every Python reference to it; the
    /// view releases its reference with its last clone.
    ///
    /// The first wrapping calls Python, a handoff of its own from another
    /// thread; a serving path makes it inside the [`Interpreter::execute`]
    /// of the call that passes the view, so the two take one handoff. A
    /// caller drops the references it makes for a call ([`Handle::to_any`])
    /// before the view or on the Python thread, so that the view's own
    /// reference is the last one and is released on the Python thread.
    pub(crate) fn handle(&self) -> Result<&Handle> {
        if let Some(handle) = self.tensor.get() {
            return Ok(handle);
        }
        let export = Any::from(self.to_tensor()?);
        let handle = Handle::new(from_dlpack()?.call(&[AnyView::from(&export)])?)?;
        Ok(self.tensor.get_or_init(|| handle))
    }
}

/// `torch.from_dlpack`, resolved once per process.
fn from_dlpack() -> Result<&'static Function> {
    static FROM_DLPACK: OnceLock<Function> = OnceLock::new();
    if let Some(function) = FROM_DLPACK.get() {
        return Ok(function);
    }
    let function = Interpreter::import("torch.from_dlpack")?;
    Ok(FROM_DLPACK.get_or_init(|| function))
}

impl std::fmt::Debug for StorageView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageView")
            .field("device", &self.device())
            .field("offset", &self.offset)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_stay_inside_their_allocation() -> Result<()> {
        let memory: Arc<dyn Backing> = Memory::host(64);
        assert!(StorageView::contiguous(Arc::clone(&memory), 0, DType::F32, &[4, 4]).is_ok());
        assert!(StorageView::contiguous(Arc::clone(&memory), 4, DType::F32, &[4, 4]).is_err());

        let rows = StorageView::contiguous(Arc::clone(&memory), 0, DType::I64, &[8])?;
        assert_eq!(rows.prefix(3)?.byte_len()?, 24);
        assert!(rows.prefix(9).is_err());
        assert_eq!(rows.reshape(DType::I32, &[16])?.byte_len()?, 64);
        assert!(rows.reshape(DType::I32, &[17]).is_err());
        Ok(())
    }

    #[test]
    fn regions_select_and_transposes_address_the_same_elements() -> Result<()> {
        let memory: Arc<dyn Backing> = Memory::host(4 * 6 * 4);
        let base = memory.address();
        let grid = StorageView::contiguous(Arc::clone(&memory), 0, DType::F32, &[4, 6])?;

        let window = grid.region(&[1..3, 2..5])?;
        assert_eq!(window.shape(), &[2, 3]);
        assert_eq!(window.address(), base + (6 + 2) * 4);
        assert_eq!(
            window.intervals(),
            vec![base + 32..base + 44, base + 56..base + 68,]
        );

        let column = grid.select(1, 5)?;
        assert_eq!(
            (column.shape(), column.strides()),
            (&[4_i64][..], &[6_i64][..])
        );
        let transposed = StorageView::strided(memory, 0, DType::F32, &[6, 4], &[1, 6])?;
        assert_eq!(transposed.shape(), &[6, 4]);
        // A transposed view addresses the same bytes: one contiguous interval.
        assert_eq!(transposed.intervals(), vec![base..base + 96]);
        assert!(grid.region(&[0..5, 0..6]).is_err());
        Ok(())
    }

    #[test]
    fn dtype_names_round_trip() -> Result<()> {
        for dtype in [
            DType::Bool,
            DType::U8,
            DType::BF16,
            DType::F8E4M3,
            DType::F64,
        ] {
            assert_eq!(DType::parse(dtype.name())?, dtype);
        }
        assert!(DType::parse("complex64").is_err());
        Ok(())
    }
}
