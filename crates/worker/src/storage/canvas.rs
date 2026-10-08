//! Resident canvas banks and the fixed buffers of canvas lanes.
//!
//! A generating token denoiser keeps, per request slot, the canvas its next
//! denoising step reads, the argmax canvases its stopping rule compares,
//! the self-conditioning embedding of its latest distribution and whether
//! its block continues ([`CanvasBanks`]). Slot zero is the padding sentinel:
//! numerical commits skip it, so it stays zero. Lanes share the banks for
//! disjoint slots.
//!
//! Each canvas lane owns its own fixed buffers ([`CanvasBuffers`]): packed
//! canvas tokens and positions, the attention metadata its sequences read,
//! readout slot and candidate columns, the final stage's attend state, and
//! on generating ranks the step columns, the gathered step views, the step
//! results and the sampler workspace. Every buffer is carved from one
//! allocation and bound to Python once. The runtime writes host values into
//! them with [`ColumnWrites`] and gathers bank rows into the step views
//! natively ([`CanvasBanks::gather`]).

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder, Stream, memset_d8};

use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{RequestSlots, SlotField};
use crate::{Error, Result};

/// Byte alignment of every carved buffer.
const ALIGNMENT: usize = 256;

/// Views of carved fields, by name.
type Views = Vec<(String, StorageView)>;

/// Fields carved from one allocation, by name.
struct Carving {
    fields: Vec<(String, DType, Vec<i64>)>,
}

impl Carving {
    fn new() -> Self {
        Self { fields: Vec::new() }
    }

    fn field(&mut self, name: impl Into<String>, dtype: DType, shape: &[i64]) {
        self.fields.push((name.into(), dtype, shape.to_vec()));
    }

    /// Byte offset of every field and the total length.
    fn layout(&self) -> (Vec<usize>, usize) {
        let mut cursor = 0;
        let offsets = self
            .fields
            .iter()
            .map(|(_, dtype, shape)| {
                let offset = cursor;
                let bytes = shape.iter().product::<i64>().max(0) as usize * dtype.bytes();
                cursor += bytes.next_multiple_of(ALIGNMENT);
                offset
            })
            .collect();
        (offsets, cursor.max(ALIGNMENT))
    }

    fn bytes(&self) -> u64 {
        self.layout().1 as u64
    }

    /// Allocate zeroed storage on `device` (host memory when `None`) and
    /// view every field. Device memory is zeroed on `stream`, which is
    /// synchronized before returning.
    fn allocate(
        &self,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<(Arc<Memory>, Views)> {
        let (offsets, bytes) = self.layout();
        let memory = match device {
            Some(device) => Memory::device(device, bytes)?,
            None => Memory::host(bytes),
        };
        if device.is_some() {
            let stream = stream.ok_or_else(|| {
                Error::Invalid("device canvas storage is zeroed on a stream".into())
            })?;
            // SAFETY: the fresh allocation spans `bytes`; the stream drains
            // before the storage is used.
            unsafe { memset_d8(memory.address(), 0, bytes, stream) }?;
            stream.synchronize()?;
        }
        let backing = Arc::clone(&memory) as Arc<dyn Backing>;
        let views = self
            .fields
            .iter()
            .zip(offsets)
            .map(|((name, dtype, shape), offset)| {
                Ok((
                    name.clone(),
                    StorageView::contiguous(Arc::clone(&backing), offset, *dtype, shape)?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok((memory, views))
    }
}

fn find(views: &[(String, StorageView)], name: &str) -> Result<StorageView> {
    views
        .iter()
        .find(|(field, _)| field == name)
        .map(|(_, view)| view.clone())
        .ok_or_else(|| Error::Invariant(format!("canvas storage has no {name} buffer")))
}

/// Dimensions of the resident banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankLayout {
    /// Request slots, excluding the padding slot zero.
    pub slots: usize,
    /// Canvas tokens `C`.
    pub length: usize,
    /// Previous argmax canvases kept per slot (the stability threshold).
    pub depth: usize,
    /// Self-conditioning embedding width `H`.
    pub hidden: usize,
    pub dtype: DType,
}

impl BankLayout {
    /// The per-slot fields, each one slot's extent.
    fn fields(&self) -> Vec<SlotField> {
        let (length, depth, hidden) = (self.length as i64, self.depth as i64, self.hidden as i64);
        let field = |name: &str, dtype, shape: &[i64]| SlotField {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            host: false,
        };
        vec![
            field("canvas", DType::I64, &[length]),
            field("history", DType::I64, &[depth, length]),
            field("self_conditioning", self.dtype, &[length, hidden]),
            field("live", DType::U8, &[]),
        ]
    }

    /// Bytes the banks allocate.
    pub fn bytes(&self) -> Result<u64> {
        let (device, host) = RequestSlots::bytes(&self.fields(), self.slots as u32)?;
        Ok(device + host)
    }

    /// Bytes of one slot's row of each gathered field: canvas, history and
    /// self-conditioning.
    fn row_bytes(&self) -> [usize; 3] {
        [
            self.length * 8,
            self.depth * self.length * 8,
            self.length * self.hidden * self.dtype.bytes(),
        ]
    }
}

/// The resident per-slot state of generating canvases.
pub struct CanvasBanks {
    layout: BankLayout,
    device: Option<i32>,
    slots: RequestSlots,
    views: Vec<(String, StorageView)>,
}

impl CanvasBanks {
    /// Allocate zeroed banks on `device` (host memory when `None`); device
    /// banks are zeroed on `stream`, which is synchronized before
    /// returning. Every slot's state starts zero, and slot zero stays zero.
    pub fn allocate(
        layout: BankLayout,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if layout.slots == 0 || layout.length == 0 || layout.hidden == 0 {
            return Err(Error::Invalid(
                "canvas bank dimensions must be positive".into(),
            ));
        }
        let slots = RequestSlots::allocate(&layout.fields(), layout.slots as u32, device)?;
        let views: Vec<(String, StorageView)> = slots
            .views()
            .into_iter()
            .map(|(name, view)| (format!("bank.{name}"), view))
            .collect();
        let mut zeros = ColumnWrites::new();
        for (_, view) in &views {
            zeros.zero(view, 0, view.numel() as usize)?;
        }
        zeros.submit(device, stream)?;
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(Self {
            layout,
            device,
            slots,
            views,
        })
    }

    pub fn layout(&self) -> BankLayout {
        self.layout
    }

    /// The banks keyed as the canvas binding receives them.
    pub fn views(&self) -> &[(String, StorageView)] {
        &self.views
    }

    /// Copy the rows of `slots` into the leading rows of `step`'s views and
    /// zero its rows `slots.len()..rows`, which then read as slot zero.
    ///
    /// Device copies are ordered on `stream` after earlier commits to the
    /// banks; host banks copy immediately (CPU lanes compute synchronously).
    pub fn gather(
        &self,
        slots: &[u32],
        rows: usize,
        step: &StepViews,
        stream: Option<&Stream>,
    ) -> Result<()> {
        if rows < slots.len() || rows > step.rows {
            return Err(Error::Invalid(format!(
                "a canvas step of {rows} rows exceeds its {} step rows",
                step.rows
            )));
        }
        let banks = [
            self.slots.bank("canvas")?,
            self.slots.bank("history")?,
            self.slots.bank("self_conditioning")?,
        ];
        let targets = [&step.canvas, &step.history, &step.self_conditioning];
        let row_bytes = self.layout.row_bytes();
        let mut copies = Vec::new();
        let mut zeros = Vec::new();
        for ((bank, target), bytes) in banks.iter().zip(targets).zip(row_bytes) {
            if bytes == 0 {
                continue;
            }
            for (row, &slot) in slots.iter().enumerate() {
                if slot == 0 || slot as usize > self.layout.slots {
                    return Err(Error::Invalid(format!(
                        "canvas slot {slot} is outside the resident banks"
                    )));
                }
                copies.push((
                    target.address() + (row * bytes) as u64,
                    bank.address() + (slot as usize * bytes) as u64,
                    bytes,
                ));
            }
            zeros.push((
                target.address() + (slots.len() * bytes) as u64,
                (rows - slots.len()) * bytes,
            ));
        }
        match (self.device, stream) {
            (Some(_), Some(stream)) => {
                let mut batch = CopyBatch::new(SourceOrder::Stream);
                for &(destination, source, bytes) in &copies {
                    batch.push(destination, source, bytes);
                }
                // SAFETY: sources are bank rows and destinations step rows
                // inside live allocations; destinations are disjoint, and
                // the stream orders the copies after earlier commits.
                unsafe {
                    batch.submit(stream)?;
                    for &(address, bytes) in &zeros {
                        if bytes > 0 {
                            memset_d8(address, 0, bytes, stream)?;
                        }
                    }
                }
                Ok(())
            }
            (None, None) => {
                for &(destination, source, bytes) in &copies {
                    // SAFETY: host banks and views of live allocations;
                    // CPU lanes serialize their numerical calls.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source as *const u8,
                            destination as *mut u8,
                            bytes,
                        );
                    }
                }
                for &(address, bytes) in &zeros {
                    // SAFETY: as above.
                    unsafe { std::ptr::write_bytes(address as *mut u8, 0, bytes) };
                }
                Ok(())
            }
            _ => Err(Error::Invalid(
                "canvas gathers use a stream exactly for device banks".into(),
            )),
        }
    }
}

/// The step buffers of a generating lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepLayout {
    /// Canvas rows of one step invocation.
    pub rows: usize,
    pub banks: BankLayout,
    /// Real vocabulary entries `V`.
    pub vocab: usize,
    /// Canvases the sampler steps at a time.
    pub chunk_rows: usize,
    /// Bytes of the product scratch of a full chunk.
    pub scratch: usize,
}

/// The fixed buffers of one canvas lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanvasLayout {
    /// Packed sequences of one invocation, padding included.
    pub rows: usize,
    /// Packed tokens of one invocation, padding included.
    pub tokens: usize,
    /// Numerical block tables.
    pub tables: usize,
    /// Page columns of every table's rows.
    pub width: usize,
    /// Slot rows of one readout tail chunk.
    pub tail_rows: usize,
    /// Candidates selected per readout pass.
    pub candidates: usize,
    /// Per-token widths of the attend state (final stage only).
    pub attend_widths: Vec<usize>,
    pub attend_dtype: DType,
    pub step: Option<StepLayout>,
}

impl CanvasLayout {
    fn carving(&self) -> Carving {
        let (rows, tokens) = (self.rows as i64, self.tokens as i64);
        let tables = self.tables as i64;
        let mut carving = Carving::new();
        carving.field("input_ids", DType::I64, &[tokens]);
        carving.field("positions", DType::I64, &[tokens]);
        carving.field("attention.query_lengths", DType::I32, &[rows]);
        carving.field("attention.query_offsets", DType::I32, &[rows + 1]);
        carving.field("attention.prefix_lengths", DType::I32, &[rows]);
        carving.field("attention.prefix_offsets", DType::I32, &[rows + 1]);
        carving.field(
            "attention.block_tables",
            DType::I32,
            &[tables, rows, self.width as i64],
        );
        carving.field("attention.start_pages", DType::I32, &[tables, rows]);
        carving.field("slot_tokens", DType::I64, &[tokens]);
        carving.field("tail_rows", DType::I64, &[self.tail_rows as i64]);
        carving.field("candidate_index", DType::I64, &[self.candidates as i64]);
        carving.field("candidate_logprobs", DType::F32, &[self.candidates as i64]);
        for (index, &width) in self.attend_widths.iter().enumerate() {
            carving.field(
                format!("attend_state.{index}"),
                self.attend_dtype,
                &[tokens, width as i64],
            );
        }
        if let Some(step) = &self.step {
            let banks = &step.banks;
            let rows = step.rows as i64;
            let (length, depth, hidden) =
                (banks.length as i64, banks.depth as i64, banks.hidden as i64);
            let positions = (step.chunk_rows * banks.length) as i64;
            carving.field("step.columns", DType::I64, &[4, rows]);
            carving.field("step.canvas", DType::I64, &[rows, length]);
            carving.field("step.history", DType::I64, &[rows, depth, length]);
            carving.field(
                "step.self_conditioning",
                banks.dtype,
                &[rows, length, hidden],
            );
            carving.field("step.results", DType::I64, &[rows, 1 + length]);
            carving.field("step.continued", DType::U8, &[rows]);
            carving.field(
                "workspace.weights",
                banks.dtype,
                &[positions, step.vocab as i64],
            );
            carving.field("workspace.normalizer", DType::F32, &[positions]);
            carving.field("workspace.product", DType::F32, &[positions, hidden]);
            carving.field("workspace.scratch", DType::U8, &[step.scratch as i64]);
        }
        carving
    }

    /// Bytes one lane's buffers allocate.
    pub fn bytes(&self) -> u64 {
        self.carving().bytes()
    }
}

/// The step views a gather fills and a step reads.
pub struct StepViews {
    pub rows: usize,
    pub canvas: StorageView,
    pub history: StorageView,
    pub self_conditioning: StorageView,
}

/// The fixed buffers of one canvas lane.
pub struct CanvasBuffers {
    layout: CanvasLayout,
    device: Option<i32>,
    views: Vec<(String, StorageView)>,
    step: Option<StepViews>,
    _memory: Arc<Memory>,
}

impl CanvasBuffers {
    /// Allocate zeroed buffers on `device` (host memory when `None`).
    pub fn allocate(
        layout: CanvasLayout,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if layout.rows == 0 || layout.tokens == 0 || layout.tables == 0 || layout.width == 0 {
            return Err(Error::Invalid(
                "canvas buffer dimensions must be positive".into(),
            ));
        }
        let (memory, views) = layout.carving().allocate(device, stream)?;
        let step = match &layout.step {
            Some(step) => Some(StepViews {
                rows: step.rows,
                canvas: find(&views, "step.canvas")?,
                history: find(&views, "step.history")?,
                self_conditioning: find(&views, "step.self_conditioning")?,
            }),
            None => None,
        };
        Ok(Self {
            layout,
            device,
            views,
            step,
            _memory: memory,
        })
    }

    pub fn layout(&self) -> &CanvasLayout {
        &self.layout
    }

    pub fn device(&self) -> Option<i32> {
        self.device
    }

    /// Every buffer keyed as the canvas binding receives it.
    pub fn views(&self) -> &[(String, StorageView)] {
        &self.views
    }

    /// One named buffer.
    pub fn view(&self, name: &str) -> Result<StorageView> {
        find(&self.views, name)
    }

    /// The gathered step views of a generating lane.
    pub fn step(&self) -> Option<&StepViews> {
        self.step.as_ref()
    }
}

/// Host values written into a lane's device columns by one invocation.
///
/// Values are staged in pageable memory and submitted as one batched copy
/// that reads its sources during the call, so the staging is reusable as
/// soon as [`ColumnWrites::submit`] returns. Every copy and fill is ordered
/// on the lane stream after the lane's earlier work, which may still read
/// the same buffers. Fills run before copies, so a copy may overwrite part
/// of a filled range; copy destinations must be disjoint.
#[derive(Default)]
pub struct ColumnWrites {
    copies: Vec<(u64, usize, usize)>,
    payload: Vec<u8>,
    fills: Vec<(u64, usize)>,
}

/// Element types a column write accepts.
pub trait Element: Copy {
    const DTYPE: DType;
    fn extend(values: &[Self], payload: &mut Vec<u8>);
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;
    fn extend(values: &[Self], payload: &mut Vec<u8>) {
        payload.extend(values.iter().flat_map(|value| value.to_ne_bytes()));
    }
}

impl Element for i32 {
    const DTYPE: DType = DType::I32;
    fn extend(values: &[Self], payload: &mut Vec<u8>) {
        payload.extend(values.iter().flat_map(|value| value.to_ne_bytes()));
    }
}

fn element_address(view: &StorageView, dtype: DType, element: usize, count: usize) -> Result<u64> {
    if view.dtype() != dtype || !view.is_contiguous() {
        return Err(Error::Invariant(format!(
            "a {dtype:?} column write needs a contiguous {dtype:?} buffer"
        )));
    }
    if element + count > view.numel() as usize {
        return Err(Error::Invalid(format!(
            "a write of {count} values at {element} exceeds a buffer of {} values",
            view.numel()
        )));
    }
    Ok(view.address() + (element * dtype.bytes()) as u64)
}

impl ColumnWrites {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write `values` into `view` starting at flat element `element`.
    pub fn write<T: Element>(
        &mut self,
        view: &StorageView,
        element: usize,
        values: &[T],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let address = element_address(view, T::DTYPE, element, values.len())?;
        let first = self.payload.len();
        T::extend(values, &mut self.payload);
        self.copies
            .push((address, first, self.payload.len() - first));
        Ok(())
    }

    /// Zero `count` elements of `view` starting at flat element `element`.
    pub fn zero(&mut self, view: &StorageView, element: usize, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let address = element_address(view, view.dtype(), element, count)?;
        self.fills.push((address, count * view.dtype().bytes()));
        Ok(())
    }

    /// Submit every fill, then every copy, on `stream` for device buffers
    /// (`device`) or immediately for host buffers.
    pub fn submit(self, device: Option<i32>, stream: Option<&Stream>) -> Result<()> {
        match (device, stream) {
            (Some(_), Some(stream)) => {
                // SAFETY: every destination lies inside a lane allocation
                // that outlives the stream's work; fills precede the copies
                // in stream order; the batch reads its pageable payload
                // during the call.
                unsafe {
                    for &(address, bytes) in &self.fills {
                        memset_d8(address, 0, bytes, stream)?;
                    }
                    let mut batch = CopyBatch::new(SourceOrder::DuringCall);
                    for &(address, first, bytes) in &self.copies {
                        batch.push(address, self.payload[first..].as_ptr() as u64, bytes);
                    }
                    batch.submit(stream)?;
                }
                Ok(())
            }
            (None, None) => {
                // SAFETY: host buffers of a CPU lane, which serializes its
                // numerical calls with these writes.
                unsafe {
                    for &(address, bytes) in &self.fills {
                        std::ptr::write_bytes(address as *mut u8, 0, bytes);
                    }
                    for &(address, first, bytes) in &self.copies {
                        std::ptr::copy_nonoverlapping(
                            self.payload[first..].as_ptr(),
                            address as *mut u8,
                            bytes,
                        );
                    }
                }
                Ok(())
            }
            _ => Err(Error::Invalid(
                "canvas column writes use a stream exactly for device buffers".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn banks() -> BankLayout {
        BankLayout {
            slots: 3,
            length: 4,
            depth: 1,
            hidden: 2,
            dtype: DType::F32,
        }
    }

    fn read<T: Copy>(view: &StorageView, count: usize) -> Vec<T> {
        // SAFETY: host views of test allocations, read after their writes.
        unsafe { std::slice::from_raw_parts(view.address() as *const T, count).to_vec() }
    }

    #[test]
    fn host_gathers_copy_slot_rows_and_zero_padding_rows() -> Result<()> {
        let layout = banks();
        let resident = CanvasBanks::allocate(layout, None, None)?;
        let mut writes = ColumnWrites::new();
        let canvas = find(resident.views(), "bank.canvas")?;
        assert_eq!(canvas.shape(), &[4, 4]);
        writes.write(&canvas, 0, &(0..16).collect::<Vec<i64>>())?;
        writes.submit(None, None)?;

        let lane = CanvasBuffers::allocate(
            CanvasLayout {
                rows: 6,
                tokens: 12,
                tables: 1,
                width: 2,
                tail_rows: 4,
                candidates: 8,
                attend_widths: vec![2],
                attend_dtype: DType::F32,
                step: Some(StepLayout {
                    rows: 3,
                    banks: layout,
                    vocab: 5,
                    chunk_rows: 1,
                    scratch: 0,
                }),
            },
            None,
            None,
        )?;
        let step = lane.step().expect("generating lanes have step views");
        // Stale contents of the padding row must not survive a gather.
        let mut stale = ColumnWrites::new();
        stale.write(&step.canvas, 8, &[7i64; 4])?;
        stale.submit(None, None)?;

        resident.gather(&[2, 1], 3, step, None)?;
        assert_eq!(
            read::<i64>(&step.canvas, 12),
            [8, 9, 10, 11, 4, 5, 6, 7, 0, 0, 0, 0]
        );
        assert!(resident.gather(&[0], 1, step, None).is_err());
        assert!(resident.gather(&[4], 1, step, None).is_err());
        assert!(resident.gather(&[1], 4, step, None).is_err());
        Ok(())
    }

    #[test]
    fn column_writes_check_their_destination() -> Result<()> {
        let lane = CanvasBuffers::allocate(
            CanvasLayout {
                rows: 2,
                tokens: 4,
                tables: 1,
                width: 1,
                tail_rows: 1,
                candidates: 1,
                attend_widths: Vec::new(),
                attend_dtype: DType::F32,
                step: None,
            },
            None,
            None,
        )?;
        let ids = lane.view("input_ids")?;
        let mut writes = ColumnWrites::new();
        assert!(writes.write(&ids, 2, &[1i64, 2, 3]).is_err());
        assert!(writes.write(&ids, 0, &[1i32]).is_err());
        writes.write(&ids, 1, &[5i64, 6])?;
        writes.zero(&ids, 3, 1)?;
        writes.submit(None, None)?;
        assert_eq!(read::<i64>(&ids, 4), [0, 5, 6, 0]);
        Ok(())
    }
}
