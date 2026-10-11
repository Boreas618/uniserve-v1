//! Storage of the diffusion LM runner: the resident banks and the input and
//! output tensors of its forwards.
//!
//! A generating diffusion LM keeps, per request slot, the block its next
//! denoising step reads, the argmax blocks its stopping rule compares, the
//! self-conditioning embedding of its latest distribution and whether its
//! block continues ([`Banks`]). Slot zero is the padding sentinel:
//! numerical commits skip it, so it stays zero. The Python runner receives
//! the banks once, at construction.
//!
//! [`Buffers`] holds the tensors of one forward: packed tokens and
//! positions, the attention metadata the sequences read, readout slot and
//! candidate columns, the final stage's attend state, and on generating
//! ranks the denoising columns, the gathered state rows and the results.
//! Every buffer is carved from one allocation; a method receives the
//! buffers it reads and writes as arguments, and a captured graph addresses
//! the buffers its capture was given, so the storage is fixed for the
//! runner's life. The runner writes host values into them with
//! [`ColumnWrites`] and gathers bank rows into the state rows natively
//! ([`Banks::gather`]).

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder, Stream, memset_d8};

use super::calls::Sequence;
use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{RequestSlots, SlotField, TablePages};
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
                Error::Invalid("device block storage is zeroed on a stream".into())
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
        .ok_or_else(|| Error::Invariant(format!("block storage has no {name} buffer")))
}

/// Dimensions of the resident banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BankLayout {
    /// Request slots, excluding the padding slot zero.
    pub(super) slots: usize,
    /// Block tokens `C`.
    pub(super) length: usize,
    /// Previous argmax blocks kept per slot (the stability threshold).
    pub(super) depth: usize,
    /// Self-conditioning embedding width `H`.
    pub(super) hidden: usize,
    pub(super) dtype: DType,
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
            field("tokens", DType::I64, &[length]),
            field("history", DType::I64, &[depth, length]),
            field("self_conditioning", self.dtype, &[length, hidden]),
            field("live", DType::U8, &[]),
        ]
    }

    /// Bytes the banks allocate.
    pub(super) fn bytes(&self) -> Result<u64> {
        let (device, host) = RequestSlots::bytes(&self.fields(), self.slots as u32)?;
        Ok(device + host)
    }

    /// Bytes of one slot's row of each gathered field: block, history and
    /// self-conditioning.
    fn row_bytes(&self) -> [usize; 3] {
        [
            self.length * 8,
            self.depth * self.length * 8,
            self.length * self.hidden * self.dtype.bytes(),
        ]
    }
}

/// The resident per-slot state of generating blocks.
pub(super) struct Banks {
    layout: BankLayout,
    device: Option<i32>,
    slots: RequestSlots,
    views: Vec<(String, StorageView)>,
}

impl Banks {
    /// Allocate zeroed banks on `device` (host memory when `None`); device
    /// banks are zeroed on `stream`, which is synchronized before
    /// returning. Every slot's state starts zero, and slot zero stays zero.
    pub(super) fn allocate(
        layout: BankLayout,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if layout.slots == 0 || layout.length == 0 || layout.hidden == 0 {
            return Err(Error::Invalid(
                "block bank dimensions must be positive".into(),
            ));
        }
        let slots = RequestSlots::allocate(&layout.fields(), layout.slots as u32, device)?;
        let views = slots.views();
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

    /// The banks by field: `tokens`, `history`, `self_conditioning` and
    /// `live`.
    pub(super) fn views(&self) -> &[(String, StorageView)] {
        &self.views
    }

    /// Copy the rows of `slots` into the leading rows of `step`'s views and
    /// zero its rows `slots.len()..rows`, which then read as slot zero.
    ///
    /// Device copies are ordered on `stream` after earlier commits to the
    /// banks; host banks copy immediately (CPU lanes compute synchronously).
    pub(super) fn gather(
        &self,
        slots: &[u32],
        rows: usize,
        step: &DenoiseViews,
        stream: Option<&Stream>,
    ) -> Result<()> {
        if rows < slots.len() || rows > step.rows {
            return Err(Error::Invalid(format!(
                "a block step of {rows} rows exceeds its {} step rows",
                step.rows
            )));
        }
        let banks = [
            self.slots.bank("tokens")?,
            self.slots.bank("history")?,
            self.slots.bank("self_conditioning")?,
        ];
        let targets = [&step.tokens, &step.history, &step.self_conditioning];
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
                        "block slot {slot} is outside the resident banks"
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
                "block gathers use a stream exactly for device banks".into(),
            )),
        }
    }
}

/// The step buffers of a generating lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DenoiseLayout {
    /// Block rows of one step invocation.
    pub(super) rows: usize,
    pub(super) banks: BankLayout,
}

/// The fixed buffers of one block lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BufferLayout {
    /// Packed sequences of one invocation, padding included.
    pub(super) rows: usize,
    /// Packed tokens of one invocation, padding included.
    pub(super) tokens: usize,
    /// Numerical block tables.
    pub(super) tables: usize,
    /// Page columns of every table's rows.
    pub(super) width: usize,
    /// Slot rows of the largest `readout` graph.
    pub(super) slot_rows: usize,
    /// Candidates selected per readout pass.
    pub(super) candidates: usize,
    /// Per-token widths of the attend state (final stage only).
    pub(super) attend_widths: Vec<usize>,
    pub(super) attend_dtype: DType,
    pub(super) denoise: Option<DenoiseLayout>,
}

impl BufferLayout {
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
        carving.field(
            "slot_rows",
            DType::I64,
            &[tokens.max(self.slot_rows as i64)],
        );
        carving.field("candidate_index", DType::I64, &[self.candidates as i64]);
        carving.field("candidate_logprobs", DType::F32, &[self.candidates as i64]);
        for (index, &width) in self.attend_widths.iter().enumerate() {
            carving.field(
                format!("attend_state.{index}"),
                self.attend_dtype,
                &[tokens, width as i64],
            );
        }
        if let Some(step) = &self.denoise {
            let banks = &step.banks;
            let rows = step.rows as i64;
            let (length, depth, hidden) =
                (banks.length as i64, banks.depth as i64, banks.hidden as i64);
            carving.field("denoise.columns", DType::I64, &[4, rows]);
            carving.field("denoise.tokens", DType::I64, &[rows, length]);
            carving.field("denoise.history", DType::I64, &[rows, depth, length]);
            carving.field(
                "denoise.self_conditioning",
                banks.dtype,
                &[rows, length, hidden],
            );
            carving.field("denoise.results", DType::I64, &[rows, 1 + length]);
            carving.field("denoise.continued", DType::U8, &[rows]);
        }
        carving
    }

    /// Bytes one lane's buffers allocate.
    pub(super) fn bytes(&self) -> u64 {
        self.carving().bytes()
    }
}

/// The step views a gather fills and a step reads.
pub(super) struct DenoiseViews {
    pub(super) rows: usize,
    pub(super) tokens: StorageView,
    pub(super) history: StorageView,
    pub(super) self_conditioning: StorageView,
}

/// The fixed buffers of one block lane.
pub(super) struct Buffers {
    layout: BufferLayout,
    device: Option<i32>,
    views: Vec<(String, StorageView)>,
    step: Option<DenoiseViews>,
    _memory: Arc<Memory>,
}

impl Buffers {
    /// Allocate zeroed buffers on `device` (host memory when `None`).
    pub(super) fn allocate(
        layout: BufferLayout,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if layout.rows == 0 || layout.tokens == 0 || layout.tables == 0 || layout.width == 0 {
            return Err(Error::Invalid(
                "block buffer dimensions must be positive".into(),
            ));
        }
        let (memory, views) = layout.carving().allocate(device, stream)?;
        let step = match &layout.denoise {
            Some(step) => Some(DenoiseViews {
                rows: step.rows,
                tokens: find(&views, "denoise.tokens")?,
                history: find(&views, "denoise.history")?,
                self_conditioning: find(&views, "denoise.self_conditioning")?,
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

    pub(super) fn layout(&self) -> &BufferLayout {
        &self.layout
    }

    pub(super) fn device(&self) -> Option<i32> {
        self.device
    }

    /// One named buffer.
    pub(super) fn view(&self, name: &str) -> Result<StorageView> {
        find(&self.views, name)
    }

    /// The gathered step views of a generating lane.
    pub(super) fn denoise(&self) -> Option<&DenoiseViews> {
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
pub(super) struct ColumnWrites {
    copies: Vec<(u64, usize, usize)>,
    payload: Vec<u8>,
    fills: Vec<(u64, usize)>,
}

/// Element types a column write accepts.
pub(super) trait Element: Copy {
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
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Write `values` into `view` starting at flat element `element`.
    pub(super) fn write<T: Element>(
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
    pub(super) fn zero(&mut self, view: &StorageView, element: usize, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let address = element_address(view, view.dtype(), element, count)?;
        self.fills.push((address, count * view.dtype().bytes()));
        Ok(())
    }

    /// Submit every fill, then every copy, on `stream` for device buffers
    /// (`device`) or immediately for host buffers.
    pub(super) fn submit(self, device: Option<i32>, stream: Option<&Stream>) -> Result<()> {
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
                "block column writes use a stream exactly for device buffers".into(),
            )),
        }
    }
}

/// The host mirror of the attention metadata the runner wrote, as the
/// runner's methods receive it.
pub(crate) struct AttentionMetadata {
    pub(crate) queries: Vec<i64>,
    pub(crate) prefixes: Vec<i64>,
    pub(crate) widths: Vec<i64>,
    /// First selected page of every row of every table, `[table][row]`.
    pub(crate) starts: Vec<i64>,
}

impl AttentionMetadata {
    pub(crate) fn rows(&self) -> usize {
        self.queries.len()
    }
}

/// Write the attention metadata of `rows` live sequences reading `pages`,
/// followed by `padding` sequences of the given query tokens reading no
/// prefix. Each table's rows hold `widths[table]` page columns: the pages a
/// live row selects, then zeros.
pub(crate) fn write_attention(
    buffers: &Buffers,
    writes: &mut ColumnWrites,
    rows: &[Sequence],
    padding: &[usize],
    pages: &[TablePages],
    widths: &[usize],
) -> Result<AttentionMetadata> {
    let layout = buffers.layout();
    let count = rows.len() + padding.len();
    if count > layout.rows {
        return Err(crate::Error::Invalid(format!(
            "a block invocation of {count} sequences exceeds its {} sequence rows",
            layout.rows
        )));
    }
    if pages.len() != layout.tables || widths.len() != layout.tables {
        return Err(crate::Error::Invariant(format!(
            "block attention reads {} tables, not {}",
            layout.tables,
            pages.len()
        )));
    }

    let queries: Vec<i64> = rows
        .iter()
        .map(|row| row.query as i64)
        .chain(padding.iter().map(|&query| query as i64))
        .collect();
    let prefixes: Vec<i64> = rows
        .iter()
        .map(|row| i64::from(row.prefix))
        .chain(padding.iter().map(|_| 0))
        .collect();
    let offsets = |lengths: &[i64]| -> Result<Vec<i32>> {
        let mut total = 0i64;
        std::iter::once(Ok(0))
            .chain(lengths.iter().map(|&length| {
                total += length;
                i32::try_from(total).map_err(|_| {
                    crate::Error::Invalid("block sequence offsets exceed int32".into())
                })
            }))
            .collect()
    };
    let narrow =
        |values: &[i64]| -> Vec<i32> { values.iter().map(|&value| value as i32).collect() };
    writes.write(
        &buffers.view("attention.query_lengths")?,
        0,
        &narrow(&queries),
    )?;
    writes.write(
        &buffers.view("attention.query_offsets")?,
        0,
        &offsets(&queries)?,
    )?;
    writes.write(
        &buffers.view("attention.prefix_lengths")?,
        0,
        &narrow(&prefixes),
    )?;
    writes.write(
        &buffers.view("attention.prefix_offsets")?,
        0,
        &offsets(&prefixes)?,
    )?;

    let tables = buffers.view("attention.block_tables")?;
    let start_pages = buffers.view("attention.start_pages")?;
    let mut starts = Vec::with_capacity(layout.tables * count);
    for (table, (selected, &width)) in pages.iter().zip(widths).enumerate() {
        if width > layout.width {
            return Err(crate::Error::Invariant(format!(
                "block table width {width} exceeds its {} page columns",
                layout.width
            )));
        }
        // Rows hold `layout.width` columns; the invocation reads `width`.
        // Zero every row first, so unselected columns and padding rows read
        // unit zero.
        let base = table * layout.rows * layout.width;
        writes.zero(&tables, base, count * layout.width)?;
        for (row, units) in selected.rows().enumerate() {
            let units: Vec<i32> = units.map(|unit| unit as i32).collect();
            if units.len() > width {
                return Err(crate::Error::Invalid(format!(
                    "a block row selects {} pages of table {table}, beyond its {width} page columns",
                    units.len()
                )));
            }
            writes.write(&tables, base + row * layout.width, &units)?;
        }
        let first: Vec<i32> = selected
            .starts
            .iter()
            .map(|&page| page as i32)
            .chain(padding.iter().map(|_| 0))
            .collect();
        writes.write(&start_pages, table * layout.rows, &first)?;
        starts.extend(first.iter().map(|&page| i64::from(page)));
    }
    Ok(AttentionMetadata {
        queries,
        prefixes,
        widths: widths.iter().map(|&width| width as i64).collect(),
        starts,
    })
}

/// Write each row's positions `position..position + query` into
/// `positions` back to back, then zeros up to `tokens`.
pub(crate) fn write_positions(
    buffers: &Buffers,
    writes: &mut ColumnWrites,
    rows: &[Sequence],
    tokens: usize,
) -> Result<()> {
    let positions: Vec<i64> = rows
        .iter()
        .flat_map(|row| (row.position..row.position + row.query as u64).map(|value| value as i64))
        .collect();
    let view = buffers.view("positions")?;
    writes.write(&view, 0, &positions)?;
    writes.zero(
        &view,
        positions.len(),
        tokens.saturating_sub(positions.len()),
    )
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
        let resident = Banks::allocate(layout, None, None)?;
        let mut writes = ColumnWrites::new();
        let tokens = find(resident.views(), "tokens")?;
        assert_eq!(tokens.shape(), &[4, 4]);
        writes.write(&tokens, 0, &(0..16).collect::<Vec<i64>>())?;
        writes.submit(None, None)?;

        let lane = Buffers::allocate(
            BufferLayout {
                rows: 6,
                tokens: 12,
                tables: 1,
                width: 2,
                slot_rows: 4,
                candidates: 8,
                attend_widths: vec![2],
                attend_dtype: DType::F32,
                denoise: Some(DenoiseLayout {
                    rows: 3,
                    banks: layout,
                }),
            },
            None,
            None,
        )?;
        let step = lane.denoise().expect("a generating runner has state rows");
        // Stale contents of the padding row must not survive a gather.
        let mut stale = ColumnWrites::new();
        stale.write(&step.tokens, 8, &[7i64; 4])?;
        stale.submit(None, None)?;

        resident.gather(&[2, 1], 3, step, None)?;
        assert_eq!(
            read::<i64>(&step.tokens, 12),
            [8, 9, 10, 11, 4, 5, 6, 7, 0, 0, 0, 0]
        );
        assert!(resident.gather(&[0], 1, step, None).is_err());
        assert!(resident.gather(&[4], 1, step, None).is_err());
        assert!(resident.gather(&[1], 4, step, None).is_err());
        Ok(())
    }

    #[test]
    fn column_writes_check_their_destination() -> Result<()> {
        let lane = Buffers::allocate(
            BufferLayout {
                rows: 2,
                tokens: 4,
                tables: 1,
                width: 1,
                slot_rows: 1,
                candidates: 1,
                attend_widths: Vec::new(),
                attend_dtype: DType::F32,
                denoise: None,
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
