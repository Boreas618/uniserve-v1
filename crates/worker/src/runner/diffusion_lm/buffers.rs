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
//! [`Dimensions`] states the columns of the runner's input buffers: packed
//! tokens and positions, the attention metadata the sequences read, readout
//! slot and candidate columns, the final stage's attend state, and on
//! generating ranks the denoising columns, the gathered state rows and the
//! results. The runner writes host values into them with [`ColumnWrites`]
//! and gathers bank rows into the state rows natively ([`Banks::gather`]).

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};

use super::calls::Sequence;
use crate::ffi::DType;
use crate::memory::allocation::StorageView;
use crate::memory::{RequestSlots, SlotField, TablePages};
use crate::runner::input_buffers::{Columns, InputBuffers};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};

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
    slots: RequestSlots,
    views: Vec<(String, StorageView)>,
}

impl Banks {
    /// Allocate zeroed banks on `device` (host memory when `None`); device
    /// banks are zeroed on `stream`, which is synchronized before
    /// returning. Every slot's state starts zero, and slot zero stays zero.
    pub(super) fn allocate(
        layout: BankLayout,
        device: Option<DeviceId>,
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
        zeros.submit(stream)?;
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(Self {
            layout,
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
    /// The copies are issued to `issue` after earlier commits to the banks:
    /// on the lane's stream for device banks, as host writes for host banks.
    pub(super) fn gather(
        &self,
        slots: &[u32],
        rows: usize,
        step: &DenoiseViews,
        issue: impl Issue,
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
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for &(destination, source, bytes) in &copies {
            batch.push(destination, source, bytes);
        }
        // SAFETY: sources are bank rows and destinations step rows inside
        // live allocations that outlive the issued work; destinations are
        // disjoint, and the issuer orders the copies after earlier commits.
        unsafe {
            issue.copy(&batch)?;
            for &(address, bytes) in &zeros {
                if bytes > 0 {
                    issue.fill(Fill {
                        address,
                        value: 0,
                        width: 1,
                        count: bytes,
                    })?;
                }
            }
        }
        Ok(())
    }
}

/// The denoising columns of a generating rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DenoiseLayout {
    /// Rows of one denoising step.
    pub(super) rows: usize,
    pub(super) banks: BankLayout,
}

/// The dimensions of the runner's columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Dimensions {
    /// Packed sequences of one forward, padding included.
    pub(super) rows: usize,
    /// Packed tokens of one forward, padding included.
    pub(super) tokens: usize,
    /// Numerical block tables.
    pub(super) tables: usize,
    /// Page columns of every table's rows.
    pub(super) width: usize,
    /// Slot rows of the largest `readout` graph.
    pub(super) slot_rows: usize,
    /// Candidates one `readout` selects.
    pub(super) candidates: usize,
    /// Per-token widths of the attend state (final stage only).
    pub(super) attend_widths: Vec<usize>,
    pub(super) attend_dtype: DType,
    pub(super) denoise: Option<DenoiseLayout>,
}

impl Columns for Dimensions {
    type Name = String;

    fn columns(&self) -> Vec<(String, DType, Vec<i64>)> {
        let (rows, tokens) = (self.rows as i64, self.tokens as i64);
        let tables = self.tables as i64;
        let candidates = self.candidates as i64;
        let mut columns = vec![
            ("input_ids".to_owned(), DType::I64, vec![tokens]),
            ("positions".to_owned(), DType::I64, vec![tokens]),
            ("attention.query_lengths".to_owned(), DType::I32, vec![rows]),
            (
                "attention.query_offsets".to_owned(),
                DType::I32,
                vec![rows + 1],
            ),
            (
                "attention.prefix_lengths".to_owned(),
                DType::I32,
                vec![rows],
            ),
            (
                "attention.prefix_offsets".to_owned(),
                DType::I32,
                vec![rows + 1],
            ),
            (
                "attention.block_tables".to_owned(),
                DType::I32,
                vec![tables, rows, self.width as i64],
            ),
            (
                "attention.start_pages".to_owned(),
                DType::I32,
                vec![tables, rows],
            ),
            (
                "slot_rows".to_owned(),
                DType::I64,
                vec![tokens.max(self.slot_rows as i64)],
            ),
            ("candidate_index".to_owned(), DType::I64, vec![candidates]),
            (
                "candidate_logprobs".to_owned(),
                DType::F32,
                vec![candidates],
            ),
        ];
        for (index, &width) in self.attend_widths.iter().enumerate() {
            columns.push((
                format!("attend_state.{index}"),
                self.attend_dtype,
                vec![tokens, width as i64],
            ));
        }
        if let Some(denoise) = &self.denoise {
            let banks = &denoise.banks;
            let rows = denoise.rows as i64;
            let (length, depth, hidden) =
                (banks.length as i64, banks.depth as i64, banks.hidden as i64);
            columns.extend([
                ("denoise.columns".to_owned(), DType::I64, vec![4, rows]),
                ("denoise.tokens".to_owned(), DType::I64, vec![rows, length]),
                (
                    "denoise.history".to_owned(),
                    DType::I64,
                    vec![rows, depth, length],
                ),
                (
                    "denoise.self_conditioning".to_owned(),
                    banks.dtype,
                    vec![rows, length, hidden],
                ),
                (
                    "denoise.results".to_owned(),
                    DType::I64,
                    vec![rows, 1 + length],
                ),
                ("denoise.continued".to_owned(), DType::U8, vec![rows]),
            ]);
        }
        columns
    }
}

/// The state rows of a denoising step: a gather fills them from the banks
/// and the step reads and steps them.
pub(super) struct DenoiseViews {
    pub(super) rows: usize,
    pub(super) tokens: StorageView,
    pub(super) history: StorageView,
    pub(super) self_conditioning: StorageView,
}

impl DenoiseViews {
    /// The state rows among `buffers`' columns, on a generating rank.
    pub(super) fn of(buffers: &InputBuffers<Dimensions>) -> Result<Option<Self>> {
        let Some(denoise) = &buffers.dimensions.denoise else {
            return Ok(None);
        };
        Ok(Some(Self {
            rows: denoise.rows,
            tokens: buffers.column("denoise.tokens")?.clone(),
            history: buffers.column("denoise.history")?.clone(),
            self_conditioning: buffers.column("denoise.self_conditioning")?.clone(),
        }))
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

    /// Issue every fill, then every copy, to `issue`: the lane's stream for
    /// device buffers, host writes for host buffers.
    pub(super) fn submit(self, issue: impl Issue) -> Result<()> {
        // SAFETY: every destination lies inside a lane allocation that
        // outlives the issued work; fills precede the copies in issue
        // order; the batch's pageable payload is read during the call.
        unsafe {
            for &(address, bytes) in &self.fills {
                issue.fill(Fill {
                    address,
                    value: 0,
                    width: 1,
                    count: bytes,
                })?;
            }
            let mut batch = CopyBatch::new(SourceOrder::DuringCall);
            for &(address, first, bytes) in &self.copies {
                batch.push(address, self.payload[first..].as_ptr() as u64, bytes);
            }
            issue.copy(&batch)?;
        }
        Ok(())
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
    buffers: &InputBuffers<Dimensions>,
    writes: &mut ColumnWrites,
    rows: &[Sequence],
    padding: &[usize],
    pages: &[TablePages],
    widths: &[usize],
) -> Result<AttentionMetadata> {
    let layout = &buffers.dimensions;
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
        buffers.column("attention.query_lengths")?,
        0,
        &narrow(&queries),
    )?;
    writes.write(
        buffers.column("attention.query_offsets")?,
        0,
        &offsets(&queries)?,
    )?;
    writes.write(
        buffers.column("attention.prefix_lengths")?,
        0,
        &narrow(&prefixes),
    )?;
    writes.write(
        buffers.column("attention.prefix_offsets")?,
        0,
        &offsets(&prefixes)?,
    )?;

    let tables = buffers.column("attention.block_tables")?;
    let start_pages = buffers.column("attention.start_pages")?;
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
        writes.zero(tables, base, count * layout.width)?;
        for (row, units) in selected.rows().enumerate() {
            let units: Vec<i32> = units.map(|unit| unit as i32).collect();
            if units.len() > width {
                return Err(crate::Error::Invalid(format!(
                    "a block row selects {} pages of table {table}, beyond its {width} page columns",
                    units.len()
                )));
            }
            writes.write(tables, base + row * layout.width, &units)?;
        }
        let first: Vec<i32> = selected
            .starts
            .iter()
            .map(|&page| page as i32)
            .chain(padding.iter().map(|_| 0))
            .collect();
        writes.write(start_pages, table * layout.rows, &first)?;
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
    buffers: &InputBuffers<Dimensions>,
    writes: &mut ColumnWrites,
    rows: &[Sequence],
    tokens: usize,
) -> Result<()> {
    let positions: Vec<i64> = rows
        .iter()
        .flat_map(|row| (row.position..row.position + row.query as u64).map(|value| value as i64))
        .collect();
    let view = buffers.column("positions")?;
    writes.write(view, 0, &positions)?;
    writes.zero(
        view,
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
        let tokens = resident.views()[0].1.clone();
        assert_eq!(tokens.shape(), &[4, 4]);
        writes.write(&tokens, 0, &(0..16).collect::<Vec<i64>>())?;
        writes.submit(None::<&Stream>)?;

        let lane = InputBuffers::allocate(
            Dimensions {
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
        let step = DenoiseViews::of(&lane)?.expect("a generating runner has state rows");
        let step = &step;
        // Stale contents of the padding row must not survive a gather.
        let mut stale = ColumnWrites::new();
        stale.write(&step.tokens, 8, &[7i64; 4])?;
        stale.submit(None::<&Stream>)?;

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
        let lane = InputBuffers::allocate(
            Dimensions {
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
        let ids = lane.column("input_ids")?;
        let mut writes = ColumnWrites::new();
        assert!(writes.write(ids, 2, &[1i64, 2, 3]).is_err());
        assert!(writes.write(ids, 0, &[1i32]).is_err());
        writes.write(ids, 1, &[5i64, 6])?;
        writes.zero(ids, 3, 1)?;
        writes.submit(None::<&Stream>)?;
        assert_eq!(read::<i64>(ids, 4), [0, 5, 6, 0]);
        Ok(())
    }
}
