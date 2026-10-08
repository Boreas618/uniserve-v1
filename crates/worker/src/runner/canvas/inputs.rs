//! Host descriptions of canvas invocations and their column writes.
//!
//! The executor validates calls and builds a [`ReadoutPass`] or a list of
//! [`StepRow`]s; the runner writes them into its lane's fixed buffers.
//! Every canvas sequence attends to its request's cached prefix through the
//! resident block tables and to its whole own canvas; nothing writes the
//! cache. Padding sequences read no prefix and table unit zero.

use crate::Result;
use crate::storage::TablePages;
use crate::storage::canvas::{CanvasBuffers, ColumnWrites};

/// One canvas sequence of an invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CanvasRow {
    /// Canvas tokens.
    pub(crate) query: usize,
    /// Cached prefix tokens the canvas reads: its request's visible KV.
    pub(crate) prefix: u32,
    /// Position of the canvas's first token.
    pub(crate) position: u64,
}

/// A readout invocation: canvas rows of one or more calls, back to back.
pub(crate) struct ReadoutPass<'a> {
    pub(crate) rows: Vec<CanvasRow>,
    /// Packed canvas tokens of every row.
    pub(crate) tokens: Vec<i64>,
    /// Packed token index of every answer slot, strictly increasing.
    pub(crate) slots: Vec<i64>,
    /// `slots + 1` offsets of each slot's candidates in `candidates`.
    pub(crate) slot_candidates: Vec<usize>,
    /// Candidate token ids, in row, slot and candidate order.
    pub(crate) candidates: Vec<i64>,
    /// The rows' selected pages, one entry per numerical table.
    pub(crate) pages: &'a [TablePages],
}

impl ReadoutPass<'_> {
    /// Tokens of the longest canvas.
    pub(crate) fn longest(&self) -> usize {
        self.rows.iter().map(|row| row.query).max().unwrap_or(0)
    }
}

/// One canvas step: the request slot whose resident canvas it denoises and
/// its sampling coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StepRow {
    pub(crate) slot: u32,
    /// The request's sampling seed, carried as its two's-complement bits.
    pub(crate) seed: u64,
    pub(crate) block: u32,
    pub(crate) step: u32,
    pub(crate) prefix: u32,
    pub(crate) position: u64,
}

/// The host mirror of the attention metadata the runner wrote, as the
/// binding's calls receive it.
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
    buffers: &CanvasBuffers,
    writes: &mut ColumnWrites,
    rows: &[CanvasRow],
    padding: &[usize],
    pages: &[TablePages],
    widths: &[usize],
) -> Result<AttentionMetadata> {
    let layout = buffers.layout();
    let count = rows.len() + padding.len();
    if count > layout.rows {
        return Err(crate::Error::Invalid(format!(
            "a canvas invocation of {count} sequences exceeds its {} sequence rows",
            layout.rows
        )));
    }
    if pages.len() != layout.tables || widths.len() != layout.tables {
        return Err(crate::Error::Invariant(format!(
            "canvas attention reads {} tables, not {}",
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
                    crate::Error::Invalid("canvas sequence offsets exceed int32".into())
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
                "canvas table width {width} exceeds its {} page columns",
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
                    "a canvas row selects {} pages of table {table}, beyond its {width} page columns",
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
    buffers: &CanvasBuffers,
    writes: &mut ColumnWrites,
    rows: &[CanvasRow],
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
