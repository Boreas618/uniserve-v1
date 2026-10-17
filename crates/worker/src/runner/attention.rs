//! Attention metadata of a runner that attends to the prefix cache.
//!
//! A runner reading the prefix cache through block tables (a causal LM, an
//! image denoiser) gathers each forward's attention metadata on the device
//! from one packed host column array (`row_columns`), sized for `R` rows and
//! `T` tokens over `Tb` numerical tables. Per forward the runner selects
//! each row's visible pages against the installed block tables and packs:
//!
//! | Section | Words |
//! | --- | --- |
//! | slots, prefixes, queries, writes | `R + 1` each |
//! | query offsets, prefix offsets | `R + 1` each (exclusive prefix sums) |
//! | each table's selected page count | `R + 1` each |
//! | each token's row | `T` |
//!
//! The Python half's [`AttentionMetadata`](`uniserve.runner.attention`)
//! derives the device block tables, lengths, offsets and write indices from
//! it and the slots' installed tables in one launch
//! (`uniserve.nn.attention.gather`), and owns those device tensors.

use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream};

use crate::ffi::{Backing, DType};
use crate::storage::PrefixCache;
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{AttentionRow, TablePages};
use crate::stream::Issue;
use crate::{Error, Result};

/// Host sections of `row_columns` before the per-table page counts.
pub(crate) const ROW_SECTIONS: usize = 6;

/// The capacity of a runner's attention metadata: rows, tokens, numerical
/// tables and the widest page width a row selects from one table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionCapacity {
    pub rows: usize,
    pub tokens: usize,
    pub tables: usize,
    pub width: usize,
}

impl AttentionCapacity {
    /// Words of the packed host columns.
    pub(crate) fn row_words(&self) -> usize {
        (ROW_SECTIONS + self.tables) * (self.rows + 1) + self.tokens
    }
}

/// One runner's attention metadata on the host: the packed row column its
/// Python half gathers from, and the host mirrors of the last staged
/// forward's rows, which an eager forward and a capture read.
pub(crate) struct AttentionMetadata {
    pub capacity: AttentionCapacity,
    /// `int64 [row_words]` on the runner's device.
    row_columns: StorageView,
    /// Rows and tokens of the last staged forward.
    pub rows: usize,
    pub tokens: usize,
    /// Each row's query and prefix length.
    pub queries: Vec<i64>,
    pub prefixes: Vec<i64>,
    /// Each windowed table's first selected pages, `rows` per table.
    pub starts: Vec<i64>,
    /// Each table's page width.
    pub widths: Vec<i64>,
}

impl AttentionMetadata {
    /// Allocate the packed row column on `device` (host memory when
    /// `None`), zeroed on `stream`.
    pub(crate) fn allocate(
        capacity: AttentionCapacity,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let words = capacity.row_words();
        let bytes = words * DType::I64.bytes();
        let memory: Arc<dyn Backing> = match device {
            Some(device) => Memory::device(device, bytes)?,
            None => Memory::host(bytes),
        };
        let row_columns = StorageView::contiguous(memory, 0, DType::I64, &[words as i64])?;
        row_columns.zero(stream)?;
        Ok(Self {
            capacity,
            row_columns,
            rows: 0,
            tokens: 0,
            queries: Vec::new(),
            prefixes: Vec::new(),
            starts: Vec::new(),
            widths: Vec::new(),
        })
    }

    pub(crate) fn row_columns(&self) -> &StorageView {
        &self.row_columns
    }

    /// Select `rows`' pages of `prefix_cache` (every write checked against
    /// its live accesses), pack the host column and copy it to the device on
    /// `stream`, and keep the forward's host mirrors. `widths` fixes each
    /// table's page width (a captured graph's configured widths); `None`
    /// uses the rows' own.
    pub(crate) fn stage(
        &mut self,
        rows: &[AttentionRow],
        prefix_cache: &mut PrefixCache,
        widths: Option<&[usize]>,
        stream: impl Issue,
    ) -> Result<()> {
        let capacity = self.capacity;
        let tokens: usize = rows.iter().map(|row| row.query as usize).sum();
        if rows.is_empty() || rows.len() > capacity.rows || tokens > capacity.tokens {
            return Err(Error::Invalid(format!(
                "{} attention rows of {tokens} tokens exceed {} rows of {} tokens",
                rows.len(),
                capacity.rows,
                capacity.tokens
            )));
        }
        let pages = prefix_cache.prepare_attention(rows)?;
        if pages.len() != capacity.tables {
            return Err(Error::Invariant(
                "attention pages disagree with the runner's tables".into(),
            ));
        }
        let widths = pages
            .iter()
            .enumerate()
            .map(|(table, selected)| {
                let width = match widths {
                    Some(widths) => {
                        let configured = widths[table];
                        if selected.width() > configured {
                            return Err(Error::Invalid(format!(
                                "table {table} selects {} pages; its graph width is {configured}",
                                selected.width()
                            )));
                        }
                        configured
                    }
                    None => selected.width(),
                };
                if width > capacity.width {
                    return Err(Error::Invalid(format!(
                        "table {table} selects {width} pages; the runner holds {}",
                        capacity.width
                    )));
                }
                Ok(width as i64)
            })
            .collect::<Result<Vec<_>>>()?;
        let words = pack_rows(rows, &pages, capacity.rows + 1, capacity.tokens);
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        self.row_columns.upload(&bytes, stream)?;

        let mut starts = vec![0; pages.len() * rows.len()];
        for (table, selected) in pages.iter().enumerate() {
            for (row, &start) in selected.starts.iter().enumerate() {
                starts[table * rows.len() + row] = i64::from(start);
            }
        }
        self.rows = rows.len();
        self.tokens = tokens;
        self.queries = rows.iter().map(|row| i64::from(row.query)).collect();
        self.prefixes = rows.iter().map(|row| i64::from(row.prefix)).collect();
        self.starts = starts;
        self.widths = widths;
        Ok(())
    }
}

/// Pack rows into the `row_columns` layout (see the module table), with
/// sections `stride` words apart and room for `tokens` token rows.
pub(crate) fn pack_rows(
    rows: &[AttentionRow],
    pages: &[TablePages],
    stride: usize,
    tokens: usize,
) -> Vec<i64> {
    let mut columns = vec![0_i64; (ROW_SECTIONS + pages.len()) * stride + tokens];
    let token_section = (ROW_SECTIONS + pages.len()) * stride;
    let (mut query_offset, mut prefix_offset) = (0_i64, 0_i64);
    for (index, row) in rows.iter().enumerate() {
        columns[index] = i64::from(row.slot);
        columns[stride + index] = i64::from(row.prefix);
        columns[2 * stride + index] = i64::from(row.query);
        columns[3 * stride + index] = i64::from(row.write);
        columns[4 * stride + index] = query_offset;
        columns[5 * stride + index] = prefix_offset;
        for (table, selected) in pages.iter().enumerate() {
            columns[(ROW_SECTIONS + table) * stride + index] =
                selected.lengths.get(index).copied().unwrap_or(0) as i64;
        }
        for token in 0..row.query as usize {
            columns[token_section + query_offset as usize + token] = index as i64;
        }
        query_offset += i64::from(row.query);
        prefix_offset += i64::from(row.prefix);
    }
    columns[4 * stride + rows.len()] = query_offset;
    columns[5 * stride + rows.len()] = prefix_offset;
    columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_rows_hold_lengths_offsets_and_token_rows() {
        let rows = [
            AttentionRow {
                slot: 3,
                prefix: 10,
                query: 2,
                write: false,
            },
            AttentionRow {
                slot: 5,
                prefix: 4,
                query: 3,
                write: true,
            },
        ];
        // Three rows of capacity (stride 4), five tokens, no tables.
        let words = pack_rows(&rows, &[], 4, 5);
        assert_eq!(&words[0..4], &[3, 5, 0, 0]); // slots
        assert_eq!(&words[4..8], &[10, 4, 0, 0]); // prefixes
        assert_eq!(&words[8..12], &[2, 3, 0, 0]); // queries
        assert_eq!(&words[12..16], &[0, 1, 0, 0]); // writes
        assert_eq!(&words[16..20], &[0, 2, 5, 0]); // query offsets
        assert_eq!(&words[20..24], &[0, 10, 14, 0]); // prefix offsets
        assert_eq!(&words[24..29], &[0, 0, 1, 1, 1]); // token rows
    }
}
