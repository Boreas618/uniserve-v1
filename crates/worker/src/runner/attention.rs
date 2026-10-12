//! Paged attention columns of a binding that attends to the KV cache.
//!
//! A binding reading the KV cache through block tables (a causal LM, an
//! image denoiser) owns one set of fixed device columns sized for `R` rows
//! and `T` tokens over `Tb` numerical tables of width `W`. Per invocation
//! the runner selects each row's visible pages against the installed block
//! tables and packs one host column array (`row_columns`):
//!
//! | Section | Words |
//! | --- | --- |
//! | slots, prefixes, queries, writes | `R + 1` each |
//! | query offsets, prefix offsets | `R + 1` each (exclusive prefix sums) |
//! | each table's selected page count | `R + 1` each |
//! | each token's row | `T` |
//!
//! The binding's gather (`uniserve_models.nn.attention.gather`) derives the
//! device block tables, lengths, offsets and write indices from it and the
//! slots' installed tables in one launch
//! (`uniserve_models.interface.attention`).

use std::sync::Arc;

use uniserve_cuda::Stream;

use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{AttentionRow, BlockTables, GroupTable, TablePages};
use crate::stream::Issue;
use crate::{Error, Result};

/// Host sections of `row_columns` before the per-table page counts.
pub const ROW_SECTIONS: usize = 6;

/// The capacity of an attention column set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionCapacity {
    pub rows: usize,
    pub tokens: usize,
    pub tables: usize,
    pub width: usize,
}

impl AttentionCapacity {
    /// Words of the packed host columns.
    pub fn row_words(&self) -> usize {
        (ROW_SECTIONS + self.tables) * (self.rows + 1) + self.tokens
    }

    fn columns(&self) -> Vec<(&'static str, DType, Vec<i64>)> {
        let (rows, tokens, tables) = (self.rows as i64, self.tokens as i64, self.tables as i64);
        vec![
            ("row_columns", DType::I64, vec![self.row_words() as i64]),
            (
                "block_tables",
                DType::I32,
                vec![tables, rows, self.width as i64],
            ),
            ("start_pages", DType::I32, vec![tables, rows]),
            ("cache_lengths", DType::I32, vec![rows]),
            ("query_lengths", DType::I32, vec![rows]),
            ("causal_values", DType::I32, vec![rows]),
            ("query_offsets", DType::I32, vec![rows + 1]),
            ("prefix_offsets", DType::I32, vec![rows + 1]),
            ("write_indices", DType::I64, vec![tables, tokens]),
        ]
    }
}

/// The host geometry of one invocation's attention rows, passed to the
/// binding beside the packed columns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub rows: usize,
    pub tokens: usize,
    pub queries: Vec<i64>,
    pub prefixes: Vec<i64>,
    /// Each windowed table's first pages, `rows` per table.
    pub starts: Vec<i64>,
    pub widths: Vec<i64>,
}

impl AttentionGeometry {
    /// `[rows, tokens, queries, prefixes, starts, widths]`.
    pub fn to_any(&self) -> tvm_ffi::Any {
        let list =
            |values: &[i64]| tvm_ffi::Any::from(crate::ffi::integer_list(values.iter().copied()));
        tvm_ffi::Any::from(crate::ffi::list([
            tvm_ffi::Any::from(self.rows as i64),
            tvm_ffi::Any::from(self.tokens as i64),
            list(&self.queries),
            list(&self.prefixes),
            list(&self.starts),
            list(&self.widths),
        ]))
    }
}

/// One binding's attention columns.
pub struct AttentionColumns {
    pub capacity: AttentionCapacity,
    columns: Vec<(&'static str, StorageView)>,
}

impl AttentionColumns {
    /// Allocate the columns on `device` (host memory when `None`), zeroed
    /// and with every write index -1 (no cache write), on `stream`.
    pub fn allocate(
        capacity: AttentionCapacity,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let mut columns = Vec::new();
        for (name, dtype, shape) in capacity.columns() {
            let bytes = shape.iter().product::<i64>().max(1) as usize * dtype.bytes();
            let memory: Arc<dyn Backing> = match device {
                Some(device) => Memory::device(device, bytes)?,
                None => Memory::host(bytes),
            };
            let view = StorageView::contiguous(memory, 0, dtype, &shape)?;
            view.zero(stream)?;
            columns.push((name, view));
        }
        let columns = Self { capacity, columns };
        let writes = columns.column("write_indices")?;
        let inert = vec![0xff_u8; writes.nbytes() as usize];
        writes.upload(&inert, stream)?;
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(columns)
    }

    pub fn column(&self, name: &str) -> Result<&StorageView> {
        self.columns
            .iter()
            .find(|(column, _)| *column == name)
            .map(|(_, view)| view)
            .ok_or_else(|| Error::Invariant(format!("attention columns have no {name}")))
    }

    /// Select `rows`' pages, pack the host columns and copy them to the
    /// device on `stream`. `widths` fixes each table's page width (a
    /// captured graph's configured widths); `None` uses the rows' own.
    /// `check` validates each table's writes against the cache's live
    /// accesses.
    pub fn stage(
        &self,
        rows: &[AttentionRow],
        tables: &BlockTables,
        widths: Option<&[usize]>,
        check: impl FnMut(&GroupTable, u32, u32) -> Result<()>,
        stream: impl Issue,
    ) -> Result<AttentionGeometry> {
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
        let pages = tables.prepare_attention(rows, check)?;
        if pages.len() != capacity.tables {
            return Err(Error::Invariant(
                "attention pages disagree with the binding's tables".into(),
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
                        "table {table} selects {width} pages; the binding holds {}",
                        capacity.width
                    )));
                }
                Ok(width as i64)
            })
            .collect::<Result<Vec<_>>>()?;
        let words = pack_rows(rows, &pages, capacity.rows + 1, capacity.tokens);
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        self.column("row_columns")?.upload(&bytes, stream)?;

        let mut starts = vec![0; pages.len() * rows.len()];
        for (table, selected) in pages.iter().enumerate() {
            for (row, &start) in selected.starts.iter().enumerate() {
                starts[table * rows.len() + row] = i64::from(start);
            }
        }
        Ok(AttentionGeometry {
            rows: rows.len(),
            tokens,
            queries: rows.iter().map(|row| i64::from(row.query)).collect(),
            prefixes: rows.iter().map(|row| i64::from(row.prefix)).collect(),
            starts,
            widths,
        })
    }
}

/// Pack rows into the `row_columns` layout (see the module table), with
/// sections `stride` words apart and room for `tokens` token rows.
pub fn pack_rows(
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
