//! Visible cache pages for numerical attention inputs.

use std::ops::Range;
use std::sync::Arc;

use super::{BlockTables, GroupTable};
use crate::{Error, Result};

/// Host lengths and placement of one numerical attention row.
pub(crate) struct AttentionRow {
    pub slot: u32,
    pub prefix: u32,
    pub query: u32,
    pub write: bool,
}

/// The pages one numerical block table selects, in batch row order: each
/// row's first visible page (counted from the table's first page, not from
/// its start) and its number of pages. Device gathers read the physical
/// units from the installed tables.
pub(crate) struct TablePages {
    pub starts: Vec<u32>,
    pub lengths: Vec<usize>,
}

impl GroupTable {
    /// The first query at `prefix` reaches furthest back into the window.
    fn first_page(&self, prefix: u32) -> u32 {
        self.shape.window.map_or(0, |window| {
            prefix.saturating_sub(window) / self.shape.block_size
        })
    }

    fn visible_pages(&self, prefix: u32, query: u32) -> Result<Range<u64>> {
        let first = u64::from(self.first_page(prefix));
        // The table no longer holds pages before its start, so a row
        // reaching back before it has no pages to read.
        if first < u64::from(self.start_page) {
            return Err(Error::Invalid(
                "attention row reads retired window pages".into(),
            ));
        }

        let end = if self.shape.window.is_some() {
            self.end_page().min(
                (u64::from(prefix) + u64::from(query)).div_ceil(u64::from(self.shape.block_size)),
            )
        } else {
            self.end_page()
        };
        Ok(first..end.max(first))
    }
}

/// Select the pages rows of resident tables read: each row's visible page
/// range in every group, for serving and startup forwards alike.
fn table_pages(
    tables: &[Vec<Arc<GroupTable>>],
    prefixes: &[u32],
    queries: &[u32],
) -> Result<Vec<TablePages>> {
    let Some(first) = tables.first() else {
        return Err(Error::Invalid(
            "attention tables require at least one row".into(),
        ));
    };
    if tables.len() != prefixes.len()
        || tables.len() != queries.len()
        || tables.iter().any(|row| row.len() != first.len())
    {
        return Err(Error::Invalid(
            "attention tables and lengths must align by row".into(),
        ));
    }

    let mut selected = Vec::new();
    for (group, table) in first.iter().enumerate() {
        let mut starts = Vec::with_capacity(tables.len());
        let mut lengths = Vec::with_capacity(tables.len());
        for ((row, &prefix), &query) in tables.iter().zip(prefixes).zip(queries) {
            let row_table = &row[group];
            if row_table.shape != table.shape {
                return Err(Error::Invalid(
                    "attention rows must share cache group dimensions".into(),
                ));
            }
            let pages = row_table.visible_pages(prefix, query)?;
            starts.push(pages.start as u32);
            lengths.push((pages.end - pages.start) as usize);
        }

        // A group holds one numerical table per unit of its pages, all
        // selecting the same pages.
        for _ in 0..table.shape.units_per_page {
            selected.push(TablePages {
                starts: starts.clone(),
                lengths: lengths.clone(),
            });
        }
    }
    Ok(selected)
}

impl BlockTables {
    /// Resolve an attention batch over the installed tables. The scheduler
    /// sizes every row within its tables; the caller supplies the cache's
    /// write check over the batch's physical intervals.
    pub(crate) fn prepare_attention(
        &self,
        rows: &[AttentionRow],
        mut require_writable: impl FnMut(&GroupTable, u32, u32) -> Result<()>,
    ) -> Result<Vec<TablePages>> {
        if rows.is_empty() {
            return Err(Error::Invalid(
                "attention metadata requires forward rows".into(),
            ));
        }

        let mut tables = Vec::with_capacity(rows.len());
        let mut prefixes = Vec::with_capacity(rows.len());
        let mut queries = Vec::with_capacity(rows.len());
        for row in rows {
            let mut groups = Vec::with_capacity(self.groups.len());
            for group in 0..self.groups.len() {
                let table = self.table(row.slot, group as u32)?;
                if row.write {
                    require_writable(&table, row.prefix, row.query)?;
                }
                groups.push(table);
            }
            tables.push(groups);
            prefixes.push(row.prefix);
            queries.push(row.query);
        }
        table_pages(&tables, &prefixes, &queries)
    }
}

#[cfg(test)]
mod tests {
    use super::super::GroupShape;
    use super::*;
    use ipc::UnitId;

    #[test]
    fn window_pages_split_layer_units_and_keep_full_history() -> Result<()> {
        let window = Arc::new(GroupTable {
            shape: GroupShape::new(4, 2, Some(4))?,
            start_page: 1,
            units: (10..18).map(UnitId).collect(),
            allocated_tokens: 20,
        });
        let full = Arc::new(GroupTable {
            shape: GroupShape::new(8, 1, None)?,
            start_page: 0,
            units: vec![UnitId(21), UnitId(22), UnitId(23)],
            allocated_tokens: 20,
        });
        let tables = vec![
            vec![Arc::clone(&window), Arc::clone(&full)],
            vec![Arc::clone(&window), Arc::clone(&full)],
        ];

        // Token 8 reaches page 1; token 12 reaches page 2. The latter row
        // extends beyond the stored prefix, as a read-only block may do.
        // The window group's two units per page are two numerical tables.
        let pages = table_pages(&tables, &[8, 12], &[1, 16])?;
        assert_eq!(pages.len(), 3);
        for window in &pages[..2] {
            assert_eq!(window.starts, [1, 2]);
            assert_eq!(window.lengths, [2, 3]);
        }
        assert_eq!(pages[2].starts, [0, 0]);
        assert_eq!(pages[2].lengths, [3, 3]);

        // A retired history page cannot be reconstructed from later units.
        assert!(table_pages(&tables, &[7, 12], &[1, 1]).is_err());
        Ok(())
    }
}
