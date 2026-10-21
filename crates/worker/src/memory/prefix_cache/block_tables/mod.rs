//! Scheduler-assigned cache pages and their numerical block-table updates.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use ipc::UnitId;
use ipc::{BlockTable, CacheTransfer, RequestKey};

use crate::{Error, Result};

mod attention;
mod backing;

pub(crate) use attention::{AttentionRow, TablePages};
pub use backing::BlockTableStorage;

/// Page dimensions needed to address one cache group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupShape {
    pub block_size: u32,
    pub units_per_page: u32,
    pub window: Option<u32>,
}

impl GroupShape {
    /// Installed tables retain the complete reservation, including history
    /// that a sliding-window reader no longer selects.
    pub(crate) fn resident_width(&self, max_sequence_tokens: u64) -> u64 {
        max_sequence_tokens
            .max(1)
            .div_ceil(u64::from(self.block_size))
    }

    /// A windowed call selects its history and queries plus the partial page
    /// at their starting offset. Full attention selects the whole sequence.
    pub(crate) fn table_width(&self, max_sequence_tokens: u64, max_query_tokens: u64) -> u64 {
        let resident = self.resident_width(max_sequence_tokens);
        self.window.map_or(resident, |window| {
            resident.min(
                (u64::from(window) + max_query_tokens.max(1)).div_ceil(u64::from(self.block_size))
                    + 1,
            )
        })
    }

    pub fn new(block_size: u32, units_per_page: u32, window: Option<u32>) -> Result<Self> {
        if block_size == 0 || units_per_page == 0 {
            return Err(Error::Invalid(
                "cache page dimensions must be positive".into(),
            ));
        }

        Ok(Self {
            block_size,
            units_per_page,
            window,
        })
    }
}

/// One slot's pages in one cache group, with page-major physical unit IDs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GroupTable {
    pub shape: GroupShape,
    pub start_page: u32,
    pub units: Vec<UnitId>,
    pub allocated_tokens: u32,
}

impl GroupTable {
    pub(crate) fn end_page(&self) -> u64 {
        u64::from(self.start_page) + self.units.len() as u64 / u64::from(self.shape.units_per_page)
    }

    /// The physical units at one position in each logical page.
    pub(crate) fn row(&self, position: usize) -> Vec<u32> {
        self.row_units(position).collect()
    }

    /// The physical units at one position in each logical page, in page
    /// order.
    pub(crate) fn row_units(&self, position: usize) -> impl Iterator<Item = u32> + '_ {
        self.units
            .iter()
            .skip(position)
            .step_by(self.shape.units_per_page as usize)
            .map(|unit| unit.0)
    }

    /// The logical pages the table holds.
    pub(crate) fn pages(&self) -> usize {
        self.units.len() / self.shape.units_per_page as usize
    }

    /// Map absolute tokens to `(unit, token offset, token count)` intervals.
    /// Every unit in a page holds these tokens for its own subset of layers.
    pub(crate) fn spans(&self, mut start: u64, mut length: u64) -> Result<Vec<(u32, u32, u32)>> {
        let tokens = u64::from(self.shape.block_size);
        let units = self.shape.units_per_page as usize;
        let end = start.checked_add(length);
        if (length > 0 && start < u64::from(self.start_page) * tokens)
            || end.is_none_or(|end| end > self.end_page() * tokens)
        {
            return Err(Error::Invalid(
                "cache token interval exceeds its unit table".into(),
            ));
        }

        let mut spans = Vec::new();
        while length > 0 {
            let page = start / tokens;
            let offset = start % tokens;
            let count = length.min(tokens - offset);
            let first = (page - u64::from(self.start_page)) as usize * units;
            spans.extend(
                self.units[first..first + units]
                    .iter()
                    .map(|unit| (unit.0, offset as u32, count as u32)),
            );
            start += count;
            length -= count;
        }

        Ok(spans)
    }
}

/// Changed rows and lengths to write to the device tables in one batch.
/// Rows are read from the accepted tables; [`BlockTableStorage::install`]
/// writes them at full width, which clears the tail of a longer previous
/// row.
pub struct BlockTableUpdate {
    rows: Vec<RowUpdate>,
    /// Changed first pages: group, slot and page.
    starts: Vec<(u32, u32, u32)>,
    /// Changed visible lengths: slot and token extent.
    extents: Vec<(u32, u32)>,
    /// The accepted tables by slot and group.
    tables: Vec<((u32, u32), Arc<GroupTable>)>,
    lengths: BTreeMap<u32, u32>,
}

/// One changed device row: the units at `position` of every page of the
/// accepted table `tables[table]`, written to device table `target`.
struct RowUpdate {
    table: usize,
    position: usize,
    target: usize,
}

/// Host ownership of the device block tables, including alternative prefixes.
/// Slot and physical unit zero are reserved for padding. The scheduler owns
/// allocation; this owner only installs complete assignments and releases them.
pub struct BlockTables {
    groups: Vec<GroupShape>,
    first_table: Vec<usize>,
    request_pool_size: u32,
    width: usize,
    num_units: u32,
    tables: HashMap<(u32, u32), Arc<GroupTable>>,
    lengths: HashMap<u32, u32>,
    prefixes: HashMap<RequestKey, BTreeSet<u32>>,
}

impl BlockTables {
    pub fn new(
        groups: Vec<GroupShape>,
        request_pool_size: u32,
        width: usize,
        num_units: u32,
    ) -> Result<Self> {
        if groups.is_empty() || request_pool_size == 0 || width == 0 || num_units == 0 {
            return Err(Error::Invalid(
                "request-to-token pool dimensions are invalid".into(),
            ));
        }

        let mut first_table = Vec::with_capacity(groups.len());
        let mut count = 0;
        for group in &groups {
            first_table.push(count);
            count += group.units_per_page as usize;
        }

        Ok(Self {
            groups,
            first_table,
            request_pool_size,
            width,
            num_units,
            tables: HashMap::new(),
            lengths: HashMap::new(),
            prefixes: HashMap::new(),
        })
    }

    pub(crate) fn groups(&self) -> &[GroupShape] {
        &self.groups
    }

    /// Resolve physical writes before an import starts. Existing prefix pages
    /// keep their addresses and initialization; only newly assigned units reset.
    /// This does not install the tables or change their visible lengths.
    pub(crate) fn import_spans(
        &self,
        slot: u32,
        tables: &[Arc<GroupTable>],
        transfer: &CacheTransfer,
        initialized: &[u32],
    ) -> Result<Vec<(u32, u32, u32)>> {
        if slot == 0 || slot > self.request_pool_size || tables.len() != self.groups.len() {
            return Err(Error::Invalid(
                "cache import requires a destination table per group".into(),
            ));
        }
        if !transfer.groups.is_empty() && transfer.groups.len() != tables.len() {
            return Err(Error::Invalid(
                "cache transfer groups do not match destination groups".into(),
            ));
        }

        let reset: HashSet<_> = initialized.iter().copied().collect();
        let mut spans = Vec::new();
        for (group, table) in tables.iter().enumerate() {
            self.validate_table(group, table)?;
            let block_size = u64::from(table.shape.block_size);
            if table.allocated_tokens < transfer.exported_extent
                || u64::from(table.start_page) * block_size > u64::from(transfer.exported_extent)
            {
                return Err(Error::Invalid(
                    "cache import exceeds its scheduler block table".into(),
                ));
            }

            if transfer.base_extent > 0 {
                let installed = self.table(slot, group as u32)?;
                let first = table.start_page.max(installed.start_page) as u64;
                let last = u64::from(transfer.base_extent)
                    .div_ceil(block_size)
                    .min(table.end_page())
                    .min(installed.end_page());
                let per_page = table.shape.units_per_page as usize;
                for page in first..last {
                    let new = (page - u64::from(table.start_page)) as usize * per_page;
                    let old = (page - u64::from(installed.start_page)) as usize * per_page;
                    let units = &table.units[new..new + per_page];
                    if units != &installed.units[old..old + per_page]
                        || units.iter().any(|unit| reset.contains(&unit.0))
                    {
                        return Err(Error::Invalid(
                            "cache import would replace its installed base units".into(),
                        ));
                    }
                }
            }

            if let Some(source) = transfer.groups.get(group) {
                let start = table.shape.window.map_or(transfer.base_extent, |window| {
                    transfer
                        .base_extent
                        .max(transfer.exported_extent.saturating_sub(window))
                });
                if source.start != start || start > transfer.exported_extent {
                    return Err(Error::Invalid(
                        "cache transfer starts outside its destination history".into(),
                    ));
                }
                spans.extend(table.spans(
                    u64::from(start),
                    u64::from(transfer.exported_extent - start),
                )?);
            }
        }

        // A physical unit can serve any group. Reset reserves its full token
        // range, while the access owner coalesces it with the copied suffix.
        let unit_tokens = self
            .groups
            .iter()
            .map(|group| group.block_size)
            .fold(0, u32::max);
        spans.extend(initialized.iter().map(|&unit| (unit, 0, unit_tokens)));
        Ok(spans)
    }

    /// Check a table against the rank's own table storage: its group's
    /// shape, whole pages within the row width, allocated tokens within its
    /// pages, and units inside the pool. The scheduler's allocator owns
    /// which units a table names (never the sentinel, never twice); the pool
    /// bound stays here because kernels address the planes by these ids, so
    /// a unit beyond the pool would read and write outside its allocation.
    fn validate_table(&self, group: usize, table: &GroupTable) -> Result<()> {
        let per_page = table.shape.units_per_page as usize;
        if self.groups.get(group) != Some(&table.shape)
            || !table.units.len().is_multiple_of(per_page)
            || table.units.len() / per_page > self.width
            || table.units.iter().any(|unit| unit.0 >= self.num_units)
            || u64::from(table.allocated_tokens)
                > table.end_page() * u64::from(table.shape.block_size)
        {
            return Err(Error::Invalid("scheduler block table is invalid".into()));
        }
        Ok(())
    }

    /// Borrow a slot's tables before a batch installs its assignments.
    /// Supplied groups replace resident groups only in this returned view.
    pub fn for_batch(&self, slot: u32, assignments: &[BlockTable]) -> Result<Vec<Arc<GroupTable>>> {
        self.groups
            .iter()
            .enumerate()
            .map(|(group, &shape)| {
                let group = group as u32;
                // The last assignment of a group is the one that counts.
                let supplied = assignments
                    .iter()
                    .rev()
                    .find(|table| table.request_pool_idx == slot && table.group_id == group);
                match supplied {
                    Some(table) => Ok(Arc::new(GroupTable {
                        shape,
                        start_page: table.start_page,
                        units: table.unit_ids.clone(),
                        allocated_tokens: table.allocated_tokens,
                    })),
                    None => self.table(slot, group),
                }
            })
            .collect()
    }

    /// Validate the whole installation before computing any device changes.
    /// After the device copy succeeds, commit on the same owner thread.
    /// A failed copy discards the update and leaves the host
    /// tables unchanged. No other update may intervene before commit.
    pub fn prepare(&self, tables: &[BlockTable]) -> Result<BlockTableUpdate> {
        let mut accepted = Vec::with_capacity(tables.len());
        let mut seen = HashSet::new();

        for value in tables {
            let slot = value.request_pool_idx;
            let group = value.group_id;
            if slot == 0 || slot > self.request_pool_size {
                return Err(Error::Invalid("scheduler block table is invalid".into()));
            }
            let shape = self
                .groups
                .get(group as usize)
                .copied()
                .ok_or_else(|| Error::Invalid("scheduler block table is invalid".into()))?;

            let table = GroupTable {
                shape,
                start_page: value.start_page,
                units: value.unit_ids.clone(),
                allocated_tokens: value.allocated_tokens,
            };
            self.validate_table(group as usize, &table)?;
            if !seen.insert((slot, group)) {
                return Err(Error::Invalid("scheduler block table is invalid".into()));
            }

            accepted.push(((slot, group), Arc::new(table)));
        }

        let mut update = BlockTableUpdate {
            rows: Vec::new(),
            starts: Vec::new(),
            extents: Vec::new(),
            tables: accepted,
            lengths: BTreeMap::new(),
        };
        for (index, (key, table)) in update.tables.iter().enumerate() {
            let (slot, group) = *key;
            let previous = self.tables.get(&(slot, group));
            if previous.is_none_or(|previous| previous.units != table.units) {
                for position in 0..table.shape.units_per_page as usize {
                    update.rows.push(RowUpdate {
                        table: index,
                        position,
                        target: self.first_table[group as usize] + position,
                    });
                }
            }
            if previous.is_none_or(|previous| previous.start_page != table.start_page) {
                update.starts.push((group, slot, table.start_page));
            }

            update
                .lengths
                .entry(slot)
                .and_modify(|length| {
                    *length = (*length).min(table.allocated_tokens);
                })
                .or_insert(table.allocated_tokens);
        }

        // Unchanged groups still constrain the slot's available token extent.
        for (&slot, length) in &mut update.lengths {
            for group in 0..self.groups.len() as u32 {
                let key = (slot, group);
                if !seen.contains(&key)
                    && let Some(table) = self.tables.get(&key)
                {
                    *length = (*length).min(table.allocated_tokens);
                }
            }
        }
        for (&slot, &length) in &update.lengths {
            if self.lengths.get(&slot) != Some(&length) {
                update.extents.push((slot, length));
            }
        }

        Ok(update)
    }

    pub fn commit(&mut self, update: BlockTableUpdate) {
        self.tables.extend(update.tables);
        self.lengths.extend(update.lengths);
    }

    pub(crate) fn table(&self, slot: u32, group: u32) -> Result<Arc<GroupTable>> {
        self.tables
            .get(&(slot, group))
            .cloned()
            .ok_or_else(|| Error::Invalid("request slot has no installed block table".into()))
    }

    pub(crate) fn allocated_length(&self, slot: u32) -> u32 {
        self.lengths.get(&slot).copied().unwrap_or(0)
    }

    /// Resolve one request's visible prefix against every installed cache group.
    /// The returned capacity is the common extent, including sliding windows.
    pub(crate) fn coordinates(&self, slot: u32, visible: u64) -> Result<(u32, u64, u32)> {
        for group in 0..self.groups.len() as u32 {
            if !self.tables.contains_key(&(slot, group)) {
                return Err(Error::Invalid(
                    "request slot has no installed block table".into(),
                ));
            }
        }

        let capacity = self.allocated_length(slot);
        if visible > u64::from(capacity) {
            return Err(Error::Invalid(
                "call visibility exceeds scheduler block table".into(),
            ));
        }
        Ok((slot, visible, capacity))
    }

    /// Validate releases together so an invalid slot cannot partially clear
    /// device or host rows. Duplicate slots require only one device write.
    pub(crate) fn release_slots(&self, slots: &[u32]) -> Result<Vec<u32>> {
        if slots
            .iter()
            .any(|&slot| slot == 0 || slot > self.request_pool_size)
        {
            return Err(Error::Invalid(
                "released request slot is outside capacity".into(),
            ));
        }

        Ok(slots
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    /// Commit a release after its device clears have been submitted.
    pub fn release(&mut self, slots: &[u32]) {
        for &slot in slots {
            for group in 0..self.groups.len() as u32 {
                self.tables.remove(&(slot, group));
            }
            self.lengths.remove(&slot);
        }
    }

    pub(crate) fn retain_prefix(&mut self, request: RequestKey, slot: u32) {
        self.prefixes.entry(request).or_default().insert(slot);
    }

    pub(crate) fn prefix_slots(&self, request: RequestKey) -> Vec<u32> {
        self.prefixes
            .get(&request)
            .map_or_else(Vec::new, |slots| slots.iter().copied().collect())
    }

    pub(crate) fn release_prefixes(&mut self, request: RequestKey, slots: &[u32]) {
        self.release(slots);
        if let Some(tracked) = self.prefixes.get_mut(&request) {
            for slot in slots {
                tracked.remove(slot);
            }
            if tracked.is_empty() {
                self.prefixes.remove(&request);
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.tables.clear();
        self.lengths.clear();
        self.prefixes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipc::RequestId;

    #[test]
    fn visible_prefix_requires_every_group_and_fits_the_common_extent() -> Result<()> {
        let groups = vec![
            GroupShape::new(4, 1, None)?,
            GroupShape::new(8, 2, Some(8))?,
        ];
        let mut tables = BlockTables::new(groups, 1, 3, 6)?;
        let update = tables.prepare(&[BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 0,
            unit_ids: vec![UnitId(1), UnitId(2), UnitId(3)],
            allocated_tokens: 12,
        }])?;
        tables.commit(update);
        assert!(tables.coordinates(1, 8).is_err());

        let update = tables.prepare(&[BlockTable {
            request_pool_idx: 1,
            group_id: 1,
            start_page: 0,
            unit_ids: vec![UnitId(4), UnitId(5)],
            allocated_tokens: 8,
        }])?;
        tables.commit(update);
        assert_eq!(tables.coordinates(1, 8)?, (1, 8, 8));
        assert!(tables.coordinates(1, 9).is_err());

        tables.release(&[1]);
        assert!(tables.coordinates(1, 0).is_err());
        Ok(())
    }

    #[test]
    fn batch_assignments_replace_only_supplied_groups_until_commit() -> Result<()> {
        let groups = vec![
            GroupShape::new(4, 1, None)?,
            GroupShape::new(8, 2, Some(8))?,
        ];
        let mut tables = BlockTables::new(groups, 1, 2, 7)?;
        let first = BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 0,
            unit_ids: vec![UnitId(1), UnitId(2)],
            allocated_tokens: 8,
        };
        assert!(tables.for_batch(1, std::slice::from_ref(&first)).is_err());

        let update = tables.prepare(&[
            first,
            BlockTable {
                request_pool_idx: 1,
                group_id: 1,
                start_page: 0,
                unit_ids: vec![UnitId(3), UnitId(4)],
                allocated_tokens: 8,
            },
        ])?;
        tables.commit(update);

        let assignments = [BlockTable {
            request_pool_idx: 1,
            group_id: 1,
            start_page: 1,
            unit_ids: vec![UnitId(5), UnitId(6)],
            allocated_tokens: 16,
        }];
        let batch = tables.for_batch(1, &assignments)?;
        assert_eq!(batch[0].spans(2, 4)?, vec![(1, 2, 2), (2, 0, 2)]);
        assert_eq!(batch[1].spans(10, 4)?, vec![(5, 2, 4), (6, 2, 4)]);
        assert_eq!(tables.table(1, 1)?.spans(2, 4)?, vec![(3, 2, 4), (4, 2, 4)]);

        let update = tables.prepare(&assignments)?;
        tables.commit(update);
        assert_eq!(tables.table(1, 1)?.spans(10, 4)?, batch[1].spans(10, 4)?);
        Ok(())
    }

    #[test]
    fn token_spans_cover_each_layers_unit_without_crossing_retired_pages() -> Result<()> {
        let table = GroupTable {
            shape: GroupShape::new(8, 2, Some(8))?,
            start_page: 3,
            units: vec![UnitId(2), UnitId(3), UnitId(4), UnitId(5)],
            allocated_tokens: 40,
        };
        assert_eq!(table.row(0), vec![2, 4]);
        assert_eq!(table.row(1), vec![3, 5]);
        assert_eq!(
            table.spans(30, 4)?,
            vec![(2, 6, 2), (3, 6, 2), (4, 0, 2), (5, 0, 2)]
        );
        assert!(table.spans(23, 1).is_err());
        assert!(table.spans(39, 2).is_err());
        assert!(table.spans(40, 0)?.is_empty());
        Ok(())
    }

    #[test]
    fn retained_prefixes_follow_the_request_epoch_and_preserve_borrowed_tables() -> Result<()> {
        let shape = GroupShape::new(4, 1, None)?;
        let mut tables = BlockTables::new(vec![shape], 3, 2, 9)?;
        let update = tables.prepare(&[BlockTable {
            request_pool_idx: 2,
            group_id: 0,
            start_page: 0,
            unit_ids: vec![UnitId(7)],
            allocated_tokens: 4,
        }])?;
        tables.commit(update);
        let borrowed = tables.table(2, 0)?;
        let request = RequestKey::new(1, RequestId(4), 1);
        let replacement = RequestKey::new(1, RequestId(4), 2);
        tables.retain_prefix(request, 2);
        tables.retain_prefix(request, 3);
        assert!(tables.prefix_slots(replacement).is_empty());

        let slots = tables.release_slots(&[2, 2])?;
        tables.release_prefixes(request, &slots);
        assert_eq!(tables.prefix_slots(request), vec![3]);
        assert_eq!(tables.allocated_length(2), 0);
        assert!(tables.table(2, 0).is_err());
        assert_eq!(borrowed.units, vec![UnitId(7)]);

        let update = tables.prepare(&[BlockTable {
            request_pool_idx: 2,
            group_id: 0,
            start_page: 1,
            unit_ids: vec![UnitId(8)],
            allocated_tokens: 8,
        }])?;
        tables.commit(update);
        assert_eq!(tables.table(2, 0)?.units, vec![UnitId(8)]);
        assert_eq!(borrowed.units, vec![UnitId(7)]);
        Ok(())
    }
}
