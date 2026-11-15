//! Runs of units and the regions they are taken from.
//!
//! A [`Run`] is the only thing that stands for units: it is made when a
//! region is created, it is split and joined, and it is never cloned, so a
//! unit is in one run at a time and a run is in one place at a time: a
//! region's free list, the runs outside the region, or one block. A pool
//! made again from its image ([`super::image`]) makes its runs again from
//! the image's, and the pool the image was taken of is not used afterwards.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use storage::protocol::Key;
use storage::runtime::Tier;

/// A contiguous range of units of one region.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Run {
    region: u8,
    first: u32,
    count: u32,
}

impl Run {
    pub(crate) fn region(&self) -> u8 {
        self.region
    }

    pub(crate) fn first(&self) -> u32 {
        self.first
    }

    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The run as a completion states it.
    pub(crate) fn stated(&self) -> storage::runtime::Run {
        (self.region, self.first, self.count)
    }

    /// Splits off the first `count` units; the rest stays in `self`.
    fn split(&mut self, count: u32) -> Run {
        debug_assert!(count < self.count);
        let head = Run {
            region: self.region,
            first: self.first,
            count,
        };
        self.first += count;
        self.count -= count;
        head
    }
}

/// The memory blocks are taken from, on one tier and in one layout.
///
/// A region's units are numbered from 0 up to a fixed limit. Each is in the
/// free list, in a block, or outside the region: given back to the
/// deployment's quota by a shrink, or not yet taken from it. Unit `i`
/// occupies the bytes from `i * unit_bytes` of the region's memory, which
/// regions of other layouts may occupy too, so a unit enters the region only
/// when its bytes are in no other region (R10).
#[derive(Serialize, Deserialize)]
pub(crate) struct Region {
    /// The name sessions state it under: a region another session states
    /// under the same name is this region.
    pub(crate) name: String,
    pub(crate) tier: Tier,
    /// The memory space whose bytes the region's units occupy; regions of
    /// two layouts over the same memory state the same space.
    pub(crate) space: u8,
    pub(crate) unit_bytes: u64,
    /// The bytes of one rank's part of a unit (`storage::runtime::Region::part_bytes`):
    /// what a block that lies in one rank's memory holds of each of its units.
    pub(crate) part_bytes: u64,
    /// The memory the region lies in, as its bit's number in a block's
    /// validity, when its units exist once (a rank's product buffer); `None`
    /// when they exist in every memory that maps it (`storage::runtime::Region::memory`).
    pub(crate) memory: Option<u8>,
    /// Whether the region was closed ([`super::BufferPool::close_region`]): its units went back
    /// to the quota and its number is free for the next region opened. A region opened and not
    /// yet named by a session (one an admission states before its session opens) is not closed,
    /// however few units it holds.
    pub(crate) closed: bool,
    /// The free runs by their first unit; neighbours are joined.
    free: BTreeMap<u32, Run>,
    /// The runs outside the region by their first unit.
    outside: BTreeMap<u32, Run>,
    /// The units in the region: free or in a block.
    units: u32,
    /// The units numbered for the region: from the first that can be
    /// allocated up to the limit.
    span: (u32, u32),
    /// The slot rows given back, the last one last: a row is taken from here
    /// first, so a group reuses the row it gave up last.
    returned: Vec<u32>,
    /// Per queue pair (its session and its scheduler's index), the keys its
    /// allocations may evict here, first first.
    pub(crate) orders: BTreeMap<(u32, u16), Vec<Key>>,
    /// The counts the capacity words are made of, kept by the handlers
    /// that change them so that a word costs no scan (`specs/buffer-pool.md`
    /// §10, F6); an image leaves them out and the pool made of it counts
    /// them again. This one is the units of the free runs.
    #[serde(skip)]
    free_units: u32,
    /// The units here of the shared blocks nothing but their entry holds.
    #[serde(skip)]
    pub(crate) evictable: u32,
    /// The units here of the retired blocks.
    #[serde(skip)]
    pub(crate) retiring: u32,
    /// Per queue pair with an order here: the keys of its order, and the
    /// units here of the shared blocks among them nothing but their entry
    /// holds.
    #[serde(skip)]
    pub(crate) installed: BTreeMap<(u32, u16), (HashSet<Key>, u32)>,
}

impl Region {
    /// A region of `limit` units numbered from 0, of which those from
    /// `reserved` to `reserved + units` are in the region and free; the units
    /// before `reserved` are never allocated, and the rest are outside. A
    /// unit is `unit_bytes` over every rank and `part_bytes` on one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: u8,
        name: &str,
        tier: Tier,
        space: u8,
        (unit_bytes, part_bytes): (u64, u64),
        reserved: u32,
        units: u32,
        limit: u32,
    ) -> Self {
        let mut region = Self {
            name: name.to_owned(),
            tier,
            space,
            unit_bytes,
            part_bytes,
            memory: None,
            closed: false,
            free: BTreeMap::new(),
            outside: BTreeMap::new(),
            units,
            span: (reserved, limit),
            returned: Vec::new(),
            orders: BTreeMap::new(),
            free_units: units,
            evictable: 0,
            retiring: 0,
            installed: BTreeMap::new(),
        };
        if units > 0 {
            region.free.insert(
                reserved,
                Run {
                    region: id,
                    first: reserved,
                    count: units,
                },
            );
        }
        let rest = limit - reserved - units;
        if rest > 0 {
            let first = reserved + units;
            region.outside.insert(
                first,
                Run {
                    region: id,
                    first,
                    count: rest,
                },
            );
        }
        region
    }

    /// The units in the region.
    pub(crate) fn units(&self) -> u32 {
        self.units
    }

    /// The first unit that can be allocated and the limit of the numbering.
    pub(crate) fn span(&self) -> (u32, u32) {
        self.span
    }

    /// The free units, as counted.
    pub(crate) fn free_units(&self) -> u32 {
        self.free_units
    }

    /// The free units, by a scan of the free runs: what [`Self::free_units`]
    /// counts.
    pub(crate) fn free_scanned(&self) -> u32 {
        self.free.values().map(Run::count).sum()
    }

    /// Counts the free units again, and starts the other counts afresh: of
    /// a region made from an image, whose blocks are counted next.
    pub(crate) fn recount(&mut self) {
        self.free_units = self.free_scanned();
        self.evictable = 0;
        self.retiring = 0;
        self.installed = self
            .orders
            .iter()
            .map(|(queue, keys)| (*queue, (keys.iter().copied().collect(), 0)))
            .collect();
    }

    /// Replaces the order a queue pair installed; the units of its keys'
    /// blocks that may be evicted are `units`.
    pub(crate) fn install(&mut self, queue: (u32, u16), keys: Vec<Key>, units: u32) {
        self.installed
            .insert(queue, (keys.iter().copied().collect(), units));
        self.orders.insert(queue, keys);
    }

    /// Drops the orders of a session's queue pairs.
    pub(crate) fn uninstall(&mut self, client: u32) {
        self.orders.retain(|queue, _| queue.0 != client);
        self.installed.retain(|queue, _| queue.0 != client);
    }

    /// Counts `units` of a shared block cached under `key` that nothing but
    /// its entry holds (`adds`), or takes them out of the count once
    /// something else holds it or it is gone: in the evictable units, and
    /// in the installed units of every queue pair whose order names the key.
    pub(crate) fn count_evictable(&mut self, key: &Key, units: u32, adds: bool) {
        let change = |count: &mut u32| {
            *count = if adds {
                *count + units
            } else {
                count.saturating_sub(units)
            };
        };
        change(&mut self.evictable);
        for (keys, installed) in self.installed.values_mut() {
            if keys.contains(key) {
                change(installed);
            }
        }
    }

    /// Counts `units` of a retired block (`adds`), or takes them out of the
    /// count once it is no longer retired or gone.
    pub(crate) fn count_retiring(&mut self, units: u32, adds: bool) {
        self.retiring = if adds {
            self.retiring + units
        } else {
            self.retiring.saturating_sub(units)
        };
    }

    /// The units an allocation of a queue pair can take: the free units,
    /// and the units of the blocks of its order that may be evicted.
    pub(crate) fn available(&self, queue: (u32, u16)) -> u32 {
        let installed = self.installed.get(&queue).map_or(0, |(_, units)| *units);
        self.free_units + installed
    }

    /// The free runs as `(first, count)`, for the exported state.
    pub(crate) fn free_runs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.free.values().map(|run| (run.first, run.count))
    }

    /// The runs outside the region as `(first, count)`.
    pub(crate) fn outside_runs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.outside.values().map(|run| (run.first, run.count))
    }

    /// Takes a run of `count` units from the free list: the lowest free run
    /// that holds it.
    pub(crate) fn take(&mut self, count: u32) -> Option<Run> {
        let first = self.free.values().find(|run| run.count >= count)?.first;
        let mut run = self.free.remove(&first)?;
        if run.count == count {
            self.free_units -= run.count;
            return Some(run);
        }
        let head = run.split(count);
        self.free.insert(run.first, run);
        self.free_units -= head.count;
        Some(head)
    }

    /// Takes one unit: the one given back last that is still free, and
    /// otherwise the lowest free one.
    pub(crate) fn take_row(&mut self) -> Option<Run> {
        while let Some(unit) = self.returned.pop() {
            // The free run that holds the unit, when a shrink has not taken it.
            let holder = self
                .free
                .range(..=unit)
                .next_back()
                .map(|(first, run)| (*first, run.count));
            let Some((first, count)) = holder.filter(|(first, count)| unit < first + count) else {
                continue;
            };
            let Some(mut run) = self.free.remove(&first) else {
                continue;
            };

            // Cut the unit out of the run: what lies before it and after it stays free.
            self.free_units -= 1;
            if unit > first {
                let head = run.split(unit - first);
                self.free.insert(head.first, head);
            }
            if first + count > unit + 1 {
                let row = run.split(1);
                self.free.insert(run.first, run);
                return Some(row);
            }
            return Some(run);
        }
        self.take(1)
    }

    /// Returns a run to the free list (R2: only [`super::BufferPool::reclaim`]
    /// calls this, with the runs of a block nothing else holds).
    pub(crate) fn give(&mut self, run: Run) {
        self.free_units += run.count;
        insert_joined(&mut self.free, run);
    }

    /// As [`Self::give`], for a slot row: it is the next one [`Self::take_row`] takes.
    pub(crate) fn give_row(&mut self, run: Run) {
        self.returned.push(run.first);
        self.give(run);
    }

    /// Shrinks the region by up to `count` free units, highest first, and
    /// returns the byte ranges that left it.
    pub(crate) fn shrink(&mut self, count: u32) -> Vec<(u64, u64)> {
        let mut left = Vec::new();
        let mut wanted = count;
        while wanted > 0 {
            let Some(first) = self.free.keys().next_back().copied() else {
                break;
            };
            let Some(mut run) = self.free.remove(&first) else {
                break;
            };
            // Take the run's tail when it is longer than what is wanted.
            let taken = if run.count > wanted {
                let keep = run.count - wanted;
                let head = run.split(keep);
                self.free.insert(head.first, head);
                run
            } else {
                run
            };
            wanted -= taken.count;
            self.units -= taken.count;
            self.free_units -= taken.count;
            left.push((
                u64::from(taken.first) * self.unit_bytes,
                u64::from(taken.count) * self.unit_bytes,
            ));
            insert_joined(&mut self.outside, taken);
        }
        left
    }

    /// Grows the region by up to `count` units that are outside it and whose
    /// bytes `takes` grants (R10: bytes that have left every other region),
    /// lowest first. `takes(offset, bytes)` returns whether it took the range
    /// out of the quota.
    pub(crate) fn grow(&mut self, count: u32, mut takes: impl FnMut(u64, u64) -> bool) -> u32 {
        let mut grown = 0;
        let firsts: Vec<u32> = self.outside.keys().copied().collect();
        for first in firsts {
            let Some(mut run) = self.outside.remove(&first) else {
                continue;
            };
            // One unit at a time: a unit enters only with all of its bytes.
            let mut joined = 0;
            while joined < run.count && grown < count {
                let unit = run.first + joined;
                if !takes(u64::from(unit) * self.unit_bytes, self.unit_bytes) {
                    break;
                }
                joined += 1;
                grown += 1;
            }
            if joined == run.count {
                insert_joined(&mut self.free, run);
            } else if joined > 0 {
                let head = run.split(joined);
                insert_joined(&mut self.free, head);
                self.outside.insert(run.first, run);
            } else {
                self.outside.insert(run.first, run);
            }
            if grown == count {
                break;
            }
        }
        self.units += grown;
        self.free_units += grown;
        grown
    }

    /// Whether every unit of `first..end` that the region numbers is
    /// outside it: in no free run and no block.
    pub(crate) fn outside_holds(&self, first: u32, end: u32) -> bool {
        let (from, to) = (first.max(self.span.0), end.min(self.span.1));
        let mut unit = from;
        while unit < to {
            let holder = self
                .outside
                .range(..=unit)
                .next_back()
                .map(|(_, run)| run.first + run.count)
                .filter(|run_end| *run_end > unit);
            let Some(run_end) = holder else {
                return false;
            };
            unit = run_end;
        }
        true
    }
}

// The enumeration's identity of a state tells apart states that differ in the order of the rows
// given back, which the next row taken follows.
#[cfg(any(test, feature = "verification"))]
impl Region {
    /// The rows given back and not taken again, the last one last.
    pub(super) fn returned(&self) -> &[u32] {
        &self.returned
    }
}

/// Inserts a run into a list of runs by first unit, joining it with the runs
/// that end where it begins or begin where it ends.
fn insert_joined(runs: &mut BTreeMap<u32, Run>, mut run: Run) {
    if let Some(next) = runs.remove(&(run.first + run.count)) {
        run.count += next.count;
    }
    let before = runs
        .range(..run.first)
        .next_back()
        .map(|(first, previous)| (*first, previous.count));
    if let Some((first, count)) = before
        && first + count == run.first
        && let Some(mut previous) = runs.remove(&first)
    {
        previous.count += run.count;
        run = previous;
    }
    runs.insert(run.first, run);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(units: u32, limit: u32) -> Region {
        Region::new(0, "units", Tier::Device, 0, (64, 64), 1, units, limit)
    }

    #[test]
    fn runs_are_taken_lowest_first_and_joined_when_given_back() {
        let mut region = region(8, 9);
        let first = region.take(3).unwrap();
        let second = region.take(2).unwrap();
        assert_eq!((first.stated(), second.stated()), ((0, 1, 3), (0, 4, 2)));
        assert_eq!(region.free_units(), 3);

        // The middle run comes back first: it joins the free tail, then the head joins both.
        region.give(second);
        assert_eq!(region.free_runs().collect::<Vec<_>>(), [(4, 5)]);
        region.give(first);
        assert_eq!(region.free_runs().collect::<Vec<_>>(), [(1, 8)]);
    }

    #[test]
    fn a_run_is_refused_when_no_free_run_holds_it() {
        let mut region = region(4, 5);
        let head = region.take(1).unwrap();
        let middle = region.take(2).unwrap();
        region.give(head);

        // Two units are free, in runs of one each.
        assert_eq!(region.free_units(), 2);
        assert!(region.take(2).is_none());
        region.give(middle);
        assert!(region.take(4).is_some());
    }

    #[test]
    fn a_row_is_the_one_given_back_last_and_otherwise_the_lowest() {
        let mut region = region(4, 5);
        let rows: Vec<Run> = (0..3).map(|_| region.take_row().unwrap()).collect();
        assert_eq!(rows.iter().map(Run::first).collect::<Vec<_>>(), [1, 2, 3]);

        // Rows 1 and then 3 come back: 3 is reused first, then 1, then the lowest free one.
        let mut rows = rows.into_iter();
        let (first, second, third) = (
            rows.next().unwrap(),
            rows.next().unwrap(),
            rows.next().unwrap(),
        );
        region.give_row(first);
        region.give_row(third);
        assert_eq!(region.take_row().unwrap().first(), 3);
        assert_eq!(region.take_row().unwrap().first(), 1);
        assert_eq!(region.take_row().unwrap().first(), 4);
        assert!(region.take_row().is_none());
        region.give_row(second);
        assert_eq!(region.free_runs().collect::<Vec<_>>(), [(2, 1)]);
    }

    #[test]
    fn a_shrink_takes_free_units_from_the_top_and_a_grow_puts_granted_ones_back() {
        let mut region = region(6, 7);
        let held = region.take(2).unwrap();

        // Four units are free; a shrink of ten stops at them.
        let left = region.shrink(10);
        assert_eq!(left, [(3 * 64, 4 * 64)]);
        assert_eq!((region.units(), region.free_units()), (2, 0));
        assert_eq!(region.outside_runs().collect::<Vec<_>>(), [(3, 4)]);

        // The quota grants the first two units' bytes only.
        let grown = region.grow(4, |offset, _| offset < 5 * 64);
        assert_eq!(grown, 2);
        assert_eq!((region.units(), region.free_units()), (4, 2));
        assert_eq!(region.outside_runs().collect::<Vec<_>>(), [(5, 2)]);
        region.give(held);
        assert_eq!(region.free_runs().collect::<Vec<_>>(), [(1, 4)]);
    }
}
