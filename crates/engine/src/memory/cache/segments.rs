//! The segments a device cache's value and state planes are made of.
//!
//! Each value plane, and each state plane of a cache that keeps recurrent state, is made of
//! segments along the units, each column's part of a segment one allocation, mapped back to back
//! so the plane keeps its layout and addresses. A segment is the fewest units whose part of a
//! plane's column is whole allocation granules ([`segment_units`]): 32 units of 64 KiB planes on
//! a device of 2 MiB granules. Each column's part of a plane is its own allocation at its own
//! address, so this is the finest step in which the planes' memory can exist, and every creation,
//! release and creation again of memory is in whole segments: a region holds what its GPU's quota
//! holds to within one granule of each of its allocations, at a start, a grow, a wake and a
//! release alike.
//!
//! The segments are the memory of the cache's buffer pool region: the epoch that creates the
//! region splits its numbering into them ([`own`]), and every other epoch that names the region
//! maps the same segments, as the region's segment table states them ([`of_table`]). The last
//! segments are given back to the device when a region of another layout needs their storage,
//! when the device must make room for other storage, or all of them while the deployment sleeps,
//! and created again at their addresses afterwards; the buffer pool gives none of their units to
//! a block meanwhile.

use crate::worker::{Error, Result};

/// One segment of a cache's value and state planes, as the rank that maps it takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneSegment {
    /// The segment: the epoch that created its storage, its ordinal there and its units.
    pub segment: crate::memory::cache::info::CacheSegment,
    /// Whether the rank creates the segment when its storage does not exist; one that is not
    /// created is left unmapped, as a segment given back.
    pub create: bool,
}

/// The units of one segment of a cache whose planes take `plane_bytes` per unit on a device of
/// allocation granules of `granularity` bytes: the fewest units whose part of a plane's column
/// fills whole granules, `granularity / gcd(plane_bytes, granularity)`. One granule a column
/// when the plane's bytes divide the granule, as a plane of a power of two bytes does. `None` for
/// planes or granules of no bytes.
pub(crate) fn segment_units(plane_bytes: u64, granularity: u64) -> Option<u32> {
    if plane_bytes == 0 || granularity == 0 {
        return None;
    }

    let (mut a, mut b) = (plane_bytes, granularity);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    u32::try_from(granularity / a).ok()
}

/// How a cache of `num_units` units splits its planes into segments ([`segment_units`]): the
/// segments' count and the units each holds. `None` when the cache holds no whole segment or is
/// not a whole number of them ([`segmented_units`] rounds it).
pub(crate) fn segment_layout(
    num_units: u32,
    plane_bytes: u64,
    granularity: u64,
) -> Option<(u32, u32)> {
    let units = segment_units(plane_bytes, granularity)?;
    let count = num_units / units;

    (count > 0 && num_units.is_multiple_of(units)).then_some((count, units))
}

/// `num_units` rounded down to whole segments ([`segment_layout`]): less by under one segment;
/// `num_units` itself when it holds no whole segment.
pub fn segmented_units(num_units: u32, plane_bytes: u64, granularity: u64) -> u32 {
    let Some(units) = segment_units(plane_bytes, granularity) else {
        return num_units;
    };
    if num_units < units {
        return num_units;
    }
    num_units / units * units
}

/// The fewest units of at least `num_units` that are whole segments ([`segment_layout`]).
pub(crate) fn segmented_units_above(num_units: u32, plane_bytes: u64, granularity: u64) -> u32 {
    let Some(units) = segment_units(plane_bytes, granularity) else {
        return num_units;
    };
    num_units.div_ceil(units).saturating_mul(units).max(units)
}

/// The segments of a cache of `num_units` units that `epoch` creates whole: its units split
/// into segments of [`segment_units`] each ([`segment_layout`]), numbered from zero. `None`
/// when the cache does not split.
pub fn own(
    epoch: u64,
    num_units: u32,
    plane_bytes: u64,
    granularity: u64,
) -> Option<Vec<PlaneSegment>> {
    let (count, units) = segment_layout(num_units, plane_bytes, granularity)?;

    Some(
        (0..count)
            .map(|ordinal| PlaneSegment {
                segment: crate::memory::cache::info::CacheSegment {
                    epoch,
                    ordinal,
                    units,
                },
                create: true,
            })
            .collect(),
    )
}

/// The segments of a region's memory as its segment table states them
/// (`storage::runtime::Opened::segments`): the segments an epoch's ranks created, which a rank
/// of every epoch that names the region maps at the same units. Every rank that maps the
/// region holds its parts of each segment of the table; a part the Storage Server does not hold
/// is created.
///
/// # Errors
///
/// A table that does not lay its segments back to back from unit 0, which the planes' layout
/// requires.
pub fn of_table(table: &[(storage::Segment, u32, u32)]) -> Result<Vec<PlaneSegment>> {
    let mut next = 0;

    table
        .iter()
        .map(|(segment, first, units)| {
            if *first != next || *units == 0 {
                return Err(Error::Invalid(format!(
                    "the region's segment {}/{} holds {units} units from unit {first}, and the \
                     planes' segments lie back to back from unit 0: the next is at unit {next}",
                    segment.epoch, segment.ordinal
                )));
            }
            next += units;

            Ok(PlaneSegment {
                segment: crate::memory::cache::info::CacheSegment {
                    epoch: segment.epoch,
                    ordinal: segment.ordinal,
                    units: *units,
                },
                create: true,
            })
        })
        .collect()
}

/// The units `segments` hold.
pub(crate) fn units(segments: &[PlaneSegment]) -> u32 {
    segments.iter().map(|part| part.segment.units).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane of a thirty-second of a granule fills a granule every thirty-two units: a segment
    /// is 32 units, and a cache fitted to the storage loses fewer than 32 of what it holds, one
    /// granule of each column's allocation.
    #[test]
    fn a_cache_rounds_to_whole_segments_of_one_granule_a_column() {
        let (plane, granule) = (1 << 16, 1 << 21);
        assert_eq!(segment_units(plane, granule), Some(32));

        // Fitted to the storage, a cache rounds down by fewer units than a segment; fitted to a
        // capacity, up by as few.
        assert_eq!(segmented_units(11_007, plane, granule), 10_976);
        assert_eq!(segmented_units(11_008, plane, granule), 11_008);
        assert_eq!(segmented_units_above(10_977, plane, granule), 11_008);
        assert_eq!(segmented_units_above(7, plane, granule), 32);
        assert_eq!(segment_layout(11_008, plane, granule), Some((344, 32)));

        // Too few units for one segment, or a cache that is not whole segments, does not split.
        assert_eq!(segmented_units(7, plane, granule), 7);
        assert_eq!(segment_layout(7, plane, granule), None);
        assert_eq!(segment_layout(11_007, plane, granule), None);

        // A plane whose bytes do not divide the granule (72 KiB against 2 MiB) fills whole
        // granules every 256 units, nine granules of a column.
        assert_eq!(segment_units(72 << 10, granule), Some(256));
        assert_eq!(segment_layout(1024, 72 << 10, granule), Some((4, 256)));
    }

    /// An epoch's own segments are equal and numbered from zero, and created where missing.
    #[test]
    fn own_segments_split_the_units() {
        let segments = own(7, 512, 1 << 16, 1 << 21).expect("512 units split into sixteen");
        assert_eq!(segments.len(), 16);
        assert_eq!(units(&segments), 512);
        assert!(
            segments
                .iter()
                .enumerate()
                .all(|(ordinal, part)| part.segment.epoch == 7
                    && part.segment.ordinal == ordinal as u32
                    && part.create)
        );
    }

    /// A rank of another epoch that names the region lays its planes out from the region's
    /// segment table: the segments the creating epoch's ranks made, each with the identity that
    /// keys its parts, so both epochs map the same storages.
    #[test]
    fn a_region_s_table_gives_the_segments_its_creating_epoch_made() -> Result<()> {
        let made = own(3, 512, 1 << 16, 1 << 21).expect("512 units split into sixteen");
        let mut first = 0;
        let table: storage::runtime::Segments = made
            .iter()
            .map(|part| {
                let entry = (
                    storage::Segment {
                        epoch: part.segment.epoch,
                        ordinal: part.segment.ordinal,
                    },
                    first,
                    part.segment.units,
                );
                first += part.segment.units;
                entry
            })
            .collect();

        assert_eq!(of_table(&table)?, made);
        assert!(of_table(&[])?.is_empty());

        // A table whose segments do not lie back to back from unit 0 lays out no planes.
        let refused = of_table(&[table[0], table[2]])
            .expect_err("a table with a gap")
            .to_string();
        assert!(refused.contains("segment 3/2"), "{refused}");
        assert!(of_table(&table[1..]).is_err());
        Ok(())
    }
}
